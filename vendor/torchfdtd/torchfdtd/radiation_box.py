"""Owned CPU snapshots of six native spectral faces for isolated far fields.

The continuum transform is unchanged. Stored fields carry no material graph.
Incident subtraction is coherent on every face and does not normalize power.
"""
from dataclasses import dataclass
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np
import torch
from .models import Project
from .radiation import project_farfield, project_nearzone, _plane, _rectangle
from .radiation_io import native_radiation_plane

_NAMES=tuple(a+s for a in 'xyz' for s in ('_min','_max'))
_ARRAYS=('fields','frequency_hz','points_um','weights')


def _budget(required,budget):
    if type(budget) is not int or budget<=0: raise ValueError('host_budget_bytes must be a positive integer.')
    from .memory_profile import host_memory
    available=host_memory()['available_bytes']
    limit=budget if available is None else min(budget,int(.8*available))
    if required>limit: raise ValueError(f'Far-field host byte budget or available RAM exceeded: {required} > {limit}.')


def _parts(result):
    if isinstance(result,Mapping): return result['project'],result['summary'],result['frequency_fields']
    return result.project,result.summary,result.frequency_fields


def _ids(mapping,open_surface=False):
    if not isinstance(mapping,Mapping): raise ValueError('Provide named box faces: '+', '.join(_NAMES))
    if open_surface:
        if not mapping or set(mapping)-set(_NAMES) or len(mapping)==6:
            raise ValueError('Open-surface projection takes one to five named box faces; use the closed box for all six.')
    elif set(mapping)!=set(_NAMES):
        raise ValueError('Provide exactly six named closed-box faces: '+', '.join(_NAMES))
    if any(not isinstance(v,str) or not v for v in mapping.values()) or len(set(mapping.values()))!=len(mapping):
        raise ValueError('Each box face must select a distinct nonempty stored monitor ID.')
    return {name:mapping[name] for name in _NAMES if name in mapping}


def _preflight(result,ids,index):
    project,summary,records=_parts(result)
    if not isinstance(summary,Mapping): raise ValueError('Native run summary is required.')
    selected=[];numeric=points=0
    for name in ids:
        matches=[v for v in records if v.get('id')==ids[name]]
        if len(matches)!=1: raise ValueError('Each selected monitor ID must identify exactly one stored face.')
        record=matches[0]
        if record.get('fourier_convention','exp(+2 pi i f t)')!='exp(+2 pi i f t)' or record.get('field_units','reduced field * s')!='reduced field * s':
            raise ValueError('Stored fields require native positive-time DFT and reduced spectral field units.')
        if record.get('time_downsample',1)!=1:
            raise ValueError('Stored frequency faces require time downsample 1.')
        arrays=[]
        for key in _ARRAYS:
            value=record.get(key)
            if not isinstance(value,np.ndarray) or value.dtype.hasobject:
                raise ValueError('Stored face arrays must be numeric NumPy arrays, not objects or lazy inputs.')
            numeric+=value.nbytes;arrays.append(value)
        fields,frequency,position,weights=arrays
        if fields.ndim!=3 or fields.shape[-1]!=6 or fields.dtype not in (np.dtype('complex64'),np.dtype('complex128')):
            raise ValueError('Stored fields require complex64/complex128 shape (F,P,6).')
        if frequency.ndim!=1 or fields.shape[0]!=len(frequency) or position.shape!=(fields.shape[1],3) or weights.shape!=(fields.shape[1],):
            raise ValueError('Stored face frequency/point/weight shapes are inconsistent.')
        if not fields.shape[0] or not fields.shape[1]: raise ValueError('Stored faces cannot be empty.')
        shape=record.get('shape');normal=record.get('normal_axis')
        if normal not in ('x','y','z') or not isinstance(shape,(list,tuple)) or len(shape)!=3 or any(type(v) is not int or v<1 for v in shape) or math.prod(shape)!=fields.shape[1] or shape['xyz'.index(normal)]!=1:
            raise ValueError('Stored face shape must describe one complete native plane.')
        if index is not None and (type(index) is not int or not 0<=index<fields.shape[0]):
            raise ValueError('frequency_index must select a stored frequency, or be None.')
        if any(np.iscomplexobj(x) or not np.issubdtype(x.dtype,np.number) for x in arrays[1:]):
            raise ValueError('Frequency, points and weights require real numeric arrays.')
        points+=fields.shape[1];selected.append(record)
    payload=project.model_dump(mode='json') if isinstance(project,Project) else project
    metadata=dict(project=payload,summary=dict(summary),records=[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in selected])
    encoded=json.dumps(metadata,allow_nan=False)
    return payload,dict(summary),selected,numeric,points,len(encoded.encode())


def _axis_workspace_bytes(project_payload):
    """Scalar-only bound for cached nodes and simultaneous FP64 axis scratch.

    A uniform axis needs at most ceil(size/step)+1 nodes. Reserve 32 FP64
    copies per axis (256 bytes/node), covering cached sample/reference nodes,
    source-coordinate work, component axes, and interpolation searches. This
    helper deliberately does not validate Project or access region.mesh_nodes.
    """
    if not isinstance(project_payload,Mapping): raise ValueError('Native project metadata must be a mapping.')
    region=project_payload.get('region',{})
    if not isinstance(region,Mapping): raise ValueError('Native region metadata must be a mapping.')
    if region.get('dimension','2d')!='3d' or region.get('mesh_type','uniform')!='uniform':
        raise ValueError('Stored far fields require isolated uniform 3D geometry.')
    size=region.get('size',(8,6,2));step=region.get('mesh_steps')
    if step is None: step=(region.get('mesh',.05),)*3
    if not isinstance(size,(tuple,list)) or len(size)!=3 or not isinstance(step,(tuple,list)) or len(step)!=3:
        raise ValueError('Uniform mesh sizes and steps need three scalar values.')
    nodes=0
    for span,h in zip(size,step):
        if any(isinstance(x,bool) or not isinstance(x,(int,float,np.number)) or not math.isfinite(x) or x<=0 for x in (span,h)):
            raise ValueError('Uniform mesh sizes and steps must be positive finite scalars.')
        ratio=float(span)/float(h)
        if not math.isfinite(ratio) or ratio>1_000_001:
            raise ValueError('Uniform mesh axis exceeds the supported native cell limit.')
        nodes+=max(1,math.ceil(ratio))+2
    return 256*nodes


def _complete(summary):
    if summary.get('cancelled',False) or summary.get('status') in ('failed','cancelled'):
        raise ValueError('Cancelled or failed native results cannot define a complete radiation surface.')
    if type(summary.get('steps')) is not int or summary['steps']<=0:
        raise ValueError('Native completed-step metadata is required.')
    reason=summary.get('termination_reason')
    if reason not in ('steps','step_limit','completed','decayed','max_steps'):
        raise ValueError('Native result must have a successful step-limit or decay termination reason.')


def _geometry(project,bounds,index,reference):
    r=project.region
    if r.dimension!='3d' or r.mesh_type!='uniform' or any(f.kind!='pml' for a in range(3) for f in r.boundaries.pair(a)):
        raise ValueError('Stored far fields require isolated uniform 3D geometry with PML on all six outer faces.')
    if r.complex_fields: raise ValueError('Periodic/Bloch native results are not isolated closed-box radiation data.')
    if not math.isclose(r.background_index,index,rel_tol=2e-7,abs_tol=1e-9):
        raise ValueError('Declared exterior index must match the native homogeneous background.')
    for a,(lo,hi) in enumerate(bounds):
        low,high=r.interior_bounds(a)
        if not low<lo<hi<high: raise ValueError('All box faces must lie strictly inside the PML-free interior.')
    materials={m.name:m for m in project.materials}
    from .geometry import object_bounds
    for obj in project.structures:
        if not obj.enabled: continue
        material=materials[obj.material]
        homogeneous=(material.model=='dielectric' and not material.oscillators and
                     math.isclose(material.instantaneous_epsilon,index**2,rel_tol=2e-7,abs_tol=1e-9))
        if homogeneous: continue
        if reference: raise ValueError('Incident reference must be homogeneous without active contrast objects.')
        center,size=object_bounds(obj)
        if any(center[a]-size[a]/2<=bounds[a][0]+r.axis_steps[a] or
               center[a]+size[a]/2>=bounds[a][1]-r.axis_steps[a] for a in range(3)):
            raise ValueError('All contrast-object support bounds must be inside the box, one mesh cell clear of its faces.')


def _sources(project,bounds,subtracted):
    """Admit source provenance; return True when every face lies in a TFSF scattered-field region."""
    from .solver import source_slice,field_axes
    r=project.region;scattered=False
    for raw in project.sources:
        source=project.resolved_source(raw)
        if not source.enabled: continue
        if source.kind=='tfsf':
            # The staggered box corrects E on the lo/hi node faces and H one
            # half cell outside them. Faces are admitted only when clear of
            # that shell by one cell on the scattered side (fields are the
            # scattered field) or on the total-field side (matched reference
            # subtraction removes the incident wave). A crossing box mixes
            # total and scattered samples and is not a closed radiation surface.
            from .tfsf import tfsf_plan
            _,lo,hi,_=tfsf_plan(source,r);nodes=r.mesh_nodes;step=r.axis_steps
            outer=[(float(nodes[a][lo[a]])-step[a],float(nodes[a][hi[a]])+step[a]) for a in range(3)]
            inner=[(float(nodes[a][lo[a]])+step[a],float(nodes[a][hi[a]])-step[a]) for a in range(3)]
            if all(bounds[a][0]+step[a]<outer[a][0] and outer[a][1]<bounds[a][1]-step[a] for a in range(3)):
                scattered=True;continue
            if not all(inner[a][0]<bounds[a][0]-step[a] and bounds[a][1]+step[a]<inner[a][1] for a in range(3)):
                raise ValueError('TFSF box faces cross or approach the measurement faces; enclose the TFSF box with one cell clearance or place the box inside its total-field region.')
            if not subtracted: raise ValueError('A measurement box inside a TFSF total-field region needs a matched reference for incident subtraction.')
            continue
        if source.injection!='soft': raise ValueError('One-way planes span a periodic transverse cell and are not isolated closed-box sources.')
        for component,_ in source.polarization_components:
            scalar=source.model_copy(update={'component':component,'theta':None})
            loc=source_slice(scalar,r);axes=field_axes(r,component)
            support=tuple((float(axis[s]),float(axis[s])) if isinstance(s,(int,np.integer)) else
                          (float(axis[s.start]),float(axis[s.stop-1])) for axis,s in zip(axes,loc))
            inside=all(bounds[a][0]+r.axis_steps[a]<support[a][0] and support[a][1]<bounds[a][1]-r.axis_steps[a] for a in range(3))
            if inside: continue
            disjoint=any(support[a][1]<bounds[a][0]-r.axis_steps[a] or support[a][0]>bounds[a][1]+r.axis_steps[a] for a in range(3))
            if not subtracted: raise ValueError('Total-field source support must be enclosed and clear of the measurement faces.')
            if not disjoint: raise ValueError('Incident source support crosses or approaches the measurement faces; move the source or box.')
    return scattered


def _support(project,plane):
    # Inspect the exact one-dimensional interpolation brackets without building
    # full eight-corner maps or a volume epsilon array.
    from .solver import field_axes
    r=project.region;points=plane.points_um.numpy()
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        for a,axis in enumerate(field_axes(r,component)):
            low=np.searchsorted(axis,points[:,a],side='right')-1
            low=np.clip(low,0,len(axis)-2);high=low+1
            if np.any(low<r.pml_layers(a,0)) or np.any(high>=r.shape[a]-r.pml_layers(a,1)):
                raise ValueError('Face interpolation support enters PML; move the box farther inside.')


def _immutable(value):
    value=np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class StoredRadiationBox:
    """Owned immutable stored arrays. Projection copies are separately admitted."""
    _records: tuple
    bounds_um: tuple
    refractive_index: float
    frequency_hz: tuple
    field_dtype: str
    _report_json: str
    open_surface: bool=False

    @property
    def report(self): return json.loads(self._report_json)

    @staticmethod
    def _rows(value,name):
        if isinstance(value,torch.Tensor) and (value.requires_grad or value.device.type!='cpu'):
            raise ValueError(f'Stored projection {name} must be fixed CPU metadata.')
        shape=getattr(value,'shape',None)
        if shape is None:
            if not isinstance(value,(list,tuple)): raise ValueError(f'{name} must have shape (N,3).')
            count=len(value)
            if any(not isinstance(row,(list,tuple)) or len(row)!=3 for row in value): raise ValueError(f'{name} must have shape (N,3).')
        else:
            if len(shape)!=2 or shape[1]!=3: raise ValueError(f'{name} must have shape (N,3).')
            count=int(shape[0])
        if count<1: raise ValueError(f'At least one row of {name} is required.')
        return count

    def _faces(self,required,host_budget_bytes):
        _budget(required,host_budget_bytes)
        faces={}
        for name,record in self._records:
            plane=native_radiation_plane({k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in record.items()},
                                         frequency_index=None,max_samples=record['fields'].size)
            faces[name]=plane
        return faces

    def project(self,directions,*,phase_origin_um=(0,0,0),edge_window=(0.,0.),direction_chunk=16,point_chunk=2048,host_budget_bytes=512*1024**2):
        if type(direction_chunk) is not int or direction_chunk<1 or type(point_chunk) is not int or point_chunk<1:
            raise ValueError('Projection chunks must be positive integers.')
        if isinstance(phase_origin_um,torch.Tensor) and (phase_origin_um.requires_grad or phase_origin_um.device.type!='cpu'):
            raise ValueError('Stored projection directions and phase origin must be fixed CPU metadata.')
        count=self._rows(directions,'directions')
        f=len(self.frequency_hz);item=self._records[0][1]['fields'].dtype.itemsize
        retained=self.report['retained_bytes'];points=self.report['points']
        d=min(count,direction_chunk);p=min(max(v['fields'].shape[1] for _,v in self._records),point_chunk)
        # All output chunks plus cat, tensor snapshots, validation unique/sort,
        # phase/factor/equivalence currents/einsum temporary engineering bound.
        output=2*f*count*3*item
        workspace=32*f*d*p*item+64*f*(p+d)*item+512*points+count*3*8+65536
        faces=self._faces(2*retained+output+workspace,host_budget_bytes)
        with torch.no_grad():
            return project_farfield(faces,directions,bounds_um=self.bounds_um,refractive_index=self.refractive_index,
                phase_origin_um=phase_origin_um,open_surface=self.open_surface,edge_window=edge_window,
                direction_chunk=direction_chunk,point_chunk=point_chunk)

    def nearzone(self,points_um,*,edge_window=(0.,0.),point_chunk=2048,observation_chunk=64,host_budget_bytes=512*1024**2):
        """Exact finite-distance E/H at fixed CPU points strictly outside the box, without autograd."""
        if type(observation_chunk) is not int or observation_chunk<1 or type(point_chunk) is not int or point_chunk<1:
            raise ValueError('Projection chunks must be positive integers.')
        count=self._rows(points_um,'observation points')
        f=len(self.frequency_hz);item=self._records[0][1]['fields'].dtype.itemsize
        retained=self.report['retained_bytes'];points=self.report['points']
        o=min(count,observation_chunk);p=min(max(v['fields'].shape[1] for _,v in self._records),point_chunk)
        # Output chunks plus cat, tensor snapshots, double-precision geometry
        # and Green-function kernels, weighted currents and the broadcast
        # (F, O, S, 3) potential/curl temporaries.
        output=2*f*count*6*item
        workspace=96*f*o*p*item+128*o*p*8+64*f*(p+o)*item+512*points+count*3*8+65536
        faces=self._faces(2*retained+output+workspace,host_budget_bytes)
        with torch.no_grad():
            return project_nearzone(faces,points_um,bounds_um=self.bounds_um,refractive_index=self.refractive_index,
                open_surface=self.open_surface,edge_window=edge_window,point_chunk=point_chunk,observation_chunk=observation_chunk)


def native_radiation_box(result,monitor_ids,*,bounds_um,refractive_index,frequency_index=0,
                         reference=None,reference_monitor_ids=None,open_surface=False,host_budget_bytes=512*1024**2):
    """Snapshot completed native faces and optionally subtract a matched incident run.

    Full selected input payloads are charged even when selecting one frequency.
    Background-only reference and conservative CAD/source support checks enforce
    the restricted isolated-domain contract. No efficiency normalization occurs.
    ``open_surface`` admits one to five faces of the declared box as the
    documented open-surface approximation; the declared box must still enclose
    the sources and contrast objects.
    """
    if type(open_surface) is not bool: raise ValueError('open_surface must be a boolean.')
    ids=_ids(monitor_ids,open_surface)
    if reference is None and reference_monitor_ids is not None: raise ValueError('Reference monitor IDs require a reference result.')
    ref_ids=_ids(ids if reference_monitor_ids is None else reference_monitor_ids,open_surface) if reference is not None else None
    if ref_ids is not None and set(ref_ids)!=set(ids): raise ValueError('Reference monitor IDs must name the same faces as the sample.')
    if isinstance(refractive_index,(bool,torch.Tensor)) or not isinstance(refractive_index,(float,int,np.floating)) or not math.isfinite(refractive_index) or refractive_index<=0:
        raise ValueError('Exterior refractive index must be a positive finite fixed scalar.')
    if np.shape(bounds_um)!=(3,2): raise ValueError('bounds_um must have shape (3,2).')
    bounds=tuple(tuple(float(x) for x in row) for row in bounds_um)
    if any(not math.isfinite(v) for row in bounds for v in row) or any(lo>=hi for lo,hi in bounds): raise ValueError('Box spans must be positive and finite.')
    data=_preflight(result,ids,frequency_index)
    ref=_preflight(reference,ref_ids,frequency_index) if reference is not None else None
    all_data=[data]+([] if ref is None else [ref])
    numeric=sum(v[3] for v in all_data);points=sum(v[4] for v in all_data);metadata=sum(v[5] for v in all_data)
    axis_workspace=sum(_axis_workspace_bytes(v[0]) for v in all_data)
    required=6*numeric+16*metadata+512*points+65536+axis_workspace
    _budget(required,host_budget_bytes)
    project=Project.model_validate(data[0]);_complete(data[1]);_geometry(project,bounds,refractive_index,False)
    tfsf_scattered=_sources(project,bounds,ref is not None)
    if ref is not None:
        reference_project=Project.model_validate(ref[0]);_complete(ref[1]);_geometry(reference_project,bounds,refractive_index,True)
        _sources(reference_project,bounds,True)
        if data[1]['steps']!=ref[1]['steps']: raise ValueError('Incident reference completed duration must match.')
        ignored={'backend','cuda_kernel','cuda_monitor_kernel','execution_mode','tiling','resident_cell_limit','field','slice_axis','slice_position','complex_display','snapshot_interval'}
        if project.region.model_dump(exclude=ignored)!=reference_project.region.model_dump(exclude=ignored) or [project.resolved_source(s).model_dump() for s in project.sources]!=[reference_project.resolved_source(s).model_dump() for s in reference_project.sources]:
            raise ValueError('Incident reference mesh/background/source metadata must match.')
    stored=[];signature=None;frequency=None;dtype=None;raw_frequency=None
    for i,name in enumerate(ids):
        record=data[2][i]
        plane=native_radiation_plane(record,frequency_index=frequency_index,max_samples=record['fields'].size)
        _rectangle(plane,bounds);_support(project,plane)
        a='xyz'.index(name[0]);side=0 if name.endswith('min') else 1
        tolerance=32*torch.finfo(plane.points_um.dtype).eps*max(1.,max(abs(v) for row in bounds for v in row))
        if plane.normal!=name[0] or not torch.allclose(plane.points_um[:,a],plane.points_um.new_full((len(plane.points_um),),bounds[a][side]),rtol=0,atol=tolerance):
            raise ValueError('Named face normal/position does not match the declared closed box.')
        selected_frequency=record['frequency_hz'] if frequency_index is None else record['frequency_hz'][frequency_index:frequency_index+1]
        if raw_frequency is None: raw_frequency=selected_frequency
        if not np.array_equal(selected_frequency,raw_frequency): raise ValueError('All box faces must have exactly matching stored frequency values before dtype conversion.')
        if signature is None: signature=plane.run_signature;frequency=plane.frequency_hz;dtype=plane.fields.dtype
        if plane.run_signature!=signature or plane.fields.dtype!=dtype or not torch.equal(plane.frequency_hz,frequency):
            raise ValueError('All box faces must share run signature, exact frequencies and field dtype.')
        fields=plane.fields.numpy()
        if ref is not None:
            rrecord=ref[2][i]
            if any(not np.array_equal(record[key],rrecord[key]) for key in ('points_um','weights')):
                raise ValueError('Incident reference raw points and weights must match exactly before dtype conversion.')
            ref_frequency=rrecord['frequency_hz'] if frequency_index is None else rrecord['frequency_hz'][frequency_index:frequency_index+1]
            if not np.array_equal(selected_frequency,ref_frequency): raise ValueError('Incident reference stored frequency values must match exactly before dtype conversion.')
            other=native_radiation_plane(rrecord,frequency_index=frequency_index,max_samples=rrecord['fields'].size)
            _plane(other)
            settings=lambda v:{k:x for k,x in v.get('settings',{}).items() if k not in ('id','name','color')}
            if plane.run_signature!=other.run_signature or plane.normal!=other.normal or plane.fields.dtype!=other.fields.dtype or settings(record)!=settings(rrecord):
                raise ValueError('Incident reference signature, dtype and monitor settings must match on all six faces.')
            if any(not torch.equal(getattr(plane,k),getattr(other,k)) for k in ('frequency_hz','points_um','weights')):
                raise ValueError('Incident reference frequencies and quadrature must match exactly on all six faces.')
            fields=fields-other.fields.numpy()
        owned=dict(id=ids[name],components=list(plane.components),flux_units=plane.flux_units,
            run_signature=signature,settings=dict(spectrum=dict(apodization='none'),time_downsample=1),
            shape=tuple(record['shape']),normal_axis=plane.normal,
            fields=_immutable(fields),frequency_hz=_immutable(plane.frequency_hz.numpy()),
            points_um=_immutable(plane.points_um.numpy()),weights=_immutable(plane.weights.numpy()))
        owned['components']=tuple(owned['components'])
        owned['settings']=MappingProxyType(dict(spectrum=MappingProxyType(dict(apodization='none')),time_downsample=1))
        stored.append((name,MappingProxyType(owned)))
    retained=sum(v[k].nbytes for _,v in stored for k in _ARRAYS)
    report=dict(field_kind='scattered' if ref is not None or tfsf_scattered else 'total',
        incident_removal='matched reference' if ref is not None else 'tfsf scattered-field region' if tfsf_scattered else None,
        retained_bytes=retained,approximation='open surface' if open_surface else None,surfaces=list(ids),
        points=data[4],monitor_ids=ids,run_signature=signature,reference_run_signature=signature if ref is not None else None,
        preparation_reservation_bytes=required,input_payload_bytes=numeric,metadata_bytes=metadata,
        axis_workspace_reservation_bytes=axis_workspace,
        source='owned native stored frequency faces',autograd=False,normalization='none',
        scope='Isolated homogeneous isotropic exterior, uniform complete closed box; continuum projection of native fields.')
    return StoredRadiationBox(tuple(stored),bounds,float(refractive_index),tuple(float(v) for v in frequency),str(dtype).removeprefix('torch.'),json.dumps(report,allow_nan=False),open_surface)
