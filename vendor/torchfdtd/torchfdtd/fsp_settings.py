"""Transactional writes of recognized FSP source, monitor and region settings.

No API/runtime is loaded. Property containers retain their scalar/matrix type.
Unknown entries and dormant settings are preserved unless explicitly edited.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from .fsp_native import FDTD, DIPOLE, PLANE, TFSF, TIME, DFT, identity
from .models import SourceTimeSettings, SpectrumSettings, BoundaryFace
from .spectra import frequency_samples
from .waveforms import C0, pulse_parameters


def _string(text):
    if '\0' in text:raise ValueError('FSP names cannot contain null characters.')
    data=text.encode('utf-8')+b'\0'
    return struct.pack('<I',len(data))+data


def _value(value,stored=None):
    if stored is not None:kind=stored.kind
    else:kind=0 if isinstance(value,str) else 2 if isinstance(value,(int,bool)) else (5 if np.asarray(value).dtype.kind in 'iu' else 3) if np.ndim(value) else 1
    head=struct.pack('<I',kind)
    if kind==0:return head+_string(value)
    if kind in (1,2):
        values=np.asarray(value).reshape(-1)
        if kind==2 and not np.equal(values,np.floor(values)).all():raise ValueError('Integer FSP setting requires integral values.')
        array=np.asarray(values,dtype='<f8' if kind==1 else '<i4')
        return head+struct.pack('<I',array.size)+array.tobytes()
    if kind not in (3,5):raise ValueError('Unsupported writable property type.')
    array=np.asarray(value,dtype='<f8' if kind==3 else '<i4')
    if array.ndim==1 and stored is not None:
        shape=np.shape(stored.value)
        if len(shape)==2 and 1 in shape:array=array.reshape((-1,1) if shape[1]==1 else (1,-1))
    return (head+struct.pack('<III',0,array.size,array.ndim)+struct.pack('<'+'I'*array.ndim,*array.shape)
            +struct.pack('<II',1,1)+array.tobytes(order='F'))


class PropertyPlan:
    def __init__(self,document):self.document=document;self.pending={}
    def set(self,node,key,value):self.pending.setdefault(node.start,(node,{}))[1][key]=value
    def update(self,node,values):
        for key,value in values.items():self.set(node,key,value)
    def patches(self):
        result=[]
        for node,values in self.pending.values():
            additions=[];object_id=identity(node)['id']
            for key,value in values.items():
                old=node.properties.get(key)
                if old is not None and np.array_equal(old.value,value):continue
                packed=_value(value,old)
                if old is None:additions.append(_string(key)+packed)
                else:result.append((old.start,old.end,packed,object_id,key))
            if additions:
                start,end,_=node.legacy_fields['__properties__']
                result.append((start,start+4,struct.pack('<I',len(node.properties)+len(additions)),object_id,'property count'))
                result.append((end,end,b''.join(additions),object_id,'new settings'))
        return result


def _changed(a,b):return {k for k in type(a).model_fields if getattr(a,k)!=getattr(b,k)}


def _temporal_values(source,*,global_=False,local=False):
    if source.time_definition=='cycles' or source.pulse=='continuous':
        raise ValueError('FSP temporal export requires a standard, ranged or sampled pulse. Native cycles/continuous definitions are not mapped.')
    if source.pulse=='sampled' and (global_ or local):
        raise ValueError('Global/inherited sampled source tables are not mapped to FSP yet.')
    p=pulse_parameters(source)
    preference={'standard':0,'frequency':1,'wavelength':2}[source.time_definition]
    names=dict(envelope='frequencyEnvelopeType',preference='sourcePreference',frequency='globalFrequency' if global_ else 'frequency',
               dc='globalEliminateDC' if global_ else 'mEliminateDC',length='pulseLength',offset='offset',
               optimize='optimizeForShortPulse',smooth='eliminateDiscontinuities',low='BBFrequencyStart',high='BBFrequencyStop')
    if local:
        names={k:'local::'+v for k,v in names.items()}
        names.update(envelope='local::pulseTypeActual',preference='local::defineSourceBy',
                     low='local::minGUIFrequency',high='local::maxGUIFrequency')
    result={names['envelope']:2 if source.pulse=='sampled' else int(p.chirped),names['preference']:preference,
            names['frequency']:p.frequency_hz,names['dc']:0,names['length']:p.as_dict()['pulse_length_s'],
            names['offset']:p.offset_s,names['optimize']:int(source.optimize_for_short_pulse),
            names['smooth']:int(source.eliminate_discontinuities)}
    if source.pulse=='broadband' or preference:
        result.update({names['low']:p.frequency_hz-p.frequency_span_hz/2,names['high']:p.frequency_hz+p.frequency_span_hz/2})
    if source.pulse=='sampled':
        result.update(userTime=np.asarray(source.signal.time_s).reshape(-1,1),
                      userAmp=np.asarray(source.signal.amplitude).reshape(-1,1),userPhs=np.asarray(source.signal.phase_rad).reshape(-1,1))
    return result


def _spectrum_values(spec,*,global_=False):
    if spec.sampling=='fft':raise ValueError('FFT-bin spectra do not map to saved FSP DFT frequency settings.')
    if spec.sampling=='chebyshev' and spec.chebyshev_nodes!='lobatto':
        # The explicit frequency table represents exactly the requested roots.
        return dict(sampleSpacing=2,useWavelengthSpacing=0,useSourceLimits=0,
                    customFrequencySamples=frequency_samples(spec).reshape(-1,1))
    result=dict(sampleSpacing=2 if spec.sampling=='custom' else 1 if spec.sampling=='chebyshev' else 0,
                useWavelengthSpacing=int(spec.sampling=='wavelength' or (spec.sampling=='chebyshev' and spec.chebyshev_wavelength)),
                useSourceLimits=int(spec.use_source_limits))
    if spec.sampling=='custom':result['customFrequencySamples']=np.asarray(spec.custom_frequencies_hz).reshape(-1,1)
    else:
        low,high=('fStart','fEnd') if global_ else ('userF1','userF2')
        result.update({low:C0/(spec.wavelength_stop*1e-6),high:C0/(spec.wavelength_start*1e-6),'nfreqDesired':spec.frequency_points})
    return result


def _apodization_values(spec):
    if spec.apodization=='hann':raise ValueError('Hann apodization is a native FFT option, not a mapped FSP monitor window.')
    return dict(apodizationType={'none':0,'full':1,'start':2,'end':3}[spec.apodization],
                apodizationCenter=spec.apodization_center,apodizationWidth=spec.apodization_time_width)


def _region(plan,node,base,project,origin,native_only):
    old,new=base.region,project.region;changes=_changed(old,new)
    local={'backend','cuda_kernel','cuda_monitor_kernel','execution_mode','tiling','resident_cell_limit','precision','snapshot_interval','field','slice_axis','slice_position','complex_display'}
    mesh_fields={'mesh','mesh_type','mesh_steps','mesh_coordinates','size','material_sampling'}
    remesh=bool(changes&mesh_fields)
    supported=local|mesh_fields|{'steps','courant_factor','time_step_override','background_index','boundaries','pml_cells'}
    inactive_mesh={'mesh_max','mesh_grading','mesh_ppw','mesh_auto_refine','mesh_refinements'} if new.mesh_type=='uniform' else set()
    supported|=inactive_mesh
    unsupported=changes-supported
    if unsupported:raise ValueError('Scene export cannot yet write region.'+sorted(unsupported)[0]+'.')
    native_only.extend('region.'+key for key in sorted(changes&local))
    native_only.extend('region.'+key+' (inactive on a uniform mesh)' for key in sorted(changes&inactive_mesh))
    if remesh:
        if new.mesh_type!='uniform':
            raise ValueError('FSP mesh export currently requires a uniform target mesh. Edited graded/explicit node generators are not mapped yet.')
        active=2 if new.dimension=='2d' else 3
        steps=new.axis_steps[:active]
        isotropic=np.allclose(steps,steps[0],rtol=1e-10,atol=0)
        # The current independent reader resolves an isotropic uniform record
        # to its legacy cell sampling, and an anisotropic one to Yee sampling.
        # Never silently change how a dielectric interface is discretized.
        expected_sampling='cell' if isotropic else 'yee'
        if new.material_sampling!=expected_sampling:
            raise ValueError('This FSP uniform mesh mapping requires '+expected_sampling+' material sampling. '
                             'The requested interface sampling cannot be preserved in this record representation.')
        plan.set(node,'customGrid',2)
        plan.update(node,{'d'+a:h*1e-6 for a,h in zip('xyz',new.axis_steps)})
        if new.mesh_coordinates is not None:native_only.append('region.mesh_coordinates (inactive on a uniform mesh)')
        if new.dimension=='2d' and not np.isclose(new.size[2],steps[0],rtol=1e-12,atol=0):
            native_only.append('region.size[2] (invariant 2D display span, imported as dx)')
    if remesh or changes&{'steps','courant_factor','time_step_override'}:
        factor=new.courant_factor
        if new.mesh_type=='uniform':
            # FSP exposes a CFL fraction, not an independent dt input. Encode
            # a smaller native fixed dt through that controlling fraction.
            factor*=new.time_step/new.cfl_time_step
            if new.time_step_override is not None:
                native_only.append('region.time_step_override (effective dt encoded through the CFL factor)')
        plan.update(node,dict(courantFactor=factor,dt=new.time_step,MaxSimTime=new.steps*new.time_step))
    if 'background_index' in changes:plan.set(node,'mRefIndex',format(new.background_index,'.17g'))
    if remesh or changes&{'boundaries','pml_cells'}:
        layers=np.array(node.properties['PMLLayersV7p0'].value).copy().reshape(-1)
        gui=np.array(node.properties['GUIboundary'].value,dtype=int).reshape(-1).copy() if 'GUIboundary' in node.properties else np.array([int(f.kind=='periodic')*2 for axis in range(3) for f in old.boundaries.pair(axis)])
        for axis in range(2 if new.dimension=='2d' else 3):
            a='xyz'[axis];pair=new.boundaries.pair(axis);previous=old.boundaries.pair(axis)
            if any(face.kind not in ('pml','periodic') for face in pair):raise ValueError('FSP Bloch wavevector export is not mapped yet.')
            for side,face in enumerate(pair):
                default=BoundaryFace(kind=face.kind,layers=face.layers)
                if face!=default:raise ValueError('Native CPML profile coefficients do not have a verified FSP conversion.')
                plan.set(node,f'BCType{2*axis+side}',int(face.kind=='periodic'))
                gui[2*axis+side]=2 if face.kind=='periodic' else 0
                if face.kind=='pml':
                    count=new.pml_layers(axis,side)
                    if node.properties.get('PMLType') is not None and node.properties['PMLType'].value==1 and count%2:
                        raise ValueError(f'{a}: this stretched-coordinate FSP profile rounds odd layer counts. An even PML count is required for exact export.')
                    for key,compare in (('minPMLLayers',lambda x:count<x),('maxPMLLayers',lambda x:count>x)):
                        if key in node.properties and compare(np.asarray(node.properties[key].value).reshape(-1)[2*axis+side]):
                            raise ValueError(f'{a}: requested PML layers are outside the stored FSP profile limits. Choose a supported layer count.')
                    layers[2*axis+side]=count
                else:layers[2*axis+side]=0
            if remesh or pair!=previous or new.pml_cells!=old.pml_cells:
                nodes=np.asarray(new.mesh_nodes[axis])*1e-6+origin[axis]
                if pair[0].kind=='periodic':
                    nodes=np.r_[nodes[0]-(nodes[-1]-nodes[-2]),nodes,nodes[-1]+nodes[1]-nodes[0]]
                plan.set(node,a+'Grid',nodes)
                low,high=new.interior_bounds(axis);low=low*1e-6+origin[axis];high=high*1e-6+origin[axis]
                if axis<2:plan.set(node,'GUI'+a,(low+high)/2);plan.set(node,'GUI'+('width' if axis==0 else 'height'),high-low)
                else:plan.update(node,dict(GUIz1=low,GUIz2=high))
        plan.set(node,'PMLLayersV7p0',layers)
        plan.set(node,'GUIboundary',gui.reshape(6,1))


def _source(plan,node,old,new,base,project,origin,native_only,*,fresh=False):
    changes=set(type(new).model_fields) if fresh else _changed(old,new)
    if new.kind!=old.kind or new.injection!=old.injection:raise ValueError('Changing source type/injection is not mapped to FSP yet.')
    if new.component.startswith('H'):raise ValueError('FSP magnetic source encoding is not mapped yet.')
    if 'name' in changes:plan.set(node,'name',new.name)
    if 'enabled' in changes:plan.set(node,'enabled',int(new.enabled))
    if 'center' in changes:
        plan.update(node,{a+'coord':new.center[i]*1e-6+origin[i] for i,a in enumerate('xyz')})
    if node.uid==DIPOLE:
        if 'amplitude' in changes:plan.set(node,'amplitude0',new.amplitude)
        if changes&{'component','theta','phi','phase'}:
            theta,phi=(new.theta,new.phi) if new.theta is not None else {'Ex':(90.,0.),'Ey':(90.,90.),'Ez':(0.,0.)}[new.component]
            plan.update(node,dict(theta=theta,angle=phi,phase=new.phase))
        native_only.extend(new.id+'.'+k for k in sorted(changes&{'normal','direction','size','incident_pml_cells'}))
    else:
        axis='xyz'.index(new.normal)
        if 'amplitude' in changes:plan.set(node,'amplitude',new.amplitude)
        if 'phase' in changes:plan.set(node,'phase',new.phase)
        if 'direction' in changes:plan.set(node,'direction',int(new.direction=='+'))
        if 'normal' in changes:plan.set(node,'type',((3,4,9) if node.uid==TFSF else (1,2,5))[axis])
        if changes&{'component','theta','phi','normal'}:
            components=dict(new.polarization_components);v=np.array([components.get('E'+a,0) for a in 'xyz'])
            if abs(v[axis])>1e-14:raise ValueError('A normal-incidence FSP source must have transverse electric polarization.')
            angle=math.degrees(math.atan2(v[(axis+2)%3],v[(axis+1)%3]))%360
            # Cyclic bases are (y,z), (z,x), (x,y).
            plan.update(node,dict(phi=0.,polarizationAngle=angle))
        if changes&{'size','center','normal'}:
            if node.uid==TFSF:
                low=np.asarray(new.center)*1e-6+origin-np.asarray(new.size)*.5e-6
                plan.update(node,dict(left=low[0],bottom=low[1],z1=low[2],width=new.size[0]*1e-6,
                                      height=new.size[1]*1e-6,z2=low[2]+new.size[2]*1e-6))
            else:plan.update(node,{'GUI'+a+'span':new.size[i]*1e-6 for i,a in enumerate('xyz') if i!=axis})
        if 'incident_pml_cells' in changes:native_only.append(new.id+'.incident_pml_cells')
    temporal=set(SourceTimeSettings.model_fields)
    changed_global=base.global_source!=project.global_source and new.use_global_source
    if changes&(temporal|{'use_global_source'}) or changed_global:
        plan.set(node,'useGlobalSource',int(new.use_global_source))
        plan.update(node,_temporal_values(new,local=new.use_global_source))
        if not new.use_global_source:
            if new.pulse=='sampled':plan.update(node,{'local::defineSourceBy':0,'local::pulseTypeActual':2})
            else:plan.update(node,_temporal_values(new,local=True))
        if new.time_definition in ('wavelength','frequency'):
            parameters=pulse_parameters(new)
            plan.update(node,{'local::bandwidth':parameters.frequency_span_hz})
            if not new.use_global_source:
                plan.update(node,dict(frequency1=parameters.frequency_hz-parameters.frequency_span_hz/2,
                                      frequency2=parameters.frequency_hz+parameters.frequency_span_hz/2))
        if new.use_global_source:plan.update(node,_temporal_values(project.global_source))
    if changes&{'pulse_cycles'}:native_only.append(new.id+'.pulse_cycles (inactive)')


def _monitor(plan,node,old_group,new_group,base,project,origin,native_only,*,fresh=False):
    old,new=old_group[0],new_group[0]
    if new.kind!=old.kind:raise ValueError('Changing a point/plane monitor class is not mapped yet.')
    # Several native point traces may represent one shared FSP monitor record.
    common={'center','enabled','time_downsample','use_global_monitor','inherit_apodization','spectrum'}
    if any(any(getattr(m,key)!=getattr(new,key) for key in common) for m in new_group):
        raise ValueError('Component traces from one FSP monitor must share position, enabled state and spectral settings.')
    changes=set(type(new).model_fields) if fresh else set().union(*(_changed(a,b) for a,b in zip(old_group,new_group)))
    if len(old_group)!=len(new_group):changes.update(('name','component'))
    if 'name' in changes:
        names=[m.name.removesuffix(' '+m.component) if len(new_group)>1 else m.name for m in new_group]
        if len(set(names))!=1:raise ValueError('Component traces from one FSP monitor must share a base name.')
        plan.set(node,'name',names[0])
    if 'enabled' in changes:plan.set(node,'enabled',int(new.enabled))
    if new.kind=='point':
        if 'center' in changes:plan.update(node,{key:new.center[i]*1e-6+origin[i] for i,key in enumerate(('left','bottom','z1'))})
        if 'component' in changes:
            components=[m.component for m in new_group]
            if len(set(components))!=len(components):raise ValueError('FSP monitor components must be unique.')
            if node.uid==TIME:plan.set(node,'outputE',np.array([int(c in components) for c in ('Ex','Ey','Ez','Hx','Hy','Hz')]))
            else:plan.update(node,{'output'+c:int(c in components) for c in ('Ex','Ey','Ez','Hx','Hy','Hz')})
    else:
        if changes&{'normal','center','size'}:
            axis='xyz'.index(new.normal);low=np.asarray(new.center)*1e-6+origin-np.asarray(new.size)*.5e-6
            spans=np.asarray(new.size)*1e-6
            if project.region.dimension=='2d':low[2]=origin[2];spans[2]=0.
            plan.update(node,dict(monitorShape=4+axis if project.region.dimension=='3d' else {'x':2,'y':1}[new.normal],
                left=low[0],bottom=low[1],z1=low[2],width=spans[0],height=spans[1],z2=low[2]+spans[2]))
        if changes&{'downsample','downsample_xyz'}:
            plan.update(node,{'downsample'+a:v for a,v in zip('XYZ',new.downsample_xyz or (new.downsample,)*3)})
        if 'record_fields' in changes:plan.update(node,{'output'+c:int(c in new.record_fields) for c in ('Ex','Ey','Ez','Hx','Hy','Hz')})
        if 'record_poynting' in changes:plan.update(node,{'outputP'+a:int(a in new.record_poynting) for a in 'xyz'})
        if 'record_flux' in changes:plan.set(node,'outputPower',int(new.record_flux))
        if 'dft_precision' in changes:plan.set(node,'highPrecisionDFT',int(new.dft_precision=='float64'))
        if 'spatial_interpolation' in changes:plan.set(node,'spatialAveraging',int(new.spatial_interpolation=='nearest'))
    if node.uid==TIME:
        if fresh:
            if new.spectrum.sampling!='fft' or new.spectrum.apodization!='none' or new.time_downsample!=1 or new.use_global_monitor:
                raise ValueError('New FSP time monitors require default FFT settings with apodization="none", stride 1 and no global inheritance. Use an explicit DFT monitor for other spectral settings.')
            if new.spectrum!=SpectrumSettings(apodization='none'):native_only.append(new.id+'.inactive FFT frequency/window parameters')
        elif changes&{'spectrum','time_downsample','use_global_monitor','inherit_apodization'}:
            raise ValueError('Time monitor FFT postprocessing settings are native-only. FSP time-window export is not mapped yet.')
        return
    temporal_change=(base.sources!=project.sources or base.global_source!=project.global_source or base.region.time_step!=project.region.time_step)
    sampling_change=bool(changes&{'time_downsample','spectrum','use_global_monitor'}) or (base.global_monitor!=project.global_monitor and new.use_global_monitor)
    if temporal_change or sampling_change:
        if new.time_downsample>1 and any(project.resolved_source(s).pulse=='sampled' and s.enabled for s in project.sources):
            raise ValueError('Sampled sources require time_downsample=1 for verified FSP monitor export. The external automatic sampling limit can override a coarser target.')
        effective=project.resolved_monitor(new).spectrum
        maximum=float(np.max(frequency_samples(effective)))
        # Encode a rate strictly inside the requested integer-stride interval.
        # A stricter external source-bandwidth limit may still force finer
        # sampling, which is reported independently from this input target.
        minimum=max(2.,1./((new.time_downsample+.25)*project.region.time_step*maximum))
        plan.update(node,dict(useGlobalAdvanced=0,minSamplingPerCycle=minimum,downsampleT=new.time_downsample))
        native_only.append(new.id+'.DFT stride encoded as local sampling rate, subject to external source Nyquist limit')
    global_change=base.global_monitor!=project.global_monitor and new.use_global_monitor
    if changes&{'spectrum','use_global_monitor','inherit_apodization'} or global_change:
        plan.set(node,'useGlobalDFT',int(new.use_global_monitor))
        effective=project.resolved_monitor(new).spectrum
        plan.update(node,_apodization_values(effective))
        if new.use_global_monitor:
            # Keep dormant local frequency properties as originally stored.
            if new.spectrum!=old.spectrum:native_only.append(new.id+'.inactive local frequency settings')
        else:plan.update(node,_spectrum_values(new.spectrum))
        if new.inherit_apodization and new.use_global_monitor:native_only.append(new.id+'.apodization inheritance (effective local window written)')
        if effective.sampling=='chebyshev' and effective.chebyshev_nodes=='roots':native_only.append(new.id+'.Chebyshev roots encoded as exact custom frequencies')


def plan_settings(document,base,project,conversion):
    plan=PropertyPlan(document);native_only=[]
    nodes={n.start:n for n in document.root.children};origin=np.asarray(conversion.origin_m)
    domain=next(n for n in nodes.values() if n.uid==FDTD)
    _region(plan,domain,base,project,origin,native_only)
    if base.global_source!=project.global_source:
        if project.global_source is None:raise ValueError('Deleting global FSP source settings is not mapped.')
        plan.update(domain,_temporal_values(project.global_source,global_=True))
    if base.global_monitor!=project.global_monitor:
        # Global apodization is a native abstraction, saved monitors have local windows.
        plan.update(domain,_spectrum_values(project.global_monitor,global_=True))
        if project.global_monitor.apodization!='none':native_only.append('global_monitor.apodization (effective local windows written)')
    from .fsp_instruments import plan_instruments
    instruments=plan_instruments(plan,document,base,project,conversion,native_only)
    return plan.patches(),native_only,instruments


def _close(a,b,*,atol=0,rtol=2e-11):
    return np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=rtol,atol=atol)


def _same_pulse(a,b):
    if (a.pulse=='sampled')!=(b.pulse=='sampled'):return False
    if a.pulse=='sampled':return a.signal==b.signal
    p,q=pulse_parameters(a),pulse_parameters(b)
    return (p.chirped==q.chirped and a.eliminate_discontinuities==b.eliminate_discontinuities
        and _close((p.frequency_hz,p.sigma_s,p.offset_s),(q.frequency_hz,q.sigma_s,q.offset_s))
        and (not p.chirped or _close(p.frequency_span_hz,q.frequency_span_hz)))


def verify_settings(expected,actual,before,after,monitor_groups=None):
    id_mapping={}
    a,b=expected.region,actual.region
    if (a.dimension,a.steps)!=(b.dimension,b.steps) or not _close(a.time_step,b.time_step) or a.background_index!=b.background_index:
        raise ValueError('FSP region writeback verification failed.')
    active=2 if a.dimension=='2d' else 3
    if any(not _close(x,y,atol=5e-12) for x,y in zip(a.mesh_nodes[:active],b.mesh_nodes[:active])):
        raise ValueError('FSP export changed the native mesh nodes.')
    if a.material_sampling!=b.material_sampling:
        raise ValueError('FSP export changed the material interface sampling.')
    for axis in range(2 if a.dimension=='2d' else 3):
        if [f.kind for f in a.boundaries.pair(axis)]!=[f.kind for f in b.boundaries.pair(axis)] or any(a.pml_layers(axis,s)!=b.pml_layers(axis,s) for s in (0,1)):
            raise ValueError('FSP boundary writeback verification failed.')
    if len(expected.sources)!=len(actual.sources) or len(expected.monitors)!=len(actual.monitors):
        raise ValueError('FSP source/monitor counts changed during export.')
    if expected.global_source is not None and (actual.global_source is None or not _same_pulse(expected.global_source,actual.global_source)):
        raise ValueError('Global source writeback verification failed.')
    if expected.global_monitor.sampling!='fft' and not expected.global_monitor.use_source_limits and not _close(frequency_samples(expected.global_monitor),frequency_samples(actual.global_monitor)):
        raise ValueError('Global monitor writeback verification failed.')
    for a,b in zip(expected.sources,actual.sources):
        id_mapping[a.id]=b.id
        if (a.kind,a.enabled,a.name,a.use_global_source)!=(b.kind,b.enabled,b.name,b.use_global_source) or not _close(a.center,b.center,atol=5e-12):
            raise ValueError(a.name+': source geometry writeback verification failed.')
        if a.kind!='point' and ((a.normal,a.direction)!=(b.normal,b.direction) or not _close(a.size,b.size,atol=5e-12)):
            raise ValueError(a.name+': paired source support writeback verification failed.')
        def polarization(s):
            weights=dict(s.polarization_components)
            return np.array([weights.get('E'+c,0) for c in 'xyz'])*s.amplitude*np.exp(1j*np.deg2rad(s.phase))
        if not _close(polarization(a),polarization(b),atol=1e-12) or not _same_pulse(expected.resolved_source(a),actual.resolved_source(b)) or not _same_pulse(a,b):
            raise ValueError(a.name+': source pulse/polarization writeback verification failed.')
    # A component change may reorder outputs inside one shared point record.
    expected_ids={m.id for m in expected.monitors};actual_ids={m.id for m in actual.monitors}
    before_monitors=([dict(native_ids=ids) for ids in monitor_groups] if monitor_groups is not None else
                     [row for row in before.mappings if expected_ids.intersection(row['native_ids'])])
    after_monitors=[row for row in after.mappings if actual_ids.intersection(row['native_ids'])]
    if len(before_monitors)!=len(after_monitors):raise ValueError('FSP monitor record count changed during export.')
    for left,right in zip(before_monitors,after_monitors):
        old_ids=left['native_ids'];new_ids=right['native_ids']
        aa=[m for m in expected.monitors if m.id in old_ids];bb=[m for m in actual.monitors if m.id in new_ids]
        if len(aa)!=len(bb):raise ValueError('FSP monitor component count changed during export.')
        if not aa:continue
        if aa[0].kind=='point':aa=sorted(aa,key=lambda m:m.component);bb=sorted(bb,key=lambda m:m.component)
        for a,b in zip(aa,bb):
            id_mapping[a.id]=b.id
            if (a.kind,a.enabled,a.name,a.use_global_monitor,a.time_downsample)!=(b.kind,b.enabled,b.name,b.use_global_monitor,b.time_downsample) or not _close(a.center,b.center,atol=5e-12):
                raise ValueError(a.name+': monitor geometry writeback verification failed.')
            if a.kind=='point' and a.component!=b.component:raise ValueError(a.name+': monitor component verification failed.')
            if a.kind=='field':
                keys=('normal','record_fields','record_poynting','record_flux','dft_precision','spatial_interpolation')
                active=2 if expected.region.dimension=='2d' else 3
                if any(getattr(a,k)!=getattr(b,k) for k in keys) or not _close(a.size[:active],b.size[:active],atol=5e-12) or (a.downsample_xyz or (a.downsample,)*3)!=(b.downsample_xyz or (b.downsample,)*3):
                    raise ValueError(a.name+': monitor output writeback verification failed.')
            sa,sb=expected.resolved_monitor(a).spectrum,actual.resolved_monitor(b).spectrum
            if sa.sampling!='fft' and not _close(frequency_samples(sa),frequency_samples(sb)):
                raise ValueError(a.name+': monitor frequency writeback verification failed.')
            if sa.apodization!=sb.apodization or (sa.apodization not in ('none','hann') and not _close((sa.apodization_center,sa.apodization_time_width),(sb.apodization_center,sb.apodization_time_width))):
                raise ValueError(a.name+': monitor apodization writeback verification failed.')
    return id_mapping
