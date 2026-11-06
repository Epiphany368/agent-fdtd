"""Independent geometry writeback into an immutable, recognized FSP layout.

Existing-object edits splice decoded fields and preserve untouched bytes.
Scene mode also reconstructs the primitive list with explicit ID/range maps.
The output must reparse and convert before it is returned. New records use
authored metadata defaults. This is not a general FSP serializer.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from .fsp_binary import FspDocument
from .fsp_native import ZERO_UUID, convert_fsp, identity
from .geometry import rotation_matrix
from .models import Project


def _string(value):
    if '\0' in value:raise ValueError('FSP strings cannot contain a null character.')
    raw=value.encode('utf-8')+b'\0'
    return struct.pack('<I',len(raw))+raw


def _matrix(value):
    a=np.asarray(value,dtype='<f8')
    return (struct.pack('<I',3)+b'\0'+struct.pack('<II',a.size,a.ndim)
            +struct.pack('<'+'I'*a.ndim,*a.shape)+struct.pack('<II',1,1)+a.tobytes(order='F'))


def _encode(codec,value):
    if codec=='string':return _string(value)
    if codec=='matrix':return _matrix(value)
    if codec=='axis':value={'none':-1,'x':0,'y':1,'z':2}[value];codec='i32'
    return struct.pack('<d' if codec=='f64' else '<i',value if codec=='f64' else int(value))


def _material_physics(m):
    return (m.model,m.instantaneous_epsilon,m.oscillators)


def _rotation(shape):
    """Serialize three fixed-world rotations, folding the legacy fourth one."""
    if shape.rotation==0:return shape.rotation_axes,shape.rotation_angles
    r=rotation_matrix(shape)
    y=math.atan2(-r[2,0],math.hypot(r[0,0],r[1,0]))
    if math.hypot(r[0,0],r[1,0])>1e-12:
        x=math.atan2(r[2,1],r[2,2]);z=math.atan2(r[1,0],r[0,0])
    else:
        x=math.atan2(-r[1,2],r[1,1]);z=0.
    return ('x','y','z'),tuple(math.degrees(a) for a in (x,y,z))


def _same_geometry(a,b):
    if (a.kind,a.enabled,a.mesh_order,a.name)!=(b.kind,b.enabled,b.mesh_order,b.name):return False
    if not np.allclose(a.center,b.center,rtol=1e-12,atol=5e-13):return False
    if not np.allclose(rotation_matrix(a),rotation_matrix(b),rtol=0,atol=3e-12):return False
    def close(x,y):return np.shape(x)==np.shape(y) and np.allclose(x,y,rtol=1e-12,atol=5e-13)
    if a.kind=='rectangle':return close(a.size,b.size)
    if a.kind=='polygon':return close(a.vertices,b.vertices) and close(a.size[2],b.size[2])
    if a.make_ellipsoid!=b.make_ellipsoid or not close(a.radius,b.radius):return False
    if a.kind!='sphere' and not close(a.size[2],b.size[2]):return False
    if a.make_ellipsoid and not close(a.radius_2,b.radius_2):return False
    if a.kind=='sphere' and a.make_ellipsoid and not close(a.radius_3,b.radius_3):return False
    if a.kind=='ring':
        return (close((a.inner_radius,a.theta_start,a.theta_stop),(b.inner_radius,b.theta_start,b.theta_stop))
                and (not a.make_ellipsoid or close(a.inner_radius_2,b.inner_radius_2)))
    return True


def write_fsp_geometry(document: FspDocument, project: Project):
    """Return ``(new_document, report)`` without writing files or loading an API.

    ``project`` must descend from this exact document's native conversion.
    Lengths use Project's micrometres. Existing primitive geometry, name,
    enabled state, priority and constant dielectric assignments are editable.
    Scene topology and nongeometry physical changes are rejected. Native run
    and display controls are retained only in JSON and listed in the report.
    """
    return _write_fsp(document,project,settings=False)


def write_fsp_scene(document: FspDocument, project: Project):
    """Write primitive scene lists, geometry and supported simulation settings.

    The immutable source fingerprint is required. Unsupported changes fail
    before any file is returned. See the report for native-only controls.
    """
    return _write_fsp(document,project,settings=True)


def _write_fsp(document,project,*,settings):
    project=Project.model_validate(project.model_dump())
    provenance=project.import_provenance
    if provenance is None or provenance.source_sha256!=document.fingerprint():
        raise ValueError('Geometry export requires the exact original FSP fingerprint from native import.')
    conversion=convert_fsp(document)
    base=conversion.project
    if base is None:raise ValueError('Original FSP has unsupported native mappings: '+str(conversion.issues))
    if tuple(provenance.origin_m)!=tuple(conversion.origin_m):raise ValueError('FSP coordinate origin was changed.')
    if any(s.holes for s in project.structures):raise ValueError('FSP polygons are single contours; polygon holes cannot be written.')
    native_only=[];instruments=None
    for material in project.materials:
        if material.model=='tensor':
            native_only.append(f'material {material.name}: Cartesian tensor coefficients (retain in native JSON)')
        if material.samples is not None or material.fit_band_um is not None:
            native_only.append(f'material {material.name}: optical samples and fit metadata (retain in native JSON)')
    if settings:
        from .fsp_settings import plan_settings, verify_settings
        settings_patches,settings_native_only,instruments=plan_settings(document,base,project,conversion)
        native_only.extend(settings_native_only)
    allowed_region={'backend','cuda_kernel','cuda_monitor_kernel','execution_mode','tiling','resident_cell_limit','precision',
                    'snapshot_interval','field','slice_axis','slice_position','complex_display'}
    region_values=project.region.model_dump();base_values=base.region.model_dump()
    # A field written only when set (an optional cap) can appear on either side.
    for key in [*base_values,*(k for k in region_values if k not in base_values)]:
        if base_values.get(key)==region_values.get(key):continue
        if not settings:
            if key not in allowed_region:raise ValueError(f'Geometry export cannot write region.{key}. Save the full native JSON project.')
            native_only.append('region.'+key)
    for key in ('sources','monitors','global_source','global_monitor'):
        if not settings and getattr(base,key)!=getattr(project,key):
            raise ValueError(f'Geometry export cannot write changes to {key}. Save the full native JSON project.')
    if project.name!=base.name:native_only.append('project.name')
    originals={s.id:s for s in base.structures}
    structure_ids=[s.id for s in project.structures]
    structure_changed=structure_ids!=list(originals)
    topology_changed=structure_changed or bool(instruments and (instruments['source_changed'] or instruments['monitor_changed']))
    if topology_changed and not settings:
        raise ValueError('Adding, deleting or reordering FSP objects is not mapped by the geometry writer yet.')
    materials={m.name:m for m in project.materials};base_materials={m.name:m for m in base.materials}
    nodes={identity(n)['id']:n for n in document.root.children if n.legacy}
    # Objects a group script generated are imported for execution only; they are never written back.
    generated={identity(c)['id']:c for n in document.root.children if n.legacy.get('kind')==13 for c in n.children}
    if any(sid in generated and (topology_changed or project.structures[i]!=originals.get(sid)) for i,sid in enumerate(structure_ids)):
        raise ValueError('Script-generated group objects are not written back to FSP. Keep them unchanged and edit the group script in Lumerical, or remove them from the export.')
    patches=list(settings_patches) if settings else [];edits=[];reparameterized=[]
    new_records=dict(instruments['new_records']) if instruments else {};new_primitives=[]
    for shape in project.structures:
        if shape.id not in originals:
            from .fsp_objects import primitive_record
            new_records[shape.id]=primitive_record(shape,materials[shape.material],conversion.origin_m,document.materials)
            new_primitives.append(shape.id)
            native_only.append(shape.id+'.material label/color and inactive primitive controls')
            if shape.rotation!=0:reparameterized.append(shape.id)
            continue
        if shape.id in generated:continue
        previous=originals[shape.id];node=nodes[shape.id]
        if shape.kind!=previous.kind:raise ValueError('Changing the FSP primitive type is not mapped yet.')
        changes={k for k in type(shape).model_fields if getattr(shape,k)!=getattr(previous,k)}
        pending={};generic={}
        def put(key,value):pending[key]=value
        center=np.asarray(shape.center)*1e-6+np.asarray(conversion.origin_m)
        if 'name' in changes:put('name',shape.name)
        if 'enabled' in changes:put('enabled',int(shape.enabled))
        if 'mesh_order' in changes:
            put('mesh order',shape.mesh_order);put('use database order',0)
        if changes & {'rotation','rotation_axes','rotation_angles'}:
            axes,angles=_rotation(shape)
            for i in range(3):put(f'rotation axis {i+1}',axes[i]);put(f'rotation {i+1}',angles[i])
            if shape.rotation!=0:reparameterized.append(shape.id)
        if shape.kind=='rectangle':
            for i,a in enumerate('xyz'):
                if shape.center[i]!=previous.center[i] or shape.size[i]!=previous.size[i]:
                    put(a+' min',center[i]-shape.size[i]*.5e-6);put(a+' max',center[i]+shape.size[i]*.5e-6)
        else:
            if shape.kind=='polygon':
                if shape.center[:2]!=previous.center[:2]:put('polygon pivot',center[:2].reshape(2,1))
                if shape.center[:2]!=previous.center[:2] or shape.vertices!=previous.vertices:
                    put('vertices_global',(np.asarray(shape.vertices)*1e-6+center[:2]).T)
            else:
                for i,a in enumerate('xyz' if shape.kind=='sphere' else 'xy'):
                    if shape.center[i]!=previous.center[i]:put(a,center[i])
                if 'radius' in changes:put('outer radius' if shape.kind=='ring' else 'radius',shape.radius*1e-6)
            if shape.kind!='sphere' and (shape.center[2]!=previous.center[2] or shape.size[2]!=previous.size[2]):
                put('z min',center[2]-shape.size[2]*.5e-6);put('z max',center[2]+shape.size[2]*.5e-6)
        if shape.kind in ('circle','sphere','ring'):
            ring=shape.kind=='ring'
            if 'make_ellipsoid' in changes:
                if ring:generic['mIsEllipse']=int(shape.make_ellipsoid)
                else:put('make ellipsoid',int(shape.make_ellipsoid))
            if shape.make_ellipsoid or 'radius_2' in changes:
                if 'radius_2' in changes or 'make_ellipsoid' in changes:
                    if ring:generic['ro2']=shape.radius_2*1e-6
                    else:put('radius 2',shape.radius_2*1e-6)
            if shape.kind=='sphere' and ('radius_3' in changes or ('make_ellipsoid' in changes and shape.make_ellipsoid)):
                put('radius 3',shape.radius_3*1e-6)
            if ring:
                if 'inner_radius' in changes:put('inner radius',shape.inner_radius*1e-6)
                for key in ('theta_start','theta_stop'):
                    if key in changes:put(key.replace('_',' '),getattr(shape,key))
                if 'inner_radius_2' in changes or ('make_ellipsoid' in changes and shape.make_ellipsoid):generic['ri2']=shape.inner_radius_2*1e-6
        desired=materials[shape.material];old=base_materials[previous.material]
        if _material_physics(desired)!=_material_physics(old):
            if desired.model!='dielectric':
                raise ValueError(f'{shape.name}: writing new dispersive material coefficients is not mapped yet.')
            put('index',desired.index);put('index_expression',format(desired.index,'.17g'))
            generic['materialuuid']=ZERO_UUID
            # Replacing a database material must preserve its resolved priority.
            put('mesh order',shape.mesh_order);put('use database order',0)
        if desired.name!=old.name or desired.color!=old.color:native_only.append(shape.id+'.material label/color')
        # Inactive shape controls do not define this primitive's geometry.
        active={'name','enabled','mesh_order','material','center','rotation','rotation_axes','rotation_angles','kind','id'}
        if shape.kind=='rectangle':active.add('size')
        elif shape.kind=='polygon':active.update(('size','vertices'))
        else:
            active.update(('radius','make_ellipsoid','radius_2'))
            if shape.kind=='sphere':active.add('radius_3')
            else:active.add('size')
            if shape.kind=='ring':active.update(('inner_radius','inner_radius_2','theta_start','theta_stop'))
        native_only.extend(shape.id+'.'+k for k in sorted(changes-active))
        if shape.kind in ('polygon','circle','ring') and shape.size[:2]!=previous.size[:2]:
            native_only.append(shape.id+'.size[:2] (derived display spans)')
        for key,val in pending.items():
            if key not in node.legacy_fields:raise ValueError(f'{shape.name}: {key} has no decoded writable field.')
            start,end,codec=node.legacy_fields[key];raw=_encode(codec,val)
            if raw!=document.data[start:end]:patches.append((start,end,raw,shape.id,key))
        additions=[]
        for key,val in generic.items():
            stored=node.properties.get(key)
            kind=0 if isinstance(val,str) else 2 if isinstance(val,int) else 1
            encoded=struct.pack('<I',kind)+(_string(val) if kind==0 else struct.pack('<I',1)+_encode('i32' if kind==2 else 'f64',val))
            if stored is None:additions.append(_string(key)+encoded)
            elif not np.array_equal(stored.value,val):patches.append((stored.start,stored.end,encoded,shape.id,key))
        if additions:
            start,end,_=node.legacy_fields['__properties__']
            patches.append((start,start+4,struct.pack('<I',len(node.properties)+len(additions)),shape.id,'property count'))
            patches.append((end,end,b''.join(additions),shape.id,'ellipse properties'))
    topology_report={}
    if topology_changed:
        from .fsp_objects import rewrite_scene
        requested=dict(structures=structure_ids,sources=instruments['sources'],monitors=instruments['monitors'])
        retained_nodes={key:node for key,node in nodes.items() if key in structure_ids}
        retained_nodes.update(instruments['nodes'])
        result,topology_report=rewrite_scene(document,patches,requested,new_records,retained_nodes)
        edits=topology_report['edits']
    else:
        chunks=[];cursor=0;new_cursor=0
        for start,end,raw,object_id,key in sorted(patches):
            if start<cursor:raise ValueError('Overlapping geometry edits.')
            chunks.append(document.data[cursor:start]);new_cursor+=start-cursor
            edits.append(dict(object_id=object_id,property=key,original_range=[start,end],output_range=[new_cursor,new_cursor+len(raw)]))
            chunks.append(raw);new_cursor+=len(raw);cursor=end
        chunks.append(document.data[cursor:]);result=FspDocument(b''.join(chunks))
    reread=convert_fsp(result)
    if reread.project is None:raise ValueError('Edited FSP cannot be converted: '+str(reread.issues))
    after=reread.project
    id_mapping=verify_settings(project,after,conversion,reread,instruments['monitor_groups']) if settings else {}
    if len(after.structures)!=len(project.structures):raise ValueError('Geometry export changed the object count.')
    for expected,actual in zip(project.structures,after.structures):
        id_mapping[expected.id]=actual.id
        if not _same_geometry(expected,actual):raise ValueError(f'{expected.name}: geometry writeback verification failed.')
        actual_material=next(m for m in after.materials if m.name==actual.material)
        if _material_physics(materials[expected.material])!=_material_physics(actual_material):
            raise ValueError(f'{expected.name}: material writeback verification failed.')
    if topology_changed:
        retained={n.start:n for n in document.root.children}
        for item,node in zip(topology_report['object_offsets'],result.root.children):
            if item['original'] is not None and retained[item['original']].uid!=node.uid:
                raise ValueError('A retained object class changed during export.')
    elif [n.uid for n in document.nodes()]!=[n.uid for n in result.nodes()]:raise ValueError('Object classes changed during export.')
    active=2 if project.region.dimension=='2d' else 3
    grid_changed=any(np.shape(a)!=np.shape(b) or not np.allclose(a,b,rtol=2e-11,atol=5e-12)
                     for a,b in zip(base.region.mesh_nodes[:active],project.region.mesh_nodes[:active]))
    instrument_changed=bool(instruments and (instruments['source_changed'] or instruments['monitor_changed']))
    return result,dict(requires_lumerical=False,scope=('scene lists and supported settings' if instrument_changed else 'primitive scene and supported settings' if topology_changed else 'existing objects and supported settings') if settings else 'existing primitive geometry',
        source_sha256=document.fingerprint(),output_sha256=result.fingerprint(),
        byte_identical=not edits,edits=edits,native_only_settings=native_only,
        rotation_reparameterized=reparameterized,
        id_mapping=id_mapping,
        structure_list=dict(changed=structure_changed,added=new_primitives,
                            removed=[key for key in originals if key not in structure_ids],
                            input_order=structure_ids,output_order=[s.id for s in after.structures],
                            new_record_metadata='authored default drawing fields' if new_primitives else None,
                            external_reader_verified=False),
        source_list=None if instruments is None else dict(changed=instruments['source_changed'],
            added=instruments['source_added'],removed=instruments['source_removed'],
            input_order=[s.id for s in project.sources],output_order=[s.id for s in after.sources]),
        monitor_list=None if instruments is None else dict(changed=instruments['monitor_changed'],
            added=instruments['monitor_added'],removed=instruments['monitor_removed'],
            input_order=[m.id for m in project.monitors],output_order=[m.id for m in after.monitors],
            groups=instruments['monitor_groups'],splits=instruments['splits']),
        property_edits=topology_report.get('property_edits',edits),
        preserved_segments=topology_report.get('preserved_segments'),
        mesh_export=dict(nodes_changed=grid_changed,shape_before=base.region.shape,shape_after=after.region.shape,
                         requested_size_um=project.region.size,actual_size_um=project.region.actual_size,
                         stored_cfl=after.region.courant_factor,time_step_s=after.region.time_step,
                         material_sampling=after.region.material_sampling,
                         external_remeshing_verified=False),
        object_offsets=topology_report.get('object_offsets',[dict(original=a.start,output=b.start) for a,b in zip(document.nodes(),result.nodes())]),
        limitations=[('Uniform target meshes can update spacing, bounds and saved nodes. External remeshing is not verified.' if settings else
                      'Saved mesh nodes remain fixed. Refinement after geometry edits is a separate native operation.'),
                     ('In-place object class changes, groups, graded/explicit mesh regeneration and new dispersive coefficients are not written. New record metadata uses authored defaults and external reader acceptance is unverified. Split monitor templates retain opaque properties without interpreting their references.' if settings else
                      'Object addition/deletion/type changes, groups, nongeometry settings and new dispersive coefficients are not written.')])
