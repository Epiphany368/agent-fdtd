"""Top-level primitive records and loss-preserving scene-list reconstruction.

New records use an authored layout for the explicitly decoded legacy versions.
Their reserved drawing bytes are zero defaults, not copied vendor assets.
External reader acceptance is unverified. Existing records retain opaque data.
"""
from __future__ import annotations

import struct
from bisect import bisect_left
import numpy as np

from .fsp_binary import FspDocument
from .fsp_native import ZERO_UUID, convert_material, identity, SOURCE_CLASSES, TIME, DFT


def primitive_record(shape, material, origin, material_records):
    from .fsp_geometry import _matrix, _string, _rotation, _material_physics
    from .fsp_settings import _value
    formats={
        'rectangle':(6,32,'{75954e61-0067-4a0e-8d16-1ee8d371e892}',192),
        'sphere':(8,25,'{23046316-141b-4111-aa2f-9e18de790b6c}',167),
        'circle':(4,25,'{921e6d99-3bcb-4063-b513-216a3bb757d9}',158),
        'ring':(5,26,'{b4a87699-109a-4252-b2c4-06be606f82f3}',144),
        'polygon':(11,22,'{49651cff-e643-43b3-b70d-0b6846faa79c}',189),
    }
    code,version,uid,tail_size=formats[shape.kind]
    material_uid=ZERO_UUID
    if material.model!='dielectric':
        for entry in material_records:
            if 'materialuuid' not in entry:continue
            candidate=entry['materialuuid'].value
            try:existing,_=convert_material(material_records,candidate)
            except ValueError:continue
            if _material_physics(existing)==_material_physics(material):
                material_uid=candidate;break
        else:raise ValueError('New FSP objects require a constant dielectric or an unchanged supported material already in the original database.')
    index=material.index if material.model=='dielectric' else 1.
    u=lambda v:struct.pack('<I',v)
    d=lambda v:struct.pack('<d',v)
    center=np.asarray(shape.center)*1e-6+np.asarray(origin)
    size=np.asarray(shape.size)*1e-6
    tail=bytearray(tail_size);pos=0
    if shape.kind in ('circle','sphere'):
        radii=b'\0'+u(int(shape.make_ellipsoid))+b'\1'+d(shape.radius_2*1e-6)
        if shape.kind=='sphere':radii+=b'\1'+d(shape.radius_3*1e-6)
        tail[:len(radii)]=radii;pos=len(radii)
    axes,angles=_rotation(shape)
    tail[pos:pos+15]=b''.join(b'\0'+struct.pack('<i',{'none':-1,'x':0,'y':1,'z':2}[a]) for a in axes[::-1])
    tail[pos+15:pos+68]=_matrix(np.asarray(angles[::-1]).reshape(3,1))
    order=88 if shape.kind=='polygon' else tail_size-56
    tail[order:order+4]=u(0)  # object mesh priority, independent of database
    tail[order+5:order+9]=u(shape.mesh_order)
    if shape.kind=='polygon':tail[97:142]=_matrix(center[:2].reshape(2,1))
    tail[-9:-5]=u(int(shape.enabled))
    if shape.kind=='polygon':
        vertices=(np.asarray(shape.vertices)*1e-6+center[:2]).T
        body=u(code)+u(version)+b'\0'+struct.pack('<i',-1)+b'\1'+d(index)+_matrix(vertices)
        body+=b'\1'+d(center[2]-size[2]/2)+b'\1'+d(center[2]+size[2]/2)
        body+=b'\2'+_string(format(index,'.17g'))+b'\0'+bytes(8)+b'\0\2'+_string(shape.name)
    else:
        values={
            'rectangle':(center[1]-size[1]/2,center[0]-size[0]/2,center[1]+size[1]/2,center[0]+size[0]/2,center[2]-size[2]/2,center[2]+size[2]/2),
            'sphere':(shape.radius*1e-6,*center),
            'circle':(shape.radius*1e-6,*center[:2],center[2]-size[2]/2,center[2]+size[2]/2),
            'ring':(shape.inner_radius*1e-6,shape.radius*1e-6,*center[:2],shape.theta_start,shape.theta_stop,center[2]-size[2]/2,center[2]+size[2]/2),
        }[shape.kind]
        body=u(code)+u(version)+struct.pack('<i',-1)+d(index)+b''.join(d(v) for v in values)
        body+=_string(format(index,'.17g'))+bytes(8)+_string(shape.name)
    props=dict(materialuuid=material_uid,use_relative_coordinates=1,gridAttributeName='')
    if shape.kind=='ring':props.update(mIsEllipse=int(shape.make_ellipsoid),ro2=shape.radius_2*1e-6,ri2=shape.inner_radius_2*1e-6)
    mapping=u(len(props))+b''.join(_string(k)+_value(v) for k,v in props.items())
    return u(1000)+u(38)+uid.encode('ascii')+body+tail+mapping+u(0)


def rewrite_scene(document, patches, requested, new_records, originals):
    """Reassemble top-level records, preserving every untouched retained slice.

    Each edited family occupies its first original position in requested order.
    Other records retain their relative order. Missing families are appended.
    Original offsets are not IDs in the exported file, so all retained record
    offsets are mapped explicitly.
    """
    root=document.root;start=root.child_count_offset+4
    def family(node):
        if node.legacy:return None if node.legacy.get('kind')==13 else 'structures'  # groups keep their place
        if node.uid in SOURCE_CLASSES:return 'sources'
        if node.uid in (TIME,DFT):return 'monitors'
        return None
    order=[];emitted=set()
    for n in root.children:
        kind=family(n)
        if kind is None:order.append(n)
        elif kind not in emitted:order.extend(requested[kind]);emitted.add(kind)
    for kind in ('structures','sources','monitors'):
        if kind not in emitted:order.extend(requested[kind])
    if len(order)>10000:raise ValueError('FSP object count exceeds the supported limit.')
    chunks=[];cursor=0;preserved=[];detail=[];locations=[]
    ordered=sorted(patches);starts=[p[0] for p in ordered];applied=set()
    def raw(data,lo=None,hi=None,object_id=None,key=None):
        nonlocal cursor
        out=[cursor,cursor+len(data)]
        chunks.append(data);cursor=out[1]
        if key is None and lo is not None:
            preserved.append(dict(original_range=[lo,hi],output_range=out))
        else:detail.append(dict(object_id=object_id,property=key,original_range=None if lo is None else [lo,hi],output_range=out))
    def segment(lo,hi):
        at=lo
        for index in range(bisect_left(starts,lo),bisect_left(starts,hi)):
            a,b,data,obj,key=ordered[index]
            if a<at or b>hi:raise ValueError('Overlapping or cross-record FSP edit.')
            if index in applied:raise ValueError('An FSP edit would be applied more than once.')
            applied.add(index)
            raw(document.data[at:a],at,a)
            raw(data,a,b,obj,key);at=b
        raw(document.data[at:hi],at,hi)
    segment(0,root.child_count_offset)
    raw(struct.pack('<I',len(order)),root.child_count_offset,start,'model','object count')
    children_start=cursor
    for item in order:
        shape_id=item if isinstance(item,str) else None
        node=originals.get(item) if shape_id is not None else item
        begin=cursor
        if node is None:
            raw(new_records[shape_id],object_id=shape_id,key='new scene record')
        else:segment(node.start,node.end)
        locations.append(dict(input_id=shape_id,original=None if node is None else node.start,output=begin))
    children_end=cursor
    segment(root.end,len(document.data))
    if len(applied)!=len(ordered):raise ValueError('An FSP edit falls outside the retained record ranges.')
    result=FspDocument(b''.join(chunks))
    if len(result.root.children)!=len(order):raise ValueError('FSP object count did not reparse correctly.')
    for loc,node in zip(locations,result.root.children):
        if node.start!=loc['output']:raise ValueError('FSP record offset verification failed.')
        loc['output_id']=identity(node)['id']
    for part in preserved:
        a,b=part['original_range'];c,d=part['output_range']
        if document.data[a:b]!=result.data[c:d]:raise ValueError('An untouched FSP byte segment changed.')
    edits=[dict(object_id='model',property='object count',original_range=[root.child_count_offset,start],output_range=[children_start-4,children_start]),
           dict(object_id='model',property='scene lists',original_range=[start,root.end],output_range=[children_start,children_end])]
    return result,dict(edits=edits,property_edits=detail,preserved_segments=preserved,object_offsets=locations)
