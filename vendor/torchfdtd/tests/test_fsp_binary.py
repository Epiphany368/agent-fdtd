"""Synthetic format tests contain no vendor material database or binaries."""
import json
import struct

import numpy as np
import pytest

from torchfdtd.fsp_binary import FspDocument, FspFormatError


def u(n):return struct.pack('<I',n)
def string(s):
    b=s.encode()+b'\0';return u(len(b))+b
def mapping(items):
    output=u(len(items))
    for key, (kind, value) in items.items():
        output+=string(key)+u(kind)
        if kind==0:output+=string(value)
        elif kind==1:output+=u(1)+struct.pack('<d',value)
        elif kind==2:output+=u(1)+struct.pack('<i',value)
        elif kind==3:
            a=np.asarray(value);real=not np.iscomplexobj(a)
            output+=u(0)+u(a.size)+u(a.ndim)+b''.join(u(x) for x in a.shape)+u(int(real))+u(1)
            output+=a.real.astype('<f8').tobytes(order='F')
            if not real:output+=a.imag.astype('<f8').tobytes(order='F')
    return output
def node(uid,items,children=()):
    return u(1001)+u(38)+uid.encode()+mapping(items)+u(0)+u(len(children))+b''.join(children)
def transform_block(axes=(-1,-1,-1),angles=(0.,0.,0.),order=2):
    # Three tagged axis codes, a 3x1 angle matrix, then the database/mesh order pair (rotation 3 first).
    block=b''.join(b'\0'+struct.pack('<i',a) for a in axes[::-1])
    block+=u(3)+b'\0'+u(3)+u(2)+u(3)+u(1)+u(1)+u(1)+struct.pack('<3d',*angles[::-1])
    return block,b'\0'+u(1)+b'\0'+u(order)
def legacy_circle(name='circle',enabled=1,index=1.5,x=0.,y=0.,radius=.1e-6,material='{00000000-0000-0000-0000-000000000000}',relative=1):
    # Headerless circle 4/25 body: also what a group script leaves behind.
    body=u(4)+u(25)+struct.pack('<i6d',-1,index,radius,x,y,-.1e-6,.1e-6)
    body+=string(f'{index:g}')+bytes(8)+string(name)
    tail=bytearray(158);tail[-9:-5]=u(enabled)
    rotation,order=transform_block();tail[:5]=b'\0'+u(0);tail[5:14]=b'\1'+struct.pack('<d',radius)
    tail[14:14+len(rotation)]=rotation;tail[-57:-47]=order
    return body+tail+mapping({'materialuuid':(0,material),'use_relative_coordinates':(2,relative),'gridAttributeName':(0,'')})
def scripted_group(name,script,properties,generated,enabled=1,axes=(-1,-1,-1),angles=(0.,0.,0.),x=0.,y=0.,z=0.):
    rotation,_=transform_block(axes,angles);drawing=rotation+bytes(97-len(rotation))
    body=u(13)+u(18)+drawing+b'\0'+struct.pack('<i',-1)+b'\1'+struct.pack('<d',1.)
    body+=b'\x02'+string('1')+b'\0'+bytes(8)+b'\0\x02'+string(name)+b''.join(b'\1'+struct.pack('<d',v) for v in (x,y,z))
    body+=bytes(4)+b'\x02'+string(script)+b'\0'+u(len(properties))
    for key,code,value in properties:body+=b'\x02'+string(key)+b'\0'+u(code)+b'\x02'+string(value)
    tail=bytearray(15);tail[6:10]=u(enabled)
    body+=tail+mapping({'use_relative_coordinates':(2,1)})+u(len(generated))+b''.join(generated)+u(0)
    return u(1000)+u(38)+b'{b1c063bf-9b6d-49e1-b1dc-2c5cddef5d3c}'+body


def fixture():
    monitor=node('{15879b8f-3efb-45dd-a50f-acd1106124bc}',{
        'name':(0,'::model::probe'),'apodizationType':(2,0),'apodizationCenter':(1,20e-15),
        'apodizationWidth':(1,10e-15),'nfreqDesired':(2,11),'userF1':(1,1e14),
        'unrecognized_future_property':(0,'Preserve this text, including Unicode: µm')})
    root=node('{ba475c44-6315-46ba-866f-0d85a97d5267}',{'name':(0,'::model')},[monitor])
    matrix=np.array([[1+4j,2+5j],[3+6j,4+7j],[5+8j,6+9j]])
    return (b'LUMERICAL file version 1.1\0'+struct.pack('<5I',3,10,4,8,1)+bytes(24)
            +b'table of contents\0end table of contents\0Lumerical material data file version 2.0\0'
            +u(1)+mapping({'name':(0,'Synthetic test material'),'coefficients':(3,matrix)})
            +root+mapping({'version':(2,2)})+bytes(12)),matrix


def test_independent_decode_complex_arrays_roundtrip_and_unknown_preservation(tmp_path,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Independent parser must not load the vendor API'))
    raw,expected=fixture();doc=FspDocument(raw)
    np.testing.assert_array_equal(doc.materials[0]['coefficients'].value,expected)
    assert len(doc.nodes())==2
    json.dumps(doc.inspect(),allow_nan=False)
    doc.save(tmp_path/'copy.fsp');assert (tmp_path/'copy.fsp').read_bytes()==raw
    with pytest.raises(FileExistsError):doc.save(tmp_path/'copy.fsp')
    target=doc.nodes()[1]
    edited=doc.with_monitor_edits([{'record_offset':target.start,'property':'apodizationCenter','value':35e-15}])
    assert edited.nodes()[1].properties['apodizationCenter'].value==35e-15
    assert edited.nodes()[1].properties['unrecognized_future_property'].value==target.properties['unrecognized_future_property'].value
    a,b=edited.changed_ranges[0];assert edited.data[:a]==raw[:a] and edited.data[b:]==raw[b:]
    assert doc.data==raw


@pytest.mark.parametrize('cut',[1,26,60,110,156,-1,-5,-20])
def test_truncated_files_fail_before_exposing_a_document(cut):
    raw,_=fixture()
    with pytest.raises(FspFormatError):FspDocument(raw[:cut])


def test_unknown_headers_and_invalid_edits_are_rejected():
    raw,_=fixture();doc=FspDocument(raw);offset=doc.nodes()[1].start
    with pytest.raises(FspFormatError):FspDocument(raw.replace(b'version 1.1',b'version 9.9',1))
    changed=bytearray(raw);changed[27:31]=u(4)
    with pytest.raises(FspFormatError):FspDocument(changed)
    for prop,value in [('apodizationWidth',0),('apodizationType',9),('nfreqDesired',3.5),
                       ('name','rename'),('apodizationCenter',float('nan')),('userF1',-1)]:
        with pytest.raises(ValueError):doc.with_monitor_edits([{'record_offset':offset,'property':prop,'value':value}])


def test_excessive_matrix_dimensions_are_rejected():
    raw,_=fixture();needle=string('coefficients')+u(3)+u(0)
    start=raw.index(needle)+len(needle)
    bad=raw[:start]+u(2**32-1)+raw[start+4:]
    with pytest.raises(FspFormatError):FspDocument(bad)


def test_legacy_enabled_flag_is_independent_of_expression_metadata():
    raw,_=fixture();base=FspDocument(raw)
    # The expression metadata preceding a legacy name is not its enabled flag.
    for enabled in (0,1):
        body=u(8)+u(25)+struct.pack('<i5d',-1,2.5,.63e-6,.1e-6,.2e-6,.3e-6)
        body+=string('2.5')+u(1)+u(0)+string('sphere')
        tail=bytearray(167);tail[-9:-5]=u(enabled)
        body+=tail+mapping({'materialuuid':(0,'{00000000-0000-0000-0000-000000000000}'),
                            'use_relative_coordinates':(2,1)})+u(0)
        legacy=u(1000)+u(38)+b'{23046316-141b-4111-aa2f-9e18de790b6c}'+body
        root=node('{ba475c44-6315-46ba-866f-0d85a97d5267}',{'name':(0,'::model')},[legacy])
        blob=raw[:base.root.start]+root+raw[base.root.end:]
        assert FspDocument(blob).nodes()[1].legacy['enabled'] is bool(enabled)


def test_scripted_structure_groups_and_sweep_stores_parse_and_roundtrip():
    raw,_=fixture();base=FspDocument(raw)
    circles=[legacy_circle(enabled=i%2) for i in range(3)]
    group=scripted_group('array','deleteall;\naddcircle;\n',[('index',0,'1.5'),('nx',0,'3')],circles)
    plain=scripted_group('empty','',[],[])  # the previously observed fixed 30-byte tail
    root=node('{ba475c44-6315-46ba-866f-0d85a97d5267}',{'name':(0,'::model')},[group,plain])
    sweep=b'opaque sweep store bytes'
    blob=raw[:base.root.start]+root+raw[base.root.end:-12]+u(0)+u(0)+u(1)+sweep
    doc=FspDocument(blob);array,empty=doc.root.children
    assert array.legacy['script']=='deleteall;\naddcircle;\n' and array.legacy['user_properties']==[('index',0,'1.5'),('nx',0,'3')]
    assert [c.name for c in array.children]==['circle']*3 and [c.legacy['enabled'] for c in array.children]==[False,True,False]
    assert all(c.legacy['script_generated'] and c.uid=='{921e6d99-3bcb-4063-b513-216a3bb757d9}' and not c.children for c in array.children)
    assert empty.legacy['script']=='' and empty.legacy['user_properties']==[] and empty.children==[] and empty.legacy['enabled']
    assert doc.sweep_records==1 and doc.trailing==sweep and len(doc.nodes())==6
    assert array.legacy['transform_metadata_decoded'] and array.legacy['rotation_axes']==['none']*3 and array.legacy['rotation_angles']==[0.,0.,0.]
    assert all(c.legacy['transform_metadata_decoded'] and c.legacy['mesh_order']==2 for c in array.children)
    twisted=FspDocument(raw[:base.root.start]+node('{ba475c44-6315-46ba-866f-0d85a97d5267}',{'name':(0,'::model')},
                        [scripted_group('twist','',[],[],axes=(2,-1,-1),angles=(90.,0.,0.),z=.2e-6)])+raw[base.root.end:]).root.children[0]
    assert twisted.legacy['rotation_axes']==['z','none','none'] and twisted.legacy['rotation_angles']==[90.,0.,0.] and twisted.legacy['z']==.2e-6
    json.dumps(doc.inspect(),allow_nan=False)
    assert doc.data==blob
    with pytest.raises(FspFormatError):FspDocument(raw[:-12]+u(0)+u(0)+u(0)+b'x')
    with pytest.raises(FspFormatError):FspDocument(raw[:-12]+u(0)+u(0)+u(2))
    with pytest.raises(FspFormatError):FspDocument(raw[:-12]+u(1)+u(0)+u(0))
