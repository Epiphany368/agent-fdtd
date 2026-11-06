"""Author-generated FSP layouts only, no commercial files or calculations."""
import struct
import subprocess
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_geometry import write_fsp_geometry, _rotation
from torchfdtd.fsp_native import convert_fsp, ZERO_UUID
from torchfdtd.geometry import rotation_matrix
from torchfdtd.models import Material, Structure
from torchfdtd.server import create_app
from torchfdtd.solver import Simulation, voxelize
from test_fsp_native import fixture, items
from test_fsp_binary import mapping, string, u


def matrix(a):
    a=np.asarray(a,dtype='<f8')
    return u(3)+b'\0'+u(a.size)+u(a.ndim)+b''.join(u(n) for n in a.shape)+u(1)+u(1)+a.tobytes(order='F')


def shape_fixture(kind='polygon', *, closed=False, ellipse=True, omit_ellipse_properties=False):
    """Independent fixture authorship, lengths below are SI metres."""
    raw=fixture();doc=FspDocument(raw);old=next(n for n in doc.nodes() if n.legacy)
    code,version,uid,size={
      'rectangle':(6,32,'{75954e61-0067-4a0e-8d16-1ee8d371e892}',192),
      'sphere':(8,25,'{23046316-141b-4111-aa2f-9e18de790b6c}',167),
      'circle':(4,25,'{921e6d99-3bcb-4063-b513-216a3bb757d9}',158),
      'ring':(5,26,'{b4a87699-109a-4252-b2c4-06be606f82f3}',144),
      'polygon':(11,22,'{49651cff-e643-43b3-b70d-0b6846faa79c}',189)}[kind]
    center=np.array([.17,-.23,.11])*1e-6
    tail=bytearray(size);pos=0
    if kind in ('sphere','circle'):
        radii=b'\0'+u(int(ellipse))+b'\1'+struct.pack('<d',.19e-6)
        if kind=='sphere':radii+=b'\1'+struct.pack('<d',.27e-6)
        tail[:len(radii)]=radii;pos=len(radii)
    # File order is reversed relative to applied fixed-world x,y,z rotations.
    tail[pos:pos+15]=b''.join(b'\0'+struct.pack('<i',v) for v in (2,1,0))
    tail[pos+15:pos+68]=matrix(np.array([-29.,31.,17.]).reshape(3,1))
    priority=88 if kind=='polygon' else size-56
    tail[priority:priority+4]=u(1);tail[priority+5:priority+9]=u(2)
    if kind=='polygon':tail[97:142]=matrix(center[:2].reshape(2,1))
    tail[-9:-5]=u(1)
    if kind=='polygon':
        vertices=np.array([[0,0],[.6,0],[.6,.2],[.2,.2],[.2,.5],[0,.5]])*1e-6+center[:2]
        if closed:vertices=np.vstack([vertices,vertices[0]])
        body=u(code)+u(version)+b'\0'+struct.pack('<i',-1)+b'\1'+struct.pack('<d',2.)+matrix(vertices.T)
        body+=b'\1'+struct.pack('<d',center[2]-.2e-6)+b'\1'+struct.pack('<d',center[2]+.2e-6)
        body+=b'\2'+string('2')+b'\0'+bytes(8)+b'\0\2'+string(kind)
    else:
        fields={'rectangle':(center[1]-.2e-6,center[0]-.4e-6,center[1]+.2e-6,center[0]+.4e-6,center[2]-.3e-6,center[2]+.3e-6),
            'sphere':(.4e-6,*center),'circle':(.4e-6,*center[:2],center[2]-.2e-6,center[2]+.2e-6),
            'ring':(.12e-6,.4e-6,*center[:2],315.,110.,center[2]-.2e-6,center[2]+.2e-6)}[kind]
        body=u(code)+u(version)+struct.pack('<i'+'d'*(len(fields)+1),-1,2.,*fields)+string('2')+bytes(8)+string(kind)
    props=dict(materialuuid=ZERO_UUID,use_relative_coordinates=1,gridAttributeName='',
               unknown_unicode='preserve μ 和 광학',unknown_matrix=np.array([[1.+2j,3.-4j]]))
    if kind=='ring' and not omit_ellipse_properties:props.update(mIsEllipse=int(ellipse),ro2=.25e-6,ri2=.08e-6)
    body+=tail+mapping(items(props))+u(0)
    return raw[:old.start]+u(1000)+u(38)+uid.encode()+body+raw[old.end:]


def imported(raw):
    doc=FspDocument(raw);report=convert_fsp(doc,backend='cpu')
    assert report.project is not None,report.issues
    return doc,report.project


def assert_untouched(original,output,report):
    a=b=0
    for edit in report['edits']:
        lo,hi=edit['original_range'];newlo,newhi=edit['output_range']
        assert original[a:lo]==output[b:newlo]
        a,b=hi,newhi
    assert original[a:]==output[b:]


@pytest.mark.parametrize('kind',['rectangle','sphere','circle','ring','polygon'])
def test_import_rotation_and_byte_identical_noop(kind,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Commercial runtime loaded'))
    doc,p=imported(shape_fixture(kind));s=p.structures[0]
    np.testing.assert_allclose(s.center,[.17,-.23,.11],atol=1e-15)
    np.testing.assert_allclose(rotation_matrix(s),Rotation.from_euler('xyz',[17,31,-29],degrees=True).as_matrix(),atol=3e-16)
    if kind=='polygon':
        np.testing.assert_allclose(s.vertices,[[0,0],[.6,0],[.6,.2],[.2,.2],[.2,.5],[0,.5]],atol=1e-15)
    if kind=='ring':assert (s.theta_start,s.theta_stop,s.radius_2,s.inner_radius_2)==(315.,110.,.25,.08)
    out,report=write_fsp_geometry(doc,p)
    assert report['byte_identical'] and report['edits']==[] and out.data==doc.data


@pytest.mark.parametrize('kind',['rectangle','sphere','circle','ring','polygon'])
def test_resize_rotate_rename_material_priority_preserves_all_other_bytes(kind):
    doc,p=imported(shape_fixture(kind));s=p.structures[0]
    s.name='길이가 다른 새 구조 μ';s.center=(-.19,.21,-.13);s.rotation_axes=('z','x','y');s.rotation_angles=(49,-33,27)
    s.size=(.66,.36,.48);s.radius=.38;s.radius_2=.24;s.radius_3=.17;s.mesh_order=7
    if kind=='polygon':s.vertices=((-0.3,-.2),(.3,-.2),(.31,.1),(0,.1),(0,.3),(-.3,.3),(-.33,0))
    if kind=='ring':s.inner_radius=.11;s.inner_radius_2=.07;s.theta_start=-25;s.theta_stop=95
    p.materials.append(Material(name='New dielectric',index=2.25));s.material='New dielectric'
    out,report=write_fsp_geometry(doc,p);q=convert_fsp(out,backend='cpu').project
    assert q is not None and q.structures[0].name==s.name
    assert q.materials[0].index==2.25 and q.structures[0].mesh_order==7
    assert_untouched(doc.data,out.data,report)
    # Same native rasterization and optical solve after the unit/format roundtrip.
    np.testing.assert_array_equal(voxelize(p)[0],voxelize(q)[0])
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_array_equal(a.electric,b.electric);np.testing.assert_array_equal(a.signals,b.signals)


def test_polygon_closed_contour_noop_and_vertex_count_changes():
    doc,p=imported(shape_fixture('polygon',closed=True))
    assert len(p.structures[0].vertices)==6
    assert write_fsp_geometry(doc,p)[0].data==doc.data
    p.structures[0].vertices=((0,0),(.5,0),(0,.5))
    out,report=write_fsp_geometry(doc,p)
    assert len(convert_fsp(out).project.structures[0].vertices)==3
    assert len(out.data)<len(doc.data);assert_untouched(doc.data,out.data,report)


def test_ring_can_add_missing_ellipse_properties_without_reencoding_unknowns():
    doc,p=imported(shape_fixture('ring',ellipse=False,omit_ellipse_properties=True));s=p.structures[0]
    s.make_ellipsoid=True;s.radius_2=.25;s.inner_radius_2=.08
    out,report=write_fsp_geometry(doc,p)
    assert convert_fsp(out).project.structures[0].make_ellipsoid
    assert_untouched(doc.data,out.data,report)


@pytest.mark.parametrize('angles',[(17,31,-29),(45,90,33),(45,-90,33),(0,90-1e-8,33)])
def test_legacy_fourth_rotation_canonicalization(angles):
    s=Structure(rotation=19,rotation_axes=('x','y','z'),rotation_angles=angles)
    axes,new=_rotation(s)
    np.testing.assert_allclose(rotation_matrix(s),Rotation.from_euler('xyz',new,degrees=True).as_matrix(),atol=3e-12)
    assert axes==('x','y','z')
    doc,p=imported(shape_fixture('rectangle'));p.structures[0].rotation=19;p.structures[0].rotation_angles=angles
    out,report=write_fsp_geometry(doc,p);assert report['rotation_reparameterized']==[p.structures[0].id]
    np.testing.assert_allclose(rotation_matrix(p.structures[0]),rotation_matrix(convert_fsp(out).project.structures[0]),atol=3e-12)


@pytest.mark.parametrize('change,match',[
    (lambda p:setattr(p.region,'steps',30),'region.steps'),
    (lambda p:setattr(p.sources[0],'phase',15),'sources'),
    (lambda p:p.structures.clear(),'Adding, deleting'),
    (lambda p:setattr(p.structures[0],'kind','circle'),'primitive type'),
    (lambda p:setattr(p.import_provenance,'source_sha256','0'*64),'fingerprint'),
    (lambda p:setattr(p.structures[0],'center',(2.,0,0)),'PML'),
    (lambda p:setattr(p.materials[0],'model','drude'),'dispersive'),
    (lambda p:setattr(p.structures[0],'name','bad\0name'),'null character')])
def test_unsupported_changes_fail_without_mutating_original(change,match):
    raw=shape_fixture('rectangle');doc,p=imported(raw);change(p)
    with pytest.raises(ValueError,match=match):write_fsp_geometry(doc,p)
    assert doc.data==raw


def test_native_execution_settings_are_explicitly_reported():
    doc,p=imported(shape_fixture());p.region.backend='cuda';p.region.cuda_kernel='fused'
    out,report=write_fsp_geometry(doc,p)
    assert out.data==doc.data and 'region.backend' in report['native_only_settings']
    assert 'region.cuda_kernel' in report['native_only_settings']


def test_malformed_polygon_metadata_and_ring_switch_are_rejected():
    doc,p=imported(shape_fixture('polygon'))
    n=next(n for n in doc.nodes() if n.legacy)
    start,end,_=n.legacy_fields['polygon pivot']
    damaged=bytearray(doc.data);damaged[start+29:start+37]=struct.pack('<d',float('nan'))
    assert convert_fsp(FspDocument(damaged)).project is None
    ring=FspDocument(shape_fixture('ring'));n=next(n for n in ring.nodes() if n.legacy)
    stored=n.properties['mIsEllipse'];damaged=bytearray(ring.data)
    damaged[stored.payload:stored.end]=struct.pack('<i',2)
    assert convert_fsp(FspDocument(damaged)).project is None


def test_changed_geometry_cpu_cuda_and_tensor_cohort():
    import torch
    from torchfdtd import run_tensor_batch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    cases=[]
    for kind in ('rectangle','sphere','circle','ring','polygon'):
        doc,p=imported(shape_fixture(kind));p.structures[0].center=(.07,-.08,.09)
        p.structures[0].rotation_angles=(19,24,37)
        q=convert_fsp(write_fsp_geometry(doc,p)[0],backend='cuda').project
        cpu=Simulation(p).run();gpu=Simulation(q).run()
        np.testing.assert_allclose(cpu.electric,gpu.electric,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(cpu.signals,gpu.signals,atol=2e-6,rtol=2e-6)
        cases.append(q)
    results=run_tensor_batch(cases);results.raise_for_errors()
    for p,item in zip(cases,results.items):
        result=item.result
        serial=Simulation(p).run()
        np.testing.assert_allclose(result.electric,serial.electric,atol=2e-6,rtol=2e-6)


def test_cli_exclusive_outputs_and_geometry_report(tmp_path):
    import json
    source=tmp_path/'source.fsp';source.write_bytes(shape_fixture())
    doc,p=imported(source.read_bytes());p.structures[0].name='CLI polygon'
    scene=tmp_path/'scene.json';p.save(scene)
    output=tmp_path/'edited.fsp';report=tmp_path/'report.json'
    command=[sys.executable,'-m','torchfdtd.cli','fsp-write-geometry',str(source),str(scene),
             '--output',str(output),'--report',str(report)]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['requires_lumerical'] is False
    assert convert_fsp(FspDocument.load(output)).project.structures[0].name=='CLI polygon'
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert source.read_bytes()==doc.data


def test_api_geometry_export_and_rejected_non_geometry_change(tmp_path,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Vendor runtime accessed'))
    app=create_app(tmp_path)
    try:
        with TestClient(app) as client:
            def wait(key):
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    job=client.get('/api/fsp/'+key).json()
                    if job['status'] in ('ready','failed'):return job
                    time.sleep(.01)
                pytest.fail('FSP job did not complete')
            raw=shape_fixture()
            key=client.post('/api/fsp/native-import',content=raw,headers={'x-filename':'synthetic.fsp'}).json()['id']
            job=wait(key);assert job['status']=='ready'
            scene=job['conversion']['project'];scene['structures'][0]['name']='API geometry'
            created=client.post('/api/fsp/'+key+'/native-export',json=scene)
            assert created.status_code==202
            exported=wait(created.json()['id']);assert exported['status']=='ready',exported
            newkey=exported['id'];data=client.get('/api/fsp/'+newkey+'/download').content
            assert convert_fsp(FspDocument(data)).project.structures[0].name=='API geometry'
            assert client.get('/api/fsp/'+key+'/download').content==raw
            assert client.get('/api/fsp/'+newkey+'/write-report').json()['output_sha256']==FspDocument(data).fingerprint()
            scene['sources'][0]['phase']=51
            failed=wait(client.post('/api/fsp/'+key+'/native-export',json=scene).json()['id'])
            assert failed['status']=='failed' and 'sources' in failed['error']
            assert client.get('/api/fsp/'+failed['id']+'/download').status_code==409
    finally:
        app.state.pool.shutdown();app.state.fsp_pool.shutdown()
