import pytest
from fastapi.testclient import TestClient
from torchfdtd.server import create_app
from torchfdtd.models import demo_project,Project
from torchfdtd.gds import import_gds,GDSLayer
from test_gds import rectangle_file


def test_browser_conversion_matches_python_native_geometry(tmp_path):
    client=TestClient(create_app(tmp_path/'results'))
    path=rectangle_file(tmp_path)
    response=client.post('/api/gds/inspect',content=path.read_bytes(),headers={'x-filename':'layout.gds'})
    assert response.status_code==200,response.text
    upload=response.json();assert upload['cells']==[{'name':'TOP','geometry_pairs':[[1,2]],'text_pairs':[]}]
    project=demo_project();material=project.materials[0].name
    body={'project':project.model_dump(),'cell':'TOP','layers':[{'layer':1,'datatype':2,'z_min':-.1,'z_max':.1,'material':material}]}
    response=client.post('/api/gds/'+upload['id']+'/convert',json=body)
    assert response.status_code==200,response.text
    native=Project.model_validate(response.json()['project'])
    expected=import_gds(path,cell='TOP',layers=[GDSLayer(1,2,-.1,.1,material)]).add_to(project)
    assert native.model_dump()==expected.model_dump()
    assert client.post('/api/validate',json=native.model_dump()).status_code==200
    script=client.post('/api/python',json=native.model_dump())
    assert script.status_code==200 and 'polygon' in script.text
    body['layers'][0]['material']='unknown'
    assert client.post('/api/gds/'+upload['id']+'/convert',json=body).status_code==422
    assert client.post('/api/gds/missing/convert',json=body).status_code==404


def test_optional_dependency_and_upload_admission(tmp_path,monkeypatch):
    from torchfdtd import gds,gds_service
    client=TestClient(create_app(tmp_path/'results'));path=rectangle_file(tmp_path)
    def missing():raise ImportError('pip install torchfdtd[gds]')
    monkeypatch.setattr(gds,'_gdstk',missing)
    response=client.post('/api/gds/inspect',content=path.read_bytes())
    assert response.status_code==503 and 'torchfdtd[gds]' in response.text
    monkeypatch.setattr(gds_service,'MAX_UPLOAD_BYTES',20)
    assert client.post('/api/gds/inspect',content=path.read_bytes()).status_code==413
    assert client.post('/api/gds/inspect',content=b'bad').status_code==422


def test_browser_conversion_carries_polygon_holes_and_etch_pairs(tmp_path):
    gdstk=pytest.importorskip('gdstk')
    from test_gds import write
    lib=gdstk.Library();cell=lib.new_cell('TOP')
    cell.add(gdstk.rectangle((-.8,-.4),(.8,.4),layer=1),gdstk.rectangle((-.2,-.1),(.2,.1),layer=2))
    path=write(tmp_path,lib)
    client=TestClient(create_app(tmp_path/'results'))
    upload=client.post('/api/gds/inspect',content=path.read_bytes()).json()
    project=demo_project();material=project.materials[0].name
    body={'project':project.model_dump(),'cell':'TOP','layers':[{'layer':1,'datatype':0,'z_min':-.1,'z_max':.1,'material':material,'etch_by':[[2,0]]}]}
    response=client.post('/api/gds/'+upload['id']+'/convert',json=body)
    assert response.status_code==200,response.text
    payload=response.json();shape=payload['project']['structures'][-1]
    assert shape['kind']=='polygon' and len(shape['holes'])==1 and len(shape['holes'][0])==4
    assert payload['report']['hole_count']==1 and payload['report']['etched_layers']==[{'layer':1,'datatype':0,'etch_by':[[2,0]],'polygons':1}]
    assert client.post('/api/validate',json=payload['project']).status_code==200
    assert '\'holes\'' in client.post('/api/python',json=payload['project']).text
    del body['layers'][0]['etch_by']
    assert client.post('/api/gds/'+upload['id']+'/convert',json=body).status_code==422
