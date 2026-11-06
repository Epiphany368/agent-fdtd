"""HTTP contracts, output precision and concurrent postprocessing admission."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import threading
from types import ModuleType

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source
from torchfdtd.farfield_api import FACES, register_farfield_routes
from torchfdtd.radiation import FarFieldResult


def payload():
    return dict(faces={name:name for name in FACES}, bounds_um=[[-.6,.6]]*3,
        confirm_homogeneous_closed_surface=True,
        theta_deg=dict(start=0.,stop=180.,count=3), phi_deg=dict(start=0.,stop=360.,count=4))


@pytest.fixture
def setup(monkeypatch):
    project=Project(region=Region(dimension='3d',size=(2.4,2.4,2.4),mesh=.1,steps=20,pml_cells=3),
        structures=[],sources=[Source(kind='point')],monitors=[])
    records=[]
    for name in FACES:
        points=np.zeros((4,3));points[:,'xyz'.index(name[0])]=-.6 if name.endswith('min') else .6
        records.append(dict(id=name,name=name,normal_axis=name[0],points_um=points,shape=[2,2],weights=np.ones(4),
            components=list(('Ex','Ey','Ez','Hx','Hy','Hz')),frequency_hz=np.array([3e14]),fields=np.ones((1,4,6),dtype=np.complex64)))
    job=dict(status='completed',project=project.model_dump(mode='json'),summary={},frequency_fields=records)
    jobs=dict(sample=job,reference=job)
    calls=[]
    def get_job(key):
        if key not in jobs:raise HTTPException(404,'Unknown job')
        return jobs[key]
    class Box:
        report=dict(field_kind='total',retained_bytes=4096,points=24)
        def project(self,directions,**kwargs):
            # A nonzero complex64 optical coefficient whose FP32 squared
            # magnitude underflows. The output reduction must preserve it.
            field=torch.full((1,len(directions),3),complex(1e-24,2e-24),dtype=torch.complex64)
            return FarFieldResult(field,torch.as_tensor(directions,dtype=torch.float32),
                torch.tensor([3e14],dtype=torch.float32),1.,torch.tensor(kwargs['phase_origin_um'],dtype=torch.float32))
    def prepare(*args,**kwargs):
        calls.append(kwargs);box=Box()
        box.report=dict(Box.report,field_kind='scattered' if kwargs.get('reference') is not None else 'total')
        return box
    module=ModuleType('torchfdtd.radiation_box');module.native_radiation_box=prepare
    monkeypatch.setitem(sys.modules,'torchfdtd.radiation_box',module)
    app=FastAPI();register_farfield_routes(app,get_job)
    return TestClient(app),jobs,calls,Box


def test_metadata_and_complex_output_precision_units_and_order(setup):
    client,jobs,calls,_=setup
    metadata=client.get('/api/jobs/sample/farfield-monitors').json()
    assert metadata['isolated_pml'] and len(metadata['monitors'])==6
    assert metadata['monitors'][0]['position_um']==-.6
    response=client.post('/api/jobs/sample/farfield',json=payload())
    assert response.status_code==200,response.text
    value=response.json();intensity=np.array(value['intensity'])
    assert intensity.shape==(3,4) and (intensity>0).all()
    np.testing.assert_allclose(intensity,7.5e-48,rtol=2e-7)
    np.testing.assert_array_equal(value['relative_intensity'],np.ones((3,4)))
    assert value['phi_deg']==[0,90,180,270]
    np.testing.assert_allclose(value['directions'][4],[1,0,0],atol=1e-6)
    assert value['intensity_units']=='reduced E*H * s^2 * m^2 / sr'
    assert value['field_kind']=='total' and 'not efficiency' in value['note']
    expected=hashlib.sha256(json.dumps(value['request'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert value['request_digest']==expected and len(calls)==1


@pytest.mark.parametrize('change',[
    dict(confirm_homogeneous_closed_surface=False),
    dict(theta_deg=dict(start=0.,stop=180.,count=True)),
    dict(theta_deg=dict(start=0.,stop=180.,count=8192)),
    dict(reference='reference',subtract_incident=False),
    dict(extra=1),
])
def test_invalid_request_rejected_before_adaptation(setup,change):
    client,_,calls,_=setup
    response=client.post('/api/jobs/sample/farfield',json=dict(payload(),**change))
    assert response.status_code==422 and not calls


def test_direction_point_gate_precedes_adaptation_and_reference_pairing(setup):
    client,jobs,calls,_=setup
    request=payload();request.update(reference='reference',subtract_incident=True)
    response=client.post('/api/jobs/sample/farfield',json=request)
    assert response.status_code==200,response.text
    assert response.json()['field_kind']=='scattered' and calls[0]['reference'] is not None
    calls.clear()
    for record in jobs['sample']['frequency_fields']:record['weights']=np.ones(3000)
    request.update(theta_deg=dict(start=0.,stop=180.,count=64),phi_deg=dict(start=0.,stop=360.,count=64))
    response=client.post('/api/jobs/sample/farfield',json=request)
    assert response.status_code==422 and 'work limit' in response.text and not calls


def test_postprocessing_gate_is_independent_bounded_and_released(setup,monkeypatch):
    client,_,_,Box=setup;started=threading.Event();release=threading.Event()
    original=Box.project
    def held(self,*args,**kwargs):
        started.set();assert release.wait(5)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Box,'project',held)
    with ThreadPoolExecutor(1) as pool:
        first=pool.submit(client.post,'/api/jobs/sample/farfield',json=payload())
        try:
            assert started.wait(5)
            response=client.post('/api/jobs/sample/farfield',json=payload())
            assert response.status_code==409
        finally:release.set()
        assert first.result(5).status_code==200
    assert client.post('/api/jobs/sample/farfield',json=payload()).status_code==200
