"""Actual native CPU job, stored adapter, and bounded diffraction API."""
import time
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd import Project, Region, Source
from torchfdtd.models import FieldMonitor
from torchfdtd.server import create_app
from torchfdtd.radiation_io import load_native_radiation_plane,native_radiation_plane
from torchfdtd.radiation import diffraction_orders


def project():
    faces={a+'_'+s:dict(kind='pml' if a=='z' else 'periodic') for a in 'xyz' for s in ('min','max')}
    return Project(region=Region(dimension='3d',size=(.6,.6,1.6),mesh=.1,steps=40,backend='cpu',
        pml_cells=3,boundaries=faces),sources=[Source(kind='plane',normal='z',component='Ex',size=(.6,.6,0),
        center=(0,0,-.3),pulse='continuous',wavelength=1.)],monitors=[FieldMonitor(kind='field',normal='z',
        size=(.6,.6,0),center=(0,0,.3),spectrum=dict(apodization='none',sampling='custom',custom_frequencies_hz=[299792458e6]))])


def finish(client,p):
    response=client.post('/api/jobs',json=p.model_dump());assert response.status_code==202
    key=response.json()['id'];deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        job=client.get('/api/jobs/'+key).json()
        if job['status'] in ('completed','failed'):break
        time.sleep(.02)
    assert job['status']=='completed',job
    return key


def test_native_cpu_saved_plane_and_api_reference_workflow(tmp_path,monkeypatch):
    app=create_app(tmp_path);p=project();monitor=p.monitors[0].id
    try:
        with TestClient(app) as client:
            key=finish(client,p);reference=finish(client,p)
            listing=client.get(f'/api/jobs/{key}/diffraction-monitors').json()
            assert listing[0]['components']==['Ex','Ey','Ez','Hx','Hy','Hz']
            payload=dict(monitor=monitor,orders=[[0,0],[1,0]],refractive_index=1.,confirm_homogeneous_exterior=True)
            response=client.post(f'/api/jobs/{key}/diffraction',json=payload)
            assert response.status_code==200,response.text
            raw=response.json();assert raw['propagating']==[True,False]
            assert raw['forward_power'][1]==raw['backward_power'][1]==0
            assert raw['normalized'] is False
            normalized=client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,reference=reference))
            assert normalized.status_code==200,normalized.text
            values=normalized.json();assert values['normalized'] is True
            assert values['forward_efficiency'][0]-values['backward_efficiency'][0]==pytest.approx(1.,rel=2e-5)
            rejected=client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,confirm_homogeneous_exterior=False))
            assert rejected.status_code==422 and 'homogeneous' in rejected.text
            assert client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,orders=[[3,0]])).status_code==422
            cutoff=client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,refractive_index=1/.6))
            assert cutoff.status_code==422 and 'grazing cutoff' in cutoff.text
            assert client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,orders=[[True,0]])).status_code==422
            ref_record=app.state.jobs[reference]['frequency_fields'][0]
            original_signature=ref_record['run_signature'];ref_record['run_signature']='different-source'
            mismatch=client.post(f'/api/jobs/{key}/diffraction',json=dict(payload,reference=reference))
            assert mismatch.status_code==422 and 'Reference mesh, source' in mismatch.text
            ref_record['run_signature']=original_signature
            archive=tmp_path/f'{key}.npz'
            with np.load(archive,allow_pickle=False) as stored:cls=type(stored)
            original=cls.__getitem__
            def safe_read(self,name):
                assert name not in ('E','H','frames','epsilon')
                return original(self,name)
            monkeypatch.setattr(cls,'__getitem__',safe_read)
            plane,restored=load_native_radiation_plane(archive,monitor,frequency_index=0)
            result=diffraction_orders(plane,[(0,0),(1,0)],period_um=(.6,.6))
            torch.testing.assert_close(result.forward_power[0],torch.tensor(raw['forward_power'],dtype=torch.float64))
            assert restored.region.dimension=='3d'
            with pytest.raises(ValueError,match='byte budget'):
                load_native_radiation_plane(archive,monitor,max_bytes=1)
    finally:app.state.pool.shutdown()


def test_adapter_rejects_flux_only_and_missing_signature():
    with pytest.raises(ValueError,match='all six'):
        native_radiation_plane(dict(components=['Ex','Hy']))
    record=dict(components=['Ex','Ey','Ez','Hx','Hy','Hz'],flux_units='reduced E*H * s^2 * m^2',
        settings=dict(spectrum=dict(apodization='none')))
    with pytest.raises(ValueError,match='run signature'):
        native_radiation_plane(record)
