import time

from fastapi.testclient import TestClient

from torchfdtd.models import demo_project
from torchfdtd.server import create_app


def test_api_job_lifecycle(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        p=demo_project();p.region.backend='cpu';p.region.steps=100
        assert client.post('/api/validate',json=p.model_dump()).status_code==200
        response=client.post('/api/jobs',json=p.model_dump());assert response.status_code==202
        key=response.json()['id']
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+key).json()
            if job['status'] in ('completed','failed'):
                break
            time.sleep(.03)
        assert job['status']=='completed',job
        assert client.get('/api/jobs/'+key+'/fields').json()['frames']
        assert client.get('/api/jobs/'+key+'/download').content[:2]==b'PK'
        assert 'time_fs' in client.get('/api/jobs/'+key+'/monitors.csv').text
        assert 'wavelength_um,real,imag,magnitude,units,apodization' in client.get('/api/jobs/'+key+'/spectra.csv').text
        assert 'Simulation(project).run()' in client.post('/api/python',json=p.model_dump()).text
        assert client.post('/api/jobs',json=p.model_dump(),headers={'Origin':'https://unrelated.example'}).status_code==403
        assert client.get('/api/health',headers={'Host':'evil.example'}).status_code==400
    app.state.pool.shutdown()


def test_health_reports_cpu_when_cuda_is_available_without_a_visible_device(tmp_path,monkeypatch):
    """CUDA_VISIBLE_DEVICES="" leaves torch.cuda.is_available() True with device_count() 0; no device query may run."""
    import torch
    from torchfdtd.solver import hardware
    from torchfdtd.execution_modes import execution_resources
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    def invalid(*args,**kwargs):raise AssertionError('Invalid device id')
    for name in ('get_device_name','get_device_properties','mem_get_info'):
        monkeypatch.setattr(torch.cuda,name,invalid)
    assert hardware()['cuda'] is False and hardware()['gpu'] is None and hardware()['gpu_memory_gb']==0
    resources=execution_resources()
    assert resources['cuda'] is False and resources['gpu'] is None and resources['cupy'] is False
    app=create_app(tmp_path)
    with TestClient(app) as client:
        response=client.get('/api/health')
        assert response.status_code==200
        assert response.json()['cuda'] is False and response.json()['gpu'] is None
    app.state.pool.shutdown()


def test_validate_reports_stored_frames_per_optical_period(tmp_path):
    import pytest
    from torchfdtd.solver import snapshot_frames
    app=create_app(tmp_path)
    with TestClient(app) as client:
        for name in ('waveguide','scatterer','3d','pmc'):
            p=demo_project(name)
            snapshot=client.post('/api/validate',json=p.model_dump()).json()['snapshot']
            assert snapshot==snapshot_frames(p) and snapshot['frames_per_period']>=4 and not snapshot['aliased'], (name, snapshot)
        p=demo_project()
        snapshot=client.post('/api/validate',json=p.model_dump()).json()['snapshot']
        # 1.55 um at a 0.05 um mesh: one period is 44.3 steps; the solver keeps at most 100 frames, so 1000 steps store one every 10.
        assert snapshot['period_steps']==pytest.approx(44.3,abs=.1) and snapshot['effective_interval']==10 and snapshot['frames']==100
        assert snapshot['frames_per_period']==pytest.approx(4.43,abs=.01)
        p.region.snapshot_interval=40
        summary=client.post('/api/validate',json=p.model_dump()).json()
        assert summary['snapshot']['aliased'] and summary['snapshot']['frames_per_period']==pytest.approx(1.107,abs=.01)
        assert any('alias the carrier' in w and 'backward motion' in w for w in summary['warnings'])
        p.sources=[]
        summary=client.post('/api/validate',json=p.model_dump()).json()
        assert summary['snapshot']['frames_per_period'] is None and not summary['snapshot']['aliased']
    app.state.pool.shutdown()
