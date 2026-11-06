import ast
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
import torch

from torchfdtd import PeriodicDesignConfig, periodic_design_plan, run_periodic_design
from torchfdtd.server import create_app


def test_adam_result_scores_evaluated_design_and_cancellation():
    calls=[]
    def model(density):
        calls.append(density.detach().clone())
        return density.mean().expand(2,4)
    config=PeriodicDesignConfig(initial_density=[[.3,.4]],iterations=2,learning_rate=.01)
    with patch('torchfdtd.periodic_design._prepare',return_value=(model,{})):
        result=run_periodic_design(config)
    assert len(calls)==3 and result['updates_completed']==2
    assert result['history'][-1]['objective'] > result['history'][0]['objective']
    torch.testing.assert_close(torch.tensor(result['last_evaluated']['density']),calls[-1])
    assert result['last_evaluated']['objective']==float(calls[-1].mean())
    assert result['pending_density'] is None
    event=threading.Event()
    with patch('torchfdtd.periodic_design._prepare',return_value=(model,{})):
        cancelled=run_periodic_design(config,cancel=event,
            on_progress=lambda p:event.set() if p['stage']=='evaluated' else None)
    assert cancelled['status']=='cancelled' and cancelled['updates_completed']==0
    assert len(calls)==4 and cancelled['last_evaluated']['update']==0


def test_memory_plan_runs_no_fields_and_revalidates_mutated_config():
    config=PeriodicDesignConfig(initial_density=[[.3,.4],[.5,.6]],quadrature_counts=(4,4))
    with patch('torchfdtd.periodic_adjoint.PeriodicLayerResponse._compute_response',
               side_effect=AssertionError('Planning must not solve fields')):
        plan=periodic_design_plan(config)
    assert plan['calibration_solves']==0 and plan['selection']['mode']=='resident'
    assert plan['total_host_reservation_bytes'] > plan['host_reservation_bytes']
    config.initial_density[0][0]=2
    with pytest.raises(ValueError,match='density'):periodic_design_plan(config)
    with pytest.raises(ValueError,match='directory'):PeriodicDesignConfig(execution='file')
    with pytest.raises(ValueError):PeriodicDesignConfig(steps=True)


def test_design_routes_share_queue_and_export_validated_python(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        config=client.post('/api/design/config',json={'iterations':1,'initial_density':[[.5,.5]]}).json()
        assert config['wavelength_um']==.5
        source=client.post('/api/design/python',json=config).text
        parsed=ast.parse(source)
        assignment=next(n for n in parsed.body if isinstance(n,ast.Assign))
        assert ast.literal_eval(assignment.value.args[0])==config
        for i in range(3):app.state.jobs[str(i)]=dict(status='running')
        assert client.post('/api/design/jobs',json=config).status_code==409
        app.state.jobs.clear()
        response=client.post('/api/design/jobs',json={'initial_density':[[2]]})
        assert response.status_code==422
        def fake_run(config,*,on_progress,cancel):
            on_progress(dict(stage='evaluated',updates_completed=1,history=[{'update':1,'objective':.6}]))
            return dict(status='completed',updates_completed=1,best={'objective':.6},last_evaluated={'objective':.6})
        with patch('torchfdtd.design_service.run_periodic_design',fake_run):
            response=client.post('/api/design/jobs',json=config)
            assert response.status_code==202
            key=response.json()['id']
            for _ in range(100):
                job=client.get('/api/jobs/'+key).json()
                if job['status'] in ('completed','failed'):break
                time.sleep(.01)
            assert job['status']=='completed',job
            assert client.get('/api/design/jobs/'+key+'/download').json()['best']['objective']==.6
            assert client.get('/api/jobs').json()[0]['name']==config['name']
        app.state.jobs['waiting']=dict(status='queued',kind='periodic_design',cancel=threading.Event())
        assert not client.get('/api/jobs/waiting').json()['cancel_requested']
        client.post('/api/jobs/waiting/cancel')
        assert client.get('/api/jobs/waiting').json()['cancel_requested']
    app.state.pool.shutdown()
