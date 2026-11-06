"""Generic cancellation and modal publication share one ordering boundary."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from fastapi.testclient import TestClient
from torchfdtd.models import demo_project
from torchfdtd.server import create_app
from torchfdtd import mode_network_service as service


def payload():
    return dict(project=demo_project('3d').model_dump(mode='json'),normal='z',
        ports=[dict(name='left',coordinate_um=-.2,source_coordinate_um=-.3,direction=1),
               dict(name='right',coordinate_um=.2,source_coordinate_um=.3,direction=-1)])


def result(snapshot):
    return dict(status='completed',channels=[['left',0],['right',0]],
        s_real=[[0.,1.],[1.,0.]],s_imag=[[0.,0.],[0.,0.]],
        request_digest=service.request_digest(snapshot))


def test_generic_cancel_cannot_interleave_modal_final_publication(monkeypatch,tmp_path):
    entered=threading.Event();release=threading.Event();requested=threading.Event();answered=threading.Event()
    responses=[];errors=[]
    monkeypatch.setattr(service,'mode_network_plan',lambda c:{})
    monkeypatch.setattr(service,'_execute_owned',lambda snapshot,*args:result(snapshot))
    original=Path.replace
    def replace(path,target):
        if str(target).endswith('.modal.npz'):
            entered.set()
            if not release.wait(5): raise RuntimeError('Test publication barrier timed out')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',replace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr('torchfdtd.server.ThreadPoolExecutor',lambda **kwargs:pool)
        with TestClient(create_app(tmp_path)) as client:
            job=client.post('/api/mode-network-jobs',json=payload()).json()['id']
            assert entered.wait(5)
            def cancel():
                requested.set()
                try: responses.append(client.post('/api/jobs/'+job+'/cancel'))
                except BaseException as exc: errors.append(exc)
                finally: answered.set()
            caller=threading.Thread(target=cancel)
            caller.start()
            try:
                assert requested.wait(2)
                # Publication already owns the shared lock. This used to
                # return immediately and set cancel on a still-running job.
                assert not answered.wait(.2)
            finally:
                release.set();caller.join(timeout=5)
            assert not caller.is_alive() and not errors
            assert responses[0].status_code==200
            assert responses[0].json()==dict(status='completed',cancel_requested=False)
            status=client.get('/api/mode-network-jobs/'+job).json()
            assert status['status']=='completed' and not status['cancel_requested']
            assert client.get('/api/mode-network-jobs/'+job+'/download').status_code==200


def test_generic_cancel_before_modal_publication_never_publishes(monkeypatch,tmp_path):
    ready=threading.Event();release=threading.Event()
    monkeypatch.setattr(service,'mode_network_plan',lambda c:{})
    def execute(snapshot,scratch,cancel,progress):
        ready.set()
        if not release.wait(5): raise RuntimeError('Test worker barrier timed out')
        return result(snapshot)
    monkeypatch.setattr(service,'_execute_owned',execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr('torchfdtd.server.ThreadPoolExecutor',lambda **kwargs:pool)
        with TestClient(create_app(tmp_path)) as client:
            job=client.post('/api/mode-network-jobs',json=payload()).json()['id']
            assert ready.wait(5)
            try:
                response=client.post('/api/jobs/'+job+'/cancel')
                assert response.status_code==200 and response.json()['cancel_requested']
            finally: release.set()
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                status=client.get('/api/mode-network-jobs/'+job).json()
                if status['status']=='cancelled': break
                time.sleep(.01)
            assert status['status']=='cancelled'
            assert client.get('/api/mode-network-jobs/'+job+'/download').status_code==409
            assert not list(tmp_path.glob('*.modal.*'))
