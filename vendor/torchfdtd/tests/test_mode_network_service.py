"""Queue, publication and interruption checks with bounded fake calculations."""
from concurrent.futures import Future, ThreadPoolExecutor
import io
import json
import multiprocessing
import os
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest


def _blocking_worker(snapshot, output_path, connection, memory_admission=False):
    from torchfdtd.mode_network_worker import send_message
    send_message(connection, {'type': 'progress', 'data': {'pid': os.getpid()}})
    while True:
        time.sleep(.1)


class DeferredPool:
    def __init__(self):
        self.calls = []
    def submit(self, function, *args):
        future = Future()
        self.calls.append((future, function, args))
        return future
    def run(self):
        future, function, args = self.calls.pop(0)
        if future.set_running_or_notify_cancel():
            function(*args)
            future.set_result(None)


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    from torchfdtd import mode_network_service as service
    from torchfdtd.models import demo_project
    # The service tests exercise lifecycle only. Physics admission has its own
    # adapter tests and is deliberately replaced here before any solve.
    monkeypatch.setattr(service, 'mode_network_plan', lambda config: {'status': 'valid'})
    project = demo_project('3d').model_dump(mode='json')
    payload = dict(version=1, project=project, normal='z',
        ports=[dict(name='left', coordinate_um=-.2, source_coordinate_um=-.3,
            direction=1, mode_indices=[0]), dict(name='right', coordinate_um=.2,
            source_coordinate_um=.3, direction=-1, mode_indices=[0])])
    return service, tmp_path, payload


def setup_app(service, root, pool):
    app, jobs, lock = FastAPI(), {}, threading.Lock()
    service.attach_mode_network_routes(app, root, pool, jobs, lock)
    return app, jobs


def result(snapshot=None):
    from torchfdtd.mode_network_service import request_digest
    return dict(status='completed', channels=[['left', 0], ['right', 0]],
        s_real=[[0., .8], [.8, 0.]], s_imag=[[0., .1], [.1, 0.]],
        phase_planes_um=[-.2, .2], objective=None, material_gradients={},
        mode_summaries=[], admission={}, request_digest='test' if snapshot is None else request_digest(snapshot))


def test_shared_queue_snapshot_export_and_npz(fixture, monkeypatch):
    service, root, payload = fixture
    pool = DeferredPool()
    app, jobs = setup_app(service, root, pool)
    client = TestClient(app)
    captured = []
    def execute(snapshot, scratch, cancel, progress):
        captured.append(snapshot)
        progress({'stage': 'modal_solve'})
        return result(snapshot)
    monkeypatch.setattr(service, '_execute_owned', execute)
    assert client.post('/api/mode-networks/validate', json=payload).json() == {'status': 'valid'}
    exported = client.post('/api/mode-networks/python', json=payload)
    assert exported.status_code == 200
    assert 'ModeNetworkConfig.model_validate' in exported.text
    assert exported.text == client.post('/api/mode-networks/python', json=payload).text
    key = client.post('/api/mode-network-jobs', json=payload).json()['id']
    original = payload['project']['name']
    payload['project']['name'] = 'changed'
    pool.run()
    assert captured[0]['project']['name'] == original
    status = client.get(f'/api/mode-network-jobs/{key}').json()
    assert status['status'] == 'completed' and status['result']['channels'] == result()['channels']
    assert not any('process' in k or 'pipe' in k for k in jobs[key])
    data = client.get(f'/api/mode-network-jobs/{key}/download').content
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        assert json.loads(str(archive['result_json'])) == result(captured[0])
        assert archive['s'].dtype == np.complex64
        assert archive['s_real'].dtype == archive['s_imag'].dtype == np.float32
    assert 'power' in client.get(f'/api/mode-network-jobs/{key}/s.csv').text
    jobs['ordinary'] = dict(status='running')
    for _ in range(2):
        assert client.post('/api/mode-network-jobs', json=payload).status_code == 202
    assert client.post('/api/mode-network-jobs', json=payload).status_code == 409


@pytest.mark.parametrize('mode', ['queued_cancel', 'publication_cancel', 'failure'])
def test_cancel_and_failure_never_publish(fixture, monkeypatch, mode):
    service, root, payload = fixture
    pool = DeferredPool()
    app, jobs = setup_app(service, root, pool)
    client = TestClient(app)
    def execute(snapshot, scratch, cancel, progress):
        if mode == 'failure':
            raise RuntimeError('owned spawn or calculation failure')
        if mode == 'queued_cancel':
            pytest.fail('Cancelled queued job launched.')
        cancel.set()
        return result(snapshot)
    monkeypatch.setattr(service, '_execute_owned', execute)
    key = client.post('/api/mode-network-jobs', json=payload).json()['id']
    if mode == 'queued_cancel':
        client.post(f'/api/mode-network-jobs/{key}/cancel')
    pool.run()
    assert jobs[key]['status'] == ('failed' if mode == 'failure' else 'cancelled')
    assert list(root.iterdir()) == []
    assert client.get(f'/api/mode-network-jobs/{key}/download').status_code == 409


def test_shutdown_waits_for_owned_runner_cleanup(fixture, monkeypatch):
    service, root, payload = fixture
    started, reaped = threading.Event(), threading.Event()
    def execute(snapshot, scratch, cancel, progress):
        started.set()
        assert cancel.wait(5)
        reaped.set()
        return None
    monkeypatch.setattr(service, '_execute_owned', execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        app, jobs = setup_app(service, root, pool)
        with TestClient(app) as client:
            key = client.post('/api/mode-network-jobs', json=payload).json()['id']
            assert started.wait(5)
        assert reaped.is_set() and jobs[key]['status'] == 'cancelled'
    assert list(root.iterdir()) == []


def test_shutdown_reaps_actual_spawned_child(fixture, monkeypatch):
    service, root, payload = fixture
    monkeypatch.setattr(service, 'mode_network_worker', _blocking_worker)
    with ThreadPoolExecutor(max_workers=1) as pool:
        app, jobs = setup_app(service, root, pool)
        with TestClient(app) as client:
            key = client.post('/api/mode-network-jobs', json=payload).json()['id']
            deadline = time.monotonic() + 20
            while not jobs[key]['progress'] and time.monotonic() < deadline:
                time.sleep(.02)
            assert 'pid' in jobs[key]['progress'], jobs[key]
            pid = jobs[key]['progress']['pid']
        assert jobs[key]['status'] == 'cancelled'
        assert all(child.pid != pid for child in multiprocessing.active_children())
    assert list(root.iterdir()) == []


def test_cancel_during_npz_preparation_wins_publication(fixture, monkeypatch):
    service, root, payload = fixture
    pool = DeferredPool()
    app, jobs = setup_app(service, root, pool)
    client = TestClient(app)
    monkeypatch.setattr(service, '_execute_owned', lambda snapshot, *args: result(snapshot))
    key = client.post('/api/mode-network-jobs', json=payload).json()['id']
    original = service.np.savez_compressed
    def save(*args, **kwargs):
        original(*args, **kwargs)
        jobs[key]['cancel'].set()
    monkeypatch.setattr(service.np, 'savez_compressed', save)
    pool.run()
    assert jobs[key]['status'] == 'cancelled' and list(root.iterdir()) == []


def test_spawn_failure_closes_pipe_and_unstarted_handle(fixture, monkeypatch):
    service, root, _ = fixture
    class Pipe:
        closed = False
        def close(self): self.closed = True
    class Process:
        pid = None
        closed = False
        def start(self): raise OSError('spawn failed')
        def close(self): self.closed = True
    receive, send, child = Pipe(), Pipe(), Process()
    class Context:
        def Pipe(self, **kwargs): return receive, send
        def Process(self, **kwargs): return child
    monkeypatch.setattr(service.multiprocessing, 'get_context', lambda kind: Context())
    with pytest.raises(OSError, match='spawn failed'):
        service._execute_owned({}, root / 'generated.json', threading.Event(), lambda data: None)
    assert receive.closed and send.closed and child.closed


@pytest.mark.parametrize('defect', ['digest', 'channels', 'shape', 'nonfinite'])
def test_invalid_worker_result_is_not_published(fixture, monkeypatch, defect):
    service, root, payload = fixture
    pool = DeferredPool()
    app, jobs = setup_app(service, root, pool)
    client = TestClient(app)
    def execute(snapshot, *args):
        value = result(snapshot)
        if defect == 'digest': value['request_digest'] = 'wrong'
        elif defect == 'channels': value['channels'].reverse()
        elif defect == 'shape': value['s_real'] = [[0.]]
        else: value['s_imag'][0][0] = float('nan')
        return value
    monkeypatch.setattr(service, '_execute_owned', execute)
    key = client.post('/api/mode-network-jobs', json=payload).json()['id']
    pool.run()
    assert jobs[key]['status'] == 'failed'
    assert list(root.iterdir()) == []
