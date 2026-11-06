"""Small process lifecycle checks without optical solver work."""
import multiprocessing
import time
import json
import sys
from types import SimpleNamespace

import pytest

from torchfdtd.mode_network_worker import close_owned_process, send_message


def _wait_forever(connection):
    connection.send_bytes(b'ready')
    while True:
        time.sleep(.1)


def test_owned_process_is_terminated_and_reaped():
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_wait_forever, args=(send,), daemon=True)
    child.start()
    send.close()
    try:
        assert receive.poll(20)
        assert receive.recv_bytes() == b'ready'
        pid = child.pid
        close_owned_process(child)
        assert all(other.pid != pid for other in multiprocessing.active_children())
        with pytest.raises(ValueError, match='closed'):
            child.is_alive()
    finally:
        receive.close()


def test_json_progress_rejects_large_or_nonfinite_payload_before_send():
    class Connection:
        def send_bytes(self, data):
            pytest.fail('Invalid progress was transmitted.')
    with pytest.raises(ValueError, match='IPC message limit'):
        send_message(Connection(), {'text': 'x' * 65536})
    with pytest.raises(ValueError):
        send_message(Connection(), {'value': float('nan')})


def test_unstarted_owned_process_can_be_closed():
    close_owned_process(multiprocessing.get_context('spawn').Process())


@pytest.mark.parametrize('fail', [False, True])
def test_worker_serializes_result_or_error_without_graph_objects(tmp_path, monkeypatch, fail):
    from torchfdtd.mode_network_worker import mode_network_worker
    class Config:
        @staticmethod
        def model_validate(snapshot):
            return SimpleNamespace(execution=SimpleNamespace(output_budget_bytes=1024))
    def run(config, on_progress):
        on_progress({'stage': 'solve'})
        if fail:
            raise ValueError('deliberate solve failure')
        return {'status': 'completed', 's_real': [[1.]], 's_imag': [[0.]]}
    monkeypatch.setitem(sys.modules, 'torchfdtd.mode_network_project',
        SimpleNamespace(ModeNetworkConfig=Config, run_mode_network=run))
    class Connection:
        closed = False
        messages = []
        def send_bytes(self, payload): self.messages.append(json.loads(payload))
        def close(self): self.closed = True
    connection = Connection()
    path = tmp_path / 'owned.json'
    mode_network_worker({}, str(path), connection)
    assert connection.closed
    assert connection.messages[0]['type'] == 'progress'
    assert connection.messages[-1]['type'] == ('error' if fail else 'completed')
    assert path.exists() != fail
