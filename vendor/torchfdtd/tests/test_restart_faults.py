"""Fault injection against the durable restart journal (G5-07).

One fault per test: a forward or backward interruption, a killed process, a
simulated ENOSPC while a record is written, a device out-of-memory error
inside a slab update, a read or write fault on a file bank, a truncated array
and a corrupted checksum in the newest record, a failed CUDA transfer and a
cancellation through the run's cancel event. After every fault the last valid
record survives, a damaged newest record is rolled back to its predecessor
with the reason named in the report, the resumed run reproduces the
uninterrupted signals and gradient within the declared limits, the journal
carries a terminal state, and scratch banks and device memory return to
their baseline.
"""
import errno
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

import torchfdtd
from torchfdtd import StreamedSimulation
from torchfdtd import streamed_restart
from torchfdtd.differentiable import _System
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd.state_store import DiskArray
from torchfdtd.streamed import StreamCancelled
from torchfdtd.streamed_restart import inspect_journal
from torchfdtd.tile_workspace import TileWorkspace
from test_differentiable import gpu
from test_streamed_restart import options, scene, Interrupted, interrupt, reference

# Declared limits: completion_gates.json proposed_thresholds discrete_dimensionless_fp32.
RTOL, ATOL = 1e-4, 1e-6


def same(actual, expected, device):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    if device == 'cpu':
        # Stricter than the declared limit: CPU resumes are bitwise.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class Baseline:
    """Scratch files and Torch device allocation before a faulty run; both must return."""
    def __init__(self, scratch, device):
        self.scratch, self.device = scratch, device
        gc.collect()
        if device == 'cuda':
            torch.cuda.synchronize()
        self.allocated = torch.cuda.memory_allocated() if device == 'cuda' else 0

    def check(self):
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == self.allocated
        if self.scratch is not None:
            assert not list(self.scratch.iterdir()), 'scratch banks were left behind'


def status(tmp_path):
    return json.loads((tmp_path / 'journal' / 'status.json').read_text())


def pointer(tmp_path, kind):
    path = tmp_path / 'journal' / f'latest-{kind}.json'
    return json.loads(path.read_text()) if path.exists() else None


def finish(resumed, eps, signals, gradient, device, tmp_path):
    same(resumed.signals.detach(), signals, device)
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    same(later, gradient, device)
    journal = tmp_path / 'journal'
    assert resumed.report['restart_state'] == 'completed' and status(tmp_path)['state'] == 'completed'
    assert (journal / 'complete.json').exists() and not (journal / 'owner.json').exists()
    assert not [d for d in journal.iterdir() if d.is_dir()] and not list(journal.glob('*.tmp'))
    return later


def setup(tmp_path, storage='host', device='cpu', **overrides):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, storage, device, tmp_path)
    settings = options(tmp_path, storage, device, **overrides)
    scratch = tmp_path / 'scratch' if storage == 'disk' else None
    if scratch is not None:
        scratch.mkdir(exist_ok=True)
    return p, eps, signals, gradient, settings, scratch


# ---- interruptions -------------------------------------------------------------------
@pytest.mark.parametrize('storage', ['host', 'disk'])
def test_forward_interruption(tmp_path, monkeypatch, storage):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, storage)
    baseline = Baseline(scratch, 'cpu')
    interrupt(monkeypatch, 'forward', 2)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    baseline.check()
    assert pointer(tmp_path, 'forward')['block'] == 2 and pointer(tmp_path, 'forward')['previous'] == 'forward-1'
    failed = status(tmp_path)
    assert failed['state'] == 'failed' and failed['phase'] == 'forward' and failed['reason'].startswith('Interrupted: interrupted forward')
    assert not (tmp_path / 'journal' / 'owner.json').exists()
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 2 and resumed.report['restart_previous_state'] == 'failed'
    assert resumed.report['restart_rollbacks'] == []
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    baseline.check()


@pytest.mark.parametrize('device,storage', [('cpu', 'host'), ('cuda', 'disk')])
def test_backward_interruption(tmp_path, monkeypatch, device, storage):
    if device == 'cuda':
        gpu()
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, storage, device)
    baseline = Baseline(scratch, device)
    result = StreamedSimulation(p, settings)(eps)
    interrupt(monkeypatch, 'transpose', 2)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    del result
    baseline.check()
    assert pointer(tmp_path, 'backward')['block'] == 3 and pointer(tmp_path, 'backward')['previous'] == 'backward-4'
    assert status(tmp_path)['state'] == 'failed' and status(tmp_path)['phase'] == 'backward'
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 'complete'
    finish(resumed, eps, signals, gradient, device, tmp_path)
    assert resumed.report['backward_resumed_from_block'] == 3
    del resumed
    baseline.check()


# ---- process kill --------------------------------------------------------------------
CHILD = '''
import sys, time, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from torchfdtd import StreamedSimulation, StreamedAdjointOptions
from torchfdtd.spacetime import SlabBlockOperator
from test_differentiable import project
phase = sys.argv[5]
original = getattr(SlabBlockOperator, phase)
def slow(self, *args, **kwargs):
    value = original(self, *args, **kwargs)
    time.sleep(0.4)
    return value
setattr(SlabBlockOperator, phase, slow)
p = project(steps=13)
eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
settings = StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=3, checkpoints=1, local_checkpoints=1,
    state_storage='disk', state_directory=sys.argv[3], disk_budget_bytes=64*1024**2, restart_directory=sys.argv[4])
result = StreamedSimulation(p, settings)(eps)
torch.autograd.grad(result.signals.square().sum(), eps)
print('child finished without being killed', flush=True)
'''


@pytest.mark.parametrize('phase', ['forward', 'transpose'])
def test_process_kill(tmp_path, phase):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk')
    source = str(Path(torchfdtd.__file__).resolve().parents[1])
    tests = str(Path(__file__).resolve().parent)
    journal = tmp_path / 'journal'
    kind = 'forward' if phase == 'forward' else 'backward'
    child = subprocess.Popen([sys.executable, '-c', CHILD, source, tests, str(scratch), str(journal), phase],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 120
    seen = None
    while time.time() < deadline and child.poll() is None:
        latest = journal / f'latest-{kind}.json'
        if latest.exists():
            try:
                seen = json.loads(latest.read_text())
            except (json.JSONDecodeError, OSError):
                seen = None
            if seen and seen['kind'] == kind and (seen['block'] >= 2 if kind == 'forward' else seen['block'] <= 3):
                break
        time.sleep(0.05)
    assert child.poll() is None, child.stderr.read().decode('utf8', 'replace')[-2000:]
    child.kill()
    child.wait(timeout=60)
    assert child.returncode != 0
    # The killed process left its lock and its scratch bank behind and no
    # terminal state: the journal reports the run as partial.
    view = inspect_journal(journal)
    # The Windows venv launcher runs the interpreter as its own child, so the lock's pid is not child.pid.
    assert view['state'] == 'partial' and view['owner']['alive'] is False and isinstance(view['owner']['pid'], int)
    assert view['records'][kind]['valid'] and view['records'][kind]['block'] == seen['block']
    leftovers = [d for d in scratch.iterdir()]
    assert len(leftovers) == 1 and leftovers[0].name.startswith('torchfdtd-state-')
    baseline = Baseline(None, 'cpu')
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['restart_previous_state'] == 'partial'
    if kind == 'forward':
        assert resumed.report['forward_resumed_from_block'] == seen['block']
    else:
        assert resumed.report['forward_resumed_from_block'] == 'complete'
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    if kind == 'backward':
        assert resumed.report['backward_resumed_from_block'] == seen['block']
    baseline.check()
    # The resumed process owns only its own scratch; the killed one's bank is left for the operator.
    assert [d for d in scratch.iterdir()] == leftovers
    import shutil
    shutil.rmtree(leftovers[0])


# ---- ENOSPC while writing a record -------------------------------------------------
class NoSpace:
    """A file object whose writes fail with ENOSPC after a number of bytes."""
    def __init__(self, handle, after):
        self.handle, self.after, self.written = handle, after, 0

    def write(self, data):
        if self.written + len(data) > self.after:
            raise OSError(errno.ENOSPC, 'No space left on device')
        self.written += len(data)
        return self.handle.write(data)

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()
        return False


def no_space(monkeypatch, name, *, occurrence=1, after=8):
    real_open = open
    seen = [0]

    def fake_open(path, mode='r', *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if 'w' in mode and Path(path).name == name:
            seen[0] += 1
            if seen[0] == occurrence:
                return NoSpace(handle, after)
        return handle
    monkeypatch.setattr(streamed_restart, 'open', fake_open, raising=False)
    return seen


@pytest.mark.parametrize('phase,name,occurrence', [('forward', 'state-0.bin', 2), ('forward', 'meta.json.tmp', 2),
                                                   ('forward', 'latest-forward.json.tmp', 2), ('backward', 'gradient.bin', 2)])
def test_enospc_while_writing_a_record(tmp_path, monkeypatch, phase, name, occurrence):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk')
    baseline = Baseline(scratch, 'cpu')
    seen = no_space(monkeypatch, name, occurrence=occurrence)
    if phase == 'forward':
        with pytest.raises(OSError) as info:
            StreamedSimulation(p, settings)(eps)
    else:
        result = StreamedSimulation(p, settings)(eps)
        with pytest.raises(OSError) as info:
            torch.autograd.grad(result.signals.square().sum(), eps)
        del result
    assert info.value.errno == errno.ENOSPC and seen[0] == occurrence
    del info
    monkeypatch.undo()
    baseline.check()
    kind = 'forward' if phase == 'forward' else 'backward'
    journal = tmp_path / 'journal'
    # The pointer still names the previous valid record; the failure is named.
    latest = pointer(tmp_path, kind)
    expected = 1 if kind == 'forward' else 4
    assert latest['block'] == expected and (journal / latest['directory']).is_dir()
    failed = status(tmp_path)
    assert failed['state'] == 'failed' and 'No space left on device' in failed['reason']
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['restart_previous_state'] == 'failed' and resumed.report['restart_rollbacks'] == []
    if kind == 'forward':
        assert resumed.report['forward_resumed_from_block'] == 1
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    if kind == 'backward':
        assert resumed.report['backward_resumed_from_block'] == 4
    baseline.check()


# ---- device faults -------------------------------------------------------------------
def fail_after(monkeypatch, owner, method, allowed, error):
    original = getattr(owner, method)
    calls = [0]

    def wrapper(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] > allowed:
            raise error
        return original(self, *args, **kwargs)
    monkeypatch.setattr(owner, method, wrapper)
    return calls


def fail_once_journal_holds(monkeypatch, owner, method, tmp_path, kind, block, error):
    """Raise from the next call of method once the journal's latest record of a kind is at block."""
    original = getattr(owner, method)
    calls = [0]

    def wrapper(self, *args, **kwargs):
        latest = pointer(tmp_path, kind)
        if latest is not None and latest['block'] == block:
            calls[0] += 1
            raise error
        return original(self, *args, **kwargs)
    monkeypatch.setattr(owner, method, wrapper)
    return calls


def test_device_out_of_memory_inside_a_slab_update(tmp_path, monkeypatch):
    gpu()
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk', 'cuda')
    baseline = Baseline(scratch, 'cuda')
    # Four tiles of three steps per block: two blocks complete, the first update of the third raises the allocator's error.
    calls = fail_after(monkeypatch, _System, 'advance', 24, torch.cuda.OutOfMemoryError('CUDA out of memory (injected)'))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    assert calls[0] == 25
    baseline.check()
    assert pointer(tmp_path, 'forward')['block'] == 2
    assert status(tmp_path)['state'] == 'failed' and 'OutOfMemoryError' in status(tmp_path)['reason']
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 2
    finish(resumed, eps, signals, gradient, 'cuda', tmp_path)
    del resumed
    baseline.check()


@pytest.mark.parametrize('direction,method', [('h2d', '_copy_into'), ('d2h', 'to_host')])
def test_cuda_transfer_failure(tmp_path, monkeypatch, direction, method):
    gpu()
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk', 'cuda')
    baseline = Baseline(scratch, 'cuda')
    result = StreamedSimulation(p, settings)(eps)
    calls = fail_once_journal_holds(monkeypatch, TileWorkspace, method, tmp_path, 'backward', 3,
                                    RuntimeError(f'CUDA error: injected {direction} transfer failure'))
    with pytest.raises(RuntimeError, match='injected .* transfer failure'):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    del result
    assert calls[0] == 1
    baseline.check()
    latest = pointer(tmp_path, 'backward')
    assert latest['block'] == 3 and latest['previous'] == 'backward-4'
    assert status(tmp_path)['state'] == 'failed' and 'transfer failure' in status(tmp_path)['reason']
    resumed = StreamedSimulation(p, settings)(eps)
    finish(resumed, eps, signals, gradient, 'cuda', tmp_path)
    assert resumed.report['backward_resumed_from_block'] == latest['block']
    del resumed
    baseline.check()


# ---- bank faults ---------------------------------------------------------------------
@pytest.mark.parametrize('fault,method', [('read', 'index_select'), ('write', 'index_copy_')])
def test_bank_fault(tmp_path, monkeypatch, fault, method):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk')
    baseline = Baseline(scratch, 'cpu')
    result = StreamedSimulation(p, settings)(eps)
    calls = fail_once_journal_holds(monkeypatch, DiskArray, method, tmp_path, 'backward', 3,
                                    OSError(errno.EIO, f'Input/output error (injected bank {fault} fault)'))
    with pytest.raises(OSError) as info:
        torch.autograd.grad(result.signals.square().sum(), eps)
    assert info.value.errno == errno.EIO
    del info, result
    monkeypatch.undo()
    assert calls[0] == 1
    # The failing store closed its banks and removed its directory.
    baseline.check()
    latest = pointer(tmp_path, 'backward')
    assert latest['block'] == 3 and (tmp_path / 'journal' / latest['directory']).is_dir()
    assert status(tmp_path)['state'] == 'failed' and f'bank {fault} fault' in status(tmp_path)['reason']
    resumed = StreamedSimulation(p, settings)(eps)
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    assert resumed.report['backward_resumed_from_block'] == latest['block']
    baseline.check()


# ---- damaged newest record -----------------------------------------------------------
def damage(path, how):
    data = bytearray(path.read_bytes())
    if how == 'truncate':
        path.write_bytes(bytes(data[:len(data) // 2]))
    else:
        data[len(data) // 3] ^= 0x5A
        path.write_bytes(bytes(data))


@pytest.mark.parametrize('how,file', [('truncate', 'state-0.bin'), ('checksum', 'signals.bin')])
def test_damaged_newest_forward_record_rolls_back_to_the_previous(tmp_path, monkeypatch, how, file):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'host')
    interrupt(monkeypatch, 'forward', 3)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    journal = tmp_path / 'journal'
    assert (journal / 'forward-3').is_dir() and (journal / 'forward-2').is_dir() and not (journal / 'forward-1').exists()
    damage(journal / 'forward-3' / file, how)
    view = inspect_journal(journal)
    assert view['records']['forward']['valid'] is (how == 'checksum')  # a truncation is visible without reading
    resumed = StreamedSimulation(p, settings)(eps)
    rollback, = resumed.report['restart_rollbacks']
    assert rollback['kind'] == 'forward' and rollback['record'] == 'forward-3' and rollback['resumed_from'] == 'forward-2'
    if how == 'truncate':
        assert rollback['reason'].startswith(f'{file} is truncated:')
    else:
        assert rollback['reason'] == f'checksum mismatch in {file}'
    assert resumed.report['forward_resumed_from_block'] == 2 and not (journal / 'forward-3').exists()
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)


@pytest.mark.parametrize('transposes,how,file,fallback', [(2, 'checksum', 'adjoint-1.bin', 'backward-4'),
                                                          (2, 'truncate', 'gradient.bin', 'backward-4'),
                                                          (1, 'checksum', 'gradient.bin', None)])
def test_damaged_newest_backward_record_rolls_back(tmp_path, monkeypatch, transposes, how, file, fallback):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'host')
    result = StreamedSimulation(p, settings)(eps)
    interrupt(monkeypatch, 'transpose', transposes)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    journal = tmp_path / 'journal'
    newest = f'backward-{5 - transposes}'
    assert (journal / newest).is_dir() and ((journal / fallback).is_dir() if fallback else True)
    damage(journal / newest / file, how)
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 'complete'
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    rollback, = resumed.report['restart_rollbacks']
    assert rollback['kind'] == 'backward' and rollback['record'] == newest and rollback['resumed_from'] == fallback
    assert (rollback['reason'] == f'checksum mismatch in {file}') if how == 'checksum' else rollback['reason'].startswith(f'{file} is truncated:')
    if fallback:
        assert resumed.report['backward_resumed_from_block'] == 4
    else:
        # No earlier backward record: the valid state is the completed forward, replayed from the end.
        assert 'backward_resumed_from_block' not in resumed.report


def test_damaged_meta_and_missing_checksum_are_named(tmp_path, monkeypatch):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'host')
    interrupt(monkeypatch, 'forward', 3)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    journal = tmp_path / 'journal'
    meta = journal / 'forward-3' / 'meta.json'
    payload = json.loads(meta.read_text())
    del payload['arrays'][1]['sha256']
    meta.write_text(json.dumps(payload))
    resumed = StreamedSimulation(p, settings)(eps)
    rollback, = resumed.report['restart_rollbacks']
    assert rollback['reason'] == 'state-1.bin has no recorded checksum' and resumed.report['forward_resumed_from_block'] == 2
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    # A second journal whose newest record and its fallback are both damaged has no valid record left.
    other = options(tmp_path / 'other', 'host')
    interrupt(monkeypatch, 'forward', 3)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, other)(eps)
    monkeypatch.undo()
    journal = tmp_path / 'other' / 'journal'
    (journal / 'forward-3' / 'meta.json').write_text('{not json')
    (journal / 'forward-2' / 'state-0.bin').write_bytes(b'')
    resumed = StreamedSimulation(p, other)(eps)
    reasons = [(r['record'], r['resumed_from']) for r in resumed.report['restart_rollbacks']]
    assert reasons == [('forward-2', None), ('forward-3', None)]
    assert resumed.report['restart_rollbacks'][0]['reason'].startswith('state-0.bin is truncated: 0 of ')
    assert resumed.report['restart_rollbacks'][1]['reason'].startswith('meta.json is unreadable')
    assert 'forward_resumed_from_block' not in resumed.report
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path / 'other')


# ---- cancellation --------------------------------------------------------------------
@pytest.mark.parametrize('phase,every', [('forward', 1), ('forward', 2), ('transpose', 1)])
def test_cancellation_records_the_boundary(tmp_path, monkeypatch, phase, every):
    p, eps, signals, gradient, settings, scratch = setup(tmp_path, 'disk', restart_every_blocks=every)
    baseline = Baseline(scratch, 'cpu')
    cancel = threading.Event()
    original = getattr(SlabBlockOperator, phase)
    calls = [0]

    def wrapper(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            cancel.set()
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SlabBlockOperator, phase, wrapper)
    journal = tmp_path / 'journal'
    if phase == 'forward':
        with pytest.raises(StreamCancelled, match='forward cancelled at block 3 of 5'):
            StreamedSimulation(p, settings, cancel=cancel)(eps)
        # With every=2 the boundary was not due; the cancellation records it.
        assert pointer(tmp_path, 'forward')['block'] == 3
    else:
        result = StreamedSimulation(p, settings, cancel=cancel)(eps)
        with pytest.raises(StreamCancelled, match='backward cancelled at block 2 of 5'):
            torch.autograd.grad(result.signals.square().sum(), eps)
        del result
        assert pointer(tmp_path, 'backward')['block'] == 2
    monkeypatch.undo()
    baseline.check()
    cancelled = status(tmp_path)
    assert cancelled['state'] == 'cancelled' and cancelled['phase'] == ('forward' if phase == 'forward' else 'backward')
    assert cancelled['block'] == (3 if phase == 'forward' else 2)
    assert inspect_journal(journal)['state'] == 'cancelled' and not (journal / 'owner.json').exists()
    cancel.clear()
    resumed = StreamedSimulation(p, settings, cancel=cancel)(eps)
    assert resumed.report['restart_previous_state'] == 'cancelled'
    assert resumed.report['forward_resumed_from_block'] == (3 if phase == 'forward' else 'complete')
    finish(resumed, eps, signals, gradient, 'cpu', tmp_path)
    if phase == 'transpose':
        assert resumed.report['backward_resumed_from_block'] == 2
    baseline.check()


def test_cancel_event_is_validated_and_without_a_journal_nothing_is_kept(tmp_path):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='is_set'):
        StreamedSimulation(p, options(tmp_path, restart_directory=None), cancel=object())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(StreamCancelled, match='cancelled at block 0 of 5'):
        StreamedSimulation(p, options(tmp_path, restart_directory=None), cancel=cancel)(eps)
    assert not (tmp_path / 'journal').exists()
