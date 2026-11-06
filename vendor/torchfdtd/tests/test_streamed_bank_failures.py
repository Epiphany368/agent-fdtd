"""File-bank cleanup after injected failures while tracebacks stay alive.

Failures are injected at bank allocation, slab reads, forward commit writes,
transpose commit reductions and host transfer waits, in the forward phase and in
a retained backward pass. The caught exception, with its traceback frames that
still reference banks and slab arrays, is held while the checks run. Explicit
store closing must remove every scratch file regardless of those references.
A later backward pass on the same graph must still succeed after the failure.
"""
import gc
import sys

import pytest
import torch

from torchfdtd import StreamedSimulation, StreamedAdjointOptions
from torchfdtd.state_store import StateStore, DiskArray, _Bank
from torchfdtd.tile_workspace import HostTransfer
from test_bloch_adjoint import scene
from test_differentiable import project, gpu


def _options(directory, **overrides):
    values = dict(device='cpu', state_storage='disk', state_directory=directory,
                  disk_budget_bytes=64 * 1024 ** 2, slab_width=8, temporal_depth=3,
                  checkpoints=2, local_checkpoints=1)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


class _Injector:
    """Make the nth call of cls.name raise, and restore the original on demand."""

    def __init__(self, cls, name, nth, error):
        self.cls, self.name, self.nth, self.error = cls, name, nth, error
        self.original = getattr(cls, name)
        self.calls = 0
        self.armed = False

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.armed and self.calls == self.nth:
            # A fresh instance each time: a shared exception object would keep
            # its traceback, and every frame it references, alive after the test.
            raise type(self.error)(*self.error.args)
        return self.original(*args, **kwargs)

    def arm(self):
        self.calls = 0
        self.armed = True
        injector = self

        def wrapper(*args, **kwargs):
            # A plain function binds as a method, so args[0] is the instance.
            return injector(*args, **kwargs)

        setattr(self.cls, self.name, wrapper)

    def disarm(self):
        self.armed = False
        setattr(self.cls, self.name, self.original)


def _frames_reference_banks(error):
    tb = error.__traceback__
    seen = 0
    while tb is not None:
        for value in tb.tb_frame.f_locals.values():
            if isinstance(value, (DiskArray, _Bank, StateStore)):
                seen += 1
        tb = tb.tb_next
    return seen


INJECTIONS = [
    ('allocation', StateStore, 'new_state', 2, MemoryError('injected bank allocation failure')),
    ('write', DiskArray, 'index_copy_', 5, OSError('injected slab write failure')),
    ('transfer', HostTransfer, 'wait', 3, RuntimeError('injected transfer wait failure')),
]
BACKWARD_INJECTIONS = INJECTIONS + [
    ('read', DiskArray, 'index_select', 4, OSError('injected slab read failure')),
    ('reduction', DiskArray, 'index_add_', 3, OSError('injected transpose reduction failure')),
]


@pytest.mark.parametrize('label,cls,name,nth,error', INJECTIONS, ids=[i[0] for i in INJECTIONS])
def test_forward_failure_cleans_scratch_with_traceback_alive(tmp_path, label, cls, name, nth, error):
    p = scene()
    p.region.steps = 13
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    sentinel = tmp_path / 'keep.bin'
    sentinel.write_bytes(b'user data')
    injector = _Injector(cls, name, nth, error)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        injector.arm()
        with pytest.raises(type(error), match='injected'):
            StreamedSimulation(p, _options(tmp_path))(eps)
        caught = sys.exc_info()
        injector.disarm()
        # Scratch is gone even though frames may still hold bank references.
        assert list(tmp_path.iterdir()) == [sentinel]
        assert injector.calls == nth
        del caught
        # The same model runs after the failure and its gradient is finite.
        result = StreamedSimulation(p, _options(tmp_path))(eps)
        gradient, = torch.autograd.grad(result.signals.abs().square().sum(), eps)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        assert result.report['forward_backing_store']['closed']
        assert result.report['backward_backing_store']['live_logical_file_bytes'] == 0
        assert list(tmp_path.iterdir()) == [sentinel]
    finally:
        injector.disarm()
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize('label,cls,name,nth,error', BACKWARD_INJECTIONS, ids=[i[0] for i in BACKWARD_INJECTIONS])
def test_retained_backward_failure_cleans_scratch_and_later_pass_succeeds(tmp_path, label, cls, name, nth, error):
    p = scene()
    p.region.steps = 13
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    injector = _Injector(cls, name, nth, error)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = StreamedSimulation(p, _options(tmp_path))(eps)
        loss = result.signals.abs().square().sum()
        first, = torch.autograd.grad(loss, eps, retain_graph=True)
        injector.arm()
        with pytest.raises(type(error), match='injected') as excinfo:
            torch.autograd.grad(loss, eps, retain_graph=True)
        injector.disarm()
        held = excinfo.value
        assert held.__traceback__ is not None
        if label in ('read', 'write', 'reduction', 'allocation'):
            assert _frames_reference_banks(held) > 0
        store = result.report['backward_backing_store']
        assert store['closed'] and store['live_logical_file_bytes'] == 0
        assert not list(tmp_path.iterdir())
        # Sequential pass after the failure, then a final pass without retention.
        second, = torch.autograd.grad(loss, eps, retain_graph=True)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        third, = torch.autograd.grad(loss, eps)
        torch.testing.assert_close(third, first, rtol=0, atol=0)
        assert not list(tmp_path.iterdir())
        del held, excinfo
    finally:
        injector.disarm()
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize('phase,cls,name,nth,error', [
    ('forward', HostTransfer, 'wait', 3, RuntimeError('injected transfer wait failure')),
    ('backward', DiskArray, 'index_add_', 3, OSError('injected transpose reduction failure')),
], ids=['forward-transfer', 'backward-reduction'])
def test_async_cuda_tiles_drain_after_failure(tmp_path, monkeypatch, phase, cls, name, nth, error):
    gpu()
    from torchfdtd import tile_workspace
    drains = []
    original_drain = tile_workspace.TileWorkspace.drain

    def counting_drain(self):
        drains.append(len(self.events))
        return original_drain(self)

    monkeypatch.setattr(tile_workspace.TileWorkspace, 'drain', counting_drain)
    p = project(steps=12, precision='float32')
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float32, requires_grad=True)
    options = _options(tmp_path, device='cuda', tile_transfers='async', tile_buffers=2,
                       slab_width=4, temporal_depth=2, checkpoints=1)
    injector = _Injector(cls, name, nth, error)
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        if phase == 'forward':
            injector.arm()
            with pytest.raises(type(error), match='injected') as excinfo:
                StreamedSimulation(p, options)(eps)
            injector.disarm()
        else:
            result = StreamedSimulation(p, options)(eps)
            loss = result.signals.square().sum()
            injector.arm()
            with pytest.raises(type(error), match='injected') as excinfo:
                torch.autograd.grad(loss, eps, retain_graph=True)
            injector.disarm()
            assert result.report['backward_backing_store']['closed']
            del result, loss
        # Scratch files are gone while the traceback still references the run.
        assert excinfo.value.__traceback__ is not None
        assert not list(tmp_path.iterdir())
        assert drains, 'workspaces were not drained after the failure'
        assert torch.cuda.current_stream().query(), 'device work still in flight after failure'
        # Dropping the exception releases every device buffer the frames held.
        del excinfo
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == baseline
        # Retry on the same device after the failure.
        result = StreamedSimulation(p, options)(eps)
        gradient, = torch.autograd.grad(result.signals.square().sum(), eps)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        assert not list(tmp_path.iterdir())
        del result, gradient
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == baseline
    finally:
        injector.disarm()
        if was_enabled:
            gc.enable()
