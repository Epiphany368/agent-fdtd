"""Distinct live file-bank identities per streamed adjoint phase.

Bank objects are counted through StateStore.banks weak references at every
allocation with cyclic GC disabled, so the peaks reflect reference lifetime and
not collector timing. Forward holds at most two banks. Backward holds at most
checkpoints+3, and reaches that bound when the block count allows the full
checkpoint nesting. Local tile checkpoints never create file banks.
"""
import gc

import pytest
import torch

from torchfdtd import StreamedSimulation, StreamedAdjointOptions
from torchfdtd import state_store
from test_bloch_adjoint import scene


class _TrackingStore(state_store.StateStore):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_live_banks = 0
        _TrackingStore.instances.append(self)

    def new_state(self, templates):
        arrays = super().new_state(templates)
        self.peak_live_banks = max(self.peak_live_banks, len(self.banks))
        return arrays


@pytest.fixture
def stores(monkeypatch):
    _TrackingStore.instances.clear()
    monkeypatch.setattr(state_store, 'StateStore', _TrackingStore)
    yield _TrackingStore.instances
    _TrackingStore.instances.clear()


def _options(directory, depth, checkpoints, local):
    return StreamedAdjointOptions(device='cpu', state_storage='disk', state_directory=directory,
                                  disk_budget_bytes=64 * 1024 ** 2, slab_width=8, temporal_depth=depth,
                                  checkpoints=checkpoints, local_checkpoints=local)


def _run(tmp_path, steps, depth, checkpoints, local, backward_passes=2):
    p = scene()
    p.region.steps = steps
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = StreamedSimulation(p, _options(tmp_path, depth, checkpoints, local))(eps)
        gradients = []
        for _ in range(backward_passes):
            gradient, = torch.autograd.grad(result.signals.abs().square().sum(), eps, retain_graph=True)
            gradients.append(gradient)
    finally:
        if was_enabled:
            gc.enable()
    return result, gradients


@pytest.mark.parametrize('steps,depth', [(10, 1), (13, 3), (29, 4)])
@pytest.mark.parametrize('checkpoints', [0, 1, 2, 4])
@pytest.mark.parametrize('local', [0, 2])
def test_live_bank_identities_stay_within_bounds(tmp_path, stores, steps, depth, checkpoints, local):
    result, gradients = _run(tmp_path, steps, depth, checkpoints, local)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    forward, *backwards = stores
    assert len(backwards) == 2
    state = result.report['state_bytes']
    assert forward.peak_live_banks == 2 and forward.peak_bytes == 2 * state
    for store in backwards:
        # Identity count and byte peak agree: every bank is one full state.
        assert store.peak_live_banks * state == store.peak_bytes
        assert store.peak_live_banks <= checkpoints + 3
        assert store.closed and store.live_bytes == 0
    assert backwards[0].peak_live_banks == backwards[1].peak_live_banks
    assert not list(tmp_path.iterdir())
    assert result.report['disk_reservation_bytes'] == result.report['state_bank_capacity'] * state


@pytest.mark.parametrize('checkpoints', [0, 1, 2, 4])
def test_backward_bound_is_reached_with_enough_blocks(tmp_path, stores, checkpoints):
    # Ten one-step blocks let every checkpoint slot nest, so the upper bound is exact.
    _run(tmp_path, 10, 1, checkpoints, 0, backward_passes=1)
    forward, backward = stores
    assert forward.peak_live_banks == 2
    assert backward.peak_live_banks == checkpoints + 3
    assert backward.peak_bytes == (checkpoints + 3) * forward.peak_bytes // 2


@pytest.mark.parametrize('checkpoints,steps,depth', [(4, 13, 3), (4, 24, 4)])
def test_fewer_blocks_than_nesting_stay_below_bound(tmp_path, stores, checkpoints, steps, depth):
    # Too few blocks to nest every checkpoint: the peak is below the bound, never above.
    _run(tmp_path, steps, depth, checkpoints, 1, backward_passes=1)
    forward, backward = stores
    assert backward.peak_live_banks < checkpoints + 3
    assert backward.peak_bytes == backward.peak_live_banks * (forward.peak_bytes // 2)


@pytest.mark.parametrize('checkpoints,expected', [(0, 3), (2, 5)])
def test_dispersive_path_keeps_the_same_bank_bounds(tmp_path, stores, checkpoints, expected):
    from torchfdtd import StreamedDispersiveSimulation
    from test_streamed_dispersive import scene as dispersive_scene, inputs
    p = dispersive_scene(bloch=True, steps=13)
    values = inputs(p, 'shared', True)
    options = StreamedAdjointOptions(device='cpu', state_storage='disk', state_directory=tmp_path,
                                     disk_budget_bytes=64 * 1024 ** 2, slab_width=3, temporal_depth=3,
                                     checkpoints=checkpoints, local_checkpoints=1)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = StreamedDispersiveSimulation(p, options)(*values)
        loss = result.signals.abs().square().sum()
        first = torch.autograd.grad(loss, values, retain_graph=True)
        second = torch.autograd.grad(loss, values, retain_graph=True)
    finally:
        if was_enabled:
            gc.enable()
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    forward, *backwards = stores
    state = result.report['state_bytes']
    assert forward.peak_live_banks == 2 and forward.peak_bytes == 2 * state
    for store in backwards:
        assert store.peak_live_banks == expected == store.peak_bytes // state
        assert store.closed and store.live_bytes == 0
    assert result.report['disk_reservation_bytes'] == (checkpoints + 3) * state
    assert not list(tmp_path.iterdir())


def test_local_checkpoints_do_not_create_file_banks(tmp_path, stores):
    counts = {}
    for local in (0, 1, 2):
        stores.clear()
        _run(tmp_path, 13, 3, 2, local, backward_passes=1)
        forward, backward = stores
        counts[local] = (forward.created_banks, backward.created_banks, backward.peak_live_banks)
    assert len(set(counts.values())) == 1, counts
