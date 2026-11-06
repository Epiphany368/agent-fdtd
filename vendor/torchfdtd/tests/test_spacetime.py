"""The tiled operator and transpose must include halo and CPML ownership."""
import pytest
import torch
import numpy as np

from torchfdtd.differentiable import _System
from torchfdtd.spacetime import SlabBlockOperator
from test_differentiable import project, gpu
from torchfdtd import (Source, Region, StreamedSimulation, StreamedAdjointOptions,
                         DifferentiableSimulation, smooth_sphere_epsilon)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('local_checkpoints', [0,2])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('periodic,depth,diagonal', [(False, 1, False), (False, 3, True), (True, 3, True), (True, 9, False)])
def test_block_and_transpose_match_resident(device, local_checkpoints, precision, periodic, depth, diagonal):
    if device == 'cuda':gpu()
    p = project('3d', precision=precision, steps=max(10, depth+2), periodic=periodic)
    torch.manual_seed(608)
    epsilon = 1.5 + torch.rand(p.region.shape + ((3,) if diagonal else ()), dtype=getattr(torch, precision))
    host = _System(p, epsilon, prepare_updates=False)
    state = tuple(torch.randn_like(s)*.01 for s in host.state())
    endpoint = tuple(torch.randn_like(s)*.02 for s in host.state())
    weights = torch.randn(depth, len(host.monitors), dtype=epsilon.dtype)
    operator = SlabBlockOperator(host, 5, device,local_checkpoints=local_checkpoints)
    result, signals = operator.forward(epsilon, state, 1, depth)
    got_bar, got_gradient = operator.transpose(epsilon, state, 1, depth, endpoint, weights)

    differentiable_eps = epsilon.clone().requires_grad_()
    inputs = tuple(s.clone().requires_grad_() for s in state)
    current = inputs
    observed = []
    for j in range(depth):
        current = host.reference_step(current, j+1, differentiable_eps)
        observed.append(host.observe(current))
    expected_signals = torch.stack(observed)
    loss = (expected_signals*weights).sum() + sum((s*b).sum() for s,b in zip(current, endpoint))
    expected = torch.autograd.grad(loss, (*inputs, differentiable_eps))
    tolerance = dict(rtol=2e-5, atol=2e-6) if precision == 'float32' else dict(rtol=2e-10, atol=2e-12)
    for got, want in zip(result, current):torch.testing.assert_close(got, want, **tolerance)
    torch.testing.assert_close(signals, expected_signals, **tolerance)
    for got, want in zip(got_bar, expected[:-1]):torch.testing.assert_close(got, want, **tolerance)
    torch.testing.assert_close(got_gradient, expected[-1], **tolerance)
    from torchfdtd.streamed_cost import replay_blocks
    assert operator.local_replayed_steps == len(list(operator.tiles(depth)))*replay_blocks(depth,local_checkpoints)
    assert operator.peak_local_checkpoints <= local_checkpoints


def test_temporal_blocks_compose_without_mutating_old_bank():
    p = project(steps=10, periodic=True)
    epsilon = torch.full(p.region.shape, 1.6, dtype=torch.float64)
    host = _System(p, epsilon)
    operator = SlabBlockOperator(host, 3, 'cpu')
    initial = tuple(s.clone() for s in host.state())
    first, a = operator.forward(epsilon, initial, 0, 3)
    second, b = operator.forward(epsilon, first, 3, 5)
    whole, c = operator.forward(epsilon, initial, 0, 8)
    for s in initial:assert torch.count_nonzero(s) == 0
    for actual, expected in zip(second, whole):torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat((a,b)), c, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('checkpoints', [0, 2])
def test_streamed_geometry_backward_across_blocks(device, checkpoints):
    if device == 'cuda':gpu()
    p = project(steps=13, periodic=True)
    radius = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
    epsilon = smooth_sphere_epsilon(p.region, radius, inside=2.7, width=.09, yee=True)
    reference = DifferentiableSimulation(p).reference(epsilon)
    expected, = torch.autograd.grad(reference.square().sum(), radius, retain_graph=True)
    simulation = StreamedSimulation(p, StreamedAdjointOptions(slab_width=5, temporal_depth=3,
                                                              checkpoints=checkpoints, device=device))
    result = simulation(epsilon)
    got, = torch.autograd.grad(result.signals.square().sum(), radius)
    torch.testing.assert_close(result.signals, reference, rtol=2e-11, atol=1e-12)
    torch.testing.assert_close(got, expected, rtol=2e-10, atol=1e-12)
    assert result.report['peak_block_checkpoints'] <= checkpoints
    assert result.report['spatial_streaming']


@pytest.mark.parametrize('periodic', [False, True])
def test_nonuniform_plane_source_and_cpml_transpose(periodic):
    gpu()
    p = project('3d', steps=10, periodic=periodic)
    nodes = []
    for length, count in zip(p.region.size, p.region.shape):
        widths = np.linspace(.85, 1.15, count)
        points = np.r_[0., np.cumsum(widths/widths.sum()*length)]-length/2
        nodes.append(tuple(points))
    data = p.region.model_dump()
    data.update(mesh_type='explicit', mesh_coordinates=tuple(nodes), material_sampling='yee')
    p.region = Region.model_validate(data)
    p.sources = [Source(kind='plane', normal='y', component='Ez', center=(0, 0, 0), size=(.5, 0, .5),
                        pulse='continuous', wavelength=1.1)]
    epsilon = torch.full(p.region.shape, 1.6, dtype=torch.float64, requires_grad=True)
    reference = DifferentiableSimulation(p).reference(epsilon)
    expected, = torch.autograd.grad(reference.square().sum(), epsilon)
    result = StreamedSimulation(p, StreamedAdjointOptions(slab_width=5, temporal_depth=2))(epsilon)
    got, = torch.autograd.grad(result.signals.square().sum(), epsilon)
    torch.testing.assert_close(result.signals, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-11)


def test_streamed_admission_precedes_state_allocation(monkeypatch):
    import torchfdtd.streamed as module
    p = project(steps=10)
    epsilon = torch.full(p.region.shape, 1.6, dtype=torch.float64)
    def unexpected(*args, **kwargs):raise AssertionError('Allocated before admission')
    monkeypatch.setattr(module, '_System', unexpected)
    sim = StreamedSimulation(p, StreamedAdjointOptions(device='cpu', host_budget_bytes=1))
    with pytest.raises(ValueError, match='host budget'):sim(epsilon)
    monkeypatch.setattr(module, 'host_memory', lambda: {'available_bytes': 1})
    sim = StreamedSimulation(p, StreamedAdjointOptions(device='cpu'))
    with pytest.raises(ValueError, match='host budget'):sim(epsilon)
    with pytest.raises(ValueError, match='temporal_depth'):StreamedAdjointOptions(temporal_depth=0)


def test_cuda_tile_storage_does_not_scale_with_slab_count():
    gpu()
    import gc
    peaks = []
    for cells in (32, 64):
        p = project(steps=10, periodic=True)
        data = p.region.model_dump()
        data['size'] = (cells*.1, 1.5, 1.4)
        p.region = Region.model_validate(data)
        epsilon = torch.full(p.region.shape, 1.6, dtype=torch.float64, requires_grad=True)
        model = StreamedSimulation(p, StreamedAdjointOptions(slab_width=4, temporal_depth=2))
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = model(epsilon)
        result.signals.square().sum().backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated()-baseline)
        del result, model, epsilon
        gc.collect()
        assert torch.cuda.memory_allocated() == baseline
    # Allow small allocator rounding differences, but no full-domain GPU bank.
    assert peaks[1] <= peaks[0]*1.05+4096, peaks
