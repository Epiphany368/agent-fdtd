"""G4-03: CUDA paths under hostile but legal use, and named rejection of unsupported inputs.

Case file: docs/validation/cases/G4-03_cuda_robustness.json. Noncontiguous
epsilon views, duplicate observers, repeated forward and backward calls on one
object, host inputs released after upload, a caller-owned stream, cancellation
mid-run and an exception mid-run are each checked against the same tiny 3D
problem, with allocator accounting after cancellation and after an exception.
"""
import gc
import json
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, FieldMonitor, Material, Monitor, Project, Region,
                       Simulation, Source, SpectrumSettings, Structure)
import fdtd  # after torchfdtd, whose import restores the grad mode that importing fdtd first would disable
from torchfdtd.boundaries import YeeGrid
from torchfdtd.cuda_kernels import FusedYeeCUDA, _direct_cuda_view
from torchfdtd.solver import voxelize

CASE = json.loads((Path(__file__).resolve().parents[1] / 'docs' / 'validation' / 'cases' / 'G4-03_cuda_robustness.json').read_text(encoding='utf-8'))
F = CASE['fixture']
FP64 = (CASE['acceptance']['float64']['rtol'], CASE['acceptance']['float64']['atol'])
LEAK_TOLERANCE_BYTES = CASE['acceptance']['allocator']['tolerance_bytes']

pytestmark = pytest.mark.cuda


def gpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


def tiny(backend='cuda', kernel='fused', monitor='fused', *, plane=True, duplicate=False, snapshot_interval=10000):
    r = Region(dimension='3d', size=tuple(F['size_um']), mesh=F['mesh_um'], pml_cells=F['pml_cells'], steps=F['steps'],
               precision='float64', backend=backend, cuda_kernel=kernel, cuda_monitor_kernel=monitor, snapshot_interval=snapshot_interval)
    point = dict(center=tuple(F['point_monitor_um']), component='Ez')
    monitors = [Monitor(id='point', **point)]
    if plane:
        monitors.append(FieldMonitor(id='plane', normal='x', center=(F['plane_x_um'], 0, 0), size=(0, *F['plane_size_um']),
                                     spectrum=SpectrumSettings(sampling='frequency', frequency_points=F['plane_frequency_points'], apodization='none')))
    if duplicate:
        monitors = monitors + [m.model_copy(update={'id': m.id + '-again'}) for m in monitors]
    return Project(region=r, materials=[Material(name='glass', index=F['sphere_index'])],
                   structures=[Structure(kind='sphere', center=tuple(F['sphere_center_um']), radius=F['sphere_radius_um'], material='glass')],
                   sources=[Source(center=tuple(F['source_um']), component='Ez', wavelength=F['wavelength_um'], pulse_cycles=1)],
                   monitors=monitors)


def host_epsilon():
    epsilon, _ = voxelize(tiny('cpu', 'torch', 'torch', plane=False))
    return np.array(epsilon, dtype=np.float64)


def adjoint(project, epsilon, **options):
    result = DifferentiableSimulation(project, AdjointOptions(checkpoints=F['adjoint_checkpoints'], **options))(epsilon)
    gradient, = torch.autograd.grad(result.signals.square().sum(), epsilon)
    return result.signals.detach(), gradient


def assert_same_forward(a, b):
    for key in ('electric', 'magnetic', 'signals', 'frames', 'frame_steps'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)
    for x, y in zip(a.frequency_fields, b.frequency_fields):
        np.testing.assert_array_equal(x['fields'], y['fields'])
        np.testing.assert_array_equal(x['flux'], y['flux'])


def settle():
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize('layout', ['transposed', 'sliced'])
def test_noncontiguous_epsilon_views_match_contiguous(layout):
    gpu()
    p = tiny('cpu', plane=False)
    host = host_epsilon()
    contiguous = torch.as_tensor(host, device='cuda').requires_grad_(True)
    if layout == 'transposed':
        view = torch.as_tensor(np.ascontiguousarray(host.transpose(2, 1, 0)), device='cuda').permute(2, 1, 0)
    else:
        big = torch.zeros((2 * host.shape[0], *host.shape[1:]), device='cuda', dtype=torch.float64)
        big[::2] = torch.as_tensor(host, device='cuda')
        view = big[::2]
    assert not view.is_contiguous() and view.shape == contiguous.shape
    view.requires_grad_(True)
    expected_signals, expected_gradient = adjoint(p, contiguous)
    signals, gradient = adjoint(p, view)
    assert torch.equal(signals, expected_signals)
    assert gradient.shape == view.shape and torch.equal(gradient, expected_gradient)
    assert float(expected_gradient.norm()) > 0


def test_fused_kernel_rejects_noncontiguous_grid_arrays():
    gpu()
    import cupy
    old = torch.get_default_dtype()
    try:
        fdtd.set_backend('torch.cuda.float64')
        fdtd.backend.float = torch.float64
        grid = YeeGrid(tiny().region)
        grid.E = grid.E.transpose(0, 1)
        with pytest.raises(ValueError, match='Fused CUDA arrays must be contiguous'):
            FusedYeeCUDA(grid)
        with pytest.raises(ValueError, match='Direct CUDA views require contiguous'):
            _direct_cuda_view(cupy, grid.H[::2])
        with pytest.raises(ValueError, match='Direct CUDA views require contiguous'):
            _direct_cuda_view(cupy, grid.H.cpu())
    finally:
        fdtd.set_backend('numpy')
        torch.set_default_dtype(old)


@pytest.mark.parametrize('kernel,monitor,graph', [('torch', 'torch', False), ('fused', 'fused', True)])
def test_duplicate_observers_record_identical_copies(kernel, monitor, graph):
    gpu()
    single = Simulation(tiny('cuda', kernel, monitor)).run(cuda_graph=graph)
    doubled = Simulation(tiny('cuda', kernel, monitor, duplicate=True)).run(cuda_graph=graph)
    assert doubled.signals.shape == (F['steps'], 2) and len(doubled.frequency_fields) == 2
    np.testing.assert_array_equal(doubled.signals[:, 0], doubled.signals[:, 1])
    np.testing.assert_array_equal(doubled.signals[:, 0], single.signals[:, 0])
    first, second = (doubled.field_monitor(name) for name in ('plane', 'plane-again'))
    np.testing.assert_array_equal(first['fields'], second['fields'])
    np.testing.assert_array_equal(first['fields'], single.field_monitor('plane')['fields'])
    np.testing.assert_array_equal(first['flux'], second['flux'])
    assert np.max(np.abs(single.signals)) > 0


@pytest.mark.parametrize('backward', ['torch', 'fused'])
def test_duplicate_point_monitor_gradients_accumulate(backward):
    gpu()
    epsilon = torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True)
    signals, gradient = adjoint(tiny('cpu', plane=False), epsilon, backward_kernel=backward)
    doubled, accumulated = adjoint(tiny('cpu', plane=False, duplicate=True), epsilon, backward_kernel=backward)
    assert torch.equal(doubled[:, 0], doubled[:, 1]) and torch.equal(doubled[:, 0], signals[:, 0])
    torch.testing.assert_close(accumulated, 2 * gradient, rtol=FP64[0], atol=FP64[1] * float(gradient.abs().max()))
    assert float(gradient.norm()) > 0


@pytest.mark.parametrize('kernel,monitor,graph', [('torch', 'torch', False), ('fused', 'fused', True)])
def test_repeated_runs_of_one_simulation_object_agree(kernel, monitor, graph):
    gpu()
    simulation = Simulation(tiny('cuda', kernel, monitor))
    first = simulation.run(cuda_graph=graph)
    second = simulation.run(cuda_graph=graph)
    third = simulation.run(cuda_graph=not graph)
    assert_same_forward(first, second)
    assert_same_forward(first, third)


@pytest.mark.parametrize('backward', ['torch', 'fused'])
def test_repeated_forward_and_backward_on_one_model(backward):
    gpu()
    p = tiny('cpu', plane=False)
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=F['adjoint_checkpoints'], backward_kernel=backward))
    epsilon = torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True)
    first, second = model(epsilon), model(epsilon)
    assert torch.equal(first.signals, second.signals)
    single, = torch.autograd.grad(first.signals.square().sum(), epsilon, retain_graph=True)
    again, = torch.autograd.grad(first.signals.square().sum(), epsilon, retain_graph=True)
    other, = torch.autograd.grad(second.signals.square().sum(), epsilon)
    torch.testing.assert_close(again, single, rtol=FP64[0], atol=FP64[1] * float(single.abs().max()))
    torch.testing.assert_close(other, single, rtol=FP64[0], atol=FP64[1] * float(single.abs().max()))
    assert epsilon.grad is None
    (2 * first.signals.square().sum()).backward()
    torch.testing.assert_close(epsilon.grad, 2 * single, rtol=FP64[0], atol=FP64[1] * float(single.abs().max()))
    assert float(single.norm()) > 0


def test_host_input_released_after_upload_still_runs():
    gpu()
    p = tiny('cpu', plane=False)
    expected_signals, expected_gradient = adjoint(p, torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True))
    host = host_epsilon()
    staging = torch.from_numpy(host)
    device = staging.cuda().requires_grad_(True)
    finalizer = weakref.ref(host)
    del host, staging
    gc.collect()
    assert finalizer() is None, 'the host array is still referenced after the upload'
    signals, gradient = adjoint(p, device)
    assert torch.equal(signals, expected_signals) and torch.equal(gradient, expected_gradient)


@pytest.mark.parametrize('kernel,monitor,graph', [('torch', 'torch', False), ('torch', 'torch', True), ('fused', 'fused', False), ('fused', 'fused', True)])
def test_caller_stream_gives_bitwise_identical_forward(kernel, monitor, graph):
    gpu()
    expected = Simulation(tiny('cuda', kernel, monitor)).run(cuda_graph=graph)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        got = Simulation(tiny('cuda', kernel, monitor)).run(cuda_graph=graph)
    torch.cuda.current_stream().wait_stream(stream)
    stream.synchronize()
    assert_same_forward(got, expected)


@pytest.mark.parametrize('backward', ['torch', 'fused'])
def test_caller_stream_gives_identical_adjoint(backward):
    gpu()
    p = tiny('cpu', plane=False)
    expected_signals, expected_gradient = adjoint(p, torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True), backward_kernel=backward)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        signals, gradient = adjoint(p, torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True), backward_kernel=backward)
    torch.cuda.current_stream().wait_stream(stream)
    stream.synchronize()
    assert torch.equal(signals, expected_signals)
    torch.testing.assert_close(gradient, expected_gradient, rtol=FP64[0], atol=FP64[1] * float(expected_gradient.abs().max()))


@pytest.mark.parametrize('kernel,monitor,graph', [('torch', 'torch', False), ('torch', 'torch', True), ('fused', 'fused', False), ('fused', 'fused', True)])
def test_cancellation_mid_run_releases_every_allocation(kernel, monitor, graph):
    gpu()
    project = tiny('cuda', kernel, monitor, snapshot_interval=F['cancel_after_steps'])
    Simulation(project).run(cuda_graph=graph)  # warm-up: kernel modules, allocator pools
    baseline = settle()
    event = threading.Event()
    seen = []

    def progress(state):
        seen.append(state['step'])
        event.set()

    result = Simulation(project).run(progress=progress, cancel=event, cuda_graph=graph)
    assert result.summary['cancelled'] and result.summary['termination_reason'] == 'cancelled'
    assert 0 < result.summary['steps'] < F['steps'] and seen == [result.summary['steps']]
    assert result.signals.shape[0] == result.summary['steps']
    del result
    leaked = settle() - baseline
    assert leaked <= LEAK_TOLERANCE_BYTES, f'{leaked} bytes remain allocated after a cancelled run'


@pytest.mark.parametrize('kernel,monitor,graph', [('torch', 'torch', False), ('torch', 'torch', True), ('fused', 'fused', False), ('fused', 'fused', True)])
def test_exception_mid_run_releases_every_allocation(kernel, monitor, graph):
    gpu()
    project = tiny('cuda', kernel, monitor, snapshot_interval=F['cancel_after_steps'])
    Simulation(project).run(cuda_graph=graph)
    baseline = settle()

    def progress(state):
        raise RuntimeError('injected mid-run failure')

    with pytest.raises(RuntimeError, match='injected mid-run failure'):
        Simulation(project).run(progress=progress, cuda_graph=graph)
    leaked = settle() - baseline
    assert leaked <= LEAK_TOLERANCE_BYTES, f'{leaked} bytes remain allocated after an exception'
    # The engine lock and the process-wide fdtd backend must be restored for the next run.
    assert fdtd.backend.__class__.__name__.startswith('Numpy')
    assert_same_forward(Simulation(project).run(cuda_graph=graph), Simulation(project).run(cuda_graph=graph))


@pytest.mark.parametrize('backward', ['torch', 'fused'])
def test_exception_in_backward_releases_every_allocation(backward):
    gpu()
    p = tiny('cpu', plane=False)
    epsilon = torch.as_tensor(host_epsilon(), device='cuda').requires_grad_(True)
    adjoint(p, epsilon, backward_kernel=backward)
    baseline = settle()
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=F['adjoint_checkpoints'], backward_kernel=backward))
    result = model(epsilon)
    with pytest.raises(RuntimeError, match='Higher-order derivatives are not supported'):
        torch.autograd.grad(result.signals.square().sum(), epsilon, create_graph=True)
    del result, model
    leaked = settle() - baseline
    assert leaked <= LEAK_TOLERANCE_BYTES, f'{leaked} bytes remain allocated after a failed backward'


def test_unsupported_inputs_are_rejected_by_name():
    gpu()
    p = tiny('cpu', plane=False)
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=F['adjoint_checkpoints']))
    host = host_epsilon()
    good = torch.as_tensor(host, device='cuda')
    with pytest.raises(ValueError, match='real float32 or float64 torch Tensor'):
        model(host)
    with pytest.raises(ValueError, match='real float32 or float64 torch Tensor'):
        model(good.half())
    with pytest.raises(ValueError, match='real float32 or float64 torch Tensor'):
        model(good.to(torch.complex128))
    with pytest.raises(ValueError, match='epsilon dtype must match the project precision'):
        model(good.float())
    with pytest.raises(ValueError, match='epsilon shape must match the scene grid'):
        model(good[1:])
    with pytest.raises(ValueError, match='finite epsilon >= 1'):
        model(good * 0.5)
    nan = good.clone()
    nan[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite epsilon >= 1'):
        model(nan)
    with pytest.raises(ValueError, match='cuda_graph_steps must be an integer'):
        Simulation(tiny()).run(cuda_graph_steps='4')
    with pytest.raises(ValueError, match='Checkpoint storage must be'):
        AdjointOptions(storage='cloud')
