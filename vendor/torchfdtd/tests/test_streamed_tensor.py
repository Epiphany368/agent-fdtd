"""Streamed full-tensor slabs: block Jacobians with 2K halos and resident parity."""
import pytest
import torch

from torchfdtd import AdjointOptions, StreamedAdjointOptions, StreamedSimulation
from torchfdtd.anisotropy import TensorDielectricSimulation, _TensorSystem
from torchfdtd.streamed_tensor import (StreamedTensorSimulation, TensorSlabBlockOperator, _tensor_reservation,
                                       estimate_streamed_tensor_memory)
from test_anisotropy_walls import scene, tensor_field


def devices():
    return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


def streamed_options(device, **overrides):
    values = dict(slab_width=3, temporal_depth=2, checkpoints=2, device=device, gpu_budget_bytes=256*1024**2,
                  host_budget_bytes=2*1024**3)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


@pytest.mark.parametrize('faces,phase,depth', [
    (dict(x='bloch', y='pml', z='pec'), (.43, 0, 0), 2),
    (dict(x='bloch', y='pml', z='pec'), (.43, 0, 0), 5),
    (dict(x='pml', y='periodic', z='pml'), (0, 0, 0), 2),
    (dict(x='pec', y='pml', z='periodic'), (0, 0, 0), 3),
])
@pytest.mark.parametrize('device', devices())
def test_complete_block_jacobian_with_doubled_halo(faces, phase, depth, device):
    p = scene(faces, 'float64', steps=12, bloch_phase=phase)
    generator = torch.Generator().manual_seed(17)
    epsilon = tensor_field(p.region, generator, torch.float64)
    host = _TensorSystem(p, epsilon, prepare_updates=False)
    state = tuple(torch.randn(s.shape, dtype=s.dtype, generator=generator)*.03 for s in host.state())
    endpoint = tuple(torch.randn(s.shape, dtype=s.dtype, generator=generator)*.04 for s in host.state())
    weights = torch.randn(depth, len(host.monitors), dtype=host.field_dtype, generator=generator)
    operator = TensorSlabBlockOperator(host, 3, device, local_checkpoints=1)
    descriptors = list(operator.tiles(depth))
    assert all(len(d[2]) >= 3+2*depth or faces['x'] != 'bloch' for d in descriptors)
    if depth == 5:
        assert any(len(d[2]) > p.region.shape[0] for d in descriptors)  # repeated windings
    actual, signals = operator.forward(epsilon, state, 1, depth)
    bars, gradient = operator.transpose(epsilon, state, 1, depth, endpoint, weights)
    live = epsilon.clone().requires_grad_()
    initial = tuple(s.clone().requires_grad_() for s in state)
    system = _TensorSystem(p, live)
    current = initial
    expected_signals = []
    for j in range(depth):
        current = system.reference_step(current, 1+j, live)
        expected_signals.append(system.observe(current))
    expected_signals = torch.stack(expected_signals)
    torch.testing.assert_close(signals, expected_signals, rtol=1e-11, atol=1e-13)
    for a, b in zip(actual, current):
        torch.testing.assert_close(a, b.detach(), rtol=1e-11, atol=1e-13)
    loss = sum((e.conj()*c).real.sum() for e, c in zip(endpoint, current))+(weights.conj()*expected_signals).real.sum()
    expected = torch.autograd.grad(loss, (*initial, live))
    for a, b in zip(bars, expected[:-1]):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    reference = (expected[-1]+expected[-1].transpose(-1, -2))/2
    torch.testing.assert_close(gradient, reference, rtol=1e-10, atol=1e-12)
    assert gradient.abs().max() > 0 and all(b.abs().max() > 0 for b in bars[:2])


@pytest.mark.parametrize('faces,phase', [
    (dict(x='bloch', y='pml', z='pec'), (.37, 0, 0)),
    (dict(x='pml', y='pml', z='pml'), (0, 0, 0)),
    (dict(x='periodic', y='pec', z='periodic'), (0, 0, 0)),
])
@pytest.mark.parametrize('device', devices())
def test_streamed_matches_resident_signals_and_material_gradient(faces, phase, device):
    p = scene(faces, 'float64', steps=14, bloch_phase=phase)
    generator = torch.Generator().manual_seed(29)
    epsilon = tensor_field(p.region, generator, torch.float64)
    resident = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2), cpml_material='tensor')
    streamed = StreamedTensorSimulation(p, streamed_options(device, temporal_depth=3))
    a = epsilon.clone().requires_grad_()
    b = epsilon.clone().requires_grad_()
    expected = resident(a)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    actual = streamed(b)
    torch.testing.assert_close(actual.signals, expected.signals, rtol=1e-11, atol=1e-13)
    seed = torch.randn(expected.signals.shape, dtype=expected.signals.dtype, generator=generator)
    loss = lambda signals: (signals.conj()*seed).real.sum()+signals.abs().square().sum()
    grad_expected, = torch.autograd.grad(loss(expected.signals), a)
    grad_actual, = torch.autograd.grad(loss(actual.signals), b)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-10, atol=1e-12)
    assert grad_actual.abs().max() > 0
    if device == 'cuda':
        assert torch.cuda.max_memory_allocated() <= actual.report['gpu_reservation_bytes']
    assert actual.report['spatial_streaming'] and actual.report['halo_cells_per_side'] == 6
    assert actual.report['tensor_tile_reservation_bytes'] > 0
    layers = p.region.pml_layers(1, 0)
    if faces['y'] == 'pml':
        assert grad_actual[:, :layers].abs().max() > 0


def test_streamed_tensor_spectrum_matches_resident_online_dft():
    p = scene(dict(x='periodic', y='pml', z='pml'), 'float64', steps=16)
    generator = torch.Generator().manual_seed(3)
    epsilon = tensor_field(p.region, generator, torch.float64)
    frequency = torch.tensor([3.5e14, 4.2e14], dtype=torch.float64)
    resident = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2), cpml_material='tensor')
    streamed = StreamedTensorSimulation(p, streamed_options('cpu'))
    a = epsilon.clone().requires_grad_()
    b = epsilon.clone().requires_grad_()
    expected = resident.spectrum(a, frequency)
    actual = streamed.spectrum(b, frequency)
    torch.testing.assert_close(actual.fields, expected.fields, rtol=1e-11, atol=1e-13)
    grad_expected, = torch.autograd.grad(expected.fields.abs().square().sum(), a)
    grad_actual, = torch.autograd.grad(actual.fields.abs().square().sum(), b)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-10, atol=1e-12)


def test_streamed_tensor_admission_reservation_and_rejections():
    p = scene(dict(x='pml', y='pml', z='pml'), 'float32')
    generator = torch.Generator().manual_seed(4)
    epsilon = tensor_field(p.region, generator, torch.float32)
    options = streamed_options('cpu')
    report = estimate_streamed_tensor_memory(p, options)
    diagonal = _tensor_reservation(p, epsilon, options)
    assert report['halo_cells_per_side'] == 4 and report['host_reservation_bytes'] == diagonal['host_reservation_bytes']
    assert report['max_extended_tile_cells'] == min(3+8, p.region.shape[0])*p.region.shape[1]*p.region.shape[2]
    with pytest.raises(ValueError, match='StreamedTensorSimulation'):
        StreamedSimulation(p, options)(epsilon)
    with pytest.raises(ValueError, match="cpml_material='tensor'"):
        StreamedTensorSimulation(p, options, cpml_material='isotropic')
    model = StreamedTensorSimulation(p, options)
    with pytest.raises(ValueError, match='CPU tensor'):
        model(epsilon.to('meta'))
    biaxial = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float32)).expand(p.region.shape+(3, 3)).contiguous()
    with pytest.raises(ValueError, match='y_min'):
        model(biaxial)
    with pytest.raises(ValueError, match='host budget'):
        StreamedTensorSimulation(p, streamed_options('cpu', host_budget_bytes=1024))(epsilon)
    q = scene(dict(x='pmc', y='pmc', z='pmc'), 'float32')
    with pytest.raises(ValueError, match='PMC'):
        StreamedTensorSimulation(q, options)
