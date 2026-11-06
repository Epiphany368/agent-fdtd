"""Online six-field planes with bounded async complex boundary traces."""
import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiablePlaneSimulation,
                      ReversibleCPMLOptions, ReversibleCPMLPlaneSimulation)
from test_reversible_cpml_planes import fixture, relative


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('complex_fields', [False, True])
def test_async_online_plane_cuda_spectrum_vjp_and_admission(complex_fields):
    torch.set_num_threads(1)
    project, base, fixed = fixture(complex_fields)
    project.region.cuda_kernel = 'fused'
    base, fixed = base.cuda(), fixed.cuda()
    counts = {'incident': (3, 4), 'detector': (3, 4)}
    options = ReversibleCPMLOptions(trace_storage='cpu', trace_transfers='async', trace_chunk_steps=7)
    model = ReversibleCPMLPlaneSimulation(project, options, quadrature_counts=counts)
    frequencies = torch.tensor([.033, .057], dtype=torch.float32, device='cuda') / project.region.time_step
    parameter = base.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    planes = model(parameter, frequencies, fixed_epsilon=fixed, block_size=7)
    fields = torch.stack([plane.fields for plane in planes.values()]) / project.region.time_step
    gradient, = torch.autograd.grad((fields.real + .37 * fields.imag).square().mean(), parameter, retain_graph=True)
    generator = torch.Generator().manual_seed(291)
    seeds = [torch.randn((6, 12, 2, 2), generator=generator, dtype=torch.complex64).cuda().permute(2, 3, 1, 0) for _ in range(2)]
    gradients = [torch.autograd.grad(fields, parameter, seed, retain_graph=True)[0] for seed in seeds]
    assert all(not seed.is_contiguous() for seed in seeds)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    report = next(iter(planes.values())).report
    assert peak <= report['gpu_reservation_bytes']
    a, b = model.interior_z
    ref_parameter = base.clone().requires_grad_()
    effective = fixed.clone()
    effective[:, :, a:b+1] = ref_parameter[:, :, a:b+1]
    reference = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=2, backward_kernel='fused'), quadrature_counts=counts)(
        effective, frequencies, block_size=7)
    expected_fields = torch.stack([plane.fields for plane in reference.values()]) / project.region.time_step
    expected, = torch.autograd.grad((expected_fields.real + .37 * expected_fields.imag).square().mean(), ref_parameter, retain_graph=True)
    errors = [relative(gradient, expected)]
    for seed, grad in zip(seeds, gradients):
        ref, = torch.autograd.grad(expected_fields, ref_parameter, seed, retain_graph=True)
        errors.append(relative(grad, ref))
    assert relative(fields, expected_fields) < 1e-4
    assert max(errors) < 1e-4
    assert gradient[:, :, :a].count_nonzero() == gradient[:, :, b+1:].count_nonzero() == 0
    assert report['output_history_bytes'] == 0
    assert report['observation_history_retained'] is False
    assert report['observation_block_shape'] == [7, len(model.observers)]
    assert report['last_backward']['seed_buffer_shape'] == [7, len(model.observers)]
    assert report['last_backward']['regenerated_seed_blocks'] == 5
    assert report['plane_layout_reservation_bytes'] == model.layout_reservation_bytes
    print(dict(complex_fields=complex_fields, fields_relative_l2=relative(fields, expected_fields),
               gradient_relative_l2=errors, torch_peak_increment_bytes=peak,
               reservation=report['gpu_reservation_bytes'], trace_bytes=report['trace_bytes'],
               observation_block_shape=report['observation_block_shape']))
