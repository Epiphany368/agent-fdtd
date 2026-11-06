"""Native recorded CPML API with device and synchronous CPU boundary history."""
import gc
import weakref

import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation,
                       ReversibleCPMLOptions, ReversibleCPMLSimulation)
from test_reversible_cpml import fixture, effective, relative


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_public_cpml_cuda_trace_vjp_retained_backward_and_admission(storage):
    project, base, fixed = fixture(96)
    base, fixed = base.cuda(), fixed.cuda()
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(trace_storage=storage))
    parameter = base.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = model(parameter, fixed_epsilon=fixed)
    context = actual.signals.grad_fn
    refs = [weakref.ref(context.system), *[weakref.ref(t) for t in context.system.state()],
            *[weakref.ref(t) for t in context.saved_tensors]]
    assert context.saved_tensors[1].device.type == ('cuda' if storage == 'device' else 'cpu')
    del context
    gradient, = torch.autograd.grad(actual.signals.square().mean(), parameter, retain_graph=True)
    generator = torch.Generator().manual_seed(19)
    seed = torch.randn((4, project.region.steps), generator=generator).cuda().T
    seeded, = torch.autograd.grad(actual.signals, parameter, seed, retain_graph=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-before
    assert peak <= actual.report['gpu_reservation_bytes']
    reference_parameter = base.clone().requires_grad_()
    expected = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))(
        effective(reference_parameter, fixed, model.interior_z))
    torch.testing.assert_close(actual.signals, expected.signals, rtol=0, atol=0)
    oracle, = torch.autograd.grad(expected.signals.square().mean(), reference_parameter, retain_graph=True)
    oracle_seeded, = torch.autograd.grad(expected.signals, reference_parameter, seed)
    assert relative(gradient, oracle) < 1e-4
    assert relative(seeded, oracle_seeded) < 1e-4
    assert gradient[:, :, :4].count_nonzero() == gradient[:, :, 15:].count_nonzero() == 0
    print(dict(storage=storage, gradient_relative_l2=relative(gradient, oracle),
               seeded_relative_l2=relative(seeded, oracle_seeded),
               initial_relative_l2=actual.report['last_backward']['initial_relative_l2'],
               torch_peak_increment_bytes=peak, reserved_solver_bytes=actual.report['gpu_reservation_bytes']))
    enabled = gc.isenabled()
    gc.disable()
    try:
        del actual
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
