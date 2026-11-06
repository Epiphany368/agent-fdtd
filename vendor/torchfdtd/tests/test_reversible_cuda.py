"""Actual native CUDA integration, distinct from external timing fixtures."""
import gc
import weakref

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation, ReversibleSimulation
from test_reversible import fixture, relative


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('size,steps', [(12, 96), (64, 512)])
def test_native_reversible_cuda_gradient_memory_and_owned_states(size, steps):
    p, _ = fixture(steps)
    p.region.size = (.1 * size,) * 3
    p.region._mesh_cache = None
    p.region.backend = 'cuda'
    p.region.cuda_kernel = 'fused'
    base = torch.full(p.region.shape, 1.3, device='cuda')
    base[:, :, size // 2:size // 2 + 2] = 2.25
    epsilon = base.clone().requires_grad_()
    model = ReversibleSimulation(p)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = model(epsilon)
    context = actual.signals.grad_fn
    owner = weakref.ref(context.system)
    grid = weakref.ref(context.system.grid)
    fields = [weakref.ref(v) for v in context.system.state()]
    terminal = [weakref.ref(v) for v in context.saved_tensors[1:]]
    del context
    loss = actual.signals.square().mean()
    gradient, = torch.autograd.grad(loss, epsilon, retain_graph=True)
    generator = torch.Generator(device='cuda').manual_seed(74)
    seed = torch.randn((3, steps), device='cuda', generator=generator).T
    seeded, = torch.autograd.grad(actual.signals, epsilon, seed, retain_graph=True)
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() - before
    assert allocated <= actual.report['gpu_reservation_bytes']
    reference_epsilon = base.clone().requires_grad_()
    reference = DifferentiableSimulation(p, AdjointOptions(checkpoints=4))
    expected = reference(reference_epsilon)
    torch.testing.assert_close(actual.signals, expected.signals, rtol=0, atol=0)
    expected_gradient, = torch.autograd.grad(expected.signals.square().mean(), reference_epsilon, retain_graph=True)
    expected_seeded, = torch.autograd.grad(expected.signals, reference_epsilon, seed)
    assert relative(gradient, expected_gradient) < 2e-4
    assert relative(seeded, expected_seeded) < 2e-4
    assert actual.report['last_backward']['initial_relative_l2'] < 1e-3
    assert actual.report['checkpoint_replays'] == 0
    print(dict(size=size, steps=steps, gradient_relative_l2=relative(gradient, expected_gradient),
               seed_relative_l2=relative(seeded, expected_seeded),
               initial_relative_l2=actual.report['last_backward']['initial_relative_l2'],
               torch_peak_increment_bytes=allocated,
               reserved_solver_bytes=actual.report['gpu_reservation_bytes']))
    enabled = gc.isenabled()
    gc.disable()
    try:
        del actual, loss
        assert owner() is None
        assert grid() is None
        assert all(ref() is None for ref in fields + terminal)
    finally:
        if enabled:
            gc.enable()
