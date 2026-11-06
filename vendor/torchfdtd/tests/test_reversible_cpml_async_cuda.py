"""Public recorded CPML async transport, complex materials and cleanup."""
import gc
import weakref

import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation,
                      ReversibleCPMLOptions, ReversibleCPMLSimulation)
from test_reversible_cpml import fixture, effective, relative

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def cuda_fixture(complex_diagonal):
    project, base, fixed = fixture(48)
    if complex_diagonal:
        for axis in (0, 1):
            for face in project.region.boundaries.pair(axis):
                face.kind = 'bloch'
        project.region.bloch_phase = (.31, -.47, 0.)
        project.region.cuda_kernel = 'fused'
        base = base[..., None].repeat(1, 1, 1, 3)
        fixed = fixed[..., None].repeat(1, 1, 1, 3)
        base += torch.tensor([0., .13, .29])
        fixed += torch.tensor([0., .07, .11])
        for source in project.sources:
            source.kind = 'plane'
            source.normal = 'z'
            source.size = (*project.region.actual_size[:2], 0.)
            source.center = (0., 0., source.center[2])
    return project, base.cuda(), fixed.cuda()


@pytest.mark.parametrize('complex_diagonal', [False, True])
def test_public_async_cpml_matches_sync_checkpoint_and_releases_owners(complex_diagonal):
    torch.set_num_threads(1)
    project, base, fixed = cuda_fixture(complex_diagonal)
    options = ReversibleCPMLOptions(trace_storage='cpu', trace_transfers='async', trace_chunk_steps=7)
    model = ReversibleCPMLSimulation(project, options)
    parameter = base.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = model(parameter, fixed_epsilon=fixed)
    context = actual.signals.grad_fn
    refs = [weakref.ref(context.system), weakref.ref(context.transport),
            *[weakref.ref(t) for t in context.saved_tensors],
            *[weakref.ref(t) for slot in context.transport.slots for t in (slot.cpu, slot.gpu)]]
    archive = context.saved_tensors[1].clone()
    assert context.transport.state == 'complete'
    del context
    dtype = torch.complex64 if complex_diagonal else torch.float32
    generator = torch.Generator().manual_seed(181)
    seeds = [torch.randn((4, 48), dtype=dtype, generator=generator).cuda().T for _ in range(2)]
    assert all(not seed.is_contiguous() for seed in seeds)
    gradient, = torch.autograd.grad(actual.signals.abs().square().mean(), parameter, retain_graph=True)
    gradients = [torch.autograd.grad(actual.signals, parameter, seed, retain_graph=True)[0] for seed in seeds]
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak <= actual.report['gpu_reservation_bytes']
    synchronous_parameter = base.clone().requires_grad_()
    synchronous = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(trace_storage='cpu'))(
        synchronous_parameter, fixed_epsilon=fixed)
    torch.testing.assert_close(archive, synchronous.signals.grad_fn.saved_tensors[1], rtol=0, atol=0)
    torch.testing.assert_close(actual.signals, synchronous.signals, rtol=0, atol=0)
    sync_gradient, = torch.autograd.grad(synchronous.signals.abs().square().mean(), synchronous_parameter)
    torch.testing.assert_close(gradient, sync_gradient, rtol=0, atol=0)
    reference_parameter = base.clone().requires_grad_()
    reference = DifferentiableSimulation(project, AdjointOptions(checkpoints=3, backward_kernel='fused'))(
        effective(reference_parameter, fixed, model.interior_z))
    torch.testing.assert_close(actual.signals, reference.signals, rtol=0, atol=0)
    expected, = torch.autograd.grad(reference.signals.abs().square().mean(), reference_parameter, retain_graph=True)
    errors = [relative(gradient, expected)]
    for seed, grad in zip(seeds, gradients):
        ref, = torch.autograd.grad(reference.signals, reference_parameter, seed, retain_graph=True)
        errors.append(relative(grad, ref))
    assert max(errors) < 1e-4
    assert gradient[:, :, :4].count_nonzero() == gradient[:, :, 15:].count_nonzero() == 0
    assert actual.report['trace_reusable_event_count'] == 8
    assert actual.report['trace_chunk_steps'] == 7
    assert actual.report['trace_item_bytes'] == (8 if complex_diagonal else 4)
    assert actual.report['material_components'] == (3 if complex_diagonal else 1)
    assert actual.report['backward_calls'] == 3
    print(dict(complex_diagonal=complex_diagonal, gradient_relative_l2=errors,
               torch_peak_increment_bytes=peak, reservation=actual.report['gpu_reservation_bytes'],
               reconstruction_relative_l2=actual.report['last_backward']['initial_relative_l2']))
    enabled = gc.isenabled()
    gc.disable()
    try:
        del actual
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()


def test_public_async_forward_failure_drains_and_backward_failure_can_retry(monkeypatch):
    import torchfdtd.reversible_cpml as public
    import torchfdtd.reversible_cpml_kernels as kernels
    import torchfdtd.reversible_trace as transport
    project, base, fixed = cuda_fixture(False)
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        trace_storage='cpu', trace_transfers='async', trace_chunk_steps=7))
    advance, close = public._advance_recorded, transport.AsyncBoundaryTrace.close
    closed = []
    def record_close(self):
        close(self)
        closed.append((self.state, len(self.slots)))
    def fail_forward(system, step, *args):
        advance(system, step, *args)
        if step == 8:
            raise RuntimeError('injected forward failure')
    monkeypatch.setattr(transport.AsyncBoundaryTrace, 'close', record_close)
    monkeypatch.setattr(public, '_advance_recorded', fail_forward)
    with pytest.raises(RuntimeError, match='injected forward failure'):
        # A differentiable input: a no-grad call would run forward only, without a transport.
        model(base.clone().requires_grad_(), fixed_epsilon=fixed)
    assert closed == [('closed', 0)]
    monkeypatch.setattr(public, '_advance_recorded', advance)
    parameter = base.clone().requires_grad_()
    result = model(parameter, fixed_epsilon=fixed)
    original = kernels.InteriorReconstruction.step
    def fail_backward(self, n, frame, **kwargs):
        original(self, n, frame, **kwargs)
        if n == 33:
            raise RuntimeError('injected backward failure')
    monkeypatch.setattr(kernels.InteriorReconstruction, 'step', fail_backward)
    with pytest.raises(RuntimeError, match='injected backward failure'):
        torch.autograd.grad(result.signals.abs().square().mean(), parameter, retain_graph=True)
    assert not result.signals.grad_fn.transport.reading
    assert result.signals.grad_fn.transport.state == 'complete'
    assert not result.signals.grad_fn.lock.locked()
    monkeypatch.setattr(kernels.InteriorReconstruction, 'step', original)
    recovered, = torch.autograd.grad(result.signals.abs().square().mean(), parameter)
    reference_parameter = base.clone().requires_grad_()
    reference = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))(
        effective(reference_parameter, fixed, model.interior_z))
    expected, = torch.autograd.grad(reference.signals.abs().square().mean(), reference_parameter)
    assert relative(recovered, expected) < 1e-4
