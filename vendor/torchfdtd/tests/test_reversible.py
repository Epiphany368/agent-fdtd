import gc
import weakref

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation, ReversibleOptions, ReversibleSimulation
from torchfdtd.models import Monitor, Project, Region, Source
from torchfdtd.solver import field_axes


def fixture(steps=80):
    r = Region(dimension='3d', size=(.6, .6, .6), mesh=.1, steps=steps,
               courant_factor=.9, backend='cpu', material_sampling='yee',
               boundaries={a + '_' + s: {'kind': 'periodic'} for a in 'xyz' for s in ('min', 'max')})
    axes = field_axes(r, 'Ex')
    def xyz(q):
        return tuple(float(axes[a][q[a]]) for a in range(3))
    sources = [Source(component='Ex', center=xyz((3, 3, 1)), wavelength=.6,
                      time_definition='standard', pulse_length=.4e-15, pulse_offset=.6e-15)]
    monitors = [Monitor(component='Ex', center=xyz((3, 3, 2))),
                Monitor(component='Ex', center=xyz((3, 3, 2))),
                Monitor(component='Hy', center=(0, 0, .1))]
    p = Project(region=r, sources=sources, monitors=monitors)
    base = torch.full(r.shape, 1.3)
    base[:, :, 3:5] = 2.25
    return p, base


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize('steps', [80, 512])
def test_reversible_full_gradient_retained_seeds_and_source_derivative(steps):
    torch.set_num_threads(1)
    p, base = fixture(steps)
    rev = ReversibleSimulation(p)
    ref = DifferentiableSimulation(p, AdjointOptions(checkpoints=4, backward_kernel='torch'))
    e = base.clone().requires_grad_()
    f = base.clone().requires_grad_()
    a, b = rev(e), ref(f)
    torch.testing.assert_close(a.signals, b.signals, rtol=0, atol=0)
    weights = torch.tensor([1., .2, .3])
    loss = (a.signals.square().mean(0) * weights).sum()
    expected_loss = (b.signals.square().mean(0) * weights).sum()
    g, = torch.autograd.grad(loss, e, retain_graph=True)
    expected, = torch.autograd.grad(expected_loss, f, retain_graph=True)
    assert relative(g, expected) < 2e-4
    generator = torch.Generator().manual_seed(37)
    for _ in range(2):
        seed = torch.randn((3, steps), generator=generator).T
        assert not seed.is_contiguous()
        actual, = torch.autograd.grad(a.signals, e, seed, retain_graph=True)
        oracle, = torch.autograd.grad(b.signals, f, seed, retain_graph=True)
        assert relative(actual, oracle) < 2e-4
    # Unlike the matched external fixture, the API permits material changes at
    # a source cell. Its impressed increment is independent of that material.
    direction = torch.zeros_like(base)
    direction[3, 3, 1] = 1
    delta = .003
    def objective(value):
        return float((ref(value).signals.square().mean(0) * weights).sum())
    finite = (objective(base + delta * direction) - objective(base - delta * direction)) / (2 * delta)
    analytic = float((g * direction).sum())
    assert abs(analytic - finite) / max(abs(finite), 1e-12) < .02
    assert a.report['checkpoint_replays'] == 0
    assert a.report['backward_calls'] == 3
    assert a.report['last_backward']['inverse_steps'] == steps
    assert a.report['last_backward']['initial_relative_l2'] < 1e-3


def test_owned_terminal_snapshot_mutation_guards_and_lifetime():
    p, base = fixture(30)
    model = ReversibleSimulation(p)
    e = base.clone().requires_grad_()
    result = model(e)
    owner = weakref.ref(result.signals.grad_fn.system)
    fields = [weakref.ref(t) for t in result.signals.grad_fn.system.state()]
    # A later project mutation cannot change an earlier forward's backward.
    model.project.sources[0].amplitude = 4
    result.signals.square().sum().backward()
    enabled = gc.isenabled()
    gc.disable()
    try:
        del result
        assert owner() is None
        assert all(ref() is None for ref in fields)
    finally:
        if enabled:
            gc.enable()
    changed = model(e)
    with torch.no_grad():
        e.add_(.001)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        changed.signals.sum().backward()
    next_result = model(base.requires_grad_())
    with pytest.raises(RuntimeError, match='first derivatives'):
        torch.autograd.grad(next_result.signals.sum(), base, create_graph=True)


def test_drift_failure_and_concurrent_backward_fail_closed(monkeypatch):
    import torchfdtd.reversible as module
    p, base = fixture(20)
    e = base.requires_grad_()
    result = ReversibleSimulation(p)(e)
    context = result.signals.grad_fn
    context.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='Concurrent backward'):
            torch.autograd.grad(result.signals.sum(), e, retain_graph=True)
    finally:
        context.lock.release()
    original = module._inverse_torch
    def corrupted(system, step):
        original(system, step)
        if step == 0:
            system.grid.E.add_(.1)
    monkeypatch.setattr(module, '_inverse_torch', corrupted)
    with pytest.raises(RuntimeError, match='reconstruction drift'):
        torch.autograd.grad(result.signals.sum(), e)
    assert result.report['backward_calls'] == 0


def test_metadata_and_budget_rejection_precede_fields(monkeypatch):
    import torchfdtd.reversible as module
    p, base = fixture(20)
    def forbidden(*args, **kwargs):
        raise AssertionError('field allocation must not occur')
    monkeypatch.setattr(module, '_System', forbidden)
    with pytest.raises(ValueError, match='budget'):
        ReversibleSimulation(p, ReversibleOptions(resident_budget_bytes=1))(base)
    p.region.mesh_type = 'graded'
    with pytest.raises(ValueError, match='uniform 3D'):
        ReversibleSimulation(p)


@pytest.mark.parametrize('tolerance', [0, float('nan'), .01, True])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match='reconstruction_tolerance'):
        ReversibleOptions(reconstruction_tolerance=tolerance)


@pytest.mark.parametrize('field,value', [('steps', 0), ('courant_factor', 2.),
                                       ('time_step_override', 1.)])
def test_numeric_project_mutation_is_revalidated_before_allocation(monkeypatch, field, value):
    import torchfdtd.reversible as module
    p, base = fixture(20)
    model = ReversibleSimulation(p)
    setattr(model.project.region, field, value)
    def forbidden(*args, **kwargs):
        raise AssertionError('field allocation must not occur')
    monkeypatch.setattr(module, '_System', forbidden)
    with pytest.raises(ValueError):
        model.plan()
    with pytest.raises(ValueError):
        model(base)
