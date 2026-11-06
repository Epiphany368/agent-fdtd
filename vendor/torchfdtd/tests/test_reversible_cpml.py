import gc
import weakref

import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, Monitor, Project, Region,
                       ReversibleCPMLOptions, ReversibleCPMLSimulation, Source)
from torchfdtd.solver import field_axes


def fixture(steps=96):
    faces = {f'{a}_{s}': {'kind': 'periodic'} for a in 'xy' for s in ('min', 'max')}
    faces.update(z_min={'kind': 'pml', 'layers': 3, 'kappa': 2, 'alpha': .03},
                 z_max={'kind': 'pml', 'layers': 4, 'kappa': 2, 'alpha': .03})
    region = Region(dimension='3d', size=(.6, .7, 2), mesh=.1, steps=steps,
                    courant_factor=.9, backend='cpu', boundaries=faces, material_sampling='yee')
    def xyz(component, indices):
        axes = field_axes(region, component)
        return tuple(float(axes[a][indices[a]]) for a in range(3))
    sources = [Source(component='Ex', center=xyz('Ex', (2, 3, 5)), wavelength=.6,
                      time_definition='standard', pulse_length=.4e-15,
                      pulse_offset=.6e-15, amplitude=amplitude) for amplitude in (1., .17)]
    monitors = [Monitor(component='Ex', center=xyz('Ex', (2, 3, 7))),
                Monitor(component='Ex', center=xyz('Ex', (2, 3, 7))),
                Monitor(component='Hy', center=xyz('Hy', (2, 3, 13))),
                Monitor(component='Ex', center=xyz('Ex', (2, 3, 15)))]
    project = Project(region=region, sources=sources, monitors=monitors)
    design = torch.full(region.shape, 1.4)
    design[:, :, 9:11] = 2.25
    fixed = torch.full(region.shape, 1.3)
    fixed[:, :, 15:] = 1.7
    return project, design, fixed


def effective(parameter, fixed, interval):
    a, b = interval
    material = fixed.clone()
    material[:, :, a:b+1] = parameter[:, :, a:b+1]
    return material


def relative(actual, expected):
    return float((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_public_cpml_full_interior_vjp_retained_seeds_and_collar_observer(storage):
    torch.set_num_threads(1)
    project, base, fixed = fixture()
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(trace_storage=storage))
    assert model.interior_z == (4, 14)
    parameter = base.clone().requires_grad_()
    reference_parameter = base.clone().requires_grad_()
    actual = model(parameter, fixed_epsilon=fixed)
    expected = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))(
        effective(reference_parameter, fixed, model.interior_z))
    torch.testing.assert_close(actual.signals, expected.signals, rtol=0, atol=0)
    assert max(float(t['psi'].abs().max()) for t in actual.signals.grad_fn.system.segments) > 1e-8
    weights = torch.tensor([1., .2, .3, .4])
    loss = (actual.signals.square().mean(0)*weights).sum()
    reference_loss = (expected.signals.square().mean(0)*weights).sum()
    gradient, = torch.autograd.grad(loss, parameter, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference_loss, reference_parameter, retain_graph=True)
    assert relative(gradient, reference_gradient) < 1e-4
    assert torch.count_nonzero(gradient[:, :, :4]) == 0
    assert torch.count_nonzero(gradient[:, :, 15:]) == 0
    assert abs(float(gradient[2, 3, 5])) > 1e-9
    # A collar-only observation must traverse the untruncated CPML transpose.
    generator = torch.Generator().manual_seed(31)
    for index in range(2):
        seed = torch.randn((4, project.region.steps), generator=generator).T
        if index == 0:
            seed[:, :3] = 0
        assert not seed.is_contiguous()
        g, = torch.autograd.grad(actual.signals, parameter, seed, retain_graph=True)
        ref, = torch.autograd.grad(expected.signals, reference_parameter, seed, retain_graph=True)
        assert relative(g, ref) < 1e-4
    if storage == 'device':
        direction = torch.zeros_like(base)
        direction[2, 3, 5] = 1
        oracle = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))
        def objective(value):
            signals = oracle(effective(value, fixed, model.interior_z)).signals
            return float((signals.square().mean(0)*weights).sum())
        delta = .003
        numerical = (objective(base+delta*direction)-objective(base-delta*direction))/(2*delta)
        assert abs(float(gradient[2, 3, 5])-numerical)/max(abs(numerical), 1e-12) < .02
    assert actual.report['checkpoint_replays'] == 0
    assert actual.report['backward_calls'] == 3
    assert actual.report['last_backward']['initial_relative_l2'] < 1e-4
    assert actual.report['trace_bytes'] == project.region.steps*4*6*7*4
    assert actual.report['terminal_state_bytes'] == 6*6*7*11*4
    assert not actual.report['cpml_primal_inverted']


def test_cpml_long_public_path_reconstructs_without_checkpoint_replay():
    torch.set_num_threads(1)
    project, base, fixed = fixture(2048)
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(trace_storage='cpu'))
    parameter = base.clone().requires_grad_()
    reference_parameter = base.clone().requires_grad_()
    result = model(parameter, fixed_epsilon=fixed)
    reference = DifferentiableSimulation(project, AdjointOptions(checkpoints=4))(
        effective(reference_parameter, fixed, model.interior_z))
    torch.testing.assert_close(result.signals, reference.signals, rtol=0, atol=0)
    actual, = torch.autograd.grad(result.signals.square().mean(), parameter)
    expected, = torch.autograd.grad(reference.signals.square().mean(), reference_parameter)
    assert relative(actual, expected) < 1e-4
    assert result.report['last_backward']['inverse_steps'] == 2048
    assert result.report['last_backward']['initial_relative_l2'] < 1e-4


def test_cpml_owned_trace_lifetime_and_project_background_snapshot():
    project, base, fixed = fixture(20)
    model = ReversibleCPMLSimulation(project)
    parameter = base.requires_grad_()
    result = model(parameter, fixed_epsilon=fixed)
    context = result.signals.grad_fn
    refs = [weakref.ref(context.system), *[weakref.ref(t) for t in context.system.state()],
            *[weakref.ref(t) for t in context.saved_tensors]]
    assert not context.system.epsilon.requires_grad
    del context
    model.project.sources[0].amplitude = 7
    fixed.add_(1)
    result.signals.square().sum().backward()
    assert torch.isfinite(parameter.grad).all()
    enabled = gc.isenabled()
    gc.disable()
    try:
        del result
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()


def test_cpml_drift_mutation_higher_order_and_concurrent_guards(monkeypatch):
    import torchfdtd.reversible_cpml_kernels as kernels
    project, base, fixed = fixture(20)
    parameter = base.requires_grad_()
    model = ReversibleCPMLSimulation(project)
    result = model(parameter, fixed_epsilon=fixed)
    context = result.signals.grad_fn
    context.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='Concurrent backward'):
            torch.autograd.grad(result.signals.sum(), parameter, retain_graph=True)
    finally:
        context.lock.release()
    with pytest.raises(RuntimeError, match='first derivatives'):
        torch.autograd.grad(result.signals.sum(), parameter, create_graph=True, retain_graph=True)
    original = kernels.InteriorReconstruction.step
    def corrupted(self, n, trace):
        original(self, n, trace)
        if n == 0:
            self.system.grid.E[:, :, self.a:self.b+1].add_(.1)
    monkeypatch.setattr(kernels.InteriorReconstruction, 'step', corrupted)
    with pytest.raises(RuntimeError, match='reconstruction drift'):
        torch.autograd.grad(result.signals.sum(), parameter)
    assert result.report['backward_calls'] == 0
    changed = model(parameter, fixed_epsilon=fixed)
    with torch.no_grad():
        changed.signals.grad_fn.saved_tensors[1].add_(.001)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        torch.autograd.grad(changed.signals.sum(), parameter)


def test_cpml_inputs_scope_and_budget_rejected_before_fields(monkeypatch):
    import torchfdtd.reversible_cpml as module
    project, base, fixed = fixture(20)
    def forbidden(*a, **k):
        pytest.fail('field allocation must not occur')
    monkeypatch.setattr(module, '_System', forbidden)
    with pytest.raises(ValueError, match='budget'):
        ReversibleCPMLSimulation(project, ReversibleCPMLOptions(resident_budget_bytes=1))(
            base, fixed_epsilon=fixed)
    with pytest.raises(ValueError, match='must not require gradients'):
        ReversibleCPMLSimulation(project)(base, fixed_epsilon=fixed.requires_grad_())
    fixed.requires_grad_(False)
    with pytest.raises(ValueError, match='contiguous resolved scalar FP32'):
        ReversibleCPMLSimulation(project)(base.double(), fixed_epsilon=fixed)
    changed = ReversibleCPMLSimulation(project)
    changed.project.region.steps = 0
    with pytest.raises(ValueError):
        changed.plan()
    project.region.mesh_type = 'graded'
    with pytest.raises(ValueError, match='uniform 3D FP32'):
        ReversibleCPMLSimulation(project)


@pytest.mark.parametrize('options', [dict(trace_storage='disk'), dict(collar_cells=True),
                                     dict(collar_cells=0), dict(reconstruction_tolerance=.1)])
def test_cpml_invalid_options(options):
    with pytest.raises(ValueError):
        ReversibleCPMLOptions(**options)


def test_cpml_descriptor_interval_and_source_component_index():
    project, base, fixed = fixture()
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(collar_cells=2))
    assert model.interior_z == (5, 13)
    # Sources outside the reconstruction are carried by the recorded halos.
    axes = field_axes(project.region, 'Ex')
    project.sources[0].center = tuple(float(axes[a][(2, 3, 3)[a]]) for a in range(3))
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(collar_cells=2))
    parameter = base.clone().requires_grad_()
    reference_parameter = base.clone().requires_grad_()
    actual = model(parameter, fixed_epsilon=fixed)
    expected = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))(
        effective(reference_parameter, fixed, model.interior_z))
    assert torch.equal(actual.signals, expected.signals)
    gradient, = torch.autograd.grad(actual.signals.square().mean(), parameter)
    reference, = torch.autograd.grad(expected.signals.square().mean(), reference_parameter)
    assert relative(gradient, reference) < 1e-4
