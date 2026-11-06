"""Explicit reconstruction cuts against full-domain checkpointed adjoints."""
import pytest
import torch

from torchfdtd import (AdjointBatchOptions, AdjointExecutionPolicy, AdjointOptions,
                       DifferentiablePlaneSimulation, DifferentiableSimulation,
                       PeriodicLayerResponse, ReversibleCPMLOptions,
                       ReversibleCPMLPlaneSimulation, ReversibleCPMLSimulation)
from torchfdtd.solver import field_axes
from tests.test_reversible_cpml import fixture as point_fixture, relative
from tests.test_reversible_cpml_planes import fixture as plane_fixture
from tests.test_recorded_periodic_response import SPEC, BUDGET, objective


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)]
CUT = (8, 11)


def require_device(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA device required')
        from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
        prepare_cuda_kernels()
    torch.set_num_threads(1)


def material_maps(project, diagonal, device):
    shape = (*project.region.shape, 3) if diagonal else project.region.shape
    fixed = torch.full(shape, 1.4, device=device)
    fixed[:, :, 15:] = 1.7
    if diagonal:
        fixed += torch.tensor([0., .1, .2], device=device)
    base = fixed.clone()
    base[:, :, 9:11] += .5
    return base, fixed


def inside(value):
    return value[:, :, CUT[0]:CUT[1]+1]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('complex_fields', [False, True])
@pytest.mark.parametrize('source_z', [5, 7, 9, 12])
def test_point_interval_forward_vjp_and_source_halos(device, diagonal, complex_fields, source_z):
    require_device(device)
    project, _, _ = point_fixture(64)
    if complex_fields:
        for axis in (0, 1):
            for face in project.region.boundaries.pair(axis):
                face.kind = 'bloch'
        project.region.bloch_phase = (.31, -.47, 0.)
    axes = field_axes(project.region, 'Ex')
    for source in project.sources:
        source.center = tuple(float(axes[a][(2, 3, source_z)[a]]) for a in range(3))
    base, fixed = material_maps(project, diagonal, device)
    wide = ReversibleCPMLSimulation(project)
    narrow = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(interior_z=CUT))
    oracle = DifferentiableSimulation(project, AdjointOptions(checkpoints=3))
    rows = []
    for model in (wide, narrow, oracle):
        value = base.clone().requires_grad_()
        result = model(value) if model is oracle else model(value, fixed_epsilon=fixed)
        gradient, = torch.autograd.grad(result.signals.abs().square().mean(), value)
        rows.append((result.signals.detach(), gradient, result.report))
    assert torch.equal(rows[0][0], rows[1][0])
    torch.testing.assert_close(rows[1][0], rows[2][0], rtol=1e-4, atol=1e-6)
    assert torch.count_nonzero(inside(rows[1][1])) > 0
    for index in (0, 1):
        assert relative(inside(rows[index][1]), inside(rows[2][1])) < 1e-4
    assert relative(inside(rows[0][1]), inside(rows[1][1])) < 1e-4
    assert torch.count_nonzero(rows[1][1][:, :, :CUT[0]]) == 0
    assert torch.count_nonzero(rows[1][1][:, :, CUT[1]+1:]) == 0
    assert rows[1][2]['last_backward']['initial_relative_l2'] < 1e-4
    assert rows[1][2]['trace_bytes'] == rows[0][2]['trace_bytes']
    assert rows[1][2]['terminal_state_bytes']*11 == rows[0][2]['terminal_state_bytes']*4
    assert narrow.interior_z == CUT


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('complex_fields', [False, True])
@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_plane_interval_spectra_vjp_and_trace_storage(device, diagonal, complex_fields, storage):
    require_device(device)
    project, _, _ = plane_fixture(complex_fields)
    base, fixed = material_maps(project, diagonal, device)
    counts = {'incident': (3, 4), 'detector': (3, 4)}
    models = [ReversibleCPMLPlaneSimulation(project,
                  ReversibleCPMLOptions(interior_z=cut, trace_storage=storage),
                  quadrature_counts=counts) for cut in (None, CUT)]
    oracle = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=2),
                                           quadrature_counts=counts)
    frequency = [.033/project.region.time_step, .057/project.region.time_step]
    rows = []
    for model in (*models, oracle):
        value = base.clone().requires_grad_()
        args = {} if model is oracle else {'fixed_epsilon': fixed}
        planes = model(value, frequency, block_size=7, **args)
        fields = torch.stack([plane.fields for plane in planes.values()])/project.region.time_step
        gradient, = torch.autograd.grad((fields.real+.37*fields.imag).square().mean(), value)
        rows.append((fields.detach(), gradient))
    assert torch.equal(rows[0][0], rows[1][0])
    torch.testing.assert_close(rows[1][0], rows[2][0], rtol=1e-4, atol=1e-6)
    for index in (0, 1):
        assert relative(inside(rows[index][1]), inside(rows[2][1])) < 1e-4
    assert relative(inside(rows[0][1]), inside(rows[1][1])) < 1e-4
    assert torch.count_nonzero(rows[1][1][:, :, :CUT[0]]) == 0
    assert torch.count_nonzero(rows[1][1][:, :, CUT[1]+1:]) == 0
    plan = models[1].plan(frequency, device=device, material_components=3 if diagonal else 1,
                          block_size=7)
    wide = models[0].plan(frequency, device=device, material_components=3 if diagonal else 1,
                          block_size=7)
    assert plan['terminal_state_bytes']*11 == wide['terminal_state_bytes']*4


def test_interval_material_gradient_matches_central_difference():
    require_device('cpu')
    project, _, _ = point_fixture(96)
    base, fixed = material_maps(project, False, 'cpu')
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(interior_z=CUT))
    value = base.clone().requires_grad_()
    result = model(value, fixed_epsilon=fixed)
    gradient, = torch.autograd.grad(result.signals.square().mean(), value)
    direction = torch.zeros_like(base)
    direction[2, 3, 9] = 1
    def loss(value):
        return float(model(value, fixed_epsilon=fixed).signals.square().mean())
    delta = .003
    numerical = (loss(base+delta*direction)-loss(base-delta*direction))/(2*delta)
    assert abs(float(gradient[2, 3, 9])-numerical)/max(abs(numerical), 1e-12) < .02


@pytest.mark.parametrize('cut', [(4, 4), (8, 7), (True, 11), (8., 11), [8, 11], (8,), '8,11'])
def test_interval_option_types(cut):
    with pytest.raises(ValueError, match='interior_z'):
        ReversibleCPMLOptions(interior_z=cut)


@pytest.mark.parametrize('cut', [(3, 11), (8, 15), (-1, 2), (18, 20)])
def test_interval_must_fit_cpml_collar(cut):
    project, _, _ = point_fixture(10)
    with pytest.raises(ValueError, match='CPML-free'):
        ReversibleCPMLSimulation(project, ReversibleCPMLOptions(interior_z=cut))


@pytest.mark.parametrize('diagonal', [False, True])
def test_explicit_interval_exterior_mismatch_rejected_before_fields(monkeypatch, diagonal):
    import torchfdtd.reversible_cpml as core
    project, _, _ = point_fixture(10)
    base, fixed = material_maps(project, diagonal, 'cpu')
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(interior_z=CUT))
    monkeypatch.setattr(core, '_System', lambda *a, **k: pytest.fail('Allocated fields before rejection'))
    for z in (0, 7, 12, 19):
        bad = base.clone()
        bad[-1, -1, z] += .01
        with pytest.raises(ValueError, match='equal fixed_epsilon outside'):
            model(bad, fixed_epsilon=fixed)
        with pytest.raises(ValueError, match='equal fixed_epsilon outside'):
            core._require_fixed_exterior(bad, fixed, CUT, 7)
    core._require_fixed_exterior(base, fixed, CUT, 7)


def test_periodic_response_uses_explicit_support_and_cache_identity(monkeypatch):
    require_device('cpu')
    from torchfdtd.density_layer import _layer_z_fraction
    import torchfdtd.periodic_adjoint as implementation
    def make(cut):
        return PeriodicLayerResponse(SPEC, density_shape=(2, 2),
            policy=AdjointExecutionPolicy(device='cpu', host_budget_bytes=BUDGET,
                recorded=ReversibleCPMLOptions(interior_z=cut, trace_storage='cpu',
                    host_budget_bytes=BUDGET, resident_budget_bytes=BUDGET)),
            batch_options=AdjointBatchOptions(host_budget_bytes=BUDGET, gpu_budget_bytes=BUDGET),
            mesh=.1, steps=160, pml_cells=6, quadrature_counts=(4, 4), forward_kernel='torch')
    wide = make(None)
    supports = []
    for component in ('Ex', 'Ey', 'Ez'):
        fraction = _layer_z_fraction(wide._project.region, component, -SPEC['height_um']/2,
                                     SPEC['height_um']/2, torch.empty(()))
        supports.extend(torch.nonzero(fraction > 0).flatten().tolist())
    cut = min(supports)-1, max(supports)+1
    narrow = make(cut)
    assert wide._reference_key != narrow._reference_key
    assert wide._response_key != narrow._response_key
    rows = []
    for model in (wide, narrow):
        density = torch.tensor([[.2, .4], [.5, .3]], requires_grad=True)
        response = model(density)
        gradient, = torch.autograd.grad(objective(response), density)
        rows.append((response.detach(), gradient))
    assert torch.equal(rows[0][0], rows[1][0])
    assert relative(rows[0][1], rows[1][1]) < 1e-4
    monkeypatch.setattr(implementation, 'RecomputedAdjointBatch',
                        lambda *a, **k: pytest.fail('Allocated cases before support rejection'))
    with pytest.raises(ValueError, match='layer support.*reconstruction interval'):
        make((cut[0], cut[0]+1))
