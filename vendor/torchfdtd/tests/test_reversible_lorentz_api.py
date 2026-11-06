import pytest
import torch

from torchfdtd import (Project, Region, Material, LorentzPole, Boundaries, BoundaryFace,
    Source, Monitor, FieldMonitor, ReversibleSimulation, ReversibleOptions,
    ReversibleCPMLSimulation, ReversibleCPMLOptions, ReversibleCPMLPlaneSimulation)


def declaration(poles):
    return Material(name='Lorentz design', model='multipole', epsilon_inf=1.2,
        poles=[LorentzPole(resonance_rad_s=5e15+2e15*i,
                           strength_rad_s_squared=4e31*(i+1), damping_rad_s=0)
               for i in range(poles)])


def configuration(cpml, planes=False):
    periodic = BoundaryFace(kind='periodic')
    pml = BoundaryFace(kind='pml', layers=3)
    return Project(region=Region(dimension='3d', size=(.6, .6, 2.0), mesh=.1,
        precision='float32', backend='cpu', steps=80, mesh_auto_refine=False, material_sampling='yee',
        boundaries=Boundaries(x_min=periodic, x_max=periodic, y_min=periodic, y_max=periodic,
                             z_min=pml if cpml else periodic, z_max=pml if cpml else periodic)),
        sources=[Source(center=(0, 0, -.2), wavelength=.6, time_definition='standard',
                        pulse_length=.4e-15, pulse_offset=.6e-15)],
        monitors=[FieldMonitor(id='plane', normal='z', center=(0, 0, .2), size=(.3, .3, 0),
                               spectrum=dict(sampling='frequency', apodization='none'))]
                 if planes else [Monitor(center=(.1, 0, .2))])


@pytest.mark.parametrize('cpml', [False, True])
@pytest.mark.parametrize('poles', [1, 2])
def test_point_density_api_matches_no_grad_and_supports_retained_vjps(cpml, poles):
    p = configuration(cpml)
    model = (ReversibleCPMLSimulation(p, material=declaration(poles)) if cpml else
             ReversibleSimulation(p, material=declaration(poles)))
    rho = torch.full(p.region.shape, .43, requires_grad=True)
    bg = torch.linspace(1, 2, p.region.shape[2])
    result = model(rho, fixed_epsilon=bg)
    assert float(result.signals.detach().abs().max()) > 1e-5
    first = torch.autograd.grad(result.signals.square().sum(), rho, retain_graph=True)[0]
    second = torch.autograd.grad(result.signals.square().sum(), rho)[0]
    assert torch.equal(first, second) and bool(torch.isfinite(first).all())
    assert result.report['backward_calls'] == 2
    with torch.no_grad():
        forward_only = model(rho, fixed_epsilon=bg)
    assert torch.equal(result.signals, forward_only.signals)
    assert forward_only.report['forward_only']
    assert result.report['pole_count'] == poles
    if cpml:
        a, b = model.interior_z
        assert not bool(first[:, :, :a].any()) and not bool(first[:, :, b+1:].any())


@pytest.mark.parametrize('poles', [1, 2])
@pytest.mark.parametrize('rho_value', [0., 1.])
def test_plane_density_extremes_produce_finite_material_vjps(poles, rho_value):
    p = configuration(True, planes=True)
    model = ReversibleCPMLPlaneSimulation(p, material=declaration(poles),
                                         quadrature_counts={'plane': (2, 2)})
    rho = torch.full((*p.region.shape[:2], 1), rho_value, requires_grad=True)
    bg = torch.linspace(1, 2, p.region.shape[2])
    frequency = torch.tensor([299792458/(.6e-6)])
    result = model(rho, frequency, fixed_epsilon=bg)['plane']
    gradient = torch.autograd.grad((result.fields/p.region.time_step).abs().square().sum(), rho)[0]
    assert bool(torch.isfinite(result.fields).all()) and bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().max()) > 1e-6
    plan = model.plan(frequency, density_layers=1)
    assert plan['pole_state_bytes'] > 0 and plan['density_layer_count'] == 1
    assert result.report['backend'] == 'torch CPU'


def test_density_reservation_rejects_a_pole_budget_before_allocating_fields(monkeypatch):
    p = configuration(True)
    ordinary = ReversibleCPMLSimulation(p).plan()
    lorentz = ReversibleCPMLSimulation(p, material=declaration(2)).plan()
    assert lorentz['memory_reservation_bytes'] > ordinary['memory_reservation_bytes']
    options = ReversibleCPMLOptions(resident_budget_bytes=ordinary['memory_reservation_bytes'])
    model = ReversibleCPMLSimulation(p, options, material=declaration(2))
    rho, bg = torch.full(p.region.shape, .3), torch.ones(p.region.shape[2])
    def fail(*args, **kwargs):
        raise AssertionError('field allocation was reached')
    monkeypatch.setattr(torch, 'zeros', fail)
    with pytest.raises(ValueError, match='reservation|budget'):
        model(rho, fixed_epsilon=bg)


def test_mutated_background_and_plane_material_are_not_silently_reused():
    p = configuration(False)
    model = ReversibleSimulation(p, material=declaration(1))
    rho, bg = torch.full(p.region.shape, .3, requires_grad=True), torch.ones(p.region.shape[2])
    output = model(rho, fixed_epsilon=bg)
    bg.add_(.1)
    with pytest.raises(RuntimeError, match='modified by an inplace'):
        output.signals.square().sum().backward()
    p = configuration(True, planes=True)
    plane = ReversibleCPMLPlaneSimulation(p, material=declaration(1), quadrature_counts={'plane': (2, 2)})
    plane.model.material.epsilon_inf = 1.8
    with pytest.raises(ValueError, match='material changed'):
        plane(torch.ones((*p.region.shape[:2], 1)), torch.tensor([5e14]), fixed_epsilon=torch.ones(p.region.shape[2]))


def test_new_options_are_keyword_only_and_legacy_positions_stay_fixed():
    options = ReversibleOptions(None, None, None, 1e-3, offload_terminal=True)
    assert options.offload_terminal
    with pytest.raises(ValueError, match='boolean'):
        ReversibleOptions(offload_terminal=1)


def test_plane_density_vjp_matches_centered_finite_difference():
    p = configuration(True,planes=True)
    model = ReversibleCPMLPlaneSimulation(p,material=declaration(2),
        quadrature_counts={'plane':(2,2)})
    rho = torch.full((*p.region.shape[:2],1),.41,requires_grad=True)
    bg = torch.linspace(1,2,p.region.shape[2])
    frequency = torch.tensor([299792458/(.6e-6)])
    def objective(values):
        field = model(values,frequency,fixed_epsilon=bg)['plane'].fields/p.region.time_step
        return field.abs().square().sum()
    gradient = torch.autograd.grad(objective(rho),rho)[0]
    index = tuple(int(v) for v in torch.unravel_index(gradient.abs().argmax(),gradient.shape))
    delta = torch.zeros_like(rho)
    delta[index] = 1e-3
    with torch.no_grad():
        finite = (objective(rho+delta)-objective(rho-delta))/(2e-3)
    assert abs(float(gradient[index])) > 1e-6
    torch.testing.assert_close(gradient[index],finite,rtol=2e-2,atol=2e-5)


def test_explicit_recording_request_and_cpu_offload_reservation_are_honored():
    p = configuration(True)
    options = ReversibleCPMLOptions(forward_only='never',offload_terminal=True)
    model = ReversibleCPMLSimulation(p,options,material=declaration(2))
    rho = torch.full((*p.region.shape[:2],1),.3)
    result = model(rho,fixed_epsilon=torch.ones(p.region.shape[2]))
    assert not result.report['forward_only']
    assert not result.report['terminal_offload_used']
    plan = model.plan(density_layers=1)
    assert plan['workspace_components_bytes']['pole_terminal'] == plan['pole_state_bytes']


def test_automatic_cpu_history_reports_its_real_placement():
    p = configuration(True)
    model = ReversibleCPMLSimulation(p,ReversibleCPMLOptions(trace_storage='auto',
        host_budget_bytes=1<<30),material=declaration(2))
    plan = model.plan(density_layers=1)
    assert plan['trace_storage'] == 'cpu' and plan['trace_transfers'] == 'sync'
    assert plan['trace_placement_attempts'][-1]['admitted']


def test_one_micrometre_tile_with_256_steps_has_a_nonzero_retained_density_vjp():
    p = configuration(True,planes=True)
    p.region.size = (1.,1.,2.)
    p.region.steps = 256
    model = ReversibleCPMLPlaneSimulation(p,material=declaration(2),
        quadrature_counts={'plane':(2,2)})
    rho = torch.full((*p.region.shape[:2],2),.37,requires_grad=True)
    bg = torch.ones(p.region.shape[2])
    frequency = torch.tensor([299792458/(.6e-6)])
    result = model(rho,frequency,fixed_epsilon=bg)['plane']
    loss = (result.fields/p.region.time_step).abs().square().sum()
    first = torch.autograd.grad(loss,rho,retain_graph=True)[0]
    second = torch.autograd.grad(loss,rho)[0]
    assert torch.equal(first,second) and float(first.abs().max()) > 1e-6
