import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Material, LorentzPole, Boundaries, BoundaryFace, Monitor
from torchfdtd.differentiable import _System
from torchfdtd.reversible_cpml import ReversibleCPMLOptions, _resolve_interval
from torchfdtd.reversible_cpml_kernels import _local_curl
from torchfdtd.reversible_lorentz import (DensityMaterial, forward_electric, inverse_electric,
                                        validate_density, proxy_project, resolve_material,
                                        advance_cpu, CPUReconstruction)


def project(cpml):
    periodic = BoundaryFace(kind='periodic')
    pml = BoundaryFace(kind='pml', layers=3)
    return Project(region=Region(dimension='3d', size=(.6, .5, 2.0), mesh=.1,
        steps=12, backend='cpu', precision='float32', mesh_auto_refine=False,
        boundaries=Boundaries(x_min=periodic, x_max=periodic, y_min=periodic, y_max=periodic,
                             z_min=pml if cpml else periodic, z_max=pml if cpml else periodic)),
        sources=[], monitors=[Monitor()])


def material(poles, damping=0):
    return Material(name='density material', model='multipole', epsilon_inf=1.3,
        poles=[LorentzPole(resonance_rad_s=4e15+2e15*i,
                           strength_rad_s_squared=1e31*(i+1), damping_rad_s=damping)
               for i in range(poles)])


@pytest.mark.parametrize('cpml', [False, True])
@pytest.mark.parametrize('poles', [1, 2])
@pytest.mark.parametrize('damping', [0, 1e12])
def test_packed_electric_and_magnetic_maps_reconstruct_the_initial_state(cpml, poles, damping):
    p = project(cpml)
    shape = p.region.shape
    interval = _resolve_interval(p.region, ReversibleCPMLOptions()) if cpml else (0, shape[2]-1)
    rho = torch.full((*shape[:2], interval[1]-interval[0]+1), .37)
    background = torch.linspace(1.0, 2.0, shape[2])
    mat = DensityMaterial(rho, background, p, interval, material(poles, damping))
    mat.allocate()
    system = _System(proxy_project(p), torch.ones(shape), prepare_kernels=False, prepare_permittivity=False)
    generator = torch.Generator().manual_seed(27)
    for field in (system.grid.E, system.grid.H, mat.state):
        field.copy_(torch.randn(field.shape, generator=generator)*.03)
    # Padding is storage only and never participates in an oscillator update.
    mat.state[..., 2*poles:] = 0
    old_e, old_h, old_poles = system.grid.E.clone(), system.grid.H.clone(), mat.state.clone()
    new_e, new_poles, psi = forward_electric(system, mat)
    system.grid.E.copy_(new_e)
    mat.state.copy_(new_poles)
    curl, _ = system.curl(new_e, psi, True)
    system.grid.H.sub_(system.grid.courant_number*curl)
    a, b = interval
    if cpml:
        hcurl = _local_curl(system.grid.E, a, b, True, system.grid.wrap)
        system.grid.H[:, :, a:b+1].add_(system.grid.courant_number*hcurl)
        # The interface algorithm restores its recorded pre-H lower halo.
        # Exterior CPML state is deliberately not inverted.
        system.grid.H[:, :, a-1, :2].copy_(old_h[:, :, a-1, :2])
    else:
        hcurl, _ = system.curl(system.grid.E, (), True)
        system.grid.H.add_(system.grid.courant_number*hcurl)
    inverse_electric(system, mat, periodic=not cpml)
    torch.testing.assert_close(system.grid.E[:, :, a:b+1], old_e[:, :, a:b+1], rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(system.grid.H[:, :, a:b+1], old_h[:, :, a:b+1], rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(mat.state, old_poles, rtol=2e-5, atol=1e-7)


def test_cell_and_layer_density_maps_produce_identical_packed_updates():
    p = project(False)
    shape = p.region.shape
    assert shape[2] % 2 == 0
    generator = torch.Generator().manual_seed(15)
    layer = torch.rand((*shape[:2], 2), generator=generator)
    cell = layer.repeat_interleave(shape[2]//2, 2)
    background = torch.linspace(1.0, 2.0, shape[2])
    outputs = []
    for rho in (layer, cell):
        mat = DensityMaterial(rho, background, p, (0, shape[2]-1), material(2))
        mat.allocate()
        system = _System(p, torch.ones(shape), prepare_kernels=False, prepare_permittivity=False)
        system.grid.H.copy_(torch.randn(system.grid.H.shape, generator=torch.Generator().manual_seed(99)))
        outputs.append(forward_electric(system, mat)[:2])
    assert torch.equal(outputs[0][0], outputs[1][0]) and torch.equal(outputs[0][1], outputs[1][1])


def test_invalid_density_and_strong_damping_fail_before_field_execution():
    p = project(False)
    shape = p.region.shape
    interval = (0, shape[2]-1)
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        validate_density(torch.full(shape, -1e-4), shape, interval)
    with pytest.raises(ValueError, match='anisotropic'):
        validate_density(torch.ones((*shape, 3)), shape, interval)
    with pytest.raises(ValueError, match='invertible|horizon'):
        DensityMaterial(torch.ones(shape), torch.ones(shape[2]), p, interval, material(2, 1e17))


def test_material_selection_freezes_the_declaration_and_rejects_tensor_materials():
    p = project(False)
    selected = material(1)
    p.materials.append(selected)
    frozen = resolve_material(p)
    selected.epsilon_inf = 2.0
    assert frozen.epsilon_inf == 1.3
    assert all(not m.oscillators for m in proxy_project(p).materials)
    with pytest.raises(ValueError, match='isotropic'):
        resolve_material(p, Material(name='tensor', model='tensor'))


def initialized_system(p):
    system = _System(proxy_project(p), torch.ones(p.region.shape),
                     prepare_kernels=False, prepare_permittivity=False)
    generator = torch.Generator().manual_seed(73)
    system.grid.E.copy_(torch.randn(system.grid.E.shape, generator=generator)*.1)
    system.grid.H.copy_(torch.randn(system.grid.H.shape, generator=generator)*.1)
    return system


def functional_signals(p, rho, interval, poles):
    mat = DensityMaterial(rho, torch.linspace(1, 2, p.region.shape[2]), p, interval, material(poles))
    mat.allocate()
    system = initialized_system(p)
    state = system.state()
    signals = []
    for step in range(p.region.steps):
        e, packed, psi = forward_electric(system, mat, state=state, rho=rho)
        e = system.inject(e, 'E', step, functional=True)
        curl, psi = system.curl(e, psi, True)
        h = system.inject(state[1]-system.grid.courant_number*curl, 'H', step, functional=True)
        state = (e, h, *psi)
        mat.state = packed
        signals.append(system.observe(state))
    return torch.stack(signals)


@pytest.mark.parametrize('cpml', [False, True])
@pytest.mark.parametrize('poles', [1, 2])
def test_reconstructed_density_vjp_matches_full_tape_and_finite_difference(cpml, poles):
    p = project(cpml)
    shape = p.region.shape
    interval = _resolve_interval(p.region, ReversibleCPMLOptions()) if cpml else (0, shape[2]-1)
    rho = torch.full((*shape[:2], interval[1]-interval[0]+1), .37, requires_grad=True)
    seed = torch.linspace(-.7, 1.1, p.region.steps).reshape(-1, 1)
    reference = functional_signals(p, rho, interval, poles)
    full_gradient = torch.autograd.grad((reference*seed).sum(), rho)[0]
    assert float(full_gradient.abs().max()) > 1e-5
    mat = DensityMaterial(rho, torch.linspace(1, 2, shape[2]), p, interval, material(poles))
    mat.allocate()
    system = initialized_system(p)
    trace = torch.empty((p.region.steps, 2, *shape[:2], 2)) if cpml else None
    outputs = []
    with torch.no_grad():
        for step in range(p.region.steps):
            advance_cpu(system, mat, step, None if trace is None else trace[step])
            outputs.append(system.observe(system.state()))
        assert torch.equal(torch.stack(outputs), reference.detach())
        gradient = torch.zeros((*shape[:2], mat.depth, 3))
        inverse = CPUReconstruction(system, mat, gradient, seed, periodic=not cpml)
        for step in range(p.region.steps-1, -1, -1):
            inverse.step(step, None if trace is None else trace[step])
        actual = mat.reduce_gradient(gradient, rho)
    torch.testing.assert_close(actual, full_gradient, atol=2e-6, rtol=2e-4)
    direction = torch.randn(rho.shape, generator=torch.Generator().manual_seed(5))
    direction = direction/direction.norm()
    h = .002
    with torch.no_grad():
        plus = (functional_signals(p, rho+h*direction, interval, poles)*seed).sum()
        minus = (functional_signals(p, rho-h*direction, interval, poles)*seed).sum()
        fd = (plus-minus)/(2*h)
    torch.testing.assert_close((actual*direction).sum(), fd, atol=1e-4, rtol=1e-2)
