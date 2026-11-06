"""Dispersive (ADE) media next to PMC/symmetric faces: face polarization banks.

The stored upper-face E nodes carry their own P/Q banks in the resident,
streamed and tensor-batch paths. Physics is checked against mirrored full
domains without PMC; every adjoint is checked against full autograd.
"""
import math
import numpy as np
import pytest
import torch

from torchfdtd import Project, Source, Monitor, AdjointOptions, LorentzPole
from torchfdtd.models import Material, Structure
from torchfdtd.boundaries import BoundaryDescription, material_shape
from torchfdtd.dispersive_adjoint import DispersiveSimulation, _DispersiveSystem
from torchfdtd.solver import Simulation, voxelize
from test_pmc_general import (CUDA, PROFILE, scene, at, random_epsilon, objective, mirrored_pair, material_samples, MIRRORS)

RATES = dict(omega0=(2e15, 3e15), gamma=(1e14, 3e14), strength=(1e31, .5e31))


def dispersive_scene(precision='float32', *, backend='cpu', steps=14):
    kinds = (('pml', 'pmc'), ('pmc', 'pec'), ('pec', 'pmc'))
    p = scene(kinds, size=(1.0, .6, .6), steps=steps, precision=precision, profiles={'x_min': PROFILE}, backend=backend)
    r = p.region;n = r.shape
    p.sources = [Source(center=at(r, 'Ez', (4, 0, 2)), component='Ez', pulse='continuous', wavelength=.5),
                 Source(center=at(r, 'Ey', (n[0], 2, n[2])), component='Ey', pulse='continuous', wavelength=.45, amplitude=.7)]
    p.monitors = [Monitor(center=at(r, 'Ez', (n[0], 3, 1)), component='Ez'), Monitor(center=at(r, 'Hx', (n[0], 1, 3)), component='Hx'),
                  Monitor(center=at(r, 'Ex', (5, 2, n[2])), component='Ex'), Monitor(center=at(r, 'Hy', (6, 0, 2)), component='Hy')]
    return Project.model_validate(p.model_dump())


def parameters(region, seed, *, device='cpu', spatial=True):
    """epsilon_inf, strength, omega0, gamma with the stored PMC rows included."""
    dtype = torch.float64 if region.precision == 'float64' else torch.float32
    generator = torch.Generator().manual_seed(seed)
    epsilon = random_epsilon(region, seed, device=device)
    if spatial:
        strength = 1e31*(.5+torch.rand((2, *material_shape(region, True)), generator=generator, dtype=torch.float64))
        strength = strength.to(device=device, dtype=dtype)
    else:
        strength = torch.tensor(RATES['strength'], dtype=dtype, device=device)
    omega0 = torch.tensor(RATES['omega0'], dtype=dtype, device=device)
    gamma = torch.tensor(RATES['gamma'], dtype=dtype, device=device)
    return epsilon, strength, omega0, gamma


def tolerance(precision, reference, *, factor=1.):
    scale = float(reference.abs().max())
    return dict(atol=1e-12*scale if precision == 'float64' else 4e-6*scale*factor, rtol=1e-9 if precision == 'float64' else 4e-5*factor)


# ---------------------------------------------------------------------------
# Physics: mirrored full domain without PMC, with spatial oscillator strength
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', list(MIRRORS))
@torch.enable_grad()
def test_dispersive_half_domain_matches_mirrored_full_domain(name):
    half, full, images = mirrored_pair(name)
    eps_half, strength_half, omega0, gamma = parameters(half.region, 7)
    eps_half.requires_grad_();strength_half.requires_grad_();omega0.requires_grad_();gamma.requires_grad_()
    eps_full = torch.ones(material_shape(full.region, True))
    strength_full = torch.zeros((2, *material_shape(full.region, True)))
    with torch.no_grad():
        for index, component, targets in material_samples(half, full, images):
            for target in targets:
                eps_full[target+(component,)] = eps_half[index+(component,)]
                strength_full[(slice(None), *target, component)] = strength_half[(slice(None), *index, component)]
    eps_full.requires_grad_();strength_full.requires_grad_()
    omega0_full = omega0.detach().clone().requires_grad_();gamma_full = gamma.detach().clone().requires_grad_()
    reference = DispersiveSimulation(full).reference(eps_full, strength_full, omega0_full, gamma_full)
    expected = torch.autograd.grad(objective(reference), (eps_full, strength_full, omega0_full, gamma_full))
    result = DispersiveSimulation(half, AdjointOptions(checkpoints=3))(eps_half, strength_half, omega0, gamma)
    assert result.report['adjoint'] == 'discrete Yee/CPML/trapezoidal ADE' and result.report['pmc_faces'] is True
    actual = torch.autograd.grad(objective(result.signals), (eps_half, strength_half, omega0, gamma))
    assert reference.abs().max() > 1e-3
    torch.testing.assert_close(result.signals, reference, atol=4e-6, rtol=4e-5)
    folded_eps = torch.zeros_like(eps_half);folded_strength = torch.zeros_like(strength_half)
    for index, component, targets in material_samples(half, full, images):
        for target in targets:
            folded_eps[index+(component,)] += expected[0][target+(component,)]
            folded_strength[(slice(None), *index, component)] += expected[1][(slice(None), *target, component)]
    for axis, side in MIRRORS[name]['mirrors']:
        if side != 1:continue
        cut = [slice(None)]*4;cut[axis] = half.region.shape[axis]
        wall = actual[1][(slice(None), *cut)]
        assert wall.abs().max() > 0 and wall[..., axis].abs().max() == 0
    torch.testing.assert_close(actual[0], folded_eps, atol=2e-6*float(expected[0].abs().max()), rtol=4e-5)
    torch.testing.assert_close(actual[1], folded_strength, atol=2e-6*float(expected[1].abs().max()), rtol=4e-5)
    # Shared oscillator rates see the same objective in both domains.
    for value, reference_value in zip(actual[2:], expected[2:]):
        assert reference_value.abs().max() > 0
        torch.testing.assert_close(value, reference_value, atol=2e-6*float(reference_value.abs().max()), rtol=4e-5)


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------
@torch.enable_grad()
def test_dispersive_adjoint_taylor_check_float64():
    p = dispersive_scene('float64')
    values = [v.requires_grad_() for v in parameters(p.region, 3)]
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
    result = model(*values)
    gradient = torch.autograd.grad(objective(result.signals), values)
    reference = torch.autograd.grad(objective(model.reference(*values)), values)
    for actual, expected in zip(gradient, reference):
        assert expected.abs().max() > 0
        torch.testing.assert_close(actual, expected, atol=1e-12*float(expected.abs().max()), rtol=1e-9)
    # Directions scaled to each parameter's magnitude so every input moves.
    generator = torch.Generator().manual_seed(11)
    directions = [torch.randn(v.shape, generator=generator, dtype=torch.float64)*(.05 if i == 0 else .05*v.detach().abs().mean())
                  for i, v in enumerate(values)]
    directional = sum(float((g*d).sum()) for g, d in zip(gradient, directions))
    with torch.no_grad():
        def value(h):return float(objective(model(*(v+h*d for v, d in zip(values, directions))).signals))
        errors = [abs((value(h)-value(-h))/(2*h)-directional) for h in (4e-2, 2e-2, 1e-2)]
    assert abs(directional) > 1e-8
    assert errors[-1] < 1e-5*abs(directional)
    assert errors[0] > 3*errors[1] > 9*errors[2]


@CUDA
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('spatial', [False, True])
@torch.enable_grad()
def test_cuda_dispersive_adjoint_matches_autograd(precision, spatial):
    p = dispersive_scene(precision)
    values = [v.requires_grad_() for v in parameters(p.region, 5, device='cuda', spatial=spatial)]
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
    result = model(*values)
    assert result.report['forward_backend'] == 'torch CUDA' and result.report['backward_backend'] == 'torch explicit ADE transpose'
    gradient = torch.autograd.grad(objective(result.signals), values)
    reference = torch.autograd.grad(objective(model.reference(*values)), values)
    for actual, expected in zip(gradient, reference):
        assert expected.abs().max() > 0
        torch.testing.assert_close(actual, expected, **tolerance(precision, expected))
    direction = [torch.randn(v.shape, generator=torch.Generator().manual_seed(13)).to(v)*(1. if i == 0 else v.detach().abs().mean()) for i, v in enumerate(values)]
    directional = sum(float((g*d).sum()) for g, d in zip(gradient, direction))
    h = 1e-3 if precision == 'float64' else 1e-2
    with torch.no_grad():
        central = (float(objective(model(*(v+h*d for v, d in zip(values, direction))).signals))
                   -float(objective(model(*(v-h*d for v, d in zip(values, direction))).signals)))/(2*h)
    assert abs(central-directional) < (2e-6 if precision == 'float64' else 3e-2)*abs(directional)


@pytest.mark.parametrize('storage', ['host', 'disk'])
@torch.enable_grad()
def test_checkpoint_tiers_carry_face_pole_state(storage, tmp_path):
    p = dispersive_scene('float32')
    values = [v.requires_grad_() for v in parameters(p.region, 9)]
    reference = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
    expected = torch.autograd.grad(objective(reference(*values).signals), values)
    options = AdjointOptions(checkpoints=2, storage=storage, checkpoint_directory=tmp_path if storage == 'disk' else None,
                             disk_budget_bytes=64*2**20 if storage == 'disk' else None)
    result = DispersiveSimulation(p, options)(*values)
    actual = torch.autograd.grad(objective(result.signals), values)
    packed, layout = DispersiveSimulation(p)._pack(*values)
    system = _DispersiveSystem(p, values[0].detach(), packed.detach(), layout)
    faces = system.grid.pmc_blocks['E']
    assert len(system.face_P) == len(faces) > 0 and all(f.shape == (2, *shape) for f, (_, _, shape) in zip(system.face_P, faces))
    assert result.report['restart_bytes'] == sum(s.numel()*s.element_size() for s in system.state())
    assert result.report['material_state_bytes'] == 4*2*(6*math.prod(p.region.shape)+2*sum(math.prod(shape) for _, _, shape in faces))
    for a, e in zip(actual, expected):torch.testing.assert_close(a, e, atol=1e-7*float(e.abs().max()), rtol=1e-5)


# ---------------------------------------------------------------------------
# Streamed X-slab execution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('storage', ['host', 'disk'])
@torch.enable_grad()
def test_streamed_dispersive_matches_resident_with_pmc_faces(device, storage, tmp_path):
    from torchfdtd import StreamedAdjointOptions
    from torchfdtd.streamed_dispersive import StreamedDispersiveSimulation, _SlabDispersiveSystem, estimate_streamed_dispersive_memory
    p = dispersive_scene('float32', steps=17)
    values = [v.requires_grad_() for v in parameters(p.region, 21)]
    reference = DispersiveSimulation(p, AdjointOptions(checkpoints=2)).reference(*values)
    expected = torch.autograd.grad(objective(reference), values)
    options = StreamedAdjointOptions(device=device, slab_width=3, temporal_depth=3, checkpoints=2, local_checkpoints=1,
                                     state_storage=storage, state_directory=tmp_path/'banks' if storage == 'disk' else None,
                                     disk_budget_bytes=256*2**20 if storage == 'disk' else None,
                                     restart_directory=tmp_path/'journal', restart_every_blocks=2)
    result = StreamedDispersiveSimulation(p, options)(*values)
    assert result.report['pmc_faces'] is True and result.report['forward_backend'] == ('torch CUDA' if device == 'cuda' else 'torch CPU')
    actual = torch.autograd.grad(objective(result.signals), values)
    assert reference.abs().max() > 1e-3
    torch.testing.assert_close(result.signals, reference, atol=3e-6, rtol=4e-5)
    for a, e in zip(actual, expected):
        assert e.abs().max() > 0
        torch.testing.assert_close(a, e, atol=3e-6*float(e.abs().max()), rtol=4e-5)
    packed, layout = DispersiveSimulation(p)._pack(*values)
    host = _SlabDispersiveSystem(p, values[0].detach(), packed.detach(), layout, prepare_updates=False)
    exact = sum(s.numel()*s.element_size() for s in host.state())
    assert result.report['state_bytes'] == exact
    assert sum(f.numel() for f in host.face_P) > 0
    assert result.report['journal_records_written'] and result.report['restart_reservation_bytes'] >= 2*exact
    planned = estimate_streamed_dispersive_memory(p, [tuple(v.shape) for v in values], options)
    assert planned['state_bytes'] == exact


# ---------------------------------------------------------------------------
# Tensor batch
# ---------------------------------------------------------------------------
@CUDA
def test_tensor_batch_face_ade_banks_match_resident():
    pytest.importorskip('cupy')
    from torchfdtd.tensor_batch import run_tensor_batch
    material = Material(name='dispersion', model='multipole', epsilon_inf=1.8,
                        poles=[LorentzPole(resonance_rad_s=1.3e15, damping_rad_s=2e14, strength_rad_s_squared=.7e30),
                               LorentzPole(resonance_rad_s=1.9e15, damping_rad_s=4e14, strength_rad_s_squared=.5e30)])
    kinds = (('pml', 'pmc'), ('pmc', 'pec'), ('pec', 'pmc'))
    p = scene(kinds, size=(1.0, .6, .6), steps=30, profiles={'x_min': PROFILE}, backend='cuda', materials=[material],
              structures=[Structure(name='block', material='dispersion', kind='rectangle', center=(.2, .1, .1), size=(.6, .5, .5))])
    r = p.region;n = r.shape
    p.sources = [Source(center=at(r, 'Ez', (4, 0, 2)), component='Ez', pulse='continuous', wavelength=.5),
                 Source(center=at(r, 'Ey', (n[0], 2, n[2])), component='Ey', pulse='continuous', wavelength=.45, amplitude=.7)]
    p.monitors = [Monitor(center=at(r, 'Ez', (n[0], 3, 1)), component='Ez'), Monitor(center=at(r, 'Hx', (n[0], 1, 3)), component='Hx'),
                  Monitor(center=at(r, 'Ex', (5, 2, n[2])), component='Ex'), Monitor(center=at(r, 'Hy', (6, 0, 2)), component='Hy')]
    p = Project.model_validate(p.model_dump())
    epsilon, _, ownership = voxelize(p, with_ownership=True)
    face = tuple(slice(nn, nn+1) if a == 0 else slice(0, nn) for a, nn in enumerate(n))
    assert int((ownership[face+(2,)] == 0).sum()) > 0   # the block reaches the x-upper face
    report = run_tensor_batch([p, p], cuda_graph=True)
    report.raise_for_errors()
    rates = np.array(material.oscillators)
    mask = torch.as_tensor(ownership == 0, dtype=torch.float32, device='cuda')
    strength = torch.as_tensor(rates[:, 1], dtype=torch.float32, device='cuda')[:, None, None, None, None]*mask[None]
    with torch.no_grad():
        single = DispersiveSimulation(p)(torch.as_tensor(epsilon, device='cuda'), strength,
                                         torch.as_tensor(rates[:, 0], dtype=torch.float32, device='cuda'),
                                         torch.as_tensor(rates[:, 2], dtype=torch.float32, device='cuda')).signals.cpu().numpy()
    assert np.abs(single).max() > 1e-2
    for item in report.items:
        np.testing.assert_allclose(item.result.signals, single, rtol=3e-6, atol=3e-7)
        assert item.summary['material_update'] == 'trapezoidal ADE'
        volume_nodes = int((ownership[:n[0], :n[1], :n[2]] == 0).sum())
        face_nodes = sum(int((ownership[tuple(slice(nn, nn+1) if a in upper else slice(0, nn) for a, nn in enumerate(n))+(component,)] == 0).sum())
                         for component, upper, _ in BoundaryDescription(r).pmc_blocks['E'])
        assert face_nodes > 0 and item.summary['dispersive_samples'] == 2*(volume_nodes+face_nodes)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------
def test_dispersive_pmc_rejections_are_exact():
    p = dispersive_scene('float32')
    values = parameters(p.region, 1)
    with pytest.raises(ValueError, match='fused CUDA ADE'):DispersiveSimulation(p, AdjointOptions(backward_kernel='fused'))
    fused = p.model_copy(deep=True);fused.region.cuda_kernel = 'fused'
    with pytest.raises(ValueError, match='fused CUDA ADE'):DispersiveSimulation(fused)
    with pytest.raises(ValueError, match='stored row'):
        DispersiveSimulation(p)(torch.ones(p.region.shape+(3,)), *values[1:])
    with pytest.raises(ValueError, match='stored upper PMC rows'):
        DispersiveSimulation(p)(values[0], torch.ones((2, *p.region.shape, 3)), *values[2:])
    # The exact-endpoint forward dispatch has no ADE path.
    material = Material(name='metal', model='drude')
    closed = scene((('pec', 'pmc'), ('pmc', 'pmc'), ('pec', 'pmc')), size=(.8, .7, .6), materials=[material],
                   structures=[Structure(material='metal')], sources=[Source(component='Ez', center=(.05, .05, .05), pulse='continuous')],
                   monitors=[Monitor(component='Ez', center=(.05, .05, .05))])
    with pytest.raises(ValueError, match='ADE'):Simulation(closed).run()
    assert BoundaryDescription(closed.region).pmc_blocks['E']
