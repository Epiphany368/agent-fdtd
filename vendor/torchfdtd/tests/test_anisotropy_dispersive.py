"""Full-tensor trapezoidal ADE: transpose, VJP, passivity, parity and admission."""
import math

import pytest
import torch

from torchfdtd import AdjointOptions, DispersiveSimulation
from torchfdtd.anisotropy import TensorDielectricSimulation, _cpml_faces
from torchfdtd.anisotropy_dispersive import (TensorDispersiveSimulation, _TensorDispersiveSystem, _TensorPoleLayout,
                                             cpml_face_dispersive_admissible)
from test_anisotropy_walls import bloch_scene, one_step_matrix, scene, spd, tensor_field


def interior_mask(region):
    mask = torch.ones(region.shape, dtype=torch.bool)
    for _, _, index in _cpml_faces(region):
        mask[index] = False
    return mask


def poles(region, dtype, count=2):
    """PSD strength tensors of distinct orientation, active in the interior only."""
    mask = interior_mask(region)
    out = []
    for i in range(count):
        angles = (.3+.5*i, -.4+.2*i, .7-.6*i)
        eigen = (0., .4, 1.1) if i == 0 else (.3, .9, .2)
        value = spd(eigen, angles, dtype)
        field = torch.where(mask[..., None, None], value, torch.zeros_like(value)).expand(region.shape+(3, 3)).clone()
        out.append(field)
    return torch.stack(out)


def parameters(p, generator, dtype, device='cpu'):
    """Order-one leaves; scale() applies the physical (rad/s)^2 and rad/s units."""
    epsilon = tensor_field(p.region, generator, dtype).to(device).requires_grad_()
    strength = poles(p.region, dtype).to(device).requires_grad_()
    omega0 = torch.tensor([0., 1.7], dtype=dtype, device=device, requires_grad=True)
    gamma = torch.tensor([.2, .3], dtype=dtype, device=device, requires_grad=True)
    return epsilon, strength, omega0, gamma


def scale(leaves):
    epsilon, strength, omega0, gamma = leaves
    return epsilon, strength*1e30, omega0*1e15, gamma*1e15


@pytest.mark.parametrize('faces,phase', [
    (dict(x='periodic', y='periodic', z='periodic'), (0, 0, 0)),
    (dict(x='bloch', y='pml', z='pec'), (.37, 0, 0)),
    (dict(x='pml', y='pml', z='pml'), (0, 0, 0)),
])
def test_tensor_ade_transpose_and_vjp_match_full_autograd(faces, phase):
    p = scene(faces, 'float64', steps=12, bloch_phase=phase)
    generator = torch.Generator().manual_seed(11)
    inputs = parameters(p, generator, torch.float64)
    model = TensorDispersiveSimulation(p, AdjointOptions(checkpoints=2))
    reference = model.reference(*scale(inputs))
    weights = torch.randn(reference.shape, dtype=reference.dtype, generator=generator)
    loss = lambda signals: (signals.conj()*weights).real.sum()+signals.abs().square().sum()
    expected = list(torch.autograd.grad(loss(reference), inputs))
    # Symmetric inputs: the explicit VJP is the symmetric projection of the
    # unconstrained derivative, which is what a symmetric perturbation sees.
    expected[:2] = [(g+g.transpose(-1, -2))/2 for g in expected[:2]]
    actual = model(*scale(inputs))
    got = torch.autograd.grad(loss(actual.signals), inputs)
    torch.testing.assert_close(actual.signals, reference, rtol=1e-12, atol=1e-13)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=1e-12)
    assert all(torch.linalg.vector_norm(g) > 1e-8 for g in got)
    assert actual.report['neumann_terms'] >= 2 and 0 < actual.report['coupling_bound'] < 1
    assert actual.report['peak_checkpoints'] <= 2 and not actual.report['full_time_autograd']


def test_isotropic_tensor_ade_matches_scalar_trapezoidal_ade():
    p = scene(dict(x='pml', y='periodic', z='pml'), 'float64', steps=30)
    shape = p.region.shape
    eye = torch.eye(3, dtype=torch.float64)
    epsilon = (1.8*eye).expand(shape+(3, 3)).clone()
    strength = torch.stack([(.7e30*eye).expand(shape+(3, 3)), (.5e30*eye).expand(shape+(3, 3))]).clone()
    omega0 = torch.tensor([1.3e15, 1.9e15], dtype=torch.float64)
    gamma = torch.tensor([2e14, 4e14], dtype=torch.float64)
    tensor = TensorDispersiveSimulation(p)(epsilon, strength, omega0, gamma)
    scalar = DispersiveSimulation(p)(torch.full(shape, 1.8, dtype=torch.float64),
                                     torch.tensor([.7e30, .5e30], dtype=torch.float64), omega0, gamma)
    # The nodal assemblies of uniform isotropic tensors are pointwise scalars, so
    # only the truncated Neumann series separates the two schemes.
    torch.testing.assert_close(tensor.signals, scalar.signals, rtol=1e-10, atol=1e-13)
    assert tensor.signals.abs().max() > 1e-3


def test_strength_rotation_taylor_and_central_difference():
    p = scene(dict(x='periodic', y='periodic', z='periodic'), 'float64', steps=14)
    shape = p.region.shape
    model = TensorDispersiveSimulation(p, AdjointOptions(checkpoints=3))
    epsilon = spd((1.4, 2.1, 3.), (.3, -.2, .5)).expand(shape+(3, 3))
    omega0, gamma = torch.tensor([1.5e15], dtype=torch.float64), torch.tensor([2e14], dtype=torch.float64)
    def loss(angle):
        strength = (1e30*spd((.2, .9, 1.4), (angle, .4, -.3)))[None].expand((1,)+shape+(3, 3))
        return model(epsilon, strength, omega0, gamma).signals.abs().square().sum()
    theta = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
    value = loss(theta)
    derivative, = torch.autograd.grad(value, theta)
    step = 1e-4
    central = (loss(theta.detach()+step)-loss(theta.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, central, rtol=1e-6, atol=1e-12)
    remainders = [float((loss(theta.detach()+h)-value.detach()-h*derivative).abs()) for h in (1e-2, 5e-3, 2.5e-3)]
    assert remainders[1] < .3*remainders[0] and remainders[2] < .3*remainders[1]
    assert abs(float(derivative)) > 1e-8


def test_lossless_tensor_ade_one_step_map_has_unit_spectral_radius():
    # Rotated epsilon_inf and two differently rotated poles, no damping, Bloch
    # phases at (pi, pi/2, 0.9) per cell: the fixed-length Neumann solve keeps
    # the trapezoidal tensor ADE marginally stable to rounding. A three-cell
    # periodic box below the scene validator keeps the dense map small; the
    # Bloch phase pi puts the Nyquist mode on the x axis.
    p = bloch_scene(dict(x='bloch', y='bloch', z='bloch'), (5, 5, 5), (math.pi, math.pi/2, .9), pml_cells=3)
    shape = p.region.shape
    epsilon = spd((1.3, 2.4, 3.6), (.4, .7, -.3)).expand(shape+(3, 3)).contiguous()
    strength = torch.stack([spd((.05, 1.2, 2.5), (-.5, .3, .8)), spd((.6, .6, 1.9), (.9, -.4, .2))])
    strength = (1e30*strength)[:, None, None, None].expand((2,)+shape+(3, 3)).contiguous()
    omega0 = torch.tensor([0., 2.1e15], dtype=torch.float64)
    gamma = torch.zeros(2, dtype=torch.float64)
    model = TensorDispersiveSimulation(p)
    _, layout, bound = model._pack(epsilon, strength, omega0, gamma)
    assert 0 < bound < 1 and layout.iterations >= 3
    # Same uniform coefficients and admitted series length on the tiny box.
    tiny = p.model_copy(update=dict(region=p.region.model_copy(update=dict(size=(.3, .3, .3)))))
    assert tiny.region.shape == (3, 3, 3) and tiny.region.time_step == p.region.time_step
    layout = _TensorPoleLayout((3, 3, 3), 2, layout.iterations)
    dt = tiny.region.time_step
    epsilon, strength = epsilon[:3, :3, :3].contiguous(), strength[:, :3, :3, :3].contiguous()
    packed = layout.flatten(epsilon, strength*dt*dt, (omega0*dt).square(), gamma*dt)
    system = _TensorDispersiveSystem(tiny, epsilon, packed, layout)
    matrix = one_step_matrix(system, packed)
    radius = float(torch.linalg.eigvals(matrix).abs().max())
    assert radius <= 1+1e-9, radius


def test_tensor_ade_admission_passivity_and_dispersive_face_criterion():
    p = scene(dict(x='pml', y='periodic', z='periodic'), 'float64')
    shape = p.region.shape
    model = TensorDispersiveSimulation(p)
    eye = torch.eye(3, dtype=torch.float64)
    epsilon = (2*eye).expand(shape+(3, 3)).clone()
    good = (.5e30*eye).expand((1,)+shape+(3, 3)).clone()
    omega0, gamma = torch.tensor([1e15], dtype=torch.float64), torch.tensor([1e14], dtype=torch.float64)
    model._pack(epsilon, good, omega0, gamma)
    with pytest.raises(ValueError, match='positive semidefinite'):
        model._pack(epsilon, -good, omega0, gamma)
    asymmetric = good.clone()
    asymmetric[..., 0, 1] = 1e29
    with pytest.raises(ValueError, match='exactly symmetric'):
        model._pack(epsilon, asymmetric, omega0, gamma)
    with pytest.raises(ValueError, match='nonnegative'):
        model._pack(epsilon, good, -omega0, gamma)
    with pytest.raises(ValueError, match='nonnegative'):
        model._pack(epsilon, good, omega0, -gamma)
    with pytest.raises(ValueError, match='one entry per pole'):
        model._pack(epsilon, good, torch.tensor([1e15, 2e15], dtype=torch.float64), gamma)
    with pytest.raises(ValueError, match='below one'):
        model._pack(epsilon, 1e4*good, omega0, gamma)
    with pytest.raises(ValueError, match="cpml_material='tensor'"):
        TensorDispersiveSimulation(p, cpml_material='isotropic')
    # A pole active in the x CPML region: aligned epsilon_inf with x extreme and
    # every strength a scalar multiple of epsilon_inf (frequency-independent anisotropy).
    aligned = torch.diag(torch.tensor([3., 2., 2.5], dtype=torch.float64)).expand(shape+(3, 3)).clone()
    proportional = (.2e30*aligned)[None].clone()
    model._pack(aligned, proportional, omega0, gamma)
    assert bool(cpml_face_dispersive_admissible(aligned, proportional, 0).all())
    # The multiple may vary per node, and eight ulps of rounding are tolerated.
    varying = proportional*(1+.1*torch.linspace(0, 1, shape[0], dtype=torch.float64))[None, :, None, None, None, None]
    varying = varying*(1+4*torch.finfo(torch.float64).eps)
    model._pack(aligned, varying, omega0, gamma)
    uniaxial = torch.diag(torch.tensor([.1e30, .5e30, .5e30], dtype=torch.float64)).expand((1,)+shape+(3, 3)).clone()
    with pytest.raises(ValueError, match='scalar multiple'):
        model._pack(torch.diag(torch.tensor([3., 2., 2.], dtype=torch.float64)).expand(shape+(3, 3)).clone(), uniaxial, omega0, gamma)
    with pytest.raises(ValueError, match='x_min'):
        model._pack(aligned, uniaxial, omega0, gamma)
    rotated = spd((3., 2., 2.5), (.4, 0., 0.)).expand(shape+(3, 3)).clone()
    with pytest.raises(ValueError, match='axis-aligned'):
        model._pack(rotated, (.2e30*rotated)[None].clone(), omega0, gamma)
    # The rotated tensor is admissible without dispersion, and with a pole that is inactive in the collars.
    TensorDielectricSimulation(p, cpml_material='tensor')._validate_epsilon(rotated)
    inactive = (.2e30*rotated)[None].clone()
    for _, _, index in _cpml_faces(p.region):
        inactive[(slice(None),)+index] = 0
    model._pack(rotated, inactive, omega0, gamma)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_tensor_ade_matches_cpu_and_reservation():
    p = scene(dict(x='pml', y='pec', z='periodic'), 'float32', steps=12)
    generator = torch.Generator().manual_seed(5)
    cpu = parameters(p, generator, torch.float32)
    gpu = tuple(v.detach().cuda().requires_grad_() for v in cpu)
    model = TensorDispersiveSimulation(p, AdjointOptions(checkpoints=2))
    expected = model(*scale(cpu))
    gradients = torch.autograd.grad(expected.signals.abs().square().sum(), cpu)
    torch.cuda.reset_peak_memory_stats()
    actual = model(*scale(gpu))
    got = torch.autograd.grad(actual.signals.abs().square().sum(), gpu)
    torch.testing.assert_close(actual.signals.cpu(), expected.signals, rtol=5e-5, atol=3e-7)
    for a, b in zip(got, gradients):
        torch.testing.assert_close(a.cpu(), b, rtol=5e-4, atol=3e-7*float(b.abs().max()))
    assert torch.cuda.max_memory_allocated() <= actual.report['gpu_reservation_bytes']
