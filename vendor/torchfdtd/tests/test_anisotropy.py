"""Independent staggered symbol, actual objective VJP, and admission gates."""
import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions, BoundaryFace, DifferentiableSimulation, Monitor, Project, Region, Source
from torchfdtd.anisotropy import TensorConstitutive, TensorDielectricSimulation, _TensorSystem


def scene(precision='float32', bloch=False, steps=12):
    r = Region(dimension='3d', size=(.6, 1., 1.25), mesh_steps=(.1, .2, .25),
               steps=steps, precision=precision, backend='cpu', material_sampling='yee',
               boundaries={a+'_'+side: {'kind': 'bloch' if bloch else 'periodic'} for a in 'xyz' for side in ('min','max')},
               bloch_phase=(.31, -.23, .19) if bloch else (0, 0, 0))
    for a in 'xyz':
        for side in ('min', 'max'):
            setattr(r.boundaries, a + '_' + side, BoundaryFace(kind='bloch' if bloch else 'periodic'))
    return Project(region=r, sources=[Source(component='Ez', center=(0, 0, 0),
                   pulse='continuous', wavelength=1.2)],
                   monitors=[Monitor(component=c, center=(.1, .1, 0)) for c in ('Ex', 'Ey', 'Ez', 'Hx')])


def matrix(dtype):
    # Six nonzero entries, with a margin above the required eigenvalue bound.
    return torch.tensor([[2.3, .27, -.18], [.27, 1.8, .21], [-.18, .21, 2.7]], dtype=dtype)


def test_constant_diagonal_native_parity():
    p = scene()
    diag = torch.tensor([1.6, 2., 2.4]).expand(p.region.shape + (3,))
    native = DifferentiableSimulation(p)(diag).signals
    result = TensorDielectricSimulation(p)(torch.diag_embed(diag))
    torch.testing.assert_close(result.signals, native, rtol=3e-5, atol=2e-7)
    assert result.report['full_time_autograd'] is False


@pytest.mark.parametrize('bloch', [False, True])
def test_tensor_adjoint_all_six_components_on_actual_field_objective(bloch):
    # FP64 is a tiny derivative diagnostic, not the production default.
    p = scene('float64', bloch=bloch)
    model = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2))
    coefficients = matrix(torch.float64).requires_grad_()
    epsilon = coefficients.expand(p.region.shape + (3, 3))
    weights = torch.linspace(.2, 1.3, p.region.steps * 4).reshape(-1, 4)
    def loss(value):
        signals = model(value.expand_as(epsilon)).signals
        return (signals.abs().square() * weights).sum()
    value = loss(coefficients)
    gradient, = torch.autograd.grad(value, coefficients)
    for a in range(3):
        for b in range(a, 3):
            direction = torch.zeros_like(coefficients)
            direction[a, b] = direction[b, a] = 1
            step = 2e-5
            difference = (loss(coefficients.detach() + step * direction)
                          - loss(coefficients.detach() - step * direction)) / (2 * step)
            derivative = (gradient * direction).sum()
            torch.testing.assert_close(derivative, difference, rtol=2e-5, atol=1e-10)
    assert gradient.norm() > 1e-6


def test_bloch_operator_hermiticity_positivity_and_local_material_vjp():
    generator = torch.Generator().manual_seed(18)
    raw = torch.randn((3, 4, 2, 3, 3), dtype=torch.float64, generator=generator)
    epsilon = (torch.eye(3) + raw @ raw.transpose(-1, -2)).requires_grad_()
    phases = tuple(np.exp(1j * x) for x in (.7, -.4, .2))
    operator = TensorConstitutive(epsilon, phases)
    x = torch.randn((3, 4, 2, 3), dtype=torch.complex128, generator=generator)
    y = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    sx, sy = operator.apply(x), operator.apply(y)
    torch.testing.assert_close((y.conj() * sx).sum(), (sy.conj() * x).sum())
    assert (x.conj() * sx).sum().real > 0
    assert (x.conj() * sx).sum().real <= x.abs().square().sum()
    expected, = torch.autograd.grad((y.conj() * sx).real.sum(), epsilon)
    expected = (expected + expected.transpose(-1, -2)) / 2
    torch.testing.assert_close(operator.epsilon_vjp(x, y), expected)


def test_rotated_tensor_independent_fourier_symbol_and_continuum_refinement():
    # Physical component phases include their own half-cell coordinates.
    # Cross-component symbol: K_ab cos(q_a/2) cos(q_b/2), diagonal K_aa.
    p = scene('float64', bloch=True, steps=10)
    eps = matrix(torch.float64).expand(p.region.shape + (3, 3))
    system = _TensorSystem(p, eps)
    q = np.array(p.region.bloch_phase) / np.array(p.region.shape)
    coordinates = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float64)
                               for n in p.region.shape), indexing='ij'), -1)
    phase = torch.exp(1j * (coordinates * torch.tensor(q)).sum(-1))
    ephase = phase[..., None] * torch.exp(.5j * torch.tensor(q))
    hphase = phase[..., None] * torch.exp(.5j * torch.tensor(q.sum() - q))
    k = np.linalg.inv(matrix(torch.float64).numpy())
    symbol = k * np.outer(np.cos(q / 2), np.cos(q / 2))
    np.fill_diagonal(symbol, np.diag(k))
    amplitude = torch.tensor([.7 + .2j, -.4j, .3], dtype=torch.complex128)
    actual = system.operator.apply(ephase * amplitude) / ephase
    expected = torch.tensor(symbol, dtype=torch.complex128) @ amplitude
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=2e-12, atol=2e-12)
    spacing = np.array(p.region.axis_steps)
    wave = 2 * np.sin(q / 2) / spacing * p.region.reference_step
    cross = torch.tensor([[0, -wave[2], wave[1]], [wave[2], 0, -wave[0]],
                          [-wave[1], wave[0], 0]], dtype=torch.complex128)
    system.sources = {'E': [], 'H': []}
    h = torch.tensor([.2j, .3, -.1j], dtype=torch.complex128)
    new_e, new_h = system.reference_step((ephase * amplitude, hphase * h), 0, eps)
    electric = amplitude + system.grid.courant_number * torch.tensor(symbol, dtype=torch.complex128) @ (1j * cross @ h)
    magnetic = h - system.grid.courant_number * (1j * cross @ electric)
    torch.testing.assert_close(new_e / ephase, electric.expand_as(new_e), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(new_h / hphase, magnetic.expand_as(new_h), rtol=2e-12, atol=2e-12)
    # Independent modified leapfrog energy for this resolved Fourier mode.
    # The inverse here is only a 3x3 diagnostic, not a global production solve.
    symbol_inverse = torch.linalg.inv(torch.tensor(symbol, dtype=torch.complex128))
    def energy(e, h):
        ea, ha = e[0, 0, 0] / ephase[0, 0, 0], h[0, 0, 0] / hphase[0, 0, 0]
        return ((ea.conj() * (symbol_inverse @ ea)).sum().real
                + ha.abs().square().sum()
                + system.grid.courant_number * (ea.conj() * (1j * cross @ ha)).sum().real)
    state = (ephase * amplitude, hphase * h)
    initial = energy(*state)
    for step in range(256):
        state = system.reference_step(state, step, eps)
    torch.testing.assert_close(energy(*state), initial, rtol=2e-12, atol=2e-12)
    physical_wave = np.array([.7, -.4, .9])
    continuum = np.sort(np.linalg.eigvals(k @ (np.dot(physical_wave, physical_wave) * np.eye(3)
                                              - np.outer(physical_wave, physical_wave))).real)[1:]
    errors = []
    for h in (.4, .2, .1):
        q = physical_wave * h
        kd = 2 * np.sin(q / 2) / h
        sd = k * np.outer(np.cos(q / 2), np.cos(q / 2))
        np.fill_diagonal(sd, np.diag(k))
        discrete = np.sort(np.linalg.eigvals(sd @ (np.dot(kd, kd) * np.eye(3) - np.outer(kd, kd))).real)[1:]
        errors.append(np.max(np.abs(discrete / continuum - 1)))
    assert errors[1] < .27 * errors[0]
    assert errors[2] < .27 * errors[1]


def test_budget_rejects_before_inverse_or_fields_and_unsupported_contracts(monkeypatch):
    p = scene()
    model = TensorDielectricSimulation(p, AdjointOptions(host_budget_bytes=1))
    epsilon = matrix(torch.float32).expand(p.region.shape + (3, 3))
    def forbidden(*args, **kwargs):
        raise AssertionError('Large allocation or linear algebra before admission')
    monkeypatch.setattr(torch.linalg, 'inv', forbidden)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', forbidden)
    with pytest.raises(ValueError, match='budget'):
        model(epsilon)
    for a in 'xyz':
        for side in ('min', 'max'):
            setattr(p.region.boundaries, a + '_' + side, BoundaryFace(kind='pmc'))
    with pytest.raises(ValueError, match='PMC/symmetric/antisymmetric'):
        TensorDielectricSimulation(p)


def test_tensor_input_validation_and_online_dft():
    p = scene(steps=10)
    model = TensorDielectricSimulation(p)
    epsilon = matrix(torch.float32).expand(p.region.shape + (3, 3)).clone()
    bad = epsilon.clone()
    bad[..., 0, 1] += .1
    with pytest.raises(ValueError, match='symmetric'):
        model(bad)
    with pytest.raises(ValueError, match='eigenvalues'):
        model(.1 * epsilon)
    epsilon.requires_grad_()
    spectral = model.spectrum(epsilon, [2e14])
    history = model(epsilon).spectrum([2e14])
    torch.testing.assert_close(spectral.fields, history, rtol=2e-5, atol=1e-20)
    gradient, = torch.autograd.grad(spectral.fields.abs().square().sum(), epsilon)
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Local CUDA unavailable')
@pytest.mark.parametrize('bloch', [False, True])
def test_cuda_tensor_runtime_and_checkpoint_gradient_match_cpu(bloch):
    p = scene(bloch=bloch, steps=10)
    model = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2))
    cpu = matrix(torch.float32).expand(p.region.shape + (3, 3)).clone().requires_grad_()
    gpu = cpu.detach().cuda().requires_grad_()
    expected = model(cpu)
    gradient, = torch.autograd.grad(expected.signals.abs().square().sum(), cpu)
    torch.cuda.reset_peak_memory_stats()
    actual = model(gpu)
    got, = torch.autograd.grad(actual.signals.abs().square().sum(), gpu)
    torch.testing.assert_close(actual.signals.cpu(), expected.signals, rtol=3e-5, atol=2e-7)
    torch.testing.assert_close(got.cpu(), gradient, rtol=5e-4, atol=2e-7)
    assert torch.cuda.max_memory_allocated() < actual.report['gpu_reservation_bytes']


def test_tensor_reservation_scales_linearly_with_cells():
    small = scene()
    large = scene()
    large.region.size = tuple(2 * value for value in small.region.size)
    first = TensorDielectricSimulation(small).reservation()
    second = TensorDielectricSimulation(large).reservation()
    assert second['tensor_reservation_bytes'] == 8 * first['tensor_reservation_bytes']
    assert second['restart_state_bytes'] == 8 * first['restart_state_bytes']
