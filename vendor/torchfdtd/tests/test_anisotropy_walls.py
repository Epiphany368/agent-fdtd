"""PEC wall closure and tensors extending into CPML: algebra, symmetry, gradients."""
from itertools import product
import math

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation, Monitor, Project, Region, Source
from torchfdtd.anisotropy import (TensorConstitutive, TensorDielectricSimulation, _TensorSystem,
                                  _cpml_faces, cpml_face_admissible)


def scene(faces, precision='float32', steps=14, mesh=.1, size=1.2, bloch_phase=(0, 0, 0), pml_cells=3):
    boundaries = {a+'_'+side: dict(kind=faces[a]) for a in 'xyz' for side in ('min', 'max')}
    r = Region(dimension='3d', size=(size,)*3, mesh=mesh, steps=steps, precision=precision,
               pml_cells=pml_cells, material_sampling='yee', boundaries=boundaries, bloch_phase=bloch_phase)
    return Project(region=r, sources=[Source(component='Ez', center=(0, 0, 0), wavelength=.7,
        pulse='gaussian', time_definition='standard', pulse_length=.5e-15, pulse_offset=.7e-15)],
        monitors=[Monitor(component=c, center=(.1, 0, 0)) for c in ('Ex', 'Ey', 'Ez', 'Hx')])


def rotation(alpha, beta, gamma, dtype=torch.float64):
    ca, sa, cb, sb, cg, sg = (f(torch.as_tensor(v, dtype=dtype)) for v in (alpha, beta, gamma) for f in (torch.cos, torch.sin))
    rx = torch.stack((torch.ones_like(ca), 0*ca, 0*ca, 0*ca, ca, -sa, 0*ca, sa, ca)).reshape(3, 3)
    ry = torch.stack((cb, 0*cb, sb, 0*cb, torch.ones_like(cb), 0*cb, -sb, 0*cb, cb)).reshape(3, 3)
    rz = torch.stack((cg, -sg, 0*cg, sg, cg, 0*cg, 0*cg, 0*cg, torch.ones_like(cg))).reshape(3, 3)
    return rz @ ry @ rx


def spd(eigenvalues, angles, dtype=torch.float64):
    r = rotation(*angles, dtype=dtype)
    t = r @ torch.diag(torch.as_tensor(eigenvalues, dtype=dtype)) @ r.T
    return (t + t.T) / 2


def dense_wall_constitutive(epsilon, periodic, phases, pec):
    """Explicit incidence matrices with ghost wall nodes, a tiny diagnostic only."""
    shape = epsilon.shape[:3]
    count = math.prod(shape)*3
    dtype = torch.complex128
    matrix = torch.zeros((count, count), dtype=dtype)
    coverage = torch.zeros(count, dtype=torch.float64)
    node_range = [range(n + (1 if pec[a][1] else 0)) for a, n in enumerate(shape)]
    for node in product(*node_range):
        walls = [a for a in range(3) if pec[a][0] and node[a] == 0 or pec[a][1] and node[a] == shape[a]]
        source = tuple(min(node[a], shape[a]-1) for a in range(3))
        eps = epsilon[source].to(dtype)
        if len(walls) == 0:
            kernel = torch.linalg.inv(eps)
        elif len(walls) == 1:
            kernel = torch.zeros((3, 3), dtype=dtype)
            kernel[walls[0], walls[0]] = 1/eps[walls[0], walls[0]]
        else:
            kernel = torch.zeros((3, 3), dtype=dtype)
        for signs in product((0, 1), repeat=3):
            rows = torch.zeros((3, count), dtype=dtype)
            for a in range(3):
                index = list(node)
                coefficient = 1.
                if signs[a]:
                    index[a] -= 1
                    if index[a] < 0:
                        if not periodic[a]:
                            continue
                        index[a] += shape[a]
                        coefficient = 1/phases[a]
                if any(index[c] >= shape[c] for c in range(3)):
                    continue  # ghost edge outside the domain
                flat = ((index[0]*shape[1]+index[1])*shape[2]+index[2])*3+a
                rows[a, flat] = coefficient
            coverage += rows.abs().square().sum(0)/8
            matrix = matrix+rows.conj().T@kernel@rows/8
    scale = torch.ones(count, dtype=torch.float64)
    for a in range(3):
        if periodic[a] or pec[a][1]:
            continue
        edge = torch.zeros(shape+(3,), dtype=torch.bool)
        index = [slice(None)]*3
        index[a] = -1
        edge[tuple(index)+(a,)] = True
        scale[edge.flatten()] = math.sqrt(2.)
    return scale[:, None]*matrix*scale[None, :], coverage


def admissible_mask(shape, pec):
    mask = torch.ones(shape+(3,), dtype=torch.bool)
    for a in range(3):
        if pec[a][0]:
            index = [slice(None)]*3
            index[a] = 0
            for b in range(3):
                if b != a:
                    mask[tuple(index)+(b,)] = False
    return mask


@pytest.mark.parametrize('pec,periodic', [
    (((True, True), (False, False), (False, False)), (False, False, True)),
    (((True, False), (False, True), (True, True)), (False, False, False)),
])
def test_pec_dense_closure_bounds_invariance_transpose_and_vjp(pec, periodic):
    torch.manual_seed(91)
    raw = torch.randn(3, 3, 3, 3, 3, dtype=torch.float64)
    epsilon = (torch.eye(3)+raw@raw.transpose(-1, -2)).requires_grad_()
    phases = (1., 1., complex(math.cos(.3), math.sin(.3)))
    operator = TensorConstitutive(epsilon, phases, periodic, pec)
    matrix, coverage = dense_wall_constitutive(epsilon, periodic, phases, pec)
    torch.testing.assert_close(matrix, matrix.conj().T, atol=2e-15, rtol=2e-15)
    mask = admissible_mask(epsilon.shape[:3], pec).flatten()
    # Wall-tangential components are invariant zero states: their rows and
    # columns vanish. On the admissible subspace S is positive definite and
    # bounded by one, which keeps the conservative vacuum CFL.
    assert torch.count_nonzero(matrix[~mask]) == 0 and torch.count_nonzero(matrix[:, ~mask]) == 0
    reduced = matrix[mask][:, mask]
    eigen = torch.linalg.eigvalsh(reduced)
    assert eigen.min() > 0 and eigen.max() <= 1+1e-12
    assert (coverage[mask] > 0).all()
    x = torch.randn(3, 3, 3, 3, dtype=torch.complex128)
    y = torch.randn_like(x)
    torch.testing.assert_close(operator.apply(x).flatten(), matrix@x.flatten(), atol=2e-15, rtol=2e-14)
    expected, = torch.autograd.grad((y.conj().flatten()*(matrix@x.flatten())).real.sum(), epsilon)
    expected = (expected+expected.transpose(-1, -2))/2
    torch.testing.assert_close(operator.epsilon_vjp(x, y), expected, atol=2e-14, rtol=2e-13)
    assert operator.epsilon_vjp(x, y).shape == epsilon.shape


def test_pec_transpose_and_material_vjp_against_full_autograd_mixed_faces():
    p = scene(dict(x='pec', y='pml', z='bloch'), 'float64', bloch_phase=(0, 0, .37))
    coefficient = spd((1.3, 2.2, 3.1), (.4, -.3, .7)).requires_grad_()
    epsilon = coefficient.expand(p.region.shape+(3, 3))
    system = _TensorSystem(p, epsilon)
    generator = torch.Generator().manual_seed(7)
    state = tuple(torch.randn(v.shape, dtype=v.dtype, generator=generator).requires_grad_() for v in system.state())
    seeds = tuple(torch.randn(v.shape, dtype=v.dtype, generator=generator) for v in state)
    output = system.reference_step(state, 0, epsilon)
    loss = sum((a.conj()*b).real.sum() for a, b in zip(seeds, output))
    expected = torch.autograd.grad(loss, (*state, coefficient))
    actual, gradient = system.transpose_step(state, tuple(v.clone() for v in seeds), torch.zeros(4, dtype=state[0].dtype))
    for a, b in zip(actual, expected[:-1]):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    reference = (expected[-1]+expected[-1].T)/2
    torch.testing.assert_close(gradient.sum((0, 1, 2)), reference, rtol=2e-12, atol=2e-12)
    assert len(state) > 2 and gradient.abs().max() > 0


def test_pec_mirror_symmetry_and_wall_states_remain_zero():
    # Reflecting x flips the sign of the xy and xz entries. Sources and
    # monitors are mirrored; upper and lower wall closures must agree.
    def build(sign, center):
        p = scene(dict(x='pec', y='periodic', z='periodic'), steps=40)
        p.sources[0].center = (sign*center, 0, 0)
        p.monitors = [Monitor(component=c, center=(sign*(center-.2), .1, 0)) for c in ('Ey', 'Ez', 'Hx')]
        return p
    base = spd((1.4, 2.3, 3.), (.5, .3, -.4), torch.float32)
    flip = torch.tensor([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]], dtype=torch.float32)
    left, right = build(1, .3), build(-1, .3)
    a = TensorDielectricSimulation(left)(base.expand(left.region.shape+(3, 3)).clone()).signals
    b = TensorDielectricSimulation(right)((base*flip).expand(right.region.shape+(3, 3)).clone()).signals
    # Ey, Ez and the pseudovector Hx are all even under the x reflection.
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    assert a.abs().max() > 1e-3
    system = _TensorSystem(left, base.expand(left.region.shape+(3, 3)).clone())
    system.advance(0, 40)
    e, h = system.state()[:2]
    assert torch.count_nonzero(e[0, ..., 1:]) == 0 and torch.count_nonzero(h[0, ..., 0]) == 0
    assert e.abs().max() > 1e-3


def test_pec_cavity_exact_discrete_standing_wave_with_transverse_coupling():
    # PEC x walls, a tensor rotated about x couples Ey and Ez. Modes vary only
    # along x, so each transverse eigenpolarization is an exact discrete
    # standing wave with the Yee/leapfrog dispersion, converging to the
    # continuum omega = c*m*pi/(L*sqrt(lambda)). The staggered H initial state
    # is derived from the same discrete relation.
    p = scene(dict(x='pec', y='periodic', z='periodic'), 'float64', steps=10, mesh=.1, size=1.2)
    epsilon = spd((1.9, 2.6, 4.1), (.55, 0., 0.))
    assert abs(epsilon[0, 1]) < 1e-12 and abs(epsilon[0, 2]) < 1e-12 and abs(epsilon[1, 2]) > .1
    system = _TensorSystem(p, epsilon.expand(p.region.shape+(3, 3)))
    system.sources = {'E': [], 'H': []}
    n = p.region.shape[0]
    transverse = torch.linalg.inv(epsilon)[1:, 1:]
    values, vectors = torch.linalg.eigh(transverse)
    courant = system.grid.courant_number
    for mode, polarization in ((2, 0), (3, 1)):
        k = mode*math.pi/n  # per cell
        omega = 2*math.asin(courant*math.sqrt(float(values[polarization]))*math.sin(k/2))  # per step
        x_nodes = torch.arange(n, dtype=torch.float64)
        v = vectors[:, polarization]
        e = torch.zeros(p.region.shape+(3,), dtype=torch.float64)
        e[..., 1] = (v[0]*torch.sin(k*x_nodes))[:, None, None]
        e[..., 2] = (v[1]*torch.sin(k*x_nodes))[:, None, None]
        # Discrete Faraday: H^{n+1/2}-H^{n-1/2} = -C*(E^n(x+1)-E^n(x)) with
        # Hy from +dEz/dx and Hz from -dEy/dx gives the staggered amplitude
        # B = +/- C*v*sin(k/2)/sin(omega/2) at half steps and half cells.
        # The electric update runs first, so the state holds (E^n, H^{n+1/2}).
        h = torch.zeros_like(e)
        amplitude = courant*math.sin(k/2)
        h[..., 1] = (amplitude*v[1]*torch.cos(k*(x_nodes+.5)))[:, None, None]
        h[..., 2] = (-amplitude*v[0]*torch.cos(k*(x_nodes+.5)))[:, None, None]
        state = (e, h)
        steps = 400
        for step in range(steps):
            state = system.reference_step(state, step, epsilon)
        torch.testing.assert_close(state[0], e*math.cos(omega*steps), rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(state[1], h*(math.sin(omega*(steps+.5))/math.sin(omega/2)), rtol=1e-9, atol=1e-11)
        continuum = mode*math.pi*math.sqrt(float(values[polarization]))*courant/n
        assert abs(omega/continuum-1) < .03  # 12-cell Yee dispersion, mode 3 is 2.2%


def tensor_field(r, generator, dtype):
    """Smooth spatially varying SPD field: arbitrary rotations in the interior,
    the geometric CPML criterion in every face's PML layers plus collar.

    A node touched by one CPML face is rotated only about that face normal,
    whose eigenvalue is the largest. A node touched by several faces is
    axis-aligned uniaxial, admissible for every face. Off-diagonals stay
    nonzero in single-face PML regions.
    """
    grid = torch.stack(torch.meshgrid(*(torch.linspace(-1, 1, n, dtype=dtype) for n in r.shape), indexing='ij'), -1)
    angles = .6*torch.sin(2*grid)+.2*torch.randn(3, dtype=dtype, generator=generator)
    values = 1.2+torch.tensor([.3, 1.1, 2.4], dtype=dtype)+.3*torch.cos(3*grid)
    faces = torch.zeros(r.shape+(3,), dtype=torch.bool)
    for axis, side, index in _cpml_faces(r):
        faces[index+(axis,)] = True
    out = torch.empty(r.shape+(3, 3), dtype=dtype)
    for index in product(*(range(n) for n in r.shape)):
        touched = [a for a in range(3) if faces[index+(a,)]]
        if not touched:
            out[index] = spd(values[index], angles[index].tolist(), dtype)
        elif len(touched) == 1:
            axis = touched[0]
            angle = [0., 0., 0.]
            angle[axis] = float(angles[index][axis])
            order = sorted(values[index].tolist())
            eigen = [order[0], order[1], order[2]]
            eigen[axis], eigen[2] = eigen[2], eigen[axis]  # largest along the face normal
            out[index] = spd(eigen, angle, dtype)
        else:
            low, high = float(values[index].min()), float(values[index].max())
            out[index] = torch.diag(torch.tensor([low, low, high], dtype=dtype))
    return out


@pytest.mark.parametrize('faces', [dict(x='pml', y='pml', z='pml'), dict(x='pml', y='pec', z='bloch')])
def test_tensor_into_cpml_checkpoint_vjp_matches_full_autograd(faces):
    p = scene(faces, 'float64', bloch_phase=(0, 0, .29) if faces['z'] == 'bloch' else (0, 0, 0))
    generator = torch.Generator().manual_seed(23)
    epsilon = tensor_field(p.region, generator, torch.float64).requires_grad_()
    system = _TensorSystem(p, epsilon)
    state = tuple(torch.zeros_like(v) for v in system.state())
    samples = []
    for step in range(p.region.steps):
        state = system.reference_step(state, step, epsilon)
        samples.append(system.observe(state))
    oracle = torch.stack(samples)
    expected, = torch.autograd.grad(oracle.abs().square().sum(), epsilon)
    expected = (expected+expected.transpose(-1, -2))/2
    model = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2), cpml_material='tensor')
    result = model(epsilon)
    actual, = torch.autograd.grad(result.signals.abs().square().sum(), epsilon)
    torch.testing.assert_close(result.signals, oracle, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-13)
    layers = p.region.pml_layers(0, 0)
    assert actual[:layers].abs().max() > 0 and 'extend into CPML' in result.report['cpml_contract']
    assert result.report['cpml_collar_material_vjp'].startswith('full')


def test_tensor_cpml_rotation_taylor_and_central_difference_and_isotropic_parity():
    # Uniform tensor through the x CPML faces: x is a principal axis with the
    # largest eigenvalue, the yz block rotates by the design angle.
    p = scene(dict(x='pml', y='periodic', z='periodic'), 'float64', steps=16)
    model = TensorDielectricSimulation(p, AdjointOptions(checkpoints=3), cpml_material='tensor')
    shape = p.region.shape+(3, 3)
    def loss(angle):
        epsilon = spd((3.3, 1.5, 2.4), (angle, 0., 0.)).expand(shape)
        return model(epsilon).signals.abs().square().sum()
    theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    value = loss(theta)
    derivative, = torch.autograd.grad(value, theta)
    step = 1e-4
    central = (loss(theta.detach()+step)-loss(theta.detach()-step))/(2*step)
    torch.testing.assert_close(derivative, central, rtol=1e-6, atol=1e-12)
    remainders = []
    for h in (1e-2, 5e-3, 2.5e-3):
        remainders.append(float((loss(theta.detach()+h)-value.detach()-h*derivative).abs()))
    assert remainders[1] < .3*remainders[0] and remainders[2] < .3*remainders[1]
    assert abs(float(derivative)) > 1e-8
    # Constant isotropic tensor through six CPML faces equals the scalar solver.
    q = scene(dict(x='pml', y='pml', z='pml'), 'float64', steps=16)
    scalar = torch.full(q.region.shape, 2., dtype=torch.float64)
    native = DifferentiableSimulation(q)(scalar).signals
    tensor = TensorDielectricSimulation(q, AdjointOptions(checkpoints=3), cpml_material='tensor')(
        (2*torch.eye(3, dtype=torch.float64)).expand(shape).clone()).signals
    torch.testing.assert_close(tensor, native, rtol=2e-12, atol=2e-14)


def test_tensor_cpml_mode_rejects_fixed_scalar_and_keeps_isotropic_default_error():
    p = scene(dict(x='pml', y='pml', z='pml'))
    with pytest.raises(ValueError, match='explicit fixed'):
        TensorDielectricSimulation(p)
    with pytest.raises(ValueError, match='takes no fixed'):
        TensorDielectricSimulation(p, cpml_material='tensor', cpml_background_epsilon=2.)
    with pytest.raises(ValueError, match='cpml_material'):
        TensorDielectricSimulation(p, cpml_material='anything')
    # Walls with symmetric/PMC faces stay explicit errors.
    q = scene(dict(x='pmc', y='pmc', z='pmc'))
    with pytest.raises(ValueError, match='PMC/symmetric/antisymmetric'):
        TensorDielectricSimulation(q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('faces', [dict(x='pml', y='pml', z='pml'), dict(x='pec', y='pml', z='periodic')])
def test_cuda_walls_and_tensor_cpml_match_cpu_and_reservation(faces):
    p = scene(faces, 'float32', steps=12)
    generator = torch.Generator().manual_seed(5)
    cpu = tensor_field(p.region, generator, torch.float32).requires_grad_()
    gpu = cpu.detach().cuda().requires_grad_()
    model = TensorDielectricSimulation(p, AdjointOptions(checkpoints=2), cpml_material='tensor')
    expected = model(cpu)
    gradient, = torch.autograd.grad(expected.signals.abs().square().sum(), cpu)
    torch.cuda.reset_peak_memory_stats()
    actual = model(gpu)
    got, = torch.autograd.grad(actual.signals.abs().square().sum(), gpu)
    torch.testing.assert_close(actual.signals.cpu(), expected.signals, rtol=5e-5, atol=3e-7)
    torch.testing.assert_close(got.cpu(), gradient, rtol=5e-4, atol=3e-7)
    assert torch.cuda.max_memory_allocated() <= actual.report['gpu_reservation_bytes']


def one_step_matrix(system, epsilon):
    """Dense one-step map of the source-free linear update, a tiny diagnostic only."""
    system.sources = {'E': [], 'H': []}
    state = system.state()
    sizes = [v.numel() for v in state]
    total = sum(sizes)
    matrix = torch.zeros((total, total), dtype=state[0].dtype)
    with torch.no_grad():
        for column in range(total):
            unit = torch.zeros(total, dtype=state[0].dtype)
            unit[column] = 1
            parts = tuple(part.reshape(v.shape) for part, v in zip(unit.split(sizes), state))
            matrix[:, column] = torch.cat([o.reshape(-1) for o in system.reference_step(parts, 0, epsilon)])
    return matrix


def bloch_scene(faces, shape, phases, pml_cells=3):
    boundaries = {a+'_'+side: dict(kind=faces[a]) for a in 'xyz' for side in ('min', 'max')}
    r = Region(dimension='3d', size=tuple(.1*n for n in shape), mesh=.1, steps=10, precision='float64',
               pml_cells=pml_cells, material_sampling='yee', boundaries=boundaries, bloch_phase=phases)
    assert r.shape == tuple(shape)
    return Project(region=r, sources=[], monitors=[Monitor(component='Ez', center=(0, 0, 0))])


@pytest.mark.parametrize('eigenvalues,angles,admissible', [
    ((2., 1., 4.), (0., .5, 0.), True),   # y principal with the smallest eigenvalue, xz block rotated
    ((1., 4., 2.), (0., .5, 0.), True),   # y principal with the largest eigenvalue
    ((1.5, 3., 6.), (0., .5, 0.), False),  # y principal but strictly intermediate: optic-axis cone
    ((1., 1., 4.), (.5, 0., 0.), False),  # y is not a principal axis
])
def test_cpml_face_criterion_matches_dense_one_step_spectral_radius(eigenvalues, angles, admissible):
    # y CPML faces, x and z Bloch phases place a mode at (kx, kz) = (pi, pi/2)
    # per cell, near the discrete optic-axis direction of these tensors. The
    # geometric criterion of Becache, Fauqueux and Joly separates exactly
    # unit spectral radius from growth, and admission reproduces the split.
    p = bloch_scene(dict(x='bloch', y='pml', z='bloch'), (5, 11, 5), (math.pi, 0., math.pi/2))
    epsilon = spd(eigenvalues, angles).expand(p.region.shape+(3, 3)).contiguous()
    assert bool(cpml_face_admissible(epsilon, 1).all()) == admissible
    model = TensorDielectricSimulation(p, cpml_material='tensor')
    if admissible:
        model._validate_epsilon(epsilon)
    else:
        with pytest.raises(ValueError, match='y_min'):
            model._validate_epsilon(epsilon)
    radius = float(torch.linalg.eigvals(one_step_matrix(_TensorSystem(p, epsilon), epsilon)).abs().max())
    if admissible:
        assert radius <= 1+1e-9, radius
    else:
        assert radius > 1+1e-3, radius


def test_tensor_cpml_admission_names_face_and_keeps_interior_free():
    p = scene(dict(x='pml', y='pml', z='pml'), 'float64')
    model = TensorDielectricSimulation(p, cpml_material='tensor')
    generator = torch.Generator().manual_seed(3)
    epsilon = tensor_field(p.region, generator, torch.float64)
    model._validate_epsilon(epsilon)
    layers = p.region.pml_layers(0, 0)
    interior = epsilon[layers+1:-layers-1, layers+1:-layers-1, layers+1:-layers-1]
    assert interior.shape[0] > 0 and (interior[..., 0, 1] != 0).any() and (interior[..., 0, 2] != 0).any()
    # A rotated interior tensor leaking one node row into the z_max collar is rejected by name.
    leaked = epsilon.clone()
    leaked[layers+1:-layers-1, layers+1:-layers-1, -layers-1] = spd((1.5, 2.4, 3.3), (.4, .2, -.5))
    with pytest.raises(ValueError, match='z_max'):
        model._validate_epsilon(leaked)
    # Biaxial with the face normal strictly intermediate is rejected even when diagonal.
    biaxial = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64)).expand(p.region.shape+(3, 3)).contiguous()
    with pytest.raises(ValueError, match='y_min'):
        model._validate_epsilon(biaxial)
    # The same biaxial tensor is admissible when only the extreme axes carry CPML.
    q = scene(dict(x='pml', y='periodic', z='pml'), 'float64')
    TensorDielectricSimulation(q, cpml_material='tensor')._validate_epsilon(biaxial)
