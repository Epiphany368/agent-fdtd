"""PMC/symmetric faces in the Yee adjoint, streamed X-slab and tensor-batch paths.

Every check keeps the wall at the mesh endpoint with the stored upper face and
edge arrays. Physics is validated against the discrete cavity dispersion, the
independent exact-endpoint solver and mirrored full domains without PMC.
"""
import math
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Monitor, AdjointOptions, DifferentiableSimulation
from torchfdtd.models import Boundaries, BoundaryFace, Material, Structure
from torchfdtd.boundaries import BoundaryDescription, YeeGrid, material_shape
from torchfdtd.differentiable import _System
from torchfdtd.solver import Simulation, field_axes, voxelize

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
PROFILE = dict(layers=4, kappa=3., alpha=.05, polynomial=2., alpha_polynomial=1., sigma_scale=.8)


def scene(kinds, *, size, steps=12, mesh=.1, precision='float32', profiles=None, sources=(), monitors=(),
          backend='cpu', materials=(), structures=()):
    faces = {}
    for axis, name in enumerate('xyz'):
        for side, kind in zip(('min', 'max'), kinds[axis]):
            key = name+'_'+side
            extra = dict(layers=3, alpha=0.) if kind == 'pml' else {}
            extra.update((profiles or {}).get(key, {}))
            faces[key] = BoundaryFace(kind=kind, **extra)
    region = Region(dimension='3d', size=size, mesh=mesh, steps=steps, boundaries=Boundaries(**faces),
                    material_sampling='yee', precision=precision, pml_cells=3, backend=backend, cuda_kernel='torch')
    project = Project(region=region, sources=list(sources), monitors=list(monitors), structures=list(structures),
                      **(dict(materials=list(materials)) if materials else {}))
    return Project.model_validate(project.model_dump())


def at(region, component, index):
    """Physical centre of one Yee sample, including stored upper PMC nodes."""
    return tuple(float(axis[i]) for axis, i in zip(field_axes(region, component), index))


def random_epsilon(region, seed, *, device='cpu', diagonal=True):
    shape = material_shape(region, diagonal)
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64 if region.precision == 'float64' else torch.float32
    return (1.5+torch.rand(shape, generator=generator, dtype=torch.float64)).to(device=device, dtype=dtype)


def objective(signals):
    return signals.square().sum()


# ---------------------------------------------------------------------------
# Physics: discrete cavity dispersion and the independent endpoint solver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('axis', range(3))
def test_pmc_cavity_eigenmode_follows_discrete_dispersion(axis):
    kinds = [None]*3
    kinds[axis] = ('pmc', 'pmc');kinds[(axis+1) % 3] = ('pec', 'pec');kinds[(axis+2) % 3] = ('pmc', 'pmc')
    p = scene(kinds, size=(1.2, .6, .6), steps=10)
    system = _System(p, torch.ones(material_shape(p.region)))
    grid = system.grid;shape = p.region.shape;n = shape[axis]
    q = math.pi/n;omega = 2*math.asin(grid.courant_number*math.sin(q/2))
    component = (axis+1) % 3
    # E_c = cos(q x_a) is tangential on the PMC faces (even) and normal on the
    # PEC faces; its curl partner is odd along axis a and vanishes on the walls.
    parts = [torch.zeros(tuple(n+(1 if a in grid.pmc_upper and (c != a) else 0) for a, n in enumerate(shape)))
             for c in range(3)]
    wave = torch.cos(q*torch.arange(parts[component].shape[axis], dtype=torch.float32))
    parts[component] += wave.reshape([-1 if a == axis else 1 for a in range(3)])
    e, faces_e = system._restricted(parts, 'E', shape)
    curl, faces_h, _ = system.curl_faces(e, faces_e, (), True)
    h = -grid.courant_number/2*curl;faces_h = tuple(-grid.courant_number/2*f for f in faces_h)
    stored = [f for f, (c, _, _) in zip(faces_e, grid.pmc_blocks['E']) if c == component]
    assert stored and all(f.abs().max() > .5 for f in stored)
    state = (e, h, *faces_e, *faces_h)
    steps = 35
    for step in range(steps):state = system.reference_step(state, step, system.epsilon)
    factor = math.cos(steps*omega)
    torch.testing.assert_close(state[0], e*factor, atol=4e-6, rtol=4e-6)
    for actual, expected in zip(state[2:2+len(faces_e)], faces_e):
        torch.testing.assert_close(actual, expected*factor, atol=4e-6, rtol=4e-6)


def cavity_scene(backend='cpu', steps=40):
    kinds = (('pec', 'pmc'), ('pmc', 'pmc'), ('pec', 'pmc'))
    p = scene(kinds, size=(.8, .7, .6), steps=steps, backend=backend, materials=[Material(name='glass', index=1.7)],
              structures=[Structure(name='block', material='glass', kind='rectangle', center=(.15, .05, .1), size=(.3, .3, .3))])
    r = p.region;n = r.shape
    p.sources = [Source(center=at(r, 'Ez', (2, 2, 1)), component='Ez', pulse='continuous', wavelength=.5),
                 Source(center=at(r, 'Ey', (n[0], 3, 2)), component='Ey', pulse='gaussian', wavelength=.5, amplitude=.5)]
    p.monitors = [Monitor(center=at(r, 'Ez', (3, 4, 2)), component='Ez'),
                  Monitor(center=at(r, 'Ez', (n[0], n[1], 3)), component='Ez'),   # upper x/y edge
                  Monitor(center=at(r, 'Ex', (2, n[1], n[2])), component='Ex'),   # upper y/z edge
                  Monitor(center=at(r, 'Hx', (n[0], 2, 3)), component='Hx'),      # upper x face, normal H
                  Monitor(center=at(r, 'Hy', (1, 0, 4)), component='Hy'),         # lower y wall, normal H
                  Monitor(center=at(r, 'Ey', (4, 3, n[2])), component='Ey')]      # upper z face
    return Project.model_validate(p.model_dump())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
def test_yee_faces_match_independent_exact_endpoint_solver(device):
    if device == 'cuda':pytest.importorskip('cupy')
    p = cavity_scene(backend=device)
    expected = Simulation(p).run()
    assert expected.summary['engine'] == 'TorchFDTD exact-endpoint PEC/PMC'
    epsilon, _ = voxelize(p)
    assert epsilon.shape == material_shape(p.region, True)
    with torch.no_grad():
        actual = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))(torch.as_tensor(epsilon, device=device))
    assert actual.report['pmc_faces'] is True
    signals = actual.signals.cpu().numpy()
    assert np.abs(expected.signals).max() > .05 and np.abs(signals[:, 1:]).max(axis=0).min() > 1e-4
    np.testing.assert_allclose(signals, expected.signals, rtol=3e-5, atol=2e-6)


# ---------------------------------------------------------------------------
# Transpose and mirrored full-domain references
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('explicit', [False, True])
@torch.enable_grad()
def test_face_curl_transpose_with_nonzero_psi(explicit):
    kinds = (('pml', 'pmc'), ('pmc', 'pml'), ('pec', 'pmc'))
    p = scene(kinds, size=(1.0, 1.0, .6), profiles={'x_min': PROFILE, 'y_max': dict(layers=3, kappa=2., alpha=.01)})
    if explicit:
        shape = p.region.shape
        p.region.mesh_type = 'explicit'
        p.region.mesh_coordinates = tuple(tuple(float(v) for v in .5*s*(.8*np.linspace(-1, 1, n+1)+.2*np.linspace(-1, 1, n+1)**3))
                                          for s, n in zip(p.region.size, shape))
        p = Project.model_validate(p.model_dump())
    system = _System(p, torch.ones(material_shape(p.region)))
    g = system.grid
    assert set(g.pmc_upper) == {0, 2} and set(g.pmc_lower) == {1}
    for forward in (False, True):
        family = 'E' if forward else 'H'
        field = torch.randn_like(g.E, requires_grad=True)
        faces = [torch.randn_like(f, requires_grad=True) for f in g.faces[family]]
        psis = [torch.randn_like(seg['psi'], requires_grad=True) for seg in system.segments]
        result, faces_out, updated = system.curl_faces(field, faces, psis, forward)
        seeds = [torch.randn_like(result), *(torch.randn_like(f) for f in faces_out), *(torch.randn_like(u) for u in updated)]
        outputs = [result, *faces_out, *updated]
        expected = torch.autograd.grad(sum((o*s).sum() for o, s in zip(outputs, seeds)), [field, *faces, *psis], allow_unused=True)
        actual, actual_faces, previous = system.curl_faces_transpose(seeds[0], seeds[1:1+len(faces_out)], seeds[1+len(faces_out):], forward)
        torch.testing.assert_close(actual, expected[0], atol=3e-6, rtol=3e-6)
        for value, reference in zip(actual_faces, expected[1:1+len(faces)]):
            torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-6)
        for value, reference in zip(previous, expected[1+len(faces):]):
            torch.testing.assert_close(value, torch.zeros_like(value) if reference is None else reference, atol=3e-6, rtol=3e-6)


MIRRORS = {
    # half kinds, mirrored (axis, side), half size, sources (component, index), monitors (component, index)
    'upper_yz_edge': dict(kinds=(('pml', 'pml'), ('pec', 'pmc'), ('pec', 'pmc')), mirrors=((1, 1), (2, 1)), size=(1.2, .6, .6),
                          sources=(('Ex', (6, 6, 6)), ('Ez', (5, 2, 1))),
                          monitors=(('Ex', (4, 6, 6)), ('Ez', (7, 6, 3)), ('Hy', (8, 6, 2)), ('Hx', (5, 2, 4)))),
    'upper_x_lower_z': dict(kinds=(('pec', 'pmc'), ('pml', 'pml'), ('pmc', 'pec')), mirrors=((0, 1), (2, 0)), size=(.6, 1.2, .6),
                            sources=(('Ey', (6, 5, 2)), ('Ez', (2, 6, 0))),
                            monitors=(('Ez', (6, 4, 1)), ('Hx', (6, 7, 2)), ('Ey', (3, 5, 0)), ('Hz', (4, 6, 0)))),
}


def mirrored_pair(name, *, precision='float32', profiles=None):
    spec = MIRRORS[name]
    half = scene(spec['kinds'], size=spec['size'], steps=16, precision=precision, profiles=profiles)
    kinds = [list(k) for k in spec['kinds']];size = list(spec['size'])
    for axis, _ in spec['mirrors']:
        kinds[axis] = ['pec', 'pec'];size[axis] *= 2
    full = scene(kinds, size=tuple(size), steps=16, precision=precision, profiles=profiles)
    n = half.region.shape

    def image_indices(component, index):
        """Full-domain indices and signs of every mirror image of one half-domain sample."""
        images = [(list(index), 1.)]
        for axis, side in spec['mirrors']:
            nodal = 'xyz'.index(component[1].lower()) != axis
            offset = n[axis] if side == 0 else 0
            extended = []
            for full_index, sign in images:
                base = full_index[axis]+offset
                mirrored = 2*n[axis]-base if nodal else 2*n[axis]-1-base
                first = list(full_index);first[axis] = base
                extended.append((first, sign))
                if mirrored != base:
                    second = list(full_index);second[axis] = mirrored
                    extended.append((second, sign*(-1. if not nodal else 1.)))
            images = extended
        return [(tuple(i), s) for i, s in images]

    for component, index in spec['sources']:
        half.sources.append(Source(component=component, center=at(half.region, component, index), pulse='continuous', wavelength=.5))
        for full_index, sign in image_indices(component, index):
            full.sources.append(Source(component=component, center=at(full.region, component, full_index), pulse='continuous',
                                       wavelength=.5, phase=0. if sign > 0 else 180.))
    for component, index in spec['monitors']:
        half.monitors.append(Monitor(component=component, center=at(half.region, component, index)))
        (full_index, _), = image_indices(component, index)[:1]
        full.monitors.append(Monitor(component=component, center=at(full.region, component, full_index)))
    half = Project.model_validate(half.model_dump());full = Project.model_validate(full.model_dump())

    def images_of_material(component, index):
        return [i for i, _ in image_indices('E'+'xyz'[component], index)]
    return half, full, images_of_material


def material_samples(half, full, images):
    """(half index, component, full-domain images) of every physical epsilon sample.

    Rows at the upper wall of a half-cell component are inert padding and are skipped.
    """
    shape = material_shape(half.region, True)
    for component in range(3):
        for index in np.ndindex(*shape[:3]):
            if any(i == n and a == component for a, (i, n) in enumerate(zip(index, half.region.shape))):continue
            yield index, component, [t for t in images(component, index) if all(v < s for v, s in zip(t, full.region.shape))]


@pytest.mark.parametrize('name', list(MIRRORS))
@torch.enable_grad()
def test_symmetric_half_domain_matches_mirrored_full_domain(name):
    half, full, images = mirrored_pair(name)
    eps_half = random_epsilon(half.region, 7).requires_grad_()
    eps_full = torch.ones(material_shape(full.region, True))
    with torch.no_grad():
        for index, component, targets in material_samples(half, full, images):
            for target in targets:eps_full[target+(component,)] = eps_half[index+(component,)]
    eps_full.requires_grad_()
    reference = DifferentiableSimulation(full).reference(eps_full)
    expected, = torch.autograd.grad(objective(reference), eps_full)
    result = DifferentiableSimulation(half, AdjointOptions(checkpoints=3))(eps_half)
    actual, = torch.autograd.grad(objective(result.signals), eps_half)
    assert reference.abs().max() > 1e-3
    torch.testing.assert_close(result.signals, reference, atol=4e-6, rtol=4e-5)
    folded = torch.zeros_like(eps_half)
    for index, component, targets in material_samples(half, full, images):
        for target in targets:folded[index+(component,)] += expected[target+(component,)]
    # Wall rows carry real gradient; inert padding rows of half-cell components stay exactly zero.
    for axis, side in MIRRORS[name]['mirrors']:
        if side != 1:continue
        cut = [slice(None)]*4;cut[axis] = half.region.shape[axis]
        wall = actual[tuple(cut)]
        assert wall.abs().max() > 0 and wall[..., axis].abs().max() == 0
    torch.testing.assert_close(actual, folded, atol=2e-6*expected.abs().max().item(), rtol=4e-5)


# ---------------------------------------------------------------------------
# Gradient checks: Taylor, central differences and CUDA against autograd
# ---------------------------------------------------------------------------
def gradient_scene(precision, *, backend='cpu'):
    kinds = (('pml', 'pmc'), ('pmc', 'pec'), ('pec', 'pmc'))
    profiles = {'x_min': PROFILE}
    p = scene(kinds, size=(1.0, .6, .6), steps=14, precision=precision, profiles=profiles, backend=backend)
    r = p.region;n = r.shape
    p.sources = [Source(center=at(r, 'Ez', (4, 0, 2)), component='Ez', pulse='continuous', wavelength=.5),
                 Source(center=at(r, 'Ey', (n[0], 2, n[2])), component='Ey', pulse='continuous', wavelength=.45, amplitude=.7)]
    p.monitors = [Monitor(center=at(r, 'Ez', (n[0], 3, 1)), component='Ez'), Monitor(center=at(r, 'Hx', (n[0], 1, 3)), component='Hx'),
                  Monitor(center=at(r, 'Ex', (5, 2, n[2])), component='Ex'), Monitor(center=at(r, 'Hy', (6, 0, 2)), component='Hy')]
    return Project.model_validate(p.model_dump())


def test_per_face_cpml_profiles_are_independent_next_to_pmc():
    p = gradient_scene('float32')
    description = BoundaryDescription(p.region)
    lower = description.cpml[True, 0, 1][0];assert lower['b'].shape[0] == PROFILE['layers']
    assert lower['inv_k'].min() < .6 and (lower['psi'] is None)
    # Auxiliary rows follow their derivative target onto the stored z-upper nodes.
    assert description.cpml[True, 0, 1][0]['shape'][2] == p.region.shape[2]+1
    assert description.cpml[True, 0, 2][0]['shape'][2] == p.region.shape[2]
    q = p.model_copy(deep=True);q.region.boundaries.x_min = BoundaryFace(kind='pml', layers=3, alpha=0.)
    plain = BoundaryDescription(Project.model_validate(q.model_dump()).region).cpml[True, 0, 1][0]
    assert plain['b'].shape[0] == 3 and float(plain['inv_k'].max()) == 1.


@torch.enable_grad()
def test_adjoint_gradient_taylor_check_float64():
    p = gradient_scene('float64')
    eps = random_epsilon(p.region, 3).requires_grad_()
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
    result = model(eps)
    gradient, = torch.autograd.grad(objective(result.signals), eps)
    reference, = torch.autograd.grad(objective(model.reference(eps)), eps)
    torch.testing.assert_close(gradient, reference, atol=1e-12, rtol=1e-9)
    direction = torch.randn(eps.shape, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    directional = float((gradient*direction).sum())
    with torch.no_grad():
        def value(h):return float(objective(model(eps+h*direction).signals))
        errors = []
        for h in (4e-3, 2e-3, 1e-3):
            errors.append(abs((value(h)-value(-h))/(2*h)-directional))
    assert abs(directional) > 1e-6
    assert errors[-1] < 1e-6*abs(directional)
    # Central differences converge at second order until round-off.
    assert errors[0] > 3*errors[1] > 9*errors[2]


@CUDA
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@torch.enable_grad()
def test_cuda_adjoint_matches_autograd_and_central_differences(precision):
    p = gradient_scene(precision)
    eps = random_epsilon(p.region, 5, device='cuda').requires_grad_()
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
    result = model(eps)
    assert result.report['forward_backend'] == 'fused CUDA' and result.report['backward_backend'] == 'torch explicit transpose'
    gradient, = torch.autograd.grad(objective(result.signals), eps)
    reference, = torch.autograd.grad(objective(model.reference(eps)), eps)
    tolerance = dict(atol=1e-12, rtol=1e-9) if precision == 'float64' else dict(atol=3e-6*float(reference.abs().max()), rtol=4e-5)
    torch.testing.assert_close(gradient, reference, **tolerance)
    direction = torch.randn(eps.shape, generator=torch.Generator().manual_seed(13)).to(eps)
    directional = float((gradient*direction).sum())
    h = 5e-4 if precision == 'float64' else 2e-2
    with torch.no_grad():
        central = (float(objective(model(eps+h*direction).signals))-float(objective(model(eps-h*direction).signals)))/(2*h)
    assert abs(central-directional) < (2e-6 if precision == 'float64' else 2e-2)*abs(directional)


@pytest.mark.parametrize('storage', ['host', 'disk'])
@torch.enable_grad()
def test_checkpoint_tiers_carry_face_state(storage, tmp_path):
    p = gradient_scene('float32')
    eps = random_epsilon(p.region, 9).requires_grad_()
    reference = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
    expected, = torch.autograd.grad(objective(reference(eps).signals), eps)
    options = AdjointOptions(checkpoints=2, storage=storage, checkpoint_directory=tmp_path if storage == 'disk' else None,
                             disk_budget_bytes=64*2**20 if storage == 'disk' else None)
    result = DifferentiableSimulation(p, options)(eps)
    actual, = torch.autograd.grad(objective(result.signals), eps)
    system = _System(p, eps.detach())
    assert result.report['restart_bytes'] == sum(s.numel()*s.element_size() for s in system.state())
    assert len(system.state()) == 2+len(system.segments)+sum(len(b) for b in system.grid.pmc_blocks.values())
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


# ---------------------------------------------------------------------------
# Streamed X-slab execution
# ---------------------------------------------------------------------------
def streamed_scene(precision='float32'):
    kinds = (('pml', 'pmc'), ('pec', 'pmc'), ('pmc', 'pec'))
    p = scene(kinds, size=(1.0, .6, .6), steps=17, precision=precision, profiles={'x_min': dict(kappa=2., alpha=.01)})
    r = p.region;n = r.shape
    p.sources = [Source(center=at(r, 'Ez', (n[0], n[1], 2)), component='Ez', pulse='continuous', wavelength=.5),   # upper x/y edge
                 Source(center=at(r, 'Ey', (3, 2, 1)), component='Ey', pulse='continuous', wavelength=.5, amplitude=.6)]
    p.monitors = [Monitor(center=at(r, 'Ez', (n[0], n[1], 3)), component='Ez'),   # upper x/y edge
                  Monitor(center=at(r, 'Ex', (4, n[1], 2)), component='Ex'),      # upper y face
                  Monitor(center=at(r, 'Hx', (n[0], 3, 2)), component='Hx'),      # upper x face, normal H
                  Monitor(center=at(r, 'Hy', (4, n[1], 1)), component='Hy'),      # upper y face, normal H
                  Monitor(center=at(r, 'Ez', (7, 1, 0)), component='Ez'),         # lower z PMC wall
                  Monitor(center=at(r, 'Ey', (n[0]-1, 4, 1)), component='Ey')]
    return Project.model_validate(p.model_dump())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('storage', ['host', 'disk'])
@torch.enable_grad()
def test_streamed_matches_resident_with_pmc_faces(device, storage, tmp_path):
    from torchfdtd import StreamedSimulation, StreamedAdjointOptions
    p = streamed_scene()
    eps = random_epsilon(p.region, 21).requires_grad_()
    resident = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
    reference = resident.reference(eps)
    expected, = torch.autograd.grad(objective(reference), eps)
    # Width three with depth three: the penultimate tile reaches the last row
    # as halo without owning it, and the last tile owns the x-upper faces.
    options = StreamedAdjointOptions(device=device, slab_width=3, temporal_depth=3, checkpoints=2, local_checkpoints=1,
                                     state_storage=storage, state_directory=tmp_path/'banks' if storage == 'disk' else None,
                                     disk_budget_bytes=256*2**20 if storage == 'disk' else None,
                                     restart_directory=tmp_path/'journal', restart_every_blocks=2)
    result = StreamedSimulation(p, options)(eps)
    actual, = torch.autograd.grad(objective(result.signals), eps)
    assert reference.abs().max() > 1e-3
    torch.testing.assert_close(result.signals, reference, atol=3e-6, rtol=4e-5)
    torch.testing.assert_close(actual, expected, atol=3e-6*float(expected.abs().max()), rtol=4e-5)
    host = _System(p, eps.detach(), prepare_updates=False)
    exact = sum(s.numel()*s.element_size() for s in host.state())
    assert result.report['state_bytes'] == exact
    faces = sum(f.numel()*f.element_size() for family in ('E', 'H') for f in host.grid.faces[family])
    assert faces > 0 and result.report['state_bytes'] > 6*math.prod(p.region.shape)*4+faces
    assert result.report['journal_records_written'] and result.report['restart_reservation_bytes'] >= 2*exact


@torch.enable_grad()
def test_streamed_cpu_float64_and_wall_source_ownership():
    from torchfdtd import StreamedSimulation, StreamedAdjointOptions
    p = streamed_scene('float64')
    eps = random_epsilon(p.region, 23).requires_grad_()
    resident = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
    expected, = torch.autograd.grad(objective(resident.reference(eps)), eps)
    for width in (2, 4):
        result = StreamedSimulation(p, StreamedAdjointOptions(device='cpu', slab_width=width, temporal_depth=2, checkpoints=3))(eps)
        actual, = torch.autograd.grad(objective(result.signals), eps)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-9)


# ---------------------------------------------------------------------------
# Tensor batch
# ---------------------------------------------------------------------------
@CUDA
def test_tensor_batch_matches_single_and_keeps_stored_faces(tmp_path):
    pytest.importorskip('cupy')
    from torchfdtd.tensor_batch import run_tensor_batch
    from torchfdtd.solver import Result
    first = gradient_scene('float32', backend='cuda');first.region.steps = 24
    second = first.model_copy(deep=True)
    second.materials = [Material(name='glass', index=1.6)]
    second.structures = [Structure(name='block', material='glass', kind='rectangle', center=(.1, .05, .1), size=(.3, .3, .3))]
    projects = [Project.model_validate(q.model_dump()) for q in (first, second)]
    report = run_tensor_batch(projects, output_dir=tmp_path, cuda_graph=True, cuda_graph_steps=2)
    report.raise_for_errors()
    for project, item in zip(projects, report.items):
        epsilon, _ = voxelize(project)
        with torch.no_grad():
            single = DifferentiableSimulation(project)(torch.as_tensor(epsilon, device='cuda')).signals.cpu().numpy()
        assert np.abs(single).max() > 1e-3
        np.testing.assert_allclose(item.result.signals, single, rtol=1e-6, atol=1e-7)
        blocks = item.summary['endpoint_blocks']
        assert sum(math.prod(b['shape']) for b in blocks['E']) == item.result.endpoint_fields['E_upper'].size > 0
        assert item.result.electric.shape == project.region.shape+(3,)
        loaded = Result.load(item.output)
        np.testing.assert_array_equal(loaded.endpoint_fields['H_upper'], item.result.endpoint_fields['H_upper'])
    closed = cavity_scene(backend='cuda')
    endpoint = Simulation(closed).run()
    batch = run_tensor_batch([closed], cuda_graph=False)
    batch.raise_for_errors()
    np.testing.assert_allclose(batch.items[0].result.signals, endpoint.signals, rtol=3e-5, atol=2e-6)


# ---------------------------------------------------------------------------
# CPML absorption next to PMC faces
# ---------------------------------------------------------------------------
def cpml_reflection(profile):
    """Reflection from an x-upper CPML whose four neighbours are PMC faces.

    The reference domain is long enough that its own CPML reflection cannot
    reach the monitors within the run, so the difference is the short domain's
    CPML return. One monitor sits on the y-upper PMC face.
    """
    def run(length):
        kinds = (('pec', 'pml'), ('pmc', 'pmc'), ('pmc', 'pmc'))
        p = scene(kinds, size=(length, .5, .5), steps=100, profiles={'x_max': profile})
        r = p.region
        p.sources = [Source(center=at(r, 'Ez', (3, 2, 2)), component='Ez', pulse='gaussian', wavelength=.8)]
        p.monitors = [Monitor(center=at(r, 'Ez', (8, 2, 2)), component='Ez'), Monitor(center=at(r, 'Ez', (8, r.shape[1], 2)), component='Ez')]
        p = Project.model_validate(p.model_dump())
        with torch.no_grad():
            return DifferentiableSimulation(p)(torch.ones(material_shape(p.region))).signals.numpy()
    short, long = run(2.4), run(6.0)
    incident = np.abs(long).max(axis=0)
    assert incident.min() > 5e-3
    return (np.abs(short-long).max(axis=0)/incident).max()


def test_cpml_reflection_next_to_pmc_faces():
    strong = cpml_reflection(dict(layers=8, alpha=0.))
    graded = cpml_reflection(dict(layers=8, kappa=3., alpha=.05, polynomial=2., alpha_polynomial=1.))
    weak = cpml_reflection(dict(layers=3, sigma_scale=.1, alpha=0.))
    assert 0 < strong < 1e-3 and 0 < graded < 1e-3
    assert weak > 3*strong   # the measurement responds to the per-face profile


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------
def test_rejected_combinations_raise_explicitly():
    p = gradient_scene('float32')
    eps = random_epsilon(p.region, 1)
    with pytest.raises(ValueError, match='stored row'):
        DifferentiableSimulation(p)(torch.ones(p.region.shape+(3,)))
    # The PMC rejection precedes the CUDA-tensor requirement, so it holds without a GPU.
    with pytest.raises(ValueError, match='fused CUDA backward'):
        DifferentiableSimulation(p, AdjointOptions(backward_kernel='fused'))(eps)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match='fused CUDA backward'):
            DifferentiableSimulation(p, AdjointOptions(backward_kernel='fused'))(eps.cuda())
    plane = p.model_copy(deep=True)
    plane.sources = [Source(kind='plane', normal='x', center=(0, 0, .05), size=(0, .6, .5), component='Ey', pulse='gaussian', wavelength=.5)]
    with pytest.raises(ValueError, match='plane sources must end below the wall'):
        DifferentiableSimulation(Project.model_validate(plane.model_dump()))(eps)
    grid = YeeGrid(p.region)
    with pytest.raises(ValueError, match='Torch/NumPy grid curl'):grid.curl(grid.H, False)
    payload = p.model_dump()
    payload['region']['boundaries']['y_min'] = {'kind': 'periodic'};payload['region']['boundaries']['y_max'] = {'kind': 'periodic'}
    with pytest.raises(ValueError, match='periodic or Bloch mixing'):Project.model_validate(payload)
    payload = p.model_dump();payload['region']['interface_method'] = 'subpixel'
    with pytest.raises(ValueError, match='staircase'):Project.model_validate(payload)

    class _Special(_System):pass
    with pytest.raises(ValueError, match='not implemented by _Special'):_Special(p, eps)
    from torchfdtd.reversible import ReversibleSimulation
    from torchfdtd.dispersive_adjoint import DispersiveSimulation
    from torchfdtd.adjoint_planes import DifferentiablePlaneSimulation
    from torchfdtd.source_adjoint import SourceWaveformSimulation
    from torchfdtd.streamed_geometry import StreamedGeometrySimulation
    from torchfdtd.adjoint_batch import RecomputedAdjointBatch, AdjointCase
    for factory in (ReversibleSimulation, DifferentiablePlaneSimulation, SourceWaveformSimulation, StreamedGeometrySimulation):
        with pytest.raises(ValueError, match='PMC/symmetric faces are not implemented by'):factory(p)
    assert DispersiveSimulation(p).project.region.boundaries.x_max.kind == 'pmc'   # face ADE banks: tests/test_pmc_dispersive.py
    from torchfdtd.execution_tuning import AdjointExecutionPolicy
    with pytest.raises(ValueError, match='PMC/symmetric faces are not implemented by'):
        RecomputedAdjointBatch([AdjointCase(p, AdjointExecutionPolicy(resident=AdjointOptions(), device='cpu'))])
    with pytest.raises(ValueError):Simulation(p).run()   # per-face profile outside the endpoint forward contract


@CUDA
def test_plane_source_on_a_stored_face_is_refused_before_any_field_allocation():
    from torchfdtd import StreamedSimulation, StreamedAdjointOptions
    p = gradient_scene('float32')
    plane = p.model_copy(deep=True)
    plane.sources = [Source(kind='plane', normal='x', center=(0, 0, .05), size=(0, .6, .5), component='Ey', pulse='gaussian', wavelength=.5)]
    plane = Project.model_validate(plane.model_dump())
    eps = random_epsilon(p.region, 1, device='cuda')
    torch.cuda.synchronize();allocated = torch.cuda.memory_allocated()
    for factory in (lambda: DifferentiableSimulation(plane)(eps), lambda: StreamedSimulation(plane, StreamedAdjointOptions(device='cuda', slab_width=4))(eps.cpu())):
        with pytest.raises(ValueError, match='plane sources must end below the wall'):factory()
        assert torch.cuda.memory_allocated() == allocated


@CUDA
def test_tensor_batch_rejects_unsupported_pmc_features():
    from torchfdtd.tensor_batch import run_tensor_batch
    base = gradient_scene('float32', backend='cuda')
    from torchfdtd.models import FieldMonitor, SpectrumSettings
    field = base.model_copy(deep=True)
    field.monitors.append(FieldMonitor(center=(0, 0, 0), size=(0, .4, .4),
                                       spectrum=SpectrumSettings(sampling='frequency', frequency_points=2, apodization='none')))
    with pytest.raises(ValueError, match='field monitors'):run_tensor_batch([Project.model_validate(field.model_dump())])
    oneway = base.model_copy(deep=True)
    oneway.sources = [Source(kind='plane', injection='oneway', normal='x', center=(-.05, 0, 0), size=(0, .2, .2), component='Ez', pulse='gaussian', wavelength=.5)]
    with pytest.raises(ValueError):run_tensor_batch([Project.model_validate(oneway.model_dump())])
    plane = base.model_copy(deep=True)
    plane.sources = [Source(kind='plane', normal='x', center=(0, 0, .05), size=(0, .6, .5), component='Ey', pulse='gaussian', wavelength=.5)]
    torch.cuda.synchronize();allocated = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match='plane sources must end below the wall'):run_tensor_batch([Project.model_validate(plane.model_dump())])
    assert torch.cuda.memory_allocated() == allocated   # refused before the cohort grids exist
