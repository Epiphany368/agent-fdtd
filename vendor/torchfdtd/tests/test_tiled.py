"""Overlapping-tile plans, stitching, angular-spectrum propagation and the tiled adjoint."""
import math
import warnings

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiablePlaneSimulation, FieldMonitor, Project, Region, Simulation, Source,
                       Structure, StitchedPlane, TiledPlaneSimulation, farfield_from_stitched, plan_tiles, propagate_plane,
                       run_tiled, stitch_planes, suggest_overlap)
from torchfdtd.adjoint_planes import COMPONENTS
from torchfdtd.field_monitors import plane_plan
from torchfdtd.run_control import source_end_time
from torchfdtd.solver import voxelize

C0 = 299792458.


def spectrum(wavelength_um):
    return dict(sampling='custom', custom_frequencies_hz=[C0 / (wavelength_um * 1e-6)], apodization='none')


def pillar_row(*, mesh=.05, size=(7., 3., 1.), period=.5, count=12, half=(.08, .22), wavelength=1.55, through_pml=True,
               precision='float32', extra_fs=40., seed=3, component='Ex', source_y=-.85, height=.6, plane_y=.5):
    """2D row of index-2 rectangles lit along +y with the output plane above them."""
    pml = round(.5 / mesh)
    region = Region(dimension='2d', size=size, mesh=mesh, pml_cells=pml, steps=100, precision=precision, backend='cpu')
    sheet = size[0] if through_pml else size[0] - 2 * pml * mesh
    source = Source(kind='plane', normal='y', center=(0., source_y, 0.), size=(sheet, 0., 0.), component=component,
                    wavelength=wavelength, extend_through_pml=through_pml)
    monitor = FieldMonitor(id='out', normal='y', center=(0., plane_y, 0.), size=(size[0] - 2 * pml * mesh, 0., 1.), spectrum=spectrum(wavelength))
    rng = np.random.default_rng(seed)
    structures = [Structure(kind='rectangle', center=(-(count - 1) * period / 2 + period * i, -height / 2, 0.),
                            size=(2 * float(rng.uniform(*half)), height, 1.), material='SiN (constant n)') for i in range(count)]
    project = Project(region=region, structures=structures, sources=[source], monitors=[monitor])
    project.region.steps = math.ceil((source_end_time(project) + extra_fs * 1e-15) / project.region.time_step)
    return Project.model_validate(project.model_dump())


def small_row(**overrides):
    """A 4 x 2.2 um row at 0.1 um mesh and 1 um wavelength for cheap CPU checks."""
    return pillar_row(**{**dict(mesh=.1, size=(4., 2.2, 1.), wavelength=1., source_y=-.5, height=.4, plane_y=.3), **overrides})


def pillar_array(*, mesh=.1, size=(3.2, 3.2, 2.), wavelength=1.3, backend='cpu', precision='float32', extra_fs=20.):
    """Small 3D array of index-2 cylinders lit along +z."""
    pml = round(.4 / mesh)
    region = Region(dimension='3d', size=size, mesh=mesh, pml_cells=pml, steps=100, precision=precision, backend=backend,
                    cuda_kernel='fused' if backend == 'cuda' else 'torch')
    source = Source(kind='plane', normal='z', center=(0., 0., -.5), size=(size[0], size[1], 0.), component='Ex',
                    wavelength=wavelength, pulse_cycles=2, extend_through_pml=True)
    inner = tuple(s - 2 * pml * mesh for s in size[:2])
    monitor = FieldMonitor(id='out', normal='z', center=(0., 0., .35), size=(inner[0], inner[1], 0.), spectrum=spectrum(wavelength))
    structures = [Structure(kind='circle', center=(x, y, -.1), radius=.14 + .04 * ((i + j) % 3), size=(1., 1., .4), material='SiN (constant n)')
                  for i, x in enumerate((-.9, -.3, .3, .9)) for j, y in enumerate((-.9, -.3, .3, .9))]
    project = Project(region=region, structures=structures, sources=[source], monitors=[monitor])
    project.region.steps = math.ceil((source_end_time(project) + extra_fs * 1e-15) / project.region.time_step)
    return Project.model_validate(project.model_dump())


def reference_plane(project, plan):
    """The whole-device plane on the stitched grid through a one-tile plan."""
    whole = plan_tiles(project, project.region.size[plan.lateral[0]], plan.overlap_um)
    assert len(whole.tiles) == 1
    return stitch_planes(whole, [Simulation(project).run().field_monitor('out')])


def gpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


def test_plan_geometry_node_alignment_edge_clipping_and_crossing_structures():
    p = pillar_array()
    plan = plan_tiles(p, 1.0, .25)
    r = p.region
    h = r.mesh
    assert plan.counts == (3, 3) and plan.tile_cells == 10 and plan.overlap_cells == 3 and plan.lateral == (0, 1)
    interior = tuple((r.pml_layers(a, 0), r.shape[a] - r.pml_layers(a, 1)) for a in (0, 1))
    assert plan.interior == interior
    # Cores partition the interior; extended and full ranges are clipped at the device edge.
    for i, (lo, hi) in enumerate(interior):
        cores = sorted({t.core[i] for t in plan.tiles})
        assert cores[0][0] == lo and cores[-1][1] == hi and all(a[1] == b[0] for a, b in zip(cores, cores[1:]))
        assert cores[-1][1] - cores[-1][0] == (hi - lo) - 2 * plan.tile_cells  # last core takes the remainder
    for tile in plan.tiles:
        q = tile.project
        for i, a in enumerate(plan.lateral):
            c0, c1 = tile.core[i]
            lo, hi = interior[i]
            assert tile.extended[i] == (max(lo, c0 - 3), min(hi, c1 + 3))
            assert tile.full[i] == (tile.extended[i][0] - r.pml_layers(a, 0), tile.extended[i][1] + r.pml_layers(a, 1))
            f0, f1 = tile.full[i]
            assert q.region.shape[a] == f1 - f0
            # Every tile node coincides with a global node.
            np.testing.assert_allclose(q.region.mesh_nodes[a] + tile.center_um[a], r.mesh_nodes[a][f0:f1 + 1], rtol=0, atol=1e-9)
            # The monitor spans the extended region at global cell centres and the sheet the whole tile grid.
            points = plane_plan(q.region, q.resolved_monitor(q.monitors[0]))['points_um'][:, a] + tile.center_um[a]
            e0, e1 = tile.extended[i]
            np.testing.assert_allclose(np.unique(points), (r.mesh_nodes[a][e0:e1] + r.mesh_nodes[a][e0 + 1:e1 + 1]) / 2, atol=1e-9)
            assert q.sources[0].size[a] == pytest.approx((f1 - f0) * h) and q.sources[0].extend_through_pml
        assert q.region.shape[2] == r.shape[2] and q.region.memory_mode == 'resident'
        assert all(face.kind == 'pml' for axis in range(3) for face in q.region.boundaries.pair(axis))
        # Kept structures are whole and shifted into the tile frame; far ones are dropped.
        for s in q.structures:
            original = next(o for o in p.structures if o.id == s.id)
            assert s.radius == original.radius
            np.testing.assert_allclose(np.add(s.center, tile.center_um), original.center, atol=1e-12)
            assert all(original.center[a] + original.radius > r.mesh_nodes[a][tile.full[i][0]] - h and
                       original.center[a] - original.radius < r.mesh_nodes[a][tile.full[i][1]] + h for i, a in enumerate(plan.lateral))
    # The pillar at (-0.3, -0.3) crosses the cut between the first two tiles along x and is present in both.
    crossing = [s.id for s in p.structures if s.center[:2] == (-.3, -.3)][0]
    owners = [t.id for t in plan.tiles if any(s.id == crossing for s in t.project.structures)]
    assert {'tile-0-0', 'tile-1-0'} <= set(owners) and 'tile-2-2' not in owners
    assert plan.report['largest_tile_cells'] < plan.report['global_cells'] < plan.report['total_tile_cells']
    # A sheet inside the non-PML region is restricted to the extended tile region.
    truncated = Project.model_validate({**p.model_dump(), 'sources': [{**p.sources[0].model_dump(), 'size': (2.4, 2.4, 0.), 'extend_through_pml': False}]})
    tile = plan_tiles(truncated, 1.0, .25).tiles[4]
    assert tile.project.sources[0].size[0] == pytest.approx((tile.extended[0][1] - tile.extended[0][0]) * h)
    # 2D: one tiled axis, z invariant.
    q = small_row(count=6)
    flat = plan_tiles(q, 1.6, .3)
    assert flat.lateral == (0,) and flat.counts == (2,) and flat.normal == 'y' and flat.direction == 1
    assert all(t.project.region.dimension == '2d' and t.project.region.shape[1:] == q.region.shape[1:] for t in flat.tiles)


def test_plan_rejections_and_overlap_suggestion():
    p = pillar_array()
    with pytest.raises(ValueError, match='uniform mesh'):
        plan_tiles(Project.model_validate({**p.model_dump(), 'region': {**p.region.model_dump(), 'mesh_type': 'graded', 'material_sampling': 'yee'}}), 1., .2)
    with pytest.raises(ValueError, match='CPML'):
        plan_tiles(Project.model_validate({**p.model_dump(), 'region': {**p.region.model_dump(), 'boundaries': {'x_min': {'kind': 'periodic'}, 'x_max': {'kind': 'periodic'}}}}), 1., .2)
    with pytest.raises(ValueError, match='soft plane sources'):
        plan_tiles(Project.model_validate({**p.model_dump(), 'sources': [dict(center=(0, 0, -.5), wavelength=1.3)]}), 1., .2)
    with pytest.raises(ValueError, match='whole non-PML lateral extent'):
        plan_tiles(Project.model_validate({**p.model_dump(), 'sources': [{**p.sources[0].model_dump(), 'size': (1., 1., 0.), 'extend_through_pml': False}]}), 1., .2)
    with pytest.raises(ValueError, match='exactly one enabled field monitor'):
        plan_tiles(Project.model_validate({**p.model_dump(), 'monitors': []}), 1., .2)
    with pytest.raises(ValueError, match='positive finite'):
        plan_tiles(p, 0., .2)
    with pytest.raises(ValueError, match='invariant axis'):
        plan_tiles(small_row(count=6), 1., .2, normal='z')
    assert suggest_overlap(2., 45., 1.) == pytest.approx(dict(overlap_um=2.5, spread_um=2., absorber_um=.5, distance_um=2., max_angle_deg=45., wavelength_um=1.))
    assert suggest_overlap(0., 60., 1.5, absorber_um=.4)['overlap_um'] == pytest.approx(.4)
    for bad in ((-1., 30., 1.), (1., 90., 1.), (1., 30., 0.)):
        with pytest.raises(ValueError):
            suggest_overlap(*bad)
    # Pillars reach z = -0.3 and the plane is at 0.35: 0.65 um at 45 degrees plus the 0.4 um absorber.
    with pytest.warns(UserWarning, match='below the suggested'):
        plan = plan_tiles(p, 1., .25, max_angle_deg=45.)
    assert plan.suggestion['overlap_um'] == pytest.approx(.65 + .4) and plan.suggestion['distance_um'] == pytest.approx(.65)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert plan_tiles(p, 1., 1.1, max_angle_deg=45.).suggestion['overlap_um'] == pytest.approx(1.05)


def synthetic(plan, grid, wavelength=1.3):
    """A smooth complex field on the global lateral grid and its per-tile extended cuts."""
    r = plan.project.region
    axes = [(r.mesh_nodes[a][lo:hi] + r.mesh_nodes[a][lo + 1:hi + 1]) / 2 for a, (lo, hi) in zip(plan.lateral, plan.interior)]
    if len(axes) == 1:
        axes.append(np.zeros(1))
    u, v = np.meshgrid(*axes, indexing='ij')
    k = 2 * math.pi / wavelength
    base = np.exp(1j * k * (.3 * u + .2 * v)) * np.exp(-(u**2 + v**2) / 2.)
    fields = torch.as_tensor(np.stack([base * (c + 1) for c in range(6)], -1)[None], dtype=torch.complex128)
    origin = [lo for lo, _ in plan.interior]
    cuts = []
    for tile in plan.tiles:
        window = tuple(slice(e0 - o, e1 - o) for (e0, e1), o in zip(tile.extended, origin))
        if len(window) == 1:
            window += (slice(None),)
        cuts.append(fields[(slice(None),) + window].clone())
    frequency = torch.tensor([C0 / (wavelength * 1e-6)], dtype=torch.float64)
    return fields, cuts, frequency


@pytest.mark.parametrize('dimension', ['2d', '3d'])
def test_stitching_is_exact_for_cuts_of_one_field(dimension):
    p = pillar_array() if dimension == '3d' else small_row(count=6)
    plan = plan_tiles(p, 1., .3)
    fields, cuts, frequency = synthetic(plan, None)
    planes = [dict(fields=cut.reshape(1, -1, 6).numpy(), shape=tuple(cut.shape[1:3]) if dimension == '3d' else (cut.shape[1], 1, 1),
                   components=list(COMPONENTS), frequency_hz=frequency.numpy()) for cut in cuts]
    if dimension == '3d':
        for plane, cut in zip(planes, cuts):
            plane['shape'] = (cut.shape[1], cut.shape[2], 1)
    for blend in ('hard', 'linear'):
        stitched = stitch_planes(plan, planes, blend=blend)
        torch.testing.assert_close(stitched.fields, fields, rtol=0, atol=1e-13)
        assert stitched.report['max_mismatch'] == 0 and stitched.report['max_mismatch_center'] == 0 and stitched.blend == blend
        assert len(stitched.report['pairs']) == (12 if dimension == '3d' else 2)
    assert stitched.fields.shape[1:3] == ((24, 24) if dimension == '3d' else (30, 1))
    # Tensor input with the frequencies missing is rejected; a wrong shape names the tile.
    with pytest.raises(ValueError, match='no frequencies'):
        stitch_planes(plan, cuts)
    with pytest.raises(ValueError, match='tile-0'):
        stitch_planes(plan, [dict(planes[0], fields=planes[0]['fields'][:, :-1])] + planes[1:])
    # Zero-distance propagation is the identity and the plane view has full midpoint quadrature.
    same = propagate_plane(stitched, 0., 1., pad=1)
    torch.testing.assert_close(same.fields, fields, rtol=0, atol=1e-13)
    plane = stitched.as_plane()
    assert plane.fields.shape == (1, fields.shape[1] * fields.shape[2], 6) and plane.normal == plan.normal
    assert float(plane.weights.sum()) == pytest.approx((.1e-6)**2 * 24 * 24 if dimension == '3d' else .1e-6 * 30)
    if dimension == '3d':
        # The open-surface radiation adapter accepts the plane, flags the approximation and keeps the graph.
        live = StitchedPlane(**{**vars(stitched), 'fields': stitched.fields.clone().requires_grad_()})
        far = farfield_from_stitched(live, [[0., 0., 1.], [math.sin(.3), 0., math.cos(.3)]])
        assert far.approximation == 'open surface' and far.surfaces == ('z_max',)
        faces, bounds = live.radiation_surface()
        assert bounds[2] == (-1., .35) and bounds[0][0] == pytest.approx(-1.2)
        gradient, = torch.autograd.grad(far.intensity().sum(), live.fields)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    else:
        with pytest.raises(ValueError, match='3D'):
            stitched.radiation_surface()


def gaussian_plane(w0, wavelength, spacing, count, *, two_dimensional=False):
    u = (torch.arange(count, dtype=torch.float64) - (count - 1) / 2) * spacing
    v = torch.zeros(1, dtype=torch.float64) if two_dimensional else u.clone()
    ug, vg = torch.meshgrid(u, v, indexing='ij')
    waist = torch.exp(-(ug**2 + vg**2) / w0**2).to(torch.complex128)
    fields = torch.zeros((1, count, len(v), 6), dtype=torch.complex128)
    fields[0, :, :, 0] = waist
    return StitchedPlane(fields, torch.tensor([C0 / (wavelength * 1e-6)], dtype=torch.float64), u, v, 'z', 0., 1,
                         (spacing, spacing if not two_dimensional else 1.), '2d' if two_dimensional else '3d', (-1., 0.), 'synthetic', {}), ug, vg


def paraxial_gaussian(w0, wavelength, distance, ug, vg, *, two_dimensional):
    k = 2 * math.pi / wavelength
    rayleigh = math.pi * w0**2 / wavelength
    width = w0 * math.sqrt(1 + (distance / rayleigh)**2)
    gouy = math.atan(distance / rayleigh)
    radius2 = ug**2 + (0 if two_dimensional else vg**2)
    curvature = distance / (distance**2 + rayleigh**2)
    amplitude = (w0 / width) ** (.5 if two_dimensional else 1.)
    return amplitude * torch.exp(-radius2 / width**2) * torch.exp(1j * (k * distance + k * radius2 * curvature / 2 - gouy * (.5 if two_dimensional else 1.)))


@pytest.mark.parametrize('two_dimensional', [False, True])
def test_angular_spectrum_matches_tilted_plane_wave_and_paraxial_gaussian(two_dimensional):
    wavelength, spacing, count = 1., .25, 192
    # A plane wave commensurate with the unpadded grid propagates with exactly exp(i k_n d).
    plane, ug, vg = gaussian_plane(1e9, wavelength, spacing, count, two_dimensional=two_dimensional)
    ku = 2 * math.pi * 3 / (count * spacing)
    kv = 0. if two_dimensional else 2 * math.pi * 2 / (count * spacing)
    k = 2 * math.pi / wavelength
    wave = torch.exp(1j * (ku * ug + kv * vg))
    plane.fields[0, :, :, 0] = wave
    plane.fields[0, :, :, 4] = 2 * wave
    distance = 7.3
    out = propagate_plane(plane, distance, 1., pad=1)
    expected = wave * complex(math.cos(math.sqrt(k**2 - ku**2 - kv**2) * distance), math.sin(math.sqrt(k**2 - ku**2 - kv**2) * distance))
    torch.testing.assert_close(out.fields[0, :, :, 0], expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(out.fields[0, :, :, 4], 2 * expected, rtol=1e-9, atol=1e-9)
    assert out.offset_um == pytest.approx(distance) and out.report['propagation']['distance_um'] == distance
    with pytest.raises(ValueError, match='nonnegative'):
        propagate_plane(plane, -1., 1.)
    # A Gaussian beam over one Rayleigh range against the paraxial solution. The
    # transform is exact, so the difference is the paraxial error, which scales
    # as (wavelength / (pi w0))^2: it falls fourfold when the waist doubles.
    errors = []
    for w0 in (2., 4., 8.):
        plane, ug, vg = gaussian_plane(w0, wavelength, spacing, count, two_dimensional=two_dimensional)
        rayleigh = math.pi * w0**2 / wavelength
        out = propagate_plane(plane, rayleigh, 1., pad=2)
        expected = paraxial_gaussian(w0, wavelength, rayleigh, ug, vg, two_dimensional=two_dimensional)
        errors.append(float((out.fields[0, :, :, 0] - expected).norm() / expected.norm()))
    assert errors[0] < 1e-2 and all(a / b > 3.5 for a, b in zip(errors, errors[1:]))
    # Evanescent components decay instead of growing and the transform keeps the graph.
    live = StitchedPlane(**{**vars(plane), 'fields': plane.fields.clone().requires_grad_()})
    far = propagate_plane(live, 30., 1.5, pad=2)
    assert float(far.fields.detach().abs().max()) <= float(plane.fields.abs().max()) * 1.01
    gradient, = torch.autograd.grad(far.intensity().sum(), live.fields)
    assert torch.isfinite(gradient).all()


def test_two_dimensional_device_error_and_indicator_decrease_with_overlap():
    p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
    plan = plan_tiles(p, 3., .25)
    reference = reference_plane(p, plan)
    errors, indicators, focal = [], [], []
    target = propagate_plane(reference, 10., 1., pad=4).intensity()
    for overlap in (.25, .5, 1., 2.):
        plan = plan_tiles(p, 3., overlap)
        stitched = run_tiled(p, plan, backend='cpu')
        assert len(plan.tiles) == 2 and stitched.fields.shape == reference.fields.shape
        errors.append(float((stitched.fields - reference.fields).norm() / reference.fields.norm()))
        indicators.append(stitched.report['max_mismatch_center'])
        focal.append(float((propagate_plane(stitched, 10., 1., pad=4).intensity() - target).norm() / target.norm()))
        linear = stitch_planes(plan, stitched.tile_planes, blend='linear')
        assert float((linear.fields - reference.fields).norm() / reference.fields.norm()) < 1.5 * errors[-1] + 1e-9
    assert errors[0] > .1 and all(a > b for a, b in zip(errors, errors[1:])) and errors[-1] < 1e-4
    assert all(a > b for a, b in zip(indicators, indicators[1:])) and indicators[-1] < 1e-4
    assert focal[0] > focal[-1] and focal[-1] < 1e-4
    # The empty device stitches exactly with the sheet through the absorber.
    empty = Project.model_validate({**p.model_dump(), 'structures': []})
    stitched = run_tiled(empty, plan_tiles(empty, 3., .5), backend='cpu')
    assert float((stitched.fields - reference_plane(empty, plan).fields).norm() / stitched.fields.norm()) < 1e-6
    assert stitched.report['max_mismatch'] < 1e-6


def test_run_tiled_rejections_and_report():
    p = small_row(count=6)
    plan = plan_tiles(p, 1.6, .3)
    other = small_row(count=5)
    with pytest.raises(ValueError, match='different project'):
        run_tiled(other, plan, backend='cpu')
    with pytest.raises(ValueError, match='executor'):
        run_tiled(p, plan, backend='cpu', executor='threads')
    stitched = run_tiled(p, plan, backend='cpu', blend='linear')
    assert stitched.report['executor'] == 'sequential' and len(stitched.report['tiles_run']) == 2 and stitched.blend == 'linear'
    assert stitched.report['tiles_run'][0]['steps'] == p.region.steps and stitched.report['seconds'] > 0


def test_batch_executor_matches_sequential():
    p = small_row(count=6, precision='float64')
    plan = plan_tiles(p, 1.6, .3)
    sequential = run_tiled(p, plan, backend='cpu')
    batched = run_tiled(p, plan, backend='cpu', executor='batch', options=dict(max_workers=2))
    torch.testing.assert_close(batched.fields, sequential.fields, rtol=1e-12, atol=1e-30)
    assert batched.report['execution']['method'] == 'spawned independent simulation processes'


def test_tiled_adjoint_matches_central_differences_and_full_device_gradient():
    p = small_row(count=3, half=(.1, .15), precision='float64', extra_fs=15., component='Ez')
    frequency = [C0 / 1e-6]
    epsilon = torch.as_tensor(voxelize(p)[0], dtype=torch.float64)
    full = DifferentiablePlaneSimulation(p, AdjointOptions(checkpoints=3))
    whole = plan_tiles(p, 4., .4)

    def objective(stitched):
        intensity = propagate_plane(stitched, 3., 1., pad=2).intensity()[0, :, 0]
        n = len(intensity) // 2
        return intensity[n - 6:n + 6].sum() / 1e-28

    design = epsilon.clone().requires_grad_()
    objective(stitch_planes(whole, [full(design, frequency)['out']])).backward()
    full_gradient = design.grad.clone()
    with torch.no_grad():
        reference = stitch_planes(whole, [full(epsilon, frequency)['out']])
    gradient_errors, field_errors = [], []
    for overlap in (.4, 1.2):
        plan = plan_tiles(p, 1.6, overlap)
        tiled = TiledPlaneSimulation(p, plan, AdjointOptions(checkpoints=3))
        design = epsilon.clone().requires_grad_()
        stitched = tiled(design, frequency)
        assert stitched.fields.requires_grad
        objective(stitched).backward()
        gradient = design.grad.clone()
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        field_errors.append(float((stitched.fields.detach() - reference.fields).norm() / reference.fields.norm()))
        # Compare on the design cells: inside a tile's own absorber the tile gradient has no meaning.
        design_cells = epsilon > 1.5
        gradient_errors.append(float((gradient[design_cells] - full_gradient[design_cells]).norm() / full_gradient[design_cells].norm()))
        # The pillar cell nearest the cut lies in the overlap band, so both tiles contribute; the leftmost lies in the core of tile-0.
        cut = plan.tiles[0].core[0][1]
        pillar_columns = torch.nonzero(epsilon[:, 9, 0] > 1.5).flatten().tolist()
        nearest = min(pillar_columns, key=lambda i: abs(i - cut))
        core = [i for i in pillar_columns if i < cut]
        assert abs(nearest - cut) < plan.overlap_cells and core
        for cell in ((nearest, 9, 0), (core[0], 9, 0)):
            assert epsilon[cell] > 1.5
            with torch.no_grad():
                plus, minus = epsilon.clone(), epsilon.clone()
                plus[cell] += 1e-3
                minus[cell] -= 1e-3
                difference = float((objective(tiled(plus, frequency)) - objective(tiled(minus, frequency))) / 2e-3)
            assert float(gradient[cell]) == pytest.approx(difference, rel=1e-5, abs=1e-12)
    with torch.no_grad():
        plus, minus = epsilon.clone(), epsilon.clone()
        plus[core[0], 9, 0] += 1e-3
        minus[core[0], 9, 0] -= 1e-3
        difference = float((objective(stitch_planes(whole, [full(plus, frequency)['out']])) -
                            objective(stitch_planes(whole, [full(minus, frequency)['out']]))) / 2e-3)
    assert float(full_gradient[core[0], 9, 0]) == pytest.approx(difference, rel=1e-5, abs=1e-12)
    # With the larger overlap every pillar lies inside both tiles and the tiled model approaches the full one.
    assert field_errors[1] < field_errors[0] and field_errors[1] < 1e-2
    assert gradient_errors[0] > 5e-2 and gradient_errors[1] < 1e-2


def test_tiled_simulation_rejections():
    p = small_row(count=3, precision='float64')
    plan = plan_tiles(p, 1.6, .4)
    model = TiledPlaneSimulation(p, plan)
    with pytest.raises(ValueError, match='global grid'):
        model(torch.ones((10, 10, 1), dtype=torch.float64), [C0 / 1e-6])
    with pytest.raises(ValueError, match='different project'):
        TiledPlaneSimulation(small_row(count=4), plan)


def test_cuda_executors_and_differentiable_forward_agree():
    gpu()
    p = pillar_array(backend='cuda')
    plan = plan_tiles(p, 1.2, .3)
    assert len(plan.tiles) == 4 and all(math.prod(t.project.region.shape) <= 96 * 96 * 48 for t in plan.tiles)
    sequential = run_tiled(p, plan, backend='cuda')
    tensor = run_tiled(p, plan, backend='cuda', executor='tensor', options=dict(cohort_size=4))
    torch.testing.assert_close(tensor.fields, sequential.fields, rtol=1e-4, atol=0)
    assert tensor.report['execution']['group_count'] == 1 and tensor.report['execution']['total_cases'] == 4
    reference = reference_plane(p, plan)
    error = float((sequential.fields - reference.fields).norm() / reference.fields.norm())
    assert error < .3 and sequential.report['max_mismatch'] > 0
    epsilon = torch.as_tensor(voxelize(p)[0], dtype=torch.float32, device='cuda').requires_grad_()
    model = TiledPlaneSimulation(p, plan, AdjointOptions(checkpoints=2))
    stitched = model(epsilon, [C0 / 1.3e-6])
    torch.testing.assert_close(stitched.fields.detach().cpu().to(torch.complex128), sequential.fields, rtol=2e-3, atol=1e-6 * float(sequential.fields.abs().max()))
    propagate_plane(stitched, 5., 1., pad=2).intensity().sum().backward()
    assert torch.isfinite(epsilon.grad).all() and epsilon.grad.abs().sum() > 0
