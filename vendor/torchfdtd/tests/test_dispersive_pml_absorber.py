"""Adiabatic absorber for dispersive media crossing the PML (Region.pml_dispersion='absorber').

The acceptance record (docs/validation/dispersive_pml_absorber.json, written by
benchmarks/dispersive_pml_absorber.py for docs/validation/cases/DISPERSIVE_PML_ABSORBER.json)
is judged again from its own numbers here. The other tests check the absorber
coefficients, the coupled trapezoidal ADE update, the refusals of the paths that
do not implement it and a short run of the diverging fixture.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import fdtd
import numpy as np
import pytest
import torch

from benchmarks import dispersive_pml_absorber as bench
from torchfdtd import Project, Region, Structure, Source, Monitor, Material, Simulation
from torchfdtd.models import Boundaries, BoundaryFace, FieldMonitor
from torchfdtd.boundaries import BoundaryDescription, YeeGrid, absorber_faces, absorber_loss, absorber_profiles
from torchfdtd.materials import configure_materials
from torchfdtd.plan import PlanInvalidated, resolve_plan
from torchfdtd.solver import estimate, run_signature, voxelize
from torchfdtd.stability_checks import stability_warnings

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT/'docs'/'validation'/'cases'/'DISPERSIVE_PML_ABSORBER.json'
RECORD_PATH = ROOT/'docs'/'validation'/'dispersive_pml_absorber.json'
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
SIN = Material(name='sin', model='lorentz', epsilon_inf=1., resonance_rad_s=1.4e16, linewidth_rad_s=1e13, delta_epsilon=3.)
DRUDE = Material(name='sin', model='drude', epsilon_inf=1., plasma_rad_s=2e15, collision_rad_s=1e14)
# Non-unit eps_inf, a Drude pole whose Re eps at the source centre is negative (eps_ref falls back to eps_inf) and two poles.
LORENTZ_225 = Material(name='sin', model='lorentz', epsilon_inf=2.25, resonance_rad_s=1.4e16, linewidth_rad_s=1e13, delta_epsilon=3.)
METAL = Material(name='sin', model='drude', epsilon_inf=2., plasma_rad_s=1.2e16, collision_rad_s=1e14)
MULTIPOLE = Material(name='sin', model='multipole', epsilon_inf=1.5, poles=[
    dict(resonance_rad_s=1.4e16, strength_rad_s_squared=3*1.4e16**2, damping_rad_s=2e13),
    dict(resonance_rad_s=5e15, strength_rad_s_squared=.5*5e15**2, damping_rad_s=1e14)])


@pytest.fixture(autouse=True)
def restore_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def slab_project(mode='absorber', *, dimension='2d', backend='cpu', precision='float64', kernel='torch', steps=40, sampling='yee',
                 everywhere=False, material=SIN):
    """A SiN slab through the y faces and the x_max face into the corners; x faces 5 and 7 layers deep, x_min stays vacuum."""
    three = dimension == '3d'
    faces = dict(x_min=BoundaryFace(layers=5), x_max=BoundaryFace(layers=7, sigma_scale=.8, polynomial=2))
    r = Region(dimension=dimension, size=(2., 1.6, 1.4 if three else 1.), mesh=.1, pml_cells=4, steps=steps, precision=precision,
               backend=backend, cuda_kernel=kernel, material_sampling=sampling, pml_dispersion=mode, boundaries=Boundaries(**faces))
    slab = Structure(name='slab', center=(0 if everywhere else .75, 0, 0), size=(100. if everywhere else .8, 100., 100. if three else 1.),
                     material='sin')
    return Project(region=r, materials=[Material(name='void', index=1), material], structures=[slab],
                   sources=[Source(kind='point', component='Ex', center=(-.3, .1, 0), wavelength=.55, pulse_cycles=2)],
                   monitors=[Monitor(component='Ex', center=(.2, 0, 0))])


def numpy_grid(p):
    fdtd.set_backend('numpy')
    fdtd.backend.float = np.float64
    eps, _, owner = voxelize(p, with_ownership=True)
    g = YeeGrid(p.region, absorber_faces(p))
    g.inverse_permittivity[:] = 1/(eps if eps.ndim == 4 else eps[..., None])
    configure_materials(g, p, owner)
    return g, owner


# ----------------------------------------------------------------------------------------------- the record
@pytest.fixture(scope='module')
def case():
    return json.loads(CASE_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def record():
    return json.loads(RECORD_PATH.read_text(encoding='utf-8'))


def test_record_covers_the_declared_case(case, record):
    assert record['case']['case_id'] == case['case_id'] and record['case']['declared_at_commit'] == case['declared_at_commit']
    assert record['acceptance'] == case['acceptance']
    assert {row['id'] for row in record['rows']} == {spec['id'] for spec in bench.rows()}
    # Every run of the record was clean, at one commit, and ran the sources hashed in it. The hashes are those of the record's
    # commit, not checked against this tree: an edit of a hashed source calls for a new record.
    runs = [record['environment'], *record['environment'].get('merged', [])]
    assert all(run['dirty_paths'] == 0 for run in runs) and len({run['commit'] for run in runs}) == 1
    assert all(run.get('source_sha256', record['source_sha256']) == record['source_sha256'] for run in runs)
    for row in record['rows']:
        if row['kind'] == 'stability':
            assert row['steps_completed'] == row['steps'] and len(row['samples']) == row['steps']//row['sample_interval'], row['id']


def test_every_verdict_follows_from_the_record_and_the_case_limits(case, record):
    for row in record['rows']:
        again = bench.verdict({k: v for k, v in row.items() if k not in ('judgement', 'passed')}, case['acceptance'])
        assert again['judgement'] == row['judgement'] and again['passed'] == row['passed'], row['id']
    assert record['failing_ids'] == [row['id'] for row in record['rows'] if not row['passed']]
    assert record['passed'] and not record['failing_ids'], record['failing_ids']


def test_document_is_rendered_from_the_record(record):
    assert (ROOT/'docs'/'DISPERSIVE_PML_ABSORBER.md').read_text(encoding='utf-8') == bench.render(record)
    assert record['passed'] == (not record['failing_ids'])


# ----------------------------------------------------------------------------------------------- coefficients
def test_faces_reached_by_dispersive_structures_become_the_absorber():
    p = slab_project()
    assert absorber_faces(p) == ((0, 1), (1, 0), (1, 1))
    assert absorber_faces(slab_project('ade')) == () and absorber_faces(slab_project('frozen')) == ()
    three = slab_project(dimension='3d')
    assert absorber_faces(three) == ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    inside = slab_project()
    inside.structures[0].size = (.2, .2, 1.)
    inside.structures[0].center = (0, 0, 0)
    assert absorber_faces(Project.model_validate(inside.model_dump())) == ()
    with pytest.raises(ValueError, match='Absorber faces require pml_dispersion="absorber"'):
        BoundaryDescription(slab_project('ade').region, ((0, 1),))


def test_profile_is_the_graded_depth_of_each_absorber_face_and_corners_add():
    p = slab_project()
    r = p.region
    courant = r.rectangular_courant
    profiles = absorber_profiles(r, courant, absorber_faces(p))
    assert set(profiles) == {0, 1}
    nodes = np.asarray(r.mesh_nodes[0])
    assert not np.any(profiles[0][0][:len(nodes)//2]) and not np.any(profiles[0][1][:len(nodes)//2])   # x_min keeps the CPML
    for side, layers, scale, power in ((1, 7, .8, 2),):
        inner, outer = (nodes[layers], nodes[0]) if side == 0 else (nodes[-1-layers], nodes[-1])
        for k, points in enumerate((nodes[:-1], (nodes[:-1]+nodes[1:])/2)):
            rho = np.clip((points-inner)/(outer-inner), 0, 1)
            half = slice(0, len(points)//2) if side == 0 else slice(len(points)//2, None)
            np.testing.assert_allclose(profiles[0][k][half], (scale*40/(layers+1)*rho**power*courant/2)[half], rtol=1e-15, atol=0)
    n = len(nodes)-1
    assert profiles[1][0][0] == pytest.approx(40/5*courant/2) and profiles[0][0][n-7] == 0 and profiles[0][1][n-7] > 0
    full = absorber_loss(profiles, r.shape, 'E')
    # Ex is a half node along x and an integer node along y; Ez an integer node on both.
    assert full[-1, 0, 0, 0] == profiles[0][1][-1]+profiles[1][0][0] and full[-1, 0, 0, 2] == profiles[0][0][-1]+profiles[1][0][0]
    h = absorber_loss(profiles, r.shape, 'H')
    assert h[-1, 0, 0, 2] == profiles[0][1][-1]+profiles[1][1][0]
    rng = np.random.default_rng(3)
    flat = rng.choice(full.size, 50, replace=False)
    np.testing.assert_array_equal(absorber_loss(profiles, r.shape, 'E', flat, True), full.reshape(-1)[flat])
    cells = rng.choice(full.size//3, 20, replace=False)
    np.testing.assert_array_equal(absorber_loss(profiles, r.shape, 'H', cells, False), h.reshape(-1, 3)[cells])


def test_absorber_replaces_the_stretched_coordinates_of_its_faces_and_enters_the_plan():
    p = slab_project()
    template = BoundaryDescription(p.region, absorber_faces(p))
    segments = [(axis, s['side']) for (_, axis, _), items in template.cpml.items() for s in items]
    assert set(segments) == {(0, 0)} and template.absorber is not None
    ade = slab_project('ade')
    assert BoundaryDescription(ade.region).absorber is None
    assert {(axis, s['side']) for (_, axis, _), items in BoundaryDescription(ade.region).cpml.items() for s in items} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    plan, reference = resolve_plan(p), resolve_plan(ade)
    assert plan.plan_hash != reference.plan_hash and plan.pml_dispersion == 'absorber'
    g, _ = numpy_grid(p)
    plan.verify_grid(g)
    assert Simulation(p).run().summary['absorber_faces'] == ['x_max', 'y_min', 'y_max']


def bilinear_reference(material, dt):
    """Re eps at the 0.55 um source centre from the poles written out here; eps_inf where it is not positive."""
    f = 299792458./.55e-6
    omega = 2/dt*np.tan(np.pi*f*dt)
    eps = material.epsilon_inf+0j
    if material.model == 'lorentz':
        eps += material.delta_epsilon*material.resonance_rad_s**2/(material.resonance_rad_s**2-omega**2-2j*material.linewidth_rad_s*omega)
    elif material.model == 'drude':
        eps -= material.plasma_rad_s**2/(omega**2+1j*material.collision_rad_s*omega)
    else:
        for pole in material.poles:
            eps += pole.strength_rad_s_squared/(pole.resonance_rad_s**2-omega**2-1j*pole.damping_rad_s*omega)
    return eps.real if eps.real > 0 else material.epsilon_inf


@pytest.mark.parametrize('material,expected', [(SIN, (4.1, 4.3)), (LORENTZ_225, (5.35, 5.55)), (METAL, None), (MULTIPOLE, (5.6, 5.85))],
                         ids=['sin', 'lorentz-eps-inf-2.25', 'metal-negative', 'multipole'])
def test_absorber_update_is_the_trapezoidal_lossy_ade_update(material, expected):
    """(eps_inf + s eps_ref) E_new + (P_new - P_old) = (eps_inf - s eps_ref) E_old + courant*curl H on every E sample,
    eps_ref the bilinear Re eps at the source centre, or eps_inf where that is negative (eps elsewhere); H decays with the same s."""
    for sampling in ('yee', 'cell'):
        # The slab fills the region, so every face is an absorber and no CPML memory advances with the extra curls below.
        p = slab_project(dimension='3d', sampling=sampling, everywhere=True, material=material)
        p.structures.append(Structure(name='hole', center=(0, 0, 0), size=(.4, .4, .4), material='void'))
        p = Project.model_validate(p.model_dump())
        assert len(absorber_faces(p)) == 6
        g, owner = numpy_grid(p)
        rng = np.random.default_rng(7)
        g.E[:] = rng.standard_normal(g.E.shape)
        g.H[:] = rng.standard_normal(g.H.shape)
        # The samples without absorber loss keep a plain ADE state; only the lossy ones carry the absorber coefficients.
        states = g.material_states
        assert [state.absorber is not None for state in states] == [False, True]
        for state in states:
            state.P[:] = rng.standard_normal(state.P.shape)
            state.Q[:] = rng.standard_normal(state.Q.shape)
        e0, h0, p0 = g.E.copy(), g.H.copy(), [state.P.copy() for state in states]
        curl_h = g.curl(g.H, False)
        g.update_E()
        s = absorber_loss(g.absorber, p.region.shape, 'E')
        take = lambda array, state: (array.reshape(-1) if state.components else array.reshape(-1, 3))[state.indices]
        assert s.max() > .5 and not np.any(take(s, states[0])) and np.all(np.any(take(s, states[1]).reshape(len(states[1].indices), -1), axis=1))
        eps = 1/g.inverse_permittivity
        reference = bilinear_reference(material, g.time_step)
        assert (reference == material.epsilon_inf) if expected is None else expected[0] < reference < expected[1]
        assert np.all(eps[eps != 1] == material.epsilon_inf)
        ref, dp = eps.copy(), np.zeros_like(g.E)
        for state, old in zip(states, p0):
            for array, value in ((ref, reference), (dp, (state.P-old).sum(axis=0) if state.multiple else state.P-old)):
                (array.reshape(-1) if state.components else array.reshape(-1, 3))[state.indices] = value
        lhs = (eps+s*ref)*g.E+dp
        rhs = (eps-s*ref)*e0+g.courant_number*curl_h
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-12)
        assert np.abs(dp).max() > 1e-3
        curl_e = g.curl(g.E, True)
        g.update_H()
        sh = absorber_loss(g.absorber, p.region.shape, 'H')
        np.testing.assert_allclose((1+sh)*g.H, (1-sh)*h0-g.courant_number*curl_e, rtol=0, atol=1e-12)


def test_default_ade_keeps_the_cpml_update_and_kernel_text():
    from torchfdtd.cuda_kernels import FusedYeeCUDA
    old_dtype = torch.get_default_dtype()
    fdtd.set_backend('torch.float64')
    try:
        texts = {}
        for mode in ('ade', 'absorber'):
            p = slab_project(mode)
            g = YeeGrid(p.region, absorber_faces(p))
            texts[mode] = FusedYeeCUDA._source(SimpleNamespace(grid=g), False)[0]+FusedYeeCUDA._source(SimpleNamespace(grid=g), True)[0]
            assert (g.absorber is None) == (mode == 'ade')
    finally:
        fdtd.set_backend('numpy')
        fdtd.backend.float = np.float64
        torch.set_default_dtype(old_dtype)
    assert 'absorber' not in texts['ade'] and {f'psi{t}_1' in texts['ade'] for t in range(4)} == {True}
    # x_min keeps its CPML memory; x_max and the y faces are absorbers.
    assert 'psi0_0' in texts['absorber'] and 'psi0_1' not in texts['absorber'] and 'psi2_0' not in texts['absorber']
    assert texts['absorber'].count('(1-s)/(1+s)') == 6


# ----------------------------------------------------------------------------------------------- refusals and warnings
def test_paths_without_the_absorber_refuse_it_only_where_it_changes_the_run():
    from torchfdtd import DifferentiableSimulation, StreamedSimulation, StreamedAdjointOptions, DifferentiablePlaneSimulation
    from torchfdtd.differentiable import _System
    from torchfdtd.open_mode_operators import plan_open_mode_operators
    for mode in ('absorber', 'frozen'):
        p = slab_project(mode)
        message = f'pml_dispersion="{mode}" is implemented by the resident Yee Simulation and run_tensor_batch only'
        for build in (lambda: DifferentiableSimulation(p), lambda: StreamedSimulation(p, StreamedAdjointOptions(device='cpu')),
                      lambda: _System(p, torch.ones(p.region.shape, dtype=torch.float64))):
            with pytest.raises(ValueError, match=message):
                build()
        plane = p.model_copy(deep=True)
        plane.monitors = [FieldMonitor(normal='x', center=(.2, 0, 0), size=(0, .6, 1))]
        with pytest.raises(ValueError, match='not by DifferentiablePlaneSimulation'):
            DifferentiablePlaneSimulation(Project.model_validate(plane.model_dump()))
        # Without a dispersive structure in a PML layer neither mode changes anything, and nothing is refused.
        dielectric = Project.model_validate(dict(plane.model_dump(), materials=[m.model_dump() for m in plane.materials[:1]]+[
            Material(name='sin', index=2.).model_dump()]))
        DifferentiablePlaneSimulation(dielectric)
        DifferentiableSimulation(dielectric.model_copy(update=dict(monitors=p.monitors)))
        plan_open_mode_operators(slab_project(mode, dimension='3d', precision='float32').region, 'x', 1.55)


def test_paths_with_oscillator_parameter_tensors_refuse_every_mode_but_ade():
    """DispersiveSimulation and its plane and streamed forms take their poles from tensors, not from structures, so
    'frozen' and 'absorber' cannot be honoured there, and a project without dispersive structures must not slip through."""
    from torchfdtd import (DispersiveSimulation, DispersivePlaneSimulation, StreamedDispersiveSimulation, StreamedAdjointOptions,
                           AdjointOptions)
    from test_differentiable import project
    from test_adjoint_planes import scene
    for mode in ('frozen', 'absorber'):
        p, plane = project(steps=10), scene()
        p.region.pml_dispersion = plane.region.pml_dispersion = mode
        p, plane = Project.model_validate(p.model_dump()), Project.model_validate(plane.model_dump())
        for build, name in ((lambda: DispersiveSimulation(p, AdjointOptions(checkpoints=2)), 'DispersiveSimulation'),
                            (lambda: StreamedDispersiveSimulation(p, StreamedAdjointOptions(device='cpu')), 'StreamedDispersiveSimulation'),
                            (lambda: DispersivePlaneSimulation(plane, AdjointOptions(checkpoints=2)), 'DispersivePlaneSimulation')):
            with pytest.raises(ValueError, match=f'pml_dispersion="{mode}" .* not by {name}, whose oscillators are parameter tensors'):
                build()
    DispersiveSimulation(project(steps=10), AdjointOptions(checkpoints=2))
    DispersivePlaneSimulation(scene(), AdjointOptions(checkpoints=2))


def test_native_tensor_projects_run_with_every_mode():
    from test_tensor_native import scene
    reference = None
    for mode in ('ade', 'frozen', 'absorber'):
        p = scene(True)
        p.region.pml_dispersion = mode
        signals = Simulation(Project.model_validate(p.model_dump())).run().signals
        reference = signals if reference is None else reference
        np.testing.assert_array_equal(signals, reference)


def extended_sheet_project(extend=True, crossing='x'):
    """2D: a SiN slab through the x faces (or the y faces) and a y-normal soft sheet, extended through the x layers or not."""
    r = Region(dimension='2d', size=(2., 2., 1.), mesh=.1, pml_cells=4, steps=20, precision='float64', backend='cpu',
               material_sampling='yee', pml_dispersion='absorber')
    slab = (Structure(name='slab', center=(0, .1, 0), size=(100., .2, 1.), material='sin') if crossing == 'x' else
            Structure(name='slab', center=(.3, 0, 0), size=(.2, 100., 1.), material='sin'))
    sheet = Source(name='sheet', kind='plane', normal='y', component='Ez', center=(0, -.4, 0), size=(2. if extend else 1.2, 0, 0),
                   extend_through_pml=extend, wavelength=.55, pulse_cycles=2)
    return Project(region=r, materials=[Material(name='void', index=1), SIN], structures=[slab], sources=[sheet],
                   monitors=[Monitor(component='Ez', center=(0, -.2, 0))])


def test_a_sheet_extended_through_an_absorber_face_is_refused():
    with pytest.raises(ValueError, match='sheet: a soft sheet with extend_through_pml crosses the absorber face x_min'):
        Simulation(extended_sheet_project()).run()
    assert absorber_faces(extended_sheet_project(extend=False)) == ((0, 0), (0, 1))
    # The sheet may still extend through CPML faces when the dispersive slab crosses only the y faces.
    crossing_y = extended_sheet_project(crossing='y')
    assert absorber_faces(crossing_y) == ((1, 0), (1, 1))
    assert Simulation(crossing_y).run().summary['absorber_faces'] == ['y_min', 'y_max']


def test_tiled_admission_refuses_extended_sheets_across_tile_absorbers_and_notes_the_faces():
    from test_tiled import small_row
    from torchfdtd.execution_modes import _tiled_candidate
    health = dict(host_available_bytes=1 << 40, gpu_free_bytes=None)
    out = {}
    for through in (True, False):
        p = small_row(through_pml=through)
        p.materials.append(SIN.model_copy(update=dict(name='dispersive sin')))
        for structure in p.structures:
            structure.material = 'dispersive sin'
        p.region.pml_dispersion = 'absorber'
        p.region.tiling = p.region.tiling.model_copy(update=dict(size_um=1.5, overlap_um=.5))
        out[through] = _tiled_candidate(Project.model_validate(p.model_dump()), 'cpu', health)
    assert not out[True]['admitted'] and 'extend_through_pml crosses the absorber face' in out[True]['reason']
    assert out[False]['admitted'] and any('absorber faces' in note for note in out[False]['notes'])


def test_run_tiled_checks_every_tile_first_and_admission_takes_the_largest_estimate(monkeypatch):
    from test_tiled import small_row
    import torchfdtd.solver as solver
    from torchfdtd.execution_modes import _tiled_candidate, _tiled_plan
    from torchfdtd.tiled import run_tiled

    def tiled(through):
        p = small_row(through_pml=through)
        p.materials.append(SIN.model_copy(update=dict(name='dispersive sin')))
        for structure in p.structures:
            structure.material = 'dispersive sin'
        p.region.pml_dispersion = 'absorber'
        p.region.tiling = p.region.tiling.model_copy(update=dict(size_um=1.5, overlap_um=.5))
        p = Project.model_validate(p.model_dump())
        return p, _tiled_plan(p)[0]
    ran = []
    monkeypatch.setattr(solver.Simulation, 'run', lambda self, **_: ran.append(self) or pytest.fail('a tile ran'))
    p, plan = tiled(True)
    with pytest.raises(ValueError, match='extend_through_pml crosses the absorber face'):
        run_tiled(p, plan, backend='cpu')
    assert ran == []
    # Two tiles of one shape, both with absorber faces: the admission quotes the larger estimate, not the first tile's.
    p, plan = tiled(False)
    assert len({t.project.region.shape for t in plan.tiles}) == 1 and all(absorber_faces(t.project) for t in plan.tiles)
    sizes = {str([s.center for s in t.project.structures]): 10.*(i+1) for i, t in enumerate(plan.tiles)}
    monkeypatch.setattr(solver, 'estimate', lambda q, **_: dict(estimated_memory_mb=sizes[str([s.center for s in q.structures])]))
    candidate = _tiled_candidate(p, 'cpu', dict(host_available_bytes=1 << 40, gpu_free_bytes=None))
    assert candidate['plan']['largest_tile_estimate_bytes'] == int(max(sizes.values())*2**20)


def test_run_and_tile_signatures_distinguish_absorber_faces():
    """A device with absorber faces and its air reference differ in their signatures; without absorber faces nothing changes."""
    from torchfdtd.tiled import plan_tiles, _signature
    from test_tiled import small_row
    metal = Material(name='metal', model='drude', epsilon_inf=1., plasma_rad_s=2e15, collision_rad_s=1e14)
    for mode in ('absorber', 'ade'):
        r = Region(dimension='2d', size=(2.0, 2.0, 1.), mesh=.05, pml_cells=6, steps=50, backend='cpu', material_sampling='yee',
                   pml_dispersion=mode)
        device = Project(region=r, materials=[Material(name='void', index=1), metal],
                         structures=[Structure(name='film', center=(0, .3, 0), size=(100., .05, 1.), material='metal')],
                         sources=[Source(kind='plane', component='Ez', center=(0, -.4, 0), size=(1.2, 0, 0), wavelength=1.55)],
                         monitors=[FieldMonitor(normal='y', center=(0, .6, 0), size=(1.2, 0, 1))])
        air = device.model_copy(update=dict(structures=[]))
        assert (run_signature(device, 50) == run_signature(air, 50)) == (mode == 'ade')
        tiled = small_row(through_pml=False)
        tiled.materials.append(SIN.model_copy(update=dict(name='dispersive sin')))
        for structure in tiled.structures:
            structure.material = 'dispersive sin'
        tiled.region.pml_dispersion = mode
        tiled = Project.model_validate(tiled.model_dump())
        empty = tiled.model_copy(update=dict(structures=[]))
        signatures = [_signature(plan_tiles(q, 1.5, .5, normal='y')) for q in (tiled, empty)]
        assert (signatures[0] == signatures[1]) == (mode == 'ade')


def test_absorber_refuses_pmc_faces_and_subpixel_interfaces():
    r = Region(dimension='3d', size=(1.2, 1.2, 1.2), mesh=.1, pml_cells=3, material_sampling='yee', pml_dispersion='absorber',
               boundaries=Boundaries(x_min=BoundaryFace(kind='pmc'), x_max=BoundaryFace(kind='pmc')))
    with pytest.raises(ValueError, match='does not implement PMC/symmetric faces'):
        BoundaryDescription(r, ((1, 0),))
    sub = Region(dimension='2d', size=(1.6, 1.6, 1.), mesh=.1, pml_cells=3, material_sampling='yee', interface_method='subpixel',
                 pml_dispersion='absorber')
    with pytest.raises(ValueError, match='requires staircase interfaces'):
        BoundaryDescription(sub, ((0, 0),))
    for face in (dict(kappa=3.), dict(alpha=.1, alpha_polynomial=1.)):
        stretched = Region(dimension='2d', size=(1.6, 1.6, 1.), mesh=.1, pml_cells=3, pml_dispersion='absorber',
                           boundaries=Boundaries(y_max=BoundaryFace(**face)))
        with pytest.raises(ValueError, match='kappa and alpha of its faces must stay at 1 and at most the default alpha'):
            BoundaryDescription(stretched, ((1, 1),))
        BoundaryDescription(stretched, ((1, 0),))
    # No stretching: alpha 0 (as the endpoint CPML requires) and an alpha polynomial without alpha are admitted.
    for face in (dict(alpha=0.), dict(alpha=0., alpha_polynomial=2.)):
        plain = Region(dimension='2d', size=(1.6, 1.6, 1.), mesh=.1, pml_cells=3, pml_dispersion='absorber',
                       boundaries=Boundaries(y_max=BoundaryFace(**face)))
        assert BoundaryDescription(plain, ((1, 1),)).absorber is not None


def post_in_layer_project(material, *, mode='ade', size=.2, span=None):
    """2D: a post of the material lying inside the x_max layer only (it ends inside the layer), or spanning `span` in x."""
    r = Region(dimension='2d', size=(2., 1.6, 1.), mesh=.1, pml_cells=4, steps=10, material_sampling='yee', pml_dispersion=mode)
    lo, hi = span or (1.-size/2, 1.+size/2)
    return Project(region=r, materials=[Material(name='void', index=1), material.model_copy(update=dict(name='post material'))],
                   structures=[Structure(name='post', center=((lo+hi)/2, 0, 0), size=(hi-lo, .2, 1.), material='post material')],
                   sources=[Source(component='Ez', center=(-.3, 0, 0), wavelength=.55)], monitors=[Monitor(component='Ez', center=(0, 0, 0))])


def test_warning_recommends_the_absorber_where_a_negative_permittivity_structure_has_an_end_in_the_layer():
    (ending,) = stability_warnings(post_in_layer_project(SIN))
    assert 'post (x_max)' in ending and 'set region.pml_dispersion="absorber"' in ending and 'grows by e^0.057 per step' in ending
    # Entering from the interior and ending two cells short of the outer edge (the record's enter rows) is an end in the layer;
    # running through the outer edge is a crossing, for which no stability is claimed.
    high = post_in_layer_project(SIN).region.interior_bounds(0)[1]
    (entering,) = stability_warnings(post_in_layer_project(SIN, span=(high-.3, 1.-.2)))
    assert entering.startswith('Dispersive material inside PML layers, ending there: post (x_max)') and 'e^0.10' in entering
    for project in (post_in_layer_project(SIN, span=(high-.3, 1.3)), slab_project('ade', everywhere=True)):
        (crossing,) = stability_warnings(project)
        assert crossing.startswith('Dispersive material inside PML layers, crossing them:') and 'not known to be stable' in crossing
        assert 'e^0.0011 per step' in crossing and '0.016 to 1.2' in crossing and 'stayed stable' not in crossing
    assert 'slab (x_min, x_max, y_min, y_max)' in stability_warnings(slab_project('ade', everywhere=True))[0]
    # A strongly damped pole keeps Re eps positive on the whole band of the grid: no recommendation either way.
    damped = Material(name='damped', model='lorentz', epsilon_inf=4., resonance_rad_s=1e15, linewidth_rad_s=2e15, delta_epsilon=1.)
    for span in (None, (high-.3, 1.3)):
        (positive,) = stability_warnings(post_in_layer_project(damped, span=span))
        assert 'keep pml_dispersion="ade"' in positive and 'stays positive' in positive
    assert stability_warnings(slab_project('absorber')) == [] and stability_warnings(post_in_layer_project(SIN, mode='absorber')) == []


@pytest.mark.parametrize('sampling', ['yee', 'cell'])
def test_a_face_counts_exactly_when_a_pole_sample_lies_in_its_layer_rows(sampling):
    """The rows 'frozen' freezes decide: a box touching the inner edge reaches an upper layer with Yee sampling (the node on
    the edge is in its rows) and a lower layer only once it covers the half node inside it; with cell sampling, the cell centres."""
    from torchfdtd.boundaries import reject_pml_dispersion
    from torchfdtd.materials import pml_cell_mask
    from torchfdtd.stability_checks import dispersive_structures_in_pml
    r = Region(dimension='2d', size=(2., 1.6, 1.), mesh=.1, pml_cells=4, steps=10, precision='float64', backend='cpu',
               material_sampling=sampling, pml_dispersion='frozen')
    low, high = r.interior_bounds(0)
    rows = pml_cell_mask(r, r.shape)
    for side in (0, 1):
        for depth in (-.3, 0., .3, .7, 1.3):
            edge = high+depth*.1 if side else low-depth*.1
            lo, hi = (-.3, edge) if side else (edge, .3)
            box = Structure(name='box', center=((lo+hi)/2, 0, 0), size=(hi-lo, .4, 1.), material='sin')
            p = Project(region=r, materials=[Material(name='void', index=1), SIN], structures=[box],
                        sources=[Source(component='Ez', center=(0, .3, 0), wavelength=.55)], monitors=[Monitor(component='Ez', center=(0, 0, 0))])
            _, _, owner = voxelize(p, with_ownership=True)
            layer = np.zeros(r.shape, dtype=bool)
            layer[(slice(0, 4) if side == 0 else slice(r.shape[0]-4, None),)] = True
            poles = owner == 1 if owner.ndim == 3 else np.any(owner == 1, axis=3)
            expected = bool(np.any(poles & layer & rows))
            reached = dispersive_structures_in_pml(p) == [('box', ['x_max' if side else 'x_min'])]
            assert reached == expected and (reached or not dispersive_structures_in_pml(p)), (side, depth)
            if expected:
                with pytest.raises(ValueError, match='pml_dispersion="frozen"'):
                    reject_pml_dispersion(p, 'a path')
            else:
                reject_pml_dispersion(p, 'a path')
    if sampling == 'yee':
        assert expected  # the last box, 1.3 cells into the lower layer, has pole samples there


@pytest.mark.parametrize('gap', [0., 1e-12, -.05])
def test_a_structure_touching_the_layer_makes_its_face_an_absorber(gap):
    """A SiN box ending on the inner edge of the x_max layer (or inside it) turns x_max into an absorber, so no pole sample
    stays in a stretched row; the upper E rows of the CPML include the node on that edge."""
    from torchfdtd.boundaries import CURL_TERMS
    r = Region(dimension='2d', size=(2., 1.6, 1.), mesh=.1, pml_cells=4, steps=10, precision='float64', backend='cpu',
               material_sampling='yee', pml_dispersion='absorber')
    high = r.interior_bounds(0)[1]
    edge = high-gap
    box = Structure(name='box', center=((edge-.2)/2, 0, 0), size=(edge+.2, .4, 1.), material='sin')
    p = Project(region=r, materials=[Material(name='void', index=1), SIN], structures=[box],
                sources=[Source(component='Ez', center=(-.3, 0, 0), wavelength=.55)], monitors=[Monitor(component='Ez', center=(0, 0, 0))])
    assert (0, 1) in absorber_faces(p)
    fdtd.set_backend('numpy')
    _, _, owner = voxelize(p, with_ownership=True)
    g = YeeGrid(r, absorber_faces(p))
    stretched = 0
    for (forward, axis, comp), segments in g.cpml.items():
        if forward:
            continue
        out = next(o for a, c, o, _ in CURL_TERMS if a == axis and c == comp)
        for segment in segments:
            index = [slice(None)]*3
            index[axis] = slice(segment['slice'][axis].start+1, segment['slice'][axis].stop+1)
            stretched += int(np.count_nonzero(owner[tuple(index)+(out,)] == 1))
    assert stretched == 0


# ----------------------------------------------------------------------------------------------- short runs
def test_short_run_of_the_diverging_fixture_grows_with_the_cpml_and_decays_with_the_absorber():
    """The case's corner post in a 0.8 um cube for 2000 steps; the pulse ends at step 1390."""
    torch.set_num_threads(2)
    out = {}
    for mode in ('ade', 'absorber'):
        samples, _, error = bench.run_samples(bench.corner_post_project('sin', mode, size=(.8, .8, .8), steps=2000, sample=200))
        assert error is None
        peak = max(s['state_norm'] for s in samples if s['step'] <= 1400)
        out[mode] = samples[-1]['state_norm']/peak
    assert out['ade'] > 1e6 and out['absorber'] < 1e-6, out


def spectral_radius(p):
    """Largest |eigenvalue| of the source-free one-step operator on E, H, the CPML memories and the pole states (float64)."""
    fdtd.set_backend('numpy')
    fdtd.backend.float = np.float64
    g, _ = numpy_grid(p)
    arrays = [g.E, g.H, *g.memory_states]
    sizes = [a.size for a in arrays]
    columns = []
    for j in range(sum(sizes)):
        vector = np.zeros(sum(sizes))
        vector[j] = 1.
        offset = 0
        for a, size in zip(arrays, sizes):
            a.reshape(-1)[:] = vector[offset:offset+size]
            offset += size
        g.update_E()
        g.update_H()
        columns.append(np.concatenate([a.reshape(-1) for a in arrays]))
    return float(np.abs(np.linalg.eigvals(np.stack(columns, axis=1))).max())


def corner_post_cell(mode, material, sampling):
    """11 x 11 x 5 cells, 3 CPML layers, z periodic, a post filling the x_max/y_max corner (after the reviewer's fixture)."""
    faces = dict(z_min=BoundaryFace(kind='periodic'), z_max=BoundaryFace(kind='periodic'))
    r = Region(dimension='3d', size=(.22, .22, .1), mesh=.02, pml_cells=3, steps=10, precision='float64', backend='cpu',
               material_sampling=sampling, pml_dispersion=mode, boundaries=Boundaries(**faces))
    return Project(region=r, materials=[Material(name='void', index=1), material],
                   structures=[Structure(name='post', center=(.11, .11, 0), size=(.12, .12, .06), material='sin')],
                   sources=[Source(kind='point', component='Ex', center=(0, 0, 0), wavelength=.55, pulse_cycles=2)],
                   monitors=[Monitor(component='Ex', center=(0, 0, 0))])


@pytest.mark.long
@pytest.mark.parametrize('material,sampling', [pytest.param(m, s, id=f'{n}-{s}') for n, m, s in
                                               (('sin', SIN, 'yee'), ('sin', SIN, 'cell'), ('drude', DRUDE, 'yee'), ('drude', DRUDE, 'cell'))])
def test_one_step_operator_grows_with_the_cpml_and_is_contractive_with_the_absorber(material, sampling):
    torch.set_num_threads(2)
    assert spectral_radius(corner_post_cell('ade', material, sampling))-1 > 5e-4
    absorber = corner_post_cell('absorber', material, sampling)
    assert absorber_faces(absorber) == ((0, 1), (1, 1))
    assert spectral_radius(absorber)-1 <= 1e-12


def test_one_step_operator_of_a_periodic_row_grows_with_the_cpml_and_is_contractive_with_the_absorber():
    """The fast form of the test above: 11 x 5 x 5 cells, y and z periodic, a Drude post ending in the x_max layer."""
    torch.set_num_threads(2)
    periodic = BoundaryFace(kind='periodic')
    def row(mode):
        r = Region(dimension='3d', size=(.22, .1, .1), mesh=.02, pml_cells=3, steps=10, precision='float64', backend='cpu',
                   material_sampling='yee', pml_dispersion=mode,
                   boundaries=Boundaries(y_min=periodic, y_max=periodic, z_min=periodic, z_max=periodic))
        return Project(region=r, materials=[Material(name='void', index=1), DRUDE],
                       structures=[Structure(name='post', center=(.11, 0, 0), size=(.12, .06, .06), material='sin')],
                       sources=[Source(kind='point', component='Ex', center=(0, 0, 0), wavelength=.55, pulse_cycles=2)],
                       monitors=[Monitor(component='Ex', center=(0, 0, 0))])
    assert spectral_radius(row('ade'))-1 > 5e-4
    absorber = row('absorber')
    assert absorber_faces(absorber) == ((0, 1),)
    assert spectral_radius(absorber)-1 <= 1e-12


def test_torch_grid_keeps_the_absorber_coefficients_on_its_slabs_and_the_estimate_counts_them():
    p = slab_project(dimension='3d', everywhere=True)
    g, _ = numpy_grid(p)
    g.update_E()
    g.update_H()
    cached = sum(decay.size+gain.size for family in ('E', 'H') for _, decay, gain in g.absorber_update(family))
    shape = p.region.shape
    interior = [n-p.region.pml_layers(a, 0)*((a, 0) in g.absorber_faces)-p.region.pml_layers(a, 1)*((a, 1) in g.absorber_faces)
                for a, n in enumerate(shape)]
    slab_cells = int(np.prod(shape)-np.prod(interior))
    assert cached == 4*3*slab_cells < 4*g.E.size
    # The pole coefficients of the absorber exist only for the pole samples with loss, all inside the slabs.
    lossy = [state for state in g.material_states if state.absorber is not None]
    poles = sum(state.P.size for state in g.material_states)
    coefficients = sum(a.size for state in lossy for a in state.absorber)
    assert len(lossy) == 1 and 0 < coefficients == 3*lossy[0].P.size < 3*poles and coefficients <= 3*3*slab_cells
    summary, ade = estimate(p), estimate(slab_project('ade', dimension='3d', everywhere=True))
    assert summary['absorber_estimated_bytes'] == (cached+3*3*slab_cells)*8 and 'absorber_estimated_bytes' not in ade
    # Building those coefficients is a host transient of the slab samples, which the CUDA host estimate counts too.
    from torchfdtd.solver import HOST_ABSORBER_SLAB_BYTES, HOST_MATERIAL_BYTES
    cells = int(np.prod(shape))
    assert cells*HOST_MATERIAL_BYTES['yee'][0]+slab_cells*HOST_ABSORBER_SLAB_BYTES['yee'] > cells*(HOST_MATERIAL_BYTES['yee'][0]+9*8)
    assert summary['host_estimated_mb'] > ade['host_estimated_mb']


def test_plan_describes_the_absorber_and_verify_grid_checks_it():
    p = slab_project(dimension='3d')
    plan = resolve_plan(p)
    absorber = plan.boundaries.absorber
    assert absorber['faces'] == [[0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    assert set(absorber['profiles']) == {0, 1, 2} and absorber['reference_epsilon']['sin'] == pytest.approx(
        bilinear_reference(SIN, p.region.time_step), rel=1e-12)
    assert 'absorber' in plan.to_json()['boundaries'] and 'absorber' not in resolve_plan(slab_project('ade')).to_json()['boundaries']
    g, _ = numpy_grid(p)
    plan.verify_grid(g)
    g.absorber[1][0][0] *= 2
    with pytest.raises(PlanInvalidated, match='absorber'):
        plan.verify_grid(g)
    g, _ = numpy_grid(p)
    g.absorber_reference['sin'] += 1e-9
    with pytest.raises(PlanInvalidated, match='reference permittivity'):
        plan.verify_grid(g)


@pytest.mark.parametrize('material', ['dilute', 'sin'])
def test_short_normal_incidence_reflection_of_a_thin_absorber(material):
    """12 layers and 60 fs: a homogeneous fill (vacuum with a 1e-6 pole, or SiN) through the x absorbers stays below the normal limit."""
    kw = dict(material=material, pol='TE', layers=12, duration_fs=60., fill='full')
    short, long = (Simulation(bench.reflection_project(long=flag, **kw)).run() for flag in (False, True))
    assert short.summary['absorber_faces'] == ['x_min', 'x_max']
    assert bench.reflection(short, long, 0)['R_max_on_band'] < 1e-6


# ----------------------------------------------------------------------------------------------- CUDA
@CUDA
@pytest.mark.parametrize('precision,tolerance', [('float64', 1e-11), ('float32', 1e-4)])
def test_fused_kernel_matches_the_torch_update(precision, tolerance):
    pytest.importorskip('cupy')
    runs = {}
    for kernel in ('torch', 'fused'):
        p = slab_project(dimension='3d', backend='cuda', precision=precision, kernel=kernel, steps=200)
        runs[kernel] = Simulation(p).run()
    reference = runs['torch'].signals[:, 0]
    scale = np.abs(reference).max()
    assert scale > 0 and np.abs(runs['fused'].signals[:, 0]-reference).max() <= tolerance*scale
    if precision == 'float64':
        cpu = Simulation(slab_project(dimension='3d', steps=200)).run().signals[:, 0]
        assert np.abs(cpu-reference).max() <= 1e-11*scale


@CUDA
def test_tensor_batch_matches_the_fused_simulation():
    pytest.importorskip('cupy')
    from torchfdtd import run_tensor_batch
    p = slab_project(dimension='3d', backend='cuda', precision='float32', kernel='fused', steps=200)
    single = Simulation(p).run().signals
    other = p.model_copy(deep=True)
    other.structures[0].center = (.65, 0, 0)
    report = run_tensor_batch([p, other], device=0)
    np.testing.assert_array_equal(report.items[0].result.signals, single)
    assert report.items[0].result.summary['absorber_faces'] == ['x_max', 'y_min', 'y_max', 'z_min', 'z_max']
    mixed = p.model_copy(deep=True)
    mixed.region.pml_dispersion = 'ade'
    with pytest.raises(ValueError, match='identical'):
        run_tensor_batch([p, Project.model_validate(mixed.model_dump())], device=0)


# ----------------------------------------------------------------------------------------------- long
@pytest.mark.long
@pytest.mark.parametrize('material', ['sin', 'drude'])
def test_twenty_thousand_steps_of_the_diverging_fixture_in_float64(case, material):
    """The judged float64 stability rows, rerun: G3-07 decay and late-growth limits and the sweep growth ratio."""
    torch.set_num_threads(2)
    spec = next(s for s in bench.rows() if s['id'] == f'stability-{material}-post-absorber-cpu-float64')
    out = bench.verdict(bench.execute(spec), case['acceptance'])
    assert out['passed'], out['judgement']
