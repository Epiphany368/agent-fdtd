"""Dispersive subpixel interfaces (torchfdtd.subpixel_dispersive): node tensors, assembly, stability and backends."""
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Material, Structure, Source, Monitor, FieldMonitor, Simulation, DifferentiableSimulation, run_tensor_batch, LorentzPole
from torchfdtd.boundaries import YeeGrid
from torchfdtd.materials import configure_materials
from torchfdtd.run_control import StateDiagnostics
from torchfdtd.subpixel import prepare_interfaces, configure_interfaces
from torchfdtd.subpixel_dispersive import MIXED, WHOLE, NEXT, WINDOW
from test_solver import small

DRUDE = dict(model='drude', epsilon_inf=5., plasma_rad_s=1.37e16, collision_rad_s=1.5e14)
TWO_POLES = [LorentzPole(resonance_rad_s=0., strength_rad_s_squared=1.6e32, damping_rad_s=1e14),
             LorentzPole(resonance_rad_s=4e15, strength_rad_s_squared=8e30, damping_rad_s=3e14)]


def periodic_scene(structures, materials, mesh=.02, size=.24, bloch=False, dimension='3d'):
    axes = 'xyz' if dimension == '3d' else 'xy'
    bounds = {a+'_'+s: dict(kind='bloch' if bloch else 'periodic') for a in axes for s in ('min', 'max')}
    r = Region(dimension=dimension, size=(size, size, size if dimension == '3d' else 1.), mesh=mesh, material_sampling='yee',
               interface_method='subpixel', boundaries=bounds, bloch_phase=(.43, -.27, .18 if dimension == '3d' else 0) if bloch else (0, 0, 0),
               precision='float64', backend='cpu', steps=100)
    return Project(region=r, materials=materials, structures=structures)


def sphere_scene(collision=1.5e14, dielectric=False, bloch=False, poles=None, mesh=.02):
    metal = Material(name='metal', **dict(DRUDE, collision_rad_s=collision)) if poles is None else \
        Material(name='metal', model='multipole', epsilon_inf=3., poles=poles)
    structures = [Structure(kind='sphere', material='metal', radius=.071, center=(.013, -.007, .004))]
    if dielectric:
        structures.append(Structure(kind='rectangle', material='glass', size=(.3, .3, .05), center=(0, 0, -.09)))
    return periodic_scene(structures, [metal, Material(name='glass', index=2)], mesh=mesh, bloch=bloch)


def grid_for(project):
    plan = prepare_interfaces(project)
    fdtd.set_backend('numpy'); fdtd.backend.float = np.float64
    grid = YeeGrid(project.region)
    grid.inverse_permittivity[:] = plan.diagonal
    configure_materials(grid, project, plan.ownership)
    configure_interfaces(grid, plan)
    return grid, plan


def permittivity_at(material, omega):
    return material.instantaneous_epsilon+sum(s/(w0*w0-omega*omega-1j*g*omega) for w0, s, g in material.oscillators)


def dispersive_operator(plan, material, omega, size):
    """E = M(omega) D of the dispersive shares, assembled from the plan with the node tensor written out."""
    M = np.zeros((size, size), complex)
    eps_m = permittivity_at(material, omega)
    for g in plan.dispersive.groups:
        M[g['half_index'], g['half_index']] += g['half_weight']/eps_m
        f, C, B, n = g['fraction'], g['arithmetic'], g['harmonic'], g['normal']
        normal = B+f/eps_m; tangential = 1/(C+f*eps_m)
        Z = n[:, :, None]*n[:, None, :]*normal[:, None, None]+(np.eye(3)-n[:, :, None]*n[:, None, :])*tangential[:, None, None]
        phase = np.ones(g['edges'].shape) if g['phases'] is None else g['phases']
        for k in range(len(f)):
            for a in range(3):
                for s, sign in ((0, 1), (1, -1)):
                    row = g['edges'][a, s, k]
                    for b in range(3):
                        for t in range(2):
                            # 1/2 [(Z Dbar)_a + sign Z_aa delta_a], Dbar = (D+ + D-)/2, delta = (D+ - D-)/2
                            weight = Z[k, a, b]/4+(sign*(1 if t == 0 else -1)*Z[k, a, a]/4 if b == a else 0)
                            M[row, g['edges'][b, t, k]] += weight*np.conj(phase[a, s, k])*phase[b, t, k]
    return M


@pytest.mark.parametrize('poles', [None, TWO_POLES], ids=['drude', 'multipole'])
def test_dispersive_state_is_the_node_tensor_at_the_bilinear_frequency(poles):
    """Drive only the D-driven state with a sinusoidal D and compare with the static share plus the assembled node tensors at (2/dt) tan(w dt/2)."""
    project = sphere_scene(poles=poles)
    grid, plan = grid_for(project)
    state = grid.subpixel.dispersive
    driven = np.flatnonzero(plan.dispersive.driven)
    size = grid.E.size; dt = grid.time_step; c = grid.courant_number; omega = 4.6e15
    rng = np.random.default_rng(4)
    amplitude = np.zeros(size, complex); amplitude[driven] = rng.normal(size=len(driven))+1j*rng.normal(size=len(driven))
    steps = 9000
    t = np.arange(steps+1)*dt
    ramp = 1-np.exp(-(t/(300*dt))**2)
    D = (amplitude[None, :]*np.exp(-1j*omega*t)[:, None]*ramp[:, None]).real
    grid.E[:] = 0
    history = []
    for n in range(steps):
        state.apply(((D[n+1]-D[n])/c).reshape(grid.E.shape))
        if n >= steps//2:
            history.append(grid.E.reshape(-1)[driven].copy())
    history = np.array(history)
    time = t[steps//2+1:steps+1]
    basis = np.stack([np.cos(omega*time), np.sin(omega*time)], 1)
    fit = np.linalg.lstsq(basis, history, rcond=None)[0]
    measured = fit[0]+1j*fit[1]  # E = Re(measured e^{-i w t})
    Omega = 2/dt*np.tan(omega*dt/2)
    static = plan.update_diagonal.reshape(-1)*amplitude
    for row, cols, values in zip(plan.rows, plan.columns, plan.values):
        static[row] += values@amplitude[cols]
    expected = static[driven]+(dispersive_operator(plan, project.materials[0], Omega, size)@amplitude)[driven]
    np.testing.assert_allclose(measured, expected, rtol=0, atol=2e-6*abs(expected).max())
    continuum = static[driven]+(dispersive_operator(plan, project.materials[0], omega, size)@amplitude)[driven]
    assert abs(measured-continuum).max() > 1e-4*abs(expected).max()


def high_frequency_operator(plan, material, size):
    static = np.diag(plan.update_diagonal.reshape(-1)).astype(complex)
    for row, cols, values in zip(plan.rows, plan.columns, plan.values):
        np.add.at(static[row], cols, values)
    return static+dispersive_operator(plan, material, 1e30, size)


@pytest.mark.parametrize('bloch', [False, True])
def test_instantaneous_operator_is_hermitian_positive_and_cfl_bounded(bloch):
    project = sphere_scene(dielectric=True, bloch=bloch, mesh=.04)
    plan = prepare_interfaces(project)
    M = high_frequency_operator(plan, project.materials[0], plan.diagonal.size)
    np.testing.assert_allclose(M, M.conj().T, atol=1e-14, rtol=0)
    eigen = np.linalg.eigvalsh(M)
    assert eigen.min() > 0 and eigen.max() <= 1+1e-12
    # The epsilon image carries the diagonal of this operator.
    np.testing.assert_allclose(np.diag(M).real, plan.diagonal.reshape(-1), rtol=1e-12)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_off_grid_slab_nodes_take_exact_fractions_and_axis_normals(axis):
    size = [3., 3., 3.]; size[axis] = .37; center = [0., 0., 0.]; center[axis] = .013
    project = periodic_scene([Structure(kind='rectangle', material='metal', size=tuple(size), center=tuple(center))],
                             [Material(name='metal', **DRUDE)], mesh=.1, size=1.)
    plan = prepare_interfaces(project); d = plan.dispersive; r = project.region; h = r.mesh
    group = d.groups[0]
    nodes = np.stack(np.unravel_index(np.flatnonzero(d.kind == MIXED), d.node_shape), 1)
    x = r.mesh_nodes[axis][nodes[:, axis]]
    lo, hi = .013-.185, .013+.185; w = WINDOW*h
    expected = np.clip(np.minimum(x+w/2, hi)-np.maximum(x-w/2, lo), 0, w)/w
    np.testing.assert_allclose(group['fraction'], expected, rtol=1e-12)
    np.testing.assert_allclose(abs(group['normal'][:, axis]), 1)
    np.testing.assert_allclose(group['arithmetic'], 1-expected, rtol=1e-12); np.testing.assert_allclose(group['harmonic'], 1-expected, rtol=1e-12)
    assert np.all(plan.ownership.reshape(-1)[d.driven] == -1) and np.all(plan.ownership.reshape(-1)[d.standard] == 0)
    np.testing.assert_allclose(d.next_tensors, np.broadcast_to(np.eye(3), d.next_tensors.shape), atol=1e-15)
    assert set(np.unique(d.kind)) == {0, MIXED, WHOLE, NEXT}


def leapfrog_norms(project, steps, every):
    """State norm of a closed box started from a random E and H. E at the D-driven samples is not state: the first
    step assigns it from D, so its random start leaves no offset."""
    grid, plan = grid_for(project)
    rng = np.random.default_rng(3)
    for field in (grid.E, grid.H):
        field[:] = rng.normal(size=field.shape)
        if project.region.complex_fields:
            field[:] += 1j*rng.normal(size=field.shape)
    diagnostics = StateDiagnostics(grid)
    norms = []
    for n in range(steps):
        grid.update_E(); grid.update_H()
        if n % every == 0:
            norms.append(diagnostics.measure()[0])
    return np.array(norms), plan


def check_bounded(norms, collision):
    assert np.all(np.isfinite(norms))
    quarter = len(norms)//4
    if collision:
        assert norms[-quarter:].max() < norms[:quarter].max()
    else:
        assert norms[-quarter:].max() <= 1.05*norms[:quarter].max()
        assert norms.min() >= .5*norms[0]


def one_step_matrix(project):
    grid, plan = grid_for(project)
    states = [grid.E, grid.H]+[a for a in grid.memory_states if a is not grid.subpixel.curl_buffer]
    sizes = [a.size for a in states]; n = sum(sizes)
    A = np.zeros((n, n), complex if project.region.complex_fields else float)
    for j in range(n):
        vector = np.zeros(n); vector[j] = 1; k = 0
        for a, m in zip(states, sizes):
            a.reshape(-1)[:] = vector[k:k+m]; k += m
        grid.update_E(); grid.update_H()
        A[:, j] = np.concatenate([a.reshape(-1) for a in states])
    return A, plan


@pytest.mark.parametrize('bloch', [False, True])
def test_one_step_map_has_no_eigenvalue_outside_the_unit_circle(bloch):
    """The complete update (fields, CPML-free periodic box, ADE and dispersive branch states) of a damped Drude sphere
    on a 6^3 grid. A lossless Drude medium has Jordan chains at 1 (persistent currents), so its spectral radius is
    1 up to a rounding-dependent 1e-8 in staircase as well; its field norms are checked instead."""
    p = periodic_scene([Structure(kind='sphere', material='metal', radius=.07, center=(.013, -.007, .004))],
                       [Material(name='metal', **DRUDE)], mesh=.04, bloch=bloch)
    A, plan = one_step_matrix(p)
    assert plan.metadata['dispersive']['mixed_nodes'] > 0
    assert abs(np.linalg.eigvals(A)).max() <= 1+1e-12


def drude(collision, plasma=3e16):
    return Material(name='metal', model='drude', epsilon_inf=1., plasma_rad_s=plasma, collision_rad_s=collision)


ADVERSARIAL = {
    # A Drude cylinder in 2D, a slab whose faces sit a hair inside and outside node windows (fractions near 0
    # and 1) and a slab tilted by 3 degrees: with incremented E at the D-driven samples, a one-time E write there
    # grew |H| linearly in each of them.
    'cylinder_2d': lambda collision: periodic_scene([Structure(kind='circle', material='metal', radius=.13, center=(.013, -.007, 0))],
                                                    [drude(collision, 1e17)], mesh=.04, size=.48, dimension='2d'),
    'slab': lambda collision: periodic_scene([Structure(kind='rectangle', material='metal', size=(3., 3., .08+2e-9), center=(0, 0, 1e-9))],
                                             [drude(collision)], mesh=.04),
    'tilted_slab': lambda collision: periodic_scene([Structure(kind='rectangle', material='metal', size=(3., 3., .06), center=(0, 0, .01),
                                                               rotation_angles=(0., 3., 0.))], [drude(collision)], mesh=.04),
}


@pytest.mark.parametrize('name', list(ADVERSARIAL))
def test_damped_cylinder_and_slabs_have_no_eigenvalue_outside_the_unit_circle(name):
    """With damping the one-step map has no Jordan chain at 1, so its whole spectrum must lie in the unit disc. A
    lossless Drude medium keeps persistent currents (P and D grow linearly while the fields stay put; the staircase
    update of the same slab has such chains as well), so the lossless cases are checked by their field norms below."""
    A, plan = one_step_matrix(ADVERSARIAL[name](1e14))
    assert plan.metadata['dispersive']['mixed_nodes'] > 0
    # The static modes form a large cluster at 1 whose computed eigenvalues carry rounding of up to about 2e-12 on
    # this non-normal map (2D cylinder); a Jordan chain would split by about 1e-8 and an instability by more.
    assert abs(np.linalg.eigvals(A)).max() <= 1+1e-10


@pytest.mark.parametrize('name', list(ADVERSARIAL))
def test_lossless_cylinder_and_slabs_keep_a_bounded_norm_from_a_random_start(name):
    norms, _ = leapfrog_norms(ADVERSARIAL[name](0.), 3000, 25)
    check_bounded(norms, 0.)


@pytest.mark.parametrize('collision', [0., 1.5e14])
@pytest.mark.parametrize('name', list(ADVERSARIAL))
def test_an_e_write_on_driven_samples_leaves_no_static_source(name, collision):
    """E written at the D-driven samples between the E and H updates kicks H once; the next step assigns E from D
    again, so no static source remains. With incremented E such a kick grew |H| linearly (lossless) or left a
    static H about 50 times the perturbation (damped)."""
    grid, plan = grid_for(ADVERSARIAL[name](collision))
    driven = np.flatnonzero(plan.dispersive.driven)
    grid.update_E()
    grid.E.reshape(-1)[driven] += 1e-3*np.random.default_rng(5).normal(size=len(driven))
    grid.update_H()
    kick = abs(grid.H).max()
    peak = 0.
    for n in range(4000):
        grid.update_E(); grid.update_H()
        peak = max(peak, abs(grid.H).max())
    assert 0 < kick and peak <= 10*kick


@pytest.mark.parametrize('bloch', [False, True])
@pytest.mark.parametrize('collision', [0., 1.5e14])
def test_closed_box_state_norm_does_not_grow(collision, bloch):
    norms, plan = leapfrog_norms(sphere_scene(collision, dielectric=True, bloch=bloch), 3000, 25)
    assert plan.metadata['dispersive']['mixed_nodes'] > 0 and len(plan.rows) > 0
    check_bounded(norms, collision)


def test_multipole_material_closed_box_does_not_grow():
    norms, _ = leapfrog_norms(sphere_scene(poles=TWO_POLES), 3000, 25)
    check_bounded(norms, 1)


@pytest.mark.long
@pytest.mark.parametrize('collision', [0., 1.5e14])
def test_closed_box_state_norm_does_not_grow_over_200000_steps(collision):
    norms, _ = leapfrog_norms(sphere_scene(collision, dielectric=True), 200000, 500)
    check_bounded(norms, collision)


def drude_scene(backend='cpu', precision='float64', dimension='3d'):
    p = small(backend, dimension, precision)
    p.region.material_sampling = 'yee'; p.region.interface_method = 'subpixel'; p.region.steps = 160
    p.materials.append(Material(name='metal', **DRUDE))
    p.structures[0].material = 'metal'; p.structures[0].center = (.013, -.021, .007)
    if dimension == '3d':
        p.region.size = (2.4, 2.4, 2.4); p.sources[0].center = (-.6, 0, 0); p.monitors[0].center = (.6, .1, 0)
    return Project.model_validate(p.model_dump())


@pytest.mark.parametrize('dimension', ['2d', '3d'])
def test_open_scene_runs_and_reports_the_node_tensors(dimension):
    result = Simulation(drude_scene(dimension=dimension)).run()
    info = result.summary['subpixel']['dispersive']
    assert info['mixed_nodes'] > 0 and info['method'] == 'node_tensor_dispersive_laminate'
    assert np.all(np.isfinite(result.electric)) and abs(result.signals).max() > 0
    assert any('dispersive averaging tensor' in w for w in result.summary['warnings'])


def two_pole_cylinder(mesh, interface, cylinder=True):
    import test_physics_g3_b as g3
    spectrum = g3.wavelength_spectrum(.30, .42, 13)
    monitors = g3.box_monitors(.2, 'xy', spectrum)+[FieldMonitor(id='incident', name='incident', normal='x', center=(0., 0., 0.), size=(0., .1, 1.), spectrum=spectrum)]
    return Project(region=g3.region('2d', (.8, .8, 1.), mesh, .1, 'cpu', 'float64', 60., interface),
                   materials=[Material(name='metal', model='multipole', epsilon_inf=3., poles=TWO_POLES)],
                   structures=[Structure(kind='circle', radius=.05, material='metal', center=(.0013, -.0021, 0.))] if cylinder else [],
                   sources=[Source(kind='tfsf', size=(.5, .5, 1.), component='Ey', normal='x', direction='+', wavelength=.36,
                                   time_definition='standard', pulse_length=1.5e-15, pulse_offset=7e-15)],
                   monitors=monitors)


def test_two_pole_cylinder_against_the_mie_series():
    """Regression guard, not a judged case: a Drude plus Lorentz metal, TE cylinder of radius 0.05 um off the node
    lattice at h = 0.01 um, against the infinite-cylinder Mie series with the same permittivity over 0.30 to 0.42 um.
    The limits were set from one development run (largest errors 3.6 percent for scattering and 47 percent for
    absorption, against 24 and 932 percent for staircase) with a margin of about two, and require at most a third
    of the staircase error."""
    import test_physics_g3_b as g3
    from torchfdtd.materials import permittivity
    reference = Simulation(two_pole_cylinder(.01, 'staircase', False)).run()
    wavelengths, intensity = g3.incident_intensity(reference)
    eps = permittivity(two_pole_cylinder(.01, 'subpixel').materials[0], g3.C0/(wavelengths*1e-6))
    parts = [g3.cylinder_cross_sections([w], .05, complex(np.sqrt(e)), 'TE') for w, e in zip(wavelengths, eps)]
    mie = {k: np.array([q[k][0] for q in parts]) for k in ('scattering', 'absorption')}
    errors = {}
    for interface in ('staircase', 'subpixel'):
        sample = Simulation(two_pole_cylinder(.01, interface)).run()
        errors[interface] = (abs(g3.scattered_power(sample, reference, 'xy')/intensity*1e6/mie['scattering']-1).max(),
                             abs(g3.total_inward_power(sample, 'xy')/intensity*1e6/mie['absorption']-1).max())
    scattering, absorption = errors['subpixel']
    assert scattering < .1 and absorption < 1.
    assert scattering < errors['staircase'][0]/3 and absorption < errors['staircase'][1]/3


def test_unsupported_dispersive_interfaces_are_refused():
    data = drude_scene().model_dump(); data['region']['pml_dispersion'] = 'frozen'
    with pytest.raises(ValueError, match='frozen'):
        Project.model_validate(data)
    p = sphere_scene()
    p.materials.append(Material(name='gold', **dict(DRUDE, plasma_rad_s=1.2e16)))
    p.structures.append(Structure(kind='sphere', material='gold', radius=.03, center=(.013, -.007, .09)))
    with pytest.raises(ValueError, match='one dispersive material per node cell'):
        prepare_interfaces(p)
    p = drude_scene(); p.structures[0].center = (0, 0, 0); p.structures[0].radius = 1.25
    with pytest.raises(ValueError, match='nonperiodic grid boundary'):
        prepare_interfaces(p)
    p = drude_scene(); p.sources[0].center = (.013+.5, -.021, .007)
    with pytest.raises(ValueError, match='overlaps node cells cut by a dispersive surface'):
        Simulation(p).run()
    with pytest.raises(ValueError, match='staircase'):
        DifferentiableSimulation(drude_scene())


def film_scene(source):
    """A 1.5-cell Drude film with eps_inf = 1 crossing the source plane: the permittivity image equals the background."""
    periodic = {} if source.kind == 'tfsf' else dict(boundaries=dict(y_min=dict(kind='periodic'), y_max=dict(kind='periodic')))
    r = Region(size=(1.2, 1.2, 1.), dimension='2d', mesh=.01, pml_cells=12, backend='cpu', precision='float64', steps=20,
               material_sampling='yee', interface_method='subpixel', **periodic)
    return Project(region=r, materials=[Material(name='metal', model='drude', epsilon_inf=1., plasma_rad_s=1.37e16, collision_rad_s=1.5e14)],
                   structures=[Structure(kind='rectangle', size=(.9, .015, 1.), center=(0., .0132, 0.), material='metal')],
                   sources=[source], monitors=[Monitor(id='e', name='e', center=(.45, .2, 0), component='Ez')])


def test_driven_samples_in_tfsf_and_oneway_neighbourhoods_are_refused():
    tfsf = Source(kind='tfsf', size=(.6, .6, 1.), component='Ez', normal='x', direction='+', wavelength=.5,
                  time_definition='standard', pulse_length=2e-15, pulse_offset=8e-15)
    with pytest.raises(ValueError, match='dispersive material cannot intersect the TFSF face neighborhood'):
        Simulation(film_scene(tfsf)).run()
    oneway = Source(kind='plane', injection='oneway', normal='x', center=(-.3, 0, 0), size=(0, 1.2, 0), component='Ez', wavelength=.5)
    with pytest.raises(ValueError, match='dispersive materials cannot intersect the injection neighborhood'):
        Simulation(film_scene(oneway)).run()
    p = film_scene(tfsf); p.structures[0].size = (.5, .015, 1.)
    assert Simulation(p).run().summary['subpixel']['dispersive']['driven_samples'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_cpu_cuda_fused_and_tensor_cohorts_agree(precision):
    pytest.importorskip('cupy')
    p = drude_scene(precision=precision)
    cpu = Simulation(p).run()
    p.region.backend = 'cuda'; p.region.cuda_kernel = 'torch'
    reference = Simulation(p).run(cuda_graph=False)
    p.region.cuda_kernel = 'fused'
    fused = Simulation(p).run()
    shifted = p.model_copy(deep=True); shifted.structures[0].center = (.05, .03, 0)
    batch = run_tensor_batch([p, shifted]); batch.raise_for_errors()
    tol = 3e-5 if precision == 'float32' else 2e-12
    peak = abs(cpu.electric).max()
    for actual in (reference, fused):
        np.testing.assert_allclose(actual.electric, cpu.electric, rtol=0, atol=tol*peak)
        np.testing.assert_allclose(actual.signals, cpu.signals, rtol=0, atol=tol*abs(cpu.signals).max())
    for item, expected in zip(batch.items, (fused, Simulation(shifted).run())):
        np.testing.assert_array_equal(item.result.electric, expected.electric)
        np.testing.assert_array_equal(item.result.signals, expected.signals)
