"""Every statement of docs/CONVENTIONS.md is fixed here by a computation on a tiny problem (G2-05)."""
import math

import numpy as np
import pytest
import torch

from torchfdtd import (Project, Region, Source, Monitor, FieldMonitor, Material, BoundaryFace, SpectrumSettings,
                       Simulation, DifferentiablePlaneSimulation)
from torchfdtd.spectra import direct_transform, point_spectrum
from torchfdtd.differentiable import DifferentiableResult, _System
from torchfdtd.field_monitors import plane_result, plane_plan, interpolation_map
from torchfdtd.boundaries import BoundaryDescription, YeeGrid
from torchfdtd.materials import permittivity
from torchfdtd.solver import source_profile, source_slice, field_axes
from torchfdtd.plan import resolve_plan, SAMPLE_TIME_STEPS, FOURIER_CONVENTION
from torchfdtd.radiation import _passive, project_farfield
from test_plan import scene

C0 = 299792458.0


# ---- DFT sign and normalization ----------------------------------------------------------------

def test_native_point_and_plane_dft_use_the_positive_sign_with_dt_normalization():
    dt, n = 1e-16, 400
    f0 = 10/(n*dt)  # ten whole cycles in the window, so the wrong-sign sums vanish exactly
    t = np.arange(1, n+1)*dt
    # A complex exponential exp(-2 pi i f0 t) transforms to n*dt at +f0 only under exp(+2 pi i f t).
    value = direct_transform(t, np.exp(-2j*np.pi*f0*t), np.array([f0]))
    assert abs(value[0] - n*dt) < 1e-12*n*dt
    assert abs(direct_transform(t, np.exp(+2j*np.pi*f0*t), np.array([f0]))[0]) < 1e-9*n*dt
    # sin(omega t) = cos(omega t - pi/2) carries phase +pi/2 under the positive sign ...
    settings = SpectrumSettings(sampling='custom', custom_frequencies_hz=[f0], apodization='none')
    native = point_spectrum(t, np.sin(2*np.pi*f0*t), settings)
    assert native['transform'] == 'dt * sum(field * window * exp(+2pi i f t))' and native['units'] == 'reduced field * s'
    assert abs(np.angle(native['value'][0]) - np.pi/2) < 1e-9
    assert abs(abs(native['value'][0]) - n*dt/2) < 1e-9*n*dt/2
    # ... and -pi/2 under the legacy FFT path, whose sign is negative and whose scale is 2/N for a real trace.
    legacy = point_spectrum(t, np.sin(2*np.pi*f0*t), SpectrumSettings(sampling='fft', apodization='none'))
    k = int(np.argmin(abs(legacy['frequency_hz'] - f0)))
    assert legacy['transform'].startswith('FFT exp(-2pi i f t); scale 2/N real') and legacy['units'] == 'reduced field'
    assert abs(np.angle(legacy['value'][k]) + np.pi/2) < 1e-9 and abs(abs(legacy['value'][k]) - 1) < 1e-9
    assert FOURIER_CONVENTION == 'exp(+2 pi i f t)'


def test_differentiable_point_spectrum_uses_the_negative_sign_and_the_half_step_for_h():
    dt, n, f0 = 1e-16, 400, 8e13
    t = torch.arange(1, n+1, dtype=torch.float64)*dt
    signals = torch.stack([torch.exp(2j*np.pi*f0*t), torch.exp(2j*np.pi*f0*(t+dt/2))], dim=1)
    result = DifferentiableResult(signals, dt, ('Ez', 'Hy'))
    value = result.spectrum([f0])[0]
    # exp(+2 pi i f0 t) transforms to n*dt only under exp(-2 pi i f t); the H sample time is t + dt/2.
    torch.testing.assert_close(value[0], torch.tensor(n*dt, dtype=torch.complex128), rtol=1e-12, atol=0)
    torch.testing.assert_close(value[1], torch.tensor(n*dt, dtype=torch.complex128), rtol=1e-12, atol=0)


def plane_reference(p, epsilon, frequency):
    """The documented plane DFT: dt * sum_n F(t_n) exp(+2 pi i f t_n), t_n = (n+1) dt for E and (n+1.5) dt for H."""
    system = _System(p, epsilon)
    state = system.state()
    monitor = p.resolved_monitor(p.monitors[0])
    plan = plane_plan(p.region, monitor)
    total = 0
    for step in range(p.region.steps):
        state = system.reference_step(state, step, epsilon)
        columns = []
        for c in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            indices, weights = interpolation_map(p.region, c, plan['points_um'])
            spatial = (state[0 if c[0] == 'E' else 1].reshape(-1)[torch.tensor(indices)]*torch.tensor(weights)).sum(0)
            t = (step+SAMPLE_TIME_STEPS[c[0]])*p.region.time_step
            columns.append(np.exp(2j*np.pi*frequency*t)*spatial.numpy()*p.region.time_step)
        total = total+np.stack(columns, axis=-1)
    return total


def test_native_and_differentiable_planes_match_the_documented_dft_and_sample_times():
    p = scene(monitors='field', precision='float64', steps=11)
    p.structures = []
    p.monitors[0].spectrum = SpectrumSettings(sampling='custom', custom_frequencies_hz=[2e14], apodization='none')
    p = Project.model_validate(p.model_dump())
    epsilon = torch.ones(p.region.shape + (3,), dtype=torch.float64)
    expected = plane_reference(p, epsilon, 2e14)
    assert np.abs(expected).max() > 0
    differentiable = DifferentiablePlaneSimulation(p)(epsilon, [2e14])['plane'].fields[0].numpy()
    np.testing.assert_allclose(differentiable, expected, rtol=1e-10, atol=1e-14*np.abs(expected).max())
    native = Simulation(p).run().frequency_fields[0]['fields'][0]
    np.testing.assert_allclose(native, expected, rtol=1e-10, atol=1e-14*np.abs(expected).max())


# ---- Bloch spatial phase -------------------------------------------------------------------------

def test_bloch_field_is_multiplied_by_exp_plus_i_phase_per_positive_period():
    phase = .7
    p = scene(precision='float64')
    p.region.boundaries.y_min = p.region.boundaries.y_max = BoundaryFace(kind='bloch')
    p.region.bloch_phase = (0, phase, 0)
    p.sources = [Source(kind='plane', normal='x', center=(-.2, 0, 0), size=(0, 1.5, 0), component='Ez')]
    p = Project.model_validate(p.model_dump())
    r = p.region
    assert BoundaryDescription(r).wrap[1] == np.exp(1j*phase)
    # The sheet profile is exp(+i phase (y - y0) / L): the injected field obeys F(y + L) = exp(+i phase) F(y).
    scalar = p.sources[0]
    profile = source_profile(scalar, source_slice(scalar, r), r)
    y = field_axes(r, 'Ez')[1]
    np.testing.assert_allclose(profile.reshape(-1), np.exp(1j*phase*(y-scalar.center[1])/r.actual_size[1]), rtol=1e-12, atol=0)
    # Interpolating above the last sample wraps to the first one multiplied by exp(+i phase).
    top = y[-1]+.25*(y[1]-y[0])
    indices, weights = interpolation_map(r, 'Ez', np.array([[0., top, 0.]]))
    wrapped = weights[np.flatnonzero(indices % (3*r.shape[2]*r.shape[1]) // 3 == 0)]
    np.testing.assert_allclose(wrapped[np.abs(wrapped) > 0], .25*np.exp(1j*phase), rtol=1e-12, atol=0)
    # Below the first sample it wraps to the last one multiplied by exp(-i phase).
    bottom = y[0]-.25*(y[1]-y[0])
    indices, weights = interpolation_map(r, 'Ez', np.array([[0., bottom, 0.]]))
    wrapped = weights[np.flatnonzero(indices % (3*r.shape[2]*r.shape[1]) // 3 == r.shape[1]-1)]
    np.testing.assert_allclose(wrapped[np.abs(wrapped) > 0], .25*np.exp(-1j*phase), rtol=1e-12, atol=0)
    # The curl seam uses the same phase: on a Bloch function exp(i k y), k = (phase + 2 pi m) / L, the
    # forward difference at the last cell equals the interior formula with F(y_n) = exp(+i phase) F(y_0).
    g = YeeGrid(r)
    k = (phase+2*np.pi)/r.actual_size[1]
    g.E[:] = 0
    g.E[:, :, :, 2] = np.exp(1j*k*y)[None, :, None]
    curl = g.curl(g.E, True)
    wrapped_end = np.r_[np.exp(1j*k*y[1:]), np.exp(1j*phase)*np.exp(1j*k*y[0])]
    expected = (wrapped_end-np.exp(1j*k*y))
    np.testing.assert_allclose(curl[2, :, 0, 0], expected, rtol=1e-12, atol=1e-14)
    # The backward (E-derivative) seam uses exp(-i phase) F[n-1] before the first cell.
    g.H[:] = 0
    g.H[:, :, :, 2] = np.exp(1j*k*y)[None, :, None]
    backward = g.curl(g.H, False)
    wrapped_start = np.r_[np.exp(-1j*phase)*np.exp(1j*k*y[-1]), np.exp(1j*k*y[:-1])]
    np.testing.assert_allclose(backward[2, :, 0, 0], np.exp(1j*k*y)-wrapped_start, rtol=1e-12, atol=1e-14)


# ---- E/H half step and sample times ---------------------------------------------------------------

def test_e_is_tagged_at_n_plus_one_dt_and_h_half_a_step_later():
    p = scene(precision='float64', steps=40)
    p.sources = [Source(center=(-.2, 0, 0), pulse='continuous', wavelength=1.1),
                 Source(center=(-.2, .3, 0), pulse='continuous', wavelength=1.1, component='Hy')]
    p.monitors = [Monitor(id='ez', component='Ez', center=(.1, 0, 0)), Monitor(id='hy', component='Hy', center=(.1, 0, 0))]
    p = Project.model_validate(p.model_dump())
    dt = p.region.time_step
    result = Simulation(p).run()
    np.testing.assert_array_equal(result.times, np.arange(1, 41)*dt)
    spectrum = result.point_monitors[1].spectrum
    np.testing.assert_array_equal(result.spectra[1]['value'], point_spectrum(result.times+dt/2, result.signals[:, 1], spectrum)['value'])
    np.testing.assert_array_equal(result.spectra[0]['value'], point_spectrum(result.times, result.signals[:, 0], spectrum)['value'])
    assert SAMPLE_TIME_STEPS == {'E': 1., 'H': 1.5}
    # Sources are sampled at their injection times: E terms at (n+1) dt, H terms at (n+1.5) dt.
    plan = resolve_plan(p)
    from torchfdtd.waveforms import source_time_signal
    e_term, h_term = plan.sources[0].terms[0], plan.sources[1].terms[0]
    np.testing.assert_array_equal(e_term.sample_times, np.arange(1, 41)*dt)
    np.testing.assert_array_equal(h_term.sample_times, np.arange(1, 41)*dt+.5*dt)
    np.testing.assert_array_equal(e_term.samples, source_time_signal(p.sources[0], np.arange(1, 41)*dt))
    np.testing.assert_array_equal(h_term.samples, source_time_signal(p.sources[1], np.arange(1, 41)*dt+.5*dt))
    assert p.sources[1].time_offset_steps == .5 and p.sources[0].time_offset_steps == 0.


# ---- Monitor normal, flux sign, reduced units and 2D power per unit length ----------------------------

def test_flux_density_is_half_re_e_cross_h_conjugate_along_the_positive_normal():
    monitor = FieldMonitor(normal='x', size=(0, 1, 1), record_fields=('Ey', 'Hz'), record_poynting=('x', 'y', 'z'))
    plan = dict(points_um=np.zeros((1, 3)), weights=np.array([2e-12]), shape=(1, 1, 1), normal=0)
    components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
    def result(ex, ey, ez, hx, hy, hz, dimension='3d'):
        return plane_result(monitor, plan, np.array([1e14]), components, np.array([[[ex, ey, ez, hx, hy, hz]]], dtype=complex), dimension)
    forward = result(0, 1, 0, 0, 0, 1)
    backward = result(0, 1, 0, 0, 0, -1)
    # E = y, H = z: E x H = +x, density .5, flux .5 * area; reversing H reverses the sign.
    np.testing.assert_allclose(forward['poynting'][0, 0, 0], .5)
    np.testing.assert_allclose(forward['flux'], [1e-12])
    np.testing.assert_allclose(backward['flux'], [-1e-12])
    # The stored components follow the same rule: z x x = +y and x x y = +z.
    np.testing.assert_allclose(result(0, 0, 1, 1, 0, 0)['poynting'][0, 0], [0., .5, 0.])
    np.testing.assert_allclose(result(1, 0, 0, 0, 1, 0)['poynting'][0, 0], [0., 0., .5])
    np.testing.assert_allclose(result(0, 1j, 0, 0, 0, 1j)['poynting'][0, 0], [.5, 0., 0.])  # E H* keeps a common phase out
    assert forward['flux_units'] == 'reduced E*H * s^2 * m^2' and forward['field_units'] == 'reduced field * s'
    assert result(0, 1, 0, 0, 0, 1, '2d')['flux_units'] == 'reduced E*H * s^2 * m per invariant length'


def sheet_project(source_x, monitors):
    r = Region(dimension='2d', size=(3, .6, 1), mesh=.05, steps=800, pml_cells=8, backend='cpu',
               material_sampling='yee', precision='float64',
               boundaries=dict(y_min=dict(kind='periodic'), y_max=dict(kind='periodic')))
    return Project(region=r, sources=[Source(kind='plane', normal='x', center=(source_x, 0, 0), size=(0, .6, 0),
                                             component='Ez', pulse='continuous', wavelength=1.1)], monitors=monitors)


def windowed(center_um, width_um, identifier):
    spectrum = SpectrumSettings(sampling='custom', custom_frequencies_hz=[C0/1.1e-6], apodization='full',
                                apodization_center=60e-15, apodization_time_width=20e-15)
    return FieldMonitor(id=identifier, center=(center_um, 0, 0), size=(0, width_um, 1), normal='x', spectrum=spectrum)


def test_two_dimensional_flux_is_signed_power_per_unit_length_with_unit_reduced_impedance():
    half, full = windowed(.5, .3, 'half'), windowed(.5, .6, 'full')
    result = Simulation(sheet_project(-1., [half, full])).run()
    planes = {m['id']: m for m in result.frequency_fields}
    # Quadrature weights are transverse widths in metres per unit invariant length, not areas.
    assert math.isclose(planes['half']['weights'].sum(), .3e-6, rel_tol=1e-9)
    assert math.isclose(planes['full']['weights'].sum(), .6e-6, rel_tol=1e-9)
    assert planes['half']['flux_units'] == 'reduced E*H * s^2 * m per invariant length'
    # The wave is uniform in y: the power through the whole period is twice the power through half of it.
    flux_half, flux_full = planes['half']['flux'][0], planes['full']['flux'][0]
    assert flux_full > 0 and math.isclose(flux_full, 2*flux_half, rel_tol=1e-6)
    # Reduced units: eta_0 = 1, so |H_y| = |E_z| for the plane wave and flux = .5 |E_z|^2 * width.
    fields = planes['full']['fields'][0]
    names = planes['full']['components']
    ez, hy = fields[:, names.index('Ez')], fields[:, names.index('Hy')]
    np.testing.assert_allclose(np.abs(hy)/np.abs(ez), 1., rtol=2e-2)
    sheet_power = .5*(np.abs(ez)**2*planes['full']['weights']).sum()
    assert math.isclose(flux_full, sheet_power, rel_tol=2e-2)
    # A source on the far side sends the wave toward -x through the same monitor: negative flux.
    reversed_run = Simulation(sheet_project(1.0, [windowed(.5, .6, 'full')])).run()
    reversed_flux = reversed_run.frequency_fields[0]['flux'][0]
    assert reversed_flux < 0 and math.isclose(-reversed_flux, flux_full, rel_tol=5e-2)


def test_reduced_units_and_si_calibration():
    p = scene(monitors='field', precision='float64')
    r = p.region
    # Time is SI seconds from the SI speed of light and micrometre mesh steps.
    assert math.isclose(r.time_step, r.courant_factor/math.sqrt(2)*r.mesh*1e-6/C0, rel_tol=1e-15)
    plan = resolve_plan(p)
    # Plane quadrature weights are SI areas (m^2) in 3D.
    box = Project(region=Region(dimension='3d', size=(1.6, 1.6, 1.6), mesh=.1, pml_cells=3, steps=10, backend='cpu'),
                  sources=[Source(center=(0, 0, 0))], monitors=[FieldMonitor(center=(.3, 0, 0), size=(0, 1, .8), normal='x')])
    assert math.isclose(resolve_plan(box).monitors[0].weights.sum(), .8e-12, rel_tol=1e-9)
    # Material rates are SI angular frequencies, permittivity is relative and the ADE coefficient .5 (w0 dt)^2 is dimensionless.
    drude = Material(name='drude', model='drude', plasma_rad_s=1.3e16, collision_rad_s=1e14, epsilon_inf=1.5)
    f = 2e14
    omega = 2*np.pi*f
    expected = 1.5-1.3e16**2/(omega**2+1j*1e14*omega)
    np.testing.assert_allclose(permittivity(drude, f), expected, rtol=1e-12)
    assert expected.imag > 0  # passive under exp(-i omega t)
    lorentz = Material(name='lorentz', model='lorentz', resonance_rad_s=2e15, linewidth_rad_s=1e14, delta_epsilon=1, epsilon_inf=2)
    q = p.model_copy(deep=True); q.materials.append(lorentz); q.structures[0].material = 'lorentz'
    ade = resolve_plan(Project.model_validate(q.model_dump())).ade[0]
    np.testing.assert_allclose(ade.a, [.5*(2e15*r.time_step)**2], rtol=1e-12)
    # Fields, source amplitudes and fluxes are reduced: the summary and the results say so.
    result = Simulation(scene()).run()
    assert result.summary['units'] == 'geometry: um; time: s; E/H: reduced fields; Bloch phase: rad'
    assert result.spectra[0]['units'] == 'reduced field'  # legacy FFT bins of the default point monitor
    assert plan.monitors[0].weights.sum() > 0 and plan.time_step == r.time_step


# ---- Lossy exterior of the radiation transforms ---------------------------------------------------------

def test_lossy_exterior_uses_exp_minus_i_omega_t_phasors_with_decaying_exp_ikr():
    from test_radiation import dipole_faces
    n = 1.3+.05j
    assert _passive(n, 'n') == n and _passive(1.3, 'n') == 1.3
    with pytest.raises(ValueError, match='growing'):
        _passive(1.3-.05j, 'n')
    faces, bounds, _ = dipole_faces(14, n=n)
    directions = torch.tensor([[0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    directions = directions/directions.norm(dim=-1, keepdim=True)
    far = project_farfield(faces, directions, bounds_um=bounds, refractive_index=n)
    r1, r2 = 50e-6, 60e-6
    near_field, far_field = far.fields_at_radius(r1)[..., :3], far.fields_at_radius(r2)[..., :3]
    imaginary_k = 2*math.pi*n.imag/1.55e-6
    ratio = far_field.abs().sum(-1)/near_field.abs().sum(-1)
    torch.testing.assert_close(ratio, torch.full_like(ratio, (r1/r2)*math.exp(-imaginary_k*(r2-r1))), rtol=1e-6, atol=0)
    assert ratio.max() < r1/r2
