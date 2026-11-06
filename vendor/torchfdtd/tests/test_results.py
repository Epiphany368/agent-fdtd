"""G6-03: the integrated results API read from the public outputs of the existing fixtures.

The slab flux fixture of tests/test_field_monitors.py gives R, T and the
balance against its matched reference run, and, with a lossy slab and a closed
box of flux monitors, an absorption that is measured rather than inferred. The
mode-port fixtures of tests/test_mode_branches.py give complex S with phase and
group delay (a three-wavelength sweep of the straight guide against the mode
solver's d beta / d omega) and the Y branch at one carrier; the slab guide of
benchmarks/mode_injection.py gives the mode decomposition; the G3-08 grating
geometry at a coarse mesh gives the diffraction orders; the analytic dipole
faces of tests/test_radiation.py give the far-field and near-zone records.
"""
import math

import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions, FieldMonitor, LorentzPole, Material, Simulation, SpectrumSettings
from torchfdtd.mode_branches import ModeBranchNetwork
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, prepare_modal_launch, _profile
from torchfdtd.mode_ports import solve_waveguide_modes
from torchfdtd.results import (RATIO, ResultRecord, diffraction_record, farfield_record, mode_decomposition, nearzone_record,
                               reflection_transmission, s_parameters)
from torchfdtd.solver import C0
from benchmarks.mode_injection import make_case
from test_field_monitors import slab_project
from test_mode_branches import (aperture_ports, branch_project, guide, straight_project, ybranch_epsilon, ybranch_ports,
                                ybranch_sections)
from test_physics_g3_b import grating_project, lift_line
from test_radiation import dipole_faces, plane, rayleigh, spherical_directions, spherical_points


def box_monitors():
    """Closed box through the reflection and transmission monitors of the slab fixture; the y faces of the
    y-uniform scene carry no flux but the closure is still formed from four recorded faces."""
    spectrum = SpectrumSettings(sampling='frequency', frequency_points=31, apodization='none')
    return [FieldMonitor(id='y_min', normal='y', center=(0, -.2, 0), size=(1.6, 0, 1), spectrum=spectrum),
            FieldMonitor(id='y_max', normal='y', center=(0, .2, 0), size=(1.6, 0, 1), spectrum=spectrum)]


@pytest.fixture(scope='module')
def slab_runs():
    # One monitor set for the three runs: the run signature of a matched reference includes the monitors.
    p = slab_project()
    p.materials[1].index = 1.5
    p.monitors += box_monitors()
    sample = Simulation(p).run()
    empty = p.model_copy(deep=True)
    empty.structures = []
    reference = Simulation(empty).run()
    lossy = p.model_copy(deep=True)
    lossy.materials.append(Material(name='lossy', model='multipole', epsilon_inf=2.25,
                                    poles=[LorentzPole(resonance_rad_s=1.2e15, strength_rad_s_squared=4e29, damping_rad_s=8e14)]))
    lossy.structures[0].material = 'lossy'
    absorbing = Simulation(lossy).run()
    return sample, reference, absorbing


def fabry_perot_transmission(frequency_hz):
    wavelength = C0/frequency_hz*1e6
    return 1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)


def test_slab_reflection_transmission_against_the_reference_run(slab_runs):
    sample, reference, _ = slab_runs
    record = reflection_transmission(sample, reference, reflection='reflection', transmission='transmission', direction='+')
    assert isinstance(record, ResultRecord) and record.kind == 'reflection_transmission'
    expected = fabry_perot_transmission(record.frequency_hz)
    assert record.valid['T'].all() and record.valid['R'].all()
    np.testing.assert_allclose(record['T'], expected, atol=.004)
    np.testing.assert_allclose(record['R'], 1-expected, atol=.004)
    np.testing.assert_allclose(record['balance'], 0, atol=.004)
    np.testing.assert_allclose(record['T_signed'], record['T'])
    for name in ('R', 'T', 'balance'):
        assert record.units[name] == 'dimensionless' and record.calibration[name] == RATIO
    assert record.units['sample_flux_T'] == 'reduced E*H * s^2 * m per invariant length' and record.calibration['sample_flux_T'] == 'reduced'
    assert not record.valid['A'].any() and record.reason('A', 0).startswith('not measured')
    assert np.isnan(record['A']).all() and 'not an absorption measurement' in record.definitions['balance']
    assert 'reference run flux is the denominator' in record.notes[0]
    payload = record.as_dict()
    assert payload['reasons']['A'][0].startswith('not measured') and payload['units']['T'] == 'dimensionless'
    # The same record from the raw monitor mappings.
    mapping = reflection_transmission({m['id']: m for m in sample.frequency_fields}, {m['id']: m for m in reference.frequency_fields},
                                      reflection='reflection', transmission='transmission')
    np.testing.assert_array_equal(mapping['T'], record['T'])
    with pytest.raises(ValueError, match="direction must be"):
        reflection_transmission(sample, reference, reflection='reflection', transmission='transmission', direction='forward')


def test_wrong_incident_direction_and_reversed_flux_are_backflow_not_a_sign_fix(slab_runs):
    sample, reference, _ = slab_runs
    # The declared direction contradicts the reference run: every ratio is refused as reference backflow.
    record = reflection_transmission(sample, reference, reflection='reflection', transmission='transmission', direction='-')
    assert not record.valid['T'].any() and not record.valid['R'].any()
    assert all(r.startswith('reference backflow') for r in record.reasons['T']) and all(r.startswith('reference backflow') for r in record.reasons['R'])
    assert np.isnan(record['T']).all() and np.isnan(record['R']).all() and np.isnan(record['balance']).all()
    assert (record['T_signed'] < 0).all()
    # A transmission monitor whose net flux runs back toward the source (injected by negating the record):
    # T is NaN with the backflow reason, the signed ratio keeps the value, R is untouched.
    flipped = {m['id']: (dict(m, flux=-m['flux']) if m['id'] == 'transmission' else m) for m in sample.frequency_fields}
    record = reflection_transmission(flipped, {m['id']: m for m in reference.frequency_fields}, reflection='reflection', transmission='transmission')
    assert not record.valid['T'].any() and all(r.startswith('backflow: net flux against the propagation direction') for r in record.reasons['T'])
    assert np.isnan(record['T']).all() and (record['T_signed'] < 0).all() and record.valid['R'].all()
    assert not record.valid['balance'].any() and record.reason('balance', 0) == 'R or T is not supported at this frequency'


def test_absorption_is_measured_from_a_closed_box_and_agrees_with_the_balance(slab_runs):
    _, reference, absorbing = slab_runs
    faces = {'x_min': absorbing.field_monitor('reflection'), 'x_max': absorbing.field_monitor('transmission'),
             'y_min': absorbing.field_monitor('y_min'), 'y_max': absorbing.field_monitor('y_max')}
    record = reflection_transmission(absorbing, reference, reflection='reflection', transmission='transmission', absorption=faces)
    assert record.valid['A'].all() and record.calibration['A'] == RATIO
    assert (record['A'] > .02).all()
    np.testing.assert_allclose(record['A'], record['balance'], atol=.01)
    assert 'closed box' in record.definitions['A'] and record.units['box_outward_flux'] == record.units['sample_flux_T']
    with pytest.raises(ValueError, match='closed box faces'):
        reflection_transmission(absorbing, reference, reflection='reflection', transmission='transmission',
                                absorption={k: v for k, v in faces.items() if k != 'y_max'})
    with pytest.raises(ValueError, match='must be y-normal'):
        reflection_transmission(absorbing, reference, reflection='reflection', transmission='transmission',
                                absorption=dict(faces, y_max=faces['x_max']))


def test_straight_guide_group_delay_matches_the_mode_solver_dispersion():
    wavelengths = (1.5, 1.55, 1.6)
    values, betas = [], []
    for wavelength in wavelengths:
        p = straight_project(steps=600)
        p.sources[0].wavelength = wavelength
        network = ModeBranchNetwork(p, aperture_ports(), guide, AdjointOptions(checkpoints=4))
        with torch.no_grad():
            values.append(network(network.reference_epsilon(port='in', device='cpu')).s.numpy())
        # The fundamental mode of the same periodic 2.4 x 0.5 um cell from the public mode solver.
        betas.append(solve_waveguide_modes(guide, shape=(24, 5), spacing_um=(.1, .1), wavelength_um=wavelength, num_modes=1)[0].beta_per_um)
    frequency = np.array([C0/(w*1e-6) for w in wavelengths[::-1]])
    s = np.stack(values[::-1])
    record = s_parameters(frequency, s, channels=('in', 'out'))
    assert record.kind == 's_parameters' and record.units['group_delay_s'] == 's' and record.units['phase_rad'] == 'rad'
    assert record.calibration['S'] == 'dimensionless ratio of mode amplitudes'
    assert record.valid['phase_rad'][:, 1, 0].all() and record.valid['group_delay_s'][:, 1, 0].all()
    # Ports 2 um apart: the group delay is 2 um * d beta_yee / d omega, where beta_yee = (2/h) asin(beta h / 2) is
    # the longitudinal Yee wavenumber whose discrete dispersion the launch realizes (docs/MODE_INJECTION.md); at
    # 0.1 um (about 8 cells per guided wavelength) it exceeds the continuum 2 um * d beta / d omega by about 27 percent.
    omega = 2*math.pi*frequency
    beta = np.array(betas[::-1])
    beta_yee = 2/.1*np.arcsin(beta*.1/2)*1e6
    expected = 2e-6*(beta_yee[2]-beta_yee[0])/(omega[2]-omega[0])
    continuum = 2e-6*(beta[2]-beta[0])*1e6/(omega[2]-omega[0])
    measured = record['group_delay_s'][1, 1, 0]
    assert measured > 0 and abs(measured-expected)/expected < .03, (measured, expected, continuum)
    assert record['group_delay_s'][1, 0, 1] == pytest.approx(measured, rel=1e-2)
    assert 1.2 < measured/continuum < 1.35
    # The reflection channels are at the rounding floor: no phase, no group delay, with the reason.
    assert not record.valid['phase_rad'][:, 0, 0].any()
    assert record.reason('phase_rad', 1, 0, 0).startswith('|S| at or below the magnitude floor')
    assert record.reason('group_delay_s', 1, 0, 0).startswith('|S| at or below the magnitude floor')
    print({'group_delay_fdtd_s': float(measured), 'group_delay_yee_dispersion_s': float(expected), 'group_delay_continuum_s': float(continuum),
           'beta_per_um': beta.tolist()})


def test_y_branch_single_carrier_s_record():
    p = branch_project()
    network = ModeBranchNetwork(p, ybranch_ports(), options=AdjointOptions(checkpoints=4), port_permittivities=ybranch_sections())
    with torch.no_grad():
        result = network(ybranch_epsilon(p.region, 'cpu'))
    record = s_parameters([C0/(p.sources[0].wavelength*1e-6)], result.s[None], channels=result.channels)
    assert record['S'].shape == (1, 3, 3) and record['magnitude'][0, 1, 0] == pytest.approx(abs(result.s[1, 0].item()))
    assert record['magnitude'][0, 1, 0] == pytest.approx(record['magnitude'][0, 2, 0], abs=1e-3)
    assert record.valid['phase_rad'][0, 1, 0] and record['phase_rad'][0, 1, 0] == pytest.approx(math.atan2(result.s[1, 0].imag.item(), result.s[1, 0].real.item()))
    assert not record.valid['group_delay_s'].any()
    assert record.reason('group_delay_s', 0, 1, 0) == 'group delay needs at least three frequencies'
    assert "channels: (('in', 0), ('up', 0), ('down', 0))" in record.notes[1]


def test_slab_guide_mode_decomposition_on_the_far_plane():
    p, epsilon = make_case('slab', steps=600)
    launch = prepare_modal_launch(p, epsilon)
    model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=4))
    section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
    base = torch.tensor(np.broadcast_to(section, p.region.shape+(3,)).copy())
    with torch.no_grad():
        planes = model(base, [C0/1.55e-6])
    record = mode_decomposition(planes['far'], [launch], names=['TE0'])
    assert record.kind == 'mode_decomposition' and record['forward'].shape == (1, 1)
    assert record.units['forward'].startswith('reduced field * s') and record.calibration['forward'] == 'reduced'
    assert record.units['plane_flux'] == planes['far'].flux_units
    assert record.valid['forward_fraction'][0, 0] and .9 < record['forward_fraction'][0, 0] < 1.15
    assert record['backward_fraction'][0, 0] < .02 or record.reason('backward_fraction', 0, 0).startswith('weak mode')
    assert record.definitions['forward_fraction'].startswith('|forward amplitude|^2 / |plane flux|')
    with pytest.raises(ValueError, match='at least one mode launch'):
        mode_decomposition(planes['far'], [])


@pytest.fixture(scope='module')
def coarse_grating():
    """The G3-08 geometry at 0.03 um and 150 fs: the interface test, not the RCWA comparison."""
    sample = Simulation(grating_project(.03, 'TE', 20., 1.02, duration_fs=150)).run()
    reference = Simulation(grating_project(.03, 'TE', 20., 1.02, duration_fs=150, grating=False)).run()
    return sample, reference


def test_grating_diffraction_record_flags_evanescent_orders_and_balances(coarse_grating):
    sample, reference = coarse_grating
    period, kx = 1.2, 2*math.pi/1.02*math.sin(math.radians(20.))
    orders = [(0, m) for m in (-2, -1, 0, 1, 2)]
    kwargs = dict(period_um=(1., period), bloch_wavevector_per_um=(0., kx))
    # The matched reference of each line is the same line of the empty run: its flux is the incident power, and on
    # the reflection line its fields are the incident field that subtract_incident removes.
    transmitted = diffraction_record(lift_line(sample.field_monitor('transmission')), lift_line(reference.field_monitor('transmission')), orders, **kwargs)
    reflected = diffraction_record(lift_line(sample.field_monitor('reflection')), lift_line(reference.field_monitor('reflection')), orders,
                            direction='backward', subtract_incident=True, **kwargs)
    for record in (transmitted, reflected):
        assert record.kind == 'diffraction' and record.units['efficiency'] == 'dimensionless' and record.calibration['efficiency'] == RATIO
        assert record.units['order_power'] == 'reduced E*H * s^2 * m^2' and record.calibration['wavevector_per_um'] == 'SI length'
        assert record['orders'].tolist() == [list(o) for o in orders]
        # At 1.02 um and 20 degrees only m = -1, 0 propagate in vacuum with period 1.2 um.
        assert record['propagating'].tolist() == [[False, True, True, False, False]]*3
        for k in (0, 3, 4):
            assert not record.valid['efficiency'][:, k].any() and record.reason('efficiency', 0, k).startswith('evanescent order')
            assert np.isnan(record['efficiency'][:, k]).all() and (record['order_power'][:, k] == 0).all()
        assert record.valid['efficiency'][:, 1:3].all()
    total = np.nansum(transmitted['efficiency'], axis=1)+np.nansum(reflected['efficiency'], axis=1)
    np.testing.assert_allclose(total, 1, atol=.05)
    assert 'evanescent order' in transmitted.reason('efficiency', 0, 0)
    with pytest.raises(ValueError, match='Direction must be'):
        diffraction_record(lift_line(sample.field_monitor('transmission')), lift_line(reference.field_monitor('transmission')), orders, direction='up', **kwargs)
    with pytest.raises(ValueError, match='points_um must match'):
        diffraction_record(lift_line(sample.field_monitor('transmission')), lift_line(reference.field_monitor('incident')), orders, **kwargs)


def test_synthetic_orders_refuse_cutoff_and_name_evanescent_orders():
    p = plane(wavelength=2.)
    reference = plane(wavelength=2.)
    p.fields, _ = rayleigh(p, (2, 0), n=1., bloch=(0., 0.))
    reference.fields, _ = rayleigh(reference, (0, 0), n=1., bloch=(0., 0.))
    record = diffraction_record(p, reference, [(2, 0), (0, 0)], period_um=(2, 2))
    assert not record.valid['efficiency'][0, 0] and 'evanescent' in record.reason('efficiency', 0, 0)
    assert np.isnan(record['efficiency'][0, 0]) and record.valid['efficiency'][0, 1]
    with pytest.raises(ValueError, match='grazing cutoff'):
        diffraction_record(p, reference, [(1, 0)], period_um=(2, 2))
    zero = plane(wavelength=2.)
    with pytest.raises(ValueError, match='zero or non-finite'):
        diffraction_record(p, zero, [(0, 0)], period_um=(2, 2))


def test_dipole_farfield_and_nearzone_records_name_reduced_units_and_the_reference_guard():
    faces, bounds, _ = dipole_faces(10)
    theta = torch.linspace(.2, math.pi-.2, 7, dtype=torch.float64)
    phi = torch.linspace(0., 2*math.pi, 9, dtype=torch.float64)[:-1]
    directions = spherical_directions(theta, phi)
    reduced = farfield_record(faces, directions, bounds_um=bounds, refractive_index=1.3)
    assert reduced.kind == 'farfield' and reduced.units['intensity'] == 'reduced E*H * s^2 * m^2 per steradian'
    assert reduced.calibration['intensity'] == 'reduced' and reduced['intensity'].shape == (1, 56) and (reduced['intensity'] > 0).all()
    assert reduced.valid['intensity'].all() and 'closed box' in reduced.notes[0]
    reference = plane(wavelength=1.55)
    reference.fields, _ = rayleigh(reference, (0, 0), n=1.3, bloch=(0., 0.))
    reference.frequency_hz = faces['x_min'].frequency_hz.clone()
    reference.run_signature = faces['x_min'].run_signature
    normalized = farfield_record(faces, directions, bounds_um=bounds, refractive_index=1.3, reference=reference)
    assert normalized.units['intensity'] == '1/sr' and normalized.calibration['intensity'] == RATIO
    incident = float(reference.flux().abs()[0])
    np.testing.assert_allclose(normalized['intensity'], reduced['intensity']/incident, rtol=1e-10)
    assert normalized['reference_flux'][0] == pytest.approx(float(reference.flux()[0]))
    np.testing.assert_allclose(normalized['electric_amplitude'], reduced['electric_amplitude'], rtol=1e-10,
                               atol=1e-12*abs(reduced['electric_amplitude']).max())
    zero = plane(wavelength=1.55)
    zero.frequency_hz = reference.frequency_hz.clone()
    zero.run_signature = reference.run_signature
    with pytest.raises(ValueError, match='zero or non-finite'):
        farfield_record(faces, directions, bounds_um=bounds, refractive_index=1.3, reference=zero)
    points = spherical_points(theta[:3], torch.zeros(1, dtype=torch.float64), 6.)
    near = nearzone_record(faces, points, bounds_um=bounds, refractive_index=1.3)
    assert near.kind == 'nearzone' and near['fields'].shape == (1, 3, 6) and near['poynting'].shape == (1, 3, 3)
    assert near.units['fields'] == 'reduced field * s' and near.units['poynting'] == 'reduced E*H * s^2' and near.units['points_um'] == 'um'
    assert near.calibration['points_um'] == 'SI length'
