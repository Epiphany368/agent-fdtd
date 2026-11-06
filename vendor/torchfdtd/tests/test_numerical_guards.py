"""G6-08: every normalization path refuses or flags what it cannot support, with the reason.

Each test injects one of the situations of docs/NUMERICAL_GUARDS.md (a zero or
weak reference, a reference flowing against its direction, backflow through a
monitor, an order at cutoff or evanescent, a phase without magnitude support, an
aliased phase step, too few frequencies for a group delay, a zero plane for a
mode decomposition, zero intensity for an allocation) and asserts the reason
that comes back, and that no value was clipped or absolute-valued instead.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions, quadrant_intensity_allocation
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.field_monitors import normalize_flux
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_s_parameters, prepare_modal_launch, _profile
from torchfdtd.radiation import diffraction_efficiency, normalized_farfield_intensity
from torchfdtd.results import (REFERENCE_FRACTION, ROUNDING_FACTOR, diffraction_record, farfield_record, guarded_ratio,
                               mode_decomposition, reflection_transmission, rounding_floor, s_parameters)
from torchfdtd.solver import C0
from benchmarks.mode_injection import make_case
from test_radiation import plane, rayleigh, spherical_directions, two_frequency_faces


def test_guarded_ratio_names_every_refusal_and_never_clips():
    reference = np.array([1., 0., 5e-3, -1., 1., 1.])
    numerator = np.array([.5, .5, .5, .5, -.2, np.nan])
    ratio, valid, reasons = guarded_ratio(numerator, reference, expected_sign=1)
    assert valid.tolist() == [True, False, False, False, False, False]
    assert ratio[0] == .5 and np.isnan(ratio[1:]).all()
    assert reasons[0] is None and reasons[1] == 'reference is zero'
    assert reasons[2].startswith(f'reference below {REFERENCE_FRACTION:g} of its band peak')
    assert reasons[3].startswith('reference backflow') and reasons[4].startswith('backflow: net flux against')
    assert reasons[5] == 'non-finite flux'
    # Without a declared sign the negative numerator is a valid signed ratio.
    ratio, valid, reasons = guarded_ratio(numerator, reference)
    assert valid[4] and ratio[4] == -.2 and reasons[4] is None
    # A caller who knows the incident direction overrides the dominant-sign inference.
    ratio, valid, reasons = guarded_ratio(np.array([.5, .5]), np.array([-1., -1.]), reference_sign=-1)
    assert valid.all() and ratio.tolist() == [.5, .5]
    ratio, valid, reasons = guarded_ratio(np.array([.5, .5]), np.array([-1., -1.]), reference_sign=1)
    assert not valid.any() and all(r.startswith('reference backflow') for r in reasons)
    ratio, valid, reasons = guarded_ratio(np.zeros(3), np.zeros(3))
    assert not valid.any() and all(r == 'reference is zero at every frequency' for r in reasons)
    with pytest.raises(ValueError, match='share one shape'):
        guarded_ratio(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match='between 0 and 1'):
        guarded_ratio(np.ones(2), np.ones(2), min_reference_fraction=1.)
    with pytest.raises(ValueError, match='expected_sign'):
        guarded_ratio(np.ones(2), np.ones(2), expected_sign=2)
    # The threshold is tied to the precision: below 32 eps of the peak a float32 reference is rounding noise.
    assert rounding_floor(np.float32) == ROUNDING_FACTOR*np.finfo(np.float32).eps
    assert rounding_floor(torch.complex128) == ROUNDING_FACTOR*np.finfo(np.float64).eps
    ratio, valid, reasons = guarded_ratio(np.ones(2), np.array([1., 1e-6]), min_reference_fraction=1e-9)
    assert valid.tolist() == [True, True]
    weak = np.array([1., 1e-8], dtype=np.float32)
    ratio, valid, reasons = guarded_ratio(np.ones(2), weak, min_reference_fraction=1e-9)
    assert valid.tolist() == [True, False] and 'rounding floor' in reasons[1]


def record(flux, *, normal='x', signature='run', fields=None, npoints=4):
    """A minimal Result.frequency_fields entry with the keys normalize_flux checks."""
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    points = np.stack([np.zeros(npoints), np.linspace(-.5, .5, npoints), np.zeros(npoints)], 1)
    return dict(id='m', name='m', frequency_hz=np.linspace(1e14, 2e14, n), points_um=points, weights=np.full(npoints, 1e-13),
                normal_axis=normal, settings=dict(time_downsample=1, spectrum=dict(apodization='none')), run_signature=signature,
                flux=flux, fields=np.zeros((n, npoints, 6), dtype=complex) if fields is None else fields,
                components=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'], flux_units='reduced E*H * s^2 * m^2', field_units='reduced field * s')


def test_normalize_flux_reports_the_reason_of_every_nan():
    sample = record([.5, .5, .5, .5])
    reference = record([1., 0., 5e-3, -1.])
    out = normalize_flux(sample, reference)
    assert out['valid'].tolist() == [True, False, False, True]
    # The ratio keeps the sample's sign over the reference magnitude (docs/CONVENTIONS.md section 4).
    assert out['ratio'][0] == .5 and out['ratio'][3] == .5 and np.isnan(out['ratio'][1:3]).all()
    assert out['reasons'][0] is None and out['reasons'][1] == 'reference is zero'
    assert out['reasons'][2].startswith('reference below 0.01 of its band peak') and out['reasons'][3] is None
    tight = normalize_flux(sample, reference, min_reference_fraction=1e-3)
    assert tight['valid'][2] and tight['ratio'][2] == 100.


def test_reflection_transmission_zero_reference_and_backflow_injections():
    # The reflected flux is formed from the sample-minus-reference fields: uniform Ey and Hz whose
    # .5 Re(Ey Hz*) times the quadrature area gives -0.3 (reflected toward -x), on a zero-field reference.
    reflected = np.zeros((3, 4, 6), dtype=complex)
    reflected[..., 1] = 1e6
    reflected[..., 5] = -.3/(.5*1e6*4e-13)
    sample = {'r': record([.7, .7, .7], normal='x', fields=reflected), 't': record([.7, .7, -.7], normal='x')}
    reference = {'r': record([1., 0., 1.], normal='x'), 't': record([1., 0., 1.], normal='x')}
    out = reflection_transmission(sample, reference, reflection='r', transmission='t')
    assert out.valid['T'].tolist() == [True, False, False] and out['T'][0] == .7
    assert out.reason('T', 1) == 'reference is zero' and out.reason('T', 2).startswith('backflow: net flux against the propagation direction')
    assert out['T_signed'][2] == -.7 and np.isnan(out['T_signed'][1]) and out.reason('T_signed', 1) == 'reference is zero'
    assert out.valid['R'].tolist() == [True, False, True] and out['R'][0] == pytest.approx(.3)
    assert out.reason('R', 1) == 'reference is zero' and np.isnan(out['balance'][1:]).all()
    assert out.reason('balance', 2) == 'R or T is not supported at this frequency'


def test_s_parameter_phase_floor_aliasing_and_frequency_count():
    frequency = np.linspace(1.9e14, 2.1e14, 9)
    omega = 2*math.pi*frequency
    tau0, dispersion, centre = 5e-14, 2e-28, 2*math.pi*2e14
    s = np.stack([np.exp(1j*(omega*tau0+.5*dispersion*(omega-centre)**2)), np.full(9, 1e-20+0j)], -1)
    out = s_parameters(frequency, s)
    assert out.valid['phase_rad'][:, 0].all() and not out.valid['phase_rad'][:, 1].any()
    assert out.reason('phase_rad', 0, 1).startswith('|S| at or below the magnitude floor')
    assert out.reason('group_delay_s', 4, 1).startswith('|S| at or below the magnitude floor')
    expected = tau0+dispersion*(omega-centre)
    np.testing.assert_allclose(out['group_delay_s'][1:-1, 0], expected[1:-1], rtol=1e-6)
    # One-sided differences at the band ends: first-order error, half a step of the quadratic phase.
    np.testing.assert_allclose(out['group_delay_s'][[0, -1], 0], expected[[0, -1]], rtol=5e-2)
    # The same line sampled so sparsely that neighbouring phases differ by about pi (3.14 rad here): unwrapping
    # cannot tell +pi from -pi, so no group delay is reported, with the reason.
    sparse = np.array([1.9e14, 2.0e14, 2.1e14])
    aliased = np.exp(1j*2*math.pi*sparse*tau0)
    out = s_parameters(sparse, aliased)
    assert out.valid['phase_rad'].all() and not out.valid['group_delay_s'].any()
    assert 'unresolved (possibly aliased) phase' in out.reason('group_delay_s', 1)
    fine = s_parameters(frequency, np.exp(1j*omega*tau0))
    np.testing.assert_allclose(fine['group_delay_s'], tau0, rtol=1e-9)
    two = s_parameters(frequency[:2], np.exp(1j*omega[:2]*tau0))
    assert two.valid['phase_rad'].all() and not two.valid['group_delay_s'].any()
    assert two.reason('group_delay_s', 0) == 'group delay needs at least three frequencies'
    with pytest.raises(ValueError, match='strictly increasing'):
        s_parameters(frequency[::-1], np.exp(1j*omega*tau0))
    with pytest.raises(ValueError, match='non-finite'):
        s_parameters(frequency, np.r_[np.exp(1j*omega[:-1]*tau0), np.nan])
    with pytest.raises(ValueError, match='one entry per frequency'):
        s_parameters(frequency, np.ones(3))


def test_diffraction_order_at_cutoff_is_refused_and_evanescent_orders_are_flagged():
    p = plane(wavelength=2.)
    reference = plane(wavelength=2.)
    p.fields, _ = rayleigh(p, (2, 0), n=1., bloch=(0., 0.))
    reference.fields, _ = rayleigh(reference, (0, 0), n=1., bloch=(0., 0.))
    with pytest.raises(ValueError, match='grazing cutoff'):
        diffraction_record(p, reference, [(1, 0)], period_um=(2, 2))
    out = diffraction_record(p, reference, [(2, 0), (0, 0)], period_um=(2, 2))
    assert not out.valid['efficiency'][0, 0] and out.reason('efficiency', 0, 0).startswith('evanescent order')
    assert np.isnan(out['efficiency'][0, 0]) and out['order_power'][0, 0] == 0 and not out['propagating'][0, 0]
    assert out.valid['efficiency'][0, 1] and out['propagating'][0, 1]
    with pytest.raises(ValueError, match='zero or non-finite'):
        diffraction_record(p, plane(wavelength=2.), [(0, 0)], period_um=(2, 2))
    # A weak reference at one of two frequencies: that frequency is flagged, the other is a ratio.
    two = replace(p, fields=torch.cat((p.fields, p.fields), 0), frequency_hz=torch.cat((p.frequency_hz, 1.5*p.frequency_hz)))
    weak = replace(reference, fields=torch.cat((reference.fields, 1e-3*reference.fields), 0), frequency_hz=two.frequency_hz)
    out = diffraction_record(two, weak, [(0, 0)], period_um=(2, 2))
    assert out.valid['efficiency'][0, 0] and not out.valid['efficiency'][1, 0]
    assert out.reason('efficiency', 1, 0).startswith('reference below 0.01 of its band peak')


def test_farfield_weak_reference_frequency_is_flagged_and_zero_reference_refused():
    faces, bounds = two_frequency_faces(8)
    directions = spherical_directions(torch.tensor([.5, 1.2], dtype=torch.float64), torch.tensor([0., 1.], dtype=torch.float64))
    reference = plane(wavelength=1.55)
    fields, _ = rayleigh(reference, (0, 0), n=1.3, bloch=(0., 0.))
    reference = replace(reference, fields=torch.cat((fields, 1e-3*fields), 0), frequency_hz=faces['x_min'].frequency_hz.clone(),
                        run_signature=faces['x_min'].run_signature)
    out = farfield_record(faces, directions, bounds_um=bounds, refractive_index=1.3, reference=reference)
    assert out.valid['intensity'][0].all() and not out.valid['intensity'][1].any()
    assert out.reason('intensity', 1, 0).startswith('reference below 0.01 of its band peak')
    assert np.isnan(out['intensity'][1]).all()
    zero = replace(reference, fields=torch.zeros_like(reference.fields))
    with pytest.raises(ValueError, match='zero or non-finite'):
        farfield_record(faces, directions, bounds_um=bounds, refractive_index=1.3, reference=zero)


def test_mode_decomposition_of_a_plane_without_signal_and_the_modal_reference_refusal():
    # Twelve steps: the pulse has not reached the far plane, so its flux is zero and no fraction is supported.
    p, epsilon = make_case('slab', steps=12)
    launch = prepare_modal_launch(p, epsilon)
    model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=1))
    section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
    base = torch.tensor(np.broadcast_to(section, p.region.shape+(3,)).copy())
    with torch.no_grad():
        planes = model(base, [C0/1.55e-6])
    far = planes['far']
    assert float(far.flux().abs().max()) == 0
    out = mode_decomposition(far, [launch])
    assert not out.valid['forward_fraction'].any() and out.reason('forward_fraction', 0, 0) == 'reference is zero at every frequency'
    assert np.isnan(out['forward_fraction']).all() and out['plane_flux'][0] == 0
    with pytest.raises(ValueError, match='Modal reference is zero'):
        modal_s_parameters(far, far, launch)


def test_allocation_ratio_refuses_zero_intensity_instead_of_dividing():
    points = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.], [1., 1., 0.]], dtype=torch.float64)
    plane_record = DifferentiablePlaneResult(torch.zeros((1, 4, 6), dtype=torch.complex128), torch.tensor([2e14], dtype=torch.float64),
                                             points, torch.full((4,), 1e-14, dtype=torch.float64), (2, 2, 1), 'z', 'run', {})
    with pytest.raises(ValueError, match='finite and positive'):
        quadrant_intensity_allocation(plane_record, torch.ones(1, dtype=torch.float64))


def test_differentiable_efficiencies_refuse_a_weak_reference_band():
    # The differentiable paths keep no per-frequency mask: a reference that is zero or below the threshold at
    # any requested frequency refuses the whole call instead of dividing there.
    p = plane(wavelength=2.)
    p.fields, _ = rayleigh(p, (0, 0), n=1., bloch=(0., 0.))
    reference = replace(p, fields=p.fields.clone())
    torch.testing.assert_close(diffraction_efficiency(p, reference, [(0, 0)], period_um=(2, 2))[0, 0], torch.tensor(1., dtype=torch.float64))
    two = replace(p, fields=torch.cat((p.fields, p.fields), 0), frequency_hz=torch.cat((p.frequency_hz, 1.5*p.frequency_hz)))
    weak = replace(two, fields=torch.cat((p.fields, 1e-3*p.fields), 0))
    with pytest.raises(ValueError, match='zero or too weak in a requested frequency band'):
        diffraction_efficiency(two, weak, [(0, 0)], period_um=(2, 2))
    with pytest.raises(ValueError, match='zero or too weak'):
        diffraction_efficiency(p, replace(p, fields=torch.zeros_like(p.fields)), [(0, 0)], period_um=(2, 2))
    faces, bounds = two_frequency_faces(8)
    directions = spherical_directions(torch.tensor([.5], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64))
    reference = replace(weak, frequency_hz=faces['x_min'].frequency_hz.clone(), run_signature=faces['x_min'].run_signature)
    with pytest.raises(ValueError, match='zero or too weak in a requested frequency band'):
        normalized_farfield_intensity(faces, reference, directions, bounds_um=bounds, refractive_index=1.3)
