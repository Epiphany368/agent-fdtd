"""Integrated results through one record type that names its units and its guards.

Reflection, transmission and absorption with the reference-run bookkeeping,
complex S-parameters with phase and group delay, mode decomposition, diffraction
orders and far-field or near-zone projections are read from the public outputs
of a run (``Result.field_monitor``, ``DifferentiablePlaneResult`` planes, mode
launches, ``ModeNetworkResult.s``) and returned as a ``ResultRecord``: every
value carries its unit and its calibration (docs/CONVENTIONS.md section 5:
reduced fields and fluxes, dimensionless ratios to a matched reference, SI
frequencies, times and lengths), and every entry a normalization cannot support
is NaN with a reason instead of a clipped number (docs/NUMERICAL_GUARDS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .field_monitors import normalize_flux

REFERENCE_FRACTION = .01      # a reference below this fraction of its band peak supports no ratio (normalize_flux default)
ROUNDING_FACTOR = 32          # |x| <= ROUNDING_FACTOR * eps(dtype) * max |x| is rounding noise, not a signal

RATIO = 'dimensionless ratio to the matched reference run'


def rounding_floor(dtype):
    """The relative magnitude below which a value of ``dtype`` is rounding noise."""
    real = torch.finfo(dtype if isinstance(dtype, torch.dtype) else torch.as_tensor(np.zeros(1, dtype=dtype)).real.dtype)
    return ROUNDING_FACTOR*real.eps


@dataclass
class ResultRecord:
    """Named values with units, calibration, validity and the reason of every invalid entry."""
    kind: str
    frequency_hz: np.ndarray
    values: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    valid: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)
    notes: tuple = ()

    def add(self, name, value, *, units, calibration, definition, valid=None, reasons=None):
        value = np.asarray(value)
        if valid is None:
            valid = np.isfinite(value) if value.dtype.kind in 'fc' else np.ones(value.shape, dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        if reasons is None:
            reasons = np.where(valid, None, 'non-finite value').astype(object)
        reasons = np.asarray(reasons, dtype=object)
        if reasons.shape != valid.shape or (value.shape and valid.shape != value.shape):
            raise ValueError(f'{name}: value, valid and reasons must share one shape.')
        self.values[name] = value
        self.units[name] = units
        self.calibration[name] = calibration
        self.valid[name] = valid
        self.reasons[name] = reasons
        self.definitions[name] = definition
        return self

    def __getitem__(self, name):
        return self.values[name]

    def __contains__(self, name):
        return name in self.values

    def reason(self, name, *index):
        """The reason of one entry, None when it is valid."""
        return self.reasons[name][index] if index else self.reasons[name]

    def as_dict(self):
        """JSON-ready copy: complex values as [real, imag], NaN kept as null through the valid masks."""
        def plain(value):
            value = np.asarray(value)
            if value.dtype.kind == 'c':
                return np.stack((value.real, value.imag), -1).tolist()
            if value.dtype == object:
                return value.tolist()
            return value.tolist()
        return dict(kind=self.kind, frequency_hz=np.asarray(self.frequency_hz).tolist(),
                    values={k: plain(v) for k, v in self.values.items()}, units=dict(self.units),
                    calibration=dict(self.calibration), valid={k: v.tolist() for k, v in self.valid.items()},
                    reasons={k: v.tolist() for k, v in self.reasons.items()}, definitions=dict(self.definitions),
                    notes=list(self.notes))


def guarded_ratio(numerator, reference, *, min_reference_fraction=REFERENCE_FRACTION, expected_sign=None,
                  reference_sign='dominant'):
    """``numerator / |reference|`` with every unsupported entry NaN and a reason.

    Reasons, in the order they are tested per entry: non-finite input; a
    reference that is exactly zero; a reference below ``min_reference_fraction``
    of the largest |reference| (or below the rounding floor of its dtype); a
    reference flowing against its dominant direction (``reference_sign``
    ``'dominant'``: the sign of the entry with the largest magnitude, or ``+1``
    / ``-1`` when the caller knows the incident direction); a numerator whose
    sign is not ``expected_sign`` (backflow). Nothing is clipped or absolute-
    valued: the denominator's magnitude is used only after its sign was checked.
    """
    # The rounding floor is that of the reference's own dtype; the arithmetic runs in float64.
    dtype = np.asarray(reference).dtype
    dtype = dtype if dtype.kind in 'fc' else np.dtype(float)
    numerator = np.asarray(numerator, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if numerator.shape != reference.shape:
        raise ValueError('Numerator and reference must share one shape.')
    if not 0 < min_reference_fraction < 1:
        raise ValueError('Reference threshold must lie between 0 and 1.')
    if expected_sign not in (None, 1, -1) or reference_sign not in ('dominant', 1, -1):
        raise ValueError('expected_sign must be None, 1 or -1 and reference_sign "dominant", 1 or -1.')
    ratio = np.full(reference.shape, np.nan)
    reasons = np.full(reference.shape, None, dtype=object)
    finite = np.isfinite(numerator) & np.isfinite(reference)
    reasons[~finite] = 'non-finite flux'
    magnitude = np.abs(reference)
    peak = float(magnitude[finite].max()) if finite.any() else 0.
    if peak <= 0:
        reasons[finite] = 'reference is zero at every frequency'
        return ratio, np.zeros(reference.shape, dtype=bool), reasons
    floor = max(min_reference_fraction, rounding_floor(dtype))*peak
    dominant = np.sign(reference[finite][np.argmax(magnitude[finite])]) if reference_sign == 'dominant' else reference_sign
    zero = finite & (magnitude == 0)
    weak = finite & ~zero & (magnitude <= floor)
    against = finite & ~zero & ~weak & (np.sign(reference) != dominant)
    reasons[zero] = 'reference is zero'
    reasons[weak] = f'reference below {min_reference_fraction:g} of its band peak (or its rounding floor): no supported ratio'
    reasons[against] = 'reference backflow: the reference flux runs against its dominant direction'
    valid = finite & ~zero & ~weak & ~against
    if expected_sign is not None:
        backflow = valid & (np.sign(numerator) == -expected_sign)
        reasons[backflow] = 'backflow: net flux against the propagation direction (the signed ratio is kept separately)'
        valid &= ~backflow
    ratio[valid] = numerator[valid]/magnitude[valid]
    return ratio, valid, reasons


def _plane(container, name):
    if isinstance(container, dict):
        if name not in container:
            raise ValueError(f'Monitor {name!r} is not in the record mapping.')
        return container[name]
    return container.field_monitor(name)


def reflection_transmission(sample, reference, *, reflection, transmission, direction='+', absorption=None,
                            min_reference_fraction=REFERENCE_FRACTION):
    """R, T and A of a run against its matched reference run, with the bookkeeping named.

    ``sample`` and ``reference`` are ``Result`` objects (or mappings of monitor
    id to ``Result.frequency_fields`` records) of the same scene with and without
    the structure. ``reflection`` and ``transmission`` name the monitors; both
    share the normal axis along which the incident wave travels in
    ``direction`` ('+' or '-'). T is the transmitted flux in that direction over
    the reference flux through the transmission monitor; R is the flux of the
    sample-minus-reference fields through the reflection monitor, against the
    direction, over the reference flux through the reflection monitor. A is
    measured only from ``absorption``: a mapping of closed-box faces
    (``x_min`` ... ``z_max`` records of the sample run) whose net inward flux
    over the reference transmitted flux is the absorbed fraction. Without it A
    is 'not measured' and ``balance`` = 1 - R - T is a balance, not absorption.
    """
    if direction not in ('+', '-'):
        raise ValueError("direction must be '+' or '-'.")
    sign = 1 if direction == '+' else -1
    r_sample, r_reference = _plane(sample, reflection), _plane(reference, reflection)
    t_sample, t_reference = _plane(sample, transmission), _plane(reference, transmission)
    if r_sample['normal_axis'] != t_sample['normal_axis']:
        raise ValueError('Reflection and transmission monitors must share the normal axis of the incident wave.')
    # normalize_flux validates the matched reference (signature, sampling, apodization, stored components) and
    # forms the reflected flux of the difference fields; its own threshold is disabled so that every guard
    # below is the declared one. Its NaN marks a zero reference, which guarded_ratio names.
    transmitted = normalize_flux(t_sample, t_reference, min_reference_fraction=1e-300)
    reflected = normalize_flux(r_sample, r_reference, subtract_incident=True, min_reference_fraction=1e-300)
    frequency = np.asarray(transmitted['frequency_hz'])
    if not np.allclose(frequency, reflected['frequency_hz'], rtol=1e-10, atol=0):
        raise ValueError('Reflection and transmission monitors must sample the same frequencies.')
    flux_units = t_sample['flux_units']
    record = ResultRecord('reflection_transmission', frequency,
                          notes=(f'incident direction {direction}{t_sample["normal_axis"]}; reference run flux is the denominator of every ratio; '
                                 'fluxes are reduced (no absolute watt calibration), ratios are dimensionless',))
    t_ratio, t_valid, t_reasons = guarded_ratio(sign*np.asarray(t_sample['flux']), np.asarray(t_reference['flux']),
                                                min_reference_fraction=min_reference_fraction, expected_sign=1, reference_sign=sign)
    r_flux = -sign*np.where(reflected['valid'], reflected['ratio'], 0.)*np.abs(np.asarray(r_reference['flux']))
    r_ratio, r_valid, r_reasons = guarded_ratio(r_flux, np.asarray(r_reference['flux']),
                                                min_reference_fraction=min_reference_fraction, expected_sign=1, reference_sign=sign)
    record.add('T', t_ratio, units='dimensionless', calibration=RATIO, valid=t_valid, reasons=t_reasons,
               definition='signed flux through the transmission monitor along the incident direction / reference flux through it')
    record.add('R', r_ratio, units='dimensionless', calibration=RATIO, valid=r_valid, reasons=r_reasons,
               definition='flux of (sample - reference) fields through the reflection monitor against the incident direction / reference flux through it')
    nonzero = np.asarray(t_reference['flux']) != 0
    record.add('T_signed', np.where(nonzero, sign*np.asarray(t_sample['flux'])/np.where(nonzero, np.abs(t_reference['flux']), 1.), np.nan),
               units='dimensionless', calibration=RATIO, valid=nonzero,
               reasons=np.where(nonzero, None, 'reference is zero').astype(object),
               definition='the same ratio with its sign kept and no threshold: negative means net backflow through the transmission monitor')
    record.add('sample_flux_T', np.asarray(t_sample['flux']), units=flux_units, calibration='reduced',
               definition='signed flux through the transmission monitor along +normal, sample run')
    record.add('reference_flux_T', np.asarray(t_reference['flux']), units=flux_units, calibration='reduced',
               definition='signed flux through the transmission monitor along +normal, reference run')
    record.add('sample_flux_R', np.asarray(r_sample['flux']), units=flux_units, calibration='reduced',
               definition='signed flux through the reflection monitor along +normal, sample run (incident plus reflected)')
    record.add('reference_flux_R', np.asarray(r_reference['flux']), units=flux_units, calibration='reduced',
               definition='signed flux through the reflection monitor along +normal, reference run')
    if absorption is None:
        record.add('A', np.full(frequency.shape, np.nan), units='dimensionless', calibration=RATIO,
                   valid=np.zeros(frequency.shape, dtype=bool),
                   reasons=np.full(frequency.shape, 'not measured: no absorption monitor or closed flux box was supplied', dtype=object),
                   definition='absorbed fraction; not measured in this record')
    else:
        faces = absorption if isinstance(absorption, dict) else {name: _plane(sample, name) for name in absorption}
        active = 'xy' if 'invariant length' in flux_units else 'xyz'
        expected = {f'{axis}_{side}' for axis in active for side in ('min', 'max')}
        if set(faces) != expected:
            raise ValueError(f'Absorption needs the closed box faces {sorted(expected)} of the sample run.')
        outward = np.zeros(frequency.shape)
        for name, plane in faces.items():
            axis, side = name.rsplit('_', 1)
            if plane['normal_axis'] != axis:
                raise ValueError(f'Absorption face {name} must be {axis}-normal.')
            if not np.allclose(plane['frequency_hz'], frequency, rtol=1e-10, atol=0) or plane['flux'] is None:
                raise ValueError(f'Absorption face {name} must record flux at the same frequencies as the monitors.')
            outward = outward+np.asarray(plane['flux'])*(1 if side == 'max' else -1)
        absorbed, a_valid, a_reasons = guarded_ratio(-outward, np.asarray(t_reference['flux']),
                                                     min_reference_fraction=min_reference_fraction, expected_sign=1, reference_sign=sign)
        a_reasons = np.where(a_valid | ~np.char.startswith(a_reasons.astype(str), 'backflow'), a_reasons,
                             'negative absorbed power: the box flux leaves the box (gain, or the box does not enclose the absorber)')
        record.add('A', absorbed, units='dimensionless', calibration=RATIO, valid=a_valid, reasons=a_reasons,
                   definition='net inward flux through the closed box of the sample run / reference flux through the transmission monitor')
        record.add('box_outward_flux', outward, units=flux_units, calibration='reduced',
                   definition='sum of the outward signed fluxes of the box faces (x_min faces -x, x_max faces +x)')
    balance = 1-np.where(r_valid, r_ratio, np.nan)-np.where(t_valid, t_ratio, np.nan)
    record.add('balance', balance, units='dimensionless', calibration=RATIO,
               valid=r_valid & t_valid, reasons=np.where(r_valid & t_valid, None, 'R or T is not supported at this frequency').astype(object),
               definition='1 - R - T: the power the two monitors do not account for (absorption, scattering out of the monitors, '
                          'numerical error); it is a balance, not an absorption measurement')
    return record


def s_parameters(frequency_hz, s, *, channels=None, magnitude_floor=None):
    """Complex S with phase and group delay from a frequency sweep of mode-port amplitudes.

    ``s`` is one S matrix per frequency (F, n, m), one column (F, n) or one
    element (F,), as ``ModeNetworkResult.s`` or ``modal_s_parameters`` return
    at each carrier of a sweep. The phase is unwrapped along frequency; the
    group delay is +d(phase)/d(omega) by central differences, one-sided at the
    ends (docs/CONVENTIONS.md: with exp(-i omega t) a delay tau carries the
    phase +omega tau, so a straight guide has arg S21 = +beta L). Guards: an
    element whose magnitude is at or below ``magnitude_floor``
    (default the rounding floor of the dtype times the largest magnitude) has
    no phase; a phase step of pi/2 or more between neighbouring frequencies
    (after unwrapping, so a true step near pi is indistinguishable from one
    near -pi) is unresolved and gives no group delay there; fewer than three
    frequencies give no group delay at all; frequencies must increase strictly.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    value = np.asarray(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s)
    if not np.iscomplexobj(value):
        value = value.astype(complex)
    if frequency.ndim != 1 or len(frequency) == 0 or value.shape[0] != len(frequency):
        raise ValueError('s needs one entry per frequency along its first axis.')
    if not np.isfinite(frequency).all() or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError('Frequencies must be finite, positive and strictly increasing.')
    if not np.isfinite(value).all():
        raise ValueError('S contains non-finite entries.')
    magnitude = np.abs(value)
    if magnitude_floor is None:
        magnitude_floor = rounding_floor(value.dtype)*float(magnitude.max() if magnitude.size else 0.)
    record = ResultRecord('s_parameters', frequency, notes=('S is an amplitude ratio between fixed port modes at their reference planes; '
                                                             'no absolute power calibration', f'channels: {channels}' if channels is not None else 'channels unnamed'))
    record.add('S', value, units='dimensionless', calibration='dimensionless ratio of mode amplitudes',
               definition='complex mode amplitude ratio, outgoing over incident, at the port reference planes')
    record.add('magnitude', magnitude, units='dimensionless', calibration='dimensionless',
               definition='|S|')
    supported = magnitude > magnitude_floor
    phase = np.full(value.shape, np.nan)
    phase_reasons = np.where(supported, None, f'|S| at or below the magnitude floor {magnitude_floor:.3e}: phase undefined').astype(object)
    raw = np.angle(value)
    # Unwrap along frequency through the supported entries only.
    flat = raw.reshape(len(frequency), -1)
    ok = supported.reshape(len(frequency), -1)
    out = phase.reshape(len(frequency), -1)
    for column in range(flat.shape[1]):
        index = np.flatnonzero(ok[:, column])
        if len(index):
            out[index, column] = np.unwrap(flat[index, column])
    phase = out.reshape(value.shape)
    record.add('phase_rad', phase, units='rad', calibration='dimensionless', valid=supported, reasons=phase_reasons,
               definition='arg S, unwrapped along frequency through the supported entries (exp(-i omega t) convention)')
    delay = np.full(value.shape, np.nan)
    delay_valid = np.zeros(value.shape, dtype=bool)
    delay_reasons = np.full(value.shape, None, dtype=object)
    if len(frequency) < 3:
        delay_reasons[...] = 'group delay needs at least three frequencies'
    else:
        omega = 2*math.pi*frequency
        step = np.abs(np.diff(phase, axis=0))
        resolved = np.isfinite(step) & (step < math.pi/2)
        for k in range(len(frequency)):
            if k == 0 or k == len(frequency)-1:
                lo, hi = (0, 1) if k == 0 else (k-1, k)
            else:
                lo, hi = k-1, k+1
            usable = supported[k] & np.all(resolved[lo:hi], axis=0)
            slope = np.where(usable, (phase[hi]-phase[lo])/(omega[hi]-omega[lo]), np.nan)
            delay[k] = slope
            delay_valid[k] = usable
            row = np.where(usable, None, np.where(supported[k], 'phase step of pi/2 or more to a neighbouring frequency: unresolved (possibly aliased) phase, '
                                                                'no group delay; sample the band more finely', phase_reasons[k]))
            delay_reasons[k] = row if row.ndim else row.item()
    record.add('group_delay_s', delay, units='s', calibration='SI (from the SI frequency axis)', valid=delay_valid, reasons=delay_reasons,
               definition='+d(arg S)/d(omega) by central differences (one-sided at the band ends); positive for a causal delay '
                          'under the exp(-i omega t) convention')
    return record


def mode_decomposition(plane, launches, *, names=None):
    """Forward and backward amplitudes of a plane in fixed mode bases, with their power fractions.

    ``plane`` is a ``DifferentiablePlaneResult`` at the single carrier of the
    launches; ``launches`` are ``ModalLaunch`` objects (``prepare_modal_launch``
    or ``prepare_aperture_modal_launch``). The amplitudes come from the public
    ``modal_plane_amplitudes`` and are referenced to the unit reduced-power
    mode; a fraction is |a|^2 over the magnitude of the plane's total flux.
    Guards: a plane whose flux is at or below the rounding floor supports no
    fraction; a mode amplitude at or below the rounding floor of the largest
    amplitude is a weak mode and its fraction is flagged.
    """
    from .mode_injection import modal_plane_amplitudes
    launches = list(launches)
    if not launches:
        raise ValueError('Give at least one mode launch.')
    names = list(names) if names is not None else [f'mode{k}' for k in range(len(launches))]
    if len(names) != len(launches):
        raise ValueError('names must match the launches.')
    frequency = plane.frequency_hz.detach().cpu().numpy().astype(float)
    flux = plane.flux().detach().cpu().numpy().astype(float)
    forward = np.zeros((len(frequency), len(launches)), dtype=complex)
    backward = np.zeros_like(forward)
    for k, launch in enumerate(launches):
        amplitudes = modal_plane_amplitudes(plane, launch)
        forward[:, k] = amplitudes['forward'].detach().cpu().numpy()
        backward[:, k] = amplitudes['backward'].detach().cpu().numpy()
    record = ResultRecord('mode_decomposition', frequency, notes=(f'modes: {names}', 'amplitudes carry the plane DFT factor (seconds) and are '
                                                                 'referenced to unit reduced-power modes; fractions are ratios to the plane flux magnitude'))
    amplitude_units = 'reduced field * s (unit reduced-power mode basis)'
    record.add('forward', forward, units=amplitude_units, calibration='reduced',
               definition='complex amplitude of the +normal propagating mode (modal_plane_amplitudes)')
    record.add('backward', backward, units=amplitude_units, calibration='reduced',
               definition='complex amplitude of the -normal propagating mode')
    record.add('plane_flux', flux, units=plane.flux_units, calibration='reduced',
               definition='signed total flux through the plane along +normal')
    floor = rounding_floor(plane.fields.dtype)
    power = np.abs(forward)**2, np.abs(backward)**2
    largest = max(float(power[0].max()), float(power[1].max()))
    denominator = np.abs(flux)[:, None]*np.ones((1, len(launches)))
    weak = np.stack([p <= floor*largest for p in power])
    for name, p, weak_mask in zip(('forward_fraction', 'backward_fraction'), power, weak):
        ratio, valid, reasons = guarded_ratio(p, denominator, min_reference_fraction=floor if floor < .5 else .01)
        reasons = np.where(valid & weak_mask, 'weak mode: |amplitude|^2 at or below the rounding floor of the largest modal power', reasons)
        valid = valid & ~weak_mask
        ratio = np.where(valid, ratio, np.nan)
        record.add(name, ratio, units='dimensionless', calibration='dimensionless ratio to the plane flux magnitude', valid=valid, reasons=reasons,
                   definition=f'|{name.split("_")[0]} amplitude|^2 / |plane flux|; the modal power in the {name.split("_")[0]} direction as a fraction of the total')
    return record


def diffraction_record(plane, reference, orders, *, period_um, refractive_index=1., bloch_wavevector_per_um=(0., 0.),
                direction='forward', subtract_incident=False, min_reference_fraction=REFERENCE_FRACTION, cutoff_tolerance=1e-5):
    """Diffraction efficiencies of the Rayleigh orders of a plane over the matched incident plane power.

    The orders come from the public ``diffraction_orders`` on the sample plane
    (minus the reference fields when ``subtract_incident``); the reference
    plane's flux magnitude is the denominator. Guards: an order at grazing
    cutoff is refused by ``diffraction_orders`` (ValueError); an evanescent
    order carries no real power and its efficiency is NaN with the reason,
    never zero; a reference below ``min_reference_fraction`` of its band peak
    supports no efficiency at that frequency.
    """
    from dataclasses import replace
    from .radiation import diffraction_orders
    if direction not in ('forward', 'backward'):
        raise ValueError('Direction must be forward or backward.')
    if plane.run_signature != reference.run_signature or plane.normal != reference.normal:
        raise ValueError('Reference mesh, source, duration and normal must match.')
    for name in ('frequency_hz', 'points_um', 'weights'):
        x, y = getattr(plane, name), getattr(reference, name)
        if x.device != y.device or x.dtype != y.dtype or not torch.equal(x, y):
            raise ValueError(f'Reference {name} must match exactly.')
    scale = reference.fields.detach().abs().amax()
    if not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError('Reference fields are zero or non-finite: no incident power to normalize by.')
    fields = plane.fields/scale
    if subtract_incident:
        fields = fields-reference.fields/scale
    result = diffraction_orders(replace(plane, fields=fields), orders, period_um=period_um, refractive_index=refractive_index,
                                bloch_wavevector_per_um=bloch_wavevector_per_um, cutoff_tolerance=cutoff_tolerance)
    incident = replace(reference, fields=reference.fields/scale).flux().detach().cpu().numpy().astype(float)
    frequency = plane.frequency_hz.detach().cpu().numpy().astype(float)
    power = getattr(result, direction+'_power').detach().cpu().numpy().astype(float)
    propagating = result.propagating.detach().cpu().numpy().astype(bool)
    denominator = np.abs(incident)[:, None]*np.ones((1, power.shape[1]))
    ratio, valid, reasons = guarded_ratio(power, denominator, min_reference_fraction=min_reference_fraction)
    reasons = np.where(valid & ~propagating, 'evanescent order: k_normal^2 < 0, no real power (amplitude available in the fields)', reasons)
    valid = valid & propagating
    ratio = np.where(valid, ratio, np.nan)
    record = ResultRecord('diffraction', frequency,
                          notes=(f'{direction} orders {np.asarray(orders).tolist()} of a {plane.normal}-normal plane; efficiency = order power / |incident plane flux| '
                                 'of the matched reference run; the exterior index is real and lossless',))
    record.add('orders', np.asarray(orders), units='integer pair', calibration='dimensionless', definition='requested (m, n) Rayleigh orders')
    record.add('efficiency', ratio, units='dimensionless', calibration=RATIO, valid=valid, reasons=reasons,
               definition=f'{direction} real power of the order / |flux of the reference plane|')
    record.add('propagating', propagating, units='boolean', calibration='dimensionless', definition='k_normal^2 > 0 for the order at this frequency')
    record.add('order_power', power*float(scale)**2, units=plane.flux_units, calibration='reduced',
               definition=f'{direction} real power of the order (zero for evanescent orders by definition)')
    record.add('reference_flux', incident*float(scale)**2, units=reference.flux_units, calibration='reduced',
               definition='signed flux of the reference plane along +normal')
    record.add('wavevector_per_um', result.wavevector_per_um.detach().cpu().numpy(), units='rad/um', calibration='SI length',
               definition='wavevector of the positive-normal branch of each order')
    return record


def farfield_record(faces, directions, *, bounds_um, reference=None, min_reference_fraction=REFERENCE_FRACTION, **kwargs):
    """Far-field intensity of a closed box, reduced or normalized to a matched incident plane.

    ``faces`` and ``directions`` are those of ``project_farfield``. Without a
    reference the intensity is the reduced spectral power per steradian; with
    ``reference`` (a plane of the matched empty run) it is divided by the
    reference flux magnitude at each frequency under the near-zero guard.
    """
    from dataclasses import replace
    from .radiation import project_farfield
    scale = 1.
    if reference is not None:
        for face in faces.values():
            if (face.run_signature != reference.run_signature or face.fields.device != reference.fields.device or
                    face.fields.dtype != reference.fields.dtype or not torch.equal(face.frequency_hz, reference.frequency_hz)):
                raise ValueError('Incident reference and surface run signatures/frequencies/dtypes/devices must match.')
        scale = reference.fields.detach().abs().amax()
        if not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise ValueError('Reference fields are zero or non-finite: no incident power to normalize by.')
        faces = {name: replace(face, fields=face.fields/scale) for name, face in faces.items()}
    result = project_farfield(faces, directions, bounds_um=bounds_um, **kwargs)
    frequency = result.frequency_hz.detach().cpu().numpy().astype(float)
    intensity = result.intensity().detach().cpu().numpy().astype(float)
    record = ResultRecord('farfield', frequency, notes=(f'exterior index {result.refractive_index}, mu_r {result.relative_permeability}; '
                                                       f'{result.approximation or "closed box"}',))
    record.add('directions', result.directions.detach().cpu().numpy(), units='unit vector', calibration='dimensionless',
               definition='observation directions')
    if reference is None:
        record.add('intensity', intensity, units='reduced E*H * s^2 * m^2 per steradian', calibration='reduced',
                   definition='.5 Re(n / mu_r) |A|^2 spectral power per steradian (FarFieldResult.intensity)')
    else:
        incident = replace(reference, fields=reference.fields/scale).flux().detach().cpu().numpy().astype(float)
        denominator = np.abs(incident)[:, None]*np.ones((1, intensity.shape[1]))
        ratio, valid, reasons = guarded_ratio(intensity, denominator, min_reference_fraction=min_reference_fraction)
        record.add('intensity', ratio, units='1/sr', calibration=RATIO, valid=valid, reasons=reasons,
                   definition='spectral power per steradian / |flux of the reference plane| (normalized_farfield_intensity)')
        record.add('reference_flux', incident*float(scale)**2, units=reference.flux_units, calibration='reduced',
                   definition='signed flux of the reference plane along +normal')
    record.add('electric_amplitude', result.electric_amplitude.detach().cpu().numpy()*(float(scale) if reference is not None else 1.),
               units='reduced field * s * m', calibration='reduced', definition='far-field electric amplitude A per direction (E r exp(-i k r))')
    return record


def nearzone_record(faces, points_um, *, bounds_um, **kwargs):
    """Near-zone fields and Poynting vector at points outside a closed box, in reduced units."""
    from .radiation import project_nearzone
    result = project_nearzone(faces, points_um, bounds_um=bounds_um, **kwargs)
    frequency = result.frequency_hz.detach().cpu().numpy().astype(float)
    record = ResultRecord('nearzone', frequency, notes=(f'exterior index {result.refractive_index}, mu_r {result.relative_permeability}; '
                                                       f'{result.approximation or "closed box"}',))
    record.add('points_um', result.points_um.detach().cpu().numpy(), units='um', calibration='SI length', definition='observation points')
    record.add('fields', result.fields.detach().cpu().numpy(), units='reduced field * s', calibration='reduced',
               definition='Ex, Ey, Ez, Hx, Hy, Hz spectral fields at the points')
    record.add('poynting', result.poynting().detach().cpu().numpy(), units='reduced E*H * s^2', calibration='reduced',
               definition='.5 Re(E x H*) per point')
    return record
