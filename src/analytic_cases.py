"""V1: the analytic case library. External references for the F6 readout.

Why this module exists: every check before it compared this solver against this solver.
An analytic law is the only reference that comes from outside. Rule V-0 (`plan.md` section 9)
allows a `verified` label only for a number that passed a `V1`, `V2` or `V4` check.

**Copy by citation, not by import.** The reference projects are read-only, and this project
never imports from them at run time. The laws below are transcribed, unchanged, from:

  `V1-fresnel`, `V1-airy`
      reference project `src/run_canary.py`, function `analyze`, the `interface` and `slab`
      branches. The optical thickness there is `2.1` micrometres and the reference uses an
      absolute tolerance of `0.01`.

  `V1-slab-mode`, `V1-mie`  --  NOT implemented by leg 3. See `DROPPED` below.

Leg 14 adds two cases, each recomputed here from a textbook law (no stored value):

  `V1-grating-zero-order`
      the zero-order transmission efficiency of a ZERO-MODULATION grating (a uniform lossless
      film inside the Bloch cell), oblique incidence, p polarisation. Law: the single-film
      transmission of M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge, 1999),
      section 1.6.4 ("a homogeneous dielectric film"), with the Fresnel p coefficient of
      section 1.5.2. A modulated grating has no closed form; it is recorded in `LEG14_DROPPED`.

  `V1-slab-mode`
      the effective index of the fundamental TM mode of a SYMMETRIC three-layer slab with sharp
      interfaces. Law: the TM eigenvalue equation of the symmetric planar guide,
      tan(kappa d/2) = (n1/n2)^2 gamma/kappa, K. Okamoto, *Fundamentals of Optical Waveguides*,
      2nd ed. (Academic Press, 2006), chapter 2 (planar waveguides, TM modes); the same equation
      is derived from the continuity of H_z and of E_x at the two interfaces in `slab_tm0_neff`.

Every law states its scope. Criterion C3 of the leg-3 registration says a law whose scope does
not match the configuration actually run must be DROPPED, never approximated.
"""
import math
import re

V1_IDS = ('V1-fresnel', 'V1-airy')
# Leg 14: cases judged on named scalar fields of the measured readout, not on R/T curves.
SCALAR_IDS = ('V1-grating-zero-order', 'V1-slab-mode')

# What each law assumes. A checker must show that the run it judges matches this text.
SCOPE = {
    'V1-fresnel': (
        'One planar interface between vacuum and a medium of index n, NORMAL incidence, with '
        'vacuum on both far sides. Law: R = ((n-1)/(n+1))^2 and T = 1 - R, independent of '
        'wavelength. A half-space of index n behind the interface satisfies this; an oblique run '
        'or a second interface does not.'),
    'V1-airy': (
        'One planar slab of index n and OPTICAL thickness 2.1 micrometres in vacuum, NORMAL '
        'incidence. Law: T = 1/(1 + ((n^2-1)/(2n))^2 * sin(2*pi*d/lambda)^2) and R = 1 - T, with '
        'd the optical thickness in the same unit as lambda. Lossless and non-dispersive.'),
    'V1-grating-zero-order': (
        'A Bloch cell whose structure is ONE uniform lossless film of index n and physical '
        'thickness d, filling the whole period (zero modulation), vacuum on both sides, a plane '
        'wave at incidence angle theta in vacuum, p polarisation (E in the plane of incidence). '
        'Only the (0,0) order propagates. Law: eta_0 = T_p = (1-r^2)^2 / |1 - r^2 exp(2i beta)|^2 '
        'with r = (n cos(theta) - cos(theta_t))/(n cos(theta) + cos(theta_t)), '
        'sin(theta_t) = sin(theta)/n, beta = 2 pi n d cos(theta_t)/lambda. A grating with any '
        'modulation, a lossy or dispersive film, or s polarisation does not satisfy this.'),
    'V1-slab-mode': (
        'A SYMMETRIC slab: core of index n1 and thickness d between two cladding half-spaces of '
        'index n2 < n1, sharp interfaces, lossless, the fundamental TM mode (E normal to the '
        'layers). Law: tan(kappa d/2) = (n1/n2)^2 gamma/kappa with kappa = k0 sqrt(n1^2-N^2), '
        'gamma = k0 sqrt(N^2-n2^2), N the effective index, solved on the first branch. An '
        'asymmetric stack, a smoothed interface, or the TE mode does not satisfy this.'),
}

# Cases registered in the leg-3 contract but not implemented in this leg, with the reason.
DROPPED = (
    {'id': 'V1-slab-mode',
     'reason': (
         'Stop condition 1: the law `u tan(u) = (nc/nb)^2 sqrt(V^2-u^2)` assumes a SYMMETRIC slab '
         'of thickness 0.11 micrometres with a guided mode and a modal launch. The copied project '
         'builder in `src/parity.py` constructs planar interface and slab cases with a plane-wave '
         'source and PML ends, not a symmetric waveguide with a modal port. Applying the law there '
         'would be an approximation. Register a symmetric-waveguide configuration as its own step.'),
     'source': 'reference project `src/run_tm_guide.py`, function `analytic_tm`'},
    {'id': 'V1-mie',
     'reason': (
         'Stop condition 3: the comparison needs a sphere and a closed-surface flux. The copied '
         'project builder has no sphere, so this case needs solver capability that does not exist '
         'in this repository yet. Register it as its own step.'),
     'source': 'reference project `src/run_kernel_scratch.py`, Mie series (cited in the audit)'},
)

# Leg 14: a case considered and NOT implemented, with the reason (stop condition 1).
LEG14_DROPPED = (
    {'id': 'V1-grating-modulated',
     'reason': (
         'Stop condition 1: a binary grating with non-zero modulation (the leg-9 fixture: fill '
         '0.5, eps 4) has no closed-form order efficiency. The exact treatments (rigorous '
         'coupled-wave analysis, the lamellar modal method) are truncated series solved '
         'numerically, and effective-medium formulas are approximations. Using either as a '
         'closed-form reference would be an approximation, so the leg-9 numbers stay unchecked.'),
     'source': 'M. G. Moharam and T. K. Gaylord, J. Opt. Soc. Am. 71, 811 (1981) (RCWA)'},
)

OPTICAL_THICKNESS_UM = 2.1  # the reference project's slab; its `analyze` hard-codes it


def fresnel_rt(index):
    """Exact single-interface reflectance and transmittance, normal incidence."""
    reflected = ((index - 1.0) / (index + 1.0)) ** 2
    return reflected, 1.0 - reflected


def airy_rt(index, wavelength_um, optical_thickness_um=OPTICAL_THICKNESS_UM):
    """Exact lossless single-slab transmittance and reflectance, normal incidence."""
    contrast = ((index ** 2 - 1.0) / (2.0 * index)) ** 2
    transmitted = 1.0 / (1.0 + contrast * math.sin(
        2.0 * math.pi * optical_thickness_um / wavelength_um) ** 2)
    return 1.0 - transmitted, transmitted


def film_tp(index, thickness_um, wavelength_um, angle_deg):
    """Exact p-polarised transmittance of one lossless film in vacuum, oblique incidence.

    Born and Wolf, Principles of Optics, 7th ed., section 1.6.4: for a film between two equal
    media, t = t12 t21 exp(i beta) / (1 + r12 r21 exp(2i beta)) with r21 = -r12 and
    t12 t21 = 1 - r12^2 (Stokes relations, section 1.6.4). The p coefficient
    r12 = (n cos(theta) - cos(theta_t)) / (n cos(theta) + cos(theta_t)) is section 1.5.2,
    written for vacuum on the incident side. The film carries the same medium on both sides,
    so T = |t|^2 with no impedance factor. At theta = 0 this reduces to `airy_rt`.
    """
    theta = math.radians(angle_deg)
    sin_t = math.sin(theta) / index
    cos_i, cos_t = math.cos(theta), math.sqrt(1.0 - sin_t ** 2)
    r = (index * cos_i - cos_t) / (index * cos_i + cos_t)
    beta = 2.0 * math.pi * index * thickness_um * cos_t / wavelength_um
    phase = complex(math.cos(2.0 * beta), math.sin(2.0 * beta))
    return (1.0 - r ** 2) ** 2 / abs(1.0 - r ** 2 * phase) ** 2


def film_ts(index, thickness_um, wavelength_um, angle_deg):
    """The s-polarised counterpart of `film_tp` (section 1.5.2 s coefficient). Control only."""
    theta = math.radians(angle_deg)
    sin_t = math.sin(theta) / index
    cos_i, cos_t = math.cos(theta), math.sqrt(1.0 - sin_t ** 2)
    r = (cos_i - index * cos_t) / (cos_i + index * cos_t)
    beta = 2.0 * math.pi * index * thickness_um * cos_t / wavelength_um
    phase = complex(math.cos(2.0 * beta), math.sin(2.0 * beta))
    return (1.0 - r ** 2) ** 2 / abs(1.0 - r ** 2 * phase) ** 2


def slab_tm0_neff(core_index, cladding_index, thickness_um, wavelength_um):
    """Effective index of the fundamental TM mode of a symmetric slab, sharp interfaces.

    Okamoto, Fundamentals of Optical Waveguides, 2nd ed., chapter 2. Derivation, so the law is
    recomputable here: with y normal to the layers and propagation exp(i(beta x - omega t)),
    the TM mode has H_z(y) = cos(kappa y) in the core (|y| < d/2, even mode) and
    H_z = cos(kappa d/2) exp(-gamma(|y| - d/2)) outside. Continuity of H_z and of the
    tangential E_x = (i / (omega eps)) dH_z/dy at y = d/2 gives
    kappa tan(kappa d/2) / n1^2 = gamma / n2^2, i.e. tan(kappa d/2) = (n1/n2)^2 gamma/kappa,
    with kappa^2 = k0^2 n1^2 - beta^2 and gamma^2 = beta^2 - k0^2 n2^2. The fundamental mode is
    the root with kappa d/2 in (0, pi/2), where the left side rises from 0 to +inf and the
    right side falls, so the root is unique and bisection on N in (n2, n1) brackets it.
    """
    if not 0 < cladding_index < core_index:
        raise ValueError('a guided slab needs 0 < cladding index < core index')
    k0 = 2.0 * math.pi / wavelength_um
    half = thickness_um / 2.0
    ratio = (core_index / cladding_index) ** 2

    def mismatch(neff):
        kappa = k0 * math.sqrt(core_index ** 2 - neff ** 2)
        gamma = k0 * math.sqrt(neff ** 2 - cladding_index ** 2)
        return math.sin(kappa * half) * kappa - ratio * gamma * math.cos(kappa * half)

    # Lower end of the first branch: kappa d/2 = pi/2 (or the cladding line, if that is higher).
    low = max(cladding_index, math.sqrt(max(core_index ** 2 - (math.pi / (2 * half * k0)) ** 2,
                                            0.0)))
    low, high = low * (1 + 1e-15) + 1e-15, core_index * (1 - 1e-15)
    f_low = mismatch(low)
    if f_low * mismatch(high) > 0:
        raise ValueError('no TM0 root bracketed; the slab parameters are outside the law')
    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = mismatch(mid)
        if f_mid * f_low > 0:
            low, f_low = mid, f_mid
        else:
            high = mid
    return 0.5 * (low + high)


LAWS = {'fresnel': fresnel_rt, 'airy': airy_rt}
# Leg 14 laws, judged on named scalar fields. Kept apart so `_law_values` (R/T curves) is
# untouched and the leg-3 cases decide exactly as before.
SCALAR_LAWS = {'film-tp': film_tp, 'film-ts': film_ts, 'slab-tm0': slab_tm0_neff}


def scalar_law_points(entry):
    """[(field, exact)] for one leg-14 case, recomputed from the entry's declared inputs.

    No analytic value is read from the entry or the readout: only the law's inputs.
    """
    law = entry.get('law')
    if law not in SCALAR_LAWS:
        raise ValueError(f'unknown law {law!r}')
    points = entry.get('points')
    if not isinstance(points, list) or not points:
        raise ValueError('a scalar analytic case needs a non-empty points list')
    out = []
    for point in points:
        field = point.get('field')
        if not isinstance(field, str) or not field:
            raise ValueError('every point names a measured field')
        if law in ('film-tp', 'film-ts'):
            exact = SCALAR_LAWS[law](float(entry['index']), float(entry['thickness_um']),
                                     float(entry['wavelength_um']), float(point['angle_deg']))
        else:
            exact = slab_tm0_neff(float(entry['core_index']), float(entry['cladding_index']),
                                  float(entry['thickness_um']), float(entry['wavelength_um']))
        out.append((field, exact))
    return out


def _evaluate_scalar(entry, case):
    measured = case['measured']
    errors = []
    for field, exact in scalar_law_points(entry):
        value = measured.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append((float('inf'), field, exact, None))
        else:
            errors.append((abs(float(value) - exact), field, exact, float(value)))
    worst_error, worst_field, worst_exact, worst_value = max(errors, key=lambda item: item[0])
    tolerance = entry.get('tolerance')
    finite = isinstance(tolerance, (int, float)) and tolerance > 0
    in_scope = entry.get('scope_match') is not False
    status = 'passed' if finite and in_scope and worst_error <= tolerance else (
        'out-of-scope' if not in_scope else 'failed')
    return dict(id=entry['id'], case_id=entry['case_id'], status=status, error=worst_error,
                tolerance=tolerance, tolerance_kind=entry.get('tolerance_kind'),
                basis=entry.get('basis'),
                margin=(tolerance / worst_error if finite and worst_error > 0 else None),
                worst=dict(key=worst_field, exact=worst_exact, measured=worst_value),
                points=[dict(field=field, exact=exact, error=error)
                        for error, field, exact, _ in errors],
                scope=entry.get('scope_match'), reason=entry.get('scope_match_reason'))

_PATH = re.compile(r'^cases\[(\d+)\]\.measured\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$')


def _law_values(entry, wavelength_um):
    """Recompute the whole law curve. No stored analytic value is ever read."""
    law = LAWS.get(entry.get('law'))
    if law is None:
        raise ValueError(f'unknown law {entry.get("law")!r}')
    index = entry.get('index')
    if not isinstance(index, (int, float)) or index <= 0:
        raise ValueError('every analytic case needs a positive index')
    if entry['law'] == 'fresnel':
        reflected, transmitted = law(float(index))
        return [reflected] * len(wavelength_um), [transmitted] * len(wavelength_um)
    out = [law(float(index), float(w)) for w in wavelength_um]
    return [row[0] for row in out], [row[1] for row in out]


def evaluate_analytic_cases(readout, config):
    """Recompute every declared analytic case against the run's own measured curves.

    Returns one row per declared case: the two errors, the worst point, the tolerance and its
    kind. A case passes only when EVERY sampled point is inside the tolerance AND the config does
    not declare it out of scope (`scope_match: false`); an out-of-scope case is `out-of-scope`,
    whatever its numbers say. A missing `scope_match` is treated as in scope, for the configs
    written before the field existed.
    """
    rows = []
    cases = readout.get('cases') if isinstance(readout, dict) else None
    declared = config.get('analytic_cases') if isinstance(config, dict) else None
    if not isinstance(cases, list) or not isinstance(declared, list):
        return rows
    by_id = {case.get('id'): case for case in cases if isinstance(case, dict)}
    for entry in declared:
        if isinstance(entry, dict) and entry.get('id') in SCALAR_IDS:
            case = by_id.get(entry.get('case_id'))
            if case is None or not isinstance(case.get('measured'), dict):
                rows.append(dict(id=entry['id'], case_id=entry.get('case_id'),
                                 status='not-executed', error=None,
                                 tolerance=entry.get('tolerance'),
                                 tolerance_kind=entry.get('tolerance_kind'),
                                 basis=entry.get('basis'), worst=None,
                                 reason=f'No readout case with id {entry.get("case_id")!r}.'))
            else:
                rows.append(_evaluate_scalar(entry, case))
            continue
        if not isinstance(entry, dict) or entry.get('id') not in V1_IDS:
            continue
        case = by_id.get(entry.get('case_id'))
        if case is None or not isinstance(case.get('measured'), dict):
            rows.append(dict(id=entry['id'], case_id=entry.get('case_id'), status='not-executed',
                             error=None, tolerance=entry.get('tolerance'),
                             tolerance_kind=entry.get('tolerance_kind'),
                             basis=entry.get('basis'), worst=None,
                             reason=f'No readout case with id {entry.get("case_id")!r}.'))
            continue
        measured = case['measured']
        wavelength = measured.get('wavelength_um')
        if not isinstance(wavelength, list) or not wavelength:
            rows.append(dict(id=entry['id'], case_id=entry['case_id'], status='not-executed',
                             error=None, tolerance=entry.get('tolerance'),
                             tolerance_kind=entry.get('tolerance_kind'),
                             basis=entry.get('basis'), worst=None,
                             reason='The measured case carries no wavelength_um curve.'))
            continue
        exact_r, exact_t = _law_values(entry, wavelength)
        errors = []
        for key, exact in (('R', exact_r), ('T', exact_t)):
            curve = measured.get(key)
            if not isinstance(curve, list) or len(curve) != len(exact):
                errors.append((float('inf'), key, None))
                continue
            for point, (value, target) in enumerate(zip(curve, exact)):
                errors.append((abs(float(value) - float(target)), key, point))
        worst_error, worst_key, worst_point = max(errors, key=lambda item: item[0])
        tolerance = entry.get('tolerance')
        finite = isinstance(tolerance, (int, float)) and tolerance >= 0
        in_scope = entry.get('scope_match') is not False
        status = 'passed' if finite and in_scope and worst_error <= tolerance else (
            'out-of-scope' if not in_scope else 'failed')
        rows.append(dict(id=entry['id'], case_id=entry['case_id'], status=status,
                         error=worst_error, tolerance=tolerance,
                         tolerance_kind=entry.get('tolerance_kind'),
                         basis=entry.get('basis'),
                         worst=None if worst_point is None else
                         dict(key=worst_key, point=worst_point,
                              wavelength_um=wavelength[worst_point]),
                         scope=entry.get('scope_match'), reason=entry.get('scope_match_reason')))
    return rows


def label_analytic_cases(readout, config, headlines):
    """Add `verified` labels for points covered by a passing analytic case.

    A FAILING CASE VERIFIES NOTHING, even at points that individually sit inside the tolerance.
    We do not know why the case failed, so promoting its other points would be false confidence.
    This is the opposite of a per-point optimisation of the verdict, and it is deliberate.

    A row is found by `(id, case_id)`, never by the rule id alone: several cases may share one
    rule id, and each case's own status, tolerance and basis must decide its own labels.
    """
    rows = {(row['id'], row['case_id']): row
            for row in evaluate_analytic_cases(readout, config)}
    declared = config.get('analytic_cases') if isinstance(config, dict) else None
    if not isinstance(declared, list):
        return headlines
    cases = readout.get('cases') if isinstance(readout, dict) else None
    if not isinstance(cases, list):
        return headlines
    index_of = {case.get('id'): position for position, case in enumerate(cases)
                if isinstance(case, dict)}
    for entry in declared:
        if not isinstance(entry, dict):
            continue
        row = rows.get((entry.get('id'), entry.get('case_id')))
        if row is None:
            continue
        position = index_of.get(entry.get('case_id'))
        if position is None or row['status'] != 'passed':
            continue
        if entry['id'] in SCALAR_IDS:
            _label_scalar(entry, row, position, headlines)
            continue
        measured = cases[position].get('measured', {})
        wavelength = measured.get('wavelength_um') or []
        exact_r, exact_t = _law_values(entry, wavelength)
        for key, exact in (('R', exact_r), ('T', exact_t)):
            curve = measured.get(key)
            if not isinstance(curve, list):
                continue
            for point, (value, target) in enumerate(zip(curve, exact)):
                if abs(float(value) - float(target)) > row['tolerance']:
                    continue
                path = f'cases[{position}].measured.{key}[{point}]'
                for headline in headlines:
                    if headline['path'] == path:
                        headline['label'] = 'verified'
                        headline.setdefault('invariants', []).append(entry['id'])
                        headline['reason'] = (
                            f'Inside the declared tolerance of {entry["id"]} on case '
                            f'{entry["case_id"]}: absolute error '
                            f'{abs(float(value) - float(target)):.6g} <= {row["tolerance"]!r} '
                            f'({row["tolerance_kind"]}). Law scope: {SCOPE.get(entry["id"], "")} '
                            f'Basis: {row["basis"]}')
    return headlines


def _label_scalar(entry, row, position, headlines):
    """A passed leg-14 case verifies exactly the fields it names, on its own case."""
    for point in row['points']:
        path = f'cases[{position}].measured.{point["field"]}'
        for headline in headlines:
            if headline['path'] != path:
                continue
            headline['label'] = 'verified'
            headline.setdefault('invariants', []).append(entry['id'])
            headline['reason'] = (
                f'Inside the declared tolerance of {entry["id"]} on case {entry["case_id"]}: '
                f'absolute error {point["error"]:.6g} <= {row["tolerance"]!r} '
                f'({row["tolerance_kind"]}). Law scope: {SCOPE.get(entry["id"], "")} '
                f'Basis: {row["basis"]}')


def negative_control_scalar(entry, readout, offset):
    """Shift the first named field of an in-memory copy by `offset`; the source is untouched."""
    import copy
    perturbed = copy.deepcopy(readout)
    field = entry['points'][0]['field']
    for case in perturbed.get('cases') or []:
        if case.get('id') == entry.get('case_id'):
            case['measured'][field] = float(case['measured'][field]) + offset
    return perturbed


def negative_control(entry, readout, offset):
    """Perturb one measured point inside an in-memory copy and return the perturbed copy.

    The source run files are never written. Used to show that the same code reports a failure.
    """
    import copy
    import json as _json
    perturbed = copy.deepcopy(readout)
    cases = perturbed.get('cases')
    declared = entry.get('case_id')
    for case in cases or []:
        if case.get('id') != declared:
            continue
        measured = case['measured']
        wavelength = measured['wavelength_um']
        exact_r, exact_t = _law_values(entry, wavelength)
        key = entry.get('negative_control_key', 'T')
        curve = measured[key]
        target = (exact_t if key == 'T' else exact_r)[0]
        curve[0] = target + offset
    return _json.loads(_json.dumps(perturbed))
