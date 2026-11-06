"""V5: the units and normalisation ledger, machine-checked.

Why this module exists. The largest unresolved discrepancy this project inherited is attributed to
the **readout definition**, not to the solver: a `5.6` percentage-point difference in
transition-region loss, and a radiation strength about `40 percent` above a published design. A
number that does not say which plane it was read on, over what span, and against what normalisation
cannot be compared with anything.

The contract, pre-declared in `plan.md` (leg 5):

* the declaration is written in the config **before** the run and mirrored into the readout; this
  module **diffs** the two, so a readout cannot describe itself differently from how it was set up;
* every numeric leaf under `measured` needs a declaration: `unit`, `plane`, `span`, `normalisation`;
* numeric case metadata is covered by the `EXEMPT` table below, which is part of this checked
  artifact, so a key that is in neither place is flagged;
* the unit vocabulary is closed, and one key may carry exactly one unit.

Nothing here is assumed to be dimensionless by default.
"""
import math

# The closed vocabulary. A unit outside this tuple is a failure, not a warning.
VOCABULARY = (
    'dimensionless',   # a ratio, a fraction, an index, a normalised amplitude
    'second',
    'femtosecond',
    'hertz',
    'micrometre',
    'nanometre',
    'count',           # a cell count, a step count, an array length
    'sha256',          # an identity digest, never a measured quantity
    'reduced-unit',    # the solver's own units, which are not claimed to be physical
    'not-applicable',
    # Added by leg 13 (S4) for registered numbers that already exist: the leg-7 drift bound is an
    # angle and its fixed wavevector is an inverse length. Adding a unit is editing the contract.
    'degree',
    'per-micrometre',
)

# Numeric case metadata, exempt from a physics declaration but NOT from having a unit.
# The reason is part of the checked artifact: editing it is editing the contract.
EXEMPT = {
    'ppw': ('count', 'points per wavelength: a resolution parameter, not a measured quantity'),
    'index': ('dimensionless', 'refractive index of the case'),
    'courant_factor': ('dimensionless', 'the time-step safety factor'),
    'steps': ('count', 'number of time updates'),
    'dt_s': ('second', 'the time step'),
    'duration_s': ('second', 'physical duration actually integrated'),
    'duration_fs': ('femtosecond', 'the requested duration'),
    'wall_seconds': ('second', 'wall-clock cost of the sample run'),
    'reference_wall_seconds': ('second', 'wall-clock cost of the reference run'),
    'sample_elapsed_seconds': ('second', 'wall-clock cost of the sample run'),
    'reference_elapsed_seconds': ('second', 'wall-clock cost of the reference run'),
    'shape': ('count', 'grid dimensions in cells'),
    'edge_count': ('count', 'number of geometry edges'),
    'launch_scores': ('dimensionless', 'mode-selection scores'),
    'launch_neff': ('dimensionless', 'effective index of the launched mode'),
    'frequency_points': ('count', 'number of sampled frequencies'),
}

DECLARATION_FIELDS = ('unit', 'plane', 'span', 'normalisation')


def _numeric_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_leaves(child, path + (key,))
    elif isinstance(value, list):
        for position, child in enumerate(value):
            yield from _numeric_leaves(child, path + (position,))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return
        yield path, value
    elif isinstance(value, str) and path and str(path[-1]).endswith('_sha256'):
        yield path, value


def leaf_key(path):
    """The declaration key for a leaf path: the measured key, or the metadata key."""
    if len(path) >= 2 and path[0] == 'measured':
        return str(path[1])
    return str(path[0]) if path else ''


def check_declaration(declaration):
    """Return (ok, reason) for one declaration entry."""
    if not isinstance(declaration, dict):
        return False, 'declaration is not an object'
    for field in DECLARATION_FIELDS:
        value = declaration.get(field)
        if not isinstance(value, str) or not value.strip():
            return False, f'missing or empty {field!r}'
    unit = declaration['unit']
    if unit not in VOCABULARY:
        return False, f'unit {unit!r} is outside the closed vocabulary'
    return True, 'declared'


def evaluate_ledger(readout, config):
    """Recompute the ledger for one readout against its archived config.

    The **readout's own mirrored declaration** is the source of truth for coverage, because that is
    the run describing itself. The archived config holds the declaration that was made *before* the
    run, and the two are diffed: a readout cannot describe itself differently from how it was set up.

    A readout with no declaration at all predates this contract (legs 1 to 4). It is reported as
    such, and it produces no failures: those legs cannot fail a contract that did not exist.
    """
    cases = readout.get('cases') if isinstance(readout, dict) else None
    declared_all = config.get('units') if isinstance(config, dict) else None
    declared_all = declared_all if isinstance(declared_all, dict) else {}
    mirrored = readout.get('units') if isinstance(readout, dict) else None

    failures = []
    if mirrored is None:
        return dict(cases=[], key_units={}, failures=[], diff='absent', vocabulary=list(VOCABULARY),
                    note=('This readout carries no units declaration. It predates the leg-5 '
                          'contract, so nothing is declared and nothing is claimed for it.'))
    diff = 'match' if mirrored == declared_all else 'mismatch'
    if diff == 'mismatch':
        failures.append('The readout declaration differs from the archived config declaration.')

    key_units = {}
    case_rows = []
    if not isinstance(cases, list):
        return dict(cases=[], key_units={}, failures=failures, diff=diff,
                    vocabulary=list(VOCABULARY),
                    note='This readout has no case list to check.')

    for position, case in enumerate(cases):
        case_id = case.get('id')
        declared = mirrored.get(case_id) if isinstance(mirrored.get(case_id), dict) else {}
        undeclared, unknown, cross_checked, exempt_used = [], [], [], []
        seen = set()
        for path, _value in _numeric_leaves(case):
            key = leaf_key(path)
            if key in seen:
                continue
            seen.add(key)
            entry = declared.get(key)
            if entry is None:
                if key in EXEMPT:
                    unit, reason = EXEMPT[key]
                    exempt_used.append(dict(key=key, unit=unit, reason=reason))
                    key_units.setdefault(key, set()).add(unit)
                    continue
                undeclared.append(f'cases[{position}].{key}')
                continue
            ok, reason = check_declaration(entry)
            if not ok:
                unknown.append(dict(key=key, reason=reason))
                continue
            unit = entry['unit']
            if unit not in VOCABULARY:
                unknown.append(dict(key=key, reason=f'unit {unit!r} is outside the closed vocabulary'))
                continue
            if key in key_units and unit not in key_units[key]:
                failures.append(f'Key {key!r} carries more than one unit: '
                                f'{sorted(key_units[key])} and {unit!r}.')
            key_units.setdefault(key, set()).add(unit)
            cross_checked.append(dict(key=key, unit=unit, plane=entry['plane'],
                                      span=entry['span'],
                                      normalisation=entry['normalisation'],
                                      cross_check='declared-only',
                                      cross_check_reason=(
                                          'Stop condition 1: the archived readout records no monitor '
                                          'span or plane geometry, so this cannot be cross-checked. '
                                          'It is reported as a declaration, not as a verified fact.')))
        if undeclared:
            failures.append(f'Case {case_id!r} has undeclared numbers: {undeclared}.')
        if unknown:
            failures.append(f'Case {case_id!r} has invalid declarations: {unknown}.')
        case_rows.append(dict(case_id=case_id, declared=cross_checked, exempt=exempt_used,
                              undeclared=undeclared, invalid=unknown))
    return dict(cases=case_rows,
                key_units={key: sorted(units) for key, units in key_units.items()},
                failures=failures, diff=diff, vocabulary=list(VOCABULARY))


def drop_declaration(readout, case_id, key):
    """Negative control 1: remove one declaration from an in-memory copy, keeping the number.

    The measured number stays in the readout, so the coverage check must flag it as undeclared.
    Removing the number as well would only test the config-readout diff, not the coverage rule.
    """
    import copy
    perturbed = copy.deepcopy(readout)
    if isinstance(perturbed.get('units'), dict) and case_id in perturbed['units']:
        perturbed['units'][case_id].pop(key, None)
    return perturbed


def invent_unit(readout, case_id, key, unit='furlong'):
    """Negative control 2: give one declaration a unit outside the closed vocabulary."""
    import copy
    perturbed = copy.deepcopy(readout)
    if isinstance(perturbed.get('units'), dict) and case_id in perturbed['units']:
        entry = perturbed['units'][case_id].get(key)
        if isinstance(entry, dict):
            entry['unit'] = unit
    return perturbed


# ================================================================================================
# Leg 13 (S4): the readout-definition ledger for every registered number, and the standing controls.
#
# Leg 5 made a run declare its own numbers. Runs from before leg 5 declare nothing, so most of the
# project's registered numbers had no stated plane, span or normalisation. This section recovers
# them from the code that produced them -- each entry names the function it was read from -- and
# checks them mechanically against the archived runs. Nothing here is inferred from a paper or a
# vendor. A definition that cannot be recovered is written as UNKNOWN with its reason, never filled.
# ================================================================================================

UNKNOWN = 'UNKNOWN'

# The N8/G8 guided fixture, `run_b9_native_shape_n8.scene`, in the DEVICE frame (x = 0 at the start
# of the first tooth period; the project frame is shifted by `xc`). `end = N * 0.9 + 0.26325` um.
_N8_NEAR = ('near plane: x-normal, at x = -0.80 um (0.8 um before the first tooth, snapped to the '
            'x mesh), centred at y = +0.12 um')
_N8_NEAR_SPAN = ('y from -1.68 to +1.92 um (3.6 um: the 0.22 um core and 1.7-1.9 um of cladding '
                 'either side) by the full periodic z cell (z_cells x 0.04 um)')
_N8_FAR = 'far plane: x-normal, at x = end + 0.80 um (snapped to the x mesh), centred at y = +0.12 um'
_N8_TOP = 'top plane: y-normal, at y = +1.50 um (inside the 2 um top cladding, 0.7 um below the air)'
_N8_TOP_SPAN = ('x from the near plane (-0.80 um) to end + 0.80 um, unsnapped (9.06325 um for N = 8) '
                'by the full periodic z cell; power leaving upward outside this x range is not counted')
_MODAL = ('projection onto the selected launch eigenmode, using the fixed detector basis of '
          '`mode_injection.modal_plane_amplitudes` (unit reduced-power mode, E and H, plane quadrature)')

# The parity 1-D family, `parity.project_for` + `parity.spectra` (legs 2 to 5): a one-way plane
# source at x = -2.5 um, periodic transverse faces, one cell in z.
_P_REFL = 'reflection monitor: x-normal, at x = -2.0 um (downstream of the one-way source at -2.5 um)'
_P_TRANS = 'transmission monitor: x-normal, at x = +2.0 um'
_P_SCAT = 'scattered monitor: x-normal, at x = -3.0 um (upstream of the one-way source)'
_P_SPAN = ('the full periodic transverse width (6 cells) by the single z cell: the monitor covers the '
           'whole cell, so no power can pass beside it')

# The leg-7/leg-9 Bloch cell, `leg7_campaign.build_project`.
_A_TRANS = 'transmitted plane: z-normal, at z = +0.6 um (source at z = -0.5 um)'
_A_REFL = 'reflected plane: z-normal, at z = -0.6 um'
_A_SPAN = ('one full Bloch period in x (0.9 um) by the full periodic y cell (0.2 um), Rayleigh orders '
           'm = -2..2 in x')


def _d(unit, plane, span, normalisation, source, family, monitors=()):
    return dict(unit=unit, plane=plane, span=span, normalisation=normalisation, source=source,
                family=family, monitors=list(monitors), status='recovered')


def _parity_block(policy, with_reference_power):
    rows = {
        (policy, 'T'): _d('dimensionless', _P_TRANS, _P_SPAN,
                          'signed sample flux divided by the signed empty-domain flux through the same '
                          'monitor, per frequency (`normalize_flux`, no incident subtraction)',
                          'src/parity.py:spectra', 'parity-1d', ['transmission']),
        (policy, 'R'): _d('dimensionless', _P_REFL, _P_SPAN,
                          'minus the flux of (sample field - empty-domain field) divided by the signed '
                          'empty-domain flux through the same monitor, per frequency '
                          '(`normalize_flux(subtract_incident=True)`, sign flipped)',
                          'src/parity.py:spectra', 'parity-1d', ['reflection']),
        (policy, 'energy_residual'): _d('dimensionless', 'reflection and transmission monitors together',
                                        _P_SPAN, 'max over the sampled band of |R + T - 1|, with R and T '
                                        'as defined on their own monitors; a closure, not an efficiency',
                                        'src/parity.py:spectra', 'parity-1d',
                                        ['reflection', 'transmission']),
        (policy, 'empty_reflection_floor'): _d('dimensionless', _P_SCAT + ', in the EMPTY run', _P_SPAN,
                                               'max over the band of |empty-run scattered-monitor flux| '
                                               'divided by |empty-run REFLECTION-monitor flux| (the '
                                               'denominator is the x = -2.0 um monitor, not the scattered one)',
                                               'src/parity.py:spectra', 'parity-1d',
                                               ['scattered', 'reflection']),
    }
    if with_reference_power:
        rows[(policy, 'reference_power_reduced')] = _d(
            'reduced-unit', _P_REFL + ', in the EMPTY run', _P_SPAN,
            'none: |signed empty-run flux| through the reflection monitor, per frequency, in solver '
            'reduced units. It is the denominator of R and of the floor, and it is not a power in watts',
            'src/parity.py:spectra (denominator)', 'parity-1d', ['reflection'])
    return rows


READOUT_DEFINITIONS = {}
READOUT_DEFINITIONS.update(_parity_block('leg2-invariants-v1', True))
READOUT_DEFINITIONS.update(_parity_block('leg3-analytic-v1', False))
READOUT_DEFINITIONS.update(_parity_block('leg4-convergence-v1', False))
READOUT_DEFINITIONS.update(_parity_block('leg5-units-ledger-v1', True))

for _policy, _source in (('leg1-n8-forward-v1', 'src/leg1_forward.py:main'),
                         ('leg10-true-2d-v1', 'src/leg10_true_2d.py:main')):
    _note = ('' if _policy == 'leg1-n8-forward-v1' else
             '; true 2-D: the z axis is invariant, so the plane carries flux per unit length, which '
             'cancels in this ratio')
    READOUT_DEFINITIONS[(_policy, 'T')] = _d(
        'dimensionless', _N8_FAR + ', numerator; ' + _N8_NEAR + ', denominator', _N8_NEAR_SPAN,
        '|forward modal amplitude at far / forward modal amplitude at near|^2: modal forward power '
        'ratio, ' + _MODAL + '. Not a net-flux ratio and not normalised to the source' + _note,
        _source, 'n8-scene', ['near', 'far'])
    READOUT_DEFINITIONS[(_policy, 'R')] = _d(
        'dimensionless', _N8_NEAR + ', both amplitudes', _N8_NEAR_SPAN,
        '|backward modal amplitude / forward modal amplitude|^2 on the same near plane, ' + _MODAL + _note,
        _source, 'n8-scene', ['near'])
READOUT_DEFINITIONS[('leg1-n8-forward-v1', 'launch_scores')] = _d(
    'dimensionless', 'the launch cross-section of `prepare_aperture_modal_launch` (no monitor)',
    'the sampled transverse mode profile of each of the two candidate modes',
    'sum |Ey|^2 / sum (|Ey|^2 + |Ez|^2) of the mode profile: a polarisation-purity score used to '
    'select the launch mode, not an accuracy claim', 'src/leg1_forward.py:main', 'n8-scene')

_LEG1 = 'leg1-objective-v1'
READOUT_DEFINITIONS[(_LEG1, 'P_up_reduced')] = _d(
    'reduced-unit', _N8_TOP, _N8_TOP_SPAN,
    'none: the signed net Poynting flux through the plane (`planes["top"].flux()`, +y counts as '
    'upward), in solver reduced units, single frequency; all directions crossing the plane count',
    'src/leg1_objective.py:main', 'n8-scene', ['top'])
READOUT_DEFINITIONS[(_LEG1, 'unit_mode_flux_reduced')] = _d(
    'reduced-unit', _N8_NEAR + ' (the detector basis sampled on it)', _N8_NEAR_SPAN,
    '0.5 Re sum (Ey Hz* - Ez Hy*) w of the unit reduced-power detector mode, with the plane quadrature '
    'weights w. Its distance from 1 is the part of the unit mode the finite span and the sampling '
    'miss; it is a property of the plane, not of the device',
    'src/leg1_objective.py:main', 'n8-scene', ['near'])
READOUT_DEFINITIONS[(_LEG1, 'P_in_reduced')] = _d(
    'reduced-unit', _N8_NEAR, _N8_NEAR_SPAN,
    '|forward modal amplitude at near|^2 x unit_mode_flux_reduced: the launched FORWARD power in the '
    'launch mode only (' + _MODAL + '). Backward and non-modal power at the plane are excluded. It is '
    'not the source power and not the net flux',
    'src/leg1_objective.py:main', 'n8-scene', ['near'])
READOUT_DEFINITIONS[(_LEG1, 'J')] = _d(
    'dimensionless', 'numerator: ' + _N8_TOP + '; denominator: ' + _N8_NEAR,
    'numerator: ' + _N8_TOP_SPAN + '; denominator: ' + _N8_NEAR_SPAN,
    'P_up_reduced / P_in_reduced: net upward flux over modal forward input. Two different planes and '
    'two different kinds of normalisation (net flux over modal power), by construction',
    'src/leg1_objective.py:main', 'n8-scene', ['top', 'near'])
READOUT_DEFINITIONS[(_LEG1, 'launch_neff')] = _d(
    'dimensionless', 'the launch cross-section of `prepare_aperture_modal_launch` (no monitor)',
    'the transverse cross-section the eigenmode solver samples', 'none: the eigenvalue n_eff of the '
    'selected launch mode', 'src/leg1_objective.py:main', 'n8-scene')

for _policy in ('leg6-fibre-overlap-v1',):
    for _key, _what in (('eta_lat', 'at the cited W = 12 um and w_lat = 5.2 um'),
                        ('model_peak_eta_lat', 'at the width that maximises it for w_lat = 5.2 um')):
        READOUT_DEFINITIONS[(_policy, _key)] = _d(
            'dimensionless', 'not applicable: a closed-form model evaluation, no monitor and no run',
            'the lateral coordinate z from -max(3 w_lat, W/2) to +max(3 w_lat, W/2), 200001 points',
            '|sum u g*|^2 / (sum |u|^2 sum |g|^2): both fields normalised to unit power on the same '
            'lateral grid, ' + _what + '. One factor of a fibre efficiency, not the efficiency',
            'src/coupling.py:lateral_overlap', 'model')

_L7 = 'leg7-asm-campaign-v1'
for _key, _norm, _mons in (
        ('forward_total', 'none: the sum of forward order powers m = -2..2 on the transmitted plane '
         '(`diffraction_orders(...).forward_power`), solver reduced units, per Bloch phase', ['transmitted']),
        ('backward_total', 'none: the sum of backward order powers m = -2..2 on the reflected plane, '
         'solver reduced units, per Bloch phase', ['reflected']),
        ('delivered_total', 'none: forward_total + backward_total, per Bloch phase; the power the soft '
         'source delivered at that phase', ['transmitted', 'reflected'])):
    READOUT_DEFINITIONS[(_L7, _key)] = _d('reduced-unit', _A_TRANS if _mons == ['transmitted'] else
                                          (_A_REFL if _mons == ['reflected'] else _A_TRANS + '; ' + _A_REFL),
                                          _A_SPAN, _norm, 'src/leg7_campaign.py:main', 'asm-cell', _mons)
READOUT_DEFINITIONS[(_L7, 'energy_spread')] = _d(
    'reduced-unit', _A_TRANS + '; ' + _A_REFL, _A_SPAN + '; over the 8 registered Bloch phases',
    'none: max - min of delivered_total over the phases, solver reduced units',
    'src/leg7_campaign.py:main', 'asm-cell', ['transmitted', 'reflected'])
READOUT_DEFINITIONS[(_L7, 'energy_spread_relative')] = _d(
    'dimensionless', _A_TRANS + '; ' + _A_REFL, _A_SPAN + '; over the 8 registered Bloch phases',
    'energy_spread divided by the mean delivered_total over the same phases',
    'src/leg7_campaign.py:main', 'asm-cell', ['transmitted', 'reflected'])
READOUT_DEFINITIONS[(_L7, 'reconstruction_error')] = _d(
    'dimensionless', _A_TRANS + ' (the profile is reconstructed in this plane, not on a far plane)',
    'the reconstructed line from -ceil(2 w0/d) to +ceil(2 w0/d) cells of 0.9 um',
    'relative RMS of |u| / max|u| against exp(-x^2/w0^2) / max, each phase response first divided by '
    'sqrt(delivered_total) of its own phase', 'src/leg7_campaign.py:main', 'asm-cell', ['transmitted'])
READOUT_DEFINITIONS[(_L7, 'k_fixed_per_um')] = _d(
    'per-micrometre', 'not applicable: arithmetic on the registered phase grid, no monitor',
    'the middle phase of the 8-point grid', 'none: phase / period', 'src/leg7_campaign.py:main', 'model')
READOUT_DEFINITIONS[(_L7, 'drift_deg')] = _d(
    'degree', 'not applicable: arithmetic, no monitor', 'the band 1.50 to 1.60 um, its two ends',
    'asin(k_fixed / k0(1.60 um)) - asin(k_fixed / k0(1.50 um)) in degrees, vacuum k0',
    'src/leg7_campaign.py:main', 'model')

_L9 = 'leg9-grating-angle-v1'
READOUT_DEFINITIONS[(_L9, 'forward_total')] = _d(
    'reduced-unit', _A_TRANS + ', grating run', _A_SPAN,
    'none: the sum of forward order powers m = -2..2, solver reduced units, per angle',
    'src/leg9_grating.py:main', 'asm-cell', ['transmitted'])
READOUT_DEFINITIONS[(_L9, 'zero_order_forward_power')] = _d(
    'reduced-unit', _A_TRANS + ', grating run', _A_SPAN + '; order (0, 0) only',
    'none: the forward power of order (0, 0), solver reduced units, per angle',
    'src/leg9_grating.py:main', 'asm-cell', ['transmitted'])
for _key, _which in (('zero_order_efficiency', 'per angle'), ('zero_order_efficiency_a', 'at angle a'),
                     ('zero_order_efficiency_b', 'at angle b')):
    READOUT_DEFINITIONS[(_L9, _key)] = _d(
        'dimensionless', _A_TRANS + ', grating run over the empty run at the same angle',
        _A_SPAN + '; order (0, 0) only',
        'order-(0, 0) forward power of the grating run divided by |net flux| of the EMPTY run through '
        'the same plane (`diffraction_efficiency`, both fields scaled by the empty run\'s peak), ' + _which,
        'src/leg9_grating.py:main', 'asm-cell', ['transmitted'])

# The monitor geometry each family's code builds, stated as numbers so it can be cross-checked
# against the Project the code actually constructs (device frame; 'full' = the whole region axis;
# 'end+' = the fixture's `end` plus the number; None = not checked on that axis).
DECLARED_GEOMETRY = {
    'n8-scene': {
        'near': dict(normal='x', center=(-0.80, 0.12, 0.0), size=(0.0, 3.6, 'full'), snapped=True),
        'far': dict(normal='x', center=('end+', 0.80, 0.12, 0.0), size=(0.0, 3.6, 'full'), snapped=True),
        'top': dict(normal='y', center=('mid', 1.5, 0.0), size=('near-far', 0.0, 'full'), snapped=False),
    },
    'parity-1d': {
        'reflection': dict(normal='x', center=(-2.0, 0.0, 0.0), size=(0.0, 'full', 'full'), snapped=True),
        'transmission': dict(normal='x', center=(2.0, 0.0, 0.0), size=(0.0, 'full', 'full'), snapped=True),
        'scattered': dict(normal='x', center=(-3.0, 0.0, 0.0), size=(0.0, 'full', 'full'), snapped=True),
    },
    'asm-cell': {
        'transmitted': dict(normal='z', center=(0.0, 0.0, 0.6), size=('full', 'full', 0.0), snapped=False),
        'reflected': dict(normal='z', center=(0.0, 0.0, -0.6), size=('full', 'full', 0.0), snapped=False),
    },
}


def check_registered(entry):
    """(ok, reason) for one registry entry: a complete declaration, or an honest UNKNOWN."""
    if not isinstance(entry, dict):
        return False, 'entry is not an object'
    if entry.get('status') == 'unknown':
        return (bool(str(entry.get('reason', '')).strip()),
                'unknown, with a reason' if str(entry.get('reason', '')).strip()
                else 'unknown without a reason')
    ok, reason = check_declaration(entry)
    if not ok:
        return ok, reason
    for field in DECLARATION_FIELDS[1:]:
        if UNKNOWN in entry[field]:
            return False, f'{field!r} contains {UNKNOWN} but the entry is not marked unknown'
    if not str(entry.get('source', '')).strip():
        return False, 'no source: a recovered definition must name the code it was read from'
    return True, 'recovered'


def audit_definitions(observed, mirrored=None, registry=None):
    """C1, mechanically: every observed (policy, field) must carry a definition.

    `observed` is an iterable of (policy, field) pairs taken from the archived readouts.
    `mirrored` maps (policy, field) to the declaration a run wrote into its own readout (leg 5
    onwards). A pair is defined if its own readout declares it validly, or if the registry holds a
    valid entry for it. A registry UNKNOWN counts as answered but is listed separately, so an
    unknown can never be mistaken for a definition.
    """
    registry = READOUT_DEFINITIONS if registry is None else registry
    mirrored = mirrored or {}
    rows = []
    for pair in sorted(set(observed)):
        own = mirrored.get(pair)
        own_ok = own is not None and check_declaration(own)[0]
        entry = registry.get(pair)
        reg_ok, reg_reason = check_registered(entry) if entry is not None else (False, 'no registry entry')
        if entry is not None and entry.get('status') == 'unknown' and reg_ok:
            status = 'unknown'
        elif own_ok and reg_ok:
            status = 'declared-and-registered'
        elif reg_ok:
            status = 'registered'
        elif own_ok:
            status = 'declared'
        else:
            status = 'missing' if entry is None else 'invalid'
        rows.append(dict(policy=pair[0], field=pair[1], status=status,
                         reason=reg_reason if status in ('missing', 'invalid') else ''))
    count = {key: sum(1 for row in rows if row['status'] == key)
             for key in ('declared-and-registered', 'registered', 'declared', 'unknown', 'missing',
                         'invalid')}
    return dict(rows=rows, count=count, total=len(rows),
                passed=count['missing'] == 0 and count['invalid'] == 0)


def cross_check_monitor(spec, monitor, region_size, frame_offset=(0.0, 0.0, 0.0), cell_um=0.0,
                        end_um=None, near_x=None, far_x=None, tolerance_um=1e-9):
    """Compare one declared monitor geometry with the monitor the code actually builds.

    `monitor` is (normal, center, size) in the PROJECT frame; `frame_offset` is added to the centre
    to reach the device frame the declaration uses. A snapped plane may sit up to half a cell from
    its nominal position along its normal; everything else must agree to `tolerance_um`.
    Returns a list of disagreements; empty means the declaration matches the built project.
    """
    normal, center, size = monitor
    problems = []
    if normal != spec['normal']:
        problems.append(f'normal {normal!r} != declared {spec["normal"]!r}')
    axis = 'xyz'.index(spec['normal'])
    declared_center = spec['center']
    if declared_center[0] == 'end+':
        declared_center = (end_um + declared_center[1],) + tuple(declared_center[2:])
    elif declared_center[0] == 'mid':
        declared_center = ((near_x + far_x) / 2,) + tuple(declared_center[1:])
    for i in range(3):
        actual = center[i] + frame_offset[i]
        slack = tolerance_um + (cell_um / 2 if spec.get('snapped') and i == axis else 0.0)
        if abs(actual - declared_center[i]) > slack:
            problems.append(f'centre[{"xyz"[i]}] {actual:.9g} != declared {declared_center[i]:.9g} '
                            f'(slack {slack:.3g})')
    for i in range(3):
        want = spec['size'][i]
        if want == 'full':
            want = region_size[i]
        elif want == 'near-far':
            want = far_x - near_x
        if abs(size[i] - want) > tolerance_um:
            problems.append(f'size[{"xyz"[i]}] {size[i]:.9g} != declared {want:.9g}')
    return problems


# ------------------------------------------------------------------------------------------------
# The standing control set (C2). Pure arithmetic on what a run measured; a control the run did not
# measure is reported as not measured, never as passed.
# ------------------------------------------------------------------------------------------------

def energy_closure(R, T, tolerance):
    """T + R = 1 in a lossless, fully monitored cell: max |R + T - 1| over the band."""
    residual = float(max(abs(r + t - 1.0) for r, t in zip(R, T)))
    return dict(control='T+R=1', value=residual, tolerance=tolerance, passed=bool(residual <= tolerance),
                margin=tolerance / residual if residual > 0 else None)


def symmetric_source_ratio(forward, backward, tolerance, min_fraction=0.01):
    """A mirror-symmetric source in a homogeneous cell sends equal power both ways (`1.000x`).

    `forward` is the +x flux on the +x monitor, `backward` the -x flux magnitude on the mirror
    monitor, per frequency. A frequency where NEITHER side carries `min_fraction` of the peak power
    is dropped (the threshold rule `normalize_flux` applies), and the count dropped is reported. A
    frequency where only one side carries power is kept: that is exactly what a one-way source does.
    """
    peak = max(max(abs(f) for f in forward), max(abs(b) for b in backward))
    used = [(f, b) for f, b in zip(forward, backward) if max(abs(f), abs(b)) > min_fraction * peak]
    if not used:
        return dict(control='symmetric-source', passed=False, reason='no frequency carries power')
    if any(b == 0 for _f, b in used):
        return dict(control='symmetric-source', passed=False, deviation=None, value=None,
                    tolerance=tolerance, reason='a used frequency has exactly zero backward power',
                    frequencies_used=len(used), frequencies_dropped=len(forward) - len(used))
    ratios = [float(f) / float(b) for f, b in used]
    worst = max(ratios, key=lambda q: abs(q - 1.0))
    deviation = abs(worst - 1.0)
    return dict(control='symmetric-source', value=worst, deviation=deviation, tolerance=tolerance,
                passed=bool(deviation <= tolerance), frequencies_used=len(used),
                frequencies_dropped=len(forward) - len(used),
                margin=tolerance / deviation if deviation > 0 else None)


# ------------------------------------------------------------------------------------------------
# C3 and C4: the two inherited terms, recorded from the material with their evidence class.
# ------------------------------------------------------------------------------------------------

# The `5.6 pt` term. Material: the predecessor project `20261007_efficiency-optimization`,
# `output/phase1-paper-caliber.md`, section (e) E.2 row 8 and section (a), and its frozen vendor copy
# `refs/lumerical-reference.json`; re-examined in $PARITY_REF `plan.md` steps 25a-25e and A2.
TRANSITION_TERM = {
    'definition': 'transition-region loss = 1 - b_in / b_src_in, in percentage points of input',
    'ours': dict(
        b_src_in=5.691739e-30, b_in=4.848624e-30,
        plane_numerator='b_in: x-normal plane at x = 51.842 um (the grating start)',
        plane_denominator='b_src_in: x-normal plane at x = -3.5 um (the input slab)',
        span='1.5 um in y centred at y = 0.11 um, both planes',
        normalisation='net flux, late window from 6 ps; 2-D, dx = 20 nm, constant index',
        evidence='D (frozen values in the predecessor record, R16 lines 130 and 134)'),
    'vendor': dict(
        b_src_in_ratio=1.1014,
        plane_denominator=UNKNOWN, span=UNKNOWN,
        normalisation=UNKNOWN,
        reason=('The vendor value is a ratio whose basis is unrecorded: the predecessor states that '
                'whether 1.1014 is referred to the net or to the modal b_in is not recorded, and no '
                'plane position or span is recorded for it at all. The vendor run stopped at an '
                'unconverged 5 ps. Evidence class X.'),
        candidate_bases_named_in_the_material=dict(
            net=dict(RE_total_flux_pct=71.450),
            modal=dict(RE_mode_pct=74.277))),
    'region_in_this_repository': ('absent. Every guided fixture here (N8, G8) starts with the grating '
                                  'at x = 0 and has no transition or taper section, so no number in '
                                  'this repository measures this term.'),
}


def transition_term_verdict(term=TRANSITION_TERM):
    """C3: bound the term, or declare it unbounded and say why. Arithmetic only.

    The candidate bases are those the material itself names. They do NOT form a bound, because the
    vendor's plane and span are unknown too and nothing says the basis is one of the two; they are
    reported as a conditional envelope so the size of the ambiguity is visible.
    """
    ours = term['ours']
    our_loss = 1.0 - ours['b_in'] / ours['b_src_in']
    ratio = term['vendor']['b_src_in_ratio']
    net = term['vendor']['candidate_bases_named_in_the_material']['net']['RE_total_flux_pct']
    modal = term['vendor']['candidate_bases_named_in_the_material']['modal']['RE_mode_pct']
    vendor_net = 1.0 - 1.0 / ratio
    # If 1.1014 were referred to the modal b_in, re-express it against the net b_in:
    # RE_mode / RE_total = b_in_net / b_in_modal.
    vendor_modal = 1.0 - 1.0 / (ratio * net / modal)
    unknown = [key for key in ('plane_denominator', 'span', 'normalisation')
               if term['vendor'][key] == UNKNOWN]
    return dict(
        our_loss_pt=100 * our_loss,
        vendor_loss_pt_if_net_basis=100 * vendor_net,
        vendor_loss_pt_if_modal_basis=100 * vendor_modal,
        difference_pt_if_net_basis=100 * (our_loss - vendor_net),
        difference_pt_if_modal_basis=100 * (our_loss - vendor_modal),
        vendor_fields_unknown=unknown,
        verdict='unbounded' if unknown else 'bounded',
        statement=('UNBOUNDED, and declared so: a limit of the comparison, not a result. The vendor '
                   'side has no recorded plane, span or normalisation basis, so no interval can be '
                   'put around the difference without inventing one. Under the two bases the '
                   'material itself names, the difference would be the two numbers above; that is '
                   'a conditional envelope, not a bound.') if unknown else 'bounded')


# The `~40 percent` radiation-strength term. Material: $PARITY_REF `plan.md` step 25e item 1 and its
# withdrawal in step 26b (`tools/alpha_refit.py`); the predecessor record, E.2 row 4.
RADIATION_TERM = {
    'claim': 'alpha is ~40 percent above the paper design (4.15-4.33 Np/mm against 3 Np/mm)',
    'attribution': 'the predecessor project\'s separate FDTD implementation, not this solver line',
    'withdrawn_at': '$PARITY_REF plan.md step 26b: "there is no 40 percent excess in this solver"',
    'reference_solver_alpha_np_per_mm': [2.87, 3.05],
    'physics_bound': dict(far_end_T=0.0238, length_mm=0.5508,
                          rule='alpha <= -ln(T_far) / (2 L): any loss other than radiation only '
                               'lowers T, so this can only over-estimate a uniform alpha'),
    'in_this_repository': 'no number here measures alpha: N8 is eight periods, not a decay length',
}


def radiation_term_bound(term=RADIATION_TERM):
    bound = term['physics_bound']
    alpha_max = -math.log(bound['far_end_T']) / (2.0 * bound['length_mm'])
    low, high = 4.15, 4.33
    return dict(alpha_upper_bound_np_per_mm=alpha_max,
                excludes_claimed_range=alpha_max < low,
                claimed_range=[low, high])
