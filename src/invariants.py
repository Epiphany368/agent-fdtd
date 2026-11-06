"""Registered V2 invariant arithmetic and conservative V-0 label decisions.

This module imports no solver. It recomputes from measured values and the run's
archived config, never from stored checks, residuals, or pass/fail flags.
Tolerances are dimensionless absolute errors for a/b, and relative errors for
c/d. Missing required monitors drop a check; no monitor is added here.
"""
import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
import sys

from verification import verify_readout


IDS = ('V2-a', 'V2-b', 'V2-c', 'V2-d', 'V2-box')
_MISSING = object()


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _samples(value, path):
    """Return numeric leaf paths and values, including scalar readouts."""
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_samples(child, f'{path}[{index}]'))
        return rows
    return [(path, value)]


def _shape(value):
    if not isinstance(value, list):
        return ()
    return tuple(_shape(child) for child in value)


def _pair(measured, names, prefix):
    values = [measured.get(name, _MISSING) for name in names]
    rows = [_samples(value, f'{prefix}.{name}') if value is not _MISSING else []
            for name, value in zip(names, values)]
    covers = [path for group in rows for path, value in group
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if any(value is _MISSING for value in values):
        return covers, None, 'Required monitor/readout data is missing; check dropped (stop condition 1).'
    if (any(not group for group in rows) or _shape(values[0]) != _shape(values[1])):
        return covers, None, 'Paired monitor arrays must be nonempty and have the same shape.'
    pairs = list(zip([value for _, value in rows[0]], [value for _, value in rows[1]]))
    if not all(_finite(value) for pair in pairs for value in pair):
        return covers, None, 'Every required monitor value must be a finite number.'
    return covers, pairs, None


def _box_quad(measured, prefix):
    """Extract the 4 box surface monitors (right, left, top, bottom) from measured data."""
    right_names = ('box-right', 'box_right', 'transmission', 'trans', 'far', 'T', 'right')
    left_names = ('box-left', 'box_left', 'reflection', 'refl', 'near', 'R', 'left')
    top_names = ('box-top', 'box_top', 'top', 'up', 'cladding')
    bottom_names = ('box-bottom', 'box_bottom', 'bottom', 'down', 'substrate')

    candidate_groups = [right_names, left_names, top_names, bottom_names]
    found_values = []
    found_paths = []

    monitors_dict = measured.get('monitors') if isinstance(measured.get('monitors'), Mapping) else None

    for group in candidate_groups:
        matched_val = _MISSING
        matched_path = None
        if monitors_dict is not None:
            for name in group:
                if name in monitors_dict:
                    raw = monitors_dict[name]
                    matched_val = raw.get('flux', raw) if isinstance(raw, Mapping) else raw
                    matched_path = f'{prefix}.monitors.{name}'
                    break
        if matched_val is _MISSING:
            for name in group:
                if name in measured:
                    raw = measured[name]
                    matched_val = raw.get('flux', raw) if isinstance(raw, Mapping) else raw
                    matched_path = f'{prefix}.{name}'
                    break
        found_values.append(matched_val)
        found_paths.append(matched_path)

    rows = [_samples(value, path) if value is not _MISSING else []
            for value, path in zip(found_values, found_paths)]
    covers = [path for group in rows for path, value in group
              if isinstance(value, (int, float)) and not isinstance(value, bool)]

    if any(value is _MISSING for value in found_values):
        missing_names = [group[0] for value, group in zip(found_values, candidate_groups) if value is _MISSING]
        return covers, None, f'Required 4-surface box monitor data is missing ({", ".join(missing_names)}); check dropped (stop condition 1).'

    shapes = [_shape(val) for val in found_values]
    if any(not group for group in rows) or len(set(shapes)) != 1:
        return covers, None, '4-surface box monitor arrays must be nonempty and have the same shape.'

    quads = list(zip(
        [value for _, value in rows[0]],
        [value for _, value in rows[1]],
        [value for _, value in rows[2]],
        [value for _, value in rows[3]],
    ))
    if not all(_finite(value) for quad in quads for value in quad):
        return covers, None, 'Every required box monitor value must be a finite number.'

    return covers, quads, None


def _lawful(invariant, case, registration, readout):
    physics = case.get('physics', {})
    if not isinstance(physics, Mapping):
        physics = {}
    geometry = case.get('geometry')
    if invariant == 'V2-b':
        return True, 'Passivity is lawful on every configuration.'
    if invariant == 'V2-box':
        identity = f'{registration["case_id"]} {readout.get("fixture", "")} {geometry}'.lower()
        guided = 'guided' in identity or re.search(r'(?<![a-z0-9])[ng]8(?![a-z0-9])', identity) or geometry in ('grating', 'guided')
        has_monitors = (physics.get('all_exits_monitored') is True or
                        physics.get('poynting_box') is True or
                        physics.get('box_enclosure') is True or
                        'box' in identity)
        lawful = bool(physics.get('lossless') is True and (guided or has_monitors or geometry in ('empty', 'slab', 'interface', 'grating', 'guided')))
        return lawful, ('Lossless structure with 4-surface enclosing box.' if lawful else
                        'V2-box requires a lossless structure with 4-surface enclosing box monitored.')
    if invariant == 'V2-a':
        identity = f'{registration["case_id"]} {readout.get("fixture", "")} {geometry}'.lower()
        guided = 'guided' in identity or re.search(r'(?<![a-z0-9])[ng]8(?![a-z0-9])', identity)
        lawful = (not guided and geometry in ('empty', 'slab', 'interface')
                  and physics.get('family') == 'closed-lossless-plane'
                  and physics.get('lossless') is True
                  and physics.get('all_exits_monitored') is True)
        return lawful, ('Closed lossless plane case with every exit monitored.' if lawful else
                        'Energy balance is unlawful here: requires a closed lossless empty/slab/interface '
                        'case with all exits monitored; guided N8/G8 are explicitly excluded.')
    if invariant == 'V2-c':
        lawful = (geometry in ('empty', 'slab') and physics.get('symmetric_source') is True
                  and physics.get('symmetric_case') is True)
        return lawful, ('Declared symmetric empty/slab case and source.' if lawful else
                        'Equal directional power requires a declared symmetric empty/slab case and source.')
    lawful = physics.get('linear_reciprocal') is True
    ports = registration.get('ports')
    exchanged = (isinstance(ports, Mapping) and
                 all(isinstance(ports.get(key), Mapping) for key in ('forward', 'reverse')))
    if exchanged:
        forward, reverse = ports['forward'], ports['reverse']
        exchanged = (isinstance(forward.get('source'), str) and bool(forward['source']) and
                     isinstance(forward.get('monitor'), str) and bool(forward['monitor']) and
                     forward['source'] != forward['monitor'] and
                     forward['source'] == reverse.get('monitor') and
                     forward['monitor'] == reverse.get('source'))
    return lawful and exchanged, ('Linear reciprocal case with registered exchanged source/monitor ports.'
                                    if lawful and exchanged else
                                    'Reciprocity requires linear reciprocal physics and explicit exchanged ports.')


def _energy_leaves(row, measured, prefix, closure):
    """Leg 12: wire V2-a to the two scalar energy fields its arithmetic determines.

    ``energy_residual`` is the stored copy of max|R+T-1|. It is covered only when
    it equals, bit for bit, the closure V2-a has just recomputed from R and T (the
    same float64 expression; JSON round-trips a float64 exactly). A mismatching
    copy is never granted: it is listed in ``rejected`` and blocks its label.

    ``empty_reflection_floor`` is a measured flux ratio: the matched empty
    reference's flux through the plane behind the one-way source, over the
    reference denominator. It is the one exit R+T does not count, so V2-a's
    "every exit monitored" premise holds only if it is within the same budget.
    It therefore enters the V2-a error, and is covered because it entered it.
    Returns the combined error, or None after marking the row failed.
    """
    error = closure
    floor = measured.get('empty_reflection_floor', _MISSING)
    if floor is not _MISSING:
        path = f'{prefix}.empty_reflection_floor'
        if not _finite(floor):
            row.update(status='failed', error=math.inf, covers=row['covers'] + [path],
                       reason='empty_reflection_floor must be a finite number.')
            return None
        row['covers'].append(path)
        error = max(error, abs(floor))
        row['reason'] += f' Unmonitored backward-exit leakage (empty_reflection_floor) {abs(floor):.17g}.'
    stored = measured.get('energy_residual', _MISSING)
    if stored is not _MISSING:
        path = f'{prefix}.energy_residual'
        if _finite(stored) and stored == closure:
            row['covers'].append(path)
        else:
            row.setdefault('rejected', []).append(dict(
                path=path, reason=f'Stored energy_residual {stored!r} differs from the closure '
                                  f'{closure!r} recomputed from R and T; a stored residual is not trusted.'))
    row['reason'] += f' Recomputed R+T closure {closure:.17g}.'
    return error


def evaluate_invariants(readout, config=None):
    """Return one recomputed row per config registration; never mutate inputs.

    Registration: config.invariants[] = {id, case_id, tolerance, basis}; lawful
    physics lives in config.cases[].physics. V2-c reads measured.forward_power
    and backward_power. V2-d reads measured.reciprocity.forward/reverse and
    requires matching measured.reciprocity.ports and registration.ports, each
    {forward: {source, monitor}, reverse: {source, monitor}}.
    """
    if not isinstance(config, Mapping):
        return []
    registrations = config.get('invariants', [])
    if not isinstance(registrations, list):
        raise ValueError('Archived invariants must be a list of registrations.')
    cases = readout.get('cases', [])
    case_map = {case.get('id'): (index, case) for index, case in enumerate(cases)
                if isinstance(case, Mapping)}
    config_cases = config.get('cases', [])
    if not isinstance(config_cases, list):
        config_cases = []
    declared = {case.get('id'): case for case in config_cases if isinstance(case, Mapping)}
    results = []
    seen = set()
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError('Every invariant registration must be an object.')
        invariant, case_id = registration.get('id'), registration.get('case_id')
        row = dict(id=invariant, case_id=case_id, status='invalid', error=None,
                   tolerance=registration.get('tolerance'), basis=registration.get('basis'),
                   lawful=False, covers=[], reason='Invalid invariant registration.')
        results.append(row)
        if invariant not in IDS or not isinstance(case_id, str) or not case_id:
            continue
        key = (invariant, case_id)
        if key in seen:
            raise ValueError(f'Duplicate invariant registration: {invariant} on {case_id}.')
        seen.add(key)
        if case_id not in case_map or case_id not in declared:
            row['reason'] = 'Registered case is absent from the archived config or readout.'
            continue
        index, measured_case = case_map[case_id]
        measured = measured_case.get('measured', {})
        if not isinstance(measured, Mapping):
            row['reason'] = 'Measured case must be an object.'
            continue
        prefix = f'cases[{index}].measured'
        if invariant in ('V2-a', 'V2-b'):
            names = ('R', 'T')
            covers, pairs, data_error = _pair(measured, names, prefix)
        elif invariant == 'V2-box':
            covers, quads, data_error = _box_quad(measured, prefix)
        elif invariant == 'V2-c':
            names = ('forward_power', 'backward_power')
            covers, pairs, data_error = _pair(measured, names, prefix)
        else:
            measured = measured.get('reciprocity', {})
            if not isinstance(measured, Mapping):
                measured = {}
            names, prefix = ('forward', 'reverse'), prefix + '.reciprocity'
            covers, pairs, data_error = _pair(measured, names, prefix)
        row['covers'] = covers
        if not config.get('policy') or config.get('policy') != readout.get('policy'):
            row['reason'] = 'Archived config policy does not match this readout.'
            continue
        tolerance, basis = row['tolerance'], row['basis']
        if not _finite(tolerance) or tolerance < 0 or not isinstance(basis, str) or not basis.strip():
            row['reason'] = 'One finite nonnegative tolerance and a nonempty basis must be pre-declared.'
            continue
        row['lawful'], row['reason'] = _lawful(invariant, declared[case_id], registration, readout)
        if not row['lawful']:
            row['status'] = 'dropped'
            continue
        if data_error:
            row['status'] = 'dropped' if 'missing' in data_error else 'failed'
            row['reason'] = data_error
            if row['status'] == 'failed':
                row['error'] = math.inf
            continue
        if invariant == 'V2-a':
            error = max(abs(reflection + transmission - 1) for reflection, transmission in pairs)
            error = _energy_leaves(row, measured, prefix, error)
            if error is None:
                continue
        elif invariant == 'V2-box':
            p_inc = measured.get('P_in', measured.get('incident_power', measured.get('incident_flux', 1.0)))
            if isinstance(p_inc, (int, float)) and not isinstance(p_inc, bool):
                inc_list = [float(p_inc)] * len(quads)
            elif isinstance(p_inc, list) and len(p_inc) == len(quads):
                inc_list = [float(x) for x in p_inc]
            else:
                inc_list = [1.0] * len(quads)
            error = max(abs((abs(r) + abs(l) + abs(t) + abs(b)) / inc - 1.0)
                        for (r, l, t, b), inc in zip(quads, inc_list))
            stored = measured.get('box_closure_residual', _MISSING)
            if stored is not _MISSING:
                path = f'{prefix}.box_closure_residual'
                if _finite(stored) and abs(stored - error) < 1e-12:
                    row['covers'].append(path)
                else:
                    row.setdefault('rejected', []).append(dict(
                        path=path, reason=f'Stored box_closure_residual {stored!r} differs from recomputed '
                                          f'closure {error!r}; stored residual is not trusted.'))
            row['reason'] += f' Recomputed 4-surface box closure {error:.17g}.'
        elif invariant == 'V2-b':
            error = max(max(0., -value, value - 1.) for pair in pairs for value in pair)
        elif invariant == 'V2-c':
            if any(forward <= 0 or backward <= 0 for forward, backward in pairs):
                error = math.inf
            else:
                error = max(abs(forward / backward - 1) for forward, backward in pairs)
        else:
            if measured.get('ports') != registration.get('ports'):
                row['status'], row['reason'] = 'dropped', 'Measured exchanged ports do not match registration.'
                continue
            error = max(abs(forward - reverse) / max(abs(forward), abs(reverse))
                        if max(abs(forward), abs(reverse)) else 0. for forward, reverse in pairs)
        row['error'] = error
        row['status'] = 'passed' if math.isfinite(error) and error <= tolerance else 'failed'
        convention = 'absolute dimensionless' if invariant in ('V2-a', 'V2-b', 'V2-box') else 'relative'
        row['reason'] += f' Recomputed maximum {convention} error {error:.17g}; tolerance {tolerance:.17g}.'
    return results


def verify_invariants(readout, config=None):
    """Keep legacy labels and promote only exact leaves covered by passed V2.

    A failed/invalid covering registration blocks promotion even if another
    covering check passes. A deliberately dropped unlawful check cannot verify.
    """
    headlines = verify_readout(readout, config)
    checks = evaluate_invariants(readout, config)
    for headline in headlines:
        covering = [check for check in checks if headline['path'] in check['covers']]
        failed = [check for check in covering if check['status'] in ('failed', 'invalid')]
        passed = [check for check in covering if check['status'] == 'passed']
        rejected = [entry['reason'] for check in checks for entry in check.get('rejected', [])
                    if entry['path'] == headline['path']]
        if rejected:
            headline['label'] = 'unchecked'
            headline['reason'] = 'Not granted: ' + ' '.join(rejected)
        elif failed:
            headline['label'] = 'unchecked'
            headline['reason'] = 'Covering registered invariant failed or is invalid: ' + ', '.join(
                f'{check["id"]} ({check["reason"]})' for check in failed)
        elif passed:
            headline['label'] = 'verified'
            headline['invariants'] = [check['id'] for check in passed]
            headline['reason'] = 'Covered by recomputed registered invariant(s): ' + ', '.join(
                f'{check["id"]} on {check["case_id"]} (error {check["error"]:.17g} <= '
                f'{check["tolerance"]:.17g}; basis: {check["basis"]})' for check in passed)
    return headlines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Existing run directory (read-only input).')
    parser.add_argument('--negative-control', action='store_true',
                        help='Corrupt one copied transmission value in memory; not a simulation.')
    args = parser.parse_args(argv)
    if not args.negative_control:
        parser.error('Specify --negative-control.')
    try:
        config = json.loads((args.run / 'config.input.json').read_text())
        readout = json.loads((args.run / 'readout.json').read_text())
        case = next(case for case in readout['cases']
                    if isinstance(case.get('measured', {}).get('T'), list)
                    and case['measured']['T'])
        before = case['measured']['T'][0]
        case['measured']['T'][0] = 1.25
        print(f'Negative control: case={case["id"]} T[0] {before!r} -> 1.25 (in-memory copy only).')
        checks = evaluate_invariants(readout, config)
        for row in checks:
            print(f'{row["id"]} case={row["case_id"]} status={row["status"]} '
                  f'error={row["error"]!r} tolerance={row["tolerance"]!r}')
            print(row['reason'])
        failed = any(row['status'] == 'failed' for row in checks)
        print(f'Negative control detected: {failed}. Source run files were not modified.')
        return 1 if failed else 0
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
        parser.exit(2, f'invariants: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
