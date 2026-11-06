#!/usr/bin/env python3
"""Leg 13 (S4): the readout-definition ledger, checked mechanically over every archived run.

What it does, in order:
  C1  every (policy, physics field) the archived runs report must have a plane, span and
      normalisation -- in the run's own readout (leg 5 onwards) or in
      `units_ledger.READOUT_DEFINITIONS`, recovered from the code that produced it;
  C1b every monitor a definition names is rebuilt from the run's ARCHIVED config with the same
      builder the run used, and its normal, centre and size are compared with the declaration;
  C3  the inherited `5.6 pt` term is bounded or declared unbounded, from the frozen material only;
  C4  the `~40 percent` term is attributed from the material before anything is concluded;
  C6  the leg-11 audit is recomputed unchanged, and then with the S4 result applied.

No simulation, no GPU. Builders are called only to construct projects; nothing is run.

Run:  ./tools/run.sh configs/leg13-readout-ledger.json src/leg13_ledger.py
      ./.venv/bin/python src/leg13_ledger.py            (dry run, nothing written)
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

# Same split as leg 11: code from the snapshot, data from the repository the run belongs to.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'src'))
_pinned = os.environ.get('RUN_DIR_ABS')
ROOT = Path(_pinned).resolve().parents[1] if _pinned else CODE_ROOT

import coverage_audit as ca
import units_ledger as ul

# The pinned scope: every run leg 11 inventoried (run ids up to and including the leg-11 audit run),
# plus this leg's own runs. Later runs from legs running in parallel are listed, not judged: their
# own leg owns their declarations.
LEG11_AUDIT_RUN = '20261018T131254Z-619690'
OWN_POLICIES = ('leg13-standing-controls-v1',)
EXCLUDED_POLICIES = ('leg11-coverage-audit-v1', 'leg13-readout-ledger-v1')


def in_scope(run, policy):
    return run.name <= LEG11_AUDIT_RUN or policy in OWN_POLICIES


def observe(runs):
    """(policy, physics field) -> run names, and the run's own mirrored declarations."""
    observed, mirrored, scope_of = {}, {}, {}
    for run in runs:
        readout, headlines = ca.audit_run(run)
        policy = readout.get('policy', 'undeclared')
        units = readout.get('units') if isinstance(readout.get('units'), dict) else {}
        for row in headlines:
            name, _kind = ca.family(row['path'])
            if ca.classify(name)[0] not in ('physics', 'unknown'):
                continue
            pair = (policy, name)
            observed.setdefault(pair, set()).add(run.name)
            scope_of.setdefault(pair, set()).add(in_scope(run, policy))
            for case_units in units.values():
                if isinstance(case_units, dict) and name in case_units:
                    mirrored[pair] = case_units[name]
    return observed, mirrored, scope_of


# ---------------------------------------------------------------------------- geometry (C1b) ---

def _monitors(project):
    return {m.id: (m.normal, tuple(m.center), tuple(m.size)) for m in project.monitors}


def build_family(family, policy, config):
    """[(label, monitors, region_size, frame_offset, cell_um, end, near_x, far_x)] for one config."""
    out = []
    if family == 'n8-scene':
        import run_b9_native_shape_n8 as b9
        cases = config.get('cases') or [config['case']]
        for case in cases:
            case = dict(case)
            case.setdefault('transition_nm', case['dx_um'] * 1000)
            sc = b9.scene(config, case)
            shift = case.get('shift_nm', 0) * 1e-3
            out.append((case['id'], _monitors(sc['p']), tuple(sc['r'].actual_size),
                        (sc['xc'], 0.0, 0.0), case['dx_um'], sc['end'], -0.8 + shift,
                        sc['end'] + 0.8))
    elif family == 'parity-1d':
        import parity
        for case in config['cases']:
            kind = case.get('kind') or case.get('geometry')
            args = dict(frequency_points=case.get('frequency_points', config.get('frequency_points', 21)))
            if 'transverse_cells' in case:
                args['transverse_cells'] = case['transverse_cells']
            project = parity.project_for(kind, case['index'], case['ppw'], case['duration_fs'], **args)
            region = project.region
            out.append((case['id'], _monitors(project), tuple(region.actual_size), (0.0, 0.0, 0.0),
                        region.axis_steps[0], None, None, None))
    elif family == 'asm-cell':
        import leg7_campaign as l7
        project = l7.build_project(config, 0.0)
        out.append(('cell', _monitors(project), tuple(project.region.actual_size), (0.0, 0.0, 0.0),
                    config['mesh_um'], None, None, None))
    return out


def cross_check(observed, scope_of, runs_by_name, geometry=None):
    """Rebuild every monitor a definition names from each in-scope run's archived config."""
    geometry = ul.DECLARED_GEOMETRY if geometry is None else geometry
    cache, results = {}, {}
    for pair, names in sorted(observed.items()):
        entry = ul.READOUT_DEFINITIONS.get(pair)
        if entry is None or not entry.get('monitors'):
            continue
        family = entry['family']
        problems, checked = [], 0
        for name in sorted(names):
            run = runs_by_name[name]
            if not in_scope(run, pair[0]):
                continue
            text = (run / 'config.input.json').read_text()
            key = (family, hashlib.sha256(text.encode()).hexdigest())
            if key not in cache:
                cache[key] = build_family(family, pair[0], json.loads(text))
            for label, monitors, size, offset, cell, end, near_x, far_x in cache[key]:
                for monitor_id in entry['monitors']:
                    if monitor_id not in monitors:
                        problems.append(f'{name}/{label}: monitor {monitor_id!r} is not built')
                        continue
                    found = ul.cross_check_monitor(geometry[family][monitor_id], monitors[monitor_id],
                                                   size, offset, cell, end, near_x, far_x)
                    problems += [f'{name}/{label}/{monitor_id}: {p}' for p in found]
                    checked += 1
        results[pair] = dict(checked=checked, problems=problems)
    return results


# --------------------------------------------------------------------------------- C6 overlay ---

def leg11_counts(runs, settled_by_s4=()):
    """The leg-11 numbers, optionally with fields S4 settled moved from open to settled."""
    families, checked, unclassified = ca.audit(runs)
    families.pop('?<unreadable>', None)
    physics = {}
    for name in families:
        role, coverage, leg, cost = ca.classify(name)
        if role != 'physics':
            continue
        if name in settled_by_s4 and coverage == 'open':
            coverage, leg = 'settled', 'S4 (leg 13): plane, span and normalisation pinned'
        physics[name] = (coverage, leg)
    open_ = sorted(n for n, (c, _l) in physics.items() if c == 'open')
    return dict(physics_fields=len(physics),
                covered=sum(1 for c, _l in physics.values() if c == 'covered'),
                open=len(open_), settled=sum(1 for c, _l in physics.values() if c == 'settled'),
                gap_s4=sum(1 for n in open_ if physics[n][1] == 'S4'),
                open_fields=open_, unclassified=len(unclassified))


def main():
    config = json.loads((ROOT / 'configs' / 'leg13-readout-ledger.json').read_text())
    own = Path(os.environ['RUN_DIR_ABS']).resolve() if os.environ.get('RUN_DIR_ABS') else None
    runs = [r for r in ca.discover_runs(ROOT, exclude_policies=EXCLUDED_POLICIES)
            if own is None or r.resolve() != own]
    runs_by_name = {r.name: r for r in runs}
    observed, mirrored, scope_of = observe(runs)
    scoped = {p: n for p, n in observed.items() if True in scope_of[p]}
    later = sorted(p for p in observed if True not in scope_of[p])

    lines = ['Readout-definition ledger (leg 13, S4)',
             f'runs read: {len(runs)}; (policy, physics field) pairs: {len(observed)}; '
             f'in the pinned scope: {len(scoped)}; from later parallel legs: {len(later)}', '']

    # C1
    c1 = ul.audit_definitions(scoped, mirrored)
    lines.append(f'C1 definitions: {c1["count"]} of {c1["total"]} -> '
                 f'{"PASS" if c1["passed"] else "FAIL"}')
    for row in c1['rows']:
        entry = ul.READOUT_DEFINITIONS.get((row['policy'], row['field']), {})
        lines.append(f'  {row["policy"]} | {row["field"]} | {row["status"]} | '
                     f'unit={entry.get("unit", "-")} | plane={entry.get("plane", "-")} | '
                     f'span={entry.get("span", "-")} | normalisation={entry.get("normalisation", "-")} | '
                     f'source={entry.get("source", "-")} {row["reason"]}')
    lines.append('  later parallel-leg pairs (listed, not judged): '
                 + (', '.join(f'{p}/{f}' for p, f in later) or 'none'))

    # C1 negative controls: the audit must be able to fail.
    dropped = dict(ul.READOUT_DEFINITIONS)
    dropped.pop(('leg1-objective-v1', 'J'))
    blanked = copy.deepcopy(ul.READOUT_DEFINITIONS)
    blanked[('leg1-objective-v1', 'P_in_reduced')]['span'] = ' '
    invented = dict(scoped)
    invented[('leg1-objective-v1', 'invented_power_ratio_xyz')] = {'x'}
    negatives = {
        'registry entry for J removed': ul.audit_definitions(scoped, {}, dropped),
        'span of P_in_reduced blanked': ul.audit_definitions(scoped, {}, blanked),
        'an invented field observed': ul.audit_definitions(invented, mirrored),
    }
    c1_neg_ok = all(not result['passed'] for result in negatives.values())
    for label, result in negatives.items():
        lines.append(f'  negative control ({label}): passed={result["passed"]} (must be False); '
                     f'missing={result["count"]["missing"]} invalid={result["count"]["invalid"]}')

    # C1b
    geo = cross_check(scoped, scope_of, runs_by_name)
    geo_problems = {p: r for p, r in geo.items() if r['problems'] or not r['checked']}
    monitors_checked = sum(r['checked'] for r in geo.values())
    lines += ['', f'C1b geometry: {len(geo)} monitor-based pairs, {monitors_checked} monitor '
                  f'comparisons against rebuilt projects, {len(geo_problems)} with a disagreement -> '
                  f'{"PASS" if not geo_problems else "FAIL"}']
    for pair, result in sorted(geo_problems.items()):
        lines.append(f'  {pair}: checked={result["checked"]} {result["problems"][:3]}')
    wrong = copy.deepcopy(ul.DECLARED_GEOMETRY)
    wrong['n8-scene']['near']['center'] = (-0.70, 0.12, 0.0)
    wrong['parity-1d']['transmission']['size'] = (0.0, 1.0, 'full')
    probe = {p: n for p, n in scoped.items() if p in (('leg1-objective-v1', 'P_in_reduced'),
                                                      ('leg5-units-ledger-v1', 'T'))}
    geo_neg = cross_check(probe, scope_of, runs_by_name, wrong)
    geo_neg_ok = len(geo_neg) == 2 and all(r['problems'] for r in geo_neg.values())
    lines.append(f'  negative control (near plane moved to -0.70 um; transmission span set to '
                 f'1 um): every probe fails={geo_neg_ok}')

    # C3, C4
    term = ul.transition_term_verdict()
    alpha = ul.radiation_term_bound()
    lines += ['', 'C3 the 5.6 pt transition-region term:',
              f'  ours: 1 - b_in/b_src_in = {term["our_loss_pt"]:.3f} pt (net flux, x = -3.5 and '
              f'51.842 um, 1.5 um span; recovered)',
              f'  vendor: 1 - 1/1.1014 = {term["vendor_loss_pt_if_net_basis"]:.3f} pt if the ratio '
              f'is referred to the net b_in; {term["vendor_loss_pt_if_modal_basis"]:.3f} pt if to the '
              f'modal b_in; plane, span and basis: {", ".join(term["vendor_fields_unknown"])} UNKNOWN',
              f'  difference: {term["difference_pt_if_net_basis"]:.3f} pt (net basis) / '
              f'{term["difference_pt_if_modal_basis"]:.3f} pt (modal basis) -- conditional, not a bound',
              f'  verdict: {term["verdict"].upper()}. {term["statement"]}',
              f'  in this repository: {ul.TRANSITION_TERM["region_in_this_repository"]}',
              '', 'C4 the ~40 percent radiation-strength term:',
              f'  attribution (from the material): {ul.RADIATION_TERM["attribution"]}; '
              f'{ul.RADIATION_TERM["withdrawn_at"]}',
              f'  physics bound on the reference solver line: alpha <= -ln(T)/(2L) = '
              f'{alpha["alpha_upper_bound_np_per_mm"]:.3f} Np/mm; excludes 4.15-4.33: '
              f'{alpha["excludes_claimed_range"]}',
              f'  in this repository: {ul.RADIATION_TERM["in_this_repository"]}']

    # C6
    target = config['c6_fields']
    settled = [f for f in target
               if all(c1_row['status'] in ('registered', 'declared-and-registered')
                      for c1_row in c1['rows'] if c1_row['field'] == f)
               and not any(pair[1] == f for pair in geo_problems)
               and any(c1_row['field'] == f for c1_row in c1['rows'])]
    leg11_runs = [r for r in ca.discover_runs(ROOT, exclude_policies=('leg11-coverage-audit-v1',))
                  if r.name <= LEG11_AUDIT_RUN]
    before = leg11_counts(leg11_runs)
    after = leg11_counts(leg11_runs, settled)
    lines += ['', 'C6 the leg-11 audit, recomputed on the runs it inventoried '
                  f'({len(leg11_runs)} runs):',
              f'  unchanged rules: physics={before["physics_fields"]} covered={before["covered"]} '
              f'open={before["open"]} settled={before["settled"]} gap_s4={before["gap_s4"]}',
              f'  with the S4 result applied: physics={after["physics_fields"]} covered={after["covered"]} '
              f'open={after["open"]} settled={after["settled"]} gap_s4={after["gap_s4"]}',
              f'  fields S4 settles: {settled}',
              f'  open after: {after["open_fields"]}',
              '  NOTE: the leg-11 rules table is src/coverage_audit.py, outside this leg\'s write set. '
              'Its unchanged re-run still lists the six as open; the second line is what it prints once '
              'that table records this result. They move to settled, NOT to covered: a pinned definition '
              'is not a physics check, and no label becomes verified.']
    text = '\n'.join(lines) + '\n'

    passed = c1['passed'] and c1_neg_ok and not geo_problems and geo_neg_ok
    measured = dict(pairs_in_scope=c1['total'], pairs_registered=c1['count']['registered'],
                    pairs_declared_and_registered=c1['count']['declared-and-registered'],
                    pairs_unknown=c1['count']['unknown'], pairs_missing=c1['count']['missing'],
                    pairs_invalid=c1['count']['invalid'], pairs_from_later_legs=len(later),
                    monitor_comparisons=monitors_checked, geometry_disagreements=len(geo_problems),
                    leg11_open_before=before['open'], leg11_open_after=after['open'],
                    leg11_gap_s4_before=before['gap_s4'], leg11_gap_s4_after=after['gap_s4'],
                    transition_term_difference_pt_net_basis=term['difference_pt_if_net_basis'],
                    transition_term_difference_pt_modal_basis=term['difference_pt_if_modal_basis'],
                    alpha_upper_bound_np_per_mm=alpha['alpha_upper_bound_np_per_mm'])
    readout = dict(policy=config['policy'], status='completed' if passed else 'ledger-failed',
                   simulation_executed=False,
                   fixture=('Every archived run up to the leg-11 inventory, plus leg 13\'s own runs. '
                            'Projects are rebuilt from archived configs; nothing is simulated.'),
                   stages=['F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout'],
                   stage_source={
                       'F1-structure': 'src/units_ledger.py:READOUT_DEFINITIONS (recovered definitions)',
                       'F2-mesh': 'src/leg13_ledger.py:build_family (projects rebuilt, not run)',
                       'F3-source': 'runs/*/readout.json and config.input.json (read-only)',
                       'F4-monitors': 'src/units_ledger.py:cross_check_monitor',
                       'F5-run': 'no simulation: this driver runs no solver',
                       'F6-readout': 'src/units_ledger.py:audit_definitions, transition_term_verdict',
                   },
                   cases=[dict(id='readout-ledger', measured=measured)],
                   units=config['units'], transition_term=term, radiation_term=alpha,
                   c6=dict(before=before, after=after, settled=settled))
    ledger = ul.evaluate_ledger(readout, config)
    if ledger['failures']:
        passed = False
        readout['status'] = 'ledger-failed'
        text += 'OWN LEDGER FAILURES: ' + '; '.join(ledger['failures']) + '\n'
    if own is not None:
        (own / 'readout-ledger.txt').write_text(text)
        (own / 'readout.json').write_text(json.dumps(readout, indent=2, sort_keys=True) + '\n')
    print(text)
    print(f'leg13 readout ledger: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 56


if __name__ == '__main__':
    sys.exit(main())
