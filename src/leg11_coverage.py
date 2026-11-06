#!/usr/bin/env python3
"""Leg 11 driver: run the physics-check coverage audit and archive its result.

Read-only with respect to every other run. Writes the F1-F6 readout contract so the
audit is itself reportable, and so its evidence sits in the ledger like a leg.
"""
import json
import os
import sys
from pathlib import Path

# The launcher runs this file from the run's source SNAPSHOT, which deliberately excludes runs/.
# The code comes from the snapshot; the data must come from the real repository.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'src'))
# RUN_DIR_ABS is <repo>/runs/<run-id>, so its grandparent is the repository the data lives in.
# PARITY_CONFIG is NOT usable here: the launcher points it at the run's own archived config.
_pinned = os.environ.get('RUN_DIR_ABS')
ROOT = Path(_pinned).resolve().parents[1] if _pinned else CODE_ROOT

from coverage_audit import (ABSENT_CHECKS, REGISTERED_CHECKS, audit, classify,
                            discover_runs, render)


def main():
    config = json.loads((ROOT / 'configs' / 'leg11-coverage-audit.json').read_text())
    # The launcher names the run directory; without it this is a dry run.
    run_dir = os.environ.get('RUN_DIR_ABS')
    own = Path(run_dir).resolve() if run_dir else None
    # Inventories are excluded: an inventory must not count itself, and leg 13's readout-definition
    # ledger is the second inventory. It reports its own counters and values recovered from the
    # predecessor's record -- not measurements of this solver -- so it is not a physics family
    # source. Every other `simulation_executed: false` run is kept: leg 6 evaluates a closed-form
    # model and leg 14's slab runs evaluate analytic laws, and both carry real physics families.
    runs = discover_runs(ROOT, exclude_policies=(config['policy'], 'leg13-readout-ledger-v1'))
    runs = [r for r in runs if own is None or r.resolve() != own]

    families, checked, unclassified = audit(runs)
    families.pop('?<unreadable>', None)
    physics = {n: classify(n) for n in families if classify(n)[0] == 'physics'}
    open_gaps = {n: v for n, v in physics.items() if v[1] == 'open'}
    covered = {n: v for n, v in physics.items() if v[1] == 'covered'}

    text = render(runs)
    if own is not None:
        (own / 'coverage-audit.txt').write_text(text)

    measured = {
        'runs_read': len(runs),
        'headline_values': checked,
        'field_families': len(families),
        'physics_fields': len(physics),
        'physics_covered': len(covered),
        'physics_open': len(open_gaps),
        'unclassified_families': len(unclassified),
        'registered_checks': len(REGISTERED_CHECKS),
        'absent_checks': len(ABSENT_CHECKS),
        'gap_s4': len([n for n, v in open_gaps.items() if v[2] == 'S4']),
        'gap_unassigned': len([n for n, v in open_gaps.items()
                               if v[2] == 'not established from this repository']),
    }
    readout = {
        'policy': config['policy'],
        'status': 'completed',
        'simulation_executed': False,
        'fixture': ('Every archived run under runs/ except this one. Inventory only: no label is '
                    'changed, no check is implemented, no physics is re-run.'),
        'stages': ['F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout'],
        'stage_source': {
            'F1-structure': 'src/coverage_audit.py:family (field-family normalisation)',
            'F2-mesh': 'src/coverage_audit.py:audit_run (the same headline pipeline as F8)',
            'F3-source': 'runs/*/readout.json of the archived runs (read-only inputs)',
            'F4-monitors': 'src/coverage_audit.py:audit (per-family aggregation)',
            'F5-run': 'no simulation: this driver imports no solver',
            'F6-readout': 'src/coverage_audit.py:classify (declared role and coverage rules)',
        },
        'cases': [{'id': 'coverage-audit', 'measured': measured,
                   'recorded': {'measured': measured}}],
        'note': ('Inventory only. ABSENT_CHECKS lists a registered check that does not exist in this '
                 'repository; it may never be cited as closing a gap. Fields with no matching rule '
                 'are reported as unclassified rather than guessed.'),
    }
    if own is not None:
        (own / 'readout.json').write_text(json.dumps(readout, indent=2, sort_keys=True) + '\n')
        (own / 'effective.json').write_text(json.dumps(
            {'policy': config['policy'], 'runs_read': len(runs),
             'excluded_own_run': own.name}, indent=2, sort_keys=True) + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
