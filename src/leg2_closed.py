"""Short closed-case control using the unchanged copied parity adapter.

Run only through tools/run.sh. No new source, monitor, material or solver code.
Two independent executions make the matched empty-domain denominator explicit.
Raw monitor fields and fluxes are retained; registered decisions are recomputed
from the measured spectra, never from a stored pass/fail flag.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

from parity import geometry_assertions, load_run, project_for, simulate, spectra, write_json
from verification import STAGES
from invariants import evaluate_invariants


def save_monitors(path, result):
    arrays = {}
    for name in ('reflection', 'transmission', 'scattered'):
        monitor = result.field_monitor(name)
        for key in ('fields', 'flux', 'frequency_hz'):
            if key in monitor:
                arrays[f'{name}_{key}'] = np.asarray(monitor[key])
    np.savez_compressed(path, **arrays)


def main():
    config, run = load_run()
    effective_path = Path(os.environ['EFFECTIVE_JSON'])
    effective = json.loads(effective_path.read_text())
    readout = dict(
        policy=config['policy'], status='in_progress', acceptance_granted=False,
        fixture='Closed lossless empty plane-wave control; periodic transverse boundaries',
        stages=list(STAGES),
        stage_source={
            'F1-structure': 'parity.project_for (empty family)',
            'F2-mesh': 'parity.project_for (unchanged uniform Region)',
            'F3-source': 'parity.project_for (unchanged one-way plane source)',
            'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
            'F5-run': 'parity.simulate',
            'F6-readout': 'parity.spectra',
        },
        cases=[],
    )
    write_json(run / 'readout.json', readout)
    for case in config['cases']:
        if case['geometry'] != 'empty':
            raise ValueError('This registered driver is restricted to the empty control.')
        args = dict(index=case['index'], ppw=case['ppw'], duration_fs=case['duration_fs'],
                    transverse_cells=case['transverse_cells'],
                    frequency_points=case['frequency_points'], precision=config['precision'])
        sample_project = project_for(case['geometry'], **args)
        reference_project = project_for('empty', **args)
        resolved = geometry_assertions(sample_project)
        geometry_assertions(reference_project)
        effective['resolved_cases'].append(dict(id=case['id'], **resolved))
        write_json(effective_path, effective)
        reference, reference_seconds = simulate(reference_project, config['graph_steps'])
        sample, sample_seconds = simulate(sample_project, config['graph_steps'])
        save_monitors(run / f'{case["id"]}-reference-monitors.npz', reference)
        save_monitors(run / f'{case["id"]}-sample-monitors.npz', sample)
        measured = spectra(sample, reference)
        readout['cases'].append(dict(
            id=case['id'], shape=resolved['shape'], steps=resolved['steps'],
            dt_s=resolved['dt_s'], sample_elapsed_seconds=sample_seconds,
            reference_elapsed_seconds=reference_seconds, measured=measured))
        print(json.dumps(dict(case_id=case['id'], measured=measured,
                              sample_elapsed_seconds=sample_seconds,
                              reference_elapsed_seconds=reference_seconds)), flush=True)
    decisions = evaluate_invariants(readout, config)
    failed = [row for row in decisions if row['status'] != 'passed']
    readout['status'] = 'invariant_failure' if failed else 'completed'
    readout['simulation_executed'] = True
    write_json(run / 'readout.json', readout)
    effective.update(simulation_executed=True, stage='completed' if not failed else 'invariant_failure')
    write_json(effective_path, effective)
    for row in decisions:
        # A failed finiteness check can carry an infinite error. Keep its size
        # visible in the log rather than failing JSON encoding before reporting.
        print(repr(row), flush=True)
    print(f'leg2 closed control: {len(decisions) - len(failed)}/{len(decisions)} registered invariants passed',
          flush=True)
    return 53 if failed or not decisions else 0


if __name__ == '__main__':
    sys.exit(main())
