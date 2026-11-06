"""Leg 12: the V2-a energy balance on a NON-EMPTY closed box, with a lossy negative control.

The empty-domain control (leg 2) closes trivially: sample and reference are the same run.
Here a lossless dielectric slab sits inside the same closed box (periodic transverse walls,
both x exits monitored), so R and T are non-trivial and R + T = 1 is a real statement.

  F1 structure   parity.project_for('slab'); the control swaps the slab material for a
                 lossy Lorentz material (solver defaults except epsilon_inf)
  F2-F4          unchanged uniform Region, one-way plane source, existing monitor planes
  F5 run         parity.simulate, matched empty reference per (index, ppw, duration)
  F6 readout     parity.spectra
  V2             invariants.evaluate_invariants recomputes every registered check

The archived config declares the control truthfully as lossy, so the report drops V2-a
there as unlawful. The negative control asks the question on an in-memory copy: if this
lossy box were claimed lossless, would V2-a catch it? It must fail.

Run:  ./tools/run.sh configs/leg12-closed-slab.json src/leg12_closed_box.py
"""
import copy
import json
import os
from pathlib import Path
import sys

from invariants import evaluate_invariants


def negative_control_rows(readout, config, case_id):
    """V2 rows for ``case_id`` with its physics claimed lossless, on copies only."""
    claimed = copy.deepcopy(config)
    for case in claimed['cases']:
        if case.get('id') == case_id:
            case['physics']['lossless'] = True
    claimed['invariants'] = [row for row in claimed['invariants']
                             if row.get('case_id') == case_id and row.get('id') == 'V2-a']
    return evaluate_invariants(copy.deepcopy(readout), claimed)


def build(case, frequency_points):
    from parity import Material, project_for
    project = project_for('slab', case['index'], case['ppw'], case['duration_fs'],
                          transverse_cells=case['transverse_cells'],
                          frequency_points=frequency_points)
    if case.get('material') == 'lossy-lorentz-control':
        # Every Lorentz parameter is the solver default except epsilon_inf, which keeps the
        # slab's real background permittivity at index**2. A control value, not a device one.
        data = project.model_dump()
        data['materials'] = [Material(name='dielectric', model='lorentz',
                                      epsilon_inf=case['index'] ** 2).model_dump()]
        project = type(project).model_validate(data)
    return project


def main():
    from parity import geometry_assertions, load_run, project_for, simulate, spectra, write_json
    from verification import STAGES
    import numpy as np
    config, run = load_run()
    effective_path = Path(os.environ['EFFECTIVE_JSON'])
    effective = json.loads(effective_path.read_text())
    readout = dict(
        policy=config['policy'], status='in_progress', acceptance_granted=False,
        fixture='Closed non-empty plane-wave box: dielectric slab, periodic transverse walls, '
                'both x exits monitored; plus one lossy control slab',
        stages=list(STAGES),
        stage_source={
            'F1-structure': 'parity.project_for (slab family); leg12_closed_box.build for the control',
            'F2-mesh': 'parity.project_for (unchanged uniform Region)',
            'F3-source': 'parity.project_for (unchanged one-way plane source)',
            'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
            'F5-run': 'parity.simulate',
            'F6-readout': 'parity.spectra',
        },
        cases=[])
    write_json(run / 'readout.json', readout)
    references = {}
    for case in config['cases']:
        points = config['frequency_points']
        project = build(case, points)
        resolved = geometry_assertions(project)
        effective['resolved_cases'].append(dict(id=case['id'], **resolved))
        write_json(effective_path, effective)
        key = (case['index'], case['ppw'], case['duration_fs'], case['transverse_cells'])
        if key not in references:
            reference_project = project_for('empty', case['index'], case['ppw'], case['duration_fs'],
                                            transverse_cells=case['transverse_cells'],
                                            frequency_points=points)
            geometry_assertions(reference_project)
            references[key] = simulate(reference_project, config['graph_steps'])
        reference, reference_seconds = references[key]
        sample, sample_seconds = simulate(project, config['graph_steps'])
        np.savez_compressed(run / f'{case["id"]}-sample-monitors.npz', **{
            f'{name}_{field}': np.asarray(sample.field_monitor(name)[field])
            for name in ('reflection', 'transmission', 'scattered') for field in ('flux', 'frequency_hz')})
        measured = spectra(sample, reference)
        readout['cases'].append(dict(
            id=case['id'], shape=resolved['shape'], steps=resolved['steps'], dt_s=resolved['dt_s'],
            sample_elapsed_seconds=sample_seconds, reference_elapsed_seconds=reference_seconds,
            measured=measured))
        print(json.dumps(dict(case_id=case['id'], energy_residual=measured['energy_residual'],
                              empty_reflection_floor=measured['empty_reflection_floor'],
                              sample_elapsed_seconds=sample_seconds)), flush=True)
    decisions = evaluate_invariants(readout, config)
    control_ids = [case['id'] for case in config['cases'] if case.get('negative_control')]
    controls = [row for case_id in control_ids for row in negative_control_rows(readout, config, case_id)]
    registered = [row for row in decisions if row['case_id'] not in control_ids]
    lossless_ok = bool(registered) and all(row['status'] == 'passed' for row in registered)
    control_ok = bool(controls) and all(row['status'] == 'failed' for row in controls)
    readout['negative_control'] = [dict(case_id=row['case_id'], id=row['id'], status=row['status'],
                                        error=row['error'], tolerance=row['tolerance'])
                                   for row in controls]
    readout['status'] = 'completed' if lossless_ok and control_ok else 'invariant_failure'
    readout['simulation_executed'] = True
    write_json(run / 'readout.json', readout)
    effective.update(simulation_executed=True, stage=readout['status'])
    write_json(effective_path, effective)
    for row in decisions + controls:
        print(repr(row), flush=True)
    print(f'leg12 closed box: registered {sum(r["status"] == "passed" for r in registered)}/'
          f'{len(registered)} passed; negative control failed as required: {control_ok}', flush=True)
    return 0 if lossless_ok and control_ok else 53


if __name__ == '__main__':
    sys.exit(main())
