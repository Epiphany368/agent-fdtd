"""Leg 5, V5: run one case with a declared units ledger and check it mechanically.

The declaration is written in the config before the run and mirrored verbatim into the readout.
`src/units_ledger.py` then recomputes coverage and diffs the two.

Run:  ./tools/run.sh configs/leg5-units-ledger.json src/leg5_ledger.py
      ./.venv/bin/python src/leg5_ledger.py <run-dir> --negative-control
"""
from parity import *
import units_ledger

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'parity.project_for (interface family)',
    'F2-mesh': 'parity.project_for (unchanged uniform Region)',
    'F3-source': 'parity.project_for (unchanged one-way plane source)',
    'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
    'F5-run': 'parity.simulate',
    'F6-readout': 'parity.spectra',
}

MEASURED_KEYS = ('wavelength_um', 'R', 'T', 'energy_residual', 'empty_reflection_floor',
                 'sourcepower_reduced', 'reference_power_reduced',
                 'sourcepower_max_relative_error')


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    points = cfg['frequency_points']
    case = cfg['cases'][0]
    project = project_for(case['kind'], case['index'], case['ppw'], case['duration_fs'],
                          frequency_points=points)
    detail = geometry_assertions(project)
    reference, reference_seconds = simulate(
        project_for('empty', case['index'], case['ppw'], case['duration_fs'],
                    frequency_points=points))
    sample, seconds = simulate(project)
    data = spectra(sample, reference)
    measured = {key: data[key] for key in MEASURED_KEYS}
    row = dict(id=case['id'], kind=case['kind'], index=case['index'], ppw=case['ppw'],
               courant_factor=float(project.region.courant_factor),
               steps=int(project.region.steps), dt_s=project.region.time_step,
               shape=list(project.region.shape), wall_seconds=seconds,
               reference_wall_seconds=reference_seconds, measured=measured)
    effective['resolved_cases'].append(dict(id=case['id'], kind=case['kind'], **detail))
    Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')

    readout = dict(policy=cfg['policy'], status='completed', simulation_executed=True,
                   fixture=('Planar interface in vacuum, normal incidence, one-way plane source, '
                            'PML ends; every reported number carries a declared readout definition'),
                   stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=[row],
                   units=cfg['units'])
    write_json(Path(run) / 'readout.json', readout)

    ledger = units_ledger.evaluate_ledger(readout, cfg)
    print(f'ledger diff={ledger["diff"]} failures={len(ledger["failures"])}', flush=True)
    for case_row in ledger['cases']:
        print(f'  case {case_row["case_id"]}: declared={[d["key"] for d in case_row["declared"]]}',
              flush=True)
        print(f'    exempt={[e["key"] for e in case_row["exempt"]]}', flush=True)
    for key, units in ledger['key_units'].items():
        print(f'  key {key}: {units}', flush=True)
    for failure in ledger['failures']:
        print(f'  FAILURE: {failure}', flush=True)
    print(f'leg5 ledger: {len(ledger["failures"])} failures', flush=True)
    return 54 if ledger['failures'] else 0


def negative_control():
    """Two controls: a dropped declaration, and a unit outside the closed vocabulary."""
    run = Path(sys.argv[1])
    readout = json.loads((run / 'readout.json').read_text())
    config = json.loads((run / 'config.input.json').read_text())
    case_id = config['cases'][0]['id']
    before = units_ledger.evaluate_ledger(readout, config)
    print(f'control 0 (unmodified): diff={before["diff"]} '
          f'failures={len(before["failures"])}')

    dropped = units_ledger.drop_declaration(readout, case_id, 'T')
    dropped_result = units_ledger.evaluate_ledger(dropped, config)
    print(f'control 1 (declaration for T removed from the readout mirror): '
          f'diff={dropped_result["diff"]} failures={len(dropped_result["failures"])}')
    for failure in dropped_result['failures']:
        print(f'  FAILURE: {failure}')

    invented = units_ledger.invent_unit(readout, case_id, 'R', 'furlong')
    invented_result = units_ledger.evaluate_ledger(invented, config)
    print(f'control 2 (unit furlong invented for R): '
          f'diff={invented_result["diff"]} failures={len(invented_result["failures"])}')
    for failure in invented_result['failures']:
        print(f'  FAILURE: {failure}')
    print('Source run files were not modified.')
    clean = not before['failures'] and before['diff'] == 'match'
    both_fail = bool(dropped_result['failures']) and bool(invented_result['failures'])
    return 0 if (clean and both_fail) else 1


if __name__ == '__main__':
    sys.exit(negative_control() if '--negative-control' in sys.argv else main())
