"""Leg 3, V1: run the analytic cases and compare them with their closed-form laws.

Two cases, one empty reference, three short forward runs:

  F1 structure   `parity.project_for` for the `interface` and `slab` families
  F2 mesh        the uniform `Region` those builders carry
  F3 source      the unchanged one-way plane source those builders carry
  F4 monitors    the reflection / transmission / scattered planes those builders carry
  F5 run         `parity.simulate`
  F6 readout     `parity.spectra` (R and T per frequency)
  V1             `analytic_cases` recomputes each closed-form law in this repository

The source run files are never written by the negative control.

Run:  ./tools/run.sh configs/leg3-analytic-cases.json src/leg3_analytic.py
      ./.venv/bin/python src/leg3_analytic.py <run-dir> --negative-control
"""
from parity import *
import analytic_cases

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'parity.project_for (interface and slab families)',
    'F2-mesh': 'parity.project_for (unchanged uniform Region)',
    'F3-source': 'parity.project_for (unchanged one-way plane source)',
    'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
    'F5-run': 'parity.simulate',
    'F6-readout': 'parity.spectra',
}


def run_cases(cfg, run):
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    points = cfg['frequency_points']
    references = {}
    rows = []
    for case in cfg['cases']:
        index, ppw, duration = case['index'], case['ppw'], case['duration_fs']
        project = project_for(case['kind'], index, ppw, duration, frequency_points=points)
        detail = geometry_assertions(project)
        key = (index, ppw, duration)
        if key not in references:
            reference, reference_seconds = simulate(
                project_for('empty', index, ppw, duration, frequency_points=points))
            references[key] = (reference, reference_seconds)
        reference, reference_seconds = references[key]
        sample, seconds = simulate(project)
        data = spectra(sample, reference)
        row = dict(
            id=case['id'], kind=case['kind'], index=index, ppw=ppw, duration_fs=duration,
            shape=list(project.region.shape), steps=project.region.steps,
            dt_s=project.region.time_step, wall_seconds=seconds,
            reference_wall_seconds=reference_seconds,
            measured=dict(wavelength_um=data['wavelength_um'], R=data['R'], T=data['T'],
                          energy_residual=data['energy_residual'],
                          empty_reflection_floor=data['empty_reflection_floor'],
                          sourcepower_max_relative_error=data['sourcepower_max_relative_error']))
        rows.append(row)
        effective['resolved_cases'].append(dict(id=case['id'], kind=case['kind'], **detail))
        Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'measured'}), flush=True)
    return rows


def write_readout(run, cfg, rows):
    readout = dict(policy=cfg['policy'], status='completed', simulation_executed=True,
                   fixture=('Planar interface and slab in vacuum, normal incidence, '
                            'one-way plane source, PML ends'),
                   stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=rows)
    write_json(Path(run) / 'readout.json', readout)
    return readout


def main():
    if '--negative-control' in sys.argv:
        return negative_control_main()
    cfg, run = load_run()
    rows = run_cases(cfg, run)
    readout = write_readout(run, cfg, rows)
    checks = analytic_cases.evaluate_analytic_cases(readout, cfg)
    for check in checks:
        print(f'{check["id"]} case={check["case_id"]} status={check["status"]} '
              f'error={check["error"]!r} tolerance={check["tolerance"]!r} '
              f'kind={check["tolerance_kind"]!r} worst={check["worst"]}', flush=True)
    failed = [check['id'] for check in checks if check['status'] != 'passed']
    print(f'leg3 analytic: {len(checks) - len(failed)}/{len(checks)} cases passed; '
          f'dropped: {[entry["id"] for entry in cfg.get("dropped_analytic_cases", [])]}', flush=True)
    return 52 if failed else 0


def negative_control_main():
    """Perturb one measured point in memory and show the same code reporting a failure."""
    run = Path(sys.argv[1])
    readout = json.loads((run / 'readout.json').read_text())
    config = json.loads((run / 'config.input.json').read_text())
    entry = config['analytic_cases'][0]
    offset = 0.25
    perturbed = analytic_cases.negative_control(entry, readout, offset)
    before = readout['cases'][0]['measured'][entry.get('negative_control_key', 'T')][0]
    after = perturbed['cases'][0]['measured'][entry.get('negative_control_key', 'T')][0]
    checks = analytic_cases.evaluate_analytic_cases(perturbed, config)
    for check in checks:
        print(f'{check["id"]} case={check["case_id"]} status={check["status"]} '
              f'error={check["error"]!r} tolerance={check["tolerance"]!r}')
    print(f'negative control: case={entry["case_id"]} '
          f'{entry.get("negative_control_key", "T")}[0] {before} -> {after} (in-memory copy only)')
    print('Source run files were not modified.')
    return 1 if any(check['status'] == 'failed' for check in checks) else 0


if __name__ == '__main__':
    sys.exit(main())
