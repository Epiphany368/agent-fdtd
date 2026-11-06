"""Leg 4, V3: the convergence ladder, the time-step ladder, and the coincidence guard.

Four mesh rungs (`ppw` 20/40/80/160) on the two cases that have an exact answer, so each rung's
error is measured against truth and not against another rung of ours. One extra rung varies the
time step through `Region.courant_factor`, with `Region.steps` recomputed so the physical duration
stays fixed.

Stage imports are unchanged from leg 3; this leg only changes the resolution and the time step.

Run:  ./tools/run.sh configs/leg4-convergence-ladder.json src/leg4_ladder.py
"""
import math

from parity import *
import analytic_cases
import convergence

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'parity.project_for (interface and slab families)',
    'F2-mesh': 'parity.project_for (unchanged uniform Region), courant_factor and steps set here',
    'F3-source': 'parity.project_for (unchanged one-way plane source)',
    'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
    'F5-run': 'parity.simulate',
    'F6-readout': 'parity.spectra',
}


def with_courant(project, courant, duration_fs):
    """Change the time step and recompute the step count so the physical duration is unchanged."""
    project.region.courant_factor = courant
    project.region.steps = math.ceil(duration_fs * 1e-15 / project.region.time_step)
    return Project.model_validate(project.model_dump())


def empty_reference(kind_index, ppw, duration_fs, courant, points, cache):
    key = (ppw, courant)
    if key not in cache:
        reference = project_for('empty', kind_index, ppw, duration_fs, frequency_points=points)
        if courant != reference.region.courant_factor:
            reference = with_courant(reference, courant, duration_fs)
        cache[key] = simulate(reference)
    return cache[key]


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    points = cfg['frequency_points']
    cache = {}
    rows = []
    for case in cfg['cases']:
        index, ppw = case['index'], case['ppw']
        courant = case['courant_factor']
        duration = case['duration_fs']
        project = project_for(case['kind'], index, ppw, duration, frequency_points=points)
        if courant != project.region.courant_factor:
            project = with_courant(project, courant, duration)
        detail = geometry_assertions(project)
        reference, reference_seconds = empty_reference(index, ppw, duration, courant, points, cache)
        sample, seconds = simulate(project)
        data = spectra(sample, reference)
        row = dict(id=case['id'], kind=case['kind'], index=index, ppw=ppw,
                   courant_factor=float(project.region.courant_factor),
                   steps=int(project.region.steps), dt_s=project.region.time_step,
                   duration_s=project.region.steps * project.region.time_step,
                   shape=list(project.region.shape), wall_seconds=seconds,
                   reference_wall_seconds=reference_seconds,
                   measured=dict(wavelength_um=data['wavelength_um'], R=data['R'], T=data['T'],
                                 energy_residual=data['energy_residual'],
                                 empty_reflection_floor=data['empty_reflection_floor']))
        rows.append(row)
        effective['resolved_cases'].append(dict(id=case['id'], kind=case['kind'], **detail))
        Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'measured'}), flush=True)

    by_id = {row['id']: row for row in rows}
    dt = cfg.get('dt_ladder') or {}
    dt_result = None
    if dt.get('case_ids'):
        first, second = (by_id[case_id] for case_id in dt['case_ids'])
        difference = max(abs(float(a) - float(b))
                         for key in ('R', 'T')
                         for a, b in zip(first['measured'][key], second['measured'][key]))
        dt_result = dict(id=dt['id'], case_ids=list(dt['case_ids']),
                         courant_factors=[first['courant_factor'], second['courant_factor']],
                         steps=[first['steps'], second['steps']],
                         duration_s=[first['duration_s'], second['duration_s']],
                         max_abs_difference=difference,
                         wall_seconds=[first['wall_seconds'], second['wall_seconds']],
                         reason=dt['reason'])

    readout = dict(policy=cfg['policy'], status='completed', simulation_executed=True,
                   fixture=('Planar interface and slab in vacuum, normal incidence, one-way plane '
                            'source, PML ends; mesh ladder ppw 20/40/80/160 and one time-step rung'),
                   stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=rows,
                   dt_ladder=dt_result)
    write_json(Path(run) / 'readout.json', readout)

    ladders = convergence.evaluate_ladders(readout, cfg)
    for row in ladders:
        print(f'{row["id"]} status={row["status"]} errors={row["errors"]} '
              f'order={row["order"]!r} residual={row["residual"]!r} '
              f'coincidence={row["coincidence"]}', flush=True)
    if dt_result:
        print(f'{dt_result["id"]} max_abs_difference={dt_result["max_abs_difference"]!r} '
              f'steps={dt_result["steps"]}', flush=True)
    for row in convergence.earned_floor(ladders, cfg['registered_ppw'], cfg['inherited_tolerance']):
        print(json.dumps(row), flush=True)
    stopped = [row['id'] for row in ladders if row['status'] in ('stopped-non-monotone',)]
    print(f'leg4 ladder: {len(ladders)} ladders, stopped={stopped}', flush=True)
    return 53 if stopped else 0


def guard_control():
    """Criterion C4: show the coincidence guard firing on a constructed input.

    The finest rung of the first declared ladder is perturbed in memory so that it lands outside
    the tolerance while the coarse rung stays inside it. The same code must then (a) flag the
    ladder as a coincidence and (b) strip every `verified` label from that case. The source run
    files are never written. Exit 0 when the guard fired correctly, 1 otherwise.
    """
    from analytic_cases import label_analytic_cases
    from verification import verify_readout

    run = Path(sys.argv[1])
    readout = json.loads((run / 'readout.json').read_text())
    config = json.loads((run / 'config.input.json').read_text())
    ladder = config['ladders'][0]
    finest = ladder['case_ids'][-1]
    for case in readout['cases']:
        if case['id'] == finest:
            case['measured']['T'] = [value + 0.5 for value in case['measured']['T']]
    rows = convergence.evaluate_ladders(readout, config)
    row = next(entry for entry in rows if entry['id'] == ladder['id'])
    headlines = label_analytic_cases(readout, config, verify_readout(readout, config))
    before = sum(1 for headline in headlines
                 if headline['label'] == 'verified' and headline.get('case_id') in ladder['case_ids'])
    headlines = convergence.apply_coincidence_guard(headlines, rows)
    after = sum(1 for headline in headlines
                if headline['label'] == 'verified' and headline.get('case_id') in ladder['case_ids'])
    print(f'guard control: perturbed {finest} transmittance by +0.5 (in-memory copy only)')
    print(f'{row["id"]} status={row["status"]} monotone={row["monotone"]} '
          f'coincidence={row["coincidence"]} errors={row["errors"]}')
    print(f'verified labels on that ladder: {before} before the guard, {after} after it')
    print('Source run files were not modified.')
    return 0 if (row['coincidence'] and before > 0 and after == 0) else 1


if __name__ == '__main__':
    sys.exit(guard_control() if '--guard-control' in sys.argv else main())
