"""Leg 13 (S4): the standing control set, run on new simulations.

Two controls, both from physical law and neither from a paper or a vendor:

  T + R = 1           a lossless cell whose two exits are fully monitored loses nothing
  symmetric 1.000x    a soft source in a homogeneous cell sends equal power both ways

plus a negative control: the one-way source must FAIL the symmetric gate, otherwise that gate
cannot tell a symmetric source from a one-way one. Every number this run reports carries a
declared plane, span and normalisation, pre-declared in the config and diffed by `units_ledger`.

The absorber depth is the config's `pml_cells` (leg 18). Absent, it is `parity.DEFAULT_PML_CELLS`,
the 32 cells every run before leg 18 used. It is recorded by the config and by the resolved projects
in `effective.json`, not in the readout: a readout field the coverage audit does not classify fails
`checks/test_coverage_audit.py`.

Run:  ./tools/run.sh configs/leg13-standing-controls.json src/leg13_controls.py
"""
from parity import *
import units_ledger

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'parity.project_for (interface, slab and empty families)',
    'F2-mesh': 'parity.project_for (unchanged uniform Region)',
    'F3-source': ('parity.project_for (one-way plane source); for the symmetric case the same '
                  'project with the source switched to soft injection at x = 0'),
    'F4-monitors': 'parity.project_for (existing reflection/transmission/scattered planes)',
    'F5-run': 'parity.simulate',
    'F6-readout': 'parity.spectra; leg13_controls.directional_fluxes',
}
ONE_WAY_KEYS = ('wavelength_um', 'R', 'T', 'energy_residual', 'empty_reflection_floor',
                'sourcepower_reduced', 'reference_power_reduced', 'sourcepower_max_relative_error')


def soft_project(index, ppw, duration_fs, points, pml_cells=DEFAULT_PML_CELLS):
    """The empty one-way project with its source made soft and moved to the mirror centre x = 0.

    Nothing else changes: same mesh, PML, monitors and spectrum. The reflection and transmission
    monitors sit at -snap(2) and +snap(2), mirror images about the source's E node.
    """
    dump = project_for('empty', index, ppw, duration_fs, frequency_points=points,
                       pml_cells=pml_cells).model_dump()
    source = dump['sources'][0]
    source.update(injection='soft', center=(0.0, 0.0, 0.0))
    project = Project.model_validate(dump)
    xs = {m.id: m.center[0] for m in project.monitors}
    assert abs(xs['reflection'] + xs['transmission']) < 1e-12, 'monitors are not mirror images'
    assert project.sources[0].injection == 'soft'
    return project


def directional_fluxes(result, forward_id, backward_id):
    """Signed +x flux on one monitor and the -x flux magnitude on the other, per frequency."""
    forward = np.asarray(result.field_monitor(forward_id)['flux'], dtype=float).reshape(-1)
    backward = -np.asarray(result.field_monitor(backward_id)['flux'], dtype=float).reshape(-1)
    return forward, backward


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    points = cfg['frequency_points']
    pml_cells = cfg.get('pml_cells', DEFAULT_PML_CELLS)
    gates = cfg['controls']
    rows, controls, negatives = [], [], []
    references = {}
    for case in cfg['cases']:
        index, ppw, duration = case['index'], case['ppw'], case['duration_fs']
        key = (index, ppw, duration)
        if key not in references:
            references[key] = simulate(project_for('empty', index, ppw, duration,
                                                   frequency_points=points, pml_cells=pml_cells))
        reference, reference_seconds = references[key]
        if case['kind'] == 'empty-soft':
            project = soft_project(index, ppw, duration, points, pml_cells)
            sample, seconds = simulate(project)
            forward, backward = directional_fluxes(sample, 'transmission', 'reflection')
            gate = gates['symmetric_source']
            control = units_ledger.symmetric_source_ratio(list(forward), list(backward),
                                                          gate['tolerance'], gate['min_fraction'])
            control['case'] = case['id']
            controls.append(control)
            # The negative control: the one-way empty run, forward past the source, backward behind it.
            ow_forward, ow_backward = directional_fluxes(reference, 'transmission', 'scattered')
            negative = units_ledger.symmetric_source_ratio(list(ow_forward), list(ow_backward),
                                                           gate['tolerance'], gate['min_fraction'])
            negative.update(case=case['id'] + ' (one-way empty run)', expected='fail',
                            behaved=not negative['passed'])
            negatives.append(negative)
            frequency = np.asarray(sample.field_monitor('transmission')['frequency_hz']).reshape(-1)
            measured = dict(wavelength_um=(C0 / frequency * 1e6).tolist(),
                            forward_flux_reduced=forward.tolist(),
                            backward_flux_reduced=backward.tolist(),
                            symmetric_ratio_max_deviation=control.get('deviation'),
                            oneway_ratio_max_deviation=negative.get('deviation'))
            detail = dict(shape=list(project.region.shape), steps=project.region.steps,
                          dt_s=project.region.time_step, project=project.model_dump())
        else:
            project = project_for(case['kind'], index, ppw, duration, frequency_points=points,
                                  pml_cells=pml_cells)
            detail = geometry_assertions(project)
            sample, seconds = simulate(project)
            data = spectra(sample, reference)
            measured = {k: data[k] for k in ONE_WAY_KEYS}
            control = units_ledger.energy_closure(data['R'], data['T'],
                                                  gates['energy_closure']['tolerance'])
            control['case'] = case['id']
            controls.append(control)
        row = dict(id=case['id'], kind=case['kind'], index=index, ppw=ppw,
                   courant_factor=float(project.region.courant_factor),
                   steps=int(project.region.steps), dt_s=project.region.time_step,
                   shape=list(project.region.shape), wall_seconds=seconds,
                   reference_wall_seconds=reference_seconds, measured=measured)
        rows.append(row)
        effective['resolved_cases'].append(dict(id=case['id'], kind=case['kind'], **detail))
        Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'measured'}), flush=True)

    readout = dict(policy=cfg['policy'], status='completed', simulation_executed=True,
                   fixture=('Interface and slab in vacuum with a one-way plane source, and the empty '
                            'cell with a soft source at its mirror centre; PML ends, periodic sides'),
                   stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=rows,
                   units=cfg['units'], standing_controls=controls, negative_controls=negatives)
    ledger = units_ledger.evaluate_ledger(readout, cfg)
    readout['ledger_failures'] = ledger['failures']
    failed = ([c['case'] for c in controls if not c['passed']]
              + [n['case'] for n in negatives if not n['behaved']] + ledger['failures'])
    readout['status'] = 'completed' if not failed else 'control-failed'
    write_json(Path(run) / 'readout.json', readout)

    for c in controls:
        value = c.get('value')
        print(f'control {c["control"]} [{c["case"]}]: value={value!r} tolerance={c.get("tolerance")!r} '
              f'passed={c["passed"]} margin={c.get("margin")!r}', flush=True)
    for n in negatives:
        print(f'negative {n["control"]} [{n["case"]}]: value={n.get("value")!r} '
              f'passed_gate={n["passed"]} (must be False) behaved={n["behaved"]}', flush=True)
    print(f'pml_cells={pml_cells}', flush=True)
    print(f'ledger diff={ledger["diff"]} failures={len(ledger["failures"])}', flush=True)
    for failure in ledger['failures']:
        print(f'  FAILURE: {failure}', flush=True)
    print(f'leg13 standing controls: {len(failed)} failures', flush=True)
    return 55 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
