"""Leg 15: attribute the `7.5e-7` closure residual of the non-empty slab fixture.

Leg 12 left `T + R = 1` failing by `7.5e-07` on a lossless non-empty box with the cause unknown.
This leg moves one declared numerical knob at a time and watches the residual. It changes no
physics case, no label, and no solver code.

The fixture is the leg-3 slab as `parity.project_for` builds it, with the knobs exposed:
`pml_cells` and `interior_half_um` (how far the boundary and the box ends are), `slab_center_um`
(where the structure sits), `duration_fs`, `frequency_points` and the slab `index` (`1.0` is the
null: no interface at all). The baseline row is `parity.project_for`'s own values, and it must
reproduce the archived `7.50792e-07`.

Three numbers carry the diagnosis, all from arrays this run stores:

  energy_residual  max |R + T - 1|               -- the quantity under investigation
  absolute_change  max |R - (1 - net/reference)| -- the gap between the two ways of composing R
                                                    from the same stored arrays (hypothesis H3)
  relative_change  max |empty_reflection/empty_transmission - 1| -- how far the two denominators
                                                    of the empty run differ (the incident side)

Run:  ./tools/run.sh configs/leg15-closure.json src/leg15_closure.py
"""
import math
import os
import sys
from pathlib import Path

from parity import *  # noqa: F401,F403  (Project, Region, ... ppw_step, load_run, spectra, simulate)


def build(kind, index, ppw, duration_fs, *, pml_cells=32, interior_half_um=4.0,
          slab_center_um=0.001, transverse_cells=6, frequency_points=41,
          precision='float64', lossy=False, mesh_index=None, slab_thickness_um=None):
    """`parity.project_for`, with the numerical knobs this leg has to move.

    Copied from `parity.project_for` (same formulas, same source and monitors) so that the
    baseline row is the same fixture; the only difference is that the four knobs are arguments
    instead of constants.
    """
    dx = ppw_step(1.2, index if mesh_index is None else mesh_index, ppw)
    nx = 2 * math.ceil(interior_half_um / dx) + 2 * pml_cells
    span = nx * dx
    width = transverse_cells * dx
    r = Region(size=(span, width, 1), mesh=dx, mesh_steps=(dx, dx, 1),
               material_sampling='yee', interface_method='staircase', pml_cells=pml_cells,
               backend='cuda', precision=precision, cuda_kernel='fused',
               cuda_monitor_kernel='fused', snapshot_interval=10000,
               boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'),
                                     y_max=BoundaryFace(kind='periodic')))
    r.steps = math.ceil(duration_fs * 1e-15 / r.time_step)
    snap = lambda x: round(x / dx) * dx
    thickness = 2.1 / index if slab_thickness_um is None else slab_thickness_um
    structures = []
    if kind == 'slab':
        structures = [Structure(center=(slab_center_um, 0, 0), size=(thickness, width, 1),
                                material='dielectric')]
    elif kind == 'interface':
        left = dx / 2
        right = span / 2 + dx
        structures = [Structure(center=((left + right) / 2, 0, 0),
                                size=(right - left, width, 1), material='dielectric')]
    elif kind != 'empty':
        raise ValueError('Unknown geometry')
    spec = SpectrumSettings(sampling='wavelength', wavelength_start=1.2, wavelength_stop=1.8,
                            frequency_points=frequency_points, apodization='none')
    materials = [Material(name='dielectric', index=index)]
    if lossy:
        # Leg 12's control material, verbatim: solver defaults except epsilon_inf.
        materials = [Material(name='dielectric', model='lorentz', epsilon_inf=index ** 2)]
    p = Project(name=f'{kind}-n{index}-ppw{ppw}-pml{pml_cells}-pad{interior_half_um}',
                region=r, materials=materials, structures=structures,
                sources=[Source(id='source', kind='plane', injection='oneway', normal='x',
                                component='Ez', center=(snap(-2.5), 0, 0), size=(0, width, 0),
                                wavelength=1.5, pulse_cycles=1)],
                monitors=[FieldMonitor(id=name, name=name, center=(snap(x), 0, 0),
                                       size=(0, width, 1), normal='x',
                                       spectrum=spec.model_copy(deep=True))
                          for name, x in [('reflection', -2.), ('transmission', 2.),
                                          ('scattered', -3.)]])
    return Project.model_validate(p.model_dump())


def case_knobs(config, overrides):
    knobs = dict(config['baseline'])
    knobs.update(overrides)
    return knobs


def empty_key(knobs, points):
    return (knobs['ppw'], knobs['duration_fs'], knobs['pml_cells'], knobs['interior_half_um'],
            knobs['index'], knobs['transverse_cells'], points)


def store_monitors(path, result, names=('reflection', 'transmission', 'scattered')):
    np.savez_compressed(path, **{f'{name}_{field}': np.asarray(result.field_monitor(name)[field])
                                 for name in names for field in ('flux', 'frequency_hz')})


def decompose(result, reference, measured):
    """The two other numbers: the R-composition gap, and the empty-run denominator asymmetry.

    `absolute_change` is the difference between the archived `R` (the solver's difference-field
    flux over the empty-run denominator) and the same ratio formed from the *stored total* fluxes:
    `1 - flux_sample(reflection) / |flux_empty(reflection)|` for a forward-going reference.
    """
    flux_sample = np.asarray(result.field_monitor('reflection')['flux'], dtype=float)
    flux_empty = np.asarray(reference.field_monitor('reflection')['flux'], dtype=float)
    r_stored = 1.0 - flux_sample / np.abs(flux_empty)
    absolute_change = float(np.max(np.abs(np.asarray(measured['R']) - r_stored)))
    refl = np.abs(flux_empty)
    trans = np.abs(np.asarray(reference.field_monitor('transmission')['flux'], dtype=float))
    relative_change = float(np.max(np.abs(refl / trans - 1.0)))
    return absolute_change, relative_change


def geometry_check(project, strict):
    """`parity.geometry_assertions` for the slab fixtures, a reduced check for the null.

    The strict check includes `realized thickness == physical thickness`, which counts cells with
    `eps > 1`. An index-`1.0` slab has no such cells -- that is exactly what makes it the null
    (no interface) -- so the strict check cannot apply to it. The null still gets the parts that
    matter for validity: a finite monotone mesh, a buildable one-way plan, monitors inside the
    interior.
    """
    if strict:
        return geometry_assertions(project)
    r = project.region
    assert np.isfinite(np.concatenate(r.mesh_nodes)).all()
    assert all(np.all(np.diff(a) > 0) for a in r.mesh_nodes)
    assert project.sources[0].injection == 'oneway'
    oneway_plan(project.sources[0], r)
    lo, hi = r.interior_bounds(0)
    assert all(lo < m.center[0] < hi for m in project.monitors)
    return dict(shape=list(r.shape), dt_s=r.time_step, steps=r.steps,
                interior_x_um=[lo, hi])


# The pre-declared strength of the negative control: the lossy slab's `energy_residual` must be
# "at least 10^4 times the effect under test" (configs/leg15-closure.json `negative_control.expect`;
# the same bound in checks/test_leg15_closure.py). The effect under test is the baseline row.
NEGATIVE_CONTROL_FACTOR = 1e4
NEGATIVE_CONTROL_FAILED = 'failed-as-required'


def negative_control_evidence(declared, rows):
    """The declared control plus what this run measured for it.

    `executed` and `observed` come from the stored rows only. Without both the control row and
    the baseline row there is no evidence, and the control stays vacuous: `verdict_exit` refuses.
    """
    residuals = {row['id']: row['measured']['energy_residual'] for row in rows}
    control = dict(declared)
    case, baseline = residuals.get(declared['case']), residuals.get('baseline')
    if case is None or baseline is None:
        return dict(control, status='not-run', executed=False, observed=None)
    failed = case >= NEGATIVE_CONTROL_FACTOR * baseline
    return dict(control, executed=True,
                status=NEGATIVE_CONTROL_FAILED if failed else 'passed',
                observed=dict(energy_residual=case, baseline_energy_residual=baseline,
                              required_factor=NEGATIVE_CONTROL_FACTOR))


def verdict_exit(readout):
    """The exit status a readout earns. Pure: reads `readout`, writes only the refusal message.

    0   every entry of `readout['criteria']` is `True`, and `readout['negative_control']` ran
        (`executed` is `True`, `observed` is non-empty) and failed as required (`status`);
    61  a criterion is not `True`; the message names every one;
    62  the negative control is missing, vacuous, or did not fail as required;
    63  there is no criteria block, or it declares nothing.

    The control is judged before the criteria: a run whose control cannot fail has not shown that
    its criteria can either.
    """
    criteria = readout.get('criteria')
    if not isinstance(criteria, dict) or not criteria:
        print('verdict 63: the readout carries no pre-declared criteria block', file=sys.stderr)
        return 63
    control = readout.get('negative_control')
    if not isinstance(control, dict):
        print('verdict 62: the negative control is missing', file=sys.stderr)
        return 62
    if control.get('executed') is not True or not control.get('observed'):
        print(f"verdict 62: the negative control {control.get('case')!r} is vacuous: "
              f"executed={control.get('executed')!r}, observed={control.get('observed')!r}",
              file=sys.stderr)
        return 62
    if control.get('status') != NEGATIVE_CONTROL_FAILED:
        print(f"verdict 62: the negative control {control.get('case')!r} has status "
              f"{control.get('status')!r}, not {NEGATIVE_CONTROL_FAILED!r}", file=sys.stderr)
        return 62
    violated = [name for name, holds in criteria.items() if holds is not True]
    if violated:
        print(f"verdict 61: pre-declared criteria violated: {', '.join(violated)}",
              file=sys.stderr)
        return 61
    return 0


def main():
    config, run = load_run()
    effective_path = Path(os.environ['EFFECTIVE_JSON'])
    effective = json.loads(effective_path.read_text())
    points = config['frequency_points']
    rows = []
    references = {}
    for overrides in config['cases']:
        case_id = overrides.pop('id')
        knobs = case_knobs(config, overrides)
        project = build(knobs.get('family', 'slab'), knobs['index'], knobs['ppw'],
                        knobs['duration_fs'],
                        pml_cells=knobs['pml_cells'], interior_half_um=knobs['interior_half_um'],
                        slab_center_um=knobs['slab_center_um'],
                        transverse_cells=knobs['transverse_cells'],
                        frequency_points=knobs.get('frequency_points', points),
                        lossy=knobs.get('lossy', False),
                        mesh_index=knobs.get('mesh_index'),
                        slab_thickness_um=knobs.get('slab_thickness_um'))
        resolved = geometry_check(project, strict=not knobs.get('null_interface'))
        effective['resolved_cases'].append(dict(id=case_id, knobs=knobs, **resolved))
        write_json(effective_path, effective)
        points_case = knobs.get('frequency_points', points)
        key = empty_key(knobs, points_case)
        if key not in references:
            empty = build('empty', knobs['index'], knobs['ppw'], knobs['duration_fs'],
                          pml_cells=knobs['pml_cells'], interior_half_um=knobs['interior_half_um'],
                          transverse_cells=knobs['transverse_cells'], frequency_points=points_case,
                          mesh_index=knobs.get('mesh_index'))
            geometry_check(empty, strict=not knobs.get('null_interface'))
            references[key] = simulate(empty, config['graph_steps'])
        reference, reference_seconds = references[key]
        sample, sample_seconds = simulate(project, config['graph_steps'])
        measured = spectra(sample, reference)
        absolute_change, relative_change = decompose(sample, reference, measured)
        store_monitors(run / f'{case_id}-sample-monitors.npz', sample)
        store_monitors(run / f'{case_id}-empty-monitors.npz', reference)
        row = dict(id=case_id, shape=resolved['shape'], steps=resolved['steps'],
                   dt_s=resolved['dt_s'], sample_elapsed_seconds=sample_seconds,
                   reference_elapsed_seconds=reference_seconds,
                   measured=dict(R=measured['R'], T=measured['T'],
                                 wavelength_um=measured['wavelength_um'],
                                 energy_residual=measured['energy_residual'],
                                 empty_reflection_floor=measured['empty_reflection_floor'],
                                 absolute_change=absolute_change,
                                 relative_change=relative_change))
        rows.append(row)
        print(json.dumps(dict(case=case_id, ppw=knobs['ppw'], pml=knobs['pml_cells'],
                              pad=knobs['interior_half_um'], center=knobs['slab_center_um'],
                              duration=knobs['duration_fs'], points=points_case,
                              index=knobs['index'],
                              energy_residual=measured['energy_residual'],
                              absolute_change=absolute_change,
                              relative_change=relative_change)) , flush=True)
    readout = dict(policy=config['policy'], status='completed', simulation_executed=True,
                   fixture=('The leg-3 slab fixture with the numerical knobs moved one at a time: '
                            'PML cells, interior half-length, slab centre, duration, frequency '
                            'points and slab index (1.0 is the no-interface null).'),
                   stages=['F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run',
                           'F6-readout'],
                   stage_source={'F1-structure': 'leg15_closure.build (parity.project_for + knobs)',
                                 'F2-mesh': 'leg15_closure.build',
                                 'F3-source': 'parity.project_for (one-way plane source)',
                                 'F4-monitors': 'parity.project_for (existing planes)',
                                 'F5-run': 'parity.simulate',
                                 'F6-readout': ('parity.spectra; leg15_closure.decompose '
                                                '(archived monitor arrays)')},
                   hypotheses=config['hypotheses'],
                   negative_control=negative_control_evidence(config['negative_control'], rows),
                   cases=rows, units=config['units'])
    # The config declares hypotheses H1-H4 (statements to attribute), not pass/fail criteria, and
    # this driver evaluates none. The readout therefore carries no `criteria` block and the verdict
    # is 63 by contract, never a silent 0.
    readout['exit_status'] = verdict_exit(readout)
    write_json(run / 'readout.json', readout)
    effective.update(simulation_executed=True, stage='completed')
    write_json(effective_path, effective)
    return readout['exit_status']


if __name__ == '__main__':
    sys.exit(main())
