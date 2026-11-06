"""Leg 36: attribute Leg 24's graded-mesh disagreement to the x or y mesh metric.

One control, as registered in plan.md LEG 36: take Leg 24's archived graded
scene and add one refinement in x over [-6.1, 6.1] um, the complete PML-free
interior. Then solve the existing uniform + graded pair in one launcher
invocation. The pre-declared decision rule in plan.md is applied verbatim; this
driver does not interpret the outcome.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PHYSICAL_KEYS = ('kind', 'core_index', 'cladding_index', 'wavelength_um', 'width_um', 'precision')
CASE_KEYS = ('id', 'dx_um', 'dy_um', 'z_cells', 'periods', 'duration_fs', 'shift_nm', 'transition_nm')
METADATA_KEYS = {'policy', 'task', 'mode', 'provenance', 'leg24_graded_result'}
TINY_UM = 1e-9


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def physical_inputs(cfg):
    return dict(**{key: cfg[key] for key in PHYSICAL_KEYS},
                cases=[{key: case[key] for key in CASE_KEYS} for case in cfg['cases']])


def realised_metrics(region):
    axes = region.mesh_nodes[:2 if region.dimension == '2d' else 3]
    widths = [np.diff(a) for a in axes]
    if any(not np.isfinite(d).all() or np.any(d <= 0) for d in widths):
        raise ValueError('Mesh nodes must be finite and strictly increasing.')
    ratios = [float(max(np.max(d[1:]/d[:-1]), np.max(d[:-1]/d[1:])))
              if len(d) > 1 else 1. for d in widths]
    return dict(shape=list(region.shape), finest_realised_spacing_um=float(min(d.min() for d in widths)),
                coarsest_realised_spacing_um=float(max(d.max() for d in widths)),
                axis_min_step_um=[float(d.min()) for d in widths],
                axis_max_step_um=[float(d.max()) for d in widths],
                max_adjacent_ratio=max(ratios), axis_adjacent_ratio=ratios,
                node_sha256=[digest(a) for a in axes], dt_s=region.time_step, steps=region.steps)


def refine_scope(region, reference_nodes):
    """Check that the one added box refines x only and leaves y and z untouched."""
    nodes = region.mesh_nodes
    dx = region.axis_steps[0]
    return dict(
        x_uniform=bool(np.allclose(np.diff(nodes[0]), dx, rtol=1e-10, atol=0)),
        x_changed=bool(not np.array_equal(nodes[0], reference_nodes[0])),
        y_unchanged=bool(np.array_equal(nodes[1], reference_nodes[1])),
        z_unchanged=bool(np.array_equal(nodes[2], reference_nodes[2])),
        reference_node_sha256=[digest(a) for a in reference_nodes],
        refined_node_sha256=[digest(a) for a in nodes])


def worker(config_path, solver_root, kind, output):
    # Import the selected solver first. parity's snapshot bootstrap cannot then
    # replace it with a different solver in this child process.
    sys.path.insert(0, solver_root)
    import torchfdtd
    import torch
    from torchfdtd import AdjointOptions, MeshRefinement
    from torchfdtd.mesh import configure_auto_mesh
    from torchfdtd.mode_branches import prepare_aperture_modal_launch
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene
    cfg = json.loads(Path(config_path).read_text())
    c = cfg['cases'][0]
    sc = scene(cfg, c)
    p = sc['p']
    if sc['r'] is not p.region:
        raise RuntimeError('Scene material builder and Project must share their Region.')
    row = dict(leg=36, kind=kind, solver_module=str(Path(torchfdtd.__file__).resolve()),
               physical_inputs=physical_inputs(cfg), simulation_executed=False)
    if kind == 'graded':
        reference = scene(cfg, c)['p']
        reference.region.mesh_type = 'graded'
        reference.region.time_step_rule = 'realised_axis_minimum'
        configure_auto_mesh(reference)
        reference_nodes = reference.region.mesh_nodes
        r = p.region
        r.mesh_type = 'graded'
        r.time_step_rule = 'realised_axis_minimum'
        configure_auto_mesh(p)
        lo, hi = cfg['x_refinement_um']
        r.mesh_refinements = list(r.mesh_refinements) + [MeshRefinement(
            name='leg36-x-refinement', center=(0.5*(lo+hi), 0., 0.), size=(hi-lo, TINY_UM, TINY_UM))]
        r.steps = math.ceil(c['duration_fs']*1e-15/r.time_step)
        scope = refine_scope(r, reference_nodes)
        row.update(refinement_scope=scope,
                   refinement_box=dict(center_um=[0.5*(lo+hi), 0., 0.],
                                       size_um=[hi-lo, TINY_UM, TINY_UM]))
        if not (scope['x_uniform'] and scope['x_changed'] and scope['y_unchanged']
                and scope['z_unchanged']):
            row['operator_stop'] = 'The x-only refinement could not be expressed without moving y or z nodes.'
            Path(output).write_text(json.dumps(row, indent=2)+'\n')
            return 55
    r = p.region
    row['mesh'] = realised_metrics(r)
    launches = []
    try:
        for i in range(2):
            kwargs = dict(mode_index=i, num_modes=2)
            if kind == 'graded':
                kwargs['allow_graded'] = True
            launches.append(prepare_aperture_modal_launch(p, sc['section'], **kwargs))
    except ValueError as error:
        if kind != 'graded':
            raise
        row['operator_stop'] = str(error)
        Path(output).write_text(json.dumps(row, indent=2)+'\n')
        return 55
    scores = [float(np.sum(abs(x.mode.fields[..., 1])**2)/np.sum(abs(x.mode.fields[..., [1, 2]])**2))
              for x in launches]
    launch = launches[int(np.argmax(scores))]
    if max(scores) <= 1-1e-8 or launch.mode.maxwell_residual >= 1e-5:
        raise RuntimeError('Archived mode selection is unresolved.')
    # Check all modal detectors before allocating simulation state.
    try:
        for m in p.monitors:
            if m.id in ('near', 'far', 'outside'):
                launch.detector_mode(m.center[0])
    except ValueError as error:
        if kind != 'graded':
            raise
        row['operator_stop'] = str(error)
        Path(output).write_text(json.dumps(row, indent=2)+'\n')
        return 55
    width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
    with torch.no_grad():
        epsilon = sc['epsilon'](width)
        model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=32, gpu_budget_bytes=32*1024**3))
        planes = model(epsilon, [299792458./(cfg['wavelength_um']*1e-6)])
    amps = {k: modal_plane_amplitudes(planes[k], launch) for k in ('near', 'far', 'outside')}
    arrays = {name: plane.fields.detach().cpu().numpy() for name, plane in planes.items()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError('Nonfinite modal monitor fields.')
    np.savez_compressed(Path(output).with_suffix('.npz'), **arrays)
    measured = dict(T=float((amps['far']['forward']/amps['near']['forward']).abs().square().sum()),
                    R=float((amps['near']['backward']/amps['near']['forward']).abs().square().sum()),
                    epsilon_sha256=digest(epsilon.detach().cpu().numpy()))
    row.update(measured=measured,
               field_sha256={k: digest(a) for k, a in arrays.items()}, mode_sha256=digest(launch.mode.fields),
               launch_identity=launch.identity, launch_scores=scores,
               launch_aperture=[list(x) for x in launch.aperture],
               mode_spacing_um=list(launch.mode.spacing_um), normal_step_um=launch.normal_step_um,
               maxwell_residual=launch.mode.maxwell_residual, simulation_executed=True)
    if kind == 'uniform':
        # The uniform case is Leg 24's unchanged baseline. Reproducing it against
        # the archived record is the physical-input identity check for this leg.
        recorded = cfg['expected']['cases'][c['id']]
        tolerance = cfg['expected']['tolerance_relative']
        row['archived_match'] = dict(
            T_relative_error=abs(measured['T']-recorded['T'])/abs(recorded['T']),
            R_relative_error=abs(measured['R']-recorded['R'])/abs(recorded['R']),
            epsilon_sha256_equal=measured['epsilon_sha256'] == recorded['epsilon_sha256'])
        ok = (row['archived_match']['T_relative_error'] <= tolerance
              and row['archived_match']['R_relative_error'] <= tolerance
              and row['archived_match']['epsilon_sha256_equal'])
        Path(output).write_text(json.dumps(row, indent=2)+'\n')
        if not ok:
            return 56
        print(json.dumps(row), flush=True)
        return 0
    Path(output).write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    return 0


def archived_config_differences(cfg, root):
    """Pre-run check of the two archived Leg 24 configs and of this config.

    Stop condition 4: if the archived pair differs in a physical input, the
    comparison is meaningless and nothing may run.
    """
    uniform = json.loads((root/'configs/leg24-uniform-baseline.json').read_text())
    graded = json.loads((root/'configs/leg24-graded-check.json').read_text())
    current = physical_inputs(cfg)
    archived_pair = physical_inputs(uniform) == physical_inputs(graded)
    current_matches_graded = current == physical_inputs(graded)
    return dict(archived_uniform_equals_archived_graded=archived_pair,
                current_equals_archived_graded=current_matches_graded,
                current=current, archived_graded=physical_inputs(graded))


def input_difference_table(cfg, root):
    """Model inputs only: every row is marked physical or numerical."""
    archived = json.loads((root/'configs/leg24-graded-check.json').read_text())
    rows = []
    for key in sorted(set(cfg) | set(archived)):
        if key in METADATA_KEYS or cfg.get(key) == archived.get(key):
            continue
        physical = key in PHYSICAL_KEYS or key == 'cases'
        rows.append(dict(input=key, leg24_graded=archived.get(key), leg36=cfg.get(key),
                         classification='physical' if physical else 'numerical',
                         note=('physical model input' if physical else
                               'mesh refinement; the only numerical change')))
    return rows


def metadata_difference_table(cfg, root):
    """Labels and prose that changed. Explicitly not model inputs."""
    archived = json.loads((root/'configs/leg24-graded-check.json').read_text())
    rows = []
    for key in sorted(METADATA_KEYS | {'leg24_graded_result'}):
        if cfg.get(key) != archived.get(key):
            rows.append(dict(field=key, leg24_graded=archived.get(key), leg36=cfg.get(key),
                             role='non-input metadata; not physical and not numerical'))
    return rows


def main():
    from parity import load_run, write_json
    cfg, run = load_run()
    root = Path(__file__).resolve().parents[1]
    snapshot_solver = Path(os.environ['PARITY_SNAPSHOT'])/'src/solver'
    report = dict(leg=36, policy=cfg['policy'], readout_kind='diagnostics',
                  readout_reason=('M4 attribution: one pre-declared x-refinement control on the '
                                  'Leg 24 graded scene; cases are worker diagnostics, not an F1-F6 pipeline readout.'),
                  acceptance_granted=False, status='running', verdict='partial',
                  decision_threshold=cfg['agreement_relative'],
                  input_differences_vs_leg24_graded=input_difference_table(cfg, root),
                  metadata_differences_vs_leg24_graded=metadata_difference_table(cfg, root),
                  archived_config_check=archived_config_differences(cfg, root), cases={})

    def evaluate(label, solver, kind):
        output = run/(label+'.json')
        result = subprocess.run([sys.executable, '-B', '-u', __file__, '--worker',
                                 os.environ['PARITY_CONFIG'], str(solver), kind, str(output)], check=False)
        if result.returncode not in (0, 55, 56):
            raise RuntimeError(f'{label} worker exited {result.returncode}; stop, no retry.')
        row = json.loads(output.read_text())
        report['cases'][label] = row
        write_json(run/'readout.json', report)
        return result.returncode, row

    if not report['archived_config_check']['archived_uniform_equals_archived_graded']:
        report.update(status='stopped', verdict='rejected', exit_code=56,
                      stop_reason='The two archived Leg 24 configs differ in a physical input.')
    elif not report['archived_config_check']['current_equals_archived_graded']:
        report.update(status='stopped', verdict='rejected', exit_code=56,
                      stop_reason='This config differs from the archived Leg 24 graded config in a physical input.')
    else:
        uniform_code, uniform = evaluate('uniform', snapshot_solver, 'uniform')
        if uniform_code == 56:
            code = 56
            report['stop_reason'] = 'The uniform baseline did not reproduce the archived physical scene.'
        else:
            graded_code, graded = evaluate('graded-x-refined', snapshot_solver, 'graded')
            if graded_code == 55:
                code = 55
                report['stop_reason'] = graded.get('operator_stop')
            elif graded_code == 56:
                code = 56
                report['stop_reason'] = 'The graded worker reported a physical-input mismatch.'
            else:
                delta = {key: graded['measured'][key]-uniform['measured'][key] for key in ('T', 'R')}
                relative = {key: abs(delta[key])/abs(uniform['measured'][key]) for key in ('T', 'R')}
                fired = all(math.isfinite(x) and x <= cfg['agreement_relative']
                            for x in relative.values())
                branch = ('x-longitudinal-coarse-segment' if fired else 'x-exonerated-y-implicated')
                code = 0
                archived = cfg['leg24_graded_result']
                reported = {key: archived['relative_error_'+key] for key in ('T', 'R')}
                reduction = {key: reported[key]/relative[key] for key in ('T', 'R')}
                if fired:
                    statement = (
                        'Both |dT|/|T| and |dR|/|R| are at or below 1e-3: the gap is the '
                        'longitudinal coarse segment in x, and candidate (b) material sampling '
                        'is bounded below 1e-3 in the same run.')
                else:
                    statement = (
                        'The first line did not fire: |dT|/|T| = %.6g and |dR|/|R| = %.6g, both '
                        'above 1e-3. The second line therefore applies in the registered binary '
                        'structure, but its registered magnitudes did not reproduce: LEG 24 '
                        'recorded %.6g (T) and %.6g (R), and the x refinement reduced those gaps '
                        'by a factor %.6g (T) and %.6g (R). The residual is still above 1e-3. '
                        'The further experiment to separate material sampling from the y metric '
                        'was not run, by rule.'
                        % (relative['T'], relative['R'], reported['T'], reported['R'],
                           reduction['T'], reduction['R']))
                report.update(delta_absolute=delta, relative_error=relative,
                              leg24_graded_relative_error=reported,
                              error_reduction_factor=reduction,
                              decision_rule_fired=branch, decision_statement=statement,
                              first_line_fired=fired)
        report.update(exit_code=code)

    report.update(status='completed' if report['exit_code'] == 0 else 'stopped',
                  verdict='accepted' if report['exit_code'] == 0 else 'rejected',
                  simulation_executed=any(x['simulation_executed'] for x in report['cases'].values()))
    write_json(run/'readout.json', report)
    (run/'verdict').write_text(report['verdict']+'\n')
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    effective.update(resolved_cases=[dict(id=k, **v['mesh']) for k, v in report['cases'].items()],
                     simulation_executed=report['simulation_executed'],
                     exit_code=report['exit_code'],
                     decision_rule_fired=report.get('decision_rule_fired'))
    write_json(os.environ['EFFECTIVE_JSON'], effective)
    print(json.dumps(report), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.exit(worker(*sys.argv[2:]))
    sys.exit(main())
