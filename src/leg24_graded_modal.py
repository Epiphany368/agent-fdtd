"""Leg 24: exact uniform modal supports in an opt-in graded exterior.

All physical inputs and scene construction are the archived Leg 1 N8 case.
No reference project is imported. The independent frozen fork supplies C1's
before fields; isolated processes avoid sharing imported solver modules.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def realised_metrics(region):
    axes = region.mesh_nodes[:2 if region.dimension == '2d' else 3]
    widths = [np.diff(a) for a in axes]
    if any(not np.isfinite(d).all() or np.any(d <= 0) for d in widths):
        raise ValueError('Mesh nodes must be finite and strictly increasing.')
    ratios = [float(max(np.max(d[1:]/d[:-1]), np.max(d[:-1]/d[1:])))
              if len(d) > 1 else 1. for d in widths]
    return dict(shape=list(region.shape), finest_realised_spacing_um=float(min(d.min() for d in widths)),
                axis_min_step_um=[float(d.min()) for d in widths],
                axis_max_step_um=[float(d.max()) for d in widths],
                max_adjacent_ratio=max(ratios), axis_adjacent_ratio=ratios,
                node_sha256=[digest(a) for a in axes], dt_s=region.time_step, steps=region.steps)


def compare_default(before, after, recorded):
    # Exact numeric equality and byte equality, with no numerical tolerance.
    return (before['measured'] == after['measured'] == {k: recorded[k] for k in ('T', 'R', 'epsilon_sha256')}
            and before['field_sha256'] == after['field_sha256']
            and before['mode_sha256'] == after['mode_sha256']
            and before['launch_identity'] == after['launch_identity'])


def graded_exit(uniform, graded, tolerance, ratio_limit):
    if graded['mesh']['max_adjacent_ratio'] > ratio_limit:
        return 58, {}
    if graded.get('operator_stop'):
        return 55, {}
    # Floating-point node subtraction has roundoff even on the fine lattice.
    # This tests realised minima, never the requested nominal steps.
    if not np.allclose(uniform['mesh']['axis_min_step_um'], graded['mesh']['axis_min_step_um'],
                       rtol=1e-10, atol=0):
        return 55, {'reason': 'Unequal realised finest spacing; C2 not comparable.'}
    errors = {key: abs(graded['measured'][key]-uniform['measured'][key])/abs(uniform['measured'][key])
              for key in ('T', 'R')}
    return (0 if all(math.isfinite(x) and x <= tolerance for x in errors.values()) else 57), errors


def worker(config_path, solver_root, kind, output):
    # Import the selected solver first. parity's normal snapshot bootstrap
    # cannot then replace it with a different solver in this child process.
    sys.path.insert(0, solver_root)
    import torchfdtd
    import torch
    from torchfdtd import AdjointOptions
    from torchfdtd.mode_branches import prepare_aperture_modal_launch
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene
    cfg = json.loads(Path(config_path).read_text())
    sc = scene(cfg, cfg['cases'][0])
    p = sc['p']
    if sc['r'] is not p.region:
        raise RuntimeError('Scene material builder and Project must share their Region.')
    if kind == 'graded':
        from torchfdtd.mesh import configure_auto_mesh
        p.region.mesh_type = 'graded'
        p.region.time_step_rule = 'realised_axis_minimum'
        configure_auto_mesh(p)
        p.region.steps = math.ceil(cfg['cases'][0]['duration_fs']*1e-15/p.region.time_step)
    r = p.region
    row = dict(solver_module=str(Path(torchfdtd.__file__).resolve()), mesh=realised_metrics(r),
               simulation_executed=False)
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
    row.update(measured=dict(T=float((amps['far']['forward']/amps['near']['forward']).abs().square().sum()),
                             R=float((amps['near']['backward']/amps['near']['forward']).abs().square().sum()),
                             epsilon_sha256=digest(epsilon.detach().cpu().numpy())),
               field_sha256={k: digest(a) for k, a in arrays.items()}, mode_sha256=digest(launch.mode.fields),
               launch_identity=launch.identity, launch_scores=scores,
               launch_aperture=[list(x) for x in launch.aperture],
               mode_spacing_um=list(launch.mode.spacing_um), normal_step_um=launch.normal_step_um,
               maxwell_residual=launch.mode.maxwell_residual, simulation_executed=True)
    Path(output).write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    return 0


def main():
    from parity import load_run, write_json
    cfg, run = load_run()
    snapshot_solver = Path(os.environ['PARITY_SNAPSHOT'])/'src/solver'
    report = dict(leg=24, policy=cfg['policy'], readout_kind='diagnostics',
                  readout_reason=('Modal default-path identity and uniform/graded comparison; '
                                  'cases are worker diagnostics, not an F1-F6 pipeline readout.'),
                  acceptance_granted=False, status='running', verdict='partial', cases={})

    def evaluate(label, solver, kind):
        output = run/(label+'.json')
        result = subprocess.run([sys.executable, '-B', '-u', __file__, '--worker',
                                 os.environ['PARITY_CONFIG'], str(solver), kind, str(output)], check=False)
        if result.returncode not in (0, 55):
            raise RuntimeError(f'{label} worker exited {result.returncode}; stop, no retry.')
        row = json.loads(output.read_text())
        report['cases'][label] = row
        write_json(run/'readout.json', report)
        return row

    if cfg['mode'] == 'uniform-baseline':
        frozen = os.environ.get('SOLVER_FROZEN')
        if not frozen:
            raise RuntimeError('SOLVER_FROZEN is required; source env/refs.env with export enabled.')
        before = evaluate('before-frozen', Path(frozen), 'uniform')
        after = evaluate('after-snapshot', snapshot_solver, 'uniform')
        recorded = cfg['expected']['cases'][cfg['cases'][0]['id']]
        code = 0 if compare_default(before, after, recorded) else 56
        report.update(recorded=recorded, bit_identical=code == 0)
    else:
        uniform = evaluate('uniform', snapshot_solver, 'uniform')
        graded = evaluate('graded', snapshot_solver, 'graded')
        code, errors = graded_exit(uniform, graded, cfg['agreement_relative'], cfg['max_realised_ratio'])
        report.update(relative_error=errors,
                      cell_reduction_percent=100*(1-math.prod(graded['mesh']['shape'])/math.prod(uniform['mesh']['shape'])))
    report.update(exit_code=code, status='completed' if code == 0 else 'stopped',
                  verdict='accepted' if code == 0 else 'partial',
                  simulation_executed=any(x['simulation_executed'] for x in report['cases'].values()))
    write_json(run/'readout.json', report)
    (run/'verdict').write_text(report['verdict']+'\n')
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    effective.update(resolved_cases=[dict(id=k, **v['mesh']) for k, v in report['cases'].items()],
                     simulation_executed=report['simulation_executed'], exit_code=code)
    write_json(os.environ['EFFECTIVE_JSON'], effective)
    print(json.dumps(report), flush=True)
    return code


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.exit(worker(*sys.argv[2:]))
    sys.exit(main())
