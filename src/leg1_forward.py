"""Leg 1, F1-F6 on the N8 fixture: one command, one forward, one readout.

Stages, in the order the report will name them:

  F1 structure  -- the N8 scene (imported, not re-implemented)
  F2 mesh       -- the Region the scene carries
  F3 source     -- the aperture eigenmode launch, solved on the same cross-section
  F4 monitors   -- the near/far/outside planes the scene carries
  F5 run        -- ModeInjectedPlaneSimulation, forward only
  F6 readout    -- T and R as the recorded row defines them, plus the epsilon hash

It compares its readout against the values recorded by the reference project and
stops with a non-zero status when they differ. Nothing here is a new implementation
of a stage: every stage is an import from a copied module.

Run:  ./tools/run.sh configs/leg1-n8-forward.json src/leg1_forward.py
"""
from parity import *
import hashlib
import torch
from torchfdtd import AdjointOptions
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
from torchfdtd.mode_branches import prepare_aperture_modal_launch

from run_b9_native_shape_n8 import scene

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')

# Which module and symbol each stage came from. The report prints this, and the
# criterion is that the list has no entry written for this leg.
STAGE_SOURCE = {
    'F1-structure': 'run_b9_native_shape_n8.scene',
    'F2-mesh': 'run_b9_native_shape_n8.scene (Region)',
    'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch',
    'F4-monitors': 'run_b9_native_shape_n8.scene (FieldMonitor list)',
    'F5-run': 'torchfdtd.mode_injection.ModeInjectedPlaneSimulation',
    'F6-readout': 'torchfdtd.mode_injection.modal_plane_amplitudes',
}


def relative_error(measured, recorded):
    if recorded == 0:
        return abs(measured)
    return abs(measured - recorded) / abs(recorded)


def launch_for(p, section):
    """The scene launch the recorded run used: best TE-like mode of the first two."""
    modes = [prepare_aperture_modal_launch(p, section, mode_index=i, num_modes=2) for i in range(2)]
    scores = [float(np.sum(abs(m.mode.fields[..., 1])**2)
                    / np.sum(abs(m.mode.fields[..., [1, 2]])**2)) for m in modes]
    launch = modes[int(np.argmax(scores))]
    assert max(scores) > 1 - 1e-8 and launch.mode.maxwell_residual < 1e-5, \
        'The launch eigenmode is not resolvable on this cross-section'
    return launch, scores


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    frequency = [C0 / (cfg['wavelength_um'] * 1e-6)]
    tolerance = cfg['expected']['tolerance_relative']
    expected = cfg['expected']['cases']

    report = {
        'policy': cfg['policy'],
        'status': 'in_progress',
        'stages': list(STAGES),
        'stage_source': dict(STAGE_SOURCE),
        'fixture': 'N8 (eight periods), the recorded guided fixture',
        'acceptance_granted': False,
        'cases': [],
    }
    Path(run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')

    failed = []
    for case in cfg['cases']:
        sc = scene(cfg, case)
        r, p = sc['r'], sc['p']
        launch, scores = launch_for(p, sc['section'])
        width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
        with torch.no_grad():
            epsilon = sc['epsilon'](width)
        model = ModeInjectedPlaneSimulation(
            p, launch, AdjointOptions(checkpoints=32, gpu_budget_bytes=32 * 1024**3))
        with torch.no_grad():
            planes = model(epsilon, frequency)
        amps = {key: modal_plane_amplitudes(planes[key], launch) for key in ('near', 'far', 'outside')}
        measured = {
            'T': float((amps['far']['forward'] / amps['near']['forward']).abs().square().sum()),
            'R': float((amps['near']['backward'] / amps['near']['forward']).abs().square().sum()),
            'epsilon_sha256': hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        recorded = expected[case['id']]
        errors = {key: relative_error(measured[key], recorded[key]) for key in ('T', 'R')}
        errors['epsilon_sha256'] = 0.0 if measured['epsilon_sha256'] == recorded['epsilon_sha256'] else 1.0
        ok = all(value <= tolerance for value in errors.values())
        row = {
            'id': case['id'],
            'shape': list(r.shape),
            'steps': r.steps,
            'dt_s': r.time_step,
            'launch_identity': launch.identity,
            'launch_scores': scores,
            'measured': measured,
            'recorded': recorded,
            'relative_error': errors,
            'within_tolerance': ok,
        }
        report['cases'].append(row)
        effective['resolved_cases'].append(dict(
            id=case['id'], shape=r.shape, dt_s=r.time_step, steps=r.steps,
            launch_identity=launch.identity, mesh_um=[case['dx_um'], case['dy_um']],
            precision=case.get('precision', cfg['precision'])))
        Path(run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')
        Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not ok:
            failed.append(case['id'])

    report['status'] = 'mismatch' if failed else 'completed'
    report['mismatched_cases'] = failed
    report['simulation_executed'] = True
    Path(run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'leg1 forward: {len(report["cases"]) - len(failed)}/{len(report["cases"])} cases within '
          f'{tolerance:g} relative', flush=True)
    return 52 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
