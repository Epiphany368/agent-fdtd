"""Leg 1, criterion C2 (default path unchanged), on the registered G8/N8 fixture.

This is the forward half of the reference project's loop evaluator
(`run_g3_loop.NativeEvaluator.__call__`, adjoint=False). It produces the four numbers
the criterion names:

  P_in   launched incident power   = |near forward modal amplitude|^2 * unit-mode flux
  P_up   upward-coupled power      = the 'top' monitor flux
  J      the objective             = P_up / P_in
  epsilon_sha256                   = hash of the built permittivity tensor

and compares them with the values recorded by the reference project. Every stage is an
import of a copied module; no stage is re-implemented here.

Run:  ./tools/run.sh configs/leg1-objective.json src/leg1_objective.py
"""
import os
import sys
from pathlib import Path

import hashlib
import json
import time

import numpy as np
import torch

import run_g1b_gradient_taylor as native
from run_g5_discretisation import g5_fixture

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'run_g5_discretisation.g5_fixture (build), on run_g1b_gradient_taylor.fixture',
    'F2-mesh': 'run_b9_native_shape_n8.scene (Region) via run_g1b_gradient_taylor.fixture',
    'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch',
    'F4-monitors': 'run_b9_native_shape_n8.scene (FieldMonitor list)',
    'F5-run': 'torchfdtd.mode_injection.ModeInjectedPlaneSimulation',
    'F6-readout': 'torchfdtd.mode_injection.modal_plane_amplitudes + the detector basis',
}


def relative_error(measured, recorded):
    return abs(measured - recorded) / abs(recorded) if recorded else abs(measured)


def main():
    case = native.load_run()
    cfg, run = case
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    frequency = [native.C0 / (cfg['wavelength_um'] * 1e-6)]
    tolerance = cfg['expected']['tolerance_relative']

    report = {
        'policy': cfg['policy'],
        'status': 'in_progress',
        'stages': list(STAGES),
        'stage_source': dict(STAGE_SOURCE),
        'fixture': 'G8/N8: eight periods, 10/5 nm mesh, five z cells, normal-harmonic bridge',
        'acceptance_granted': False,
        'cases': [],
    }
    (run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')

    failed = []
    for fixture_case in cfg['cases']:
        sc, section, p0, build = g5_fixture(cfg, fixture_case)
        r, project = sc['r'], sc['p']
        modes = [native.prepare_aperture_modal_launch(project, section, mode_index=i, num_modes=2)
                 for i in range(2)]
        scores = [float(np.sum(abs(m.mode.fields[..., 1])**2)
                        / np.sum(abs(m.mode.fields[..., [1, 2]])**2)) for m in modes]
        launch = modes[int(np.argmax(scores))]
        if max(scores) <= 1 - 1e-8 or launch.mode.maxwell_residual >= 1e-5:
            raise SystemExit('The launch eigenmode is not resolvable on this cross-section')

        model = native.ModeInjectedPlaneSimulation(project, launch, native.AdjointOptions(
            checkpoints=32, gpu_budget_bytes=16 * 1024**3, backward_kernel='fused',
            **fixture_case['checkpoint_options']))

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        started = time.perf_counter()
        parameters = torch.tensor(p0.detach().cpu().numpy(), dtype=torch.float64, device='cuda')
        with torch.no_grad():
            epsilon = build(parameters, 0.)
            planes = model(epsilon, frequency)
            amps = {key: native.modal_plane_amplitudes(planes[key], launch)
                    for key in ('near', 'far', 'outside')}
            near = planes['near']
            coordinate = near.points_um_geometry if near.points_um_geometry is not None \
                else near.points_um
            basis = torch.as_tensor(launch.detector_mode(float(coordinate[0, 0])).sample_plane(
                near.points_um.detach().cpu().numpy()), device='cuda')
            unit_flux = .5 * (basis[..., 1] * basis[..., 5].conj()
                              - basis[..., 2] * basis[..., 4].conj()).real @ near.weights
            incident = amps['near']['forward'].abs().square().reshape(-1)[0] * unit_flux
            upward = planes['top'].flux()[0]
            objective = upward / incident
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        epsilon_array = epsilon.detach().cpu().numpy()
        measured = {
            'J': float(objective.detach()),
            'P_up_reduced': float(upward.detach()),
            'P_in_reduced': float(incident.detach()),
            'unit_mode_flux_reduced': float(unit_flux.detach()),
            'epsilon_sha256': hashlib.sha256(epsilon_array.tobytes(order='C')).hexdigest(),
        }
        recorded = cfg['expected']['cases'][fixture_case['id']]
        errors = {key: relative_error(measured[key], recorded[key])
                  for key in ('J', 'P_up_reduced', 'P_in_reduced', 'unit_mode_flux_reduced')}
        errors['epsilon_sha256'] = 0.0 if measured['epsilon_sha256'] == recorded['epsilon_sha256'] else 1.0
        ok = all(value <= tolerance for value in errors.values())

        row = {
            'id': fixture_case['id'],
            'shape': list(r.shape),
            'steps': r.steps,
            'dt_s': r.time_step,
            'edge_count': int(parameters.numel()),
            'launch_identity': launch.identity,
            'launch_neff': launch.mode.neff,
            'elapsed_seconds': elapsed,
            'measured': measured,
            'recorded': recorded,
            'relative_error': errors,
            'within_tolerance': ok,
        }
        report['cases'].append(row)
        effective['resolved_cases'].append(dict(
            id=fixture_case['id'], shape=r.shape, dt_s=r.time_step, steps=r.steps,
            edge_count=int(parameters.numel()), launch_identity=launch.identity,
            launch_neff=launch.mode.neff, sampling=fixture_case['sampling'],
            mesh_um=[fixture_case['dx_um'], fixture_case['dy_um']], precision=cfg['precision']))
        (run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')
        Path(os.environ['EFFECTIVE_JSON']).write_text(json.dumps(effective, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not ok:
            failed.append(fixture_case['id'])

    report['status'] = 'mismatch' if failed else 'completed'
    report['mismatched_cases'] = failed
    report['simulation_executed'] = True
    (run / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'leg1 objective: {len(report["cases"]) - len(failed)}/{len(report["cases"])} cases within '
          f'{tolerance:g} relative', flush=True)
    return 52 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
