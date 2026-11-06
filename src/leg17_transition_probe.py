"""Leg 17: the preregistered transition-resolution lattice, no fitted quantities.

The scene and CUDA geometry derivative instrument are imported without changes.
All execution, including --checks, requires tools/run.sh and its source snapshot.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MATCHED = {'0.5': ('a', 'd'), '1.0': ('f', 'b', 'g'), '2.0': ('e', 'c')}
LATTICE = {'a': (.04, .02, 20), 'b': (.02, .01, 20), 'c': (.01, .005, 20),
           'd': (.02, .01, 10), 'e': (.02, .01, 40), 'f': (.04, .02, 40),
           'g': (.01, .005, 10)}


def validate_config(cfg, baseline):
    for key in ('core_index', 'cladding_index', 'wavelength_um', 'width_um',
                'precision', 'expected'):
        if cfg[key] != baseline[key]:
            raise ValueError(f'Fixture or control drift: {key}')
    if cfg['cost_valve_seconds'] != 1800:
        raise ValueError('The registered cost ceiling is 1800 seconds')
    if [c['id'] for c in cfg['cases']] != list('bdefg'):
        raise ValueError('Exactly b control then d/e/f/g are authorized')
    if [c['id'] for c in cfg['anchors']] != list('abc'):
        raise ValueError('Exactly the three leg-16 anchors are required')
    for case in cfg['cases'] + cfg['anchors']:
        if tuple(case[k] for k in ('dx_um', 'dy_um', 'transition_nm')) != LATTICE[case['id']]:
            raise ValueError('Unregistered lattice point')
        for key in ('z_cells', 'periods', 'duration_fs', 'shift_nm'):
            if case[key] != baseline['cases'][0][key]:
                raise ValueError(f'Fixed case input drift: {key}')


def differences(ids, rows, observable):
    """Available adjacent differences only; never bridge a missing lattice point."""
    result = []
    for left, right in zip(ids, ids[1:]):
        if left not in rows or right not in rows:
            result.append(dict(left=left, right=right, signed=None, absolute=None))
            continue
        x, y = rows[left]['measured'][observable], rows[right]['measured'][observable]
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Non-finite measured observable')
        result.append(dict(left=left, right=right, signed=y-x, absolute=abs(y-x)))
    return dict(increments=result, maximum_absolute=max(r['absolute'] for r in result)
                if all(r['absolute'] is not None for r in result) else None)


def classify(rows, control_passed):
    result = dict(label='unevaluable', evaluable=False, observables={})
    for key in ('T', 'n_eff'):
        result['observables'][key] = dict(
            smoothing=differences(('d', 'b', 'e'), rows, key),
            mesh={s: differences(ids, rows, key) for s, ids in MATCHED.items()},
            leg16_reference=None)
    if not control_passed or set(rows) != set(LATTICE):
        return result
    above, below = [], []
    for value in result['observables'].values():
        delta = value['smoothing']['maximum_absolute']
        for mesh in value['mesh'].values():
            above.append(delta > mesh['maximum_absolute'])
            below.append(3 * delta < mesh['maximum_absolute'])
    result.update(evaluable=True, label='explained' if all(above) else
                  'not-the-driver' if all(below) else 'inconclusive')
    return result


def control_result(measured, expected):
    from leg1_forward import relative_error
    recorded = next(iter(expected['cases'].values()))
    errors = {k: relative_error(measured[k], recorded[k]) for k in ('T', 'R')}
    exact_hash = measured['epsilon_sha256'] == recorded['epsilon_sha256']
    return dict(relative_errors=errors, epsilon_sha256_exact=exact_hash,
                passed=exact_hash and all(math.isfinite(v) and v <= expected['tolerance_relative']
                                          for v in errors.values()))


def raster_counts(epsilon, sc, cfg, case):
    """Count the actual material samples, not an analytic second rasterizer.

    The windows isolate each narrow tooth using the copied scene's tooth left
    edges. All values come from epsilon at the nearest core-interior y/z plane.
    Counts are conventions and do not replace the smooth material coefficients.
    """
    import numpy as np
    from torchfdtd.solver import field_axes
    result = {}
    for component, name in enumerate(('Ex', 'Ey', 'Ez')):
        x, y, z = field_axes(sc['r'], name)
        yi, zi = int(np.argmin(abs(y - .12))), int(np.argmin(abs(z)))
        values = epsilon[:, yi, zi, component].detach().cpu().numpy()
        counts = []
        for tooth in range(sc['N']):
            left = sc['shift'] + tooth * .9 + .26325 - sc['xc']
            right = left + cfg['width_um']
            mask = (x >= left - sc['transition']) & (x <= right + sc['transition'])
            occupancy = (values[mask] - sc['eb']) / (sc['ec'] - sc['eb'])
            counts.append(dict(tooth=tooth, occupied_ge_half=int((occupancy >= .5).sum()),
                               nonzero_support=int((occupancy > 0).sum()),
                               weighted_cells=float(occupancy.sum())))
        result[name] = dict(y_um=float(y[yi]), z_um=float(z[zi]), teeth=counts)
    return dict(basis='Measured epsilon slice; core interior nearest y=0.12 um, z=0; '
                'tooth windows from unchanged copied scene positions', components=result)


def standard_readout(report):
    from leg16_tooth_convergence import standard_readout as envelope
    from leg1_forward import STAGES, STAGE_SOURCE
    result = envelope(report, STAGES, STAGE_SOURCE)
    result.update(fixture='N8 guided tooth transition probe; diagnostics in transition-probe.json',
                  verdict=report['verdict'])
    return result


def run_checks(cfg, run):
    """Run the real workspace suite through this launcher's recorded process."""
    import unittest
    # Existing archive-sensitive checks deliberately refer to the workspace.
    suite = unittest.defaultTestLoader.discover(str(Path.cwd() / 'checks'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    summary = dict(tests_run=result.testsRun, skipped=len(result.skipped),
                   failures=len(result.failures), errors=len(result.errors),
                   successful=result.wasSuccessful(), simulation_executed=False,
                   verdict='accepted' if result.wasSuccessful() else 'rejected')
    (run / 'checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    (run / 'verdict').write_text(summary['verdict'] + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if result.wasSuccessful() else 1


def main():
    if 'RUN_DIR_ABS' not in os.environ:
        raise RuntimeError('Use tools/run.sh, including for --checks')
    from parity import load_run, C0
    cfg, run = load_run()
    baseline = json.loads((ROOT / 'configs/leg16-tooth-convergence.json').read_text())
    validate_config(cfg, baseline)
    if sys.argv[1:] == ['--checks']:
        return run_checks(cfg, run)
    if sys.argv[1:]:
        raise ValueError('Only --checks or the registered campaign is supported')

    import torch
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene, geometry_only
    from leg1_forward import launch_for
    from leg16_tooth_convergence import refusal_details

    effective_path = Path(os.environ['EFFECTIVE_JSON'])
    effective = json.loads(effective_path.read_text())
    report = dict(policy=cfg['policy'], status='in_progress', verdict='partial',
                  run_id=run.name, cases=[], anchors=cfg['anchors'], geometry_only=[],
                  control=None, simulation_executed=False,
                  registration=cfg['registration'], hardware=torch.cuda.get_device_name(),
                  geometry_instrument='run_b9_native_shape_n8.geometry_only (unchanged)',
                  geometry_instrument_sha256=hashlib.sha256(
                      (ROOT / 'src/run_b9_native_shape_n8.py').read_bytes()).hexdigest())
    campaign_started = time.perf_counter()

    def save():
        # The mandatory new b control supplies the current lattice value. The old
        # b stays archived in anchors and in the leg-16 reference increments.
        rows = {r['id']: r for r in cfg['anchors'] if r['id'] != 'b'}
        rows.update({r['id']: r for r in report['cases']})
        report['C1'] = classify(rows, bool(report['control'] and report['control']['passed']))
        for key, data in report['C1']['observables'].items():
            data['leg16_reference'] = differences(tuple('abc'),
                {r['id']: r for r in cfg['anchors']}, key)
        report['campaign_wall_seconds'] = time.perf_counter() - campaign_started
        (run / 'transition-probe.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (run / 'readout.json').write_text(json.dumps(standard_readout(report), indent=2, allow_nan=False) + '\n')
        (run / 'verdict').write_text(report['verdict'] + '\n')
        effective_path.write_text(json.dumps(effective, indent=2, allow_nan=False) + '\n')

    def geometry(case):
        diagnostic = {}
        previous = list(report['geometry_only'])
        one = dict(cfg, geometry_cases=[dict(case, transition_nm=[case['transition_nm']],
                                            shift_nm=[case['shift_nm']])])
        def sink():
            report['geometry_only'] = previous + [dict(r, lattice_case=case['id'])
                                                   for r in diagnostic.get('geometry_only', [])]
            save()
        geometry_only(one, diagnostic, sink)

    def cost_exceeded():
        return time.perf_counter() - campaign_started >= cfg['cost_valve_seconds']

    save()
    try:
        for case in cfg['cases']:
            report['active_case'] = case['id']
            if cost_exceeded():
                report.update(status='stopped-cost-valve', verdict='partial')
                save()
                return 56
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            sc = scene(cfg, case)
            region, project = sc['r'], sc['p']
            if list(region.size) != [13.48, 6., .2]:
                report.update(status='stopped-fixture-change', verdict='rejected',
                              actual_domain_um=list(region.size))
                save()
                return 57
            row = dict(id=case['id'], dx_um=case['dx_um'], dy_um=case['dy_um'],
                       transition_nm=case['transition_nm'], s=case['transition_nm']/(1000*case['dx_um']),
                       shape=list(region.shape), cells=math.prod(region.shape),
                       dt_s=region.time_step, steps=region.steps, domain_um=list(region.size),
                       provenance='measured in leg 17 run ' + run.name)
            report['pending_case'] = row
            effective['resolved_cases'].append(dict(row))
            save()
            launch, scores = launch_for(project, sc['section'])
            width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
            with torch.no_grad():
                epsilon = sc['epsilon'](width)
                model = ModeInjectedPlaneSimulation(project, launch)
                planes = model(epsilon, [C0/(cfg['wavelength_um']*1e-6)])
                amps = {k: modal_plane_amplitudes(planes[k], launch) for k in ('near','far','outside')}
                measured = dict(
                    T=float((amps['far']['forward']/amps['near']['forward']).abs().square().sum()),
                    R=float((amps['near']['backward']/amps['near']['forward']).abs().square().sum()),
                    n_eff=float(launch.mode.beta_per_um*cfg['wavelength_um']/(2*math.pi)),
                    epsilon_sha256=hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest())
            torch.cuda.synchronize()
            row.update(measured=measured, launch_identity=launch.identity, launch_scores=scores,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       wall_seconds=time.perf_counter()-started,
                       solver_report=dict(planes['near'].report))
            report['cases'].append(row)
            report['simulation_executed'] = effective['simulation_executed'] = True
            report.pop('pending_case', None)
            if case['id'] == 'b':
                report['control'] = control_result(measured, cfg['expected'])
                if not report['control']['passed']:
                    report.update(status='stopped-control-mismatch', verdict='rejected')
                    save()
                    return 52
            row['raster'] = raster_counts(epsilon, sc, cfg, case)
            print(json.dumps(row), flush=True)
            del amps, planes, model, epsilon, width, launch, sc, project, region
            torch.cuda.empty_cache()
            if cost_exceeded():
                report.update(status='stopped-cost-valve', verdict='partial')
                save()
                return 56
            geometry(case)
            save()
            if cost_exceeded():
                report.update(status='stopped-cost-valve', verdict='partial')
                save()
                return 56
    except (ValueError, RuntimeError, MemoryError) as error:
        report.update(status='stopped-refusal', verdict='partial', refusal=refusal_details(error))
        save()
        print(json.dumps(report['refusal']), flush=True)
        return 55
    report.pop('active_case', None)
    report.update(status='completed', verdict='accepted')
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
