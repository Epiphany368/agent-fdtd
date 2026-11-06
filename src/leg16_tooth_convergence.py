"""Leg 16: forward-only N8 tooth ladder and the unchanged copied geometry instrument.

Run with tools/run.sh. Imports of the numerical stack are delayed so that the
ladder arithmetic and sourced configuration can be tested without CUDA.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def ladder_statistics(values, complete=False):
    """Measured consecutive orders; never extrapolate a reversing/worsening ladder."""
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Observable values must be finite')
    signed = [b - a for a, b in zip(values, values[1:])]
    differences = [abs(x) for x in signed]
    reversal = any(a * b < 0 for a, b in zip(signed, signed[1:]))
    worsening = any(b > a for a, b in zip(differences, differences[1:]))
    stopped = reversal or worsening
    orders = [] if stopped else [math.log2(a / b) if a > 0 and b > 0 else None
                                for a, b in zip(differences, differences[1:])]
    p = orders[-1] if orders else None
    extrapolated = None
    if p is not None and p > 0:
        extrapolated = values[-1] + signed[-1] / math.expm1(p * math.log(2))
    return dict(values=list(values), differences=differences, non_monotone=stopped,
                reversal=reversal, worsening=worsening, observed_orders=orders,
                finest_order=p, richardson_extrapolated=extrapolated,
                # Steward amendment 3cea489: evaluate C1 on completed rungs.
                # Two differences (three values) are the minimum evidence.
                C1_pass=(not stopped and differences[-1] <= .5 * differences[0])
                if len(values) >= 3 else None,
                all_four_rungs=complete and len(values) == 4)


def validate_config(cfg, baseline):
    """Reject accidental fixture drift before constructing a scene."""
    for key in ('core_index', 'cladding_index', 'wavelength_um', 'width_um', 'precision'):
        if cfg[key] != baseline[key]:
            raise ValueError(f'Fixture input changed: {key}')
    if cfg['expected'] != baseline['expected']:
        raise ValueError('Recorded control changed')
    reference = baseline['cases'][0]
    ladder = [(.04, .02), (.02, .01), (.01, .005), (.005, .0025)]
    if len(cfg['cases']) != 4:
        raise ValueError('Exactly four rungs are registered')
    for case, mesh in zip(cfg['cases'], ladder):
        if (case['dx_um'], case['dy_um']) != mesh:
            raise ValueError('Mesh ladder differs from the registration')
        for key in ('z_cells', 'periods', 'duration_fs', 'shift_nm', 'transition_nm'):
            if case[key] != reference[key]:
                raise ValueError(f'Fixture input changed: {key}')


def refusal_details(error):
    """Preserve numeric requirements from the actual refusing solver frame."""
    details = dict(type=type(error).__name__, message=str(error), requirements=[])
    trace = error.__traceback__
    keys = ('required', 'limit', 'device_required', 'host_required', 'extra',
            'host_checkpoint', 'disk_checkpoint', 'source_budget_bytes', 'layers')
    while trace:
        local = trace.tb_frame.f_locals
        numbers = {key: local[key] for key in keys
                   if key in local and isinstance(local[key], (int, float))}
        if numbers:
            details['requirements'].append(dict(function=trace.tb_frame.f_code.co_name,
                                                 values=numbers))
        trace = trace.tb_next
    return details


def standard_readout(report, stages, sources):
    """Common F1-F6 envelope; full leg diagnostics stay in their own artifact.

    Use the existing launch_neff field for the same effective-index observable.
    This keeps tools/report.py and the shared coverage inventory usable without
    changing their shared schema or assigning verification to new diagnostics.
    """
    cases = []
    for row in report['cases']:
        measured = {key: row['measured'][key] for key in ('T', 'R', 'epsilon_sha256')}
        measured['launch_neff'] = row['measured']['n_eff']
        cases.append(dict(id=row['id'], measured=measured,
                          **{key: row[key] for key in ('shape', 'steps', 'dt_s', 'wall_seconds',
                                                       'launch_identity', 'launch_scores')}))
    return dict(policy=report['policy'], status=report['status'], stages=list(stages),
                stage_source=dict(sources), cases=cases,
                fixture='Unchanged N8 tooth; full diagnostics in tooth-convergence.json')


def main():
    from parity import load_run, C0
    import torch
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene, geometry_only
    from leg1_forward import launch_for, relative_error, STAGES, STAGE_SOURCE

    cfg, run = load_run()
    baseline = json.loads((Path(__file__).resolve().parents[1] /
                           'configs/leg1-n8-forward.json').read_text())
    validate_config(cfg, baseline)
    effective_path = Path(os.environ['EFFECTIVE_JSON'])
    effective = json.loads(effective_path.read_text())
    if len(sys.argv) == 3 and sys.argv[1] == '--repair-report':
        # Reporting only, through run.sh. Preserve the original diagnostic JSON
        # bytes and every measurement; do not alter its manifest or exit status.
        source = run.parent / sys.argv[2]
        if source.parent != run.parent or source == run:
            raise ValueError('Repair source must be an earlier run in this workspace')
        diagnostic_path = source / 'tooth-convergence.json'
        original_path = source / 'readout.json'
        original = diagnostic_path.read_text() if diagnostic_path.exists() else original_path.read_text()
        archived = json.loads(original)
        if archived.get('policy') != 'leg16-tooth-convergence-v1':
            raise ValueError('Only this leg diagnostic envelope may be repaired')
        diagnostic_path.write_text(original)
        original_path.write_text(json.dumps(standard_readout(archived, STAGES, STAGE_SOURCE),
                                             indent=2, allow_nan=False) + '\n')
        (run / 'readout.json').write_text(json.dumps(dict(
            policy='leg16-report-repair-v1', status='completed-reporting-only',
            stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=[],
            fixture='Reporting repair only; no simulation or new observations',
            source_run=source.name), indent=2) + '\n')
        return 0
    report = dict(policy=cfg['policy'], status='in_progress', verdict=None, cases=[],
                  geometry_only=[], simulation_executed=False,
                  scope='In-plane tooth resolution only; z mesh fixed; no device claim',
                  geometry_instrument='run_b9_native_shape_n8.geometry_only (unchanged)',
                  forward_instrument='leg1_forward.launch_for and identical modal readout',
                  run_id=run.name)

    def save():
        (run / 'tooth-convergence.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (run / 'readout.json').write_text(json.dumps(standard_readout(report, STAGES, STAGE_SOURCE),
                                                    indent=2, allow_nan=False) + '\n')
        effective_path.write_text(json.dumps(effective, indent=2, allow_nan=False) + '\n')

    def geometry(case):
        # The imported function, including its CUDA autograd and normalization,
        # is executed unchanged. Only its one-rung input and report sink differ.
        one = dict(cfg, geometry_cases=[dict(case, transition_nm=[case['transition_nm']],
                                            shift_nm=[case['shift_nm']])])
        previous = list(report['geometry_only'])
        diagnostic = {}

        def save_geometry():
            report['geometry_only'] = previous + diagnostic.get('geometry_only', [])
            save()
        geometry_only(one, diagnostic, save_geometry)

    save()
    campaign_started = time.perf_counter()
    physical_domain = None
    try:
        for case in cfg['cases']:
            if time.perf_counter() - campaign_started >= cfg['cost_valve_seconds']:
                report.update(status='stopped-cost-valve', verdict='partial',
                              stopped_rung=case['id'])
                save()
                return 56
            started = time.perf_counter()
            report['active_rung'] = case['id']
            report['required_boundary_layers'] = dict(x=round(.64 / case['dx_um']),
                                                       y=round(.4 / case['dy_um']))
            save()
            if case['dx_um'] == .005:
                # Steward amendment a2f5ab2: no fourth-rung re-attempt. Keep the
                # native profile and record its known requirement before scene()
                # or any fourth-rung CUDA work is invoked.
                report.update(status='stopped-known-boundary-cap', verdict='partial',
                              stopped_rung=case['id'], refusal=dict(
                                  type='KnownNativeBoundaryValidationRefusal',
                                  message='Native CPML supports at most 100 layers',
                                  attempted=False, native_layer_cap=100,
                                  required_layers=dict(report['required_boundary_layers'])))
                report.pop('active_rung', None)
                save()
                return 55
            if cfg['mode'] == 'geometry':
                geometry(case)
                continue
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sc = scene(cfg, case)
            region, project = sc['r'], sc['p']
            domain = list(region.size)
            if physical_domain is None:
                physical_domain = domain
            if domain != physical_domain:
                report.update(status='stopped-fixture-change', verdict='stopped',
                              domain_um=domain, expected_domain_um=physical_domain)
                save()
                return 57
            row = dict(id=case['id'], dx_um=case['dx_um'], dy_um=case['dy_um'],
                       shape=list(region.shape), cells=math.prod(region.shape),
                       dt_s=region.time_step, steps=region.steps, domain_um=domain,
                       tooth_width_cells=cfg['width_um'] / case['dx_um'])
            report['pending_rung'] = row
            effective['resolved_cases'].append(dict(row))
            save()
            launch, scores = launch_for(project, sc['section'])
            width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
            with torch.no_grad():
                epsilon = sc['epsilon'](width)
                # Amendment a2f5ab2: forward construction uses library defaults;
                # do not reserve the leg-1 adjoint's 32 checkpoint slots.
                model = ModeInjectedPlaneSimulation(project, launch)
                planes = model(epsilon, [C0 / (cfg['wavelength_um'] * 1e-6)])
                amps = {key: modal_plane_amplitudes(planes[key], launch)
                        for key in ('near', 'far', 'outside')}
                measured = dict(
                    T=float((amps['far']['forward'] / amps['near']['forward']).abs().square().sum()),
                    R=float((amps['near']['backward'] / amps['near']['forward']).abs().square().sum()),
                    n_eff=float(launch.mode.beta_per_um * cfg['wavelength_um'] / (2 * math.pi)),
                    epsilon_sha256=hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest())
            torch.cuda.synchronize()
            row.update(measured=measured, launch_identity=launch.identity, launch_scores=scores,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       wall_seconds=time.perf_counter() - started,
                       solver_report=dict(planes['near'].report))
            report['cases'].append(row)
            report['simulation_executed'] = effective['simulation_executed'] = True
            report.pop('pending_rung', None)
            if case['dx_um'] == .02:
                recorded = next(iter(cfg['expected']['cases'].values()))
                errors = {key: relative_error(measured[key], recorded[key]) for key in ('T', 'R')}
                errors['epsilon_sha256'] = int(measured['epsilon_sha256'] != recorded['epsilon_sha256'])
                report['control'] = dict(relative_errors=errors,
                    passed=all(x <= cfg['expected']['tolerance_relative'] for x in errors.values()))
                if not report['control']['passed']:
                    report.update(status='stopped-control-mismatch', verdict='failed')
                    save()
                    return 52
            print(json.dumps(row), flush=True)
            del amps, planes, model, epsilon, width, launch, sc, project, region
            torch.cuda.empty_cache()
            geometry(case)
            report['observables'] = {key: ladder_statistics(
                [r['measured'][key] for r in report['cases']], len(report['cases']) == 4)
                for key in ('T', 'n_eff')}
            save()
            if any(x['non_monotone'] for x in report['observables'].values()):
                # A stop in either observable forbids fitting either ladder.
                for stats in report['observables'].values():
                    stats.update(observed_orders=[], finest_order=None, richardson_extrapolated=None)
                report.update(status='stopped-non-monotone', verdict='failed')
                save()
                return 53
    except (ValueError, RuntimeError, MemoryError) as error:
        report.update(status='stopped-refusal', verdict='partial', refusal=refusal_details(error))
        save()
        print(json.dumps(report['refusal']), flush=True)
        return 55

    if cfg['mode'] == 'forward':
        deviations = [max(abs(x - 1) for x in r['normalized_edge_weight'].values())
                      for r in report['geometry_only']]
        report['criteria'] = dict(
            C1=all(x['C1_pass'] for x in report['observables'].values()),
            C2=report['control']['passed'],
            C3=deviations[0] > max(deviations[1:]),
            C4=len(report['cases']) == 4)
        report['geometry_max_deviations'] = deviations
        passed = all(report['criteria'].values())
        report.update(status='completed' if passed else 'criteria-failed',
                      verdict='passed' if passed else 'failed')
    else:
        passed = True
        report.update(status='completed', verdict='diagnostic-only')
    report.pop('active_rung', None)
    save()
    return 0 if passed else 54


if __name__ == '__main__':
    sys.exit(main())
