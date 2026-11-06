"""Measure leg 19's unchanged workers and gate their reserved peaks.

Run through tools/run.sh. Never call leg19_admission.main here: its OOM
handler writes the live fork. An OOM in any worker is an immediate stop.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

JOBS = (('old', 2), ('frozen', 2), ('new', 0), ('new', 1),
        ('new', 2), ('low-budget', 2), ('occupied', 2))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def byte_count(value):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid byte count: {value!r}')
    return value


def relation(row):
    """Same gate for admitted and refused rows; no allocated-peak substitute."""
    prediction = byte_count(row['prediction_bytes'])
    reserved = byte_count(row['gpu_peak_reserved_bytes'])
    allocated = byte_count(row['gpu_peak_allocated_bytes'])
    if reserved < allocated:
        raise ValueError('Reserved peak cannot be below allocated peak.')
    margin = prediction - reserved
    return dict(row, reserved_margin_bytes=margin,
                reserved_margin_fraction=None if reserved == 0 else margin / reserved,
                relation_holds=margin >= 0,
                relation_verdict='holds' if margin >= 0 else 'VIOLATED')


def gate_exit(rows):
    """57 is a finding on an admitted row; incomplete evidence is an error."""
    if not rows:
        raise ValueError('No measured rows.')
    evaluated = [relation(row) for row in rows]
    if any(not r['relation_holds'] and r['admitted'] for r in evaluated):
        return 57
    if not all(r['relation_holds'] for r in evaluated):
        raise ValueError('Relation violated only on refused rows; no admitted finding.')
    return 0


def fork_state(fork):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(fork), *args], text=True,
                                       env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).strip()
    return dict(head=git('rev-parse', 'HEAD'), status_porcelain=git('status', '--porcelain'))


def measured_worker(variant, index):
    """Observe CUDA queries without changing the original worker's decisions."""
    if variant == 'frozen':
        sys.path.insert(0, os.environ['SOLVER_FROZEN'])
        import torchfdtd  # Select the same independent baseline as leg 19.
    import parity
    import torch
    import torchfdtd
    from torchfdtd import adjoint_memory, cuda_memory
    import leg19_admission as legacy

    run = Path(os.environ['RUN_DIR_ABS'])
    telemetry = dict(reservation_calls=[], free_queries=[], solver_module=torchfdtd.__file__)
    original_reservation = adjoint_memory._resident_reservation
    original_info = cuda_memory.cuda_mem_info

    def observe_info(device=None):
        free, total = original_info(device)
        telemetry['free_queries'].append(dict(free_bytes=free, total_bytes=total,
            live_allocated_bytes=torch.cuda.memory_allocated(device)))
        return free, total

    def observe_reservation(*args, **kwargs):
        call = dict(live_allocated_bytes=torch.cuda.memory_allocated(args[2]))
        telemetry['reservation_calls'].append(call)
        try:
            result = original_reservation(*args, **kwargs)
        except ValueError as error:
            call['refusal'] = str(error)
            # Read the model's process-peak prediction even when admission
            # refuses it. This metadata-only second call creates no fields and
            # cannot change the already-made refusal. W3 may change the printed
            # required_bytes to incremental demand; that is not the gate metric.
            with patch.object(adjoint_memory, 'cuda_budget_limit', return_value=sys.maxsize):
                call['capacity_bypass_report'] = dict(original_reservation(*args, **kwargs))
            raise
        call['report'] = dict(result)
        return result

    with patch.object(adjoint_memory, '_resident_reservation', observe_reservation), \
            patch.object(cuda_memory, 'cuda_mem_info', observe_info):
        code = legacy.worker(variant, index)
    telemetry.update(worker_exit_status=code,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(run / f'relation-{variant}-{index}.json', telemetry)
    return code


def measured_relation(raw, telemetry, variant, index):
    admitted = not raw.get('refused', False)
    if admitted:
        prediction = raw['solver_report']['memory_reservation_bytes']
        for key in ('gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes'):
            if raw[key] != telemetry[key]:
                raise ValueError(f'Worker peak changed: {key}')
        scope = 'memory_reservation_bytes; includes modal source packet'
    else:
        match = re.search(r'\brequired_bytes=(\d+)', raw['refusal'])
        if match is None:
            raise ValueError('Refusal has no required_bytes; cannot infer a prediction.')
        prediction = telemetry['reservation_calls'][-1]['capacity_bypass_report']['memory_reservation_bytes']
        scope = 'memory_reservation_bytes in metadata-only capacity bypass after refusal; base frame, excludes modal source packet'
    return relation(dict(id=f'{variant}-{index}', case_id=raw['case_id'], admitted=admitted,
        prediction_bytes=prediction, prediction_scope=scope,
        gpu_peak_reserved_bytes=telemetry['gpu_peak_reserved_bytes'],
        gpu_peak_allocated_bytes=telemetry['gpu_peak_allocated_bytes']))


def double_count(rows, raw, telemetry):
    """Exact arithmetic on measured terms; no corrected model is installed."""
    new = raw['new-2']['solver_report']
    occupied = raw['occupied-2']
    calls = telemetry['occupied-2']['reservation_calls']
    live = byte_count(calls[-1]['live_allocated_bytes'])
    original_live = new['workspace_components_bytes']['live_inputs_and_external_allocations']
    subtotal = new['memory_reservation_bytes'] - new['allocation_headroom_bytes'] \
        - new['modal_source_extra_reservation_bytes']
    replaced_subtotal = subtotal - original_live + live
    reconstructed = replaced_subtotal + 8 * 1024**2 + (replaced_subtotal + 19) // 20
    occupied_row = next(r for r in rows if r['id'] == 'occupied-2')
    match = re.search(r'\blimit_bytes=(\d+)', occupied['refusal'])
    queries = telemetry['occupied-2']['free_queries']
    limit = int(match[1])
    return dict(new_base_subtotal_bytes=subtotal, new_live_component_bytes=original_live,
        occupied_live_at_admission_bytes=live, occupied_allocated_before_bytes=occupied['allocated_before_admission_bytes'],
        held_bytes=occupied['held_bytes'], substituted_subtotal_bytes=replaced_subtotal,
        reconstructed_required_bytes=reconstructed,
        observed_required_bytes=int(re.search(r'\brequired_bytes=(\d+)', occupied['refusal'])[1]),
        process_prediction_bytes=occupied_row['prediction_bytes'],
        reconstruction_matches=reconstructed == int(re.search(r'\brequired_bytes=(\d+)', occupied['refusal'])[1]),
        observed_limit_bytes=limit, free_queries=queries,
        limit_matches_eighty_percent_free=any(int(q['free_bytes'] * .8) == limit for q in queries),
        reference_new_prediction_bytes=new['memory_reservation_bytes'],
        reference_new_reserved_peak_bytes=raw['new-2']['gpu_peak_reserved_bytes'],
        explanation='The live term is included in required with headroom, while the same live '
                    'allocations have already reduced free bytes used for limit=0.8*free. '
                    'This is a diagnostic of forward-v1, not a change to reservation semantics.')


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        return measured_worker(sys.argv[2], int(sys.argv[3]))
    root = Path.cwd()
    for line in (root / 'env/refs.env').read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key, value)
    cfg = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())
    run = Path(os.environ['RUN_DIR_ABS'])
    snapshot = Path(os.environ['PARITY_SNAPSHOT'])
    worker_config = snapshot / cfg['worker_config']
    fork = root / 'src/torchfdtd'
    before = fork_state(fork)
    report = dict(policy=cfg['policy'], run_id=run.name, verdict='partial',
        status='in_progress', simulation_executed=False, rows=[], worker_exit_statuses=[],
        fork_before=before, measurement_basis='Fresh unchanged leg-19 workers; historical runs are not edited.',
        worker_config_sha256=hashlib.sha256(worker_config.read_bytes()).hexdigest())
    raw, telemetry = {}, {}
    started = time.perf_counter()

    def finish(code, status):
        report.update(status=status, exit_status=code, fork_after=fork_state(fork),
                      wall_seconds=time.perf_counter()-started)
        report['fork_unchanged'] = before == report['fork_after'] and not before['status_porcelain']
        if not report['fork_unchanged']:
            code = 58
            report.update(status='STOP-fork-changed', exit_status=code, verdict='rejected')
        dump(run / 'relation.json', report)
        from leg19_admission import readout_envelope
        dump(run / 'readout.json', readout_envelope(report) | dict(detail='relation.json'))
        effective_path = Path(os.environ['EFFECTIVE_JSON'])
        effective = json.loads(effective_path.read_text())
        effective.update(stage='completed', simulation_executed=report['simulation_executed'],
                         verdict=report['verdict'], resolved_cases=report['rows'])
        dump(effective_path, effective)
        print(json.dumps(dict(exit_status=code, status=report['status'],
                              fork_unchanged=report['fork_unchanged'])), flush=True)
        return code

    if before['status_porcelain']:
        return finish(58, 'STOP-dirty-fork')
    for variant, index in JOBS:
        if time.perf_counter() - started > cfg['cost_valve_seconds']:
            return finish(56, 'stopped-cost-valve')
        proc = subprocess.run([sys.executable, '-B', __file__, '--worker', variant, str(index)],
            env=dict(os.environ, PARITY_CONFIG=str(worker_config)))
        report['worker_exit_statuses'].append(dict(variant=variant, index=index, exit_status=proc.returncode))
        if proc.returncode:
            # In particular, do not restore the fork and do not retry an OOM.
            return finish(proc.returncode, 'STOP-worker-OOM' if proc.returncode == 59 else 'stopped-worker-failure')
        key = f'{variant}-{index}'
        raw[key] = json.loads((run / f'{key}.json').read_text())
        telemetry[key] = json.loads((run / f'relation-{key}.json').read_text())
        row = measured_relation(raw[key], telemetry[key], variant, index)
        report['rows'].append(row)
        report['simulation_executed'] |= row['admitted']
        print(json.dumps(row), flush=True)
        dump(run / 'relation.json', report)
    report['double_count'] = double_count(report['rows'], raw, telemetry)
    print(json.dumps(dict(double_count=report['double_count'])), flush=True)
    try:
        code = gate_exit(report['rows'])
    except ValueError as error:
        report['evidence_error'] = str(error)
        return finish(58, 'stopped-invalid-evidence')
    report['verdict'] = 'accepted' if code == 0 else 'rejected'
    return finish(code, 'relation-holds-every-row' if code == 0 else 'finding-admitted-underprediction')


if __name__ == '__main__':
    sys.exit(main())
