"""Leg 19: isolated, measured forward admission and conservative safety controls.

Run only through tools/run.sh. The launcher owns all generated artifacts.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative_error(value, truth):
    return abs(value-truth)/abs(truth) if truth else abs(value-truth)


def verdict(report):
    """Amendment 1: unknown historical ratios are context, never a C1 gate."""
    if not all(report['criteria'].values()):
        return 'rejected'
    return 'partial' if report.get('historical_full_device',{}).get('memory_ratio_after') is None else 'accepted'


def readout_envelope(report):
    """Common reporting contract; scientific details remain in admission.json."""
    from leg1_forward import STAGES, STAGE_SOURCE
    return dict(policy=report['policy'],status=report['status'],verdict=report['verdict'],cases=[],
                stages=list(STAGES),stage_source=dict(STAGE_SOURCE),
                fixture='Unchanged N8 forward admission measurement; details in admission.json',
                detail='admission.json',simulation_executed=report['simulation_executed'])


def repair_envelopes(run, policy):
    """Launcher-owned schema repair; preserve original bytes and all outcomes."""
    repaired=[]
    for prior in sorted(run.parent.glob('*/readout.json')):
        if prior.parent==run:
            continue
        original=prior.read_bytes()
        data=json.loads(original)
        if data.get('policy')!=policy or 'stages' in data:
            continue
        backup=prior.with_name('readout.before-envelope-repair.json')
        if backup.exists():
            raise ValueError('Refusing to overwrite an earlier envelope backup.')
        backup.write_bytes(original)
        dump(prior,readout_envelope(data))
        repaired.append(dict(run_id=prior.parent.name,
            original_sha256=hashlib.sha256(original).hexdigest(),
            repaired_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
            preserved='readout.before-envelope-repair.json; admission.json and manifest unchanged'))
    return repaired


def worker(variant, case_index):
    # Import the independently frozen solver before the snapshot adapter only
    # for the frozen control. Never import either read-only reference project.
    if variant == 'frozen':
        sys.path.insert(0, os.environ['SOLVER_FROZEN'])
        import torchfdtd
    from parity import C0
    import numpy as np
    import torch
    from torchfdtd import AdjointOptions
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene
    from leg1_forward import launch_for

    cfg = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())
    run = Path(os.environ['RUN_DIR_ABS'])
    case = cfg['cases'][case_index]
    os.environ.pop('TORCHFDTD_ADMISSION_MODEL', None)
    if variant in ('new', 'low-budget', 'occupied'):
        os.environ['TORCHFDTD_ADMISSION_MODEL'] = 'forward-v1'
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sc = scene(cfg, case)
    launch, scores = launch_for(sc['p'], sc['section'])
    with torch.no_grad():
        width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
        epsilon = sc['epsilon'](width)
        torch.cuda.synchronize()
        geometry_peak = torch.cuda.max_memory_allocated()
        model = ModeInjectedPlaneSimulation(sc['p'], launch)
        frequency = [C0/(cfg['wavelength_um']*1e-6)]
        holder = None
        if variant == 'low-budget':
            # Use half this case's previously measured prediction.
            previous = json.loads((run/f'new-{case_index}.json').read_text())
            model.model.options = AdjointOptions(gpu_budget_bytes=previous['solver_report']['memory_reservation_bytes']//2)
        if variant == 'occupied':
            from torchfdtd.cuda_memory import cuda_mem_info
            previous = json.loads((run/f'new-{case_index}.json').read_text())
            torch.cuda.empty_cache()
            free, _ = cuda_mem_info('cuda')
            # Leave less free than the observed forward peak. This allocation
            # is resource pressure, not a new physical fixture or OOM trial.
            hold_bytes = max(0, free-previous['gpu_peak_allocated_bytes']//2)
            holder = torch.empty(hold_bytes, dtype=torch.uint8, device='cuda')
        before = torch.cuda.memory_allocated()
        try:
            planes = model(epsilon, frequency)
        except torch.cuda.OutOfMemoryError as error:
            dump(run/f'{variant}-{case_index}.json',dict(oom=True,message=str(error)))
            return 59
        except ValueError as error:
            if variant not in ('low-budget', 'occupied'):
                raise
            row = dict(variant=variant, case_id=case['id'], refused=True,
                       refusal=str(error), allocated_before_admission_bytes=before,
                       allocated_after_refusal_bytes=torch.cuda.memory_allocated(),
                       held_bytes=0 if holder is None else holder.numel())
            row['allocation_increase_bytes'] = row['allocated_after_refusal_bytes']-before
            dump(run/f'{variant}-{case_index}.json', row)
            return 0 if 'required_bytes=' in str(error) and 'unused_slot_reservation_bytes=0' in str(error) else 57
        if variant in ('low-budget', 'occupied'):
            dump(run/f'{variant}-{case_index}.json', dict(refused=False))
            return 58
        amps = {key: modal_plane_amplitudes(planes[key], launch) for key in ('near', 'far', 'outside')}
        measured = dict(T=float((amps['far']['forward']/amps['near']['forward']).abs().square().sum()),
                        R=float((amps['near']['backward']/amps['near']['forward']).abs().square().sum()),
                        n_eff=float(launch.mode.beta_per_um*cfg['wavelength_um']/(2*np.pi)),
                        epsilon_sha256=hashlib.sha256(epsilon.cpu().numpy().tobytes()).hexdigest())
        arrays = {key: plane.fields.cpu().numpy() for key, plane in planes.items()}
        np.savez(run/f'{variant}-{case_index}-planes.npz', **arrays)
        torch.cuda.synchronize()
        cells = int(np.prod(sc['r'].shape))
        row = dict(variant=variant, case_id=case['id'], shape=list(sc['r'].shape),
                   cells=cells, steps=sc['r'].steps, dt_s=sc['r'].time_step,
                   measured=measured, launch_identity=launch.identity, launch_scores=scores,
                   plane_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                   solver_report=dict(planes['near'].report),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), geometry_peak_bytes=geometry_peak,
                   allocated_before_admission_bytes=before, wall_seconds=time.perf_counter()-started,
                   ns_per_cell_step=planes['near'].report['forward_seconds']*1e9/(cells*sc['r'].steps))
        dump(run/f'{variant}-{case_index}.json', row)
    print(json.dumps({k: row[k] for k in ('variant','case_id','gpu_peak_allocated_bytes','ns_per_cell_step')}), flush=True)
    return 0


def inventory(fork, frozen):
    files = subprocess.check_output(['git', '-C', str(frozen), 'ls-files', '-z']).decode().split('\0')
    files = [p for p in files if p]
    differences = [p for p in files if not (fork/p).is_file() or (fork/p).read_bytes() != (frozen/p).read_bytes()]
    return dict(tracked_files=len(files), identical_files=len(files)-len(differences), differences=differences,
                note='models.py is the pre-existing, separately committed leg-20 opt-in change. Numerical defaults are checked by the frozen worker.')


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        return worker(sys.argv[2], int(sys.argv[3]))
    # Load machine roots as symbols. Never embed their values in project files.
    root = Path.cwd()
    for line in (root/'env/refs.env').read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key, value)
    cfg = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())
    run = Path(os.environ['RUN_DIR_ABS'])
    report = dict(policy=cfg['policy'], verdict='partial', status='in_progress', run_id=run.name,
                  criteria={}, rows=[], worker_exit_statuses=[], simulation_executed=False,
                  historical_full_device=cfg['historical_full_device'])
    save = lambda: dump(run/'admission.json', report)
    save()
    started = time.perf_counter()
    jobs = [('old', 2), ('frozen', 2), ('new', 0), ('new', 1), ('new', 2), ('low-budget', 2), ('occupied', 2)]
    for variant, index in jobs:
        if time.perf_counter()-started > cfg['cost_valve_seconds']:
            report.update(status='stopped-cost-valve')
            save()
            return 56
        proc = subprocess.run([sys.executable, '-B', __file__, '--worker', variant, str(index)])
        report['worker_exit_statuses'].append(dict(variant=variant, index=index, exit_status=proc.returncode))
        if proc.returncode:
            # Stop and report every failure, including OOM (59). The solver
            # fork is shared: a run must never restore source or retry an OOM.
            report.update(status='stopped-worker-failure')
            save()
            return proc.returncode
        report['rows'].append(json.loads((run/f'{variant}-{index}.json').read_text()))
        report['simulation_executed'] = True
        save()
        if variant == 'frozen':
            default, frozen = report['rows'][:2]
            if default['measured'] != frozen['measured'] or default['plane_sha256'] != frozen['plane_sha256']:
                report.update(status='stopped-default-path-difference',verdict='rejected')
                save()
                return 52
    old, frozen = report['rows'][:2]
    new = report['rows'][4]
    truth = cfg['leg16_truth']['peak_allocated_bytes']
    prediction = new['solver_report']['memory_reservation_bytes']
    report['C1'] = dict(prediction_bytes=prediction, historical_truth_bytes=truth,
                        new_measured_bytes=new['gpu_peak_allocated_bytes'],
                        error_relative=relative_error(prediction,truth),
                        before_ratio=cfg['leg16_truth']['reserved_bytes']/truth,
                        after_ratio=prediction/truth,
                        source=cfg['leg16_truth']['source'])
    import numpy as np
    comparisons = []
    for label, reference in (('old-versus-new', old), ('frozen-versus-default', frozen)):
        target = new if label == 'old-versus-new' else old
        errors = {k: relative_error(target['measured'][k],reference['measured'][k]) for k in ('T','R','n_eff')}
        plane_errors = {}
        with np.load(run/f'{reference["variant"]}-2-planes.npz') as a, np.load(run/f'{target["variant"]}-2-planes.npz') as b:
            for key in a.files:
                plane_errors[key] = float(np.max(np.abs(a[key]-b[key]))/max(np.max(np.abs(a[key])),np.finfo(float).tiny))
        comparisons.append(dict(comparison=label, relative_errors=errors, plane_relative_errors=plane_errors,
                                plane_hashes_identical=target['plane_sha256']==reference['plane_sha256'],
                                material_hash_identical=target['measured']['epsilon_sha256']==reference['measured']['epsilon_sha256'],
                                ns_per_cell_step=[reference['ns_per_cell_step'],target['ns_per_cell_step']]))
    report['C2'] = comparisons
    report['envelope_repairs']=repair_envelopes(run,cfg['policy'])
    dump(run/'readout.json',readout_envelope(report))
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'checks', '-v'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (run/'checks.log').write_text(tests.stdout)
    print(tests.stdout[-1500:], flush=True)
    report['checks_exit_status'] = tests.returncode
    report['frozen_inventory'] = inventory(Path(os.environ['SOLVER_FORK']),Path(os.environ['SOLVER_FROZEN']))
    report['criteria'] = dict(C1=report['C1']['error_relative']<=.25,
        C2=all(max([*c['relative_errors'].values(),*c['plane_relative_errors'].values()])<=1e-12 and c['plane_hashes_identical'] and c['material_hash_identical'] for c in comparisons),
        C3=all(r.get('refused') for r in report['rows'][-2:]),
        C4=all(r['solver_report']['peak_checkpoints']==0 for r in report['rows'][2:5])
           and all(r.get('refused') and r['allocation_increase_bytes']<new['solver_report']['restart_state_bytes'] for r in report['rows'][-2:])
           and report['rows'][-1]['held_bytes']>0,
        C5=tests.returncode==0 and set(report['frozen_inventory']['differences'])=={
            'torchfdtd/models.py','torchfdtd/adjoint_memory.py','torchfdtd/plane_execution.py'})
    report['unknown'] = ['The full-device 2.19x/2.87x empirical models are distinct from this solver admission model; their after ratios are unmeasured and remain uncorrected.',
                         'No universal no-OOM guarantee follows from these three meshes and one real-memory-pressure refusal. Other devices, meshes and adjoint workloads are unmeasured.']
    report.update(status='completed-with-unmeasured-full-device-calibration', verdict=verdict(report),
                  wall_seconds=time.perf_counter()-started)
    save()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    effective.update(simulation_executed=True, resolved_cases=[{k:r[k] for k in ('variant','case_id','shape','steps','dt_s')} for r in report['rows'][:5]],
                     verdict=report['verdict'])
    dump(Path(os.environ['EFFECTIVE_JSON']), effective)
    dump(run/'readout.json',readout_envelope(report))
    print(json.dumps(dict(C1=report['C1'],criteria=report['criteria'],verdict=report['verdict'])),flush=True)
    # All five gates must pass. Amendment 1 allows the contextual unknowns to
    # retain a partial verdict; exit 54 records that finding explicitly.
    return 54 if all(report['criteria'].values()) else 52


if __name__ == '__main__':
    sys.exit(main())
