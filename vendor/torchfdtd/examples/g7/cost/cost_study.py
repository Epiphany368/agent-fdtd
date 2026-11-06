"""G7-05 five-process stage timing from an installed TorchFDTD wheel.

Run this script by absolute path with an external wheel environment and a
working directory outside the repository. The first import is the installed
package; only then are the repository-only examples added to sys.path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import torchfdtd

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(torchfdtd.__file__).resolve()
if ROOT == PACKAGE or ROOT in PACKAGE.parents or 'site-packages' not in PACKAGE.parts:
    raise RuntimeError(f'G7-05 requires an installed wheel outside the checkout, got {PACKAGE}.')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from examples.g7.cost.g701_iteration import G701Iteration  # noqa: E402
from examples.g7.cost.g703_iteration import G703Iteration  # noqa: E402
from examples.g7.cost.timing import STAGES, stage_regressions, summarize_fresh_processes  # noqa: E402


SCENARIOS = {
    'g701-resident-broadband': dict(workload='g701', storage='resident', frequencies='broadband'),
    'g701-host-broadband': dict(workload='g701', storage='host', frequencies='broadband'),
    'g701-resident-sequential': dict(workload='g701', storage='resident', frequencies='sequential'),
    'g703-resident': dict(workload='g703'),
}


def worker(scenario, *, reduced=False, backend='cuda'):
    spec = SCENARIOS[scenario]
    if spec['workload'] == 'g701':
        runner = G701Iteration(seed=1, storage=spec['storage'],
                               frequencies=spec['frequencies'], reduced=reduced, backend=backend)
    else:
        runner = G703Iteration(seed=1, reduced=reduced, backend=backend)
    started = time.perf_counter()
    observations = []
    for iteration in range(1, 7):
        observation = runner.step()
        observations.append(observation)
        # Progress is emitted after the synchronized stage timers have stopped.
        print(json.dumps(dict(scenario=scenario, iteration=iteration,
                              complete_iteration_seconds=sum(observation['stages'].values()),
                              objective=observation['objective'])), flush=True)
    policy = asdict(runner.objective.policy) if spec['workload'] == 'g701' else None
    return dict(scenario=scenario, reduced=reduced, backend=backend,
                package_version=importlib.metadata.version('torchfdtd'),
                package_path=str(PACKAGE), interpreter=sys.executable,
                cpu_threads=dict(omp=os.environ['OMP_NUM_THREADS'],
                                 mkl=os.environ['MKL_NUM_THREADS']),
                execution_policy=policy,
                iterations=[row['stages'] for row in observations],
                observations=[{key: value for key, value in row.items() if key != 'stages'}
                              for row in observations],
                process_wall_seconds=time.perf_counter()-started,
                stage_definition=('Every declared stage is synchronized before and after. '
                                  'Explicit host copies are in T_transfer_io; internal host-streamed '
                                  'transfers remain within T_forward or T_backward, so no overlap is '
                                  'counted twice. T_setup is cold-only model/reference preparation.'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=SCENARIOS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--reduced', action='store_true', help='Development fixture only; not acceptance data.')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--wheel', type=Path, help='Installed release wheel; required for the full five-process run.')
    parser.add_argument('--compare-to', type=Path,
                        help='A previous summary of this scenario; fail if a stage median exceeds 1.25 times it.')
    args = parser.parse_args(argv)
    if not args.reduced:
        case_path = ROOT/'docs/validation/cases/G7-05r2.json'
        if not case_path.exists():
            parser.error('The approved G7-05r2 case has not been committed.')
        case = json.loads(case_path.read_text(encoding='utf-8'))
        if (case.get('declared_before_run') is not True
                or not case.get('revision', {}).get('approved_by')
                or 'draft_status' in case):
            parser.error('The G7-05r2 case needs owner approval before full timing.')
    if args.worker:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(worker(args.scenario, reduced=args.reduced,
                                              backend=args.backend), indent=1)+'\n', encoding='utf-8')
        print(args.out, flush=True)
        return
    if args.wheel is None and not args.reduced:
        parser.error('--wheel is required for the full five-process study')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.wheel is not None:
        wheel = args.wheel.resolve(strict=True)
        if ROOT == wheel or ROOT in wheel.parents:
            parser.error('The release wheel must be outside the checkout.')
        wheel_hash = sha256(wheel)
    else:
        wheel, wheel_hash = None, None
    records = []
    for index in range(1, 6):
        path = args.out/f'process{index}.json'
        if path.exists():
            parser.error(f'{path} already exists; use a new output directory for a fresh run')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--scenario', args.scenario, '--out', str(path), '--backend', args.backend]
        if args.reduced:
            command.append('--reduced')
        subprocess.run(command, cwd=args.out.resolve(), check=True)
        records.append(json.loads(path.read_text(encoding='utf-8')))
    versions = {row['package_version'] for row in records}
    paths = {row['package_path'] for row in records}
    thread_policies = {tuple(sorted(row['cpu_threads'].items())) for row in records}
    if len(versions) != 1 or len(paths) != 1 or len(thread_policies) != 1:
        raise RuntimeError('The fresh processes used different installations or CPU thread settings.')
    policies = [row['execution_policy'] for row in records]
    if any(policy != policies[0] for policy in policies):
        raise RuntimeError('The fresh processes used different execution policies.')
    if wheel is not None and not next(iter(versions)) in wheel.name:
        raise RuntimeError('Installed TorchFDTD version does not match the wheel filename.')
    summary = dict(scenario=args.scenario, reduced=args.reduced, backend=args.backend,
                   package_version=next(iter(versions)), package_path=next(iter(paths)),
                   cpu_threads=records[0]['cpu_threads'],
                   torchfdtd_import='installed', wheel=None if wheel is None else str(wheel),
                   wheel_sha256=wheel_hash, execution_policy=policies[0],
                   stage_names=list(STAGES),
                   statistics=summarize_fresh_processes(records),
                   process_wall_seconds=[row['process_wall_seconds'] for row in records],
                   observations=[row['observations'] for row in records])
    if args.compare_to is not None:
        baseline = json.loads(args.compare_to.read_text(encoding='utf-8'))
        if (baseline.get('scenario'), baseline.get('reduced'), baseline.get('backend')) != (
                args.scenario, args.reduced, args.backend):
            parser.error('The regression baseline must have the same scenario and execution fixture.')
        summary['regression_factor'] = 1.25
        summary['regressions'] = stage_regressions(baseline['statistics'], summary['statistics'])
    target = args.out/'summary.json'
    target.write_text(json.dumps(summary, indent=1)+'\n', encoding='utf-8')
    print(target, flush=True)
    if summary.get('regressions'):
        raise SystemExit(f'{len(summary["regressions"])} stage medians exceed the recorded baseline by more than 1.25x')


if __name__ == '__main__':
    main()
