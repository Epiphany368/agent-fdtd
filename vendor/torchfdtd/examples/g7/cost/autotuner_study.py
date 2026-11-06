"""Measure G7-01 execution-policy tuning and its full-iteration break-even.

Run by absolute path from an external wheel environment, under the GPU timing
lock. A reduced fixture is available for development and is never acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import torch
import torchfdtd
from torchfdtd import tune_adjoint_execution
from torchfdtd.design_parameterization import DensityParameterization

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(torchfdtd.__file__).resolve()
if ROOT == PACKAGE or ROOT in PACKAGE.parents or 'site-packages' not in PACKAGE.parts:
    raise RuntimeError(f'G7-05 requires an installed wheel outside the checkout, got {PACKAGE}.')
sys.path.append(str(ROOT))

from examples.g7.cost.g701_iteration import G701Iteration, execution_policy  # noqa: E402
from examples.g7.cost.timing import autotuner_break_even  # noqa: E402
from examples.g7.metagrating import workflow as w  # noqa: E402


def prepared_input(*, reduced, backend):
    runner = G701Iteration(seed=1, storage='resident', reduced=reduced, backend=backend)
    settings, fixture = runner.settings, runner.case['fixture']
    initial = .5 * torch.randn((fixture['pixels'], 1),
                                 generator=torch.Generator().manual_seed(1))
    options = dict(spacing_um=(fixture['pixel_um'],
                               fixture['design_layer_um'][1]-fixture['design_layer_um'][0]),
                   initial=initial, mode='logits', filter_radius_um=settings.filter_radius_um,
                   boundary='periodic', beta=settings.betas[0], eta=.5)
    if settings.threshold_shift is None:
        design = DensityParameterization((fixture['pixels'], 1), **options)
    else:
        design = w.RobustDensity((fixture['pixels'], 1),
                                 threshold_shift=settings.threshold_shift, **options)
    project = w.build_project(runner.geometry, settings.design_mesh_um, 'TE',
                              'normal', settings, settings.physical_time_fs,
                              monitors=('transmission',))
    density = design().detach()
    epsilon = w.layer_epsilon(density, project.region, runner.geometry)
    frequency = w.C0/(np.asarray(w.DESIGN_WAVELENGTHS_UM)*1e-6)
    return runner, project, epsilon, frequency


def checked_summary(path, scenario, *, reduced, backend):
    summary = json.loads(path.read_text(encoding='utf-8'))
    if (summary.get('scenario'), summary.get('reduced'), summary.get('backend')) != (
            scenario, reduced, backend):
        raise ValueError(f'{path} is not a matching {scenario} cost summary')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resident-summary', type=Path)
    parser.add_argument('--host-summary', type=Path)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    case_path = ROOT/'docs/validation/cases/G7-05r2.json'
    if not args.reduced:
        if not case_path.exists():
            parser.error('The approved G7-05r2 case has not been committed.')
        case = json.loads(case_path.read_text(encoding='utf-8'))
        if (case.get('declared_before_run') is not True
                or not case.get('revision', {}).get('approved_by')
                or 'draft_status' in case):
            parser.error('The G7-05r2 case needs owner approval before full timing.')
        if args.resident_summary is None or args.host_summary is None:
            parser.error('Both five-process scenario summaries are required for break-even.')
    runner, project, epsilon, frequency = prepared_input(
        reduced=args.reduced, backend=args.backend)
    candidates = [execution_policy(runner.settings, storage)
                  for storage in ('resident', 'host')]
    started = time.perf_counter()
    selection = tune_adjoint_execution(
        project, epsilon, candidates=candidates, frequency_hz=frequency,
        probe_steps=min(24, project.region.steps), repeats=2,
        reference_cache_bytes=64*1024**2)
    if args.backend == 'cuda':
        torch.cuda.synchronize()
    tuning_seconds = time.perf_counter()-started
    if selection.policy not in candidates:
        raise RuntimeError('The tuner selected a policy outside the declared comparison.')
    selected_storage = 'resident' if selection.policy == candidates[0] else 'host'
    measured = None
    if args.resident_summary is not None and args.host_summary is not None:
        resident = checked_summary(args.resident_summary, 'g701-resident-broadband',
                                   reduced=args.reduced, backend=args.backend)
        host = checked_summary(args.host_summary, 'g701-host-broadband',
                               reduced=args.reduced, backend=args.backend)
        if (resident.get('package_version'), resident.get('wheel_sha256')) != (
                host.get('package_version'), host.get('wheel_sha256')) or (
                resident.get('package_version') != importlib.metadata.version('torchfdtd')):
            raise ValueError('The tuner and compared summaries must use the same installed wheel.')
        if (resident.get('cpu_threads') != host.get('cpu_threads')
                or resident.get('cpu_threads') != dict(omp='2', mkl='2')):
            raise ValueError('The compared summaries must use the same two-thread CPU policy.')
        selected = resident if selected_storage == 'resident' else host
        before = resident['statistics']['full_iteration']['warm']['median']
        after = selected['statistics']['full_iteration']['warm']['median']
        before_cold = resident['statistics']['full_iteration']['cold']['median']
        after_cold = selected['statistics']['full_iteration']['cold']['median']
        measured = dict(untuned_resident_warm_seconds=before,
                        tuned_policy_warm_seconds=after,
                        untuned_resident_cold_seconds=before_cold,
                        tuned_policy_cold_seconds=after_cold,
                        selected_storage=selected_storage,
                        break_even_iterations=autotuner_break_even(
                            tuning_seconds, before, after,
                            untuned_cold_seconds=before_cold,
                            tuned_cold_seconds=after_cold),
                        resident_summary=str(args.resident_summary),
                        host_summary=str(args.host_summary))
    result = dict(task='G7-05', status='development' if args.reduced else 'acceptance',
                  reduced=args.reduced, backend=args.backend,
                  package_version=importlib.metadata.version('torchfdtd'),
                  package_path=str(PACKAGE), interpreter=sys.executable,
                  cpu_threads=dict(omp=os.environ['OMP_NUM_THREADS'],
                                   mkl=os.environ['MKL_NUM_THREADS']),
                  case=None if not case_path.exists() else str(case_path),
                  case_sha256=None if not case_path.exists() else hashlib.sha256(
                      case_path.read_bytes()).hexdigest(),
                  tuner=dict(seconds=tuning_seconds, selected_storage=selected_storage,
                             selected_policy=asdict(selection.policy), report=selection.report),
                  measured_full_iterations=measured,
                  scope=('The tuner uses short forward/backward probes on the seed-1 '
                         'G7-01 initial geometry. Break-even uses independent five-process '
                         'medians of complete seven-stage iterations. This is a measured '
                         'policy choice, not a guarantee of global optimality.'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error(f'{args.out} already exists; use a new output path')
    args.out.write_text(json.dumps(result, indent=1, default=str)+'\n', encoding='utf-8')
    print(json.dumps(dict(selected_storage=selected_storage,
                          tuning_seconds=tuning_seconds,
                          measured_full_iterations=measured)), flush=True)


if __name__ == '__main__':
    main()
