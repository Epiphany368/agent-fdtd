"""Group heterogeneous fixed-duration projects into exact CUDA cohorts.

No padding, resampling, shortened runs or precision changes are introduced.
Different compatible groups run successively on one GPU. Cases within each
cohort share native CUDA launches. This is not multi-GPU domain decomposition.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import torch

from .batch import BatchCase, BatchItem, BatchReport, _write_json
from .boundaries import absorber_faces
from .cuda_graph import validate_graph_steps
from .models import Project
from .solver import estimate
from .tensor_batch import _topology, run_tensor_batch
from .cuda_memory import cuda_mem_info


def _cases(cases):
    result = [BatchCase(f'case-{i:04d}', c) if isinstance(c, Project) else c
              for i, c in enumerate(cases)]
    if not result:
        raise ValueError('Grouped batch requires at least one case.')
    if any(not isinstance(c, BatchCase) for c in result):
        raise TypeError('Supply BatchCase or Project instances.')
    if len({c.id for c in result}) != len(result):
        raise ValueError('Duplicate case ids.')
    # Snapshot input before planning and execution so callbacks cannot mutate it.
    return [BatchCase(c.id, c.project, dict(c.parameters)) for c in result]


def _plan(cases, cohort_size, memory_limit_bytes):
    if isinstance(cohort_size, bool) or not isinstance(cohort_size, int) or not 1 <= cohort_size <= 64:
        raise ValueError('cohort_size must be an integer from 1 to 64.')
    if memory_limit_bytes is not None and (isinstance(memory_limit_bytes, bool)
            or not math.isfinite(memory_limit_bytes) or memory_limit_bytes <= 0):
        raise ValueError('memory_limit_bytes must be finite and positive.')
    groups = {}
    for index, case in enumerate(cases):
        r = case.project.region
        if r.backend == 'cpu':
            raise ValueError(f'{case.id}: grouped CUDA batch cannot execute a CPU project.')
        if r.complex_fields:
            raise ValueError(f'{case.id}: use BatchRunner for complex Bloch fields.')
        if r.run_control.auto_shutoff:
            raise ValueError(f'{case.id}: grouped CUDA batch requires fixed-duration runs.')
        cost = math.ceil(estimate(case.project)['estimated_memory_mb'] * 2**20)
        if memory_limit_bytes is not None and cost > memory_limit_bytes:
            raise ValueError(f'{case.id}: one case exceeds the GPU memory allowance.')
        key = _topology(r, absorber_faces(case.project))
        if key not in groups:
            groups[key] = dict(group=len(groups), signature=hashlib.sha256(repr(key).encode()).hexdigest(),
                               shape=list(r.shape), precision=r.precision, steps=r.steps,
                               indices=[], estimated_bytes=[])
        groups[key]['indices'].append(index)
        groups[key]['estimated_bytes'].append(cost)
    cohorts = []
    for group in groups.values():
        indices, used = [], 0
        for index, cost in zip(group['indices'], group['estimated_bytes']):
            if indices and (len(indices) == cohort_size or
                           memory_limit_bytes is not None and used + cost > memory_limit_bytes):
                cohorts.append(dict(group=group['group'], indices=indices, estimated_bytes=used))
                indices, used = [], 0
            indices.append(index)
            used += cost
        cohorts.append(dict(group=group['group'], indices=indices, estimated_bytes=used))
    return dict(execution='grouped_fused_cuda', total_cases=len(cases), cohort_size=cohort_size,
                group_count=len(groups), groups=list(groups.values()), cohorts=cohorts,
                memory_limit_bytes=memory_limit_bytes, padding_cells=0, resume=False,
                result_order='input', objective_order='cohort execution')


def plan_grouped_batch(cases, *, cohort_size=4, memory_limit_bytes=None):
    """Inspect exact groups and estimated memory without initializing CUDA.

    Cohort size is a cap, not an autotuned performance prediction. Optional
    memory_limit_bytes splits groups conservatively using the native estimator.
    The estimate does not guarantee sufficient driver or graph memory.
    """
    return _plan(_cases(cases), cohort_size, memory_limit_bytes)


def run_grouped_batch(cases, *, cohort_size=4, objective=None, output_dir=None,
                      keep_results=True, device=0, memory_fraction=.6,
                      cuda_graph=True, cuda_graph_steps=1, cancel=None, progress=None):
    """Run mixed meshes, durations, boundaries and precisions in exact groups.

    Returns BatchReport items in original input order with original case IDs.
    Objectives execute in cohort order and should not depend on call ordering.
    All groups use real, fixed-duration native CUDA tensor execution. Unsupported
    project types fail rather than silently falling back to a different solver.
    Cancellation leaves not-yet-started cases cancelled. A numerical failure
    raises, while an objective failure marks only its own result item failed.
    Output directories must be new. Saving uses one subdirectory per cohort.
    Resume and GUI ensemble submission are provided by other workflows only.
    """
    start = time.perf_counter()
    validate_graph_steps(cuda_graph_steps, cuda_graph)
    cases = _cases(cases)
    _plan(cases, cohort_size, None)  # Reject unsupported input before any GPU work.
    if not 0 < memory_fraction <= .9:
        raise ValueError('memory_fraction must be positive and at most .9.')
    if not torch.cuda.is_available():
        raise RuntimeError('Grouped batch requires CUDA.')
    if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < torch.cuda.device_count():
        raise ValueError('Invalid CUDA device.')
    root = Path(output_dir) if output_dir is not None else None
    if root is not None and root.exists():
        raise FileExistsError('Grouped batch output directory exists. Use a new directory.')
    with torch.cuda.device(device):
        free, _ = cuda_mem_info()
        plan = _plan(cases, cohort_size, int(free * memory_fraction))
        plan['planning_seconds'] = time.perf_counter() - start
        if root is not None:
            root.mkdir(parents=True, exist_ok=False)
            _write_json(root / 'plan.json', plan)
        items, executions = {}, []
        for number, cohort in enumerate(plan['cohorts']):
            if cancel is not None and cancel.is_set():
                break
            indices = cohort['indices']
            def on_progress(state):
                progress(dict(state, cohort=number, group=cohort['group'],
                              input_indices=indices, total_cases=len(cases)))
            result = run_tensor_batch([cases[i] for i in indices], cohort_size=len(indices),
                objective=objective, output_dir=root / f'cohort-{number:04d}' if root else None,
                keep_results=keep_results, device=device, memory_fraction=memory_fraction,
                cuda_graph=cuda_graph, cuda_graph_steps=cuda_graph_steps, cancel=cancel,
                progress=on_progress if progress is not None else None)
            items.update((item.id, item) for item in result.items)
            executions.append(dict(cohort=number, **result.plan))
        ordered = [items.get(c.id, BatchItem(c.id, 'cancelled', c.parameters)) for c in cases]
        plan.update(executions=executions, cohorts_started=len(executions),
                    loop_seconds=sum(p['loop_seconds'] for p in executions),
                    setup_seconds=sum(p['setup_seconds'] for p in executions))
        report = BatchReport(ordered, time.perf_counter() - start, plan)
        if root is not None:
            _write_json(root / 'grouped-batch.json', report.as_dict())
        return report
