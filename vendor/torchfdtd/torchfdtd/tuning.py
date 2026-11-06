"""Explicit, measured cohort selection for repeated forward ensembles.

No estimates are advertised as measured throughput. Trials execute the complete
supplied workload, including result transfer, without invoking user objectives.
Tuning has a visible up-front cost and never runs implicitly in a solve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .batch import BatchCase, _write_json
from .models import Project
from .tensor_batch import run_tensor_batch


@dataclass
class TensorBatchTuning:
    cohort_size: int
    seconds: float
    workload_sha256: str
    hardware: dict
    candidates: list[dict]
    repeats: int
    cuda_graph: bool
    cuda_graph_steps: int = 1

    def as_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Tuning report exists. Choose a new path.')
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, self.as_dict())


def _result_digest(result):
    """Compare physical outputs only. Timing and cohort metadata must differ."""
    h = hashlib.sha256()
    arrays = [getattr(result, name) for name in
        ('electric', 'magnetic', 'signals', 'times', 'frames', 'frame_steps', 'epsilon')]
    for monitor in result.frequency_fields:
        h.update(str((monitor.get('components'),monitor.get('poynting_components'))).encode())
        arrays.extend(monitor[key] for key in ('frequency_hz','points_um','weights','fields','poynting','flux')
                      if monitor[key] is not None)
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def tune_tensor_batch(cases, *, candidates=(1, 2, 4, 8, 16), repeats=3,
                      device=0, memory_fraction=.6, cuda_graph=True, cancel=None,
                      progress=None, cuda_graph_steps=1):
    """Select the lowest median full-wall cost among actual cohort trials.

    Each candidate gets one warmup plus ``repeats`` measured full-ensemble runs.
    The order alternates. Every trial must reproduce identical physical arrays
    across candidates. Sizes larger than the ensemble are deduplicated. Sizes
    rejected by memory admission are recorded and excluded. Other errors stop
    tuning. No source time truncation, file saves or user-objective calls occur.

    ``seconds`` includes warmups, hashing and all trials. Timed solve samples
    exclude hashing and cold compilation. Peak allocation uses and resets the
    device's Torch peak-memory counters. Run on an otherwise idle GPU. Selected
    sizes are workload/hardware-specific, with no speed guarantee for later
    workloads. Pass ``report.cohort_size`` to run_tensor_batch or optimize.
    """
    started = time.perf_counter()
    cases = list(cases)
    if not cases:
        raise ValueError('Tuning requires at least one case.')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer.')
    sizes = list(candidates)
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 64 for n in sizes):
        raise ValueError('Candidates must be integers from 1 to 64.')
    sizes = sorted({min(n, len(cases)) for n in sizes})
    # Snapshot the input before any trials. Caller edits cannot change a trial.
    cases = [BatchCase(f'case-{i:04d}', c) if isinstance(c, Project) else
             BatchCase(c.id, c.project, dict(c.parameters)) if isinstance(c, BatchCase) else c
             for i, c in enumerate(cases)]
    if any(not isinstance(c, BatchCase) for c in cases):
        raise TypeError('Supply BatchCase or Project instances.')
    if not torch.cuda.is_available():
        raise RuntimeError('Cohort tuning requires CUDA.')
    if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < torch.cuda.device_count():
        raise ValueError('Invalid CUDA device.')
    fingerprint = hashlib.sha256(json.dumps([c.project.model_dump() for c in cases],
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    records = {n: dict(cohort_size=n, status='pending', samples=[], warmup_seconds=None) for n in sizes}
    reference = None
    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        hardware = dict(device=device, gpu=props.name, total_memory_bytes=props.total_memory,
                        capability=list(torch.cuda.get_device_capability(device)),
                        torch=torch.__version__, cuda=torch.version.cuda)
        for repeat in range(-1, repeats):
            for n in (sizes if repeat % 2 == 0 else sizes[::-1]):
                row = records[n]
                if row['status'] == 'memory_rejected':
                    continue
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('Cohort tuning cancelled.')
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                try:
                    report = run_tensor_batch(cases, cohort_size=n, device=device,
                        memory_fraction=memory_fraction, cuda_graph=cuda_graph, cancel=cancel,
                        cuda_graph_steps=cuda_graph_steps)
                except ValueError as exc:
                    if 'GPU memory allowance' not in str(exc):
                        raise
                    row.update(status='memory_rejected', reason=str(exc))
                    continue
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('Cohort tuning cancelled.')
                report.raise_for_errors()
                torch.cuda.synchronize(device)
                sample = dict(wall_seconds=time.perf_counter()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                hashes = [_result_digest(item.load()) for item in report.items]
                del report
                if reference is None:
                    reference = hashes
                elif hashes != reference:
                    raise ArithmeticError('Cohort tuning changed physical output arrays.')
                row['status'] = 'completed'
                if repeat < 0:
                    row['warmup_seconds'] = sample['wall_seconds']
                else:
                    row['samples'].append(sample)
                if progress:
                    progress(dict(cohort_size=n, repeat=repeat, repeats=repeats, **sample))
        valid = []
        for row in records.values():
            if row['status'] != 'completed':
                continue
            row['median_wall_seconds'] = statistics.median(s['wall_seconds'] for s in row['samples'])
            row['native_bitwise_identical'] = True
            valid.append(row)
        if not valid:
            raise ValueError('No candidate fits the GPU memory allowance.')
        winner = min(valid, key=lambda r: (r['median_wall_seconds'], r['cohort_size']))
    return TensorBatchTuning(winner['cohort_size'], time.perf_counter()-started,
        fingerprint, hardware, list(records.values()), repeats, cuda_graph, cuda_graph_steps)
