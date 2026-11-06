"""G7-05 timing aggregation and baseline comparison.

Each fresh process supplies six complete design iterations. Its first is cold;
its other five are reduced to one warm median before the five processes are
combined. A stage timer synchronizes CUDA on both sides of its boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import statistics
import time


STAGES = (
    'T_geometry', 'T_setup', 'T_forward', 'T_monitor', 'T_backward',
    'T_transfer_io', 'T_optimizer',
)


class StageTimes:
    def __init__(self, device='cpu'):
        self.device = device
        self.values = {stage: 0. for stage in STAGES}
        self._active = None

    def synchronize(self):
        if str(self.device).startswith('cuda'):
            import torch

            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if name not in self.values:
            raise KeyError(name)
        if self._active is not None:
            raise RuntimeError(f'Nested stage {name} would double-count {self._active}.')
        self.synchronize()
        started = time.perf_counter()
        self._active = name
        try:
            yield
        finally:
            self.synchronize()
            self.values[name] += time.perf_counter()-started
            self._active = None

    def call(self, name, function, *args, **kwargs):
        with self.stage(name):
            return function(*args, **kwargs)

    def report(self):
        return dict(self.values)


def _range(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values),
                samples=list(values))


def summarize_fresh_processes(records):
    """Five process records, each with cold iteration 1 and warm iterations 2-6."""
    if len(records) != 5:
        raise ValueError('G7-05 needs exactly five fresh process records.')
    cold, warm = {s: [] for s in STAGES}, {s: [] for s in STAGES}
    full_cold, full_warm = [], []
    for process in records:
        iterations = process.get('iterations')
        if not isinstance(iterations, list) or len(iterations) != 6:
            raise ValueError('Each process needs exactly six iterations.')
        for row in iterations:
            if set(row) != set(STAGES) or any(not isinstance(row[s], (float, int))
                                              or not math.isfinite(row[s]) or row[s] < 0 for s in STAGES):
                raise ValueError('Every stage time must be finite and nonnegative.')
        for stage in STAGES:
            cold[stage].append(float(iterations[0][stage]))
            warm[stage].append(statistics.median(float(row[stage]) for row in iterations[1:]))
        full_cold.append(sum(float(iterations[0][stage]) for stage in STAGES))
        full_warm.append(statistics.median(
            sum(float(row[stage]) for stage in STAGES) for row in iterations[1:]))
    return dict(cold={s: _range(cold[s]) for s in STAGES},
                warm={s: _range(warm[s]) for s in STAGES},
                full_iteration=dict(cold=_range(full_cold), warm=_range(full_warm)),
                process_count=5, iterations_per_process=6)


def stage_regressions(baseline, candidate, *, factor=1.25):
    """Return stages whose later median exceeds the recorded baseline by factor."""
    if factor <= 1 or not math.isfinite(factor):
        raise ValueError('Regression factor must exceed one.')
    failures = []
    for temperature in ('cold', 'warm'):
        for stage in STAGES:
            before = baseline[temperature][stage]['median']
            after = candidate[temperature][stage]['median']
            if not all(math.isfinite(v) and v >= 0 for v in (before, after)):
                raise ValueError('Stage medians must be finite and nonnegative.')
            if after > factor*before:
                failures.append(dict(temperature=temperature, stage=stage,
                                     baseline_seconds=before, candidate_seconds=after,
                                     factor=None if before == 0 else after/before))
    return failures


def autotuner_break_even(tuning_seconds, untuned_iteration_seconds, tuned_iteration_seconds,
                         *, untuned_cold_seconds=None, tuned_cold_seconds=None):
    """First positive iteration count whose accumulated saving repays tuning.

    A policy with no warm saving breaks even only if its cold saving pays for tuning.
    The full iteration times supplied here must include the same seven stages.
    Cold times default to warm times for simple repeated-iteration comparisons.
    """
    if untuned_cold_seconds is None:
        untuned_cold_seconds = untuned_iteration_seconds
    if tuned_cold_seconds is None:
        tuned_cold_seconds = tuned_iteration_seconds
    values = (tuning_seconds, untuned_iteration_seconds, tuned_iteration_seconds,
              untuned_cold_seconds, tuned_cold_seconds)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in values):
        raise ValueError('Tuning and iteration times must be finite and nonnegative.')
    first_saving = untuned_cold_seconds-tuned_cold_seconds
    still_to_repay = tuning_seconds-first_saving
    if still_to_repay <= 0:
        return 1
    warm_saving = untuned_iteration_seconds-tuned_iteration_seconds
    if warm_saving <= 0:
        return None
    return 1+math.ceil(still_to_repay/warm_saving)
