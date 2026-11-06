"""Independent simulation ensembles with bounded process/GPU concurrency.

Processes isolate fdtd's global backend. This is concurrent independent-case
execution, not MPI domain decomposition or a fused leading-batch-dimension kernel.
Use a main guard on Windows and top-level, pickleable objective functions.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
import hashlib
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import time

import numpy as np
import torch

from .models import Project
from .solver import Result, Simulation, estimate
from .cuda_memory import cuda_mem_info


@dataclass
class BatchCase:
    id: str
    project: Project
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', self.id):
            raise ValueError('Case ids need 1–96 ASCII letters/digits/dot/underscore/hyphen, starting with a letter or digit.')
        self.project = Project.model_validate(self.project.model_dump())
        json.dumps(self.parameters, allow_nan=False)


@dataclass
class BatchItem:
    id: str
    status: str
    parameters: dict
    summary: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    output: str | None = None
    error: str | None = None
    pid: int | None = None
    device: str | None = None
    started: float | None = None
    finished: float | None = None
    resumed: bool = False
    result: Result | None = field(default=None, repr=False)

    def load(self):
        if self.result is not None:return self.result
        if self.output is None:raise ValueError('No saved or retained result. Set output_dir or keep_results=True.')
        return Result.load(self.output)

    def as_dict(self):
        return {k:v for k,v in vars(self).items() if k != 'result'}


@dataclass
class BatchReport:
    items: list[BatchItem]
    seconds: float
    plan: dict

    @property
    def successful(self):return all(item.status == 'completed' for item in self.items)

    def raise_for_errors(self):
        errors = [f'{x.id}: {x.error or x.status}' for x in self.items if x.status != 'completed']
        if errors:raise RuntimeError('Batch did not complete: ' + '; '.join(errors))
        return self

    def as_dict(self):
        return dict(seconds=self.seconds, plan=self.plan, successful=self.successful,
                    items=[item.as_dict() for item in self.items])


_WORKER_CANCEL = None


def _initialize_worker(device, cancel, cpu_threads, ready):
    global _WORKER_CANCEL
    _WORKER_CANCEL = cancel
    torch.set_num_threads(cpu_threads)
    if device is not None:torch.cuda.set_device(device)
    ready.wait(timeout=60)


def _worker_ready():
    return os.getpid()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20), b''):h.update(chunk)
    return h.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _execute(case, device, output_dir, objective, fingerprint, keep_results, cuda_graph):
    item = BatchItem(case.id, 'running', case.parameters, pid=os.getpid(),
                     device=f'cuda:{device}' if device is not None else 'cpu', started=time.time())
    try:
        result = Simulation(case.project).run(cancel=_WORKER_CANCEL, cuda_graph=cuda_graph)
        item.summary = result.summary
        item.status = 'cancelled' if result.summary['cancelled'] else 'completed'
        if output_dir:
            path = Path(output_dir) / (case.id + '.npz')
            temporary = path.with_suffix('.tmp.npz')
            result.save(temporary)
            temporary.replace(path)
            item.output = str(path)
        if item.status == 'completed' and objective is not None:
            metrics = objective(result)
            if not isinstance(metrics, dict):metrics = {'objective':metrics}
            metrics = {str(k):float(v) for k,v in metrics.items()}
            if not metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError('Objective must return a finite scalar or a nonempty dict of finite scalars.')
            item.metrics = metrics
        if keep_results:item.result = result
    except Exception as exc:
        item.status = 'failed'
        item.error = f'{type(exc).__name__}: {exc}'
    finally:
        item.finished = time.time()
        if device is not None:
            gc.collect()
            torch.cuda.empty_cache()
    if output_dir:
        record = dict(fingerprint=fingerprint, item=item.as_dict(),
                      result_sha256=_sha(item.output) if item.output else None)
        _write_json(Path(output_dir)/(case.id+'.json'), record)
    return item


class BatchRunner:
    """Persistent spawn workers for sweeps and successive optimizer generations.

    ``max_workers`` is the maximum per device. GPU memory admission uses 1.5 times
    the solver estimate plus 768 MiB per worker for context/allocator overhead.
    This is an estimate, not an allocation guarantee; other applications can still
    consume memory. ``memory_fraction`` reserves a share of currently free VRAM.
    CPU workers divide 80% of the available host memory, or the aggregate memory_limit_mb.
    """
    def __init__(self, *, backend='auto', devices=None, max_workers=2,
                 memory_fraction=.6, memory_limit_mb=None, cpu_threads=1, cuda_graph=True):
        if backend not in ('auto','cpu','cuda'):raise ValueError('backend must be auto, cpu or cuda.')
        if max_workers < 1 or cpu_threads < 1:raise ValueError('Worker and CPU thread counts must be positive.')
        if not 0 < memory_fraction <= .85:raise ValueError('memory_fraction must be in (0, .85].')
        if memory_limit_mb is not None and memory_limit_mb <= 0:raise ValueError('memory_limit_mb must be positive.')
        self.backend = 'cuda' if backend == 'cuda' or (backend == 'auto' and torch.cuda.is_available()) else 'cpu'
        if self.backend == 'cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
        self.devices = tuple(devices or (0,)) if self.backend == 'cuda' else (None,)
        if len(set(self.devices)) != len(self.devices):raise ValueError('Duplicate device ids.')
        if self.backend == 'cpu' and devices is not None:raise ValueError('devices is only valid for CUDA.')
        if self.backend == 'cuda' and any(not isinstance(d,int) or d < 0 or d >= torch.cuda.device_count() for d in self.devices):
            raise ValueError('Invalid CUDA device id.')
        self.max_workers, self.cpu_threads = max_workers, cpu_threads
        self.memory_fraction, self.memory_limit_mb = memory_fraction, memory_limit_mb
        self.cuda_graph = cuda_graph
        self._context = mp.get_context('spawn')
        self._cancel = self._context.Event()
        self._pools, self._pool_limits = {}, {}
        self._busy, self._closed = False, False

    def __enter__(self):return self

    def __exit__(self, *args):self.close()

    def close(self):
        self._cancel.set()
        for pool in self._pools.values():pool.shutdown(wait=True, cancel_futures=True)
        self._pools.clear()
        self._closed = True

    def cancel(self):
        """May be called by another thread; running cases stop at a time-step boundary."""
        self._cancel.set()

    def _plan(self, cases):
        max_mb = max(estimate(case.project)['estimated_memory_mb'] for case in cases)
        reserve_mb = max_mb*1.5 + (768 if self.backend=='cuda' else 64)
        counts, budgets = {}, {}
        for device in self.devices:
            budget = float('inf')
            if device is not None:
                free, _ = cuda_mem_info(device)
                # Workers release unused allocator blocks after each case. Existing
                # contexts remain: reserving overhead again is conservative.
                budget = free/2**20*self.memory_fraction
            else:
                # CPU workers share the host: each case's Simulation admits itself against the same
                # available memory, so the workers divide 80% of it among themselves.
                from .memory_profile import host_memory
                available = host_memory()['available_bytes']
                if available is not None:
                    budget = available/2**20*.8
            if self.memory_limit_mb is not None:budget = min(budget, self.memory_limit_mb)
            count = min(self.max_workers, len(cases), int(budget/reserve_mb) if math.isfinite(budget) else self.max_workers)
            if count < 1:raise ValueError(f'Insufficient batch memory on {device}: budget {budget:.1f} MiB, estimated worker {reserve_mb:.1f} MiB.')
            counts[device], budgets[str(device)] = count, budget if math.isfinite(budget) else None
        return counts, dict(backend=self.backend, devices=list(self.devices), workers={str(k):v for k,v in counts.items()},
                            largest_solver_estimate_mb=max_mb, estimated_worker_mb=reserve_mb, memory_budget_mb=budgets,
                            method='spawned independent simulation processes', cuda_graph=self.cuda_graph)

    def run(self, cases, *, objective=None, objective_key=None, output_dir=None, resume=False,
            keep_results=False, fail_fast=False, on_complete=None, cancel=None):
        if self._closed:raise RuntimeError('BatchRunner is closed.')
        if self._busy:raise RuntimeError('One run at a time per BatchRunner.')
        items = [BatchCase(case.id,case.project,dict(case.parameters)) if isinstance(case,BatchCase)
                 else BatchCase(f'case-{i:05d}',case) for i,case in enumerate(cases)]
        if not items:raise ValueError('At least one case is required.')
        if len({c.id for c in items}) != len(items):raise ValueError('Duplicate case ids.')
        for case in items:
            # Copy again so backend selection never mutates a caller's case.
            case.project = Project.model_validate({**case.project.model_dump(), 'region':{**case.project.region.model_dump(),'backend':self.backend}})
        if objective is not None:
            try:pickle.dumps(objective)
            except Exception as exc:raise ValueError('Objective must be a pickleable module-level function or callable.') from exc
        if resume and not output_dir:raise ValueError('Resume requires output_dir.')
        if resume and objective is not None and not objective_key:raise ValueError('Resume with an objective requires an explicit objective_key; change it when the objective changes.')
        directory = Path(output_dir).resolve() if output_dir else None
        if directory:directory.mkdir(parents=True,exist_ok=True)
        code = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob('*.py')):code.update(path.name.encode());code.update(path.read_bytes())
        completed, pending, fingerprints = {}, [], {}
        for case in items:
            payload = dict(project=case.project.model_dump(),parameters=case.parameters,code=code.hexdigest(),
                           objective=objective_key,has_objective=objective is not None,cuda_graph=self.cuda_graph,
                           numpy=np.__version__,torch=torch.__version__)
            digest = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
            fingerprints[case.id] = digest
            meta = directory/(case.id+'.json') if directory else None
            data = directory/(case.id+'.npz') if directory else None
            if meta and (meta.exists() or data.exists()):
                if not resume or not meta.exists():raise FileExistsError(f'Existing case output: {case.id}. Use a new directory or resume matching cases.')
                record = json.loads(meta.read_text(encoding='utf-8'))
                if record['fingerprint'] != digest:raise ValueError(f'Resume fingerprint mismatch: {case.id}. Use a new case id or output directory.')
                if record['item']['status'] == 'completed':
                    if not data.exists() or _sha(data) != record['result_sha256']:raise ValueError(f'Result checksum mismatch: {case.id}.')
                    item = BatchItem(**record['item']);item.output = str(data);item.resumed = True
                    if keep_results:item.result = Result.load(data)
                    completed[case.id] = item
                    continue
            pending.append(case)
        started = time.perf_counter()
        plan = dict(backend=self.backend,devices=list(self.devices),workers={},method='resume only')
        if not pending:return BatchReport([completed[c.id] for c in items],time.perf_counter()-started,plan)
        counts, plan = self._plan(pending)
        self._busy=True;self._cancel.clear()
        active, available = {}, dict(counts)
        try:
            for device,count in counts.items():
                if device in self._pools and self._pool_limits[device] != count:
                    self._pools.pop(device).shutdown(wait=True)
                if device not in self._pools:
                    self._pools[device] = ProcessPoolExecutor(max_workers=count,mp_context=self._context,
                        initializer=_initialize_worker,initargs=(device,self._cancel,self.cpu_threads,self._context.Barrier(count)))
                    self._pool_limits[device] = count
                    # Start every worker before submitting short simulations.
                    # Otherwise one early Windows spawn can consume the batch
                    # while the other workers are still importing Torch.
                    ready = [self._pools[device].submit(_worker_ready) for _ in range(count)]
                    for future in ready:future.result(timeout=90)
            while pending or active:
                if cancel is not None and cancel.is_set():self._cancel.set()
                if not self._cancel.is_set():
                    for device in self.devices:
                        while available[device] and pending:
                            case = pending.pop(0)
                            future = self._pools[device].submit(_execute,case,device,str(directory) if directory else None,
                                objective,fingerprints[case.id],keep_results,self.cuda_graph)
                            active[future] = (case,device);available[device]-=1
                else:
                    for case in pending:completed[case.id] = BatchItem(case.id,'cancelled',case.parameters)
                    pending.clear()
                if not active:break
                done,_ = wait(active,timeout=.1,return_when=FIRST_COMPLETED)
                for future in done:
                    case,device = active.pop(future);available[device]+=1
                    try:item = future.result()
                    except Exception as exc:item = BatchItem(case.id,'failed',case.parameters,error=f'{type(exc).__name__}: {exc}')
                    completed[case.id] = item
                    if fail_fast and item.status == 'failed':self._cancel.set()
                    if on_complete is not None:on_complete(item)
        except BaseException:
            self._cancel.set()
            if active:wait(active)
            raise
        finally:self._busy=False
        report = BatchReport([completed[c.id] for c in items],time.perf_counter()-started,plan)
        if directory:_write_json(directory/'batch-report.json',report.as_dict())
        return report


def run_batch(cases, *, objective=None, objective_key=None, output_dir=None, resume=False,
              keep_results=False, fail_fast=False, on_complete=None, cancel=None, **runner_options):
    """Convenience wrapper. Use BatchRunner for several evaluation generations."""
    with BatchRunner(**runner_options) as runner:
        return runner.run(cases,objective=objective,objective_key=objective_key,output_dir=output_dir,
                          resume=resume,keep_results=keep_results,fail_fast=fail_fast,on_complete=on_complete,cancel=cancel)


def parameter_case(base, parameters, case_id):
    """Set explicit dotted Project paths, then validate the whole scene once.

    Paths use native micrometre geometry and SI time, e.g. structures.0.radius.
    Multiple coupled parameters are set atomically before validation.
    """
    data = base.model_dump(mode='json')
    for path,value in parameters.items():
        parts = path.split('.');target = data
        try:
            for key in parts[:-1]:target = target[int(key)] if isinstance(target,(list,tuple)) else target[key]
            key = int(parts[-1]) if isinstance(target,list) else parts[-1]
            if isinstance(target,dict) and key not in target:raise KeyError(key)
            target[key] = value
        except (KeyError,IndexError,TypeError,ValueError) as exc:raise ValueError(f'Unknown or non-editable parameter path: {path}') from exc
    return BatchCase(case_id,Project.model_validate(data),dict(parameters))


def parameter_sweep(base, parameters, *, prefix='sweep'):
    """Cartesian/nested sweep in insertion order, with reproducible case ids."""
    names = list(parameters)
    values = [list(parameters[name]) for name in names]
    if not names or any(not v for v in values):raise ValueError('Supply at least one parameter and a nonempty list for every axis.')
    return [parameter_case(base,dict(zip(names,v)),f'{prefix}-{i:05d}') for i,v in enumerate(itertools.product(*values))]
