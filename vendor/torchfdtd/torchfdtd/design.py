"""Reproducible black-box inverse design using parallel differential evolution.

This API does not claim differentiable FDTD or an adjoint gradient. Each proposal
is a complete independent simulation. Objectives receive the same Result as a
single run, including complex monitor fields and physical mesh coordinates.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np

from .batch import BatchRunner, parameter_case, _write_json


@dataclass
class DesignResult:
    parameters: dict
    objective: float
    history: list[dict]
    evaluations: int
    seed: int

    def as_dict(self):return vars(self).copy()


def optimize(base, bounds, objective, *, runner=None, population=8, generations=10,
             seed=0, mutation=.7, crossover=.8, maximize=False, metric='objective',
             output_dir=None, resume=False, objective_key=None, on_generation=None,
             cancel=None, execution='process', cohort_size=None, **runner_options):
    """DE/rand/1/bin with simultaneous population evaluation and bounded proposals.

    ``bounds`` maps dotted native Project paths to (lower, upper), in native units.
    ``generations`` counts trial generations after the initial population.
    Failed simulations/invalid objective values stop optimization explicitly.
    Mesh, time and objective convergence remain the caller's responsibility.
    ``execution='tensor'`` evaluates each population through shared CUDA
    launches. It accepts cohort_size, device, memory_fraction, cuda_graph and
    cuda_graph_steps for optional multi-step graph replays.
    Tensor execution requires compatible real fixed-duration projects and does
    not support resume or a process runner. Objectives may be local callables.
    """
    if not bounds:raise ValueError('At least one design parameter is required.')
    names = list(bounds);limits = np.asarray(list(bounds.values()),dtype=float)
    if limits.shape != (len(names),2) or not np.isfinite(limits).all() or not np.all(limits[:,0] < limits[:,1]):
        raise ValueError('Every bound must contain finite lower < upper values.')
    if not isinstance(population,int) or population < 4:raise ValueError('Population must be an integer >= 4.')
    if not isinstance(generations,int) or generations < 0:raise ValueError('Generations must be a nonnegative integer.')
    if not 0 < mutation <= 2 or not 0 <= crossover <= 1:raise ValueError('Require 0 < mutation <= 2 and 0 <= crossover <= 1.')
    if execution not in ('process','tensor'):raise ValueError('execution must be process or tensor.')
    if execution=='tensor':
        if runner is not None:raise ValueError('Tensor execution does not accept a process runner.')
        if resume:raise ValueError('Tensor design does not support resume. Use process execution.')
        invalid=set(runner_options)-{'device','memory_fraction','cuda_graph','cuda_graph_steps'}
        if invalid:raise ValueError('Unsupported tensor options: '+', '.join(sorted(invalid)))
    else:
        if cohort_size is not None:raise ValueError('cohort_size requires tensor execution.')
        if runner is not None and runner_options:raise ValueError('Configure resources on the supplied runner.')
    owned = execution=='process' and runner is None
    if owned:runner=BatchRunner(**runner_options)
    rng = np.random.default_rng(seed)
    lo,hi = limits.T
    vectors = rng.uniform(lo,hi,(population,len(names)))
    directory = Path(output_dir) if output_dir else None
    history, evaluations = [], 0
    sign = -1 if maximize else 1
    config = dict(bounds={k:list(v) for k,v in bounds.items()},population=population,seed=seed,mutation=mutation,
                  crossover=crossover,maximize=maximize,metric=metric,objective_key=objective_key)
    if execution=='tensor':config.update(execution=execution,cohort_size=cohort_size,resources=runner_options)
    if directory:
        import json
        directory.mkdir(parents=True,exist_ok=True)
        config_path = directory/'design-config.json'
        if config_path.exists():
            if not resume or json.loads(config_path.read_text(encoding='utf-8')) != config:
                raise ValueError('Existing design configuration differs, or resume is disabled. Use a new directory.')
        else:_write_json(config_path,config)
    def evaluate(proposals,generation):
        nonlocal evaluations
        if cancel is not None and cancel.is_set():raise InterruptedError('Design cancelled before evaluation.')
        cases = [parameter_case(base,dict(zip(names,row.tolist())),f'design-{i:05d}') for i,row in enumerate(proposals)]
        destination=directory/f'generation-{generation:04d}' if directory else None
        if execution=='tensor':
            from .tensor_batch import run_tensor_batch
            report=run_tensor_batch(cases,objective=objective,output_dir=destination,
                keep_results=False,cohort_size=cohort_size,cancel=cancel,**runner_options)
        else:
            report = runner.run(cases,objective=objective,objective_key=objective_key,
                                output_dir=destination,resume=resume,fail_fast=True,cancel=cancel)
        if cancel is not None and cancel.is_set():raise InterruptedError('Design cancelled during evaluation.')
        report.raise_for_errors()
        values = np.asarray([item.metrics[metric] for item in report.items],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Objective values must be finite.')
        evaluations += len(cases)
        return values
    def record(generation,values):
        best = int(np.argmin(sign*values))
        entry = dict(generation=generation,best=float(values[best]),mean=float(np.mean(values)),
                     parameters=dict(zip(names,vectors[best].tolist())),evaluations=evaluations)
        history.append(entry)
        if directory:_write_json(directory/'design-history.json',history)
        if on_generation:on_generation(entry)
        return best
    try:
        values = evaluate(vectors,0)
        best = record(0,values)
        for generation in range(1,generations+1):
            proposals = np.empty_like(vectors)
            for i in range(population):
                candidates = np.delete(np.arange(population),i)
                a,b,c = rng.choice(candidates,3,replace=False)
                mutant = np.clip(vectors[a]+mutation*(vectors[b]-vectors[c]),lo,hi)
                mask = rng.random(len(names)) < crossover
                mask[rng.integers(len(names))] = True
                proposals[i] = np.where(mask,mutant,vectors[i])
            trial = evaluate(proposals,generation)
            improved = sign*trial <= sign*values
            vectors[improved],values[improved] = proposals[improved],trial[improved]
            best = record(generation,values)
        result = DesignResult(dict(zip(names,vectors[best].tolist())),float(values[best]),history,evaluations,seed)
        if directory:_write_json(directory/'design-result.json',result.as_dict())
        return result
    finally:
        if owned:runner.close()
