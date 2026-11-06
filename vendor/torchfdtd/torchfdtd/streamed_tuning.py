"""Measured, budget-constrained selection among streamed execution policies.

This is a bounded two-duration experiment, not a globally optimal runtime model. The
selected policy must still pass admission when the full simulation is invoked.
"""
from dataclasses import dataclass, replace, asdict
from collections import OrderedDict
import statistics
import time

import torch

from .streamed import StreamedAdjointOptions, StreamedSimulation, _reservation
from .streamed_cost import predict_duration


class _ReferenceCache:
    """Bound full-gradient references by bytes, independent of candidate count."""
    def __init__(self, budget):
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:self.misses += 1
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return None if value is None else value[0]

    def put(self, key, value):
        size = sum(t.numel()*t.element_size() for t in (value[0], *value[1]))
        if size > self.budget:return
        if key in self.entries:
            _, old = self.entries.pop(key);self.bytes -= old
        while self.bytes+size > self.budget:
            _, (_, old) = self.entries.popitem(last=False);self.bytes -= old
        # Gradients returned by CatBackward can be small views retaining a
        # larger packed carrier. Own compact copies so this byte cap is real.
        value = (value[0].detach().clone(),tuple(t.detach().clone(memory_format=torch.contiguous_format) for t in value[1]))
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


class _TuningWorkload:
    physics = 'Yee/CPML'
    gradient_coordinates = ('epsilon',)

    def __init__(self, project, epsilon, frequency_hz=None, window=None):
        if torch.is_inference_mode_enabled():raise ValueError('Run gradient policy tuning outside torch.inference_mode().')
        if not isinstance(epsilon,torch.Tensor) or epsilon.device.type != 'cpu' or epsilon.dtype not in (torch.float32,torch.float64):
            raise ValueError('Tuning requires a real CPU epsilon tensor.')
        self.project, self.epsilon = project, epsilon
        self.design = (epsilon.detach().requires_grad_(),)
        self.spectral = None
        if window is not None and frequency_hz is None:raise ValueError('A spectral window requires frequency_hz.')
        if frequency_hz is not None:
            from .adjoint_spectrum import SpectralObservation
            self.spectral = SpectralObservation(epsilon,project.region,
                [m.component for m in project.monitors if m.enabled],frequency_hz,window)

    @property
    def parameter_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.design)

    def output_bytes(self, steps):
        if self.spectral is not None:return self.spectral.reservation(1)['spectral_output_bytes']
        return steps*sum(m.enabled for m in self.project.monitors)*self.epsilon.element_size()*(2 if self.project.region.complex_fields else 1)

    def reservation(self, options):
        return _reservation(self.project,self.epsilon,options,self.spectral)

    def model(self, project, options):
        return StreamedSimulation(project,options)

    def evaluate(self, model, length):
        if self.spectral is None:
            result = model(*self.design)
            values = result.signals
        else:
            window = None if self.spectral.window is None else self.spectral.window[:length]
            result = model.spectrum(*self.design,self.spectral.frequency,window=window)
            # Compare spectra and VJPs in field-amplitude units. Raw DFT values
            # carry seconds, so a fixed absolute tolerance could hide errors.
            values = result.fields/(length*self.project.region.time_step)
        gradients = torch.autograd.grad(values.abs().square().sum(),self.design)
        return result, values.detach(), gradients

    def comparison_gradients(self, gradients):
        return gradients


class _DispersiveTuningWorkload(_TuningWorkload):
    physics = 'Yee/CPML/ADE'
    gradient_coordinates = ('epsilon_inf','strength * dt^2','omega0 * dt','gamma * dt')

    def __init__(self, project, epsilon, strength, omega0, gamma, frequency_hz=None, window=None):
        super().__init__(project,epsilon,frequency_hz,window)
        from .dispersive_adjoint import DispersiveSimulation
        # Validate compact inputs without allocating a normalized full-domain
        # carrier. Admission must precede material packing and solver state.
        for value in (strength,omega0,gamma):
            if isinstance(value,torch.Tensor) and (value.device.type!='cpu' or value.dtype!=epsilon.dtype):
                raise ValueError('Tuning material tensors must match epsilon CPU device and dtype.')
        values,self.layout = DispersiveSimulation._inputs(self,epsilon,strength,omega0,gamma,streamed=True)
        self.design = tuple(t.detach().requires_grad_() for t in values)

    def reservation(self, options):
        return _reservation(self.project,self.epsilon,options,self.spectral,
            pole_count=self.layout.pole_count,parameter_shapes=self.layout.shapes)

    def model(self, project, options):
        from .streamed_dispersive import StreamedDispersiveSimulation
        return StreamedDispersiveSimulation(project,options)

    def comparison_gradients(self, gradients):
        # A derivative wrt SI strength can be around 1e-30. Convert to the
        # dimensionless coordinates before comparing each material separately.
        dt = self.project.region.time_step
        return gradients[0].clone(), gradients[1]/dt/dt, gradients[2]/dt, gradients[3]/dt


@dataclass(frozen=True)
class StreamedTuning:
    options: StreamedAdjointOptions
    report: dict


def _calibration_lengths(steps, probe, depth, strategy, limit):
    first = min(steps,((probe+depth-1)//depth)*depth)
    lengths = [min(steps,probe)] if strategy == 'prefix' else sorted({first,min(steps,2*first)})
    if lengths[-1] > limit:
        raise ValueError('Whole-block calibration exceeds max_calibration_steps. Reduce temporal depth or explicitly increase the calibration limit.')
    return lengths


def _fit_default_policy(workload, policy, overhead):
    """Shrink generated policies to current budgets, keeping explicit ones fixed.

    Reservation is monotone in slab width for a fixed temporal depth and tier.
    Use binary search for the largest admitted core width up to the proposed
    width. Reduce depth only if even one core cell does not fit. File fallback
    is permitted only with an explicit directory and budget.
    """
    tiers = [policy.state_storage]
    if policy.state_storage == 'host' and policy.state_directory and policy.disk_budget_bytes:
        tiers.append('disk')
    last_error = ''
    attempts = 0
    def admitted(candidate):
        nonlocal last_error,attempts
        attempts += 1
        if candidate.host_budget_bytes <= overhead:
            last_error = 'Host budget cannot hold tuning reference buffers.'
            return False
        try:workload.reservation(replace(candidate,host_budget_bytes=candidate.host_budget_bytes-overhead))
        except ValueError as exc:
            last_error = str(exc)
            return False
        return True
    for tier in tiers:
        depth = min(policy.temporal_depth,workload.project.region.steps)
        while True:
            candidate = replace(policy,state_storage=tier,temporal_depth=depth,
                                slab_width=min(policy.slab_width,workload.project.region.shape[0]))
            if admitted(candidate):return candidate,dict(admission_probes=attempts)
            narrow = replace(candidate,slab_width=1)
            if admitted(narrow):
                low,high = 1,candidate.slab_width
                while low+1 < high:
                    middle = (low+high)//2
                    if admitted(replace(candidate,slab_width=middle)):low = middle
                    else:high = middle
                return replace(candidate,slab_width=low),dict(admission_probes=attempts)
            if depth == 1:break
            depth = max(1,depth//2)
    return None,dict(admission_probes=attempts,reason=last_error)


def tune_streamed(project, epsilon, *, options=None, candidates=None, probe_steps=24, repeats=2,
                  strategy='replay_cost', max_calibration_steps=512, refine_candidates=0,
                  frequency_hz=None, window=None, reference_cache_bytes=64*1024**2):
    """Return a measured slab/K/transfer policy without changing design gradients.

    Costs include preparation, forward, checkpoint replay and backward over the
    whole-block calibration durations for each candidate. An explicit candidate list allows checkpoint
    counts and buffer counts to vary. Optional frequencies calibrate online DFT
    output and gradients instead of time signals. Reference gradients use a
    bounded host cache. Generated candidates are fitted to budgets, with file
    fallback only when a directory and budget were explicitly configured.
    Multi-GPU policies and unconfigured storage locations are never inferred.
    """
    workload = _TuningWorkload(project,epsilon,frequency_hz,window)
    return _tune_streamed(project,workload,options=options,candidates=candidates,
        probe_steps=probe_steps,repeats=repeats,strategy=strategy,max_calibration_steps=max_calibration_steps,
        refine_candidates=refine_candidates,reference_cache_bytes=reference_cache_bytes)


def tune_streamed_dispersive(project, epsilon_inf, strength, omega0, gamma, *, options=None,
        candidates=None, probe_steps=24, repeats=2, strategy='replay_cost',
        max_calibration_steps=512, refine_candidates=0, frequency_hz=None, window=None,
        reference_cache_bytes=64*1024**2):
    """Select admitted ADE slab/replay policies from measured forward and VJP.

    All four original material inputs and their .grad values remain untouched.
    Each calibration compares dimensionless material derivatives independently.
    Packing, forward, the sample objective, replay and backward are timed.
    Policy tuning itself consumes time and host memory. No optimum is guaranteed.
    """
    workload = _DispersiveTuningWorkload(project,epsilon_inf,strength,omega0,gamma,frequency_hz,window)
    return _tune_streamed(project,workload,options=options,candidates=candidates,
        probe_steps=probe_steps,repeats=repeats,strategy=strategy,max_calibration_steps=max_calibration_steps,
        refine_candidates=refine_candidates,reference_cache_bytes=reference_cache_bytes)


def _generated_candidates(project, workload, base, reference_cache_bytes, max_calibration_steps):
    planning = []
    n = project.region.shape[0]
    candidates = [replace(base, slab_width=min(n,width), temporal_depth=depth, tile_transfers='sync')
                  for width,depth in ((base.slab_width,1),(base.slab_width,base.temporal_depth),
                                     (2*base.slab_width,base.temporal_depth))]
    if torch.device(base.device).type == 'cuda' and base.reuse_tile_buffers:
        candidates.append(replace(base, slab_width=min(n,2*base.slab_width), tile_transfers='async', tile_buffers=2))
    # Deeper tiles can save forward replay at the cost of local state copies.
    # Keep the zero-slot policies in the race, since shallow/small workloads
    # can regress when copies cost more than the eliminated updates.
    if base.local_checkpoints == 0 and base.temporal_depth >= 8:
        candidates.extend(replace(p,local_checkpoints=1) for p in list(candidates)
                          if p.slab_width == min(n,2*base.slab_width) and p.temporal_depth == base.temporal_depth)
    # Include a low-memory global schedule and a less replay-heavy one.
    # These are admitted independently because saved banks can dominate
    # DRAM/file capacity even when the CUDA tile itself fits.
    candidates.extend(replace(base,slab_width=min(n,base.slab_width),checkpoints=count,tile_transfers='sync')
                      for count in {0,min(32,base.checkpoints+2)} if count != base.checkpoints)
    candidates = list(dict.fromkeys(candidates))
    longest = min(project.region.steps,max_calibration_steps)
    reference_upper = workload.parameter_bytes+workload.output_bytes(longest)
    overhead = min(reference_cache_bytes,3*len(candidates)*reference_upper)+4*reference_upper
    fitted = []
    for proposed in candidates:
        policy,detail = _fit_default_policy(workload,proposed,overhead)
        planning.append(dict(proposed=asdict(proposed),fitted=None if policy is None else asdict(policy),**detail))
        # Keep an unfitted proposal for a precise rejection at measurement
        # admission. Conservative planning overhead can exceed the final
        # reference working set for short calibrations.
        fitted.append(proposed if policy is None else policy)
    candidates = list(dict.fromkeys(fitted))
    return candidates, planning


def _validate_tuning_arguments(probe_steps, repeats, strategy, max_calibration_steps,
                               refine_candidates, reference_cache_bytes):
    if isinstance(probe_steps,bool) or not isinstance(probe_steps,int) or not 10 <= probe_steps <= 256:
        raise ValueError('probe_steps must be an integer between 10 and 256.')
    if isinstance(repeats,bool) or not isinstance(repeats,int) or not 1 <= repeats <= 5:
        raise ValueError('repeats must be an integer between one and five.')
    if strategy not in ('prefix','replay_cost'):raise ValueError('strategy must be prefix or replay_cost.')
    if isinstance(max_calibration_steps,bool) or not isinstance(max_calibration_steps,int) or not 10 <= max_calibration_steps <= 100000:
        raise ValueError('max_calibration_steps must be an integer between 10 and 100000.')
    if isinstance(refine_candidates,bool) or not isinstance(refine_candidates,int) or not 0 <= refine_candidates <= 12:
        raise ValueError('refine_candidates must be an integer between zero and twelve.')
    if isinstance(reference_cache_bytes,bool) or not isinstance(reference_cache_bytes,int) or reference_cache_bytes<0:
        raise ValueError('reference_cache_bytes must be a nonnegative integer.')


def _tune_streamed(project, workload, *, options=None, candidates=None, probe_steps=24, repeats=2,
                   strategy='replay_cost', max_calibration_steps=512, refine_candidates=0,
                   reference_cache_bytes=64*1024**2, _policy_type=StreamedAdjointOptions):
    _validate_tuning_arguments(probe_steps,repeats,strategy,max_calibration_steps,
                               refine_candidates,reference_cache_bytes)
    epsilon = workload.epsilon
    base = options or StreamedAdjointOptions()
    if candidates is None:
        candidates,planning = _generated_candidates(project,workload,base,reference_cache_bytes,max_calibration_steps)
    else:candidates,planning = list(candidates),[]
    if not 1 <= len(candidates) <= 12 or not all(isinstance(p,_policy_type) for p in candidates):
        raise ValueError(f'Supply one to twelve {_policy_type.__name__} candidates.')
    if len({torch.device(p.device) for p in candidates}) != 1:
        raise ValueError('Compare policies on a single selected execution device.')
    started = time.perf_counter()
    plans, errors = {}, {}
    for index,policy in enumerate(candidates):
        try:plans[index] = _calibration_lengths(project.region.steps,probe_steps,policy.temporal_depth,strategy,max_calibration_steps)
        except ValueError as exc:errors[index] = str(exc)
    refinement = {}
    if strategy == 'replay_cost' and refine_candidates:
        for index,lengths in plans.items():
            longer = min(project.region.steps,2*lengths[-1])
            if lengths[-1] < longer <= max_calibration_steps:refinement[index] = longer
    all_lengths = sorted({length for lengths in plans.values() for length in lengths} | set(refinement.values()))
    references = _ReferenceCache(reference_cache_bytes)
    evaluated_lengths = set()
    records, feasible = [], []
    reference_policy, reference_index = None, None
    extra_reference_evaluations = 0
    # The tuner holds a reference signal/gradient while evaluating later modes.
    cache_reservation = min(reference_cache_bytes,
        sum(workload.parameter_bytes+workload.output_bytes(length) for length in all_lengths))
    # A cache miss can hold the candidate, a newly computed reference and its
    # dimensionless material-gradient conversion concurrently. Cache eviction
    # never allows this comparison workspace to exceed the reserved budget.
    extra = cache_reservation+4*workload.parameter_bytes+4*max((workload.output_bytes(n) for n in all_lengths),default=0)

    def measure(index,admitted,length):
        nonlocal extra_reference_evaluations
        prefix = project.model_copy(deep=True)
        prefix.region.steps = length
        model = workload.model(prefix,admitted)
        samples = []
        for repeat in range(repeats+1):
            before = time.perf_counter()
            with torch.enable_grad():
                result, values, raw_gradients = workload.evaluate(model,length)
            elapsed = time.perf_counter()-before
            forward_seconds=result.report['forward_seconds']
            backward_seconds=result.report['backward_seconds']
            # An exhausted result graph can still own its native system. Drop
            # it before a cache miss starts a second reference solve, so the
            # admission budget need not hold two simultaneous solver workspaces.
            del result
            gradient = workload.comparison_gradients(raw_gradients)
            del raw_gradients
            reference = references.get(length)
            if reference is None:
                if index == reference_index:
                    reference = (values,gradient)
                else:
                    # Compare unique durations with the same reference policy.
                    with torch.enable_grad():
                        reference_result,reference_values,raw_reference = workload.evaluate(workload.model(prefix,reference_policy),length)
                    reference = (reference_values,workload.comparison_gradients(raw_reference))
                    extra_reference_evaluations += 1
                    del reference_result, reference_values, raw_reference
                references.put(length,reference)
            tolerance = dict(rtol=5e-5,atol=2e-6) if epsilon.dtype == torch.float32 else dict(rtol=1e-9,atol=1e-11)
            torch.testing.assert_close(values,reference[0],**tolerance)
            for actual,expected in zip(gradient,reference[1]):
                torch.testing.assert_close(actual,expected,**tolerance)
            del actual, expected
            evaluated_lengths.add(length)
            if repeat:samples.append(dict(total=elapsed,forward=forward_seconds,backward=backward_seconds))
            del gradient, values, reference
        return dict(steps=length,samples=samples,**{k:statistics.median(s[k] for s in samples) for k in ('total','forward','backward')})

    admitted_policies = {}
    for index,policy in enumerate(candidates):
        row = dict(index=index, policy=asdict(policy))
        try:
            if index in errors:raise ValueError(errors[index])
            if policy.host_budget_bytes <= extra:raise ValueError('Host budget cannot hold tuning reference buffers.')
            admitted = replace(policy,host_budget_bytes=policy.host_budget_bytes-extra)
            # Check the real duration's source/output storage, not only the prefix.
            reservation = workload.reservation(admitted)
        except ValueError as exc:
            row.update(status='rejected',reason=str(exc))
            records.append(row)
            continue
        if reference_policy is None:reference_policy, reference_index = admitted, index
        admitted_policies[index] = admitted
        probes = [measure(index,admitted,length) for length in plans[index]]
        prediction = predict_duration(probes[0],probes[-1],steps=project.region.steps,
                                      depth=policy.temporal_depth,checkpoints=policy.checkpoints)
        score = probes[0]['total'] if strategy == 'prefix' else prediction['seconds']
        row.update(status='measured',seconds=[s['total'] for s in probes[0]['samples']],
                   median_seconds=probes[0]['total'],reservation=reservation,probes=probes,
                   calibration_steps=plans[index],
                   prediction=prediction,selection_score_seconds=score)
        records.append(row)
        feasible.append((score,index))
    if not feasible:
        reasons = ' '.join(f"Candidate {row['index']}: {row['reason']}" for row in records)
        raise ValueError('No streamed tuning candidate fits the full-workload memory budgets and calibration limits. '+reasons)
    refined = []
    for _,index in sorted(feasible)[:refine_candidates]:
        if index not in refinement:continue
        row,policy = records[index],candidates[index]
        row['initial_prediction'] = row['prediction']
        row['probes'].append(measure(index,admitted_policies[index],refinement[index]))
        row['calibration_steps'] = [probe['steps'] for probe in row['probes']]
        row['prediction'] = predict_duration(*row['probes'][-2:],steps=project.region.steps,
                                             depth=policy.temporal_depth,checkpoints=policy.checkpoints)
        row['selection_score_seconds'] = row['prediction']['seconds']
        refined.append(index)
    _, selected = min((row['selection_score_seconds'],row['index']) for row in records if row['status']=='measured')
    device = torch.device(candidates[selected].device)
    report = dict(selected_index=selected, candidates=records, probe_steps=min(project.region.steps,probe_steps),
                  calibration_steps=sorted(evaluated_lengths),strategy=strategy,max_calibration_steps=max_calibration_steps,
                  reference_policy_index=reference_index,
                  refine_candidates=refine_candidates,refined_indices=refined,
                  extra_reference_evaluations=extra_reference_evaluations,
                  full_steps=project.region.steps, warmups_per_probe=1,repeats=repeats,
                  total_warmup_runs=sum(len(row['probes']) for row in records if row['status']=='measured'),
                  tuning_seconds=time.perf_counter()-started,tuning_reference_reservation_bytes=extra,
                  reference_cache_limit_bytes=reference_cache_bytes,reference_cache_reservation_bytes=cache_reservation,
                  reference_cache_peak_bytes=references.peak_bytes,reference_cache_hits=references.hits,reference_cache_misses=references.misses,
                  physics=workload.physics,gradient_comparison_coordinates=workload.gradient_coordinates,
                  observation='online_spectrum' if workload.spectral is not None else 'time_history',
                  default_candidate_planning=planning,
                  torch_version=torch.__version__,device=str(device),
                  hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  scope='Selection from measured prefixes with optional two-duration checkpoint-replay extrapolation. Generated candidates fit current budgets and may fall back to explicitly configured file banks. File timings include OS cache effects. Full-duration optimality, resident-vs-streamed selection and sustained physical-storage performance are not established.')
    return StreamedTuning(candidates[selected],report)
