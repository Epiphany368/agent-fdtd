"""Shared-budget, sequential case replay for coupled FDTD objectives."""
from dataclasses import dataclass, replace
from copy import deepcopy
import math
import time

import torch

from .adjoint_spectrum import SpectralObservation
from .execution_tuning import AdjointExecutionPolicy, _resident_reservation, _streamed_options
from .memory_profile import host_memory
from .plane_execution import plane_mode, plane_reservation
from .streamed import _reservation


@dataclass(frozen=True)
class AdjointCase:
    """A fixed project and execution policy with explicit material bindings.

    parameter_indices selects epsilon, or epsilon_inf/strength/omega0/gamma,
    from the batch call's CPU tensors. Shared indices accumulate derivatives.
    Frequencies, mesh, monitors and sources are fixed during each graph.
    Recorded plane cases require an explicit homogeneous fixed exterior value
    in fixed_background_epsilon. Other algorithms reject that argument.
    """
    project: object
    policy: AdjointExecutionPolicy
    parameter_indices: tuple = (0,)
    frequency_hz: object = None
    quadrature_counts: dict | None = None
    block_size: int = 32
    fixed_background_epsilon: float | None = None

    def __post_init__(self):
        if not isinstance(self.policy,AdjointExecutionPolicy):raise ValueError('Each case needs an AdjointExecutionPolicy.')
        indices=tuple(self.parameter_indices)
        if len(indices) not in (1,4) or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in indices):
            raise ValueError('Bind one dielectric or four ADE parameter indices.')
        object.__setattr__(self,'parameter_indices',indices)
        recorded=getattr(self.policy,'recorded',None) is not None
        background=self.fixed_background_epsilon
        if recorded:
            if type(background) is not float or not math.isfinite(background) or background<1:
                raise ValueError('Recorded cases require finite Python float fixed_background_epsilon >= 1.')
            if len(indices)!=1 or not plane_mode(self.project):
                raise ValueError('Recorded cases support nondispersive field planes only.')
        elif background is not None:
            raise ValueError('fixed_background_epsilon requires a recorded execution policy.')
        if isinstance(self.block_size,bool) or not isinstance(self.block_size,int) or self.block_size<1:
            raise ValueError('block_size must be a positive integer.')


@dataclass(frozen=True)
class AdjointBatchOptions:
    """Total solver-owned budgets, with CPU observations and accumulators.

    Geometry/input storage, optimizer and user objective graphs, CUDA context,
    unrelated allocations and OS file cache are outside these budgets.
    """
    host_budget_bytes: int = 8*1024**3
    gpu_budget_bytes: int = 1024**3
    disk_budget_bytes: int | None = None
    output_budget_bytes: int = 64*1024**2
    replay_rtol: float = 1e-6
    replay_atol: float = 0.

    def __post_init__(self):
        for name in ('host_budget_bytes','gpu_budget_bytes','disk_budget_bytes','output_budget_bytes'):
            value=getattr(self,name)
            if name=='disk_budget_bytes' and value is None:continue
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer byte count.')
        if any(not math.isfinite(v) or v<0 for v in (self.replay_rtol,self.replay_atol)):
            raise ValueError('Replay tolerances must be finite and nonnegative.')


@dataclass
class AdjointBatchResult:
    cases: tuple
    report: dict


def _tensors(result):
    if isinstance(result,dict):return tuple(p.fields for p in result.values())
    return (result.fields if hasattr(result,'fields') else result.signals,)


def _replace_tensors(result,values):
    if isinstance(result,dict):return {key:replace(plane,fields=value) for (key,plane),value in zip(result.items(),values)}
    return replace(result,**{'fields' if hasattr(result,'fields') else 'signals':values[0]})


class _PreparedCase:
    def __init__(self,spec):
        self.spec=spec
        self.dispersive=len(spec.parameter_indices)==4
        self.planes=plane_mode(spec.project)
        self.configuration=deepcopy((spec.policy,spec.fixed_background_epsilon,
            spec.parameter_indices,spec.block_size,spec.quadrature_counts))
        settings=dict(dispersive=self.dispersive,quadrature_counts=spec.quadrature_counts)
        if getattr(spec.policy,'recorded',None) is not None:
            settings['fixed_background_epsilon']=spec.fixed_background_epsilon
        self.model=spec.policy.simulation(spec.project,**settings)
        self.project=self.model.project
        self.snapshot=self.project.model_dump()
        self.spectral=None
        scalar=torch.empty((),dtype=getattr(torch,self.project.region.precision),device='cpu')
        if self.planes:
            if spec.frequency_hz is None:raise ValueError('Plane cases require frequency_hz.')
            self.spectral=self.model.model._spectral(scalar,spec.frequency_hz,spec.block_size)
            self.output_shapes=[(self.spectral.frequency.numel(),len(plan['weights']),6)
                for _,_,plan,_ in self.model.model.plans]
            self.layout_bytes=self.model.model.layout_reservation_bytes
            points=sum(shape[1] for shape in self.output_shapes)
            self.metadata_bytes=(4*points+len(self.output_shapes)*self.spectral.frequency.numel())*scalar.element_size()
        else:
            self.layout_bytes=0
            monitors=[m for m in self.project.monitors if m.enabled]
            if spec.frequency_hz is not None:
                self.spectral=SpectralObservation(scalar,self.project.region,[m.component for m in monitors],
                    spec.frequency_hz,block_size=spec.block_size)
            self.output_shapes=[(self.project.region.steps if self.spectral is None else self.spectral.frequency.numel(),len(monitors))]
            self.metadata_bytes=0 if self.spectral is None else self.spectral.frequency.numel()*scalar.element_size()
        self.output_dtype=(torch.complex128 if scalar.dtype==torch.float64 else torch.complex64) if self.spectral is not None or self.project.region.complex_fields else scalar.dtype
        item=torch.empty((),dtype=self.output_dtype,device='cpu').element_size()
        self.output_bytes=sum(math.prod(shape)*item for shape in self.output_shapes)

    def check_fixed(self):
        spec=self.spec
        configuration=(spec.policy,spec.fixed_background_epsilon,
            spec.parameter_indices,spec.block_size,spec.quadrature_counts)
        if self.project.model_dump()!=self.snapshot or configuration!=self.configuration:
            raise RuntimeError('Batch case configuration changed. Rebuild the batch.')
        if getattr(spec.policy,'recorded',None) is not None:
            self.model.check_fixed()

    def inputs(self,parameters):
        if max(self.spec.parameter_indices)>=len(parameters):raise ValueError('Case parameter index exceeds the supplied tensors.')
        values=tuple(parameters[i] for i in self.spec.parameter_indices)
        dtype=getattr(torch,self.project.region.precision)
        if any(v.device.type!='cpu' or v.dtype!=dtype for v in values):
            raise ValueError('Case materials must be CPU tensors matching project precision.')
        return values

    def reservation(self,parameters):
        self.check_fixed()
        values=self.inputs(parameters)
        shapes=tuple(tuple(v.shape) for v in values)
        frequency=None if self.spectral is None else self.spectral.frequency
        policy=self.spec.policy
        if getattr(policy,'recorded',None) is not None:
            from .recorded_execution import _recorded_reservation
            return _recorded_reservation(self.project,shapes,policy,frequency,
                block_size=self.spec.block_size,quadrature_counts=self.spec.quadrature_counts,
                plane=self.model.model,fixed_background_epsilon=self.spec.fixed_background_epsilon)
        if policy.resident is not None:
            return _resident_reservation(self.project,shapes,policy,frequency,block_size=self.spec.block_size,
                quadrature_counts=self.spec.quadrature_counts,plane=self.model.model if self.planes else None)
        options=_streamed_options(policy)
        if self.planes:
            return plane_reservation(self.model.model,shapes,options,frequency,block_size=self.spec.block_size)
        poles=0
        if self.dispersive:
            from .adjoint_memory import _material_shapes
            shapes,poles=_material_shapes(self.project.region,shapes)
        elif shapes[0] not in (self.project.region.shape,self.project.region.shape+(3,)):
            raise ValueError('epsilon shape must match the grid.')
        return _reservation(self.project,values[0],options,self.spectral,pole_count=poles,
            parameter_shapes=shapes if self.dispersive else None)

    def evaluate(self,parameters):
        self.check_fixed()
        values=self.inputs(parameters)
        if self.planes:return self.model(*values,frequency_hz=self.spectral.frequency,block_size=self.spec.block_size)
        if self.spectral is not None:
            return self.model.spectrum(*values,frequency_hz=self.spectral.frequency,block_size=self.spec.block_size)
        return self.model(*values)


class _BatchRun:
    def __init__(self,cases,options,parameters):
        self.cases,self.options=cases,options
        self.output_bytes=sum(case.output_bytes for case in cases)
        if self.output_bytes>options.output_budget_bytes:raise ValueError('Batch output budget exceeded before simulation.')
        parameters_bytes=sum(v.numel()*v.element_size() for v in parameters)
        metadata=sum(case.layout_bytes+case.metadata_bytes+64*1024 for case in cases)
        # All results + incoming seeds + a comparison/copy allowance coexist
        # with an active case. Reserve accumulation and gradient-copy carriers.
        self.overhead=3*self.output_bytes+4*parameters_bytes+metadata
        self.templates=[]
        self.report=dict(execution='sequential_case_replay',case_count=len(cases),
            full_case_graph_retention=False,output_bytes=self.output_bytes,
            parameter_bytes=parameters_bytes,metadata_reservation_bytes=metadata,
            batch_overhead_bytes=self.overhead,forward_cases=0,replayed_cases=0,
            scope='Shared solver/output/gradient-carrier admission. Caller geometry, input storage, objective and optimizer graphs, CUDA context and OS file cache excluded. Sequential, first-order execution.')
        reservations=[self.admit(case,parameters) for case in cases]
        self.report.update(case_reservations=reservations,
            host_reservation_bytes=self.overhead+max(r['host_reservation_bytes'] for r in reservations),
            gpu_reservation_bytes=max(r['gpu_reservation_bytes'] for r in reservations),
            disk_reservation_bytes=max(r.get('disk_reservation_bytes',r.get('disk_checkpoint_reservation_bytes',0)) for r in reservations))

    def admit(self,case,parameters):
        reservation=case.reservation(parameters)
        host=self.overhead+reservation['host_reservation_bytes']
        available=host_memory()['available_bytes']
        limit=self.options.host_budget_bytes
        if available is not None:limit=min(limit,int(available*.8))
        if host>limit:raise ValueError('Batch outputs, gradients and active solver exceed the shared host budget.')
        if reservation['gpu_reservation_bytes']>self.options.gpu_budget_bytes:
            raise ValueError('Active solver exceeds the shared batch GPU budget.')
        disk=reservation.get('disk_reservation_bytes',reservation.get('disk_checkpoint_reservation_bytes',0))
        if disk and (self.options.disk_budget_bytes is None or disk>self.options.disk_budget_bytes):
            raise ValueError('File-backed cases require a sufficient shared batch disk budget.')
        return reservation


def _replay(run,case,parameters,expected,seeds):
    # Local scope must release the native system before another case starts.
    run.admit(case,parameters)
    with torch.enable_grad():
        values=case.inputs(parameters)
        inputs=tuple(p.detach().requires_grad_(p.requires_grad) for p in values)
        # Evaluate the case's local material tuple without copying the whole
        # global parameter set. Repeated index bindings remain independent leaves.
        if case.planes:result=case.model(*inputs,frequency_hz=case.spectral.frequency,block_size=case.spec.block_size)
        elif case.spectral is not None:result=case.model.spectrum(*inputs,frequency_hz=case.spectral.frequency,block_size=case.spec.block_size)
        else:result=case.model(*inputs)
        outputs=_tensors(result)
        active_outputs=[];active_seeds=[]
        for value,want,seed in zip(outputs,expected,seeds):
            if value.shape!=want.shape or value.dtype!=want.dtype or not torch.allclose(value.detach(),want,
                    rtol=run.options.replay_rtol,atol=run.options.replay_atol):
                raise RuntimeError('Recomputed FDTD case output drifted. Keep all case settings fixed.')
            if seed is not None and value.requires_grad:
                active_outputs.append(value);active_seeds.append(seed)
        active_inputs=tuple(p for p in inputs if p.requires_grad)
        gradients=torch.autograd.grad(active_outputs,active_inputs,active_seeds,allow_unused=True) if active_outputs and active_inputs else (None,)*len(active_inputs)
        iterator=iter(gradients)
        report=next(iter(result.values())).report if isinstance(result,dict) else result.report
        return tuple(next(iterator) if p.requires_grad else None for p in inputs),dict(report)


class _Batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,run,*parameters):
        started=time.perf_counter()
        outputs=[]
        for case in run.cases:
            run.admit(case,parameters)
            result=case.evaluate(parameters)
            tensors=_tensors(result)
            if len(tensors)!=len(case.output_shapes):raise RuntimeError('Unexpected case output count.')
            for value,shape in zip(tensors,case.output_shapes):
                if value.shape!=shape or value.dtype!=case.output_dtype or value.device.type!='cpu':
                    raise RuntimeError('Case output differs from its admitted layout.')
                if not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite FDTD case output.')
            outputs.extend(tensors)
            run.templates.append(_replace_tensors(result,[v.new_empty(0) for v in tensors]))
            run.report['forward_cases']+=1
            del result,tensors,value
        ctx.run,ctx.output_count=run,len(outputs)
        ctx.save_for_backward(*outputs,*parameters)
        ctx.set_materialize_grads(False)
        run.report['batch_forward_seconds']=time.perf_counter()-started
        return tuple(outputs)

    @staticmethod
    def backward(ctx,*seeds):
        if torch.is_grad_enabled():raise RuntimeError('Adjoint batches support first-order derivatives only.')
        saved=ctx.saved_tensors
        expected,parameters=saved[:ctx.output_count],saved[ctx.output_count:]
        run=ctx.run
        started=time.perf_counter()
        accumulated=[None]*len(parameters)
        offset=0
        run.report['replayed_cases']=0
        run.report['replay_reports']=[]
        for case in run.cases:
            count=len(case.output_shapes)
            group=seeds[offset:offset+count]
            if any(seed is not None for seed in group) and any(parameters[i].requires_grad for i in case.spec.parameter_indices):
                gradients,report=_replay(run,case,parameters,expected[offset:offset+count],group)
                for index,gradient in zip(case.spec.parameter_indices,gradients):
                    if gradient is not None:
                        if accumulated[index] is None:accumulated[index]=gradient.clone()
                        else:accumulated[index].add_(gradient)
                del gradients
                run.report['replayed_cases']+=1
                run.report['replay_reports'].append(report)
            offset+=count
        run.report['batch_backward_seconds']=time.perf_counter()-started
        return None,*accumulated


class RecomputedAdjointBatch(torch.nn.Module):
    """Return ordinary FDTD result objects with one shared first-order backward.

    All cases are admitted together before any simulation. Results remain on
    CPU and can feed a nonseparable Torch objective. Backward replays one case
    graph at a time and sums derivatives of shared material parameters. This
    is sequential case replay, not concurrent CUDA execution or automatic
    policy tuning. Select case policies with tune_adjoint_execution first.
    """
    def __init__(self,cases,options=None):
        super().__init__()
        specs=tuple(cases)
        if not specs or not all(isinstance(case,AdjointCase) for case in specs):
            raise ValueError('Provide at least one AdjointCase.')
        self.options=options or AdjointBatchOptions()
        if not isinstance(self.options,AdjointBatchOptions):raise ValueError('Use AdjointBatchOptions.')
        from .boundaries import reject_pmc_faces
        for case in specs:reject_pmc_faces(case.project.region,'RecomputedAdjointBatch')
        self._cases=tuple(_PreparedCase(case) for case in specs)

    def _plan(self,parameters):
        if not parameters or not all(isinstance(p,torch.Tensor) and p.device.type=='cpu' and p.dtype in (torch.float32,torch.float64) for p in parameters):
            raise ValueError('Pass explicit real CPU material tensors.')
        return _BatchRun(self._cases,self.options,parameters)

    def plan(self,*parameters):
        """Check metadata and live shared budgets without allocating fields."""
        return self._plan(parameters).report

    def forward(self,*parameters):
        if torch.is_inference_mode_enabled():raise ValueError('Use torch.no_grad(), not inference_mode, for adjoint batches.')
        run=self._plan(parameters)
        flat=_Batch.apply(run,*parameters)
        results=[];offset=0
        for case,template in zip(run.cases,run.templates):
            count=len(case.output_shapes)
            results.append(_replace_tensors(template,flat[offset:offset+count]))
            offset+=count
        return AdjointBatchResult(tuple(results),run.report)
