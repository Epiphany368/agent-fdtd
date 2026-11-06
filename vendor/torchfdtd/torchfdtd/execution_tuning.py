"""Measured resident/streamed adjoint selection with a common CPU design API."""
from dataclasses import dataclass, replace
import math

import torch

from .adjoint_memory import estimate_adjoint_memory
from .differentiable import AdjointOptions, DifferentiableSimulation
from .memory_profile import host_memory
from .cuda_memory import cuda_budget_limit
from .reversible_cpml import ReversibleCPMLOptions
from .streamed import StreamedAdjointOptions, StreamedSimulation
from .streamed_tuning import (_TuningWorkload, _DispersiveTuningWorkload,
                             _generated_candidates, _tune_streamed, _validate_tuning_arguments)


@dataclass(frozen=True)
class AdjointExecutionPolicy:
    """One explicit execution candidate. host_budget_bytes bounds solver RAM.

    CPU caller inputs and geometry/optimizer graphs remain outside the budget.
    Exactly one of resident, streamed or recorded must be supplied. Resident
    and recorded CUDA use differentiable copies with CPU design tensors.
    Recorded CPML is an explicit fixed-plane policy, excluded from tuning.
    """
    resident: AdjointOptions | None = None
    streamed: StreamedAdjointOptions | None = None
    device: str = 'cuda'
    host_budget_bytes: int = 8*1024**3
    recorded: ReversibleCPMLOptions | None = None

    def __post_init__(self):
        if sum(value is not None for value in (self.resident, self.streamed, self.recorded)) != 1:
            raise ValueError('Supply exactly one resident, streamed or recorded execution policy.')
        if self.resident is not None and not isinstance(self.resident,AdjointOptions):
            raise ValueError('resident must be AdjointOptions.')
        if self.streamed is not None and not isinstance(self.streamed,StreamedAdjointOptions):
            raise ValueError('streamed must be StreamedAdjointOptions.')
        if self.recorded is not None and not isinstance(self.recorded,ReversibleCPMLOptions):
            raise ValueError('recorded must be ReversibleCPMLOptions.')
        if torch.device(self.device).type not in ('cpu','cuda'):
            raise ValueError('Execution selection supports CPU or CUDA.')
        if self.streamed is not None and torch.device(self.streamed.device)!=torch.device(self.device):
            raise ValueError('Streamed policy must use the selected execution device.')
        if isinstance(self.host_budget_bytes,bool) or not isinstance(self.host_budget_bytes,int) or self.host_budget_bytes<1:
            raise ValueError('host_budget_bytes must be a positive integer.')

    @property
    def temporal_depth(self):return self.streamed.temporal_depth if self.streamed is not None else 1

    @property
    def checkpoints(self):return 0 if self.recorded is not None else (self.resident or self.streamed).checkpoints

    def simulation(self,project,*,dispersive=False,quadrature_counts=None,fixed_background_epsilon=None):
        from .plane_execution import plane_mode
        planes=plane_mode(project)
        if not planes and quadrature_counts is not None:
            raise ValueError('Quadrature counts require fixed field planes.')
        if self.recorded is not None:
            if not planes or dispersive:
                raise ValueError('Recorded execution currently requires nondispersive fixed field planes.')
            from .recorded_execution import _RecordedPlanesFromHost
            return _RecordedPlanesFromHost(project,self,dispersive,quadrature_counts,
                fixed_background_epsilon=fixed_background_epsilon)
        if fixed_background_epsilon is not None:
            raise ValueError('fixed_background_epsilon applies only to recorded execution.')
        if self.resident is not None:return _ResidentFromHost(project,self,dispersive,quadrature_counts)
        if planes:return _StreamedPlanesFromHost(project,self,dispersive,quadrature_counts)
        model=StreamedSimulation
        if dispersive:
            from .streamed_dispersive import StreamedDispersiveSimulation
            model=StreamedDispersiveSimulation
        return model(project,_streamed_options(self))


def _streamed_options(policy):
    return replace(policy.streamed,host_budget_bytes=min(policy.host_budget_bytes,policy.streamed.host_budget_bytes))


def _resident_project(project,options):
    # Deferred large-grid projects are admitted through explicit byte budgets.
    # Legacy candidates without that budget retain the workbench cell guard.
    project=project.model_copy(deep=True)
    project.region.memory_mode='budgeted' if options.resident_budget_bytes is not None else 'resident'
    from .adjoint_memory import _resident_contract
    _resident_contract(project.region,options)
    return project


def _resident_reservation(project,shapes,policy,frequency_hz=None,window=None,block_size=32,*,quadrature_counts=None,plane=None):
    from .plane_execution import plane_mode, plane_model, plane_reservation
    project=_resident_project(project,policy.resident)
    if shapes[0] not in (project.region.shape,project.region.shape+(3,)):
        raise ValueError('epsilon shape must match the grid, optionally with three components.')
    if plane_mode(project):
        if window is not None:raise ValueError('Plane execution does not yet support a temporal window.')
        plane=plane if plane is not None else plane_model(project,policy.resident,len(shapes)==4,quadrature_counts)
        result=plane_reservation(plane,shapes,policy.resident,frequency_hz,device=policy.device,block_size=block_size)
    else:
        if quadrature_counts is not None:raise ValueError('Quadrature counts require fixed field planes.')
        settings=dict(device=policy.device,frequency_hz=frequency_hz,window=window,block_size=block_size)
        if len(shapes)==4:settings['parameter_shapes']=shapes
        result=estimate_adjoint_memory(project,policy.resident,**settings)
    item=8 if project.region.precision=='float64' else 4
    parameters=sum(math.prod(shape) for shape in shapes)*item
    output=result.get('plane_output_bytes',result.get('spectral_output_bytes',result['output_history_bytes']))
    cuda=torch.device(policy.device).type=='cuda'
    # Inputs copied to CUDA and their incoming gradients coexist with the
    # solver. CPU output/gradient copies must also fit outside caller storage.
    gpu_copy=2*parameters if cuda else 0
    host_copy=2*parameters+2*output+result.get('plane_result_metadata_bytes',0) if cuda else 0
    gpu=result['gpu_reservation_bytes']+gpu_copy
    host=result['host_reservation_bytes']+host_copy
    available=host_memory()['available_bytes']
    limit=policy.host_budget_bytes
    if available is not None:limit=min(limit,int(available*.8))
    if host>limit:raise ValueError('Resident solver and transfer reservation exceed the unified host budget.')
    active=host if not cuda else gpu
    if policy.resident.resident_budget_bytes is not None and active>policy.resident.resident_budget_bytes:
        raise ValueError('Resident solver and transfers exceed the explicit resident byte budget.')
    if cuda:
        limit=cuda_budget_limit(policy.device,gpu,policy.resident.gpu_budget_bytes)
        if gpu>limit:raise ValueError('Resident solver and transfer reservation exceed the GPU budget.')
    return dict(result,host_reservation_bytes=host,gpu_reservation_bytes=gpu,
                host_transfer_reservation_bytes=host_copy,gpu_transfer_reservation_bytes=gpu_copy)


def _host_design(project,values,dispersive):
    if len(values)!=(4 if dispersive else 1):raise ValueError('Wrong number of material inputs.')
    epsilon=values[0]
    if not isinstance(epsilon,torch.Tensor) or epsilon.device.type!='cpu' or epsilon.dtype not in (torch.float32,torch.float64):
        raise ValueError('Unified execution requires real CPU design tensors.')
    if (epsilon.dtype==torch.float64)!=(project.region.precision=='float64'):
        raise ValueError('epsilon dtype must match the project precision.')
    for value in values:
        if isinstance(value,torch.Tensor) and (value.device.type!='cpu' or value.dtype!=epsilon.dtype):
            raise ValueError('Material tensors must match epsilon CPU device and dtype.')
    return tuple(torch.as_tensor(v,dtype=epsilon.dtype,device='cpu') for v in values)


class _StreamedPlanesFromHost(torch.nn.Module):
    def __init__(self,project,policy,dispersive,quadrature_counts):
        super().__init__()
        from .plane_execution import plane_model
        self.policy,self.dispersive=policy,dispersive
        self.options=_streamed_options(policy)
        self.model=plane_model(project,self.options,dispersive,quadrature_counts)
        self.project=self.model.project

    def forward(self,*values,frequency_hz=None,block_size=32):
        from .plane_execution import plane_reservation
        values=_host_design(self.project,values,self.dispersive)
        reservation=plane_reservation(self.model,tuple(tuple(v.shape) for v in values),
            self.options,frequency_hz,block_size=block_size)
        planes=self.model(*values,frequency_hz=frequency_hz,block_size=block_size)
        for plane in planes.values():
            plane.report.update(unified_execution='streamed',host_input_interface=True,
                                execution_reservation=reservation)
        return planes


class _ResidentFromHost(torch.nn.Module):
    def __init__(self,project,policy,dispersive,quadrature_counts=None):
        super().__init__()
        from .plane_execution import plane_mode, plane_model
        self.project=_resident_project(project,policy.resident)
        self.policy,self.dispersive=policy,dispersive
        self.planes=plane_mode(self.project)
        self.quadrature_counts=quadrature_counts
        if self.planes:
            self.model=plane_model(self.project,policy.resident,dispersive,quadrature_counts)
            return
        model=DifferentiableSimulation
        if dispersive:
            from .dispersive_adjoint import DispersiveSimulation
            model=DispersiveSimulation
        self.model=model(self.project,policy.resident)

    def _run(self,values,frequency_hz=None,window=None,block_size=32):
        values=_host_design(self.project,values,self.dispersive)
        shapes=tuple(tuple(v.shape) for v in values)
        reservation=_resident_reservation(self.project,shapes,self.policy,frequency_hz,window,block_size,
            quadrature_counts=self.quadrature_counts,plane=self.model if self.planes else None)
        # Do not detach: copies are part of the user's geometry-to-loss graph.
        copied=tuple(v.to(self.policy.device) for v in values)
        if self.planes:
            result=self.model(*copied,frequency_hz=frequency_hz,block_size=block_size)
            output={}
            for key,plane in result.items():
                output[key]=replace(plane,fields=plane.fields.cpu(),frequency_hz=plane.frequency_hz.cpu(),
                    points_um=plane.points_um.cpu(),weights=plane.weights.cpu())
                plane.report.update(unified_execution='resident',host_input_interface=True,
                                    execution_reservation=reservation)
            return output
        if frequency_hz is None:
            result=self.model(*copied)
            result=replace(result,signals=result.signals.cpu())
        else:
            result=self.model.spectrum(*copied,frequency_hz,window=window,block_size=block_size)
            result=replace(result,fields=result.fields.cpu(),frequency_hz=result.frequency_hz.cpu())
        result.report.update(unified_execution='resident',host_input_interface=True,
                             execution_reservation=reservation)
        return result

    def forward(self,*values,frequency_hz=None,block_size=32):return self._run(values,frequency_hz,block_size=block_size)

    def spectrum(self,*values,frequency_hz=None,window=None,block_size=32):
        if frequency_hz is None:
            if len(values)!=(5 if self.dispersive else 2):raise ValueError('Provide materials followed by frequency_hz.')
            values,frequency_hz=values[:-1],values[-1]
        return self._run(values,frequency_hz,window,block_size)


class _ExecutionWorkload:
    def __init__(self,workload):self.workload=workload

    def __getattr__(self,name):return getattr(self.workload,name)

    def reservation(self,policy):
        if policy.recorded is not None:
            raise ValueError('Recorded CPML policies are explicit and are not supported by execution tuning.')
        from .plane_execution import PlaneTuningWorkload
        if isinstance(self.workload,PlaneTuningWorkload):return self.workload.execution_reservation(policy)
        if policy.streamed is not None:return self.workload.reservation(_streamed_options(policy))
        spectral=self.workload.spectral
        return _resident_reservation(self.workload.project,
            tuple(tuple(v.shape) for v in self.workload.design),policy,
            None if spectral is None else spectral.frequency,
            None if spectral is None else spectral.window)

    def model(self,project,policy):
        from .plane_execution import PlaneTuningWorkload
        if isinstance(self.workload,PlaneTuningWorkload):return self.workload.execution_model(project,policy)
        return policy.simulation(project,dispersive=isinstance(self.workload,_DispersiveTuningWorkload))


@dataclass(frozen=True)
class AdjointExecutionSelection:
    policy: AdjointExecutionPolicy
    report: dict
    dispersive: bool
    quadrature_counts: dict | None = None

    def simulation(self,project):return self.policy.simulation(project,dispersive=self.dispersive,quadrature_counts=self.quadrature_counts)


def tune_adjoint_execution(project,epsilon,*material_parameters,options=None,candidates=None,
        frequency_hz=None,window=None,quadrature_counts=None,probe_steps=24,repeats=2,max_calibration_steps=512,
        refine_candidates=0,reference_cache_bytes=64*1024**2):
    """Compare resident and streamed full-iteration cost under shared budgets.

    Supply epsilon only, or epsilon_inf, strength, omega0 and gamma for ADE.
    Defaults combine generated streamed policies with device/host resident
    checkpoints. Input and result tensors remain on CPU with intact gradients.
    All-field projects return fixed spectral planes from model(...,
    frequency_hz=...). Optional quadrature_counts maps monitor IDs to transverse
    point counts. Mixed point/plane projects and plane windows are unsupported.
    Tuning uses detached leaves and a proxy energy objective, never modifies
    caller .grad, and does not establish a globally optimal execution policy.
    File banks require explicit streamed directory, budget and headroom settings.
    """
    if len(material_parameters) not in (0,3):raise ValueError('Supply epsilon or four ADE material inputs.')
    _validate_tuning_arguments(probe_steps,repeats,'replay_cost',max_calibration_steps,
                               refine_candidates,reference_cache_bytes)
    if candidates is None and options is not None and not isinstance(options,StreamedAdjointOptions):
        raise ValueError('Default execution proposals require StreamedAdjointOptions budgets.')
    if candidates is not None:
        candidates=tuple(candidates)
        if any(isinstance(p,AdjointExecutionPolicy) and p.recorded is not None for p in candidates):
            raise ValueError('Recorded CPML policies are explicit and are not supported by execution tuning.')
    from .plane_execution import plane_mode, PlaneTuningWorkload
    planes=plane_mode(project)
    quadrature_counts=None if quadrature_counts is None else dict(quadrature_counts)
    if planes:
        workload=PlaneTuningWorkload(project,epsilon,material_parameters,frequency_hz,window,quadrature_counts)
    else:
        if quadrature_counts is not None:raise ValueError('Quadrature counts require fixed field planes.')
        workload=(_DispersiveTuningWorkload(project,epsilon,*material_parameters,frequency_hz,window)
                  if material_parameters else _TuningWorkload(project,epsilon,frequency_hz,window))
    planning=[]
    if candidates is None:
        base=options or StreamedAdjointOptions()
        streamed,planning=_generated_candidates(project,workload,base,reference_cache_bytes,max_calibration_steps)
        device=base.device
        kernel='fused' if torch.device(device).type=='cuda' else 'torch'
        resident=AdjointOptions(checkpoints=base.checkpoints,backward_kernel=kernel,
            gpu_budget_bytes=base.gpu_budget_bytes,host_budget_bytes=base.host_budget_bytes,
            resident_budget_bytes=base.gpu_budget_bytes if torch.device(device).type=='cuda' else base.host_budget_bytes)
        residents=[resident,replace(resident,checkpoints=0)]
        if torch.device(device).type=='cuda':
            residents.append(replace(resident,storage='host',checkpoint_transfers='async'))
        candidates=[AdjointExecutionPolicy(resident=p,device=device,host_budget_bytes=base.host_budget_bytes)
                    for p in dict.fromkeys(residents)]
        candidates.extend(AdjointExecutionPolicy(streamed=p,device=device,host_budget_bytes=base.host_budget_bytes)
                          for p in streamed)
    tuned=_tune_streamed(project,_ExecutionWorkload(workload),candidates=candidates,
        probe_steps=probe_steps,repeats=repeats,max_calibration_steps=max_calibration_steps,
        refine_candidates=refine_candidates,reference_cache_bytes=reference_cache_bytes,
        _policy_type=AdjointExecutionPolicy)
    tuned.report.update(default_candidate_planning=planning,
        scope='Measured resident and streamed prefixes with checkpoint-replay extrapolation. CPU-to-device design and CPU-result copies are included for resident CUDA. Full-duration optimality, geometry/optimizer cost, microbatch budgeting, multi-GPU and sustained physical-storage performance are not established.')
    if planes:
        tuned.report.update(observation='fixed_plane_spectrum_and_flux',quadrature_counts=quadrature_counts,
            proxy_objective='Sum of squared duration-normalized complex plane fields and area-normalized signed flux.')
    return AdjointExecutionSelection(tuned.options,tuned.report,bool(material_parameters),quadrature_counts)
