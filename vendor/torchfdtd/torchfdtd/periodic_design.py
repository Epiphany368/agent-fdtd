"""Shared Python/UI workflow for single-frequency periodic density design."""
from copy import deepcopy
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from .adjoint_batch import AdjointBatchOptions
from .differentiable import AdjointOptions
from .execution_tuning import AdjointExecutionPolicy
from .periodic_adjoint import PeriodicLayerResponse
from .reference_cache import PlaneReferenceCache
from .reversible_cpml import ReversibleCPMLOptions
from .streamed import StreamedAdjointOptions


class PeriodicDesignConfig(BaseModel):
    """Independent periodic layer, not a conversion of an arbitrary CAD scene.

    Density axes are x,y. The fixed wavelength objective maximizes a weighted
    sum of R/G2/G1/B quadrant powers averaged over two polarizations. The
    density remains continuous. Physical convergence and fabrication are
    separate from this projected Adam workflow.
    """
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, revalidate_instances='always')
    name: str = Field(default='Periodic density design', min_length=1, max_length=120)
    wavelength_um: float = Field(default=.5, gt=0)
    period_um: tuple[float, float] = (.8, .8)
    height_um: float = Field(default=.2, gt=0)
    detector_offset_um: float = Field(default=.5, gt=0)
    background_index: float = Field(default=1.4, ge=1)
    design_index: float = Field(default=1.8, ge=1)
    theta_deg: float = Field(default=0, ge=0, lt=90)
    phi_deg: float = Field(default=0, ge=-360, le=360)
    mesh_um: float = Field(default=.1, gt=0)
    steps: int = Field(default=160, ge=1, strict=True)
    pml_cells: int = Field(default=6, ge=1, strict=True)
    quadrature_counts: tuple[int, int] = (24, 24)
    initial_density: list[list[float]] = Field(default_factory=lambda:[[.5]*16 for _ in range(16)])
    objective_weights: tuple[float, float, float, float] = (1., 0., 0., 0.)
    iterations: int = Field(default=5, ge=1, le=10000, strict=True)
    learning_rate: float = Field(default=.01, gt=0, lt=1)
    device: Literal['cpu', 'cuda'] = 'cpu'
    execution: Literal['auto', 'resident', 'dram', 'file', 'recorded'] = 'auto'
    gpu_budget_gib: float = Field(default=4, gt=0)
    host_budget_gib: float = Field(default=8, gt=0)
    checkpoints: int = Field(default=2, ge=0, strict=True)
    slab_width: int = Field(default=64, ge=1, strict=True)
    temporal_depth: int = Field(default=8, ge=1, strict=True)
    state_directory: str | None = None
    disk_budget_gib: float | None = Field(default=None, gt=0)
    disk_free_reserve_gib: float = Field(default=100, ge=0)
    recorded_trace_storage: Literal['cpu', 'device'] = 'cpu'
    recorded_trace_chunk_steps: int = Field(default=32, ge=1, le=1024, strict=True)
    recorded_collar_cells: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode='after')
    def valid(self):
        if any(v <= 0 for v in self.period_um):
            raise ValueError('Period lengths must be positive.')
        if any(v < 2 for v in self.quadrature_counts):
            raise ValueError('Use at least two detector quadrature points per axis.')
        rows = self.initial_density
        if (not rows or not rows[0] or len(rows)*len(rows[0]) > 1024**2
                or any(len(row) != len(rows[0]) for row in rows)):
            raise ValueError('Density must be rectangular, nonempty and at most 1,048,576 pixels.')
        if any(not 0 <= v <= 1 for row in rows for v in row):
            raise ValueError('Initial density must lie in [0,1].')
        if not any(self.objective_weights):
            raise ValueError('At least one objective weight must be nonzero.')
        if self.execution == 'file' and not self.state_directory:
            raise ValueError('File execution requires a state directory and disk budget.')
        if (self.state_directory is None) != (self.disk_budget_gib is None):
            raise ValueError('Supply both a state directory and disk budget, or neither.')
        if self.state_directory is not None and not self.state_directory.strip():
            raise ValueError('State directory cannot be empty.')
        return self

    def spec(self):
        return dict(wavelength_um=self.wavelength_um,period_um=self.period_um,
            height_um=self.height_um,detector_offset_um=self.detector_offset_um,
            background_index=self.background_index,design_index=self.design_index,
            theta_inside_rad=math.radians(self.theta_deg),phi_rad=math.radians(self.phi_deg))


def _prepare(config):
    shape = (len(config.initial_density), len(config.initial_density[0]))
    # Parameter, Adam/gradient temporaries and current/best/result tensor copies.
    # JSON/Python configuration and progress records are outside this allowance.
    optimizer_bytes = 16*math.prod(shape)*4
    host = int(config.host_budget_gib*1024**3)-optimizer_bytes
    gpu = int(config.gpu_budget_gib*1024**3)
    if host < 1 or gpu < 1:
        raise ValueError('Memory budget cannot hold the design/optimizer allowance.')
    disk = None if config.disk_budget_gib is None else int(config.disk_budget_gib*1024**3)
    settings = dict(density_shape=shape,mesh=config.mesh_um,steps=config.steps,
        pml_cells=config.pml_cells,quadrature_counts=config.quadrature_counts,
        reference_cache=PlaneReferenceCache(),dtype=torch.float32)
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable on the connected solver.')
    if config.execution == 'auto':
        model = PeriodicLayerResponse.auto(config.spec(),**settings,device=config.device,
            gpu_budget_bytes=gpu,host_budget_bytes=host,checkpoints=config.checkpoints,
            max_slab_width=config.slab_width,temporal_depth=config.temporal_depth,
            state_directory=config.state_directory,disk_budget_bytes=disk,
            disk_free_reserve_bytes=int(config.disk_free_reserve_gib*1024**3))
    else:
        if config.execution == 'recorded':
            policy = AdjointExecutionPolicy(device=config.device,host_budget_bytes=host,
                recorded=ReversibleCPMLOptions(
                    trace_storage=config.recorded_trace_storage,
                    trace_transfers='async' if config.device=='cuda' and config.recorded_trace_storage=='cpu' else 'sync',
                    trace_chunk_steps=config.recorded_trace_chunk_steps,
                    collar_cells=config.recorded_collar_cells,
                    gpu_budget_bytes=gpu,host_budget_bytes=host,
                    resident_budget_bytes=gpu if config.device=='cuda' else host))
        elif config.execution == 'resident':
            policy = AdjointExecutionPolicy(device=config.device,host_budget_bytes=host,
                resident=AdjointOptions(checkpoints=config.checkpoints,
                    gpu_budget_bytes=gpu,host_budget_bytes=host,
                    resident_budget_bytes=gpu if config.device=='cuda' else host,
                    backward_kernel='fused' if config.device=='cuda' else 'torch'))
        else:
            policy = AdjointExecutionPolicy(device=config.device,host_budget_bytes=host,
                streamed=StreamedAdjointOptions(device=config.device,checkpoints=config.checkpoints,
                    slab_width=config.slab_width,temporal_depth=config.temporal_depth,
                    host_budget_bytes=host,gpu_budget_bytes=gpu,
                    tile_transfers='async' if config.device=='cuda' else 'sync',
                    state_storage='disk' if config.execution=='file' else 'host',
                    state_directory=config.state_directory,disk_budget_bytes=disk,
                    disk_free_reserve_bytes=int(config.disk_free_reserve_gib*1024**3)))
        model = PeriodicLayerResponse(config.spec(),**settings,policy=policy,
            batch_options=AdjointBatchOptions(host_budget_bytes=host,gpu_budget_bytes=gpu,
                disk_budget_bytes=disk))
    plan = model.plan()
    plan.update(optimizer_allowance_bytes=optimizer_bytes,
        total_host_reservation_bytes=plan['host_reservation_bytes']+optimizer_bytes,
        selection=model.selection_report,execution=config.execution,device=config.device,
        precision='float32',calibration_solves=0)
    return model, plan


def periodic_design_plan(config):
    """Admit the design without allocating fields or running trial solves."""
    config = PeriodicDesignConfig.model_validate(config).model_copy(deep=True)
    return _prepare(config)[1]


def run_periodic_design(config, *, on_progress=None, cancel=None):
    """Run projected Adam and return JSON-compatible evaluated designs.

    Uses the same PeriodicLayerResponse as custom Torch optimization loops.
    Cancellation is checked between forward and backward solves. It cannot
    interrupt an in-flight solve. No graph history is retained across updates.
    This function does not implement optimizer restart or binary projection.
    """
    config = PeriodicDesignConfig.model_validate(config).model_copy(deep=True)
    model, plan = _prepare(config)
    density = torch.nn.Parameter(torch.tensor(config.initial_density,dtype=torch.float32))
    weights = density.new_tensor(config.objective_weights)
    optimizer = torch.optim.Adam([density],lr=config.learning_rate,foreach=False)
    history, best, last = [], None, None
    updates, stopped = 0, False
    def cancelled():return cancel is not None and cancel.is_set()
    def emit(stage):
        if on_progress:
            preview = None if last is None else last['density']
            if preview is not None:
                sx,sy = max(1,math.ceil(len(preview)/128)),max(1,math.ceil(len(preview[0])/128))
                preview = [row[::sy] for row in preview[::sx]]
            on_progress(dict(stage=stage,updates_completed=updates,total_updates=config.iterations,
                history=deepcopy(history),density_preview=preview,plan=plan))
    for index in range(config.iterations+1):
        if cancelled():stopped=True;break
        optimizer.zero_grad(set_to_none=True)
        emit('forward')
        with torch.set_grad_enabled(index < config.iterations):
            response = model(density)
            score = (response.mean(0)*weights).sum()
        if not bool(torch.isfinite(score)):
            raise RuntimeError('Nonfinite design objective.')
        last = dict(update=updates,objective=float(score.detach()),
            density=density.detach().tolist(),response=response.detach().tolist())
        history.append(dict(update=updates,objective=last['objective']))
        if best is None or last['objective'] > best['objective']:best=deepcopy(last)
        emit('evaluated')
        if cancelled():stopped=True;break
        if index == config.iterations:break
        emit('backward')
        (-score).backward()
        if not bool(torch.isfinite(density.grad).all()):
            raise RuntimeError('Nonfinite density gradient.')
        history[-1]['gradient_l2'] = float(density.grad.norm())
        if cancelled():stopped=True;break
        optimizer.step()
        with torch.no_grad():density.clamp_(0,1)
        updates += 1
        del response, score
    emit('cancelled' if stopped else 'completed')
    return dict(status='cancelled' if stopped else 'completed',config=config.model_dump(mode='json'),
        plan=plan,updates_completed=updates,history=history,best=best,last_evaluated=last,
        pending_density=density.detach().tolist() if last is None or last['update'] != updates else None,
        scope='Continuous single-frequency periodic density optimization. Physical convergence, fabrication and arbitrary CAD conversion are not established.')
