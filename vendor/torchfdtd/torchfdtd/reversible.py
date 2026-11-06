"""Experimental lossless periodic FDTD with terminal-state reconstruction.

This is an explicit alternative to checkpointed differentiation. It supports
confined isotropic Lorentz density mixtures and does not invert absorbing states. Floating-point reconstruction is
approximate, and its drift diagnostic is not a gradient-error bound.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import threading
import time

import torch

from .differentiable import DifferentiableResult, _System
from .models import Project


@dataclass(frozen=True)
class ReversibleOptions:
    """Total solver budgets and a fail-closed reconstruction drift threshold."""

    gpu_budget_bytes: int | None = None
    host_budget_bytes: int | None = None
    resident_budget_bytes: int | None = None
    reconstruction_tolerance: float = 1e-3
    offload_terminal: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        if type(self.offload_terminal) is not bool:
            raise ValueError('offload_terminal must be a boolean.')
        for name in ('gpu_budget_bytes', 'host_budget_bytes', 'resident_budget_bytes'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f'{name} must be a positive integer byte budget.')
        value = self.reconstruction_tolerance
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1e-3:
            raise ValueError('reconstruction_tolerance must be finite and in (0, 1e-3].')


def _validate_project(project):
    r = project.region
    recommendation = ' Use checkpointed DifferentiableSimulation for other supported physics.'
    if (r.dimension != '3d' or r.precision != 'float32' or r.mesh_type != 'uniform'
            or r.mesh_steps is not None or r.material_sampling != 'yee'
            or r.interface_method != 'staircase' or r.run_control.auto_shutoff
            or r.memory_mode == 'streamed'):
        raise ValueError('ReversibleSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.' + recommendation)
    if any(f.kind != 'periodic' for a in range(3) for f in r.boundaries.pair(a)) or any(r.bloch_phase):
        raise ValueError('ReversibleSimulation requires all six boundaries to be periodic, without Bloch phase.' + recommendation)
    if any(m.model != 'dielectric' or m.oscillators for m in project.materials):
        raise ValueError('ReversibleSimulation supports only nondispersive dielectric material declarations.' + recommendation)
    for raw in project.sources:
        s = project.resolved_source(raw)
        if s.enabled and (s.kind != 'point' or s.injection != 'soft'
                          or any(not name.startswith('E') for name, _ in s.polarization_components)):
            raise ValueError('ReversibleSimulation supports fixed soft electric point sources only.' + recommendation)
    monitors = [m for m in project.monitors if m.enabled]
    if not monitors or any(m.kind != 'point' or m.time_downsample != 1 for m in monitors):
        raise ValueError('ReversibleSimulation requires point E/H monitors sampled every timestep.' + recommendation)


def _field_scale(values):
    """Bounded FP64 diagnostic reductions, never full-volume FP64 fields."""
    square = torch.zeros((), device=values[0].device, dtype=torch.float64)
    peak = torch.zeros((), device=values[0].device, dtype=torch.float32)
    for value in values:
        flat = value.reshape(-1)
        for start in range(0, flat.numel(), 65536):
            block = flat[start:start + 65536]
            peak = torch.maximum(peak, block.abs().amax())
            converted = block.to(torch.float64)
            square.add_(converted.square().sum())
            del converted
    maximum, norm = float(peak), math.sqrt(float(square))
    if not math.isfinite(maximum) or not math.isfinite(norm):
        raise RuntimeError('Reversible fields became nonfinite. Use a stable checkpointed configuration.')
    return maximum, norm


def _undo_sources(system, field, family, step):
    for loc, component, wave, profile in reversed(system.sources[family]):
        amplitude = wave[step] if profile is None else wave[step] * profile
        field[loc + (component,)] -= amplitude


def _inverse_torch(system, step):
    e, h = system.state()
    _undo_sources(system, h, 'H', step)
    curl, _ = system.curl(e, (), True)
    h.add_(system.grid.courant_number * curl)
    del curl
    _undo_sources(system, e, 'E', step)
    curl, _ = system.curl(h, (), False)
    e.sub_(system.grid.courant_number / system.eps4 * curl)


class _Reversible(torch.autograd.Function):
    @staticmethod
    def forward(ctx, epsilon, project, options, report):
        # The system never owns an autograd-connected input. save_for_backward
        # below separately preserves the input's normal mutation/version check.
        system = _System(project, epsilon.detach(), prepare_kernels=False)
        if epsilon.is_cuda:
            from .cuda_kernels import FusedYeeCUDA
            system.kernel = FusedYeeCUDA(system.grid, direct_views=True)
        signals = epsilon.new_empty((project.region.steps, len(system.monitors)))
        maximum = norm = 0.
        started = time.perf_counter()
        for step in range(project.region.steps):
            system.advance(step, step + 1)
            signals[step] = system.observe(system.state())
            if (step + 1) % 64 == 0 or step + 1 == project.region.steps:
                current_maximum, current_norm = _field_scale(system.state())
                maximum = max(maximum, current_maximum)
                norm = max(norm, current_norm)
        terminal = tuple(value.clone() for value in system.state())
        ctx.save_for_backward(epsilon, *terminal)
        ctx.system, ctx.options, ctx.report = system, options, report
        ctx.lock = threading.Lock()
        ctx.scale = maximum, norm
        if epsilon.is_cuda:
            torch.cuda.synchronize(epsilon.device)
        report.update(forward_seconds=time.perf_counter() - started,
                      sampled_forward_peak=maximum, sampled_forward_l2=norm,
                      diagnostic_cadence_steps=64, terminal_copies=1,
                      checkpoint_replays=0, backward_calls=0)
        return signals

    @staticmethod
    def backward(ctx, signal_bar):
        if torch.is_grad_enabled():
            raise RuntimeError('ReversibleSimulation supports first derivatives only.')
        if not ctx.lock.acquire(blocking=False):
            raise RuntimeError('Concurrent backward calls on one reversible result are not supported.')
        try:
            epsilon, *terminal = ctx.saved_tensors
            system, report = ctx.system, ctx.report
            started = time.perf_counter()
            for target, value in zip(system.state(), terminal):
                target.copy_(value)
            gradient = torch.zeros_like(epsilon)
            fused = inverse = inverse_grid = None
            if epsilon.is_cuda:
                from .cuda_kernels import FusedYeeCUDA
                from .cuda_adjoint import FusedAdjointCUDA
                inverse_grid = copy.copy(system.grid)
                inverse_grid.courant_number = -system.grid.courant_number
                inverse = FusedYeeCUDA(inverse_grid, direct_views=True)
                fused = FusedAdjointCUDA(system, gradient, signal_bar, direct_views=True)
            bars = None if fused is not None else tuple(torch.zeros_like(x) for x in terminal)
            for step in range(system.region.steps - 1, -1, -1):
                if inverse is not None:
                    _undo_sources(system, system.grid.H, 'H', step)
                    inverse.update_H()
                    _undo_sources(system, system.grid.E, 'E', step)
                    inverse.update_E()
                    fused.step(step)
                else:
                    _inverse_torch(system, step)
                    bars, part = system.transpose_step(system.state(), bars, signal_bar[step])
                    gradient.add_(part)
                    del part
            absolute, l2 = _field_scale(system.state())
            forward_peak, forward_l2 = ctx.scale
            relative_peak = absolute / forward_peak if forward_peak else (0. if absolute == 0 else math.inf)
            relative_l2 = l2 / forward_l2 if forward_l2 else (0. if l2 == 0 else math.inf)
            # A final residual is a drift alarm, not a proof that every objective
            # has a small gradient error. Never silently switch algorithms.
            diagnostics = dict(initial_max_abs=absolute, initial_l2=l2,
                               initial_relative_peak=relative_peak,
                               initial_relative_l2=relative_l2,
                               inverse_steps=system.region.steps,
                               transpose_steps=system.region.steps)
            report['last_backward'] = diagnostics
            if max(relative_peak, relative_l2) > ctx.options.reconstruction_tolerance:
                raise RuntimeError('Reversible reconstruction drift exceeds the declared tolerance. Rerun with checkpointed DifferentiableSimulation.')
            # Bounded blocks also check the returned material derivative.
            for start in range(0, gradient.numel(), 65536):
                if not bool(torch.isfinite(gradient.reshape(-1)[start:start + 65536]).all()):
                    raise RuntimeError('Reversible material gradient is nonfinite.')
            if epsilon.is_cuda:
                torch.cuda.synchronize(epsilon.device)
            report['backward_calls'] += 1
            diagnostics['seconds'] = time.perf_counter() - started
            return gradient, None, None, None
        finally:
            ctx.lock.release()


class ReversibleSimulation(torch.nn.Module):
    """Opt-in FP32 scalar, lossless periodic FDTD and first material VJP.

    Input is a contiguous (Nx, Ny, Nz) epsilon tensor. Impressed electric
    increments are fixed independently of epsilon, including source cells.
    All material cells receive their actual derivative. Constrain design cells
    explicitly with torch.where when a source or exterior must remain fixed.
    Field storage is O(cells). Point/source histories remain O(steps).
    Selecting ``material`` changes the input to scalar density rho in [0,1],
    with a fixed scalar/per-z background supplied as ``fixed_epsilon``.
    """

    def __init__(self, project: Project, options: ReversibleOptions | None = None, *, material=None):
        super().__init__()
        self.project = Project.model_validate(project.model_dump())
        self.options = options or ReversibleOptions()
        from .reversible_lorentz import resolve_material, proxy_project
        if not isinstance(self.options, ReversibleOptions):
            raise ValueError('ReversibleSimulation requires ReversibleOptions.')
        from .boundaries import reject_pmc_faces
        reject_pmc_faces(self.project.region, 'ReversibleSimulation')
        _validate_project(proxy_project(self.project) if any(m.oscillators for m in self.project.materials) else self.project)
        self.material = resolve_material(self.project, material)
        if self.material is None and self.options.offload_terminal:
            raise ValueError('Terminal offload is currently available for reversible Lorentz density mixtures only.')
        from .adjoint_memory import _resident_contract
        _resident_contract(self.project.region, self.options)

    def plan(self, *, device='cpu', density_layers=None):
        """Metadata-only admission, repeated before each execution."""
        project = self._snapshot()
        if self.material is not None:
            from .reversible_lorentz import resolve_material, reservation
            material = resolve_material(project, self.material)
            shape = (*project.region.shape[:2], project.region.shape[2] if density_layers is None else density_layers)
            return reservation(project, self.options, torch.device(device), (0, project.region.shape[2]-1),
                               material, density_shape=shape, periodic=True)[0]
        from .reversible_memory import _reversible_reservation
        return _reversible_reservation(project, self.options, torch.device(device))

    def _snapshot(self):
        if not isinstance(self.options, ReversibleOptions):
            raise ValueError('ReversibleSimulation requires ReversibleOptions.')
        # Pydantic objects are mutable. Revalidate numeric/CFL settings as well
        # as categorical scope, then use this same snapshot throughout a call.
        project = Project.model_validate(self.project.model_dump())
        from .reversible_lorentz import resolve_material, proxy_project
        _validate_project(proxy_project(project) if any(m.oscillators for m in project.materials) else project)
        material = resolve_material(project, self.material)
        return project

    def forward(self, epsilon, *, fixed_epsilon=None):
        project = self._snapshot()
        if self.material is not None:
            from .reversible_lorentz import resolve_material, execute_density
            material = resolve_material(project, self.material)
            background = epsilon.new_tensor(1.) if fixed_epsilon is None and isinstance(epsilon, torch.Tensor) else fixed_epsilon
            return execute_density(epsilon, background, project, self.options,
                (0, project.region.shape[2]-1), material, periodic=True)
        if fixed_epsilon is not None:
            raise ValueError('fixed_epsilon is used only for the explicit Lorentz density mixture.')
        if (not isinstance(epsilon, torch.Tensor) or epsilon.dtype != torch.float32
                or epsilon.device.type not in ('cpu', 'cuda') or epsilon.layout != torch.strided
                or not epsilon.is_contiguous() or epsilon.is_conj() or epsilon.is_neg()
                or tuple(epsilon.shape) != project.region.shape):
            raise ValueError('epsilon must be a contiguous resolved scalar FP32 CPU/CUDA tensor matching the region shape.')
        from .reversible_memory import _reversible_reservation
        reservation = _reversible_reservation(project, self.options, epsilon.device)
        flattened = epsilon.reshape(-1)
        for start in range(0, flattened.numel(), 65536):
            block = flattened[start:start + 65536]
            if not bool(torch.isfinite(block).all()) or bool((block < 1).any()):
                raise ValueError('The reversible CFL contract requires finite epsilon >= 1.')
        if epsilon.is_cuda:
            from .cuda_bootstrap import prepare_cuda_kernels
            prepare_cuda_kernels()
        report = dict(experimental=True, adjoint='lossless periodic reconstruction',
                      higher_order=False, full_time_autograd=False, spatial_streaming=False,
                      checkpoint_capacity=0,
                      reconstruction_tolerance=self.options.reconstruction_tolerance,
                      material_input_retained_bytes=epsilon.numel() * epsilon.element_size(),
                      source_material_gradient='unrestricted fixed impressed increments',
                      backend='fused CUDA' if epsilon.is_cuda else 'torch CPU', **reservation)
        signals = _Reversible.apply(epsilon, project, self.options, report)
        return DifferentiableResult(signals, project.region.time_step,
                                    tuple(m.component for m in project.monitors if m.enabled), report)
