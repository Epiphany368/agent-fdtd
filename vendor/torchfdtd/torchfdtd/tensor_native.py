"""Native result/progress execution for the admitted full-tensor Torch solver."""
import math
import time

import numpy as np
import torch


def _device(region):
    use_cuda = region.backend == 'cuda' or (region.backend == 'auto' and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Select CPU for native tensor execution.')
    return 'cuda' if use_cuda else 'cpu'


def _output_reservation(project):
    r = project.region
    interval = max(r.snapshot_interval, math.ceil(r.steps/100))
    axis = 'xyz'.index(r.slice_axis)
    plane = math.prod(math.ceil(n/max(1, math.ceil(n/256))) for a, n in enumerate(r.shape) if a != axis)
    item = 8 if r.complex_fields else 4
    monitors = sum(m.enabled for m in project.monitors)
    return 6*math.prod(r.shape)*item + r.steps*monitors*item + 4*plane*(2*math.ceil(r.steps/interval)+3)


def estimate_tensor(project):
    """Admit metadata, solver working tensors and native outputs before rasterization."""
    from .tensor_project import tensor_from_project, validate_tensor_project
    from .mesh import mesh_summary
    from .memory_profile import host_memory
    validate_tensor_project(project)
    r = project.region
    device = _device(r)
    adapter = tensor_from_project(project, device=device, checkpoints=0)
    reservation = adapter.plan()['memory']
    outputs = _output_reservation(project)
    host = reservation['host_reservation_bytes'] + outputs
    available = host_memory()['available_bytes']
    if available is not None and host > .8*available:
        raise ValueError('Native tensor outputs and solver exceed available host memory.')
    return dict(mesh_summary(project), shape=r.shape, actual_size_um=r.actual_size,
        cells=math.prod(r.shape), dt_fs=r.time_step*1e15, duration_fs=r.steps*r.time_step*1e15,
        estimated_memory_mb=(reservation['memory_reservation_bytes']+outputs)/2**20,
        tensor_solver_reservation=reservation, tensor_native_output_bytes=outputs,
        tensor_native_host_reservation_bytes=host,
        warnings=['Full-tensor native execution uses Torch operations, not fused kernels. '
                  'Material preview is the plotted component diagonal at common nodes, not scalar epsilon.'],
        oneway_planes=[], tfsf_boxes=[], tfsf_auxiliary_estimated_bytes=0)


def run_tensor(project, progress=None, cancel=None, **graph_options):
    """Run one fixed native tensor scene without retaining a timestep autograd graph."""
    from .tensor_project import tensor_from_project
    from .anisotropy import _TensorSystem
    from .solver import Result, field_axes, index_at
    from .run_control import DecayDecision, source_end_time
    started = time.perf_counter()
    stats = estimate_tensor(project)
    device = _device(project.region)
    adapter = tensor_from_project(project, device=device, checkpoints=0,
        tensor_budget_bytes=stats['tensor_solver_reservation']['memory_reservation_bytes'])
    p = adapter.project
    r = p.region
    with torch.no_grad():
        epsilon = adapter.rasterize()
        adapter.validate_material(epsilon)
        system = _TensorSystem(p.model_copy(deep=True), epsilon)
    axis = 'xyz'.index(r.slice_axis)
    component = 'xyz'.index(r.field[1].lower())
    location = tuple(r.slice_position if a == axis else 0 for a in range(3))
    slice_index = index_at(location, r, r.field)[axis]
    def plane(value):
        data = value.select(axis, slice_index)
        data = data[::max(1, math.ceil(data.shape[0]/256)), ::max(1, math.ceil(data.shape[1]/256))]
        return data.detach().cpu().numpy().copy()
    epsilon_plane = plane(epsilon[..., component, component])
    trace = torch.empty((r.steps, len(system.monitors)), dtype=system.field_dtype, device=device)
    interval = max(r.snapshot_interval, math.ceil(r.steps/100))
    frames, frame_steps = [], []
    completed, reason = 0, 'max_steps'
    decision = DecayDecision(r.run_control, source_end_time(p), r.time_step)
    if device == 'cuda':
        torch.cuda.synchronize(system.device)
    setup = time.perf_counter()-started
    begin = time.perf_counter()
    with torch.no_grad():
        for step in range(r.steps):
            if cancel is not None and cancel.is_set():
                reason = 'cancelled'
                break
            system.advance(step, step+1)
            completed = step+1
            trace[step] = system.observe(system.state())
            if completed % r.run_control.check_interval == 0 or completed == r.steps:
                state = system.state()
                if any(not bool(torch.isfinite(value).all()) for value in state):
                    raise FloatingPointError('Non-finite tensor field or CPML auxiliary state detected.')
                # A diagnostic state norm, not a tensor-energy conservation claim.
                norm = sum(float(value.abs().double().square().sum()) for value in state)
                peak = max(float(value.abs().max()) for value in state[:2])
                decision.update(completed, norm, peak)
            if completed % interval == 0 or completed == r.steps:
                field = system.grid.E if r.field[0] == 'E' else system.grid.H
                image = plane(field[..., component])
                image = {'real': np.real, 'imag': np.imag, 'magnitude': np.abs, 'phase': np.angle}[r.complex_display](image)
                # Own only the displayed FP32 plane, not a complex backing array.
                image = np.array(image, dtype=np.float32, copy=True)
                frames.append(image)
                frame_steps.append(completed)
                if progress:
                    progress(dict(step=completed, total=r.steps, frame=image.tolist(),
                        elapsed=time.perf_counter()-begin, termination_reason=None,
                        diagnostics=decision.history[-1] if decision.history else None))
        if any(not bool(torch.isfinite(value).all()) for value in system.state()):
            raise FloatingPointError('Non-finite tensor field or CPML auxiliary state detected.')
    if device == 'cuda':
        torch.cuda.synchronize(system.device)
    seconds = time.perf_counter()-begin
    stats.update(engine='TorchFDTD full-tensor dielectric', backend=device, precision='float32',
        gpu=torch.cuda.get_device_name(system.device) if device == 'cuda' else None,
        steps=completed, requested_steps=r.steps, seconds=seconds, setup_seconds=setup,
        cancelled=reason == 'cancelled', termination_reason=reason, auto_shutoff=False,
        cuda_graph=False, cuda_graph_steps=1, cuda_graph_replays=0,
        cuda_kernel='torch', cuda_monitor_kernel='torch',
        field_peak=float(system.grid.E.abs().max()),
        mcells_per_second=math.prod(r.shape)*completed/max(seconds, 1e-9)/1e6,
        slice_index=slice_index, slice_position=float(field_axes(r, r.field)[axis][slice_index]),
        complex_fields=r.complex_fields, complex_display=r.complex_display,
        material_update='nondispersive full tensor', material_sampling='common mesh nodes',
        epsilon_preview=f'epsilon_{r.field[1].lower()}{r.field[1].lower()} at common nodes',
        boundaries=r.boundaries.model_dump(), diagnostics=decision.history,
        diagnostic_backend='Torch full-field and CPML auxiliary state norm',
        tensor_plan=adapter.plan(), units='geometry: um; time: s; E/H: reduced fields')
    return Result(p, stats, np.asarray(frames, dtype=np.float32).reshape(-1, *epsilon_plane.shape),
        np.asarray(frame_steps, dtype=np.int64), epsilon_plane,
        trace[:completed].detach().cpu().numpy(), np.arange(1, completed+1)*r.time_step,
        system.grid.E.detach().cpu().numpy().copy(), system.grid.H.detach().cpu().numpy().copy())
