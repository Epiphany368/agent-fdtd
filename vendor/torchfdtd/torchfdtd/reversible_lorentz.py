"""Isotropic density mixtures and local reversible trapezoidal pole state."""
from __future__ import annotations

import math
from dataclasses import replace
import threading
import time
import numpy as np
import torch

from .models import Material, Project
from .reversible_cpml_kernels import _local_curl


def pole_coefficients(material, dt, steps):
    values = []
    for rate, strength, gamma in material.oscillators:
        square = (rate*dt)**2
        plus = 1 + .5*gamma*dt + .25*square
        minus = 1 - .5*gamma*dt + .25*square
        if minus <= 0 or not math.isfinite(plus/minus):
            raise ValueError('The damped Lorentz update has no safely invertible trapezoidal pole map. Use checkpointed DispersiveSimulation.')
        if steps*math.log(plus/minus) > math.log(1000):
            raise ValueError('Lorentz damping amplifies inverse roundoff beyond the reversible horizon. Use checkpointed DispersiveSimulation.')
        row = tuple(float(np.float32(x)) for x in
                    (.5*square, 1/plus, strength*dt*dt/(4*plus), 1/minus, strength*dt*dt/(4*minus)))
        if not all(math.isfinite(x) for x in row):
            raise ValueError('Pole coefficients exceed FP32 range at this timestep.')
        values.append(row)
    return tuple(values)


def resolve_material(project, material=None):
    """Freeze a selected declaration, without changing dielectric callers."""
    if isinstance(material, str):
        found = [m for m in project.materials if m.name == material]
        if not found:
            raise ValueError(f'Unknown reversible density material {material!r}.')
        material = found[0]
    if material is None:
        found = [m for m in project.materials if m.oscillators]
        if not found:
            return None
        if len(found) != 1:
            raise ValueError('Select one isotropic oscillator material explicitly for the reversible density mixture.')
        material = found[0]
    if not isinstance(material, Material):
        raise ValueError('material must be a Material or a declared material name.')
    material = Material.model_validate(material.model_dump())
    if material.model == 'tensor' or not material.oscillators:
        raise ValueError('Reversible density mixtures require an isotropic Lorentz/Drude pole declaration.')
    if project.region.complex_fields:
        raise ValueError('Reversible Lorentz density mixtures currently require real fields; use checkpointed DispersiveSimulation for Bloch fields.')
    if material.fit_dt_s is not None and not math.isclose(material.fit_dt_s, project.region.time_step, rel_tol=1e-12, abs_tol=0):
        raise ValueError('The discrete material fit belongs to another timestep. Refit at region.time_step.')
    return material


def proxy_project(project):
    """Scene assignments do not replace the explicit density/pole inputs."""
    project = Project.model_validate(project.model_dump())
    project.materials = [Material(name=m.name, index=math.sqrt(m.instantaneous_epsilon), color=m.color)
                         if m.oscillators else m.model_copy(deep=True) for m in project.materials]
    return project


def validate_density(rho, shape, interval):
    if (not isinstance(rho, torch.Tensor) or rho.dtype != torch.float32
            or rho.device.type not in ('cpu', 'cuda') or rho.layout != torch.strided
            or rho.ndim != 3 or not rho.is_contiguous() or rho.is_conj() or rho.is_neg()
            or tuple(rho.shape[:2]) != tuple(shape[:2])):
        raise ValueError('Density must be a contiguous scalar FP32 CPU/CUDA (Nx,Ny,M) tensor; anisotropic dispersive maps are unsupported.')
    depth = interval[1] - interval[0] + 1
    if rho.shape[2] != shape[2] and (rho.shape[2] <= 0 or depth % rho.shape[2]):
        raise ValueError('Density z samples must match the whole grid or evenly divide the reconstruction interval.')
    flat = rho.reshape(-1)
    for start in range(0, flat.numel(), 65536):
        block = flat[start:start+65536]
        if not bool(torch.isfinite(block).all()) or bool(((block < 0) | (block > 1)).any()):
            raise ValueError('Density must be finite and lie in [0,1].')


def background_z(background, shape, device):
    if (not isinstance(background, torch.Tensor) or background.dtype != torch.float32
            or background.layout != torch.strided or background.requires_grad
            or background.device.type not in ('cpu', 'cuda') or background.is_conj() or background.is_neg()):
        raise ValueError('Fixed background must be a real FP32 tensor without gradients.')
    if background.device != device and background.device.type != 'cpu':
        raise ValueError('Fixed background must use the density device or CPU.')
    if background.ndim == 0:
        value = background.expand(shape[2])
    elif tuple(background.shape) == (shape[2],):
        value = background
    elif tuple(background.shape) == tuple(shape):
        value = background[0, 0, :]
        for start in range(0, shape[0], max(1, 65536//max(1, shape[1]*shape[2]))):
            block = background[start:start+max(1, 65536//max(1, shape[1]*shape[2]))]
            if not torch.equal(block, value[None, None, :].expand_as(block)):
                raise ValueError('Dispersive fixed background must be laterally uniform; z-dependent scalar backgrounds are supported.')
    else:
        raise ValueError('Fixed background must be scalar, (Nz,) or an isotropic (Nx,Ny,Nz) tensor.')
    if not bool(torch.isfinite(value).all()) or bool((value < 1).any()):
        raise ValueError('The reversible CFL contract requires fixed background epsilon >= 1.')
    return value.detach().to(device).contiguous()


class DensityMaterial:
    """Compact rho view and packed (P0,Q0,P1,Q1,...) per component."""

    def __init__(self, rho, background, project, interval, material):
        # Conditioning checks precede even the compact rho/background copies.
        self.constants = pole_coefficients(material, project.region.time_step, project.region.steps)
        self.shape = tuple(project.region.shape)
        self.a, self.b = interval
        self.depth = self.b - self.a + 1
        self.full_input = rho.shape[2] == self.shape[2]
        self.rho = (rho[:, :, self.a:self.b+1].detach().contiguous() if self.full_input else rho.detach())
        self.layers = self.rho.shape[2]
        self.cells_per_layer = self.depth // self.layers
        self.background = background_z(background, self.shape, rho.device)
        self.inverse_background = self.background.reciprocal()
        self.epsilon_inf = float(np.float32(material.instantaneous_epsilon))
        self.delta_epsilon = (torch.tensor(self.epsilon_inf, dtype=torch.float32, device=rho.device) - self.background).contiguous()
        self.material = material
        self.dt = project.region.time_step
        self.pole_count = len(self.constants)
        self.padded_poles = 2*((self.pole_count+1)//2)
        self.state = None


    def allocate(self):
        self.state = torch.zeros((*self.shape[:2], self.depth, 3, 2*self.padded_poles),
                                 device=self.rho.device, dtype=torch.float32)

    def dense_rho(self, value=None):
        value = self.rho if value is None else (value[:, :, self.a:self.b+1] if self.full_input else value)
        return value if self.cells_per_layer == 1 else value.repeat_interleave(self.cells_per_layer, 2)

    def coefficients(self, inverse=False, rho=None):
        density = self.dense_rho(rho)
        bg = self.background[self.a:self.b+1][None, None, :]
        difference = self.delta_epsilon[self.a:self.b+1][None, None, :]
        epsilon = bg + density*difference
        ks = [density*values[4 if inverse else 2] for values in self.constants]
        total = ks[0]
        for k in ks[1:]:
            total = total+k
        return epsilon-total, (epsilon+total).reciprocal(), ks

    def state_views(self, state=None):
        state = self.state if state is None else state
        return [(state[..., 2*i], state[..., 2*i+1]) for i in range(self.pole_count)]

    def reduce_gradient(self, value, original):
        result = value.sum(3)
        if self.cells_per_layer > 1:
            result = result.reshape(*self.shape[:2], self.layers, self.cells_per_layer).sum(3)
        if self.full_input:
            full = torch.zeros_like(original)
            full[:, :, self.a:self.b+1] = result
            return full
        return result.contiguous()


def forward_electric(system, mat, *, state=None, rho=None):
    """Functional CPU reference, also usable with a local Torch graph."""
    e, h, *psis = system.state() if state is None else state
    curl, psis = system.curl(h, tuple(psis), False)
    cn = system.grid.courant_number
    new = e + (cn*mat.inverse_background)[None, None, :, None]*curl
    old = e[:, :, mat.a:mat.b+1]
    pqs = mat.state_views()
    responses = [(q-values[0]*p)*values[1] for (p, q), values in zip(pqs, mat.constants)]
    response = responses[0]
    for part in responses[1:]:
        response = response+part
    em, iv, ks = mat.coefficients(rho=rho)
    next_e = (em[..., None]*old + cn*curl[:, :, mat.a:mat.b+1] - response)*iv[..., None]
    next_state = mat.state.clone()
    for i, ((p, q), response, k) in enumerate(zip(pqs, responses, ks)):
        delta = response+k[..., None]*(next_e+old)
        next_state[..., 2*i] = p+delta
        next_state[..., 2*i+1] = -q+2*delta
    new = new.clone()
    new[:, :, mat.a:mat.b+1] = next_e
    return new, next_state, tuple(psis)


def inverse_electric(system, mat, *, periodic=False):
    """Reconstruct E and every packed local pole after undoing its sources."""
    if periodic:
        curl, _ = system.curl(system.grid.H, (), False)
        curl = curl[:, :, mat.a:mat.b+1]
    else:
        curl = _local_curl(system.grid.H, mat.a, mat.b, False, system.grid.wrap)
    next_e = system.grid.E[:, :, mat.a:mat.b+1]
    pqs = mat.state_views()
    us = [(q+values[0]*p)*values[3] for (p, q), values in zip(pqs, mat.constants)]
    total = us[0]
    for u in us[1:]:
        total = total+u
    em, iv, ks = mat.coefficients(inverse=True)
    old = (em[..., None]*next_e-system.grid.courant_number*curl+total)*iv[..., None]
    for (p, q), u, k in zip(pqs, us, ks):
        delta = u-k[..., None]*(next_e+old)
        p.sub_(delta)
        q.copy_(2*delta-q)
    system.grid.E[:, :, mat.a:mat.b+1].copy_(old)
    return old


def advance_cpu(system, mat, step, frame=None):
    """Native E/source/frame/H/source order with local pole state."""
    e, h, *_ = system.state()
    next_e, next_poles, psis = forward_electric(system, mat)
    next_e = system.inject(next_e, 'E', step, functional=True)
    if frame is not None:
        frame[0].copy_(next_e[:, :, mat.b+1, :2])
        frame[1].copy_(h[:, :, mat.a-1, :2])
    curl, psis = system.curl(next_e, psis, True)
    next_h = system.inject(h-system.grid.courant_number*curl, 'H', step, functional=True)
    for target, value in zip(system.state(), (next_e, next_h, *psis)):
        target.copy_(value)
    mat.state.copy_(next_poles)
    system.current_step = step+1


class CPUReconstruction:
    """Full CPML field transpose and local ADE reconstruction/transpose.

    E cotangents carry the next pole coupling before observation/H transpose,
    matching the packed CUDA reference's ordering. Exterior primal fields and
    CPML memory are not inverted or consumed by the material derivative.
    """

    def __init__(self, system, mat, gradient, signal_bar, *, periodic=False):
        self.system, self.mat, self.gradient = system, mat, gradient
        self.signal_bar = signal_bar.contiguous()
        self.periodic = periodic
        self.next_step = system.region.steps-1
        self.e_bar = torch.zeros_like(system.grid.E)
        self.h_bar = torch.zeros_like(system.grid.H)
        self.psi_bars = tuple(torch.zeros_like(s['psi']) for s in system.segments)
        self.pole_bars = torch.zeros_like(mat.state)
        # Outside the design the reference transpose uses cn / fixed epsilon.
        self.scale = (system.grid.courant_number/mat.background)[None, None, :, None].expand_as(system.grid.E).clone()
        _, iv, _ = mat.coefficients()
        self.scale[:, :, mat.a:mat.b+1] = (system.grid.courant_number*iv)[..., None]

    def observe(self, index):
        for target, (positions, indices) in zip((self.e_bar, self.h_bar), self.system.observation_maps):
            if indices.numel():
                target.reshape(-1).index_add_(0, indices, self.signal_bar[index].index_select(0, positions))

    def step(self, step, frame=None, *, observation_index=None):
        from .reversible import _undo_sources
        system, mat = self.system, self.mat
        if type(step) is not int or step != self.next_step or step < 0:
            raise ValueError('Lorentz reconstruction requires descending steps exactly once.')
        row = step if observation_index is None else observation_index
        if type(row) is not int or not 0 <= row < self.signal_bar.shape[0]:
            raise ValueError('Lorentz observation row lies outside the seed block.')
        if not self.periodic and (frame is None or tuple(frame.shape) != (2,*mat.shape[:2],2)):
            raise ValueError('CPML Lorentz reconstruction requires a complete boundary frame.')
        a, b = mat.a, mat.b
        e, h = system.grid.E, system.grid.H
        if frame is not None:
            e[:, :, b+1, :2].copy_(frame[0])
        _undo_sources(system, h, 'H', step)
        if self.periodic:
            curl, _ = system.curl(e, (), True)
            h.add_(system.grid.courant_number*curl)
        else:
            curl = _local_curl(e, a, b, True, system.grid.wrap)
            h[:, :, a:b+1].add_(system.grid.courant_number*curl)
            h[:, :, a-1, :2].copy_(frame[1])
        _undo_sources(system, e, 'E', step)
        next_e = e[:, :, a:b+1].clone()
        old = inverse_electric(system, mat, periodic=self.periodic)
        self.observe(step if observation_index is None else observation_index)
        contribution, self.psi_bars = system.curl_transpose(-system.grid.courant_number*self.h_bar,
                                                          self.psi_bars, True)
        self.e_bar.add_(contribution)
        contribution, self.psi_bars = system.curl_transpose(self.scale*self.e_bar,
                                                          self.psi_bars, False)
        self.h_bar.add_(contribution)
        em, iv, ks = mat.coefficients()
        bars = mat.state_views(self.pole_bars)
        deltas = [pb+2*qb for pb, qb in bars]
        common = ks[0][..., None]*deltas[0]
        for k, delta in zip(ks[1:], deltas[1:]):
            common = common+k[..., None]*delta
        nb = self.e_bar[:, :, a:b+1]
        numerator_bar = nb*iv[..., None]
        denominator_bar = -(nb*next_e)*iv[..., None]
        old_bar = common+em[..., None]*numerator_bar
        epsilon_bar = numerator_bar*old+denominator_bar
        part = epsilon_bar*mat.delta_epsilon[a:b+1][None, None, :, None]
        updated = []
        for i, ((pb, qb), delta, values) in enumerate(zip(bars, deltas, mat.constants)):
            k_bar = delta*(next_e+old)-numerator_bar*old+denominator_bar
            response_bar = delta-numerator_bar
            new_pb = pb-values[0]*response_bar*values[1]
            new_qb = -qb+response_bar*values[1]
            part = part+k_bar*values[2]
            updated.append((new_pb, new_qb))
        self.gradient.add_(part)
        next_common = ks[0][..., None]*(updated[0][0]+2*updated[0][1])
        for k, (pb, qb) in zip(ks[1:], updated[1:]):
            next_common = next_common+k[..., None]*(pb+2*qb)
        self.e_bar[:, :, a:b+1].copy_(old_bar+next_common)
        for (pb, qb), (new_pb, new_qb) in zip(bars, updated):
            pb.copy_(new_pb)
            qb.copy_(new_qb)
        self.next_step -= 1


def reservation(project, options, device, interval, material, *, density_shape=None,
                spectral=None, periodic=False):
    """Metadata-only complete totals, including padded pole state and scratch."""
    from .memory_profile import host_memory
    from .cuda_memory import cuda_budget_limit
    device = torch.device(device)
    pole_coefficients(material, project.region.time_step, project.region.steps)
    if project.region.complex_fields:
        raise ValueError('Reversible Lorentz mixtures require real fields.')
    if (getattr(options, 'forward_kernel', 'split') != 'split'
            or getattr(options, 'adjoint_kernel', 'split') != 'split'):
        raise ValueError('Lorentz reconstruction requires split electric/magnetic and field-adjoint kernels.')
    cells = math.prod(project.region.shape)
    plane = math.prod(project.region.shape[:2])
    depth = interval[1]-interval[0]+1
    inside = plane*depth
    count = len(material.oscillators)
    padded = 2*((count+1)//2)
    poles = inside*3*2*padded*4
    shape = project.region.shape if density_shape is None else tuple(density_shape)
    if len(shape) != 3 or shape[:2] != project.region.shape[:2]:
        raise ValueError('Density planning shape must be (Nx,Ny,M).')
    layers = depth if shape[2] == project.region.shape[2] else shape[2]
    if type(layers) is not int or layers <= 0 or depth % layers:
        raise ValueError('Density z samples must evenly divide the reconstructed interval.')
    owned_density = 4*inside if shape[2] == project.region.shape[2] else 0
    maps = 12*project.region.shape[2]+4*plane*layers
    gradient = 12*inside
    tuning = 2*poles+gradient if (getattr(options, 'cells_per_thread', 1) == 'auto'
        or getattr(options, 'block_size', None) == 'auto') else 0
    # CPU functional updates and transpose use bounded volume temporaries.
    # CUDA kernels use the packed banks in place; the scalar scale table is
    # retained as a conservative allowance even when generated reads avoid it.
    scratch = (6*poles+4*(count+8)*inside+12*cells if device.type == 'cpu' else 4*inside)
    offload = getattr(options, 'offload_terminal', False)
    offload_used = offload and device.type == 'cuda'
    extra = dict(pole_state=poles, pole_terminal=0 if offload_used else poles,
                 pole_adjoint=poles, density_copy=owned_density,
                 density_coefficient_views=maps, density_gradient=gradient,
                 pole_update_scratch=scratch, pole_tuning_scratch=tuning)
    extra_active = sum(extra.values())
    extra_host = poles+24*inside+owned_density+maps if offload and device.type == 'cuda' else 0
    attempts = []
    if not periodic and getattr(options, 'trace_storage', 'device') == 'auto':
        candidates = [('device', 'sync'), ('cpu', 'async')] if device.type == 'cuda' else [('cpu', 'sync')]
    else:
        candidates = [(getattr(options, 'trace_storage', 'device'), getattr(options, 'trace_transfers', 'sync'))]
    for storage, transfers in candidates:
        try:
            effective = options if periodic else replace(options, trace_storage=storage, trace_transfers=transfers)
            if periodic:
                from .reversible_memory import _reversible_reservation
                base = _reversible_reservation(project, effective, device)
            else:
                from .reversible_cpml_memory import _cpml_reversible_reservation
                base = _cpml_reversible_reservation(project, effective, device, interval, spectral=spectral)
            headroom = (extra_active+19)//20+4096 if device.type == 'cuda' else 0
            host = base['host_reservation_bytes']+extra_host
            active = base['memory_reservation_bytes']+extra_active+headroom
            if device.type == 'cpu':
                host += extra_active
                active = host
            if options.host_budget_bytes is not None and host > options.host_budget_bytes:
                raise ValueError('Lorentz complete host reservation exceeds the explicit total budget.')
            if options.resident_budget_bytes is not None and active > options.resident_budget_bytes:
                raise ValueError('Lorentz complete resident reservation exceeds the explicit total budget.')
            available = host_memory()['available_bytes']
            if available is not None and host > int(.8*available):
                raise ValueError('Lorentz complete reservation exceeds available host memory.')
            if device.type == 'cuda' and active > cuda_budget_limit(device, active, options.gpu_budget_bytes):
                raise ValueError('Lorentz complete CUDA reservation exceeds the device budget.')
            parts = dict(base['workspace_components_bytes'])
            parts.update(extra)
            attempts.append(dict(storage=storage,transfers=transfers,admitted=True))
            result = dict(base, memory_reservation_bytes=active, host_reservation_bytes=host,
                gpu_reservation_bytes=active if device.type == 'cuda' else 0,
                workspace_reservation_bytes=base['workspace_reservation_bytes']+extra_active,
                allocation_headroom_bytes=base['allocation_headroom_bytes']+headroom,
                workspace_components_bytes=parts, pole_state_bytes=poles,
                pole_terminal_bytes=poles, pole_count=count, padded_pole_count=padded,
                pole_offload_host_bytes=extra_host, offload_terminal=offload,
                density_input_shape=list(shape), density_layer_count=layers,
                density_cells_per_layer=depth//layers, density_input_parameterization='linear isotropic mixture',
                workspace_model='reversible_lorentz_'+base['workspace_model'])
            if not periodic:
                result.update(trace_storage_requested=options.trace_storage,
                    trace_storage=storage, trace_transfers=transfers,
                    trace_placement_attempts=attempts)
            return result, effective
        except ValueError as error:
            attempts.append(dict(storage=storage, transfers=transfers, admitted=False, reason=str(error)))
            if periodic or getattr(options, 'trace_storage', None) != 'auto':
                raise
    raise ValueError('No complete Lorentz trace placement fits: '+str(attempts))


def _system(project, rho, mat, spectral, options=None, report=None):
    from .differentiable import _System
    proxy = rho.new_ones(()).expand(project.region.shape)
    system = _System(proxy_project(project), proxy, prepare_kernels=False,
        prepare_permittivity=False,
        observation_monitors=None if spectral is None else spectral.observers,
        observation_cache=getattr(spectral, 'observation_cache', None))
    system.grid.inverse_permittivity = rho.new_ones(1)
    mat.allocate()
    if rho.is_cuda:
        from .reversible_lorentz_cuda import LorentzYeeCUDA
        system.kernel = LorentzYeeCUDA(system.grid, mat, options=options, report=report)
    return system


def _advance(system, mat, step, frame):
    if system.kernel is None:
        advance_cpu(system, mat, step, frame)
    else:
        e, h = system.grid.E, system.grid.H
        system.kernel.update_E()
        system.inject(e, 'E', step)
        if frame is not None:
            frame[0].copy_(e[:, :, mat.b+1, :2])
            frame[1].copy_(h[:, :, mat.a-1, :2])
        system.kernel.update_H()
        system.inject(h, 'H', step)
        system.current_step = step+1


def _scale(system, mat):
    from .reversible_cpml import _interior_scale
    from .reversible import _field_scale
    peak, norm = _interior_scale(system, mat.a, mat.b)
    ppeak, pnorm = _field_scale((mat.state,))
    return max(peak, ppeak), math.sqrt(norm*norm+pnorm*pnorm)


def _forward(rho, background, project, options, interval, material, report, spectral, recorded):
    mat = DensityMaterial(rho, background, project, interval, material)
    system = _system(project, rho, mat, spectral, options, report)
    steps = project.region.steps
    periodic = report['periodic']
    transport = None
    archive = None
    if recorded and not periodic:
        shape = (steps, 2, *project.region.shape[:2], 2)
        if options.trace_transfers == 'async':
            from .reversible_trace import AsyncBoundaryTrace
            transport = AsyncBoundaryTrace(shape, rho.device, chunk_steps=report['trace_chunk_steps'])
            archive = transport.archive
        else:
            archive = torch.empty(shape, dtype=rho.dtype,
                device=rho.device if options.trace_storage == 'device' else torch.device('cpu'))
    block = steps if spectral is None else spectral.block_size
    samples = rho.new_empty((block, len(system.monitors)))
    signals = samples if spectral is None else spectral.zeros()
    maximum = norm = 0.0
    start = time.perf_counter()
    try:
        for step in range(steps):
            frame = (None if archive is None else transport.frame(step) if transport is not None else archive[step])
            _advance(system, mat, step, frame)
            if transport is not None:
                transport.commit(step)
            row = step % block
            samples[row] = system.observe(system.state())
            if spectral is not None and (row+1 == block or step+1 == steps):
                spectral.accumulate(signals, samples[:row+1], step-row)
            if recorded and ((step+1) % 64 == 0 or step+1 == steps):
                peak, total = _scale(system, mat)
                maximum, norm = max(maximum, peak), max(norm, total)
        if transport is not None:
            transport.finish()
        from .reversible_cpml import _require_finite
        _require_finite(signals, 'Lorentz observations became nonfinite.', 65536)
        _require_finite(mat.state, 'Lorentz pole state became nonfinite.', 65536)
        if archive is not None:
            _require_finite(archive, 'Lorentz boundary trace became nonfinite.', 65536)
    except BaseException as error:
        if transport is not None:
            transport.__exit__(type(error), error, error.__traceback__)
        raise
    report.update(forward_seconds=time.perf_counter()-start,
        sampled_forward_peak=maximum, sampled_forward_l2=norm,
        backward_calls=0, checkpoint_replays=0, observation_history_retained=spectral is None,
        boundary_pole_state='implicit zero outside the confined density interval',
        pole_terminal_required=True, forward_only=not recorded)
    return signals, system, mat, archive, transport, (maximum, norm)


class _RecordedLorentz(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, background, project, options, interval, material, report, spectral):
        signals, system, mat, archive, transport, scale = _forward(
            rho, background, project, options, interval, material, report, spectral, True)
        terminal = [v[:, :, mat.a:mat.b+1] for v in system.state()[:2]]
        terminal.append(mat.state)
        offload = options.offload_terminal and rho.is_cuda
        if offload:
            copies = [torch.empty(t.shape, dtype=t.dtype, device='cpu', pin_memory=True).copy_(t, non_blocking=True)
                      for t in terminal]
            torch.cuda.current_stream(rho.device).synchronize()
            terminal = copies
            system.kernel = None
            system.grid.E = system.grid.H = None
            mat.state = None
        else:
            terminal = [v.clone() for v in terminal]
        saved = [rho, background, *terminal]
        if archive is not None:
            saved.append(archive)
        ctx.save_for_backward(*saved)
        ctx.system, ctx.mat, ctx.transport, ctx.options = system, mat, transport, options
        ctx.report, ctx.spectral, ctx.scale = report, spectral, scale
        ctx.lock = threading.Lock()
        ctx.has_archive, ctx.offload = archive is not None, offload
        report['terminal_offload_used'] = offload
        report['terminal_state_count'] = 1
        if rho.is_cuda:
            torch.cuda.synchronize(rho.device)
        return signals

    @staticmethod
    def backward(ctx, signal_bar):
        if torch.is_grad_enabled():
            raise RuntimeError('Reversible Lorentz mixtures support first derivatives only.')
        if not ctx.lock.acquire(blocking=False):
            raise RuntimeError('Concurrent backward calls on one reversible Lorentz result are unsupported.')
        try:
            rho, background, te, th, poles, *archives = ctx.saved_tensors
            system, mat, report = ctx.system, ctx.mat, ctx.report
            a, b = mat.a, mat.b
            if system.grid.E is None:
                system.grid.E = rho.new_empty((*system.region.shape, 3))
                system.grid.H = torch.empty_like(system.grid.E)
                mat.allocate()
            for target, terminal in zip(system.state()[:2], (te, th)):
                target.fill_(float('nan'))
                target[:, :, a:b+1].copy_(terminal.to(rho.device, non_blocking=ctx.offload))
            mat.state.copy_(poles.to(rho.device, non_blocking=ctx.offload))
            gradient = rho.new_zeros((*system.region.shape[:2], mat.depth, 3))
            spectral = ctx.spectral
            seed = signal_bar.contiguous() if spectral is None else rho.new_empty(
                (spectral.block_size, len(system.monitors)))
            if rho.is_cuda:
                from .reversible_lorentz_cuda import LorentzReconstructionCUDA
                inverse = LorentzReconstructionCUDA(system, mat, gradient, seed,
                    periodic=report['periodic'], options=ctx.options, report=report)
            else:
                inverse = CPUReconstruction(system, mat, gradient, seed, periodic=report['periodic'])
            archive = archives[0] if ctx.has_archive else None
            staging = (rho.new_empty(archive.shape[1:]) if archive is not None
                and ctx.transport is None and archive.device != rho.device else None)
            from contextlib import nullcontext
            reader = ctx.transport.reverse() if ctx.transport is not None else nullcontext(archive)
            current, blocks = -1, 0
            start = time.perf_counter()
            with reader as frames:
                for step in range(system.region.steps-1, -1, -1):
                    frame = None if archive is None else frames[step]
                    if staging is not None:
                        staging.copy_(frame)
                        frame = staging
                    if spectral is None:
                        inverse.step(step, frame)
                    else:
                        begin = step//spectral.block_size*spectral.block_size
                        if begin != current:
                            value = spectral.transpose(signal_bar, begin, min(begin+spectral.block_size, system.region.steps))
                            seed[:len(value)].copy_(value)
                            current, blocks = begin, blocks+1
                        inverse.step(step, frame, observation_index=step-begin)
            residual_peak, residual_norm = _scale(system, mat)
            peak, norm = ctx.scale
            relative_peak = residual_peak/peak if peak else (0.0 if residual_peak == 0 else math.inf)
            relative_norm = residual_norm/norm if norm else (0.0 if residual_norm == 0 else math.inf)
            report['last_backward'] = dict(initial_relative_peak=relative_peak,
                initial_relative_l2=relative_norm, pole_initial_max_abs=float(mat.state.abs().max()),
                inverse_steps=system.region.steps, regenerated_seed_blocks=blocks,
                seconds=time.perf_counter()-start)
            if max(relative_peak, relative_norm) > ctx.options.reconstruction_tolerance:
                raise RuntimeError('Lorentz reconstruction drift exceeds the declared tolerance. Use checkpointed DispersiveSimulation.')
            from .reversible_cpml import _require_finite
            _require_finite(gradient, 'Lorentz density gradient became nonfinite.', 65536)
            result = mat.reduce_gradient(gradient, rho)
            report['backward_calls'] += 1
            if ctx.offload:
                del inverse
                system.grid.E = system.grid.H = None
                mat.state = None
            return result, None, None, None, None, None, None, None
        finally:
            ctx.lock.release()


def execute_density(rho, background, project, options, interval, material,
                    spectral=None, *, periodic=False):
    from .differentiable import DifferentiableResult
    # Resolve the full admission before effective maps, fields or pole banks.
    if not isinstance(rho, torch.Tensor):
        raise ValueError('Density must be a Tensor.')
    admitted, effective = reservation(proxy_project(project), options, rho.device, interval,
        material, density_shape=tuple(rho.shape), spectral=spectral, periodic=periodic)
    validate_density(rho, project.region.shape, interval)
    if not isinstance(background, torch.Tensor):
        background = torch.as_tensor(background, dtype=torch.float32, device='cpu')
    # Numerical background checks precede system allocation. The tiny per-z
    # view is recreated inside the immutable material binding after admission.
    background_z(background, project.region.shape, rho.device)
    report = dict(admitted, experimental=True, adjoint='reversible trapezoidal Lorentz density mixture',
        periodic=periodic, higher_order=False, full_time_autograd=False, spatial_streaming=False,
        reconstruction_interval_z=list(interval), reconstruction_tolerance=options.reconstruction_tolerance,
        material_gradient_scope='scalar density in the explicit interval',
        source_material_gradient='fixed impressed increments', cpml_primal_inverted=False,
        material_declaration=material.model_dump(mode='json', exclude={'samples'}),
        backend='packed fused CUDA' if rho.is_cuda else 'torch CPU')
    recorded = ((torch.is_grad_enabled() and rho.requires_grad)
                or getattr(effective, 'forward_only', 'auto') == 'never')
    if recorded:
        signals = _RecordedLorentz.apply(rho, background, project, effective, interval, material, report, spectral)
    else:
        with torch.no_grad():
            signals, _, _, _, _, _ = _forward(rho, background, project, effective, interval, material, report, spectral, False)
    if spectral is not None:
        return spectral.result(signals, report)
    return DifferentiableResult(signals, project.region.time_step,
        tuple(m.component for m in project.monitors if m.enabled), report)
