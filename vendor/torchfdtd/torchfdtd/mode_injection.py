"""Discrete single-frequency modal Huygens sheets on the native Yee adjoint.

Uniform 3D meshes, periodic transverse axes, longitudinal CPML, real isotropic
fixed launch materials only. Eigensystem and source profiles are fixed inputs.
"""
from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import numpy as np
import torch
from .mode_ports import solve_waveguide_modes, _amplitudes, _uniform_cell_quadrature, _sample_permittivity_component, C0
from .adjoint_planes import DifferentiablePlaneSimulation
from .differentiable import DifferentiableSimulation, _System
from .injection import oneway_plan
from .streamed import StreamedSimulation, _StreamedExecution, _reservation
from .waveforms import source_time_signal


def _profile(value, normal):
    w='xyz'.index(normal)
    return value[None].transpose(np.argsort((w,(w+1)%3,(w+2)%3)))


@dataclass(frozen=True)
class ModalLaunch:
    mode: object
    epsilon: np.ndarray
    region_signature: str
    source_signature: str
    normal_origin_um: float
    normal_step_um: float
    electric_index: int
    direction: int
    beta_tilde_per_um: float
    temporal_k_per_um: float
    terms: tuple
    identity: str

    def detector_mode(self, coordinate_um):
        q=(coordinate_um-self.normal_origin_um)/self.normal_step_um
        if not math.isclose(q,round(q),abs_tol=2e-5,rel_tol=0):
            raise ValueError('Validated modal detectors must lie on longitudinal E-node planes.')
        fields=self.mode.fields.copy()
        w='xyz'.index(self.mode.normal)
        factor=math.cos(self.mode.beta_per_um*self.normal_step_um/2)
        fields[...,w]*=factor
        for axis in ((w+1)%3,(w+2)%3):fields[...,3+axis]*=factor
        fields.setflags(write=False)
        return replace(self.mode,fields=fields)

    @property
    def beta_yee_spacetime_per_um(self):
        return float(np.real(self.mode.beta_per_um))

    @property
    def storage_bytes(self):
        return self.epsilon.nbytes+self.mode.fields.nbytes+sum(w.nbytes+p.nbytes for _,_,w,p in self.terms)


@dataclass(frozen=True)
class ApertureModalLaunch(ModalLaunch):
    """A periodic-supercell mode on a transverse sub-rectangle of the cell.

    aperture holds (begin, end) Yee cell index ranges in cyclic transverse
    order. Injection, the fixed-material check and the detector quadrature are
    restricted to that rectangle. The profile is a fixed periodic eigenmode of
    the aperture, so its tails must be confined inside it; edge_energy_fraction
    records the share of squared field amplitude in the outermost cell ring.
    """
    aperture: tuple
    edge_energy_fraction: float


def _check_launch_signatures(project,launch,epsilon):
    region=project.region
    if json.dumps(region.model_dump(mode='json'),sort_keys=True)!=launch.region_signature:
        raise ValueError('Modal launch region changed. Rebuild its fixed tables.')
    active=[project.resolved_source(s) for s in project.sources if s.enabled]
    if len(active)!=1 or json.dumps(active[0].model_dump(mode='json'),sort_keys=True)!=launch.source_signature:
        raise ValueError('Modal launch source changed. Rebuild its fixed tables.')
    if not isinstance(epsilon,torch.Tensor):raise ValueError('Modal epsilon must be a Torch tensor.')
    if epsilon.shape!=region.shape+(3,):raise ValueError('Modal propagation requires explicit Yee diagonal epsilon.')


def _frozen_launch_material(launch,epsilon):
    """Check the fixed source neighbourhood/collars and detach their derivatives."""
    w='xyz'.index(launch.mode.normal)
    selection=[slice(None)]*3
    selection[w]=slice(launch.electric_index-1,launch.electric_index+2)
    for axis,(begin,end) in zip(((w+1)%3,(w+2)%3),getattr(launch,'aperture',((None,None),(None,None)))):
        selection[axis]=slice(begin,end)
    selection=tuple(selection)
    expected=torch.tensor(np.stack([_profile(launch.epsilon[...,c],launch.mode.normal) for c in range(3)],axis=-1),
                          dtype=epsilon.dtype,device=epsilon.device)
    if not torch.allclose(epsilon[selection],expected.expand_as(epsilon[selection]),rtol=2e-6,atol=1e-7):
        raise ValueError('Injection-neighborhood epsilon must match the fixed modal cross-section.')
    fixed_transverse=[]
    for axis,begin,end in getattr(launch,'fixed_transverse_slices',()):
        slab=[slice(None)]*3
        slab[axis]=slice(begin,end)
        slab=tuple(slab)
        if not torch.allclose(epsilon[slab],epsilon.new_tensor(launch.cladding_epsilon),rtol=2e-6,atol=1e-7):
            raise ValueError('Transverse CPML and its fixed collar must retain the modal cladding epsilon.')
        fixed_transverse.append(slab)
    # Keep source-neighborhood material derivatives excluded, since its fixed
    # electric sheet contains inverse epsilon and the eigenmode is frozen.
    if epsilon.requires_grad:
        epsilon=epsilon.clone()
        epsilon[selection]=epsilon[selection].detach()
        for slab in fixed_transverse:
            epsilon[slab]=epsilon[slab].detach()
    return epsilon,selection,tuple(fixed_transverse)


def prepare_modal_launch(project, permittivity, *, mode_index=0, num_modes=2, source_budget_bytes=256*1024**2):
    """Solve a temporally/longitudinally discrete mode and prepare fixed currents.

    The sole project source is a Gaussian plane carrier with cycle-based timing.
    Its normal/direction/center/size/time settings define the launch. The spatial
    Cartesian polarization is replaced by the selected full-vector mode.
    """
    r=project.region
    active=[project.resolved_source(s) for s in project.sources if s.enabled]
    if len(active)!=1:raise ValueError('Modal launch requires exactly one enabled source.')
    s=active[0]
    if r.mesh_type!='uniform' or r.complex_fields or r.material_sampling!='yee':
        raise ValueError('Modal launch requires a real uniform Yee-sampled region.')
    if r.dimension == '2d':
        from .invariant_modes import validate_invariant_launch
        validate_invariant_launch(r, s)
    if r.interface_method!='staircase' or any(m.oscillators for m in project.materials):
        raise ValueError('Only real nondispersive staircase materials are supported.')
    if s.kind!='plane' or s.injection!='soft' or s.pulse!='gaussian' or s.time_definition!='cycles':
        raise ValueError('Use one soft Gaussian plane source with cycle-based timing.')
    if s.pulse_cycles<2:raise ValueError('Validated modal pulses require at least two carrier cycles.')
    w,k,d=oneway_plan(s,r)
    u,v=(w+1)%3,(w+2)%3
    if not math.isclose(s.center[w],r.mesh_nodes[w][k],abs_tol=1e-7,rel_tol=0):
        raise ValueError('The source must lie on a longitudinal E-node plane.')
    shape=(r.shape[u],r.shape[v])
    dtype=np.float32 if r.precision=='float32' else np.float64
    # Admission precedes source tables. Sparse eigensystem memory is separate.
    required=(8*r.steps+40*math.prod(shape))*np.dtype(dtype).itemsize
    if required>source_budget_bytes:raise ValueError('Modal source tables exceed the source byte budget.')
    dt=r.time_step
    omega=2*np.pi*C0/(s.wavelength*1e-6)
    kt=2*np.sin(omega*dt/2)/(C0*dt)*1e-6
    mode_solver = solve_waveguide_modes
    if r.dimension == '2d':
        from .invariant_modes import solve_invariant_waveguide_modes
        mode_solver = solve_invariant_waveguide_modes
    modes=mode_solver(permittivity,shape=shape,spacing_um=(r.axis_steps[u],r.axis_steps[v]),
        origin_um=(r.mesh_nodes[u][0],r.mesh_nodes[v][0]),wavelength_um=2*np.pi/kt,
        normal=s.normal,num_modes=num_modes,precision=r.precision)
    if not isinstance(mode_index,int) or not 0<=mode_index<len(modes):raise ValueError('Invalid mode index.')
    base=modes[mode_index]
    h=r.axis_steps[w]
    ratio=base.beta_per_um*h/2
    if not 0<ratio<1:raise ValueError('Mode exceeds the longitudinal Yee propagation band.')
    beta=2*np.arcsin(ratio)/h
    mode=replace(base,wavelength_um=s.wavelength,beta_per_um=float(beta))
    epsilon=np.empty((*shape,3),dtype=dtype)
    for c in range(3):
        axes=mode.component_axes('E'+'xyz'[c])
        coords=np.meshgrid(*axes,indexing='ij')
        epsilon[...,c]=_sample_permittivity_component(permittivity,coords,c,shape,dtype)
    if np.min(epsilon)<1:raise ValueError('Native FDTD CFL requires epsilon at least one.')
    amplitude_scale=float(np.max(np.abs(mode.fields[...,:3])))
    incident=mode if d==1 else mode.backward()
    fields=incident.fields/amplitude_scale
    courant=C0*dt/(h*1e-6)
    phase=np.exp(-1j*beta*h/2)
    loc_e=[slice(None)]*3
    loc_h=[slice(None)]*3
    loc_e[w]=slice(k,k+1)
    hat=k-1 if d==1 else k
    loc_h[w]=slice(hat,hat+1)
    terms=[]
    for c,partner,sign in ((u,v,-1),(v,u,1)):
        # (w cross H)_u=-H_v and (w cross H)_v=H_u.
        electric=-d*courant*sign*fields[...,3+partner]*phase/epsilon[...,c]
        magnetic=d*courant*sign*fields[...,partner]
        for family,component,loc,profile,tshift in (
                ('E',c,tuple(loc_e),electric,.5),('H',c,tuple(loc_h),magnetic,1.)):
            times=(np.arange(r.steps)+tshift)*dt
            for imag,offset in ((False,90.),(True,0.)):
                waveform=source_time_signal(s.model_copy(update={'phase':s.phase+offset}),times).astype(dtype)
                spatial=_profile(profile.imag if imag else profile.real,s.normal).astype(dtype)
                waveform.setflags(write=False)
                spatial.setflags(write=False)
                terms.append((family+'xyz'[component],loc,waveform,spatial))
    region_signature=json.dumps(r.model_dump(mode='json'),sort_keys=True)
    source_signature=json.dumps(s.model_dump(mode='json'),sort_keys=True)
    digest=hashlib.sha256(region_signature.encode()+source_signature.encode()+mode.fields.tobytes()+epsilon.tobytes()).hexdigest()
    epsilon.setflags(write=False)
    return ModalLaunch(mode,epsilon,region_signature,source_signature,float(r.mesh_nodes[w][0]),h,k,d,
                       base.beta_per_um,float(kt),tuple(terms),digest)


class _ModalSystem(_System):
    def __init__(self,project,epsilon,*,launch,options=None,observation_monitors=None,**kwargs):
        super().__init__(project,epsilon,observation_monitors=observation_monitors,**kwargs)
        self.sources={'E':[],'H':[]}
        for component,loc,waveform,profile in launch.terms:
            self.sources[component[0]].append((loc,'xyz'.index(component[1].lower()),
                self.tensor(np.array(waveform,copy=True)),self.tensor(np.array(profile,copy=True))))

        self._fused_modal = {'E': None, 'H': None}
        self._modal_scratch = {'E': None, 'H': None}
        profile_mode = getattr(options, 'execution_profile', 'auto')
        if os.environ.get('TORCHFDTD_BLOCK_FUSED') == '0':
            profile_mode = 'reference'

        if profile_mode in ('auto', 'block_fused'):
            self._prepare_block_fused()

    def _prepare_block_fused(self):
        for family in ('E', 'H'):
            items = self.sources[family]
            if len(items) == 4:
                loc0, comp0, wave0, prof0 = items[0]
                loc1, comp1, wave1, prof1 = items[1]
                loc2, comp2, wave2, prof2 = items[2]
                loc3, comp3, wave3, prof3 = items[3]
                grid_field = self.grid.E if family == 'E' else self.grid.H
                target_u = grid_field[loc0 + (comp0,)]
                target_v = grid_field[loc2 + (comp2,)]
                waveforms = torch.stack([wave0, wave1], dim=-1)
                profiles = torch.stack([prof0, prof1, prof2, prof3], dim=-1)
                self._modal_scratch[family] = torch.empty_like(prof0)
                self._fused_modal[family] = (target_u, target_v, profiles, waveforms)
            elif items:
                grid_field = self.grid.E if family == 'E' else self.grid.H
                fused_items = []
                for loc, comp, wave, prof in items:
                    target = grid_field[loc + (comp,)] if prof is not None else None
                    fused_items.append((target, loc, comp, wave, prof))
                self._modal_scratch[family] = torch.empty_like(items[0][3]) if items[0][3] is not None else None
                self._fused_modal[family] = fused_items

    def inject(self, value, family, step, *, functional=False):
        fused = self._fused_modal.get(family)
        target_field = self.grid.E if family == 'E' else self.grid.H
        if functional or fused is None or value is not target_field:
            return super().inject(value, family, step, functional=functional)

        if isinstance(fused, tuple) and len(fused) == 4:
            target_u, target_v, profiles, waveforms = fused
            w0 = waveforms[step, 0]
            w1 = waveforms[step, 1]
            scratch = self._modal_scratch[family]

            torch.mul(profiles[..., 0], w0, out=scratch)
            target_u.add_(scratch)
            torch.mul(profiles[..., 1], w1, out=scratch)
            target_u.add_(scratch)

            torch.mul(profiles[..., 2], w0, out=scratch)
            target_v.add_(scratch)
            torch.mul(profiles[..., 3], w1, out=scratch)
            target_v.add_(scratch)
        else:
            scratch = self._modal_scratch[family]
            for target, loc, comp, wave, prof in fused:
                if prof is None or target is None:
                    value[loc + (comp,)] = value[loc + (comp,)] + (wave[step] if prof is None else wave[step] * prof)
                else:
                    torch.mul(prof, wave[step], out=scratch)
                    target.add_(scratch)
        return value


class _ModalSimulation(DifferentiableSimulation):
    def _run(self,epsilon,spectral):
        launch=self.launch
        _check_launch_signatures(self.project,launch,epsilon)
        from .adjoint_memory import _resident_reservation
        from .memory_profile import host_memory
        base=_resident_reservation(self.project,self.options,epsilon.device,spectral)
        extra=2*launch.storage_bytes+(epsilon.numel()*epsilon.element_size() if epsilon.requires_grad else 0)
        # Use explicit budget fields below rather than depending on report labels.
        from .cuda_memory import cuda_budget_limit
        device_required=base.get('own_required_bytes',base['gpu_reservation_bytes'])
        host_required=base.get('host_reservation_bytes',0)+extra
        # The modal API treats an explicit host budget as the full admitted
        # host footprint, including its packet and gradient-freezing carrier.
        if self.options.host_budget_bytes is not None and host_required>self.options.host_budget_bytes:
            raise ValueError('Modal source storage exceeds the explicit host byte budget.')
        if epsilon.is_cuda:
            limit=cuda_budget_limit(str(epsilon.device),device_required+extra,self.options.gpu_budget_bytes)
            if base.get('forward_only') and base.get('live_allocated_bytes',0)>0:
                from .cuda_memory import cuda_mem_info
                free,total=cuda_mem_info(str(epsilon.device))
                live_alloc=base['live_allocated_bytes']
                if free is not None and total is not None:
                    cap=total if self.options.gpu_budget_bytes is None else self.options.gpu_budget_bytes
                    combined_limit=int(min(cap,free+live_alloc)*.88)-live_alloc
                    safe_free=max(0,free-200*1024**2)
                    limit=max(limit,min(cap,combined_limit,safe_free))
            if device_required+extra>limit:raise ValueError('Modal source storage exceeds the GPU budget.')
        available=host_memory()['available_bytes']
        if available is not None and host_required>int(available*.8):raise ValueError('Modal source storage exceeds available host memory.')
        if self.options.resident_budget_bytes is not None and (device_required+extra if epsilon.is_cuda else host_required)>self.options.resident_budget_bytes:
            raise ValueError('Modal source storage exceeds the resident budget.')
        epsilon,_,fixed_transverse=_frozen_launch_material(launch,epsilon)
        factory=lambda p,e,**kwargs:_ModalSystem(p,e,launch=launch,options=self.options,**kwargs)
        result=super()._run(epsilon,spectral,system_factory=factory)
        result.report['memory_reservation_bytes']+=extra
        result.report['host_reservation_bytes']+=extra
        if epsilon.is_cuda:result.report['gpu_reservation_bytes']+=extra
        _modal_report(result.report,launch,extra,fixed_transverse)
        return result


def _modal_report(report,launch,extra,fixed_transverse):
    report.update(modal_source=True,modal_source_storage_bytes=launch.storage_bytes,modal_source_extra_reservation_bytes=extra,
        modal_beta_per_um=float(launch.mode.beta_per_um.real),modal_beta_tilde_per_um=float(launch.beta_tilde_per_um.real),
        source_neighborhood_gradient='frozen',modal_source_identity=launch.identity)
    aperture=getattr(launch,'aperture',None)
    if aperture is not None:
        report.update(modal_aperture_cells=[list(item) for item in aperture],
                      modal_aperture_edge_energy_fraction=launch.edge_energy_fraction)
    if fixed_transverse:
        report.update(transverse_boundary='cpml',transverse_cpml_cladding_gradient='frozen',
                      mode_beta_complex=[float(launch.mode.beta_per_um.real),float(launch.mode.beta_per_um.imag)])


class _ModalStreamedExecution(_StreamedExecution):
    """Streamed physics factories whose host system carries the fixed sheets."""
    def __init__(self,launch):
        self.launch=launch

    def host(self,project,value,spectral):
        return _ModalSystem(project,value,launch=self.launch,prepare_updates=False,
                            observation_monitors=None if spectral is None else spectral.observers)


class _ModalStreamedSimulation(StreamedSimulation):
    """Streamed X-slab execution of the fixed modal sheets and plane observers.

    Tiles receive only their rows of each sheet and accumulate only their own
    plane observers; the transposes are the existing tile transposes. The
    source neighbourhood is frozen exactly as in the resident path.
    """
    def _run(self,epsilon,spectral):
        launch=self.launch
        _check_launch_signatures(self.project,launch,epsilon)
        if epsilon.device.type!='cpu':raise ValueError('Streamed modal epsilon must be a CPU tensor.')
        options=self.streaming_options
        base=_reservation(self.project,epsilon,options,spectral)
        # The host system and every packed tile slot copy the fixed sheets; the
        # gradient-freezing carrier is one more full host volume.
        buffers=options.tile_buffers if options.tile_transfers=='async' else 1
        extra=(1+buffers)*launch.storage_bytes+(epsilon.numel()*epsilon.element_size() if epsilon.requires_grad else 0)
        from .memory_profile import host_memory
        available=host_memory()['available_bytes']
        limit=min(options.host_budget_bytes,int(available*.8)) if available is not None else options.host_budget_bytes
        if base['host_reservation_bytes']+extra>limit:
            raise ValueError('Modal source storage exceeds the streamed host budget.')
        epsilon,_,fixed_transverse=_frozen_launch_material(launch,epsilon)
        result=super()._run(epsilon,spectral,execution=_ModalStreamedExecution(launch))
        result.report['host_reservation_bytes']+=extra
        _modal_report(result.report,launch,extra,fixed_transverse)
        return result


class ModeInjectedPlaneSimulation(DifferentiablePlaneSimulation):
    """Native Yee/CPML forward and adjoint with fixed modal sheets.

    Resident AdjointOptions use the checkpointed resident adjoint. Streamed
    options run the same sheets and plane observers through bounded X slabs.
    """
    _resident_model_type=_ModalSimulation
    _streamed_model_type=_ModalStreamedSimulation

    def __init__(self,project,launch,options=None,*,quadrature_counts=None):
        super().__init__(project,options,quadrature_counts=quadrature_counts)
        self.model.launch=launch
        self.launch=launch
        self.signature=hashlib.sha256((self.signature+launch.identity).encode()).hexdigest()
        if launch.mode.boundary == 'cpml':
            from types import SimpleNamespace
            normal_axis='xyz'.index(launch.mode.normal)
            for _,normal,plan,_ in self.plans:
                mode=launch.detector_mode(float(plan['points_um'][0,normal_axis]))
                mode.validate_quadrature(SimpleNamespace(normal=normal,
                    points_um=plan['points_um'],weights=plan['weights']))

    def reference(self,epsilon,frequency_hz,*,block_size=32,graph_budget_bytes=None):
        """Small-problem full Torch autograd oracle through the same fixed sheets.

        Every timestep stays in the autograd graph. This is the check for the
        checkpointed resident adjoint, not a large-simulation path. The graph
        is admitted by its estimated memory; graph_budget_bytes caps it.
        """
        if isinstance(self.model,_ModalStreamedSimulation):
            raise ValueError('The full-autograd oracle uses resident systems.')
        region=self.model.project.region
        from .oracle_memory import admit_oracle,check_oracle_dtype,oracle_graph_bytes
        # The graph runs at epsilon's dtype and the estimate counts the project precision.
        check_oracle_dtype(epsilon,region)
        launch=self.launch
        admit_oracle(oracle_graph_bytes(region,'yee',material_elements=epsilon.numel(),source_terms=len(launch.terms)),
                     epsilon.device,graph_budget_bytes)
        _check_launch_signatures(self.model.project,launch,epsilon)
        material,_,_=_frozen_launch_material(launch,epsilon)
        def run(spectral):
            system=_ModalSystem(self.model.project.model_copy(deep=True),material,launch=launch,
                                observation_monitors=spectral.observers,prepare_kernels=False)
            state=tuple(torch.zeros_like(x) for x in system.state())
            signals=spectral.zeros()
            for step in range(region.steps):
                state=system.reference_step(state,step,material)
                spectral.accumulate(signals,system.observe(state)[None],step)
            return spectral.result(signals,dict(full_time_autograd=True,modal_source=True))
        return self._planes(epsilon,frequency_hz,block_size,run)


def modal_plane_amplitudes(plane,launch,*,check_frequency=True):
    """Signed-axis forward/backward complex amplitudes in the fixed mode basis.

    Amplitudes include the native time-integral DFT factor (seconds). They are
    referenced to the underlying unit reduced-power mode, not a peak-E basis.
    Supports single-frequency planes, broadband multi-frequency planes against
    a carrier `ModalLaunch`, or a `Sequence[ModalLaunch]` matching `plane.frequency_hz`.
    """
    if isinstance(launch, Sequence) and not isinstance(launch, ModalLaunch):
        launches = tuple(launch)
        if len(launches) != plane.frequency_hz.numel():
            raise ValueError('A sequence of modal launches must match the number of plane frequencies.')
        fields_flat = plane.fields.reshape(plane.frequency_hz.numel(), -1, 6) if plane.fields.ndim > 3 else plane.fields
        forwards, backwards = [], []
        for idx, item in enumerate(launches):
            single_plane = replace(plane, fields=fields_flat[idx:idx+1], frequency_hz=plane.frequency_hz[idx:idx+1])
            res = modal_plane_amplitudes(single_plane, item, check_frequency=check_frequency)
            forwards.append(res['forward'][0])
            backwards.append(res['backward'][0])
        return dict(forward=torch.stack(forwards, dim=0), backward=torch.stack(backwards, dim=0))

    if plane.normal!=launch.mode.normal:raise ValueError('Modal plane normal mismatch.')
    normal='xyz'.index(plane.normal)
    geometry = getattr(plane, 'points_um_geometry', None)
    coordinate = geometry[0, normal] if geometry is not None else plane.points_um[0, normal]
    mode=launch.detector_mode(float(coordinate))
    target_freq = plane.frequency_hz.new_tensor(C0/(mode.wavelength_um*1e-6))
    if check_frequency:
        if plane.frequency_hz.numel() == 1:
            if not torch.isclose(plane.frequency_hz[0], target_freq, rtol=1e-6):
                raise ValueError('Modal amplitudes require the solved carrier frequency.')
        else:
            f_min = plane.frequency_hz.min() * (1.0 - 1e-3)
            f_max = plane.frequency_hz.max() * (1.0 + 1e-3)
            if not bool((target_freq >= f_min) & (target_freq <= f_max)):
                raise ValueError('Modal amplitudes require the solved carrier frequency.')
    _uniform_cell_quadrature(plane,mode)
    basis=torch.tensor(mode.sample_plane(plane.points_um.detach().cpu().numpy()),device=plane.fields.device,dtype=plane.fields.dtype)
    basis_scale=basis.abs().max()
    basis=basis/basis_scale
    fields_flat = plane.fields.reshape(plane.frequency_hz.numel(), -1, 6) if plane.fields.ndim > 3 else plane.fields
    scale=fields_flat.detach().abs().max()
    if not bool(torch.isfinite(scale)):raise ValueError('Modal fields must be finite.')
    if not bool(scale>0):scale=scale.new_ones(())
    weights=plane.weights/plane.weights.detach().abs().max()
    forward,backward=_amplitudes(fields_flat/scale,basis,weights,normal)
    return dict(forward=forward*scale/basis_scale,backward=backward*scale/basis_scale)


def modal_s_parameters(plane,reference,launch,*,subtract_reference_backward=True,check_frequency=True):
    """Complex t and r, referenced to propagation in the launch direction.

    t uses the matched reference at the same plane, so an unchanged guide has
    t=1. r subtracts the matched counterpropagating baseline. For a minus launch,
    the incident amplitude is the negative-axis mode. Source eigenmodes stay fixed.
    """
    if plane.run_signature!=reference.run_signature or plane.normal!=reference.normal:
        raise ValueError('Matched modal reference configuration required.')
    for name in ('points_um','weights','frequency_hz'):
        a,b=getattr(plane,name),getattr(reference,name)
        if a.device!=b.device or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise ValueError('Matched modal reference sampling required.')
    # Common detached spectral-field scaling keeps division seeds in FP32 range.
    scale=reference.fields.detach().abs().max()
    if not bool(torch.isfinite(scale)) or not bool(scale>0):raise ValueError('Modal reference is zero or nonfinite.')
    sample=modal_plane_amplitudes(replace(plane,fields=plane.fields/scale),launch,check_frequency=check_frequency)
    incident=modal_plane_amplitudes(replace(reference,fields=reference.fields/scale),launch,check_frequency=check_frequency)
    direction = launch[0].direction if isinstance(launch, Sequence) and not isinstance(launch, ModalLaunch) else launch.direction
    f,b=('forward','backward') if direction==1 else ('backward','forward')
    normalization=incident[f]
    # Threshold relative to modal reference support, independent of SI units.
    support=incident[f].abs()+incident[b].abs()
    if not bool(torch.isfinite(support).all()) or bool((support<=0).any()) or bool((normalization.abs()<1e-6*support).any()):
        raise ValueError('Reference has no supported mode in the launch direction.')
    return dict(transmission=sample[f]/normalization,
                reflection=(sample[b]-incident[b] if subtract_reference_backward else sample[b])/normalization,
                forward_amplitude=sample['forward']*scale,backward_amplitude=sample['backward']*scale)
