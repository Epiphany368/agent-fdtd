"""Native Project geometry, fixed modal ports and material-epsilon objectives.

This adapter exposes no arbitrary Python expressions. Material derivatives hold
port profiles, calibration fields and source/CPML exterior materials fixed.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator
from .models import Project, Monitor
from .differentiable import AdjointOptions
from .mode_network import ModeNetwork, FixedModePort
from .open_mode_injection import OpenPortOptions


class _Config(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)


class _Port(_Config):
    name: str=Field(min_length=1,max_length=100)
    coordinate_um: StrictFloat
    source_coordinate_um: StrictFloat
    direction: StrictInt
    mode_indices: list[StrictInt]=Field(default_factory=lambda:[0],min_length=1,max_length=8)


class _Open(_Config):
    cladding_epsilon: StrictFloat=Field(ge=1)
    mode_budget_bytes: StrictInt=Field(default=2*1024**3,gt=0)
    target_neff: StrictFloat | None=None
    confinement_tolerance: StrictFloat=Field(default=1e-4,gt=0,le=1e-3)


class _Execution(_Config):
    device: Literal['cpu','cuda']='cpu'
    checkpoints: StrictInt=Field(default=4,ge=0,le=64)
    gpu_budget_bytes: StrictInt | None=Field(default=None,gt=0)
    host_budget_bytes: StrictInt=Field(default=8*1024**3,gt=0)
    resident_budget_bytes: StrictInt | None=Field(default=None,gt=0)
    network_budget_bytes: StrictInt=Field(default=256*1024**2,gt=0)
    output_budget_bytes: StrictInt=Field(default=64*1024**2,gt=0)


class _Objective(_Config):
    output_channel: tuple[str,StrictInt]
    input_channel: tuple[str,StrictInt]
    quantity: Literal['power','real','imag']


class ModeNetworkConfig(_Config):
    version: StrictInt=1
    project: Project
    normal: Literal['x','y','z']
    ports: list[_Port]=Field(min_length=2,max_length=2)
    num_modes: StrictInt=Field(default=1,ge=1,le=8)
    open_ports: _Open | None=None
    execution: _Execution=Field(default_factory=_Execution)
    objective: _Objective | None=None
    differentiate_materials: list[str]=Field(default_factory=list,max_length=128)

    @model_validator(mode='after')
    def valid(self):
        if self.version!=1: raise ValueError('Only mode-network configuration version 1 is supported.')
        left,right=self.ports
        if left.name==right.name: raise ValueError('Port names must be distinct.')
        if (left.direction,right.direction)!=(1,-1): raise ValueError('Order ports left(+1), right(-1).')
        if not left.source_coordinate_um<left.coordinate_um<right.coordinate_um<right.source_coordinate_um:
            raise ValueError('Source planes must be outside the two ordered phase planes.')
        for port in self.ports:
            if len(set(port.mode_indices))!=len(port.mode_indices) or any(i<0 or i>=self.num_modes for i in port.mode_indices):
                raise ValueError('Mode indices must be unique and inside num_modes.')
        names=[m.name for m in self.project.materials]
        if len(set(names))!=len(names): raise ValueError('Material names must be unique.')
        if len(set(self.differentiate_materials))!=len(self.differentiate_materials) or any(n not in names for n in self.differentiate_materials):
            raise ValueError('Differentiate unique existing material names only.')
        if self.differentiate_materials and self.objective is None:
            raise ValueError('Material gradients require one explicit real objective.')
        if self.objective is not None:
            channels={(p.name,i) for p in self.ports for i in p.mode_indices}
            if self.objective.output_channel not in channels or self.objective.input_channel not in channels:
                raise ValueError('Objective channels must occur in the selected mode basis.')
        return self


def _snapshot(config):
    return ModeNetworkConfig.model_validate(config.model_dump() if isinstance(config,ModeNetworkConfig) else config)


def mode_network_request_digest(config_or_snapshot):
    """Canonical identity shared by preparation, worker and published result."""
    snapshot=_snapshot(config_or_snapshot).model_dump(mode='json')
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _options(config):
    e=config.execution
    return AdjointOptions(checkpoints=e.checkpoints,gpu_budget_bytes=e.gpu_budget_bytes,
        host_budget_bytes=e.host_budget_bytes,resident_budget_bytes=e.resident_budget_bytes)


def _scope(c):
    r=c.project.region;w='xyz'.index(c.normal)
    if r.dimension!='3d' or r.mesh_type!='uniform' or r.complex_fields or r.precision!='float32' or r.material_sampling!='yee' or r.interface_method!='staircase':
        raise ValueError('Mode-network projects require real FP32 uniform 3D staircase Yee sampling.')
    if any(m.model!='dielectric' or m.oscillators for m in c.project.materials):
        raise ValueError('Only nondispersive isotropic dielectric materials are supported.')
    active=[c.project.resolved_source(s) for s in c.project.sources if s.enabled]
    if len(active)!=1: raise ValueError('Provide exactly one enabled plane timing source.')
    source=active[0]
    if source.normal!=c.normal or source.kind!='plane' or source.injection!='soft' or source.pulse!='gaussian' or source.time_definition!='cycles' or source.pulse_cycles<2:
        raise ValueError('Use a matching-normal soft Gaussian plane timing source with at least two cycles.')
    if any(f.kind!='pml' for f in r.boundaries.pair(w)): raise ValueError('Propagation requires longitudinal PML on both faces.')
    opened=False
    for axis in range(3):
        if axis==w: continue
        kinds=tuple(f.kind for f in r.boundaries.pair(axis))
        if kinds==('pml','pml') and c.open_ports is not None: opened=True
        elif kinds!=('periodic','periodic') or r.bloch_phase[axis]!=0:
            raise ValueError('Transverse faces require zero-phase periodic pairs or explicit open CPML pairs.')
    if c.open_ports is not None and not opened: raise ValueError('Open ports require at least one transverse CPML pair.')
    nodes=r.mesh_nodes[w]
    indices=[]
    for port in c.ports:
        for coordinate,source_plane in ((port.coordinate_um,False),(port.source_coordinate_um,True)):
            k=int(np.argmin(abs(nodes[:-1]-coordinate)))
            if not math.isclose(float(nodes[k]),coordinate,abs_tol=1e-7,rel_tol=0): raise ValueError('Port/source coordinates must lie on E-node planes.')
            lo,hi=r.interior_bounds(w)
            if k<2 or k+2>=len(nodes) or nodes[k-1]<lo or nodes[k+2]>hi: raise ValueError('Port/source planes require two interior cells before PML.')
            if not source_plane: indices.append(k)
    if indices[1]-indices[0]<4: raise ValueError('Leave at least four cells between phase planes.')
    return source,tuple(indices)


class _SpectralBound:
    """Metadata upper bound: six components, eight corners, two full planes."""
    def __init__(self,points,steps):
        self.points=points;self.components=range(48*points);self.block_size=min(32,steps)
    def reservation(self,depth):
        m=len(self.components);q=self.points;output=8*m
        workspace=(8*m+12*depth+4*depth*m+4*depth+1)*4
        maps=6*8*q*16
        return dict(spectral_output_bytes=output,
            spectral_reservation_bytes=2*output+workspace+4+128*q*4+2*maps)


def _admit(c,report):
    from .memory_profile import host_memory
    e=c.execution;host=report['host_reservation_bytes'];gpu=report['gpu_reservation_bytes']
    available=host_memory()['available_bytes']
    limit=e.host_budget_bytes if available is None else min(e.host_budget_bytes,int(.8*available))
    if host>limit: raise ValueError('Mode-network combined preparation/execution exceeds the host byte budget or available RAM.')
    active=host if e.device=='cpu' else gpu
    if e.resident_budget_bytes is not None and active>e.resident_budget_bytes:
        raise ValueError('Mode-network combined reservation exceeds the resident byte budget.')
    if e.device=='cuda':
        if not torch.cuda.is_available(): raise ValueError('CUDA execution requested but CUDA is unavailable.')
        from .cuda_memory import cuda_budget_limit
        if gpu>cuda_budget_limit('cuda',gpu,e.gpu_budget_bytes): raise ValueError('Mode-network combined reservation exceeds the GPU byte budget.')


def mode_network_plan(config):
    """Conservative metadata-only plan, before modes, rasterization or fields."""
    c=_snapshot(config);source,indices=_scope(c);r=c.project.region;e=c.execution
    w='xyz'.index(c.normal);transverse=[a for a in range(3) if a!=w]
    n=math.prod(r.shape);q=math.prod(r.shape[a] for a in transverse)
    count=sum(len(p.mode_indices) for p in c.ports);dimension=2*q
    if c.num_modes>=dimension-1: raise ValueError('Too many modes for the transverse eigensystem.')
    candidates=min(dimension-2,max(c.num_modes+6,2*c.num_modes))
    ncv=min(dimension,max(4*candidates+1,40))
    mode=8*dimension*dimension*8+8*dimension*(ncv+4*candidates)+2048*q
    if c.open_ports is not None:
        from .open_mode_operators import plan_open_mode_operators
        op=plan_open_mode_operators(r,normal=c.normal,wavelength_um=source.wavelength)
        mode+=op['required_workspace_bytes']
        if mode>c.open_ports.mode_budget_bytes: raise ValueError('Open mode preparation exceeds mode_budget_bytes.')
    # Includes all packets retained while preparing the last incident channel.
    packets=count*(8*r.steps+80*q)*4
    wrapper=packets+24*n*4+8192*q+128*count*q*4+16*count*count*4
    if wrapper>e.network_budget_bytes: raise ValueError('Mode-network wrapper exceeds network_budget_bytes.')
    s_bytes=count*count*8
    if s_bytes>e.output_budget_bytes: raise ValueError('S matrix exceeds output_budget_bytes.')
    spectral=_SpectralBound(2*q,r.steps)
    layout=2*q*(4*8+6*8*16)+1024*len(spectral.components)+8192
    # Native component arrays + stacked epsilon/ownership + contains temporaries,
    # int64 gather indices, table graph, full gradient/CPU-device copies.
    raster=256*n+4096*len(c.project.structures)+64*len(c.project.materials)
    material_graph=128*n+4096
    project=c.project.model_copy(deep=True);project.monitors=[Monitor()]
    if e.resident_budget_bytes is not None: project.region.memory_mode='budgeted'
    if e.device=='cuda': project.region.cuda_kernel='fused'
    from .adjoint_memory import _resident_reservation
    base=_resident_reservation(project,_options(c),e.device,spectral)
    # Prepared per-channel project snapshots, dictionaries and replay closures
    # coexist. Charge their count, not just the single incoming JSON payload.
    metadata=len(c.model_dump_json().encode())*(32+64*count)+65536
    extra_host=wrapper+layout+raster+material_graph+metadata+2*packets
    # Add the complete network wrapper again on CUDA, not merely transfer
    # bytes: its 21*N*4 material/gradient carriers, 3*N*4 calibration,
    # modal scratch and packet components can coexist on device.
    extra_gpu=(wrapper+material_graph+24*n+2*packets+layout) if e.device=='cuda' else 0
    host=max(mode+packets+layout+metadata,base['host_reservation_bytes']+extra_host)
    gpu=base['gpu_reservation_bytes']+extra_gpu
    report=dict(host_reservation_bytes=host,gpu_reservation_bytes=gpu,
        mode_engineering_reservation_bytes=mode,network_wrapper_reservation_bytes=wrapper,
        packet_upper_bound_bytes=packets,plane_layout_upper_bound_bytes=layout,
        native_rasterizer_reservation_bytes=raster,material_graph_reservation_bytes=material_graph,
        metadata_reservation_bytes=metadata,resident=base,shape=list(r.shape),channels=count,
        observation_history_retained=False,scope='Fixed opposing dielectric mode ports; first-order interior material epsilon VJP.',
        budget_type='Conservative engineering allocation bound, not a process RSS guarantee.')
    # JSON carries the full request, admission and mode/calibration metadata,
    # not just the complex64 S array. Use the worker's exact JSON options.
    # Fixed schema/report overhead is charged 64 KiB, each selected channel's
    # mode plus calibration diagnostics 16 KiB, and both S floats 128 B/entry.
    # Escaped user names and the complete request/admission are counted exactly.
    request_bytes=len(json.dumps(c.model_dump(mode='json'),allow_nan=False).encode('utf8'))
    admission_bytes=len(json.dumps(report,allow_nan=False).encode('utf8'))
    names_bytes=len(json.dumps([[p.name,i] for p in c.ports for i in p.mode_indices],allow_nan=False).encode('utf8'))
    gradient_names_bytes=len(json.dumps(c.differentiate_materials,allow_nan=False).encode('utf8'))
    output_bound=request_bytes+admission_bytes+65536+16384*count+128*count*count+4*names_bytes+2*gradient_names_bytes
    report.update(s_matrix_bytes=s_bytes,result_json_reservation_bytes=output_bound,
        output_budget_scope='Complete serialized result JSON, including request and diagnostics.')
    if output_bound>e.output_budget_bytes:
        raise ValueError('Mode-network result JSON reservation exceeds output_budget_bytes before solving.')
    # Encoding, JSON container preparation and publication copies coexist.
    report['host_reservation_bytes']+=4*output_bound
    _admit(c,report)
    report['request_digest']=mode_network_request_digest(c)
    return report


def _section(project,normal,coordinate):
    """Native membership/mesh-order semantics evaluated on one fixed section."""
    from .geometry import contains
    w='xyz'.index(normal);u,v=(w+1)%3,(w+2)%3
    materials={m.name:m.instantaneous_epsilon for m in project.materials}
    objects=tuple(sorted((s for s in project.structures if s.enabled),key=lambda s:-s.mesh_order))
    def sample(a,b):
        a,b=np.broadcast_arrays(a,b);xyz=[None]*3;xyz[u]=a;xyz[v]=b;xyz[w]=np.full(a.shape,coordinate)
        value=np.full(a.shape,project.region.background_index**2,dtype=np.float32)
        for obj in objects:
            value[np.broadcast_to(contains(obj,*xyz),a.shape)]=materials[obj.material]
        return value
    return sample


def _material(c,epsilon,ownership):
    """One small material table feeds exact native ownership, with no shape VJP."""
    with torch.set_grad_enabled(bool(c.differentiate_materials)):
        device=c.execution.device
        table=torch.tensor([m.instantaneous_epsilon for m in c.project.materials],dtype=torch.float32,device=device,
                           requires_grad=bool(c.differentiate_materials))
        # A constant background row avoids clipping ownership -1 into a material.
        all_values=torch.cat((table,table.new_tensor([c.project.region.background_index**2])))
        indices=torch.tensor(ownership.astype(np.int64),device=device)
        indices[indices<0]=len(c.project.materials)
        value=all_values[indices].contiguous()
        return value,table


def run_mode_network(config,on_progress=None):
    """Execute a snapshot and return compact JSON, retaining no field graphs."""
    c=_snapshot(config)
    if c.differentiate_materials and torch.is_inference_mode_enabled():
        raise ValueError('Material derivatives require inference_mode to be disabled; no_grad is supported.')
    def progress(phase):
        if on_progress is not None: on_progress(dict(phase=phase))
    progress('validating');admission=mode_network_plan(c)
    source,_=_scope(c);project=c.project.model_copy(deep=True)
    if c.execution.resident_budget_bytes is not None: project.region.memory_mode='budgeted'
    if c.execution.device=='cuda': project.region.cuda_kernel='fused'
    progress('preparing_modes')
    ports=tuple(FixedModePort(**p.model_dump()) for p in c.ports)
    sections={p.name:_section(project,c.normal,p.source_coordinate_um) for p in c.ports}
    opened=None if c.open_ports is None else OpenPortOptions(**c.open_ports.model_dump())
    network=ModeNetwork(project,ports,options=_options(c),num_modes=c.num_modes,
        port_permittivities=sections,open_ports=opened,network_budget_bytes=c.execution.network_budget_bytes)
    progress('admitting_material');_admit(c,admission)
    from .solver import voxelize
    progress('rasterizing')
    epsilon,_,ownership=voxelize(project,with_ownership=True)
    material,table=_material(c,epsilon,ownership)
    # Assembly must remain exactly the native rasterizer, including overlaps.
    if not np.array_equal(material.detach().cpu().numpy(),epsilon): raise ValueError('Native material table assembly disagrees with Yee voxelization.')
    del epsilon,ownership
    progress('calibration_and_forward')
    with torch.set_grad_enabled(bool(c.differentiate_materials)):
        result=network(material,output_budget_bytes=c.execution.output_budget_bytes)
        objective=None;gradients={}
        if c.objective is not None:
            channels=list(result.channels)
            z=result.s[channels.index(c.objective.output_channel),channels.index(c.objective.input_channel)]
            score=z.abs().square() if c.objective.quantity=='power' else (z.real if c.objective.quantity=='real' else z.imag)
            objective=float(score.detach())
            if c.differentiate_materials:
                progress('backward')
                gradient,=torch.autograd.grad(score,table)
                gradients={name:float(gradient[[m.name for m in project.materials].index(name)].detach()) for name in c.differentiate_materials}
    s=result.s.detach().cpu()
    modes=[]
    for channel,launch in zip(network.channels,network._launches):
        mode=launch.mode
        modes.append(dict(channel=list(channel),beta_per_um=[float(mode.beta_per_um.real),float(mode.beta_per_um.imag)],
            neff=[float(mode.neff.real),float(mode.neff.imag)],maxwell_residual=mode.maxwell_residual,
            eigenpair_residual=mode.eigenpair_residual,boundary=mode.boundary,
            diagnostics=dict(getattr(mode,'diagnostics',{}))))
    output=dict(status='completed',channels=[list(x) for x in result.channels],s_real=s.real.tolist(),s_imag=s.imag.tolist(),
        phase_planes_um=list(result.port_coordinates_um),objective=objective,material_gradients=gradients,
        mode_summaries=modes,admission=admission,request_digest=admission['request_digest'],
        request=c.model_dump(mode='json'),report=result.report,
        gradient_contract='d objective / d interior material epsilon; fixed mode profiles, calibration and source/CPML exteriors')
    payload=json.dumps(output,allow_nan=False).encode('utf8')
    if len(payload)>c.execution.output_budget_bytes:
        raise ValueError('Mode-network result JSON exceeds output_budget_bytes.')
    if len(payload)>admission['result_json_reservation_bytes']:
        raise ValueError('Mode-network result JSON exceeds its admitted serialization bound.')
    progress('completed')
    return output
