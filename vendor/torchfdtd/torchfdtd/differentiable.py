"""Discrete Yee/CPML derivatives with bounded, recomputed physical checkpoints.

The supported contract is real nondispersive diagonal epsilon, fixed mesh,
prepared point/plane sources and indexed observations. Real CUDA forward/replay
uses native fused kernels. Complex Bloch fields use Torch operations. Backward
transposes the actual staggered updates, including conjugate Bloch seams.
This module does not promise derivatives for unsupported scene features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .boundaries import (BoundaryDescription, CURL_TERMS, reject_pml_dispersion, _slice, extended_shape, face_index,
                         is_nodal, material_shape)
from .models import Project
from .plan import check_snapshot, project_snapshot
from .solver import field_axes, index_at


@dataclass(frozen=True)
class AdjointOptions:
    checkpoints: int = 4
    storage: str = 'device'
    gpu_budget_bytes: int | None = None
    host_budget_bytes: int | None = None
    disk_budget_bytes: int | None = None
    checkpoint_directory: str | Path | None = None
    device_checkpoints: int = 0
    host_checkpoints: int = 0
    backward_kernel: str = 'auto'
    checkpoint_transfers: str = 'sync'
    staging_slots: int = 2
    resident_budget_bytes: int | None = None
    restart_directory: str | Path | None = None
    restart_every_blocks: int = 64
    restart_every_seconds: float | None = None
    stop_at_step: int | None = None
    allow_auto_shutoff: bool = False
    auto_shutoff: bool = False
    execution_profile: str = 'auto'

    def __post_init__(self):
        if self.execution_profile not in ('auto', 'block_fused', 'legacy', 'reference'):
            raise ValueError('execution_profile must be auto, block_fused, legacy or reference.')
        if self.backward_kernel not in ('auto','torch','fused'):
            raise ValueError('backward_kernel must be auto, torch or fused.')
        if self.checkpoint_transfers not in ('sync','async'):
            raise ValueError('checkpoint_transfers must be sync or async.')
        if isinstance(self.staging_slots,bool) or not isinstance(self.staging_slots,int) or not 1<=self.staging_slots<=4:
            raise ValueError('staging_slots must be an integer from one to four.')
        if isinstance(self.checkpoints, bool) or not isinstance(self.checkpoints,int) or not 0 <= self.checkpoints <= 64:
            raise ValueError('checkpoints must be an integer between 0 and 64.')
        if self.storage not in ('device','host','disk','hierarchical'):
            raise ValueError('Checkpoint storage must be device, host, disk or hierarchical.')
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (self.device_checkpoints,self.host_checkpoints)):
            raise ValueError('Tier checkpoint counts must be nonnegative integers.')
        if self.device_checkpoints+self.host_checkpoints>self.checkpoints:
            raise ValueError('Tier checkpoint counts exceed the total checkpoint count.')
        for name in ('gpu_budget_bytes','host_budget_bytes','disk_budget_bytes','resident_budget_bytes'):
            value=getattr(self,name)
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value <= 0):
                raise ValueError(f'{name} must be a positive integer byte budget.')
        disk=self.checkpoints if self.storage=='disk' else self.checkpoints-self.device_checkpoints-self.host_checkpoints if self.storage=='hierarchical' else 0
        if (self.storage=='disk' or disk) and (self.checkpoint_directory is None or self.disk_budget_bytes is None):
            raise ValueError('Disk checkpoints require a directory and an explicit disk byte budget.')


@dataclass
class DifferentiableResult:
    signals: torch.Tensor
    time_step: float
    monitor_components: tuple[str,...]
    report: dict = field(default_factory=dict)

    def spectrum(self, frequency_hz, *, window=None):
        """Time-integral DFT, including the existing half-step H time convention."""
        frequency=torch.as_tensor(frequency_hz,device=self.signals.device,dtype=self.signals.real.dtype).reshape(-1)
        if not bool(torch.isfinite(frequency).all()) or bool((frequency<=0).any()):
            raise ValueError('Frequencies must be finite and positive.')
        if bool((frequency*self.time_step>=.5).any()):
            raise ValueError('Frequencies must be below temporal Nyquist.')
        t=(torch.arange(self.signals.shape[0],device=frequency.device,dtype=frequency.dtype)+1)*self.time_step
        if window is None:
            weights=torch.ones_like(t)
        else:
            weights=torch.as_tensor(window,device=t.device,dtype=t.dtype)
            if weights.shape!=t.shape:raise ValueError('Window must have one weight per timestep.')
        values=[]
        for i,comp in enumerate(self.monitor_components):
            times=t+(self.time_step/2 if comp[0]=='H' else 0)
            kernel=torch.exp(-2j*torch.pi*frequency[:,None]*times[None,:])
            values.append(kernel@(self.signals[:,i]*weights).to(kernel.dtype)*self.time_step)
        return torch.stack(values,dim=-1)


class _Grid:
    """Weak-referenceable owner for the existing fused CUDA kernel interface."""
    pass


class _System:
    # Declared on each class that implements the stored PMC face topology; not inherited.
    _pmc_faces=True

    def __init__(self, project, epsilon, *, prepare_updates=True, observation_monitors=None,
                 prepare_kernels=True, prepare_permittivity=True, observation_cache=None):
        self.project=project
        self.region=r=project.region
        self.epsilon=epsilon
        self.device,self.dtype=epsilon.device,epsilon.dtype
        self.field_dtype=(torch.complex128 if self.dtype==torch.float64 else torch.complex64) if r.complex_fields else self.dtype
        if prepare_kernels and self.device.type=='cuda' and not r.complex_fields:
            from .cuda_bootstrap import prepare_cuda_kernels
            prepare_cuda_kernels()
        if not prepare_updates and self.device.type != 'cpu':
            raise ValueError('Storage-only systems require CPU epsilon.')
        self.current_step=0
        # Preparing coefficients must not allocate another full-volume E/H
        # field or touch fdtd's process-global backend.
        reject_pml_dispersion(project,type(self).__name__)
        template=BoundaryDescription(r)
        # PMC/symmetric walls: lower walls mirror inside the volume arrays,
        # upper walls keep their tangential E, normal H and E edges in stored
        # blocks. Epsilon then carries one extra sampled row per upper wall.
        self.pmc=bool(template.pmc_lower or template.pmc_upper)
        if self.pmc and not type(self).__dict__.get('_pmc_faces',False):
            raise ValueError(f'PMC/symmetric faces are not implemented by {type(self).__name__}; only the plain and dispersive Yee systems carry the stored face topology.')
        if tuple(epsilon.shape[:3])!=material_shape(r) or epsilon.ndim not in (3,4) or (epsilon.ndim==4 and epsilon.shape[3]!=3):
            raise ValueError(f'epsilon shape must be {material_shape(r)} or {material_shape(r,True)}, including stored upper PMC rows.')
        from .injection import source_terms,validate_oneway_materials
        # Prepared source terms and the stored-face contract precede every field allocation.
        terms=[(source,list(source_terms(project,source))) for source in project.sources]
        if self.pmc:
            for source,items in terms:
                for component,loc,waveform,profile in items:
                    if any((sl.stop if isinstance(sl,slice) else sl+1)>n for sl,n in zip(loc,r.shape)) and not all(isinstance(sl,int) for sl in loc):
                        raise ValueError(f'{source.name}: only point sources may address a stored upper PMC/symmetric face; plane sources must end below the wall.')
        volume=self._volume(epsilon)
        self.eps4=volume[...,None] if epsilon.ndim==3 else volume
        g=self.grid=_Grid()
        def initial(shape):
            if not prepare_updates:return torch.zeros((),device=self.device,dtype=self.field_dtype).expand(shape)
            return torch.zeros(shape,device=self.device,dtype=self.field_dtype)
        g.E=initial((*r.shape,3))
        g.H=initial(g.E.shape)
        g.inverse_permeability=torch.ones(1,device=self.device,dtype=self.dtype)
        g.is_torch=True
        g.courant_number=r.rectangular_courant
        g.time_step=r.time_step
        g.material_states=[]
        g.wrap=template.wrap.copy()
        g.pec_upper=template.pec_upper.copy()
        g.pmc_lower=template.pmc_lower.copy()
        g.pmc_upper=template.pmc_upper.copy()
        g.pmc_blocks=template.pmc_blocks
        g.faces={family:[initial(shape) for _,_,shape in blocks] for family,blocks in g.pmc_blocks.items()}
        self.face_slices=[tuple(slice(n,n+1) if a in upper else slice(0,n) for a,n in enumerate(r.shape))
                          for _,upper,_ in g.pmc_blocks['E']]
        if prepare_updates and prepare_permittivity:self.configure_material()
        else:g.inverse_permittivity=None;g.face_inverse_permittivity=[]
        g.metric={key:(self.tensor(value),edge) for key,(value,edge) in template.metric.items()}
        g.cpml={}
        self.segments=[]
        self.keys={}
        for key,segs in template.cpml.items():
            g.cpml[key]=[]
            self.keys[key]=[]
            for seg in segs:
                item={'slice':seg['slice'], 'psi':initial(seg['shape']),
                      **{name:self.tensor(seg[name]) for name in ('b','c','inv_k')}}
                self.keys[key].append(len(self.segments))
                self.segments.append(item)
                g.cpml[key].append(item)
        self.sources={'E':[],'H':[]}
        self.face_sources={'E':[],'H':[]}
        if any(s.enabled and s.injection=='oneway' for s in project.sources):
            validate_oneway_materials(project,epsilon.detach().cpu().numpy(),np.broadcast_to(np.array(-1,dtype=np.int32),r.shape))
        for source,items in terms:
            for component,loc,waveform,profile in items:
                family=component[0];comp='xyz'.index(component[1].lower())
                if self.pmc and any((sl.stop if isinstance(sl,slice) else sl+1)>n for sl,n in zip(loc,r.shape)):
                    block,index=face_index(g.pmc_blocks,r.shape,family,comp,loc)
                    self.face_sources[family].append((block,loc,index,self.tensor(waveform)))
                    continue
                self.sources[family].append((loc,comp,self.tensor(waveform),None if profile is None else self.tensor(profile)))
        self.monitors=[(m.component,index_at(m.center,r,m.component),'xyz'.index(m.component[1].lower()))
                       for m in project.monitors if m.enabled]
        self.observation_cache=observation_cache
        if observation_monitors is not None:
            # A plane model owns an immutable tuple. Preserve its identity for
            # O(1) cache checks instead of rescanning every observation at each
            # solve. Other callers retain their existing mutable-list contract.
            self.monitors=(observation_monitors if observation_cache is not None
                           and isinstance(observation_monitors,tuple)
                           else list(observation_monitors))
        self.prepare_observations()
        self.kernel=None
        if prepare_kernels and self.device.type=='cuda' and not r.complex_fields:
            from .cuda_kernels import FusedYeeCUDA
            self.kernel=FusedYeeCUDA(g)
        elif prepare_kernels and self.device.type=='cuda' and r.cuda_kernel=='fused':
            from .cuda_complex import FusedComplexYeeCUDA
            self.kernel=FusedComplexYeeCUDA(g)

    def _volume(self,epsilon):
        """Cell-indexed material rows, excluding stored upper PMC rows."""
        if not self.pmc:return epsilon
        n=self.region.shape
        return epsilon[:n[0],:n[1],:n[2]]

    def face_epsilon(self,epsilon,position):
        """Sampled epsilon at every node of one stored upper E face or edge."""
        component=self.grid.pmc_blocks['E'][position][0]
        sl=self.face_slices[position]
        return epsilon[sl+(component,)] if epsilon.ndim==4 else epsilon[sl]

    def configure_material(self,workspace=None):
        g=self.grid
        volume=self._volume(self.epsilon.detach());eps4=volume[...,None] if volume.ndim==3 else volume
        if workspace is None:g.inverse_permittivity=(1/eps4).expand_as(g.E).contiguous()
        else:
            g.inverse_permittivity=workspace.array('inverse',g.E.shape,self.dtype)
            g.inverse_permittivity.copy_(eps4.expand_as(g.E)).reciprocal_()
        faces=[]
        for position,(_,_,shape) in enumerate(g.pmc_blocks['E']):
            value=self.face_epsilon(self.epsilon.detach(),position)
            if workspace is None:faces.append((1/value).contiguous())
            else:faces.append(workspace.array(f'inverse_face:{position}',shape,self.dtype).copy_(value).reciprocal_())
        g.face_inverse_permittivity=faces

    def observation_cache_key(self,kind):
        # Stored PMC faces have topology-dependent state offsets. Keep that
        # path uncached. Plane models already reject changes to their mesh and
        # monitor configuration, and the entry also retains/checks the exact
        # tuple owner, preventing stale reuse after a layout is replaced.
        if (getattr(self,'observation_cache',None) is None
                or not isinstance(self.monitors,tuple)
                or any(self.grid.pmc_blocks.get(family) for family in ('E','H'))):
            return None
        return kind,tuple(self.region.shape)

    def prepare_observations(self):
        key=self.observation_cache_key('prepare')
        cached=None if key is None else self.observation_cache.get(key)
        if cached is not None and cached[0] is self.monitors:
            self.observation_maps=[tuple(t.to(self.device) for t in pair) for pair in cached[1]]
            self.face_observation_maps=[]
            self._setup_observation_buffers()
            return
        build_device=self.device if key is None else torch.device('cpu')
        self.observation_maps=[]
        self.face_observation_maps=[]
        shape=self.region.shape
        groups={}
        for family in ('E','H'):
            positions=[];indices=[]
            for position,(name,loc,component) in enumerate(self.monitors):
                if name[0]!=family:continue
                block,index=face_index(self.grid.pmc_blocks,shape,family,component,loc)
                if block is None:
                    positions.append(position);indices.append(index)
                else:
                    offset=2+len(self.segments)+(block if family=='E' else len(self.grid.pmc_blocks['E'])+block)
                    groups.setdefault(offset,([],[]))
                    groups[offset][0].append(position);groups[offset][1].append(index)
            self.observation_maps.append((torch.tensor(positions,device=build_device,dtype=torch.long),
                                          torch.tensor(indices,device=build_device,dtype=torch.long)))
        for offset,(positions,indices) in sorted(groups.items()):
            self.face_observation_maps.append((offset,torch.tensor(positions,device=self.device,dtype=torch.long),
                                               torch.tensor(indices,device=self.device,dtype=torch.long)))
        if key is not None:
            self.observation_cache[key]=(self.monitors,tuple(self.observation_maps))
            self.observation_maps=[tuple(t.to(self.device) for t in pair) for pair in self.observation_maps]
        self._setup_observation_buffers()

    def _setup_observation_buffers(self):
        field_dtype = self.grid.E.dtype
        self._obs_buffers = []
        self._obs_fast_slices = []
        self._trans_buffers = []
        for positions, indices in self.observation_maps:
            if indices.numel() == 0:
                self._obs_buffers.append(None)
                self._obs_fast_slices.append(None)
                self._trans_buffers.append(None)
                continue
            self._obs_buffers.append(torch.empty(indices.numel(), device=self.device, dtype=field_dtype))
            self._trans_buffers.append(torch.empty(positions.numel(), device=self.device, dtype=field_dtype))
            pos_cpu = positions.cpu()
            start = int(pos_cpu[0])
            stop = int(pos_cpu[-1]) + 1
            if len(pos_cpu) == (stop - start) and (pos_cpu == torch.arange(start, stop)).all():
                self._obs_fast_slices.append(slice(start, stop))
            else:
                self._obs_fast_slices.append(None)

        self._face_obs_buffers = []
        for offset, positions, indices in self.face_observation_maps:
            if indices.numel() > 0:
                self._face_obs_buffers.append(torch.empty(indices.numel(), device=self.device, dtype=field_dtype))
            else:
                self._face_obs_buffers.append(None)

    def tensor(self,value):
        is_complex=value.is_complex() if isinstance(value,torch.Tensor) else np.iscomplexobj(value)
        dtype=(torch.complex128 if self.dtype==torch.float64 else torch.complex64) if is_complex else self.dtype
        return torch.as_tensor(value,device=self.device,dtype=dtype)

    def state(self):
        return (self.grid.E,self.grid.H,*(seg['psi'] for seg in self.segments),*self.grid.faces['E'],*self.grid.faces['H'])

    def split_state(self,state):
        """(psi states, E face blocks, H face blocks) of a complete state tuple."""
        k=2+len(self.segments);ne=len(self.grid.pmc_blocks['E'])
        return tuple(state[2:k]),tuple(state[k:k+ne]),tuple(state[k+ne:])

    def zero(self):
        for value in self.state():value.zero_()
        self.current_step=0

    def _extended(self,field,faces,family,shape):
        """One array per component including its stored upper PMC nodes.

        Concatenation is the exact adjoint of the restriction below, so the
        face topology transposes without any separate incidence tables.
        """
        parts=[field[...,c] for c in range(3)]
        if not self.grid.pmc_upper:return parts
        blocks=self.grid.pmc_blocks[family]
        for c in range(3):
            stored={upper:faces[i] for i,(comp,upper,_) in enumerate(blocks) if comp==c}
            if not stored:continue
            axes=sorted({a for upper in stored for a in upper})
            if len(axes)==1:
                a=axes[0];parts[c]=torch.cat((parts[c],stored[(a,)]),dim=a)
            else:
                a,b=axes
                parts[c]=torch.cat((torch.cat((parts[c],stored[(b,)]),dim=b),
                                    torch.cat((stored[(a,)],stored[(a,b)]),dim=b)),dim=a)
        return parts

    def _restricted(self,parts,family,shape):
        volume=torch.stack([p[:shape[0],:shape[1],:shape[2]] for p in parts],dim=-1)
        faces=[parts[comp][tuple(slice(n,n+1) if a in upper else slice(0,n) for a,n in enumerate(shape))]
               for comp,upper,_ in self.grid.pmc_blocks[family]]
        return volume,tuple(faces)

    def _outputs(self,like,family,shape):
        if not self.grid.pmc_upper:
            result=torch.zeros_like(like)
            return result,[result[...,c] for c in range(3)]
        return None,[torch.zeros(extended_shape(shape,self.grid.pmc_upper,family,c),device=like.device,dtype=like.dtype)
                     for c in range(3)]

    def curl(self,field,psis,forward):
        if self.pmc:raise ValueError('PMC/symmetric faces require the face-aware curl; this caller does not implement stored upper faces.')
        result,_,updated=self.curl_faces(field,(),psis,forward)
        return result,updated

    def curl_faces(self,field,faces,psis,forward):
        """Yee curl on volume arrays plus stored PMC faces, with CPML memories."""
        g=self.grid;shape=tuple(field.shape[:3])
        source,target=('E','H') if forward else ('H','E')
        inputs=self._extended(field,faces,source,shape)
        result,outputs=self._outputs(field,target,shape)
        updated=list(psis)
        for axis,component,out,sign in CURL_TERMS:
            n=shape[axis]
            if n==1:continue
            src=inputs[component];dst=outputs[out]
            derivative=src[_slice(axis,slice(1,n))]-src[_slice(axis,slice(0,n-1))]
            metric=g.metric.get((forward,axis))
            if metric:derivative=derivative*metric[0]
            for i in self.keys[forward,axis,component]:
                seg=self.segments[i];sl=seg['slice']
                data=derivative[sl]
                psi=seg['b']*psis[i]+seg['c']*data
                updated[i]=psi
                derivative=derivative.clone()
                derivative[sl]=data*seg['inv_k']+psi
            dst[_slice(axis,slice(0,n-1) if forward else slice(1,n))]+=sign*derivative
            if axis in g.wrap:
                phase=g.wrap[axis]
                first,last=src[_slice(axis,0)],src[_slice(axis,n-1)]
                edge=phase*first-last if forward else first-last/phase
                if metric:edge=edge*metric[1]
                dst[_slice(axis,n-1 if forward else 0)]+=sign*edge
            elif forward and axis in g.pmc_upper:
                dst[_slice(axis,n-1)]+=sign*g.pmc_upper[axis]*(src[_slice(axis,n)]-src[_slice(axis,n-1)])
            elif forward and axis in g.pec_upper:
                dst[_slice(axis,n-1)]-=sign*g.pec_upper[axis]*src[_slice(axis,n-1)]
            elif not forward:
                # Mirrored half cells: the wall derivative is +-2 H_wall / dx_wall.
                if axis in g.pmc_lower:dst[_slice(axis,0)]+=sign*2*g.pmc_lower[axis]*src[_slice(axis,0)]
                if axis in g.pmc_upper:dst[_slice(axis,n)]-=sign*2*g.pmc_upper[axis]*src[_slice(axis,n-1)]
        if result is None:result,faces_out=self._restricted(outputs,target,shape)
        else:faces_out=()
        return result,faces_out,tuple(updated)

    def curl_transpose(self,bar,psi_bar,forward):
        if self.pmc:raise ValueError('PMC/symmetric faces require the face-aware curl transpose; this caller does not implement stored upper faces.')
        result,_,previous=self.curl_faces_transpose(bar,(),psi_bar,forward)
        return result,previous

    def curl_faces_transpose(self,bar,faces_bar,psi_bar,forward):
        g=self.grid;shape=tuple(bar.shape[:3])
        source,target=('E','H') if forward else ('H','E')
        outputs=self._extended(bar,faces_bar,target,shape)
        result,inputs=self._outputs(bar,source,shape)
        previous=list(psi_bar)
        for axis,component,out,sign in CURL_TERMS:
            n=shape[axis]
            if n==1:continue
            dst=outputs[out];src=inputs[component]
            derivative=sign*dst[_slice(axis,slice(0,n-1) if forward else slice(1,n))]
            for i in self.keys[forward,axis,component]:
                seg=self.segments[i];sl=seg['slice']
                value=derivative[sl]
                memory_bar=psi_bar[i]+value
                previous[i]=seg['b']*memory_bar
                derivative=derivative.clone()
                derivative[sl]=value*seg['inv_k']+seg['c']*memory_bar
            metric=g.metric.get((forward,axis))
            if metric:derivative=derivative*metric[0]
            src[_slice(axis,slice(1,n))]+=derivative
            src[_slice(axis,slice(0,n-1))]-=derivative
            if axis in g.wrap:
                edge=sign*dst[_slice(axis,n-1 if forward else 0)]
                if metric:edge=edge*metric[1]
                phase=g.wrap[axis]
                src[_slice(axis,0)]+=np.conj(phase)*edge if forward else edge
                src[_slice(axis,n-1)]-=edge if forward else np.conj(1/phase)*edge
            elif forward and axis in g.pmc_upper:
                edge=sign*g.pmc_upper[axis]*dst[_slice(axis,n-1)]
                src[_slice(axis,n)]+=edge
                src[_slice(axis,n-1)]-=edge
            elif forward and axis in g.pec_upper:
                src[_slice(axis,n-1)]-=sign*g.pec_upper[axis]*dst[_slice(axis,n-1)]
            elif not forward:
                if axis in g.pmc_lower:src[_slice(axis,0)]+=sign*2*g.pmc_lower[axis]*dst[_slice(axis,0)]
                if axis in g.pmc_upper:src[_slice(axis,n-1)]-=sign*2*g.pmc_upper[axis]*dst[_slice(axis,n)]
        if result is None:result,faces_out=self._restricted(inputs,source,shape)
        else:faces_out=()
        return result,faces_out,tuple(previous)

    def inject(self,value,family,step,*,functional=False):
        if functional and self.sources[family]:value=value.clone()
        for loc,component,wave,profile in self.sources[family]:
            amplitude=wave[step] if profile is None else wave[step]*profile
            value[loc+(component,)]=value[loc+(component,)]+amplitude
        return value

    def inject_faces(self,faces,family,step,*,functional=False):
        faces=list(faces)
        for block,_,index,wave in self.face_sources[family]:
            if functional:faces[block]=faces[block].clone()
            faces[block].reshape(-1)[index]+=wave[step]
        return tuple(faces)

    def reference_step(self,state,step,epsilon):
        e,h=state[:2];psis,fe,fh=self.split_state(state)
        ce,cf,psis=self.curl_faces(h,fh,psis,False)
        volume=self._volume(epsilon)
        eps=volume[...,None] if epsilon.ndim==3 else volume
        courant=self.grid.courant_number
        e=self.inject(e+courant/eps*ce,'E',step,functional=True)
        fe=self.inject_faces([f+courant/self.face_epsilon(epsilon,i)*v for i,(f,v) in enumerate(zip(fe,cf))],'E',step,functional=True)
        ch,cf,psis=self.curl_faces(e,fe,psis,True)
        h=self.inject(h-courant*ch,'H',step,functional=True)
        fh=self.inject_faces([f-courant*v for f,v in zip(fh,cf)],'H',step,functional=True)
        return (e,h,*psis,*fe,*fh)

    def advance(self,start,end):
        if self.grid.inverse_permittivity is None:
            raise RuntimeError('Storage-only initial states cannot be advanced in place. Use a tile operator.')
        for step in range(start,end):
            if self.kernel is not None:
                self.kernel.update_E();self.inject(self.grid.E,'E',step);self.inject_faces(self.grid.faces['E'],'E',step)
                self.kernel.update_H();self.inject(self.grid.H,'H',step);self.inject_faces(self.grid.faces['H'],'H',step)
            else:
                new=self.reference_step(self.state(),step,self.epsilon)
                for target,value in zip(self.state(),new):target.copy_(value)
        self.current_step=end

    def observe(self, state=None, out=None):
        if out is None:
            st = state if state is not None else self.state()
            result = st[0].new_empty(len(self.monitors))
            for field, (positions, indices) in zip(st[:2], self.observation_maps):
                if indices.numel():
                    result.index_copy_(0, positions, field.reshape(-1).index_select(0, indices))
            for offset, positions, indices in self.face_observation_maps:
                result.index_copy_(0, positions, st[offset].reshape(-1).index_select(0, indices))
            return result

        fields = (self.grid.E, self.grid.H) if state is None else state[:2]
        obs_buffers = getattr(self, '_obs_buffers', None)
        obs_fast = getattr(self, '_obs_fast_slices', None)
        if obs_buffers is None:
            self._setup_observation_buffers()
            obs_buffers = self._obs_buffers
            obs_fast = self._obs_fast_slices

        for field, (positions, indices), buf, fast_slice in zip(
            fields, self.observation_maps, obs_buffers, obs_fast
        ):
            if indices.numel() == 0:
                continue
            flat_field = field.reshape(-1)
            if fast_slice is not None:
                torch.index_select(flat_field, 0, indices, out=out[fast_slice])
            else:
                torch.index_select(flat_field, 0, indices, out=buf)
                out.index_copy_(0, positions, buf)

        for (offset, positions, indices), buf in zip(self.face_observation_maps, getattr(self, '_face_obs_buffers', ())):
            if indices.numel() == 0:
                continue
            src = self.state()[offset] if state is None else state[offset]
            torch.index_select(src.reshape(-1), 0, indices, out=buf)
            out.index_copy_(0, positions, buf)

        return out

    def transpose_step(self, state, adjoint, signal_bar):
        e_bar, h_bar = adjoint[:2]; psi_bar, fe_bar, fh_bar = self.split_state(adjoint)
        # Every point sample is taken after both source injections of this step.
        # Multiple monitors may coincide, so their adjoints must accumulate.
        trans_bufs = getattr(self, '_trans_buffers', (None, None))
        for target, (positions, indices), buf in zip((e_bar, h_bar), self.observation_maps, trans_bufs):
            if indices.numel():
                if buf is not None and buf.shape == positions.shape:
                    torch.index_select(signal_bar, 0, positions, out=buf)
                    target.reshape(-1).index_add_(0, indices, buf)
                else:
                    target.reshape(-1).index_add_(0, indices, signal_bar.index_select(0, positions))
        for offset, positions, indices in self.face_observation_maps:
            adjoint[offset].reshape(-1).index_add_(0, indices, signal_bar.index_select(0, positions))
        courant=self.grid.courant_number
        contribution,faces,psi_bar=self.curl_faces_transpose(-courant*h_bar,[-courant*f for f in fh_bar],psi_bar,True)
        e_bar=e_bar+contribution
        fe_bar=[f+v for f,v in zip(fe_bar,faces)]
        _,_,fh_state=self.split_state(state)
        curl_e,curl_faces,_=self.curl_faces(state[1],fh_state,state[2:2+len(self.segments)],False)
        full=-courant*(e_bar.conj()*curl_e).real/self.eps4.square()
        volume=full.sum_to_size(self.eps4.shape)
        if self.epsilon.ndim==3:volume=volume[...,0]
        if self.pmc:
            gradient=torch.zeros(self.epsilon.shape,device=self.device,dtype=self.dtype)
            n=self.region.shape
            gradient[:n[0],:n[1],:n[2]]=volume
            for position,(component,_,_) in enumerate(self.grid.pmc_blocks['E']):
                part=-courant*(fe_bar[position].conj()*curl_faces[position]).real/self.face_epsilon(self.epsilon,position).square()
                sl=self.face_slices[position]
                if self.epsilon.ndim==4:gradient[sl+(component,)]+=part
                else:gradient[sl]+=part
        else:gradient=volume
        scaled=[courant/self.face_epsilon(self.epsilon,position)*f for position,f in enumerate(fe_bar)]
        contribution,faces,psi_bar=self.curl_faces_transpose(courant/self.eps4*e_bar,scaled,psi_bar,False)
        fh_bar=[f+v for f,v in zip(fh_bar,faces)]
        return (e_bar,h_bar+contribution,*psi_bar,*fe_bar,*fh_bar),gradient


class _Checkpoints:
    def __init__(self,system,options,report,*,admission=False):
        self.system,self.options,self.report=system,options,report
        self.values={};self.times={};self.prefetched={};self.serial=0;self.directory=None;self.staging=None
        self.size=sum(x.numel()*x.element_size() for x in system.state())
        if options.storage=='hierarchical':
            disk=options.checkpoints-options.device_checkpoints-options.host_checkpoints
            # Outer recursion has the longest reuse distance. Keep those states
            # on slower tiers and the innermost, frequently reused states nearby.
            self.tiers=['disk']*disk+['host']*options.host_checkpoints+['device']*options.device_checkpoints
        else:self.tiers=[options.storage]*options.checkpoints
        disk_count=self.tiers.count('disk');host_count=self.tiers.count('host')
        asynchronous=options.checkpoint_transfers=='async' and (disk_count or host_count)
        host_bytes=self.size*(host_count+(options.staging_slots if asynchronous else 0)+(1 if disk_count else 0))
        archive_bound=self.size+4096+512*len(system.state())
        if options.host_budget_bytes is not None and host_bytes>options.host_budget_bytes:
            raise ValueError('Host checkpoint budget cannot hold its slots and disk staging state.')
        if disk_count and archive_bound*disk_count>options.disk_budget_bytes:
            raise ValueError('Disk checkpoint budget must include its slots and archive headers.')
        if disk_count:
            root=Path(options.checkpoint_directory).resolve()
            root.mkdir(parents=True,exist_ok=True)
            self.directory=Path(tempfile.mkdtemp(prefix='torchfdtd-adjoint-',dir=root)).resolve()
            if self.directory.parent!=root:raise RuntimeError('Checkpoint path escaped the selected directory.')
        report.update(restart_bytes=self.size,checkpoint_capacity=options.checkpoints,
                      checkpoint_storage=options.storage,peak_checkpoints=0,checkpoint_bytes_written=0,
                      checkpoint_bytes_read=0,checkpoint_io_seconds=0.,replayed_steps=0,
                      checkpoint_tiers=self.tiers.copy(),peak_checkpoints_by_tier={t:0 for t in ('device','host','disk')})
        if asynchronous and not admission:
            from .staging import AsyncStateStaging
            self.staging=AsyncStateStaging(system.state(),options.staging_slots,report)

    def save(self):
        if len(self.values)>=self.options.checkpoints:raise RuntimeError('Checkpoint scheduler exceeded its slot budget.')
        started=time.perf_counter();key=self.serial;self.serial+=1
        state=self.system.state()
        tier=self.tiers[len(self.values)]
        if self.staging is not None and tier!='device':
            path=self.directory/f'{key}.npz' if tier=='disk' else None
            value=self.staging.save(state,path=path,max_file_bytes=self.size+4096+512*len(state))
        elif tier=='device':
            value=tuple(x.clone() for x in state)
        elif tier=='host':
            value=tuple(x.detach().to('cpu',copy=True) for x in state)
        else:
            value=self.directory/f'{key}.npz'
            # Uncompressed arrays preserve exact field bits. Only this owned
            # file is later removed. No existing user directory is deleted.
            try:
                np.savez(value,**{f's{i}':x.detach().cpu().numpy() for i,x in enumerate(state)})
                if sum(p.stat().st_size for p in self.directory.iterdir())>self.options.disk_budget_bytes:
                    raise ValueError('Disk checkpoint budget must include archive headers.')
            except BaseException:
                value.unlink(missing_ok=True)
                raise
        self.values[key]=(tier,value)
        self.times[key]=self.system.current_step
        self.report['peak_checkpoints']=max(self.report['peak_checkpoints'],len(self.values))
        count=sum(t==tier for t,_ in self.values.values())
        self.report['peak_checkpoints_by_tier'][tier]=max(self.report['peak_checkpoints_by_tier'][tier],count)
        self.report['checkpoint_bytes_written']+=self.size
        self.report['checkpoint_io_seconds']+=time.perf_counter()-started
        return key

    def restore(self,key):
        target_time=0 if key is None else self.times[key]
        if self.system.current_step==target_time:
            if key in self.prefetched:self.staging.consume(self.prefetched.pop(key))
            return
        if key is None:
            self.system.zero();return
        started=time.perf_counter();tier,value=self.values[key]
        if self.staging is not None and tier!='device':
            ticket=self.prefetched.pop(key,None)
            if ticket is None:ticket=self.staging.load(value,disk=tier=='disk')
            self.staging.consume(ticket,self.system.state())
        elif tier=='disk':
            with np.load(value,allow_pickle=False) as archive:
                for i,target in enumerate(self.system.state()):
                    target.copy_(torch.from_numpy(archive[f's{i}']))
        else:
            for target,source in zip(self.system.state(),value):target.copy_(source)
        self.report['checkpoint_bytes_read']+=self.size
        self.report['checkpoint_io_seconds']+=time.perf_counter()-started
        self.system.current_step=target_time

    def prefetch(self,key):
        if self.staging is None or key is None or key in self.prefetched:return
        tier,value=self.values[key]
        if tier!='device':self.prefetched[key]=self.staging.load(value,disk=tier=='disk')

    def drop(self,key):
        error=None
        if key in self.prefetched:
            try:self.staging.consume(self.prefetched.pop(key))
            except BaseException as exc:error=exc
        tier,value=self.values.pop(key)
        self.times.pop(key)
        future=value if hasattr(value,'result') else None
        try:
            if future is not None:value=future.result()
            if tier=='disk':value.unlink()
        except BaseException as exc:
            if error is None:error=exc
        finally:
            if self.staging is not None and future is not None:self.staging.forget(future)
        if error is not None:raise error

    def close(self):
        error=None
        for key in list(self.values):
            try:self.drop(key)
            except BaseException as exc:
                if error is None:error=exc
        if self.staging is not None:
            try:self.staging.close()
            except BaseException as exc:
                if error is None:error=exc
        if self.directory is not None:self.directory.rmdir()
        if error is not None:raise error


def _split(length,slots):
    """Binomial-capacity split, with a bounded depth in checkpoint slots."""
    # The interval's restart state is available in addition to the free slots.
    # At the root it is the implicit zero state. Count that level as well, so
    # even one saved checkpoint reduces replay instead of repeatedly saving
    # the penultimate state. Pascal's identity gives the left/right capacities.
    repetitions=0
    while math.comb(slots+repetitions+1,slots+1)<length:repetitions+=1
    return min(length-1,math.comb(slots+repetitions,slots+1))


class _FDTD(torch.autograd.Function):
    @staticmethod
    def forward(ctx,epsilon,system,options,report,spectral):
        ctx.save_for_backward(epsilon)
        ctx.system,ctx.options,ctx.report=system,options,report
        ctx.spectral=spectral
        from .telemetry import get_run_dir, hash_array
        run_dir = get_run_dir()
        track_telemetry = run_dir is not None

        restart_dir = (
            getattr(options, 'restart_directory', None)
            or os.environ.get('TORCHFDTD_RESUME_JOURNAL')
            or os.environ.get('TORCHFDTD_SAVE_JOURNAL')
        )
        if restart_dir is None and track_telemetry and os.environ.get('TORCHFDTD_CHECKPOINT') == '1':
            restart_dir = run_dir / 'journal'

        journal = None
        resumed_from_step = 0
        peak_energy = 0.0
        end_energy = 0.0

        enable_auto_shutoff = bool(
            getattr(options, 'auto_shutoff', False)
            or (getattr(system.region.run_control, 'auto_shutoff', False) and getattr(options, 'allow_auto_shutoff', False))
        )
        decay_decision = None
        auto_shutoff_triggered = False
        shutoff_step = None
        shutoff_energy_ratio = 0.0
        shutoff_flux_delta = 0.0

        if enable_auto_shutoff:
            from .run_control import source_end_time, DecayDecision
            source_end = source_end_time(system.project)
            if math.isfinite(source_end):
                decay_decision = DecayDecision(system.region.run_control, source_end, system.region.time_step)
            else:
                enable_auto_shutoff = False

        if track_telemetry or restart_dir is not None or enable_auto_shutoff:
            sample_interval = max(1, system.region.steps // 200)
            r = system.region
            widths = [np.diff(nodes)/r.reference_step if n > 1 else np.ones(1)
                      for nodes, n in zip(r.mesh_nodes, r.shape)]
            vol_np = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
            vol_tensor = torch.as_tensor(vol_np, device=epsilon.device, dtype=epsilon.dtype)[..., None]

        signals = system.grid.E.new_empty((system.region.steps, len(system.monitors))) if spectral is None else spectral.zeros()
        samples = None if spectral is None else system.grid.E.new_empty((spectral.block_size, len(system.monitors)))
        sample_views = None if spectral is None else tuple(samples[i] for i in range(spectral.block_size))

        if restart_dir is not None:
            from .resident_restart import ResidentRestartJournal, build_resident_contract
            contract = build_resident_contract(system, epsilon, spectral, options)
            journal = ResidentRestartJournal(
                restart_dir,
                contract=contract,
                every_blocks=getattr(options, 'restart_every_blocks', 64),
                every_seconds=getattr(options, 'restart_every_seconds', None),
            )
            if journal.has_checkpoint():
                resumed_step, resumed_blk, loaded_peak, is_complete = journal.load_latest(system.state(), signals)
                if is_complete:
                    if epsilon.is_cuda: torch.cuda.synchronize(epsilon.device)
                    return signals
                resumed_from_step = resumed_step
                peak_energy = loaded_peak

        if resumed_from_step == 0:
            system.zero()

        started = time.perf_counter()
        completed_step = resumed_from_step
        for step in range(resumed_from_step, system.region.steps):
            if getattr(options, 'stop_at_step', None) is not None and step >= options.stop_at_step:
                break
            system.advance(step, step + 1)
            completed_step = step + 1
            if spectral is None:
                system.observe(out=signals[step])
            else:
                local = step % spectral.block_size
                system.observe(out=sample_views[local])
                if local + 1 == spectral.block_size or step + 1 == system.region.steps:
                    spectral.accumulate(signals, samples[:local + 1], step - local)
                    completed_block = completed_step // spectral.block_size
                    if journal is not None and journal.should_commit(completed_block, time.time()):
                        journal.save_checkpoint(completed_step, completed_block, system.state(), signals, peak_energy)

                    if decay_decision is not None and (local + 1 == spectral.block_size or step + 1 == system.region.steps):
                        e_sq = (system.grid.E.real**2 + system.grid.E.imag**2) if system.grid.E.is_complex() else system.grid.E**2
                        h_sq = (system.grid.H.real**2 + system.grid.H.imag**2) if system.grid.H.is_complex() else system.grid.H**2
                        cur_e = float(((e_sq * system.eps4 + h_sq) * vol_tensor).sum().item())
                        if cur_e > peak_energy: peak_energy = cur_e

                        cur_fluxes = (signals.real**2 + signals.imag**2).sum(dim=0).cpu().tolist()
                        if decay_decision.update_with_flux(step, cur_e, field_peak=0.0, monitor_fluxes=cur_fluxes):
                            auto_shutoff_triggered = True
                            shutoff_step = completed_step
                            shutoff_energy_ratio = decay_decision.last_energy_ratio
                            shutoff_flux_delta = decay_decision.last_flux_delta
                            end_energy = cur_e
                            break

            if (track_telemetry or journal is not None) and (step % sample_interval == 0 or step + 1 == system.region.steps):
                e_sq = (system.grid.E.real**2 + system.grid.E.imag**2) if system.grid.E.is_complex() else system.grid.E**2
                h_sq = (system.grid.H.real**2 + system.grid.H.imag**2) if system.grid.H.is_complex() else system.grid.H**2
                cur_e = float(((e_sq * system.eps4 + h_sq) * vol_tensor).sum().item())
                if cur_e > peak_energy: peak_energy = cur_e
                if step + 1 == system.region.steps: end_energy = cur_e

        if journal is not None:
            if completed_step == system.region.steps:
                journal.complete()
            else:
                journal.close()

        if epsilon.is_cuda: torch.cuda.synchronize(epsilon.device)
        report['forward_seconds'] = time.perf_counter() - started
        ctx.actual_steps = completed_step
        if track_telemetry or journal is not None or resumed_from_step > 0 or auto_shutoff_triggered:
            executed_steps = completed_step - resumed_from_step
            report['telemetry_kernel'] = {
                'executed_steps': executed_steps,
                'resumed_from_step': resumed_from_step,
                'energy_decay_ratio': float(end_energy / peak_energy) if peak_energy > 0 else 0.0,
                'is_lossless': bool(not epsilon.is_complex() or bool((epsilon.imag == 0).all().item())),
                'epsilon_sha256': hash_array(epsilon),
                'run_dir': run_dir,
                'auto_shutoff_triggered': auto_shutoff_triggered,
                'shutoff_step': shutoff_step,
                'shutoff_energy_ratio': shutoff_energy_ratio,
                'shutoff_flux_delta': shutoff_flux_delta,
                'execution_profile': getattr(options, 'execution_profile', 'auto'),
            }
        return signals

    @staticmethod
    def backward(ctx,signal_bar):
        if torch.is_grad_enabled():
            raise RuntimeError('Higher-order derivatives are not supported by the discrete adjoint yet.')
        epsilon,=ctx.saved_tensors
        system,options,report=ctx.system,ctx.options,ctx.report
        started=time.perf_counter()
        gradient=torch.zeros(epsilon.shape,device=epsilon.device,dtype=epsilon.dtype)
        fused=None
        if system.pmc and options.backward_kernel=='fused':
            raise ValueError('The fused CUDA backward kernel does not implement PMC/symmetric faces. Use backward_kernel="auto" or "torch".')
        if epsilon.is_cuda and not system.pmc and options.backward_kernel!='torch' and (not system.region.complex_fields or options.backward_kernel=='fused'):
            if system.region.complex_fields:
                from .cuda_complex_adjoint import FusedComplexAdjointCUDA as Kernel
            else:
                from .cuda_adjoint import FusedAdjointCUDA as Kernel
            fused_seed=signal_bar if ctx.spectral is None else torch.empty((ctx.spectral.block_size,len(system.monitors)),device=epsilon.device,dtype=system.field_dtype)
            factory=getattr(system,'fused_adjoint',None)
            fused=Kernel(system,gradient,fused_seed) if factory is None else factory(gradient,fused_seed)
        adjoint=None if fused else tuple(torch.zeros_like(x) for x in system.state())
        checkpoints=_Checkpoints(system,options,report)

        def replay(key,start,end):
            checkpoints.restore(key)
            system.advance(start,end)
            report['replayed_steps']+=end-start

        spectral_start=-1
        spectral_seed=None
        def reverse(start,end,key,slots):
            nonlocal adjoint,spectral_start,spectral_seed
            # The left recursion is a loop. Only the right branch consumes a
            # checkpoint slot, so Python call depth never grows with duration.
            while end>start:
                if end-start==1 or slots==0:
                    for step in range(end-1,start-1,-1):
                        replay(key,start,step)
                        if step>start:checkpoints.prefetch(key)
                        if ctx.spectral is None:
                            seed=signal_bar[step]
                            observation_index=None
                        else:
                            begin=(step//ctx.spectral.block_size)*ctx.spectral.block_size
                            if begin!=spectral_start:
                                spectral_seed=ctx.spectral.transpose(signal_bar,begin,min(begin+ctx.spectral.block_size,system.region.steps))
                                spectral_start=begin
                                if fused:fused.signal_bar[:len(spectral_seed)].copy_(spectral_seed)
                            observation_index=step-begin
                            seed=spectral_seed[observation_index]
                        if fused:
                            fused.step(step,observation_index=observation_index)
                        elif hasattr(system,"transpose_accumulate"):
                            adjoint=system.transpose_accumulate(system.state(),adjoint,seed,gradient,step)
                        else:
                            adjoint,part=system.transpose_step(system.state(),adjoint,seed)
                            gradient.add_(part)
                    return
                middle=start+_split(end-start,slots)
                replay(key,start,middle)
                saved=checkpoints.save()
                try:reverse(middle,end,saved,slots-1)
                finally:checkpoints.drop(saved)
                end=middle

        try:
            actual_steps = getattr(ctx, 'actual_steps', system.region.steps)
            reverse(0, actual_steps, None, options.checkpoints)
            if fused is not None and hasattr(fused,'finalize'):fused.finalize(report)
            if epsilon.is_cuda:torch.cuda.synchronize(epsilon.device)
            report['backward_seconds']=time.perf_counter()-started
        finally:
            # Break the recursive closure cycle so solver buffers do not wait
            # for cyclic GC after the caller releases its result graph.
            reverse=None
            checkpoints.close()
        return gradient,None,None,None,None


class DifferentiableSimulation(torch.nn.Module):
    """Torch-native epsilon-to-point-signal operation with a discrete adjoint.

    Epsilon has shape (Nx,Ny,Nz), or (Nx,Ny,Nz,3) for diagonal Yee values.
    Geometry uses the existing micrometre convention. This is an experimental
    fixed-mesh design API, not a claim that every forward feature differentiates.
    """
    _explicit_dispersive_parameters = False

    def __init__(self,project: Project,options: AdjointOptions | None=None):
        super().__init__()
        self.project=Project.model_validate(project.model_dump())
        self.options=options or AdjointOptions()
        p=self.project;r=p.region
        if r.interface_method!='staircase':
            raise ValueError('DifferentiableSimulation currently requires staircase coefficients.')
        reject_pml_dispersion(p,type(self).__name__,self._explicit_dispersive_parameters)
        active={s.material for s in p.structures if s.enabled}
        if not self._explicit_dispersive_parameters and any(m.oscillators and m.name in active for m in p.materials):
            raise ValueError('Dispersive ADE derivatives are not implemented yet.')
        if any(s.enabled and s.kind=='tfsf' for s in p.sources):
            raise ValueError('Live TFSF incident-state derivatives are not implemented yet.')
        if not any(m.enabled for m in p.monitors) or any(m.enabled and m.kind!='point' for m in p.monitors):
            raise ValueError('The differentiable API currently requires point monitors only.')
        if r.run_control.auto_shutoff and not getattr(self.options, 'allow_auto_shutoff', False) and not getattr(self.options, 'auto_shutoff', False):
            raise ValueError('Differentiable simulations require a fixed number of timesteps.')
        # The scene and its realized mesh are fixed here; _run rejects any later change.
        self._plan_snapshot=project_snapshot(self.project)

    @cached_property
    def plan(self):
        """The resolved plan of the fixed project (torchfdtd.plan.resolve_plan), shared with every entry point."""
        from .plan import resolve_plan
        return resolve_plan(self.project)

    @property
    def plan_hash(self):
        return self.plan.plan_hash

    def spectrum(self,epsilon,frequency_hz,*,window=None,block_size=32):
        """Online DFT with fixed settings and a bounded observation transpose."""
        from .adjoint_spectrum import SpectralObservation
        spectral=SpectralObservation(epsilon,self.project.region,
                                     [m.component for m in self.project.monitors if m.enabled],frequency_hz,window,block_size)
        return self._run(epsilon,spectral)

    def forward(self,epsilon: torch.Tensor):
        return self._run(epsilon,None)

    def _run(self,epsilon,spectral,*,system_factory=None,autograd_input=None,
             pole_count=0,material_parameter_elements=0):
        check_snapshot(self.project,self._plan_snapshot,type(self).__name__)
        r=self.project.region
        from .adjoint_memory import _resident_contract
        _resident_contract(r,self.options)
        if not isinstance(epsilon,torch.Tensor) or epsilon.dtype not in (torch.float32,torch.float64):
            raise ValueError('epsilon must be a real float32 or float64 torch Tensor.')
        if epsilon.device.type not in ('cpu','cuda'):
            raise ValueError('Only CPU and CUDA tensors are supported.')
        from .endpoint_native import uses_endpoint
        pmc=uses_endpoint(r)
        if pmc and self.options.backward_kernel=='fused':
            raise ValueError('The fused CUDA backward kernel does not implement PMC/symmetric faces. Use backward_kernel="auto" or "torch".')
        if epsilon.device.type!='cuda' and self.options.backward_kernel=='fused':
            raise ValueError('The fused backward requires a CUDA tensor.')
        if epsilon.device.type!='cuda' and self.options.checkpoint_transfers=='async':
            raise ValueError('Asynchronous checkpoints require a CUDA tensor.')
        if tuple(epsilon.shape) not in (material_shape(r),material_shape(r,True)):
            raise ValueError('epsilon shape must match the scene grid plus one stored row on every upper PMC/symmetric axis, optionally with three Yee components.'
                             if pmc else 'epsilon shape must match the scene grid, optionally with three Yee components.')
        if not bool(torch.isfinite(epsilon).all()) or bool((epsilon<1).any()):
            raise ValueError('This conservative CFL contract requires finite epsilon >= 1.')
        if (epsilon.dtype==torch.float64)!=(r.precision=='float64'):
            raise ValueError('epsilon dtype must match the project precision.')
        # Check GPU, CPU working memory and checkpoint tiers before creating
        # native fields. The public estimator uses this same calculation.
        from .adjoint_memory import _resident_reservation
        reservation=_resident_reservation(self.project,self.options,epsilon.device,spectral,
            pole_count=pole_count,parameter_elements=material_parameter_elements)
        report=dict(experimental=True,adjoint='discrete Yee/CPML',higher_order=False,
                    spatial_streaming=False,full_time_autograd=False,steps=r.steps,
                    forward_backend='fused CUDA' if epsilon.is_cuda and (not r.complex_fields or r.cuda_kernel=='fused') else 'torch CUDA' if epsilon.is_cuda else 'torch CPU',
                    backward_backend='fused CUDA complex transpose' if epsilon.is_cuda and r.complex_fields and self.options.backward_kernel=='fused' else 'fused CUDA transpose' if epsilon.is_cuda and not r.complex_fields and not pmc and self.options.backward_kernel!='torch' else 'torch explicit transpose',
                    pmc_faces=pmc,
                    checkpoint_transfers=self.options.checkpoint_transfers,**reservation)
        # A later wavelength/ray configuration must not alter an earlier graph's replay.
        factory=system_factory or _System
        with torch.no_grad():system=factory(self.project.model_copy(deep=True),epsilon,observation_monitors=None if spectral is None else spectral.observers)
        # Check host/disk admission before spending the forward compute time.
        admission=_Checkpoints(system,self.options,report,admission=True)
        admission.close()
        report['observation_storage']='time_history' if spectral is None else 'online_spectrum'
        if spectral is not None:report.update(spectral.reservation(spectral.block_size),spectral_block_size=spectral.block_size)
        signals=_FDTD.apply(epsilon if autograd_input is None else autograd_input,system,self.options,report,spectral)
        if spectral is not None:return spectral.result(signals,report)
        k_telem = report.get('telemetry_kernel')
        if k_telem and k_telem.get('run_dir'):
            from .telemetry import emit_telemetry, hash_array
            monitors_data = {}
            for idx, m in enumerate(self.project.monitors):
                if m.enabled:
                    sig = signals[:, idx].detach().cpu().numpy()
                    monitors_data[m.id] = {
                        "flux": 0.0,
                        "digest": hash_array(sig),
                    }
            emit_telemetry(
                k_telem['run_dir'],
                project_canonical_sha256=self.plan.plan_hash,
                epsilon_sha256=k_telem['epsilon_sha256'],
                is_lossless=k_telem['is_lossless'],
                executed_steps=k_telem['executed_steps'],
                resumed_from_step=k_telem.get('resumed_from_step', 0),
                energy_decay_ratio=k_telem['energy_decay_ratio'],
                monitors=monitors_data,
                auto_shutoff_triggered=k_telem.get('auto_shutoff_triggered', False),
                shutoff_step=k_telem.get('shutoff_step'),
                shutoff_energy_ratio=k_telem.get('shutoff_energy_ratio', 0.0),
                shutoff_flux_delta=k_telem.get('shutoff_flux_delta', 0.0),
                execution_profile=k_telem.get('execution_profile', 'reference'),
            )
        return DifferentiableResult(signals,r.time_step,tuple(m.component for m in self.project.monitors if m.enabled),report)

    def reference(self,epsilon,*,graph_budget_bytes=None):
        """Small-problem full-autograd oracle. Not the large-simulation path.

        The retained graph is admitted by its estimated memory against free
        device or host memory. graph_budget_bytes caps it further.
        """
        from .oracle_memory import admit_oracle,check_oracle_dtype,oracle_graph_bytes,oracle_source_terms
        r=self.project.region
        # The graph runs at epsilon's dtype and the estimate counts the project precision.
        check_oracle_dtype(epsilon,r)
        admit_oracle(oracle_graph_bytes(r,'yee',material_elements=epsilon.numel(),source_terms=oracle_source_terms(self.project)),
                     epsilon.device,graph_budget_bytes)
        system=_System(self.project,epsilon)
        state=tuple(torch.zeros_like(x) for x in system.state())
        signals=[]
        for step in range(system.region.steps):
            state=system.reference_step(state,step,epsilon)
            signals.append(system.observe(state))
        return torch.stack(signals)


def smooth_sphere_epsilon(region,radius,*,center=None,inside=4.,outside=1.,width=None,yee=False):
    """Differentiable sigmoid sphere, in um. A regularized geometry, not CAD subpixel.

    Radius/center/material parameters retain their Torch gradients. Decrease
    smoothing width only with an independent mesh/physical-gradient study.
    """
    if not isinstance(radius,torch.Tensor) or radius.ndim!=0 or not radius.is_floating_point():
        raise ValueError('radius must be a scalar floating-point Torch tensor.')
    width=region.reference_step if width is None else width
    if not math.isfinite(width) or width<=0:raise ValueError('Smoothing width must be positive.')
    center=radius.new_zeros(3) if center is None else torch.as_tensor(center,device=radius.device,dtype=radius.dtype)
    if center.shape!=(3,):raise ValueError('center must contain three coordinates in um.')
    if not bool(torch.isfinite(radius)) or not bool(radius>0):raise ValueError('Radius must be finite and positive.')
    def at(axes):
        coords=torch.meshgrid(*(torch.tensor(a,device=radius.device,dtype=radius.dtype) for a in axes),indexing='ij')
        square=sum((c-center[i]).square() for i,c in enumerate(coords))
        # Avoid the undefined derivative of sqrt at the sphere centre.
        distance=torch.sqrt(square+torch.finfo(radius.dtype).tiny)
        fraction=torch.sigmoid((radius-distance)/width)
        return outside+(inside-outside)*fraction
    target=material_shape(region)
    if yee:
        def padded(value):
            # Half-cell axes have no upper PMC sample; replicate an inert edge row.
            for axis,(n,t) in enumerate(zip(value.shape,target)):
                if t>n:value=torch.cat((value,value.narrow(axis,n-1,1)),dim=axis)
            return value
        return torch.stack([padded(at(field_axes(region,c))) for c in ('Ex','Ey','Ez')],dim=-1)
    return at([np.r_[(a[:-1]+a[1:])/2,a[-1:]] if t>len(a)-1 else (a[:-1]+a[1:])/2 if len(a)>2 else np.array([0.])
               for a,t in zip(region.mesh_nodes,target)])


class _BatchSystem:
    """Cohort of _System instances evaluated concurrently via FusedBatchYeeCUDA."""

    def __init__(
        self,
        systems: list[_System],
        base_project: Project,
        traces: list[torch.Tensor],
        counter: torch.Tensor,
    ):
        from .cuda_batch import FusedBatchYeeCUDA, FusedBatchIO

        self.systems = list(systems)
        self.base_project = base_project
        self.region = systems[0].region
        self.device = systems[0].device
        self.dtype = systems[0].dtype
        self.current_step = 0
        self.batch_size = len(systems)
        self.traces = traces
        self.counter = counter
        self.batch_kernel = FusedBatchYeeCUDA([s.grid for s in self.systems])
        self.batch_io = FusedBatchIO(
            [s.grid for s in self.systems],
            [base_project] * self.batch_size,
            traces,
            counter,
        )

    def state(self):
        states = []
        for s in self.systems:
            states.extend(s.state())
        return tuple(states)

    def zero(self):
        for value in self.state():
            value.zero_()
        self.counter.zero_()
        self.current_step = 0

    def advance(self, start: int, end: int):
        self.counter[0] = start
        for step in range(start, end):
            self.batch_kernel.update_E()
            self.batch_io.inject('E')
            self.batch_kernel.update_H()
            self.batch_io.inject('H')
            self.batch_io.record()
        self.current_step = end


class _BatchFDTD(torch.autograd.Function):
    """Autograd function coordinating batched forward simulation and curl-transpose backward."""

    @staticmethod
    def forward(ctx, adapter, options, report, spectral, *eps_batch):
        ctx.save_for_backward(*eps_batch)
        ctx.adapter = adapter
        ctx.options = options
        ctx.report = report
        ctx.spectral = spectral

        B = len(eps_batch)
        base_project = adapter.base_project
        device = eps_batch[0].device
        dtype = eps_batch[0].dtype

        systems = [_System(base_project, eps) for eps in eps_batch]
        num_monitors = len(systems[0].monitors)
        steps = base_project.region.steps

        traces = [
            torch.zeros((steps, num_monitors), device=device, dtype=dtype)
            for _ in range(B)
        ]
        counter = torch.zeros(1, device=device, dtype=torch.int64)

        batch_system = _BatchSystem(systems, base_project, traces, counter)
        ctx.batch_system = batch_system

        batch_system.advance(0, steps)

        signals_batch = tuple(traces[b].clone() for b in range(B))
        return signals_batch

    @staticmethod
    def backward(ctx, *signal_bars):
        if torch.is_grad_enabled():
            raise RuntimeError('Higher-order derivatives are not supported.')
        eps_batch = ctx.saved_tensors
        B = len(eps_batch)
        adapter, options, report = ctx.adapter, ctx.options, ctx.report
        batch_system = ctx.batch_system
        systems = batch_system.systems

        started = time.perf_counter()
        gradients = [
            torch.zeros(eps.shape, device=eps.device, dtype=eps.dtype)
            for eps in eps_batch
        ]

        from .cuda_batch import FusedBatchTransposeYeeCUDA

        fused = FusedBatchTransposeYeeCUDA(systems, gradients, signal_bars)
        checkpoints = _Checkpoints(batch_system, options, report)

        def replay(key, start, end):
            checkpoints.restore(key)
            batch_system.advance(start, end)
            report['replayed_steps'] = report.get('replayed_steps', 0) + (end - start)

        def reverse(start, end, key, slots):
            while end > start:
                if end - start == 1 or slots == 0:
                    for step in range(end - 1, start - 1, -1):
                        replay(key, start, step)
                        if step > start:
                            checkpoints.prefetch(key)
                        fused.step(step)
                    return
                middle = start + _split(end - start, slots)
                replay(key, start, middle)
                saved = checkpoints.save()
                try:
                    reverse(middle, end, saved, slots - 1)
                finally:
                    checkpoints.drop(saved)
                end = middle

        try:
            actual_steps = getattr(ctx, 'actual_steps', batch_system.region.steps)
            reverse(0, actual_steps, None, options.checkpoints)
            if eps_batch[0].is_cuda:
                torch.cuda.synchronize(eps_batch[0].device)
            report['backward_seconds'] = time.perf_counter() - started
        finally:
            reverse = None
            checkpoints.close()

        return (None, None, None, None, *gradients)


@lru_cache(maxsize=8192)
def count_recompute_steps(length: int, slots: int) -> int:
    """Exact binomial recompute forward steps under treeverse scheduling."""
    if length <= 1:
        return 0
    if slots == 0:
        return length * (length - 1) // 2
    middle = _split(length, slots)
    return (
        middle
        + count_recompute_steps(length - middle, slots - 1)
        + count_recompute_steps(middle, slots)
    )


def count_tier_transfers(length: int, slots: int, host_slots: int) -> int:
    """Counts host checkpoint save/restore I/O transfers during binomial traversal."""
    if slots == 0 or host_slots == 0 or length <= 1:
        return 0
    transfers = 0

    def _traverse(l, s, active_depth):
        nonlocal transfers
        if l <= 1 or s == 0:
            return
        middle = _split(l, s)
        if active_depth < host_slots:
            transfers += 2  # 1 save (D2H) + 1 restore (H2D)
        _traverse(l - middle, s - 1, active_depth + 1)
        _traverse(middle, s, active_depth)

    _traverse(length, slots, 0)
    return transfers


def select_optimal_adjoint_options(
    project: Project,
    *,
    gpu_budget_bytes: int,
    host_budget_bytes: int | None = None,
    candidate_checkpoints: tuple[int, ...] = (2, 4, 8, 12, 16, 24, 32),
    backward_kernel: str = 'fused',
    empirical_profile: dict[str, Any] | None = None,
) -> tuple[AdjointOptions, dict[str, Any]]:
    """Selects checkpoint configuration minimizing end-to-end wall time subject to VRAM budget.

    Guarantees non-inferiority to fixed default (c=4, device) whenever default is feasible.
    """
    from .adjoint_memory import _resident_reservation
    from .memory_profile import host_memory
    from .boundaries import BoundaryDescription

    if host_budget_bytes is None:
        avail = host_memory()["available_bytes"]
        host_budget_bytes = int(avail * 0.8) if avail else 32 * 1024**3

    steps = project.region.steps
    shape = project.region.shape
    real_item = 8 if project.region.precision == "float64" else 4
    item = real_item * (2 if project.region.complex_fields else 1)
    n = math.prod(shape)

    boundary = BoundaryDescription(project.region)
    segments = [s for group in boundary.cpml.values() for s in group]
    cpml = sum(math.prod(s['shape']) for s in segments)
    faces = sum(
        math.prod(fshape)
        for blocks in boundary.pmc_blocks.values()
        for _, _, fshape in blocks
    )
    state_bytes = (6 * n + cpml + faces) * item

    prof = empirical_profile or {}
    t_fwd_step = float(prof.get("t_fwd_step", 0.000144))
    t_adj_step = float(prof.get("t_adj_step", 0.000180))
    pcie_bw_gbps = float(prof.get("pcie_bw_gbps", 22.0))
    pcie_bw_bps = pcie_bw_gbps * 1e9

    feasible_candidates = []

    for c in candidate_checkpoints:
        if c > steps:
            continue

        partitions = [
            ("device", c, 0),
        ]
        for c_dev in sorted({max(1, c // 4), max(1, c // 2), max(1, 3 * c // 4)}):
            if 0 < c_dev < c:
                partitions.append(("hierarchical", c_dev, c - c_dev))
        partitions.append(("host", 0, c))

        for storage, dev_cp, host_cp in partitions:
            opt = AdjointOptions(
                checkpoints=c,
                storage=storage,
                device_checkpoints=dev_cp,
                host_checkpoints=host_cp,
                gpu_budget_bytes=gpu_budget_bytes,
                host_budget_bytes=host_budget_bytes,
                backward_kernel=backward_kernel,
            )
            try:
                mem_report = _resident_reservation(
                    project, opt, device="cuda:0", forward_admission=False
                )
                gpu_req = mem_report["gpu_reservation_bytes"]
                host_req = mem_report["host_reservation_bytes"]
            except ValueError:
                continue

            if gpu_req > gpu_budget_bytes or host_req > host_budget_bytes:
                continue

            recomp_steps = count_recompute_steps(steps, c)
            host_transfers = (
                count_tier_transfers(steps, c, host_cp) if host_cp > 0 else 0
            )
            transfer_sec = (host_transfers * state_bytes) / pcie_bw_bps

            t_pred = (
                steps * t_fwd_step
                + steps * t_adj_step
                + recomp_steps * t_fwd_step
                + transfer_sec
            )

            feasible_candidates.append({
                "options": opt,
                "checkpoints": c,
                "storage": storage,
                "device_checkpoints": dev_cp,
                "host_checkpoints": host_cp,
                "gpu_required_bytes": gpu_req,
                "host_required_bytes": host_req,
                "recompute_steps": recomp_steps,
                "predicted_wall_time_sec": t_pred,
            })

    if not feasible_candidates:
        raise ValueError(
            f"No feasible checkpoint configuration fits within gpu_budget_bytes={gpu_budget_bytes}."
        )

    feasible_candidates.sort(key=lambda x: x["predicted_wall_time_sec"])
    optimal = feasible_candidates[0]

    default_opt = AdjointOptions(
        checkpoints=4,
        storage="device",
        device_checkpoints=4,
        host_checkpoints=0,
        gpu_budget_bytes=gpu_budget_bytes,
        host_budget_bytes=host_budget_bytes,
        backward_kernel=backward_kernel,
    )
    default_latency = None
    try:
        def_mem = _resident_reservation(
            project, default_opt, device="cuda:0", forward_admission=False
        )
        if def_mem["gpu_reservation_bytes"] <= gpu_budget_bytes:
            def_recomp = count_recompute_steps(steps, 4)
            default_latency = (
                steps * t_fwd_step + steps * t_adj_step + def_recomp * t_fwd_step
            )
    except ValueError:
        pass

    report = {
        "selected": {
            "checkpoints": optimal["checkpoints"],
            "storage": optimal["storage"],
            "device_checkpoints": optimal["device_checkpoints"],
            "host_checkpoints": optimal["host_checkpoints"],
            "gpu_required_bytes": optimal["gpu_required_bytes"],
            "recompute_steps": optimal["recompute_steps"],
            "predicted_wall_time_sec": optimal["predicted_wall_time_sec"],
        },
        "default_latency_sec": default_latency,
        "non_inferior": (default_latency is None)
        or (optimal["predicted_wall_time_sec"] <= default_latency + 1e-9),
        "speedup_vs_default": (
            (default_latency / optimal["predicted_wall_time_sec"])
            if default_latency
            else "default_infeasible"
        ),
        "total_evaluated_candidates": len(feasible_candidates),
        "candidates": feasible_candidates,
    }

    return optimal["options"], report

