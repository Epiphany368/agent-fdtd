"""Experimental endpoint PMC/PEC with fixed isotropic CPML exteriors.

This explicit API is separate from native Project dispatch. Sparse operators
are the CPU reference. CUDA uses compact direct gathers, not sparse incidence.
Neither backend implements streaming or native Project dispatch.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .boundaries import CURL_TERMS
from .pmc_simulation import EndpointSimulation, _reverse_schedule
from .pmc_cuda import CompactEndpointTopology


@dataclass(frozen=True)
class CPMLState:
    electric: torch.Tensor
    magnetic: torch.Tensor
    psi: tuple


class _EndpointCPML:
    def __init__(self, closed, faces, layers, dt, background, reflection):
        self.topology=closed.topology
        self.active=closed.active
        self.operators={}
        self.forward_groups={False:[],True:[]}
        t=self.topology
        for forward in (False,True):
            source,target=('E','H') if forward else ('H','E')
            row,col,weight=closed.operators[forward]
            for axis,component,out,sign in CURL_TERMS:
                # A source/target component pair uniquely determines the curl
                # derivative axis. Preserve the endpoint reference incidence,
                # including the derivative to the omitted upper PEC zero.
                selected=(torch.tensor(t.components[source])[col]==component)&(torch.tensor(t.components[target])[row]==out)
                r,c,w=row[selected],col[selected],weight[selected]
                node=(out!=axis if target=='E' else out==axis)
                lattice=t.indices[target][:,axis].astype(float)+(0 if node else .5)
                rho=np.zeros(len(lattice))
                width=float(t.nodes[axis][1]-t.nodes[axis][0])*layers
                for side in (0,1):
                    if faces[axis][side]=='pml':
                        distance=lattice if side==0 else t.shape[axis]-lattice
                        # Uniform integer/half indices make zero-depth interface
                        # rows exact, avoiding phantom psi from coordinate roundoff.
                        rho=np.maximum(rho,np.maximum(1-distance/layers,0))
                use=(rho>0)&(t.components[target]==out)&t.active[target]
                ids=torch.tensor(np.flatnonzero(use),dtype=torch.long)
                # kappa=1, alpha=0 polynomial CPML. Both E and H stretching
                # use identical physical profiles sampled on their own rows.
                rate=-4*math.log(reflection)/(2*width*math.sqrt(background))*rho[use]**3
                b=torch.tensor(np.exp(-rate*dt),dtype=torch.float32)
                cpsi=torch.expm1(torch.tensor(-rate*dt,dtype=torch.float32))
                key=len(self.operators)
                self.operators[key]=(r,c,w,ids,b,cpsi)
                self.forward_groups[forward].append(key)
        self.psi_count=sum(len(v[3]) for v in self.operators.values())
        self.psi_lengths={key:len(v[3]) for key,v in self.operators.items()}

    def project(self,value,family):
        return value*self.active[family]

    def curl(self, field, psi, forward):
        target='H' if forward else 'E'
        result=field.new_zeros(len(self.topology.components[target]))
        updated=list(psi)
        for key in self.forward_groups[forward]:
            row,col,weight,ids,b,c=self.operators[key]
            derivative=torch.zeros_like(result).index_add(0,row,weight*field[col])
            updated[key]=b*psi[key]+c*derivative[ids]
            result=result+derivative.index_add(0,ids,updated[key])
        return result,tuple(updated)

    def transpose(self, seed, psi_seed, forward):
        source='E' if forward else 'H'
        result=seed.new_zeros(len(self.topology.components[source]))
        previous=list(psi_seed)
        for key in self.forward_groups[forward]:
            row,col,weight,ids,b,c=self.operators[key]
            memory=psi_seed[key]+seed[ids]
            previous[key]=b*memory
            derivative=seed.index_add(0,ids,c*memory)
            result=result.index_add(0,col,weight*derivative[row])
        return result,tuple(previous)


class EndpointCPMLSimulation(EndpointSimulation):
    """FP32 uniform endpoint grid, closed walls and isotropic CPML.

    PML faces terminate at exact PEC endpoints. ``pml_cells`` is a common
    positive layer count. The PML plus one cell of sampled electric epsilon
    must equal ``background_epsilon`` and has zero material VJP. CPML profile
    parameters, mesh and sources are fixed. No native Project dispatch.
    ``tensor_budget_bytes=None`` derives the budget at every admission, as
    EndpointSimulation does.
    """
    def __init__(self,nodes_um,faces,*,dt_seconds,sources,observations,pml_cells,
                 background_epsilon,reflection=1e-6,device='cpu',checkpoints=4,
                 tensor_budget_bytes=None):
        if torch.device(device).type not in ('cpu','cuda'):
            raise ValueError('Endpoint CPML supports CPU reference or direct CUDA.')
        aliases={'pec':'pec','antisymmetric':'pec','pmc':'pmc','symmetric':'pmc','pml':'pml'}
        try:physical=tuple(tuple(aliases[x.lower()] for x in pair) for pair in faces)
        except (KeyError,AttributeError) as exc:raise ValueError('Use explicit PEC/PMC/PML faces.') from exc
        closed=tuple(tuple('pec' if x=='pml' else x for x in pair) for pair in physical)
        topology=CompactEndpointTopology(nodes_um,closed)
        if not any('pml' in pair for pair in physical):raise ValueError('At least one PML face is required.')
        if isinstance(pml_cells,bool) or not isinstance(pml_cells,int) or pml_cells<1:
            raise ValueError('pml_cells must be a positive integer.')
        if not math.isfinite(background_epsilon) or background_epsilon<1:
            raise ValueError('Fixed background_epsilon must be finite and at least one.')
        if not math.isfinite(reflection) or not 0<reflection<1:
            raise ValueError('reflection must be finite and between zero and one.')
        for axis,nodes in enumerate(topology.nodes):
            if not np.allclose(np.diff(nodes),np.diff(nodes)[0],rtol=1e-12,atol=1e-14):
                raise ValueError('Endpoint CPML initially requires uniform axes.')
            if 'pml' in physical[axis] and pml_cells*physical[axis].count('pml')+2>=topology.shape[axis]:
                raise ValueError('Leave at least three cells between PML interiors or the opposite wall.')
        self.physical_faces=physical
        self.pml_cells=pml_cells
        self.background_epsilon=float(background_epsilon)
        self.reflection=float(reflection)
        # Before sparse construction, assume two psi entries per field DOF.
        # Extra derivative metadata is at most 20 bytes per potential psi.
        self._psi_count=2*sum(topology.counts.values())
        minimum_extra=20*self._psi_count+100*sum(topology.counts.values())
        if tensor_budget_bytes is not None and (isinstance(tensor_budget_bytes,bool) or not isinstance(tensor_budget_bytes,int) or tensor_budget_bytes<=minimum_extra):
            raise ValueError('Endpoint CPML tensor budget is too small for auxiliary metadata.')
        # The base constructor admits its preliminary bound plus this metadata.
        self._construction_bytes=minimum_extra
        super().__init__(topology.nodes,closed,dt_seconds=dt_seconds,sources=sources,
            observations=observations,device=device,checkpoints=checkpoints,
            tensor_budget_bytes=tensor_budget_bytes)
        reference=self.backend
        if self.device.type=='cuda':
            from .pmc_cpml_cuda import EndpointCPMLCUDA
            self.backend=EndpointCPMLCUDA(reference,physical,pml_cells,self.dt,background_epsilon,reflection)
        else:
            self.backend=_EndpointCPML(reference,physical,pml_cells,self.dt,background_epsilon,reflection)
        self._psi_count=self.backend.psi_count
        if self.device.type=='cuda':
            self.collar=self.backend.collar_mask()
            self.metadata_bytes=self.backend.metadata_bytes
        else:
            xyz=reference.topology.coordinates['E']
            collar=np.zeros(len(xyz),dtype=bool)
            for axis,nodes in enumerate(topology.nodes):
                distance=(pml_cells+1)*float(nodes[1]-nodes[0])
                if physical[axis][0]=='pml':collar |= xyz[:,axis]<=nodes[0]+distance+1e-12
                if physical[axis][1]=='pml':collar |= xyz[:,axis]>=nodes[-1]-distance-1e-12
            self.collar=torch.tensor(collar)
            self.metadata_bytes=sum(v.numel()*v.element_size() for group in self.backend.operators.values() for v in group)
            self.metadata_bytes+=sum(v.numel()*v.element_size() for v in self.backend.active.values())
        self.metadata_bytes+=self.source_tensor.numel()*8+self.collar.numel()
        for family,index in self.observation_ids+tuple(('E',i) for i in self.source_ids):
            coordinate=self._coordinate(family,index)
            for axis,nodes in enumerate(topology.nodes):
                width=pml_cells*float(nodes[1]-nodes[0])
                if ((physical[axis][0]=='pml' and coordinate[axis]<nodes[0]+width-1e-12)
                    or (physical[axis][1]=='pml' and coordinate[axis]>nodes[-1]-width+1e-12)):
                    raise ValueError('Sources and observations must be outside CPML layers.')

    def _coordinate(self,family,index):
        for block in self.topology.blocks[family]:
            if block.start<=index<block.stop:
                offset=index-block.start
                component=offset%3 if block.component is None else block.component
                linear=offset//3 if block.component is None else offset
                q=list(np.unravel_index(linear,block.shape[:3]))
                for axis in block.upper_axes:q[axis]=self.topology.shape[axis]
                return tuple(nodes[i] if (component!=axis if family=='E' else component==axis)
                    else (nodes[i]+nodes[i+1])/2 for axis,(nodes,i) in enumerate(zip(self.topology.nodes,q)))
        raise ValueError('Unknown endpoint degree of freedom.')

    def _payload(self,steps,saved):
        return super()._payload(steps,saved)+(saved+12)*4*self._psi_count+self._rounding(saved)

    def _rounding(self,saved):
        # Tiny CUDA psi/profile allocations are rounded independently. Reserve
        # one 512-byte quantum per live state component and auxiliary buffer.
        return ((saved+16)*14+64)*512 if self.device.type=='cuda' else 0

    def memory_plan(self,steps):
        report=super().memory_plan(steps)
        report.update(backend='cuda-endpoint-cpml-direct' if self.device.type=='cuda' else 'cpu-endpoint-cpml-reference',psi_elements=self._psi_count,
            cuda_rounding_allowance_bytes=self._rounding(report['checkpoint_states']),
            psi_state_bytes=4*self._psi_count,
            field_state_bytes=4*sum(self.topology.counts.values()),
            complete_state_bytes=4*(sum(self.topology.counts.values())+self._psi_count))
        return report

    def _configuration(self):
        profiles=tuple((key,tuple((id(v),v._version) for v in value))
                       for key,value in self.backend.profiles.items()) if self.device.type=='cuda' else ()
        return super()._configuration()+(self.physical_faces,self.pml_cells,self.background_epsilon,
            self.reflection,id(self.collar),self.collar._version,
            tuple((key,tuple(value)) for key,value in self.backend.forward_groups.items()),
            tuple(self.backend.psi_lengths.items()),profiles)

    def _zero(self):
        base=super()._zero()
        return CPMLState(base.electric,base.magnetic,
            tuple(torch.zeros(length,dtype=torch.float32,device=self.device) for length in self.backend.psi_lengths.values()))

    def _step(self,state,epsilon,drive):
        curl,psi=self.backend.curl(state.magnetic,state.psi,False)
        e=self.backend.project(state.electric,'E')+self.dt/epsilon*curl
        e=e.index_add(0,self.source_tensor,drive)
        curl,psi=self.backend.curl(e,psi,True)
        h=self.backend.project(state.magnetic,'H')-self.dt*curl
        return CPMLState(e,h,psi)

    def _transpose(self,state,seed,epsilon):
        contribution,psi=self.backend.transpose(-self.dt*seed.magnetic,seed.psi,True)
        ebar=seed.electric+contribution
        curl,_=self.backend.curl(state.magnetic,state.psi,False)
        material=-self.dt/epsilon.square()*ebar*curl
        material=material.masked_fill(self.collar,0)
        contribution,psi=self.backend.transpose(self.dt/epsilon*ebar,psi,False)
        previous=CPMLState(self.backend.project(ebar,'E'),
            self.backend.project(seed.magnetic,'H')+contribution,psi)
        return previous,material,ebar[self.source_tensor]

    def __call__(self,epsilon,waveforms):
        for value in (epsilon,waveforms):
            if not isinstance(value,torch.Tensor) or value.device!=self.device or value.dtype!=torch.float32 or not value.is_contiguous():
                raise ValueError('Inputs must be contiguous FP32 tensors on the configured device.')
        if epsilon.shape!=(self.topology.counts['E'],):raise ValueError('One epsilon per endpoint E DOF is required.')
        if waveforms.ndim!=2 or waveforms.shape[1]!=len(self.source_ids):raise ValueError('Waveforms require [steps,sources].')
        self.admit_tensors(self.memory_plan(len(waveforms))['tensor_upper_bound_bytes'],
                           'Requested simulation exceeds endpoint CPML tensor budget')
        if any(not bool(torch.isfinite(v).all()) for v in (epsilon,waveforms)):
            raise ValueError('Inputs must be finite.')
        if bool((epsilon<1).any()):raise ValueError('Endpoint CPML requires epsilon at least one.')
        if not bool((epsilon[self.collar]==self.background_epsilon).all()):
            raise ValueError('CPML and its one-cell collar require fixed isotropic background epsilon.')
        if self.dt>math.sqrt(float(epsilon.detach().min()))*self.topology.cfl_unit*(1+1e-7):
            raise ValueError('dt_seconds exceeds conservative Yee CFL.')
        return _CPMLTrace.apply(epsilon,waveforms,self)


class _CPMLTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx,epsilon,waveforms,sim):
        state=sim._zero();trace=epsilon.new_empty((len(waveforms),len(sim.observation_ids)))
        report={'forward_steps':len(waveforms),'replayed_steps':0,'reverse_steps':0,
                'peak_checkpoints':0,'checkpoint_saves':0,**sim.memory_plan(len(waveforms))}
        sim.last_report=report
        for n,drive in enumerate(waveforms):
            state=sim._step(state,epsilon,drive)
            for j,(family,index) in enumerate(sim.observation_ids):
                trace[n,j]=(state.electric if family=='E' else state.magnetic)[index]
        ctx.save_for_backward(epsilon,waveforms)
        ctx.sim,ctx.report,ctx.configuration=sim,report,sim._configuration()
        return trace

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        epsilon,waveforms=ctx.saved_tensors;sim=ctx.sim
        if sim._configuration()!=ctx.configuration:raise RuntimeError('Endpoint CPML configuration changed before backward.')
        seed=sim._zero();ge=torch.zeros_like(epsilon);gw=torch.zeros_like(waveforms)
        saved={};state=None;report=ctx.report
        try:
            for action in _reverse_schedule(len(waveforms),sim.checkpoints):
                if action[0]=='replay':
                    _,key,start,end=action
                    state=sim._zero() if key is None else saved[key]
                    for step in range(start,end):state=sim._step(state,epsilon,waveforms[step])
                    report['replayed_steps']+=end-start
                elif action[0]=='save':
                    saved[action[1]]=state;report['checkpoint_saves']+=1
                    report['peak_checkpoints']=max(report['peak_checkpoints'],len(saved))
                elif action[0]=='drop':del saved[action[1]]
                else:
                    n=action[1]
                    for j,(family,index) in enumerate(sim.observation_ids):
                        (seed.electric if family=='E' else seed.magnetic)[index]+=gradient[n,j]
                    seed,local,gw[n]=sim._transpose(state,seed,epsilon)
                    ge.add_(local);report['reverse_steps']+=1
        finally:saved.clear()
        return ge,gw,None
