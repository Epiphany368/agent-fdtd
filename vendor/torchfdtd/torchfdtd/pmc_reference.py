"""Experimental exact-endpoint PEC/PMC state and transpose reference.

This module is not connected to public simulation dispatch. It uses sparse
incidence lists for correctness, not production memory or performance claims.
Units use c=1. Materials are positive scalar or diagonal, nondispersive epsilon.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math

import numpy as np
import torch

from .boundaries import CURL_TERMS


@dataclass(frozen=True)
class BoundaryBlock:
    family: str
    component: int | None
    upper_axes: tuple[int, ...]
    shape: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True)
class ReferenceState:
    electric: torch.Tensor
    magnetic: torch.Tensor


class EndpointTopology:
    """Disjoint volume prefix plus upper PMC faces and E edges.

    ``faces[a]`` is (lower, upper), each 'pec' or 'pmc'. Node coordinates
    include physical endpoints. Half coordinates are actual cell midpoints.
    Lower PEC constrained entries remain allocated and are projected to zero.
    """
    def __init__(self, nodes, faces):
        self.nodes=tuple(np.asarray(x,dtype=float) for x in nodes)
        self.faces=tuple(tuple(x) for x in faces)
        if len(self.nodes)!=3 or len(self.faces)!=3:
            raise ValueError('The reference requires three coordinate axes and face pairs.')
        if any(x.ndim!=1 or len(x)<3 or not np.isfinite(x).all() or np.any(np.diff(x)<=0) for x in self.nodes):
            raise ValueError('Each axis needs at least two finite, strictly increasing cells.')
        if any(len(p)!=2 or any(k not in ('pec','pmc') for k in p) for p in self.faces):
            raise ValueError('Only explicit PEC/PMC face pairs are admitted by this reference.')
        self.shape=tuple(len(x)-1 for x in self.nodes)
        self.blocks={};self.coordinates={};self.components={};self.indices={};self.active={};self.lookup={}
        for family in ('E','H'):
            entries=[];blocks=[]
            for coord in product(*(range(n) for n in self.shape)):
                for comp in range(3):entries.append((comp,coord))
            blocks.append(BoundaryBlock(family,None,(),(*self.shape,3),0,len(entries)))
            for comp in range(3):
                axes=[a for a in range(3) if self.is_node(family,comp,a) and self.faces[a][1]=='pmc']
                for count in range(1,len(axes)+1):
                    for upper in combinations(axes,count):
                        start=len(entries)
                        shape=tuple(1 if a in upper else n for a,n in enumerate(self.shape))
                        for local in product(*(range(n) for n in shape)):
                            coord=tuple(self.shape[a] if a in upper else local[a] for a in range(3))
                            entries.append((comp,coord))
                        blocks.append(BoundaryBlock(family,comp,upper,shape,start,len(entries)))
            self.blocks[family]=tuple(blocks)
            self.lookup[family]={entry:i for i,entry in enumerate(entries)}
            assert len(self.lookup[family])==len(entries)
            self.components[family]=np.array([c for c,_ in entries],dtype=np.int64)
            self.indices[family]=np.array([q for _,q in entries],dtype=np.int64)
            self.coordinates[family]=np.array([[self.nodes[a][q[a]] if self.is_node(family,c,a)
                else (self.nodes[a][q[a]]+self.nodes[a][q[a]+1])/2 for a in range(3)] for c,q in entries])
            self.active[family]=np.array([not any(q[a]==0 and self.is_node(family,c,a) and self.faces[a][0]=='pec'
                for a in range(3)) for c,q in entries])

    @staticmethod
    def is_node(family,component,axis):
        return component!=axis if family=='E' else component==axis

    def zeros(self, *, dtype=torch.float32, device='cpu'):
        return ReferenceState(*(torch.zeros(len(self.components[f]),dtype=dtype,device=device) for f in ('E','H')))

    def block_views(self, values, family):
        return tuple(values[b.start:b.stop].view(b.shape) for b in self.blocks[family])

    def state_bytes(self, dtype=torch.float32):
        """Exact field payload only, excluding operators, parameters and workspaces."""
        return sum(len(x) for x in self.components.values())*torch.empty((),dtype=dtype).element_size()

    def sample_epsilon(self, sampler, *, dtype=torch.float32, device='cpu'):
        """Sample actual E positions. Sampler(xyz, component) returns one value/DOF.

        A differentiable geometry sampler may return a tensor linked to its
        geometry parameters. No nearest-cell replication is implied here.
        """
        xyz=torch.as_tensor(self.coordinates['E'],dtype=dtype,device=device)
        component=torch.as_tensor(self.components['E'],device=device)
        values=sampler(xyz,component)
        if not isinstance(values,torch.Tensor) or values.shape!=(len(xyz),):
            raise ValueError('The material sampler must return one tensor value per electric degree of freedom.')
        return values

    def shared_volume_epsilon(self, epsilon):
        """Explicit nearest-interior sharing model, not a general geometry sampler.

        Endpoint face/edge parameters alias the final interior cell parameter
        along each upper axis. Gradients accumulate into that same parameter.
        """
        if tuple(epsilon.shape) not in (self.shape,(*self.shape,3)):
            raise ValueError('Volume epsilon must be scalar or diagonal on the base cells.')
        q=np.minimum(self.indices['E'],np.asarray(self.shape)-1)
        ids=(q[:,0]*self.shape[1]+q[:,1])*self.shape[2]+q[:,2]
        if epsilon.ndim==4:ids=3*ids+self.components['E']
        index=torch.as_tensor(ids,device=epsilon.device)
        return epsilon.reshape(-1)[index],index


class EndpointReference:
    """Matrix-free sparse reference step and explicit Euclidean transpose."""
    def __init__(self, topology, *, dtype=torch.float32, device='cpu'):
        self.topology=topology
        self.dtype=dtype;self.device=torch.device(device)
        self.active={f:torch.as_tensor(m,device=device) for f,m in topology.active.items()}
        self.operators={forward:self._operator(forward) for forward in (False,True)}

    def _operator(self, forward):
        t=self.topology;source,target=('E','H') if forward else ('H','E')
        rows=[];cols=[];values=[]
        for axis,comp,out,sign in CURL_TERMS:
            for row,(target_comp,coord) in enumerate(zip(t.components[target],t.indices[target])):
                if target_comp!=out or not t.active[target][row]:continue
                i=coord[axis];width=np.diff(t.nodes[axis])
                if forward:
                    terms=((i+1,1/width[i]),(i,-1/width[i]))
                elif i==0:
                    terms=((0,2/width[0]),)
                elif i==t.shape[axis]:
                    terms=((i-1,-2/width[-1]),)
                else:
                    scale=2/(width[i-1]+width[i]);terms=((i,scale),(i-1,-scale))
                for index,scale in terms:
                    q=list(coord);q[axis]=index
                    col=t.lookup[source].get((comp,tuple(q)))
                    # Omitted upper PEC nodal fields are exact zeros.
                    if col is not None and t.active[source][col]:
                        rows.append(row);cols.append(col);values.append(sign*scale)
        return (torch.tensor(rows,device=self.device),torch.tensor(cols,device=self.device),
                torch.tensor(values,device=self.device,dtype=self.dtype))

    def project(self, value, family):return value*self.active[family]

    def curl(self, value, forward, *, transpose=False):
        row,col,weight=self.operators[forward]
        source,target=('E','H') if forward else ('H','E')
        if transpose:row,col=col,row;target=source
        result=value.new_zeros(len(self.topology.components[target]))
        return result.index_add(0,row,weight*value[col])

    def step(self, state, epsilon, dt):
        if epsilon.shape!=state.electric.shape or epsilon.is_complex() or not bool(torch.isfinite(epsilon).all() & (epsilon>0).all()):
            raise ValueError('Supply finite positive real epsilon at every electric degree of freedom.')
        e=self.project(state.electric,'E')+dt/epsilon*self.curl(state.magnetic,False)
        h=self.project(state.magnetic,'H')-dt*self.curl(e,True)
        return ReferenceState(e,h)

    def transpose_step(self, state, seed, epsilon, dt):
        """Return old-state cotangents and gradient of sampled E epsilon."""
        ebar=seed.electric-dt*self.curl(seed.magnetic,True,transpose=True)
        gradient=(-dt/epsilon.square()*(ebar.conj()*self.curl(state.magnetic,False)).real)
        hbar=self.project(seed.magnetic,'H')+self.curl(dt/epsilon*ebar,False,transpose=True)
        return ReferenceState(self.project(ebar,'E'),hbar),gradient

    def cfl_limit(self, epsilon_min=1.):
        return math.sqrt(epsilon_min)/math.sqrt(sum(1/np.diff(x).min()**2 for x in self.topology.nodes))
