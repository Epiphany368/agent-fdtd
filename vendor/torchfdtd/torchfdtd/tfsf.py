"""Closed, grid-aligned total-field/scattered-field boxes.

The discrete source is a mask/curl commutator, not a continuum impedance
approximation. Sparse corrections on six faces sample a live scalar Yee line.
Its state is O(normal span), never a time-by-surface history. Incident fields
remain independent of the scatterer and every source has its own line.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .boundaries import CURL_TERMS
from .waveforms import source_time_signal
from .mesh import local_uniform_step


def tfsf_plan(source, region):
    s,r=source,region
    axis='xyz'.index(s.normal)
    active=2 if r.dimension=='2d' else 3
    if axis>=active:raise ValueError(f'{s.name}: TFSF propagation must follow an active axis.')
    if any(face.kind!='pml' for a in range(active) for face in r.boundaries.pair(a)):
        raise ValueError(f'{s.name}: the current isolated TFSF box requires PML on every active face.')
    lo,hi=[],[]
    for a,nodes in enumerate(r.mesh_nodes):
        if a>=active:
            lo.append(0);hi.append(0);continue
        if s.size[a]<=0:raise ValueError(f'{s.name}: TFSF spans must be positive on every active axis.')
        low=s.center[a]-s.size[a]/2;high=s.center[a]+s.size[a]/2
        l=int(np.argmin(abs(nodes-low)));h=int(np.argmin(abs(nodes-high)))
        if h-l<3:raise ValueError(f'{s.name}: the TFSF box needs at least three cells along each active axis.')
        if l<r.pml_layers(a,0)+2 or h>r.shape[a]-r.pml_layers(a,1)-2:
            raise ValueError(f'{s.name}: leave at least two interior cells between each TFSF face and PML.')
        local_uniform_step(r,a,l-1,h+2)
        lo.append(l);hi.append(h)
    return axis,tuple(lo),tuple(hi),1 if s.direction=='+' else -1


def tfsf_metadata(source,region):
    axis,lo,hi,d=tfsf_plan(source,region)
    length=hi[axis]-lo[axis]
    return dict(source_id=source.id,normal=source.normal,direction=source.direction,
        lower_node_indices=list(lo),upper_node_indices=list(hi),
        lower_um=[float(n[l]) if len(n)>2 else 0. for n,l in zip(region.mesh_nodes,lo)],
        upper_um=[float(n[h]) if len(n)>2 else 0. for n,h in zip(region.mesh_nodes,hi)],
        incident_line_cells=2*source.incident_pml_cells+64+length,
        incident_pml_cells=source.incident_pml_cells,incident_drive_offset_cells=8,
        total_field='inside the staggered box',scattered_field='outside the staggered box')


def validate_tfsf_materials(project,epsilon,ownership):
    for s in project.sources:
        if not s.enabled or s.kind!='tfsf':continue
        _,lo,hi,_=tfsf_plan(s,project.region)
        active=2 if project.region.dimension=='2d' else 3
        for axis in range(active):
            for boundary in (lo[axis],hi[axis]):
                sl=[slice(l-1,h+2) if a<active else slice(None) for a,(l,h) in enumerate(zip(lo,hi))]
                sl[axis]=slice(boundary-1,boundary+2);sl=tuple(sl)
                if not np.allclose(epsilon[sl],project.region.background_index**2,
                                   rtol=1e-7 if epsilon.dtype==np.float32 else 1e-13,atol=0):
                    raise ValueError(f'{s.name}: the TFSF face neighborhood must be homogeneous background. Move the scatterer inside the box.')
                occupied=set(np.unique(ownership[sl]))
                if any(i in occupied and m.oscillators for i,m in enumerate(project.materials)):
                    raise ValueError(f'{s.name}: dispersive material cannot intersect the TFSF face neighborhood.')


def surface_maps(source,region):
    """Sparse exact commutator at Yee faces, with corner sums merged.

    E_o receives C/eps * (M_E curl(H_i) - curl(M_H H_i)).
    H_o receives -C * (M_H curl(E_i) - curl(M_E E_i)).
    The masks differ only on the derivative axis for each curl term, so only
    two faces per term are needed. This construction allocates O(surface).
    """
    axis,lo,hi,d=tfsf_plan(source,region)
    active=2 if region.dimension=='2d' else 3
    probe=source.incident_pml_cells+32
    entry=lo[axis] if d==1 else hi[axis]
    electric={int('xyz'.index(f[1])):w for f,w in source.polarization_components}
    magnetic={3-axis-c:d*w*(1 if (axis+1)%3==c else -1) for c,w in electric.items()}
    output={}
    for family,incident in (('E',magnetic),('H',electric)):
        targets=[];samples=[];weights=[]
        for a,c,o,sign in CURL_TERMS:
            if a>=active or c not in incident:continue
            courant=region.rectangular_courant*region.reference_step/local_uniform_step(region,a,lo[a]-1,hi[a]+2)
            for side in (0,1):
                # E tangential faces lie at lo/hi. H tangential faces lie just
                # outside those planes: lo-1/2 and hi+1/2 on their Yee axes.
                face=(lo[a] if side==0 else hi[a])-(1 if family=='H' and side==0 else 0)
                coords=[]
                for b in range(3):
                    if b>=active:coords.append(np.array([0]))
                    elif b==a:coords.append(np.array([face]))
                    else:
                        half=(b==o) if family=='E' else (b!=o)
                        coords.append(np.arange(lo[b],hi[b]+(0 if half else 1)))
                xyz=np.meshgrid(*coords,indexing='ij')
                flat=np.ravel_multi_index(tuple(v.ravel() for v in xyz),region.shape)*3+o
                j=xyz[axis].ravel().copy()
                if a==axis and side==0:j+=-1 if family=='E' else 1
                inc=probe+d*(j-entry)-(1 if family=='E' and d==-1 else 0)
                factor=courant/(region.background_index**2) if family=='E' else courant
                orientation=(-1 if side==0 else 1) if family=='E' else (1 if side==0 else -1)
                targets.append(flat);samples.append(inc)
                weights.append(np.full(len(flat),factor*sign*orientation*incident[c]))
        target=np.concatenate(targets);sample=np.concatenate(samples);weight=np.concatenate(weights)
        order=np.lexsort((sample,target));target,sample,weight=(v[order] for v in (target,sample,weight))
        starts=np.flatnonzero(np.r_[True,(np.diff(target)!=0)|(np.diff(sample)!=0)])
        weight=np.add.reduceat(weight,starts);target=target[starts];sample=sample[starts]
        keep=weight!=0;target,sample,weight=(v[keep] for v in (target,sample,weight))
        if len(np.unique(target))!=len(target):
            raise AssertionError('A TFSF target unexpectedly needs multiple incident samples.')
        output[family]=(target,sample,weight)
    return output


def line_coefficients(count,layers,courant,index):
    coefficients=[]
    j=np.arange(count,dtype=float)
    for forward in (False,True):
        depth=np.maximum(layers-j-(1 if forward else .5),j-(count-layers)+(1 if forward else .5))
        rho=np.maximum(depth,0)/(layers+1)
        sigma=40*rho**3/(layers+1)
        b=np.exp(-(sigma+1e-8)*courant/index)
        coefficients.append((b,(b-1)*sigma/(sigma+1e-8)))
    return coefficients


class IncidentLine:
    def __init__(self,source,region,grid=None):
        axis,lo,hi,_=tfsf_plan(source,region)
        self.source=source;self.region=region
        self.count=2*source.incident_pml_cells+64+hi[axis]-lo[axis]
        self.drive_at=source.incident_pml_cells+24
        self.probe=source.incident_pml_cells+32
        self.c=region.rectangular_courant*region.reference_step/local_uniform_step(region,axis,lo[axis]-1,hi[axis]+2)
        self.ce=self.c/region.background_index**2
        self.torch=grid is not None and grid.is_torch
        convert=grid._coefficient if grid is not None else lambda a:np.asarray(a,dtype=region.precision)
        self.e,self.h,self.pe,self.ph,self.diff=(convert(np.zeros(self.count)) for _ in range(5))
        self.coefficients=[tuple(convert(v) for v in pair) for pair in
            line_coefficients(self.count,source.incident_pml_cells,self.c,region.background_index)]
        self.drive=convert(source_time_signal(source,np.arange(1,region.steps+1)*region.time_step))
        # Positive norm of the complete incident line prevents premature decay
        # termination while a prepared pulse is still travelling toward a face.
        area=math.prod((hi[a]-lo[a])*local_uniform_step(region,a,lo[a]-1,hi[a]+2)/region.reference_step
                       for a in range(2 if region.dimension=='2d' else 3) if a!=axis)
        area*=local_uniform_step(region,axis,lo[axis]-1,hi[axis]+2)/region.reference_step
        self.norm_weight=convert(np.full(self.count,area,dtype=float))

    def advance(self,family,counter):
        forward=family=='H';field=self.h if forward else self.e
        psi=self.ph if forward else self.pe
        b,c=self.coefficients[int(forward)]
        if forward:
            self.diff[:-1]=self.e[1:]-self.e[:-1]
            if self.torch:self.diff[-1:].zero_()
            else:self.diff[-1]=0
        else:
            self.diff[1:]=self.h[1:]-self.h[:-1]
            if self.torch:self.diff[:1].zero_()
            else:self.diff[0]=0
        psi*=b;psi+=c*self.diff
        field-=(self.c if forward else self.ce)*(self.diff+psi)
        if not forward:
            value=self.drive.index_select(0,counter)[0] if self.torch else self.drive[int(counter[0])]
            self.e[self.drive_at]+=value


class TfsfState(IncidentLine):
    def __init__(self,source,region,grid):
        super().__init__(source,region,grid)
        self.grid=grid
        self.maps={}
        for family,(target,sample,weight) in surface_maps(source,region).items():
            if grid.is_torch:
                target=torch.as_tensor(target,device=grid.E.device,dtype=torch.int64)
                sample=torch.as_tensor(sample,device=grid.E.device,dtype=torch.int64)
            self.maps[family]=(target,sample,grid._coefficient(weight))
        grid.memory_states.extend((self.e,self.h,self.pe,self.ph,self.diff))
        grid.incident_states.append(self)

    def inject(self,family,counter):
        self.advance(family,counter)
        target,sample,weight=self.maps[family]
        field=(self.grid.E if family=='E' else self.grid.H).reshape(-1)
        incident=self.h if family=='E' else self.e
        field[target]+=weight*incident[sample]


def prepare_tfsf(grid,project,epsilon,ownership):
    validate_tfsf_materials(project,epsilon,ownership)
    return [TfsfState(project.resolved_source(s),project.region,grid)
            for s in project.sources if s.enabled and s.kind=='tfsf']


class TfsfInjection:
    def __init__(self,grids,counter,*,fused=False):
        self.groups=[[g.incident_states[i] for g in grids if len(g.incident_states)>i]
                     for i in range(max((len(g.incident_states) for g in grids),default=0))]
        self.counter=counter;self.cuda=None
        if self.groups and fused and grids[0].is_torch:
            from .cuda_tfsf import CudaTfsfInjection
            self.cuda=CudaTfsfInjection(self.groups,counter)

    def inject(self,family):
        if self.cuda is not None:
            self.cuda.inject(family)
        else:
            for group in self.groups:
                for state in group:state.inject(family,self.counter)


def incident_preview(source,region):
    line=IncidentLine(source,region)
    axis,lo,hi,_=tfsf_plan(source,region)
    e=np.zeros(region.steps);h=np.zeros(region.steps)
    far=np.zeros(region.steps)
    for q in range(region.steps):
        line.advance('E',np.array([q]))
        line.advance('H',np.array([q]))
        e[q]=line.e[line.probe];h[q]=line.h[line.probe]
        far[q]=line.e[line.probe+hi[axis]-lo[axis]]
    return e,h,far
