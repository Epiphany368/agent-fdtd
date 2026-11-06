"""Deterministic rectilinear mesh with smooth metric equidistribution.

Refined intervals stay on the original fine lattice. Only gaps are coarsened.
PML cells and one adjacent interior cell remain uniform. No hanging nodes,
subcycling, cached FSP mesh values, or vendor meshing code are used.
"""
from __future__ import annotations

import math
import numpy as np


def object_bounds(obj):
    from .geometry import object_bounds as bounds
    return bounds(obj)


def configure_auto_mesh(project):
    r=project.region
    boxes=[]
    if r.mesh_type == 'graded' and r.mesh_auto_refine:
        for obj in project.structures:
            if obj.enabled:
                center,size=object_bounds(obj)
                boxes.append((center,tuple(v+4*h for v,h in zip(size,r.axis_steps))))
        for obj in project.sources+project.monitors:
            if not obj.enabled:continue
            # A plane's complete transverse support remains refined, avoiding
            # a spatially varying sheet injection on a coarse transverse grid.
            size=obj.size if getattr(obj,'kind',None) in ('plane','field','tfsf') else (0,0,0)
            boxes.append((tuple(obj.center),tuple(v+4*h for v,h in zip(size,r.axis_steps))))
    r._auto_boxes=tuple(boxes)
    shortest=[]
    for source in project.sources:
        if not source.enabled:continue
        s=project.resolved_source(source)
        shortest.append(s.wavelength_start if s.time_definition in ('wavelength','frequency') else s.wavelength)
    r._coarse_limit=max(r.reference_step,min(r.mesh_max,min(shortest)/(r.background_index*r.mesh_ppw))) if shortest else max(r.reference_step,r.mesh_max)


def coarse_gap(length, h, maximum, grading):
    """Return positions including both anchors, with no cell below h.

The requested spacing rises linearly with distance to either refined edge.
Its reciprocal integral is analytic. Equally spaced metric coordinates give
smooth neighboring cell ratios without introducing small remainder cells.
"""
    count=round(length/h)
    if count<5 or maximum<=h*(1+1e-12):return np.arange(count+1)*h
    slope=math.log(grading)
    half=length/2
    ramp=min(half,(maximum-h)/slope)
    ramp_metric=math.log1p(slope*ramp/h)/slope
    half_metric=ramp_metric+(half-ramp)/maximum
    n=int(math.ceil(2*half_metric))
    if n>=count:return np.arange(count+1)*h
    metric=np.linspace(0,2*half_metric,n+1)
    mirrored=np.minimum(metric,2*half_metric-metric)
    positions=np.where(mirrored<=ramp_metric,
        h*np.expm1(slope*np.minimum(mirrored,ramp_metric))/slope,
        ramp+(mirrored-ramp_metric)*maximum)
    positions=np.where(metric<=half_metric,positions,length-positions)
    widths=np.diff(positions)
    all_widths=np.r_[h,widths,h]
    ratios=all_widths[1:]/all_widths[:-1]
    # Very short gaps can round to a metric bin smaller than one fine cell.
    # Retain the fine lattice in that case instead of shrinking the timestep.
    if widths.min()<h*(1-1e-10) or widths.max()>maximum*(1+1e-10) or max(ratios.max(),1/ratios.min())>grading*(1+1e-9):
        return np.arange(count+1)*h
    positions[0]=0;positions[-1]=length
    return positions


def mesh_nodes(region):
    r=region;base=r.base_shape
    if r.mesh_type!='explicit' and max(base)>1_000_000:raise ValueError('Requested fine lattice exceeds one million cells on an axis.')
    boxes=tuple((tuple(b.center),tuple(b.size)) for b in r.mesh_refinements if b.enabled)+r._auto_boxes
    key=(r.mesh_type,r.dimension,tuple(r.size),r.mesh,r.mesh_steps,r.mesh_coordinates,r.mesh_max,r.mesh_grading,r._coarse_limit,boxes,
         tuple(r.pml_layers(a,s) for a in range(3) for s in (0,1)))
    if r._mesh_cache is not None and r._mesh_cache[0]==key:return r._mesh_cache[1]
    result=[]
    for axis,n in enumerate(base):
        if r.mesh_type=='explicit':nodes=np.asarray(r.mesh_coordinates[axis],dtype=float)
        elif r.dimension=='2d' and axis==2:
            nodes=np.array([-r.size[2]/2,r.size[2]/2])
        elif r.mesh_type=='uniform':nodes=np.arange(n+1)*r.axis_steps[axis]-n*r.axis_steps[axis]/2
        else:
            h=r.axis_steps[axis];low=-n*h/2
            fine=np.zeros(n,dtype=bool)
            fine[:r.pml_layers(axis,0)+1]=True
            fine[-r.pml_layers(axis,1)-1:]=True
            for center,size in boxes:
                lo=max(0,min(n,math.floor((center[axis]-size[axis]/2-low)/h+1e-10)))
                hi=max(0,min(n,math.ceil((center[axis]+size[axis]/2-low)/h-1e-10)))
                fine[lo:hi]=True
            transitions=np.flatnonzero(np.r_[True,fine[1:]!=fine[:-1],True])
            pieces=[]
            maximum=max(h,min(r.mesh_max,r._coarse_limit or r.mesh_max))
            for lo,hi in zip(transitions[:-1],transitions[1:]):
                segment=np.arange(hi-lo+1)*h if fine[lo] else coarse_gap((hi-lo)*h,h,maximum,r.mesh_grading)
                pieces.append(low+lo*h+segment[:-1])
            nodes=np.r_[np.concatenate(pieces),-low]
        nodes.setflags(write=False);result.append(nodes)
    r._mesh_cache=(key,tuple(result))
    return tuple(result)


def mesh_summary(project):
    r=project.region
    widths=[np.diff(a) for a in r.mesh_nodes[:2 if r.dimension=='2d' else 3]]
    cells=math.prod(r.shape);uniform=math.prod(r.base_shape)
    return dict(mesh_type=r.mesh_type,material_sampling=r.material_sampling,uniform_reference_cells=uniform,
        cell_reduction_percent=100*(1-cells/uniform),axis_min_step_um=[float(v.min()) for v in widths],
        axis_max_step_um=[float(v.max()) for v in widths],coarse_limit_um=r._coarse_limit,
        max_adjacent_ratio=max(float(max((v[1:]/v[:-1]).max(),(v[:-1]/v[1:]).max())) for v in widths))


def freeze_refinements(project):
    """Retain geometry refinements when removing objects for an air reference."""
    from .models import Project, MeshRefinement
    p=Project.model_validate(project.model_dump())
    r=p.region
    r.mesh_refinements += [MeshRefinement(name=f'Frozen {i+1}',center=c,size=s) for i,(c,s) in enumerate(r._auto_boxes)]
    r.mesh_auto_refine=False
    return Project.model_validate(p.model_dump())


def local_uniform_step(region,axis,start,stop):
    """Spacing of a source support that is uniform along this axis."""
    widths=np.diff(region.mesh_nodes[axis])[start:stop]
    if not len(widths) or not np.allclose(widths,widths[0],rtol=1e-10,atol=0):
        raise ValueError('The source support and neighboring cells must have constant spacing along each active axis.')
    fine=region.axis_steps[axis]
    return fine if np.allclose(widths,fine,rtol=1e-10,atol=0) else float(np.mean(widths))
