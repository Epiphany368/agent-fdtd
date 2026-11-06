"""Empirical mesh convergence with fixed geometry and physical run duration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from .models import Project
from .solver import Simulation


def mesh_refinement_projects(project, meshes, *, duration_s=None, mesh_type='uniform'):
    """Return validated projects, preserving the domain and PML thickness.

    Each active span and each PML thickness must be divisible by every requested
    step. Run durations are rounded up by less than one timestep. Auto shutoff
    is disabled so sample/reference transforms use matching integration times.
    """
    base = Project.model_validate(project.model_dump())
    meshes = [float(h) for h in meshes]
    if len(meshes)<2 or any(not math.isfinite(h) or h<=0 for h in meshes):
        raise ValueError('Supply at least two finite positive mesh steps.')
    if any(a<=b for a,b in zip(meshes,meshes[1:])):
        raise ValueError('Mesh steps must be strictly decreasing from coarse to fine.')
    duration = base.region.steps*base.region.time_step if duration_s is None else float(duration_s)
    if not math.isfinite(duration) or duration<=0:
        raise ValueError('Physical duration must be finite and positive.')
    if mesh_type not in ('uniform','graded'):
        raise ValueError('Mesh study type must be uniform or graded.')
    r = base.region
    active = 2 if r.dimension=='2d' else 3
    spans = r.actual_size
    result = []
    def integer_ratio(length, h, name):
        ratio=length/h
        if not math.isclose(ratio,round(ratio),rel_tol=0,abs_tol=1e-8):
            raise ValueError(f'{name} must be divisible by mesh {h:g} um to preserve physical geometry.')
        return round(ratio)
    for h in meshes:
        data=base.model_dump()
        region=data['region']
        region.update(mesh=h,size=spans,mesh_type=mesh_type,mesh_steps=None,mesh_coordinates=None,time_step_override=None)
        # Keep explicit legacy cell sampling for uniform studies. Graded grids
        # require the component-wise Yee material convention.
        if mesh_type=='graded':region['material_sampling']='yee'
        region['mesh_max']=max(h,r.mesh_max)
        region['run_control']['auto_shutoff']=False
        dt=r.courant_factor/math.sqrt(active)*h*1e-6/299792458.
        ratio=duration/dt
        region['steps']=round(ratio) if math.isclose(ratio,round(ratio),rel_tol=0,abs_tol=1e-9) else math.ceil(ratio)
        if region['steps']<10:
            raise ValueError('Requested physical duration needs fewer than ten steps on this mesh.')
        region['snapshot_interval']=min(10000,region['steps'])
        for axis in range(active):
            integer_ratio(spans[axis],h,'Domain span')
            for side,face in enumerate(r.boundaries.pair(axis)):
                if face.kind=='pml':
                    nodes=r.mesh_nodes[axis];count=r.pml_layers(axis,side)
                    thickness=nodes[count]-nodes[0] if side==0 else nodes[-1]-nodes[-count-1]
                    layers=integer_ratio(thickness,h,'PML thickness')
                    region['boundaries']['xyz'[axis]+('_min' if side==0 else '_max')]['layers']=layers
        result.append(Project.model_validate(data))
    return result


def _values(observable):
    if not isinstance(observable,dict):observable={'observable':observable}
    if not observable or any(not isinstance(k,str) or not k for k in observable):
        raise ValueError('Observable must be a nonempty real scalar, array or named mapping.')
    values={}
    for key in sorted(observable):
        array=np.asarray(observable[key])
        if np.iscomplexobj(array):raise ValueError('Choose explicit real observables for convergence.')
        array=np.asarray(array,dtype=np.float64)
        if not array.size or not np.isfinite(array).all():
            raise ValueError('Convergence observables must be nonempty and finite.')
        values[key]=array
    return values


@dataclass
class ConvergenceReport:
    status: str
    levels: list[dict]
    relative_tolerance: float
    absolute_tolerance: float
    required_consecutive: int
    requested_duration_s: float
    interpretation: str = 'Agreement of selected observables on successive meshes, not a general error bound.'

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(asdict(self),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def mesh_convergence(project, meshes, observable, *, reference_project=None,
                     duration_s=None, mesh_type='uniform', rtol=.01, atol=1e-6,
                     consecutive=2, output_dir=None, cancel=None, progress=None,
                     cuda_graph=True):
    """Run matched refinement levels and compare a user-selected observable.

    observable(result) returns finite real values. When reference_project is
    supplied, observable(sample, reference) can use normalize_flux for R/T.
    Reference refinements are frozen from the sample at each level, preserving
    identical graded nodes. Values must keep identical names and shapes.
    The final `consecutive` comparisons must all satisfy
    abs(a-b) <= atol + rtol * max(abs(a), abs(b)).
    """
    if not callable(observable):raise ValueError('Provide a callable physical observable.')
    if not all(math.isfinite(t) and t>=0 for t in (rtol,atol)) or rtol+atol==0:
        raise ValueError('Tolerances must be finite, nonnegative and not both zero.')
    if not isinstance(consecutive,int) or consecutive<1:
        raise ValueError('consecutive must be a positive integer.')
    projects=mesh_refinement_projects(project,meshes,duration_s=duration_s,mesh_type=mesh_type)
    if len(projects)<consecutive+1:
        raise ValueError('Provide enough mesh levels for the required consecutive comparisons.')
    duration=project.region.steps*project.region.time_step if duration_s is None else float(duration_s)
    references=None
    if reference_project is not None:
        reference_project=Project.model_validate(reference_project.model_dump())
        references=[]
        from .mesh import freeze_refinements
        for sample in projects:
            frozen=freeze_refinements(sample) if mesh_type=='graded' else sample
            sample.region=frozen.region
            ref=reference_project.model_dump()
            ref['region']=sample.region.model_dump()
            references.append(Project.model_validate(ref))
    root=Path(output_dir) if output_dir is not None else None
    if root:root.mkdir(parents=True,exist_ok=True)
    report=ConvergenceReport('not_converged',[],rtol,atol,consecutive,duration)
    previous=None
    passed=0
    for i,p in enumerate(projects):
        if cancel is not None and cancel.is_set():
            report.status='cancelled'
            break
        started=time.perf_counter()
        sample=Simulation(p).run(cancel=cancel,cuda_graph=cuda_graph)
        ref=Simulation(references[i]).run(cancel=cancel,cuda_graph=cuda_graph) if references is not None and not sample.summary['cancelled'] else None
        if sample.summary['cancelled'] or (ref is not None and ref.summary['cancelled']):
            report.status='cancelled'
            break
        if ref is not None and sample.summary['steps']!=ref.summary['steps']:
            raise ValueError('Sample and reference must complete identical time-step counts.')
        values=_values(observable(sample,ref) if ref is not None else observable(sample))
        comparison={}
        if previous is not None:
            if values.keys()!=previous.keys() or any(values[k].shape!=previous[k].shape for k in values):
                raise ValueError('Observable names and array shapes must remain identical across meshes.')
            for key,value in values.items():
                difference=np.abs(value-previous[key])
                scale=np.maximum(np.abs(value),np.abs(previous[key]))
                relative=np.divide(difference,scale,out=np.zeros_like(difference),where=scale!=0)
                comparison[key]=dict(max_absolute_change=float(np.max(difference)),
                                     max_relative_change=float(np.max(relative)),
                                     passed=bool(np.all(difference<=atol+rtol*scale)))
            passed=passed+1 if all(v['passed'] for v in comparison.values()) else 0
        level=dict(mesh_um=p.region.mesh,shape=list(p.region.shape),steps=sample.summary['steps'],
                   time_step_s=p.region.time_step,duration_s=sample.summary['steps']*p.region.time_step,
                   wall_seconds=time.perf_counter()-started,summary=sample.summary,
                   values={k:v.tolist() for k,v in values.items()},comparison=comparison)
        if root:
            sample.save(root/f'level-{i:02d}-sample.npz')
            if ref is not None:ref.save(root/f'level-{i:02d}-reference.npz')
        report.levels.append(level)
        previous=values
        report.status='converged' if passed>=consecutive else 'not_converged'
        if root:report.save(root/'convergence.json')
        if progress:progress(dict(level=i+1,total=len(projects),status=report.status,**level))
        del sample,ref
    if root:report.save(root/'convergence.json')
    return report
