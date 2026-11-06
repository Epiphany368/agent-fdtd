"""Coupled plane-flux optimization across fixed Bloch-phase cases.

The 3D dipole example demonstrates shared geometry/material VJPs and budgeted
case replay. It is not converged plane-wave angular performance or CR design.
"""
import argparse
import json
from pathlib import Path

import torch

from torchfdtd import (AdjointBatchOptions, AdjointCase, AdjointExecutionPolicy,
    AdjointOptions, BoundaryFace, FieldMonitor, Project, RecomputedAdjointBatch,
    Region, Source, StreamedAdjointOptions, smooth_sphere_epsilon)


def run(*,device='cpu',execution='mixed',cases=3,iterations=2,dispersive=False):
    if cases<2 or iterations<1:raise ValueError('Use at least two cases and one iteration.')
    if execution not in ('resident','streamed','mixed'):raise ValueError('Unknown execution mode.')
    budget=512*1024**2
    resident=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=2,
        resident_budget_bytes=budget,gpu_budget_bytes=budget,
        backward_kernel='fused' if device=='cuda' else 'torch'),device=device,host_budget_bytes=budget)
    streamed=AdjointExecutionPolicy(streamed=StreamedAdjointOptions(device=device,
        slab_width=6,temporal_depth=4,checkpoints=2,gpu_budget_bytes=budget,host_budget_bytes=budget,
        tile_transfers='async' if device=='cuda' else 'sync'),device=device,host_budget_bytes=budget)
    specs=[]
    for index in range(cases):
        region=Region(dimension='3d',size=(2.,1.6,1.4),mesh=.1,pml_cells=3,
            steps=64,precision='float64',cuda_kernel='fused' if device=='cuda' else 'torch')
        region.boundaries.x_min=BoundaryFace(kind='bloch')
        region.boundaries.x_max=BoundaryFace(kind='bloch')
        region.bloch_phase=(.2+.3*index,0,0)
        wavelength=1.+.1*index
        project=Project(region=region,sources=[Source(center=(-.45,0,0),pulse='continuous',wavelength=wavelength)],
            monitors=[FieldMonitor(id='detector',center=(.4,0,0),size=(0,.6,.6),normal='x',downsample=2)])
        policy=streamed if execution=='streamed' or (execution=='mixed' and index%2) else resident
        specs.append(AdjointCase(project,policy,(0,1,2,3) if dispersive else (0,),[299792458/(wavelength*1e-6)]))
    batch=RecomputedAdjointBatch(specs,AdjointBatchOptions(host_budget_bytes=budget,gpu_budget_bytes=budget))
    radius=torch.nn.Parameter(torch.tensor(.22,dtype=torch.float64))
    damping=torch.nn.Parameter(torch.tensor(-1.,dtype=torch.float64))
    optimizer=torch.optim.Adam([radius,damping] if dispersive else [radius],lr=.002)
    def parameters():
        density=smooth_sphere_epsilon(specs[0].project.region,radius,inside=1.,outside=0.,width=.08,yee=True)
        return (1+density,(.8e30*density)[None],density.new_tensor(1.5e15),
            (torch.nn.functional.softplus(damping)+.01)*1e15) if dispersive else (1+density,)
    with torch.no_grad():
        background=torch.ones_like(parameters()[0])
        reference_parameters=(background,torch.zeros(1,dtype=background.dtype),
            background.new_tensor(1.5e15),background.new_tensor(1e14)) if dispersive else (background,)
        reference=batch(*reference_parameters)
    history=[]
    for iteration in range(iterations):
        optimizer.zero_grad()
        result=batch(*parameters())
        ratios=torch.stack([case['detector'].normalized_flux(ref['detector'])
            for case,ref in zip(result.cases,reference.cases)])
        loss=-ratios.mean()+.2*ratios.var(unbiased=False)
        loss.backward()
        history.append(dict(iteration=iteration,loss=float(loss.detach()),
            normalized_flux=ratios.detach().flatten().tolist(),radius_um=float(radius.detach()),
            radius_gradient=float(radius.grad),raw_damping_gradient=float(damping.grad) if dispersive else None,
            batch=result.report))
        optimizer.step()
        with torch.no_grad():radius.clamp_(.08,.35)
    return dict(scope=__doc__,device=device,execution=execution,dispersive=dispersive,
        history=history,final_radius_um=float(radius.detach()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--execution',choices=['resident','streamed','mixed'],default='mixed')
    parser.add_argument('--cases',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=2)
    parser.add_argument('--dispersive',action='store_true')
    parser.add_argument('--output',default='results/ensemble-design.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    result=run(device=args.device,execution=args.execution,cases=args.cases,iterations=args.iterations,dispersive=args.dispersive)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(loss=[row['loss'] for row in result['history']],final_radius_um=result['final_radius_um'])))


if __name__=='__main__':main()
