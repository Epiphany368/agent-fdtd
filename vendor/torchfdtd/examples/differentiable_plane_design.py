"""Optimize a sphere radius and optional damping using matched detector flux.

This short 2D example demonstrates a complete geometry/material optimizer
graph. Its finite-duration dipole illumination is not a converged device
efficiency or a color-router design. Frequencies and detector planes are fixed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from torchfdtd import (AdjointExecutionPolicy, AdjointOptions, FieldMonitor,
    Project, Region, Source, StreamedAdjointOptions, smooth_sphere_epsilon,
    tune_adjoint_execution)


def run(*,device='cpu',execution='auto',dispersive=False,iterations=3):
    if iterations<1:raise ValueError('iterations must be positive')
    if execution not in ('auto','resident','streamed'):raise ValueError('Unknown execution mode')
    project=Project(region=Region(dimension='2d',size=(2.,1.6,1.4),mesh=.1,
        pml_cells=3,steps=80,precision='float64',cuda_kernel='fused' if device=='cuda' else 'torch'),
        sources=[Source(center=(-.45,0,0),pulse='continuous',wavelength=1.1)],
        monitors=[FieldMonitor(id='detector',center=(.4,0,0),size=(0,.7,.6),normal='x',downsample=2)])
    frequency=[299792458/1.1e-6]
    radius=torch.nn.Parameter(torch.tensor(.22,dtype=torch.float64))
    damping=torch.nn.Parameter(torch.tensor(-1.,dtype=torch.float64))
    optimizer=torch.optim.Adam([radius,damping] if dispersive else [radius],lr=.002)
    budget=256*1024**2
    streamed=StreamedAdjointOptions(device=device,slab_width=6,temporal_depth=4,
        checkpoints=2,host_budget_bytes=budget,gpu_budget_bytes=budget,
        tile_transfers='async' if device=='cuda' else 'sync')

    def materials():
        density=smooth_sphere_epsilon(project.region,radius,inside=1.,outside=0.,width=.08,yee=True)
        epsilon=1.+density
        if not dispersive:return (epsilon,)
        return epsilon,(.8e30*density)[None],1.5e15,(torch.nn.functional.softplus(damping)+.01)*1e15

    tuning=None
    if execution=='auto':
        selection=tune_adjoint_execution(project,*materials(),options=streamed,
            frequency_hz=frequency,probe_steps=10,repeats=1)
        policy,tuning=selection.policy,selection.report
    elif execution=='resident':
        policy=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=2,
            resident_budget_bytes=budget,gpu_budget_bytes=budget,
            backward_kernel='fused' if device=='cuda' else 'torch'),device=device,host_budget_bytes=budget)
    else:policy=AdjointExecutionPolicy(streamed=streamed,device=device,host_budget_bytes=budget)
    model=policy.simulation(project,dispersive=dispersive)
    with torch.no_grad():
        background=torch.ones_like(materials()[0])
        reference_materials=(background,[0.],1.5e15,1e14) if dispersive else (background,)
        reference=model(*reference_materials,frequency_hz=frequency)['detector']
    history=[]
    for iteration in range(iterations):
        optimizer.zero_grad()
        result=model(*materials(),frequency_hz=frequency)['detector']
        ratio=result.normalized_flux(reference)
        loss=-ratio.mean()
        loss.backward()
        history.append(dict(iteration=iteration,loss=float(loss.detach()),
            normalized_flux=ratio.detach().tolist(),radius_um=float(radius.detach()),
            radius_gradient=float(radius.grad),
            raw_damping=float(damping.detach()) if dispersive else None,
            raw_damping_gradient=float(damping.grad) if dispersive else None,
            execution=result.report.copy()))
        optimizer.step()
        with torch.no_grad():radius.clamp_(.08,.35)
    return dict(scope=__doc__,device=device,execution=execution,dispersive=dispersive,
        policy=asdict(policy),tuning=tuning,history=history,final_radius_um=float(radius.detach()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--execution',choices=['auto','resident','streamed'],default='auto')
    parser.add_argument('--dispersive',action='store_true')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--output',default='results/plane-design.json')
    args=parser.parse_args()
    result=run(device=args.device,execution=args.execution,dispersive=args.dispersive,iterations=args.iterations)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(initial_loss=result['history'][0]['loss'],
        last_evaluated_loss=result['history'][-1]['loss'],final_radius_um=result['final_radius_um'])))


if __name__=='__main__':main()
