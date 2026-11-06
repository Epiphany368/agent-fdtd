"""A complete Torch geometry -> epsilon -> FDTD -> loss -> Adam example.

This small regularized-shape example minimizes point-field energy. It is not
port-normalized transmission, a manufactured device, or a physical shape-
gradient convergence validation.
"""
import argparse
import json
from pathlib import Path

import torch

from torchfdtd import (AdjointOptions,DifferentiableSimulation,Monitor,Project,
                        Region,Source,smooth_sphere_epsilon,StreamedSimulation,StreamedAdjointOptions,
                        estimate_streamed_memory,select_streamed_storage)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--iterations',type=int,default=8)
    ap.add_argument('--execution',choices=('resident','streamed','disk','auto'),default='resident')
    ap.add_argument('--state-directory',default='results/design-state-scratch')
    ap.add_argument('--output',default='results/differentiable-design.json')
    ap.add_argument('--frequency-hz',type=float,nargs='+',
                    help='Optimize selected point spectra instead of time-domain field energy')
    args=ap.parse_args()
    if args.iterations<1:ap.error('--iterations must be positive')
    project=Project(region=Region(size=(1.6,1.5,1.4),mesh=.1,pml_cells=3,
                                  steps=40,precision='float64'),
                    sources=[Source(center=(-.2,0,0),pulse='continuous',wavelength=1.1)],
                    monitors=[Monitor(center=(.1,0,0))])
    geometry_device=args.device if args.execution=='resident' else 'cpu'
    radius=torch.nn.Parameter(torch.tensor(.25,device=geometry_device,dtype=torch.float64))
    memory_plan=None
    storage_selection=None
    observation={} if args.frequency_hz is None else dict(frequency_hz=args.frequency_hz)
    if args.execution=='resident':
        model=DifferentiableSimulation(project,AdjointOptions(checkpoints=4,storage='host',host_budget_bytes=128*1024**2))
    else:
        options=StreamedAdjointOptions(device=args.device,slab_width=4,temporal_depth=4,
                                gpu_budget_bytes=128*1024**2,host_budget_bytes=128*1024**2,
                                state_storage='disk' if args.execution=='disk' else 'host',
                                state_directory=args.state_directory if args.execution in ('disk','auto') else None,
                                disk_budget_bytes=128*1024**2 if args.execution in ('disk','auto') else None)
        if args.execution=='auto':
            plan=select_streamed_storage(project,options,**observation)
            options=plan.options
            storage_selection=dict(selected=options.state_storage,rejected=plan.rejected)
        # Check capacity before constructing the full-domain geometry tensor.
        # Geometry and optimizer allocations are additional caller-owned memory.
        memory_plan=estimate_streamed_memory(project,options,**observation)
        model=StreamedSimulation(project,options)
    optimizer=torch.optim.Adam([radius],lr=.003)
    history=[]
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        epsilon=smooth_sphere_epsilon(project.region,radius,width=.09,inside=3.)
        if args.frequency_hz is None:
            result=model(epsilon)
            loss=result.signals[:,0].abs().square().mean()
        else:
            result=model.spectrum(epsilon,**observation)
            # DFT includes dt. Normalize by observation duration to keep the
            # point-field objective on a useful scale for Adam's epsilon.
            # This is not incident-power or port transmission normalization.
            loss=(result.fields[:,0]/(project.region.steps*project.region.time_step)).abs().square().mean()
        loss.backward()
        history.append(dict(iteration=iteration,radius_um=float(radius.detach()),
                            loss=float(loss.detach()),gradient=float(radius.grad),execution=result.report.copy()))
        optimizer.step()
        with torch.no_grad():radius.clamp_(.1,.4)
    output=dict(description=__doc__,device=args.device,geometry_device=geometry_device,execution=args.execution,
                observation='time_energy' if args.frequency_hz is None else 'duration_normalized_point_spectrum',
                frequency_hz=args.frequency_hz,
                memory_plan=memory_plan,storage_selection=storage_selection,
                history=history,final_radius_um=float(radius.detach()))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'initial_loss':history[0]['loss'],'last_evaluated_loss':history[-1]['loss'],
                      'final_radius_um':output['final_radius_um']}))


if __name__=='__main__':main()
