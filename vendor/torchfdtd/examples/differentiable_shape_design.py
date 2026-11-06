"""FP32 analytic shape optimization using a local, unnormalized field objective."""
import argparse
import json
from pathlib import Path

import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, DifferentiableSolid,
    Monitor, Project, Region, Source, StreamedAdjointOptions, StreamedSimulation,
    smooth_geometry_epsilon)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--execution',choices=('resident','dram'),default='resident')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--output',default='results/shape-design.json')
    args=parser.parse_args()
    if args.iterations<1:raise ValueError('iterations must be positive.')
    project=Project(region=Region(dimension='3d',size=(1.6,1.5,1.4),mesh=.1,pml_cells=3,
        steps=60,precision='float32',material_sampling='yee'),
        sources=[Source(center=(-.3,0,0),pulse='continuous',wavelength=1.1)],
        monitors=[Monitor(center=(.3,.1,0)),Monitor(center=(.2,-.1,.1))])
    geometry_device='cpu' if args.execution=='dram' else args.device
    shape=torch.nn.Parameter(torch.tensor([.22,.36,15.],device=geometry_device,dtype=torch.float32))
    if args.execution=='dram':
        model=StreamedSimulation(project,StreamedAdjointOptions(device=args.device,
            slab_width=8,temporal_depth=4,checkpoints=2,gpu_budget_bytes=512*1024**2,
            host_budget_bytes=1024**3,tile_transfers='async' if args.device=='cuda' else 'sync'))
    else:
        model=DifferentiableSimulation(project,AdjointOptions(checkpoints=3))
    optimizer=torch.optim.Adam([shape],lr=.002)
    history=[]
    for step in range(args.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        solids=[DifferentiableSolid.cylinder(shape[0],shape[1],epsilon=3.,
                    radius_y=.17,rotation=(0.,shape[2],10.)),
                DifferentiableSolid.box((.12,.3,.25),epsilon=2.,center=(.22,-.1,0.))]
        epsilon=smooth_geometry_epsilon(project.region,solids,width=.1,chunk_cells=1024)
        result=model(epsilon)
        score=result.signals.square().mean()
        row=dict(update=step,score=float(score.detach()),parameters=shape.detach().cpu().tolist())
        history.append(row)
        if step==args.iterations:break
        (-score).backward()
        row['gradient']=shape.grad.detach().cpu().tolist()
        optimizer.step()
        with torch.no_grad():
            shape[:2].clamp_(.1,.45)
            shape[2].clamp_(-45.,45.)
    record=dict(precision='float32',execution=args.execution,device=args.device,history=history,
        scope='Regularized shape and optimizer connectivity. Local point-field energy is not normalized transmission or a converged physical device.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':main()
