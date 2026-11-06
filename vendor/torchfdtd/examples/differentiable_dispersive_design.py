"""Small geometry/material optimizer through checkpointed Drude/Lorentz FDTD.

This demonstrates a point-spectrum loss, not a normalized optical device score.
"""
import argparse
import json

import torch
from torchfdtd import (AdjointOptions, DispersiveSimulation, StreamedAdjointOptions, StreamedDispersiveSimulation, Monitor, Project,
                        Region, Source, smooth_sphere_epsilon, tune_streamed_dispersive, tune_adjoint_execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--kernel', choices=['torch','fused'], default='torch')
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--streamed', action='store_true', help='Keep geometry and field banks on CPU, stream CUDA tiles')
    parser.add_argument('--state-directory', help='Optional file-backed field banks, requires --streamed')
    parser.add_argument('--tune', action='store_true', help='Measure and select the streamed policy before optimization')
    parser.add_argument('--auto-execution', action='store_true', help='Compare resident and streamed policies with CPU design parameters')
    args = parser.parse_args()
    if args.auto_execution:args.streamed=args.tune=True
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.kernel == 'fused' and args.device != 'cuda':
        parser.error('--kernel fused requires --device cuda')
    if args.streamed and args.device == 'cuda' and args.kernel != 'fused':
        parser.error('CUDA --streamed requires --kernel fused')
    if args.state_directory and not args.streamed:parser.error('--state-directory requires --streamed')
    if args.tune and not args.streamed:parser.error('--tune requires --streamed')
    project = Project(region=Region(dimension='2d',size=(1.6,1.5,1.4),mesh=.1,
        pml_cells=3,steps=50,precision='float64',cuda_kernel=args.kernel),
        sources=[Source(center=(-.2,0,0),pulse='continuous',wavelength=1.1)],
        monitors=[Monitor(component='Ez',center=(.1,0,0))])
    parameter_device = 'cpu' if args.streamed else args.device
    radius = torch.nn.Parameter(torch.tensor(.25,device=parameter_device,dtype=torch.float64))
    raw_damping = torch.nn.Parameter(torch.tensor(-1.,device=parameter_device,dtype=torch.float64))
    optimizer = torch.optim.Adam([radius,raw_damping],lr=.003)
    if args.streamed:
        options = StreamedAdjointOptions(device=args.device,slab_width=5,temporal_depth=4,checkpoints=2,
            local_checkpoints=1,tile_transfers='async' if args.device=='cuda' else 'sync',
            state_storage='disk' if args.state_directory else 'host',state_directory=args.state_directory,
            disk_budget_bytes=256*1024**2 if args.state_directory else None)
        model = StreamedDispersiveSimulation(project,options)
    else:
        model = DispersiveSimulation(project,AdjointOptions(checkpoints=3,storage='host',host_budget_bytes=64*1024**2,
                                                           backward_kernel=args.kernel))
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        density = smooth_sphere_epsilon(project.region,radius,inside=1.,outside=0.,width=.08,yee=True)
        epsilon = 1.2+.6*density
        strength = (.8e30*density)[None]
        damping = (torch.nn.functional.softplus(raw_damping)+.01)*1e15
        if args.tune and iteration == 0:
            tuner = tune_adjoint_execution if args.auto_execution else tune_streamed_dispersive
            selected = tuner(project,epsilon,strength,1.5e15,damping,
                options=model.streaming_options,probe_steps=10,repeats=1,frequency_hz=[2e14])
            model = selected.simulation(project) if args.auto_execution else StreamedDispersiveSimulation(project,selected.options)
            print(json.dumps(dict(tuning=selected.report)),flush=True)
        result = model.spectrum(epsilon,strength,1.5e15,damping,[2e14],block_size=32 if args.auto_execution else 8)
        loss = (result.fields.abs()/ (project.region.steps*project.region.time_step)).square().mean()
        loss.backward()
        print(json.dumps(dict(iteration=iteration,point_spectrum_loss=float(loss.detach()),
            radius_um=float(radius.detach()),damping_rad_s=float(damping.detach()),
            radius_gradient=float(radius.grad),raw_damping_gradient=float(raw_damping.grad),
            material_state_bytes=result.report['material_state_bytes'],
            material_parameter_bytes=result.report['material_parameter_bytes'],
            forward_backend=result.report['forward_backend'],backward_backend=result.report['backward_backend'],
            checkpoint_bytes=result.report.get('restart_bytes'),
            spatial_streaming=result.report.get('spatial_streaming',False),
            state_bytes=result.report.get('state_bytes'))),flush=True)
        optimizer.step()
        with torch.no_grad():
            radius.clamp_(.05,.5)


if __name__ == '__main__':
    main()
