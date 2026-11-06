"""Small CPU optimizer example, not an optically converged device design.

Run: python -m examples.parameterized_density --iterations 2
Resume: add --checkpoint scratch/design.pt, then --resume with a larger total.
"""
import argparse
import json
from pathlib import Path

import torch

from torchfdtd import (AdjointBatchOptions, AdjointExecutionPolicy,
    PeriodicLayerResponse, PlaneReferenceCache, StreamedAdjointOptions)
from torchfdtd.design_parameterization import DensityParameterization


SPEC = dict(wavelength_um=.5, background_index=1.4, design_index=1.8,
    period_um=(1.2, 1.2), height_um=.3, detector_offset_um=.5,
    theta_inside_rad=.1, phi_rad=.3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=2, help='Total optimizer iterations, including resumed iterations.')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.steps < 10:
        parser.error('Use positive iterations and at least ten timesteps.')
    if args.resume and args.checkpoint is None:
        parser.error('--resume requires --checkpoint.')
    torch.set_num_threads(1)
    budget = 256*1024**2
    tiles = StreamedAdjointOptions(device='cpu', slab_width=2, temporal_depth=2,
        checkpoints=1, host_budget_bytes=budget, gpu_budget_bytes=budget)
    solver = PeriodicLayerResponse(SPEC, density_shape=(3, 2), mesh=.2,
        steps=args.steps, pml_cells=3, quadrature_counts=(3, 3),
        forward_kernel='torch', reference_cache=PlaneReferenceCache(1024**2),
        policy=AdjointExecutionPolicy(device='cpu', streamed=tiles, host_budget_bytes=budget),
        batch_options=AdjointBatchOptions(host_budget_bytes=budget, gpu_budget_bytes=budget))
    design = DensityParameterization((3, 2), spacing_um=(.4, .6),
        initial=torch.tensor([[-.8, -.2], [.3, .1], [.6, -.4]]),
        filter_radius_um=.65, symmetry='mirror_x', beta=2.)
    optimizer = torch.optim.SGD(design.parameters(), lr=.1, momentum=.8)
    signature = dict(spec=SPEC, steps=args.steps)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['signature'] != signature:
            raise ValueError('Checkpoint optical specification or timestep count differs.')
        design.load_state_dict(checkpoint['design'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['iteration']
    weights = torch.tensor([[.1, .3, -.2, .7], [.4, -.1, .5, .2]])
    for iteration in range(start, args.iterations):
        optimizer.zero_grad()
        beta_used = float(design.beta)
        response = solver(design())
        # An illustrative coupled optical objective, chosen explicitly here.
        loss = -(response*weights).sum()
        loss.backward()
        gradient_norm = float(design.design.grad.norm())
        optimizer.step()
        # This example owns the continuation schedule. The module never advances it.
        if (iteration+1) % 2 == 0:
            design.advance_beta(2., maximum=8.)
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix+'.tmp')
            torch.save(dict(signature=signature, iteration=iteration+1,
                design=design.state_dict(), optimizer=optimizer.state_dict()), temporary)
            temporary.replace(args.checkpoint)
        print(json.dumps(dict(iteration=iteration+1, loss=float(loss.detach()),
            gradient_norm=gradient_norm, beta_used=beta_used, beta_next=float(design.beta),
            material_input=solver.last_report['plan']['material_input'])))


if __name__ == '__main__':
    main()
