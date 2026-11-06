"""Small fixed-port, open-cladding inverse-design example.

Run from the repository with Python. This scalar demonstration uses the same
validated fiber geometry as the open-mode acceptance case. Its optimization
objective is a selected modal transmission, not total transmitted energy.
"""
import argparse
import numpy as np
import torch
from torchfdtd import (Project, Region, Source, AdjointOptions,
                       FixedModePort, ModeNetwork, OpenPortOptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    project = Project(
        region=Region(dimension='3d', size=(8.,4.,4.), mesh=.1, pml_cells=8,
                      steps=1300, material_sampling='yee', precision='float32'),
        sources=[Source(kind='plane', normal='x', center=(-2.,0.,0.),
                        size=(0.,2.4,2.4), pulse_cycles=2)], monitors=[])
    ports = (FixedModePort('left',-1.,-2.,1), FixedModePort('right',1.,2.,-1))
    def section(y,z):
        return np.where(y*y+z*z < .4**2-1e-10, 2.2**2, 1.)
    network = ModeNetwork(project, ports, section, AdjointOptions(checkpoints=8),
                          num_modes=1, open_ports=OpenPortOptions(1.))
    base = network.reference_epsilon(device=args.device)
    mask = torch.zeros_like(base)
    mask[38:42,17:23,17:23] = 1
    increment = torch.tensor(.3,device=base.device,requires_grad=True)
    optimizer = torch.optim.Adam([increment],lr=.02)
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        result = network(base+increment*mask)
        transmission = result.s[1,0].abs().square()
        (-transmission).backward()
        print(dict(iteration=iteration,epsilon_increment=float(increment.detach()),
                   selected_modal_transmission=float(transmission.detach()),
                   objective_gradient=float(increment.grad)),flush=True)
        optimizer.step()
        with torch.no_grad():
            increment.clamp_(0.,1.)


if __name__ == '__main__':
    main()
