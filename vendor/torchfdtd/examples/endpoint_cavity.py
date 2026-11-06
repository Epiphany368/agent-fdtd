"""PMC/PEC cavity with exact endpoints and material/source gradients.

Run: python examples/endpoint_cavity.py --device cuda
The CPU backend is a bounded reference. CUDA uses the direct endpoint kernels.
"""
import argparse
import numpy as np
import torch
from torchfdtd import EndpointSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    simulation = EndpointSimulation([np.linspace(-.4, .4, 9)]*3,
        [('pmc', 'pmc')]*3, dt_seconds=.04/(299792458.*1e6),
        sources=[(0, (4, 4, 4))], observations=[('E', 0, (4, 4, 4))],
        checkpoints=4, device=args.device, tensor_budget_bytes=64*1024**2)
    material = torch.tensor(2., device=args.device, requires_grad=True)
    epsilon = simulation.sample_epsilon(lambda xyz, component:material.expand(len(xyz)))
    amplitude = torch.tensor(1., device=args.device, requires_grad=True)
    time = torch.arange(24, device=args.device, dtype=torch.float32)
    drive = (amplitude*torch.exp(-((time-6)/2)**2))[:, None]
    trace = simulation(epsilon, drive)
    objective = trace.square().mean()
    objective.backward()
    print({'objective':objective.item(), 'd_objective_d_epsilon':material.grad.item(),
           'd_objective_d_amplitude':amplitude.grad.item(),
           'replayed_steps':simulation.last_report['replayed_steps'],
           'tensor_budget':simulation.memory_plan(len(drive))})


if __name__ == '__main__':
    main()
