"""Differentiate a rotated bulk tensor on a periodic 3D Yee grid.

Run: python examples/tensor_dielectric.py --device cuda
This bounded example uses impressed-field sources, without CPML or dispersion.
"""
import argparse
import torch
from torchfdtd import (AdjointOptions, Monitor, Project, Region, Source,
                       TensorDielectricSimulation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    region = Region(dimension='3d', size=(.8, .8, .8), mesh=.1, steps=24,
        material_sampling='yee', boundaries={axis+'_'+side: {'kind':'periodic'}
            for axis in 'xyz' for side in ('min', 'max')})
    project = Project(region=region, sources=[Source(component='Ex',
        wavelength=1.55, pulse='continuous')],
        monitors=[Monitor(component='Ey', center=(.1, 0, 0))])
    angle = torch.tensor(.35, device=args.device, requires_grad=True)
    zero, one = angle*0, angle*0+1
    rotation = torch.stack((torch.stack((angle.cos(), -angle.sin(), zero)),
        torch.stack((angle.sin(), angle.cos(), zero)), torch.stack((zero, zero, one))))
    tensor = rotation @ torch.diag(angle.new_tensor([1.5, 2.5, 3.])) @ rotation.T
    tensor = (tensor+tensor.T)/2
    epsilon = tensor.expand(region.shape+(3, 3))
    simulation = TensorDielectricSimulation(project, AdjointOptions(checkpoints=4))
    result = simulation(epsilon)
    objective = result.signals.abs().square().sum()
    objective.backward()
    print({'objective':objective.item(), 'd_objective_d_angle':angle.grad.item(),
           'replayed_steps':result.report['replayed_steps'],
           'scope':'periodic nondispersive bulk tensor, first derivative'})


if __name__ == '__main__':
    main()
