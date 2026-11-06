"""Joint fixed-source temporal and material optimization using the public API.

This small periodic example fits a chosen point-field trace. It illustrates
amplitude/phase and epsilon VJPs, not a calibrated device objective or a physical
convergence result. Spatial positions, source profiles and the mesh remain fixed.
No calculation happens at import. Run explicitly, for example:
    python examples/differentiable_sources.py --device cpu --iterations 3
"""
import argparse
import json

import torch

from torchfdtd import AdjointOptions, Monitor, Project, Region, Source
from torchfdtd.source_adjoint import SourceWaveformSimulation
from torchfdtd.source_parameters import gaussian_waveform
from torchfdtd.solver import field_axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    torch.set_num_threads(1)
    dtype = torch.float32
    region = Region(dimension='3d', size=(.8, .8, 1.), mesh=.1, steps=48,
        precision='float32', material_sampling='yee', backend=args.device,
        boundaries={f'{axis}_{side}': {'kind':'periodic'}
                    for axis in 'xyz' for side in ('min', 'max')})

    def position(component, index):
        axes = field_axes(region, component)
        return tuple(float(axes[a][index[a]]) for a in range(3))

    project = Project(name='Differentiable electric and magnetic pulses', region=region,
        sources=[
            Source(id='electric', name='Electric point', kind='point', injection='soft',
                component='Ex', center=position('Ex', (3, 3, 2)), wavelength=.6),
            Source(id='magnetic', name='Magnetic point', kind='point', injection='soft',
                component='Hy', center=position('Hy', (3, 3, 7)), wavelength=.6)],
        monitors=[Monitor(id='target', name='Target Ex', component='Ex',
                          center=position('Ex', (3, 3, 5)))])
    model = SourceWaveformSimulation(project, AdjointOptions(checkpoints=4,
        host_budget_bytes=256*1024**2, gpu_budget_bytes=512*1024**2,
        resident_budget_bytes=512*1024**2))
    times = model.source_times(device=args.device)
    layout = model.term_layout
    assert [(term['source_id'], term['component']) for term in layout] == [
        ('electric', 'Ex'), ('magnetic', 'Hy')]
    assert times.shape == (region.steps, 2) and times.dtype == dtype
    # Each column is an actual additive increment. Point polarization weights
    # are one here. This custom envelope does not copy native pulse amplitudes.
    amplitude = torch.nn.Parameter(torch.tensor([.02, .01], device=args.device, dtype=dtype))
    phase = torch.nn.Parameter(torch.tensor([0., .4], device=args.device, dtype=dtype))
    material_parameter = torch.nn.Parameter(torch.tensor(.2, device=args.device, dtype=dtype))
    material_mask = torch.zeros(region.shape, device=args.device, dtype=dtype)
    material_mask[:, :, 4:6] = 1.
    observer_times = (torch.arange(region.steps, device=args.device, dtype=dtype)+1)*region.time_step
    target = gaussian_waveform(observer_times, frequency_hz=299792458/.6e-6,
        sigma_s=4*region.time_step, delay_s=14*region.time_step, amplitude=.004)
    optimizer = torch.optim.Adam([amplitude, phase, material_parameter], lr=.01)
    records = []
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        epsilon = 1.2 + material_mask*torch.nn.functional.softplus(material_parameter)
        waveforms = torch.stack([gaussian_waveform(times[:, i],
            frequency_hz=299792458/.6e-6, sigma_s=4*region.time_step,
            delay_s=8*region.time_step, amplitude=amplitude[i], phase_rad=phase[i])
            for i in range(2)], dim=1)
        response = model(epsilon, waveforms)
        loss = (response.signals[:, 0]-target).square().mean()
        loss.backward()
        records.append(dict(iteration=iteration, loss=float(loss.detach()),
            amplitude=amplitude.detach().cpu().tolist(), phase_rad=phase.detach().cpu().tolist(),
            material_parameter=float(material_parameter.detach()),
            amplitude_gradient=amplitude.grad.detach().cpu().tolist(),
            phase_gradient=phase.grad.detach().cpu().tolist(),
            material_gradient=float(material_parameter.grad.detach())))
        optimizer.step()
        del response, loss, epsilon, waveforms
    print(json.dumps(dict(scope='Fixed positions and profiles, joint first-order temporal/material derivatives',
        device=args.device, precision='float32', term_layout=layout, iterations=records), indent=2))


if __name__ == '__main__':
    main()
