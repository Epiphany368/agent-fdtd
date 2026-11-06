"""Small isotropic dispersive voxel lens and its density gradient.

Run on CPU: python -m examples.reversible_lorentz_voxel
Run on one GPU: python -m examples.reversible_lorentz_voxel --device cuda
The coarse grid demonstrates the API, not a converged optical lens design.
"""
import argparse
import json
import torch

from torchfdtd import (Project, Region, Boundaries, BoundaryFace, Source, FieldMonitor,
    ReversibleCPMLPlaneSimulation, ReversibleCPMLOptions, fit_discrete_lorentz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args = parser.parse_args()
    periodic = BoundaryFace(kind='periodic')
    pml = BoundaryFace(kind='pml',layers=3)
    project = Project(name='Small dispersive voxel lens',region=Region(dimension='3d',
        size=(1.,1.,2.),mesh=.05,steps=256,mesh_auto_refine=False,material_sampling='yee',
        precision='float32',backend=args.device,boundaries=Boundaries(x_min=periodic,
            x_max=periodic,y_min=periodic,y_max=periodic,z_min=pml,z_max=pml)),
        sources=[Source(kind='plane',normal='z',center=(0,0,-.6),size=(1.,1.,0),
            component='Ex', wavelength=.55,time_definition='standard',pulse_length=.6e-15,pulse_offset=1.2e-15)],
        monitors=[FieldMonitor(id='exit',normal='z',center=(0,0,.6),size=(.8,.8,0),
            spectrum=dict(sampling='frequency',apodization='none'))])
    # Campaign two-term approximation to the Siefke ALD TiO2 table.
    # These coefficients are not a formula published in the source paper.
    fit = fit_discrete_lorentz(sellmeier_coefficients=(
        (.6377579417631474,.09932777016261009),
        (3.5250338280915408,.034835477625238975)),
        dt_s=project.region.time_step,wavelength_range_um=(.42,.67),name='TiO2 approximation')
    material = fit.require_tolerance()
    model = ReversibleCPMLPlaneSimulation(project,ReversibleCPMLOptions(
        offload_terminal=args.device=='cuda'),material=material,
        quadrature_counts={'exit':(3,3)})
    a,b = model.interior_z
    nx,ny,nz = project.region.shape
    x = (torch.arange(nx,device=args.device)+.5)/nx-.5
    y = (torch.arange(ny,device=args.device)+.5)/ny-.5
    z = (torch.arange(b-a+1,device=args.device)+.5)/(b-a+1)
    radius_squared = x[:,None,None].square()+y[None,:,None].square()
    rho = ((radius_squared < (.32-.1*z[None,None,:]).square()) &
        (z[None,None,:]>.35) & (z[None,None,:]<.65)).float().contiguous().requires_grad_()
    background = torch.where(torch.arange(nz,device=args.device)<nz//2,1.444**2,1.).float()
    frequencies = rho.new_tensor([299792458/(v*1e-9) for v in (450,550,650)])
    result = model(rho,frequencies,fixed_epsilon=background)['exit']
    # Normalize the Fourier integral by dt only to keep this example's loss
    # and density derivative on a readable numerical scale.
    objective = (result.fields/project.region.time_step).abs().square().mean()
    gradient = torch.autograd.grad(objective,rho)[0]
    if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().max()) == 0:
        raise RuntimeError('The example must produce a finite, nonzero density derivative.')
    print(json.dumps(dict(shape=project.region.shape,steps=project.region.steps,
        fit_max_abs_n_error=fit.report['max_abs_n_error'],objective=float(objective.detach()),
        max_abs_density_gradient=float(gradient.abs().max()),
        pole_state_bytes=result.report['pole_state_bytes'],
        terminal_offload_used=result.report['terminal_offload_used'],backend=result.report['backend']),indent=2))


if __name__ == '__main__':
    main()
