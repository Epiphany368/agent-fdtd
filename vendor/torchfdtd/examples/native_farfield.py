"""Create a six-face dipole Project or postprocess its saved native NPZ.

The coarse example demonstrates stored-field workflow, not optical convergence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from torchfdtd import Project, Region, Source
from torchfdtd.models import FieldMonitor, SpectrumSettings


def make_project():
    region=Region(dimension='3d',size=(2.4,2.4,2.4),mesh=.1,precision='float32',
        material_sampling='yee',backend='cpu',pml_cells=3,steps=100,snapshot_interval=500)
    region.steps=round(100e-15/region.time_step)
    spectrum=SpectrumSettings(sampling='custom',custom_frequencies_hz=[299792458./1.1e-6],apodization='none')
    monitors=[]
    for axis,normal in enumerate('xyz'):
        for side,coordinate in (('min',-.6),('max',.6)):
            center,size=[0.,0.,0.],[1.2,1.2,1.2]
            center[axis],size[axis]=coordinate,0.
            monitors.append(FieldMonitor(id=normal+'_'+side,name=normal+' '+side+' radiation face',
                normal=normal,center=tuple(center),size=tuple(size),spectrum=spectrum.model_copy(deep=True),
                time_downsample=1,downsample=1,spatial_interpolation='specified',
                record_fields=('Ex','Ey','Ez','Hx','Hy','Hz')))
    return Project(name='Isolated dipole far field',region=region,structures=[],
        sources=[Source(id='dipole',kind='point',component='Ez',center=(0.,0.,0.),
            wavelength=1.1,pulse='gaussian',pulse_cycles=2.)],monitors=monitors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-project')
    parser.add_argument('--result',help='Previously computed native NPZ with all six fields on six faces')
    parser.add_argument('--output',default='farfield.json')
    args=parser.parse_args()
    if args.export_project:
        make_project().save(args.export_project);return
    if not args.result:parser.error('Choose --export-project or --result. This command never starts FDTD.')
    from torchfdtd.radiation_box_io import load_native_radiation_box
    faces={a+'_'+s:a+'_'+s for a in 'xyz' for s in ('min','max')}
    box=load_native_radiation_box(args.result,faces,bounds_um=[[-.6,.6]]*3,refractive_index=1.)
    theta=np.linspace(0,np.pi,37)
    directions=np.stack((np.sin(theta),np.zeros_like(theta),np.cos(theta)),axis=-1)
    far=box.project(directions)
    amplitude=far.electric_amplitude[0]
    intensity=.5*(amplitude.real.double().square()+amplitude.imag.double().square()).sum(-1).numpy()
    result=dict(theta_deg=np.rad2deg(theta).tolist(),electric_real=amplitude.real.tolist(),
        electric_imag=amplitude.imag.tolist(),intensity=intensity.tolist(),
        relative_intensity=(intensity/intensity.max() if intensity.max()>0 else intensity).tolist(),
        intensity_units='reduced E*H * s^2 * m^2 / sr',report=box.report)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':main()
