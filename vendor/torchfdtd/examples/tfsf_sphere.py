"""Closed-box TFSF sphere scattering and independent analytic Mie series.

Run: python -m examples.tfsf_sphere --backend cuda --meshes 0.1 0.05 0.025
The meshes retain domain size, PML thickness and physical duration. Sources,
geometry and results are native and independently authored.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import spherical_jn,spherical_yn

from torchfdtd import Project,Region,Source,Structure,Material,FieldMonitor,SpectrumSettings,Simulation,RunControl


def mie_cross_section(wavelength_um,radius_um,index):
    """Lossless nonmagnetic sphere in vacuum, using direct Bessel functions.

    Standard Riccati-Bessel coefficients, independently evaluated with SciPy.
    This small-size-parameter reference is not a general robust Mie package.
    """
    values=[]
    for wavelength in np.atleast_1d(wavelength_um):
        x=2*np.pi*radius_um/wavelength;m=index
        orders=np.arange(1,math.ceil(x+4*np.cbrt(x)+10)+1)
        def psi(z):
            j=spherical_jn(orders,z)
            return z*j,j+z*spherical_jn(orders,z,derivative=True)
        px,dx=psi(x);pm,dm=psi(m*x)
        y=spherical_yn(orders,x);dy=spherical_yn(orders,x,derivative=True)
        xi=px+1j*x*y;dxi=dx+1j*(y+x*dy)
        a=(m*pm*dx-px*dm)/(m*pm*dxi-xi*dm)
        b=(pm*dx-m*px*dm)/(pm*dxi-m*xi*dm)
        values.append(2*np.pi/(2*np.pi/wavelength)**2*np.sum((2*orders+1)*(abs(a)**2+abs(b)**2)))
    return np.asarray(values)


def make_project(mesh=.05,backend='auto',direction='+',duration_fs=120.):
    spectrum=SpectrumSettings(sampling='wavelength',wavelength_start=1.3,wavelength_stop=1.8,
                              frequency_points=9,apodization='none')
    r=Region(dimension='3d',size=(3.2,3.2,3.2),mesh=mesh,pml_cells=round(.4/mesh),
        backend=backend,cuda_kernel='fused' if backend!='cpu' else 'torch',precision='float32',
        material_sampling='yee',snapshot_interval=10000,run_control=RunControl(divergence_check=True))
    r.steps=math.ceil(duration_fs*1e-15/r.time_step)
    monitors=[]
    for axis in range(3):
        for side,sign in (('min',-1),('max',1)):
            center=tuple(sign*1.1 if a==axis else 0 for a in range(3))
            size=tuple(0 if a==axis else 2.2 for a in range(3))
            monitors.append(FieldMonitor(id='xyz'[axis]+'_'+side,name='xyz'[axis]+'_'+side,
                normal='xyz'[axis],center=center,size=size,spectrum=spectrum))
    monitors.append(FieldMonitor(id='incident',name='incident',center=(0,0,0),size=(0,.4,.4),spectrum=spectrum))
    return Project(name='TFSF dielectric sphere',region=r,
        materials=[Material(name='sphere',index=1.5)],
        structures=[Structure(kind='sphere',radius=.3,material='sphere')],
        sources=[Source(kind='tfsf',size=(1.6,1.6,1.6),component='Ez',normal='x',direction=direction,
            wavelength=1.55,time_definition='standard',pulse_length=8e-15,pulse_offset=30e-15)],
        monitors=monitors)


def evaluate(sample,reference,*,radius=.3,index=1.5):
    incident=reference.field_monitor('incident')
    intensity=abs(incident['flux'])/sum(incident['weights'])
    if not np.all(intensity>.01*max(intensity)):raise ValueError('Incident spectrum is too weak.')
    power=np.zeros_like(intensity);empty=np.zeros_like(intensity)
    for axis in 'xyz':
        for side,sign in (('min',-1),('max',1)):
            a=sample.field_monitor(axis+'_'+side);b=reference.field_monitor(axis+'_'+side)
            fields=a['fields']-b['fields']
            flux=.5*np.real(np.cross(fields[...,:3],fields[...,3:].conj()))[...,'xyz'.index(axis)]@a['weights']
            power+=sign*flux
            empty+=abs(b['flux'])
    wavelength=299792458/incident['frequency_hz']*1e6
    actual=power/intensity*1e12
    analytic=mie_cross_section(wavelength,radius,index)
    return dict(wavelength_um=wavelength.tolist(),scattering_cross_section_um2=actual.tolist(),
        mie_cross_section_um2=analytic.tolist(),
        max_relative_error=float(max(abs(actual/analytic-1))),
        max_empty_box_cross_section_um2=float(max(empty/intensity*1e12)),
        project=sample.project.model_dump(),sample_summary=sample.summary,reference_summary=reference.summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--meshes',nargs='+',type=float,default=[.1,.05,.025])
    parser.add_argument('--direction',choices=['+','-'],default='+')
    parser.add_argument('--duration-fs',type=float,default=120.)
    parser.add_argument('--output',default='results/tfsf-sphere.json')
    args=parser.parse_args();rows=[]
    for mesh in args.meshes:
        p=make_project(mesh,args.backend,args.direction,args.duration_fs)
        sample=Simulation(p).run();p.structures=[];reference=Simulation(p).run()
        row=evaluate(sample,reference);rows.append(row)
        print(json.dumps(dict(mesh=mesh,max_relative_error=row['max_relative_error'],
            empty_cross_section=row['max_empty_box_cross_section_um2'])),flush=True)
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(dict(method='Closed scattered-field surface flux divided by homogeneous incident intensity, versus direct analytic Mie series. Matched empty-box reference, unchanged physical domain/PML/time.',cases=rows),indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
