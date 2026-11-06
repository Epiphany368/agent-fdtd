"""Run with: python -m examples.inverse_design --backend cuda --workers 2.

Optimize a dielectric cylinder's radius for a local, unnormalized intensity
objective. This demonstrates a black-box design workflow, not a calibrated
collection-efficiency objective or a claim of a globally optimal structure.
"""
import argparse
import numpy as np
from torchfdtd import Project,Region,Structure,Source,Monitor,SpectrumSettings,optimize


def local_intensity(result):
    amplitude=result.spectra[0]['value']
    # Fixed source and simulation duration. Scaling only avoids tiny s^2 values.
    return float(np.mean(abs(amplitude/1e-15)**2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--generations',type=int,default=4)
    parser.add_argument('--output',default='results/inverse-design')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    project=Project(name='Cylinder local-intensity design',
        region=Region(size=(5,3,1),mesh=.05,pml_cells=8,steps=800,material_sampling='yee',snapshot_interval=800),
        structures=[Structure(id='cylinder',kind='circle',radius=.4,material='SiN (constant n)')],
        sources=[Source(id='source',center=(-1,0,0),pulse_cycles=1)],
        monitors=[Monitor(id='output',center=(1,0,0),spectrum=SpectrumSettings(sampling='frequency',frequency_points=11,apodization='none'))])
    result=optimize(project,{'structures.0.radius':(.2,.65)},local_intensity,
        backend=args.backend,max_workers=args.workers,population=6,generations=args.generations,seed=42,
        maximize=True,output_dir=args.output,resume=args.resume,objective_key='mean-local-dft-intensity-v1',
        on_generation=lambda info:print(info,flush=True))
    print(result.as_dict())


if __name__=='__main__':main()
