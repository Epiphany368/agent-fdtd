"""Create four independent analytic CAD scenes, export them and solve in Python.

GPU example: python examples/analytic_solids.py --backend cuda
CPU example: python examples/analytic_solids.py --backend cpu
"""
import argparse
from pathlib import Path
import numpy as np
from torchfdtd import Project,Region,Structure,Source,Monitor,Simulation,run_tensor_batch


def projects(backend='cuda'):
    objects=[
        Structure(kind='polygon',vertices=((- .5,-.4),(.5,-.4),(.5,0),(0,0),(0,.4),(-.5,.4)),size=(1,1,.6)),
        Structure(kind='sphere',radius=.5,radius_2=.3,radius_3=.2,make_ellipsoid=True),
        Structure(kind='circle',radius=.5,radius_2=.3,make_ellipsoid=True,size=(1,1,.6)),
        Structure(kind='ring',radius=.5,radius_2=.4,inner_radius=.3,inner_radius_2=.2,make_ellipsoid=True,theta_start=315,theta_stop=180)]
    result=[]
    for i,obj in enumerate(objects):
        obj.rotation_axes=('x','y','z');obj.rotation_angles=(17,31,-29)
        result.append(Project(name=f'analytic-solid-{i}',
            region=Region(dimension='3d',size=(4,4,4),mesh=.1,pml_cells=5,steps=180,material_sampling='yee',
                          backend=backend,cuda_kernel='fused' if backend=='cuda' else 'torch',snapshot_interval=60),structures=[obj],
            sources=[Source(center=(-.8,0,0),wavelength=1,pulse_cycles=1)],monitors=[Monitor(center=(.8,0,0))]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--output',default='results/analytic-solids');args=ap.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);scenes=projects(args.backend)
    for i,p in enumerate(scenes):p.save(output/f'scene-{i}.json')
    if args.backend=='cuda':
        report=run_tensor_batch(scenes,cohort_size=4,output_dir=output,objective=lambda r:float(np.max(abs(r.signals))))
        report.raise_for_errors()
        print([(v.id,v.metrics) for v in report.items])
    else:
        for i,p in enumerate(scenes):
            r=Simulation(p).run();r.save(output/f'cpu-{i}.npz');print(p.name,float(np.max(abs(r.signals))))


if __name__=='__main__':main()
