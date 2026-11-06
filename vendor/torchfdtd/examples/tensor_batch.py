"""Python-only independent sphere sweep using a CUDA batch axis."""
import argparse
import numpy as np
from torchfdtd import Project, Region, Material, Structure, Source, Monitor, parameter_sweep, run_tensor_batch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort-size',type=int,default=4)
    parser.add_argument('--output')
    args=parser.parse_args()
    base=Project(region=Region(dimension='3d',size=(4.8,4.8,4.8),mesh=.075,
        backend='cuda',steps=800,snapshot_interval=800,material_sampling='yee',pml_cells=8),
        materials=[Material(name='glass',index=1.5)],
        structures=[Structure(kind='sphere',radius=.4,material='glass')],
        sources=[Source(center=(-1.2,0,0),wavelength=1.55,pulse_cycles=2)],
        monitors=[Monitor(center=(1.2,0,0))])
    cases=parameter_sweep(base,{'structures.0.radius':np.linspace(.35,.5,16).tolist()})
    report=run_tensor_batch(cases,cohort_size=args.cohort_size,
        objective=lambda r:float(np.max(abs(r.signals))),output_dir=args.output,keep_results=False)
    report.raise_for_errors()
    print(f'{len(cases)} cases in {report.seconds:.4f} s, {len(cases)/report.seconds:.2f} objective evaluations/s')
    for item in report.items:print(item.id,item.parameters,item.metrics)


if __name__=='__main__':main()
