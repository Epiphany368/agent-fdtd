"""Independent GPU sphere spectra with shared monitor kernels and Python only."""
import argparse
import json

from torchfdtd import (Project, Region, Material, Structure, Source, FieldMonitor,
                         SpectrumSettings, BatchCase, run_tensor_batch)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--cohort',type=int,default=4)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--graph-steps',type=int,default=1,
                        help='Unchanged time steps per CUDA graph replay, 1--64. Larger graphs can be slower.')
    args=parser.parse_args()
    cases=[]
    for i in range(args.count):
        radius=.3+.04*i
        project=Project(name=f'Sphere spectrum {i}',region=Region(
            dimension='3d',size=(4.8,4.8,4.8),mesh=.15,pml_cells=4,
            steps=args.steps,snapshot_interval=args.steps,material_sampling='yee',
            backend='cuda',cuda_kernel='fused',cuda_monitor_kernel='fused'),
            materials=[Material(name='sphere dielectric',index=1.5)],
            structures=[Structure(kind='sphere',radius=radius,material='sphere dielectric')],
            sources=[Source(kind='tfsf',size=(2.4,2.4,2.4),normal='x',
                            wavelength=1.55,pulse_cycles=2)],
            monitors=[FieldMonitor(id='plane',center=(1.5,0,0),size=(0,2.4,2.4),
                spectrum=SpectrumSettings(sampling='frequency',frequency_points=9,apodization='none'))])
        cases.append(BatchCase(f'sphere-{i:03d}',project,parameters={'radius_um':radius}))
    report=run_tensor_batch(cases,cohort_size=args.cohort,cuda_graph_steps=args.graph_steps,
        objective=lambda r:{'signed_spectral_flux':float(r.field_monitor('plane')['flux'][4])})
    report.raise_for_errors()
    print(json.dumps(dict(seconds=report.seconds,cases=[
        dict(id=item.id,parameters=item.parameters,metrics=item.metrics,
             monitor_kernel=item.summary['cuda_monitor_kernel']) for item in report.items]),indent=2))
    # This is reduced raw scattered-side spectral flux, not a calibrated
    # cross section. Physical power ratios need matched incident references.


if __name__=='__main__':
    main()
