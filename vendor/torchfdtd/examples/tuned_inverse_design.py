"""Explicit cohort measurement and CUDA population evaluation, without a UI.

Run: python -m examples.tuned_inverse_design
The preliminary timing trials are intentionally separate from optimization.
"""
import numpy as np

from torchfdtd import (Project, Region, Structure, Source, Monitor, RunControl,
                        parameter_sweep, tune_tensor_batch, optimize)


def intensity(result):
    # Local unnormalized field objective, not transmission efficiency.
    return float(np.sum(result.signals**2, dtype=np.float64))


def main():
    p=Project(name='CUDA population design',
        region=Region(dimension='3d',size=(4.8,4.8,4.8),mesh=.15,steps=800,
            pml_cells=4,backend='cuda',cuda_kernel='fused',material_sampling='yee',
            snapshot_interval=800,run_control=RunControl(auto_shutoff=False)),
        structures=[Structure(kind='sphere',radius=.5)],
        sources=[Source(center=(-1.2,0,0),pulse_cycles=2)],
        monitors=[Monitor(center=(1.2,0,0))])
    pilot=parameter_sweep(p,{'structures.0.radius':np.linspace(.25,.75,16).tolist()})
    tuning=tune_tensor_batch(pilot,candidates=(1,2,4,8,16),repeats=3)
    print('Cohort:',tuning.cohort_size,'Tuning cost (s):',tuning.seconds)
    result=optimize(p,{'structures.0.radius':(.25,.75)},intensity,
        execution='tensor',cohort_size=tuning.cohort_size,population=16,generations=3,
        seed=73,maximize=True,on_generation=print)
    print(result.as_dict())


if __name__=='__main__':main()
