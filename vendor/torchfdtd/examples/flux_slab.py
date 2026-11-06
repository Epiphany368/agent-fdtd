"""Python-only slab reflection/transmission, with an identical air reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from torchfdtd import (Project,Region,Structure,Source,FieldMonitor,SpectrumSettings,
                         Boundaries,BoundaryFace,Simulation,normalize_flux)


def make_project(backend='auto'):
    return Project(name='Analytic dielectric slab',region=Region(
        size=(8,.5,1),mesh=.025,steps=1600,pml_cells=16,backend=backend,precision='float64',
        material_sampling='yee',snapshot_interval=1600,
        boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'),y_max=BoundaryFace(kind='periodic'))),
        structures=[Structure(center=(.0125,0,0),size=(.2,.5,1),material='SiO2 (constant n)')],
        sources=[Source(id='source',kind='plane',center=(-1.5,0,0),size=(0,.5,0),pulse_cycles=1)],
        monitors=[FieldMonitor(id=label,name=label,center=(x,0,0),size=(0,.5,1),
                  spectrum=SpectrumSettings(sampling='frequency',frequency_points=31,apodization='none'))
                  for label,x in [('reflection',-.8),('transmission',.8)]])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--output',default='results/flux');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    p=make_project(args.backend);p.materials[1].index=1.5
    sample=Simulation(p).run();sample.save(out/'slab.npz')
    air=Project.model_validate(p.model_dump());air.structures=[]
    reference=Simulation(air).run();reference.save(out/'air.npz')
    reflect=normalize_flux(sample.field_monitor('reflection'),reference.field_monitor('reflection'),subtract_incident=True)
    transmit=normalize_flux(sample.field_monitor('transmission'),reference.field_monitor('transmission'))
    valid=reflect['valid'] & transmit['valid'];wavelength=299792458/transmit['frequency_hz']*1e6
    R=-reflect['ratio'];T=transmit['ratio']
    exactT=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    report=dict(wavelength_um=wavelength.tolist(),R=R.tolist(),T=T.tolist(),analytic_T=exactT.tolist(),
                valid=valid.tolist(),max_T_absolute_error=float(max(abs(T[valid]-exactT[valid]))),
                max_R_absolute_error=float(max(abs(R[valid]-(1-exactT[valid])))),
                max_energy_residual=float(max(abs(R[valid]+T[valid]-1))),
                sample_summary=sample.summary,reference_summary=reference.summary)
    (out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k.startswith('max_')},indent=2))


if __name__=='__main__':main()
