"""Normal-incidence one-way plane, Fresnel R/T and a scattered-side monitor."""
import argparse
import json
from pathlib import Path
import numpy as np
from torchfdtd import Project, Simulation, FieldMonitor, normalize_flux
from examples.flux_slab import make_project as soft_slab


def make_project(backend='auto',direction='+'):
    p=soft_slab(backend);p.name='One-way dielectric slab';p.materials[1].index=1.5
    p.sources[0].injection='oneway';p.sources[0].direction=direction
    p.monitors.append(FieldMonitor(id='scattered',name='scattered',center=(-2.2,0,0),size=(0,.5,1),
                                  spectrum=p.monitors[0].spectrum.model_copy(deep=True)))
    if direction=='-':
        for o in p.sources+p.monitors+p.structures:o.center=(-o.center[0],o.center[1],o.center[2])
    return Project.model_validate(p.model_dump())


def evaluate(sample,reference):
    reflect=normalize_flux(sample.field_monitor('reflection'),reference.field_monitor('reflection'),subtract_incident=True)
    transmit=normalize_flux(sample.field_monitor('transmission'),reference.field_monitor('transmission'))
    d=1 if sample.project.sources[0].direction=='+' else -1
    wavelength=299792458/transmit['frequency_hz']*1e6
    R=-d*reflect['ratio'];T=d*transmit['ratio']
    R_scattered=-d*sample.field_monitor('scattered')['flux']/abs(reference.field_monitor('reflection')['flux'])
    floor=abs(reference.field_monitor('scattered')['flux'])/abs(reference.field_monitor('reflection')['flux'])
    exact=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    valid=reflect['valid'] & transmit['valid']
    return dict(direction=sample.project.sources[0].direction,wavelength_um=wavelength.tolist(),R=R.tolist(),T=T.tolist(),
        R_scattered=R_scattered.tolist(),analytic_T=exact.tolist(),valid=valid.tolist(),
        max_T_absolute_error=float(max(abs(T[valid]-exact[valid]))),
        max_R_absolute_error=float(max(abs(R[valid]-(1-exact[valid])))),
        max_energy_residual=float(max(abs(R[valid]+T[valid]-1))),
        max_scattered_R_difference=float(max(abs(R_scattered[valid]-R[valid]))),
        max_empty_scattered_flux_ratio=float(max(floor[valid])),
        project=sample.project.model_dump(),sample_summary=sample.summary,reference_summary=reference.summary)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--direction',choices=['+','-'],default='+');ap.add_argument('--output',default='results/oneway-slab')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    p=make_project(args.backend,args.direction);sample=Simulation(p).run();sample.save(out/'slab.npz')
    p.structures=[];reference=Simulation(p).run();reference.save(out/'reference.npz')
    report=evaluate(sample,reference)
    (out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k.startswith('max_')},indent=2))


if __name__=='__main__':main()
