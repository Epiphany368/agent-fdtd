"""A matched-reference transmission study, with no external solver data."""
import argparse
from pathlib import Path

from torchfdtd import mesh_convergence, normalize_flux
from .flux_slab import make_project


def transmission(sample,reference):
    result=normalize_flux(sample.field_monitor('transmission'),reference.field_monitor('transmission'))
    if not result['valid'].all():raise ValueError('Reference flux is too weak for a convergence test.')
    return {'transmission':result['ratio']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',default='results/mesh-convergence')
    args=parser.parse_args()
    p=make_project(args.backend)
    p.materials[1].index=1.5
    p.region.size=(8,.8,1)
    p.structures[0].center=(.013,0,0)
    p.structures[0].size=(.2,.8,1)
    p.sources[0].size=(0,.8,0)
    for m in p.monitors:m.size=(0,.8,1)
    reference=p.model_copy(deep=True);reference.structures=[]
    report=mesh_convergence(p,[.05,.025,.0125],transmission,reference_project=reference,
                            atol=.006,rtol=0,output_dir=Path(args.output),
                            progress=lambda item:print(f"Mesh {item['mesh_um']} um: {item['status']}",flush=True))
    print(report.status)


if __name__=='__main__':main()
