"""Native CAD slab, opposing open ports and an interior material derivative.

This compact workflow example is not a mesh-converged optical design.
"""
import argparse
import json
from pathlib import Path
from torchfdtd import (Project, Region, Source, Material, Structure,
                       ModeNetworkConfig, run_mode_network)


def make_config():
    boundaries={axis+'_'+side:dict(kind='periodic' if axis=='y' else 'pml',layers=4)
                for axis in 'xyz' for side in ('min','max')}
    project=Project(name='Open slab mode network',
        region=Region(dimension='3d',size=(4.,1.,6.),mesh=.2,steps=420,
                      boundaries=boundaries,material_sampling='yee',precision='float32',backend='cpu'),
        materials=[Material(name='Guide',index=2.2,color='#69a1e8'),
                   Material(name='Design',index=1.7,color='#f0af63')],
        structures=[Structure(id='guide',name='Straight guide',size=(.8,1.,6.8),material='Guide'),
                    Structure(id='design',name='Interior design',size=(.4,1.,.4),material='Design')],
        sources=[Source(id='timing',name='Port pulse',kind='plane',normal='z',center=(0,0,-1.4),
                        size=(2.4,1.,0),pulse_cycles=2,wavelength=1.55)],monitors=[])
    return ModeNetworkConfig(project=project,normal='z',num_modes=1,
        ports=[dict(name='left',coordinate_um=-.6,source_coordinate_um=-1.4,direction=1,mode_indices=[0]),
               dict(name='right',coordinate_um=.6,source_coordinate_um=1.4,direction=-1,mode_indices=[0])],
        open_ports=dict(cladding_epsilon=1.),execution=dict(device='cpu',checkpoints=2),
        objective=dict(output_channel=('right',0),input_channel=('left',0),quantity='real'),
        differentiate_materials=['Design'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-config')
    parser.add_argument('--output',default='mode-network-result.json')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    config=make_config()
    config.execution.device=args.device
    destination=Path(args.export_config or args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if args.export_config:
        destination.write_text(config.model_dump_json(indent=2)+'\n',encoding='utf8')
    else:
        result=run_mode_network(config,on_progress=lambda p:print(p['phase'],flush=True))
        destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
