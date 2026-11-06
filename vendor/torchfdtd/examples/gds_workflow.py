"""Convert supplied GDS and explicit process metadata into a native layout.

python examples/gds_workflow.py chip.gds TOP stack.json --output scene.json

stack.json: {"layers": [{"layer": 1, "datatype": 0, "z_min": -0.11,
"z_max": 0.11, "material": "core", "etch_by": [[2, 0]]}], "materials":
[{"name": "core", "index": 2.0}], "port_layers": []}

"etch_by" is optional: polygons on the listed pairs are subtracted from the
layer before extrusion. Bridged hole contours import as explicit holes.

The index above is illustrative. Supply the actual device materials. No
excitation or detector is invented from the layout or its text markers.
"""
import argparse
import json
from pathlib import Path

from torchfdtd import Project,Region,Material
from torchfdtd.gds import GDSLayer,GDSPortLayer,import_gds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('cell');parser.add_argument('stack')
    parser.add_argument('--output',default='scene.json')
    parser.add_argument('--mesh',type=float,default=.05)
    parser.add_argument('--padding',type=float,default=1.)
    args=parser.parse_args()
    if args.padding<=0:parser.error('padding must be positive')
    settings=json.loads(Path(args.stack).read_text(encoding='utf-8'))
    layout=import_gds(args.input,cell=args.cell,layers=[GDSLayer(**x) for x in settings['layers']],
                      port_layers=[GDSPortLayer(**x) for x in settings.get('port_layers',[])])
    lower,upper=layout.report['bounds_um']
    size=tuple(2*(max(abs(a),abs(b))+args.padding) for a,b in zip(lower,upper))
    project=Project(region=Region(dimension='3d',size=size,mesh=args.mesh,pml_cells=5,material_sampling='yee'),
                    materials=[Material.model_validate(x) for x in settings['materials']],sources=[],monitors=[])
    project=layout.add_to(project)
    output=Path(args.output);project.save(output)
    output.with_suffix('.gds-report.json').write_text(json.dumps(layout.report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved {len(layout.structures)} native polygons and {len(layout.ports)} port metadata records to {output}.')
    print('Add physically appropriate sources and detectors before running the scene.')


if __name__=='__main__':main()
