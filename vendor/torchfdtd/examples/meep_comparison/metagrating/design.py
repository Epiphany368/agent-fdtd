"""Design step (TorchFDTD only, not part of the comparison): choose two silicon ridges per period.

    PYTHONPATH=<worktree> python examples/meep_comparison/metagrating/design.py

Coarse parametric sweep of two ridge widths and their gap on the comparison grid (edges midway
between Ez nodes), then a local refinement in two-node steps. The objective is the +1 transmitted
order efficiency at 1.55 um, evaluated with the same Fourier decomposition compare.py applies to
the comparison records. Writes design.json and the final ridges into geometry.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torchfdtd_metagrating as tm  # noqa: E402
from compare import order_powers  # noqa: E402
from torchfdtd import Simulation  # noqa: E402


def ridges_from_nodes(g, a0, wa, gap, wb):
    """Two ridges covering Ez nodes [a0, a0+wa-1] and [b0, b0+wb-1], b0 = a0+wa+gap; edges midway between nodes."""
    dx = g['mesh_um']
    x0 = -g['period_um'] / 2
    b0 = a0 + wa + gap
    out = []
    for start, width in ((a0, wa), (b0, wb)):
        out.append(dict(center_x_um=round(x0 + dx * (start + (width - 1) / 2), 6), width_um=round(dx * width, 6), first_node=start, nodes=width))
    return out


def line(monitor):
    names = list(monitor['components'])
    return monitor['points_um'][:, 0], monitor['fields'][..., names.index('Ez')], monitor['fields'][..., names.index('Hx')]


class Evaluator:
    def __init__(self, g, steps):
        self.g = g
        self.steps = steps
        self.spectrum = dict(sampling='wavelength', wavelength_start_um=g['design_wavelength_um'], wavelength_stop_um=g['design_wavelength_um'], points=1)
        ref = Simulation(tm.make_project(g, ridges=[], steps=steps, spectrum=self.spectrum)).run()
        x, ez, hx = line(ref.field_monitor('reflection'))
        f, b, _, _ = order_powers(x, ez, hx, [g['design_wavelength_um']], g['period_um'], g['substrate_index'], [0])
        self.incident = float(f[0, 0])
        self.reference_R0 = float(b[0, 0] / self.incident)

    def __call__(self, ridges):
        g = dict(self.g, ridges=ridges)
        result = Simulation(tm.make_project(g, steps=self.steps, spectrum=self.spectrum)).run()
        x, ez, hx = line(result.field_monitor('transmission'))
        tf, _, _, _ = order_powers(x, ez, hx, [g['design_wavelength_um']], g['period_um'], 1.0, [-1, 0, 1])
        x, ez, hx = line(result.field_monitor('reflection'))
        _, rb, _, _ = order_powers(x, ez, hx, [g['design_wavelength_um']], g['period_um'], g['substrate_index'], [-1, 0, 1])
        T = tf[0] / self.incident
        R = rb[0] / self.incident
        return dict(T_minus1=float(T[0]), T0=float(T[1]), T_plus1=float(T[2]), R_minus1=float(R[0]), R0=float(R[1]), R_plus1=float(R[2]),
                    total=float(T.sum() + R.sum()), loop_seconds=result.summary['seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--widths', type=int, nargs='+', default=[6, 12, 18, 24, 30, 36], help='ridge widths in Ez nodes (0.02 um each)')
    parser.add_argument('--gaps', type=int, nargs='+', default=[4, 8, 12, 16, 20, 28, 36], help='gaps in nodes')
    parser.add_argument('--refine-step', type=int, default=2)
    parser.add_argument('--refine-range', type=int, default=4)
    parser.add_argument('--out', default=str(HERE / 'design.json'))
    parser.add_argument('--write-geometry', action='store_true')
    args = parser.parse_args()
    g = tm.load_geometry()
    steps = args.steps or g['steps']
    started = time.perf_counter()
    evaluate = Evaluator(g, steps)
    print(json.dumps(dict(incident=evaluate.incident, reference_R0=evaluate.reference_R0)), flush=True)
    a0 = 2
    rows = []

    def run(a0, wa, gap, wb, stage):
        if wa + gap + wb > g['cells'][0] - 4:
            return None
        ridges = ridges_from_nodes(g, a0, wa, gap, wb)
        row = dict(stage=stage, a0=a0, wa=wa, gap=gap, wb=wb, ridges=ridges, **evaluate(ridges))
        # The mirror image (wb, gap, wa) swaps the +1 and -1 orders: score both, keep the best orientation.
        row['best_order'] = 'plus' if row['T_plus1'] >= row['T_minus1'] else 'minus'
        row['objective'] = max(row['T_plus1'], row['T_minus1'])
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('stage', 'wa', 'gap', 'wb', 'T_plus1', 'T_minus1', 'T0', 'total', 'loop_seconds')}), flush=True)
        return row

    for wa, wb, gap in itertools.product(args.widths, args.widths, args.gaps):
        if wa > wb:
            continue
        run(a0, wa, gap, wb, 'coarse')
    best = max(rows, key=lambda r: r['objective'])
    seen = {(r['wa'], r['gap'], r['wb']) for r in rows}
    offsets = range(-args.refine_range, args.refine_range + 1, args.refine_step)
    for dwa, dgap, dwb in itertools.product(offsets, offsets, offsets):
        wa, gap, wb = best['wa'] + dwa, best['gap'] + dgap, best['wb'] + dwb
        if min(wa, wb) < 2 or gap < 2 or (wa, gap, wb) in seen:
            continue
        seen.add((wa, gap, wb))
        run(a0, wa, gap, wb, 'refine')
    best = max(rows, key=lambda r: r['objective'])
    # Orient the winner so that the +1 order is the strong one (mirror x -> -x if needed).
    if best['best_order'] == 'minus':
        final = ridges_from_nodes(g, a0, best['wb'], best['gap'], best['wa'])
        mirrored = True
    else:
        final = best['ridges']
        mirrored = False
    check = evaluate(final)
    design = dict(method='coarse parametric sweep of two ridge widths and their gap in Ez-node units, then a local refinement; '
                         'objective = +1 transmitted order efficiency at the design wavelength from the same decomposition as compare.py; '
                         'mirror images score the -1 order and are kept mirrored',
                  steps=steps, design_wavelength_um=g['design_wavelength_um'], incident_power=evaluate.incident, reference_R0=evaluate.reference_R0,
                  widths_nodes=args.widths, gaps_nodes=args.gaps, refine_step=args.refine_step, refine_range=args.refine_range,
                  evaluations=len(rows), seconds=time.perf_counter() - started, best=best, mirrored=mirrored, final_ridges=final, final_check=check,
                  rows=rows)
    Path(args.out).write_bytes((json.dumps(design, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(dict(best={k: best[k] for k in ('wa', 'gap', 'wb', 'objective', 'best_order')}, final=final, check=check)))
    if args.write_geometry:
        raw = json.loads((HERE / 'geometry.json').read_text(encoding='utf-8'))
        raw['ridges'] = [dict(center_x_um=r['center_x_um'], width_um=r['width_um']) for r in final]
        (HERE / 'geometry.json').write_bytes((json.dumps(raw, indent=2) + '\n').encode('utf-8'))
        print('updated geometry.json')


if __name__ == '__main__':
    main()
