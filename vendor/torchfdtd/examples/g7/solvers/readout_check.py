"""G7-04 development check, not part of the declared study: sensitivity of the order readout to the DFT-line position.

At 0.01 um the fixture's DFT lines (y = -0.81 and 1.01 um) lie on Ez rows, while at 0.02 um they lie midway
between rows. This check runs the staircase point at 0.01 um once with the fixture lines and once with both
lines moved by +h/2 onto pixel centres (one reference and one grating run each, no timing):

    OMP_NUM_THREADS=1 mpirun -np 4 python examples/g7/solvers/readout_check.py --solver meep       (Meep 1.34)
    python examples/g7/solvers/readout_check.py --solver torchfdtd                                 (CUDA GPU)

Writes docs/validation/g7/G7-04/readout_check_<solver>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

MESH_UM = 0.01
CASES = (('fixture lines (on Ez rows)', 0.0), ('lines moved by +h/2 (pixel centres)', MESH_UM / 2))


def lines_meep(g):
    import meep_sweep as ms
    gate = (None, None, 0)
    sim, dfts, _, _ = ms.timed_run(g, [], False, gate)
    reference = ms.lines(sim, dfts, g)
    sim.reset_meep()
    sim, dfts, _, _ = ms.timed_run(g, None, False, gate)
    monitors = ms.lines(sim, dfts, g)
    sim.reset_meep()
    return dict(monitors=monitors, reference_monitors=reference), ms.mp.am_master()


def lines_torchfdtd(g):
    import torchfdtd_sweep as ts
    reference, _ = ts.tm.timed_run(ts.make_project(g, ridges=[], precision='float32', series='staircase'))
    sample, _ = ts.tm.timed_run(ts.make_project(g, precision='float32', series='staircase'))
    return ts.full_record(sample, reference), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=('meep', 'torchfdtd'), required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    base = common.load_geometry()
    cases = {}
    master = True
    for label, dy in CASES:
        g = common.derive_geometry(base, MESH_UM, 'staircase')
        g['monitors'] = {k: round(v + dy, 12) for k, v in g['monitors'].items()}
        record, master = (lines_meep if args.solver == 'meep' else lines_torchfdtd)(g)
        common.check_lines(record, g)
        T, R = common.efficiencies_from_amplitudes(common.point_amplitudes(record), g['substrate_index'])
        total = T.sum(axis=1) + R.sum(axis=1)
        obs = common.observables(T, record['monitors']['transmission']['wavelength_um'])
        cases[label] = dict(monitors_y_um=g['monitors'], line_offset_um=dy, t1_design=obs['t1_design'], t1_band_mean=obs['t1_band_mean'],
                            order_sum_design=float(total[obs['design_index']]), order_sum_band=[float(total.min()), float(total.max())])
    if not master:
        return
    a, b = (cases[label] for label, _ in CASES)
    out = dict(schema='g7-04-readout-check-v1', case='G7-04', kind='development check, not a declared G7-04 quantity', solver=args.solver,
               series='staircase', precision='float64' if args.solver == 'meep' else 'float32', mesh_um=MESH_UM, date=time.strftime('%Y-%m-%d %H:%M:%S'),
               cases=cases, t1_design_change=b['t1_design'] - a['t1_design'], t1_band_mean_change=b['t1_band_mean'] - a['t1_band_mean'])
    path = Path(args.out) if args.out else common.RECORDS / f'readout_check_{args.solver}.json'
    path.write_bytes((json.dumps(out, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
