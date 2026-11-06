"""G7-04 reference (case G7-04r2): TORCWA 0.1.4.2 with 101 Fourier orders, checked against 241 orders.

    python examples/g7/solvers/rcwa_reference.py

(in a Python environment with TORCWA 0.1.4.2; TORCWA runs on CUDA and needs an otherwise idle GPU only
for a representative recorded run time). Reuses
solve() of examples/meep_comparison/metagrating/rcwa_metagrating.py on the unshifted base geometry:
s polarisation (E along the ridges), normal incidence from the substrate, the ridge layer sampled on
20,000 real-space cells whose boundaries hold the ridge edges. 101 orders are 50 harmonics and 241
orders are 120 harmonics (order = 2 * harmonics + 1). Every efficiency (T and R of orders -1, 0, +1)
at every one of the 41 wavelengths is compared between the two counts; the declared check
(docs/validation/cases/G7-04r2.json) is a largest difference of at most 1e-4.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

CHECK_LIMIT = 1e-4
REFERENCE_ORDERS = 101
CHECK_ORDERS = 241
ORDER_COUNTS = (REFERENCE_ORDERS, CHECK_ORDERS)


def load_rcwa():
    spec = importlib.util.spec_from_file_location('rcwa_metagrating', common.METAGRATING / 'rcwa_metagrating.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(common.RECORDS / 'rcwa_reference.json'))
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    rcwa = load_rcwa()
    g = common.load_geometry()
    spectrum = g['spectrum']
    wavelengths = np.linspace(spectrum['wavelength_start_um'], spectrum['wavelength_stop_um'], spectrum['points'])
    device = torch.device(args.device)
    started = time.perf_counter()
    bands = {}
    for count in ORDER_COUNTS:
        harmonics = (count - 1) // 2
        T = {str(m): [] for m in common.ORDERS}
        R = {str(m): [] for m in common.ORDERS}
        for w in wavelengths:
            t, r = rcwa.solve(g, float(w), harmonics, args.samples, device, torch.complex128)
            for m in common.ORDERS:
                T[str(m)].append(t[m])
                R[str(m)].append(r[m])
        total = [sum(T[str(m)][i] + R[str(m)][i] for m in common.ORDERS) for i in range(len(wavelengths))]
        Tm = np.stack([T[str(m)] for m in common.ORDERS], axis=1)
        bands[count] = dict(order_count=count, harmonics=harmonics, T=T, R=R, total=total,
                            max_abs_total_minus_one=float(np.max(abs(np.asarray(total) - 1))),
                            **{k: v for k, v in common.observables(Tm, wavelengths).items() if k != 'design_index'})
        print(count, 'orders: T+1(1.55) =', bands[count]['t1_design'], 'band mean', bands[count]['t1_band_mean'], flush=True)
    ref, chk = bands[REFERENCE_ORDERS], bands[CHECK_ORDERS]
    diff = {f'{kind}{int(m):+d}': abs(np.asarray(ref[kind][m]) - np.asarray(chk[kind][m])) for kind in ('T', 'R') for m in ref['T']}
    diffs = {k: float(v.max()) for k, v in diff.items()}
    largest = max(diffs.values())
    where = max(diff, key=lambda k: diffs[k])
    record = dict(
        schema='g7-04-rcwa-reference-v1', case='G7-04r2', solver='rcwa', geometry_sha256=g['_sha256'], date=time.strftime('%Y-%m-%d'),
        package=dict(torcwa=importlib.metadata.version('torcwa'), torch=torch.__version__, device=str(device),
                     device_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, dtype='complex128', python=sys.version.split()[0]),
        citation='C. Kim and B. Lee, TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design, '
                 'Comput. Phys. Commun. 282, 108552 (2023)',
        method='solve() of examples/meep_comparison/metagrating/rcwa_metagrating.py: input layer SiO2, one patterned layer, output layer air, '
               'normal incidence, s polarisation, permittivity on a real-space grid with the ridge edges on cell boundaries (Laurent rule), '
               'S-parameters with power normalisation; order count = 2 * harmonics + 1',
        samples=args.samples, wavelength_um=wavelengths.tolist(), bands={str(k): v for k, v in bands.items()},
        reference=dict(order_count=REFERENCE_ORDERS, t1_design=ref['t1_design'], t1_band_mean=ref['t1_band_mean']),
        check=dict(case='docs/validation/cases/G7-04r2.json', against_order_count=CHECK_ORDERS, max_abs_difference_per_efficiency=diffs,
                   max_abs_difference=largest, at_efficiency=where, at_wavelength_um=float(wavelengths[int(np.argmax(diff[where]))]),
                   limit_abs=CHECK_LIMIT, passed=bool(largest <= CHECK_LIMIT),
                   t1_design_difference=ref['t1_design'] - chk['t1_design'], t1_band_mean_difference=ref['t1_band_mean'] - chk['t1_band_mean']),
        seconds=time.perf_counter() - started)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out)
    print(json.dumps(dict(reference=record['reference'], check=record['check'])))


if __name__ == '__main__':
    main()
