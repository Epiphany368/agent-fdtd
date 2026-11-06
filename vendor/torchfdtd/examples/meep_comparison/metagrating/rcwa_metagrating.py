"""Third oracle for the metagrating: rigorous coupled-wave analysis with TORCWA 0.1.4.2.

    C:\\anaconda3\\python.exe examples/meep_comparison/metagrating/rcwa_metagrating.py \
        --out docs/validation/meep_comparison/metagrating_rcwa.json

TORCWA: C. Kim and B. Lee, "TORCWA: GPU-accelerated Fourier modal method and gradient-based
optimization for metasurface design", Comput. Phys. Commun. 282, 108552 (2023).

Same geometry.json. The FDTD axes (x periodic, y propagation, z invariant with E_z) map onto the
RCWA axes (x periodic, z propagation, y invariant with E_y): the incident wave is s-polarised
at normal incidence from the substrate (input layer), the ridges form one patterned layer, air is
the output layer. The ridge permittivity is sampled on a 20000-point real-space grid whose cell
boundaries coincide with the ridge edges; TORCWA forms the Toeplitz matrix from its FFT
(Laurent rule; for E parallel to the ridges no inverse rule is needed). The harmonic count is
increased until the six efficiencies at the design wavelength change by less than the recorded
tolerance, and the whole band is evaluated at that count.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
import torcwa

HERE = Path(__file__).resolve().parent


def load_geometry(path):
    raw = Path(path).read_bytes()
    geometry = json.loads(raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(raw).hexdigest()
    return geometry


def epsilon_grid(g, samples, device, dtype):
    """Permittivity of the ridge layer on `samples` cells across one period, x from -L/2 upward."""
    L = g['period_um']
    edges = -L / 2 + L * (np.arange(samples + 1) / samples)
    centres = (edges[:-1] + edges[1:]) / 2
    eps = np.ones(samples)
    for ridge in g['ridges']:
        lo, hi = ridge['center_x_um'] - ridge['width_um'] / 2, ridge['center_x_um'] + ridge['width_um'] / 2
        assert any(abs(edges - lo) < 1e-12) and any(abs(edges - hi) < 1e-12), 'ridge edges must fall on sample boundaries'
        eps[(centres > lo) & (centres < hi)] = g['ridge_index'] ** 2
    return torch.as_tensor(eps, dtype=dtype, device=device).reshape(samples, 1)


def solve(g, wavelength_um, harmonics, samples, device, dtype, polarization='ss'):
    orders = g['orders']
    sim = torcwa.rcwa(freq=1 / wavelength_um, order=[harmonics, 0], L=[g['period_um'], g['period_um']], dtype=dtype, device=device)
    sim.add_input_layer(eps=g['substrate_index'] ** 2)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=g['ridge_height_um'], eps=epsilon_grid(g, samples, device, dtype))
    sim.solve_global_smatrix()
    T = {}
    R = {}
    for m in orders:
        t = sim.S_parameters(orders=[m, 0], direction='forward', port='transmission', polarization=polarization, ref_order=[0, 0], power_norm=True)
        r = sim.S_parameters(orders=[m, 0], direction='forward', port='reflection', polarization=polarization, ref_order=[0, 0], power_norm=True)
        T[m] = float(abs(t.reshape(-1)[0]) ** 2)
        R[m] = float(abs(r.reshape(-1)[0]) ** 2)
    return T, R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--geometry', default=str(HERE / 'geometry.json'))
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--harmonics', type=int, nargs='+', default=[5, 10, 15, 20, 30, 40, 60, 80, 120])
    parser.add_argument('--tolerance', type=float, default=1e-5, help='converged when every efficiency changes by less than this from the previous count')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    g = load_geometry(args.geometry)
    device = torch.device(args.device)
    dtype = torch.complex128
    spectrum = g['spectrum']
    wavelengths = np.linspace(spectrum['wavelength_start_um'], spectrum['wavelength_stop_um'], spectrum['points'])
    # Rayleigh anomalies (an order exactly grazing) make the free-space reference singular; none may lie in the band.
    for m in range(1, 6):
        for index in (1.0, g['substrate_index']):
            anomaly = index * g['period_um'] / m
            assert not (spectrum['wavelength_start_um'] - 1e-3 <= anomaly <= spectrum['wavelength_stop_um'] + 1e-3), (m, index, anomaly)
    orders = g['orders']
    started = time.perf_counter()
    convergence = []
    previous = None
    chosen = None
    for n in args.harmonics:
        T, R = solve(g, g['design_wavelength_um'], n, args.samples, device, dtype)
        row = dict(harmonics=n, order_count=2 * n + 1, T={str(m): T[m] for m in orders}, R={str(m): R[m] for m in orders},
                   total=sum(T.values()) + sum(R.values()))
        if previous is not None:
            row['max_abs_change_from_previous'] = max(abs(row[k][str(m)] - previous[k][str(m)]) for k in ('T', 'R') for m in orders)
            if chosen is None and row['max_abs_change_from_previous'] < args.tolerance:
                chosen = n
        convergence.append(row)
        previous = row
        print(json.dumps(row), flush=True)
    if chosen is None:
        chosen = args.harmonics[-1]
    band_T = {str(m): [] for m in orders}
    band_R = {str(m): [] for m in orders}
    band_yy = []
    for w in wavelengths:
        T, R = solve(g, float(w), chosen, args.samples, device, dtype)
        for m in orders:
            band_T[str(m)].append(T[m])
            band_R[str(m)].append(R[m])
        Ty, Ry = solve(g, float(w), chosen, args.samples, device, dtype, polarization='yy')
        band_yy.append(max(abs(Ty[m] - T[m]) for m in orders) + max(abs(Ry[m] - R[m]) for m in orders))
    total = [sum(band_T[str(m)][i] + band_R[str(m)][i] for m in orders) for i in range(len(wavelengths))]
    record = dict(
        schema='torchfdtd-meep-comparison-v1', example='metagrating', solver='rcwa', geometry_sha256=g['_sha256'], date=time.strftime('%Y-%m-%d'),
        package=dict(torcwa=importlib.metadata.version('torcwa'), torch=torch.__version__, device=str(device), dtype='complex128'),
        citation='C. Kim and B. Lee, TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design, '
                 'Comput. Phys. Commun. 282, 108552 (2023)',
        method='Input layer SiO2 (incident side), one patterned layer of the ridge height, output layer air; normal incidence, s polarisation '
               '(E along the ridges); permittivity sampled on a real-space grid with the ridge edges on cell boundaries, Toeplitz matrix from '
               'its FFT (Laurent rule); S-parameters in the s-p basis with power normalisation; the yy basis is evaluated as a check.',
        samples=args.samples, harmonics=chosen, order_count=2 * chosen + 1, convergence_tolerance=args.tolerance, convergence=convergence,
        wavelength_um=wavelengths.tolist(), T=band_T, R=band_R, total=total, max_abs_total_minus_one=float(np.max(abs(np.asarray(total) - 1))),
        max_abs_ss_minus_yy=float(max(band_yy)), seconds=time.perf_counter() - started)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    i = int(np.argmin(abs(wavelengths - g['design_wavelength_um'])))
    print('wrote', out)
    print(json.dumps(dict(harmonics=chosen, T={m: band_T[m][i] for m in band_T}, R={m: band_R[m][i] for m in band_R}, total=total[i],
                          max_abs_ss_minus_yy=record['max_abs_ss_minus_yy'])))


if __name__ == '__main__':
    main()
