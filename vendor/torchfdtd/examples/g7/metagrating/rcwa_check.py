"""Independent check of the G7-01 binary designs with TORCWA 0.1.4.2 (rigorous coupled-wave analysis).

    C:/anaconda3/python.exe examples/g7/metagrating/rcwa_check.py --job job.json --out result.json

TORCWA: C. Kim and B. Lee, Comput. Phys. Commun. 282, 108552 (2023). It is not a TorchFDTD
dependency and lives in another interpreter, so workflow.py writes a job file and runs this
script as a subprocess. The FDTD axes (x periodic, y propagation, z invariant) map onto the RCWA
axes (x periodic, z propagation, y invariant): the substrate is the input layer, the pixel layer
the one patterned layer, air the output layer; TE (E along the ridges) is s and TM (H along the
ridges) is p polarisation. The fixed Bloch wavevector k_x becomes the incidence angle
asin(k_x lambda / (2 pi n_sub)) in the input layer, so the angle follows the wavelength. Each
0.02 um pixel is sampled by samples_per_pixel real-space cells, so every pixel edge lies on a
sample boundary. TORCWA forms the permittivity Toeplitz matrix from the FFT of the samples
(Laurent rule for both polarisations; it has no inverse rule), so TM converges as 1/N with the
harmonic count N. For every case the count rises through the job's sequence until the estimated
error at the convergence wavelengths is at most the job's error target, and the band is evaluated
at that count. The estimate assumes an error decaying as 1/N: the largest change of any order
efficiency from the previous count N_prev, times N_prev / (N - N_prev). Efficiencies use the s-p
basis with power normalisation; an order that does not propagate in its medium is reported as zero.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torcwa


def layer_permittivity(pixels, job, device):
    eps = 1+(job['ridge_index']**2-1)*np.repeat(np.asarray(pixels, dtype=float), job['samples_per_pixel'])
    return torch.as_tensor(eps, dtype=torch.complex128, device=device).reshape(-1, 1)


def propagating(job, wavelength_um, kx_per_um, order, index):
    return (index*2*math.pi/wavelength_um)**2-(kx_per_um+2*math.pi*order/job['period_um'])**2 > 0


def solve(job, pixels, wavelength_um, harmonics, polarization, kx_per_um, device):
    """T and R of every job order at one wavelength and harmonic count."""
    period, n_sub = job['period_um'], job['substrate_index']
    sim = torcwa.rcwa(freq=1/wavelength_um, order=[harmonics, 0], L=[period, period], dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=n_sub**2)
    sim.add_output_layer(eps=1.)
    sim.set_incident_angle(inc_ang=math.asin(kx_per_um*wavelength_um/(2*math.pi*n_sub)), azi_ang=0.)
    sim.add_layer(thickness=job['thickness_um'], eps=layer_permittivity(pixels, job, device))
    sim.solve_global_smatrix()
    basis = 'ss' if polarization == 'TE' else 'pp'
    T, R = {}, {}
    for m in job['orders']:
        for port, index, out in (('transmission', 1., T), ('reflection', n_sub, R)):
            if propagating(job, wavelength_um, kx_per_um, m, index):
                s = sim.S_parameters(orders=[m, 0], direction='forward', port=port, polarization=basis, ref_order=[0, 0], power_norm=True)
                out[m] = float(abs(s.reshape(-1)[0])**2)
            else:
                out[m] = 0.
    if not all(math.isfinite(v) for v in list(T.values())+list(R.values())):
        raise RuntimeError(f'TORCWA returned a nonfinite efficiency at {wavelength_um} um, {harmonics} harmonics, {polarization}.')
    return T, R


def run_case(job, case, device):
    pixels = job['designs'][case['design']]
    kx, pol = case['kx_per_um'], case['polarization']
    started = time.perf_counter()
    sweep = []
    chosen = None
    previous = None
    for n in job['harmonics']:
        rows = {str(w): solve(job, pixels, w, n, pol, kx, device) for w in case['convergence_wavelengths_um']}
        values = np.array([[v[m] for m in job['orders']] for T, R in rows.values() for v in (T, R)])
        entry = dict(harmonics=n, efficiencies={w: dict(T={str(m): T[m] for m in job['orders']}, R={str(m): R[m] for m in job['orders']})
                                                for w, (T, R) in rows.items()})
        if previous is not None:
            entry['max_abs_change_from_previous'] = float(np.max(abs(values-previous)))
            entry['first_order_error_estimate'] = entry['max_abs_change_from_previous']*sweep[-1]['harmonics']/(n-sweep[-1]['harmonics'])
            print(json.dumps(dict(design=case['design'], polarization=pol, incidence=case['incidence'], harmonics=n,
                                  first_order_error_estimate=entry['first_order_error_estimate'])), flush=True)
        sweep.append(entry)
        previous = values
        if len(sweep) > 1 and entry['first_order_error_estimate'] <= job['error_target']:
            chosen = n
            break
    converged = chosen is not None
    chosen = chosen or job['harmonics'][-1]
    error_estimate = sweep[-1].get('first_order_error_estimate')
    T = {str(m): [] for m in job['orders']}
    R = {str(m): [] for m in job['orders']}
    for w in case['wavelengths_um']:
        t, r = solve(job, pixels, w, chosen, pol, kx, device)
        for m in job['orders']:
            T[str(m)].append(t[m])
            R[str(m)].append(r[m])
    total = [sum(T[str(m)][i]+R[str(m)][i] for m in job['orders']) for i in range(len(case['wavelengths_um']))]
    return dict(case, harmonics=chosen, order_count=2*chosen+1, converged=converged, first_order_error_estimate=error_estimate,
                convergence=sweep, T=T, R=R, total=total,
                max_abs_total_minus_one=float(np.max(abs(np.asarray(total)-1))), seconds=time.perf_counter()-started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if job['device'] == 'auto' else job['device'])
    started = time.perf_counter()
    cases = []
    for case in job['cases']:
        cases.append(run_case(job, case, device))
        print(json.dumps({k: cases[-1][k] for k in ('design', 'polarization', 'incidence', 'harmonics', 'converged', 'max_abs_total_minus_one', 'seconds')}),
              flush=True)
    result = dict(schema='torchfdtd-g7-01-rcwa-v2', package=dict(torcwa=importlib.metadata.version('torcwa'), torch=torch.__version__,
                  python=platform.python_version(), interpreter=Path(sys.executable).name, device=str(device),
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(), dtype='complex128'),
                  citation='C. Kim and B. Lee, TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface '
                           'design, Comput. Phys. Commun. 282, 108552 (2023)',
                  method=__doc__.split('\n\n', 2)[2].strip(), samples_per_pixel=job['samples_per_pixel'], error_target=job['error_target'],
                  harmonic_sequence=job['harmonics'], cases=cases, seconds=time.perf_counter()-started)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(result, indent=1, allow_nan=False)+'\n').encode('utf-8'))


if __name__ == '__main__':
    main()
