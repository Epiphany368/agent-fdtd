"""Meep side of the metagrating comparison (see BRIEF.md and README.md).

Run inside the micromamba "meep" environment from the worktree root:
    OMP_NUM_THREADS=1 mpirun -np 4 python examples/meep_comparison/metagrating/meep_metagrating.py \
        --ranks 4 --out docs/validation/meep_comparison/metagrating_meep.json [--timing]

Same geometry.json, same mesh, time step, step count, pulse, PML thickness and DFT lines as the
TorchFDTD script. Only the MPI master writes the record. Meep is a double-precision CPU solver
with its own PML formulation; the record states every difference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import meep as mp

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE / 'geometry.json'
C0 = 299792458.0
UM = 1e-6
TIME_UNIT = UM / C0  # one Meep time unit in seconds (a = 1 um)


def log(*args):
    if mp.am_master():
        print(*args, flush=True)


def load_geometry(path=GEOMETRY):
    raw = Path(path).read_bytes()
    geometry = json.loads(raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(raw).hexdigest()
    return geometry


def waveform(wave, times_s):
    """The shared pulse (identical function in torchfdtd_metagrating.py)."""
    times_s = np.asarray(times_s, dtype=np.float64)
    frequency = C0 / (wave['wavelength_um'] * 1e-6)
    sigma = wave['pulse_cycles'] / frequency
    u = (times_s - 4 * sigma) / sigma
    return wave['amplitude'] * np.exp(-.5 * u * u) * np.sin(2 * math.pi * frequency * times_s)


def custom_source(wave):
    def src_func(t):
        return float(waveform(wave, np.array([t * TIME_UNIT]))[0])
    return src_func


def frequencies_per_um(spectrum):
    a, b, n = spectrum['wavelength_start_um'], spectrum['wavelength_stop_um'], spectrum['points']
    assert spectrum['sampling'] == 'wavelength'
    return (1 / np.linspace(a, b, n)).tolist()


def make_simulation(g, ridges=None):
    ridges = g['ridges'] if ridges is None else ridges
    lx, ly = g['cell_size_um']
    resolution = 1 / g['mesh_um']
    top = g['substrate_top_y_um']
    geometry = [mp.Block(size=mp.Vector3(mp.inf, top + ly / 2, mp.inf), center=mp.Vector3(0, (top - ly / 2) / 2, 0),
                         material=mp.Medium(index=g['substrate_index']))]
    for ridge in ridges:
        geometry.append(mp.Block(size=mp.Vector3(ridge['width_um'], g['ridge_height_um'], mp.inf),
                                 center=mp.Vector3(ridge['center_x_um'], top + g['ridge_height_um'] / 2, 0),
                                 material=mp.Medium(index=g['ridge_index'])))
    wave = g['source']['waveform']
    dt_meep = g['courant_number'] / resolution
    duration = g['steps'] * dt_meep
    src = mp.Source(mp.CustomSource(src_func=custom_source(wave), start_time=0, end_time=duration + 1, is_integrated=False),
                    component=mp.Ez, center=mp.Vector3(0, g['source']['y_um'], 0), size=mp.Vector3(lx, 0, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(lx, ly, 0), resolution=resolution,
                        boundary_layers=[mp.PML(g['pml_cells'] * g['mesh_um'], direction=mp.Y)], geometry=geometry, sources=[src],
                        k_point=mp.Vector3(0, 0, 0), Courant=g['courant_number'], eps_averaging=False, dimensions=2)
    freqs = frequencies_per_um(g['spectrum'])
    dfts = {name: sim.add_dft_fields([mp.Ez, mp.Hx], freqs, center=mp.Vector3(0, g['monitors'][name + '_y_um'], 0), size=mp.Vector3(lx, 0, 0))
            for name in ('reflection', 'transmission')}
    return sim, dfts, dt_meep


def dft_line(sim, dft, g):
    """Ez and Hx on the 100 pixel centres of the line (Meep returns two extra periodic-image points)."""
    x, y, z, w = sim.get_array_metadata(dft_cell=dft)
    x = np.asarray(x)
    freqs = np.asarray(dft.freq)
    ez = np.stack([np.asarray(sim.get_dft_array(dft, mp.Ez, i)) for i in range(len(freqs))])
    hx = np.stack([np.asarray(sim.get_dft_array(dft, mp.Hx, i)) for i in range(len(freqs))])
    n = g['cells'][0]
    assert ez.shape == (len(freqs), n + 2), ez.shape
    # The two outer points are the periodic images of the two inner ones.
    assert np.allclose(ez[:, 0], ez[:, n], rtol=1e-9, atol=0) and np.allclose(ez[:, n + 1], ez[:, 1], rtol=1e-9, atol=0)
    assert np.allclose(hx[:, 0], hx[:, n], rtol=1e-9, atol=0) and np.allclose(hx[:, n + 1], hx[:, 1], rtol=1e-9, atol=0)
    x, ez, hx = x[1:n + 1], ez[:, 1:n + 1], hx[:, 1:n + 1]
    expected = -g['period_um'] / 2 + g['mesh_um'] * (np.arange(n) + .5)
    assert np.allclose(x, expected, rtol=0, atol=1e-9), (x[:3], expected[:3])
    wavelength = 1 / freqs
    flux = .5 * np.real(ez * hx.conj()).mean(axis=1) * g['period_um'] * UM  # reduced units, per invariant length
    return dict(y_um=float(np.asarray(y).ravel()[0]), x_um=x.tolist(), weights_m=[g['mesh_um'] * UM] * n,
                frequency_hz=(freqs / TIME_UNIT).tolist(), wavelength_um=wavelength.tolist(),
                ez_real=ez.real.tolist(), ez_imag=ez.imag.tolist(), hx_real=hx.real.tolist(), hx_imag=hx.imag.tolist(),
                flux=flux.tolist(), flux_units='Meep units E*H * (a/c) * m per invariant length', field_units='Meep field * (a/c)')


def staircase(sim, g):
    """Silicon columns of Ez nodes on the row through the ridge middle, read from the permittivity Meep stores for Ez.

    fields.get_chi1inv returns the stored 1/epsilon of the Ez component; at a Yee node it is the
    value the update uses (between nodes it interpolates). Meep places its Ez nodes at
    -L/2 + i*dx on an axis with an even cell count, the same nodes TorchFDTD uses.
    """
    dx = g['mesh_um']
    lx, ly = g['cell_size_um']
    xs = -lx / 2 + dx * np.arange(g['cells'][0])
    ys = -ly / 2 + dx * np.arange(g['cells'][1])

    def eps(x, y):
        return 1 / sim.fields.get_chi1inv(mp.Ez, mp.Z, mp.py_v3_to_vec(sim.dimensions, mp.Vector3(x, y, 0), sim.is_cylindrical)).real
    row = int(np.argmin(abs(ys - (g['substrate_top_y_um'] + g['ridge_height_um'] / 2))))
    threshold = (g['ridge_index'] ** 2 + g['substrate_index'] ** 2) / 2
    eps_row = np.array([eps(x, ys[row]) for x in xs])
    silicon = np.flatnonzero(eps_row > threshold)
    eps_col0 = np.array([eps(xs[0], y) for y in ys])
    substrate_rows = np.flatnonzero(abs(eps_col0 - g['substrate_index'] ** 2) < 1e-6)
    if len(silicon):
        eps_col = np.array([eps(xs[silicon[0]], y) for y in ys])
        ridge_rows = np.flatnonzero(eps_col > threshold)
    else:
        ridge_rows = np.array([], dtype=int)
    return dict(row_y_um=float(ys[row]), silicon_columns=silicon.tolist(), silicon_rows=ridge_rows.tolist(),
                substrate_rows=[int(substrate_rows[0]), int(substrate_rows[-1])] if len(substrate_rows) else [],
                ez_x_nodes_um=xs.tolist(), ez_y_nodes_um=ys.tolist())


def host_cpu_load_percent():
    try:
        out = subprocess.run(['powershell.exe', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).LoadPercentage'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def nvidia_smi():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(('name', 'utilization_percent', 'memory_used_mb'), [v.strip() for v in out.split(',')]))


def versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def timed_run(g, ridges):
    mp.all_wait()
    started = time.perf_counter()
    sim, dfts, dt_meep = make_simulation(g, ridges)
    sim.init_sim()
    mp.all_wait()
    setup = time.perf_counter() - started
    stepping_started = time.perf_counter()
    # (steps - 0.5) * dt: run() steps while t < until, so exactly `steps` steps are taken regardless of round-off.
    sim.run(until=(g['steps'] - .5) * dt_meep)
    mp.all_wait()
    stepping = time.perf_counter() - stepping_started
    full = time.perf_counter() - started
    steps_run = int(sim.fields.t)
    assert steps_run == g['steps'], (steps_run, g['steps'])
    return sim, dfts, dt_meep, dict(setup_seconds=setup, stepping_seconds=stepping, full_seconds=full, steps_run=steps_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ranks', type=int, required=True, help='MPI ranks this script was launched with (asserted)')
    parser.add_argument('--timing', action='store_true', help='one warm-up, then three timed grating solves')
    parser.add_argument('--geometry', default=str(GEOMETRY))
    args = parser.parse_args()
    mp.verbosity(0)
    assert int(mp.count_processors()) == args.ranks, (mp.count_processors(), args.ranks)
    g = load_geometry(args.geometry)
    assert g['ridges'], 'geometry.json carries no ridges; run design.py first'
    cpu_before = host_cpu_load_percent() if mp.am_master() else None
    gpu_before = nvidia_smi() if mp.am_master() else None

    sim, dfts, dt_meep, ref_timing = timed_run(g, [])
    dt = dt_meep * TIME_UNIT
    assert math.isclose(dt, g['courant_number'] * g['mesh_um'] * 1e-6 / C0, rel_tol=1e-9)
    reference_monitors = {name: dft_line(sim, dfts[name], g) for name in ('reflection', 'transmission')}
    reference_stairs = staircase(sim, g)
    assert not reference_stairs['silicon_columns']
    sim.reset_meep()

    repeats = 3 if args.timing else 1
    if args.timing:
        sim, dfts, _, _ = timed_run(g, None)  # warm-up
        sim.reset_meep()
    samples = []
    for _ in range(repeats):
        sim, dfts, _, timing = timed_run(g, None)
        samples.append(timing)
        if len(samples) < repeats:
            sim.reset_meep()
    monitors = {name: dft_line(sim, dfts[name], g) for name in ('reflection', 'transmission')}
    stairs = staircase(sim, g)
    eps_shape = list(np.shape(sim.get_epsilon()))
    for name in ('reflection', 'transmission'):
        assert math.isclose(monitors[name]['y_um'], g['monitors'][name + '_y_um'], abs_tol=1e-9), (monitors[name]['y_um'], name)
    if not mp.am_master():
        return
    record = dict(
        schema='torchfdtd-meep-comparison-v1', example='metagrating', solver='meep', geometry_sha256=g['_sha256'],
        date=time.strftime('%Y-%m-%d'),
        method='2D TMz cell (Ez, Hx, Hy), x periodic through k_point=0 (real fields), Meep PML of the same thickness in y, '
               'Ez CustomSource current sheet across the full period with the shared pulse (is_integrated=False), eps_averaging=False, '
               'add_dft_fields lines of Ez and Hx read on the centred grid (pixel centres; the two periodic-image points are dropped), '
               'bare-substrate reference run with the same settings. Fields are Meep units; compare.py forms ratios.',
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], wsl_distribution=os.environ.get('WSL_DISTRO_NAME'),
                         meep_version=mp.__version__, packages=versions(('meep', 'numpy', 'mpi4py')), meep_mpi=bool(mp.with_mpi()),
                         mpi_processes=int(mp.count_processors()), omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                         precision='float64 (Meep build)', gpu=gpu_before),
        grid=dict(cells=list(g['cells']), epsilon_array_shape=eps_shape, mesh_um=g['mesh_um'], resolution_per_um=1 / g['mesh_um'], dt_s=dt,
                  steps=g['steps'], steps_run=samples[-1]['steps_run'], courant_number=g['courant_number'], pml_cells=g['pml_cells'], pml_axis='y',
                  periodic_axis='x', precision='float64', mpi_processes=int(mp.count_processors()),
                  boundary='Meep PML (stretched-coordinate, quadratic profile by default, R_asymptotic=1e-15), same thickness in cells as the CPML',
                  source_row=int(round((g['source']['y_um'] + g['cell_size_um'][1] / 2) / g['mesh_um'])), source_columns=[0, g['cells'][0]]),
        staircase=stairs,
        phasor_convention='Meep DFT = sum(f(t) exp(+i omega t)) dt; a +y travelling wave is exp(+i k_y y); Meep applies the half-step H time',
        phasor_time_sign=1,
        monitors=monitors, reference_monitors=reference_monitors,
        timing=dict(timing_mode='timing' if args.timing else 'development', repeats=repeats, samples=samples,
                    setup_seconds=float(np.median([s['setup_seconds'] for s in samples])),
                    stepping_seconds=float(np.median([s['stepping_seconds'] for s in samples])),
                    full_seconds=float(np.median([s['full_seconds'] for s in samples])),
                    reference=ref_timing, ranks=int(mp.count_processors()), host_cpu_percent_before=cpu_before,
                    host_cpu_percent_after=host_cpu_load_percent(),
                    host_load_note=(f'development run with {mp.count_processors()} ranks on a shared host: host CPU {cpu_before}% before the run, '
                                    f'GPU utilisation {(gpu_before or {}).get("utilization_percent")}% (other jobs); not a timing measurement')
                    if not args.timing else f'timing run with {mp.count_processors()} ranks: host CPU {cpu_before}% before the run'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out)
    print(json.dumps(dict(timing=samples, silicon_columns=stairs['silicon_columns'], eps_shape=eps_shape)))


if __name__ == '__main__':
    main()
