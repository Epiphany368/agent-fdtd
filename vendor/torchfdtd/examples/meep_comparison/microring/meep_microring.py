"""Meep side of the microring comparison with TorchFDTD.

Reads geometry.json, runs the ring-plus-bus scene and the straight-bus normalisation
scene on the CPU with MPI, and writes docs/validation/meep_comparison/microring_meep.json.

    OMP_NUM_THREADS=1 mpirun -np <ranks> python examples/meep_comparison/microring/meep_microring.py --ranks <ranks> [--out PATH] [--timing]

Meep works in units of 1 um = 1; the script converts the shared time step and asserts that
the cell counts and dt agree with geometry.json to 1e-9 relative. Only the MPI master writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import meep as mp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C0 = 299792458.0
UM = 1e-6
TIME_UNIT = UM / C0  # one Meep time unit in seconds
DEFAULT_OUT = REPO / 'docs' / 'validation' / 'meep_comparison' / 'microring_meep.json'


def log(*args):
    if mp.am_master():
        print(*args, flush=True)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def waveform(source, times):
    """Shared pulse: amplitude*exp(-0.5*((t-4*sigma)/sigma)^2)*sin(2*pi*f0*t), sigma = pulse_cycles/f0 (times in seconds)."""
    f0 = C0 / (source['wavelength_um'] * 1e-6)
    sigma = source['pulse_cycles'] / f0
    u = (np.asarray(times, dtype=np.float64) - 4 * sigma) / sigma
    return source['amplitude'] * np.exp(-.5 * u * u) * np.sin(2 * math.pi * f0 * times)


def waveform_fingerprint(values):
    """Portable fingerprint of the sampled pulse: values below 1e-30 set to +0, then float32.

    The Meep process runs with flush-to-zero, so the Gaussian's tail (float64 or float32 denormals, signed
    zeros) differs from the TorchFDTD process at the byte level while every value above 1e-30 is identical.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.where(abs(values) < 1e-30, 0.0, values).astype(np.float32)


def wavelengths_um(spectrum):
    return np.linspace(spectrum['wavelength_start_um'], spectrum['wavelength_stop_um'], spectrum['points'])


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_snapshot():
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return dict(load_average=load)


def node_axes(g):
    dx = g['mesh_um']
    return [np.arange(n) * dx - s / 2 for n, s in zip(g['cells'][:2], g['size_um'][:2])]


def build(g, with_ring):
    dx = g['mesh_um']
    core, clad = mp.Medium(index=g['core_index']), mp.Medium(index=g['background_index'])
    bus, src, mon, ring = g['bus'], g['source'], g['monitors'], g['ring']
    geometry = [mp.Block(size=mp.Vector3(mp.inf, bus['width_um'], mp.inf), center=mp.Vector3(0, bus['center_y_um']), material=core)]
    if with_ring:
        geometry += [mp.Cylinder(radius=ring['outer_radius_um'], center=mp.Vector3(*ring['center_um']), material=core),
                     mp.Cylinder(radius=ring['inner_radius_um'], center=mp.Vector3(*ring['center_um']), material=clad)]
    duration = g['steps'] * g['dt_s'] / TIME_UNIT

    def src_func(t):
        return float(waveform(src, np.array([t * TIME_UNIT]))[0])
    eps_core, eps_clad, half_bus = g['core_index']**2, g['background_index']**2, bus['width_um'] / 2

    def amp_func(p):
        # Meep injects a current, E += -dt J / eps; TorchFDTD adds the pulse to E directly. Scaling J by the node
        # permittivity gives both solvers the same E increment on every row of the line (no node lies on the bus edges).
        # Meep passes the position relative to the source centre, which lies on the bus axis.
        return eps_core if abs(p.y) < half_bus else eps_clad
    sources = [mp.Source(mp.CustomSource(src_func=src_func, start_time=0, end_time=duration + 1, is_integrated=False), component=mp.Ez,
                         center=mp.Vector3(src['x_um'], src['center_y_um']), size=mp.Vector3(0, src['width_um']), amp_func=amp_func)]
    sim = mp.Simulation(cell_size=mp.Vector3(g['size_um'][0], g['size_um'][1], 0), resolution=1 / dx, geometry=geometry,
                        default_material=clad, sources=sources, boundary_layers=[mp.PML(g['pml_cells'] * dx)],
                        Courant=g['courant_number'], eps_averaging=False, dimensions=2)
    freqs = (1 / wavelengths_um(g['spectrum'])[::-1]).tolist()  # increasing frequency, 1/um
    flux = {label: sim.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(x, mon['center_y_um']), size=mp.Vector3(0, mon['width_um'])))
            for label, x in (('in', mon['in_x_um']), ('out', mon['out_x_um']))}
    return sim, flux, duration


def check_grid(g, sim):
    gv = sim.fields.gv
    assert [gv.nx(), gv.ny()] == g['cells'][:2], (gv.nx(), gv.ny())
    dt = sim.fields.dt * TIME_UNIT
    assert math.isclose(dt, g['dt_s'], rel_tol=1e-9), (dt, g['dt_s'])
    assert math.isclose(sim.fields.dt, sim.Courant / sim.resolution, rel_tol=1e-12)
    assert math.isclose(sim.resolution * g['mesh_um'], 1, rel_tol=1e-9)
    return dt


def staircase(g, sim):
    """Ez-node permittivity of the interior (nodes strictly inside the PML), the same nodes the TorchFDTD script hashes."""
    x, y = node_axes(g)
    grid = np.asarray(sim.get_epsilon_grid(x, y, np.array([0.0]))).real
    grid = grid[..., 0] if grid.ndim == 3 else grid
    n = g['pml_cells']
    interior = grid[n + 1:g['cells'][0] - n, n + 1:g['cells'][1] - n]
    # The geometry evaluation must equal the permittivity the solver holds at the Ez nodes: spot check 2000 interior nodes.
    rng = np.random.default_rng(0)
    ii = rng.integers(n + 1, g['cells'][0] - n, 2000)
    jj = rng.integers(n + 1, g['cells'][1] - n, 2000)
    held = np.array([1 / sim.fields.get_chi1inv(mp.Ez, mp.Z, mp.py_v3_to_vec(2, mp.Vector3(x[i], y[j])), 0.0).real for i, j in zip(ii, jj)])
    mismatch = int(np.count_nonzero(abs(held - grid[ii, jj]) > 1e-6))
    assert mismatch == 0, mismatch
    xx, yy = np.meshgrid(x, y, indexing='ij')
    radius = np.hypot(xx - g['ring']['center_um'][0], yy - g['ring']['center_um'][1])
    return dict(interior_cells=[[n + 1, g['cells'][0] - n], [n + 1, g['cells'][1] - n]],
                interior_ez_epsilon_sha256=sha256_array(np.round(interior, 4)),
                interior_core_nodes=int(np.count_nonzero(interior > g['background_index']**2 + 1e-6)),
                min_node_distance_to_circles_um=float(min(np.min(abs(radius - g['ring']['outer_radius_um'])),
                                                          np.min(abs(radius - g['ring']['inner_radius_um'])))),
                spot_check=dict(nodes=2000, chi1inv_mismatches=mismatch))


def source_rows(g, sim):
    """Rows of the Ez node column that the restricted line source touches, and their weights, from the field after one step."""
    x, y = node_axes(g)
    xc = g['source']['x_cell']
    lo, hi = g['source']['y_cells'][0] - 3, g['source']['y_cells'][1] + 3
    sim.run(until=.5 * sim.fields.dt)
    assert int(sim.fields.t) == 1
    column = np.array([sim.get_field_point(mp.Ez, mp.Vector3(x[xc], y[j])).real for j in range(lo, hi)])
    middle = column[len(column) // 2]
    weights = np.round(column / middle, 6)
    rows = np.flatnonzero(abs(weights) > 0) + lo
    return dict(rows=[int(rows[0]), int(rows[-1]) + 1], weights_at_ends=[float(weights[rows[0] - lo]), float(weights[rows[-1] - lo])],
                interior_weights_all_one=bool(np.all(weights[rows[0] - lo + 1:rows[-1] - lo] == 1)))


def solve(g, with_ring):
    mp.all_wait()
    start = time.perf_counter()
    sim, flux, duration = build(g, with_ring)
    sim.init_sim()
    dt = check_grid(g, sim)
    mp.all_wait()
    setup = time.perf_counter() - start
    step_start = time.perf_counter()
    sim.run(until=(g['steps'] - .5) * sim.fields.dt)  # Meep's loop runs while t*dt < until; this yields exactly steps steps
    mp.all_wait()
    stepping = time.perf_counter() - step_start
    steps_run = int(sim.fields.t)
    fluxes = {label: np.asarray(mp.get_fluxes(f)) for label, f in flux.items()}
    freqs = np.asarray(mp.get_flux_freqs(flux['out']))
    mp.all_wait()
    full = time.perf_counter() - start
    assert steps_run == g['steps'], (steps_run, g['steps'])
    return sim, dict(setup_seconds=setup, stepping_seconds=stepping, full_seconds=full, steps=steps_run), fluxes, freqs, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--geometry', default=str(HERE / 'geometry.json'))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--ranks', type=int, default=None, help='MPI ranks used in the mpirun command (recorded; checked against Meep)')
    parser.add_argument('--timing', action='store_true', help='one warm-up ring solve, then three timed ring solves')
    args = parser.parse_args()
    mp.verbosity(0)
    geometry_path = Path(args.geometry)
    g = json.loads(geometry_path.read_text(encoding='utf-8'))
    ranks = int(mp.count_processors())
    assert args.ranks is None or args.ranks == ranks, (args.ranks, ranks)

    sim, _, _ = build(g, True)
    sim.init_sim()
    stairs = staircase(g, sim)
    rows = source_rows(g, sim)
    assert rows['rows'] == g['source']['y_cells'] and rows['weights_at_ends'] == [.5, .5] and rows['interior_weights_all_one'], rows
    log(json.dumps(dict(staircase=stairs, source_rows=rows)))
    sim.reset_meep()

    before = load_snapshot()
    repeats = 3 if args.timing else 1
    samples = []
    if args.timing:
        sim, warm, _, _, _ = solve(g, True)
        log(json.dumps(dict(warm_up=warm)))
        sim.reset_meep()
    for repeat in range(repeats):
        sim, timing, ring_flux, freqs, dt = solve(g, True)
        samples.append(timing)
        log(json.dumps(dict(repeat=repeat, **timing)))
        sim.reset_meep()
    sim, straight_timing, straight_flux, freqs_straight, _ = solve(g, False)
    log(json.dumps(dict(straight=straight_timing)))
    after = load_snapshot()
    assert np.array_equal(freqs, freqs_straight)

    order = np.argsort(1 / freqs)  # ascending wavelength, as in geometry.json
    wavelength = 1 / freqs[order]
    assert np.allclose(wavelength, wavelengths_um(g['spectrum']), rtol=1e-12, atol=0)
    flux = {label: dict(ring=ring_flux[label][order], straight=straight_flux[label][order]) for label in ('in', 'out')}
    T = flux['out']['ring'] / flux['out']['straight']
    median = {k: float(np.median([s[k] for s in samples])) for k in ('setup_seconds', 'stepping_seconds', 'full_seconds')}
    record = dict(
        schema='torchfdtd-meep-comparison-v1', example='microring', solver='meep', date=time.strftime('%Y-%m-%d'),
        geometry_file=str(geometry_path.name), geometry_sha256=sha256_file(geometry_path), script_sha256=sha256_file(__file__),
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], cpu=cpu_model(), logical_cpus=os.cpu_count(),
                         wsl_distribution=os.environ.get('WSL_DISTRO_NAME'), meep=mp.__version__, meep_mpi=bool(mp.with_mpi()),
                         mpi_processes=ranks, omp_num_threads=os.environ.get('OMP_NUM_THREADS'), numpy=package_version('numpy'),
                         device='cpu', precision='float64 (Meep build)'),
        method='2D Ez (TM) cell, geometry rasterised by Meep at the Yee Ez nodes with eps_averaging=False, Meep PML of the declared '
               'thickness on all four sides, Ez CustomSource line with the shared sampled pulse (is_integrated=False) and amp_func = '
               'node permittivity so that the E increment per step matches the direct E addition of TorchFDTD on every row, DFT flux '
               'monitors over the whole run, T = out flux of the ring run / out flux of the straight-bus run.',
        grid=dict(cells=g['cells'], mesh_um=1 / sim.resolution, dt_s=dt, steps=g['steps'], courant_number=sim.Courant,
                  pml_cells=g['pml_cells'], run_time_ps=g['steps'] * dt * 1e12, precision='float64', mpi_processes=ranks),
        pml=dict(formulation='Meep PML (stretched-coordinate, quadratic profile by default, R_asymptotic=1e-15)', cells=g['pml_cells'],
                 thickness_um=g['pml_cells'] * g['mesh_um']),
        staircase=stairs, source=dict(**g['source'], shared_waveform_sha256=sha256_array(waveform_fingerprint(waveform(g['source'], np.arange(1, g['steps'] + 1) * g['dt_s']))),
                                      restricted_rows=rows),
        monitors=dict(in_x_cell=g['monitors']['in_x_cell'], out_x_cell=g['monitors']['out_x_cell'], y_cells=g['monitors']['y_cells'],
                      quadrature='Meep DFT flux region: Ez nodes of the plane with Meep restriction weights at the two ends, H interpolated to the plane'),
        wavelength_um=wavelength.tolist(), frequency_hz=(C0 / (wavelength * 1e-6)).tolist(),
        flux_out_ring=flux['out']['ring'].tolist(), flux_out_straight=flux['out']['straight'].tolist(),
        flux_in_ring=flux['in']['ring'].tolist(), flux_in_straight=flux['in']['straight'].tolist(),
        T=T.tolist(), T_in=(flux['in']['ring'] / flux['in']['straight']).tolist(),
        timing=dict(timing_mode='timing' if args.timing else 'development', repeats=repeats, samples=samples, **median,
                    straight_run=straight_timing, load_before=before, load_after=after,
                    host_load_note=(f'{"development" if not args.timing else "timed"} run with {ranks} MPI ranks on a shared workstation '
                                    f'(20 logical CPUs; other jobs may have been running); WSL load average before {before["load_average"]}, '
                                    f'after {after["load_average"]}.')))
    if mp.am_master():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        print('wrote', out, flush=True)
        print(json.dumps(dict(T_min=float(T.min()), T_max=float(T.max()), **median)), flush=True)


if __name__ == '__main__':
    main()
