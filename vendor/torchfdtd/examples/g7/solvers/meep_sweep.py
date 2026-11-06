"""G7-04 Meep sweep: the metagrating at resolutions 25, 50, 100 and 200 per um, one series per call.

Inside the WSL distribution torchfdtd-bench, from the worktree root, in the micromamba "meep" environment:
    OMP_NUM_THREADS=1 mpirun -np 4 python examples/g7/solvers/meep_sweep.py --ranks 4 --series staircase

Each resolution uses the derived geometry of common.derive_geometry (the same one TorchFDTD receives).
The scene is the one of examples/meep_comparison/metagrating/meep_metagrating.py (shared pulse as a
CustomSource, Meep PML of the same 0.4 um thickness, add_dft_fields lines on the centred grid) with
eps_averaging=False in the staircase series and Meep's default subpixel averaging in the smoothed
series. Per point: one bare-substrate reference run, one warm-up and three timed grating runs.
Before every run the ranks wait while the optional --pause-file exists and, with --max-host-cpu-percent,
until the Windows host load is below that value; the load and the wait are recorded.
Only the MPI master writes the record.
"""
from __future__ import annotations

import argparse
import importlib.util
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
sys.path.insert(0, str(HERE))
import common  # noqa: E402

spec = importlib.util.spec_from_file_location('meep_metagrating', common.METAGRATING / 'meep_metagrating.py')
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

LOAD_SAMPLE_SLEEP = 3.0  # seconds the non-master ranks sleep while the master samples the host load


def make_simulation(g, ridges=None, *, eps_averaging):
    """meep_metagrating.make_simulation with the point's resolution, absorber and material sampling."""
    ridges = g['ridges'] if ridges is None else ridges
    lx, ly = g['cell_size_um']
    resolution = g['g7_04']['resolution_per_um']
    top = g['substrate_top_y_um']
    geometry = [mp.Block(size=mp.Vector3(mp.inf, top + ly / 2, mp.inf), center=mp.Vector3(0, (top - ly / 2) / 2, 0),
                         material=mp.Medium(index=g['substrate_index']))]
    for ridge in ridges:
        geometry.append(mp.Block(size=mp.Vector3(ridge['width_um'], g['ridge_height_um'], mp.inf),
                                 center=mp.Vector3(ridge['center_x_um'], top + g['ridge_height_um'] / 2, 0),
                                 material=mp.Medium(index=g['ridge_index'])))
    dt_meep = g['courant_number'] / resolution
    duration = g['steps'] * dt_meep
    src = mp.Source(mp.CustomSource(src_func=mm.custom_source(g['source']['waveform']), start_time=0, end_time=duration + 1, is_integrated=False),
                    component=mp.Ez, center=mp.Vector3(0, g['source']['y_um'], 0), size=mp.Vector3(lx, 0, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(lx, ly, 0), resolution=resolution,
                        boundary_layers=[mp.PML(g['pml_cells'] * g['mesh_um'], direction=mp.Y)], geometry=geometry, sources=[src],
                        k_point=mp.Vector3(0, 0, 0), Courant=g['courant_number'], eps_averaging=eps_averaging, dimensions=2)
    freqs = mm.frequencies_per_um(g['spectrum'])
    dfts = {name: sim.add_dft_fields([mp.Ez, mp.Hx], freqs, center=mp.Vector3(0, g['monitors'][name + '_y_um'], 0), size=mp.Vector3(lx, 0, 0))
            for name in ('reflection', 'transmission')}
    return sim, dfts, dt_meep


def wait_for_host(pause_file, max_cpu_percent, max_wait_seconds):
    """Wait while the optional pause file exists and, with max_cpu_percent, until the Windows host load is below it.

    Only the master samples the load (powershell.exe, about a second); the other ranks sleep meanwhile
    instead of spinning in an MPI barrier, so the sample does not count this job's own waiting ranks.
    The decision is broadcast from the master. After max_wait_seconds the run starts regardless and the
    record says so.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    waited, started = 0.0, time.perf_counter()
    while True:
        while pause_file and os.path.exists(pause_file):
            time.sleep(30)
        if mp.am_master():
            cpu = mm.host_cpu_load_percent()
        else:
            cpu = None
            time.sleep(LOAD_SAMPLE_SLEEP)
        cpu = comm.bcast(cpu, root=0)
        waited = time.perf_counter() - started
        busy = max_cpu_percent is not None and cpu is not None and cpu > max_cpu_percent
        if not busy or waited >= max_wait_seconds:
            comm.Barrier()
            return dict(waited_seconds=waited, host_cpu_percent_before=cpu, gate_timed_out=bool(busy))
        time.sleep(30)


def timed_run(g, ridges, eps_averaging, gate):
    host = wait_for_host(*gate)
    started = time.perf_counter()
    sim, dfts, dt_meep = make_simulation(g, ridges, eps_averaging=eps_averaging)
    sim.init_sim()
    mp.all_wait()
    setup = time.perf_counter() - started
    stepping_started = time.perf_counter()
    sim.run(until=(g['steps'] - .5) * dt_meep)  # exactly `steps` steps, as in meep_metagrating.timed_run
    mp.all_wait()
    stepping = time.perf_counter() - stepping_started
    full = time.perf_counter() - started
    steps_run = int(sim.fields.t)
    assert steps_run == g['steps'], (steps_run, g['steps'])
    return sim, dfts, dt_meep, dict(setup_seconds=setup, stepping_seconds=stepping, full_seconds=full, steps_run=steps_run, **host)


def staircase(sim, g):
    """Silicon Ez nodes on the row through the ridge middle, read from the 1/epsilon Meep stores for Ez at its own nodes."""
    xs = common.ez_nodes('meep', g['cell_size_um'][0], g['mesh_um'], g['cells'][0])
    ys = common.ez_nodes('meep', g['cell_size_um'][1], g['mesh_um'], g['cells'][1])

    def eps(x, y):
        return 1 / sim.fields.get_chi1inv(mp.Ez, mp.Z, mp.py_v3_to_vec(sim.dimensions, mp.Vector3(x, y, 0), sim.is_cylindrical)).real
    threshold = (g['ridge_index'] ** 2 + g['substrate_index'] ** 2) / 2
    row = int(np.argmin(abs(ys - (g['substrate_top_y_um'] + g['ridge_height_um'] / 2))))
    silicon = np.flatnonzero(np.array([eps(x, ys[row]) for x in xs]) > threshold)
    substrate_rows = np.flatnonzero(abs(np.array([eps(xs[0], y) for y in ys]) - g['substrate_index'] ** 2) < 1e-6)
    ridge_rows = np.flatnonzero(np.array([eps(xs[silicon[0]], y) for y in ys]) > threshold) if len(silicon) else np.array([], dtype=int)
    return dict(row_y_um=float(ys[row]), silicon_columns=silicon.tolist(), silicon_rows=ridge_rows.tolist(),
                substrate_rows=[int(substrate_rows[0]), int(substrate_rows[-1])] if len(substrate_rows) else [],
                ez_node_rule='x_i = h (i - N//2), y_j = h (j - M//2): integer multiples of h from the cell centre',
                silicon_x_um=[round(float(xs[i]), 12) for i in silicon], silicon_y_um=[round(float(ys[j]), 12) for j in ridge_rows])


def lines(sim, dfts, g):
    return {name: mm.dft_line(sim, dfts[name], g) for name in ('reflection', 'transmission')}


def run_point(base, resolution, series, gate, repeats, compare):
    mesh = {r: h for r, h in zip(common.MEEP_RESOLUTIONS, common.TORCHFDTD_MESHES_UM)}[resolution]
    g = common.derive_geometry(base, mesh, series)
    assert g['g7_04']['resolution_per_um'] == resolution
    eps_averaging = series == 'smoothed'
    cpu_before = mm.host_cpu_load_percent() if mp.am_master() else None
    gpu_before = mm.nvidia_smi() if mp.am_master() else None

    sim, dfts, dt_meep, ref_timing = timed_run(g, [], eps_averaging, gate)
    assert math.isclose(dt_meep * mm.TIME_UNIT, g['g7_04']['dt_s'], rel_tol=1e-9), (dt_meep * mm.TIME_UNIT, g['g7_04']['dt_s'])
    reference_monitors = lines(sim, dfts, g)
    sim.reset_meep()
    sim, dfts, _, warmup = timed_run(g, None, eps_averaging, gate)
    sim.reset_meep()
    samples, t1_runs = [], []
    for k in range(repeats):
        sim, dfts, _, timing = timed_run(g, None, eps_averaging, gate)
        samples.append(timing)
        monitors = lines(sim, dfts, g)
        amps = common.point_amplitudes(dict(monitors=monitors, reference_monitors=reference_monitors))
        T, _ = common.efficiencies_from_amplitudes(amps, g['substrate_index'])
        t1_runs.append(T[:, common.ORDERS.index(1)])
        if k < repeats - 1:
            sim.reset_meep()
    stairs = staircase(sim, g) if series == 'staircase' else None
    eps_shape = list(np.shape(sim.get_epsilon()))
    sim.reset_meep()
    if not mp.am_master():
        return None
    record_full = dict(phasor_time_sign=1, monitors=monitors, reference_monitors=reference_monitors)
    common.check_lines(record_full, g)
    amplitudes = common.point_amplitudes(record_full)
    T, R = common.efficiencies_from_amplitudes(amplitudes, g['substrate_index'])
    wavelength = np.asarray(monitors['transmission']['wavelength_um'])
    _, T_c, R_c, diagnostics = compare.solver_efficiencies(record_full, g, list(common.ORDERS))
    assert np.max(abs(T - T_c)) < 1e-12 and np.max(abs(R - R_c)) < 1e-12, 'amplitude path differs from compare.py'
    full = [s['full_seconds'] for s in samples]
    cells = list(g['cells'])
    return dict(
        schema='g7-04-point-v1', case='G7-04', solver='meep', series=series, precision='float64', mesh_um=mesh, resolution_per_um=resolution,
        date=time.strftime('%Y-%m-%d %H:%M:%S'), geometry=g, geometry_sha256=g['_sha256'],
        grid=dict(cells=cells, cell_count=int(np.prod(cells)), epsilon_array_shape=eps_shape, steps=g['steps'], steps_run=samples[-1]['steps_run'],
                  dt_s=dt_meep * mm.TIME_UNIT, courant_number=g['courant_number'], pml_cells=g['pml_cells'], pml_um=g['pml_cells'] * mesh,
                  physical_time_s=g['steps'] * dt_meep * mm.TIME_UNIT, precision='float64', eps_averaging=eps_averaging,
                  subpixel=dict(subpixel_tol=1e-4, subpixel_maxeval=100000, note='Meep defaults') if eps_averaging else None,
                  mpi_processes=int(mp.count_processors()),
                  boundary='Meep PML (stretched-coordinate, quadratic profile by default, R_asymptotic=1e-15), 0.4 um thick',
                  source_y_um=g['source']['y_um']),
        staircase=stairs, amplitudes=amplitudes, wavelength_um=wavelength.tolist(),
        efficiencies=dict(T={str(m): T[:, j].tolist() for j, m in enumerate(common.ORDERS)},
                          R={str(m): R[:, j].tolist() for j, m in enumerate(common.ORDERS)}, total=(T.sum(axis=1) + R.sum(axis=1)).tolist()),
        observables={k: v for k, v in common.observables(T, wavelength).items() if k != 'design_index'},
        repeat_max_abs_t1_difference=float(max(np.max(abs(t - t1_runs[-1])) for t in t1_runs)),
        diagnostics=diagnostics,
        timing=dict(cost='full_seconds: construction, init_sim and run of one grating solve on all ranks (barrier-bounded wall time on the master); '
                         'the bare-substrate reference run is recorded separately and not counted',
                    repeats=repeats, warmup=warmup, samples=samples, median_full_seconds=float(np.median(full)),
                    min_full_seconds=float(min(full)), max_full_seconds=float(max(full)),
                    median_stepping_seconds=float(np.median([s['stepping_seconds'] for s in samples])),
                    median_setup_seconds=float(np.median([s['setup_seconds'] for s in samples])), reference=ref_timing,
                    device=f'CPU, {mp.count_processors()} MPI ranks'),
        host=dict(shared=True, host_cpu_percent_before=cpu_before, host_cpu_percent_after=mm.host_cpu_load_percent(), gpu_before=gpu_before,
                  pause_file=gate[0], max_host_cpu_percent=gate[1], max_wait_seconds=gate[2],
                  note='shared workstation: other agents use the GPU and other sessions may run CPU jobs; the Windows host CPU load (all cores) is '
                       'sampled before every run and recorded; runs wait while the pause file exists and, with max_host_cpu_percent, until '
                       'the load is below it (at most max_wait_seconds)'),
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], wsl_distribution=os.environ.get('WSL_DISTRO_NAME'),
                         meep_version=mp.__version__, packages=mm.versions(('meep', 'numpy', 'mpi4py')), meep_mpi=bool(mp.with_mpi()),
                         mpi_processes=int(mp.count_processors()), omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                         cpu=cpu_model()))


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, required=True, help='MPI ranks this script was launched with (asserted)')
    parser.add_argument('--series', choices=common.SERIES, required=True)
    parser.add_argument('--resolutions', type=int, nargs='+', default=list(common.MEEP_RESOLUTIONS))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pause-file', default=None,
                        help='optional file whose existence delays the next run until it is removed (default: no pause)')
    parser.add_argument('--max-host-cpu-percent', type=float, default=None, help='wait before each run until the host load is at most this')
    parser.add_argument('--max-wait-seconds', type=float, default=3600, help='longest wait for a quiet host before a run starts anyway')
    parser.add_argument('--out-dir', default=str(common.RECORDS))
    args = parser.parse_args()
    mp.verbosity(0)
    assert int(mp.count_processors()) == args.ranks, (mp.count_processors(), args.ranks)
    base = common.load_geometry()
    compare = common.load_compare()
    for resolution in args.resolutions:
        assert resolution in common.MEEP_RESOLUTIONS, resolution
        gate = (args.pause_file, args.max_host_cpu_percent, args.max_wait_seconds)
        record = run_point(base, resolution, args.series, gate, args.repeats, compare)
        if record is None:
            continue
        out = Path(args.out_dir) / f'meep-{args.series}-float64-{resolution}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
        t = record['timing']
        print(f"{out.name}: T+1(1.55)={record['observables']['t1_design']:.6f} band={record['observables']['t1_band_mean']:.6f} "
              f"median {t['median_full_seconds']:.3f} s [{t['min_full_seconds']:.3f}, {t['max_full_seconds']:.3f}] "
              f"cells {record['grid']['cell_count']} steps {record['grid']['steps']} repeat spread {record['repeat_max_abs_t1_difference']:.2e}",
              flush=True)


if __name__ == '__main__':
    main()
