"""Meep side of the metalens comparison (Part A: 2D ridge lens, Part B: 3D pillar lens).

Run inside the micromamba "meep" environment of the WSL2 distribution torchfdtd-bench, from the worktree root:
    OMP_NUM_THREADS=1 mpirun -np 4 python examples/meep_comparison/metalens/meep_metalens.py --part 2d --ranks 4
    OMP_NUM_THREADS=1 mpirun -np 4 python examples/meep_comparison/metalens/meep_metalens.py --part 3d --ranks 4
Only the MPI master writes docs/validation/meep_comparison/metalens_<part>_meep.json. Meep is a double-precision
CPU solver with its own PML; the record states every setting. The DFT fields are read on the Yee grid and
collocated at the cell-centred monitor points with the same averaging stencil as TorchFDTD's plane monitors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import meep as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metalens_common as mc  # noqa: E402

UM = 1e-6
TIME_UNIT = UM / mc.C0          # one Meep time unit in seconds (a = 1 um)
PACKAGES = ('meep', 'numpy', 'scipy', 'mpi4py')


def log(*args):
    if mp.am_master():
        print(*args, flush=True)


def conda_versions(names, listing=None):
    """Versions of conda-forge packages from a `micromamba list` dump named by TORCHFDTD_MEEP_LIST, when present."""
    out = {}
    listing = Path(os.environ['TORCHFDTD_MEEP_LIST']) if listing is None and os.environ.get('TORCHFDTD_MEEP_LIST') else listing
    if listing is None:
        return out
    try:
        for line in listing.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in names:
                out[parts[0]] = parts[1] + ' ' + parts[2]
    except OSError:
        pass
    return out


def custom_source(wave):
    def src_func(t):
        return float(mc.waveform(wave, np.array([t * TIME_UNIT]))[0])
    return src_func


def meep_frequencies(spec):
    return [f * TIME_UNIT for f in mc.frequencies_hz(spec['frequencies'])]   # 1/um


def source_spec(spec, duration):
    wave = spec['source']['waveform']
    return mp.CustomSource(src_func=custom_source(wave), start_time=0, end_time=duration + 1, is_integrated=False)


# ----------------------------------------------------------------------------
# Yee-grid DFT arrays collocated at cell-centred points (TorchFDTD's plane-monitor stencil)
# ----------------------------------------------------------------------------
def pair_mean(a, axis):
    """Average of neighbouring samples along one axis (node grid -> centred grid)."""
    lo = [slice(None)] * a.ndim
    hi = [slice(None)] * a.ndim
    lo[axis], hi[axis] = slice(0, -1), slice(1, None)
    return .5 * (a[tuple(lo)] + a[tuple(hi)])


def inner(a, axis):
    """Drop the two outer half-cell samples along one axis (half grid of n+2 -> centred grid of n)."""
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(1, -1)
    return a[tuple(sl)]


def line_y_normal_2d(sim, dft, k, n):
    """Ez, Hx, Hy at (x_{j+1/2}, y_line) for a y-normal line of n cells; Meep collapses the half-row components onto the line."""
    ez = sim.get_dft_array(dft, mp.Ez, k)
    hx = sim.get_dft_array(dft, mp.Hx, k)
    hy = sim.get_dft_array(dft, mp.Hy, k)
    assert ez.shape == (n + 1,) and hx.shape == (n + 1,) and hy.shape == (n + 2,), (ez.shape, hx.shape, hy.shape)
    return pair_mean(ez, 0), pair_mean(hx, 0), inner(hy, 0)


def line_x_normal_2d(sim, dft, k, n):
    ez = sim.get_dft_array(dft, mp.Ez, k)
    assert ez.shape == (n + 1,), ez.shape
    return pair_mean(ez, 0)


def plane_z_normal_3d(sim, dft, k, nx, ny):
    """Six components at (x_{i+1/2}, y_{j+1/2}, z_plane) for a z-normal plane of nx by ny cells."""
    a = {c: sim.get_dft_array(dft, getattr(mp, c), k) for c in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')}
    expect = dict(Ex=(nx + 2, ny + 1), Ey=(nx + 1, ny + 2), Ez=(nx + 1, ny + 1), Hx=(nx + 1, ny + 2), Hy=(nx + 2, ny + 1), Hz=(nx + 2, ny + 2))
    for c, shape in expect.items():
        assert a[c].shape == shape, (c, a[c].shape, shape)
    return dict(Ex=pair_mean(inner(a['Ex'], 0), 1), Ey=pair_mean(inner(a['Ey'], 1), 0), Ez=pair_mean(pair_mean(a['Ez'], 0), 1),
                Hx=pair_mean(inner(a['Hx'], 1), 0), Hy=pair_mean(inner(a['Hy'], 0), 1), Hz=inner(inner(a['Hz'], 0), 1))


def section_3d(sim, dft, k, normal, nt, nz):
    """Ex, Ey, Ez at (t_{i+1/2}, z_{k+1/2}) for a y-normal (normal='y', t=x) or x-normal (normal='x', t=y) section."""
    a = {c: sim.get_dft_array(dft, getattr(mp, c), k) for c in ('Ex', 'Ey', 'Ez')}
    half = 'Ex' if normal == 'y' else 'Ey'      # the E component along the transverse axis lives on its half grid
    node = 'Ey' if normal == 'y' else 'Ex'
    assert a[half].shape == (nt + 2, nz + 1) and a[node].shape == (nt + 1, nz + 1) and a['Ez'].shape == (nt + 1, nz + 2), {c: v.shape for c, v in a.items()}
    return {half: pair_mean(inner(a[half], 0), 1), node: pair_mean(pair_mean(a[node], 0), 1), 'Ez': pair_mean(inner(a['Ez'], 1), 0)}


def centred(lo, n, dx):
    return lo + (np.arange(n) + .5) * dx


# ----------------------------------------------------------------------------
# Part A: 2D
# ----------------------------------------------------------------------------
def simulation_2d(spec, with_lens):
    s = spec
    dx = s['mesh_um']
    cell = mp.Vector3(s['cell_um'][0], s['cell_um'][1], 0)
    duration = s['steps'] * s['dt_s'] / TIME_UNIT
    src = s['source']
    sources = [mp.Source(source_spec(s, duration), component=mp.Ez, center=mp.Vector3(0, src['y_um'], 0), size=mp.Vector3(src['x_span_um'], 0, 0))]
    si = mp.Medium(index=s['material']['index'])
    geometry = [mp.Block(size=mp.Vector3(r['width_um'], s['ridge_height_um'], mp.inf), center=mp.Vector3(r['x_um'], s['ridge_y_center_um'], 0), material=si)
                for r in s['ridges']] if with_lens else []
    sim = mp.Simulation(cell_size=cell, resolution=1 / dx, boundary_layers=[mp.PML(s['pml_cells'] * dx)], geometry=geometry, sources=sources,
                        Courant=s['courant_number'], eps_averaging=False, dimensions=2)
    freqs = meep_frequencies(s)
    m = s['monitors']
    dfts = {}
    for name in ('incident', 'focal'):
        dfts[name] = sim.add_dft_fields([mp.Ez, mp.Hx, mp.Hy], freqs, center=mp.Vector3(0, m[name]['y_um'], 0), size=mp.Vector3(m[name]['x_span_um'], 0, 0), yee_grid=True)
    y0, y1 = m['axis']['y_range_um']
    dfts['axis'] = sim.add_dft_fields([mp.Ez], freqs, center=mp.Vector3(m['axis']['x_um'], (y0 + y1) / 2, 0), size=mp.Vector3(0, y1 - y0, 0), yee_grid=True)
    return sim, dfts, duration


def collect_2d(sim, dfts, spec):
    s = spec
    dx = s['mesh_um']
    m = s['monitors']
    nf = len(s['frequencies']['wavelengths_um'])
    out = {}
    for name in ('incident', 'focal'):
        n = int(round(m[name]['x_span_um'] / dx))
        rows = [line_y_normal_2d(sim, dfts[name], k, n) for k in range(nf)]
        out[name] = dict(x_um=centred(-m[name]['x_span_um'] / 2, n, dx), weight_m=dx * UM,
                         Ez=np.stack([r[0] for r in rows]), Hx=np.stack([r[1] for r in rows]), Hy=np.stack([r[2] for r in rows]))
    y0, y1 = m['axis']['y_range_um']
    n = int(round((y1 - y0) / dx))
    out['axis'] = dict(y_um=centred(y0, n, dx), Ez=np.stack([line_x_normal_2d(sim, dfts['axis'], k, n) for k in range(nf)]))
    return out


def observables_2d(lens, bare, spec):
    wl = spec['frequencies']['wavelengths_um']
    out = dict(wavelengths_um=wl)
    inc_density = mc.poynting_2d(bare['incident']['Ez'], bare['incident']['Hx'])
    inc_power = inc_density.sum(axis=1) * bare['incident']['weight_m']
    inc_intensity = np.mean(np.abs(bare['incident']['Ez']) ** 2, axis=1)
    trans_power = mc.poynting_2d(lens['incident']['Ez'], lens['incident']['Hx']).sum(axis=1) * lens['incident']['weight_m']
    out['incident'] = dict(x_um=mc.as_list(bare['incident']['x_um']), weight_m=bare['incident']['weight_m'], bare_power=mc.as_list(inc_power),
                           lens_power=mc.as_list(trans_power), transmission=mc.as_list(trans_power / inc_power), bare_mean_intensity=mc.as_list(inc_intensity))
    ez, hx = lens['focal']['Ez'], lens['focal']['Hx']
    out['focal'] = dict(y_um=spec['monitors']['focal']['y_um'], x_um=mc.as_list(lens['focal']['x_um']), weight_m=lens['focal']['weight_m'],
                        intensity=mc.as_list(np.abs(ez) ** 2 / inc_intensity[:, None]), poynting_y=mc.as_list(mc.poynting_2d(ez, hx) / inc_power[:, None]))
    out['axis'] = dict(x_um=spec['monitors']['axis']['x_um'], y_um=mc.as_list(lens['axis']['y_um']),
                       intensity=mc.as_list(np.abs(lens['axis']['Ez']) ** 2 / inc_intensity[:, None]))
    out.update(mc.summary_2d(out, spec))
    return out


def silicon_cells_2d(sim, spec):
    """Meep's material function sampled at the Ez node positions of the ridge rows (eps_averaging=False)."""
    s = spec
    dx = s['mesh_um']
    nx = int(round(s['cell_um'][0] / dx))
    x = -s['cell_um'][0] / 2 + np.arange(nx) * dx
    y = np.arange(math.floor((s['ridge_bottom_um'] + s['cell_um'][1] / 2) / dx) - 1, math.ceil((s['ridge_top_um'] + s['cell_um'][1] / 2) / dx) + 2) * dx - s['cell_um'][1] / 2
    eps = np.asarray(sim.get_epsilon_grid(x, y, np.array([0.])))
    return dict(per_component=dict(Ez=int(np.count_nonzero(eps > 1.5))), rows_sampled=len(y))


# ----------------------------------------------------------------------------
# Part B: 3D
# ----------------------------------------------------------------------------
def simulation_3d(spec, with_lens):
    s = spec
    dx = s['mesh_um']
    cell = mp.Vector3(*s['cell_um'])
    duration = s['steps'] * s['dt_s'] / TIME_UNIT
    src = s['source']
    # The sheet is (x_span - dx) by y_span so that its x boundaries fall on Ex samples (half cells) and its y boundaries on nodes:
    # Meep then weights the boundary samples by 1/2 (1/4 at the corners) and injects nothing outside; TorchFDTD applies the same weights.
    sources = [mp.Source(source_spec(s, duration), component=mp.Ex, center=mp.Vector3(0, 0, src['z_um']), size=mp.Vector3(src['x_span_um'] - dx, src['y_span_um'], 0))]
    si = mp.Medium(index=s['material']['index'])
    geometry = [mp.Cylinder(radius=p['radius_um'], height=s['pillar_height_um'], center=mp.Vector3(p['x_um'], p['y_um'], s['pillar_z_center_um']), material=si)
                for p in s['pillars']] if with_lens else []
    sim = mp.Simulation(cell_size=cell, resolution=1 / dx, boundary_layers=[mp.PML(s['pml_cells'] * dx)], geometry=geometry, sources=sources,
                        Courant=s['courant_number'], eps_averaging=False, dimensions=3)
    freqs = meep_frequencies(s)
    m = s['monitors']
    dfts = {}
    six = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
    for name in ('incident', 'focal'):
        span = m[name]['span_um']
        dfts[name] = sim.add_dft_fields(six, freqs, center=mp.Vector3(0, 0, m[name]['z_um']), size=mp.Vector3(span, span, 0), yee_grid=True)
    (x0, x1), (z0, z1) = m['sections']['x_range_um'], m['sections']['z_range_um']
    dfts['xz'] = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], freqs, center=mp.Vector3((x0 + x1) / 2, 0, (z0 + z1) / 2), size=mp.Vector3(x1 - x0, 0, z1 - z0), yee_grid=True)
    dfts['yz'] = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], freqs, center=mp.Vector3(0, (x0 + x1) / 2, (z0 + z1) / 2), size=mp.Vector3(0, x1 - x0, z1 - z0), yee_grid=True)
    return sim, dfts, duration


def collect_3d(sim, dfts, spec):
    s = spec
    dx = s['mesh_um']
    m = s['monitors']
    nf = len(s['frequencies']['wavelengths_um'])
    out = {}
    for name in ('incident', 'focal'):
        n = int(round(m[name]['span_um'] / dx))
        planes = [plane_z_normal_3d(sim, dfts[name], k, n, n) for k in range(nf)]
        u = centred(-m[name]['span_um'] / 2, n, dx)
        out[name] = dict(x_um=u, y_um=u, shape=(n, n), weight_m2=(dx * UM) ** 2,
                         **{c: np.stack([p[c].reshape(-1) for p in planes]) for c in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')})
    (x0, x1), (z0, z1) = m['sections']['x_range_um'], m['sections']['z_range_um']
    nt, nz = int(round((x1 - x0) / dx)), int(round((z1 - z0) / dx))
    for name, normal in (('xz', 'y'), ('yz', 'x')):
        secs = [section_3d(sim, dfts[name], k, normal, nt, nz) for k in range(nf)]
        out[name] = dict(transverse_um=centred(x0, nt, dx), z_um=centred(z0, nz, dx), shape=(nt, nz),
                         **{c: np.stack([sec[c] for sec in secs]) for c in ('Ex', 'Ey', 'Ez')})
    return out


def observables_3d(lens, bare, spec):
    wl = spec['frequencies']['wavelengths_um']
    out = dict(wavelengths_um=wl)
    b = bare['incident']
    inc_power = mc.poynting_3d_z(b['Ex'], b['Ey'], b['Hx'], b['Hy']).sum(axis=1) * b['weight_m2']
    inc_intensity = np.mean(np.abs(b['Ex']) ** 2 + np.abs(b['Ey']) ** 2 + np.abs(b['Ez']) ** 2, axis=1)
    t = lens['incident']
    trans_power = mc.poynting_3d_z(t['Ex'], t['Ey'], t['Hx'], t['Hy']).sum(axis=1) * t['weight_m2']
    out['incident'] = dict(shape=[b['shape'][0], b['shape'][1], 1], bare_power=mc.as_list(inc_power), lens_power=mc.as_list(trans_power),
                           transmission=mc.as_list(trans_power / inc_power), bare_mean_intensity=mc.as_list(inc_intensity))
    f = lens['focal']
    intensity = (np.abs(f['Ex']) ** 2 + np.abs(f['Ey']) ** 2 + np.abs(f['Ez']) ** 2) / inc_intensity[:, None]
    density = mc.poynting_3d_z(f['Ex'], f['Ey'], f['Hx'], f['Hy']) / inc_power[:, None]
    out['focal'] = dict(z_um=spec['monitors']['focal']['z_um'], shape=[f['shape'][0], f['shape'][1], 1], x_um=mc.as_list(f['x_um']), y_um=mc.as_list(f['y_um']),
                        weight_m2=f['weight_m2'], intensity=mc.as_list(intensity), poynting_z=mc.as_list(density))
    for name in ('xz', 'yz'):
        sec = lens[name]
        inten = (np.abs(sec['Ex']) ** 2 + np.abs(sec['Ey']) ** 2 + np.abs(sec['Ez']) ** 2) / inc_intensity[:, None, None]
        shape = [sec['shape'][0], 1, sec['shape'][1]] if name == 'xz' else [1, sec['shape'][0], sec['shape'][1]]
        out[name] = dict(shape=shape, transverse_um=mc.as_list(sec['transverse_um']), z_um=mc.as_list(sec['z_um']), intensity=[mc.as_list(inten[k]) for k in range(len(wl))])
    out.update(mc.summary_3d(out, spec))
    return mc.trim_3d(out)


def source_support(part, spec):
    """Cell slices of the Meep source samples and their weights, from the sheet extent and Meep's linear volume interpolation
    (a boundary sample that lies on a Yee position of the component gets weight 1/2); the same layout as the TorchFDTD record."""
    s = spec
    dx = s['mesh_um']
    src = s['source']
    if part == '2d':
        # Ez on nodes: the sheet |x| <= x_span/2 covers the nodes [lo, hi], the two end nodes at 1/2.
        lo = int(round((s['cell_um'][0] / 2 - src['x_span_um'] / 2) / dx))
        hi = int(round((s['cell_um'][0] / 2 + src['x_span_um'] / 2) / dx))
        j = int(round((s['cell_um'][1] / 2 + src['y_um']) / dx))
        return dict(sheet=dict(cell_slices=[[lo + 1, hi], [j, j + 1], [0, 1]], amplitude=1), end_neg=dict(cell_slices=[[lo, lo + 1], [j, j + 1], [0, 1]], amplitude=.5),
                    end_pos=dict(cell_slices=[[hi, hi + 1], [j, j + 1], [0, 1]], amplitude=.5))
    # Ex on x half cells and y nodes: the sheet (x_span - dx) by y_span has its x boundaries on the half cells lo_x, hi_x - 1 and its
    # y boundaries on the nodes lo_y, hi_y - 1, each at 1/2; the interior at 1; the four corners at 1/4.
    lo_x = int(round((s['cell_um'][0] / 2 - (src['x_span_um'] - dx) / 2) / dx - .5))
    hi_x = int(round((s['cell_um'][0] / 2 + (src['x_span_um'] - dx) / 2) / dx - .5)) + 1
    lo_y = int(round((s['cell_um'][1] / 2 - src['y_span_um'] / 2) / dx))
    hi_y = int(round((s['cell_um'][1] / 2 + src['y_span_um'] / 2) / dx)) + 1
    k = int(round((s['cell_um'][2] / 2 + src['z_um']) / dx))
    zs = [k, k + 1]
    out = dict(sheet=dict(cell_slices=[[lo_x + 1, hi_x - 1], [lo_y + 1, hi_y - 1], zs], amplitude=1))
    for name, xs, ys in (('edge_x-1', [lo_x, lo_x + 1], [lo_y + 1, hi_y - 1]), ('edge_x+1', [hi_x - 1, hi_x], [lo_y + 1, hi_y - 1]),
                         ('edge_y-1', [lo_x + 1, hi_x - 1], [lo_y, lo_y + 1]), ('edge_y+1', [lo_x + 1, hi_x - 1], [hi_y - 1, hi_y])):
        out[name] = dict(cell_slices=[xs, ys, zs], amplitude=.5)
    for name, xs, ys in (('corner_x-1_y-1', [lo_x, lo_x + 1], [lo_y, lo_y + 1]), ('corner_x-1_y+1', [lo_x, lo_x + 1], [hi_y - 1, hi_y]),
                         ('corner_x+1_y-1', [hi_x - 1, hi_x], [lo_y, lo_y + 1]), ('corner_x+1_y+1', [hi_x - 1, hi_x], [hi_y - 1, hi_y])):
        out[name] = dict(cell_slices=[xs, ys, zs], amplitude=.25)
    return out


def silicon_cells_3d(sim, spec):
    """Meep's material function at the Ex, Ey and Ez Yee positions of the pillar layers (eps_averaging=False)."""
    s = spec
    dx = s['mesh_um']
    n = [int(round(v / dx)) for v in s['cell_um']]
    nodes = [-s['cell_um'][a] / 2 + np.arange(n[a]) * dx for a in range(3)]
    halves = [v + dx / 2 for v in nodes]
    zsel = (nodes[2] > s['pillar_bottom_um'] - 2 * dx) & (nodes[2] < s['pillar_top_um'] + 2 * dx)
    out = {}
    for comp, axis in (('Ex', 0), ('Ey', 1), ('Ez', 2)):
        grids = [halves[a] if a == axis else nodes[a] for a in range(3)]
        zs = (halves[2] if axis == 2 else nodes[2])[zsel]
        eps = np.asarray(sim.get_epsilon_grid(grids[0], grids[1], zs))
        out[comp] = int(np.count_nonzero(eps > 1.5))
    return dict(per_component=out, layers_sampled=int(zsel.sum()))


# ----------------------------------------------------------------------------
def run_part(part, spec, timing):
    simulation = simulation_2d if part == '2d' else simulation_3d
    collect = collect_2d if part == '2d' else collect_3d
    mp.verbosity(0)

    def solve(with_lens):
        mp.all_wait()
        t0 = time.perf_counter()
        sim, dfts, duration = simulation(spec, with_lens)
        dt = sim.Courant / sim.resolution * TIME_UNIT
        assert math.isclose(dt, spec['dt_s'], rel_tol=1e-12), (dt, spec['dt_s'])
        sim.init_sim()
        sim._evaluate_dft_objects()   # what Simulation.run() does before stepping: creates the DFT chunks
        mp.all_wait()
        setup = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(spec['steps']):
            sim.fields.step()
        mp.all_wait()
        stepping = time.perf_counter() - t1
        full = time.perf_counter() - t0
        steps_run = int(sim.fields.t)
        assert steps_run == spec['steps'], steps_run
        data = collect(sim, dfts, spec)
        row = dict(setup_seconds=setup, stepping_seconds=stepping, full_seconds=full, steps=steps_run, dt_s=dt,
                   mcells_per_second=math.prod(mc.cells(spec)) * steps_run / stepping / 1e6)
        log(json.dumps(dict(with_lens=with_lens, **row)))
        return sim, data, row
    log('bare cell run')
    sim, bare, bare_timing = solve(False)
    sim.reset_meep()
    samples = []
    repeats = 4 if timing else 1
    for i in range(repeats):
        log(f'lens run {i}')
        sim, lens, row = solve(True)
        samples.append(row)
        if i < repeats - 1:
            sim.reset_meep()
    silicon = (silicon_cells_2d if part == '2d' else silicon_cells_3d)(sim, spec)
    observables = (observables_2d if part == '2d' else observables_3d)(lens, bare, spec)
    return sim, observables, silicon, bare_timing, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('2d', '3d'), required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--ranks', type=int, default=None, help='expected MPI rank count (asserted against mpirun -np)')
    parser.add_argument('--timing', action='store_true', help='one warm-up, then three timed lens solves; medians are reported')
    args = parser.parse_args()
    if args.ranks is not None:
        assert int(mp.count_processors()) == args.ranks, (mp.count_processors(), args.ranks)
    spec = mc.load_geometry(args.part)
    load_note = mc.host_load_note('timing' if args.timing else 'development') if mp.am_master() else None
    sim, observables, silicon, bare_timing, samples = run_part(args.part, spec, args.timing)
    timed = samples[1:] if args.timing else samples
    if not mp.am_master():
        return
    conda = conda_versions(('mpich', 'pymeep', 'libmeep', 'python', 'numpy', 'scipy'))
    dx = spec['mesh_um']
    record = dict(schema='torchfdtd-meep-comparison-v1', solver='meep', example='metalens', part=args.part, geometry_file=Path(spec['_path']).name,
                  geometry_sha256=spec['_sha256'], date=time.strftime('%Y-%m-%d'),
                  environment=mc.environment(dict(packages=mc.package_versions(PACKAGES), meep_version=mp.__version__, conda_packages=conda,
                                                  meep_mpi=bool(mp.with_mpi()), mpi_processes=int(mp.count_processors()),
                                                  omp_num_threads=os.environ.get('OMP_NUM_THREADS'), precision='float64 (Meep build)')),
                  method=('Meep 1.34 Yee grid, eps_averaging=False (material sampled at each Yee component position), double precision, MPI ranks on the '
                          'CPU, Meep PML of the shared thickness, CustomSource current sheet with the shared waveform (is_integrated=False), DFT fields on '
                          'the Yee grid (yee_grid=True) accumulated over the whole run, collocated at the cell-centred monitor points by the same '
                          'neighbour averaging as TorchFDTD. Bare-cell normalisation run on the same grid with the same monitors.'),
                  grid=dict(shape=mc.cells(spec) + ([1] if args.part == '2d' else []), cells=int(math.prod(mc.cells(spec))), mesh_um=dx, resolution_per_um=sim.resolution,
                            dt_s=samples[-1]['dt_s'], courant_number=sim.Courant, steps=samples[-1]['steps'], pml_cells=spec['pml_cells'], precision='float64',
                            backend='cpu', mpi_processes=int(mp.count_processors())),
                  pml=dict(formulation='Meep PML (stretched-coordinate, quadratic profile by default, R_asymptotic=1e-15)', layers=spec['pml_cells'],
                           thickness_um=spec['pml_cells'] * dx),
                  source=dict(waveform=spec['source']['waveform'], support=source_support(args.part, spec),
                              support_note='volume source whose boundaries lie on Yee positions of the component; Meep weights those boundary samples '
                                           'by 1/2 (linear volume interpolation, measured on a test cell), the corners by 1/4, and injects nothing outside',
                              injection='CustomSource current, is_integrated=False'),
                  silicon_cells=silicon,
                  observables=observables,
                  timing=dict(timing_mode='timing' if args.timing else 'development', samples=timed, warmup=samples[0] if args.timing else None,
                              bare_run=bare_timing, host_load_note=load_note, **mc.timing_summary(timed)))
    out = Path(args.out) if args.out else mc.RECORDS / f'metalens_{args.part}_meep.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out, flush=True)
    print(json.dumps(observables['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
