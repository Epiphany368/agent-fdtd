"""Shared, solver-free parts of the G7-04 accuracy-versus-cost study (docs/G7_WORKFLOWS.md, G7-04).

The fixed two-ridge metagrating of examples/meep_comparison/metagrating/geometry.json is defined
in micrometres. derive_geometry() turns it into the fixture of one mesh and one material-sampling
series: cells = size / mesh, a 0.4 um absorber, the base physical time and Courant number, and in
the staircase series a half-cell shift of the whole structure in x and y at a mesh where a
material edge would fall on an Ez node of either solver. Both solver drivers import this module,
so TorchFDTD and Meep receive the same derived geometry.

Order efficiencies use the decomposition of examples/meep_comparison/metagrating/compare.py. The
records keep the per-order spatial Fourier means of Ez and Hx on each DFT line (the only quantities
that decomposition reads), so they stay small at every mesh and reproduce the efficiencies exactly.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METAGRATING = ROOT / 'examples' / 'meep_comparison' / 'metagrating'
BASE_GEOMETRY = METAGRATING / 'geometry.json'
RECORDS = ROOT / 'docs' / 'validation' / 'g7' / 'G7-04'
FIGURE = ROOT / 'docs' / 'figures' / 'g7' / 'G7-04.png'
C0 = 299792458.0

TORCHFDTD_MESHES_UM = (0.04, 0.02, 0.01, 0.005)
MEEP_RESOLUTIONS = (25, 50, 100, 200)
SERIES = ('staircase', 'smoothed')
ABSORBER_UM = 0.4
PHYSICAL_TIME_FS = 560
ORDERS = (-1, 0, 1)
PERIOD_UM = 2.0
DESIGN_WAVELENGTH_UM = 1.55
TARGETS = (0.01, 0.005, 0.002)
TIE_TOLERANCE_UM = 1e-9


def load_geometry(path=BASE_GEOMETRY):
    raw = Path(path).read_bytes()
    geometry = json.loads(raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(raw).hexdigest()
    return geometry


def load_compare():
    """examples/meep_comparison/metagrating/compare.py as a module (numpy only at import)."""
    spec = importlib.util.spec_from_file_location('metagrating_compare', METAGRATING / 'compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def material_edges(g):
    """Ridge sides (x) and the substrate and ridge tops (y) in micrometres."""
    xs = sorted(round(r['center_x_um'] + s * r['width_um'] / 2, 12) for r in g['ridges'] for s in (-1, 1))
    top = g['substrate_top_y_um']
    return dict(x=xs, y=[round(top, 12), round(top + g['ridge_height_um'], 12)])


def ez_nodes(solver, length_um, mesh_um, count):
    """Ez node coordinates along one axis of a cell centred on the origin.

    TorchFDTD places node i at -L/2 + i*h. Meep places Ez at integer multiples of h from the cell
    centre (checked with fields.get_chi1inv), which is the same set only for an even cell count.
    """
    if solver == 'torchfdtd':
        return -length_um / 2 + mesh_um * np.arange(count)
    if solver == 'meep':
        return mesh_um * (np.arange(count) - count // 2)
    raise ValueError(solver)


def edge_ties(g, solvers=('torchfdtd', 'meep')):
    """Material edges that fall on an Ez node, per solver and axis."""
    edges = material_edges(g)
    out = {}
    for solver in solvers:
        out[solver] = {}
        for axis, name in enumerate('xy'):
            nodes = ez_nodes(solver, g['cell_size_um'][axis], g['mesh_um'], g['cells'][axis])
            out[solver][name] = [e for e in edges[name] if np.min(abs(nodes - e)) < TIE_TOLERANCE_UM]
    return out


def any_tie(ties):
    return any(v for solver in ties.values() for v in solver.values())


def cells_for(length_um, mesh_um):
    n = round(length_um / mesh_um)
    assert math.isclose(n * mesh_um, length_um, rel_tol=0, abs_tol=1e-9), (length_um, mesh_um)
    return n


def derive_geometry(base, mesh_um, series):
    """The fixture of one mesh and one series, derived from the base geometry (see the module docstring)."""
    assert series in SERIES, series
    lx, ly = base['cell_size_um']
    base_dt = base['courant_number'] * base['mesh_um'] * 1e-6 / C0
    base_time = base['steps'] * base_dt
    dt = base['courant_number'] * mesh_um * 1e-6 / C0
    steps = round(base_time / dt)
    assert math.isclose(steps * dt, base_time, rel_tol=1e-9), (steps, dt, base_time)
    assert math.isclose(base_time * 1e15, PHYSICAL_TIME_FS, rel_tol=1e-3), base_time
    base_absorber = base['pml_cells'] * base['mesh_um']
    assert math.isclose(base_absorber, ABSORBER_UM, rel_tol=1e-12), base_absorber
    g = copy.deepcopy({k: v for k, v in base.items() if not k.startswith('_') and k != 'notes'})
    g.update(mesh_um=mesh_um, cells=[cells_for(lx, mesh_um), cells_for(ly, mesh_um)], pml_cells=cells_for(ABSORBER_UM, mesh_um), steps=steps)
    ties = edge_ties(g)
    shift = mesh_um / 2 if series == 'staircase' and any_tie(ties) else 0.0
    if shift:
        for ridge in g['ridges']:
            ridge['center_x_um'] = round(ridge['center_x_um'] + shift, 12)
        g['substrate_top_y_um'] = round(g['substrate_top_y_um'] + shift, 12)
    ties_after = edge_ties(g)
    if series == 'staircase':
        assert not any_tie(ties_after), ties_after
    g['notes'] = derived_notes(g, series, shift)
    g['g7_04'] = dict(series=series, base_geometry_sha256=base['_sha256'], mesh_um=mesh_um, resolution_per_um=round(1 / mesh_um),
                      shift_um=[shift, shift], edges_on_nodes_before_shift=ties, edges_on_nodes=ties_after,
                      material_edges_um=material_edges(g), physical_time_s=steps * dt, dt_s=dt,
                      rule=('staircase: the whole structure (ridges and substrate top) moves by half a cell in +x and +y when a material edge '
                            'lies on an Ez node of TorchFDTD or Meep; source and DFT lines stay; smoothed: no shift'))
    raw = json.dumps(g, sort_keys=True).encode('utf-8')
    g['_sha256'] = hashlib.sha256(raw).hexdigest()
    return g


def derived_notes(g, series, shift):
    h = g['mesh_um']
    nx, ny = g['cells']
    notes = [
        f"Derived for G7-04 from examples/meep_comparison/metagrating/geometry.json at mesh {h:g} um: {nx} x {ny} cells (cell size / mesh), "
        f"{g['pml_cells']} absorber cells on each y face ({ABSORBER_UM:g} um), {g['steps']} steps of the base Courant number "
        f"{g['courant_number']:.6f} (the base physical time). Source and DFT lines keep their positions in um.",
        "Ez nodes: TorchFDTD at -L/2 + i*h, Meep at integer multiples of h from the cell centre; the two sets coincide on an axis with an "
        "even cell count and lie half a cell apart on an axis with an odd count."
        + (f" Here the y axis has {ny} cells, so the two solvers sample the ridges and the substrate top on different rows." if ny % 2 else ''),
        f"TorchFDTD: Region.pml_cells is capped at 50, so the absorber is set per face with BoundaryFace(kind='pml', layers={g['pml_cells']}) "
        "on y_min and y_max (no package change); Meep: PML of pml_cells * mesh_um = 0.4 um in y.",
    ]
    if series == 'staircase':
        notes.append(f"Staircase series: the structure moves by {shift:g} um in +x and +y because material edges would lie on Ez nodes at this mesh."
                     if shift else 'Staircase series: no material edge lies on an Ez node at this mesh, no shift.')
    else:
        notes.append('Smoothed series: no shift; Meep subpixel averaging (eps_averaging=True), TorchFDTD experimental subpixel interfaces.')
    return notes


def order_amplitudes(monitor, orders=ORDERS):
    """Spatial Fourier means of Ez and Hx per order and wavelength, exactly as compare.order_powers forms them."""
    x = np.asarray(monitor['x_um'], dtype=np.float64)
    ez = np.asarray(monitor['ez_real'], dtype=np.float64) + 1j * np.asarray(monitor['ez_imag'], dtype=np.float64)
    hx = np.asarray(monitor['hx_real'], dtype=np.float64) + 1j * np.asarray(monitor['hx_imag'], dtype=np.float64)
    out = dict(y_um=monitor['y_um'], samples=len(x), x_first_um=float(x[0]), x_step_um=float(x[1] - x[0]),
               wavelength_um=list(monitor['wavelength_um']))
    for m in orders:
        kx = 2 * np.pi * m / PERIOD_UM
        phase = np.exp(-1j * kx * x)
        ae = (ez * phase).mean(axis=1)
        ah = (hx * phase).mean(axis=1)
        out[str(m)] = dict(ez_real=ae.real.tolist(), ez_imag=ae.imag.tolist(), hx_real=ah.real.tolist(), hx_imag=ah.imag.tolist())
    return out



def powers_from_amplitudes(amp, index, orders=ORDERS):
    """Forward and backward order power from the recorded Fourier means (the compare.order_powers formula)."""
    wavelength = np.asarray(amp['wavelength_um'], dtype=np.float64)
    k0 = 2 * np.pi / wavelength
    forward = np.zeros((len(wavelength), len(orders)))
    backward = np.zeros_like(forward)
    for j, m in enumerate(orders):
        kx = 2 * np.pi * m / PERIOD_UM
        ky2 = (index * k0) ** 2 - kx ** 2
        assert np.all(ky2 > 0), 'every recorded order propagates over the band'
        ky = np.sqrt(ky2)
        a = amp[str(m)]
        ae = np.asarray(a['ez_real']) + 1j * np.asarray(a['ez_imag'])
        ah = np.asarray(a['hx_real']) + 1j * np.asarray(a['hx_imag'])
        forward[:, j] = .5 * (ky / k0) * abs((ae + (k0 / ky) * ah) / 2) ** 2
        backward[:, j] = .5 * (ky / k0) * abs((ae - (k0 / ky) * ah) / 2) ** 2
    return forward, backward


def efficiencies_from_amplitudes(amplitudes, substrate_index, orders=ORDERS):
    """T_m and R_m per wavelength from the recorded amplitudes of one point (sample and bare-substrate reference)."""
    i0 = orders.index(0)
    inc_f, _ = powers_from_amplitudes(amplitudes['reference']['reflection'], substrate_index, orders)
    incident = inc_f[:, i0]
    _, s_r_b = powers_from_amplitudes(amplitudes['sample']['reflection'], substrate_index, orders)
    s_t_f, _ = powers_from_amplitudes(amplitudes['sample']['transmission'], 1.0, orders)
    return s_t_f / incident[:, None], s_r_b / incident[:, None]


def point_amplitudes(record):
    """Amplitudes of the sample and reference lines of a full (compare.py-format) monitor record."""
    return {kind: {name: order_amplitudes(record[key][name]) for name in ('reflection', 'transmission')}
            for kind, key in (('sample', 'monitors'), ('reference', 'reference_monitors'))}


def check_lines(record, g):
    """Every DFT line samples the pixel centres of the point's mesh at the declared height."""
    expected = -g['cell_size_um'][0] / 2 + g['mesh_um'] * (np.arange(g['cells'][0]) + .5)
    for kind in ('monitors', 'reference_monitors'):
        for name in ('reflection', 'transmission'):
            line = record[kind][name]
            assert np.allclose(line['x_um'], expected, rtol=0, atol=1e-9), (name, line['x_um'][:3], expected[:3])
            assert abs(line['y_um'] - g['monitors'][name + '_y_um']) < 1e-9, (name, line['y_um'])


def observables(T, wavelength_um, orders=ORDERS):
    """T+1 at the design wavelength and the band mean of T+1."""
    wavelength = np.asarray(wavelength_um)
    i = int(np.argmin(abs(wavelength - DESIGN_WAVELENGTH_UM)))
    assert math.isclose(wavelength[i], DESIGN_WAVELENGTH_UM, rel_tol=1e-9)
    t1 = np.asarray(T)[:, list(orders).index(1)]
    return dict(t1_design=float(t1[i]), t1_band_mean=float(t1.mean()), design_index=i)


def cost_to_reach(costs, errors, target):
    """Cost at which an error series reaches `target`, by log-log interpolation in refinement order.

    `costs` and `errors` run from the coarsest to the finest point. The crossing is the last one:
    every finer point stays at or below the target, so a coarse point that meets the target by a
    cancellation and a finer one that exceeds it again do not count. Returns a dict with status
    'interpolated', 'coarsest point' (the coarsest point already meets the target; its cost is an
    upper bound) or 'not reached' (the finest point exceeds the target).
    """
    costs = [float(c) for c in costs]
    errors = [float(e) for e in errors]
    assert len(costs) == len(errors) >= 2 and all(c > 0 for c in costs)
    if errors[-1] > target:
        return dict(status='not reached', cost=None, between=None)
    k = len(errors) - 1
    while k > 0 and errors[k - 1] <= target:
        k -= 1
    if k == 0:
        return dict(status='coarsest point', cost=costs[0], between=[0, 0])
    e0, e1 = max(errors[k - 1], 1e-15), max(errors[k], 1e-15)
    c0, c1 = costs[k - 1], costs[k]
    t = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
    return dict(status='interpolated', cost=math.exp(math.log(c0) + t * (math.log(c1) - math.log(c0))), between=[k - 1, k])
