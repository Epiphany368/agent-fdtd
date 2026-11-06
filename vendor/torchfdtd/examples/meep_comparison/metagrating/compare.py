"""Compare the recorded metagrating runs of TorchFDTD, Meep and the RCWA oracle. Reads records only.

    python examples/meep_comparison/metagrating/compare.py

Both FDTD records hold the complex Ez/Hx DFT lines of the grating run and of the bare-substrate
reference run. This script applies one Fourier decomposition to both, forms diffraction-order
efficiencies as ratios to the incident power of the reference run of the same solver, evaluates
criteria.json, writes docs/validation/meep_comparison/metagrating_comparison.json and renders
docs/figures/meep_comparison/metagrating.png from the records.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'
FIGURES = ROOT / 'docs' / 'figures' / 'meep_comparison'
C0 = 299792458.0


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def complex_line(monitor, key):
    return np.asarray(monitor[key + '_real'], dtype=np.float64) + 1j * np.asarray(monitor[key + '_imag'], dtype=np.float64)


def order_powers(x_um, ez, hx, wavelength_um, period_um, index, orders, *, conjugate=False):
    """Forward and backward power of each order on a line of uniform midpoint samples.

    Phasors follow the DFT convention sum f(t) exp(+i omega t): a +y wave is exp(+i k_y y) and its
    magnetic field is H_x = (k_y / k_0) E_z in the vacuum-normalised units of both solvers. A record
    written with the opposite convention is conjugated first. Returns (F, M) arrays of forward and
    backward power (0.5 * (k_y / k_0) * |e_pm|^2 per order), the propagating mask and the order
    amplitudes.
    """
    x = np.asarray(x_um, dtype=np.float64)
    ez = np.asarray(ez, dtype=np.complex128)
    hx = np.asarray(hx, dtype=np.complex128)
    if conjugate:
        ez, hx = ez.conj(), hx.conj()
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    k0 = 2 * np.pi / wavelength                                # per um
    forward = np.zeros((len(wavelength), len(orders)))
    backward = np.zeros_like(forward)
    propagating = np.zeros(forward.shape, dtype=bool)
    amplitudes = {}
    for j, m in enumerate(orders):
        kx = 2 * np.pi * m / period_um
        ky2 = (index * k0) ** 2 - kx ** 2
        ok = ky2 > 0
        ky = np.sqrt(np.where(ok, ky2, 1.0))
        phase = np.exp(-1j * kx * x)
        ae = (ez * phase).mean(axis=1)
        ah = (hx * phase).mean(axis=1)
        e_plus = (ae + (k0 / ky) * ah) / 2
        e_minus = (ae - (k0 / ky) * ah) / 2
        forward[:, j] = np.where(ok, .5 * (ky / k0) * abs(e_plus) ** 2, 0.0)
        backward[:, j] = np.where(ok, .5 * (ky / k0) * abs(e_minus) ** 2, 0.0)
        propagating[:, j] = ok
        amplitudes[m] = (e_plus, e_minus)
    return forward, backward, propagating, amplitudes


def solver_efficiencies(record, geometry, orders):
    """T_m, R_m per wavelength from one FDTD record, plus reference diagnostics."""
    conjugate = record.get('phasor_time_sign', +1) == -1
    n_sub = geometry['substrate_index']
    period = geometry['period_um']
    refl = record['monitors']['reflection']
    tran = record['monitors']['transmission']
    ref_refl = record['reference_monitors']['reflection']
    ref_tran = record['reference_monitors']['transmission']
    wavelength = np.asarray(tran['wavelength_um'])
    assert np.allclose(wavelength, refl['wavelength_um']) and np.allclose(wavelength, ref_refl['wavelength_um'])
    # incident power: forward branch of order 0 at the reflection line of the bare-substrate run
    inc_f, inc_b, _, _ = order_powers(ref_refl['x_um'], complex_line(ref_refl, 'ez'), complex_line(ref_refl, 'hx'), wavelength, period, n_sub, orders, conjugate=conjugate)
    i0 = orders.index(0)
    incident = inc_f[:, i0]
    ref_t_f, ref_t_b, _, _ = order_powers(ref_tran['x_um'], complex_line(ref_tran, 'ez'), complex_line(ref_tran, 'hx'), wavelength, period, 1.0, orders, conjugate=conjugate)
    s_r_f, s_r_b, prop_r, _ = order_powers(refl['x_um'], complex_line(refl, 'ez'), complex_line(refl, 'hx'), wavelength, period, n_sub, orders, conjugate=conjugate)
    s_t_f, s_t_b, prop_t, _ = order_powers(tran['x_um'], complex_line(tran, 'ez'), complex_line(tran, 'hx'), wavelength, period, 1.0, orders, conjugate=conjugate)
    T = s_t_f / incident[:, None]
    R = s_r_b / incident[:, None]
    fresnel_r = ((n_sub - 1) / (n_sub + 1)) ** 2
    diagnostics = dict(
        reference_T0=(ref_t_f[:, i0] / incident).tolist(), reference_R0=(inc_b[:, i0] / incident).tolist(),
        fresnel_T0=1 - fresnel_r, fresnel_R0=fresnel_r,
        max_abs_reference_T0_minus_fresnel=float(np.max(abs(ref_t_f[:, i0] / incident - (1 - fresnel_r)))),
        max_abs_reference_R0_minus_fresnel=float(np.max(abs(inc_b[:, i0] / incident - fresnel_r))),
        max_reference_nonzero_order_power_over_incident=float(np.max(np.delete((inc_f + inc_b + ref_t_f + ref_t_b) / incident[:, None], i0, axis=1))),
        max_reference_transmission_backward_over_incident=float(np.max(ref_t_b[:, i0] / incident)),
        max_sample_transmission_backward_over_incident=float(np.max(s_t_b / incident[:, None])),
        all_orders_propagating=bool(prop_r.all() and prop_t.all()),
        sample_flux_ratio_transmission=(np.asarray(tran['flux']) / np.asarray(ref_refl['flux'])).tolist(),
    )
    return wavelength, T, R, diagnostics


def rcwa_efficiencies(rcwa, orders):
    wavelength = np.asarray(rcwa['wavelength_um'])
    T = np.stack([np.asarray(rcwa['T'][str(m)]) for m in orders], axis=1)
    R = np.stack([np.asarray(rcwa['R'][str(m)]) for m in orders], axis=1)
    return wavelength, T, R


def grid_match(a, b, criteria):
    ga, gb = a['grid'], b['grid']
    checks = dict(
        cells=(ga['cells'] == gb['cells']),
        dt=math.isclose(ga['dt_s'], gb['dt_s'], rel_tol=criteria['grid_match']['dt_relative']),
        steps=(ga['steps'] == gb['steps']),
        pml_cells=(ga['pml_cells'] == gb['pml_cells']),
        courant=math.isclose(ga['courant_number'], gb['courant_number'], rel_tol=criteria['grid_match']['dt_relative']),
        geometry_sha256=(a['geometry_sha256'] == b['geometry_sha256']),
        staircase=(a['staircase']['silicon_columns'] == b['staircase']['silicon_columns'] and a['staircase']['silicon_rows'] == b['staircase']['silicon_rows']),
    )
    tol = criteria['grid_match']['monitor_sample_positions_um_abs']
    for name in ('reflection', 'transmission'):
        ma, mb = a['monitors'][name], b['monitors'][name]
        checks[name + '_x_samples'] = bool(np.allclose(ma['x_um'], mb['x_um'], rtol=0, atol=tol))
        checks[name + '_y'] = bool(abs(ma['y_um'] - mb['y_um']) <= tol)
        checks[name + '_wavelengths'] = bool(np.allclose(ma['wavelength_um'], mb['wavelength_um'], rtol=1e-9, atol=0))
    return checks


def table_rows(wavelength, orders, results, i_design):
    """Markdown table of the six efficiencies at the design wavelength and their band extrema."""
    lines = ['| quantity | TorchFDTD | Meep | RCWA | TorchFDTD-Meep | TorchFDTD-RCWA | Meep-RCWA |', '|---|---|---|---|---|---|---|']
    for kind in ('T', 'R'):
        for j, m in enumerate(orders):
            vals = [results[s][kind][i_design, j] for s in ('torchfdtd', 'meep', 'rcwa')]
            lines.append(f'| {kind}{m:+d} at {wavelength[i_design]:.4g} um | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | '
                         f'{vals[0]-vals[1]:+.4f} | {vals[0]-vals[2]:+.4f} | {vals[1]-vals[2]:+.4f} |')
    for s in ('torchfdtd', 'meep', 'rcwa'):
        total = results[s]['T'].sum(axis=1) + results[s]['R'].sum(axis=1)
        lines.append(f'| sum of orders, {s} | {total[i_design]:.4f} at {wavelength[i_design]:.4g} um | band min {total.min():.4f} | band max {total.max():.4f} | | | |')
    return lines


def render_figure(path, wavelength, orders, results, i_design):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {-1: '#1f77b4', 0: '#2ca02c', 1: '#d62728'}
    styles = dict(torchfdtd=dict(ls='-', lw=1.8), meep=dict(ls='--', lw=1.4), rcwa=dict(ls=':', lw=1.8))
    labels = dict(torchfdtd='TorchFDTD', meep='Meep', rcwa='RCWA')
    for ax, kind in zip(axes[:2], ('T', 'R')):
        for j, m in enumerate(orders):
            for s in ('torchfdtd', 'meep', 'rcwa'):
                ax.plot(wavelength, results[s][kind][:, j], color=colors[m], label=f'{labels[s]} {kind}{m:+d}', **styles[s])
        ax.axvline(wavelength[i_design], color='0.6', lw=0.8)
        ax.set_xlabel('wavelength (um)')
        ax.set_ylabel(f'{kind} order efficiency')
        ax.set_title('transmitted orders' if kind == 'T' else 'reflected orders')
        ax.legend(fontsize=6, ncol=3)
    ax = axes[2]
    for j, m in enumerate(orders):
        for kind, ls in (('T', '-'), ('R', '--')):
            ax.plot(wavelength, results['torchfdtd'][kind][:, j] - results['meep'][kind][:, j], color=colors[m], ls=ls, label=f'{kind}{m:+d} TorchFDTD-Meep')
    for s, ls in (('torchfdtd', '-'), ('meep', '--')):
        total = results[s]['T'].sum(axis=1) + results[s]['R'].sum(axis=1)
        ax.plot(wavelength, total - 1, color='k', ls=ls, lw=0.9, label=f'{labels[s]} sum-1')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel('wavelength (um)')
    ax.set_ylabel('difference')
    ax.set_title('TorchFDTD minus Meep, energy residual')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(torchfdtd_record, meep_record, rcwa_record, geometry, criteria):
    orders = list(criteria['orders'])
    results = {}
    diagnostics = {}
    wl_t, T, R, diag = solver_efficiencies(torchfdtd_record, geometry, orders)
    results['torchfdtd'] = dict(T=T, R=R)
    diagnostics['torchfdtd'] = diag
    wl_m, T, R, diag = solver_efficiencies(meep_record, geometry, orders)
    results['meep'] = dict(T=T, R=R)
    diagnostics['meep'] = diag
    wl_r, T, R = rcwa_efficiencies(rcwa_record, orders)
    results['rcwa'] = dict(T=T, R=R)
    assert np.allclose(wl_t, wl_m, rtol=1e-9, atol=0) and np.allclose(wl_t, wl_r, rtol=1e-9, atol=0), 'wavelength samples differ'
    wavelength = wl_t
    assert len(wavelength) == criteria['band_points']
    i_design = int(np.argmin(abs(wavelength - criteria['design_wavelength_um'])))
    assert math.isclose(wavelength[i_design], criteria['design_wavelength_um'], rel_tol=1e-9)

    def stack(s):
        return np.concatenate([results[s]['T'], results[s]['R']], axis=1)
    metrics = {}
    for name, (a, b) in dict(torchfdtd_vs_meep_order_efficiency=('torchfdtd', 'meep'), torchfdtd_vs_rcwa_order_efficiency=('torchfdtd', 'rcwa'),
                             meep_vs_rcwa_order_efficiency=('meep', 'rcwa')).items():
        diff = abs(stack(a) - stack(b))
        limit = criteria['criteria'][name]['limit_abs']
        k = np.unravel_index(int(np.argmax(diff)), diff.shape)
        label = ('T' if k[1] < len(orders) else 'R') + f'{orders[k[1] % len(orders)]:+d}'
        metrics[name] = dict(value=float(diff.max()), limit_abs=limit, passed=bool(diff.max() <= limit), at_wavelength_um=float(wavelength[k[0]]),
                             at_order=label, at_design_wavelength=float(diff[i_design].max()))
    for s in ('torchfdtd', 'meep'):
        name = f'{s}_energy_balance'
        total = results[s]['T'].sum(axis=1) + results[s]['R'].sum(axis=1)
        limit = criteria['criteria'][name]['limit_abs']
        metrics[name] = dict(value=float(abs(total - 1).max()), limit_abs=limit, passed=bool(abs(total - 1).max() <= limit),
                             at_wavelength_um=float(wavelength[int(np.argmax(abs(total - 1)))]), at_design_wavelength=float(abs(total[i_design] - 1)))
    grid = grid_match(torchfdtd_record, meep_record, criteria)
    table = table_rows(wavelength, orders, results, i_design)
    timing = {s: {k: rec['timing'].get(k) for k in ('timing_mode', 'repeats', 'setup_seconds', 'stepping_seconds', 'full_seconds', 'host_load_note')}
              for s, rec in (('torchfdtd', torchfdtd_record), ('meep', meep_record))}
    timing['meep']['ranks'] = meep_record['grid'].get('mpi_processes')
    timing['torchfdtd']['device'] = torchfdtd_record['environment'].get('device')
    cells = int(np.prod(torchfdtd_record['grid']['cells']))
    timing['cells'] = cells
    timing['steps'] = torchfdtd_record['grid']['steps']
    comparison = dict(
        schema='torchfdtd-meep-comparison-result-v1', example='metagrating', geometry_sha256=geometry['_sha256'],
        criteria=criteria, wavelength_um=wavelength.tolist(), orders=orders, design_index=i_design,
        efficiencies={s: dict(T={str(m): results[s]['T'][:, j].tolist() for j, m in enumerate(orders)},
                              R={str(m): results[s]['R'][:, j].tolist() for j, m in enumerate(orders)},
                              total=(results[s]['T'].sum(axis=1) + results[s]['R'].sum(axis=1)).tolist()) for s in results},
        metrics=metrics, all_criteria_passed=bool(all(m['passed'] for m in metrics.values())), grid_match=grid,
        all_grid_checks_passed=bool(all(grid.values())), diagnostics=diagnostics,
        rcwa=dict(harmonics=rcwa_record.get('harmonics'), convergence=rcwa_record.get('convergence'), citation=rcwa_record.get('citation'),
                  package=rcwa_record.get('package')),
        timing=timing, table_markdown=table)
    return comparison, results, wavelength, i_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', default=str(RECORDS))
    parser.add_argument('--out', default=None)
    parser.add_argument('--figure', default=str(FIGURES / 'metagrating.png'))
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    records = Path(args.records)
    geometry_raw = (HERE / 'geometry.json').read_bytes()
    import hashlib
    geometry = json.loads(geometry_raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(geometry_raw).hexdigest()
    criteria = load_json(HERE / 'criteria.json')
    t = load_json(records / 'metagrating_torchfdtd.json')
    m = load_json(records / 'metagrating_meep.json')
    r = load_json(records / 'metagrating_rcwa.json')
    comparison, results, wavelength, i_design = compare(t, m, r, geometry, criteria)
    out = Path(args.out) if args.out else records / 'metagrating_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(comparison, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    if not args.no_figure:
        render_figure(Path(args.figure), wavelength, comparison['orders'], results, i_design)
    print('\n'.join(comparison['table_markdown']))
    for name, metric in comparison['metrics'].items():
        print(f"{name}: {metric['value']:.5f} (limit {metric['limit_abs']}) {'pass' if metric['passed'] else 'FAIL'} at {metric['at_wavelength_um']:.4g} um {metric.get('at_order', '')}")
    print('grid checks:', comparison['grid_match'])
    print('wrote', out)


if __name__ == '__main__':
    main()
