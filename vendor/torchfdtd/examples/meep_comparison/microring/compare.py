"""Compare the TorchFDTD and Meep microring records: no simulation, records in, metrics and figure out.

    python examples/meep_comparison/microring/compare.py

Reads docs/validation/meep_comparison/microring_torchfdtd.json and microring_meep.json, evaluates the
metrics declared in criteria.json, writes microring_comparison.json next to the records, prints the
README table and renders docs/figures/meep_comparison/microring.png from the records.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECORDS = REPO / 'docs' / 'validation' / 'meep_comparison'
FIGURE = REPO / 'docs' / 'figures' / 'meep_comparison' / 'microring.png'
SOLVERS = ('torchfdtd', 'meep')


def load_records(directory=RECORDS):
    return {solver: json.loads((Path(directory) / f'microring_{solver}.json').read_text(encoding='utf-8')) for solver in SOLVERS}


def lorentzian(wavelength, center, fwhm, depth, base):
    return base - depth / (1 + (2 * (wavelength - center) / fwhm)**2)


def parabolic_minimum_nm(wavelength, T, index):
    """Vertex of the parabola through the sample minimum and its two neighbours."""
    x = wavelength[index - 1:index + 2] * 1e3
    y = T[index - 1:index + 2]
    a, b, _ = np.polyfit(x - x[1], y, 2)
    return float(x[1] - b / (2 * a)) if a > 0 else float(x[1])


def fit_resonance(wavelength, T, index, half_window_um=0.004):
    """Lorentzian fit around one detected minimum. A minimum without Lorentzian width (a truncation sidelobe) is kept with valid=False."""
    sel = (wavelength > wavelength[index] - half_window_um) & (wavelength < wavelength[index] + half_window_um)
    spacing = float(np.mean(np.diff(wavelength)))
    out = dict(minimum_sample_nm=float(wavelength[index] * 1e3), minimum_T=float(T[index]), fit_points=int(sel.sum()),
               center_nm=parabolic_minimum_nm(wavelength, T, index), fwhm_nm=None, depth=None, base=None, q_loaded=None,
               extinction_db=None, fit_rms=None, valid=False)
    try:
        params, _ = curve_fit(lorentzian, wavelength[sel], T[sel], p0=[wavelength[index], 0.0015, 1 - T[index], 1.0], maxfev=20000)
    except RuntimeError:
        return out
    center, fwhm, depth, base = params
    fwhm = abs(fwhm)
    valid = fwhm >= 2 * spacing and 0 < depth < base and abs(center - wavelength[index]) < half_window_um
    if not valid:
        return out
    residual = float(np.sqrt(np.mean((lorentzian(wavelength[sel], *params) - T[sel])**2)))
    out.update(center_nm=float(center * 1e3), fwhm_nm=float(fwhm * 1e3), depth=float(depth), base=float(base), q_loaded=float(center / fwhm),
               extinction_db=float(10 * math.log10(base / (base - depth))), fit_rms=residual, valid=True)
    return out


def resonances(wavelength, T, criteria):
    """Local minima of T below the declared level with the declared prominence, each refined by a Lorentzian fit."""
    level, prominence = criteria['detection']['max_T'], criteria['detection']['prominence']
    minima, _ = find_peaks(-T, height=-level, prominence=prominence)
    return [fit_resonance(wavelength, T, int(i)) for i in minima]


def timing_row(record):
    t = record['timing']
    return dict(mode=t['timing_mode'], repeats=t['repeats'], setup_seconds=t['setup_seconds'], stepping_seconds=t['stepping_seconds'],
                full_seconds=t['full_seconds'], straight_run_stepping_seconds=t['straight_run']['stepping_seconds'],
                straight_run_full_seconds=t['straight_run']['full_seconds'], host_load_note=t['host_load_note'],
                samples=[s['stepping_seconds'] for s in t['samples']])


def compute(records, criteria):
    tf, me = records['torchfdtd'], records['meep']
    wavelength = np.asarray(tf['wavelength_um'])
    assert np.allclose(wavelength, me['wavelength_um'], rtol=0, atol=1e-12)
    T = {s: np.asarray(records[s]['T']) for s in SOLVERS}
    found = {s: resonances(wavelength, T[s], criteria) for s in SOLVERS}
    valid = {s: [r for r in found[s] if r['valid']] for s in SOLVERS}
    nearest = {s: min(valid[s], key=lambda r: abs(r['center_nm'] - 1550)) for s in SOLVERS}
    matched = len(found['torchfdtd']) == len(found['meep']) and [r['valid'] for r in found['torchfdtd']] == [r['valid'] for r in found['meep']]
    centers = {s: [r['center_nm'] for r in found[s]] for s in SOLVERS}
    max_center_diff = float(max(abs(a - b) for a, b in zip(centers['torchfdtd'], centers['meep']))) if matched else None
    fsr = {s: float(np.mean(np.diff([r['center_nm'] for r in valid[s]]))) if len(valid[s]) > 1 else None for s in SOLVERS}
    rms = float(np.sqrt(np.mean((T['torchfdtd'] - T['meep'])**2)))
    q_rel = abs(nearest['torchfdtd']['q_loaded'] - nearest['meep']['q_loaded']) / nearest['meep']['q_loaded']
    er_diff = abs(nearest['torchfdtd']['extinction_db'] - nearest['meep']['extinction_db'])
    limits = criteria['criteria']
    grid_keys = ('cells', 'mesh_um', 'dt_s', 'steps', 'pml_cells')
    grid_equal = (tf['grid']['cells'] == me['grid']['cells'] and tf['grid']['steps'] == me['grid']['steps']
                  and tf['grid']['pml_cells'] == me['grid']['pml_cells']
                  and math.isclose(tf['grid']['mesh_um'], me['grid']['mesh_um'], rel_tol=1e-9)
                  and math.isclose(tf['grid']['dt_s'], me['grid']['dt_s'], rel_tol=1e-9)
                  and tf['geometry_sha256'] == me['geometry_sha256']
                  and tf['staircase']['interior_ez_epsilon_sha256'] == me['staircase']['interior_ez_epsilon_sha256'])
    results = dict(
        resonance_wavelength_nm=dict(value=max_center_diff, limit=limits['resonance_wavelength_nm']['limit'],
                                     passed=bool(matched and max_center_diff <= limits['resonance_wavelength_nm']['limit']),
                                     count={s: len(found[s]) for s in SOLVERS}, valid_fits={s: len(valid[s]) for s in SOLVERS}),
        q_relative=dict(value=float(q_rel), limit=limits['q_relative']['limit'], passed=bool(q_rel <= limits['q_relative']['limit'])),
        extinction_db=dict(value=float(er_diff), limit=limits['extinction_db']['limit'], passed=bool(er_diff <= limits['extinction_db']['limit'])),
        rms_T=dict(value=rms, limit=limits['rms_T']['limit'], passed=bool(rms <= limits['rms_T']['limit'])),
        grid_match=dict(value=grid_equal, limit='equal', passed=bool(grid_equal)))
    return dict(
        schema='torchfdtd-meep-comparison-v1', example='microring',
        records={s: dict(date=records[s]['date'], geometry_sha256=records[s]['geometry_sha256'], script_sha256=records[s]['script_sha256'])
                 for s in SOLVERS},
        grid={k: tf['grid'][k] for k in grid_keys} | dict(run_time_ps=tf['grid']['run_time_ps'], courant_number=tf['grid']['courant_number'],
                                                          cells_total=int(np.prod(tf['grid']['cells']))),
        staircase={s: dict(interior_ez_epsilon_sha256=records[s]['staircase']['interior_ez_epsilon_sha256'],
                           interior_core_nodes=records[s]['staircase']['interior_core_nodes']) for s in SOLVERS},
        resonances=found, nearest_1550=nearest, fsr_nm=fsr,
        T_stats={s: dict(min=float(T[s].min()), max=float(T[s].max())) for s in SOLVERS},
        max_abs_T_difference=float(np.max(abs(T['torchfdtd'] - T['meep']))),
        criteria=results, all_passed=bool(all(r['passed'] for r in results.values())),
        timing={s: timing_row(records[s]) for s in SOLVERS},
        environment=dict(torchfdtd=dict(gpu=tf['environment']['gpu']['name'] if tf['environment'].get('gpu') else None, torch=tf['environment']['torch'],
                                        precision=tf['environment']['precision']),
                         meep=dict(version=me['environment']['meep'], mpi_processes=me['environment']['mpi_processes'], cpu=me['environment']['cpu'],
                                   precision=me['environment']['precision'])))


def fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def render_table(c):
    """The README tables, rendered from the comparison dict only."""
    tf, me = c['nearest_1550']['torchfdtd'], c['nearest_1550']['meep']
    cr = c['criteria']
    lines = ['| Quantity | TorchFDTD | Meep | Difference | Criterion | Result |', '|---|---|---|---|---|---|']
    lines.append(f"| Resonances found in 1.50-1.60 um | {cr['resonance_wavelength_nm']['count']['torchfdtd']} | {cr['resonance_wavelength_nm']['count']['meep']} | "
                 f"same count | same count | {'pass' if cr['resonance_wavelength_nm']['count']['torchfdtd'] == cr['resonance_wavelength_nm']['count']['meep'] else 'fail'} |")
    for k, (a, b) in enumerate(zip(c['resonances']['torchfdtd'], c['resonances']['meep'])):
        kind = 'Lorentzian centre' if a['valid'] and b['valid'] else 'parabolic minimum, no Lorentzian width (truncation sidelobe)'
        lines.append(f"| Minimum {k + 1} (nm), {kind} | {a['center_nm']:.3f} | {b['center_nm']:.3f} | {abs(a['center_nm'] - b['center_nm']):.4f} | "
                     f"<= {cr['resonance_wavelength_nm']['limit']} | {'pass' if abs(a['center_nm'] - b['center_nm']) <= cr['resonance_wavelength_nm']['limit'] else 'fail'} |")
    lines.append(f"| Free spectral range (nm), minima with a valid fit | {fmt(c['fsr_nm']['torchfdtd'], 2)} | {fmt(c['fsr_nm']['meep'], 2)} | "
                 f"{fmt(None if None in c['fsr_nm'].values() else abs(c['fsr_nm']['torchfdtd'] - c['fsr_nm']['meep']), 4)} | reported | - |")
    lines.append(f"| Nearest-1.55 um resonance centre (nm) | {tf['center_nm']:.3f} | {me['center_nm']:.3f} | {abs(tf['center_nm'] - me['center_nm']):.4f} | "
                 f"<= {cr['resonance_wavelength_nm']['limit']} | {'pass' if abs(tf['center_nm'] - me['center_nm']) <= cr['resonance_wavelength_nm']['limit'] else 'fail'} |")
    lines.append(f"| Loaded Q (Lorentzian fit) | {tf['q_loaded']:.1f} | {me['q_loaded']:.1f} | {100 * cr['q_relative']['value']:.3f} % | "
                 f"<= {100 * cr['q_relative']['limit']:.0f} % | {'pass' if cr['q_relative']['passed'] else 'fail'} |")
    lines.append(f"| Extinction ratio (dB) | {tf['extinction_db']:.3f} | {me['extinction_db']:.3f} | {cr['extinction_db']['value']:.4f} | "
                 f"<= {cr['extinction_db']['limit']:.1f} | {'pass' if cr['extinction_db']['passed'] else 'fail'} |")
    lines.append(f"| Full width at half depth (nm) | {tf['fwhm_nm']:.3f} | {me['fwhm_nm']:.3f} | {abs(tf['fwhm_nm'] - me['fwhm_nm']):.4f} | reported | - |")
    lines.append(f"| Fit depth | {tf['depth']:.3f} | {me['depth']:.3f} | {abs(tf['depth'] - me['depth']):.4f} | reported | - |")
    lines.append(f"| Fit residual, RMS of T over the +-4 nm fit window | {tf['fit_rms']:.4f} | {me['fit_rms']:.4f} | {abs(tf['fit_rms'] - me['fit_rms']):.2e} | reported | - |")
    lines.append(f"| RMS of T_torchfdtd - T_meep over 401 samples | - | - | {cr['rms_T']['value']:.2e} | <= {cr['rms_T']['limit']} | {'pass' if cr['rms_T']['passed'] else 'fail'} |")
    lines.append(f"| Max abs T difference | - | - | {c['max_abs_T_difference']:.2e} | reported | - |")
    lines.append(f"| Grid, dt, steps, PML, geometry and staircase hash | {c['staircase']['torchfdtd']['interior_ez_epsilon_sha256'][:12]} | "
                 f"{c['staircase']['meep']['interior_ez_epsilon_sha256'][:12]} | - | equal | {'pass' if cr['grid_match']['passed'] else 'fail'} |")
    lines += ['', '| Lorentzian fits (minima with a valid fit) | Centre TorchFDTD (nm) | Centre Meep (nm) | Q TorchFDTD | Q Meep | ER TorchFDTD (dB) | ER Meep (dB) | FWHM TorchFDTD (nm) | FWHM Meep (nm) | Fit RMS TorchFDTD | Fit RMS Meep |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, (a, b) in enumerate(zip(c['resonances']['torchfdtd'], c['resonances']['meep'])):
        if a['valid'] and b['valid']:
            lines.append(f"| Minimum {k + 1} | {a['center_nm']:.3f} | {b['center_nm']:.3f} | {a['q_loaded']:.1f} | {b['q_loaded']:.1f} | "
                         f"{a['extinction_db']:.3f} | {b['extinction_db']:.3f} | {a['fwhm_nm']:.3f} | {b['fwhm_nm']:.3f} | {a['fit_rms']:.4f} | {b['fit_rms']:.4f} |")
    t = c['timing']
    lines += ['', '| Timing (ring run) | TorchFDTD (RTX 3060, float32) | Meep (CPU, float64, ' + str(c['environment']['meep']['mpi_processes']) + ' ranks) |',
              '|---|---|---|',
              f"| Cells x steps | {c['grid']['cells'][0]} x {c['grid']['cells'][1]} = {c['grid']['cells_total']} cells, {c['grid']['steps']} steps | same |",
              f"| Stepping (s) | {t['torchfdtd']['stepping_seconds']:.2f} | {t['meep']['stepping_seconds']:.2f} |",
              f"| Full solve (s) | {t['torchfdtd']['full_seconds']:.2f} | {t['meep']['full_seconds']:.2f} |",
              f"| Straight-bus run, stepping (s) | {t['torchfdtd']['straight_run_stepping_seconds']:.2f} | {t['meep']['straight_run_stepping_seconds']:.2f} |",
              f"| Timing mode | {t['torchfdtd']['mode']} ({t['torchfdtd']['repeats']} sample) | {t['meep']['mode']} ({t['meep']['repeats']} sample) |",
              f"| Load note | {t['torchfdtd']['host_load_note']} | {t['meep']['host_load_note']} |"]
    return '\n'.join(lines) + '\n'


def render_figure(records, comparison, path=FIGURE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wavelength = np.asarray(records['torchfdtd']['wavelength_um'])
    T = {s: np.asarray(records[s]['T']) for s in SOLVERS}
    near = comparison['nearest_1550']
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), gridspec_kw=dict(height_ratios=[3, 2, 1.5]))
    ax = axes[0]
    ax.plot(wavelength, T['torchfdtd'], color='#1f77b4', lw=1.6, label='TorchFDTD (GPU, float32)')
    ax.plot(wavelength, T['meep'], color='#d62728', lw=1.0, ls='--', label='Meep 1.34 (CPU, float64)')
    for r in comparison['resonances']['torchfdtd']:
        if r['valid']:
            ax.axvline(r['center_nm'] * 1e-3, color='0.7', lw=0.6, zorder=0)
    ax.set_ylabel('T (through port)')
    ax.set_xlim(wavelength[0], wavelength[-1])
    ax.set_title('Microring through-port transmission, %d x %d cells, %d steps, %.1f ps' % (*comparison['grid']['cells'][:2], comparison['grid']['steps'],
                                                                                          comparison['grid']['run_time_ps']), fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax = axes[1]
    c = near['torchfdtd']['center_nm'] * 1e-3
    sel = (wavelength > c - 0.006) & (wavelength < c + 0.006)
    ax.plot(wavelength[sel] * 1e3, T['torchfdtd'][sel], 'o', ms=3, color='#1f77b4', label='TorchFDTD samples')
    ax.plot(wavelength[sel] * 1e3, T['meep'][sel], 'x', ms=4, color='#d62728', label='Meep samples')
    fine = np.linspace(wavelength[sel][0], wavelength[sel][-1], 400)
    for s, color in (('torchfdtd', '#1f77b4'), ('meep', '#d62728')):
        r = near[s]
        ax.plot(fine * 1e3, lorentzian(fine, r['center_nm'] * 1e-3, r['fwhm_nm'] * 1e-3, r['depth'], r['base']), color=color, lw=0.8,
                label=f"{s} fit: Q = {r['q_loaded']:.0f}, ER = {r['extinction_db']:.2f} dB")
    ax.set_ylabel('T near 1.55 um')
    ax.set_xlabel('wavelength (nm)')
    ax.legend(fontsize=7, loc='lower left')
    ax = axes[2]
    ax.plot(wavelength, T['torchfdtd'] - T['meep'], color='0.2', lw=0.8)
    ax.set_ylabel('T_torchfdtd - T_meep')
    ax.set_xlabel('wavelength (um)')
    ax.set_xlim(wavelength[0], wavelength[-1])
    ax.text(0.01, 0.9, f"rms {comparison['criteria']['rms_T']['value']:.1e}, max {comparison['max_abs_T_difference']:.1e}", transform=ax.transAxes, fontsize=8, va='top')
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--records', default=str(RECORDS))
    parser.add_argument('--out', default=str(RECORDS / 'microring_comparison.json'))
    parser.add_argument('--figure', default=str(FIGURE))
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    criteria = json.loads((HERE / 'criteria.json').read_text(encoding='utf-8'))
    records = load_records(args.records)
    comparison = compute(records, criteria)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(comparison, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(render_table(comparison))
    print('wrote', out)
    if not args.no_figure:
        print('wrote', render_figure(records, comparison, args.figure))


if __name__ == '__main__':
    main()
