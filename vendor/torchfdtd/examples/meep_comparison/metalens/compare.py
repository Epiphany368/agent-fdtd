"""Compare the TorchFDTD and Meep metalens records: no simulation, records in, comparison JSON, tables and figure out.

    python examples/meep_comparison/metalens/compare.py
reads docs/validation/meep_comparison/metalens_{2d,3d}_{torchfdtd,meep}.json and criteria.json, writes
docs/validation/meep_comparison/metalens_comparison.json and docs/figures/meep_comparison/metalens.png, and
prints the Markdown tables that the README carries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metalens_common as mc  # noqa: E402

CRITERIA = HERE / 'criteria.json'
PARTS = ('2d', '3d')
SOLVERS = ('torchfdtd', 'meep')


def load_records(part):
    return {s: json.loads((mc.RECORDS / f'metalens_{part}_{s}.json').read_text(encoding='utf-8')) for s in SOLVERS}


def grid_agreement(rec):
    t, m = rec['torchfdtd']['grid'], rec['meep']['grid']
    dt_rel = abs(t['dt_s'] - m['dt_s']) / t['dt_s']
    checks = dict(cells_per_axis=t['shape'] == m['shape'], dt_relative_difference=dt_rel, dt=dt_rel <= 1e-9, steps=t['steps'] == m['steps'],
                  pml_cells=t['pml_cells'] == m['pml_cells'], geometry_sha256=rec['torchfdtd']['geometry_sha256'] == rec['meep']['geometry_sha256'])
    si_t, si_m = rec['torchfdtd']['silicon_cells']['per_component'], rec['meep']['silicon_cells']['per_component']
    checks['silicon_cells'] = all(si_t[c] == si_m[c] for c in si_m)
    checks['silicon_cells_per_component'] = dict(torchfdtd=si_t, meep=si_m)
    checks['source_support'] = rec['torchfdtd']['source']['support'] == rec['meep']['source']['support']
    checks['all'] = all(checks[k] for k in ('cells_per_axis', 'dt', 'steps', 'pml_cells', 'geometry_sha256', 'silicon_cells', 'source_support'))
    return checks


def rel(a, b):
    return abs(a - b) / (.5 * (abs(a) + abs(b)))


def metrics_at_centre(part, rec, wavelength=1.55):
    out = {}
    for s in SOLVERS:
        o = rec[s]['observables']
        k = o['wavelengths_um'].index(wavelength)
        if part == '2d':
            f = o['focal']
            out[s] = mc.metrics_2d(o['axis']['y_um'], np.asarray(o['axis']['intensity'])[k], f['x_um'], np.asarray(f['intensity'])[k],
                                   np.asarray(f['poynting_y'])[k], f['weight_m'], 0.)
            out[s]['profile'] = np.asarray(f['intensity'])[k]
        else:
            f = o['focal']
            assert f['stored_wavelength_um'] == wavelength
            out[s] = mc.metrics_3d(o['xz']['z_um'], np.asarray(o['xz']['axis_intensity'])[k], f['x_um'], f['y_um'], f['intensity'], f['poynting_z'],
                                   f['weight_m2'], 0.)
            out[s]['profile'] = np.asarray(f['intensity'])
        out[s]['transmission'] = o['incident']['transmission'][k]
    return out


def evaluate(part, rec, criteria):
    crit = criteria['parts'][part]
    m = metrics_at_centre(part, rec, criteria['evaluated_at_wavelength_um'])
    t, p = m['torchfdtd'], m['meep']
    peak = .5 * (np.max(t['profile']) + np.max(p['profile']))
    rows = []
    axis_key = 'axis_peak_y_um' if part == '2d' else 'axis_peak_z_um'
    rows.append(dict(name='focal_position', label='Focal position (um)', torchfdtd=t[axis_key], meep=p[axis_key], difference=abs(t[axis_key] - p[axis_key]),
                     unit='um', limit=crit['focal_position']['limit']))
    if part == '2d':
        rows.append(dict(name='fwhm', label='Focal-plane FWHM (um)', torchfdtd=t['fwhm_um'], meep=p['fwhm_um'], difference=rel(t['fwhm_um'], p['fwhm_um']),
                         unit='relative', limit=crit['fwhm']['limit']))
    else:
        for axis in ('x', 'y'):
            key = f'fwhm_{axis}_um'
            rows.append(dict(name=f'fwhm_{axis}', label=f'Focal-plane FWHM along {axis} (um)', torchfdtd=t[key], meep=p[key], difference=rel(t[key], p[key]),
                             unit='relative', limit=crit[f'fwhm_{axis}']['limit']))
    rms = float(np.sqrt(np.mean((t['profile'] - p['profile']) ** 2)) / peak)
    rows.append(dict(name='profile_rms', label='Focal-plane profile RMS difference / peak', torchfdtd=float(np.max(t['profile'])), meep=float(np.max(p['profile'])),
                     difference=rms, unit='relative to the peak', limit=crit['profile_rms']['limit'], note='the two columns are the peak intensities relative to the incident intensity'))
    rows.append(dict(name='efficiency', label='Focusing efficiency', torchfdtd=t['efficiency'], meep=p['efficiency'], difference=abs(t['efficiency'] - p['efficiency']),
                     unit='absolute', limit=crit['efficiency']['limit']))
    for row in rows:
        row['pass'] = bool(row['difference'] <= row['limit'])
        row['criterion'] = crit[row['name']]['metric']
    info = [dict(label='Transmission through the aperture plane', torchfdtd=t['transmission'], meep=p['transmission'], difference=abs(t['transmission'] - p['transmission'])),
            dict(label='On-axis peak intensity / incident', torchfdtd=t['axis_peak_intensity'], meep=p['axis_peak_intensity'],
                 difference=abs(t['axis_peak_intensity'] - p['axis_peak_intensity']))]
    return rows, info


def extra_wavelengths(part, rec, criteria):
    """Per-solver summary rows at the extra wavelengths, straight from the records (information, no criterion)."""
    rows = []
    for wl in criteria['extra_wavelengths_um']:
        entry = dict(wavelength_um=wl)
        for s in SOLVERS:
            row = next(r for r in rec[s]['observables']['summary'] if r['wavelength_um'] == wl)
            entry[s] = dict(focal_position_um=row['axis_peak_y_um' if part == '2d' else 'axis_peak_z_um'],
                            fwhm_um=row['fwhm_um'] if part == '2d' else [row['fwhm_x_um'], row['fwhm_y_um']], efficiency=row['efficiency'], transmission=row['transmission'])
        rows.append(entry)
    return rows


def summary_consistency(part, rec, criteria):
    """The scripts' own summary rows at the centre wavelength against the recomputation from the stored (7-digit) arrays."""
    wl = criteria['evaluated_at_wavelength_um']
    m = metrics_at_centre(part, rec, wl)
    out = {}
    for s in SOLVERS:
        row = next(r for r in rec[s]['observables']['summary'] if r['wavelength_um'] == wl)
        keys = ('axis_peak_y_um', 'fwhm_um', 'efficiency') if part == '2d' else ('axis_peak_z_um', 'fwhm_x_um', 'fwhm_y_um', 'efficiency')
        out[s] = {k: dict(script=row[k], recomputed=m[s][k], relative_difference=rel(row[k], m[s][k])) for k in keys}
    return out


def timing(rec):
    out = {}
    for s in SOLVERS:
        t = rec[s]['timing']
        g = rec[s]['grid']
        out[s] = dict(timing_mode=t['timing_mode'], setup_seconds=t['setup_seconds'], stepping_seconds=t['stepping_seconds'], full_seconds=t['full_seconds'],
                      samples=len(t['samples']), host_load_note=t['host_load_note']['note'], load_average_1_5_15=t['host_load_note']['load_average_1_5_15'],
                      gpu_at_start=t['host_load_note'].get('gpu'), hardware=(g.get('device') if s == 'torchfdtd' else f"{g['mpi_processes']} MPI ranks, {rec[s]['environment']['cpu']}"),
                      precision=g['precision'], cells=g['cells'], steps=g['steps'])
    out['stepping_ratio_meep_over_torchfdtd'] = out['meep']['stepping_seconds'] / out['torchfdtd']['stepping_seconds']
    out['full_ratio_meep_over_torchfdtd'] = out['meep']['full_seconds'] / out['torchfdtd']['full_seconds']
    return out


def fmt(v, unit=None):
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if unit == 'um':
        return f'{v:.4f}'
    return f'{v:.4f}' if abs(v) >= 1e-3 or v == 0 else f'{v:.2e}'


def render_tables(result):
    lines = []
    for part in PARTS:
        r = result[part]
        title = '2D ridge lens (Part A)' if part == '2d' else '3D pillar lens (Part B)'
        lines.append(f'### {title}: agreement at 1.55 um')
        lines.append('')
        lines.append('| Metric | TorchFDTD | Meep | Difference | Criterion | Pass |')
        lines.append('|---|---|---|---|---|---|')
        for row in r['criteria_rows']:
            unit = {'um': ' um', 'relative': ' (relative)', 'relative to the peak': ' of the peak', 'absolute': ''}[row['unit']]
            lines.append(f"| {row['label']} | {fmt(row['torchfdtd'])} | {fmt(row['meep'])} | {fmt(row['difference'])}{unit} | <= {row['limit']}{unit} | {fmt(row['pass'])} |")
        for row in r['information_rows']:
            lines.append(f"| {row['label']} | {fmt(row['torchfdtd'])} | {fmt(row['meep'])} | {fmt(row['difference'])} | (information) | - |")
        lines.append('')
        lines.append(f'{title}: extra wavelengths (information, from the records)')
        lines.append('')
        head = 'FWHM (um)' if part == '2d' else 'FWHM x, y (um)'
        lines.append(f'| Wavelength (um) | Focal position TorchFDTD / Meep (um) | {head} TorchFDTD / Meep | Efficiency TorchFDTD / Meep | Transmission TorchFDTD / Meep |')
        lines.append('|---|---|---|---|---|')
        for e in r['extra_wavelengths']:
            t, m = e['torchfdtd'], e['meep']
            f = (f"{t['fwhm_um']:.4f} / {m['fwhm_um']:.4f}" if part == '2d' else
                 f"{t['fwhm_um'][0]:.4f}, {t['fwhm_um'][1]:.4f} / {m['fwhm_um'][0]:.4f}, {m['fwhm_um'][1]:.4f}")
            lines.append(f"| {e['wavelength_um']:.2f} | {t['focal_position_um']:.4f} / {m['focal_position_um']:.4f} | {f} | {t['efficiency']:.4f} / {m['efficiency']:.4f} | "
                         f"{t['transmission']:.4f} / {m['transmission']:.4f} |")
        lines.append('')
        lines.append(f'{title}: wall time of the lens run')
        lines.append('')
        lines.append('| Solver | Hardware | Precision | Cells | Steps | Setup (s) | Stepping (s) | Full (s) | Timing mode |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for s in SOLVERS:
            t = r['timing'][s]
            lines.append(f"| {'TorchFDTD' if s == 'torchfdtd' else 'Meep'} | {t['hardware']} | {t['precision']} | {t['cells']} | {t['steps']} | {t['setup_seconds']:.2f} | "
                         f"{t['stepping_seconds']:.2f} | {t['full_seconds']:.2f} | {t['timing_mode']} |")
        lines.append('')
        lines.append(f"Stepping ratio Meep / TorchFDTD: {r['timing']['stepping_ratio_meep_over_torchfdtd']:.1f}; full-run ratio: {r['timing']['full_ratio_meep_over_torchfdtd']:.1f}. "
                     f"Load condition: {r['timing']['torchfdtd']['host_load_note']}")
        lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def compare_all(criteria_path=CRITERIA):
    criteria = json.loads(Path(criteria_path).read_text(encoding='utf-8'))
    result = dict(schema='torchfdtd-meep-comparison-result-v1', example='metalens', criteria_file=Path(criteria_path).name, criteria_sha256=mc.sha256_file(criteria_path),
                  evaluated_at_wavelength_um=criteria['evaluated_at_wavelength_um'])
    records = {}
    for part in PARTS:
        rec = load_records(part)
        records[part] = rec
        rows, info = evaluate(part, rec, criteria)
        result[part] = dict(records={s: f'metalens_{part}_{s}.json' for s in SOLVERS}, geometry_sha256=rec['torchfdtd']['geometry_sha256'],
                            grid_agreement=grid_agreement(rec), criteria_rows=rows, information_rows=info, all_pass=all(r['pass'] for r in rows),
                            extra_wavelengths=extra_wavelengths(part, rec, criteria), summary_consistency=summary_consistency(part, rec, criteria), timing=timing(rec))
    result['all_pass'] = all(result[p]['all_pass'] and result[p]['grid_agreement']['all'] for p in PARTS)
    result['tables_markdown'] = render_tables(result)
    return result, records


def render_figure(result, records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(torchfdtd='#1f77b4', meep='#d62728')
    labels = dict(torchfdtd='TorchFDTD (GPU, float32)', meep='Meep (CPU, float64)')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    # Part A
    rec = records['2d']
    wl = result['evaluated_at_wavelength_um']
    ax = axes[0, 0]
    for s in SOLVERS:
        o = rec[s]['observables']
        k = o['wavelengths_um'].index(wl)
        ax.plot(o['axis']['y_um'], np.asarray(o['axis']['intensity'])[k], color=colors[s], lw=1.2, ls='-' if s == 'torchfdtd' else '--', label=labels[s])
    ax.set_xlabel('y (um)')
    ax.set_ylabel('on-axis |Ez|^2 / incident')
    ax.set_title('2D ridge lens: on-axis intensity, 1.55 um')
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[0, 1]
    prof = {}
    for s in SOLVERS:
        o = rec[s]['observables']
        k = o['wavelengths_um'].index(wl)
        prof[s] = (np.asarray(o['focal']['x_um']), np.asarray(o['focal']['intensity'])[k])
        ax.plot(*prof[s], color=colors[s], lw=1.2, ls='-' if s == 'torchfdtd' else '--', label=labels[s])
    ax.set_xlim(-4, 4)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('|Ez|^2 / incident')
    ax.set_title(f"2D: focal plane y = {rec['torchfdtd']['observables']['focal']['y_um']} um")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    peak = .5 * (prof['torchfdtd'][1].max() + prof['meep'][1].max())
    ax.plot(prof['torchfdtd'][0], (prof['meep'][1] - prof['torchfdtd'][1]) / peak, color='k', lw=1)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('(Meep - TorchFDTD) / peak')
    rms = next(r for r in result['2d']['criteria_rows'] if r['name'] == 'profile_rms')
    ax.set_title(f"2D: focal-plane residual, RMS {rms['difference']:.2e} of the peak")
    # Part B
    rec = records['3d']
    ax = axes[1, 0]
    for s in SOLVERS:
        o = rec[s]['observables']
        k = o['wavelengths_um'].index(wl)
        ax.plot(o['xz']['z_um'], np.asarray(o['xz']['axis_intensity'])[k], color=colors[s], lw=1.2, ls='-' if s == 'torchfdtd' else '--', label=labels[s])
    ax.set_xlabel('z (um)')
    ax.set_ylabel('on-axis |E|^2 / incident')
    ax.set_title('3D pillar lens: on-axis intensity, 1.55 um')
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1, 1]
    maps = {}
    for s in SOLVERS:
        f = rec[s]['observables']['focal']
        nx, ny = f['shape'][:2]
        maps[s] = (np.asarray(f['x_um']), np.asarray(f['y_um']), np.asarray(f['intensity']).reshape(nx, ny))
        x, y, grid = maps[s]
        ix, iy = np.unravel_index(int(np.argmax(grid)), grid.shape)
        ax.plot(x, grid[:, iy], color=colors[s], lw=1.2, ls='-' if s == 'torchfdtd' else '--', label=f'{labels[s]}, along x')
        ax.plot(y, grid[ix, :], color=colors[s], lw=1.2, ls=':' if s == 'torchfdtd' else '-.', label=f'{labels[s]}, along y')
    ax.set_xlim(-3, 3)
    ax.set_xlabel('x or y (um)')
    ax.set_ylabel('|E|^2 / incident')
    ax.set_title(f"3D: focal plane z = {rec['torchfdtd']['observables']['focal']['z_um']} um, cuts through the peak")
    ax.legend(fontsize=7)
    ax = axes[1, 2]
    x, y, gt = maps['torchfdtd']
    gm = maps['meep'][2]
    peak = .5 * (gt.max() + gm.max())
    resid = (gm - gt) / peak
    vmax = float(np.max(np.abs(resid)))
    im = ax.imshow(resid.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label='(Meep - TorchFDTD) / peak')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    rms = next(r for r in result['3d']['criteria_rows'] if r['name'] == 'profile_rms')
    ax.set_title(f"3D: focal-plane residual, RMS {rms['difference']:.2e} of the peak")
    fig.suptitle('TorchFDTD versus Meep: metalens focus (records in docs/validation/meep_comparison)', fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=None, help='comparison JSON path')
    parser.add_argument('--figure', default=None, help='figure path')
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    result, records = compare_all()
    out = Path(args.out) if args.out else mc.RECORDS / 'metalens_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(result, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    if not args.no_figure:
        render_figure(result, records, Path(args.figure) if args.figure else mc.FIGURES / 'metalens.png')
    print(result['tables_markdown'])
    print('all criteria pass:', result['all_pass'])
    print('wrote', out)


if __name__ == '__main__':
    main()
