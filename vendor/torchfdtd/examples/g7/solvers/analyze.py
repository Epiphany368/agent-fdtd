"""G7-04 analysis: errors against the TORCWA reference, matched-error costs, tables and the figure. Reads records only.

    python examples/g7/solvers/analyze.py

Reads docs/validation/g7/G7-04/rcwa_reference.json and every point record written by
torchfdtd_sweep.py and meep_sweep.py, recomputes the efficiencies from the recorded Fourier means,
writes docs/validation/g7/G7-04/summary.json (tables, matched costs, acceptance) and renders
docs/figures/g7/G7-04.png from the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OBSERVABLES = ('t1_design', 't1_band_mean')
OBSERVABLE_LABELS = dict(t1_design='T+1 at 1.55 um', t1_band_mean='band-mean T+1')
CURVES = (('torchfdtd', 'staircase', 'float32'), ('torchfdtd', 'staircase', 'float64'), ('torchfdtd', 'smoothed', 'float32'),
          ('torchfdtd', 'smoothed', 'float64'), ('meep', 'staircase', 'float64'), ('meep', 'smoothed', 'float64'))
SOLVER_LABELS = dict(torchfdtd='TorchFDTD', meep='Meep')
HARDWARE_STATEMENT = ('TorchFDTD ran on a GPU (NVIDIA RTX 3060, fused CUDA kernels) and Meep 1.34 on a CPU (Intel i7-12700, 4 MPI ranks in WSL). '
                      'Wall times therefore compare two solver-and-hardware combinations on a shared workstation; no conclusion about the '
                      'algorithms is drawn from the cost ratio. The cell-count axis is hardware-independent.')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record_path(records, solver, series, precision, mesh):
    if solver == 'torchfdtd':
        return Path(records) / f'torchfdtd-{series}-{precision}-{mesh:g}.json'
    resolution = dict(zip(common.TORCHFDTD_MESHES_UM, common.MEEP_RESOLUTIONS))[mesh]
    return Path(records) / f'meep-{series}-float64-{resolution}.json'


def curve_key(solver, series, precision):
    return f'{solver}/{series}/{precision}'


def curve_label(solver, series, precision):
    return f'{SOLVER_LABELS[solver]} {series}' + (f' {precision}' if solver == 'torchfdtd' else '')


def point_summary(record, reference):
    """Efficiencies recomputed from the recorded Fourier means, errors against the reference and the cost of one point."""
    g = record['geometry']
    T, R = common.efficiencies_from_amplitudes(record['amplitudes'], g['substrate_index'])
    for j, m in enumerate(common.ORDERS):
        assert np.allclose(T[:, j], record['efficiencies']['T'][str(m)], rtol=0, atol=1e-12), (record['solver'], m)
        assert np.allclose(R[:, j], record['efficiencies']['R'][str(m)], rtol=0, atol=1e-12), (record['solver'], m)
    obs = common.observables(T, record['wavelength_um'])
    t = record['timing']
    full = [s['full_seconds'] for s in t['samples']]
    return dict(
        solver=record['solver'], series=record['series'], precision=record['precision'], mesh_um=record['mesh_um'],
        resolution_per_um=record['resolution_per_um'], cells=record['grid']['cells'], cell_count=record['grid']['cell_count'],
        steps=record['grid']['steps'], cell_steps=record['grid']['cell_count'] * record['grid']['steps'],
        shift_um=g['g7_04']['shift_um'], pml_cells=g['pml_cells'], dt_s=g['g7_04']['dt_s'], t1_design=obs['t1_design'], t1_band_mean=obs['t1_band_mean'],
        error_t1_design=abs(obs['t1_design'] - reference['t1_design']), error_t1_band_mean=abs(obs['t1_band_mean'] - reference['t1_band_mean']),
        samples_full_seconds=full, median_full_seconds=float(np.median(full)), min_full_seconds=float(min(full)), max_full_seconds=float(max(full)),
        median_stepping_seconds=float(np.median([s['stepping_seconds'] for s in t['samples']])), repeats=len(full),
        solver_seconds=float(sum(full) + t['warmup']['full_seconds'] + t['reference']['full_seconds']),
        warmup_full_seconds=t['warmup']['full_seconds'], reference_run_full_seconds=t['reference']['full_seconds'],
        host_cpu_percent_before_samples=[s.get('host_cpu_percent_before') for s in t['samples']],
        device=t['device'], energy_sum_design=float(T[obs['design_index']].sum() + R[obs['design_index']].sum()),
        ridge_rows=len(record['staircase']['silicon_rows']) if record.get('staircase') else None,
        repeat_max_abs_t1_difference=record['repeat_max_abs_t1_difference'])


def matched_costs(points):
    """Cost to reach every target, per observable, in wall time and in cell-steps (refinement order)."""
    out = {}
    for obs in OBSERVABLES:
        errors = [p['error_' + obs] for p in points]
        out[obs] = {}
        for target in common.TARGETS:
            wall = common.cost_to_reach([p['median_full_seconds'] for p in points], errors, target)
            work = common.cost_to_reach([p['cell_steps'] for p in points], errors, target)
            out[obs][f'{target:g}'] = dict(status=wall['status'], wall_seconds=wall['cost'], cell_steps=work['cost'], between_meshes_um=(
                None if wall['between'] is None else [points[i]['mesh_um'] for i in wall['between']]))
    return out


def fmt_cost(entry):
    if entry['status'] == 'not reached':
        return 'not reached'
    prefix = '<= ' if entry['status'] == 'coarsest point' else ''
    return f"{prefix}{entry['wall_seconds']:.3g} s / {prefix}{entry['cell_steps']:.3g}"


def points_table(curves):
    lines = ['| solver | series | precision | mesh (um) | resolution (1/um) | cells | steps | median time (s) | range (s) | T+1(1.55) | error | '
             'band-mean T+1 | error |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key, points in curves.items():
        for p in points:
            lines.append(f"| {SOLVER_LABELS[p['solver']]} | {p['series']} | {p['precision']} | {p['mesh_um']:g} | {p['resolution_per_um']} | "
                         f"{p['cells'][0]} x {p['cells'][1]} = {p['cell_count']:,} | {p['steps']:,} | {p['median_full_seconds']:.3f} | "
                         f"{p['min_full_seconds']:.3f}-{p['max_full_seconds']:.3f} | {p['t1_design']:.5f} | {p['error_t1_design']:.2e} | "
                         f"{p['t1_band_mean']:.5f} | {p['error_t1_band_mean']:.2e} |")
    return lines


def geometry_table(curves):
    """The derived fixture of every mesh, read back from the records (both series, both solvers agree)."""
    lines = ['| mesh (um) | Meep resolution (1/um) | cells | absorber cells per y face | steps | dt (s) | staircase shift (um) | '
             'Ez rows in the ridge, TorchFDTD / Meep (staircase) |', '|---|---|---|---|---|---|---|---|']
    for i, mesh in enumerate(common.TORCHFDTD_MESHES_UM):
        t = curves[curve_key('torchfdtd', 'staircase', 'float32')][i]
        m = curves[curve_key('meep', 'staircase', 'float64')][i]
        for key in ('cells', 'pml_cells', 'steps'):
            assert all(curves[k][i][key] == t[key] for k in curves), (mesh, key)
        assert m['shift_um'] == t['shift_um']
        lines.append(f"| {mesh:g} | {m['resolution_per_um']} | {t['cells'][0]} x {t['cells'][1]} | {t['pml_cells']} | {t['steps']:,} | "
                     f"{t['dt_s']:.6e} | {t['shift_um'][0]:g} | {t['ridge_rows']} / {m['ridge_rows']} |")
    return lines


def equal_cell_table(curves):
    """Same mesh, same cells and steps: errors and wall times side by side."""
    columns = ('TorchFDTD float32', 'TorchFDTD float64', 'Meep float64')
    lines = ['| mesh (um) | cells | steps | series | ' + ' | '.join(f'{c}: error 1.55 / error band / time (s)' for c in columns) + ' |',
             '|---|---|---|---|---|---|---|']
    for mesh_index, mesh in enumerate(common.TORCHFDTD_MESHES_UM):
        for series in common.SERIES:
            row = [c for c in CURVES if c[1] == series]
            cells = curves[curve_key(*row[0])][mesh_index]
            cols = []
            for c in row:
                p = curves[curve_key(*c)][mesh_index]
                cols.append(f"{p['error_t1_design']:.2e} / {p['error_t1_band_mean']:.2e} / {p['median_full_seconds']:.3f}")
            lines.append(f"| {mesh:g} | {cells['cell_count']:,} | {cells['steps']:,} | {series} | " + ' | '.join(cols) + ' |')
    return lines


def equal_error_table(costs):
    head = ' | '.join(f'error {t:g}' for t in common.TARGETS)
    lines = [f'| curve | observable | {head} |', '|---|---|' + '---|' * len(common.TARGETS)]
    for c in CURVES:
        for obs in OBSERVABLES:
            entries = costs[curve_key(*c)][obs]
            lines.append(f'| {curve_label(*c)} | {OBSERVABLE_LABELS[obs]} | ' + ' | '.join(fmt_cost(entries[f'{t:g}']) for t in common.TARGETS) + ' |')
    return lines


def reference_table(rcwa):
    ref, chk = rcwa['reference'], rcwa['check']
    other = rcwa['bands'][str(chk['against_order_count'])]
    return ['| TORCWA orders | T+1(1.55) | band-mean T+1 | role |', '|---|---|---|---|',
            f"| {ref['order_count']} | {ref['t1_design']:.6f} | {ref['t1_band_mean']:.6f} | reference |",
            f"| {chk['against_order_count']} | {other['t1_design']:.6f} | {other['t1_band_mean']:.6f} | check (case G7-04r2) |",
            f"| largest difference, six efficiencies x 41 wavelengths | {chk['max_abs_difference']:.1e} ({chk['at_efficiency']} at "
            f"{chk['at_wavelength_um']:.2f} um) | limit {chk['limit_abs']:.0e} | {'pass' if chk['passed'] else 'fail'} |"]


def analyze(records=common.RECORDS):
    rcwa = load_json(Path(records) / 'rcwa_reference.json')
    reference = rcwa['reference']
    curves = {}
    for solver, series, precision in CURVES:
        curves[curve_key(solver, series, precision)] = [point_summary(load_json(record_path(records, solver, series, precision, mesh)), reference)
                                                         for mesh in common.TORCHFDTD_MESHES_UM]
    costs = {key: matched_costs(points) for key, points in curves.items()}
    all_points = [p for points in curves.values() for p in points]
    acceptance = dict(
        records=dict(passed=all(p['repeats'] == 3 and p['min_full_seconds'] <= p['median_full_seconds'] <= p['max_full_seconds'] for p in all_points),
                     points=len(all_points), detail='every point has three timed runs after one warm-up; median and range recorded'),
        curves=dict(passed=True, figure=str(common.FIGURE.relative_to(common.ROOT).as_posix()),
                    detail='rendered by this script from the summary numbers; tests/test_g7_solvers.py re-derives them from the point records'),
        matched_cost=dict(passed=all(costs[k][o][f'{t:g}']['status'] in ('interpolated', 'coarsest point', 'not reached')
                                     for k in costs for o in OBSERVABLES for t in common.TARGETS),
                          detail='errors 0.01, 0.005, 0.002 for each solver, sampling series and TorchFDTD precision, both observables; '
                                 'log-log interpolation in refinement order, last crossing (common.cost_to_reach)'),
        separation=dict(passed=True, detail='equal-cell-count table (same mesh) and equal-error table (matched costs) are separate'),
        hardware=dict(passed=True, statement=HARDWARE_STATEMENT))
    summary = dict(
        schema='g7-04-summary-v1', case='G7-04', declaration='docs/G7_WORKFLOWS.md', case_file='docs/validation/cases/G7-04r2.json',
        reference=dict(source='rcwa_reference.json', order_count=reference['order_count'], t1_design=reference['t1_design'],
                       t1_band_mean=reference['t1_band_mean'], check=rcwa['check']),
        targets=list(common.TARGETS), cost_definition=('wall time: median of three full solves (setup and stepping) after one warm-up; '
                                                       'cell-steps: cells times steps of the point; matched costs by log-log interpolation '
                                                       'between the two meshes that bracket the last crossing of the target in refinement order; '
                                                       '"<=" marks a target already met at the coarsest mesh (its cost is an upper bound)'),
        curves=curves, matched_costs=costs, hardware_statement=HARDWARE_STATEMENT, acceptance=acceptance,
        solver_seconds={solver: float(sum(p['solver_seconds'] for p in all_points if p['solver'] == solver)) for solver in SOLVER_LABELS},
        reference_check_passed=bool(rcwa['check']['passed']),
        tables=dict(reference=reference_table(rcwa), geometry=geometry_table(curves), points=points_table(curves), equal_cell_count=equal_cell_table(curves),
                    equal_error=equal_error_table(costs)))
    return summary


def render_figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(torchfdtd='#2a78d6', meep='#eb6834')
    styles = dict(staircase='-', smoothed='--')
    markers = dict(torchfdtd='o', meep='s')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for row, (xkey, xlabel) in enumerate((('median_full_seconds', 'wall time of one solve, median of 3 (s)'), ('cell_count', 'cells'))):
        for col, obs in enumerate(OBSERVABLES):
            ax = axes[row, col]
            for solver, series, precision in CURVES:
                points = summary['curves'][curve_key(solver, series, precision)]
                x = np.array([p[xkey] for p in points], dtype=float)
                y = np.array([max(p['error_' + obs], 1e-7) for p in points])
                filled = precision == 'float32' or solver == 'meep'
                kw = dict(color=colors[solver], ls=styles[series], lw=1.6, marker=markers[solver], ms=6,
                          mfc=colors[solver] if filled else 'white', mec=colors[solver], label=curve_label(solver, series, precision))
                if xkey == 'median_full_seconds':
                    lo = x - np.array([p['min_full_seconds'] for p in points])
                    hi = np.array([p['max_full_seconds'] for p in points]) - x
                    ax.errorbar(x, y, xerr=[lo, hi], capsize=2, elinewidth=0.8, **kw)
                else:
                    ax.plot(x, y, **kw)
            for target in summary['targets']:
                ax.axhline(target, color='0.55', lw=0.8, ls=':')
                ax.text(1.0, target, f' {target:g}', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=7, color='0.35')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"|{OBSERVABLE_LABELS[obs]} - RCWA|")
            ax.grid(True, which='major', color='0.9', lw=0.6)
            ax.set_title(f"{OBSERVABLE_LABELS[obs]}: error against TORCWA {summary['reference']['order_count']} orders", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.text(0.5, 0.955, 'GPU (TorchFDTD, RTX 3060) against CPU (Meep 1.34, i7-12700, 4 MPI ranks): wall times compare solver-and-hardware '
                         'pairs, not algorithms', ha='center', fontsize=8.5, color='0.25')
    fig.suptitle('G7-04: two-ridge metagrating, accuracy against cost', fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.08, 0.97, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', default=str(common.RECORDS))
    parser.add_argument('--figure', default=str(common.FIGURE))
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    summary = analyze(args.records)
    out = Path(args.records) / 'summary.json'
    out.write_bytes((json.dumps(summary, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    if not args.no_figure:
        render_figure(summary, Path(args.figure))
    for name, table in summary['tables'].items():
        print(f'\n{name}')
        print('\n'.join(table))
    print('\nreference check passed:', summary['reference_check_passed'], summary['reference']['check']['max_abs_difference'])
    print('acceptance:', {k: v['passed'] for k, v in summary['acceptance'].items()})
    print('wrote', out)


if __name__ == '__main__':
    main()
