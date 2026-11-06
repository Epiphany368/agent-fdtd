#!/usr/bin/env python3
"""F8: report any archived run declaring the common F1-F6 readout contract.

Read-only; no solver imports, simulations, ledger writes, or acceptance grants.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verification import verify_readout
from invariants import evaluate_invariants, verify_invariants
from analytic_cases import evaluate_analytic_cases, label_analytic_cases
from convergence import apply_coincidence_guard, earned_floor, evaluate_ladders
from units_ledger import evaluate_ledger


def render_report(run):
    run = Path(run)
    readout = json.loads((run / 'readout.json').read_text())
    config_path = run / 'config.input.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    headlines = verify_invariants(readout, config)
    headlines = label_analytic_cases(readout, config, headlines)
    checks = evaluate_invariants(readout, config)
    analytic = evaluate_analytic_cases(readout, config)
    ladders = evaluate_ladders(readout, config)
    headlines = apply_coincidence_guard(headlines, ladders)
    ledger = evaluate_ledger(readout, config)
    lines = [f'Run: {run.name}', f'Policy: {readout.get("policy", "undeclared")}',
             f'Readout status: {readout.get("status", "undeclared")}',
             f'Fixture: {readout.get("fixture", "undeclared")}',
             'Acceptance: not granted by this read-only report.',
             'Stages (declared implementation imports):']
    for stage in readout['stages']:
        lines.append(f'  {stage}: {readout["stage_source"][stage]}')
    lines.extend(['  F7-verify: verification.verify_readout',
                  '  F7-invariants: invariants.verify_invariants (legacy decisions plus registered V2)',
                  '  F7-analytic: analytic_cases.label_analytic_cases (registered V1)',
                  '  F7-convergence: convergence.evaluate_ladders (registered V3, with the coincidence guard)',
                  '  F7-ledger: units_ledger.evaluate_ledger (registered V5)',
                  '  F8-report: tools/report.py.render_report',
                  'V-0: verified requires a registered analytic, invariant, or independent',
                  'cross-code physics check (V1/V2/V4). Registered V1 and V2 checks are recomputed below.',
                  'verified V1-* means the point passed a closed-form law whose scope the leg matched;',
                  'it establishes that case, not quantitative accuracy elsewhere.',
                  'consistent means historical solver reproduction; unchecked means no passed check.',
                  'verified denotes only the named invariant property; it does not establish quantitative accuracy.',
                  'All numeric metadata is included. SHA-256 is an identity check, not a number.',
                  'Units: dt_s, elapsed_seconds and *_elapsed_seconds are seconds; shape counts grid cells;',
                  'steps counts time updates; edge_count counts geometry edges;',
                  'T, R, J, launch_neff and launch_scores are dimensionless;',
                  'wavelength_um is micrometres (one millionth of a metre); frequency_hz is cycles per second;',
                  'energy_residual, empty_reflection_floor and sourcepower_max_relative_error are dimensionless;',
                  '*_reduced denotes solver reduced units, not watts. Unknown fields have undeclared units.',
                  'T = transmitted power fraction; R = reflected power fraction (modal where specified by the fixture);',
                  'J = upward/incident power ratio; P_up_reduced = upward-coupled power;',
                  'P_in_reduced = launched incident power; unit_mode_flux_reduced = unit detector-mode flux;',
                  'launch_neff = effective refractive index; launch_scores = mode-selection scores;',
                  'epsilon_sha256 = SHA-256 fingerprint of the built material-permittivity tensor.',
                  'Registered invariant checks (recomputed from measured data; no stored flags trusted):'])
    if not checks:
        lines.append('  None registered in the archived config; no invariant verification is claimed.')
    for check in checks:
        lines.append(f'  {check["id"]} (case {check["case_id"]}): {check["status"]}; '
                     f'lawful={check["lawful"]}; error={check["error"]!r}; '
                     f'tolerance={check["tolerance"]!r}; basis={check["basis"]!r}. {check["reason"]}')
    if isinstance(config, dict):
        for dropped in config.get('dropped_invariants', []):
            lines.append(f'  Dropped invariant (not executed): {json.dumps(dropped, sort_keys=True)}')
    lines.append('Registered analytic checks (V1; the law is recomputed in this repository):')
    if not analytic:
        lines.append('  None declared in the archived config; no analytic verification is claimed.')
    for check in analytic:
        lines.append(f'  {check["id"]} (case {check["case_id"]}): {check["status"]}; '
                     f'error={check["error"]!r}; tolerance={check["tolerance"]!r}; '
                     f'kind={check["tolerance_kind"]!r}; worst={json.dumps(check["worst"], sort_keys=True)}; '
                     f'basis={check["basis"]!r}. {check["reason"] or ""}')
    if isinstance(config, dict):
        for dropped in config.get('dropped_analytic_cases', []):
            lines.append(f'  Dropped analytic case (not executed): {json.dumps(dropped, sort_keys=True)}')
    lines.append('Registered convergence ladders (V3; every rung recomputed against the closed form):')
    if not ladders:
        lines.append('  None declared in the archived config; no convergence claim is made.')
    for row in ladders:
        lines.append(f'  {row["id"]}: {row["status"]}; ppw={row["ppw"]}; errors={row["errors"]}; '
                     f'order={row["order"]!r}; residual={row["residual"]!r}; '
                     f'implied_orders={row["implied_orders"]}; monotone={row["monotone"]}; '
                     f'coincidence={row["coincidence"]}; tolerance={row["tolerance"]!r}. '
                     f'{row["fit_reason"]}')
        if row['coincidence']:
            lines.append('    GUARD: a coarse-only agreement. No point of these cases can be verified.')
    dt = readout.get('dt_ladder')
    if isinstance(dt, dict):
        lines.append(f'  {dt["id"]} (time-step ladder): max_abs_difference={dt["max_abs_difference"]!r}; '
                     f'courant_factors={dt["courant_factors"]}; steps={dt["steps"]}; '
                     f'duration_s={dt["duration_s"]}. {dt["reason"]}')
    floors = earned_floor(ladders, config.get('registered_ppw'), config.get('inherited_tolerance')) \
        if isinstance(config, dict) else []
    for floor in floors:
        lines.append(f'  earned floor (registered ppw {floor["registered_ppw"]}): '
                     f'{floor["ladder"]} error={floor["measured_error"]!r}; '
                     f'margin vs inherited {floor.get("inherited_tolerance")!r} = '
                     f'{floor.get("margin_vs_inherited")!r}; extrapolation={floor["extrapolation"]}')
    lines.append('Units and normalisation ledger (V5; every reported number names its readout):')
    if ledger.get('diff') == 'absent':
        lines.append('  This readout carries no units declaration. It predates the leg-5 contract, so')
        lines.append('  nothing is declared and nothing is claimed for it. It is not a leg-5 failure.')
    else:
        lines.append(f'  declaration diff against the archived config: {ledger["diff"]}; '
                     f'failures: {len(ledger["failures"])}')
        for case_row in ledger['cases']:
            lines.append(f'  case {case_row["case_id"]}:')
            for entry in case_row['declared']:
                lines.append(f'    {entry["key"]}: unit={entry["unit"]!r}; plane={entry["plane"]!r}; '
                             f'span={entry["span"]!r}; normalisation={entry["normalisation"]!r}; '
                             f'cross_check={entry["cross_check"]}')
            for entry in case_row['exempt']:
                lines.append(f'    {entry["key"]}: unit={entry["unit"]!r} '
                             f'(exemption table; {entry["reason"]})')
        for failure in ledger['failures']:
            lines.append(f'  LEDGER FAILURE: {failure}')
    lines.append('Headlines (recorded values and stored error/pass claims are comparison evidence):')
    for row in headlines:
        value = json.dumps(row['value'], allow_nan=True)
        case = f' (case {row["case_id"]})' if row['case_id'] else ''
        label = row['label']
        if label == 'verified':
            label += ' ' + ','.join(row['invariants'])
        lines.append(f'  {row["path"]}{case} = {value} [{label}] -- {row["reason"]}')
    if not headlines:
        lines.append('  No numeric headlines were emitted; no verification is claimed.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run directory containing readout.json')
    args = parser.parse_args(argv)
    try:
        report = render_report(args.run)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'report: {error}\n')
    print(report, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
