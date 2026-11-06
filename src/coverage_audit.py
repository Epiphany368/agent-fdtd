"""Leg 11: physics-check coverage audit. Inventory only; no label is changed.

For every readout field family emitted by the archived runs, report:
  * which labels it receives today,
  * which registered check granted any of them,
  * what role it plays, and whether a physics check is meaningful for it,
  * for a physics field with no coverage, which leg would close the gap and at what cost.

Rule V-0 is not touched. This module computes the same headline pipeline as
``tools/report.py`` and never writes a label back.
"""
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from analytic_cases import label_analytic_cases
from convergence import apply_coincidence_guard, evaluate_ladders
from invariants import verify_invariants
from verification import STAGES

INDEX = re.compile(r'\[\d+\]')
# A registered check must exist in the repository to be named in a row.
REGISTERED_CHECKS = ('V1-fresnel', 'V1-airy', 'V1-grating-zero-order', 'V1-slab-mode',
                     'V2-a', 'V2-b', 'V2-c', 'V2-d')
ABSENT_CHECKS = ('V4',)

# leg 13 (S4) pinned the readout definition of these fields: plane, span and normalisation are
# registered in `units_ledger.READOUT_DEFINITIONS` and diffed mechanically by the leg-13 ledger
# (48 of 48 pairs, 0 missing, 0 invalid). A pinned definition is not a physics check, so the field
# leaves the gap list but stays `consistent`: no label becomes `verified`.
_S4_SETTLED = ('S4 (leg 13) pinned the readout: plane, span and normalisation are registered and '
               'mechanically diffed. A pinned definition is not a physics check, so the number '
               'stays `consistent` and no label becomes `verified`')

# role: what kind of number is this; checkable is whether physical law can cover it.
# coverage: 'covered' (a check grants it) | 'open' (a physics number nobody checks, owner + cost required)
#         | 'settled' (a physics number that is not a gap because a decision closed it) | 'not-applicable'
# Every entry is the auditor's judgement and is asserted by checks/test_coverage_audit.py.
RULES = (
    # --- physical observables -------------------------------------------------
    (r'^T$', 'physics', 'covered', '-', '-'),
    (r'^R$', 'physics', 'covered', '-', '-'),
    # The five fields leg 13 (S4) pinned. A pinned definition is not a physics check: these stay
    # `consistent`, no label becomes `verified`, and no closed form covers them.
    (r'^J$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^P_up_reduced$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^P_in_reduced$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^unit_mode_flux_reduced$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^energy_residual$', 'physics', 'covered', '-', '-'),
    (r'^empty_reflection_floor$', 'physics', 'covered', '-', '-'),
    (r'^eta_lat$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^zero_order_efficiency_[ab]$', 'physics', 'covered', '-', '-'),
    (r'^launch_neff$', 'physics', 'covered', '-', '-'),
    (r'^launch_scores\[\]$', 'physics', 'settled', 'a selection diagnostic, not an accuracy claim; no check target', '-'),
    (r'^energy_spread(_relative)?$', 'physics', 'settled',
     'leg 12 corrected the registration: this is the ASM campaign delivered-energy spread across phases, not a closed-box conservation quantity, so it belongs with the parked ASM route. Leg 12 was asked to cover it with a closed box and correctly refused, because doing so would grant by field name', '-'),
    (r'^drift_deg$', 'physics', 'settled', 'S1 measured it and closed the leg; the number is a declared error bound, not a check target', '-'),
    (r'^k_fixed_per_um$', 'physics', 'settled', 'the same S1 bound', '-'),
    (r'^model_peak_eta_lat$', 'physics', 'settled', 'S2: the cited model, not this solver; comparison only', '-'),
    (r'^reconstruction_error$', 'physics', 'settled', 'S2: a reconstruction residual of cited inputs, not a solver observable', '-'),
    (r'^(rel_l2_error|absolute_change|relative_change)$', 'statistic', 'not-applicable', '-', '-'),
    # --- inputs and geometry --------------------------------------------------
    (r'^(wavelength_um|frequency_hz|period_um|device_width_um|fibre_waist_um|fibre_mode_field_diameter_um|model_peak_width_um|w0_um|angle_[ab]_deg|realised_separation_deg|m|registered_M|periods_covered_by_beam|ber)[\[\]]*$', 'input', 'not-applicable', '-', '-'),
    (r'^(value)$', 'input', 'not-applicable', '-', '-'),
    (r'^(ppw|shape|steps|dt_s|courant_factor|edge_count|index|width_um|w_lat_[0-9.]+|profile_points)$', 'grid-metadata', 'not-applicable', '-', '-'),
    (r'^(three_d_cells|two_d_cells|three_d_shape|two_d_shape|cell_ratio|time_step_pinned)$', 'grid-metadata', 'not-applicable', '-', '-'),
    (r'^(sensitivity|source_sweep|sweep|cost_rules|refusal_probes|dimension|drift|cited_inputs|band_um)$', 'container', 'not-applicable', '-', '-'),
    (r'^(relative_difference|shapes_differ|ran|finite|refused)$', 'comparison', 'not-applicable', '-', '-'),
    # --- measurement and budget bookkeeping -----------------------------------
    (r'^(tolerance|tolerance_analytic|tolerance_energy|tolerance_relative|tolerance_known_discrepancy|tolerance_reconstruction|min_weight|peak_weight|budget_runs|first_m_within_tolerance)$', 'budget', 'not-applicable', '-', '-'),
    (r'^(wall_seconds|elapsed_seconds|reference_wall_seconds|reference_elapsed_seconds|sample_elapsed_seconds|duration_s|duration_fs|max_abs_difference|energy_spread_relative)$', 'timing', 'not-applicable', '-', '-'),
    (r'^sourcepower_(reduced|max_relative_error)$', 'diagnostic', 'not-applicable', '-', '-'),
    (r'^spectrum_prediction$', 'prediction', 'not-applicable', '-', '-'),
    (r'^window_prediction$', 'prediction', 'not-applicable', '-', '-'),
    (r'^epsilon_sha256$', 'identity', 'not-applicable', '-', '-'),
    (r'^error$', 'statistic', 'not-applicable', '-', '-'),
    (r'^(T|R)_relative_error$', 'comparison', 'not-applicable', '-', '-'),
    # leg 13's standing controls: the control's own bookkeeping, not a device observable. The flux
    # pair is what the control reads; the two ratios and their deviation are the comparison itself.
    (r'^(forward_flux_reduced|backward_flux_reduced)$', 'diagnostic', 'not-applicable', '-', '-'),
    (r'^(symmetric_ratio_max_deviation|oneway_ratio_max_deviation|deviation)$', 'comparison',
     'not-applicable', '-', '-'),
    (r'^(margin|frequencies_used|frequencies_dropped)$', 'statistic', 'not-applicable', '-', '-'),
    # --- families from the earlier legs, whose meaning is established by their drivers -------
    (r'^(forward_total|backward_total|delivered_total)$', 'physics', 'settled',
     'S1 is closed and the ASM route is parked; these are evidence for that decision, not a check target', '-'),
    (r'^(phase_rad|k_per_um|k_parallel_per_um)$', 'input', 'not-applicable', '-', '-'),
    (r'^(empty_seconds|grating_seconds)$', 'timing', 'not-applicable', '-', '-'),
    (r'^courant_factors$', 'grid-metadata', 'not-applicable', '-', '-'),
    (r'^reference_power_reduced$', 'physics', 'settled', _S4_SETTLED, '-'),
    (r'^launch_scores$', 'physics', 'settled',
     'a selection diagnostic, not an accuracy claim; no check target', '-'),
    (r'^(absorbed|reflected|transmitted)$', 'physics', 'open',
     'V2 extends the closed-box balance to an open domain with flux monitors (proposed, leg not yet registered)', '~0.3 GPU-h'),
    (r'^(delta_field_norm|field_norm|residual)$', 'statistic', 'not-applicable', '-', '-'),
    (r'^weight$', 'budget', 'not-applicable', '-', '-'),
    (r'^seconds$', 'timing', 'not-applicable', '-', '-'),
    (r'^angle_deg$', 'input', 'not-applicable', '-', '-'),
    (r'^zero_order_efficiency$', 'physics', 'covered', '-', '-'),
    (r'^zero_order_forward_power$', 'physics', 'open',
     'leg 14 built V1-grating-zero-order for the efficiency; the absolute power stays open because it needs the S4 normalisation pinned first', 'CPU + ~0.3 GPU-h'),
)
COMPILED = tuple((re.compile(pattern), role, coverage, leg, cost)
                 for pattern, role, coverage, leg, cost in RULES)

# Fields whose role could not be established from this repository. Named explicitly so that a
# new, unclassified field is never silently passed: see stop condition 1.
KNOWN_UNCLASSIFIED = ('drift', 'band_um', 'angle_deg', 'k_fixed_per_um', 'first_m_within_tolerance')


# Finite archive exception, approved for leg 30. Each tuple pins the bytes of
# (readout.json, config.input.json, manifest.json), in that order. These are the
# six pre-classification leg-24 diagnostics, NOT a rule for other leg-24 output.
# A seventh run, renamed copy, changed byte or missing sidecar must fail closed.
LEGACY_DIAGNOSTICS = {
    '20261019T164935Z-648159': (
        '47f55533662c4a6d9306b8c6b824580ef086f30e80c0d233587e90cb485653c9',
        'bf8f350f5aa50bcfdcc1e61c4a08203ab956191d2deb2476f7bdb988ef7f79e2',
        'fdccd6789f7fe3b349e9462b23b6b378817338e22054cd87196299d4f6ff62fd'),
    '20261019T165011Z-649157': (
        '8156160221f95ef2ea6952f1864167cdaf160ff1f80d4704361cab954aaa1912',
        'b9fdb7b81b08c6b138d2bab3d65d50be4a5742308f6501261b8e8280357a71f3',
        'e91627e505eb752c28e77d1ef2037c12075698ec24604eff85413dde1df22036'),
    '20261019T170024Z-674169': (
        'cb6611478d0fc5ea14b47fc32c3f2efb71edeeb56d55871d409275c90c9fa4c8',
        'bf8f350f5aa50bcfdcc1e61c4a08203ab956191d2deb2476f7bdb988ef7f79e2',
        'dd1428e70a26d54bf635f4be6906ed2a632c729778a98382ce1491f8615fbcca'),
    '20261019T170056Z-675457': (
        '95d11747ec7c44c57cbc2b376f8a729a493d5a2c373fb9b06c1a2bfff4583675',
        'b9fdb7b81b08c6b138d2bab3d65d50be4a5742308f6501261b8e8280357a71f3',
        '69dde5db6ae09ea6e09b596632f9e2f3a66ee46d6e98689808d4030733b3398f'),
    '20261019T170239Z-679381': (
        '7ac11b707058bdd277509b24df4afd6fa64bda8c473a49ade05423f7ff6c2e69',
        'bf8f350f5aa50bcfdcc1e61c4a08203ab956191d2deb2476f7bdb988ef7f79e2',
        'e62f3310b0b84fbe8b66c0a2540be55edd6276d97dd056c8a48ada0df2b47d46'),
    '20261019T170318Z-680640': (
        '441245233bf684282d957bca04c06071ba01b73eaeee8474d55b41131cc2fe91',
        'bf8f350f5aa50bcfdcc1e61c4a08203ab956191d2deb2476f7bdb988ef7f79e2',
        '97a1b52c3779cace3f70494392511e396e15a8fca32cb4324425b7ca3a31b1ef'),
}
AUDIT_POLICIES = ('leg11-coverage-audit-v1', 'leg13-readout-ledger-v1')


class ReadoutClassificationError(ValueError):
    """A named archive contract failure; never an inventory exclusion."""


class RunInventory(list):
    """Pipeline paths, with the explicitly excluded diagnostics still attached."""

    def __init__(self, pipelines=(), diagnostics=()):
        super().__init__(sorted(pipelines))
        self.diagnostics = sorted(diagnostics, key=lambda row: row['run'])


def _read_object(run, filename):
    try:
        raw = (run / filename).read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        raise ReadoutClassificationError(f'{run}: {filename}: {error}') from error
    if not isinstance(data, dict):
        raise ReadoutClassificationError(f'{run}: {filename}: expected a JSON object')
    return data, raw


def classify_readout(run):
    """Return (kind, policy, reason), or fail naming the directory and field.

    Diagnostics are an explicit declaration, never inferred from missing stages,
    a leg number, a config policy, a verdict or simulation_executed alone.
    The only historical exception is the finite, byte-pinned table above.
    """
    run = Path(run)
    data, raw = _read_object(run, 'readout.json')
    if run.name in LEGACY_DIAGNOSTICS and 'policy' not in data:
        for filename, expected in zip(('readout.json', 'config.input.json', 'manifest.json'),
                                      LEGACY_DIAGNOSTICS[run.name]):
            if filename == 'readout.json':
                content = raw
            else:
                _, content = _read_object(run, filename)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ReadoutClassificationError(
                    f'{run}: missing policy; legacy {filename} SHA-256 does not match')
        config, _ = _read_object(run, 'config.input.json')
        manifest, _ = _read_object(run, 'manifest.json')
        # State the identity that the pinned sidecars establish, not just a hash.
        if (data.get('leg') != 24 or config.get('policy') != 'leg24-graded-modal-v1'
                or manifest.get('run_id') != run.name
                or manifest.get('script') != 'src/leg24_graded_modal.py'):
            raise ReadoutClassificationError(f'{run}: missing policy; legacy identity mismatch')
        return ('diagnostics', config['policy'],
                'Legacy leg 24 modal baseline/graded comparison, not a staged pipeline; '
                'finite run-id and readout/config/manifest SHA-256 exception '
                f'(readout {hashlib.sha256(raw).hexdigest()}).')
    policy = data.get('policy')
    if not isinstance(policy, str) or not policy.strip():
        raise ReadoutClassificationError(f'{run}: missing or invalid policy (non-empty string required)')
    kind = data.get('readout_kind')
    if kind == 'diagnostics':
        reason = data.get('readout_reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ReadoutClassificationError(
                f'{run}: missing or invalid readout_reason for readout_kind diagnostics')
        return 'diagnostics', policy, reason
    if 'readout_kind' in data and kind != 'pipeline':
        raise ReadoutClassificationError(f'{run}: invalid readout_kind {kind!r}')
    if data.get('stages') != list(STAGES):
        raise ReadoutClassificationError(
            f'{run}: missing or invalid stages (ordered F1-F6 stage list required); '
            'diagnostics require readout_kind and readout_reason')
    return 'pipeline', policy, ''


def _partition_runs(run_dirs, exclude_policies=()):
    if isinstance(run_dirs, RunInventory):
        run_dirs = list(run_dirs) + [row['run'] for row in run_dirs.diagnostics]
    pipelines, diagnostics = [], []
    for run in sorted(set(map(Path, run_dirs))):
        kind, policy, reason = classify_readout(run)
        # Classification precedes policy exclusions: an invalid audit ledger is
        # still invalid, and a declared diagnostic must remain visible.
        if kind == 'diagnostics':
            diagnostics.append(dict(run=run, policy=policy, reason=reason))
        elif policy not in exclude_policies:
            pipelines.append(run)
    return RunInventory(pipelines, diagnostics)


def discover_runs(root, exclude_policies=()):
    """Classify every archived readout; return only the pipeline inventory.

    The list-compatible result also carries ``diagnostics`` for reporting.
    Invalid readouts raise instead of disappearing or becoming unreadable rows.
    Audit-ledger policies are excluded only after their shape is validated.
    """
    return _partition_runs((path.parent for path in (Path(root) / 'runs').glob('*/readout.json')),
                           exclude_policies)


def normalise(path):
    """Collapse array indices so that one field family is one row."""
    return INDEX.sub('[]', path)


def family(path):
    """The readout field this headline belongs to, and whether it is measured or metadata.

    A field name may itself contain a dot (``w_lat_2.6``), so a trailing numeric component is
    rejoined to the one before it. Without this, one field splits into three bogus families.
    """
    parts = [part for part in normalise(path).split('.') if part]
    if not parts:
        return '?', '?'
    name = parts[-1]
    if re.fullmatch(r'[0-9]+(\.[0-9]+)?', name) and len(parts) > 1:
        name = parts[-2] + '.' + name
    if name.endswith('[]'):
        name = name[:-2]
    kind = 'measured' if 'measured' in parts else 'metadata'
    return name, kind


def audit_run(run):
    """The same headline pipeline as tools/report.py, with no mutation."""
    run = Path(run)
    kind, _, _ = classify_readout(run)
    if kind != 'pipeline':
        raise ReadoutClassificationError(f'{run}: readout_kind diagnostics is outside the pipeline inventory')
    readout, _ = _read_object(run, 'readout.json')
    config_path = run / 'config.input.json'
    config = _read_object(run, 'config.input.json')[0] if config_path.is_file() else None
    try:
        headlines = verify_invariants(readout, config)
        headlines = label_analytic_cases(readout, config, headlines)
        headlines = apply_coincidence_guard(headlines, evaluate_ladders(readout, config))
    except (ValueError, TypeError) as error:
        raise ReadoutClassificationError(f'{run}: pipeline verification: {error}') from error
    return readout, headlines


def audit(run_dirs):
    """Aggregate pipeline coverage. Return (families, checked, unclassified)."""
    families = defaultdict(lambda: {'labels': set(), 'checks': set(), 'runs': set(),
                                    'kind': set(), 'samples': []})
    checked = 0
    for run in _partition_runs(run_dirs):
        readout, headlines = audit_run(run)
        checked += len(headlines)
        policy = readout.get('policy', 'undeclared')
        for row in headlines:
            name, kind = family(row['path'])
            entry = families[name]
            entry['labels'].add(row['label'])
            entry['kind'].add(kind)
            entry['runs'].add(policy)
            for check in row.get('invariants') or []:
                entry['checks'].add(check)
            if len(entry['samples']) < 2:
                entry['samples'].append(row['path'])
    unclassified = []
    for name in sorted(families):
        if not any(rule.match(name) for rule, *_ in COMPILED):
            unclassified.append(name)
    return families, checked, unclassified


def classify(name):
    for rule, role, coverage, leg, cost in COMPILED:
        if rule.match(name):
            return role, coverage, leg, cost
    return 'unknown', 'open', 'not established from this repository', '-'


def render(run_dirs):
    # The two historical drivers make a plain list to exclude their own run,
    # losing RunInventory's diagnostics. Recover that reporting context from
    # the input archive roots; for an empty driver list use its launcher root
    # (or the repository for a dry run). Discovery results, including empty
    # isolated inventories, always retain their own explicit context.
    if isinstance(run_dirs, RunInventory):
        inventory = _partition_runs(run_dirs)
    else:
        run_dirs = list(map(Path, run_dirs))
        roots = {run.parent.parent for run in run_dirs}
        if not roots:
            pinned = os.environ.get('RUN_DIR_ABS')
            roots = {Path(pinned).resolve().parents[1] if pinned else Path(__file__).resolve().parents[1]}
        diagnostics = []
        for root in sorted(roots):
            diagnostics.extend(discover_runs(root, AUDIT_POLICIES).diagnostics)
        inventory = _partition_runs(run_dirs + [row['run'] for row in diagnostics])
    families, checked, unclassified = audit(inventory)
    lines = ['Physics-check coverage audit (leg 11)',
             f'Runs read: {len(inventory)}; headline values seen: {checked}; field families: '
             f'{len(families)}',
             f'Diagnostics excluded from pipeline inventory: {len(inventory.diagnostics)}',
             *[f'  {row["run"]} | policy={row["policy"]} | {row["reason"]}'
               for row in inventory.diagnostics],
             f'Registered checks in this repository: {", ".join(REGISTERED_CHECKS)}',
             f'Registered but absent: {", ".join(ABSENT_CHECKS)} (cannot be cited as a closure)',
             '', 'family | kind | labels seen | checks that granted them | role | coverage | closing leg | cost']
    gaps = []
    for name in sorted(families):
        entry = families[name]
        role, coverage, leg, cost = classify(name)
        labels = ','.join(sorted(entry['labels']))
        checks = ','.join(sorted(entry['checks'])) or 'none'
        lines.append(f'{name} | {"/".join(sorted(entry["kind"]))} | {labels} | {checks} | '
                     f'{role} | {coverage} | {leg} | {cost}')
        if role in ('physics', 'unknown') and coverage == 'open':
            gaps.append((role, name, labels, leg, cost, len(entry['runs'])))
    lines.append('')
    lines.append('GAP LIST (physics fields with no passed check; costed and assigned)')
    for role, name, labels, leg, cost, runs in sorted(gaps):
        lines.append(f'  {name} [{labels}] -- {role} -- closed by: {leg} -- cost: {cost} -- '
                     f'seen in {runs} policy(-ies)')
    lines.append('')
    lines.append('WIRED BUT NOT GRANTING (a check exists for the field, yet no label was granted)')
    wired = [(name, families[name], classify(name)) for name in sorted(families)
             if classify(name)[1] == 'covered'
             and 'verified' not in families[name]['labels']]
    for name, entry, (_r, _c, leg, cost) in wired:
        lines.append(f'  {name} [{",".join(sorted(entry["labels"]))}] -- a check exists for this field '
                     f'and did not grant it; this is a wiring gap, not a missing check')
    if not wired:
        lines.append('  none')
    lines.append('')
    lines.append('UNCLASSIFIED (stop condition 1: named, never guessed)')
    for name in unclassified:
        lines.append(f'  {name} -- no rule matches; reported as unknown, not assigned a check')
    if not unclassified:
        lines.append('  none')
    return '\n'.join(lines) + '\n'


def main():
    """CPU-only, read-only audit entry point; contract failures exit nonzero."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(render(discover_runs(args.root, AUDIT_POLICIES)), end='')
    except (OSError, ValueError, TypeError) as error:
        print(f'Coverage audit FAILED: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
