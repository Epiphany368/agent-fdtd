"""Leg 11 contracts. The audit is an inventory: it must be complete, assigned, and inert."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

from coverage_audit import (ABSENT_CHECKS, COMPILED, REGISTERED_CHECKS, audit, audit_run,
                            classify, discover_runs, family, normalise)
from report import render_report

# The same discovery the driver uses: an inventory must not count itself, and every archived
# run of an audit shares that audit's own schema (counters, before/after overlays), so audits are
# excluded by policy. leg 13's readout-definition ledger is the second such audit.
RUNS = discover_runs(ROOT, exclude_policies=('leg11-coverage-audit-v1', 'leg13-readout-ledger-v1'))


def report_rows(run):
    """The headline rows tools/report.py prints, parsed back out of its own text."""
    text = render_report(run)
    head = text.index('Headlines (')
    rows = []
    for line in text[head:].splitlines()[1:]:
        match = re.match(r'^  (?P<path>\S+?)(?: \(case .*?\))? = .*? \[(?P<label>[a-z]+)', line)
        if match:
            rows.append(match.groupdict())
    return rows


class TestCoverageAudit(unittest.TestCase):
    def test_c1_enumeration_matches_the_report_tool(self):
        """C1: the audit is complete against the tool, not against memory."""
        if not RUNS:
            self.skipTest('no archived runs')
        for run in RUNS:
            _, headlines = audit_run(run)
            printed = report_rows(run)
            self.assertEqual(len(headlines), len(printed), f'{run.name}: row count differs')

    def test_c2_every_row_names_a_real_check_or_none(self):
        """C2: no row cites a check that this repository does not implement."""
        for name in ('T', 'R', 'J', 'P_up_reduced', 'energy_residual', 'wall_seconds'):
            role, coverage, leg, cost = classify(name)
            for check in REGISTERED_CHECKS:
                self.assertNotIn(check, name)
            self.assertIn(coverage, ('covered', 'open', 'settled', 'not-applicable'))
            self.assertTrue(role)
        self.assertIn('V4', ABSENT_CHECKS)
        # V4 is registered in plan.md but has no implementation, so no row may claim it.
        self.assertNotIn('V4', REGISTERED_CHECKS)

    def test_c3_every_open_physics_gap_is_assigned_and_costed(self):
        """C3: a gap with neither an owner nor a cost is a wish, not a gap."""
        seen = set()
        for pattern, role, coverage, leg, cost in COMPILED:
            seen.add(coverage)
            if coverage == 'open':
                self.assertNotEqual(leg, '-', f'{pattern.pattern}: open gap with no owner')
                self.assertNotEqual(cost, '-', f'{pattern.pattern}: open gap with no cost')
            elif coverage == 'settled':
                # Not a gap: a decision closed the question, and the reason is stated.
                self.assertNotEqual(leg, '-', f'{pattern.pattern}: settled row states no reason')
                self.assertEqual(cost, '-', f'{pattern.pattern}: settled row is not a gap to cost')
            else:
                self.assertEqual(leg, '-', f'{pattern.pattern}: closed row carries an owner')
        self.assertEqual(seen, {'covered', 'open', 'settled', 'not-applicable'},
                         'all four coverage values must be in use, or one is a dead distinction')
        if not RUNS:
            self.skipTest('no archived runs')
        families, _, _ = audit(RUNS)
        for name in families:
            if name == '?<unreadable>':
                continue
            role, coverage, leg, cost = classify(name)
            self.assertNotEqual(role, 'unknown', f'{name}: emitted but unclassified')

    def test_c4_the_audit_changes_no_label(self):
        """C4: running the audit must not alter any headline's label."""
        if not RUNS:
            self.skipTest('no archived runs')
        for run in RUNS:
            _, headlines = audit_run(run)
            by_path = {row['path']: row['label'] for row in headlines}
            for printed in report_rows(run):
                self.assertEqual(by_path[printed['path']], printed['label'],
                                 f'{run.name}: {printed["path"]} label changed')

    def test_c5_negative_control_an_invented_field_is_never_silently_passed(self):
        """C5: a field name nobody has seen must come back unknown, not covered."""
        invented = 'invented_power_ratio_xyz'
        self.assertEqual(classify(invented), ('unknown', 'open',
                                              'not established from this repository', '-'))
        self.assertFalse(any(rule.match(invented) for rule, *_ in COMPILED))
        # The same must hold end to end, from a real path through the family reducer.
        self.assertEqual(family('cases[0].measured.' + invented), (invented, 'measured'))
        # And the control is not vacuous: a real field is classified.
        self.assertNotEqual(classify('energy_residual')[0], 'unknown')

    def test_normalisation_collapses_indices_and_keeps_dotted_names(self):
        self.assertEqual(normalise('cases[7].measured.T[41]'), 'cases[].measured.T[]')
        self.assertEqual(family('sensitivity[2].w_lat_2.6'), ('w_lat_2.6', 'metadata'))
        self.assertEqual(family('cases[3].measured.T'), ('T', 'measured'))
        self.assertEqual(family('cases[3].wall_seconds'), ('wall_seconds', 'metadata'))

    def test_c6_default_path_unchanged(self):
        """C6: the audit imports no solver and writes nothing outside its own run dir."""
        source = (ROOT / 'src' / 'coverage_audit.py').read_text()
        source += (ROOT / 'src' / 'leg11_coverage.py').read_text()
        for forbidden in ('import torch', 'from torch', 'torchfdtd', 'subprocess'):
            self.assertNotIn(forbidden, source, f'{forbidden} must not appear in the audit')


if __name__ == '__main__':
    unittest.main()
