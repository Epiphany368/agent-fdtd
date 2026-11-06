"""Leg 26 CPU tests: declared scope and rule identity must both reach the verdict.

Two defects are pinned here, both in `src/analytic_cases.py`:

1. `evaluate_analytic_cases` decides the status from the tolerance alone (`:316`, scalar path
   `:232`). The config field `scope_match` is only copied onto the row (`:240`, `:324`) and no
   consumer reads it, so a case that declares itself out of scope can be stamped `passed` when its
   numbers happen to agree.
2. `label_analytic_cases` keys its rows on the rule id alone (`:335`). Two cases that share one
   rule id collapse to whichever was evaluated last, and the labelling loop (`:344-374`) reads
   `status`, `tolerance` and `basis` from that row without checking its `case_id`. The failing
   case can therefore be labelled `verified` with its sibling's tolerance and basis.

The fixtures are synthetic and in memory: analytic laws are recomputed by `analytic_cases` itself,
never read from `runs/`, and no simulation is imported. Each data point below is an in-memory
`fresnel` curve, exact or shifted by a declared offset.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import analytic_cases as ac


WAVELENGTH_UM = [1.2, 1.35, 1.5]
EXACT_R, EXACT_T = ac._law_values(dict(law='fresnel', index=3.5), WAVELENGTH_UM)
OFFSET = 0.03


def readout(*cases):
    return dict(stages=list(('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                             'F5-run', 'F6-readout')),
                stage_source={stage: 'test' for stage in
                              ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                               'F5-run', 'F6-readout')},
                cases=list(cases))


def measured(R=None, T=None):
    return dict(wavelength_um=list(WAVELENGTH_UM),
                R=list(EXACT_R if R is None else R),
                T=list(EXACT_T if T is None else T))


def case(case_id, R=None, T=None):
    return dict(id=case_id, measured=measured(R, T))


def entry(case_id, tolerance, scope_match=True, basis=None):
    return dict(id='V1-fresnel', law='fresnel', case_id=case_id, index=3.5,
                tolerance=tolerance, tolerance_kind='test',
                basis=basis or f'test basis for case {case_id}',
                scope_match=scope_match,
                scope_match_reason=('synthetic fixture; see '
                                    'checks/test_leg26_scope_and_rule_identity.py'))


SCALAR_ENTRY = dict(id='V1-grating-zero-order', law='film-tp', case_id='g', index=2.0,
                    thickness_um=0.6, wavelength_um=1.54925, tolerance=0.01,
                    tolerance_kind='test', basis='test basis for case g', scope_match=False,
                    scope_match_reason='synthetic out-of-scope fixture',
                    points=[dict(field='zero_order_efficiency_a', angle_deg=8.5),
                            dict(field='zero_order_efficiency_b', angle_deg=8.9002)])


def headlines(data):
    rows = []
    for index, readout_case in enumerate(data['cases']):
        for key, value in readout_case['measured'].items():
            if key == 'wavelength_um':
                continue
            if isinstance(value, list):
                for point, item in enumerate(value):
                    rows.append(dict(path=f'cases[{index}].measured.{key}[{point}]',
                                     case_id=readout_case['id'], value=item,
                                     label='unchecked', reason='no check'))
            else:
                rows.append(dict(path=f'cases[{index}].measured.{key}',
                                 case_id=readout_case['id'], value=value,
                                 label='unchecked', reason='no check'))
    return rows


def paths_of(data, case_id):
    return {row['path'] for row in headlines(data) if row['case_id'] == case_id}


def verified_paths(rows):
    return {row['path'] for row in rows if row['label'] == 'verified'}


def shifted(values, index, offset=OFFSET):
    curve = list(values)
    curve[index] += offset
    return curve


class ScopeReachesTheVerdict(unittest.TestCase):
    def test_an_out_of_scope_case_is_neither_passed_nor_verified(self):
        data = readout(case('c'))
        config = dict(analytic_cases=[entry('c', tolerance=0.01, scope_match=False)])
        rows = ac.evaluate_analytic_cases(data, config)
        with self.subTest('status'):
            self.assertNotEqual(
                rows[0]['status'], 'passed',
                'a case declaring scope_match: false with an exact numeric match was stamped '
                f"{rows[0]['status']!r}; scope_match is not part of the status condition "
                '(`src/analytic_cases.py:316`)')
        labels = ac.label_analytic_cases(data, config, headlines(data))
        with self.subTest('curve labels'):
            self.assertEqual(
                sorted(verified_paths(labels)), [],
                'an out-of-scope case still promoted its measured points to verified')

        exact = dict(ac.scalar_law_points(SCALAR_ENTRY))
        scalar_data = readout(dict(id='g', measured=exact))
        scalar_config = dict(analytic_cases=[SCALAR_ENTRY])
        scalar_rows = ac.evaluate_analytic_cases(scalar_data, scalar_config)
        with self.subTest('scalar status'):
            self.assertNotEqual(
                scalar_rows[0]['status'], 'passed',
                'an out-of-scope scalar case with exact numeric values was stamped '
                f"{scalar_rows[0]['status']!r}; the scalar path at `src/analytic_cases.py:232` "
                'also ignores scope_match')
        scalar_labels = ac.label_analytic_cases(scalar_data, scalar_config,
                                                headlines(scalar_data))
        with self.subTest('scalar labels'):
            self.assertEqual(
                sorted(verified_paths(scalar_labels)), [],
                'an out-of-scope scalar case still promoted its named fields to verified')


class SharedRuleIdDoesNotShareAVerdict(unittest.TestCase):
    def setUp(self):
        self.data = readout(case('a', T=shifted(EXACT_T, 1)),
                            case('b'))
        self.failing = entry('a', tolerance=0.001, basis='test basis for case a')
        self.passing = entry('b', tolerance=0.05, basis='test basis for case b')

    def labels_for(self, ordered_entries):
        return ac.label_analytic_cases(self.data,
                                       dict(analytic_cases=ordered_entries),
                                       headlines(self.data))

    def own_verdict(self, entries):
        row = ac.evaluate_analytic_cases(self.data, dict(analytic_cases=entries))[0]
        return row['status']

    def test_a_failing_case_is_not_labelled_by_its_passing_sibling(self):
        self.assertEqual(self.own_verdict([self.failing]), 'failed')
        self.assertEqual(self.own_verdict([self.passing]), 'passed')
        failing_paths = paths_of(self.data, 'a')
        seen = []
        for name, entries in (('failing first', [self.failing, self.passing]),
                              ('passing first', [self.passing, self.failing])):
            labels = self.labels_for(entries)
            promoted = sorted(verified_paths(labels) & failing_paths)
            seen.append(verified_paths(labels))
            with self.subTest(order=name):
                self.assertEqual(
                    promoted, [],
                    'the failing case a, which shares the rule id V1-fresnel with the passing '
                    f'case b, had {len(promoted)} points labelled verified under the {name} order '
                    '(rows are keyed on the rule id alone, `src/analytic_cases.py:335`)')
        with self.subTest('order independence'):
            self.assertEqual(
                seen[0], seen[1],
                'swapping the two cases that share rule id V1-fresnel changed which points are '
                'labelled verified')


class LabelReasonsCiteTheirOwnCase(unittest.TestCase):
    def test_a_labels_reason_quotes_its_own_case_not_its_sibling(self):
        data = readout(case('a', T=shifted(EXACT_T, 1)),
                       case('b'))
        entries = [entry('a', tolerance=0.001, basis='test basis for case a'),
                   entry('b', tolerance=0.05, basis='test basis for case b')]
        by_case = {item['case_id']: item for item in entries}
        owner = {path: item['case_id']
                 for item in entries for path in paths_of(data, item['case_id'])}
        labels = ac.label_analytic_cases(data, dict(analytic_cases=entries), headlines(data))
        promoted = [row for row in labels if row['label'] == 'verified']
        self.assertNotEqual(promoted, [], 'no label was produced, so the citation cannot be tested')
        for row in promoted:
            own = by_case[owner[row['path']]]
            with self.subTest(path=row['path']):
                self.assertIn(
                    own['basis'], row['reason'],
                    f"the label on case {own['case_id']} cites another case's basis: "
                    f"{row['reason']!r}")
                self.assertIn(
                    repr(own['tolerance']), row['reason'],
                    f"the label on case {own['case_id']} quotes a tolerance other than its own "
                    f"{own['tolerance']!r}: {row['reason']!r}")


class NegativeControl(unittest.TestCase):
    def test_an_in_scope_matching_case_is_still_verified(self):
        plain = readout(case('solo'))
        plain_config = dict(analytic_cases=[entry('solo', tolerance=0.01)])
        plain_verified = verified_paths(
            ac.label_analytic_cases(plain, plain_config, headlines(plain)))
        self.assertEqual(
            plain_verified, paths_of(plain, 'solo'),
            'the negative control failed before the shared rule id was introduced: an in-scope '
            'case with exact numbers must be labelled verified')

        data = readout(case('a', T=shifted(EXACT_T, 1)),
                       case('b'))
        failing = entry('a', tolerance=0.001, basis='test basis for case a')
        passing = entry('b', tolerance=0.05, basis='test basis for case b')
        passing_paths = paths_of(data, 'b')
        for name, entries in (('failing first', [failing, passing]),
                              ('passing first', [passing, failing])):
            labels = ac.label_analytic_cases(data, dict(analytic_cases=entries),
                                             headlines(data))
            with self.subTest(order=name):
                self.assertEqual(
                    sorted(verified_paths(labels) & passing_paths), sorted(passing_paths),
                    f'the in-scope exact-match case b lost its verified labels under the {name} '
                    'order; a fix may not disable labelling to dodge the shared-rule-id defect')


if __name__ == '__main__':
    unittest.main()
