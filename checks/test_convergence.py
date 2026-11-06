"""Leg 4 CPU tests: the fit, the monotonicity gate, the coincidence guard, the earned floor.

Arithmetic and decision logic only; the GPU ladder is not exercised here.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import analytic_cases as ac
import convergence as cv

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
WAVELENGTH = [1.2, 1.35, 1.5]
INDEX = 3.5
EXACT_R, EXACT_T = ac._law_values(dict(law='fresnel', index=INDEX), WAVELENGTH)


def readout_with_rungs(ids_and_offsets):
    """One interface case per rung; the transmittance is offset by the requested absolute error."""
    cases = []
    for case_id, offset in ids_and_offsets:
        cases.append(dict(id=case_id, ppw=0,
                          measured=dict(wavelength_um=list(WAVELENGTH), R=list(EXACT_R),
                                        T=[value + offset for value in EXACT_T])))
    return dict(stages=list(STAGES), stage_source={stage: 'test' for stage in STAGES}, cases=cases)


def ladder_config(case_ids, ppw, tolerance):
    analytic = [dict(id='V1-fresnel', law='fresnel', case_id=case_id, index=INDEX,
                     tolerance=tolerance, tolerance_kind='test', basis='test')
                for case_id in case_ids]
    return dict(policy='test', analytic_cases=analytic,
                ladders=[dict(id='L', case_ids=list(case_ids), ppw=list(ppw), refinement=2.0,
                              tolerance=tolerance, coarse_rungs=1)])


class Monotonicity(unittest.TestCase):
    def test_strictly_decreasing_is_accepted(self):
        self.assertTrue(cv.monotone_decreasing([1.0, 0.25, 0.0625]))

    def test_equal_errors_are_rejected(self):
        self.assertFalse(cv.monotone_decreasing([1.0, 1.0]))

    def test_a_rising_rung_is_rejected(self):
        self.assertFalse(cv.monotone_decreasing([1.0, 0.25, 0.3]))


class Order(unittest.TestCase):
    def test_second_order_is_recovered_exactly(self):
        # error ~ h^2 and h halves each rung.
        fit = cv.fit_order([1.0, 0.25, 0.0625, 0.015625])
        self.assertAlmostEqual(fit['order'], 2.0, places=12)
        self.assertAlmostEqual(fit['residual'], 0.0, places=12)

    def test_first_order_is_recovered(self):
        fit = cv.fit_order([1.0, 0.5, 0.25, 0.125])
        self.assertAlmostEqual(fit['order'], 1.0, places=12)

    def test_consecutive_orders_match_the_fit(self):
        fit = cv.fit_order([1.0, 0.25, 0.0625])
        for implied in fit['implied']:
            self.assertAlmostEqual(implied, 2.0, places=12)

    def test_one_rung_yields_no_order(self):
        fit = cv.fit_order([1.0])
        self.assertIsNone(fit['order'])
        self.assertIsNone(fit['residual'])

    def test_a_zero_error_yields_no_order(self):
        fit = cv.fit_order([1.0, 0.0])
        self.assertIsNone(fit['order'])

    def test_an_imperfect_ladder_reports_a_nonzero_residual(self):
        fit = cv.fit_order([1.0, 0.2, 0.0625, 0.015625])
        self.assertGreater(fit['residual'], 0.0)


class Guard(unittest.TestCase):
    def test_coarse_inside_and_fine_outside_is_flagged(self):
        self.assertTrue(cv.coincidence([0.004, 0.001, 0.02], 0.01))

    def test_all_inside_is_not_flagged(self):
        self.assertFalse(cv.coincidence([0.004, 0.001, 0.0002], 0.01))

    def test_all_outside_is_not_flagged(self):
        self.assertFalse(cv.coincidence([0.4, 0.1, 0.02], 0.01))

    def test_no_tolerance_is_not_flagged(self):
        self.assertFalse(cv.coincidence([0.004, 0.001, 0.02], None))


class Ladders(unittest.TestCase):
    def test_clean_second_order_ladder_converges(self):
        data = readout_with_rungs([('r20', 0.004), ('r40', 0.001),
                                   ('r80', 0.00025), ('r160', 0.0000625)])
        config = ladder_config(['r20', 'r40', 'r80', 'r160'], [20, 40, 80, 160], 0.01)
        row = cv.evaluate_ladders(data, config)[0]
        self.assertEqual(row['status'], 'converged')
        self.assertAlmostEqual(row['order'], 2.0, places=6)
        self.assertFalse(row['coincidence'])

    def test_a_rising_rung_stops_the_ladder_without_a_fit(self):
        data = readout_with_rungs([('r20', 0.004), ('r40', 0.001), ('r80', 0.02)])
        config = ladder_config(['r20', 'r40', 'r80'], [20, 40, 80], 0.01)
        row = cv.evaluate_ladders(data, config)[0]
        self.assertEqual(row['status'], 'stopped-non-monotone')
        self.assertIsNone(row['order'])
        self.assertFalse(row['monotone'])

    def test_coarse_only_agreement_is_a_coincidence_not_a_pass(self):
        data = readout_with_rungs([('r20', 0.004), ('r40', 0.02), ('r80', 0.03)])
        config = ladder_config(['r20', 'r40', 'r80'], [20, 40, 80], 0.01)
        row = cv.evaluate_ladders(data, config)[0]
        self.assertTrue(row['coincidence'])
        self.assertNotEqual(row['status'], 'converged')

    def test_a_rung_without_a_measured_error_is_not_executed(self):
        data = readout_with_rungs([('r20', 0.004)])
        config = ladder_config(['r20', 'missing'], [20, 40], 0.01)
        row = cv.evaluate_ladders(data, config)[0]
        self.assertEqual(row['status'], 'not-executed')

    def test_no_declared_ladders_means_no_rows(self):
        data = readout_with_rungs([('r20', 0.004)])
        self.assertEqual(cv.evaluate_ladders(data, {}), [])

    def test_the_guard_strips_verified_only_from_flagged_cases(self):
        rows = [dict(id='L', coincidence=True, case_ids=['r20', 'r40'])]
        headlines = [dict(case_id='r20', path='p', label='verified', reason='r', invariants=['x']),
                     dict(case_id='other', path='q', label='verified', reason='r', invariants=['x']),
                     dict(case_id='r40', path='s', label='consistent', reason='r')]
        result = cv.apply_coincidence_guard(headlines, rows)
        labels = {row['path']: row['label'] for row in result}
        self.assertEqual(labels['p'], 'unchecked')
        self.assertEqual(labels['q'], 'verified')
        self.assertEqual(labels['s'], 'consistent')
        self.assertIn('Coarse-mesh coincidence', result[0]['reason'])

    def test_the_guard_does_nothing_without_a_flag(self):
        rows = [dict(id='L', coincidence=False, case_ids=['r20'])]
        headlines = [dict(case_id='r20', path='p', label='verified', reason='r')]
        self.assertEqual(cv.apply_coincidence_guard(headlines, rows)[0]['label'], 'verified')


class EarnedFloor(unittest.TestCase):
    def test_margin_and_extrapolation_are_reported(self):
        data = readout_with_rungs([('r20', 0.004), ('r40', 0.001),
                                   ('r80', 0.00025), ('r160', 0.0000625)])
        config = ladder_config(['r20', 'r40', 'r80', 'r160'], [20, 40, 80, 160], 0.01)
        config['registered_ppw'] = 40
        config['inherited_tolerance'] = 0.01
        rows = cv.evaluate_ladders(data, config)
        floor = cv.earned_floor(rows, 40, 0.01)[0]
        self.assertAlmostEqual(floor['measured_error'], 0.001, places=12)
        self.assertAlmostEqual(floor['margin_vs_inherited'], 10.0, places=9)
        self.assertEqual(floor['extrapolation'], 'supported')

    def test_no_registered_rung_yields_no_floor(self):
        rows = [dict(id='L', status='converged', ppw=[20, 40], errors=[0.004, 0.001],
                     tolerance=0.01, order=2.0, residual=0.001)]
        self.assertEqual(cv.earned_floor(rows, 80, 0.01), [])


class Config(unittest.TestCase):
    def test_the_leg_config_declares_four_rungs_and_a_time_step_ladder(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg4-convergence-ladder.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        self.assertEqual(len(config['ladders']), 2)
        for ladder in config['ladders']:
            self.assertEqual(len(ladder['case_ids']), 4)
            self.assertEqual(ladder['ppw'], [20, 40, 80, 160])
        self.assertEqual(config['dt_ladder']['courant_factors'], [0.99, 0.5])
        self.assertIn(config['registered_ppw'], config['ladders'][0]['ppw'])
        for entry in config['analytic_cases']:
            self.assertTrue(entry['tolerance_kind'])
            self.assertTrue(entry['basis'])
            self.assertTrue(entry['scope_match_reason'])

    def test_the_time_step_rung_keeps_the_physical_duration(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg4-convergence-ladder.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        by_id = {case['id']: case for case in config['cases']}
        first, second = (by_id[case_id] for case_id in config['dt_ladder']['case_ids'])
        self.assertEqual(first['ppw'], second['ppw'])
        self.assertEqual(first['duration_fs'], second['duration_fs'])
        self.assertNotEqual(first['courant_factor'], second['courant_factor'])


if __name__ == '__main__':
    unittest.main()
