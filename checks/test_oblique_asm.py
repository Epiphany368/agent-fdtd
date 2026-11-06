"""Leg 7 part 1 CPU tests: the ASM cost rules, the reconstruction, and the controls.

Arithmetic only. No solver, no GPU, no FDTD.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import oblique_asm as oa

K0 = 2.0 * math.pi / 1.54925
PERIOD = 0.9


class Midpoints(unittest.TestCase):
    def test_the_samples_span_the_brillouin_zone(self):
        k = oa.brillouin_midpoints(PERIOD, 8)
        self.assertEqual(len(k), 8)
        self.assertAlmostEqual(k[0], -math.pi / PERIOD + (math.pi / PERIOD) / 8, places=12)
        self.assertLess(k[-1], math.pi / PERIOD)

    def test_the_samples_are_midpoints_and_symmetric(self):
        k = oa.brillouin_midpoints(PERIOD, 16)
        self.assertAlmostEqual(float(k.sum()), 0.0, places=12)

    def test_a_non_positive_order_is_refused(self):
        for m in (0, -4):
            with self.assertRaises(ValueError):
                oa.brillouin_midpoints(PERIOD, m)

    def test_a_non_positive_period_is_refused(self):
        for period in (0.0, -0.9):
            with self.assertRaises(ValueError):
                oa.brillouin_midpoints(period, 8)


class CostRules(unittest.TestCase):
    def test_the_two_rules_match_their_formulas(self):
        rules = oa.cost_rules(2.0, 0.9)
        self.assertAlmostEqual(rules['window_prediction'], 2 * 2.0 / 0.9, places=12)
        self.assertAlmostEqual(rules['spectrum_prediction'], math.pi * 2.0 / 0.9, places=12)

    def test_the_spectrum_rule_is_tighter_for_every_positive_waist(self):
        # pi > 2, so this holds for any waist and any period; it is not a coincidence of numbers.
        for waist in (0.5, 2.0, 5.0, 20.0):
            self.assertEqual(oa.cost_rules(waist, PERIOD)['tighter_rule'], 'spectrum')

    def test_a_non_positive_waist_is_refused(self):
        with self.assertRaises(ValueError):
            oa.cost_rules(0.0, PERIOD)


class Reconstruction(unittest.TestCase):
    def test_the_small_order_error_matches_the_cited_sweep(self):
        x = np.linspace(-2.0, 2.0, 801)
        self.assertAlmostEqual(oa.reconstruction_error(x, 2.0, PERIOD, 4, K0), 0.3194, places=3)
        self.assertAlmostEqual(oa.reconstruction_error(x, 2.0, PERIOD, 8, K0), 4.58e-4, places=6)

    def test_the_error_falls_with_the_order(self):
        x = np.linspace(-2.0, 2.0, 801)
        errors = [oa.reconstruction_error(x, 2.0, PERIOD, m, K0) for m in (4, 6, 8, 12)]
        self.assertEqual(errors, sorted(errors, reverse=True))

    def test_the_registered_orders_meet_the_registered_tolerance(self):
        self.assertEqual(oa.first_m_within(2.0, PERIOD, K0, 0.01, (4, 6, 8, 12, 16)), 8)
        self.assertEqual(oa.first_m_within(5.0, PERIOD, K0, 0.01, (4, 6, 8, 12, 16, 24)), 24)

    def test_a_too_small_order_does_not_meet_the_tolerance(self):
        self.assertIsNone(oa.first_m_within(5.0, PERIOD, K0, 0.01, (4, 6, 8)))

    def test_a_non_positive_tolerance_is_refused(self):
        with self.assertRaises(ValueError):
            oa.first_m_within(2.0, PERIOD, K0, 0.0)

    def test_a_narrowed_brillouin_span_does_not_converge(self):
        x = np.linspace(-2.0, 2.0, 801)
        wrong = oa.reconstruction_error(x, 2.0, PERIOD, 32, K0, correct_span=False)
        right = oa.reconstruction_error(x, 2.0, PERIOD, 32, K0)
        self.assertGreater(wrong, 0.01)
        self.assertLess(right, 1e-5)

    def test_a_widened_span_is_not_an_error_here_because_k0_exceeds_pi_over_d(self):
        # k0 = 4.056 /um against pi/d = 3.491 /um, so widening the sample span captures MORE of the
        # propagating spectrum and the reconstruction improves. This is why the control narrows.
        self.assertGreater(K0, math.pi / PERIOD)
        x = np.linspace(-2.0, 2.0, 801)
        k = oa.brillouin_midpoints(PERIOD, 32) * 2.0
        amplitude = np.exp(-(k ** 2) * 4.0 / 4.0)
        amplitude = np.where(np.abs(k) <= K0, amplitude, 0.0)
        field = (amplitude[None, :] * np.exp(1j * k[None, :] * x[:, None])).sum(axis=1) / 32
        exact = np.exp(-(x ** 2) / 4.0)
        error = float(np.sqrt(np.mean(np.abs(field / np.abs(field).max() - exact / np.abs(exact).max()) ** 2)))
        self.assertLess(error, 0.01)

    def test_the_error_floor_is_the_reconstruction_not_the_sampling(self):
        # Past the first order that meets tolerance the error stops falling: the residual is the
        # truncated window, so adding orders beyond that buys nothing. Reporting this stops a
        # reader from buying M for nothing.
        x = np.linspace(-2.0, 2.0, 801)
        a = oa.reconstruction_error(x, 2.0, PERIOD, 24, K0)
        b = oa.reconstruction_error(x, 2.0, PERIOD, 96, K0)
        self.assertLess(abs(b - a), 1e-6)


class Config(unittest.TestCase):
    def test_the_config_declares_the_tolerance_and_the_budget(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg7-asm-feasibility.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        self.assertEqual(config['kind'], 'audit')
        self.assertGreater(config['tolerance'], 0)
        self.assertTrue(config['tolerance_basis'])
        self.assertGreaterEqual(config['budget_runs'], 24)
        self.assertIn(0.9, [config['period_um']])


if __name__ == '__main__':
    unittest.main()


class CellWeights(unittest.TestCase):
    """The step the failed campaign got wrong, pinned so it cannot regress."""

    def test_the_weights_peak_at_the_centre_of_the_window(self):
        weights, x = oa.cell_weights(PERIOD, 16, 2.0, K0)
        self.assertAlmostEqual(float(x[np.argmax(np.abs(weights))]), 0.0, places=12)

    def test_the_centred_weights_reproduce_the_sampled_gaussian(self):
        # M = 16: the envelope error is at the numerical floor.
        self.assertLess(oa.centred_window_error(PERIOD, 16, 2.0, K0), 1e-5)

    def test_the_envelope_error_falls_with_m(self):
        errors = [oa.centred_window_error(PERIOD, m, 2.0, K0) for m in (8, 12, 16)]
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1], errors[0])

    def test_the_uncentred_mapping_does_not_converge(self):
        # This is the signature of the failed campaign: the error stays put as M rises, because
        # the window covers only the right half of the beam.
        def uncentred(m):
            k = oa.brillouin_midpoints(PERIOD, m)
            amplitude = np.exp(-(k ** 2) * 4.0 / 4.0)
            magnitude = np.abs(np.fft.ifft(amplitude))
            magnitude = magnitude / magnitude.max()
            exact = np.exp(-((np.arange(m) * PERIOD) ** 2) / 4.0)
            exact = exact / exact.max()
            return float(np.sqrt(np.mean((magnitude - exact) ** 2)) / np.sqrt(np.mean(exact ** 2)))
        a, b = uncentred(8), uncentred(16)
        self.assertGreater(a, 0.5)
        self.assertAlmostEqual(a, b, places=1)

    def test_the_window_half_width_is_m_d_over_two(self):
        self.assertAlmostEqual(oa.window_um(0.9, 8), 3.6, places=12)

    def test_at_m_eight_the_window_covers_the_two_waist_beam(self):
        # 2 w0 = 4 micrometres against a window of +-3.6 micrometres: covered, but only just.
        self.assertGreater(oa.window_um(PERIOD, 8), 2.0)
