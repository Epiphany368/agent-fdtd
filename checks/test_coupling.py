"""Leg 6 CPU tests: the cited inputs, the overlap model, its limits and its refusals.

Arithmetic and decision logic only. No solver, no GPU, no simulation.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import coupling as cp


class CitedInputs(unittest.TestCase):
    def test_every_input_carries_its_source(self):
        self.assertIn('Marchetti', cp.CITED_INPUTS['source'])
        self.assertIn('10.1038/s41598-017-16505-z', cp.CITED_INPUTS['source'])

    def test_the_quoted_inputs_carry_a_verbatim_quote(self):
        for key in ('device_width_um', 'fibre_mfd_um'):
            entry = cp.CITED_INPUTS[key]
            self.assertEqual(entry['kind'], 'quoted', key)
            self.assertGreater(len(entry['quote']), 30, key)
            self.assertIn(str(entry['value']).rstrip('0').rstrip('.'), entry['quote'].replace('um', ''), key)

    def test_the_waist_is_labelled_derived_not_quoted(self):
        entry = cp.CITED_INPUTS['fibre_waist_um']
        self.assertEqual(entry['kind'], 'derived')
        self.assertIn('MFD / 2', entry['derivation'])
        self.assertEqual(entry['from'], 'fibre_mfd_um')

    def test_the_waist_really_is_half_the_quoted_mfd(self):
        self.assertAlmostEqual(cp.CITED_INPUTS['fibre_waist_um']['value'],
                               cp.CITED_INPUTS['fibre_mfd_um']['value'] / 2, places=12)

    def test_the_platform_note_states_the_shared_modelling_limitation(self):
        self.assertIn('2D-FDTD', cp.CITED_INPUTS['platform_note'])


class Model(unittest.TestCase):
    def test_the_cited_inputs_give_the_registered_factor(self):
        self.assertAlmostEqual(cp.lateral_overlap(12.0, 5.2), 0.874507, places=6)

    def test_the_factor_stays_inside_the_unit_interval(self):
        for width in (0.5, 1, 5, 12, 50, 500):
            for waist in (0.5, 2.6, 5.2, 20.0):
                value = cp.lateral_overlap(width, waist)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_a_very_narrow_device_captures_little(self):
        # A narrow uniform aperture delivers little total power, so the normalised overlap is small.
        self.assertLess(cp.lateral_overlap(0.2, 5.2), 0.05)

    def test_a_very_wide_uniform_device_captures_almost_nothing(self):
        # The model's own limitation: a uniform aperture spreads power where the mode is not.
        self.assertLess(cp.lateral_overlap(1000.0, 5.2), 0.02)

    def test_both_limits_fall_away_from_a_single_peak(self):
        # The model has one optimum in width, near twice the waist. Neither a narrower nor a
        # wider uniform aperture does better. This is the statement the leg must not overstate.
        peak = cp.peak_width(5.2)
        self.assertLess(cp.lateral_overlap(0.5, 5.2), peak['peak_factor'])
        self.assertLess(cp.lateral_overlap(500.0, 5.2), peak['peak_factor'])

    def test_the_factor_is_not_monotonic_in_width(self):
        wide = cp.lateral_overlap(100.0, 5.2)
        near_peak = cp.lateral_overlap(10.0, 5.2)
        self.assertGreater(near_peak, wide)

    def test_the_peak_sits_at_about_twice_the_waist(self):
        peak = cp.peak_width(5.2)
        self.assertAlmostEqual(peak['peak_width_um'] / 5.2, 1.99, delta=0.05)
        self.assertGreater(peak['peak_factor'], 0.87)

    def test_the_cited_width_is_close_to_the_peak(self):
        peak = cp.peak_width(5.2)
        cited = cp.lateral_overlap(12.0, 5.2)
        self.assertGreater(cited, 0.98 * peak['peak_factor'])


class Refusals(unittest.TestCase):
    def test_a_zero_width_is_refused(self):
        with self.assertRaises(ValueError):
            cp.lateral_overlap(0.0, 5.2)

    def test_a_negative_width_is_refused(self):
        with self.assertRaises(ValueError):
            cp.lateral_overlap(-12.0, 5.2)

    def test_a_non_positive_waist_is_refused(self):
        with self.assertRaises(ValueError):
            cp.lateral_overlap(12.0, 0.0)
        with self.assertRaises(ValueError):
            cp.lateral_overlap(12.0, -5.2)

    def test_non_finite_inputs_are_refused(self):
        for width, waist in ((float('nan'), 5.2), (12.0, float('inf'))):
            with self.assertRaises(ValueError):
                cp.lateral_overlap(width, waist)

    def test_a_non_number_is_refused(self):
        with self.assertRaises(ValueError):
            cp.lateral_overlap('12', 5.2)


class Sweep(unittest.TestCase):
    def test_the_sweep_covers_every_requested_width(self):
        rows = cp.sweep([5, 12, 50], [2.6, 5.2])
        self.assertEqual([row['width_um'] for row in rows], [5.0, 12.0, 50.0])
        for row in rows:
            self.assertIn('w_lat_2.6', row)
            self.assertIn('w_lat_5.2', row)

    def test_the_config_labels_the_sweep_as_sensitivity_only(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg6-fibre-overlap.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        self.assertEqual(config['kind'], 'audit')
        self.assertIn('Sensitivity only', config['sweep_note'])
        self.assertIn(12, config['sweep_widths_um'])
        self.assertIn(5.2, config['sweep_waists_um'])


if __name__ == '__main__':
    unittest.main()
