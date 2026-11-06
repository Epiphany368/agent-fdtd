"""CPU-only numerical and provenance checks for the registered tooth ladder."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('leg16', ROOT / 'src/leg16_tooth_convergence.py')
LEG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEG)


class ToothLadder(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / 'configs/leg1-n8-forward.json').read_text())
        self.cfg = json.loads((ROOT / 'configs/leg16-tooth-convergence.json').read_text())

    def test_registered_ladder_and_fixture_provenance(self):
        for filename in ('leg16-tooth-convergence.json', 'leg16-tooth-geometry.json'):
            config = json.loads((ROOT / 'configs' / filename).read_text())
            LEG.validate_config(config, self.base)
            self.assertTrue(config['input_basis'])
            widths = [config['width_um'] / row['dx_um'] for row in config['cases']]
            for actual, expected in zip(widths, (3.15, 6.3, 12.6, 25.2)):
                self.assertAlmostEqual(actual, expected)

    def test_fixture_drift_and_control_drift_are_rejected(self):
        for target, key in ((self.cfg, 'wavelength_um'), (self.cfg['cases'][2], 'z_cells'),
                            (self.cfg['cases'][1], 'transition_nm')):
            original = target[key]
            target[key] = original * 2
            with self.assertRaises(ValueError):
                LEG.validate_config(self.cfg, self.base)
            target[key] = original
        self.cfg['expected']['cases'][next(iter(self.cfg['expected']['cases']))]['T'] += .01
        with self.assertRaises(ValueError):
            LEG.validate_config(self.cfg, self.base)

    def test_richardson_recovers_known_limits_at_measured_orders(self):
        for order in (1., 1.5, 2., 3.):
            # Analytic manufactured sequence with known continuum limit 0.625.
            values = [.625 + .125 * 2 ** (-order * rung) for rung in range(4)]
            stats = LEG.ladder_statistics(values, complete=True)
            self.assertTrue(stats['C1_pass'])
            self.assertAlmostEqual(stats['finest_order'], order)
            self.assertAlmostEqual(stats['richardson_extrapolated'], .625)

    def test_partial_three_rungs_use_observed_order(self):
        stats = LEG.ladder_statistics([1., .5, .25])
        self.assertEqual(stats['finest_order'], 1.)
        self.assertAlmostEqual(stats['richardson_extrapolated'], 0.)
        self.assertTrue(stats['C1_pass'])
        self.assertFalse(stats['all_four_rungs'])

    def test_non_monotonicity_forbids_fit(self):
        for values in ([.5, .6, .55], [1., .9, .7]):
            stats = LEG.ladder_statistics(values)
            self.assertTrue(stats['non_monotone'])
            self.assertEqual(stats['observed_orders'], [])
            self.assertIsNone(stats['richardson_extrapolated'])

    def test_constant_or_insufficient_data_does_not_invent_order(self):
        for values in ([], [1.], [1., 1.], [1., 1., 1., 1.], [1., .5, 0.]):
            self.assertIsNone(LEG.ladder_statistics(values)['richardson_extrapolated'])
        with self.assertRaises(ValueError):
            LEG.ladder_statistics([float('nan')])

    def test_common_report_remains_compatible_with_verification_and_inventory(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from verification import verify_readout, STAGES
        from coverage_audit import classify, family
        full = dict(policy='leg16-test', status='stopped-refusal', cases=[dict(
            id='test-rung', measured=dict(T=.5, R=.1, n_eff=2., epsilon_sha256='abc'),
            shape=[10, 20, 5], steps=100, dt_s=1e-17, wall_seconds=1.,
            launch_identity='identity', launch_scores=[0., 1.])])
        before = json.dumps(full, sort_keys=True)
        readout = LEG.standard_readout(full, STAGES,
                                      {stage: 'leg1_forward imported stage' for stage in STAGES})
        headlines = verify_readout(readout)
        self.assertTrue(headlines)
        for row in headlines:
            name, _ = family(row['path'])
            self.assertNotEqual(classify(name)[0], 'unknown', name)
        self.assertEqual(readout['cases'][0]['measured']['launch_neff'], 2.)
        self.assertEqual(json.dumps(full, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
