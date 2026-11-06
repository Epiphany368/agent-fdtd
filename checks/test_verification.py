"""Decisions and reports tested with this project's synthetic CPU fixtures.

These are schema/arithmetic fixtures, not invented simulation inputs or physical
validation cases. No claim here qualifies as a registered V1/V2/V4 check.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from verification import LABELS, STAGES, verify_readout

spec = importlib.util.spec_from_file_location('platform_report', ROOT / 'tools/report.py')
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.readout = {
            'policy': 'cpu-schema-fixture', 'status': 'completed',
            'simulation_executed': True, 'acceptance_granted': False,
            'stages': list(STAGES), 'stage_source': {stage: 'existing.module.symbol' for stage in STAGES},
            'cases': [{'id': 'case', 'shape': [2, 3, 1], 'steps': 4, 'dt_s': .25,
                       'measured': {'result': 1., 'epsilon_sha256': 'a' * 64},
                       'recorded': {'result': 1., 'epsilon_sha256': 'a' * 64},
                       'relative_error': {'result': 0., 'epsilon_sha256': 0.},
                       'within_tolerance': True}],
        }
        self.config = {
            'policy': 'cpu-schema-fixture',
            'expected': {'source': 'Historical solver run (CPU schema fixture only)',
                         'tolerance_relative': 1e-12,
                         'cases': {'case': copy.deepcopy(self.readout['cases'][0]['recorded'])}},
        }

    def result(self, suffix='measured.result', config=True):
        rows = verify_readout(self.readout, self.config if config else None)
        return next(row for row in rows if row['path'].endswith(suffix))

    def test_reproduced_value_is_consistent_never_verified(self):
        row = self.result()
        self.assertEqual(row['label'], 'consistent')
        self.assertIn('Recomputed relative error 0', row['reason'])
        self.assertIn('no registered V1/V2/V4', row['reason'])

    def test_no_config_means_unchecked_despite_stored_pass(self):
        self.assertEqual(self.result(config=False)['label'], 'unchecked')
        self.assertIn('No archived config', self.result(config=False)['reason'])

    def test_changed_value_does_not_trust_stale_zero_error(self):
        self.readout['cases'][0]['measured']['result'] = 2.
        row = self.result()
        self.assertEqual(row['label'], 'unchecked')
        self.assertIn('Reproduction failed', row['reason'])

    def test_stored_failed_flag_does_not_replace_recalculation(self):
        self.readout['cases'][0]['within_tolerance'] = False
        self.readout['cases'][0]['relative_error']['result'] = 100.
        self.assertEqual(self.result()['label'], 'consistent')

    def test_self_declared_verified_check_cannot_promote(self):
        self.readout['verification'] = {'label': 'verified', 'check': 'V1', 'passed': True}
        self.readout['cases'][0]['checks'] = [{'track': 'V2', 'passed': True}]
        self.assertNotIn('verified', [row['label'] for row in verify_readout(self.readout, self.config)])

    def test_unknown_measured_number_is_unchecked(self):
        self.readout['cases'][0]['measured']['new_result'] = .75
        row = self.result('measured.new_result')
        self.assertEqual(row['label'], 'unchecked')
        self.assertIn('No historical target', row['reason'])

    def test_nested_values_and_arrays_are_discovered(self):
        self.readout['cases'][0]['measured']['new_result'] = {'samples': [1., 2.]}
        rows = [row for row in verify_readout(self.readout, self.config)
                if '.new_result.samples[' in row['path']]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['label'] for row in rows], ['unchecked', 'unchecked'])

    def test_nested_archived_target_can_be_compared(self):
        self.readout['cases'][0]['measured']['vector'] = [1., 2.]
        self.config['expected']['cases']['case']['vector'] = [1., 2.]
        self.assertEqual(self.result('measured.vector[1]')['label'], 'consistent')

    def test_numeric_metadata_has_no_check(self):
        self.readout['extra_metric'] = 8
        self.readout['cases'][0]['diagnostic'] = {'runtime': 9.}
        rows = verify_readout(self.readout, self.config)
        metadata = [row for row in rows if '.measured.' not in row['path']]
        self.assertEqual(len(metadata), 7)
        self.assertTrue(all(row['label'] == 'unchecked' for row in metadata))
        self.assertTrue(all('metadata' in row['reason'] for row in metadata))

    def test_boolean_status_is_not_a_numeric_headline(self):
        rows = verify_readout(self.readout, self.config)
        self.assertFalse(any(isinstance(row['value'], bool) for row in rows))
        self.assertFalse(any('.recorded.' in row['path'] or '.relative_error.' in row['path']
                             for row in rows))

    def test_exact_digest_is_consistent_identity_only(self):
        row = self.result('measured.epsilon_sha256')
        self.assertEqual(row['label'], 'consistent')
        self.assertIn('identity only', row['reason'])

    def test_digest_mismatch_or_invalid_digest_is_unchecked(self):
        for value in ('b' * 64, 'invalid', 123):
            with self.subTest(value=value):
                self.readout['cases'][0]['measured']['epsilon_sha256'] = value
                self.assertEqual(self.result('measured.epsilon_sha256')['label'], 'unchecked')

    def test_nonfinite_or_nonnumeric_target_never_passes(self):
        for measured, target in ((float('nan'), 1.), (float('inf'), 1.),
                (1., float('nan')), (1., float('inf')), (1., True), (1., '1')):
            with self.subTest(measured=measured, target=target):
                self.readout['cases'][0]['measured']['result'] = measured
                self.config['expected']['cases']['case']['result'] = target
                self.readout['cases'][0]['recorded']['result'] = target
                self.assertEqual(self.result()['label'], 'unchecked')

    def test_missing_invalid_tolerance_cannot_grant_consistency(self):
        for tolerance in (None, -1., float('nan'), float('inf'), True, '1e-12'):
            with self.subTest(tolerance=tolerance):
                self.config['expected']['tolerance_relative'] = tolerance
                self.assertEqual(self.result()['label'], 'unchecked')

    def test_comparison_tolerance_boundary_and_zero_target(self):
        self.config['expected']['tolerance_relative'] = .25
        self.readout['cases'][0]['measured']['result'] = 1.25
        self.assertEqual(self.result()['label'], 'consistent')
        self.readout['cases'][0]['measured']['result'] = 1.5
        self.assertEqual(self.result()['label'], 'unchecked')
        self.config['expected']['cases']['case']['result'] = 0.
        self.readout['cases'][0]['recorded']['result'] = 0.
        self.readout['cases'][0]['measured']['result'] = .25
        row = self.result()
        self.assertEqual(row['label'], 'consistent')
        self.assertIn('absolute error (zero target)', row['reason'])

    def test_tiny_power_difference_does_not_pass_as_absolute_error(self):
        self.readout['cases'][0]['measured']['result'] = 4e-42
        self.readout['cases'][0]['recorded']['result'] = 3e-42
        self.config['expected']['cases']['case']['result'] = 3e-42
        self.assertEqual(self.result()['label'], 'unchecked')

    def test_archived_policy_mismatch_is_unchecked(self):
        self.config['policy'] = 'different-policy'
        self.assertEqual(self.result()['label'], 'unchecked')

    def test_missing_source_is_unchecked(self):
        del self.config['expected']['source']
        self.assertEqual(self.result()['label'], 'unchecked')

    def test_missing_expected_or_case_is_unchecked(self):
        self.config['expected']['cases'] = {}
        self.assertEqual(self.result()['label'], 'unchecked')
        self.config['expected'] = None
        self.assertEqual(self.result()['label'], 'unchecked')

    def test_conflicting_recorded_target_is_unchecked(self):
        self.readout['cases'][0]['recorded']['result'] = 2.
        self.assertEqual(self.result()['label'], 'unchecked')
        self.assertIn('contradicts', self.result()['reason'])

    def test_missing_stored_recorded_still_uses_archived_target(self):
        del self.readout['cases'][0]['recorded']
        self.assertEqual(self.result()['label'], 'consistent')

    def test_wrong_stage_order_is_rejected(self):
        self.readout['stages'].reverse()
        with self.assertRaisesRegex(ValueError, 'ordered F1-F6'):
            verify_readout(self.readout, self.config)

    def test_missing_stage_source_is_rejected(self):
        del self.readout['stage_source'][STAGES[0]]
        with self.assertRaisesRegex(ValueError, 'implementation source'):
            verify_readout(self.readout, self.config)

    def test_bad_cases_and_duplicate_ids_are_rejected(self):
        for cases in (None, [1], [{'measured': {}}],
                [self.readout['cases'][0], self.readout['cases'][0]]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                verify_readout(dict(self.readout, cases=cases), self.config)

    def test_nonobject_measured_is_rejected(self):
        for value in (1, 'bad', None, [1., 2.]):
            with self.subTest(value=value):
                self.readout['cases'][0]['measured'] = value
                with self.assertRaisesRegex(ValueError, 'measured object'):
                    verify_readout(self.readout, self.config)

    def test_every_headline_has_allowed_label_and_reason_without_mutation(self):
        original_readout, original_config = copy.deepcopy(self.readout), copy.deepcopy(self.config)
        rows = verify_readout(self.readout, self.config)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row['label'] in LABELS and row['reason'] for row in rows))
        self.assertEqual((self.readout, self.config), (original_readout, original_config))

    def test_report_is_generic_read_only_and_labels_all_rows(self):
        self.readout['policy'] = self.config['policy'] = 'another-config-policy'
        self.readout['cases'][0]['measured']['new_result'] = .5
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'readout.json').write_text(json.dumps(self.readout))
            (run / 'config.input.json').write_text(json.dumps(self.config))
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            report = report_module.render_report(run)
            after = {path.name: path.read_bytes() for path in run.iterdir()}
        self.assertEqual(before, after)
        self.assertIn('another-config-policy', report)
        self.assertEqual(report.count('[consistent]'), 2)
        self.assertEqual(report.count('[unchecked]'), 6)
        self.assertNotIn('[verified]', report)
        positions = [report.index(f'  {stage}:') for stage in STAGES]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('F7-verify: verification.verify_readout', report)
        self.assertIn('F8-report: tools/report.py.render_report', report)

    def test_report_without_config_prints_unchecked(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'readout.json').write_text(json.dumps(self.readout))
            report = report_module.render_report(run)
        self.assertEqual(report.count('[unchecked]'), 7)
        self.assertNotIn('[consistent]', report)

    def test_cli_rejects_bad_stage_contract_with_nonzero_exit(self):
        self.readout['stages'] = []
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'readout.json').write_text(json.dumps(self.readout))
            result = subprocess.run([sys.executable, '-B', 'tools/report.py', directory],
                                    cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('ordered F1-F6', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_cli_missing_readout_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', 'tools/report.py', directory],
                                    cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('report:', result.stderr)


if __name__ == '__main__':
    unittest.main()
