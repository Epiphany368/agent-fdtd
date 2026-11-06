"""CPU arithmetic/schema tests. Synthetic arrays are not simulation evidence."""
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
from invariants import evaluate_invariants, verify_invariants
from verification import STAGES

spec = importlib.util.spec_from_file_location('leg2_report', ROOT / 'tools/report.py')
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)


class InvariantTests(unittest.TestCase):
    def setUp(self):
        self.readout = {
            'policy': 'cpu-invariant-fixture', 'stages': list(STAGES),
            'stage_source': {stage: 'cpu.fixture' for stage in STAGES},
            'cases': [{'id': 'empty', 'measured': {'R': [.25, .125], 'T': [.75, .875],
                       'energy_residual': 0., 'J': .75}, 'passed': True}],
        }
        self.config = {
            'policy': self.readout['policy'],
            'cases': [{'id': 'empty', 'geometry': 'empty', 'physics': {
                'family': 'closed-lossless-plane', 'lossless': True,
                'all_exits_monitored': True, 'symmetric_source': True,
                'symmetric_case': True, 'linear_reciprocal': True}}],
            'invariants': [{'id': 'V2-a', 'case_id': 'empty', 'tolerance': 1e-12,
                            'basis': 'Synthetic arithmetic budget declared before evaluation.'},
                           {'id': 'V2-b', 'case_id': 'empty', 'tolerance': 1e-12,
                            'basis': 'Synthetic arithmetic budget declared before evaluation.'}],
        }

    def check(self, index=0):
        return evaluate_invariants(self.readout, self.config)[index]

    def measured(self):
        return self.readout['cases'][0]['measured']

    def registration(self, invariant):
        registration = copy.deepcopy(self.config['invariants'][0])
        registration['id'] = invariant
        self.config['invariants'] = [registration]
        return registration

    def labels(self):
        return {row['path']: row for row in verify_invariants(self.readout, self.config)}

    def test_energy_recomputes_full_array_ignoring_stored_pass_and_residual(self):
        self.measured()['T'][1] = .9
        row = self.check()
        self.assertEqual(row['status'], 'failed')
        self.assertAlmostEqual(row['error'], .025)
        self.assertIn('cases[0].measured.T[1]', row['covers'])

    def test_energy_roundoff_pass_and_coverage_is_exact(self):
        row = self.check()
        self.assertEqual(row['status'], 'passed')
        self.assertEqual(row['error'], 0)
        labels = self.labels()
        self.assertEqual(labels['cases[0].measured.T[0]']['label'], 'verified')
        # Leg 12: the stored residual equals the recomputed closure, so V2-a now grants it.
        self.assertEqual(labels['cases[0].measured.energy_residual']['label'], 'verified')
        self.assertEqual(labels['cases[0].measured.J']['label'], 'unchecked')

    def test_guided_fixtures_are_rejected_even_with_closed_flags(self):
        for fixture in ('G8/N8 fixture', 'guided mode fixture'):
            with self.subTest(fixture=fixture):
                self.readout['fixture'] = fixture
                self.assertEqual(self.check()['status'], 'dropped')
                self.assertFalse(self.check()['lawful'])

    def test_guided_case_id_is_rejected(self):
        self.readout['cases'][0]['id'] = 'g3-g8-nh-n8-x10-y5-z5'
        self.config['cases'][0]['id'] = self.readout['cases'][0]['id']
        self.config['invariants'][0]['case_id'] = self.readout['cases'][0]['id']
        self.assertEqual(self.check()['status'], 'dropped')

    def test_energy_requires_closed_lossless_every_exit_declarations(self):
        for key in ('family', 'lossless', 'all_exits_monitored'):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                del config['cases'][0]['physics'][key]
                self.assertEqual(evaluate_invariants(self.readout, config)[0]['status'], 'dropped')

    def test_energy_whitelists_existing_closed_family(self):
        for geometry in ('empty', 'slab', 'interface', 'grating'):
            with self.subTest(geometry=geometry):
                self.config['cases'][0]['geometry'] = geometry
                expected = 'dropped' if geometry == 'grating' else 'passed'
                self.assertEqual(self.check()['status'], expected)

    def test_passivity_all_bounds_and_finiteness(self):
        self.registration('V2-b')
        for value, error in ((-.2, .2), (1.25, .25), (0., 0.), (1., 0.)):
            with self.subTest(value=value):
                self.measured()['T'][0] = value
                row = self.check()
                self.assertAlmostEqual(row['error'], error)
                self.assertEqual(row['status'], 'passed' if error == 0 else 'failed')
        for value in (float('nan'), float('inf'), True, '1'):
            with self.subTest(value=value):
                self.measured()['R'][0] = value
                self.assertEqual(self.check()['status'], 'failed')

    def test_passivity_is_lawful_on_guided_case(self):
        self.registration('V2-b')
        self.config['cases'][0] = {'id': 'empty', 'geometry': 'guided'}
        self.assertEqual(self.check()['status'], 'passed')

    def test_missing_monitors_are_dropped(self):
        del self.measured()['T']
        self.assertEqual(self.check()['status'], 'dropped')
        self.assertIn('stop condition 1', self.check()['reason'])

    def test_malformed_array_shapes_or_empty_arrays_fail(self):
        for value in ([], [.75], [[.75, .875]]):
            with self.subTest(value=value):
                self.measured()['T'] = value
                self.assertEqual(self.check()['status'], 'failed')

    def test_scalar_readout_supported(self):
        self.measured().update(R=.25, T=.75)
        self.assertEqual(self.check()['status'], 'passed')
        self.assertEqual(self.check()['covers'], ['cases[0].measured.R', 'cases[0].measured.T',
                                                  'cases[0].measured.energy_residual'])

    def test_tolerance_boundary_is_inclusive_without_widening(self):
        self.config['invariants'][0]['tolerance'] = .125
        self.measured()['T'][0] = .875
        self.assertEqual(self.check()['status'], 'passed')
        self.measured()['T'][0] = 1.
        self.assertEqual(self.check()['status'], 'failed')

    def test_invalid_or_missing_tolerance_and_basis_do_not_verify(self):
        for value in (None, -1., float('nan'), float('inf'), True, '1e-12'):
            with self.subTest(value=value):
                self.config['invariants'][0]['tolerance'] = value
                self.assertEqual(self.check()['status'], 'invalid')
                self.assertNotIn('verified', [row['label'] for row in self.labels().values()])
        self.config['invariants'][0]['tolerance'] = 1e-12
        for value in (None, '', 42):
            with self.subTest(basis=value):
                self.config['invariants'][0]['basis'] = value
                self.assertEqual(self.check()['status'], 'invalid')

    def test_policy_match_and_registration_are_required(self):
        self.config['policy'] = 'other'
        self.assertEqual(self.check()['status'], 'invalid')
        self.assertNotIn('verified', [row['label'] for row in self.labels().values()])
        self.assertEqual(evaluate_invariants(self.readout, None), [])
        self.config['invariants'] = []
        self.assertNotIn('verified', [row['label'] for row in self.labels().values()])

    def test_no_registered_case_or_unknown_invariant_cannot_pass(self):
        self.config['cases'] = []
        self.assertEqual(self.check()['status'], 'invalid')
        self.config['invariants'][0]['id'] = 'V2-invented'
        self.assertEqual(self.check()['status'], 'invalid')

    def test_duplicate_registration_is_rejected(self):
        self.config['invariants'].append(copy.deepcopy(self.config['invariants'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.check()

    def test_failed_covering_check_blocks_passing_check(self):
        self.measured()['T'][0] = .5  # passive but not conserved
        self.assertEqual(self.check(1)['status'], 'passed')
        self.assertEqual(self.check()['status'], 'failed')
        row = self.labels()['cases[0].measured.T[0]']
        self.assertEqual(row['label'], 'unchecked')
        self.assertIn('V2-a', row['reason'])

    def test_symmetric_power_relative_arithmetic_and_coverage(self):
        self.registration('V2-c')
        self.measured().update(forward_power=[2e-30, 3e-30], backward_power=[2e-30, 3e-30])
        self.assertEqual(self.check()['status'], 'passed')
        self.measured()['forward_power'][0] = 3e-30
        self.assertEqual(self.check()['status'], 'failed')
        self.assertAlmostEqual(self.check()['error'], .5)
        self.assertNotIn('cases[0].measured.T[0]', self.check()['covers'])

    def test_symmetric_power_requires_positive_nonzero_data(self):
        self.registration('V2-c')
        for value in (0., -1.):
            with self.subTest(value=value):
                self.measured().update(forward_power=[value], backward_power=[value])
                self.assertEqual(self.check()['status'], 'failed')

    def test_symmetric_power_missing_data_or_asymmetry_is_dropped(self):
        self.registration('V2-c')
        self.assertEqual(self.check()['status'], 'dropped')
        self.measured().update(forward_power=[1.], backward_power=[1.])
        self.config['cases'][0]['physics']['symmetric_source'] = False
        self.assertEqual(self.check()['status'], 'dropped')
        self.config['cases'][0]['physics']['symmetric_source'] = True
        self.config['cases'][0]['physics']['symmetric_case'] = False
        self.assertEqual(self.check()['status'], 'dropped')
        self.config['cases'][0]['physics']['symmetric_case'] = True
        self.config['cases'][0]['geometry'] = 'interface'
        self.assertEqual(self.check()['status'], 'dropped')

    def reciprocity(self):
        registration = self.registration('V2-d')
        ports = {'forward': {'source': 'left', 'monitor': 'right'},
                 'reverse': {'source': 'right', 'monitor': 'left'}}
        registration['ports'] = copy.deepcopy(ports)
        self.measured()['reciprocity'] = {'forward': [2e-30, 0.], 'reverse': [2e-30, 0.],
                                        'ports': ports}

    def test_reciprocity_relative_error_including_both_zero(self):
        self.reciprocity()
        self.assertEqual(self.check()['status'], 'passed')
        self.measured()['reciprocity']['reverse'][0] = 1e-30
        self.assertEqual(self.check()['status'], 'failed')
        self.assertAlmostEqual(self.check()['error'], .5)

    def test_reciprocity_needs_exchanged_registered_and_measured_ports(self):
        self.reciprocity()
        self.measured()['reciprocity']['ports']['reverse']['source'] = 'third'
        self.assertEqual(self.check()['status'], 'dropped')
        self.reciprocity()
        self.config['invariants'][0]['ports']['reverse']['monitor'] = 'third'
        self.assertEqual(self.check()['status'], 'dropped')

    def test_reciprocity_missing_data_and_nonlinear_cases_are_dropped(self):
        self.reciprocity()
        del self.measured()['reciprocity']['reverse']
        self.assertEqual(self.check()['status'], 'dropped')
        self.reciprocity()
        self.config['cases'][0]['physics']['linear_reciprocal'] = False
        self.assertEqual(self.check()['status'], 'dropped')

    def test_no_input_mutation_and_stored_failure_is_ignored(self):
        self.readout['cases'][0]['passed'] = False
        before = copy.deepcopy((self.readout, self.config))
        self.assertEqual(self.check()['status'], 'passed')
        self.labels()
        self.assertEqual((self.readout, self.config), before)

    def test_report_names_invariants_and_keeps_uncovered_numbers_unchecked(self):
        self.config['dropped_invariants'] = [{'id': 'V2-c', 'reason': 'Missing symmetric monitors.'}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'readout.json').write_text(json.dumps(self.readout))
            (run / 'config.input.json').write_text(json.dumps(self.config))
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            report = report_module.render_report(run)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})
        self.assertEqual(report.count('[verified V2-a,V2-b]'), 4)
        self.assertIn('measured.J (case empty) = 0.75 [unchecked]', report)
        self.assertIn('Dropped invariant (not executed)', report)
        self.assertIn('does not establish quantitative accuracy', report)
        self.assertIn('lawful=True; error=0.0; tolerance=1e-12', report)

    def test_negative_cli_detects_corruption_and_leaves_run_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'readout.json').write_text(json.dumps(self.readout))
            (run / 'config.input.json').write_text(json.dumps(self.config))
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            result = subprocess.run([sys.executable, '-B', 'src/invariants.py', directory,
                                     '--negative-control'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})
        self.assertEqual(result.returncode, 1)
        self.assertIn('T[0] 0.75 -> 1.25', result.stdout)
        self.assertIn('V2-a case=empty status=failed error=0.5', result.stdout)
        self.assertIn('V2-b case=empty status=failed error=0.25', result.stdout)


class Leg12WiringTests(unittest.TestCase):
    """Leg 12: V2-a grants energy_residual and empty_reflection_floor only by arithmetic."""
    check, measured, labels = InvariantTests.check, InvariantTests.measured, InvariantTests.labels

    def setUp(self):
        InvariantTests.setUp(self)
        self.measured()['empty_reflection_floor'] = 1e-13

    def test_both_energy_fields_verified_when_recomputed(self):
        row = self.check()
        self.assertEqual(row['status'], 'passed')
        self.assertEqual(row['error'], 1e-13)
        labels = self.labels()
        for leaf in ('energy_residual', 'empty_reflection_floor'):
            self.assertEqual(labels[f'cases[0].measured.{leaf}']['label'], 'verified')
            self.assertEqual(labels[f'cases[0].measured.{leaf}']['invariants'], ['V2-a'])

    def test_tampered_stored_residual_is_not_granted(self):
        self.measured()['energy_residual'] = 1e-15
        row = self.check()
        self.assertEqual(row['status'], 'passed')
        self.assertNotIn('cases[0].measured.energy_residual', row['covers'])
        label = self.labels()['cases[0].measured.energy_residual']
        self.assertEqual(label['label'], 'unchecked')
        self.assertIn('not trusted', label['reason'])

    def test_non_finite_residual_is_not_granted(self):
        self.measured()['energy_residual'] = float('nan')
        self.assertEqual(self.labels()['cases[0].measured.energy_residual']['label'], 'unchecked')

    def test_floor_above_budget_fails_the_balance_and_grants_nothing(self):
        self.measured()['empty_reflection_floor'] = 2e-12
        row = self.check()
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error'], 2e-12)
        verified = [path for path, item in self.labels().items() if item['label'] == 'verified']
        # V2-b (passivity) still covers R and T; nothing V2-a covers is verified.
        self.assertNotIn('cases[0].measured.empty_reflection_floor', verified)
        self.assertNotIn('cases[0].measured.energy_residual', verified)
        self.assertNotIn('cases[0].measured.T[0]', verified)

    def test_non_finite_floor_fails(self):
        self.measured()['empty_reflection_floor'] = float('inf')
        self.assertEqual(self.check()['status'], 'failed')

    def test_failing_closure_grants_neither_field(self):
        self.measured()['T'][1] = .9
        self.measured()['energy_residual'] = .025
        labels = self.labels()
        self.assertEqual(labels['cases[0].measured.energy_residual']['label'], 'unchecked')
        self.assertEqual(labels['cases[0].measured.empty_reflection_floor']['label'], 'unchecked')

    def test_field_name_alone_grants_nothing(self):
        # Same names, no V2-a registration: nothing is verified (C4).
        self.config['invariants'] = [r for r in self.config['invariants'] if r['id'] != 'V2-a']
        labels = self.labels()
        self.assertEqual(labels['cases[0].measured.energy_residual']['label'], 'unchecked')
        self.assertEqual(labels['cases[0].measured.empty_reflection_floor']['label'], 'unchecked')
        # Same names, V2-a registered but unlawful (lossy): dropped, nothing verified.
        self.setUp()
        self.config['cases'][0]['physics']['lossless'] = False
        self.assertEqual(self.check()['status'], 'dropped')
        labels = self.labels()
        self.assertEqual(labels['cases[0].measured.energy_residual']['label'], 'unchecked')

    def test_stored_pass_flag_is_ignored(self):
        self.readout['cases'][0]['passed'] = True
        self.readout['cases'][0]['measured']['T'][0] = .5
        self.assertEqual(self.check()['status'], 'failed')
        self.assertEqual(self.labels()['cases[0].measured.energy_residual']['label'], 'unchecked')

    def test_leg12_negative_control_closure(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from leg12_closed_box import negative_control_rows
        lossy = copy.deepcopy(self.readout)
        lossy['cases'][0]['id'] = 'lossy'
        lossy['cases'][0]['measured'].update(R=[.2, .2], T=[.7, .6], energy_residual=.2)
        config = copy.deepcopy(self.config)
        config['cases'][0].update(id='lossy')
        config['cases'][0]['physics']['lossless'] = False
        config['invariants'] = [dict(r, case_id='lossy') for r in config['invariants']]
        rows = negative_control_rows(lossy, config, 'lossy')
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertAlmostEqual(rows[0]['error'], .2)
        # The archived (truthful) config drops the check; it is never "passed".
        self.assertEqual(evaluate_invariants(lossy, config)[0]['status'], 'dropped')

    def test_v2_box_closure_passes_on_lossless_guided_grating(self):
        readout = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-grating',
                'geometry': 'grating',
                'measured': {
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0362995},
                    },
                    'T': 0.5743797,
                    'R': 0.1038960,
                },
            }],
        }
        config = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-grating',
                'geometry': 'grating',
                'physics': {'lossless': True, 'all_exits_monitored': True},
            }],
            'invariants': [{
                'id': 'V2-box',
                'case_id': 'n8-grating',
                'tolerance': 1e-5,
                'basis': 'Poynting 4-surface enclosure closure for lossless guided grating.',
            }],
        }
        rows = evaluate_invariants(readout, config)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['status'], 'passed')
        self.assertTrue(row['lawful'])
        self.assertLessEqual(row['error'], 1e-5)
        self.assertIn('cases[0].measured.monitors.box-right', row['covers'])
        self.assertIn('cases[0].measured.monitors.bottom', row['covers'])

    def test_v2_box_closure_fails_on_unclosed_flux(self):
        readout = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-grating',
                'geometry': 'grating',
                'measured': {
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0862995},  # Unclosed! ~1.05
                    },
                },
            }],
        }
        config = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-grating',
                'geometry': 'grating',
                'physics': {'lossless': True, 'all_exits_monitored': True},
            }],
            'invariants': [{
                'id': 'V2-box',
                'case_id': 'n8-grating',
                'tolerance': 1e-5,
                'basis': 'Poynting 4-surface enclosure closure.',
            }],
        }
        rows = evaluate_invariants(readout, config)
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertGreater(rows[0]['error'], 1e-5)

    def test_v2_box_dropped_if_lossy(self):
        readout = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-lossy',
                'geometry': 'grating',
                'measured': {
                    'box-right': 0.40, 'box-left': 0.10, 'top': 0.15, 'bottom': 0.05,
                },
            }],
        }
        config = {
            'policy': 'leg44-box-policy',
            'cases': [{
                'id': 'n8-lossy',
                'geometry': 'grating',
                'physics': {'lossless': False},
            }],
            'invariants': [{
                'id': 'V2-box',
                'case_id': 'n8-lossy',
                'tolerance': 1e-5,
                'basis': 'Poynting 4-surface enclosure closure.',
            }],
        }
        rows = evaluate_invariants(readout, config)
        self.assertEqual(rows[0]['status'], 'dropped')
        self.assertFalse(rows[0]['lawful'])


if __name__ == '__main__':
    unittest.main()

