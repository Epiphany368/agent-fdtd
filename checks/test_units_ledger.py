"""Leg 5 CPU tests: the units ledger vocabulary, coverage, diff and controls.

Arithmetic and decision logic only. No solver and no GPU.
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import units_ledger as ul


def matched(data):
    """The archived config that agrees with this readout's own declaration."""
    return dict(units=data['units'])


def declaration(unit='dimensionless'):
    return dict(unit=unit, plane='a plane', span='a span', normalisation='a normalisation')


def readout(case_id='c', unit='dimensionless', declared=True, extra=None):
    measured = dict(R=0.5, T=0.5)
    if extra:
        measured.update(extra)
    units = {case_id: {key: declaration(unit) for key in measured}} if declared else None
    body = dict(cases=[dict(id=case_id, ppw=40, steps=10, dt_s=1e-17, measured=measured)])
    if units is not None:
        body['units'] = units
    return body


class Vocabulary(unittest.TestCase):
    def test_the_vocabulary_is_closed_and_small(self):
        self.assertIn('dimensionless', ul.VOCABULARY)
        self.assertIn('sha256', ul.VOCABULARY)
        self.assertLessEqual(len(ul.VOCABULARY), 12)

    def test_a_known_unit_passes(self):
        ok, reason = ul.check_declaration(declaration('micrometre'))
        self.assertTrue(ok, reason)

    def test_an_unknown_unit_fails(self):
        ok, reason = ul.check_declaration(declaration('furlong'))
        self.assertFalse(ok)
        self.assertIn('vocabulary', reason)

    def test_a_missing_field_fails(self):
        entry = declaration()
        entry.pop('span')
        ok, reason = ul.check_declaration(entry)
        self.assertFalse(ok)
        self.assertIn('span', reason)

    def test_an_empty_field_fails(self):
        entry = declaration()
        entry['plane'] = '   '
        ok, _ = ul.check_declaration(entry)
        self.assertFalse(ok)

    def test_a_non_object_fails(self):
        ok, _ = ul.check_declaration('dimensionless')
        self.assertFalse(ok)


class Coverage(unittest.TestCase):
    def test_a_fully_declared_readout_passes(self):
        data = readout()
        result = ul.evaluate_ledger(data, matched(data))
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['diff'], 'match')
        keys = {entry['key'] for entry in result['cases'][0]['declared']}
        self.assertEqual(keys, {'R', 'T'})

    def test_an_undeclared_measured_number_fails(self):
        data = readout()
        data['units']['c'].pop('T')
        result = ul.evaluate_ledger(data, {})
        self.assertTrue(any('undeclared' in failure for failure in result['failures']))

    def test_a_readout_without_a_declaration_is_reported_as_pre_contract(self):
        result = ul.evaluate_ledger(readout(declared=False), {})
        self.assertEqual(result['diff'], 'absent')
        self.assertEqual(result['failures'], [])
        self.assertIn('predates', result['note'])

    def test_metadata_keys_are_exempt_but_must_be_in_the_table(self):
        result = ul.evaluate_ledger(readout(), {})
        exempt = {entry['key'] for entry in result['cases'][0]['exempt']}
        self.assertEqual(exempt, {'ppw', 'steps', 'dt_s'})
        self.assertNotIn('R', exempt)

    def test_an_unknown_metadata_key_is_not_exempt(self):
        data = readout()
        data['cases'][0]['mystery'] = 3.0
        result = ul.evaluate_ledger(data, {})
        self.assertTrue(any('undeclared' in failure for failure in result['failures']))

    def test_every_exemption_carries_a_reason(self):
        for key, (unit, reason) in ul.EXEMPT.items():
            self.assertIn(unit, ul.VOCABULARY, key)
            self.assertGreater(len(reason), 8, key)

    def test_a_declaration_is_reported_as_declared_only(self):
        result = ul.evaluate_ledger(readout(), {})
        entry = result['cases'][0]['declared'][0]
        self.assertEqual(entry['cross_check'], 'declared-only')
        self.assertIn('Stop condition 1', entry['cross_check_reason'])


class Diff(unittest.TestCase):
    def test_a_matching_archived_config_gives_no_failure(self):
        data = readout()
        result = ul.evaluate_ledger(data, dict(units=data['units']))
        self.assertEqual(result['diff'], 'match')
        self.assertEqual(result['failures'], [])

    def test_a_disagreeing_archived_config_fails(self):
        data = readout()
        config = dict(units={'c': {'R': declaration('micrometre'), 'T': declaration()}})
        result = ul.evaluate_ledger(data, config)
        self.assertEqual(result['diff'], 'mismatch')
        self.assertTrue(result['failures'])

    def test_a_config_without_units_is_a_mismatch_not_a_silent_pass(self):
        result = ul.evaluate_ledger(readout(), {})
        self.assertEqual(result['diff'], 'mismatch')


class OneKeyOneUnit(unittest.TestCase):
    def test_one_key_with_one_unit_passes(self):
        data = readout()
        data['units']['c']['R'] = declaration('dimensionless')
        self.assertEqual(ul.evaluate_ledger(data, matched(data))['failures'], [])

    def test_the_same_key_with_two_units_fails(self):
        data = readout(case_id='c1')
        second = readout(case_id='c2')
        data['cases'].append(second['cases'][0])
        data['units'].update(second['units'])
        data['units']['c2']['R'] = declaration('micrometre')
        result = ul.evaluate_ledger(data, {})
        self.assertTrue(any('more than one unit' in failure for failure in result['failures']))

    def test_key_units_are_reported(self):
        result = ul.evaluate_ledger(readout(), {})
        self.assertEqual(result['key_units']['R'], ['dimensionless'])


class Shape(unittest.TestCase):
    def test_the_declaration_covers_arrays_by_key(self):
        data = readout()
        data['cases'][0]['measured']['wavelength_um'] = [1.2, 1.35, 1.5]
        data['units']['c']['wavelength_um'] = declaration('micrometre')
        result = ul.evaluate_ledger(data, matched(data))
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['key_units']['wavelength_um'], ['micrometre'])

    def test_an_array_without_a_declaration_fails(self):
        data = readout()
        data['cases'][0]['measured']['wavelength_um'] = [1.2, 1.35, 1.5]
        result = ul.evaluate_ledger(data, matched(data))
        self.assertTrue(any('wavelength_um' in failure for failure in result['failures']))

    def test_a_sha256_leaf_is_checked_like_any_other(self):
        data = readout()
        data['cases'][0]['measured']['epsilon_sha256'] = 'a' * 64
        result = ul.evaluate_ledger(data, matched(data))
        self.assertTrue(any('epsilon_sha256' in failure for failure in result['failures']))
        data['units']['c']['epsilon_sha256'] = declaration('sha256')
        self.assertEqual(ul.evaluate_ledger(data, matched(data))['failures'], [])


class Controls(unittest.TestCase):
    def setUp(self):
        self.data = readout()
        self.config = dict(units=copy.deepcopy(self.data['units']))

    def test_the_unmodified_readout_passes(self):
        result = ul.evaluate_ledger(self.data, self.config)
        self.assertEqual(result['failures'], [])

    def test_dropping_a_declaration_is_caught_by_coverage_and_by_the_diff(self):
        dropped = ul.drop_declaration(self.data, 'c', 'T')
        result = ul.evaluate_ledger(dropped, self.config)
        self.assertGreaterEqual(len(result['failures']), 2)
        self.assertTrue(any('undeclared' in failure for failure in result['failures']))
        self.assertTrue(any('differs' in failure for failure in result['failures']))

    def test_dropping_a_declaration_keeps_the_number(self):
        dropped = ul.drop_declaration(self.data, 'c', 'T')
        self.assertIn('T', dropped['cases'][0]['measured'])
        self.assertNotIn('T', dropped['units']['c'])

    def test_inventing_a_unit_is_caught(self):
        invented = ul.invent_unit(self.data, 'c', 'R')
        result = ul.evaluate_ledger(invented, self.config)
        self.assertTrue(any('vocabulary' in failure for failure in result['failures']))

    def test_the_source_readout_is_not_mutated(self):
        before = copy.deepcopy(self.data)
        ul.drop_declaration(self.data, 'c', 'T')
        ul.invent_unit(self.data, 'c', 'R')
        self.assertEqual(self.data, before)


class Config(unittest.TestCase):
    def test_the_leg_config_declares_every_measured_key(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg5-units-ledger.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        case_id = config['cases'][0]['id']
        declared = config['units'][case_id]
        for key, entry in declared.items():
            ok, reason = ul.check_declaration(entry)
            self.assertTrue(ok, f'{key}: {reason}')



class Leg13RegistryTest(unittest.TestCase):
    """Leg 13 (S4): the recovered readout definitions are complete and can fail."""

    SIX = (('leg1-objective-v1', 'J'), ('leg1-objective-v1', 'P_up_reduced'),
           ('leg1-objective-v1', 'P_in_reduced'), ('leg1-objective-v1', 'unit_mode_flux_reduced'),
           ('leg2-invariants-v1', 'reference_power_reduced'),
           ('leg5-units-ledger-v1', 'reference_power_reduced'), ('leg6-fibre-overlap-v1', 'eta_lat'))

    def test_every_registry_entry_is_complete_and_names_its_source(self):
        for pair, entry in ul.READOUT_DEFINITIONS.items():
            ok, reason = ul.check_registered(entry)
            self.assertTrue(ok, f'{pair}: {reason}')
            self.assertIn(entry['unit'], ul.VOCABULARY, pair)

    def test_the_six_leg11_fields_are_registered(self):
        for pair in self.SIX:
            self.assertIn(pair, ul.READOUT_DEFINITIONS, pair)

    def test_monitor_names_exist_in_the_declared_geometry(self):
        for pair, entry in ul.READOUT_DEFINITIONS.items():
            for monitor in entry['monitors']:
                self.assertIn(monitor, ul.DECLARED_GEOMETRY[entry['family']], pair)

    def test_audit_passes_on_the_registry(self):
        result = ul.audit_definitions(ul.READOUT_DEFINITIONS.keys())
        self.assertTrue(result['passed'])
        self.assertEqual(result['count']['registered'], len(ul.READOUT_DEFINITIONS))

    def test_audit_flags_a_missing_and_an_invalid_entry(self):
        registry = copy.deepcopy(ul.READOUT_DEFINITIONS)
        registry.pop(('leg1-objective-v1', 'J'))
        registry[('leg1-objective-v1', 'P_in_reduced')]['plane'] = ''
        result = ul.audit_definitions(ul.READOUT_DEFINITIONS.keys(), registry=registry)
        self.assertFalse(result['passed'])
        self.assertEqual(result['count']['missing'], 1)
        self.assertEqual(result['count']['invalid'], 1)

    def test_an_invented_field_is_missing_not_defaulted(self):
        observed = list(ul.READOUT_DEFINITIONS) + [('leg1-objective-v1', 'invented_ratio')]
        self.assertEqual(ul.audit_definitions(observed)['count']['missing'], 1)

    def test_unknown_is_answered_but_counted_apart(self):
        registry = {('p', 'x'): dict(status='unknown', reason='the basis is unrecorded')}
        result = ul.audit_definitions([('p', 'x')], registry=registry)
        self.assertTrue(result['passed'])
        self.assertEqual(result['count']['unknown'], 1)
        registry[('p', 'x')]['reason'] = ' '
        self.assertFalse(ul.audit_definitions([('p', 'x')], registry=registry)['passed'])

    def test_unknown_text_inside_a_recovered_entry_is_rejected(self):
        entry = dict(ul.READOUT_DEFINITIONS[('leg1-objective-v1', 'J')])
        entry['span'] = ul.UNKNOWN
        self.assertFalse(ul.check_registered(entry)[0])

    def test_a_readout_declaration_alone_counts_as_declared(self):
        pair = ('legX', 'T')
        result = ul.audit_definitions([pair], {pair: declaration()}, registry={})
        self.assertEqual(result['count']['declared'], 1)
        self.assertTrue(result['passed'])

    def test_vocabulary_stays_closed_and_small(self):
        self.assertIn('degree', ul.VOCABULARY)
        self.assertIn('per-micrometre', ul.VOCABULARY)
        self.assertLessEqual(len(ul.VOCABULARY), 12)


class Leg13GeometryTest(unittest.TestCase):
    NEAR = ul.DECLARED_GEOMETRY['n8-scene']['near']

    def test_matching_monitor_passes_with_frame_offset(self):
        monitor = ('x', (-4.54, 0.12, 0.0), (0.0, 3.6, 0.2))
        self.assertEqual(ul.cross_check_monitor(self.NEAR, monitor, (13.48, 6.0, 0.2),
                                                (3.74, 0, 0), 0.01), [])

    def test_snapped_plane_gets_half_a_cell_and_no_more(self):
        inside = ('x', (-4.536, 0.12, 0.0), (0.0, 3.6, 0.2))
        outside = ('x', (-4.53, 0.12, 0.0), (0.0, 3.6, 0.2))
        args = ((13.48, 6.0, 0.2), (3.74, 0, 0), 0.01)
        self.assertEqual(ul.cross_check_monitor(self.NEAR, inside, *args), [])
        self.assertTrue(ul.cross_check_monitor(self.NEAR, outside, *args))

    def test_wrong_normal_and_wrong_span_fail(self):
        args = ((13.48, 6.0, 0.2), (3.74, 0, 0), 0.01)
        self.assertTrue(ul.cross_check_monitor(self.NEAR, ('y', (-4.54, 0.12, 0), (0, 3.6, 0.2)), *args))
        self.assertTrue(ul.cross_check_monitor(self.NEAR, ('x', (-4.54, 0.12, 0), (0, 3.0, 0.2)), *args))

    def test_end_and_mid_references_resolve(self):
        far = ul.DECLARED_GEOMETRY['n8-scene']['far']
        top = ul.DECLARED_GEOMETRY['n8-scene']['top']
        end = 7.46325
        self.assertEqual(ul.cross_check_monitor(far, ('x', (8.26, 0.12, 0), (0, 3.6, 0.2)), (1, 1, 0.2),
                                                cell_um=0.01, end_um=end), [])
        self.assertEqual(ul.cross_check_monitor(top, ('y', ((-0.8 + end + 0.8) / 2, 1.5, 0),
                                                      (end + 1.6, 0, 0.2)), (1, 1, 0.2),
                                                near_x=-0.8, far_x=end + 0.8), [])


class Leg13ControlsTest(unittest.TestCase):

    def test_energy_closure_passes_and_fails(self):
        self.assertTrue(ul.energy_closure([0.3, 0.4], [0.7, 0.6], 1e-6)['passed'])
        failed = ul.energy_closure([0.3, 0.4], [0.69, 0.6], 1e-6)
        self.assertFalse(failed['passed'])
        self.assertAlmostEqual(failed['value'], 0.01)

    def test_symmetric_source_passes_and_reports_margin(self):
        result = ul.symmetric_source_ratio([1.0, 2.0, 1e-6], [1.0, 2.0004, 1e-6], 1e-3)
        self.assertTrue(result['passed'])
        self.assertEqual(result['frequencies_dropped'], 1)
        self.assertGreater(result['margin'], 1)

    def test_one_way_source_fails_the_symmetric_gate(self):
        result = ul.symmetric_source_ratio([1.0, 2.0], [1e-9, 2e-9], 1e-3)
        self.assertFalse(result['passed'])
        self.assertGreater(result['deviation'], 1e6)

    def test_exactly_zero_backward_fails_without_dividing(self):
        result = ul.symmetric_source_ratio([1.0, 2.0], [0.0, 0.0], 1e-3)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['deviation'])


class Leg13TermsTest(unittest.TestCase):

    def test_transition_term_reproduces_the_recorded_5_6_points(self):
        verdict = ul.transition_term_verdict()
        self.assertAlmostEqual(verdict['our_loss_pt'], 14.813, places=3)
        self.assertAlmostEqual(verdict['vendor_loss_pt_if_net_basis'], 9.206, places=3)
        self.assertAlmostEqual(verdict['difference_pt_if_net_basis'], 5.606, places=3)

    def test_transition_term_is_unbounded_because_the_vendor_basis_is_unknown(self):
        self.assertEqual(ul.transition_term_verdict()['verdict'], 'unbounded')
        filled = copy.deepcopy(ul.TRANSITION_TERM)
        for key in ('plane_denominator', 'span', 'normalisation'):
            filled['vendor'][key] = 'hypothetically recorded'
        self.assertEqual(ul.transition_term_verdict(filled)['verdict'], 'bounded')

    def test_the_conditional_envelope_is_wider_than_the_headline(self):
        verdict = ul.transition_term_verdict()
        self.assertGreater(verdict['difference_pt_if_modal_basis'], verdict['difference_pt_if_net_basis'])

    def test_radiation_bound_excludes_the_withdrawn_range(self):
        bound = ul.radiation_term_bound()
        self.assertAlmostEqual(bound['alpha_upper_bound_np_per_mm'], 3.393, places=3)
        self.assertTrue(bound['excludes_claimed_range'])


if __name__ == '__main__':
    unittest.main()
