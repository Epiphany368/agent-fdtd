"""Leg 3 CPU tests: the analytic laws, the scope rule, and the label coverage.

These tests pin THIS project's contracts. They do not copy the reference project's suite.
They cover arithmetic and decision logic only; the GPU forward is not exercised here.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import analytic_cases as ac


def readout(case_id, wavelength, R, T):
    return dict(stages=list(('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                             'F5-run', 'F6-readout')),
                stage_source={stage: 'test' for stage in
                              ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                               'F5-run', 'F6-readout')},
                cases=[dict(id=case_id, measured=dict(wavelength_um=list(wavelength),
                                                      R=list(R), T=list(T)))])


def config_for(law, case_id, index, tolerance, **extra):
    entry = dict(id=ac.V1_IDS[0] if law == 'fresnel' else ac.V1_IDS[1], law=law,
                 case_id=case_id, index=index, tolerance=tolerance,
                 tolerance_kind='test', basis='test basis')
    entry.update(extra)
    return dict(analytic_cases=[entry])


class Laws(unittest.TestCase):
    def test_fresnel_energy_conserved(self):
        for index in (1.0, 1.5, 3.5, 12.0):
            R, T = ac.fresnel_rt(index)
            self.assertAlmostEqual(R + T, 1.0, places=15)

    def test_fresnel_matches_the_cited_form(self):
        index = 3.5
        self.assertAlmostEqual(ac.fresnel_rt(index)[0], ((index - 1) / (index + 1)) ** 2, places=15)

    def test_fresnel_is_wavelength_independent(self):
        wavelength = [1.2, 1.5, 1.8]
        R, T = ac._law_values(dict(law='fresnel', index=2.0), wavelength)
        self.assertEqual(len(R), 3)
        self.assertEqual(len(set(R)), 1)
        self.assertEqual(len(set(T)), 1)

    def test_airy_energy_conserved_everywhere(self):
        for step in range(1, 41):
            R, T = ac.airy_rt(3.5, 1.2 + 0.015 * step)
            self.assertAlmostEqual(R + T, 1.0, places=15)

    def test_airy_is_total_transmission_at_half_integer_optical_depths(self):
        # sin(2 pi d / lambda) = 0 when 2 d / lambda is an integer.
        for m in (2, 3, 4, 5):
            R, T = ac.airy_rt(3.5, 2 * ac.OPTICAL_THICKNESS_UM / m)
            self.assertAlmostEqual(T, 1.0, places=12)
            self.assertAlmostEqual(R, 0.0, places=12)

    def test_airy_minimum_transmission_matches_the_closed_form(self):
        index = 3.5
        # sin^2 = 1 when 4 d / lambda is odd.
        wavelength = 4 * ac.OPTICAL_THICKNESS_UM / 3
        contrast = ((index ** 2 - 1) / (2 * index)) ** 2
        self.assertAlmostEqual(ac.airy_rt(index, wavelength)[1], 1 / (1 + contrast), places=12)

    def test_unknown_law_is_refused(self):
        with self.assertRaises(ValueError):
            ac._law_values(dict(law='nope', index=1.5), [1.2])

    def test_nonpositive_index_is_refused(self):
        with self.assertRaises(ValueError):
            ac._law_values(dict(law='fresnel', index=0), [1.2])


class Scope(unittest.TestCase):
    def test_every_law_states_its_scope(self):
        for case_id in ac.V1_IDS:
            self.assertIn(case_id, ac.SCOPE)
            self.assertGreater(len(ac.SCOPE[case_id]), 40)

    def test_the_two_unimplemented_cases_are_recorded_with_a_reason(self):
        ids = {entry['id'] for entry in ac.DROPPED}
        self.assertEqual(ids, {'V1-slab-mode', 'V1-mie'})
        for entry in ac.DROPPED:
            self.assertIn('Stop condition', entry['reason'])
            self.assertTrue(entry['source'])

    def test_the_symmetric_slab_law_is_not_implemented_as_an_approximation(self):
        # Criterion C3: a symmetric-slab law must not be silently applied elsewhere. Leg 14's
        # scope-matched slab law lives in SCALAR_LAWS and only judges named scalar fields.
        self.assertNotIn('slab-mode', ac.LAWS)
        self.assertNotIn('mie', ac.LAWS)


class Decision(unittest.TestCase):
    def setUp(self):
        self.wavelength = [1.2, 1.35, 1.5]
        self.exact_R, self.exact_T = ac._law_values(dict(law='fresnel', index=3.5), self.wavelength)

    def exact_readout(self, delta=0.0):
        return readout('c', self.wavelength, self.exact_R,
                       [value + delta for value in self.exact_T])

    def test_exact_curves_pass(self):
        checks = ac.evaluate_analytic_cases(self.exact_readout(),
                                            config_for('fresnel', 'c', 3.5, 0.01))
        self.assertEqual([check['status'] for check in checks], ['passed'])
        self.assertAlmostEqual(checks[0]['error'], 0.0, places=15)

    def test_error_just_inside_tolerance_passes(self):
        checks = ac.evaluate_analytic_cases(self.exact_readout(0.009),
                                            config_for('fresnel', 'c', 3.5, 0.01))
        self.assertEqual(checks[0]['status'], 'passed')

    def test_error_outside_tolerance_fails(self):
        checks = ac.evaluate_analytic_cases(self.exact_readout(0.02),
                                            config_for('fresnel', 'c', 3.5, 0.01))
        self.assertEqual(checks[0]['status'], 'failed')

    def test_worst_point_is_reported_with_its_wavelength(self):
        T = list(self.exact_T)
        T[1] += 0.03
        checks = ac.evaluate_analytic_cases(readout('c', self.wavelength, self.exact_R, T),
                                            config_for('fresnel', 'c', 3.5, 0.01))
        self.assertEqual(checks[0]['worst']['key'], 'T')
        self.assertEqual(checks[0]['worst']['point'], 1)
        self.assertEqual(checks[0]['worst']['wavelength_um'], 1.35)

    def test_a_case_without_a_readout_case_is_not_executed(self):
        checks = ac.evaluate_analytic_cases(self.exact_readout(),
                                            config_for('fresnel', 'missing', 3.5, 0.01))
        self.assertEqual(checks[0]['status'], 'not-executed')

    def test_no_declared_cases_means_no_rows(self):
        self.assertEqual(ac.evaluate_analytic_cases(self.exact_readout(), {}), [])
        self.assertEqual(ac.evaluate_analytic_cases(self.exact_readout(), None), [])


class Labels(unittest.TestCase):
    def headlines(self, readout_object):
        rows = []
        for index, case in enumerate(readout_object['cases']):
            for key, curve in (('R', case['measured']['R']), ('T', case['measured']['T'])):
                for point, value in enumerate(curve):
                    rows.append(dict(path=f'cases[{index}].measured.{key}[{point}]',
                                     case_id=case['id'], value=value, label='unchecked',
                                     reason='no check'))
        return rows

    def setUp(self):
        self.wavelength = [1.2, 1.35, 1.5]
        self.exact_R, self.exact_T = ac._law_values(dict(law='fresnel', index=3.5), self.wavelength)

    def test_a_passing_case_verifies_every_covered_point(self):
        data = readout('c', self.wavelength, self.exact_R, self.exact_T)
        rows = ac.label_analytic_cases(data, config_for('fresnel', 'c', 3.5, 0.01),
                                       self.headlines(data))
        verified = [row for row in rows if row['label'] == 'verified']
        self.assertEqual(len(verified), 6)
        self.assertTrue(all('V1-fresnel' in row['invariants'] for row in verified))

    def test_a_failing_case_verifies_nothing(self):
        data = readout('c', self.wavelength, self.exact_R,
                       [value + 0.03 for value in self.exact_T])
        rows = ac.label_analytic_cases(data, config_for('fresnel', 'c', 3.5, 0.01),
                                       self.headlines(data))
        self.assertEqual([row for row in rows if row['label'] == 'verified'], [])

    def test_a_failing_case_verifies_nothing_even_where_a_point_passes(self):
        # One point is out of tolerance, the other five are inside it. The case fails, so no
        # point from it is promoted: we do not know why it failed.
        T = list(self.exact_T)
        T[1] += 0.03
        data = readout('c', self.wavelength, self.exact_R, T)
        rows = ac.label_analytic_cases(data, config_for('fresnel', 'c', 3.5, 0.02),
                                       self.headlines(data))
        self.assertEqual([row for row in rows if row['label'] == 'verified'], [])
        checks = ac.evaluate_analytic_cases(data, config_for('fresnel', 'c', 3.5, 0.02))
        self.assertEqual(checks[0]['status'], 'failed')

    def test_no_declared_case_leaves_labels_untouched(self):
        data = readout('c', self.wavelength, self.exact_R, self.exact_T)
        rows = self.headlines(data)
        self.assertEqual(ac.label_analytic_cases(data, {}, rows), rows)
        self.assertEqual([row for row in rows if row['label'] == 'verified'], [])


class NegativeControl(unittest.TestCase):
    def setUp(self):
        self.wavelength = [1.2, 1.35, 1.5]
        self.exact_R, self.exact_T = ac._law_values(dict(law='fresnel', index=3.5), self.wavelength)
        self.data = readout('c', self.wavelength, self.exact_R, self.exact_T)

    def test_perturbation_makes_the_check_fail_by_the_offset(self):
        entry = config_for('fresnel', 'c', 3.5, 0.01)['analytic_cases'][0]
        entry['negative_control_key'] = 'T'
        perturbed = ac.negative_control(entry, self.data, 0.25)
        checks = ac.evaluate_analytic_cases(perturbed, config_for('fresnel', 'c', 3.5, 0.01))
        self.assertEqual(checks[0]['status'], 'failed')
        self.assertAlmostEqual(checks[0]['error'], 0.25, places=12)

    def test_the_source_readout_is_not_mutated(self):
        entry = config_for('fresnel', 'c', 3.5, 0.01)['analytic_cases'][0]
        before = copy.deepcopy(self.data)
        ac.negative_control(entry, self.data, 0.25)
        self.assertEqual(self.data, before)

    def test_the_unperturbed_case_still_passes(self):
        entry = config_for('fresnel', 'c', 3.5, 0.01)['analytic_cases'][0]
        perturbed = ac.negative_control(entry, self.data, 0.25)
        checks = ac.evaluate_analytic_cases(perturbed, config_for('airy', 'c', 3.5, 0.01))
        # The Airy law is not the law these curves follow, so this must NOT pass.
        self.assertEqual(checks[0]['status'], 'failed')

    def test_the_leg_config_declares_the_tolerance_kind_and_basis(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'configs/leg3-analytic-cases.json'
        if not path.is_file():
            self.skipTest('leg config not present')
        config = json.loads(path.read_text())
        for entry in config['analytic_cases']:
            self.assertTrue(entry['tolerance_kind'])
            self.assertTrue(entry['basis'])
            self.assertTrue(entry['scope_match'])
            self.assertTrue(entry['scope_match_reason'])
        self.assertEqual(len(config['dropped_analytic_cases']), 2)


# ---------------------------------------------------------------- leg 14
def scalar_readout(case_id, **measured):
    return dict(stages=list(('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                             'F5-run', 'F6-readout')),
                stage_source={stage: 'test' for stage in
                              ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors',
                               'F5-run', 'F6-readout')},
                cases=[dict(id=case_id, measured=dict(measured))])


GRATING = dict(id='V1-grating-zero-order', law='film-tp', case_id='g', index=2.0,
               thickness_um=0.6, wavelength_um=1.54925, tolerance=0.01,
               tolerance_kind='test', basis='test basis',
               points=[dict(field='zero_order_efficiency_a', angle_deg=8.5),
                       dict(field='zero_order_efficiency_b', angle_deg=8.9002)])
SLAB = dict(id='V1-slab-mode', law='slab-tm0', case_id='s', core_index=3.476558727709872,
            cladding_index=1.4440231286977714, thickness_um=0.22, wavelength_um=1.54925,
            tolerance=0.00202, tolerance_kind='test', basis='test basis',
            points=[dict(field='launch_neff')])


class Leg14Laws(unittest.TestCase):
    def test_film_law_reduces_to_airy_at_normal_incidence(self):
        for wavelength in (1.2, 1.4, 1.54925, 1.8):
            self.assertAlmostEqual(ac.film_tp(2.0, 0.6, wavelength, 0.0),
                                   ac.airy_rt(2.0, wavelength, 1.2)[1], places=13)
            self.assertAlmostEqual(ac.film_ts(2.0, 0.6, wavelength, 0.0),
                                   ac.film_tp(2.0, 0.6, wavelength, 0.0), places=13)

    def test_film_law_is_passive_and_full_at_resonance(self):
        for angle in (0.0, 8.5, 30.0, 60.0):
            self.assertLessEqual(ac.film_tp(2.0, 0.6, 1.5, angle), 1.0 + 1e-15)
        # beta = m pi gives T = 1 (half-wave film), at any angle.
        angle = 20.0
        cos_t = math.sqrt(1 - (math.sin(math.radians(angle)) / 2.0) ** 2)
        wavelength = 2 * 2.0 * 0.6 * cos_t / 3
        self.assertAlmostEqual(ac.film_tp(2.0, 0.6, wavelength, angle), 1.0, places=12)

    def test_film_p_is_full_at_brewster(self):
        # r_p = 0 at tan(theta) = n: the film is transparent whatever its thickness.
        brewster = math.degrees(math.atan(2.0))
        self.assertAlmostEqual(ac.film_tp(2.0, 0.6, 1.54925, brewster), 1.0, places=12)
        self.assertLess(ac.film_ts(2.0, 0.6, 1.54925, brewster), 0.99)

    def test_tm0_root_satisfies_the_full_thickness_form(self):
        # Independent form of the same boundary conditions, over the whole thickness:
        # tan(kappa d) = 2 p q / (p^2 - q^2), p = kappa / n1^2, q = gamma / n2^2.
        n1, n2, d, lam = 3.476558727709872, 1.4440231286977714, 0.22, 1.54925
        neff = ac.slab_tm0_neff(n1, n2, d, lam)
        k0 = 2 * math.pi / lam
        kappa, gamma = k0 * math.sqrt(n1 ** 2 - neff ** 2), k0 * math.sqrt(neff ** 2 - n2 ** 2)
        p, q = kappa / n1 ** 2, gamma / n2 ** 2
        lhs = math.sin(kappa * d) * (p ** 2 - q ** 2)
        rhs = 2 * p * q * math.cos(kappa * d)
        self.assertAlmostEqual(lhs, rhs, delta=1e-9 * abs(rhs))
        self.assertTrue(n2 < neff < n1)

    def test_tm0_is_below_te0_and_tends_to_the_cladding_for_a_thin_core(self):
        n1, n2, lam = 3.476558727709872, 1.4440231286977714, 1.54925
        thin = ac.slab_tm0_neff(n1, n2, 0.01, lam)
        thick = ac.slab_tm0_neff(n1, n2, 2.0, lam)
        self.assertLess(thin - n2, 0.01)
        self.assertGreater(thick, 3.3)
        # TE0 of the same slab (tan(kappa d/2) = gamma/kappa) must sit above TM0.
        k0, half = 2 * math.pi / lam, 0.11
        low, high = n2 + 1e-12, n1 - 1e-12
        for _ in range(200):
            mid = (low + high) / 2
            kappa, gamma = k0 * math.sqrt(n1 ** 2 - mid ** 2), k0 * math.sqrt(mid ** 2 - n2 ** 2)
            if math.sin(kappa * half) * kappa - gamma * math.cos(kappa * half) > 0:
                low = mid
            else:
                high = mid
        self.assertGreater(low, ac.slab_tm0_neff(n1, n2, 0.22, lam) + 0.5)

    def test_unguided_slab_is_refused(self):
        with self.assertRaises(ValueError):
            ac.slab_tm0_neff(1.4, 1.5, 0.22, 1.55)

    def test_the_new_laws_state_their_scope_and_the_drop_is_recorded(self):
        for case_id in ac.SCALAR_IDS:
            self.assertGreater(len(ac.SCOPE[case_id]), 40)
        self.assertEqual([entry['id'] for entry in ac.LEG14_DROPPED], ['V1-grating-modulated'])
        self.assertIn('Stop condition 1', ac.LEG14_DROPPED[0]['reason'])

    def test_leg3_laws_are_untouched(self):
        self.assertEqual(set(ac.LAWS), {'fresnel', 'airy'})
        self.assertEqual(ac.V1_IDS, ('V1-fresnel', 'V1-airy'))


class Leg14Decision(unittest.TestCase):
    def exact(self, entry):
        return dict(ac.scalar_law_points(entry))

    def test_exact_values_pass_and_report_margin(self):
        values = self.exact(GRATING)
        rows = ac.evaluate_analytic_cases(scalar_readout('g', **{k: v + 0.004 for k, v in
                                                                 values.items()}),
                                          dict(analytic_cases=[GRATING]))
        self.assertEqual(rows[0]['status'], 'passed')
        self.assertAlmostEqual(rows[0]['margin'], 2.5, places=9)

    def test_outside_tolerance_fails(self):
        values = self.exact(SLAB)
        rows = ac.evaluate_analytic_cases(scalar_readout('s', launch_neff=values['launch_neff']
                                                         + 0.003),
                                          dict(analytic_cases=[SLAB]))
        self.assertEqual(rows[0]['status'], 'failed')

    def test_a_missing_field_fails_rather_than_passing(self):
        values = self.exact(GRATING)
        rows = ac.evaluate_analytic_cases(
            scalar_readout('g', zero_order_efficiency_a=values['zero_order_efficiency_a']),
            dict(analytic_cases=[GRATING]))
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertEqual(rows[0]['worst']['key'], 'zero_order_efficiency_b')

    def test_missing_case_is_not_executed(self):
        rows = ac.evaluate_analytic_cases(scalar_readout('other', launch_neff=2.0),
                                          dict(analytic_cases=[SLAB]))
        self.assertEqual(rows[0]['status'], 'not-executed')

    def test_zero_tolerance_never_passes(self):
        entry = dict(SLAB, tolerance=0)
        value = self.exact(SLAB)['launch_neff']
        rows = ac.evaluate_analytic_cases(scalar_readout('s', launch_neff=value),
                                          dict(analytic_cases=[entry]))
        self.assertEqual(rows[0]['status'], 'failed')


class Leg14Labels(unittest.TestCase):
    def headlines(self, data):
        return [dict(path=f'cases[0].measured.{key}', case_id=data['cases'][0]['id'], value=v,
                     label='unchecked', reason='no check')
                for key, v in data['cases'][0]['measured'].items()]

    def test_a_pass_verifies_only_the_named_fields(self):
        values = self.exact = dict(ac.scalar_law_points(GRATING))
        data = scalar_readout('g', zero_order_forward_power=3e-41, **values)
        rows = ac.label_analytic_cases(data, dict(analytic_cases=[GRATING]), self.headlines(data))
        labels = {row['path'].split('.')[-1]: row['label'] for row in rows}
        self.assertEqual(labels, {'zero_order_efficiency_a': 'verified',
                                  'zero_order_efficiency_b': 'verified',
                                  'zero_order_forward_power': 'unchecked'})

    def test_a_failure_verifies_nothing(self):
        values = dict(ac.scalar_law_points(GRATING))
        values['zero_order_efficiency_b'] += 0.02
        data = scalar_readout('g', **values)
        rows = ac.label_analytic_cases(data, dict(analytic_cases=[GRATING]), self.headlines(data))
        self.assertEqual([r for r in rows if r['label'] == 'verified'], [])


class Leg14NegativeControls(unittest.TestCase):
    def test_offset_control_fails_and_leaves_the_source_alone(self):
        value = dict(ac.scalar_law_points(SLAB))['launch_neff']
        data = scalar_readout('s', launch_neff=value)
        before = copy.deepcopy(data)
        perturbed = ac.negative_control_scalar(SLAB, data, 0.25)
        self.assertEqual(data, before)
        rows = ac.evaluate_analytic_cases(perturbed, dict(analytic_cases=[SLAB]))
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertAlmostEqual(rows[0]['error'], 0.25, places=12)

    def test_wrong_polarisation_law_fails_on_p_data(self):
        values = dict(ac.scalar_law_points(GRATING))
        rows = ac.evaluate_analytic_cases(scalar_readout('g', **values),
                                          dict(analytic_cases=[dict(GRATING, law='film-ts')]))
        self.assertEqual(rows[0]['status'], 'failed')

    def test_an_asymmetric_stack_is_outside_the_symmetric_law(self):
        # The air-clad control value measured by run 20261018T132628Z-638525 (dx 0.01, dy 0.005)
        # must fail the symmetric law: the case cannot pass a stack outside its scope.
        rows = ac.evaluate_analytic_cases(scalar_readout('s', launch_neff=1.8932165380882775),
                                          dict(analytic_cases=[SLAB]))
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertGreater(rows[0]['error'], 50 * SLAB['tolerance'])


class Leg14Configs(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1] / 'configs'

    def load(self, name):
        path = self.ROOT / name
        if not path.is_file():
            self.skipTest(f'{name} not present')
        return json.loads(path.read_text())

    def test_each_case_declares_scope_tolerance_and_controls_and_stores_no_value(self):
        for name, case_id in (('leg14-grating-zero-order.json', 'V1-grating-zero-order'),
                              ('leg14-slab-check.json', 'V1-slab-mode')):
            entries = self.load(name)['analytic_cases']
            self.assertEqual([entry['id'] for entry in entries], [case_id])
            entry = entries[0]
            for key in ('tolerance_kind', 'basis', 'scope_match', 'scope_match_reason'):
                self.assertTrue(entry[key])
            self.assertGreaterEqual(len(entry['negative_controls']), 1)
            text = str(entry)
            for forbidden in ('exact', 'expected', 'analytic_value'):
                self.assertNotIn(f"'{forbidden}'", text)
            ac.scalar_law_points(entry)  # the law is computable from the declared inputs

    def test_the_slab_tolerance_is_the_declared_floor(self):
        floor = self.load('leg14-slab-floor.json')['slab']
        check = self.load('leg14-slab-check.json')
        self.assertEqual(floor['fixture_dx_dy_um'], check['slab']['fixture_dx_dy_um'])
        self.assertNotIn('analytic_cases', floor)


if __name__ == '__main__':
    unittest.main()
