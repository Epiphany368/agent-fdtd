"""Leg-17 registration, missing-data and strict decision-boundary checks."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SPEC = importlib.util.spec_from_file_location('leg17', ROOT / 'src/leg17_transition_probe.py')
LEG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEG)


def rows(values):
    return {k: dict(measured=dict(T=v, n_eff=v)) for k, v in zip('abcdefg', values)}


class TransitionProbe(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'configs/leg17-transition-probe.json').read_text())
        self.base = json.loads((ROOT / 'configs/leg16-tooth-convergence.json').read_text())

    def test_fixture_and_true_constant_resolution_sets(self):
        LEG.validate_config(self.cfg, self.base)
        for s, ids in LEG.MATCHED.items():
            for name in ids:
                dx, _, transition = LEG.LATTICE[name]
                self.assertEqual(transition/(1000*dx), float(s))

    def test_changes_to_fixed_inputs_or_lattice_are_rejected(self):
        changes = [('core_index', None), ('precision', None), ('duration_fs', 2),
                   ('periods', 3), ('z_cells', 4), ('shift_nm', 0), ('transition_nm', 1),
                   ('dx_um', 4), ('dy_um', 3)]
        for key, case in changes:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                target = cfg if case is None else cfg['cases'][case]
                target[key] = 'float32' if key == 'precision' else target[key] + 1
                with self.assertRaises(ValueError):
                    LEG.validate_config(cfg, self.base)
        for key in ('cases', 'anchors'):
            cfg = copy.deepcopy(self.cfg)
            cfg[key].pop()
            with self.assertRaises(ValueError):
                LEG.validate_config(cfg, self.base)

    def test_maximum_single_step_is_not_endpoint_or_sum(self):
        result = LEG.differences(tuple('dbe'), rows([0, 3, 0, 0, 1, 0, 0]), 'T')
        self.assertEqual([r['signed'] for r in result['increments']], [3, -2])
        self.assertEqual(result['maximum_absolute'], 3)

    def test_explained_requires_every_set_for_both_observables(self):
        data = rows([-10, 0, 10, -10, 10, 0, 0])
        self.assertEqual(LEG.classify(data, True)['label'], 'explained')
        data['a']['measured']['n_eff'] = -100
        self.assertEqual(LEG.classify(data, True)['label'], 'inconclusive')

    def test_not_driver_requires_strict_factor_three_everywhere(self):
        data = rows([-100, 0, 100, -1, 1, -100, 100])
        self.assertEqual(LEG.classify(data, True)['label'], 'not-the-driver')
        data['a']['measured']['T'] = -4  # equality: 3*Delta_s == Delta_mesh
        self.assertEqual(LEG.classify(data, True)['label'], 'inconclusive')

    def test_equality_is_inconclusive(self):
        self.assertEqual(LEG.classify(rows([0]*7), True)['label'], 'inconclusive')

    def test_missing_case_never_bridges_or_uses_archived_b_as_control(self):
        data = rows([-10, 0, 10, -10, 10, 0, 0])
        del data['b']
        result = LEG.classify(data, True)
        self.assertEqual(result['label'], 'unevaluable')
        self.assertIsNone(result['observables']['T']['mesh']['1.0']['maximum_absolute'])
        self.assertEqual(result['observables']['T']['mesh']['0.5']['maximum_absolute'], 0)
        self.assertEqual(LEG.classify(rows([0]*7), False)['label'], 'unevaluable')

    def test_nonfinite_measurement_is_rejected(self):
        for bad in (float('nan'), float('inf')):
            data = rows([0]*7)
            data['d']['measured']['T'] = bad
            with self.assertRaises(ValueError):
                LEG.classify(data, True)

    def test_control_requires_numeric_tolerance_and_exact_material_hash(self):
        measured = copy.deepcopy(next(iter(self.cfg['expected']['cases'].values())))
        self.assertTrue(LEG.control_result(measured, self.cfg['expected'])['passed'])
        measured['T'] += 1e-8
        self.assertFalse(LEG.control_result(measured, self.cfg['expected'])['passed'])
        measured = copy.deepcopy(next(iter(self.cfg['expected']['cases'].values())))
        measured['epsilon_sha256'] = 'different'
        self.assertFalse(LEG.control_result(measured, self.cfg['expected'])['passed'])


if __name__ == '__main__':
    unittest.main()
