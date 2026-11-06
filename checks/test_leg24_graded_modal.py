"""Leg 24 CPU contracts and exercised negative controls; no FDTD run here."""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from parity import Project
from run_b9_native_shape_n8 import scene
from torchfdtd.mesh import configure_auto_mesh, freeze_refinements
from torchfdtd.mode_branches import prepare_aperture_modal_launch
from torchfdtd.mode_injection import _check_launch_signatures
from leg24_graded_modal import compare_default, graded_exit, realised_metrics

CFG = json.loads((ROOT/'configs/leg24-graded-check.json').read_text())


def graded_scene():
    sc = scene(CFG, CFG['cases'][0])
    sc['p'].region.mesh_type = 'graded'
    sc['p'].region.time_step_rule = 'realised_axis_minimum'
    configure_auto_mesh(sc['p'])
    return sc


@lru_cache(maxsize=1)
def launch():
    sc = graded_scene()
    return sc, prepare_aperture_modal_launch(sc['p'], sc['section'], num_modes=2, allow_graded=True)


class ModalSupport(unittest.TestCase):
    def test_graded_remains_default_off(self):
        sc = graded_scene()
        with self.assertRaisesRegex(ValueError, 'uniform'):
            prepare_aperture_modal_launch(sc['p'], sc['section'])

    def test_explicit_switch_required(self):
        sc = graded_scene()
        with self.assertRaises(ValueError):
            prepare_aperture_modal_launch(sc['p'], sc['section'], allow_graded='yes')

    def test_native_grid_really_coarsens(self):
        sc, _ = launch()
        m = realised_metrics(sc['p'].region)
        self.assertLess(m['finest_realised_spacing_um'], .01)
        self.assertLessEqual(m['max_adjacent_ratio'], 1.25)
        self.assertGreater(m['axis_max_step_um'][0], 3*m['axis_min_step_um'][0])
        self.assertLess(np.prod(m['shape']), np.prod(sc['p'].region.base_shape))

    def test_mode_axes_match_realised_yee_samples(self):
        sc, item = launch()
        from torchfdtd.solver import field_axes
        r = sc['p'].region
        for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            for axis, interval, actual in zip((1, 2), item.aperture, item.mode.component_axes(component)):
                expected = field_axes(r, component)[axis][slice(*interval)]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)

    def test_nonuniform_aperture_refused_before_eigensolve(self):
        sc = graded_scene()
        r = sc['p'].region
        arrays = [a.copy() for a in r.mesh_nodes]
        arrays[1][len(arrays[1])//2] += .001
        r._mesh_cache = (r._mesh_cache[0], tuple(arrays))
        with patch('torchfdtd.mode_branches.solve_waveguide_modes') as solve:
            with self.assertRaisesRegex(ValueError, 'constant spacing'):
                prepare_aperture_modal_launch(sc['p'], sc['section'], allow_graded=True)
            solve.assert_not_called()

    def test_detector_uses_real_nodes(self):
        sc, item = launch()
        for m in sc['p'].monitors:
            if m.id in ('near', 'far', 'outside'):
                mode = item.detector_mode(m.center[0])
                self.assertEqual(mode.beta_per_um, item.mode.beta_per_um)

    def test_detector_rejects_coarse_neighbour(self):
        sc, item = launch()
        nodes = sc['p'].region.mesh_nodes[0]
        k = int(np.argmax(np.diff(nodes)))
        with self.assertRaisesRegex(ValueError, 'longitudinal spacing'):
            item.detector_mode(float(nodes[k]))

    def test_detector_rejects_non_node(self):
        sc, item = launch()
        with self.assertRaisesRegex(ValueError, 'realised E nodes'):
            item.detector_mode(sc['p'].sources[0].center[0]+.001)

    def test_signature_matches_frozen_internal_region(self):
        sc, item = launch()
        import torch
        p = freeze_refinements(sc['p'])
        self.assertEqual(item.region_signature, json.dumps(p.region.model_dump(mode='json'), sort_keys=True))
        # meta device verifies the shape/signature guard without allocating a volume.
        _check_launch_signatures(p, item, torch.empty(p.region.shape+(3,), device='meta'))
        p.region.steps += 1
        with self.assertRaisesRegex(ValueError, 'region changed'):
            _check_launch_signatures(p, item, torch.empty(p.region.shape+(3,), device='meta'))


class AcceptanceControls(unittest.TestCase):
    def setUp(self):
        self.row = dict(measured=dict(T=.5, R=.1, epsilon_sha256='epsilon'),
                        field_sha256={'near': 'field'}, mode_sha256='mode', launch_identity='launch',
                        mesh=dict(axis_min_step_um=[.02, .01, .04], max_adjacent_ratio=1.24))

    def test_default_bit_gate_and_negative_controls(self):
        self.assertTrue(compare_default(self.row, self.row, self.row['measured']))
        for key in ('T', 'R'):
            other = copy.deepcopy(self.row)
            other['measured'][key] = np.nextafter(other['measured'][key], 1.)
            self.assertFalse(compare_default(self.row, other, self.row['measured']))
        for key in ('field_sha256', 'mode_sha256', 'launch_identity'):
            other = copy.deepcopy(self.row)
            other[key] = 'changed'
            self.assertFalse(compare_default(self.row, other, self.row['measured']))

    def test_realised_ratio_negative_control(self):
        other = copy.deepcopy(self.row)
        other['mesh']['max_adjacent_ratio'] = np.nextafter(1.25, 2.)
        self.assertEqual(graded_exit(self.row, other, 1e-3, 1.25)[0], 58)

    def test_agreement_negative_control(self):
        other = copy.deepcopy(self.row)
        other['measured']['T'] *= 1.002
        code, errors = graded_exit(self.row, other, 1e-3, 1.25)
        self.assertEqual(code, 57)
        self.assertGreater(errors['T'], 1e-3)
        self.assertEqual(graded_exit(self.row, self.row, 1e-3, 1.25)[0], 0)

    def test_operator_stop_negative_control(self):
        other = copy.deepcopy(self.row)
        other['operator_stop'] = 'Nonuniform transverse eigenproblem requires unsupported metrics.'
        self.assertEqual(graded_exit(self.row, other, 1e-3, 1.25)[0], 55)

    def test_unequal_realised_resolution_refused(self):
        other = copy.deepcopy(self.row)
        other['mesh']['axis_min_step_um'][1] = .009
        self.assertEqual(graded_exit(self.row, other, 1e-3, 1.25)[0], 55)

    def test_physical_inputs_unchanged(self):
        original = json.loads((ROOT/'configs/leg1-n8-forward.json').read_text())
        for filename in ('leg24-uniform-baseline.json', 'leg24-graded-check.json'):
            current = json.loads((ROOT/'configs'/filename).read_text())
            for key in ('cases', 'expected', 'core_index', 'cladding_index', 'wavelength_um', 'width_um', 'precision'):
                self.assertEqual(current[key], original[key])


if __name__ == '__main__':
    unittest.main()
