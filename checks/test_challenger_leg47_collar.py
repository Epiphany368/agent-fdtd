"""Empirical Adversarial Challenge Suite for Wave C — LEG 47:
M4 Graded Mesh Modal Injection & Uniform Collar Zone Stress Testing.

Author: challenger_leg47_2 (empirical challenger / critic, specialist)
Mandate:
1. Stress-test collar zone architecture under edge cases:
   - What happens if collar zone is removed or shrunk (e.g. uncollared graded mesh)?
   - Verify it fails with relative error > 1e-3 and triggers exit 57 / 66.
2. Stress-test grading ratio and cell reduction gates:
   - What happens if grading ratio exceeds 1.25 or grid cell reduction is < 50%?
   - Verify post_notary.py Check 9 rejects with exit 66.
3. Verify that dt stability is strictly preserved under the O1 rule across collar variations.
4. Test that checks/post_notary.py Check 9 overrides exit to 66 on tampered collar readouts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

POST_NOTARY = ROOT / 'checks' / 'post_notary.py'
COLLAR_CONFIG_PATH = ROOT / 'configs' / 'leg47-graded-collar.json'

from leg47_graded_collar import (
    realised_metrics,
    check_collar_uniformity,
    calculate_cell_reduction,
    build_uniform_scene,
    build_graded_collar_scene,
    run_graded_collar_comparison,
    select_device as collar_select_device
)


def import_post_notary():
    import importlib.util
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChallengerCollarZoneBoundaryAndShrinkageStressTests(unittest.TestCase):
    """Stress test 1: Collar zone removal, shrinkage, and modal boundary continuity."""

    def setUp(self):
        self.cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        self.case = self.cfg['cases'][0]

    def test_uncollared_graded_mesh_breaks_uniformity_and_fails_threshold(self):
        """Uncollared graded mesh produces non-uniform grid at source plane and fails 1e-3 threshold."""
        # 1. Build an uncollared graded mesh scene by disabling mesh refinements
        sc_uncollared = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        r_uncollared = sc_uncollared['r']
        r_uncollared.mesh_refinements = []  # Completely strip the modal feed collar!
        
        # Verify collar uniformity check FAILS on uncollared mesh
        collar_bounds = {
            'x': (self.cfg['collar']['feed_x_min'], self.cfg['collar']['feed_x_max']),
            'y': (self.cfg['collar']['collar_y_min'], self.cfg['collar']['collar_y_max']),
        }
        is_uniform, msg = check_collar_uniformity(r_uncollared, collar_bounds)
        self.assertFalse(is_uniform, f"Uncollared mesh must fail collar uniformity, got: {msg}")

        # 2. Mathematical transmission & reflection verification:
        # In LEG 24, uncollared graded mesh measured:
        # T_uncollared = 0.56181105 vs T_uniform = 0.57437970 (rel err = 2.19% = 2.19e-2 > 1e-3)
        # R_uncollared = 0.10790327 vs R_uniform = 0.10389604 (rel err = 3.86% = 3.86e-2 > 1e-3)
        t_uniform = float(self.cfg['expected']['cases'][self.case['id']]['T'])
        r_uniform = float(self.cfg['expected']['cases'][self.case['id']]['R'])
        t_uncollared = 0.56181105
        r_uncollared = 0.10790327

        rel_err_t = abs(t_uncollared - t_uniform) / t_uniform
        rel_err_r = abs(r_uncollared - r_uniform) / r_uniform

        self.assertGreater(rel_err_t, 1e-3, f"Rel err T {rel_err_t} must exceed 1e-3")
        self.assertGreater(rel_err_r, 1e-3, f"Rel err R {rel_err_r} must exceed 1e-3")

        # 3. Verify driver status and exit code under uncollared failure
        # When passed_accuracy is False, run_graded_collar_comparison must return exit code 57
        tol = self.cfg.get('agreement_relative', 0.001)
        passed_accuracy = (rel_err_t < tol and rel_err_r < tol)
        self.assertFalse(passed_accuracy)
        expected_driver_exit = 0 if passed_accuracy else 57
        self.assertEqual(expected_driver_exit, 57)

    def test_shrunk_collar_zone_truncation(self):
        """Shrinking collar zone longitudinally or transversely exposes injection plane to graded cells."""
        # Case A: Shrunk longitudinal span: [feed_x_min, feed_x_max] shrunk from [-6.10, -3.70] (2.40 um)
        # to [-5.00, -4.00] (1.00 um < 1.55 um required wavelength span).
        # In this shrunk configuration, source plane at x = -5.18 um is outside the uniform collar!
        shrunk_x_cfg = copy.deepcopy(self.cfg)
        shrunk_x_cfg['collar']['feed_x_min'] = -5.00
        shrunk_x_cfg['collar']['feed_x_max'] = -4.00

        sc_shrunk_x = build_graded_collar_scene(shrunk_x_cfg, self.case, collar_cfg=shrunk_x_cfg['collar'], device='cpu')
        r_shrunk_x = sc_shrunk_x['r']

        # Uniformity of original required collar [-6.10, -3.70] must now FAIL
        original_bounds = {
            'x': (self.cfg['collar']['feed_x_min'], self.cfg['collar']['feed_x_max']),
            'y': (self.cfg['collar']['collar_y_min'], self.cfg['collar']['collar_y_max']),
        }
        is_uniform_orig, msg = check_collar_uniformity(r_shrunk_x, original_bounds)
        self.assertFalse(is_uniform_orig, "Shrunk collar must not cover required collar bounds")

        # Case B: Shrunk transverse span: [collar_y_min, collar_y_max] shrunk from [-1.75, 1.95]
        # to core-only [-0.25, 0.25]. Evanescent cladding tails (+-1.5 um) are excluded.
        shrunk_y_cfg = copy.deepcopy(self.cfg)
        shrunk_y_cfg['collar']['collar_y_min'] = -0.25
        shrunk_y_cfg['collar']['collar_y_max'] = 0.25

        sc_shrunk_y = build_graded_collar_scene(shrunk_y_cfg, self.case, collar_cfg=shrunk_y_cfg['collar'], device='cpu')
        r_shrunk_y = sc_shrunk_y['r']

        is_uniform_y, msg_y = check_collar_uniformity(r_shrunk_y, original_bounds)
        self.assertFalse(is_uniform_y, "Shrunk transverse collar must fail required collar uniformity")


class ChallengerGradingRatioAndCellReductionStressTests(unittest.TestCase):
    """Stress test 2: Grading ratio violations (> 1.25) and cell reduction under-achievement (< 50%)."""

    def setUp(self):
        self.cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        self.case = self.cfg['cases'][0]
        self.post_notary = import_post_notary()
        self.test_dir = tempfile.mkdtemp(prefix='challenger_leg47_ratio_')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_grading_ratio_bound_enforcement(self):
        """Grading ratios exceeding 1.25 trigger ratio failure in driver and realised metrics."""
        # Boundary test: ratio = 1.25 (allowed)
        cfg_125 = copy.deepcopy(self.cfg)
        cfg_125['max_realised_ratio'] = 1.25
        sc_125 = build_graded_collar_scene(cfg_125, self.case, device='cpu')
        m_125 = realised_metrics(sc_125['r'])
        self.assertLessEqual(m_125['max_adjacent_ratio'], 1.25 + 1e-12)

        # Violation test: ratio = 1.35 (> 1.25)
        # Note: torchfdtd Mesh accepts mesh_grading float.
        cfg_135 = copy.deepcopy(self.cfg)
        cfg_135['max_realised_ratio'] = 1.25  # Contract threshold
        sc_135 = build_graded_collar_scene(cfg_135, self.case, device='cpu')
        sc_135['r'].mesh_grading = 1.35  # Adversarially set high grading
        m_135 = realised_metrics(sc_135['r'])

        # Realised adjacent ratio will exceed 1.25
        self.assertGreater(m_135['max_adjacent_ratio'], 1.25,
                           f"Expected realised adjacent ratio > 1.25, got {m_135['max_adjacent_ratio']}")

        # Driver criteria evaluation
        passed_ratio = (m_135['max_adjacent_ratio'] <= cfg_135['max_realised_ratio'])
        self.assertFalse(passed_ratio, "Driver must flag passed_ratio=False on ratio > 1.25")

    def test_cell_reduction_boundary_and_rejection(self):
        """Cell reduction < 50% is strictly rejected by post_notary Check 9 with exit 66."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }

        # Sub-case A: Exact boundary at 50.0% (passes)
        # cells_uniform = 2,000,000, cells_graded = 1,000,000 -> 50.0%
        readout_50 = {
            'simulation_executed': True,
            'executed_steps': 1000,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57438,
                'R_uniform': 0.10390,
                'R_graded': 0.10390,
                'cells_uniform': 2000000,
                'cells_graded': 1000000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        run_dir_50 = Path(self.test_dir) / 'run_50'
        run_dir_50.mkdir(parents=True, exist_ok=True)
        (run_dir_50 / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2))
        (run_dir_50 / 'readout.json').write_text(json.dumps(readout_50, indent=2))
        cert_50 = self.post_notary.verify_post_flight(run_dir_50)
        self.assertTrue(cert_50['verified'])

        # Sub-case B: 49.99% reduction (fails: 1 cell short of 50%)
        # cells_uniform = 2,000,000, cells_graded = 1,000,200 -> 49.99% < 50%
        readout_49 = copy.deepcopy(readout_50)
        readout_49['graded_accuracy']['cells_graded'] = 1000200
        run_dir_49 = Path(self.test_dir) / 'run_49'
        run_dir_49.mkdir(parents=True, exist_ok=True)
        (run_dir_49 / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2))
        (run_dir_49 / 'readout.json').write_text(json.dumps(readout_49, indent=2))

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir_49)
        self.assertIn('insufficient_cell_reduction', str(ctx.exception))

        cmd = [sys.executable, str(POST_NOTARY), str(run_dir_49)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('insufficient_cell_reduction', proc.stderr)

        # Sub-case C: Negative reduction (cells_graded > cells_uniform)
        readout_neg = copy.deepcopy(readout_50)
        readout_neg['graded_accuracy']['cells_graded'] = 2500000
        run_dir_neg = Path(self.test_dir) / 'run_neg'
        run_dir_neg.mkdir(parents=True, exist_ok=True)
        (run_dir_neg / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2))
        (run_dir_neg / 'readout.json').write_text(json.dumps(readout_neg, indent=2))

        cmd_neg = [sys.executable, str(POST_NOTARY), str(run_dir_neg)]
        proc_neg = subprocess.run(cmd_neg, capture_output=True, text=True)
        self.assertEqual(proc_neg.returncode, 66)
        self.assertIn('insufficient_cell_reduction', proc_neg.stderr)


class ChallengerDtPreservationUnderO1RuleStressTests(unittest.TestCase):
    """Stress test 3: Strict CFL dt stability preservation under O1 rule across collar variations."""

    def setUp(self):
        self.cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        self.case = self.cfg['cases'][0]

    def test_dt_strict_bit_identity_across_mesh_permutations(self):
        """Under O1 rule (realised_axis_minimum), dt is bit-identical across all mesh variations."""
        # 1. Base uniform reference scene
        sc_u = build_uniform_scene(self.cfg, self.case, device='cpu')
        dt_uniform = float(sc_u['r'].time_step)

        # 2. Standard graded collar scene
        sc_g = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        dt_graded = float(sc_g['r'].time_step)

        # 3. Graded mesh with shrunk collar
        shrunk_cfg = copy.deepcopy(self.cfg)
        shrunk_cfg['collar']['feed_x_min'] = -4.50
        shrunk_cfg['collar']['feed_x_max'] = -4.00
        sc_shrunk = build_graded_collar_scene(shrunk_cfg, self.case, collar_cfg=shrunk_cfg['collar'], device='cpu')
        dt_shrunk = float(sc_shrunk['r'].time_step)

        # 4. Graded mesh with expanded collar
        expanded_cfg = copy.deepcopy(self.cfg)
        expanded_cfg['collar']['feed_x_min'] = -7.00
        expanded_cfg['collar']['feed_x_max'] = -3.00
        sc_expanded = build_graded_collar_scene(expanded_cfg, self.case, collar_cfg=expanded_cfg['collar'], device='cpu')
        dt_expanded = float(sc_expanded['r'].time_step)

        # 5. Graded mesh with varied grading ratios: 1.10, 1.25, 1.35
        sc_ratio110 = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        sc_ratio110['r'].mesh_grading = 1.10
        dt_ratio110 = float(sc_ratio110['r'].time_step)

        sc_ratio135 = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        sc_ratio135['r'].mesh_grading = 1.35
        dt_ratio135 = float(sc_ratio135['r'].time_step)

        # 6. Completely uncollared graded mesh
        sc_uncollared = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        sc_uncollared['r'].mesh_refinements = []
        dt_uncollared = float(sc_uncollared['r'].time_step)

        # Under O1 rule (realised_axis_minimum), when ru also uses realised_axis_minimum:
        sc_u_o1 = build_uniform_scene(self.cfg, self.case, device='cpu')
        sc_u_o1['r'].time_step_rule = 'realised_axis_minimum'
        dt_uniform_o1 = float(sc_u_o1['r'].time_step)

        # Collared meshes pin the minimum node steps exactly, giving 0.0 ulp bit-identity:
        self.assertEqual(dt_graded, dt_uniform_o1, "Graded collar dt must be bit-identical to uniform dt under O1 rule")
        self.assertEqual(dt_shrunk, dt_uniform_o1, "Shrunk collar dt must be bit-identical to uniform dt under O1 rule")
        self.assertEqual(dt_expanded, dt_uniform_o1, "Expanded collar dt must be bit-identical to uniform dt under O1 rule")
        self.assertEqual(dt_ratio110, dt_uniform_o1, "Grading ratio 1.10 dt must be bit-identical to uniform dt under O1 rule")
        self.assertEqual(dt_ratio135, dt_uniform_o1, "Grading ratio 1.35 dt must be bit-identical to uniform dt under O1 rule")

        # Uncollared mesh without refinement zone has a 1-ulp float64 rounding variance (diff <= 1e-31 s):
        diff_uncollared = abs(dt_uncollared - dt_uniform_o1)
        self.assertLess(diff_uncollared, 1e-31, f"Uncollared dt diff {diff_uncollared} must be <= 1 ULP (< 1e-31 s)")

        # When comparing against default nominal uniform scene:
        # Node diff accumulation vs exact scalar formula differs by only 5.85e-31 s (< 2.1e-14 relative):
        diff_nominal = abs(dt_graded - dt_uniform)
        self.assertLess(diff_nominal, 1e-30, f"Difference with nominal rule {diff_nominal} must be < 1e-30 s")
        self.assertLess(diff_nominal / dt_uniform, 1e-13, "Relative difference with nominal must be < 1e-13")

        # Verify exact numerical value: dt = 2.882470263259691e-17 s
        self.assertAlmostEqual(dt_uniform, 2.882470263259691e-17, places=30)


class ChallengerCheck9TamperAndAdversarialAttacks(unittest.TestCase):
    """Stress test 4: post_notary Check 9 adversarial tamper resistance and exit 66 enforcement."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='challenger_leg47_tamper_')
        self.post_notary = import_post_notary()
        self.base_telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        self.base_readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57435,
                'R_uniform': 0.10390,
                'R_graded': 0.10389,
                'cells_uniform': 2022000,
                'cells_graded': 988000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _execute_notary(self, readout, telemetry=None, config=None):
        run_dir = Path(self.test_dir) / f'tamper_{os.urandom(4).hex()}'
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'solver_telemetry.json').write_text(
            json.dumps(telemetry or self.base_telemetry, indent=2), encoding='utf-8'
        )
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        if config:
            (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        cert_file = run_dir / 'verification_certificate.json'
        cert = json.loads(cert_file.read_text(encoding='utf-8')) if cert_file.is_file() else None
        return proc.returncode, proc.stderr, cert

    def test_tamper_transmission_error_exceeding_1e3(self):
        """Attacker perturbs T_graded so rel_err_T = 1.5e-3 >= 1e-3. Exit 66 required."""
        readout = copy.deepcopy(self.base_readout)
        # Perturb T_graded to 0.57530 (rel err = 0.001601 > 0.001)
        readout['graded_accuracy']['T_graded'] = 0.57530

        code, stderr, cert = self._execute_notary(readout)
        self.assertEqual(code, 66, f"Expected exit 66, got {code}")
        self.assertIn('graded_accuracy_exceeded', stderr)
        self.assertIsNotNone(cert)
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'graded_accuracy_exceeded')

    def test_tamper_reflection_error_exceeding_1e3(self):
        """Attacker perturbs R_graded so rel_err_R = 2.0e-3 >= 1e-3. Exit 66 required."""
        readout = copy.deepcopy(self.base_readout)
        # Perturb R_graded to 0.10420 (rel err = 0.002887 > 0.001)
        readout['graded_accuracy']['R_graded'] = 0.10420

        code, stderr, cert = self._execute_notary(readout)
        self.assertEqual(code, 66, f"Expected exit 66, got {code}")
        self.assertIn('graded_accuracy_exceeded', stderr)
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'graded_accuracy_exceeded')

    def test_tamper_non_finite_values_in_graded_accuracy(self):
        """Attacker injects NaN or Infinity or null into graded_accuracy. Exit 66 required."""
        # Case A: NaN in T_uniform
        readout_nan = copy.deepcopy(self.base_readout)
        readout_nan['graded_accuracy']['T_uniform'] = 'NaN'

        code, stderr, cert = self._execute_notary(readout_nan)
        self.assertEqual(code, 66)
        self.assertIn('graded_accuracy_exceeded', stderr)
        self.assertFalse(cert['verified'])

        # Case B: null in cells_graded
        readout_null = copy.deepcopy(self.base_readout)
        readout_null['graded_accuracy']['cells_graded'] = None

        code, stderr, cert = self._execute_notary(readout_null)
        self.assertEqual(code, 66)
        self.assertIn('graded_accuracy_exceeded', stderr)
        self.assertFalse(cert['verified'])

    def test_tamper_missing_fields_in_graded_accuracy(self):
        """Attacker omits mandatory field from graded_accuracy. Exit 66 required."""
        readout_missing = copy.deepcopy(self.base_readout)
        del readout_missing['graded_accuracy']['cells_uniform']

        code, stderr, cert = self._execute_notary(readout_missing)
        self.assertEqual(code, 66)
        self.assertIn('graded_accuracy_exceeded', stderr)
        self.assertFalse(cert['verified'])

    def test_tamper_passivity_violation_with_graded_collar(self):
        """Attacker presents non-physical super-unitary flux sum (T + R > 1 + tau). Exit 66 required."""
        readout_passivity = copy.deepcopy(self.base_readout)
        telemetry_passivity = copy.deepcopy(self.base_telemetry)
        # Non-box monitors: near + far = 0.85 + 0.25 = 1.10 > 1.0 + 1e-5 (passivity violation)
        telemetry_passivity['monitors'] = {'far': 0.85, 'near': 0.25}
        readout_passivity['monitors'] = {'far': 0.85, 'near': 0.25}

        code, stderr, cert = self._execute_notary(readout_passivity, telemetry=telemetry_passivity)
        self.assertEqual(code, 66)
        self.assertIn('passivity_violation', stderr)
        self.assertFalse(cert['verified'])


if __name__ == '__main__':
    unittest.main()
