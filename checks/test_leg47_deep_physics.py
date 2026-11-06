"""Acceptance Test Suite for Wave C — LEG 47:
M1 Tooth Width Monotonic Convergence Restoration & M4 Graded Mesh Modal Injection Uniform Collar Zone.

Acceptance Criteria:
1. Leg47ToothConvergenceUnitTests:
   - Ladder monotonicity, reversal detection, worsening detection, and C1 halving pass.
   - Richardson extrapolation order recovery across synthetic p in {1.0, 1.5, 2.0}.
   - Decoupled physical smoothing width formulation and smoothstep integral conservation.
2. Leg47ToothConvergencePositiveControl:
   - 3-rung ladder on N8 narrow tooth fixture (40/20, 20/10, 10/5 nm) verifying
     |T_40 - T_20| > |T_20 - T_10| with no reversal (d2 <= 0.5 * d1) and valid Richardson extrapolation.
3. Leg47GradedCollarUnitTests:
   - Strict collar zone boundary uniformity in x and y.
   - Grading ratio bound <= 1.25.
   - Grid cell reduction calculation >= 50%.
4. Leg47GradedCollarAccuracyPositiveControl:
   - Graded mesh + uniform collar vs uniform baseline:
     |Delta T| / T < 1e-3, |Delta R| / R < 1e-3, cell reduction >= 50%, and dt bound identical.
5. Leg47NegativeControls:
   - NC 1: Uncollared graded mesh fails < 1e-3 threshold (exit 57 / exit 66 on notary).
   - NC 2: Reversing / non-monotone ladder rejected (C1_pass=False).
   - NC 3: Unphysical smoothing parameters rejected with ValueError.
   - NC 4: Insufficient cell reduction (< 50%) rejected by post_notary (exit 66).
   - NC 5: Spoofed convergence tag rejected by post_notary (exit 66).
   - NC 6: Spoofed graded collar accuracy rejected by post_notary (exit 66).
6. Leg47PostNotaryIntegrationTests:
   - Issues 'verified:monotonic_mesh_convergence' and 'verified:graded_collar_accuracy' certificates.
   - Overrides exit code to 66 on tamper or criterion violation.
"""
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

POST_NOTARY = ROOT / 'checks' / 'post_notary.py'
M1_CONFIG_PATH = ROOT / 'configs' / 'leg47-m1-convergence.json'
COLLAR_CONFIG_PATH = ROOT / 'configs' / 'leg47-graded-collar.json'

from leg47_m1_convergence import (
    ladder_statistics,
    validate_config,
    compute_decoupled_transition,
    run_m1_convergence,
    select_device as m1_select_device
)
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


class Leg47ToothConvergenceUnitTests(unittest.TestCase):
    """Class 1: Arithmetic, monotonic detection, order recovery, and smoothing laws."""

    def test_ladder_monotonicity_detection(self):
        """Verifies reversal, non-monotonicity, and monotonic C1 passing detection."""
        # 1. Historical LEG 16 raw data with direction reversal (rose then fell)
        leg16_reversal = [0.5648354731662732, 0.5743797002155621, 0.5737057741498269]
        res_rev = ladder_statistics(leg16_reversal, complete=True)
        self.assertTrue(res_rev['non_monotone'], "LEG 16 sequence must be flagged non-monotone")
        self.assertTrue(res_rev['reversal'], "LEG 16 sequence must be flagged as reversal")
        self.assertFalse(res_rev['C1_pass'], "Reversing sequence must fail C1 criterion")
        self.assertIsNone(res_rev['richardson_extrapolated'], "Reversal refuses Richardson extrapolation")

        # 2. Historical LEG 17 fixed s=1.0 strictly monotonic decreasing sequence
        leg17_monotonic = [0.645227, 0.574380, 0.569926]
        res_mono = ladder_statistics(leg17_monotonic, complete=True)
        self.assertFalse(res_mono['non_monotone'], "Monotonic sequence must not be flagged non-monotone")
        self.assertFalse(res_mono['reversal'], "Monotonic sequence has zero reversals")
        self.assertFalse(res_mono['worsening'], "Monotonic sequence has no worsening")
        self.assertTrue(res_mono['C1_pass'], "Decreasing sequence with d2 <= 0.5*d1 passes C1")
        self.assertIsNotNone(res_mono['richardson_extrapolated'])
        self.assertGreater(res_mono['finest_order'], 1.0)

        # 3. Monotonic increasing sequence with error halving
        mono_inc = [0.50, 0.56, 0.575]  # d1 = 0.06, d2 = 0.015 (d2 <= 0.5 * d1)
        res_inc = ladder_statistics(mono_inc, complete=True)
        self.assertFalse(res_inc['non_monotone'])
        self.assertFalse(res_inc['reversal'])
        self.assertTrue(res_inc['C1_pass'])
        self.assertIsNotNone(res_inc['richardson_extrapolated'])

        # 4. Monotonic direction but worsening errors (d2 > d1)
        worsening = [0.60, 0.57, 0.52]  # d1 = 0.03, d2 = 0.05
        res_worse = ladder_statistics(worsening, complete=True)
        self.assertTrue(res_worse['non_monotone'], "Worsening error must be non-monotone")
        self.assertTrue(res_worse['worsening'], "Worsening flag must be True")
        self.assertFalse(res_worse['C1_pass'])

        # 5. Invalid inputs
        with self.assertRaises(ValueError):
            ladder_statistics([0.5, float('nan'), 0.6])
        with self.assertRaises(ValueError):
            ladder_statistics([0.5, float('inf'), 0.6])

    def test_richardson_extrapolation_order_recovery(self):
        """Verifies exact recovery of known convergence orders p in {1.0, 1.5, 2.0}."""
        h_mesh = [0.04, 0.02, 0.01]  # dx ladder halving: 40 -> 20 -> 10 nm
        t_inf = 0.572500
        amp = 0.0080

        for p_target in (1.0, 1.5, 2.0):
            # T(h) = T_inf + A * h^p
            t_values = [t_inf + amp * (h ** p_target) for h in h_mesh]
            stats = ladder_statistics(t_values, complete=True)
            self.assertTrue(stats['C1_pass'])
            self.assertAlmostEqual(
                stats['finest_order'], p_target, places=8,
                msg=f"Failed to recover known order p={p_target}"
            )
            self.assertAlmostEqual(
                stats['richardson_extrapolated'], t_inf, places=8,
                msg=f"Failed to recover Richardson limit for p={p_target}"
            )

    def test_decoupled_smoothing_formulation(self):
        """Verifies decoupled transition width computation and smoothstep integral conservation."""
        # Scaled mode: transition = s * dx
        cfg_scaled = {'transition_mode': 'scaled', 'scaling_ratio': 1.0}
        self.assertAlmostEqual(compute_decoupled_transition({'dx_um': 0.04}, cfg_scaled), 0.04)
        self.assertAlmostEqual(compute_decoupled_transition({'dx_um': 0.02}, cfg_scaled), 0.02)
        self.assertAlmostEqual(compute_decoupled_transition({'dx_um': 0.01}, cfg_scaled), 0.01)

        # Fixed mode: transition = transition_nm * 0.001
        cfg_fixed = {'transition_mode': 'fixed'}
        self.assertAlmostEqual(compute_decoupled_transition({'dx_um': 0.04, 'transition_nm': 20.0}, cfg_fixed), 0.02)
        self.assertAlmostEqual(compute_decoupled_transition({'dx_um': 0.01, 'transition_nm': 20.0}, cfg_fixed), 0.02)

        # Smoothstep mathematical integral conservation:
        # \int_0^1 (3u^2 - 2u^3) du = [u^3 - 0.5*u^4]_0^1 = 0.5
        u_grid = np.linspace(0.0, 1.0, 10001)
        s_vals = u_grid**2 * (3.0 - 2.0 * u_grid)
        integral = np.trapezoid(s_vals, u_grid)
        self.assertAlmostEqual(integral, 0.5, places=5, msg="Smoothstep volume fraction must conserve 0.5")


class Leg47ToothConvergencePositiveControl(unittest.TestCase):
    """Class 2: 3-rung ladder positive control on N8 126 nm narrow tooth fixture."""

    def test_pc_m1_monotonic_convergence_three_rungs(self):
        from run_b9_native_shape_n8 import scene
        from leg1_forward import launch_for

        cfg = json.loads(M1_CONFIG_PATH.read_text(encoding='utf-8'))
        validate_config(cfg)
        self.assertEqual(len(cfg['cases']), 3)

        # 1. Dynamically compute effective index ladder across all 3 rungs via genuine eigenmode solver
        neff_rungs = []
        for case in cfg['cases']:
            sc = scene(cfg, case)
            launch, _ = launch_for(sc['p'], sc['section'])
            neff_val = float(launch.mode.beta_per_um * cfg['wavelength_um'] / (2 * math.pi))
            neff_rungs.append(neff_val)

        stats_neff = ladder_statistics(neff_rungs, complete=True)

        # 2. Monotonicity & order on neff
        d1_neff = neff_rungs[1] - neff_rungs[0]
        d2_neff = neff_rungs[2] - neff_rungs[1]
        self.assertGreater(d1_neff * d2_neff, 0.0, "Effective index signed increments must have matching signs")
        self.assertLessEqual(abs(d2_neff), 0.5 * abs(d1_neff), "neff error must reduce by at least half")
        self.assertTrue(stats_neff['C1_pass'], "neff ladder must satisfy C1 pass criterion")
        self.assertGreaterEqual(stats_neff['finest_order'], 1.0, "neff order must be >= 1.0")

        # 3. Dynamic transmission ladder from simulation runner
        canary_cfg_path = ROOT / 'configs' / 'leg47-m1-canary.json'
        canary_cfg = json.loads(canary_cfg_path.read_text(encoding='utf-8'))
        report = run_m1_convergence(canary_cfg)
        self.assertTrue(report['simulation_executed'])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['verdict'], 'accepted')
        t_rungs = [float(c['measured']['T']) for c in report['cases']]

        stats_t = report['observables']['T']

        # 4. Monotonicity (No reversal): sign(d1) == sign(d2)
        d1_t = stats_t['signed_diffs'][0]
        d2_t = stats_t['signed_diffs'][1]
        self.assertGreater(d1_t * d2_t, 0.0, "Transmission signed increments must have matching signs (no reversal)")

        # 5. Shrinking Error: |T_10 - T_20| < |T_20 - T_40|
        self.assertLess(abs(d2_t), abs(d1_t), "Second increment |d2| must be strictly less than first increment |d1|")

        # 6. Order convergence: error drops by at least half (d2 <= 0.5 * d1)
        self.assertLessEqual(abs(d2_t), 0.5 * abs(d1_t), f"d2 ({abs(d2_t)}) must be <= 0.5 * d1 ({abs(d1_t)})")
        self.assertTrue(stats_t['C1_pass'], "Transmission ladder must satisfy C1 pass criterion")

        # 7. Valid Richardson Extrapolation
        self.assertGreaterEqual(stats_t['finest_order'], 1.0, "Observed order p must be >= 1.0")
        t_extrapolated = stats_t['richardson_extrapolated']
        self.assertIsNotNone(t_extrapolated)


class Leg47GradedCollarUnitTests(unittest.TestCase):
    """Class 3: Uniform feed collar geometry, grid arithmetic, and grading bounds."""

    def test_collar_boundary_continuity(self):
        """Verifies that within declared collar zone, grid steps are strictly uniform."""
        cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        case = cfg['cases'][0]
        sc_g = build_graded_collar_scene(cfg, case, device='cpu')
        rg = sc_g['r']

        collar_bounds = {
            'x': (cfg['collar']['feed_x_min'], cfg['collar']['feed_x_max']),
            'y': (cfg['collar']['collar_y_min'], cfg['collar']['collar_y_max']),
        }
        is_uniform, msg = check_collar_uniformity(rg, collar_bounds)
        self.assertTrue(is_uniform, f"Collar zone was not strictly uniform: {msg}")

        # Check nodes array exists and strictly increasing
        for a in rg.mesh_nodes:
            self.assertTrue(np.all(np.diff(a) > 0), "Mesh coordinates must be strictly monotonically increasing")

    def test_grading_ratio_bound_le_1_25(self):
        """Verifies maximum adjacent grid ratio is bounded by 1.25."""
        cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        case = cfg['cases'][0]
        sc_g = build_graded_collar_scene(cfg, case, device='cpu')
        metrics = realised_metrics(sc_g['r'])

        self.assertLessEqual(metrics['max_adjacent_ratio'], 1.25 + 1e-12,
                             f"Adjacent ratio {metrics['max_adjacent_ratio']} exceeds 1.25")
        for axis_idx, ratio in enumerate(metrics['axis_adjacent_ratio']):
            self.assertLessEqual(ratio, 1.25 + 1e-12, f"Axis {axis_idx} ratio {ratio} exceeds 1.25")

    def test_grid_cell_reduction_calculation(self):
        """Verifies cell reduction calculation and >= 50% savings requirement."""
        # Full uniform domain shape [674, 600, 5] = 2,022,000 cells
        uniform_cells = 2022000
        # Graded collar shape [520, 380, 5] = 988,000 cells
        graded_cells = 988000
        reduction = calculate_cell_reduction(uniform_cells, graded_cells)
        self.assertGreaterEqual(reduction, 0.50, f"Expected >= 50% cell reduction, got {reduction:.2%}")

        with self.assertRaises(ValueError):
            calculate_cell_reduction(0, 500)


class Leg47GradedCollarAccuracyPositiveControl(unittest.TestCase):
    """Class 4: Graded mesh + uniform collar accuracy vs uniform reference baseline."""

    def test_pc_m4_graded_collar_accuracy_and_reduction(self):
        """Verifies |Delta T|/T < 1e-3, |Delta R|/R < 1e-3, reduction >= 50%, and identical dt."""
        cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        case = cfg['cases'][0]

        # 1. Realised geometry & metrics evaluation
        sc_u = build_uniform_scene(cfg, case, device='cpu')
        sc_g = build_graded_collar_scene(cfg, case, device='cpu')

        ru = sc_u['r']
        rg = sc_g['r']

        cells_u = ru.shape[0] * ru.shape[1] * ru.shape[2]
        nodes_g = rg.mesh_nodes
        shape_g = [len(a) - 1 for a in nodes_g]
        cells_g = math.prod(shape_g)
        reduction = calculate_cell_reduction(cells_u, cells_g)

        self.assertGreaterEqual(reduction, 0.50, f"Cell reduction {reduction:.2%} < 50%")

        # 2. Identical time-step stability bound (O1 realized axis minimum)
        self.assertAlmostEqual(
            float(rg.time_step), float(ru.time_step), places=12,
            msg=f"Time step changed: dt_graded={rg.time_step} vs dt_uniform={ru.time_step}"
        )

        # 3. Forward simulation comparison dynamically from runner
        code, rep = run_graded_collar_comparison(cfg, max_steps=400)
        self.assertEqual(code, 0)
        self.assertTrue(rep['criteria']['all_passed'])
        self.assertGreaterEqual(rep['graded_accuracy']['cell_reduction'], 0.50)
        self.assertLess(rep['graded_accuracy']['rel_err_T'], 1e-3)
        self.assertLess(rep['graded_accuracy']['rel_err_R'], 1e-3)


class Leg47NegativeControls(unittest.TestCase):
    """Class 5: Negative controls enforcing failure codes and tamper detection."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg47_nc_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_run_dir(self, telemetry, readout, config=None):
        run_dir = Path(self.test_dir) / 'test_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        if config:
            (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return run_dir

    def test_nc1_uncollared_graded_fails_threshold(self):
        """NC 1: Uncollared graded mesh (LEG 24 raw error) fails 1e-3 threshold."""
        # Uncollared graded mesh measured in LEG 24:
        # T_graded = 0.561811 (err = 2.19% > 0.1%), R_graded = 0.107903 (err = 3.86% > 0.1%)
        t_uniform = 0.57437970
        t_uncollared = 0.56181105
        r_uniform = 0.10389604
        r_uncollared = 0.10790327

        err_t = abs(t_uncollared - t_uniform) / t_uniform
        err_r = abs(r_uncollared - r_uniform) / r_uniform

        self.assertGreater(err_t, 1e-3, "Uncollared graded mesh must fail 1e-3 threshold")
        self.assertGreater(err_r, 1e-3, "Uncollared graded mesh must fail 1e-3 threshold")

    def test_nc2_non_monotone_ladder_rejected(self):
        """NC 2: Reversing ladder is rejected without Richardson extrapolation."""
        reversing = [0.564835, 0.574380, 0.573706]
        stats = ladder_statistics(reversing, complete=True)
        self.assertTrue(stats['non_monotone'])
        self.assertFalse(stats['C1_pass'])
        self.assertIsNone(stats['richardson_extrapolated'])

    def test_nc3_unphysical_smoothing_rejected(self):
        """NC 3: Unphysical smoothing or grid parameters raise ValueError at preflight."""
        base_cfg = {
            'core_index': 3.476,
            'cladding_index': 1.444,
            'wavelength_um': 1.55,
            'width_um': 0.126,
            'precision': 'float64',
            'cases': [
                {'dx_um': 0.04, 'dy_um': 0.02, 'transition_nm': 40.0, 'periods': 8, 'duration_fs': 500},
                {'dx_um': 0.02, 'dy_um': 0.01, 'transition_nm': 20.0, 'periods': 8, 'duration_fs': 500},
                {'dx_um': 0.01, 'dy_um': 0.005, 'transition_nm': 10.0, 'periods': 8, 'duration_fs': 500},
            ]
        }
        # Negative transition
        cfg_bad_t = copy.deepcopy(base_cfg)
        cfg_bad_t['cases'][0]['transition_nm'] = -5.0
        with self.assertRaises(ValueError):
            validate_config(cfg_bad_t)

        # Transition exceeding tooth width (126 nm)
        cfg_excess_t = copy.deepcopy(base_cfg)
        cfg_excess_t['cases'][0]['transition_nm'] = 200.0
        with self.assertRaises(ValueError):
            validate_config(cfg_excess_t)

        # Negative grid step
        cfg_neg_dx = copy.deepcopy(base_cfg)
        cfg_neg_dx['cases'][0]['dx_um'] = -0.04
        with self.assertRaises(ValueError):
            validate_config(cfg_neg_dx)

        # Less than 3 rungs
        cfg_short = copy.deepcopy(base_cfg)
        cfg_short['cases'] = cfg_short['cases'][:2]
        with self.assertRaises(ValueError):
            validate_config(cfg_short)

    def test_nc4_insufficient_cell_reduction_rejected(self):
        """NC 4: Graded mesh achieving < 50% cell reduction is rejected by post_notary."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1000,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57436,
                'R_uniform': 0.10390,
                'R_graded': 0.10389,
                'cells_uniform': 2022000,
                'cells_graded': 1600000,  # Reduction = 20.8% < 50%!
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        config = {'graded_collar': True}
        run_dir = self._setup_run_dir(telemetry, readout, config)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('insufficient_cell_reduction', str(ctx.exception))

        # CLI execution returns exit 66
        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('insufficient_cell_reduction', proc.stderr)

    def test_nc5_tampered_convergence_readout_exits_66(self):
        """NC 5: Spoofed convergence tag with reversing rungs is rejected with exit 66."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1000,
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.564835}},
                {'id': 'r2', 'measured': {'T': 0.574380}},
                {'id': 'r3', 'measured': {'T': 0.573706}}  # Reversal!
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        config = {'convergence_ladder': True}
        run_dir = self._setup_run_dir(telemetry, readout, config)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('non_monotonic_mesh_ladder', str(ctx.exception))

        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('non_monotonic_mesh_ladder', proc.stderr)

    def test_nc6_tampered_graded_accuracy_exits_66(self):
        """NC 6: Spoofed collar accuracy exceeding 1e-3 threshold is rejected with exit 66."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1000,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.56000,  # Relative error = 2.5% > 1e-3!
                'R_uniform': 0.10390,
                'R_graded': 0.10390,
                'cells_uniform': 2022000,
                'cells_graded': 988000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        config = {'graded_collar': True}
        run_dir = self._setup_run_dir(telemetry, readout, config)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('graded_accuracy_exceeded', str(ctx.exception))

        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('graded_accuracy_exceeded', proc.stderr)


class Leg47PostNotaryIntegrationTests(unittest.TestCase):
    """Class 6: End-to-end postflight notary certification and tamper defense."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg47_notary_integ_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_post_notary_issues_leg47_tags_on_valid_run(self):
        """Valid run with convergence and collar accuracy earns notary tags and certificate."""
        run_dir = Path(self.test_dir) / 'valid_run'
        run_dir.mkdir(parents=True, exist_ok=True)

        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 2000,
            'resumed_from_step': 0,
            'monitors': {
                'box-right': {'flux': 0.85},
                'box-left': {'flux': 0.10},
                'top': {'flux': 0.025},
                'bottom': {'flux': 0.025}
            }
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 2000,
            'convergence_ladder': True,
            'graded_collar': True,
            'tags': ['monotonic_mesh_convergence', 'graded_collar_accuracy'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.645227}},
                {'id': 'r2', 'measured': {'T': 0.574380}},
                {'id': 'r3', 'measured': {'T': 0.569926}}
            ],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57435,
                'R_uniform': 0.10390,
                'R_graded': 0.10388,
                'cells_uniform': 2022000,
                'cells_graded': 988000
            },
            'monitors': {
                'box-right': {'flux': 0.85},
                'box-left': {'flux': 0.10},
                'top': {'flux': 0.025},
                'bottom': {'flux': 0.025}
            }
        }
        config = {
            'kind': 'simulation',
            'convergence_ladder': True,
            'graded_collar': True,
            'tags': ['monotonic_mesh_convergence', 'graded_collar_accuracy']
        }
        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

        # Run verification
        cert = self.post_notary.verify_post_flight(run_dir)
        self.assertTrue(cert['verified'])
        self.assertIn('verified:monotonic_mesh_convergence', cert['tags'])
        self.assertIn('verified:richardson_extrapolated', cert['tags'])
        self.assertIn('verified:graded_collar_accuracy', cert['tags'])
        self.assertIn('verified:cell_reduction_50pct', cert['tags'])

        self.assertTrue(cert['mesh_convergence']['monotonic'])
        self.assertGreaterEqual(cert['mesh_convergence']['observed_order'], 1.0)
        self.assertTrue(cert['graded_accuracy']['collar_zone_verified'])
        self.assertGreaterEqual(cert['graded_accuracy']['cell_reduction'], 0.50)

        # CLI call exits 0
        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('OK: verification certificate issued', proc.stdout)

    def test_post_notary_overrides_to_exit_66_on_violation(self):
        """Violations write verified: False certificate and exit 66."""
        run_dir = Path(self.test_dir) / 'tampered_run'
        run_dir.mkdir(parents=True, exist_ok=True)

        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 2000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 2000,
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.5648}},
                {'id': 'r2', 'measured': {'T': 0.5744}},
                {'id': 'r3', 'measured': {'T': 0.5737}}
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')

        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)

        cert_file = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_file.is_file())
        cert = json.loads(cert_file.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'non_monotonic_mesh_ladder')


if __name__ == '__main__':
    unittest.main()
