"""Adversarial Empirical Attack Harness for Wave C — LEG 47:
Canary Monotonicity, Graded Mesh Collar Bounds, and Kernel Telemetry Tamper Resistance.

Author: challenger_leg47_4 (Empirical Challenger / Critic, Specialist)
Mandate:
Empirically stress-test commit 9bf3634 with adversarial inputs, perturbed configs,
boundary violations, and malicious telemetry tampering, verifying that all failure
modes trigger exit code 66 and zero-trust verification certificates record verified=False.
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
M1_CANARY_CONFIG = ROOT / 'configs' / 'leg47-m1-canary.json'
M1_CONFIG_PATH = ROOT / 'configs' / 'leg47-m1-convergence.json'
COLLAR_CONFIG_PATH = ROOT / 'configs' / 'leg47-graded-collar.json'

from leg47_m1_convergence import (
    ladder_statistics,
    validate_config,
    compute_decoupled_transition,
    run_m1_convergence,
)
from leg47_graded_collar import (
    realised_metrics,
    check_collar_uniformity,
    calculate_cell_reduction,
    build_uniform_scene,
    build_graded_collar_scene,
    run_graded_collar_comparison,
)


def import_post_notary():
    import importlib.util
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CanaryParameterMonotonicityAdversarialTests(unittest.TestCase):
    """Attack 1: Canary parameter perturbations, ladder reversals, and exit 66 validation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='challenger_attack_m1_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_run(self, ladder_readout, telemetry_ladder=None, executed_steps=1000, eps_sha=None):
        run_dir = Path(self.test_dir) / f'run_{os.urandom(4).hex()}'
        run_dir.mkdir(parents=True, exist_ok=True)
        sha = eps_sha or 'e' * 64

        t_ladder = telemetry_ladder
        if t_ladder is None:
            t_ladder = []
            for r in ladder_readout:
                m = r.get('measured', r)
                t_ladder.append({
                    'id': r.get('id', 'case'),
                    'dx_um': r.get('dx_um', 0.02),
                    'dy_um': r.get('dy_um', 0.02),
                    'shape': [100, 100, 5],
                    'steps': 500,
                    'measured': {
                        'T': m.get('T', 0.5),
                        'epsilon_sha256': m.get('epsilon_sha256', sha)
                    }
                })

        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'c' * 64,
            'epsilon_sha256': sha,
            'is_lossless': True,
            'executed_steps': executed_steps,
            'resumed_from_step': 0,
            'mesh_ladder': t_ladder,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': executed_steps,
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': ladder_readout,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        config = {'convergence_ladder': True}

        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return run_dir

    def test_attack_reversal_up_then_down_exits_66(self):
        """Attacking ladder with an upward jump followed by downward drop -> exit 66."""
        # Ladder jumps up then drops: +0.02 then -0.01 -> reversal
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.50, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': 0.52, 'epsilon_sha256': 'e'*64}},
            {'id': 'r3', 'dx_um': 0.01, 'measured': {'T': 0.51, 'epsilon_sha256': 'e'*64}},
        ]
        stats = ladder_statistics([0.50, 0.52, 0.51], complete=True)
        self.assertTrue(stats['reversal'])
        self.assertTrue(stats['non_monotone'])
        self.assertFalse(stats['C1_pass'])

        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('non_monotonic_mesh_ladder', proc.stderr)
        cert = json.loads((run_dir / 'verification_certificate.json').read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'non_monotonic_mesh_ladder')

    def test_attack_reversal_down_then_up_exits_66(self):
        """Attacking ladder with classic LEG 16 pitfall: down then up -> exit 66."""
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.5648, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': 0.5500, 'epsilon_sha256': 'e'*64}},
            {'id': 'r3', 'dx_um': 0.01, 'measured': {'T': 0.5580, 'epsilon_sha256': 'e'*64}},
        ]
        stats = ladder_statistics([0.5648, 0.5500, 0.5580], complete=True)
        self.assertTrue(stats['reversal'])
        self.assertFalse(stats['C1_pass'])

        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('non_monotonic_mesh_ladder', proc.stderr)

    def test_attack_worsening_increments_exits_66(self):
        """Monotonic in direction, but subsequent error increment is larger (|d2| > |d1|) -> exit 66."""
        # d1 = -0.01, d2 = -0.03 (|d2| > |d1|)
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.60, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': 0.59, 'epsilon_sha256': 'e'*64}},
            {'id': 'r3', 'dx_um': 0.01, 'measured': {'T': 0.56, 'epsilon_sha256': 'e'*64}},
        ]
        stats = ladder_statistics([0.60, 0.59, 0.56], complete=True)
        self.assertTrue(stats['worsening'])
        self.assertTrue(stats['non_monotone'])
        self.assertFalse(stats['C1_pass'])

        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('non_monotonic_mesh_ladder', proc.stderr)

    def test_attack_stagnant_ladder_zero_delta_exits_66(self):
        """Attacking ladder with stagnant identical values (zero progress) -> exit 66."""
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.55, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': 0.55, 'epsilon_sha256': 'e'*64}},
            {'id': 'r3', 'dx_um': 0.01, 'measured': {'T': 0.54, 'epsilon_sha256': 'e'*64}},
        ]
        stats = ladder_statistics([0.55, 0.55, 0.54], complete=True)
        self.assertTrue(stats['non_monotone'])

        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('non_monotonic_mesh_ladder', proc.stderr)

    def test_attack_insufficient_rungs_exits_66(self):
        """Attacking ladder with only 2 rungs -> exit 66."""
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.60, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': 0.55, 'epsilon_sha256': 'e'*64}},
        ]
        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('requires at least 3 completed rungs', proc.stderr)

    def test_attack_non_finite_transmission_exits_66(self):
        """Attacking ladder with NaN or null transmission -> exit 66."""
        ladder = [
            {'id': 'r1', 'dx_um': 0.04, 'measured': {'T': 0.60, 'epsilon_sha256': 'e'*64}},
            {'id': 'r2', 'dx_um': 0.02, 'measured': {'T': float('nan'), 'epsilon_sha256': 'e'*64}},
            {'id': 'r3', 'dx_um': 0.01, 'measured': {'T': 0.50, 'epsilon_sha256': 'e'*64}},
        ]
        run_dir = self._setup_run(ladder)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('missing or non-finite transmission', proc.stderr)

    def test_attack_unphysical_transition_width_rejected(self):
        """Attacking validate_config with smoothing transition > tooth width -> ValueError."""
        cfg = json.loads(M1_CANARY_CONFIG.read_text(encoding='utf-8'))
        cfg['cases'][0]['transition_nm'] = 200.0  # 200 nm > 126 nm tooth width
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn('exceeds tooth width', str(ctx.exception))


class GradedMeshCollarAdversarialTests(unittest.TestCase):
    """Attack 2: Graded mesh collar zone bounds, grading ratios, and error bounds."""

    def setUp(self):
        self.cfg = json.loads(COLLAR_CONFIG_PATH.read_text(encoding='utf-8'))
        self.case = self.cfg['cases'][0]
        self.test_dir = tempfile.mkdtemp(prefix='challenger_attack_collar_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_collar_run(self, readout, telemetry=None, config=None):
        run_dir = Path(self.test_dir) / f'collar_{os.urandom(4).hex()}'
        run_dir.mkdir(parents=True, exist_ok=True)
        tel = telemetry or {
            'schema_version': '1.0',
            'project_canonical_sha256': 'p' * 64,
            'epsilon_sha256': 'e' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        (run_dir / 'solver_telemetry.json').write_text(json.dumps(tel, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        if config:
            (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return run_dir

    def test_attack_uncollared_mesh_breaks_uniformity(self):
        """Stripping modal feed collar breaks uniformity across injection plane."""
        sc_uncollared = build_graded_collar_scene(self.cfg, self.case, device='cpu')
        sc_uncollared['r'].mesh_refinements = []  # Completely strip collar
        collar_bounds = {
            'x': (self.cfg['collar']['feed_x_min'], self.cfg['collar']['feed_x_max']),
            'y': (self.cfg['collar']['collar_y_min'], self.cfg['collar']['collar_y_max']),
        }
        is_uniform, msg = check_collar_uniformity(sc_uncollared['r'], collar_bounds)
        self.assertFalse(is_uniform)
        self.assertIn('Non-uniform', msg)

    def test_attack_shrunk_collar_truncates_mode_field(self):
        """Shrinking longitudinal collar exposes injection plane to graded cells."""
        shrunk_cfg = copy.deepcopy(self.cfg)
        shrunk_cfg['collar']['feed_x_min'] = -5.00
        shrunk_cfg['collar']['feed_x_max'] = -4.00
        sc = build_graded_collar_scene(shrunk_cfg, self.case, collar_cfg=shrunk_cfg['collar'], device='cpu')
        collar_bounds = {
            'x': (self.cfg['collar']['feed_x_min'], self.cfg['collar']['feed_x_max']),
            'y': (self.cfg['collar']['collar_y_min'], self.cfg['collar']['collar_y_max']),
        }
        is_uniform, msg = check_collar_uniformity(sc['r'], collar_bounds)
        self.assertFalse(is_uniform)

    def test_attack_grading_ratio_exceeding_1_25_exits_66(self):
        """Attacking with adjacent ratio > 1.25 triggers exit 66 on post_notary."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57438,
                'R_uniform': 0.10390,
                'R_graded': 0.10390,
                'cells_uniform': 2000000,
                'cells_graded': 900000,
            },
            'mesh_metrics': {
                'max_adjacent_ratio': 1.35  # > 1.25 violation!
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        run_dir = self._setup_collar_run(readout)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('exceeds 1.25', proc.stderr)
        self.assertIn('graded_accuracy_exceeded', proc.stderr)

    def test_attack_transmission_relative_error_exceeding_1e3_exits_66(self):
        """Attacking with transmission error >= 1e-3 triggers exit 66."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.50000,
                'T_graded': 0.50060,  # rel err = 0.00060 / 0.50 = 1.2e-3 >= 1e-3
                'R_uniform': 0.10000,
                'R_graded': 0.10000,
                'cells_uniform': 2000000,
                'cells_graded': 900000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        run_dir = self._setup_collar_run(readout)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('graded mesh transmission error', proc.stderr)
        self.assertIn('graded_accuracy_exceeded', proc.stderr)

    def test_attack_reflection_relative_error_exceeding_1e3_exits_66(self):
        """Attacking with reflection error >= 1e-3 triggers exit 66."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.50000,
                'T_graded': 0.50000,
                'R_uniform': 0.10000,
                'R_graded': 0.10015,  # rel err = 0.00015 / 0.10 = 1.5e-3 >= 1e-3
                'cells_uniform': 2000000,
                'cells_graded': 900000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        run_dir = self._setup_collar_run(readout)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('graded mesh reflection error', proc.stderr)
        self.assertIn('graded_accuracy_exceeded', proc.stderr)

    def test_attack_cell_reduction_under_50pct_exits_66(self):
        """Attacking with cell reduction < 50% (e.g. 48%) triggers exit 66."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.50000,
                'T_graded': 0.50000,
                'R_uniform': 0.10000,
                'R_graded': 0.10000,
                'cells_uniform': 1000000,
                'cells_graded': 520000,  # 48% reduction < 50%
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        run_dir = self._setup_collar_run(readout)
        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('insufficient grid cell reduction', proc.stderr)
        self.assertIn('insufficient_cell_reduction', proc.stderr)


class KernelTelemetryTamperResistanceAdversarialTests(unittest.TestCase):
    """Attack 3: Zero-trust kernel telemetry tamper resistance and exit 66 enforcement."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='challenger_attack_telemetry_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _execute(self, telemetry, readout, config=None):
        run_dir = Path(self.test_dir) / f'tamper_{os.urandom(4).hex()}'
        run_dir.mkdir(parents=True, exist_ok=True)
        if telemetry is not None:
            (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        if readout is not None:
            (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        if config is not None:
            (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

        proc = subprocess.run([sys.executable, str(POST_NOTARY), str(run_dir)], capture_output=True, text=True)
        cert_path = run_dir / 'verification_certificate.json'
        cert = json.loads(cert_path.read_text(encoding='utf-8')) if cert_path.is_file() else None
        return proc.returncode, proc.stderr, cert

    def test_attack_executed_steps_zero_fails_with_tamper_detected(self):
        """Telemetry executed_steps = 0 must abort with tamper_detected and exit 66."""
        readout = {
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.60}},
                {'id': 'r2', 'measured': {'T': 0.55}},
                {'id': 'r3', 'measured': {'T': 0.52}},
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 0,  # Zero steps stepped!
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        code, stderr, cert = self._execute(telemetry, readout, config={'convergence_ladder': True})
        self.assertEqual(code, 66)
        self.assertIn('verified positive executed_steps', stderr)
        self.assertIn('tamper_detected', stderr)
        self.assertIsNotNone(cert)
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'tamper_detected')

    def test_attack_epsilon_sha256_mismatch_fails_with_tamper_detected(self):
        """Telemetry epsilon_sha256 != readout epsilon_sha256 -> exit 66 tamper_detected."""
        readout = {
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.60, 'epsilon_sha256': '1'*64}},
                {'id': 'r2', 'measured': {'T': 0.55, 'epsilon_sha256': '2'*64}},
                {'id': 'r3', 'measured': {'T': 0.52, 'epsilon_sha256': '3'*64}},
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': '9' * 64,  # Maliciously altered hash!
            'is_lossless': True,
            'executed_steps': 5000,
            'resumed_from_step': 0,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        code, stderr, cert = self._execute(telemetry, readout, config={'convergence_ladder': True})
        self.assertEqual(code, 66)
        self.assertIn('epsilon_sha256 mismatch', stderr)
        self.assertIn('tamper_detected', stderr)
        self.assertFalse(cert['verified'])

    def test_attack_mesh_ladder_transmission_tamper_fails_with_tamper_detected(self):
        """Driver readout transmission spoofed relative to kernel telemetry -> exit 66."""
        readout = {
            'convergence_ladder': True,
            'tags': ['monotonic_mesh_convergence'],
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.60}},
                {'id': 'r2', 'measured': {'T': 0.55}},
                {'id': 'r3', 'measured': {'T': 0.52}},  # Spoofed to pass monotonicity
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 5000,
            'resumed_from_step': 0,
            'mesh_ladder': [
                {'id': 'r1', 'measured': {'T': 0.60}},
                {'id': 'r2', 'measured': {'T': 0.55}},
                {'id': 'r3', 'measured': {'T': 0.58}},  # Kernel actually measured 0.58 (reversal!)
            ],
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        code, stderr, cert = self._execute(telemetry, readout, config={'convergence_ladder': True})
        self.assertEqual(code, 66)
        self.assertIn('transmission mismatch', stderr)
        self.assertIn('tamper_detected', stderr)

    def test_attack_graded_collar_accuracy_tamper_fails_with_tamper_detected(self):
        """Driver readout claims better error than kernel telemetry -> exit 66."""
        readout = {
            'graded_collar': True,
            'tags': ['graded_collar_accuracy'],
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.57438,  # Driver claims 0.0 error
                'R_uniform': 0.10390,
                'R_graded': 0.10390,
                'cells_uniform': 2000000,
                'cells_graded': 900000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'graded_accuracy': {
                'T_uniform': 0.57438,
                'T_graded': 0.58500,  # Kernel actually had significant error
                'R_uniform': 0.10390,
                'R_graded': 0.10390,
                'cells_uniform': 2000000,
                'cells_graded': 900000,
            },
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        code, stderr, cert = self._execute(telemetry, readout, config={'graded_collar': True})
        self.assertEqual(code, 66)
        self.assertIn('graded accuracy metric T_graded mismatch', stderr)
        self.assertIn('tamper_detected', stderr)

    def test_attack_missing_solver_telemetry_fails_with_tamper_detected(self):
        """Simulation run omitting solver_telemetry.json -> exit 66 tamper_detected."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 1000,
            'monitors': {'box-right': 0.85, 'box-left': 0.10, 'top': 0.025, 'bottom': 0.025}
        }
        code, stderr, cert = self._execute(None, readout)
        self.assertEqual(code, 66)
        self.assertIn('missing solver_telemetry.json', stderr)
        self.assertIn('tamper_detected', stderr)


if __name__ == '__main__':
    unittest.main()
