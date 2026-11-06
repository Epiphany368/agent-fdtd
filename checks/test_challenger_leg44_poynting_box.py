"""Empirical Challenger 1 Test Suite for LEG 44: V2-box Poynting 4-Surface Box Enclosure Gate.

Zero-Trust Verification:
1. Positive Control:
   Lossless 4-surface box satisfying R_box <= 1e-5 and decay <= 1e-6 -> verifies exit 0 and
   certificate tags ['verified:v2_box_closure', 'verified:v2_decay'].
2. Negative Control 1 (Dropped/Omitted Monitor):
   Missing bottom substrate leakage monitor -> notary raises poynting_box_unclosed and launcher
   overrides exit code to 66 and status failed:postflight=poynting_box_unclosed.
3. Negative Control 2 (Mutated Surface Flux):
   Surface flux mutation violating R_box <= 1e-5 (e.g. 5e-5 deviation, marginal 1.0001e-5 violation,
   spectral array violation) -> notary raises poynting_box_unclosed and launcher overrides to exit 66.
4. Negative Control 3 (Lossy Grating with Unclosed Flux):
   Lossy device falsely claiming lossless or experiencing unmonitored material absorption -> exit 66.
5. Invariants Integration:
   Evaluate V2-box invariant under src/invariants.py, verifying recomputed closure and rejection of
   untrusted stored residuals.
"""
import copy
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / 'configs' / 'baseline-pin.json').is_file():
            return parent
    return cur.parents[1]


ROOT = find_project_root(Path(__file__))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
LAUNCH = ROOT / 'tools' / 'launch.py'
POST_NOTARY = ROOT / 'checks' / 'post_notary.py'
INVARIANTS = ROOT / 'src' / 'invariants.py'


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChallengerLeg44PoyntingBoxUnitTests(unittest.TestCase):
    """Direct empirical tests of verify_post_flight() across V2-box edge cases."""

    def setUp(self):
        self.notary = load_module('post_notary', POST_NOTARY)

    def _create_sample_run(self, td: Path, *,
                           telemetry_data=None,
                           readout_data=None,
                           config_data=None):
        run_dir = td / 'sample_run'
        run_dir.mkdir(parents=True, exist_ok=True)

        if config_data is not None:
            (run_dir / 'config.input.json').write_text(
                json.dumps(config_data, indent=2), encoding='utf-8')
        if telemetry_data is not None:
            (run_dir / 'solver_telemetry.json').write_text(
                json.dumps(telemetry_data, indent=2), encoding='utf-8')
        if readout_data is not None:
            (run_dir / 'readout.json').write_text(
                json.dumps(readout_data, indent=2), encoding='utf-8')
        return run_dir

    def test_positive_control_lossless_box_passes_and_tags_certificate(self):
        """Positive Control: Lossless 4-surface box satisfying R_box <= 1e-5 and decay <= 1e-6.

        Calibrated N8 grating flux values:
          box-right: 0.5743797
          box-left:  0.1038960
          top:       0.2854245
          bottom:    0.0362995
          Sum:       0.9999997 (R_box = 3.0e-7 <= 1e-5)
          Decay:     2.5e-7 <= 1e-6
        """
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'Challenger PC Lossless Box',
                    'kind': 'physics',
                    'tau_box': 1e-5,
                    'decay_floor': 1e-6,
                    'tau_pml': 1e-5,
                },
                telemetry_data={
                    'schema_version': '1.0',
                    'project_canonical_sha256': 'a' * 64,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 17347,
                    'energy_decay_ratio': 2.5e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0362995},
                    },
                },
                readout_data={
                    'leg': 44,
                    'simulation_executed': True,
                    'executed_steps': 17347,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'cases': [{
                        'id': 'pc-n8-box',
                        'measured': {
                            'T': 0.5743797,
                            'R': 0.1038960,
                            'monitors': {
                                'box-right': {'flux': 0.5743797},
                                'box-left': {'flux': 0.1038960},
                                'top': {'flux': 0.2854245},
                                'bottom': {'flux': 0.0362995},
                            },
                        },
                    }],
                },
            )

            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'], 'Certificate must be verified')
            self.assertTrue(cert.get('box_closure_passed', False), 'Box closure must pass')
            self.assertLessEqual(cert.get('box_closure_residual', 1.0), 1e-5)
            self.assertTrue(cert.get('spectral_decay_passed', False), 'Spectral decay must pass')
            self.assertLessEqual(cert.get('decay_ratio', 1.0), 1e-6)

            tags = cert.get('tags', [])
            self.assertIn('verified:postflight_notary', tags)
            self.assertIn('verified:v2_box_closure', tags)
            self.assertIn('verified:poynting_box_closed', tags)
            self.assertIn('verified:v2_decay', tags)

            monitors_record = cert.get('box_monitors', {})
            self.assertAlmostEqual(monitors_record['trans'], 0.5743797, places=6)
            self.assertAlmostEqual(monitors_record['refl'], 0.1038960, places=6)
            self.assertAlmostEqual(monitors_record['up'], 0.2854245, places=6)
            self.assertAlmostEqual(monitors_record['down'], 0.0362995, places=6)
            self.assertAlmostEqual(monitors_record['sum'], 0.9999997, places=6)

    def test_positive_control_spectral_arrays_pass(self):
        """Positive Control: Multi-frequency / array monitors across 5 wavelength bins satisfying R_box <= 1e-5."""
        with tempfile.TemporaryDirectory() as td:
            right = [0.5743797] * 5
            left  = [0.1038960] * 5
            top   = [0.2854245] * 5
            bot   = [0.0362995] * 5

            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Spectral Array Test', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': right},
                        'box-left': {'flux': left},
                        'top': {'flux': top},
                        'bottom': {'flux': bot},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': right, 'box-left': left, 'top': top, 'bottom': bot},
                },
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['box_closure_passed'])
            self.assertIn('verified:v2_box_closure', cert['tags'])

    def test_caveat_spectral_passivity_gate_limitation(self):
        """Caveat / Limitation Test: Demonstrates limitation in Check 5b where shifted spectral peaks trigger passivity violation.

        Even though sum at every lambda is exactly 1.0, Check 5b calculates total_power = sum(max(norm_val)) = 1.04 > 1.0 + 1e-5.
        """
        with tempfile.TemporaryDirectory() as td:
            right = [0.55, 0.56, 0.57438, 0.58, 0.59]
            left  = [0.12, 0.11, 0.10390, 0.10, 0.09]
            top   = [0.29, 0.29, 0.28542, 0.28, 0.28]
            bot   = [0.04, 0.04, 0.03630, 0.04, 0.04]

            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Spectral Shifted Peaks Caveat', 'kind': 'physics', 'tau_box': 1e-4},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': right},
                        'box-left': {'flux': left},
                        'top': {'flux': top},
                        'bottom': {'flux': bot},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': right, 'box-left': left, 'top': top, 'bottom': bot},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            self.assertIn('passivity_violation', str(ctx.exception))

    def test_negative_1a_missing_bottom_substrate_monitor(self):
        """Negative Control 1a: Missing bottom substrate leakage monitor fails with poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC1a Missing Bottom', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        # 'bottom' omitted
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.5743797, 'box-left': 0.1038960, 'top': 0.2854245},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)
            self.assertIn('bottom/substrate', err)

    def test_negative_1b_missing_top_cladding_monitor(self):
        """Negative Control 1b: Missing top cladding monitor fails with poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC1b Missing Top', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'bottom': {'flux': 0.0362995},
                        # 'top' omitted
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.5743797, 'box-left': 0.1038960, 'bottom': 0.0362995},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)
            self.assertIn('top/cladding', err)

    def test_negative_2a_mutated_surface_flux_excess(self):
        """Negative Control 2a: Outward flux exceeds incident by 5e-5 > 1e-5 -> fails poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC2a Excess Flux', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5744297},  # +5e-5 deviation
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0362995},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.5744297, 'box-left': 0.1038960, 'top': 0.2854245, 'bottom': 0.0362995},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)
            self.assertIn('poynting box closure failure', err)

    def test_negative_2b_marginal_box_residual_violation(self):
        """Negative Control 2b: Marginal box residual 1.0001e-5 > 1.0e-5 fails sharply."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC2b Marginal Violation', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743902},  # residual = 1.02e-5 > 1.0e-5
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0362995},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.5743902, 'box-left': 0.1038960, 'top': 0.2854245, 'bottom': 0.0362995},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)

    def test_negative_2c_spectral_array_single_bin_violation(self):
        """Negative Control 2c: In a spectral array, 3 bins pass but 1 bin has R_box = 2e-4 -> fails."""
        with tempfile.TemporaryDirectory() as td:
            right = [0.57438, 0.57438, 0.57438, 0.57438]
            left  = [0.10390, 0.10390, 0.10390, 0.10390]
            top   = [0.28542, 0.28542, 0.28562, 0.28542]  # Bin 2 has +2e-4 error
            bot   = [0.03630, 0.03630, 0.03630, 0.03630]

            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC2c Spectral Single Bin', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': right},
                        'box-left': {'flux': left},
                        'top': {'flux': top},
                        'bottom': {'flux': bot},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': right, 'box-left': left, 'top': top, 'bottom': bot},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)

    def test_negative_3a_lossy_grating_unclosed_flux(self):
        """Negative Control 3a: Lossy grating absorbs 25% power -> outward sum 0.75 -> fails poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3a Unclosed Lossy Test', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'c' * 64,
                    'is_lossless': True,  # Unmodelled loss / claiming lossless
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.45},
                        'box-left': {'flux': 0.10},
                        'top': {'flux': 0.15},
                        'bottom': {'flux': 0.05},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'c' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.45, 'box-left': 0.10, 'top': 0.15, 'bottom': 0.05},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)

    def test_negative_3b_lossy_telemetry_contradicted_by_readout(self):
        """Negative Control 3b: Kernel telemetry reports lossy (False) but readout claims lossless (True)."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3b Lossless Contradiction', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': False,  # Kernel evaluated lossy medium
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.40},
                        'box-left': {'flux': 0.10},
                        'top': {'flux': 0.15},
                        'bottom': {'flux': 0.05},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': True,  # False claim
                    'measured': {'box-right': 0.40, 'box-left': 0.10, 'top': 0.15, 'bottom': 0.05},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('tamper_detected', err)
            self.assertIn('lossless contradiction', err)

    def test_negative_3c_lossy_attempted_flux_spoofing(self):
        """Negative Control 3c: Lossy simulation where driver fakes readout transmission to sum to 1.0, but telemetry caught true flux."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3c Spoofed Flux', 'kind': 'physics', 'tau_box': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.40},  # True flux
                        'box-left': {'flux': 0.10},
                        'top': {'flux': 0.15},
                        'bottom': {'flux': 0.05},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'measured': {
                        'box-right': 0.70,  # Driver spoofed 0.40 -> 0.70 to forge sum = 1.0!
                        'box-left': 0.10,
                        'top': 0.15,
                        'bottom': 0.05,
                    },
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('tamper_detected', err)
            self.assertIn('flux mismatch on box-right', err)


class ChallengerLeg44InvariantsUnitTests(unittest.TestCase):
    """Direct tests verifying V2-box under src/invariants.py."""

    def setUp(self):
        self.inv_mod = load_module('invariants', INVARIANTS)

    def test_invariants_v2_box_positive_pass(self):
        config = {
            'policy': 'v2',
            'invariants': [{'id': 'V2-box', 'case_id': 'c1', 'tolerance': 1e-5, 'basis': 'Poynting closure'}],
            'cases': [{'id': 'c1', 'geometry': 'grating', 'physics': {'lossless': True, 'poynting_box': True}}],
        }
        readout = {
            'policy': 'v2',
            'fixture': 'grating',
            'cases': [{
                'id': 'c1',
                'measured': {
                    'box-right': 0.5743797,
                    'box-left': 0.1038960,
                    'top': 0.2854245,
                    'bottom': 0.0362995,
                    'box_closure_residual': abs((0.5743797 + 0.1038960 + 0.2854245 + 0.0362995) - 1.0),
                },
            }],
        }
        res = self.inv_mod.evaluate_invariants(readout, config)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['status'], 'passed')
        self.assertTrue(res[0]['lawful'])
        self.assertLessEqual(res[0]['error'], 1e-5)

    def test_invariants_v2_box_rejects_untrusted_stored_residual(self):
        """If stored box_closure_residual does not match recomputed closure, it is flagged as untrusted."""
        config = {
            'policy': 'v2',
            'invariants': [{'id': 'V2-box', 'case_id': 'c1', 'tolerance': 1e-5, 'basis': 'Poynting closure'}],
            'cases': [{'id': 'c1', 'geometry': 'grating', 'physics': {'lossless': True, 'poynting_box': True}}],
        }
        readout = {
            'policy': 'v2',
            'fixture': 'grating',
            'cases': [{
                'id': 'c1',
                'measured': {
                    'box-right': 0.5744297,  # error = 5e-5 > 1e-5
                    'box-left': 0.1038960,
                    'top': 0.2854245,
                    'bottom': 0.0362995,
                    'box_closure_residual': 1e-8,  # Fake claimed residual!
                },
            }],
        }
        res = self.inv_mod.evaluate_invariants(readout, config)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['status'], 'failed')
        rejected = res[0].get('rejected', [])
        self.assertTrue(any('differs from recomputed closure' in r.get('reason', '') for r in rejected))


class ChallengerLeg44LauncherIntegrationTests(unittest.TestCase):
    """End-to-end integration tests through tools/launch.py verifying exit code overrides."""

    def setUp(self):
        self.ledger_file = ROOT / 'runs' / 'ledger.jsonl'
        self.orig_ledger = self.ledger_file.read_text(encoding='utf-8') if self.ledger_file.is_file() else None
        self.created_run_dirs = []
        self.created_scripts = []

    def tearDown(self):
        if self.orig_ledger is not None and self.ledger_file.is_file():
            self.ledger_file.write_text(self.orig_ledger, encoding='utf-8')
        for run_dir in self.created_run_dirs:
            if run_dir.is_dir():
                shutil.rmtree(run_dir, ignore_errors=True)
        for script in self.created_scripts:
            if script.is_file():
                try:
                    script.unlink()
                except OSError:
                    pass

    def _track_run_dir(self, output):
        for line in output.splitlines():
            if 'run_dir=' in line:
                for token in line.split():
                    if token.startswith('run_dir='):
                        path = Path(token.split('=', 1)[1])
                        self.created_run_dirs.append(path)

    def _run_launch_subproc(self, script_body: str, config_data: dict):
        script_name = f'_test_challenger1_mock_{os.getpid()}_{len(self.created_scripts)}.py'
        script_file = ROOT / 'src' / script_name
        script_file.write_text(script_body, encoding='utf-8')
        self.created_scripts.append(script_file)

        import fcntl
        def _wait_for_lock(timeout=300):
            lock_path = ROOT / 'runs' / '.lock'
            if not lock_path.is_file():
                return True
            start = time.time()
            while time.time() - start < timeout:
                try:
                    with lock_path.open('a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f, fcntl.LOCK_UN)
                    return True
                except (BlockingIOError, OSError):
                    time.sleep(2)
            return False

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / 'config.json'
            cfg_file.write_text(json.dumps(config_data), encoding='utf-8')

            env = dict(os.environ)

            # Retry with lock waiting in case of concurrent test runs
            for attempt in range(10):
                _wait_for_lock(timeout=180)
                res = subprocess.run(
                    [sys.executable, '-B', str(LAUNCH), str(cfg_file), str(script_file.relative_to(ROOT))],
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
                if 'Another run holds the project lock' in res.stdout:
                    time.sleep(3)
                    continue
                self._track_run_dir(res.stdout)
                return res
            self._track_run_dir(res.stdout)
            return res

    def test_launcher_positive_lossless_box_exits_0(self):
        """PC Launch Test: 4-surface box satisfying R_box <= 1e-5 and decay <= 1e-6 exits 0."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 17347,
    'energy_decay_ratio': 2.5e-7,
    'monitors': {
        'box-right': {'flux': 0.5743797},
        'box-left': {'flux': 0.1038960},
        'top': {'flux': 0.2854245},
        'bottom': {'flux': 0.0362995},
    },
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'leg': 44,
    'simulation_executed': True,
    'executed_steps': 17347,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'pc-n8-box',
        'measured': {
            'T': 0.5743797,
            'R': 0.1038960,
            'monitors': {
                'box-right': {'flux': 0.5743797},
                'box-left': {'flux': 0.1038960},
                'top': {'flux': 0.2854245},
                'bottom': {'flux': 0.0362995},
            },
        },
    }],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'Challenger PC Launch Test', 'kind': 'audit', 'tau_box': 1e-5, 'decay_floor': 1e-6}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertIn('exit=0 status=completed', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 0)
        self.assertEqual(manifest['status'], 'completed')

        cert_file = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_file.is_file())
        cert = json.loads(cert_file.read_text(encoding='utf-8'))
        self.assertTrue(cert['verified'])
        self.assertTrue(cert.get('box_closure_passed', False))
        self.assertIn('verified:v2_box_closure', cert.get('tags', []))

    def test_launcher_negative_1_omitted_bottom_monitor_exits_66(self):
        """NC1 Launch Test: Omitted bottom monitor -> launcher overrides exit to 66 and status failed:postflight=poynting_box_unclosed."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 10000,
    'energy_decay_ratio': 1e-7,
    'monitors': {
        'box-right': {'flux': 0.5743797},
        'box-left': {'flux': 0.1038960},
        'top': {'flux': 0.2854245},
        # bottom monitor omitted
    },
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'leg': 44,
    'simulation_executed': True,
    'executed_steps': 10000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'nc1-unclosed-box',
        'measured': {
            'T': 0.5743797,
            'R': 0.1038960,
            'monitors': {
                'box-right': {'flux': 0.5743797},
                'box-left': {'flux': 0.1038960},
                'top': {'flux': 0.2854245},
            },
        },
    }],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'Challenger NC1 Launch Test', 'kind': 'audit', 'tau_box': 1e-5}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=poynting_box_unclosed', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)
        self.assertEqual(manifest['status'], 'failed:postflight=poynting_box_unclosed')

        # Check status file
        status_file = run_dir / 'status'
        self.assertEqual(status_file.read_text(encoding='utf-8').strip(), 'failed:postflight=poynting_box_unclosed')

    def test_launcher_negative_2_mutated_flux_exits_66(self):
        """NC2 Launch Test: Mutated surface flux violating R_box <= 1e-5 -> overrides to exit 66 and status failed:postflight=poynting_box_unclosed."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 10000,
    'energy_decay_ratio': 1e-7,
    'monitors': {
        'box-right': {'flux': 0.5744297},  # +5e-5 deviation
        'box-left': {'flux': 0.1038960},
        'top': {'flux': 0.2854245},
        'bottom': {'flux': 0.0362995},
    },
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'leg': 44,
    'simulation_executed': True,
    'executed_steps': 10000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'nc2-mutated-flux',
        'measured': {
            'T': 0.5744297,
            'R': 0.1038960,
            'monitors': {
                'box-right': {'flux': 0.5744297},
                'box-left': {'flux': 0.1038960},
                'top': {'flux': 0.2854245},
                'bottom': {'flux': 0.0362995},
            },
        },
    }],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'Challenger NC2 Launch Test', 'kind': 'audit', 'tau_box': 1e-5}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=poynting_box_unclosed', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)
        self.assertEqual(manifest['status'], 'failed:postflight=poynting_box_unclosed')

    def test_launcher_negative_3_lossy_unclosed_flux_exits_66(self):
        """NC3 Launch Test: Lossy grating with unclosed flux claiming lossless -> overrides to exit 66 and status failed:postflight=poynting_box_unclosed."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,  # False lossless claim
    'executed_steps': 5000,
    'energy_decay_ratio': 1e-7,
    'monitors': {
        'box-right': {'flux': 0.45},
        'box-left': {'flux': 0.10},
        'top': {'flux': 0.15},
        'bottom': {'flux': 0.05},
        # Sum = 0.75 != 1.0 (missing 0.25 absorption)
    },
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'leg': 44,
    'simulation_executed': True,
    'executed_steps': 5000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'nc3-lossy-unclosed',
        'measured': {
            'T': 0.45,
            'R': 0.10,
            'monitors': {
                'box-right': {'flux': 0.45},
                'box-left': {'flux': 0.10},
                'top': {'flux': 0.15},
                'bottom': {'flux': 0.05},
            },
        },
    }],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'Challenger NC3 Launch Test', 'kind': 'audit', 'tau_box': 1e-5}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=poynting_box_unclosed', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)
        self.assertEqual(manifest['status'], 'failed:postflight=poynting_box_unclosed')


if __name__ == '__main__':
    unittest.main()
