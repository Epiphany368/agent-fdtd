"""Comprehensive test suite for LEG 44: Poynting 4-Surface Box Enclosure and Energy Decay Notary.

Zero-Trust Rule: Agents possess propose rights, never dispose rights. A grating simulation's
physical credibility, completion, and merge eligibility are verified by unforgeable kernel
telemetry (solver_telemetry.json) tested against:
1. Guided Poynting Box Closure (V2-box):
     R_box = |(Phi_trans + Phi_refl + Phi_up + Phi_down) / Phi_inc - 1| <= tau_box (1e-5)
2. Spectral Time-Domain Decay (V2-decay):
     energy_decay_ratio = E(T_end) / E_max <= decay_floor (1e-6)

Test Structure:
- Leg44PoyntingBoxUnitTests: In-memory tests of verify_post_flight() across positive and negative controls.
- Leg44LauncherPoyntingBoxIntegrationTests: End-to-end launch.py integration tests enforcing exit 66 overrides.
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
import unittest


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / 'configs' / 'baseline-pin.json').is_file():
            return parent
    return cur.parents[1]


ROOT = find_project_root(Path(__file__))
LAUNCH = ROOT / 'tools' / 'launch.py'
POST_NOTARY = ROOT / 'checks' / 'post_notary.py'


def load_post_notary_module():
    """Dynamically load checks/post_notary.py or fallback peer proposed module."""
    if POST_NOTARY.is_file():
        spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    peer_paths = [
        ROOT / '.agents' / 'teamwork' / 'explorer_leg44_2' / 'proposed_post_notary.py',
        ROOT / '.agents' / 'teamwork' / 'worker_leg44_1' / 'proposed_post_notary.py',
    ]
    for peer in peer_paths:
        if peer.is_file():
            spec = importlib.util.spec_from_file_location('post_notary', peer)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f'Could not find {POST_NOTARY} or any peer implementation')


class Leg44PoyntingBoxUnitTests(unittest.TestCase):
    """Direct unit tests for V2-box Poynting closure and V2-decay in verify_post_flight()."""

    def setUp(self):
        self.notary = load_post_notary_module()

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
            if isinstance(telemetry_data, str):
                (run_dir / 'solver_telemetry.json').write_text(telemetry_data, encoding='utf-8')
            else:
                (run_dir / 'solver_telemetry.json').write_text(
                    json.dumps(telemetry_data, indent=2), encoding='utf-8')
        if readout_data is not None:
            if isinstance(readout_data, str):
                (run_dir / 'readout.json').write_text(readout_data, encoding='utf-8')
            else:
                (run_dir / 'readout.json').write_text(
                    json.dumps(readout_data, indent=2), encoding='utf-8')
        return run_dir

    def test_pc_lossless_n8_4surface_box_passes(self):
        """Positive Control: Lossless N8 grating with 4-surface box achieving R_box <= 1e-5 and decay <= 1e-6.

        Calibrated N8 flux values:
          box-right (transmission): 0.5743797
          box-left (reflection):    0.1038960
          top (cladding radiation): 0.2854245
          bottom (substrate leak):  0.0362995
          Sum:                      0.9999997 (R_box = 3.0e-7 <= 1e-5)
          Decay:                    2.5e-7 <= 1e-6
        """
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'PC Lossless N8 Box Test',
                    'kind': 'physics',
                    'tau_box': 1e-5,
                    'decay_floor': 1e-6,
                    'tau_pml': 1e-5,
                },
                telemetry_data={
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
                },
                readout_data={
                    'leg': 44,
                    'simulation_executed': True,
                    'executed_steps': 17347,
                    'epsilon_sha256': '2' * 64,
                    'is_lossless': True,
                    'cases': [{
                        'id': 'native-n8-box',
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
            self.assertTrue(cert.get('passivity_passed', False), 'Passivity must pass')
            self.assertTrue(cert.get('box_closure_passed', False), 'Box closure must pass')
            self.assertLessEqual(cert.get('box_closure_residual', 1.0), 1e-5)
            self.assertTrue(cert.get('spectral_decay_passed', False), 'Spectral decay must pass')
            self.assertLessEqual(cert.get('energy_decay_ratio', 1.0), 1e-6)

            # Check tags
            tags = cert.get('tags', [])
            self.assertIn('verified:postflight_notary', tags)
            self.assertIn('verified:v2_box_closure', tags)
            self.assertIn('verified:v2_decay', tags)

            # Check certificate artifact
            cert_file = run_dir / 'verification_certificate.json'
            self.assertTrue(cert_file.is_file())
            loaded = json.loads(cert_file.read_text(encoding='utf-8'))
            self.assertEqual(loaded['verified'], True)

    def test_nc1a_omitted_bottom_monitor_fails(self):
        """Negative Control 1a (Unclosed Box): Missing bottom monitor leaves substrate leakage (~0.0363) uncounted.

        Outward flux sum = 0.9637 -> R_box = 0.0363 > 1e-5.
        Must raise RuntimeError containing 'poynting_box_unclosed'.
        """
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'NC1a Omitted Bottom Test',
                    'kind': 'physics',
                    'tau_box': 1e-5,
                },
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
                        # 'bottom' omitted!
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 10000,
                    'epsilon_sha256': 'a' * 64,
                    'is_lossless': True,
                    'measured': {
                        'box-right': 0.5743797,
                        'box-left': 0.1038960,
                        'top': 0.2854245,
                    },
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)

    def test_nc1b_mutated_flux_fails(self):
        """Negative Control 1b (Unclosed Box): Box flux sum fails closure by 5e-5 > 1e-5.

        Outward flux sum = 1.000050 -> R_box = 5.0e-5 > 1.0e-5.
        Must raise RuntimeError containing 'poynting_box_unclosed'.
        """
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'NC1b Mutated Flux Test',
                    'kind': 'physics',
                    'tau_box': 1e-5,
                },
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
                    'measured': {
                        'box-right': 0.5744297,
                        'box-left': 0.1038960,
                        'top': 0.2854245,
                        'bottom': 0.0362995,
                    },
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)

    def test_nc2a_truncated_duration_decay_fails(self):
        """Negative Control 2a (Under-Converged Decay): Premature truncation leaves 12% energy remaining.

        energy_decay_ratio = 0.12 > decay_floor (1e-6).
        Must raise RuntimeError containing 'under_converged_time'.
        """
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'NC2a Premature Truncation Test',
                    'kind': 'physics',
                    'decay_floor': 1e-6,
                },
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 1000,
                    'energy_decay_ratio': 0.12,  # 12% remaining >> 1e-6
                    'monitors': {
                        'box-right': {'flux': 0.50},
                        'box-left': {'flux': 0.10},
                        'top': {'flux': 0.25},
                        'bottom': {'flux': 0.15},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 1000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {'box-right': 0.50, 'box-left': 0.10, 'top': 0.25, 'bottom': 0.15},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('under_converged_time', err)

    def test_nc2b_marginal_decay_violation_fails(self):
        """Negative Control 2b (Under-Converged Decay): Marginal ringdown 5e-6 > 1e-6 fails sharply."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'NC2b Marginal Decay Test',
                    'kind': 'physics',
                    'decay_floor': 1e-6,
                },
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 15000,
                    'energy_decay_ratio': 5.0e-6,  # 5e-6 > 1e-6
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2854245},
                        'bottom': {'flux': 0.0362995},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 15000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {
                        'box-right': 0.5743797,
                        'box-left': 0.1038960,
                        'top': 0.2854245,
                        'bottom': 0.0362995,
                    },
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('under_converged_time', err)

    def test_nc3a_lossy_telemetry_with_false_lossless_claim_fails(self):
        """Negative Control 3a (Lossy Claiming Lossless): Kernel reports is_lossless=False, readout claims True."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3a Lossy Contradiction', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'c' * 64,
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
                    'epsilon_sha256': 'c' * 64,
                    'is_lossless': True,  # False claim
                    'measured': {
                        'box-right': 0.40,
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
            self.assertIn('lossless contradiction', err)

    def test_nc3b_lossy_grating_unclosed_box_fails(self):
        """Negative Control 3b: Lossy medium absorbs 25% power -> outward sum 0.75 -> fails R_box <= 1e-5."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': 'NC3b Unclosed Lossy Test',
                    'kind': 'physics',
                    'tau_box': 1e-5,
                },
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': True,  # Falsely marked or unverified
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.45},
                        'box-left': {'flux': 0.10},
                        'top': {'flux': 0.15},
                        'bottom': {'flux': 0.05},
                        # Sum = 0.75, missing 0.25 absorbed power
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': True,
                    'measured': {
                        'box-right': 0.45,
                        'box-left': 0.10,
                        'top': 0.15,
                        'bottom': 0.05,
                    },
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)


class Leg44LauncherPoyntingBoxIntegrationTests(unittest.TestCase):
    """End-to-end integration tests driving tools/launch.py.

    Verifies launcher overrides exit code to 66 and updates manifest / runs/ledger.jsonl
    with granular status codes:
      - 'failed:postflight=poynting_box_unclosed'
      - 'failed:postflight=under_converged_time'
      - 'failed:postflight=tamper_detected'
    """

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
        script_name = f'_test_leg44_mock_driver_{os.getpid()}_{len(self.created_scripts)}.py'
        script_file = ROOT / 'src' / script_name
        script_file.write_text(script_body, encoding='utf-8')
        self.created_scripts.append(script_file)

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / 'config.json'
            cfg_file.write_text(json.dumps(config_data), encoding='utf-8')

            env = dict(os.environ)

            res = subprocess.run(
                [sys.executable, '-B', str(LAUNCH), str(cfg_file), str(script_file.relative_to(ROOT))],
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
            )
            self._track_run_dir(res.stdout)
            return res

    def test_pc_launch_lossless_n8_box_exits_0(self):
        """PC Launch Test: Genuine 4-surface box simulation exits 0 and issues certificate."""
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
        config = {'task': 'LEG 44 PC Launch Test', 'kind': 'audit', 'tau_box': 1e-5, 'decay_floor': 1e-6}
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
        self.assertTrue(cert.get('spectral_decay_passed', False))

    def test_nc1_launch_unclosed_box_exits_66(self):
        """NC1 Launch Test: Omitted bottom monitor -> launcher exits 66 with poynting_box_unclosed."""
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
        # bottom omitted -> sum = 0.9637 != 1.0
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
        config = {'task': 'LEG 44 NC1 Launch Test', 'kind': 'audit', 'tau_box': 1e-5}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertTrue(
            'failed:postflight=poynting_box_unclosed' in res.stdout or
            'failed:postflight=tamper_detected' in res.stdout,
            res.stdout
        )

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)

    def test_nc2_launch_under_converged_decay_exits_66(self):
        """NC2 Launch Test: Incomplete decay -> launcher exits 66 with under_converged_time."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 1000,
    'energy_decay_ratio': 0.05,  # 5% remaining > 1e-6
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
    'executed_steps': 1000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'nc2-under-converged',
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
        config = {'task': 'LEG 44 NC2 Launch Test', 'kind': 'audit', 'decay_floor': 1e-6}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertTrue(
            'failed:postflight=under_converged_time' in res.stdout or
            'failed:postflight=tamper_detected' in res.stdout,
            res.stdout
        )

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)

    def test_nc3_launch_lossy_claiming_lossless_exits_66(self):
        """NC3 Launch Test: Lossy medium falsely claiming lossless -> launcher exits 66."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': False,  # Evaluated lossy by kernel
    'executed_steps': 5000,
    'energy_decay_ratio': 1e-7,
    'monitors': {
        'box-right': {'flux': 0.40},
        'box-left': {'flux': 0.10},
        'top': {'flux': 0.15},
        'bottom': {'flux': 0.05},
    },
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'leg': 44,
    'simulation_executed': True,
    'executed_steps': 5000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,  # False claim
    'cases': [{
        'id': 'nc3-lossy-claim',
        'measured': {
            'T': 0.40,
            'R': 0.10,
            'monitors': {
                'box-right': {'flux': 0.40},
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
        config = {'task': 'LEG 44 NC3 Launch Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)


if __name__ == '__main__':
    unittest.main()
