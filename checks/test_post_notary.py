"""Negative and positive control test suite for launcher post-flight notary (LEG 43).

Verifies the Zero-Trust rule:
- Agents possess propose rights, never dispose rights.
- A simulation's completion, credibility, and merge eligibility are decided by
  unforgeable kernel telemetry (solver_telemetry.json) and verified by the
  mandatory launcher post-flight notary gate (checks/post_notary.py).
- Even if a driver script exits 0, any readout tampering, false physics claim,
  missing telemetry, or passivity violation overrides the launcher exit code
  to 66 ('failed:postflight=tamper_detected') in manifest.json and runs/ledger.jsonl.
- When genuine matching telemetry and readout are present, launcher exits 0 and
  issues verification_certificate.json.
- When a driver script fails or exits non-zero (e.g. exit 53 stop condition),
  the launcher preserves the driver's exit code and does not overwrite it with 66.
"""
import datetime
import importlib.util
import json
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
    """Dynamically load checks/post_notary.py."""
    if not POST_NOTARY.is_file():
        peer_path = ROOT / '.agents' / 'teamwork' / 'explorer_leg43_2' / 'proposed_post_notary.py'
        if peer_path.is_file():
            spec = importlib.util.spec_from_file_location('post_notary', peer_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        raise FileNotFoundError(f'Neither {POST_NOTARY} nor {peer_path} exists')
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PostNotaryUnitTests(unittest.TestCase):
    """Direct unit tests for post_notary.verify_post_flight()."""

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

    def test_nc1_inflated_transmission_tamper_fails(self):
        """Negative Control 1: Readout inflates T (0.950 vs telemetry 0.574) -> tamper detected."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC1 unit test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'project_canonical_sha256': 'a' * 64,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 17347,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                    },
                },
                readout_data={
                    'leg': 43,
                    'policy': 'test-policy',
                    'simulation_executed': True,
                    'executed_steps': 17347,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'cases': [{
                        'id': 'test-case',
                        'measured': {
                            'T': 0.9500000,
                            'R': 0.1038960,
                            'monitors': {
                                'box-right': {'flux': 0.9500000},
                                'box-left': {'flux': 0.1038960},
                            },
                        },
                    }],
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('tamper_detected', err)
            self.assertIn('flux mismatch', err)

    def test_nc2_false_lossless_claim_fails(self):
        """Negative Control 2: Readout claims lossless=True, but kernel telemetry says is_lossless=False."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC2 unit test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'project_canonical_sha256': 'a' * 64,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': False,  # Kernel evaluated lossy material or conductivity
                    'executed_steps': 1000,
                    'monitors': {'T': {'flux': 0.40}, 'R': {'flux': 0.10}},
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 1000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,  # False claim
                    'measured': {'T': 0.40, 'R': 0.10},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('tamper_detected', err)
            self.assertIn('lossless contradiction', err)

    def test_nc3a_missing_telemetry_fails(self):
        """Negative Control 3a: solver_telemetry.json is absent on simulation run -> tamper detected."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3a unit test', 'kind': 'physics'},
                telemetry_data=None,  # Missing
                readout_data={
                    'simulation_executed': True,
                    'measured': {'T': 0.50},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('missing solver_telemetry.json', err)
            self.assertIn('tamper_detected', err)

    def test_nc3b_corrupt_telemetry_fails(self):
        """Negative Control 3b: solver_telemetry.json is malformed/truncated -> fails."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC3b unit test', 'kind': 'physics'},
                telemetry_data='{"schema_version": "1.0", "executed_steps": ',  # Syntax error
                readout_data={'simulation_executed': True},
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('corrupted solver_telemetry.json', err)

    def test_nc4_passivity_violation_fails(self):
        """Negative Control 4: Lossless simulation with total outward power > 1.0 -> fails."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC4 unit test', 'kind': 'physics', 'tau_pml': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 1000,
                    'monitors': {
                        'box-right': {'flux': 0.85},
                        'box-left': {'flux': 0.40},  # Sum = 1.25 > 1.0 + 1e-5
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 1000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {'T': 0.85, 'R': 0.40},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('passivity violation', err)
            self.assertTrue('passivity_violation' in err or 'tamper_detected' in err)

    def test_nc4_passivity_within_tolerance_passes(self):
        """Passivity boundary: total power = 1.000005 <= 1.0 + 1e-5 -> passes."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NC4 tolerance test', 'kind': 'physics', 'tau_pml': 1e-5},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 1000,
                    'monitors': {
                        'box-right': {'flux': 0.900000},
                        'box-left': {'flux': 0.100005},  # Sum = 1.000005
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 1000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {'T': 0.900000, 'R': 0.100005},
                },
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['passivity_passed'])

    def test_pc_genuine_matching_run_produces_certificate(self):
        """Positive Control: Genuine matching telemetry and readout -> verified certificate issued."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'PC unit test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'project_canonical_sha256': 'c' * 64,
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': True,
                    'executed_steps': 17347,
                    'energy_decay_ratio': 1.2e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5743797},
                        'box-left': {'flux': 0.1038960},
                        'top': {'flux': 0.2817243},
                        'bottom': {'flux': 0.0400000},
                    },
                },
                readout_data={
                    'leg': 43,
                    'simulation_executed': True,
                    'executed_steps': 17347,
                    'epsilon_sha256': 'd' * 64,
                    'is_lossless': True,
                    'cases': [{
                        'id': 'native-n8',
                        'measured': {
                            'T': 0.5743797,
                            'R': 0.1038960,
                            'epsilon_sha256': 'd' * 64,
                        },
                    }],
                },
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['passivity_passed'])
            self.assertTrue(cert.get('box_closure_passed', False))
            self.assertLessEqual(cert.get('box_closure_residual', 1.0), 1e-5)
            self.assertEqual(cert.get('decay_ratio'), 1.2e-7)
            self.assertEqual(cert['executed_steps'], 17347)
            self.assertEqual(cert['epsilon_sha256'], 'd' * 64)
            self.assertIn('verified:postflight_notary', cert['tags'])
            self.assertIn('verified:v2_box_closure', cert['tags'])
            self.assertIn('verified:v2_decay', cert['tags'])

            cert_path = run_dir / 'verification_certificate.json'
            self.assertTrue(cert_path.is_file())
            loaded = json.loads(cert_path.read_text(encoding='utf-8'))
            self.assertEqual(loaded['verified'], True)
            self.assertTrue(loaded.get('box_closure_passed', False))
            self.assertLessEqual(loaded.get('box_closure_residual', 1.0), 1e-5)
            self.assertEqual(loaded.get('decay_ratio'), 1.2e-7)

    def test_spectral_decay_violation_fails(self):
        """Under-converged time domain: energy_decay_ratio > decay_floor -> fails."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Decay test', 'kind': 'physics', 'decay_floor': 1e-6},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 1000,
                    'energy_decay_ratio': 0.05,  # 5% remaining > 1e-6
                    'monitors': {'T': {'flux': 0.50}, 'R': {'flux': 0.10}},
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 1000,
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {'T': 0.50, 'R': 0.10},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('spectral convergence failure', err)
            self.assertIn('under_converged_time', err)

    def test_step_count_mismatch_fails(self):
        """Readout claims 40000 steps, telemetry records 10000 -> tamper detected."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Step test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'monitors': {'T': {'flux': 0.50}, 'R': {'flux': 0.10}},
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 40000,  # Mismatch
                    'epsilon_sha256': 'b' * 64,
                    'is_lossless': True,
                    'measured': {'T': 0.50, 'R': 0.10},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('step count mismatch', err)
            self.assertIn('tamper_detected', err)

    def test_audit_task_without_simulation_passes(self):
        """Audit tasks do not run simulation and are granted audit certificate."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Audit task', 'kind': 'audit'},
                telemetry_data=None,
                readout_data={'simulation_executed': False, 'status': 'completed'},
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertEqual(cert['kind'], 'audit')
            self.assertIn('audit_only', cert['tags'])

    def test_3d_six_surface_poynting_box_closure_passes(self):
        """3D simulation with all 6 Poynting box surfaces closing to P_in = 1.0 -> passes."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': '3D Box Closure PC', 'kind': 'physics', 'dimension': '3d'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5000000},
                        'box-left': {'flux': 0.1000000},
                        'box-top': {'flux': 0.1500000},
                        'box-bottom': {'flux': 0.0500000},
                        'box-front': {'flux': 0.1200000},
                        'box-back': {'flux': 0.0800000},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'dimension': '3d',
                    'measured': {'T': 0.5000000, 'R': 0.1000000},
                },
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['box_closure_passed'])
            self.assertLessEqual(cert['box_closure_residual'], 1e-6)
            self.assertIn('verified:v2_box_closure', cert['tags'])
            self.assertIn('verified:v2_box_closure_3d', cert['tags'])
            self.assertTrue(bool(cert.get('certificate_payload_sha256')))

    def test_3d_six_surface_poynting_box_unclosed_fails(self):
        """3D simulation where 6-surface Poynting box leaks 15% power -> fails with poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': '3D Box Closure NC', 'kind': 'physics', 'dimension': '3d'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5000000},
                        'box-left': {'flux': 0.1000000},
                        'box-top': {'flux': 0.1000000},
                        'box-bottom': {'flux': 0.0500000},
                        'box-front': {'flux': 0.0500000},
                        'box-back': {'flux': 0.0500000},  # Total = 0.85 != 1.0
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'dimension': '3d',
                    'measured': {'T': 0.5000000, 'R': 0.1000000},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)
            self.assertIn('6-surface 3D', err)

    def test_3d_six_surface_poynting_box_missing_front_back_when_required_fails(self):
        """3D simulation with require_box_closure=True but only 4 monitors -> fails with poynting_box_unclosed."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={
                    'task': '3D Box Missing Surfaces NC',
                    'kind': 'physics',
                    'dimension': '3d',
                    'require_box_closure': True,
                },
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 5000,
                    'energy_decay_ratio': 1e-7,
                    'monitors': {
                        'box-right': {'flux': 0.5000000},
                        'box-left': {'flux': 0.2000000},
                        'box-top': {'flux': 0.1500000},
                        'box-bottom': {'flux': 0.1500000},
                    },
                },
                readout_data={
                    'simulation_executed': True,
                    'executed_steps': 5000,
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'dimension': '3d',
                    'measured': {'T': 0.5000000, 'R': 0.2000000},
                },
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('poynting_box_unclosed', err)
            self.assertIn('6-surface 3D', err)


class LauncherPostFlightNegativeControlTests(unittest.TestCase):
    """End-to-end integration tests driving tools/launch.py.

    Verifies that launch.py intercepts post-flight notary failures and rewrites:
      manifest['status'] -> 'failed:postflight=tamper_detected'
      manifest['exit_code'] -> 66
      runs/ledger.jsonl -> exit_code: 66
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
        """Run launch.py with an isolated script and config."""
        script_name = f'_test_mock_driver_{os.getpid()}_{len(self.created_scripts)}.py'
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

    def test_nc1_launch_exits_66_on_tampered_readout(self):
        """Driver exits 0 but mutates readout transmission -> launcher exits 66."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
# Write genuine kernel telemetry
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'executed_steps': 1000,
    'energy_decay_ratio': 1e-7,
    'monitors': {'box-right': {'flux': 0.5743797}, 'box-left': {'flux': 0.1038960}},
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

# Tampered readout: inflated transmission
readout = {
    'simulation_executed': True,
    'executed_steps': 1000,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'cases': [{'id': 'c1', 'measured': {'T': 0.9500000, 'R': 0.1038960}}],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'NC1 Launch Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=tamper_detected', res.stdout)

        # Inspect recorded manifest
        self.assertTrue(self.created_run_dirs, 'Run directory was not tracked')
        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)
        self.assertEqual(manifest['status'], 'failed:postflight=tamper_detected')

    def test_nc2_launch_exits_66_on_false_lossless_claim(self):
        """Driver exits 0 but claims lossless=True on lossy kernel -> launcher exits 66."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': False,  # Kernel evaluated lossy material
    'executed_steps': 1000,
    'energy_decay_ratio': 1e-7,
    'monitors': {'T': {'flux': 0.40}, 'R': {'flux': 0.10}},
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'simulation_executed': True,
    'executed_steps': 1000,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,  # False claim
    'measured': {'T': 0.40, 'R': 0.10},
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'NC2 Launch Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=tamper_detected', res.stdout)

    def test_nc3_launch_exits_66_on_missing_telemetry(self):
        """Driver exits 0 but omits solver_telemetry.json -> launcher exits 66."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
# Omit solver_telemetry.json
readout = {
    'simulation_executed': True,
    'measured': {'T': 0.50},
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'NC3 Launch Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=tamper_detected', res.stdout)

    def test_nc4_launch_exits_66_on_passivity_violation(self):
        """Driver exits 0 on lossless case with outward power > 1.0 -> launcher exits 66."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'executed_steps': 1000,
    'energy_decay_ratio': 1e-7,
    'monitors': {'T': {'flux': 0.85}, 'R': {'flux': 0.40}},  # Sum = 1.25 > 1.0
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'simulation_executed': True,
    'executed_steps': 1000,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'measured': {'T': 0.85, 'R': 0.40},
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'NC4 Launch Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('exit=66', res.stdout)
        self.assertTrue(
            'failed:postflight=passivity_violation' in res.stdout or
            'failed:postflight=tamper_detected' in res.stdout,
            res.stdout
        )

    def test_pc_launch_exits_0_on_genuine_matching_run(self):
        """Positive Control: Genuine matching telemetry and readout -> launcher exits 0, certificate issued."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'executed_steps': 17347,
    'energy_decay_ratio': 1.2e-7,
    'monitors': {'box-right': {'flux': 0.5743797}, 'box-left': {'flux': 0.1038960}},
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'simulation_executed': True,
    'executed_steps': 17347,
    'epsilon_sha256': 'b' * 64,
    'is_lossless': True,
    'cases': [{'id': 'c1', 'measured': {'T': 0.5743797, 'R': 0.1038960}}],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'PC Launch Test', 'kind': 'audit'}
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
        self.assertTrue(cert['passivity_passed'])

    def test_driver_exit_error_preserved_without_notary_overwrite(self):
        """If driver script exits with error (e.g. exit 53 stop condition), launcher preserves 53."""
        script_body = """import sys
sys.exit(53)
"""
        config = {'task': 'Driver Error Preservation Test', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)
        self.assertEqual(res.returncode, 53, res.stdout)
        self.assertIn('exit=53 status=failed:exit=53', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 53)
        self.assertEqual(manifest['status'], 'failed:exit=53')

    def test_audit_task_completes_exit_0(self):
        """Audit tasks do not run simulation and exit 0 cleanly with audit certificate."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
readout = {
    'task': 'audit task',
    'simulation_executed': False,
    'status': 'completed',
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {'task': 'Audit Task Launch Test', 'kind': 'audit'}
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
        self.assertEqual(cert['kind'], 'audit')


if __name__ == '__main__':
    unittest.main()
