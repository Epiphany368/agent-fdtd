"""Empirical Challenger 2 test harness for LEG 44: V2-decay gate and launcher status propagation.

This suite empirically stress-tests:
1. Time-domain convergence gate (V2-decay):
   - Under-converged premature truncation (energy_decay_ratio > decay_floor) -> exit 66 & status failed:postflight=under_converged_time.
   - Legitimate ringdown (energy_decay_ratio <= decay_floor) -> exit 0 & verified:v2_decay tag.
   - Custom decay_floor configuration override.
   - Boundary condition testing (exact decay_floor and marginal 1e-6 + 1e-12 violation).
   - Non-finite (NaN / inf) ratio handling.
2. Launcher status propagation and driver non-zero exit code preservation:
   - Driver exit 53 (stopped-non-monotone) is strictly preserved as exit 53 and status failed:exit=53.
   - Driver exit 55 and exit 1 are strictly preserved without postflight clobbering.
   - Four-channel status consistency: process exit code, status file, manifest.json, and ledger.jsonl row.
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
LAUNCH = ROOT / 'tools' / 'launch.py'
POST_NOTARY = ROOT / 'checks' / 'post_notary.py'


def load_post_notary():
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChallengerLeg44DecayUnitTests(unittest.TestCase):
    """Direct empirical tests of verify_post_flight() across V2-decay edge cases."""

    def setUp(self):
        self.notary = load_post_notary()

    def _create_sample_run(self, td: Path, *, telemetry_data=None, readout_data=None, config_data=None):
        run_dir = td / 'sample_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        if config_data is not None:
            (run_dir / 'config.input.json').write_text(json.dumps(config_data, indent=2), encoding='utf-8')
        if telemetry_data is not None:
            (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry_data, indent=2), encoding='utf-8')
        if readout_data is not None:
            (run_dir / 'readout.json').write_text(json.dumps(readout_data, indent=2), encoding='utf-8')
        return run_dir

    def _default_monitors(self):
        return {
            'box-right': {'flux': 0.5743797},
            'box-left': {'flux': 0.1038960},
            'top': {'flux': 0.2854245},
            'bottom': {'flux': 0.0362995},
        }

    def _default_readout(self, steps=10000):
        return {
            'leg': 44,
            'simulation_executed': True,
            'executed_steps': steps,
            'epsilon_sha256': 'e' * 64,
            'is_lossless': True,
            'cases': [{
                'id': 'test-case',
                'measured': {
                    'T': 0.5743797,
                    'R': 0.1038960,
                    'monitors': self._default_monitors(),
                },
            }],
        }

    def test_under_converged_truncation_ratio_1e4_fails(self):
        """Stress: Premature duration truncation with energy_decay_ratio = 1e-4 > 1e-6 must fail with under_converged_time."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Decay Truncation 1e-4', 'kind': 'physics', 'decay_floor': 1e-6},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1e-4,  # Violates 1e-6 floor
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('under_converged_time', err)
            self.assertIn('spectral convergence failure', err)

    def test_under_converged_custom_decay_floor_fails(self):
        """Stress: Custom tight decay_floor (1e-8) correctly triggers failure when ratio is 5e-7."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Custom Decay Floor 1e-8', 'kind': 'physics', 'decay_floor': 1e-8},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 5e-7,  # Less than default 1e-6, but exceeds custom 1e-8!
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('under_converged_time', err)

    def test_legitimate_decay_passes_and_issues_tags(self):
        """Positive Control: ratio 5e-7 <= 1e-6 issues verified:v2_decay and verified:spectral_decay."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Legitimate Decay Test', 'kind': 'physics', 'decay_floor': 1e-6},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 5e-7,
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['spectral_decay_passed'])
            self.assertEqual(cert['energy_decay_ratio'], 5e-7)
            self.assertIn('verified:v2_decay', cert['tags'])
            self.assertIn('verified:spectral_decay', cert['tags'])

    def test_boundary_decay_exact_floor_passes(self):
        """Boundary Test: ratio exactly equal to decay_floor (1.0e-6 == 1.0e-6) must pass cleanly."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Exact Floor Boundary', 'kind': 'physics', 'decay_floor': 1e-6},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1.0e-6,
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertTrue(cert['spectral_decay_passed'])
            self.assertIn('verified:v2_decay', cert['tags'])

    def test_marginal_decay_violation_fails(self):
        """Adversarial Boundary Test: ratio 1.000001e-6 (> 1.0e-6 by 1e-12) must fail sharply."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Marginal Violation Boundary', 'kind': 'physics', 'decay_floor': 1e-6},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': 1.000001e-6,
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            err = str(ctx.exception)
            self.assertIn('under_converged_time', err)

    def test_non_finite_decay_ratio_fails(self):
        """Adversarial: Non-finite decay ratio (NaN or inf) must be rejected."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'NaN Decay Test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    'energy_decay_ratio': float('nan'),
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            with self.assertRaises(RuntimeError) as ctx:
                self.notary.verify_post_flight(run_dir)
            self.assertIn('under_converged_time', str(ctx.exception))

    def test_omitted_decay_ratio_does_not_tag_v2_decay(self):
        """Negative Invariant: When energy_decay_ratio is omitted, verified:v2_decay tag must NOT be granted."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = self._create_sample_run(
                Path(td),
                config_data={'task': 'Omitted Decay Test', 'kind': 'physics'},
                telemetry_data={
                    'schema_version': '1.0',
                    'epsilon_sha256': 'e' * 64,
                    'is_lossless': True,
                    'executed_steps': 10000,
                    # 'energy_decay_ratio' omitted
                    'monitors': self._default_monitors(),
                },
                readout_data=self._default_readout(),
            )
            cert = self.notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'])
            self.assertNotIn('verified:v2_decay', cert['tags'])
            self.assertNotIn('verified:spectral_decay', cert['tags'])


class ChallengerLeg44LauncherIntegrationTests(unittest.TestCase):
    """End-to-end integration tests through tools/launch.py.

    Verifies the complete 4-channel contract:
    1. Process exit code
    2. run_dir/status file
    3. run_dir/manifest.json (exit_code & status)
    4. runs/ledger.jsonl (row exit & status)
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
        script_name = f'_test_challenger_driver_{os.getpid()}_{len(self.created_scripts)}.py'
        script_file = ROOT / 'src' / script_name
        script_file.write_text(script_body, encoding='utf-8')
        self.created_scripts.append(script_file)

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / 'config.json'
            cfg_file.write_text(json.dumps(config_data), encoding='utf-8')

            env = dict(os.environ)

            for attempt in range(60):
                res = subprocess.run(
                    [sys.executable, '-B', str(LAUNCH), str(cfg_file), str(script_file.relative_to(ROOT))],
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                )
                if 'Another run holds the project lock' in res.stdout:
                    time.sleep(2)
                    continue
                self._track_run_dir(res.stdout)
                return res
            self._track_run_dir(res.stdout)
            return res

    def test_launch_under_converged_overrides_to_exit_66_and_status_under_converged_time(self):
        """Stress: Premature truncation (decay 1e-4 > 1e-6) MUST override launcher rc to 66 and status to failed:postflight=under_converged_time."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 5000,
    'energy_decay_ratio': 1e-4,  # Truncated simulation: 1e-4 > 1e-6
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
    'executed_steps': 5000,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'cases': [{
        'id': 'truncation-case',
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
        config = {'task': 'Challenger NC Truncation', 'kind': 'audit', 'decay_floor': 1e-6}
        res = self._run_launch_subproc(script_body, config)

        # 1. Process exit code
        self.assertEqual(res.returncode, 66, f"Expected launcher exit 66, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('exit=66', res.stdout)
        self.assertIn('failed:postflight=under_converged_time', res.stdout)

        # 2. Inspect created run artifacts
        self.assertTrue(len(self.created_run_dirs) > 0, "No run directory tracked from stdout")
        run_dir = self.created_run_dirs[-1]

        # 3. Status file
        status_file = run_dir / 'status'
        self.assertTrue(status_file.is_file())
        self.assertEqual(status_file.read_text(encoding='utf-8').strip(), 'failed:postflight=under_converged_time')

        # 4. Manifest.json
        manifest_file = run_dir / 'manifest.json'
        self.assertTrue(manifest_file.is_file())
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 66)
        self.assertEqual(manifest['status'], 'failed:postflight=under_converged_time')

        # 5. Ledger row
        ledger_lines = self.ledger_file.read_text(encoding='utf-8').strip().splitlines()
        last_row = json.loads(ledger_lines[-1])
        self.assertEqual(last_row['exit_code'], 66)
        self.assertEqual(last_row['status'], 'failed:postflight=under_converged_time')

        # 6. Verification certificate
        cert_file = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_file.is_file())
        cert = json.loads(cert_file.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'under_converged_time')
        self.assertIn('under_converged_time', cert['error'])

    def test_launch_legitimate_decay_exits_0_with_v2_decay_certificate(self):
        """Positive Control: Full simulation with decay 5e-7 <= 1e-6 exits 0 with verified:v2_decay."""
        script_body = """import json, os, sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR_ABS'])
telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': '1' * 64,
    'epsilon_sha256': '2' * 64,
    'is_lossless': True,
    'executed_steps': 17347,
    'energy_decay_ratio': 5e-7,  # Satisfies 1e-6 floor
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
        'id': 'pc-case',
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
        config = {'task': 'Challenger PC Decay', 'kind': 'audit', 'decay_floor': 1e-6, 'tau_box': 1e-5}
        res = self._run_launch_subproc(script_body, config)

        self.assertEqual(res.returncode, 0, f"Expected exit 0, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('exit=0 status=completed', res.stdout)

        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 0)
        self.assertEqual(manifest['status'], 'completed')

        cert_file = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_file.is_file())
        cert = json.loads(cert_file.read_text(encoding='utf-8'))
        self.assertTrue(cert['verified'])
        self.assertTrue(cert['spectral_decay_passed'])
        self.assertEqual(cert['energy_decay_ratio'], 5e-7)
        self.assertIn('verified:v2_decay', cert['tags'])
        self.assertIn('verified:spectral_decay', cert['tags'])

    def test_launch_driver_exit_53_stop_condition_preserved(self):
        """Critical Invariant: Driver exit 53 (stopped-non-monotone) MUST NOT be overridden to 66 or 70.

        The scientific exit code 53 must propagate faithfully to process exit code, status file,
        manifest.json, and runs/ledger.jsonl.
        """
        script_body = """import json, os, sys
from pathlib import Path

# Driver detects non-monotone stop condition and exits 53
sys.stderr.write('[driver] Non-monotone tooth condition triggered: stop condition exit 53\\n')
sys.exit(53)
"""
        config = {'task': 'Challenger Stop Condition 53', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)

        # 1. Process exit code MUST be 53
        self.assertEqual(res.returncode, 53, f"Driver exit 53 was not preserved! Got rc={res.returncode}. Output:\n{res.stdout}")
        self.assertIn('exit=53 status=failed:exit=53', res.stdout)

        # 2. Status file
        run_dir = self.created_run_dirs[-1]
        status_file = run_dir / 'status'
        self.assertTrue(status_file.is_file())
        self.assertEqual(status_file.read_text(encoding='utf-8').strip(), 'failed:exit=53')

        # 3. Manifest
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 53)
        self.assertEqual(manifest['status'], 'failed:exit=53')

        # 4. Ledger row
        ledger_lines = self.ledger_file.read_text(encoding='utf-8').strip().splitlines()
        last_row = json.loads(ledger_lines[-1])
        self.assertEqual(last_row['exit_code'], 53)
        self.assertEqual(last_row['status'], 'failed:exit=53')

        # 5. Verification certificate MUST NOT exist (post-flight check bypassed on non-zero driver exit)
        cert_file = run_dir / 'verification_certificate.json'
        self.assertFalse(cert_file.is_file(), "Post-flight notary should NOT issue certificate on failed driver exit")

    def test_launch_driver_exit_55_precondition_preserved(self):
        """Critical Invariant: Driver exit 55 (precondition failure) is preserved without postflight clobbering."""
        script_body = """import sys
sys.stderr.write('[driver] Precondition violation: exit 55\\n')
sys.exit(55)
"""
        config = {'task': 'Challenger Precondition 55', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)

        self.assertEqual(res.returncode, 55, f"Driver exit 55 not preserved: rc={res.returncode}")
        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 55)
        self.assertEqual(manifest['status'], 'failed:exit=55')

    def test_launch_driver_exit_1_general_failure_preserved(self):
        """Critical Invariant: Driver exit 1 (general unhandled failure) is preserved without postflight clobbering."""
        script_body = """import sys
sys.stderr.write('[driver] Unhandled exception: exit 1\\n')
sys.exit(1)
"""
        config = {'task': 'Challenger General Exit 1', 'kind': 'audit'}
        res = self._run_launch_subproc(script_body, config)

        self.assertEqual(res.returncode, 1, f"Driver exit 1 not preserved: rc={res.returncode}")
        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 1)
        self.assertEqual(manifest['status'], 'failed:exit=1')


if __name__ == '__main__':
    unittest.main()
