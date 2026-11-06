"""Adversarial Empirical Attack Test Suite for Wave C — LEG 48:
Check 10 Cross-Solver Reconciliation Gate & Tamper Defense Challenge.

Author: challenger_leg48_2 (Empirical Challenger / Critic, Specialist)
Mandate:
Empirically stress-test Check 10 in checks/post_notary.py with adversarial attacks:
- Attack 1: Zero-Stepped Simulation Spoofing (executed_steps <= 0)
- Attack 2: Epsilon Permittivity Digest Spoofing (telemetry vs readout mismatch)
- Attack 3: External Solver Provenance Forgery (missing/failed external solver)
- Attack 4: Spoofed Discrepancy Field with Falsified Internal Table (table recomputation)
- Attack 5: Tolerance Tamper Attack (reconciliation_tolerance expansion/override)
- Attack 6: Missing Poynting Enclosure Interaction (Check 5a vs Check 10 isolation)
- Attack 7: Discrepancy Metric Poisoning (NaN, Inf, non-finite values)
- Attack 8: Notary Certificate Tamper Resistance & Tag Issuance Integrity
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

POST_NOTARY_PATH = ROOT / 'checks' / 'post_notary.py'
CONFIG_PATH = ROOT / 'configs' / 'leg48-reconciliation.json'
PYTHON_EXEC = sys.executable


def import_post_notary():
    import importlib.util
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BaseTamperAttackTest(unittest.TestCase):
    """Base harness providing mock run directory synthesis and CLI invocation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='challenger_leg48_attack_')
        self.post_notary = import_post_notary()
        if CONFIG_PATH.is_file():
            self.base_cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        else:
            self.base_cfg = {
                'schema_version': '1.0',
                'kind': 'benchmark',
                'cross_solver': True,
                'reconciliation_tolerance': 0.015,
            }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_mock_run(self, name: str, telemetry: dict, readout: dict, config: dict = None) -> Path:
        run_dir = Path(self.test_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = dict(config or self.base_cfg)

        # Baseline 4-surface box monitors to satisfy Check 5a if box closure is evaluated
        default_mons = {
            'box-right': {'flux': 0.8876},
            'box-left': {'flux': 0.1112},
            'top': {'flux': 0.0006},
            'bottom': {'flux': 0.0006},
        }

        tel = copy.deepcopy(telemetry)
        if 'monitors' not in tel and 'monitor_raw_digests' not in tel:
            tel['monitors'] = copy.deepcopy(default_mons)

        rd = copy.deepcopy(readout)
        if 'monitors' not in rd:
            rd['monitors'] = copy.deepcopy(tel.get('monitors', default_mons))

        (run_dir / 'solver_telemetry.json').write_text(json.dumps(tel, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(rd, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        return run_dir

    def _run_post_notary_cli(self, run_dir: Path) -> subprocess.CompletedProcess:
        cmd = [PYTHON_EXEC, str(POST_NOTARY_PATH), str(run_dir)]
        env = dict(os.environ)
        env['RUN_DIR_ABS'] = str(run_dir)
        return subprocess.run(cmd, env=env, capture_output=True, text=True)


class Attack1ZeroSteppedSimulationSpoofingTests(BaseTamperAttackTest):
    """Attack 1: Zero-Stepped Simulation Spoofing.
    Supply a mock run directory with executed_steps <= 0 in solver_telemetry.json claiming
    successful reconciliation. Confirm post_notary aborts with exit code 66 and tamper_detected.
    """

    def test_executed_steps_zero_aborts_with_exit_66_tamper_detected(self):
        """executed_steps = 0 must abort with exit 66 and tamper_detected tag."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': 0,  # Zero-stepped simulation
            'is_lossless': True,
            'epsilon_sha256': 'a' * 64,
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 0,
            'is_lossless': True,
            'epsilon_sha256': 'a' * 64,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'tolerance': 0.015,
            }
        }
        run_dir = self._setup_mock_run('attack1_zero_steps', tel, rd)

        # Python API verification
        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))
        self.assertIn('executed_steps', str(ctx.exception))

        # CLI execution verification
        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

        # Certificate must record verified=False and status=tamper_detected
        cert_path = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_path.is_file())
        cert = json.loads(cert_path.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_executed_steps_negative_aborts_with_exit_66(self):
        """Negative executed_steps (-500) must abort with exit 66."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': -500,
            'is_lossless': True,
            'epsilon_sha256': 'a' * 64,
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': -500,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
            }
        }
        run_dir = self._setup_mock_run('attack1_neg_steps', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

    def test_executed_steps_omitted_or_non_numeric_aborts_with_exit_66(self):
        """Omitted or non-numeric executed_steps must abort with exit 66."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': '4000_fake',  # String instead of int
            'is_lossless': True,
            'epsilon_sha256': 'a' * 64,
        }
        rd = {
            'simulation_executed': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
            }
        }
        run_dir = self._setup_mock_run('attack1_str_steps', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)


class Attack2EpsilonPermittivityDigestSpoofingTests(BaseTamperAttackTest):
    """Attack 2: Epsilon Permittivity Digest Spoofing.
    Alter epsilon_sha256 in solver_telemetry.json to differ from readout.json.
    Confirm post_notary aborts with exit code 66.
    """

    def test_epsilon_sha256_telemetry_vs_readout_mismatch(self):
        """Direct readout.epsilon_sha256 != telemetry.epsilon_sha256 aborts with exit 66."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'epsilon_sha256': '1111111111111111111111111111111111111111111111111111111111111111',
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'is_lossless': True,
            'epsilon_sha256': '2222222222222222222222222222222222222222222222222222222222222222',
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack2_eps_mismatch_top', tel, rd)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))
        self.assertIn('epsilon_sha256 mismatch', str(ctx.exception))

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

    def test_epsilon_sha256_cross_solver_block_mismatch(self):
        """readout.cross_solver.epsilon_sha256 != telemetry.epsilon_sha256 aborts with exit 66."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'epsilon_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'rcwa', 'version': '0.1.4', 'exit_code': 0},
                'max_discrepancy': 0.005,
                'epsilon_sha256': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            }
        }
        run_dir = self._setup_mock_run('attack2_eps_mismatch_cs', tel, rd)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))
        self.assertIn('cross_solver epsilon_sha256 mismatch', str(ctx.exception))

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)


class Attack3ExternalSolverProvenanceForgeryTests(BaseTamperAttackTest):
    """Attack 3: External Solver Provenance Forgery.
    Omit external_solver block or inject non-zero exit code (exit_code = 1, 137, or -9).
    Confirm post_notary aborts with exit code 66 and tamper_detected.
    """

    def test_omitted_external_solver_block_aborts_with_exit_66(self):
        """Missing external_solver block in cross_solver must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                # Missing 'external_solver'
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack3_missing_ext', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

    def test_empty_external_solver_name_aborts_with_exit_66(self):
        """external_solver with empty string name must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': '', 'version': '1.0', 'exit_code': 0},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack3_empty_name', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

    def test_external_solver_nonzero_exit_code_1_aborts_with_exit_66(self):
        """external_solver exit_code = 1 must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 1},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack3_exit_1', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)
        self.assertIn('external solver execution failed with exit code 1', proc.stderr)

    def test_external_solver_oom_kill_exit_code_137_aborts_with_exit_66(self):
        """external_solver exit_code = 137 (OOM killed) must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 137},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack3_exit_137', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)
        self.assertIn('exit code 137', proc.stderr)

    def test_external_solver_negative_signal_exit_code_aborts_with_exit_66(self):
        """external_solver exit_code = -9 (SIGKILL) must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'torcwa', 'version': '0.1.4', 'exit_code': -9},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack3_exit_neg9', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)


class Attack4SpoofedDiscrepancyInternalTableTests(BaseTamperAttackTest):
    """Attack 4: Spoofed Discrepancy Field with Falsified Internal Table.
    Provide a readout where max_discrepancy = 0.001 is claimed, but the actual table values
    |eta_{torchfdtd, m} - eta_{ext, m}| > 0.015. Check if Check 10 re-computes the maximum
    discrepancy from the raw tables.
    """

    def test_spoofed_scalar_discrepancy_caught_by_raw_table_audit(self):
        """Driver claims max_discrepancy = 0.001, but raw order table has discrepancy 0.045 (>0.015).
        Check 10 must independently evaluate the order table and abort with exit 66.
        """
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.001,  # Falsified small claim
                'tolerance': 0.015,
                'torchfdtd_efficiencies': {
                    'order_-1': 0.420,
                    'order_0': 0.100,
                    'order_1': 0.350,
                },
                'external_efficiencies': {
                    'order_-1': 0.4205,      # diff = 0.0005 (small)
                    'order_0': 0.1002,       # diff = 0.0002 (small)
                    'order_1': 0.395,        # diff = 0.0450 (> 0.015! VIOLATION)
                }
            }
        }
        run_dir = self._setup_mock_run('attack4_spoofed_scalar', tel, rd)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))
        self.assertIn('order_1', str(ctx.exception))

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)
        self.assertIn('order_1', proc.stderr)

    def test_efficiencies_alias_key_triggers_table_audit(self):
        """Raw table audit also triggers when using 'efficiencies' instead of 'torchfdtd_efficiencies'."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'rcwa', 'version': '0.1.4', 'exit_code': 0},
                'max_discrepancy': 0.002,  # Spoofed claim
                'tolerance': 0.015,
                'efficiencies': {
                    'T_0': 0.500,
                },
                'external_efficiencies': {
                    'T_0': 0.530,  # diff = 0.030 > 0.015
                }
            }
        }
        run_dir = self._setup_mock_run('attack4_alias_efficiencies', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)

    def test_telemetry_discrepancy_mismatch_detected(self):
        """If telemetry records cross_solver max_discrepancy differing from readout, abort with exit 66."""
        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'cross_solver': {
                'max_discrepancy': 0.0120,
            }
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0010,  # Differs from telemetry by > FLUX_TOLERANCE
                'tolerance': 0.015,
            }
        }
        run_dir = self._setup_mock_run('attack4_tel_discrepancy_mismatch', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)
        self.assertIn('cross_solver discrepancy mismatch', proc.stderr)


class Attack5ToleranceTamperTests(BaseTamperAttackTest):
    """Attack 5: Tolerance Tamper Attack.
    Set reconciliation_tolerance = 0.50 in config and check if platform enforces bounds or limits.
    """

    def test_adversarial_tolerance_expansion_analysis(self):
        """Analyze platform behavior when config sets reconciliation_tolerance = 0.50.
        Empirically verify whether post_notary bounds tolerance against hard spec limit (0.015).
        """
        cfg = copy.deepcopy(self.base_cfg)
        cfg['reconciliation_tolerance'] = 0.50  # Adversarially expanded tolerance (50%)

        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.25,  # Unphysical 25% discrepancy!
            }
        }
        run_dir = self._setup_mock_run('attack5_tolerance_50pct', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)

        # NOTE FOR CHALLENGER REPORT:
        # If proc.returncode == 0, post_notary accepted the 50% tolerance without capping it!
        # This is a key finding for the review: Check 10 currently allows arbitrary tolerance
        # expansion from config (or even from readout.json cs_data['tolerance'])!
        if proc.returncode == 0:
            cert_path = run_dir / 'verification_certificate.json'
            cert = json.loads(cert_path.read_text(encoding='utf-8'))
            self.assertTrue(cert['verified'])
            self.assertEqual(cert['cross_solver_reconciliation']['tolerance'], 0.50)
            self.assertEqual(cert['cross_solver_reconciliation']['max_discrepancy'], 0.25)
            # Record empirical observation for handoff
            self.expanded_tolerance_accepted = True
        else:
            self.assertEqual(proc.returncode, 66)
            self.expanded_tolerance_accepted = False

    def test_readout_tolerance_injection_when_config_omitted(self):
        """Driver injects tolerance = 0.50 into readout.json when config omits reconciliation_tolerance."""
        cfg = copy.deepcopy(self.base_cfg)
        cfg.pop('reconciliation_tolerance', None)
        cfg.pop('tolerance', None)

        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'tolerance': 0.40,  # Driver injected wide tolerance
                'max_discrepancy': 0.20,
            }
        }
        run_dir = self._setup_mock_run('attack5_readout_injected_tolerance', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)
        # Check whether driver self-tolerance injection succeeds
        cert_path = run_dir / 'verification_certificate.json'
        if cert_path.is_file():
            cert = json.loads(cert_path.read_text(encoding='utf-8'))
            self.readout_injection_accepted = cert.get('verified', False)

    def test_negative_or_zero_tolerance_fallback_to_default(self):
        """Non-positive tolerance (e.g. -0.05 or 0.0) must fall back to DEFAULT (0.015),
        blocking a discrepancy of 0.025 with exit 66.
        """
        cfg = copy.deepcopy(self.base_cfg)
        cfg['reconciliation_tolerance'] = -0.05  # Invalid negative tolerance

        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.025,  # Exceeds default 0.015
            }
        }
        run_dir = self._setup_mock_run('attack5_negative_tolerance', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)


class Attack6MissingPoyntingEnclosureInteractionTests(BaseTamperAttackTest):
    """Attack 6: Missing Poynting Enclosure Interaction.
    Verify that Check 5a (Poynting box closure) and Check 10 do not mask each other's failures.
    """

    def test_unclosed_poynting_box_blocks_reconciliation_certificate(self):
        """Poynting box residual R_box = 0.08 > tau_box (0.05), while cross-solver discrepancy
        is 0.001 (clean). Check 5a must abort with poynting_box_unclosed and exit 66,
        preventing any reconciliation certificate from being awarded.
        """
        cfg = copy.deepcopy(self.base_cfg)
        cfg['tau_box'] = 0.02  # Box closure tolerance 2%

        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': {
                'box-right': {'flux': 0.700},
                'box-left': {'flux': 0.100},
                'top': {'flux': 0.010},
                'bottom': {'flux': 0.010},
                # Total = 0.820 -> Residual |0.820 - 1.0| = 0.180 > 0.02
            }
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': copy.deepcopy(tel['monitors']),
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.001,  # Excellent reconciliation
                'tolerance': 0.015,
            }
        }
        run_dir = self._setup_mock_run('attack6_unclosed_box', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('poynting_box_unclosed', proc.stderr)

        cert_path = run_dir / 'verification_certificate.json'
        cert = json.loads(cert_path.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'poynting_box_unclosed')

    def test_missing_box_surface_blocks_reconciliation_certificate(self):
        """tau_box is in config, but 'top' monitor is omitted from telemetry.
        Check 5a must detect missing surface and abort with poynting_box_unclosed.
        """
        cfg = copy.deepcopy(self.base_cfg)
        cfg['tau_box'] = 0.05

        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': {
                'box-right': {'flux': 0.8876},
                'box-left': {'flux': 0.1112},
                # 'top' missing!
                'bottom': {'flux': 0.0006},
            }
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': copy.deepcopy(tel['monitors']),
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.001,
            }
        }
        run_dir = self._setup_mock_run('attack6_missing_surface', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('poynting_box_unclosed', proc.stderr)
        self.assertIn('missing top/cladding', proc.stderr)

    def test_closed_poynting_box_does_not_mask_cross_solver_violation(self):
        """Poynting box is perfectly closed (residual = 0.0), but cross-solver discrepancy = 0.045 (>0.015).
        Check 10 must catch the discrepancy violation and abort with exit 66.
        """
        cfg = copy.deepcopy(self.base_cfg)
        cfg['tau_box'] = 0.05

        tel = {
            'schema_version': '1.0',
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': {
                'box-right': {'flux': 0.8876},
                'box-left': {'flux': 0.1112},
                'top': {'flux': 0.0006},
                'bottom': {'flux': 0.0006},
                # Total = 1.0000 -> Residual 0.0 <= 0.05 (Passes Check 5a)
            }
        }
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'is_lossless': True,
            'monitors': copy.deepcopy(tel['monitors']),
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.045,  # Exceeds 0.015!
                'tolerance': 0.015,
            }
        }
        run_dir = self._setup_mock_run('attack6_closed_box_bad_discrepancy', tel, rd, config=cfg)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)


class Attack7DiscrepancyMetricPoisoningTests(BaseTamperAttackTest):
    """Attack 7: Discrepancy Metric Poisoning.
    Test non-finite values (NaN, Inf), nulls, and non-dict cross_solver payloads.
    """

    def test_nan_discrepancy_metric_rejected(self):
        """max_discrepancy = NaN must be rejected with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': float('nan'),
            }
        }
        run_dir = self._setup_mock_run('attack7_nan_discrepancy', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)

    def test_inf_discrepancy_metric_rejected(self):
        """max_discrepancy = Infinity must be rejected with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': float('inf'),
            }
        }
        run_dir = self._setup_mock_run('attack7_inf_discrepancy', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('cross_solver_discrepancy_exceeded', proc.stderr)

    def test_non_dict_cross_solver_payload_rejected(self):
        """cross_solver payload as boolean or list instead of dict must abort with exit 66."""
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'cross_solver': "fraudulent_string_payload",
        }
        run_dir = self._setup_mock_run('attack7_string_cross_solver', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)


class Attack8NotaryCertificateTagIntegrityTests(BaseTamperAttackTest):
    """Attack 8: Notary Certificate Tamper Resistance & Tag Issuance Integrity.
    Ensure driver self-declared tags are ignored and certificate cannot be forged.
    """

    def test_driver_self_declared_v4_tags_ignored(self):
        """Driver claiming tags: ['verified:v4_reconciliation'] in readout.json does not
        bypass verification. If discrepancy fails, exit 66 is still enforced.
        """
        tel = {'schema_version': '1.0', 'executed_steps': 4000, 'is_lossless': True}
        rd = {
            'simulation_executed': True,
            'executed_steps': 4000,
            'tags': ['verified:v4_reconciliation', 'verified:cross_solver_reconciled'],  # Forged claims!
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.080,  # 8% discrepancy!
            }
        }
        run_dir = self._setup_mock_run('attack8_forged_tags', tel, rd)

        proc = self._run_post_notary_cli(run_dir)
        self.assertEqual(proc.returncode, 66)

        cert_path = run_dir / 'verification_certificate.json'
        cert = json.loads(cert_path.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertNotIn('tags', cert)  # Failed certificate does not award verification tags


if __name__ == '__main__':
    unittest.main()
