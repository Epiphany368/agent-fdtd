"""Adversarial Empirical Challenge Test Suite for Wave C — LEG 48:
Post-Flight Notary Check 10 Cross-Solver Reconciliation Gate & Tamper Defenses.

Author: challenger_leg48_4 (Empirical Challenger / Critic, Specialist)
Mandate:
Perform code-executing adversarial stress testing against post-flight notary Check 10
in checks/post_notary.py and checks/test_leg48_reconciliation.py:
- Attack 1: Fabricated readout.json claiming max_discrepancy=0.0001 and reconciled=true
  while solver_telemetry.json has executed_steps: 0 or missing/invalid steps.
  Verify Check 10 rejects with exit 66 (tamper_detected).
- Attack 2: Corrupt epsilon_sha256 in solver_telemetry.json or swap geometry parameter
  in effective.json / config.input.json. Verify hash tampering detection and exit 66 (tamper_detected).
- Attack 3: Artificially alter diffracted orders so that discrepancy against Meep oracle
  is 0.020 > 0.015. Verify Check 10 overrides exit status to 66 (cross_solver_discrepancy_exceeded).
- Attack 4: Missing external solver provenance or malformed benchmark schema.
  Verify exit 66 defense.
- Control: Authentic reconciliation passes cleanly with exit 0 and verified:v4_reconciliation.
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

POST_NOTARY = ROOT / 'checks' / 'post_notary.py'
CONFIG_PATH = ROOT / 'configs' / 'leg48-reconciliation.json'


def load_post_notary():
    import importlib.util
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChallengerLeg48NotaryAttacks(unittest.TestCase):
    """Adversarial challenge test suite executing empirical attacks against Check 10."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='challenger_leg48_notary_')
        self.post_notary = load_post_notary()
        if CONFIG_PATH.is_file():
            self.base_cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        else:
            self.base_cfg = {
                'schema_version': '1.0',
                'kind': 'benchmark',
                'cross_solver': True,
                'reconciliation_tolerance': 0.015,
                'tau_box': 0.05,
            }
        self.canonical_hash = 'a' * 64
        self.default_telemetry_monitors = {
            'reflection': {'flux': 0.1112},
            'transmission': {'flux': 0.8876},
        }
        self.default_readout_monitors = {
            'reflection': {'flux': 0.1112},
            'transmission': {'flux': 0.8876},
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_run(
        self,
        name: str,
        telemetry: dict,
        readout: dict,
        config: dict = None,
        effective: dict = None,
    ) -> Path:
        """Create mock run directory populated with test artifacts."""
        run_dir = Path(self.test_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = dict(config if config is not None else self.base_cfg)
        cfg.setdefault('schema_version', '1.0')
        cfg.setdefault('kind', 'benchmark')
        cfg.setdefault('cross_solver', True)
        cfg.setdefault('reconciliation_tolerance', 0.015)

        tel = copy.deepcopy(telemetry)
        tel.setdefault('schema_version', '1.0')
        tel.setdefault('is_lossless', True)
        if 'monitors' not in tel:
            tel['monitors'] = copy.deepcopy(self.default_telemetry_monitors)

        rd = copy.deepcopy(readout)
        rd.setdefault('schema_version', '1.0')
        rd.setdefault('is_lossless', True)
        if 'monitors' not in rd:
            rd['monitors'] = copy.deepcopy(self.default_readout_monitors)

        (run_dir / 'solver_telemetry.json').write_text(json.dumps(tel, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(rd, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

        if effective is not None:
            (run_dir / 'effective.json').write_text(json.dumps(effective, indent=2), encoding='utf-8')

        return run_dir

    def _run_cli(self, run_dir: Path) -> tuple[int, str, dict]:
        """Invoke post_notary CLI in a separate process and return (exit_code, output, cert)."""
        cmd = [sys.executable, '-B', str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        cert = {}
        cert_file = run_dir / 'verification_certificate.json'
        if cert_file.is_file():
            try:
                cert = json.loads(cert_file.read_text(encoding='utf-8'))
            except Exception:
                pass
        output = proc.stdout + '\n' + proc.stderr
        return proc.returncode, output, cert

    # =========================================================================
    # Attack 1: Zero-Stepped Simulation Spoofing
    # =========================================================================

    def test_attack_1a_executed_steps_zero_rejected_with_exit_66(self):
        """Attack 1a: Fraudulent readout claiming discrepancy=0.0001 with executed_steps=0 exits 66."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 0,  # Fraudulent claim that reconciliation passed with 0 steps
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'reconciled': True,
            },
        }
        telemetry = {
            'executed_steps': 0,  # Zero steps executed in kernel
        }

        run_dir = self._setup_run('attack_1a_zero_steps', telemetry=telemetry, readout=readout)

        # 1. Direct function call verification triggers Check 10
        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))
        self.assertIn('verified positive executed_steps', str(ctx.exception))

        # 2. CLI process execution verification
        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertFalse(cert.get('verified', True))
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_1b_missing_executed_steps_rejected_with_exit_66(self):
        """Attack 1b: Telemetry entirely omitting executed_steps key is rejected with exit 66."""
        readout = {
            'simulation_executed': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'reconciled': True,
            },
        }
        telemetry = {}  # Missing executed_steps

        run_dir = self._setup_run('attack_1b_missing_steps', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_1c_negative_or_nan_executed_steps_rejected(self):
        """Attack 1c: Negative or non-numeric executed_steps rejected as tampering with exit 66."""
        readout = {
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
            },
        }

        for bad_steps in [-100, 'non_numeric', None]:
            tel = {'executed_steps': bad_steps}
            run_dir = self._setup_run(f'attack_1c_bad_steps_{bad_steps}', telemetry=tel, readout=readout)

            rc, out, cert = self._run_cli(run_dir)
            self.assertEqual(rc, 66, f'Failed to reject bad steps: {bad_steps}')
            self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_1d_step_count_mismatch_between_telemetry_and_readout(self):
        """Attack 1d: Readout claiming 12000 steps while telemetry reports 6000 exits 66 (tamper_detected)."""
        telemetry = {'executed_steps': 6000}
        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
            },
        }

        run_dir = self._setup_run('attack_1d_step_mismatch', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('step count mismatch', str(ctx.exception))
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    # =========================================================================
    # Attack 2: Permittivity SHA-256 Digest & Geometry Tampering
    # =========================================================================

    def test_attack_2a_corrupted_telemetry_epsilon_hash_detected(self):
        """Attack 2a: Flipped bits in solver_telemetry.json epsilon_sha256 trigger exit 66."""
        true_hash = hashlib.sha256(b'authentic_grating_mesh_v4').hexdigest()
        tampered_hash = hashlib.sha256(b'tampered_grating_mesh_v4').hexdigest()

        telemetry = {
            'executed_steps': 12000,
            'epsilon_sha256': tampered_hash,  # Corrupted kernel digest
        }
        readout = {
            'executed_steps': 12000,
            'epsilon_sha256': true_hash,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'epsilon_sha256': true_hash,
            },
        }

        run_dir = self._setup_run('attack_2a_corrupted_hash', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('epsilon_sha256 mismatch', str(ctx.exception))
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_2b_cross_solver_readout_hash_mismatch(self):
        """Attack 2b: Readout cross_solver section claiming divergent epsilon_sha256 exits 66."""
        tel_hash = '1' * 64
        cs_hash = '2' * 64

        telemetry = {
            'executed_steps': 12000,
            'epsilon_sha256': tel_hash,
        }
        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'epsilon_sha256': cs_hash,  # Discrepant hash declared in cross_solver block
            },
        }

        run_dir = self._setup_run('attack_2b_cs_hash_mismatch', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver epsilon_sha256 mismatch', str(ctx.exception))
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_2c_swapped_geometry_effective_json_tamper(self):
        """Attack 2c: Swapped geometry parameters in effective.json/config detected as tampering."""
        authentic_hash = hashlib.sha256(b'authentic_metagrating_geometry').hexdigest()
        swapped_hash = hashlib.sha256(b'swapped_perturbed_geometry_w_0.25').hexdigest()

        # Attacker ran with perturbed geometry producing swapped_hash,
        # but claims authentic_hash in readout to mask perturbation
        telemetry = {
            'executed_steps': 12000,
            'epsilon_sha256': swapped_hash,
        }
        readout = {
            'executed_steps': 12000,
            'epsilon_sha256': authentic_hash,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,
                'epsilon_sha256': authentic_hash,
            },
        }
        effective = {
            'ridges': [{'width_um': 0.25, 'height_um': 0.22}],  # Swapped geometry
        }

        run_dir = self._setup_run(
            'attack_2c_swapped_geometry',
            telemetry=telemetry,
            readout=readout,
            effective=effective,
        )

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    # =========================================================================
    # Attack 3: Artificially Altered Diffracted Orders (Discrepancy Violation)
    # =========================================================================

    def test_attack_3a_discrepancy_exceeds_threshold_exits_66(self):
        """Attack 3a: Explicit max_discrepancy = 0.020 > 0.015 overrides exit code to 66."""
        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.020,  # 2.0% > 1.5% tolerance
                'tolerance': 0.015,
                'reconciled': False,
            },
        }
        telemetry = {'executed_steps': 12000}

        run_dir = self._setup_run('attack_3a_disc_0.020', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))
        self.assertIn('exceeds tolerance', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertFalse(cert.get('verified', True))
        self.assertEqual(cert.get('status'), 'cross_solver_discrepancy_exceeded')

    def test_attack_3b_order_level_tamper_detected_despite_fraudulent_summary(self):
        """Attack 3b: Fraudulent summary claims 0.001 but altered order T_+1 discrepancy is 0.020."""
        # Genuine Meep efficiencies for benchmark
        meep_orders = {
            'T_-1': 0.0405,
            'T_0': 0.0709,
            'T_+1': 0.7761,
            'R_-1': 0.0445,
            'R_0': 0.0157,
            'R_+1': 0.0508,
        }
        # Attacker maliciously alters T_+1 in TorchFDTD efficiencies by +0.020
        altered_torch_orders = copy.deepcopy(meep_orders)
        altered_torch_orders['T_+1'] = 0.7961  # diff = 0.0200 > 0.0150!

        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.001,  # Fraudulent claim in summary!
                'tolerance': 0.015,
                'torchfdtd_efficiencies': altered_torch_orders,
                'external_efficiencies': meep_orders,
            },
        }
        telemetry = {'executed_steps': 12000}

        run_dir = self._setup_run('attack_3b_order_level_tamper', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))
        self.assertIn('cross-solver efficiency discrepancy for order T_+1', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'cross_solver_discrepancy_exceeded')

    def test_attack_3c_discrepancy_metric_poisoning_nan_or_inf(self):
        """Attack 3c: Poisoning max_discrepancy with NaN, Inf, or string exits 66."""
        telemetry = {'executed_steps': 12000}

        for bad_val in [float('nan'), float('inf'), 'invalid_metric', None]:
            readout = {
                'executed_steps': 12000,
                'cross_solver': {
                    'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                    'max_discrepancy': bad_val,
                },
            }
            run_dir = self._setup_run(f'attack_3c_poison_{type(bad_val).__name__}', telemetry=telemetry, readout=readout)

            rc, out, cert = self._run_cli(run_dir)
            self.assertEqual(rc, 66)
            self.assertEqual(cert.get('status'), 'cross_solver_discrepancy_exceeded')

    def test_attack_3d_tolerance_expansion_attempt_rejected(self):
        """Attack 3d: Driver expanding tolerance to 0.05 in readout while config pins 0.015 exits 66."""
        config = {
            'kind': 'benchmark',
            'cross_solver': True,
            'reconciliation_tolerance': 0.015,  # Pinned by config
        }
        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.020,  # 2.0%
                'tolerance': 0.050,  # Attacker attempts to loosen tolerance to 5%
            },
        }
        telemetry = {'executed_steps': 12000}

        run_dir = self._setup_run('attack_3d_tol_expansion', telemetry=telemetry, readout=readout, config=config)

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'cross_solver_discrepancy_exceeded')

    # =========================================================================
    # Attack 4: Missing Provenance or Malformed Benchmark Schema
    # =========================================================================

    def test_attack_4a_missing_cross_solver_section_in_reconciliation_run(self):
        """Attack 4a: Reconciliation run completely omitting cross_solver section exits 66."""
        readout = {
            'executed_steps': 12000,
            'simulation_executed': True,
            # Missing 'cross_solver' object
        }
        telemetry = {'executed_steps': 12000}

        run_dir = self._setup_run('attack_4a_missing_cs_dict', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('missing cross_solver dictionary', str(ctx.exception))
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_4b_missing_or_empty_external_solver_provenance(self):
        """Attack 4b: Omitting external_solver provenance or empty solver name exits 66."""
        telemetry = {'executed_steps': 12000}

        for bad_ext in [None, {}, {'version': '1.34.0'}, {'name': ''}]:
            readout = {
                'executed_steps': 12000,
                'cross_solver': {
                    'external_solver': bad_ext,
                    'max_discrepancy': 0.0001,
                },
            }
            run_dir = self._setup_run(f'attack_4b_bad_ext_{type(bad_ext).__name__}', telemetry=telemetry, readout=readout)

            rc, out, cert = self._run_cli(run_dir)
            self.assertEqual(rc, 66)
            self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_4c_external_solver_execution_failure_exit_code_non_zero(self):
        """Attack 4c: External solver failure (exit_code != 0, e.g. 1 or 139) exits 66."""
        telemetry = {'executed_steps': 12000}

        for fail_code in [1, 139, -9]:
            readout = {
                'executed_steps': 12000,
                'cross_solver': {
                    'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': fail_code},
                    'max_discrepancy': 0.0001,
                },
            }
            run_dir = self._setup_run(f'attack_4c_fail_code_{fail_code}', telemetry=telemetry, readout=readout)

            with self.assertRaises(RuntimeError) as ctx:
                self.post_notary.verify_post_flight(run_dir)
            self.assertIn('external solver execution failed', str(ctx.exception))
            self.assertIn('tamper_detected', str(ctx.exception))

            rc, out, cert = self._run_cli(run_dir)
            self.assertEqual(rc, 66)
            self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_4d_telemetry_discrepancy_conflict_detected(self):
        """Attack 4d: Conflict between telemetry cross_solver discrepancy and readout exits 66."""
        telemetry = {
            'executed_steps': 12000,
            'cross_solver': {
                'max_discrepancy': 0.025,  # Kernel recorded failure
            },
        }
        readout = {
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.0001,  # Readout claims success!
            },
        }

        run_dir = self._setup_run('attack_4d_conflict', telemetry=telemetry, readout=readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver discrepancy mismatch', str(ctx.exception))
        self.assertIn('tamper_detected', str(ctx.exception))

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertEqual(cert.get('status'), 'tamper_detected')

    def test_attack_4e_forged_tags_injection_resisted(self):
        """Attack 4e: Driver injecting fake verified:v4_reconciliation tags rejected on violation."""
        readout = {
            'executed_steps': 12000,
            'tags': ['verified:cross_solver_reconciled', 'verified:v4_reconciliation'],  # Forged claims!
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.035,  # Exceeds tolerance
            },
        }
        telemetry = {'executed_steps': 12000}

        run_dir = self._setup_run('attack_4e_forged_tags', telemetry=telemetry, readout=readout)

        rc, out, cert = self._run_cli(run_dir)
        self.assertEqual(rc, 66)
        self.assertFalse(cert.get('verified', True))
        self.assertNotIn('verified:cross_solver_reconciled', cert.get('tags', []))
        self.assertNotIn('verified:v4_reconciliation', cert.get('tags', []))

    # =========================================================================
    # Positive Control: Genuine Reconciliation Verification
    # =========================================================================

    def test_positive_control_genuine_reconciliation_issues_v4_tags(self):
        """Positive Control: Genuine reconciliation run within 1.5% tolerance earns certificate."""
        ep_hash = hashlib.sha256(b'authentic_silicon_metagrating').hexdigest()
        telemetry = {
            'executed_steps': 12000,
            'is_lossless': True,
            'epsilon_sha256': ep_hash,
            'monitors': copy.deepcopy(self.default_telemetry_monitors),
        }
        meep_orders = {
            'T_-1': 0.040572,
            'T_0': 0.070913,
            'T_+1': 0.776073,
            'R_-1': 0.044553,
            'R_0': 0.015728,
            'R_+1': 0.050831,
        }
        # TorchFDTD outputs closely tracking Meep within 0.01%
        torch_orders = {
            'T_-1': 0.040600,
            'T_0': 0.070950,
            'T_+1': 0.776050,
            'R_-1': 0.044580,
            'R_0': 0.015700,
            'R_+1': 0.050800,
        }
        readout = {
            'executed_steps': 12000,
            'is_lossless': True,
            'epsilon_sha256': ep_hash,
            'monitors': copy.deepcopy(self.default_readout_monitors),
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0, 'provenance': 'conda-forge'},
                'max_discrepancy': 0.000045,
                'tolerance': 0.015,
                'reconciled': True,
                'torchfdtd_efficiencies': torch_orders,
                'external_efficiencies': meep_orders,
                'epsilon_sha256': ep_hash,
            },
        }

        run_dir = self._setup_run('positive_control', telemetry=telemetry, readout=readout)

        cert = self.post_notary.verify_post_flight(run_dir)
        self.assertTrue(cert['verified'])
        self.assertIn('verified:cross_solver_reconciled', cert['tags'])
        self.assertIn('verified:v4_reconciliation', cert['tags'])
        self.assertEqual(cert['cross_solver_reconciliation']['external_solver'], 'meep')
        self.assertTrue(cert['cross_solver_reconciliation']['reconciled'])

        rc, out, cli_cert = self._run_cli(run_dir)
        self.assertEqual(rc, 0)
        self.assertTrue(cli_cert['verified'])
        self.assertIn('OK: verification certificate issued', out)


if __name__ == '__main__':
    unittest.main()
