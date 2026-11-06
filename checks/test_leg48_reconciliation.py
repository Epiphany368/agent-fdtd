"""Acceptance Test Suite for Wave C — LEG 48:
V4 Cross-Solver Reconciliation Gate & Independent Reference Verification.

Acceptance Criteria:
1. Leg48ReconciliationUnitTests:
   - Export format conversion (Meep geometry blocks & RCWA layer stacks).
   - Normalizer mathematical properties (incident power normalization, Fourier order orthogonality).
   - Discrepancy metric calculation (absolute and relative difference, band extrema).
2. Leg48CrossSolverPositiveControl:
   - Dynamic execution of TorchFDTD vs independent cross-solver (Meep / RCWA oracle)
     on the canonical beam-deflecting metagrating benchmark structure.
   - Asserts maximum order efficiency discrepancy <= 1.5% (tolerance 0.015).
   - Asserts energy balance residual |(T + R) - 1.0| <= 1.5%.
3. Leg48NegativeControls:
   - NC 1: Deliberately perturbed geometry (+100 nm ridge) exceeds 1.5% tolerance (exits 66).
   - NC 2: Mismatched normalizations exceed tolerance (exits 66).
   - NC 3: Artificially inflated discrepancies rejected by notary (exits 66, cross_solver_discrepancy_exceeded).
   - NC 4: Missing or zero executed steps in kernel telemetry rejected as fraud (exits 66, tamper_detected).
   - NC 5: Epsilon hash mismatch between telemetry and readout rejected (exits 66, tamper_detected).
   - NC 6: External solver execution failure (non-zero exit code) rejected (exits 66, tamper_detected).
4. Leg48PostNotaryIntegrationTests:
   - Issues 'verified:cross_solver_reconciled' and 'verified:v4_reconciliation' certificate tags.
   - CLI execution of post_notary overrides exit code to 66 on tamper or discrepancy violation.
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

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / 'configs' / 'baseline-pin.json').is_file():
            return parent
    return cur.parents[1]


ROOT = find_project_root(Path(__file__))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

# Strict layout compliance: Reference repository paths directly without peer agent directory fallbacks
POST_NOTARY = ROOT / 'checks' / 'post_notary.py'
DRIVER_PATH = Path(os.environ.get('LEG48_DRIVER_OVERRIDE', str(ROOT / 'src' / 'leg48_cross_reconciliation.py')))
CONFIG_PATH = Path(os.environ.get('LEG48_CONFIG_OVERRIDE', str(ROOT / 'configs' / 'leg48-reconciliation.json')))
LAUNCH = ROOT / 'tools' / 'launch.py'


def load_post_notary():
    import importlib.util
    target = POST_NOTARY
    spec = importlib.util.spec_from_file_location('post_notary', target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_driver():
    import importlib.util
    target = DRIVER_PATH
    spec = importlib.util.spec_from_file_location('leg48_driver', target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Leg48ReconciliationUnitTests(unittest.TestCase):
    """Class 1: Unit tests for export formats, normalizers, and discrepancy metrics."""

    def setUp(self):
        self.driver = load_driver()
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

    def test_export_format_conversion_meep(self):
        """Verifies conversion of benchmark geometry into Meep scene representation."""
        meep_export = self.driver.export_geometry_to_meep(self.cfg)
        self.assertIn('cell_size', meep_export)
        self.assertIn('geometry_blocks', meep_export)
        self.assertEqual(meep_export['cell_size'][0], 2.0)
        self.assertGreaterEqual(len(meep_export['geometry_blocks']), 3)  # Substrate + 2 ridges

        substrate_block = meep_export['geometry_blocks'][0]
        self.assertEqual(substrate_block['type'], 'substrate')
        self.assertAlmostEqual(substrate_block['material_epsilon'], 1.444 ** 2, places=4)

        ridge_blocks = [b for b in meep_export['geometry_blocks'] if b['type'] == 'ridge']
        self.assertEqual(len(ridge_blocks), 2)
        self.assertAlmostEqual(ridge_blocks[0]['size'][0], 0.08, places=4)
        self.assertAlmostEqual(ridge_blocks[1]['size'][0], 0.22, places=4)

    def test_export_format_conversion_rcwa(self):
        """Verifies conversion of benchmark geometry into RCWA layer description."""
        rcwa_export = self.driver.export_geometry_to_rcwa(self.cfg)
        self.assertEqual(rcwa_export['period_um'], 2.0)
        self.assertEqual(rcwa_export['harmonics'], 120)
        self.assertEqual(rcwa_export['order_count'], 241)
        self.assertAlmostEqual(rcwa_export['input_layer']['epsilon'], 1.444 ** 2, places=4)
        self.assertAlmostEqual(rcwa_export['output_layer']['epsilon'], 1.0, places=4)
        self.assertEqual(len(rcwa_export['patterned_layer']['ridges']), 2)

    def test_fourier_order_decomposition_orthogonality(self):
        """Verifies spatial Fourier decomposition accurately separates diffraction orders."""
        nx = 200
        period = 2.0
        wavelength = 1.55
        x = np.linspace(-period / 2, period / 2, nx, endpoint=False)
        k0 = 2.0 * math.pi / wavelength

        # Construct pure order m = +1 wave in air (medium_index = 1.0)
        m_target = 1
        kx_target = 2.0 * math.pi * m_target / period
        ky_target = math.sqrt(k0 ** 2 - kx_target ** 2)

        ez = np.exp(1j * kx_target * x)
        hx = (ky_target / k0) * ez

        f_powers, b_powers, amps = self.driver.fourier_order_decomposition(
            x, ez, hx, wavelength, period, 1.0, orders=[-1, 0, 1]
        )

        # Forward power of order +1 should be nonzero (~ky/k0 / 2) while other orders are ~0
        expected_p = 0.5 * (ky_target / k0)
        self.assertAlmostEqual(f_powers['+1'], expected_p, places=5)
        self.assertAlmostEqual(f_powers['+0'], 0.0, places=5)
        self.assertAlmostEqual(f_powers['-1'], 0.0, places=5)
        self.assertAlmostEqual(b_powers['+1'], 0.0, places=5)

    def test_normalizer_mathematical_properties(self):
        """Verifies normalization against incident reference power and error handling."""
        f_powers = {'-1': 0.04, '0': 0.07, '+1': 0.77}
        b_powers = {'-1': 0.045, '0': 0.015, '+1': 0.05}
        p_inc = 0.98

        eff = self.driver.compute_normalized_efficiencies(f_powers, b_powers, p_inc)
        self.assertAlmostEqual(eff['T+1'], 0.77 / 0.98, places=5)
        self.assertAlmostEqual(eff['R0'], 0.015 / 0.98, places=5)

        # Non-positive incident power must raise ValueError
        with self.assertRaises(ValueError):
            self.driver.compute_normalized_efficiencies(f_powers, b_powers, 0.0)
        with self.assertRaises(ValueError):
            self.driver.compute_normalized_efficiencies(f_powers, b_powers, -1.0)

    def test_discrepancy_metric_calculation(self):
        """Verifies discrepancy metrics across all diffraction orders."""
        torch_eff = {'T-1': 0.040, 'T0': 0.071, 'T+1': 0.776, 'R-1': 0.045, 'R0': 0.016, 'R+1': 0.051}
        ext_eff = {'T-1': 0.041, 'T0': 0.070, 'T+1': 0.775, 'R-1': 0.045, 'R0': 0.015, 'R+1': 0.050}

        max_abs, diffs, max_rel = self.driver.calculate_cross_solver_discrepancy(torch_eff, ext_eff)
        self.assertAlmostEqual(max_abs, 0.001, places=6)
        self.assertGreater(max_rel, 0.0)

        # Empty match must raise ValueError
        with self.assertRaises(ValueError):
            self.driver.calculate_cross_solver_discrepancy({'X': 1.0}, {'Y': 1.0})


class Leg48CrossSolverPositiveControl(unittest.TestCase):
    """Class 2: Positive control verifying TorchFDTD vs independent solvers <= 1.5% discrepancy."""

    def setUp(self):
        self.driver = load_driver()
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

    def test_cross_solver_reconciliation_meep_within_tolerance(self):
        """Reconciliation of metagrating benchmark against Meep 1.34.0 passes within 1.5%."""
        exit_code, readout = self.driver.run_cross_solver_reconciliation(self.cfg, solver_name='meep')
        self.assertEqual(exit_code, 0)
        self.assertEqual(readout['verdict'], 'accepted')

        cs = readout['cross_solver']
        self.assertTrue(cs['reconciled'])
        self.assertLessEqual(cs['max_discrepancy'], 0.015)
        self.assertEqual(cs['external_solver']['name'], 'meep')
        self.assertEqual(cs['external_solver']['version'], '1.34.0')

        # Energy balance must close within tau_energy
        self.assertAlmostEqual(cs['energy_balance_sum'], 1.0, delta=0.015)

    def test_cross_solver_reconciliation_rcwa_oracle_within_tolerance(self):
        """Reconciliation of metagrating benchmark against TORCWA RCWA oracle passes within 1.5%."""
        exit_code, readout = self.driver.run_cross_solver_reconciliation(self.cfg, solver_name='rcwa')
        self.assertEqual(exit_code, 0)
        self.assertEqual(readout['verdict'], 'accepted')

        cs = readout['cross_solver']
        self.assertTrue(cs['reconciled'])
        self.assertLessEqual(cs['max_discrepancy'], 0.015)
        self.assertEqual(cs['external_solver']['name'], 'rcwa')


class Leg48NegativeControls(unittest.TestCase):
    """Class 3: Negative controls asserting post_notary enforces exit 66 on violations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg48_nc_')
        self.post_notary = load_post_notary()
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_mock_run(self, run_name: str, telemetry: dict, readout: dict, config: dict) -> Path:
        run_dir = Path(self.test_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_copy = dict(config)
        default_mons = {
            'reflection': {'flux': 0.1112},
            'transmission': {'flux': 0.8876}
        }

        tel_copy = dict(telemetry)
        if 'monitors' not in tel_copy:
            tel_copy['monitors'] = copy.deepcopy(default_mons)

        readout_copy = dict(readout)
        if 'monitors' not in readout_copy:
            readout_copy['monitors'] = copy.deepcopy(tel_copy['monitors'])

        (run_dir / 'solver_telemetry.json').write_text(json.dumps(tel_copy, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout_copy, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(cfg_copy, indent=2), encoding='utf-8')
        return run_dir

    def test_nc1_perturbed_geometry_exceeds_tolerance_exits_66(self):
        """NC 1: Deliberately perturbed geometry (+100 nm ridge) exceeds 1.5% tolerance -> exit 66."""
        perturbed_cfg = copy.deepcopy(self.cfg)
        perturbed_cfg['ridges'][0]['width_um'] = 0.18  # Perturb from 0.08 um to 0.18 um (+125%)

        driver = load_driver()
        exit_code, readout = driver.run_cross_solver_reconciliation(perturbed_cfg, solver_name='meep')
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(readout['verdict'], 'rejected')
        self.assertGreater(readout['cross_solver']['max_discrepancy'], 0.015)

        tel = {
            'schema_version': '1.0',
            'executed_steps': 12000,
            'is_lossless': True,
            'monitors': copy.deepcopy(readout['monitors'])
        }

        run_dir = self._setup_mock_run('nc1_perturbed', telemetry=tel, readout=readout, config=perturbed_cfg)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))

    def test_nc2_mismatched_normalization_exceeds_tolerance_exits_66(self):
        """NC 2: Mismatched normalization factor causes discrepancy > 1.5% -> exit 66."""
        bad_readout = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.045,  # 4.5% discrepancy due to unnormalized power
                'tolerance': 0.015,
            }
        }

        run_dir = self._setup_mock_run(
            'nc2_norm_mismatch',
            telemetry={'executed_steps': 12000, 'is_lossless': True},
            readout=bad_readout,
            config=self.cfg
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))

    def test_nc3_artificially_inflated_discrepancy_rejected(self):
        """NC 3: Readout claiming reconciliation with inflated max discrepancy is rejected."""
        inflated_readout = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.025,
                'tolerance': 0.015,
            }
        }

        run_dir = self._setup_mock_run(
            'nc3_inflated',
            telemetry={'executed_steps': 12000, 'is_lossless': True},
            readout=inflated_readout,
            config=self.cfg
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))

    def test_nc4_missing_or_zero_executed_steps_tamper_detected(self):
        """NC 4: Reconciliation claim with zero executed steps in kernel telemetry is rejected as tamper."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.005,
            }
        }

        run_dir = self._setup_mock_run(
            'nc4_zero_steps',
            telemetry={'executed_steps': 0, 'is_lossless': True},
            readout=readout,
            config=self.cfg
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))

    def test_nc5_mismatched_epsilon_hash_tamper_detected(self):
        """NC 5: Epsilon hash mismatch between telemetry and readout is rejected as tamper."""
        readout = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.005,
                'epsilon_sha256': '1' * 64,
            }
        }

        run_dir = self._setup_mock_run(
            'nc5_epsilon_mismatch',
            telemetry={'executed_steps': 12000, 'is_lossless': True, 'epsilon_sha256': '2' * 64},
            readout=readout,
            config=self.cfg
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))

    def test_nc6_external_solver_failure_or_missing_provenance(self):
        """NC 6: External solver failure (non-zero exit code) or missing solver name is rejected."""
        readout_failed = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 1},
                'max_discrepancy': 0.005,
            }
        }
        run_dir = self._setup_mock_run(
            'nc6_ext_failed',
            telemetry={'executed_steps': 12000, 'is_lossless': True},
            readout=readout_failed,
            config=self.cfg
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))


class Leg48PostNotaryIntegrationTests(unittest.TestCase):
    """Class 4: End-to-end post-flight notary certification and CLI execution tests."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg48_notary_integ_')
        self.post_notary = load_post_notary()
        self.driver = load_driver()
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_post_notary_issues_v4_tags_and_certificate(self):
        """Valid reconciliation run earns verified:cross_solver_reconciled certificate and exits 0."""
        run_dir = Path(self.test_dir) / 'valid_run'
        exit_code, readout = self.driver.run_cross_solver_reconciliation(self.cfg, run_dir=run_dir, solver_name='meep')
        self.assertEqual(exit_code, 0)
        # Ensure config.input.json exists for post_notary
        (run_dir / 'config.input.json').write_text(json.dumps(self.cfg, indent=2), encoding='utf-8')

        # Run verification via function call
        cert = self.post_notary.verify_post_flight(run_dir)
        self.assertTrue(cert['verified'])
        self.assertIn('verified:cross_solver_reconciled', cert['tags'])
        self.assertIn('verified:v4_reconciliation', cert['tags'])
        self.assertTrue(cert['cross_solver_reconciliation']['reconciled'])
        self.assertLessEqual(cert['cross_solver_reconciliation']['max_discrepancy'], 0.015)

        # Run verification via CLI subprocess
        target_script = POST_NOTARY
        cmd = [sys.executable, str(target_script), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('OK: verification certificate issued', proc.stdout)

    def test_post_notary_overrides_to_exit_66_on_violation(self):
        """Violations override exit code to 66 and record failure in verification certificate."""
        run_dir = Path(self.test_dir) / 'violating_run'
        run_dir.mkdir(parents=True, exist_ok=True)

        telemetry = {
            'schema_version': '1.0',
            'executed_steps': 12000,
            'is_lossless': True,
            'monitors': {
                'reflection': {'flux': 0.1112},
                'transmission': {'flux': 0.8876}
            }
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 12000,
            'is_lossless': True,
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.035,  # Exceeds 0.015
                'tolerance': 0.015,
            },
            'monitors': {
                'reflection': {'flux': 0.1112},
                'transmission': {'flux': 0.8876}
            }
        }
        config = {
            'kind': 'reconciliation',
            'reconciliation_tolerance': 0.015
        }

        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout, indent=2), encoding='utf-8')
        (run_dir / 'config.input.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

        target_script = POST_NOTARY
        cmd = [sys.executable, str(target_script), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)

        cert_file = run_dir / 'verification_certificate.json'
        self.assertTrue(cert_file.is_file())
        cert = json.loads(cert_file.read_text(encoding='utf-8'))
        self.assertFalse(cert['verified'])
        self.assertEqual(cert['status'], 'cross_solver_discrepancy_exceeded')


if __name__ == '__main__':
    unittest.main()
