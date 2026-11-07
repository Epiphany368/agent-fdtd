"""Adversarial Empirical Challenge Test Suite for Wave D — LEG 49.

Target: M3 3-Tier Fabrication Constraints & Check 11 (I-notary) Tamper Defenses.
Role: challenger_leg49_2 (EMPIRICAL CHALLENGER).

This challenge suite stress-tests:
1. High spatial frequency noise, Nyquist comb patterns, and alternating sub-150 nm structures
   against Tier 3 morphological filter guarantees (zero violations, min feature >= 150 nm).
2. Boundary condition slivers positioned at simulation domain edges.
3. 2D checkerboards and isolated single-cell pinholes / islands.
4. Tamper attacks on solver_telemetry.json and readout.json:
   - Spoofed / out-of-tolerance gradient relative error and omitted verification records.
   - Omitted, zero, or negative executed steps in telemetry or forward/adjoint records.
   - Passivity violation injections (P_out > 1 + tau_pml) and non-lossless spoofing.
   - Sub-150 nm tooth and trench injections (including 149 nm near-miss) and conflicting violation reports.
   - Non-monotonic objective regressions (negative gain, flat trajectory, spoofed gain).
   - Mismatched forward and adjoint epsilon SHA-256 tensor digests.
   - Missing or corrupted inverse design sections.
5. Strict verification that ALL attacks are defeated by Check 11 triggering exit code 66
   with verified: false in verification_certificate.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.leg49_m3_constraints import M3FilterProjection


class AdversarialM3FilterMorphologyStressTests(unittest.TestCase):
    """Stress tests challenging Tier 1-3 M3 fabrication filters with pathological inputs."""

    def setUp(self) -> None:
        self.dx_um = 0.020  # 20 nm cell size
        self.min_feature_um = 0.150  # 150 nm
        self.m3 = M3FilterProjection(
            dx_um=self.dx_um,
            min_feature_um=self.min_feature_um,
            filter_radius_um=self.min_feature_um,
            dtype=torch.float64,
        )

    def test_attack_nyquist_alternating_comb(self) -> None:
        """Attack: 1-cell alternating comb [1, 0, 1, 0, ...] (20 nm features << 150 nm)."""
        n = 300
        # Alternating pattern: every cell flips between solid and void
        comb = np.zeros(n, dtype=bool)
        comb[::2] = True

        # Pre-filter: must fail M3 with massive violations
        pre_check = self.m3.check_violations(comb)
        self.assertFalse(pre_check["compliant"])
        self.assertLess(pre_check["min_tooth_width_nm"], 150.0)
        self.assertLess(pre_check["min_trench_width_nm"], 150.0)

        # Apply Tier 3 morphological filter
        filtered = self.m3.project_m3_discrete(comb)
        post_check = self.m3.check_violations(filtered)

        # Defense guarantee: zero violations post-filter
        self.assertTrue(
            post_check["compliant"],
            f"Filtered Nyquist comb must satisfy M3 constraints: {post_check}",
        )
        self.assertEqual(len(post_check["violations"]), 0)
        self.assertGreaterEqual(post_check["min_tooth_width_nm"], 150.0)
        self.assertGreaterEqual(post_check["min_trench_width_nm"], 150.0)

    def test_attack_sub_150nm_near_threshold_teeth_trenches(self) -> None:
        """Attack: Sub-threshold comb with teeth/trenches at 100 nm (5 cells) and 120 nm (6 cells)."""
        n = 400
        arr = np.zeros(n, dtype=bool)
        # Construct alternating blocks: 6 cells 1 (120 nm), 6 cells 0 (120 nm), 5 cells 1 (100 nm)
        idx = 20
        while idx + 20 < n - 20:
            arr[idx : idx + 6] = True  # 120 nm tooth (< 150 nm)
            idx += 12                  # 120 nm trench (< 150 nm)
            arr[idx : idx + 5] = True  # 100 nm tooth (< 150 nm)
            idx += 10                  # 100 nm trench (< 150 nm)

        pre_check = self.m3.check_violations(arr)
        self.assertFalse(pre_check["compliant"])
        self.assertTrue(len(pre_check["violations"]) > 0)

        filtered = self.m3.project_m3_discrete(arr)
        post_check = self.m3.check_violations(filtered)

        self.assertTrue(
            post_check["compliant"],
            f"Near-threshold sub-150nm structures must be eradicated: {post_check}",
        )
        self.assertEqual(len(post_check["violations"]), 0)
        self.assertGreaterEqual(post_check["min_tooth_width_nm"], 150.0)
        self.assertGreaterEqual(post_check["min_trench_width_nm"], 150.0)

    def test_attack_boundary_slivers_at_domain_edges(self) -> None:
        """Attack: Isolated sub-150 nm slivers at domain boundaries index 0 and index N-1."""
        n = 250
        # Valid central body
        arr = np.zeros(n, dtype=bool)
        arr[50:150] = True  # 2000 nm solid body

        # Inject left-boundary sliver: 1 cell (20 nm) tooth at index 0
        arr[0] = True
        # Inject right-boundary sliver: 2 cells (40 nm) tooth at index n-2..n-1
        arr[n - 2 : n] = True

        pre_check = self.m3.check_violations(arr)
        self.assertFalse(pre_check["compliant"])

        filtered = self.m3.project_m3_discrete(arr)
        post_check = self.m3.check_violations(filtered)

        self.assertTrue(
            post_check["compliant"],
            f"Boundary slivers must be eliminated without residual edge artifacts: {post_check}",
        )
        self.assertEqual(len(post_check["violations"]), 0)
        self.assertGreaterEqual(post_check["min_tooth_width_nm"], 150.0)
        self.assertGreaterEqual(post_check["min_trench_width_nm"], 150.0)

    def test_attack_boundary_void_slivers_at_domain_edges(self) -> None:
        """Attack: Isolated void slivers (narrow trenches) directly abutting domain boundaries."""
        n = 250
        # Fully solid domain
        arr = np.ones(n, dtype=bool)

        # Inject 1-cell void trench at left edge (index 0)
        arr[0] = False
        # Inject 2-cell void trench at right edge (indices n-2, n-1)
        arr[n - 2 :] = False

        pre_check = self.m3.check_violations(arr)
        self.assertFalse(pre_check["compliant"])

        filtered = self.m3.project_m3_discrete(arr)
        post_check = self.m3.check_violations(filtered)

        self.assertTrue(
            post_check["compliant"],
            f"Boundary void slivers must be closed: {post_check}",
        )
        self.assertEqual(len(post_check["violations"]), 0)
        self.assertGreaterEqual(post_check["min_tooth_width_nm"], 150.0)
        self.assertGreaterEqual(post_check["min_trench_width_nm"], 150.0)

    def test_attack_isolated_pinholes_and_islands(self) -> None:
        """Attack: Isolated single-cell pinhole (void in solid) and island (solid in void)."""
        n = 200
        # Pinhole: solid region with 1 cell hole
        arr_pinhole = np.ones(n, dtype=bool)
        arr_pinhole[100] = False  # 20 nm hole

        filt_pinhole = self.m3.project_m3_discrete(arr_pinhole)
        check_pinhole = self.m3.check_violations(filt_pinhole)
        self.assertTrue(check_pinhole["compliant"], f"Pinhole must be closed: {check_pinhole}")
        self.assertEqual(len(check_pinhole["violations"]), 0)

        # Island: void region with 1 cell solid
        arr_island = np.zeros(n, dtype=bool)
        arr_island[100] = True  # 20 nm island

        filt_island = self.m3.project_m3_discrete(arr_island)
        check_island = self.m3.check_violations(filt_island)
        self.assertTrue(check_island["compliant"], f"Island must be removed: {check_island}")
        self.assertEqual(len(check_island["violations"]), 0)

    def test_attack_extreme_beta_projection_numerical_stability(self) -> None:
        """Attack: Sharp continuation beta (beta=16.0, 32.0) on random high-frequency density."""
        torch.manual_seed(12345)
        n = 60
        rho = torch.rand(n, dtype=torch.float64)

        for beta in [8.0, 16.0, 32.0]:
            # Forward projection
            proj = self.m3.forward(rho, beta=beta)
            self.assertFalse(torch.isnan(proj).any(), f"NaN in projection at beta={beta}")
            self.assertFalse(torch.isinf(proj).any(), f"Inf in projection at beta={beta}")
            self.assertTrue((proj >= 0.0 - 1e-9).all() and (proj <= 1.0 + 1e-9).all())

            # Jacobian computation
            jac = self.m3.compute_analytic_jacobian(rho, beta=beta)
            self.assertFalse(torch.isnan(jac).any(), f"NaN in Jacobian at beta={beta}")
            self.assertFalse(torch.isinf(jac).any(), f"Inf in Jacobian at beta={beta}")


class AdversarialCheck11NotaryTamperAttacks(unittest.TestCase):
    """Adversarial tamper attacks against Check 11 (I-notary) in checks/post_notary.py."""

    def setUp(self) -> None:
        self.temp_dirs: list[Path] = []

    def tearDown(self) -> None:
        for p in self.temp_dirs:
            shutil.rmtree(p, ignore_errors=True)

    def _create_base_valid_run(self) -> tuple[Path, dict, dict, dict]:
        """Create a valid baseline inverse design run structure."""
        td = Path(tempfile.mkdtemp(prefix="adv_notary_run_"))
        self.temp_dirs.append(td)

        config = {
            "kind": "inverse_design",
            "inverse_design": True,
            "iterations": 2,
            "wavelength_um": 1.55,
            "tags": ["inverse_design", "inverse_design_step"],
        }
        steps = [
            {
                "step_index": 0,
                "objective_value": 0.50,
                "forward": {
                    "executed_steps": 300,
                    "is_lossless": True,
                    "epsilon_sha256": "fwd_adj_matched_hash_001",
                    "power_ratio": 0.92,
                    "monitors": {"top": {"flux": 0.50}},
                },
                "adjoint": {
                    "executed_steps": 300,
                    "is_lossless": True,
                    "epsilon_sha256": "fwd_adj_matched_hash_001",
                },
                "gradient_verification": {
                    "relative_error": 2.1e-08,
                    "passed": True,
                },
                "m3_fabrication": {
                    "min_tooth_width_nm": 180.0,
                    "min_trench_width_nm": 220.0,
                    "violations": [],
                    "compliant": True,
                },
            },
            {
                "step_index": 1,
                "objective_value": 0.58,
                "forward": {
                    "executed_steps": 300,
                    "is_lossless": True,
                    "epsilon_sha256": "fwd_adj_matched_hash_002",
                    "power_ratio": 0.94,
                    "monitors": {"top": {"flux": 0.58}},
                },
                "adjoint": {
                    "executed_steps": 300,
                    "is_lossless": True,
                    "epsilon_sha256": "fwd_adj_matched_hash_002",
                },
                "gradient_verification": {
                    "relative_error": 3.4e-08,
                    "passed": True,
                },
                "m3_fabrication": {
                    "min_tooth_width_nm": 180.0,
                    "min_trench_width_nm": 220.0,
                    "violations": [],
                    "compliant": True,
                },
            },
        ]
        readout = {
            "schema_version": "1.0",
            "kind": "inverse_design",
            "simulation_executed": True,
            "inverse_design": {
                "device": "canonical_grating_2d",
                "iterations_completed": 2,
                "initial_objective": 0.50,
                "final_objective": 0.58,
                "objective_gain": 0.08,
                "gradient_tolerance": 1e-06,
                "steps": steps,
            },
        }
        telemetry = {
            "schema_version": "1.0",
            "project_canonical_sha256": "plan_hash_canonical_leg49",
            "epsilon_sha256": "fwd_adj_matched_hash_002",
            "is_lossless": True,
            "executed_steps": 600,
            "energy_decay_ratio": 5.0e-8,
            "monitors": {"top": {"flux": 0.58}},
        }
        return td, config, readout, telemetry

    def _execute_post_notary(
        self, td: Path, cfg: dict, rdo: dict, tel: dict
    ) -> tuple[int, dict]:
        """Write files, execute post_notary CLI, and return (returncode, certificate)."""
        (td / "config.input.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (td / "readout.json").write_text(json.dumps(rdo, indent=2), encoding="utf-8")
        (td / "solver_telemetry.json").write_text(json.dumps(tel, indent=2), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "checks" / "post_notary.py"), str(td)],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        cert_path = td / "verification_certificate.json"
        cert = {}
        if cert_path.is_file():
            cert = json.loads(cert_path.read_text(encoding="utf-8"))
        return proc.returncode, cert

    def test_attack_gradient_relative_error_spoofed_or_exceeded(self) -> None:
        """Attack: Fabricate gradient relative error to 1.05e-6 (> 1e-6 tolerance)."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][0]["gradient_verification"]["relative_error"] = 1.05e-06

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on gradient tolerance breach")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "gradient_verification_failed")

    def test_attack_gradient_relative_error_omitted_or_nan(self) -> None:
        """Attack: Omit gradient verification relative error or set to None/NaN."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][1]["gradient_verification"]["relative_error"] = None

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on missing gradient verification")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "gradient_verification_failed")

    def test_attack_passivity_violation_injection_above_tau_pml(self) -> None:
        """Attack: Inject passivity breach power_ratio = 1.00005 > 1.0 + 1e-5 (tau_pml)."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][1]["forward"]["power_ratio"] = 1.00005

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on passivity breach above tau_pml")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "passivity_violation")

    def test_attack_non_lossless_telemetry_spoofing(self) -> None:
        """Attack: Spoof non-lossless simulation in kernel telemetry for lossless grating."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        tel["is_lossless"] = False

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 when telemetry reports non-lossless grating")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "passivity_violation")

    def test_attack_m3_sub_150nm_near_miss_149nm(self) -> None:
        """Attack: Sneak sub-150 nm feature width (149.0 nm, near-miss by 1 nm) into M3 report."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][0]["m3_fabrication"]["min_tooth_width_nm"] = 149.0

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on 149 nm sub-threshold tooth width")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "fabrication_violation_m3")

    def test_attack_m3_conflicting_report_with_violations_list(self) -> None:
        """Attack: Claim 200 nm widths but report non-empty violations list."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][0]["m3_fabrication"]["min_tooth_width_nm"] = 200.0
        rdo["inverse_design"]["steps"][0]["m3_fabrication"]["violations"] = ["min_gap"]

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 when violations list is non-empty")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "fabrication_violation_m3")

    def test_attack_non_monotonic_objective_regression(self) -> None:
        """Attack: Inject objective regression (final objective 0.45 < initial 0.50)."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["final_objective"] = 0.45
        rdo["inverse_design"]["objective_gain"] = -0.05
        rdo["inverse_design"]["steps"][1]["objective_value"] = 0.45

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on regressed final objective")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "non_monotonic_objective")

    def test_attack_flat_objective_zero_gain(self) -> None:
        """Attack: Inject flat objective across iterations (gain = 0.0, no improvement)."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["final_objective"] = 0.50
        rdo["inverse_design"]["objective_gain"] = 0.0
        rdo["inverse_design"]["steps"][1]["objective_value"] = 0.50

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on zero objective gain")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "non_monotonic_objective")

    def test_attack_mismatched_forward_adjoint_epsilon_digests(self) -> None:
        """Attack: Forge adjoint step on a different permittivity tensor (hash mismatch)."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][0]["forward"]["epsilon_sha256"] = "hash_geometry_alpha"
        rdo["inverse_design"]["steps"][0]["adjoint"]["epsilon_sha256"] = "hash_geometry_beta"

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on mismatched forward/adjoint epsilon hashes")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")

    def test_attack_omitted_or_zero_executed_telemetry_steps(self) -> None:
        """Attack: Zero or omitted executed_steps in solver_telemetry.json."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        tel["executed_steps"] = 0

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on zero telemetry executed_steps")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")

    def test_attack_negative_telemetry_steps(self) -> None:
        """Attack: Negative executed_steps in solver_telemetry.json."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        tel["executed_steps"] = -500

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on negative telemetry executed_steps")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")

    def test_attack_zero_forward_or_adjoint_step_execution(self) -> None:
        """Attack: Omit or set executed_steps to 0 in forward or adjoint step record."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"][0]["forward"]["executed_steps"] = 0

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on zero forward executed steps")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")

    def test_attack_empty_steps_list_in_readout(self) -> None:
        """Attack: Readout reports empty steps list []."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        rdo["inverse_design"]["steps"] = []

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on empty steps list")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")

    def test_attack_missing_inverse_design_section(self) -> None:
        """Attack: Readout omits inverse_design dictionary entirely."""
        td, cfg, rdo, tel = self._create_base_valid_run()
        del rdo["inverse_design"]

        rc, cert = self._execute_post_notary(td, cfg, rdo, tel)
        self.assertEqual(rc, 66, "Must exit 66 on missing inverse_design section")
        self.assertFalse(cert.get("verified", True))
        self.assertEqual(cert.get("status"), "tamper_detected")


if __name__ == "__main__":
    unittest.main()
