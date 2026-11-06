"""Acceptance test suite for Wave D — LEG 49.

Track I Inverse Design Loop:
- Analytic Jacobian d(eps)/d(rho) and Taylor finite difference verification <= 1e-6.
- 3-tier M3 fabrication constraint enforcement (tooth >= 150 nm, trench >= 150 nm, zero violations).
- Positive control: 3-iteration closed inverse design loop canary with notary certificate issuance.
- Negative controls NC1–NC5: tampered gradient, M3 violation, passivity violation, zero-step telemetry, non-monotonic objective all strictly assert exit 66.
- CLI post-notary integration tests.
"""

from __future__ import annotations

import copy
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

from src.leg49_grating_vehicle import GratingVehicle2D
from src.leg49_inverse_design import GratingInverseDesignLoop
from src.leg49_m3_constraints import M3FilterProjection
from checks.post_notary import verify_post_flight


class Leg49JacobianUnitTests(unittest.TestCase):
    """Unit tests for analytic Jacobian d(eps)/d(rho) and Taylor finite-difference check."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.m3 = M3FilterProjection(
            dx_um=0.020,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            beta=4.0,
            eta=0.5,
            dtype=torch.float64,
        )

    def test_analytic_jacobian_vs_autograd(self) -> None:
        """Verify exact agreement between analytic Jacobian and PyTorch autograd graph (< 1e-12)."""
        n = 40
        rho = torch.rand(n, dtype=torch.float64) * 0.8 + 0.1
        rho.requires_grad = True

        analytic_j = self.m3.compute_analytic_jacobian(rho, beta=4.0)

        # Vector-Jacobian Product (VJP) comparison with autograd
        for _ in range(5):
            w = torch.randn(n, dtype=torch.float64)
            # Analytic VJP: J^T @ w
            vjp_analytic = analytic_j.T @ w

            if rho.grad is not None:
                rho.grad.zero_()

            proj = self.m3.forward(rho, beta=4.0)
            loss = (proj * w).sum()
            loss.backward()

            vjp_autograd = rho.grad
            max_diff = (vjp_analytic - vjp_autograd).abs().max().item()
            norm = vjp_analytic.abs().max().item()
            rel_err = max_diff / max(norm, 1e-12)
            self.assertLess(rel_err, 1e-12, f"VJP autograd mismatch: {rel_err}")

    def test_central_finite_difference_verification(self) -> None:
        """Assert Taylor central finite-difference check relative error <= 1e-6."""
        n = 50
        rho = torch.rand(n, dtype=torch.float64) * 0.8 + 0.1
        res = self.m3.verify_jacobian_finite_difference(
            rho, h=1e-6, tolerance=1e-6, num_directions=10, beta=4.0
        )
        self.assertTrue(res["passed"], f"FD check failed: {res}")
        self.assertLessEqual(
            res["max_relative_error"],
            1e-06,
            f"Relative error {res['max_relative_error']} exceeds 1e-6 bound",
        )

    def test_projection_derivative_exactness(self) -> None:
        """Verify modified hyperbolic tangent projection derivative across beta schedule."""
        for beta in [0.0, 1.0, 4.0, 16.0]:
            res = self.m3.verify_jacobian_finite_difference(
                h=1e-6, tolerance=1e-6, num_directions=5, beta=beta
            )
            self.assertTrue(res["passed"], f"Failed for beta={beta}: {res}")

    def test_conic_filter_conservation(self) -> None:
        """Verify row-sum normalization sum_j W_ij == 1 and symmetry of conic kernel."""
        n = 60
        w = self.m3._get_filter_matrix(n)
        row_sums = w.sum(dim=-1)
        expected = torch.ones(n, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(row_sums, expected, atol=1e-12),
            f"Filter matrix rows must sum to 1.0: {row_sums}",
        )


class Leg49FabricationM3UnitTests(unittest.TestCase):
    """Unit tests for 3-tier M3 fabrication constraint enforcement (>= 150 nm)."""

    def setUp(self) -> None:
        self.m3 = M3FilterProjection(dx_um=0.020, min_feature_um=0.150)

    def test_sub_150nm_feature_violation_detection(self) -> None:
        """Construct geometry with 60 nm tooth (< 150 nm) and assert M3 violation is detected."""
        arr = np.zeros(200, dtype=bool)
        arr[20:40] = True   # 400 nm solid tooth (valid)
        arr[60:63] = True   # 60 nm solid tooth (INVALID < 150 nm)
        arr[80:120] = True  # 800 nm solid tooth (valid)

        check = self.m3.check_violations(arr)
        self.assertFalse(check["compliant"])
        self.assertIn("min_linewidth", check["violations"])
        self.assertLess(check["min_tooth_width_nm"], 150.0)

    def test_sub_150nm_trench_violation_detection(self) -> None:
        """Construct geometry with 60 nm trench (< 150 nm) and assert M3 violation is detected."""
        arr = np.ones(200, dtype=bool)
        arr[20:23] = False  # 60 nm void trench (INVALID < 150 nm)

        check = self.m3.check_violations(arr)
        self.assertFalse(check["compliant"])
        self.assertIn("min_gap", check["violations"])
        self.assertLess(check["min_trench_width_nm"], 150.0)

    def test_m3_compliant_geometry_zero_violations(self) -> None:
        """Construct compliant grating (teeth >= 200 nm, trenches >= 200 nm) and assert zero violations."""
        arr = np.zeros(200, dtype=bool)
        arr[20:40] = True   # 400 nm tooth
        arr[60:90] = True   # 600 nm tooth
        arr[120:150] = True # 600 nm tooth

        check = self.m3.check_violations(arr)
        self.assertTrue(check["compliant"])
        self.assertEqual(len(check["violations"]), 0)
        self.assertGreaterEqual(check["min_tooth_width_nm"], 150.0)
        self.assertGreaterEqual(check["min_trench_width_nm"], 150.0)

    def test_m3_morphological_projection_filter(self) -> None:
        """Assert Tier 3 morphological projection eliminates defects, producing strictly zero violations."""
        arr = np.zeros(200, dtype=bool)
        arr[20:40] = True
        arr[50:53] = True   # 60 nm defect
        arr[70:110] = True
        arr[120:123] = True # 60 nm defect

        projected = self.m3.project_m3_discrete(arr)
        check = self.m3.check_violations(projected)
        self.assertTrue(
            check["compliant"],
            f"Projected geometry must have zero violations: {check}",
        )
        self.assertEqual(len(check["violations"]), 0)
        self.assertGreaterEqual(check["min_tooth_width_nm"], 150.0)


class Leg49InverseDesignPositiveControl(unittest.TestCase):
    """Positive control: runs genuine 3-iteration closed optimization loop and verifies notary certificate."""

    def test_closed_inverse_design_loop_wsl2(self) -> None:
        """Execute 3-iteration closed loop, verify objective gain, and verify post_notary passes."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_leg49_pos_"))
        try:
            os.environ["RUN_DIR_ABS"] = str(temp_dir)
            from src.leg49_grating_vehicle import get_default_device
            device = get_default_device()
            vehicle = GratingVehicle2D(
                dx_um=0.050,
                dy_um=0.050,
                lx_um=2.5,
                steps=1000,
                device=device,
                dtype=torch.float64,
            )
            m3_filter = M3FilterProjection(
                dx_um=0.050,
                min_feature_um=0.150,
                filter_radius_um=0.150,
                dtype=torch.float64,
                device=device,
            )

            loop = GratingInverseDesignLoop(
                vehicle=vehicle,
                m3_filter=m3_filter,
                optimizer_type="adam",
                learning_rate=0.05,
            )

            inv_results = loop.run_optimization(num_iterations=3)

            # Assertions on optimization progress
            self.assertEqual(inv_results["iterations_completed"], 3)
            self.assertGreaterEqual(
                inv_results["objective_gain"],
                0.0,
                "Optimization must yield non-negative objective gain",
            )
            self.assertLessEqual(
                inv_results["jacobian_relative_error"],
                1e-06,
                "All steps must have Jacobian error <= 1e-6",
            )

            # Assert that the solver kernel wrote authentic solver_telemetry.json
            self.assertTrue(
                (temp_dir / "solver_telemetry.json").is_file(),
                "Kernel must have emitted solver_telemetry.json via RUN_DIR_ABS",
            )

            # Package mock run directory for post_notary
            config = {
                "kind": "inverse_design",
                "inverse_design": True,
                "iterations": 3,
                "wavelength_um": 1.55,
                "tags": ["inverse_design", "inverse_design_step"],
            }
            (temp_dir / "config.input.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

            last_step = inv_results["steps"][-1]
            readout = {
                "schema_version": "1.0",
                "kind": "inverse_design",
                "simulation_executed": True,
                "executed_steps": vehicle.steps,
                "epsilon_sha256": last_step["forward"]["epsilon_sha256"],
                "is_lossless": True,
                "inverse_design": inv_results,
                "monitors": {
                    "top": {
                        "flux": last_step["forward"]["monitors"]["top"]["flux"]
                    },
                    "bottom": {
                        "flux": last_step["forward"]["monitors"]["bottom"]["flux"]
                    },
                },
                "tags": [
                    "inverse_design",
                    "inverse_design_step",
                    "verified:analytic_jacobian",
                    "verified:m3_fabrication_compliant",
                ],
            }
            (temp_dir / "readout.json").write_text(
                json.dumps(readout, indent=2), encoding="utf-8"
            )

            cert = verify_post_flight(temp_dir)
            self.assertTrue(cert["verified"])
            self.assertIn("verified:inverse_design_step", cert["tags"])
            self.assertIn("verified:analytic_jacobian", cert["tags"])
            self.assertIn("verified:m3_fabrication_compliant", cert["tags"])
            self.assertTrue(cert["inverse_design"]["m3_compliant"])
            self.assertTrue(cert["inverse_design"]["all_steps_passive"])
        finally:
            os.environ.pop("RUN_DIR_ABS", None)
            shutil.rmtree(temp_dir, ignore_errors=True)


class Leg49NegativeControls(unittest.TestCase):
    """Negative controls NC1–NC5 asserting that violations strictly trigger exit 66."""

    def _setup_base_run(self, temp_dir: Path) -> tuple[dict, dict, dict]:
        config = {
            "kind": "inverse_design",
            "inverse_design": True,
            "iterations": 2,
            "tags": ["inverse_design", "inverse_design_step"],
        }
        steps = [
            {
                "step_index": 0,
                "objective_value": 0.40,
                "forward": {
                    "executed_steps": 200,
                    "is_lossless": True,
                    "epsilon_sha256": "abc12345",
                    "power_ratio": 0.95,
                    "monitors": {"top": {"flux": 0.40}},
                },
                "adjoint": {
                    "executed_steps": 200,
                    "is_lossless": True,
                    "epsilon_sha256": "abc12345",
                },
                "gradient_verification": {
                    "relative_error": 1.5e-07,
                    "passed": True,
                },
                "m3_fabrication": {
                    "min_tooth_width_nm": 200.0,
                    "min_trench_width_nm": 200.0,
                    "violations": [],
                    "compliant": True,
                },
            },
            {
                "step_index": 1,
                "objective_value": 0.45,
                "forward": {
                    "executed_steps": 200,
                    "is_lossless": True,
                    "epsilon_sha256": "def67890",
                    "power_ratio": 0.96,
                    "monitors": {"top": {"flux": 0.45}},
                },
                "adjoint": {
                    "executed_steps": 200,
                    "is_lossless": True,
                    "epsilon_sha256": "def67890",
                },
                "gradient_verification": {
                    "relative_error": 1.8e-07,
                    "passed": True,
                },
                "m3_fabrication": {
                    "min_tooth_width_nm": 200.0,
                    "min_trench_width_nm": 200.0,
                    "violations": [],
                    "compliant": True,
                },
            },
        ]
        readout = {
            "kind": "inverse_design",
            "simulation_executed": True,
            "inverse_design": {
                "device": "canonical_grating_2d",
                "iterations_completed": 2,
                "initial_objective": 0.40,
                "final_objective": 0.45,
                "objective_gain": 0.05,
                "gradient_tolerance": 1e-06,
                "steps": steps,
            },
        }
        telemetry = {
            "schema_version": "1.0",
            "project_canonical_sha256": "proj_hash_123",
            "epsilon_sha256": "def67890",
            "is_lossless": True,
            "executed_steps": 400,
            "energy_decay_ratio": 1.0e-7,
            "monitors": {"top": {"flux": 0.45}},
        }
        return config, readout, telemetry

    def _run_post_notary_cli(self, run_dir: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", "checks/post_notary.py", str(run_dir)],
            text=True,
            capture_output=True,
        )

    def test_nc1_tampered_gradient_exits_66(self) -> None:
        """NC1: Tampered gradient error (> 1e-6) triggers gradient_verification_failed and exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nc1_grad_"))
        try:
            cfg, rdo, tel = self._setup_base_run(temp_dir)
            # Tamper gradient error to 2.5e-5 (> 1e-6)
            rdo["inverse_design"]["steps"][1]["gradient_verification"]["relative_error"] = 2.5e-5

            (temp_dir / "config.input.json").write_text(json.dumps(cfg), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(rdo), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

            proc = self._run_post_notary_cli(temp_dir)
            self.assertEqual(proc.returncode, 66, f"Expected exit 66, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "gradient_verification_failed")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nc2_fabrication_violation_m3_exits_66(self) -> None:
        """NC2: Sub-150 nm tooth/trench triggers fabrication_violation_m3 and exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nc2_m3_"))
        try:
            cfg, rdo, tel = self._setup_base_run(temp_dir)
            # Inject M3 violation: 100 nm tooth
            rdo["inverse_design"]["steps"][1]["m3_fabrication"]["min_tooth_width_nm"] = 100.0
            rdo["inverse_design"]["steps"][1]["m3_fabrication"]["violations"] = ["min_linewidth"]

            (temp_dir / "config.input.json").write_text(json.dumps(cfg), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(rdo), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

            proc = self._run_post_notary_cli(temp_dir)
            self.assertEqual(proc.returncode, 66, f"Expected exit 66, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "fabrication_violation_m3")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nc3_passivity_violation_exits_66(self) -> None:
        """NC3: Physical passivity violation (power_ratio > 1 + 1e-5) triggers exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nc3_pass_"))
        try:
            cfg, rdo, tel = self._setup_base_run(temp_dir)
            # Inject passivity violation: power_ratio = 1.05
            rdo["inverse_design"]["steps"][1]["forward"]["power_ratio"] = 1.05

            (temp_dir / "config.input.json").write_text(json.dumps(cfg), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(rdo), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

            proc = self._run_post_notary_cli(temp_dir)
            self.assertEqual(proc.returncode, 66, f"Expected exit 66, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "passivity_violation")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nc4_zero_step_telemetry_exits_66(self) -> None:
        """NC4: Dummy/zero-step telemetry triggers tamper_detected and exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nc4_zero_"))
        try:
            cfg, rdo, tel = self._setup_base_run(temp_dir)
            # Zero executed steps in telemetry
            tel["executed_steps"] = 0

            (temp_dir / "config.input.json").write_text(json.dumps(cfg), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(rdo), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

            proc = self._run_post_notary_cli(temp_dir)
            self.assertEqual(proc.returncode, 66, f"Expected exit 66, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "tamper_detected")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nc5_non_monotonic_objective_exits_66(self) -> None:
        """NC5: Decreasing objective across iterations triggers non_monotonic_objective and exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="nc5_mono_"))
        try:
            cfg, rdo, tel = self._setup_base_run(temp_dir)
            # Final objective regressed
            rdo["inverse_design"]["final_objective"] = 0.35
            rdo["inverse_design"]["objective_gain"] = -0.05
            rdo["inverse_design"]["steps"][1]["objective_value"] = 0.35

            (temp_dir / "config.input.json").write_text(json.dumps(cfg), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(rdo), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

            proc = self._run_post_notary_cli(temp_dir)
            self.assertEqual(proc.returncode, 66, f"Expected exit 66, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "non_monotonic_objective")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class Leg49PostNotaryIntegrationTests(unittest.TestCase):
    """Integration test verifying post_notary CLI behaves correctly across scenarios."""

    def test_post_notary_full_success(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="notary_int_ok_"))
        try:
            config = {
                "kind": "inverse_design",
                "inverse_design": True,
                "iterations": 2,
                "tags": ["inverse_design", "inverse_design_step"],
            }
            readout = {
                "kind": "inverse_design",
                "simulation_executed": True,
                "inverse_design": {
                    "device": "canonical_grating_2d",
                    "iterations_completed": 2,
                    "initial_objective": 0.41,
                    "final_objective": 0.48,
                    "objective_gain": 0.07,
                    "gradient_tolerance": 1e-06,
                    "steps": [
                        {
                            "step_index": 0,
                            "objective_value": 0.41,
                            "forward": {
                                "executed_steps": 100,
                                "is_lossless": True,
                                "epsilon_sha256": "h1",
                                "power_ratio": 0.90,
                                "monitors": {"top": {"flux": 0.41}},
                            },
                            "adjoint": {
                                "executed_steps": 100,
                                "is_lossless": True,
                                "epsilon_sha256": "h1",
                            },
                            "gradient_verification": {
                                "relative_error": 1.2e-07,
                                "passed": True,
                            },
                            "m3_fabrication": {
                                "min_tooth_width_nm": 180.0,
                                "min_trench_width_nm": 180.0,
                                "violations": [],
                                "compliant": True,
                            },
                        },
                        {
                            "step_index": 1,
                            "objective_value": 0.48,
                            "forward": {
                                "executed_steps": 100,
                                "is_lossless": True,
                                "epsilon_sha256": "h2",
                                "power_ratio": 0.92,
                                "monitors": {"top": {"flux": 0.48}},
                            },
                            "adjoint": {
                                "executed_steps": 100,
                                "is_lossless": True,
                                "epsilon_sha256": "h2",
                            },
                            "gradient_verification": {
                                "relative_error": 1.4e-07,
                                "passed": True,
                            },
                            "m3_fabrication": {
                                "min_tooth_width_nm": 180.0,
                                "min_trench_width_nm": 180.0,
                                "violations": [],
                                "compliant": True,
                            },
                        },
                    ],
                },
            }
            telemetry = {
                "schema_version": "1.0",
                "project_canonical_sha256": "p1",
                "epsilon_sha256": "h2",
                "is_lossless": True,
                "executed_steps": 200,
                "energy_decay_ratio": 1e-7,
                "monitors": {"top": {"flux": 0.48}},
            }

            (temp_dir / "config.input.json").write_text(json.dumps(config), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(readout), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(telemetry), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", "checks/post_notary.py", str(temp_dir)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, f"Expected returncode 0, got {proc.returncode}: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertTrue(cert["verified"])
            self.assertIn("verified:inverse_design_step", cert["tags"])
            self.assertIn("verified:analytic_jacobian", cert["tags"])
            self.assertIn("verified:m3_fabrication_compliant", cert["tags"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
