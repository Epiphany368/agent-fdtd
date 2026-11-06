"""Empirical Adversarial Security & Robustness Challenge Test Suite for Wave D — LEG 50.

Author: challenger_leg50_1
Target commit: acce961 (feat(leg50): MCP Server v2 declarative agent tools, resume and notary queries)

Challenge Dimensions:
1. Attack 1: Path traversal attacks on run_id in get_certificate, inspect_breakpoint, resume_run
             ("../../etc/passwd", "../runs/foo", "/etc", "C:\\Windows").
             Must return structured refusal 'invalid_run_id' with ok: False.
2. Attack 2: Path traversal and illegal characters in create_project
             (project_id = "../../malicious", "", None). Must refuse cleanly.
3. Attack 3: Malformed declarative scene geometry: negative refractive indices,
             zero/negative domain dimensions, negative wavelength, non-PML unclosed boundaries,
             invalid material models. Verify validate_project catches without unhandled exceptions.
4. Attack 4: Enormous cell count / memory exhaustion attack: proposing 1000x1000x1000 cells (10^9).
             Verify validate_project refuses or warns on exceeding platform limits.
5. Attack 5: Tamper defense attack: mutate a single float in solver_telemetry.json or readout.json,
             verify get_certificate returns tamper_detected: True, verified: False.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from mcp_server import declarative, platform, paths


class TestAttack1PathTraversalOnRunId(unittest.TestCase):
    """Attack 1: Path traversal attacks on run_id across all query and resume APIs.

    Requirement: get_certificate, inspect_breakpoint, resume_run must reject any
    traversal patterns ('../../etc/passwd', '../runs/foo', '/etc', 'C:\\Windows')
    with structured refusal kind: 'invalid_run_id' and ok: False.
    """

    ATTACK_PAYLOADS = [
        "../../etc/passwd",
        "../runs/foo",
        "/etc",
        "/etc/passwd",
        "C:\\Windows",
        "..\\runs\\bar",
        "nested/../../secret",
    ]

    def test_get_certificate_path_traversal_refusal(self):
        """get_certificate must reject all traversal attacks with invalid_run_id."""
        for payload in self.ATTACK_PAYLOADS:
            res = platform.get_certificate(payload)
            self.assertFalse(res["ok"], f"Expected ok: False for {payload}, got {res}")
            self.assertEqual(
                res.get("kind"),
                "invalid_run_id",
                f"Expected kind 'invalid_run_id' for {payload}, got {res.get('kind')}",
            )

    def test_inspect_breakpoint_path_traversal_refusal(self):
        """inspect_breakpoint must reject all traversal attacks with invalid_run_id."""
        for payload in self.ATTACK_PAYLOADS:
            res = platform.inspect_breakpoint(payload)
            self.assertFalse(res["ok"], f"Expected ok: False for {payload}, got {res}")
            self.assertEqual(
                res.get("kind"),
                "invalid_run_id",
                f"VULNERABILITY: inspect_breakpoint accepted traversal or returned non-invalid_run_id kind "
                f"'{res.get('kind')}' for payload '{payload}': {res}",
            )

    def test_resume_run_path_traversal_refusal(self):
        """resume_run must reject all traversal attacks with invalid_run_id."""
        for payload in self.ATTACK_PAYLOADS:
            res = platform.resume_run(payload, dry_run=True)
            self.assertFalse(res["ok"], f"Expected ok: False for {payload}, got {res}")
            self.assertEqual(
                res.get("kind"),
                "invalid_run_id",
                f"VULNERABILITY: resume_run accepted traversal or returned non-invalid_run_id kind "
                f"'{res.get('kind')}' for payload '{payload}': {res}",
            )


class TestAttack2PathTraversalAndIllegalProjectId(unittest.TestCase):
    """Attack 2: Path traversal and illegal characters in create_project.

    Requirement: create_project must refuse cleanly when given:
    - project_id = '../../malicious' (path traversal writing outside configs/projects/)
    - project_id = '' or whitespace (empty name)
    - project_id = None
    """

    def setUp(self):
        self.temp_root = Path(tempfile.mkdtemp(prefix="fdtd_challenger_attack2_"))
        (self.temp_root / "configs" / "projects").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_create_project_path_traversal_refusal(self):
        """create_project must refuse path traversal in project_id or name."""
        traversal_ids = [
            "../../malicious",
            "../escaped_project",
            "/absolute/root_project",
            "C:\\Windows\\system32_proj",
            "sub/dir/project",
        ]
        for bad_id in traversal_ids:
            res = declarative.create_project(
                {"project_id": bad_id, "size": [2.0, 1.0], "mesh": 0.02},
                root=self.temp_root,
            )
            # Must NOT succeed and must NOT write files outside configs/projects
            projects_dir = self.temp_root / "configs" / "projects"
            escaped_files = [
                f for f in self.temp_root.rglob("*.json")
                if not str(f.resolve()).startswith(str(projects_dir.resolve()))
            ]
            self.assertEqual(
                escaped_files,
                [],
                f"VULNERABILITY: create_project wrote files outside configs/projects/: {escaped_files}",
            )
            self.assertFalse(
                res.get("ok"),
                f"VULNERABILITY: create_project returned ok: True for traversal project_id '{bad_id}': {res}",
            )

    def test_create_project_empty_or_none_refusal(self):
        """create_project must refuse empty or None project_id when explicitly provided."""
        for empty_val in ("", "   ", None):
            res = declarative.create_project(
                {"project_id": empty_val, "size": [2.0, 1.0], "mesh": 0.02},
                root=self.temp_root,
            )
            self.assertFalse(
                res.get("ok"),
                f"Expected create_project to refuse empty/None project_id '{empty_val}', got: {res}",
            )

    def test_create_project_kwarg_signature_consistency(self):
        """platform.create_project must accept both project_spec and project_dict kwargs."""
        spec1 = {"project_id": "kwarg_spec_test", "size": [2.0, 1.0], "mesh": 0.02}
        res1 = platform.create_project(project_spec=spec1, root=self.temp_root)
        self.assertTrue(res1.get("ok"), f"project_spec kwarg failed: {res1}")
        self.assertTrue((self.temp_root / "configs" / "projects" / "kwarg_spec_test.json").is_file())

        spec2 = {"project_id": "kwarg_dict_test", "size": [2.0, 1.0], "mesh": 0.02}
        res2 = platform.create_project(project_dict=spec2, root=self.temp_root)
        self.assertTrue(res2.get("ok"), f"project_dict kwarg failed: {res2}")
        self.assertTrue((self.temp_root / "configs" / "projects" / "kwarg_dict_test.json").is_file())


class TestAttack3MalformedDeclarativeSceneGeometry(unittest.TestCase):
    """Attack 3: Malformed declarative scene geometry.

    Verify validate_project catches and returns structured errors without unhandled exceptions for:
    - Negative refractive indices (index = -1.5)
    - Zero / negative domain dimensions (size = [0, -1.0])
    - Negative wavelength (wavelength = -1.55)
    - Non-PML unclosed boundary configurations
    - Invalid material models
    """

    def test_negative_refractive_index_refusal(self):
        """validate_project must reject negative refractive index in dielectric materials."""
        spec = {
            "name": "attack_neg_index",
            "region": {
                "size": [2.0, 1.0, 0.04],
                "mesh": 0.02,
            },
            "materials": [
                {"name": "si", "index": -3.477, "model": "dielectric"},
            ],
        }
        res = declarative.validate_project(project=spec)
        self.assertFalse(res["ok"], "validate_project should reject negative refractive index")
        self.assertIn("validation_failed", res.get("kind", ""))
        self.assertTrue(any("index" in e.lower() or "material" in e.lower() for e in res.get("errors", [])))

    def test_zero_or_negative_dimensions_refusal(self):
        """validate_project must reject zero or negative domain size."""
        for bad_size in ([0.0, 1.0], [-2.0, 1.0], [2.0, -1.0], [0.0, 0.0]):
            res = declarative.validate_project(
                project={"name": "bad_size", "region": {"size": bad_size, "mesh": 0.02}}
            )
            self.assertFalse(res["ok"], f"Expected refusal for size {bad_size}")
            self.assertEqual(res.get("kind"), "validation_failed")

    def test_negative_wavelength_refusal(self):
        """validate_project / parse_and_build_project must reject negative or zero wavelength."""
        for bad_wl in (-1.55, 0.0, -0.001):
            res = declarative.validate_project(
                project={
                    "name": "bad_wl",
                    "size": [2.0, 1.0],
                    "mesh": 0.02,
                    "wavelength": bad_wl,
                    "sources": [{
                        "name": "src",
                        "wavelength": bad_wl,
                    }],
                }
            )
            self.assertFalse(res["ok"], f"Expected refusal for wavelength {bad_wl}")

    def test_unclosed_or_invalid_boundaries_refusal(self):
        """validate_project must detect unclosed Poynting boxes or invalid boundary kinds."""
        # Invalid boundary kind
        res = declarative.validate_project(
            project={
                "name": "bad_bnd",
                "region": {
                    "size": [2.0, 1.0],
                    "mesh": 0.02,
                    "boundaries": {
                        "x_min": {"kind": "unsupported_nonsense"},
                    },
                },
            }
        )
        self.assertFalse(res["ok"], "Expected refusal for invalid boundary kind")

    def test_invalid_material_model_refusal(self):
        """validate_project must reject invalid material models."""
        spec = {
            "name": "bad_model",
            "size": [2.0, 1.0],
            "mesh": 0.02,
            "materials": [
                {"name": "alien_matter", "index": 2.0, "model": "anti_gravity_permittivity"},
            ],
        }
        res = declarative.validate_project(project=spec)
        self.assertFalse(res["ok"], "validate_project should reject invalid material model")


class TestAttack4EnormousCellCountExhaustion(unittest.TestCase):
    """Attack 4: Enormous cell count / memory exhaustion attack.

    Proposing 1000 x 1000 x 1000 cells (10^9 cells).
    Verify validate_project refuses or warns on exceeding platform limits.
    """

    def test_billion_cells_gpu_ceiling_refusal(self):
        """proposing 10^9 cells must be rejected cleanly by validate_project."""
        spec = {
            "name": "dos_billion_cells",
            "region": {
                "dimension": "3d",
                "size": [20.0, 20.0, 20.0],
                "mesh": 0.02,  # 1000 x 1000 x 1000 = 1,000,000,000 cells
                "steps": 100,
            },
        }
        res = declarative.validate_project(project=spec)
        self.assertFalse(res["ok"], "Expected refusal for 10^9 cells")
        self.assertEqual(res.get("kind"), "validation_failed")
        self.assertTrue(
            any("ceiling" in e.lower() or "exceeds" in e.lower() for e in res.get("errors", [])),
            f"Expected error mentioning ceiling or exceeds, got: {res.get('errors')}",
        )


class TestAttack5NotaryTamperDefense(unittest.TestCase):
    """Attack 5: Tamper defense attack on simulation results.

    Take a valid completed run, mutate a single float in solver_telemetry.json or readout.json,
    and call get_certificate(run_id).
    Verify get_certificate returns tamper_detected: True, verified: False.
    """

    def setUp(self):
        self.temp_root = Path(tempfile.mkdtemp(prefix="fdtd_challenger_attack5_"))
        self.run_id = "run_challenger_tamper_target"
        self.run_dir = self.temp_root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.telemetry_data = {
            "schema_version": "1.0",
            "executed_steps": 200,
            "energy_decay_ratio": 1e-7,
            "is_lossless": True,
            "project_canonical_sha256": "abcdef123456",
        }
        self.readout_data = {
            "steps": 200,
            "flux": 0.99987,
            "T": 0.65432,
            "R": 0.34555,
        }
        self.tel_bytes = json.dumps(self.telemetry_data).encode("utf-8")
        self.rdo_bytes = json.dumps(self.readout_data).encode("utf-8")

        (self.run_dir / "solver_telemetry.json").write_bytes(self.tel_bytes)
        (self.run_dir / "readout.json").write_bytes(self.rdo_bytes)
        (self.run_dir / "manifest.json").write_text(
            json.dumps({"run_id": self.run_id, "exit_code": 0, "status": "completed"})
        )

        self.cert_data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "verified": True,
            "telemetry_sha256": hashlib.sha256(self.tel_bytes).hexdigest(),
            "readout_sha256": hashlib.sha256(self.rdo_bytes).hexdigest(),
            "tags": ["verified:postflight_notary"],
            "status": "completed",
        }
        (self.run_dir / "verification_certificate.json").write_text(json.dumps(self.cert_data))

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_untampered_certificate_is_verified(self):
        """Untampered run must return verified: True, tamper_detected: False."""
        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["verified"], res)
        self.assertFalse(res["tamper_detected"], res)

    def test_tampered_telemetry_float_detected(self):
        """Mutating a single float in solver_telemetry.json must trigger tamper_detected: True, verified: False."""
        tampered_tel = dict(self.telemetry_data)
        tampered_tel["energy_decay_ratio"] = 1.0000001e-7  # Mutate float
        (self.run_dir / "solver_telemetry.json").write_text(json.dumps(tampered_tel))

        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["tamper_detected"], f"Expected tamper_detected: True, got: {res}")
        self.assertFalse(res["verified"], f"Expected verified: False when tampered, got: {res}")
        self.assertTrue(
            any("telemetry" in err.lower() for err in res.get("tamper_errors", [])),
            f"Expected telemetry in tamper_errors, got: {res.get('tamper_errors')}",
        )

    def test_tampered_readout_float_detected(self):
        """Mutating a single float in readout.json must trigger tamper_detected: True, verified: False."""
        tampered_rdo = dict(self.readout_data)
        tampered_rdo["T"] = 0.65433  # Mutate float
        (self.run_dir / "readout.json").write_text(json.dumps(tampered_rdo))

        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["tamper_detected"], f"Expected tamper_detected: True, got: {res}")
        self.assertFalse(res["verified"], f"Expected verified: False when tampered, got: {res}")
        self.assertTrue(
            any("readout" in err.lower() for err in res.get("tamper_errors", [])),
            f"Expected readout in tamper_errors, got: {res.get('tamper_errors')}",
        )

    def test_deleted_telemetry_file_detected(self):
        """Deleting solver_telemetry.json on disk must trigger tamper_detected: True, verified: False."""
        (self.run_dir / "solver_telemetry.json").unlink()
        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["tamper_detected"], f"Expected tamper_detected: True on deletion, got: {res}")
        self.assertFalse(res["verified"], f"Expected verified: False on deletion, got: {res}")
        self.assertTrue(
            any("solver_telemetry.json is missing on disk" in err for err in res.get("tamper_errors", [])),
            f"Expected missing telemetry error, got: {res.get('tamper_errors')}",
        )

    def test_deleted_readout_file_detected(self):
        """Deleting readout.json on disk must trigger tamper_detected: True, verified: False."""
        (self.run_dir / "readout.json").unlink()
        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["tamper_detected"], f"Expected tamper_detected: True on deletion, got: {res}")
        self.assertFalse(res["verified"], f"Expected verified: False on deletion, got: {res}")
        self.assertTrue(
            any("readout.json is missing on disk" in err for err in res.get("tamper_errors", [])),
            f"Expected missing readout error, got: {res.get('tamper_errors')}",
        )


if __name__ == "__main__":
    unittest.main()
