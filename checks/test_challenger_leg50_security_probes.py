"""Empirical Adversarial Security & Input Sanitization Challenger Suite for Wave D — LEG 50.

Author: challenger_leg50_5
Commit: 532ef848842b4a89c2774d8f7197eeabe2070d7c

Probes:
1. Path traversal attacks on run_id and project_id across all tools:
   - URL-encoded traversal (%2e%2e, %2e%2e%2f, %252e%252e)
   - Null-byte injections (foo\x00bar, \x00)
   - Dots, newlines, and absolute paths
   - Confirms zero filesystem write escape outside configs/projects
2. Manifest injection and malformed declarative scene inputs:
   - Non-float coordinates and dimensions
   - Missing required fields, negative wavelength/mesh/size
   - Prototype / dictionary injection fields
3. Anti-tamper defenses on certificates, journals, and manifests:
   - Manifest exit_code overrides verified state
   - Corrupted and deleted certificate files
   - Active lock contention and completed run sealing
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("PY", sys.executable)

from mcp_server import declarative, platform, paths


class TestAdvancedPathTraversalProbes(unittest.TestCase):
    """Stress-test path traversal across create_project, get_certificate, inspect_breakpoint, resume_run."""

    TRAVERSAL_PAYLOADS = [
        "../../etc/passwd",
        "../../../../../../../../etc/shadow",
        "..\\..\\Windows\\System32\\calc.exe",
        "foo/bar/baz",
        "/tmp/evil",
        "C:\\test",
        "%2e%2e%2f",
        "%2e%2e",
        "..%2f",
        "..",
        ".",
        "   ",
        "\t\n",
        "\x00",
        "foo\x00bar",
        "foo\nbar",
        "....//....//",
        "runs/../../etc/passwd",
        "C:/boot.ini",
        "\\\\?\\C:\\Windows",
        "//wsl$/Ubuntu/etc/shadow",
    ]

    def setUp(self):
        self.temp_root = Path(tempfile.mkdtemp(prefix="challenger50_traversal_"))
        (self.temp_root / "configs" / "projects").mkdir(parents=True, exist_ok=True)
        (self.temp_root / "runs").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_run_id_traversal_refusal_across_all_apis(self):
        """get_certificate, inspect_breakpoint, and resume_run must reject traversal run_ids."""
        for payload in self.TRAVERSAL_PAYLOADS:
            # get_certificate
            try:
                res_cert = platform.get_certificate(payload, root=self.temp_root)
                self.assertFalse(res_cert["ok"], f"get_certificate accepted traversal {repr(payload)}: {res_cert}")
                self.assertIn(
                    res_cert.get("kind"),
                    ("invalid_run_id", "bad_request", "run_not_found", "certificate_missing"),
                    f"Unexpected kind for {repr(payload)}: {res_cert}",
                )
            except ValueError as ve:
                # Embedded null byte raises ValueError in Python's os.path.realpath
                self.assertIn("null byte", str(ve).lower(), f"Unexpected ValueError: {ve}")

            # inspect_breakpoint
            try:
                res_bp = platform.inspect_breakpoint(payload, root=self.temp_root)
                self.assertFalse(res_bp["ok"], f"inspect_breakpoint accepted traversal {repr(payload)}: {res_bp}")
                self.assertIn(
                    res_bp.get("kind"),
                    ("invalid_run_id", "bad_request", "run_not_found", "no_checkpoint"),
                    f"Unexpected kind for {repr(payload)}: {res_bp}",
                )
            except ValueError as ve:
                self.assertIn("null byte", str(ve).lower(), f"Unexpected ValueError: {ve}")

            # resume_run
            try:
                res_res = platform.resume_run(payload, dry_run=True, root=self.temp_root)
                self.assertFalse(res_res["ok"], f"resume_run accepted traversal {repr(payload)}: {res_res}")
                self.assertIn(
                    res_res.get("kind"),
                    ("invalid_run_id", "bad_request", "run_not_found", "no_checkpoint"),
                    f"Unexpected kind for {repr(payload)}: {res_res}",
                )
            except ValueError as ve:
                self.assertIn("null byte", str(ve).lower(), f"Unexpected ValueError: {ve}")

    def test_create_project_zero_filesystem_escape(self):
        """create_project must NEVER write files outside configs/projects/ under any traversal payload."""
        projects_dir = self.temp_root / "configs" / "projects"
        for payload in self.TRAVERSAL_PAYLOADS:
            try:
                res = declarative.create_project(
                    {"project_id": payload, "size": [2.0, 1.0], "mesh": 0.02},
                    root=self.temp_root,
                )
            except Exception as e:
                # Catching exception cleanly is acceptable
                res = {"ok": False, "error": str(e)}

            # Assert no files written anywhere outside configs/projects
            escaped = [
                f for f in self.temp_root.rglob("*")
                if f.is_file() and not str(f.resolve()).startswith(str(projects_dir.resolve()))
            ]
            self.assertEqual(
                escaped,
                [],
                f"VULNERABILITY: create_project wrote files outside configs/projects/ for payload {repr(payload)}: {escaped}",
            )

            # Traversal patterns containing slashes, backslashes, dots, or control characters must be rejected
            if any(c in payload for c in ("/", "\\", "..")) or not payload.strip() or "\x00" in payload:
                self.assertFalse(
                    res.get("ok"),
                    f"VULNERABILITY: create_project returned ok: True for malicious project_id {repr(payload)}: {res}",
                )


class TestMalformedSceneAndManifestInjectionProbes(unittest.TestCase):
    """Stress-test declarative parser against malformed scenes, type confusion, and manifest injections."""

    def test_non_float_coordinates_refusal(self):
        """validate_project and create_project must handle non-float coordinates without crash."""
        malformed_specs = [
            {"name": "bad_coord_1", "size": ["invalid", "string"]},
            {"name": "bad_coord_2", "size": [None, 1.0]},
            {"name": "bad_coord_3", "size": [1.0]},  # only 1 dim
            {"name": "bad_coord_4", "size": []},
            {"name": "bad_mesh_1", "mesh": "not_a_float"},
            {"name": "bad_mesh_2", "mesh": None},
            {"name": "bad_wl_1", "wavelength": "not_a_wavelength"},
            {"name": "bad_wl_2", "wavelength": None},
            {"name": "bad_steps_1", "steps": "five_hundred"},
            {"name": "bad_steps_2", "steps": -10},
        ]
        for spec in malformed_specs:
            res = declarative.validate_project(project=spec)
            self.assertFalse(res["ok"], f"Expected refusal for {spec}, got {res}")
            self.assertEqual(res.get("kind"), "validation_failed")

    def test_manifest_injection_extra_fields(self):
        """create_project must safely serialize valid fields without executing injected attributes."""
        temp_root = Path(tempfile.mkdtemp(prefix="challenger50_inj_"))
        try:
            injection_spec = {
                "project_id": "clean_id_with_injection",
                "size": [2.0, 1.0],
                "mesh": 0.02,
                "__proto__": {"injected": "malicious"},
                "cmd": "rm -rf /",
                "admin": True,
            }
            res = declarative.create_project(injection_spec, root=temp_root)
            self.assertTrue(res["ok"], res)
            cfg_file = temp_root / "configs" / "projects" / "clean_id_with_injection.json"
            self.assertTrue(cfg_file.is_file())
            content = json.loads(cfg_file.read_text(encoding="utf-8"))
            # Injected properties must not corrupt schema
            self.assertEqual(content.get("policy"), "declarative-scene-v2")
            self.assertIn("project", content)
            self.assertEqual(content["project"]["name"], "clean_id_with_injection")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


class TestTamperDefenseProbes(unittest.TestCase):
    """Stress-test anti-tamper and integrity verification."""

    def setUp(self):
        self.temp_root = Path(tempfile.mkdtemp(prefix="challenger50_tamper_"))
        self.run_id = "run_integrity_probe"
        self.run_dir = self.temp_root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.telemetry = {
            "schema_version": "1.0",
            "executed_steps": 100,
            "energy_decay_ratio": 2e-8,
            "is_lossless": True,
        }
        self.readout = {"steps": 100, "flux": 0.9999, "T": 0.6, "R": 0.4}
        self.tel_bytes = json.dumps(self.telemetry).encode("utf-8")
        self.rdo_bytes = json.dumps(self.readout).encode("utf-8")

        (self.run_dir / "solver_telemetry.json").write_bytes(self.tel_bytes)
        (self.run_dir / "readout.json").write_bytes(self.rdo_bytes)
        (self.run_dir / "manifest.json").write_text(
            json.dumps({"run_id": self.run_id, "exit_code": 0, "status": "completed"})
        )

        self.cert = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "verified": True,
            "telemetry_sha256": hashlib.sha256(self.tel_bytes).hexdigest(),
            "readout_sha256": hashlib.sha256(self.rdo_bytes).hexdigest(),
            "status": "completed",
        }
        (self.run_dir / "verification_certificate.json").write_text(json.dumps(self.cert))

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_manifest_exit_code_failure_overrides_verified(self):
        """Even if certificate claims verified: True, manifest exit_code != 0 must force verified: False."""
        for fail_code in (1, 65, 66, 137):
            (self.run_dir / "manifest.json").write_text(
                json.dumps({"run_id": self.run_id, "exit_code": fail_code, "status": "failed"})
            )
            res = platform.get_certificate(self.run_id, root=self.temp_root)
            self.assertTrue(res["ok"])
            self.assertFalse(
                res["verified"],
                f"VULNERABILITY: get_certificate returned verified: True when manifest exit_code was {fail_code}",
            )

    def test_corrupted_certificate_json_refusal(self):
        """Corrupted verification_certificate.json must return ok: False, kind: certificate_corrupted."""
        (self.run_dir / "verification_certificate.json").write_text("{corrupt: json truncated")
        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertFalse(res["ok"])
        self.assertEqual(res.get("kind"), "certificate_corrupted")

    def test_missing_certificate_json_refusal(self):
        """Missing verification_certificate.json must return ok: False, kind: certificate_missing."""
        (self.run_dir / "verification_certificate.json").unlink()
        res = platform.get_certificate(self.run_id, root=self.temp_root)
        self.assertFalse(res["ok"])
        self.assertEqual(res.get("kind"), "certificate_missing")

    def test_active_lock_in_journal_blocks_resume(self):
        """Active process holding owner.json must prevent resume_run."""
        journal_dir = self.run_dir / "journal"
        journal_dir.mkdir()
        (journal_dir / "contract.json").write_text(json.dumps({"steps": 100, "dt": 1e-16}))
        (journal_dir / "latest-forward.json").write_text(json.dumps({"step": 50, "directory": "slot_a"}))
        (journal_dir / "owner.json").write_text(json.dumps({"pid": os.getpid(), "run_id": self.run_id}))

        bp = platform.inspect_breakpoint(self.run_id, root=self.temp_root)
        self.assertTrue(bp["ok"])
        self.assertTrue(bp["is_locked"])
        self.assertFalse(bp["can_resume"])

        res = platform.resume_run(self.run_id, dry_run=True, root=self.temp_root)
        self.assertFalse(res["ok"])
        self.assertEqual(res.get("kind"), "unresumable_run")

    def test_complete_json_in_journal_blocks_resume(self):
        """Sealed journal containing complete.json must prevent resume_run."""
        journal_dir = self.run_dir / "journal"
        journal_dir.mkdir()
        (journal_dir / "contract.json").write_text(json.dumps({"steps": 100, "dt": 1e-16}))
        (journal_dir / "latest-forward.json").write_text(json.dumps({"step": 100, "directory": "slot_a"}))
        (journal_dir / "complete.json").write_text(json.dumps({"status": "completed"}))

        bp = platform.inspect_breakpoint(self.run_id, root=self.temp_root)
        self.assertTrue(bp["ok"])
        self.assertTrue(bp["is_completed"])
        self.assertFalse(bp["can_resume"])

        res = platform.resume_run(self.run_id, dry_run=True, root=self.temp_root)
        self.assertFalse(res["ok"])
        self.assertEqual(res.get("kind"), "unresumable_run")


if __name__ == "__main__":
    unittest.main()
