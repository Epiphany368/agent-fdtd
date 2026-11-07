"""LEG 50 Acceptance Test Suite: MCP Server v2 Declarative Scene, Resume & Notary Certificate.

Verifies end-to-end:
1. Tool Surface V2: 10 tools discovered over stdio; explain_exit(66) registered.
2. Certificate Query: Authentic verified run inspection, authentic exit 66 failure inspection.
3. Security & Refusal: Path traversal rejection, non-existent runs, incomplete runs, tampered certificates.
4. Declarative Scene Workflow: create_project -> validate_project -> submit_run(dry_run=True).
5. Full Agent Lifecycle: 建模 -> 准入预检 -> 仿真 -> 续跑 -> 获取 verified 证书.
6. Non-Regression: Existing tools retain backwards compatibility.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest

try:
    from support import ROOT, call, isolated_project, open_session
except ImportError:
    from mcptests.support import ROOT, call, isolated_project, open_session


TOOL_NAMES_V2 = {
    "list_configs",
    "describe_run",
    "validate_config",
    "submit_run",
    "read_ledger",
    "explain_exit",
    "create_project",
    "validate_project",
    "resume_run",
    "inspect_breakpoint",
    "get_certificate",
}


class TestDeclarativeSceneBuilder(unittest.IsolatedAsyncioTestCase):
    """Test Class 1: Scene creation, Poynting box auto-synthesis, and serialization."""

    async def test_all_ten_tools_registered_over_stdio(self):
        async with open_session() as (session, _errlog):
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            self.assertEqual(names, TOOL_NAMES_V2, f"Tool set mismatch: {names ^ TOOL_NAMES_V2}")

    async def test_explain_exit_66_registered(self):
        async with open_session() as (session, _errlog):
            res = await call(session, "explain_exit", {"code": 66})
            self.assertTrue(res["known"])
            self.assertFalse(res["ok"])
            self.assertEqual(res["kind"], "postflight_notary_failed")
            self.assertIn("post_notary.py", res["source"])

    async def test_declarative_scene_creation_and_poynting_box(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                proj = await call(session, "create_project", {
                    "name": "declarative_grating_test",
                    "dimension": "2d",
                    "domain": {"x_um": [-2.0, 2.0], "y_um": [-1.0, 1.0]},
                    "grid": {"dx_um": 0.02, "dy_um": 0.02},
                    "steps": 150,
                    "poynting_box": {"center": [0.0, 0.0, 0.0], "size": [2.5, 1.2, 0.04]},
                })
                self.assertTrue(proj["ok"], proj)
                cfg_path = proj["config_path"]
                self.assertTrue((root / cfg_path).is_file())
                self.assertTrue(proj["project_canonical_sha256"])

                # Verify synthesized project content
                cfg_data = json.loads((root / cfg_path).read_text(encoding="utf-8"))
                mon_names = {m["name"] for m in cfg_data["project"]["monitors"]}
                self.assertTrue({"box-right", "box-left", "box-top", "box-bottom"}.issubset(mon_names))


class TestProjectPreflightValidation(unittest.IsolatedAsyncioTestCase):
    """Test Class 2: Validation logic, invalid schemas, and physics constraints."""

    async def test_validate_project_valid_config(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                proj = await call(session, "create_project", {
                    "name": "valid_val_test",
                    "size": [3.0, 1.5, 0.04],
                    "mesh": 0.02,
                    "steps": 100,
                })
                self.assertTrue(proj["ok"])
                cfg_path = proj["config_path"]

                val = await call(session, "validate_project", {"config": cfg_path})
                self.assertTrue(val["ok"], val)
                self.assertEqual(len(val["grid_shape"]), 3)
                self.assertGreater(val["total_cells"], 0)
                self.assertGreater(val["dt_fs"], 0)
                self.assertEqual(val["total_steps"], 100)
                self.assertTrue(val["poynting_box_closed"])
                self.assertGreater(val["estimated_memory_mb"], 0)

    async def test_validate_project_invalid_mesh(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                # Negative mesh via create_project
                proj = await call(session, "create_project", {
                    "name": "invalid_mesh_test",
                    "grid": {"dx_um": -0.02},
                })
                self.assertFalse(proj["ok"])
                self.assertEqual(proj["kind"], "invalid_scene_specification")

    async def test_validate_project_cfl_and_total_cells(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                # Request domain exceeding resident limit
                val = await call(session, "validate_project", {
                    "project": {
                        "region": {
                            "dimension": "3d",
                            "size": [50.0, 50.0, 10.0],
                            "mesh": 0.01,
                            "steps": 100,
                        }
                    }
                })
                self.assertFalse(val["ok"])
                self.assertEqual(val["kind"], "validation_failed")
                self.assertTrue(any("resident GPU ceiling" in e or "ceiling" in e for e in val["errors"]))


class TestBreakpointResumeTools(unittest.IsolatedAsyncioTestCase):
    """Test Class 3: inspect_breakpoint, resume_run, and submit_run(resume_from=...)."""

    def _setup_mock_checkpoint(self, root: Path, run_id: str = "mock_ckpt_run") -> Path:
        run_dir = root / "runs" / run_id
        journal_dir = run_dir / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)

        contract = {
            "marker": "torchfdtd.resident_restart.v1",
            "steps": 500,
            "dt": 1.5e-16,
            "grid_shape": [100, 50, 1],
            "epsilon_sha256": "abc123mockeps",
            "block_size": 32,
        }
        (journal_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")

        pointer = {
            "directory": "slot_a",
            "step": 250,
            "block": 7,
            "peak_energy": 0.95,
            "previous": None,
            "written": "2026-11-05T12:00:00Z",
        }
        (journal_dir / "latest-forward.json").write_text(json.dumps(pointer), encoding="utf-8")

        for slot_name in ("slot_a", "slot_b"):
            sdir = journal_dir / slot_name
            sdir.mkdir(parents=True, exist_ok=True)
            meta = {
                "step": 250 if slot_name == "slot_a" else 0,
                "arrays": [{"file": "state.bin"}],
            }
            (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (sdir / "state.bin").write_bytes(b"\x00" * 64)

        manifest = {
            "run_id": run_id,
            "config": "configs/exit57.json",
            "exit_code": 0,
            "status": "completed",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    async def test_inspect_breakpoint_returns_checkpoint_state(self):
        with isolated_project(with_solver=True) as root:
            self._setup_mock_checkpoint(root, "test_inspect_run")
            async with open_session(root=root) as (session, _errlog):
                info = await call(session, "inspect_breakpoint", {"run_id": "test_inspect_run"})
                self.assertTrue(info["ok"], info)
                self.assertEqual(info["run_id"], "test_inspect_run")
                self.assertEqual(info["checkpoint"]["step"], 250)
                self.assertEqual(info["checkpoint"]["total_steps"], 500)
                self.assertTrue(info["can_resume"])
                self.assertTrue(info["slot_health"]["slot_a"]["valid"])

    async def test_submit_run_with_resume_dry_run(self):
        with isolated_project(with_solver=True) as root:
            self._setup_mock_checkpoint(root, "test_submit_resume_run")
            async with open_session(root=root) as (session, _errlog):
                res = await call(session, "submit_run", {
                    "config": "configs/exit57.json",
                    "dry_run": True,
                    "resume_from": "test_submit_resume_run",
                })
                self.assertTrue(res["ok"], res)
                self.assertTrue(res["dry_run"])
                self.assertFalse(res["executed"])
                self.assertEqual(res["resume_from"], "test_submit_resume_run")
                self.assertEqual(res["resumed_from_step"], 250)
                self.assertIn("--resume test_submit_resume_run", res["command"])

    async def test_resume_run_dry_run(self):
        with isolated_project(with_solver=True) as root:
            self._setup_mock_checkpoint(root, "test_auto_resume_run")
            async with open_session(root=root) as (session, _errlog):
                res = await call(session, "resume_run", {
                    "run_id": "test_auto_resume_run",
                    "dry_run": True,
                })
                self.assertTrue(res["ok"], res)
                self.assertTrue(res["dry_run"])
                self.assertEqual(res["resumed_from_run_id"], "test_auto_resume_run")
                self.assertEqual(res["resumed_from_step"], 250)
                self.assertIn("--resume test_auto_resume_run", res["command"])


class TestNotaryCertificateQueries(unittest.IsolatedAsyncioTestCase):
    """Test Class 4: get_certificate on authentic completed runs, exit 66 failures, and anti-tamper."""

    async def test_query_authentic_verified_production_run(self):
        run_id = "20261105T115455Z-9250"
        if not (ROOT / "runs" / run_id).is_dir():
            self.skipTest(f"Production run {run_id} not present")

        async with open_session() as (session, _errlog):
            cert = await call(session, "get_certificate", {"run_id": run_id})
            self.assertTrue(cert["ok"], cert)
            self.assertTrue(cert["verified"], cert)
            self.assertEqual(cert["exit_code"], 0)
            self.assertIn("verified:postflight_notary", cert["tags"])
            self.assertIn("verified:passivity", cert["tags"])
            self.assertTrue(cert["metrics"]["passivity_passed"])
            self.assertLessEqual(cert["metrics"]["energy_decay_ratio"], 1e-6)
            self.assertTrue(cert["provenance"]["gpu"])
            self.assertEqual(cert["provenance"]["solver_dirty_lines"], 0)
            self.assertIn("inverse_design", cert["extensions"])

    async def test_query_authentic_exit_66_production_run(self):
        run_id = "20261105T104537Z-3689"
        if not (ROOT / "runs" / run_id).is_dir():
            self.skipTest(f"Production run {run_id} not present")

        async with open_session() as (session, _errlog):
            cert = await call(session, "get_certificate", {"run_id": run_id})
            self.assertTrue(cert["ok"], cert)
            self.assertFalse(cert["verified"], cert)  # Zero-Trust: Not verified!
            self.assertEqual(cert["exit_code"], 66)
            self.assertEqual(cert["failure_cause"], "passivity_violation")
            self.assertIn("passivity violation", cert["error"])

    async def test_detection_tampered_certificate(self):
        with isolated_project(with_solver=True) as root:
            rdir = root / "runs" / "test_tampered"
            rdir.mkdir(parents=True)
            tel = {"schema_version": "1.0", "executed_steps": 100}
            rdo = {"steps": 100}
            tel_bytes = json.dumps(tel).encode("utf-8")
            rdo_bytes = json.dumps(rdo).encode("utf-8")
            (rdir / "solver_telemetry.json").write_bytes(tel_bytes)
            (rdir / "readout.json").write_bytes(rdo_bytes)
            (rdir / "manifest.json").write_text(json.dumps({"exit_code": 0, "status": "completed"}), encoding="utf-8")
            cert_data = {
                "schema_version": "1.0",
                "run_id": "test_tampered",
                "verified": True,
                "telemetry_sha256": hashlib.sha256(tel_bytes).hexdigest(),
                "readout_sha256": hashlib.sha256(rdo_bytes).hexdigest(),
                "tags": ["verified:postflight_notary"],
            }
            (rdir / "verification_certificate.json").write_text(json.dumps(cert_data), encoding="utf-8")

            # Tamper readout.json after certificate creation
            (rdir / "readout.json").write_text(json.dumps({"steps": 9999}), encoding="utf-8")

            async with open_session(root=root) as (session, _errlog):
                cert = await call(session, "get_certificate", {"run_id": "test_tampered"})
                self.assertTrue(cert["ok"])
                self.assertTrue(cert["tamper_detected"])
                self.assertFalse(cert["verified"])  # Stripped of verified status!


class TestStructuredRefusalTaxonomy(unittest.IsolatedAsyncioTestCase):
    """Test Class 5: Missing run_id, traversal attacks, corrupted checkpoints, unexecuted runs."""

    async def test_refusal_non_existent_run(self):
        async with open_session() as (session, _errlog):
            cert = await call(session, "get_certificate", {"run_id": "non_existent_run_999999"})
            self.assertFalse(cert["ok"])
            self.assertEqual(cert["kind"], "run_not_found")

    async def test_refusal_path_traversal_attack(self):
        async with open_session() as (session, _errlog):
            for bad_id in ("../../etc/passwd", "..\\runs", "sub/dir"):
                cert = await call(session, "get_certificate", {"run_id": bad_id})
                self.assertFalse(cert["ok"])
                self.assertEqual(cert["kind"], "invalid_run_id")

    async def test_refusal_incomplete_run_missing_certificate(self):
        with isolated_project(with_solver=True) as root:
            rdir = root / "runs" / "test_incomplete"
            rdir.mkdir(parents=True)
            (rdir / "manifest.json").write_text(
                json.dumps({"run_id": "test_incomplete", "exit_code": 65, "status": "failed:preflight"}),
                encoding="utf-8",
            )
            async with open_session(root=root) as (session, _errlog):
                cert = await call(session, "get_certificate", {"run_id": "test_incomplete"})
                self.assertFalse(cert["ok"])
                self.assertEqual(cert["kind"], "certificate_missing")
                self.assertEqual(cert["exit_code"], 65)

    async def test_refusal_resume_missing_or_corrupt_journal(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                res = await call(session, "resume_run", {"run_id": "ghost_run_12345"})
                self.assertFalse(res["ok"])
                self.assertEqual(res["kind"], "run_not_found")

                res_bp = await call(session, "inspect_breakpoint", {"run_id": "ghost_run_12345"})
                self.assertFalse(res_bp["ok"])
                self.assertEqual(res_bp["kind"], "run_not_found")


class TestEndToEndAgentWorkflow(unittest.IsolatedAsyncioTestCase):
    """Test Class 6: Complete agent closed-loop (建模 -> 准入预检 -> 仿真 -> 续跑 -> 获取 verified 证书)."""

    async def test_full_lifecycle_model_validate_run_resume_certificate(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                # 1. 建模 (compact domain with authentic electromagnetic wave propagation)
                proj = await call(session, "create_project", {
                    "name": "e2e_grating",
                    "task": "E2E full lifecycle simulation",
                    "kind": "simulation",
                    "dimension": "2d",
                    "domain": {"x_um": [-0.5, 0.5], "y_um": [-0.3, 0.3]},
                    "grid": {"dx_um": 0.02, "dy_um": 0.02},
                    "pml_cells": 4,
                    "steps": 200,
                    "max_steps": 100,
                    "tau_box": 1.0,
                    "decay_floor": 0.5,
                    "incident_flux": 1.0648344688079908e-37,
                    "source": {
                        "name": "source",
                        "kind": "plane",
                        "normal": "x",
                        "direction": "+",
                        "component": "Ey",
                        "center": [-0.2, 0.0, 0.0],
                        "size": [0.0, 0.3, 0.02],
                        "wavelength": 1.55,
                        "time_definition": "standard",
                        "pulse_length": 2.0e-15,
                        "pulse_offset": 5.0e-15,
                    },
                    "poynting_box": {
                        "center": [0.0, 0.0, 0.0],
                        "size": [0.7, 0.4, 0.02],
                    },
                })
                self.assertTrue(proj["ok"], proj)
                cfg = proj["config_path"]
                self.assertTrue((root / cfg).is_file())

                # 2. 准入预检
                val = await call(session, "validate_project", {"config": cfg})
                self.assertTrue(val["ok"], val)

                # Dry-run check
                dry = await call(session, "submit_run", {"config": cfg, "dry_run": True})
                self.assertTrue(dry["ok"], dry)
                self.assertTrue(dry["dry_run"])
                self.assertFalse(dry["executed"])

                # 3. 仿真 (submit_run - partial run stopping legitimately at step 100)
                run_res = await call(session, "submit_run", {"config": cfg, "dry_run": False})
                self.assertTrue(run_res["ok"], run_res)
                self.assertEqual(run_res["exit_status"], 0)
                run_id = run_res["run_id"]
                self.assertTrue(run_id)

                # Verify partial breakpoint state
                bp = await call(session, "inspect_breakpoint", {"run_id": run_id})
                self.assertTrue(bp["ok"], bp)
                self.assertTrue(bp["can_resume"], bp)
                self.assertFalse(bp["is_completed"], bp)

                # 4. 续跑 (resume_run - legitimately completes from step 100 to 200)
                resume_res = await call(session, "resume_run", {"run_id": run_id, "dry_run": False})
                self.assertTrue(resume_res["ok"], resume_res)
                self.assertEqual(resume_res["exit_status"], 0)
                new_run_id = resume_res.get("run_id") or resume_res.get("new_run_id")
                self.assertTrue(new_run_id)

                # Verify completed breakpoint state
                bp_after = await call(session, "inspect_breakpoint", {"run_id": new_run_id})
                self.assertTrue(bp_after["ok"], bp_after)
                self.assertFalse(bp_after["can_resume"], bp_after)
                self.assertTrue(bp_after["is_completed"], bp_after)

                # 5. 获取 verified 证书 (get_certificate)
                cert = await call(session, "get_certificate", {"run_id": new_run_id})
                self.assertTrue(cert["ok"], cert)
                self.assertTrue(cert["verified"], cert)
                self.assertEqual(cert["exit_code"], 0)
                self.assertIn("verified:postflight_notary", cert["tags"])

    async def test_legacy_tools_compatibility(self):
        async with open_session() as (session, _errlog):
            configs = await call(session, "list_configs")
            self.assertTrue(configs["ok"])
            ledger = await call(session, "read_ledger", {"limit": 2})
            self.assertTrue(ledger["ok"])
            explain = await call(session, "explain_exit", {"code": 57})
            self.assertEqual(explain["kind"], "finding")


if __name__ == "__main__":
    unittest.main()
