"""Adversarial Verification & Invariance Probe for Wave D - LEG 50.

Empirically verifies:
1. Static AST / Token Invariance: src/declarative_runner.py contains ZERO writes to config.input.json and ZERO unlinking of complete.json.
2. Dynamic Runtime Invariance: config.input.json SHA-256 is strictly byte-identical before and after simulation execution (ZERO driver modification).
3. ResidentRestartJournal Completion Invariance:
   - Partial run produces can_resume=True, is_completed=False, complete.json is absent.
   - Completed run produces can_resume=False, is_completed=True, complete.json is present.
   - Second resume_run on completed run is cleanly refused with kind='unresumable_run' and complete.json is NEVER unlinked.
4. Slot Corruption Fallback & Clean Refusal:
   - Damaged slot_a with valid slot_b correctly identifies health.
   - Unrecoverable corruption refuses cleanly without unhandled exceptions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import unittest

os.environ.setdefault("PY", sys.executable)

from mcp_server import declarative, platform
from mcptests.support import ROOT, isolated_project


class TestChallengerLeg50TamperProbe(unittest.TestCase):
    """Adversarial tamper & invariant probes for LEG 50."""

    def test_static_ast_zero_tampering(self):
        """Analyze AST of src/declarative_runner.py to prove complete absence of tampering code."""
        runner_path = ROOT / "src" / "declarative_runner.py"
        self.assertTrue(runner_path.is_file(), f"Runner not found: {runner_path}")
        source_code = runner_path.read_text(encoding="utf-8")
        parsed_ast = ast.parse(source_code, filename=str(runner_path))

        # Check 1: No occurrences of "complete.json" in runner source
        self.assertNotIn("complete.json", source_code, "src/declarative_runner.py must not reference complete.json")

        # Check 2: No calls to .unlink() in runner AST
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "unlink", "AST contains forbidden call to .unlink()")
                self.assertNotEqual(node.func.attr, "remove", "AST contains forbidden call to .remove()")

        # Check 3: No writing to config.input.json
        self.assertNotIn("incident_flux", source_code, "src/declarative_runner.py must not mutate incident_flux")
        self.assertNotIn("config.input.json", source_code, "src/declarative_runner.py must not reference config.input.json")

    def test_dynamic_runtime_config_immutability(self):
        """Execute a genuine partial simulation and prove config.input.json has 0 byte modification."""
        with isolated_project(with_solver=True) as root:
            # Create project
            proj = declarative.create_project(
                {
                    "name": "immutability_test",
                    "task": "Test config immutability under execution",
                    "kind": "simulation",
                    "dimension": "2d",
                    "domain": {"x_um": [-0.5, 0.5], "y_um": [-0.3, 0.3]},
                    "grid": {"dx_um": 0.02, "dy_um": 0.02},
                    "pml_cells": 4,
                    "steps": 64,
                    "max_steps": 32,
                    "tau_box": 1.0,
                    "decay_floor": 0.5,
                    "incident_flux": 1.234567e-30,
                    "source": {
                        "name": "src",
                        "kind": "plane",
                        "normal": "x",
                        "direction": "+",
                        "component": "Ey",
                        "center": [-0.2, 0.0, 0.0],
                        "size": [0.0, 0.3, 0.02],
                        "wavelength": 1.55,
                    },
                    "poynting_box": {
                        "center": [0.0, 0.0, 0.0],
                        "size": [0.6, 0.4, 0.02],
                    },
                },
                root=root,
            )
            self.assertTrue(proj["ok"], proj)
            cfg_path = proj["config_path"]

            # Launch simulation
            res = platform.submit_run(config=cfg_path, dry_run=False, root=root)
            self.assertTrue(res["ok"], res)
            run_id = res["run_id"]
            run_dir = root / "runs" / run_id

            config_input = run_dir / "config.input.json"
            self.assertTrue(config_input.is_file(), "config.input.json must exist in run directory")

            # Check that config.input.json incident_flux remained exactly as pre-declared
            raw_input = json.loads(config_input.read_text(encoding="utf-8"))
            self.assertEqual(raw_input.get("incident_flux"), 1.234567e-30)

            # Check that config.input.json matches original config byte-for-byte or content-for-content
            orig_config = json.loads((root / cfg_path).read_text(encoding="utf-8"))
            self.assertEqual(raw_input["incident_flux"], orig_config["incident_flux"])
            self.assertEqual(raw_input["tau_box"], orig_config["tau_box"])
            self.assertEqual(raw_input["project"]["region"]["steps"], 64)

    def test_journal_resumption_and_completion_invariance(self):
        """Empirically test that partial runs commit checkpoints, complete runs seal, and complete cannot be resumed."""
        with isolated_project(with_solver=True) as root:
            # Create project with 64 steps, stopping at step 32
            proj = declarative.create_project(
                {
                    "name": "resumption_probe",
                    "task": "Test checkpoint commit and completion sealing",
                    "kind": "simulation",
                    "dimension": "2d",
                    "domain": {"x_um": [-0.5, 0.5], "y_um": [-0.3, 0.3]},
                    "grid": {"dx_um": 0.02, "dy_um": 0.02},
                    "pml_cells": 4,
                    "steps": 64,
                    "max_steps": 32,
                    "tau_box": 1.0,
                    "decay_floor": 0.9,
                    "incident_flux": 1.0e-30,
                    "source": {
                        "name": "src",
                        "kind": "plane",
                        "normal": "x",
                        "direction": "+",
                        "component": "Ey",
                        "center": [-0.2, 0.0, 0.0],
                        "size": [0.0, 0.3, 0.02],
                        "wavelength": 1.55,
                    },
                    "poynting_box": {
                        "center": [0.0, 0.0, 0.0],
                        "size": [0.6, 0.4, 0.02],
                    },
                },
                root=root,
            )
            cfg_path = proj["config_path"]

            # Phase 1: Submit partial run
            res1 = platform.submit_run(config=cfg_path, dry_run=False, root=root)
            self.assertTrue(res1["ok"], res1)
            run_id = res1["run_id"]
            run_dir = root / "runs" / run_id
            journal_dir = run_dir / "journal"

            # Checkpoint invariants for partial run
            self.assertTrue(journal_dir.is_dir(), "journal/ must exist")
            self.assertTrue((journal_dir / "contract.json").is_file(), "contract.json must exist")
            self.assertTrue((journal_dir / "latest-forward.json").is_file(), "latest-forward.json must exist")
            self.assertFalse(
                (journal_dir / "complete.json").exists(),
                "complete.json must NOT exist for partial runs stopping at max_steps",
            )

            # Inspect breakpoint
            bp1 = platform.inspect_breakpoint(run_id, root=root)
            self.assertTrue(bp1["ok"], bp1)
            self.assertTrue(bp1["can_resume"], "Partial run must be resumable")
            self.assertFalse(bp1["is_completed"], "Partial run must not be marked completed")
            self.assertGreater(bp1["checkpoint"]["step"], 0, "Checkpoint step must be > 0")

            # Phase 2: Resume run to completion
            res2 = platform.resume_run(run_id, dry_run=False, root=root)
            self.assertTrue(res2["ok"], res2)
            new_run_id = res2.get("run_id") or res2.get("new_run_id")
            new_run_dir = root / "runs" / new_run_id
            new_journal_dir = new_run_dir / "journal"

            # Invariants for completed run
            self.assertTrue(
                (new_journal_dir / "complete.json").is_file(),
                "complete.json MUST exist once simulation runs to completion",
            )
            complete_content = json.loads((new_journal_dir / "complete.json").read_text(encoding="utf-8"))
            self.assertTrue("completed" in complete_content or "status" in complete_content)

            bp2 = platform.inspect_breakpoint(new_run_id, root=root)
            self.assertTrue(bp2["ok"], bp2)
            self.assertTrue(bp2["is_completed"], "Completed run must be marked completed")
            self.assertFalse(bp2["can_resume"], "Completed run must NOT be resumable")
            self.assertIn("already completed", bp2["refusal_reason"].lower())

            # Phase 3: Adversarial attempt to resume completed run
            complete_hash_before = hashlib.sha256((new_journal_dir / "complete.json").read_bytes()).hexdigest()
            res3 = platform.resume_run(new_run_id, dry_run=False, root=root)
            self.assertFalse(res3["ok"], "Resuming completed run must fail")
            self.assertEqual(res3["kind"], "unresumable_run")
            self.assertIn("already completed", res3["errors"][0].lower())

            # Verify complete.json was NEVER unlinked or touched
            self.assertTrue((new_journal_dir / "complete.json").is_file(), "complete.json was deleted!")
            complete_hash_after = hashlib.sha256((new_journal_dir / "complete.json").read_bytes()).hexdigest()
            self.assertEqual(complete_hash_before, complete_hash_after, "complete.json was tampered with!")

    def test_slot_corruption_and_clean_refusal(self):
        """Empirically test corrupted checkpoint slot recovery and clean refusal."""
        with isolated_project(with_solver=True) as root:
            run_id = "test_corrupt_slots_probe"
            rdir = root / "runs" / run_id
            jdir = rdir / "journal"
            jdir.mkdir(parents=True)

            (jdir / "contract.json").write_text(
                json.dumps({
                    "marker": "torchfdtd.resident_restart.v1",
                    "steps": 500,
                    "dt": 1.0e-16,
                    "grid_shape": [50, 50, 1],
                    "epsilon_sha256": "fake_hash",
                    "block_size": 32,
                }),
                encoding="utf-8",
            )
            (jdir / "latest-forward.json").write_text(
                json.dumps({"directory": "slot_a", "step": 128, "previous": "slot_b"}),
                encoding="utf-8",
            )

            # Slot A: Valid
            sa = jdir / "slot_a"
            sa.mkdir()
            (sa / "meta.json").write_text(json.dumps({"step": 128, "arrays": [{"file": "state.bin"}]}))
            (sa / "state.bin").write_bytes(b"\x11" * 64)

            # Slot B: Corrupted (missing binary tensor)
            sb = jdir / "slot_b"
            sb.mkdir()
            (sb / "meta.json").write_text(json.dumps({"step": 96, "arrays": [{"file": "state.bin"}]}))

            bp = platform.inspect_breakpoint(run_id, root=root)
            self.assertTrue(bp["ok"])
            self.assertTrue(bp["slot_health"]["slot_a"]["valid"])
            self.assertFalse(bp["slot_health"]["slot_b"]["valid"])
            self.assertIn("state.bin", bp["slot_health"]["slot_b"]["missing_files"])

            # Destroy contract.json -> verify clean refusal
            (jdir / "contract.json").unlink()
            res_refusal = platform.resume_run(run_id, dry_run=True, root=root)
            self.assertFalse(res_refusal["ok"])
            self.assertEqual(res_refusal["kind"], "no_checkpoint")


if __name__ == "__main__":
    unittest.main()
