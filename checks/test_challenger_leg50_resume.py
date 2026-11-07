"""Adversarial Empirical Challenge Suite for Wave D - LEG 50.

Stress-tests:
1. Attack 1: Non-existent run_id resume refusal (run_not_found / no_checkpoint without crashing).
2. Attack 2: Completed run resume refusal (complete.json prevents unresumable execution).
3. Attack 3: Corrupted checkpoint slot recovery (invalid JSON, missing tensors, damaged journal metadata).
4. Attack 4: Advisory lock contention (exclusive lock on runs/.lock refuses cleanly without hung deadlock).
5. Attack 5: Postflight refusal query (exit 66 authentic run query returns ok:True, verified:False, failure_cause:"passivity_violation").
6. Attack 6: Dry-run invariant (zero OS processes spawned, zero filesystem modifications).
7. Attack 7: File deletion tamper detection in get_certificate (missing telemetry/readout triggers tamper_detected:True, verified:False).
"""
from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

# Ensure tools/run.sh uses current python virtual environment when running in isolated test dirs
os.environ.setdefault("PY", sys.executable)

from mcp_server import platform
from mcptests.support import ROOT, isolated_project


class TestChallengerLeg50Resume(unittest.TestCase):
    """Adversarial challenge test cases for MCP Server v2."""

    def _create_mock_checkpoint(
        self,
        root: Path,
        run_id: str,
        is_completed: bool = False,
        corrupt_meta: bool = False,
        missing_tensor: bool = False,
        corrupt_latest: bool = False,
    ) -> Path:
        run_dir = root / "runs" / run_id
        journal_dir = run_dir / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)

        contract = {
            "marker": "torchfdtd.resident_restart.v1",
            "steps": 600,
            "dt": 1.5e-16,
            "grid_shape": [120, 60, 1],
            "epsilon_sha256": "mock_eps_hash_12345",
            "block_size": 32,
        }
        (journal_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")

        if corrupt_latest:
            (journal_dir / "latest-forward.json").write_text('{"step": 300, "corrupt_trailing_comma":', encoding="utf-8")
        else:
            pointer = {
                "directory": "slot_a",
                "step": 300,
                "block": 9,
                "peak_energy": 0.88,
                "previous": "slot_b",
                "written": "2026-11-05T13:00:00Z",
            }
            (journal_dir / "latest-forward.json").write_text(json.dumps(pointer), encoding="utf-8")

        for slot_name in ("slot_a", "slot_b"):
            sdir = journal_dir / slot_name
            sdir.mkdir(parents=True, exist_ok=True)

            if slot_name == "slot_a" and corrupt_meta:
                (sdir / "meta.json").write_text("{ unquoted_key: invalid_json ", encoding="utf-8")
            else:
                meta = {
                    "step": 300 if slot_name == "slot_a" else 268,
                    "arrays": [{"file": "state.bin"}],
                }
                (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            if not (slot_name == "slot_a" and missing_tensor):
                (sdir / "state.bin").write_bytes(b"\x00" * 128)

        if is_completed:
            (journal_dir / "complete.json").write_text(
                json.dumps({"status": "completed", "steps": 600, "finished_utc": "2026-11-05T13:05:00Z"}),
                encoding="utf-8",
            )

        manifest = {
            "run_id": run_id,
            "config": "configs/exit57.json",
            "exit_code": 0,
            "status": "completed" if is_completed else "interrupted",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    # -------------------------------------------------------------------------
    # Attack 1: Non-existent run_id resume
    # -------------------------------------------------------------------------
    def test_attack1_non_existent_run_resume_refusal(self):
        """Call resume_run with non-existent run_id; must refuse cleanly without crashing."""
        with isolated_project(with_solver=True) as root:
            # 1a. Completely non-existent run directory
            res = platform.resume_run("non_existent_20261105T999999Z", root=root)
            self.assertFalse(res["ok"])
            self.assertEqual(res["kind"], "run_not_found")
            self.assertFalse(res["executed"])
            self.assertIn("not found", res["errors"][0].lower())

            # 1b. Non-existent run inspect_breakpoint
            inspect_res = platform.inspect_breakpoint("non_existent_20261105T999999Z", root=root)
            self.assertFalse(inspect_res["ok"])
            self.assertEqual(inspect_res["kind"], "run_not_found")
            self.assertIn("not found", inspect_res["errors"][0].lower())

            # 1c. Run directory exists under runs/ but lacks journal/contract
            empty_run = root / "runs" / "empty_run_20261105"
            empty_run.mkdir(parents=True, exist_ok=True)
            res_empty = platform.resume_run("empty_run_20261105", root=root)
            self.assertFalse(res_empty["ok"])
            self.assertEqual(res_empty["kind"], "no_checkpoint")
            self.assertIn("no contract.json", res_empty["errors"][0].lower())

    # -------------------------------------------------------------------------
    # Attack 2: Completed run resume
    # -------------------------------------------------------------------------
    def test_attack2_completed_run_resume_refusal(self):
        """Call resume_run on a run that already completed; must refuse cleanly."""
        with isolated_project(with_solver=True) as root:
            self._create_mock_checkpoint(root, "completed_run_leg50", is_completed=True)

            # 2a. inspect_breakpoint detects completion
            bp = platform.inspect_breakpoint("completed_run_leg50", root=root)
            self.assertTrue(bp["ok"])
            self.assertTrue(bp["is_completed"])
            self.assertFalse(bp["can_resume"])
            self.assertIn("already completed", bp["refusal_reason"].lower())

            # 2b. resume_run with dry_run=True refuses
            res_dry = platform.resume_run("completed_run_leg50", dry_run=True, root=root)
            self.assertFalse(res_dry["ok"])
            self.assertEqual(res_dry["kind"], "unresumable_run")
            self.assertFalse(res_dry["executed"])
            self.assertIn("already completed", res_dry["errors"][0].lower())

            # 2c. resume_run with dry_run=False refuses without launching process
            res_live = platform.resume_run("completed_run_leg50", dry_run=False, root=root)
            self.assertFalse(res_live["ok"])
            self.assertEqual(res_live["kind"], "unresumable_run")
            self.assertFalse(res_live["executed"])
            self.assertIn("already completed", res_live["errors"][0].lower())

            # 2d. submit_run with resume_from on completed run refuses
            res_sub = platform.submit_run(config="configs/exit57.json", resume_from="completed_run_leg50", root=root)
            self.assertFalse(res_sub["ok"])
            self.assertEqual(res_sub["kind"], "unresumable_run")

    # -------------------------------------------------------------------------
    # Attack 3: Corrupted checkpoint slot recovery
    # -------------------------------------------------------------------------
    def test_attack3_corrupted_checkpoint_slot_recovery(self):
        """Simulate damaged slot_a/slot_b and verify journal inspection reports cleanly."""
        with isolated_project(with_solver=True) as root:
            # 3a. Invalid JSON in slot_a/meta.json
            self._create_mock_checkpoint(root, "corrupt_meta_run", corrupt_meta=True)
            bp1 = platform.inspect_breakpoint("corrupt_meta_run", root=root)
            self.assertTrue(bp1["ok"])
            self.assertTrue(bp1["slot_health"]["slot_a"]["exists"])
            self.assertFalse(bp1["slot_health"]["slot_a"]["valid"])
            # slot_b remains valid
            self.assertTrue(bp1["slot_health"]["slot_b"]["valid"])

            # 3b. Missing tensor array in slot_a
            self._create_mock_checkpoint(root, "missing_tensor_run", missing_tensor=True)
            bp2 = platform.inspect_breakpoint("missing_tensor_run", root=root)
            self.assertTrue(bp2["ok"])
            self.assertFalse(bp2["slot_health"]["slot_a"]["valid"])
            self.assertIn("state.bin", bp2["slot_health"]["slot_a"]["missing_files"])

            # 3c. Corrupted latest-forward.json metadata
            self._create_mock_checkpoint(root, "corrupt_pointer_run", corrupt_latest=True)
            bp3 = platform.inspect_breakpoint("corrupt_pointer_run", root=root)
            self.assertFalse(bp3["ok"])
            self.assertEqual(bp3["kind"], "corrupted_checkpoint")
            self.assertIn("unreadable journal metadata", bp3["errors"][0].lower())

    # -------------------------------------------------------------------------
    # Attack 4: Advisory lock contention
    # -------------------------------------------------------------------------
    def test_attack4_advisory_lock_contention_no_deadlock(self):
        """Acquire runs/.lock externally and invoke submit_run(dry_run=False).

        Must raise clean refusal ('Another run holds the project lock') and not deadlock.
        """
        with isolated_project(with_solver=True) as root:
            runs_dir = root / "runs"
            runs_dir.mkdir(exist_ok=True)
            lock_path = runs_dir / ".lock"

            with open(lock_path, "a+") as lock_file:
                # Acquire exclusive non-blocking lock in test process
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    lock_file.seek(0, os.SEEK_END)
                    if lock_file.tell() == 0:
                        lock_file.write("\0")
                        lock_file.flush()
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    t0 = time.time()
                    res = platform.submit_run(
                        config="configs/exit57.json",
                        dry_run=False,
                        root=root,
                    )
                    elapsed = time.time() - t0
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    else:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

            # Assert execution returned immediately without hung deadlock
            self.assertLess(elapsed, 10.0, f"Lock contention took {elapsed:.2f}s, potential deadlock")
            self.assertFalse(res["ok"])
            self.assertEqual(res["exit_status"], 1)
            self.assertIn("Another run holds the project lock", res["stderr_tail"])

    # -------------------------------------------------------------------------
    # Attack 5: Postflight refusal query
    # -------------------------------------------------------------------------
    def test_attack5_postflight_refusal_query(self):
        """Invoke get_certificate on an authentic run that failed postflight (exit 66)."""
        run_id = "20261105T104537Z-3689"
        if not (ROOT / "runs" / run_id).is_dir():
            self.skipTest(f"Authentic exit 66 run {run_id} not available on host")

        cert = platform.get_certificate(run_id, root=ROOT)
        self.assertTrue(cert["ok"], cert)
        self.assertFalse(cert["verified"], "Postflight failure must NOT be verified")
        self.assertEqual(cert["exit_code"], 66)
        self.assertEqual(cert["failure_cause"], "passivity_violation")
        self.assertIn("passivity violation", cert["error"])
        self.assertEqual(cert["status"], "failed:postflight=passivity_violation")
        self.assertFalse(cert["tamper_detected"])

    # -------------------------------------------------------------------------
    # Attack 6: Dry-run invariant
    # -------------------------------------------------------------------------
    def test_attack6_dry_run_invariant(self):
        """Verify dry_run=True starts zero OS processes and modifies zero filesystem files."""
        with isolated_project(with_solver=True) as root:
            self._create_mock_checkpoint(root, "dry_run_checkpoint")

            def _snapshot_fs(directory: Path):
                snap = {}
                for p in sorted(directory.rglob("*")):
                    try:
                        stat = p.stat()
                        snap[str(p.relative_to(directory))] = (stat.st_mtime_ns, stat.st_size)
                    except OSError:
                        pass
                return snap

            before_snap = _snapshot_fs(root)

            # Patch subprocess.run to raise if called during dry-run
            with patch("subprocess.run") as mock_subproc:
                res_submit = platform.submit_run(config="configs/exit57.json", dry_run=True, root=root)
                res_resume = platform.resume_run("dry_run_checkpoint", dry_run=True, root=root)

                # Assert zero subprocesses launched
                mock_subproc.assert_not_called()

            # Verify submit_run dry-run payload
            self.assertTrue(res_submit["ok"])
            self.assertTrue(res_submit["dry_run"])
            self.assertFalse(res_submit["executed"])
            self.assertIsNone(res_submit.get("process_started"))
            self.assertIsInstance(res_submit["command"], str)
            self.assertIsInstance(res_submit["argv"], list)
            self.assertTrue(res_submit["preflight"]["ok"])

            # Verify resume_run dry-run payload
            self.assertTrue(res_resume["ok"])
            self.assertTrue(res_resume["dry_run"])
            self.assertFalse(res_resume["executed"])
            self.assertEqual(res_resume["resumed_from_step"], 300)
            self.assertIn("--resume dry_run_checkpoint", res_resume["command"])

            after_snap = _snapshot_fs(root)

            # Assert filesystem immutability
            self.assertEqual(
                set(before_snap.keys()),
                set(after_snap.keys()),
                f"Files created or deleted during dry-run: {set(after_snap.keys()) ^ set(before_snap.keys())}",
            )
            for file_path, before_stat in before_snap.items():
                after_stat = after_snap[file_path]
                self.assertEqual(
                    before_stat,
                    after_stat,
                    f"File modified during dry-run: {file_path}",
                )

    # -------------------------------------------------------------------------
    # Additional Stress: Active vs Stale PID locks
    # -------------------------------------------------------------------------
    def test_journal_pid_lock_liveness_and_stale_lock_recovery(self):
        """Verify journal owner.json correctly detects live vs stale PID locks."""
        with isolated_project(with_solver=True) as root:
            run_dir = self._create_mock_checkpoint(root, "lock_liveness_run")
            journal_dir = run_dir / "journal"

            # 1. Live PID lock (current test process PID)
            (journal_dir / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "run_id": "lock_liveness_run"}),
                encoding="utf-8",
            )
            bp_live = platform.inspect_breakpoint("lock_liveness_run", root=root)
            self.assertTrue(bp_live["ok"])
            self.assertTrue(bp_live["is_locked"])
            self.assertFalse(bp_live["is_stale_lock"])
            self.assertFalse(bp_live["can_resume"])
            self.assertIn(f"actively locked by PID {os.getpid()}", bp_live["refusal_reason"])

            res_resume = platform.resume_run("lock_liveness_run", dry_run=True, root=root)
            self.assertFalse(res_resume["ok"])
            self.assertEqual(res_resume["kind"], "unresumable_run")

            # 2. Stale PID lock (defunct PID)
            stale_pid = 4194300  # Above Linux default max pid or unused high pid
            (journal_dir / "owner.json").write_text(
                json.dumps({"pid": stale_pid, "run_id": "lock_liveness_run"}),
                encoding="utf-8",
            )
            bp_stale = platform.inspect_breakpoint("lock_liveness_run", root=root)
            self.assertTrue(bp_stale["ok"])
            self.assertFalse(bp_stale["is_locked"])
            self.assertTrue(bp_stale["is_stale_lock"])
            self.assertTrue(bp_stale["can_resume"])

    # -------------------------------------------------------------------------
    # Additional Stress: Contract dt mismatch rejection
    # -------------------------------------------------------------------------
    def test_contract_dt_mismatch_refusal(self):
        """Verify submit_run(resume_from=...) rejects when config dt differs from checkpoint contract."""
        with isolated_project(with_solver=True) as root:
            self._create_mock_checkpoint(root, "dt_mismatch_ckpt")
            # exit57.json does not define dt, so let's create a custom config with conflicting dt
            cfg = {
                "task": "Physics simulation",
                "kind": "physics",
                "driver": "src/exit57.py",
                "dt": 3.0e-16,  # Differs from contract dt (1.5e-16)
            }
            (root / "configs" / "custom_dt.json").write_text(json.dumps(cfg), encoding="utf-8")

            res = platform.submit_run(
                config="configs/custom_dt.json",
                resume_from="dt_mismatch_ckpt",
                dry_run=True,
                root=root,
            )
            self.assertFalse(res["ok"])
            self.assertEqual(res["kind"], "contract_mismatch")
            self.assertIn("does not match checkpoint contract", res["errors"][0])

    # -------------------------------------------------------------------------
    # Additional Stress: Anti-Tamper telemetry & readout detection
    # -------------------------------------------------------------------------
    def test_get_certificate_anti_tamper_detection(self):
        """Verify get_certificate strips verified status if telemetry or readout is mutated."""
        with isolated_project(with_solver=True) as root:
            run_id = "test_tampered_telemetry"
            rdir = root / "runs" / run_id
            rdir.mkdir(parents=True)

            tel = {"schema_version": "1.0", "executed_steps": 200, "is_lossless": True}
            rdo = {"steps": 200, "passivity": True}
            tel_bytes = json.dumps(tel).encode("utf-8")
            rdo_bytes = json.dumps(rdo).encode("utf-8")
            (rdir / "solver_telemetry.json").write_bytes(tel_bytes)
            (rdir / "readout.json").write_bytes(rdo_bytes)
            (rdir / "manifest.json").write_text(json.dumps({"exit_code": 0, "status": "completed"}))

            cert_data = {
                "schema_version": "1.0",
                "run_id": run_id,
                "verified": True,
                "telemetry_sha256": hashlib.sha256(tel_bytes).hexdigest(),
                "readout_sha256": hashlib.sha256(rdo_bytes).hexdigest(),
                "tags": ["verified:postflight_notary"],
            }
            (rdir / "verification_certificate.json").write_text(json.dumps(cert_data))

            # Untampered: passes
            c0 = platform.get_certificate(run_id, root=root)
            self.assertTrue(c0["ok"])
            self.assertTrue(c0["verified"])
            self.assertFalse(c0["tamper_detected"])

            # Mutate solver_telemetry.json
            (rdir / "solver_telemetry.json").write_text(json.dumps({"executed_steps": 9999}))
            c1 = platform.get_certificate(run_id, root=root)
            self.assertTrue(c1["ok"])
            self.assertTrue(c1["tamper_detected"])
            self.assertFalse(c1["verified"])
            self.assertIn("solver_telemetry.json hash does not match certificate", c1["tamper_errors"])

    # -------------------------------------------------------------------------
    # Attack 7: File deletion tamper detection in get_certificate
    # -------------------------------------------------------------------------
    def test_attack7_file_deletion_tamper_detection(self):
        """Verify get_certificate strips verified status if telemetry or readout is deleted on disk."""
        with isolated_project(with_solver=True) as root:
            run_id = "test_deleted_telemetry_run"
            rdir = root / "runs" / run_id
            rdir.mkdir(parents=True)

            tel = {"schema_version": "1.0", "executed_steps": 200, "is_lossless": True}
            rdo = {"steps": 200, "passivity": True}
            tel_bytes = json.dumps(tel).encode("utf-8")
            rdo_bytes = json.dumps(rdo).encode("utf-8")
            (rdir / "solver_telemetry.json").write_bytes(tel_bytes)
            (rdir / "readout.json").write_bytes(rdo_bytes)
            (rdir / "manifest.json").write_text(json.dumps({"exit_code": 0, "status": "completed"}))

            cert_data = {
                "schema_version": "1.0",
                "run_id": run_id,
                "verified": True,
                "telemetry_sha256": hashlib.sha256(tel_bytes).hexdigest(),
                "readout_sha256": hashlib.sha256(rdo_bytes).hexdigest(),
                "tags": ["verified:postflight_notary"],
            }
            (rdir / "verification_certificate.json").write_text(json.dumps(cert_data))

            # Case A: Delete solver_telemetry.json
            (rdir / "solver_telemetry.json").unlink()
            cA = platform.get_certificate(run_id, root=root)
            self.assertTrue(cA["ok"])
            self.assertTrue(cA["tamper_detected"])
            self.assertFalse(cA["verified"])
            self.assertIn("solver_telemetry.json is missing on disk (deletion detected)", cA["tamper_errors"])

            # Case B: Delete readout.json as well
            (rdir / "readout.json").unlink()
            cB = platform.get_certificate(run_id, root=root)
            self.assertTrue(cB["ok"])
            self.assertTrue(cB["tamper_detected"])
            self.assertFalse(cB["verified"])
            self.assertIn("solver_telemetry.json is missing on disk (deletion detected)", cB["tamper_errors"])
            self.assertIn("readout.json is missing on disk (deletion detected)", cB["tamper_errors"])



if __name__ == "__main__":
    unittest.main()
