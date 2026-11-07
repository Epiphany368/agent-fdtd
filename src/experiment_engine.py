"""Authoritative SQLite WAL Experiment Engine, Transactional Outbox & Worker Fencing (LEG 55).

Single source of truth for photonic experiments, multi-stage DAGs, monotonic
worker fencing tokens, zero-duplicate recovery, and idempotent projection to runs/ledger.jsonl.
"""

from __future__ import annotations

import contextlib
import datetime
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

PRAGMAS = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA busy_timeout = 10000;",
    "PRAGMA temp_store = MEMORY;",
]

SCHEMA_DDL = """
-- 1. Master Experiments Table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    ir_spec_sha256 TEXT NOT NULL,
    ir_spec_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('CREATED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    target_claim TEXT,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL,
    completed_utc TEXT
);

-- 2. Monotonic Fencing Token Sequence Generator
CREATE TABLE IF NOT EXISTS fencing_sequence (
    sequence_name TEXT PRIMARY KEY,
    current_token INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO fencing_sequence (sequence_name, current_token) VALUES ('global_worker_fencing', 0);

-- 3. Stages Table
CREATE TABLE IF NOT EXISTS stages (
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    stage_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL DEFAULT 'default',
    fidelity_level INTEGER NOT NULL DEFAULT 1,
    stage_order INTEGER NOT NULL DEFAULT 0,
    backend TEXT NOT NULL CHECK (backend IN ('TORCHFDTD', 'RCWA', 'MEEP', 'GDS', 'ANALYTIC')),
    state TEXT NOT NULL CHECK (state IN ('PENDING', 'RUNNING', 'NOTARIZED', 'FAILED')),
    config_hash TEXT NOT NULL,
    run_id TEXT,
    run_dir TEXT,
    journal_path TEXT,
    current_fencing_token INTEGER NOT NULL DEFAULT 0,
    restart_step INTEGER NOT NULL DEFAULT 0,
    executed_steps INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    exit_code INTEGER,
    error_message TEXT,
    readout_json TEXT,
    certificate_json TEXT,
    certificate_sha256 TEXT,
    created_utc TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT,
    PRIMARY KEY (experiment_id, stage_id, candidate_id, fidelity_level)
);

CREATE INDEX IF NOT EXISTS idx_stages_state ON stages(state);
CREATE INDEX IF NOT EXISTS idx_stages_run_id ON stages(run_id);

-- 4. Worker Leases Table
CREATE TABLE IF NOT EXISTS worker_leases (
    lease_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL DEFAULT 'default',
    fidelity_level INTEGER NOT NULL DEFAULT 1,
    worker_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    heartbeat_utc TEXT NOT NULL,
    expires_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'EXPIRED', 'RELEASED', 'REVOKED')),
    created_utc TEXT NOT NULL,
    FOREIGN KEY (experiment_id, stage_id, candidate_id, fidelity_level)
        REFERENCES stages(experiment_id, stage_id, candidate_id, fidelity_level) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_leases_stage_status 
    ON worker_leases(experiment_id, stage_id, candidate_id, fidelity_level, status);
CREATE INDEX IF NOT EXISTS idx_leases_fencing_token 
    ON worker_leases(fencing_token);

-- 5. Transactional Outbox Table
CREATE TABLE IF NOT EXISTS ledger_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    fidelity_level INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('STAGE_NOTARIZED', 'STAGE_FAILED', 'RUN_COMPLETED')),
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'PROJECTED', 'FAILED')) DEFAULT 'PENDING',
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL,
    projected_utc TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending ON ledger_outbox(status, outbox_id);

-- 6. Stage Audit Log Table
CREATE TABLE IF NOT EXISTS stage_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    fidelity_level INTEGER NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    fencing_token INTEGER,
    worker_id TEXT,
    reason TEXT NOT NULL,
    details_json TEXT,
    timestamp_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_stage 
    ON stage_audit_log(experiment_id, stage_id, candidate_id, fidelity_level);
"""


# ==============================================================================
# Exceptions
# ==============================================================================
class ExperimentEngineError(Exception):
    """Base error for experiment engine operations."""


class FencingTokenRevokedError(ExperimentEngineError):
    """Raised when a worker attempts mutation with a superseded or expired fencing token."""


class StageAlreadyNotarizedError(ExperimentEngineError):
    """Raised when attempting to execute an already completed/notarized stage."""


class LeaseAlreadyHeldError(ExperimentEngineError):
    """Raised when a stage lease is currently held by an active live worker."""


class CheckpointDamagedError(ExperimentEngineError):
    """Raised when both ping-pong checkpoint slots fail SHA-256 integrity validation."""


# ==============================================================================
# Database Connection & Initialization
# ==============================================================================
class ExperimentDatabase:
    """Manages SQLite connection pooling, WAL pragma enforcement, and schema initialization."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,  # Autocommit mode; explicit BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row
        with conn:
            for pragma in PRAGMAS:
                conn.execute(pragma)
        return conn

    def init_schema(self) -> None:
        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.executescript(SCHEMA_DDL)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ==============================================================================
# Worker Fencing & Mutual Exclusion
# ==============================================================================
class WorkerFencingManager:
    """Acquires, validates, and manages strictly monotonic fencing tokens."""

    @staticmethod
    def acquire_lease(
        conn: sqlite3.Connection,
        experiment_id: str,
        stage_id: str,
        candidate_id: str,
        fidelity_level: int,
        worker_id: str,
        ttl_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Atomically acquires a stage lease and increments the monotonic fencing token."""
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_dt.isoformat()
        expires_iso = (now_dt + datetime.timedelta(seconds=ttl_seconds)).isoformat()

        # Check stage state
        row = conn.execute(
            """
            SELECT state, current_fencing_token FROM stages
            WHERE experiment_id = ? AND stage_id = ? AND candidate_id = ? AND fidelity_level = ?
            """,
            (experiment_id, stage_id, candidate_id, fidelity_level),
        ).fetchone()

        if not row:
            raise ExperimentEngineError(f"Stage not found: {experiment_id}:{stage_id}")
        if row["state"] == "NOTARIZED":
            raise StageAlreadyNotarizedError(
                f"Stage {stage_id} already NOTARIZED; 0 duplicate recompute allowed."
            )

        # Check active leases
        active_lease = conn.execute(
            """
            SELECT lease_id, worker_id, expires_utc FROM worker_leases
            WHERE experiment_id = ? AND stage_id = ? AND candidate_id = ? AND fidelity_level = ?
              AND status = 'ACTIVE'
            """,
            (experiment_id, stage_id, candidate_id, fidelity_level),
        ).fetchone()

        if active_lease:
            if active_lease["expires_utc"] > now_iso and active_lease["worker_id"] != worker_id:
                raise LeaseAlreadyHeldError(
                    f"Lease active on worker {active_lease['worker_id']} until {active_lease['expires_utc']}"
                )
            else:
                conn.execute(
                    "UPDATE worker_leases SET status = 'EXPIRED' WHERE lease_id = ?",
                    (active_lease["lease_id"],),
                )

        # Monotonically increment fencing sequence
        conn.execute(
            """
            UPDATE fencing_sequence
            SET current_token = current_token + 1
            WHERE sequence_name = 'global_worker_fencing'
            """
        )
        token_row = conn.execute(
            "SELECT current_token FROM fencing_sequence WHERE sequence_name = 'global_worker_fencing'"
        ).fetchone()
        fencing_token = int(token_row["current_token"])

        # Create lease
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO worker_leases (
                lease_id, experiment_id, stage_id, candidate_id, fidelity_level,
                worker_id, fencing_token, heartbeat_utc, expires_utc, status, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
            """,
            (lease_id, experiment_id, stage_id, candidate_id, fidelity_level,
             worker_id, fencing_token, now_iso, expires_iso, now_iso),
        )

        # Update stage state and current fencing token
        conn.execute(
            """
            UPDATE stages
            SET current_fencing_token = ?, state = 'RUNNING', started_utc = ?
            WHERE experiment_id = ? AND stage_id = ? AND candidate_id = ? AND fidelity_level = ?
            """,
            (fencing_token, now_iso, experiment_id, stage_id, candidate_id, fidelity_level),
        )

        # Audit log
        conn.execute(
            """
            INSERT INTO stage_audit_log (
                experiment_id, stage_id, candidate_id, fidelity_level,
                from_state, to_state, fencing_token, worker_id, reason, timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, 'RUNNING', ?, ?, 'Lease acquired with fencing token', ?)
            """,
            (experiment_id, stage_id, candidate_id, fidelity_level,
             row["state"], fencing_token, worker_id, now_iso),
        )

        return {
            "lease_id": lease_id,
            "fencing_token": fencing_token,
            "expires_utc": expires_iso,
            "worker_id": worker_id,
        }

    @staticmethod
    def renew_heartbeat(
        conn: sqlite3.Connection,
        lease_id: str,
        fencing_token: int,
        ttl_seconds: int = 300,
    ) -> str:
        """Extends active lease TTL if fencing token is current."""
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_dt.isoformat()
        expires_iso = (now_dt + datetime.timedelta(seconds=ttl_seconds)).isoformat()

        row = conn.execute(
            """
            SELECT l.experiment_id, l.stage_id, l.candidate_id, l.fidelity_level,
                   s.current_fencing_token, l.status
            FROM worker_leases l
            JOIN stages s ON l.experiment_id = s.experiment_id AND l.stage_id = s.stage_id
                          AND l.candidate_id = s.candidate_id AND l.fidelity_level = s.fidelity_level
            WHERE l.lease_id = ? AND l.fencing_token = ?
            """,
            (lease_id, fencing_token),
        ).fetchone()

        if not row or row["status"] != "ACTIVE" or row["current_fencing_token"] != fencing_token:
            raise FencingTokenRevokedError(
                f"Lease {lease_id} with token {fencing_token} is no longer active or has been revoked."
            )

        conn.execute(
            "UPDATE worker_leases SET heartbeat_utc = ?, expires_utc = ? WHERE lease_id = ?",
            (now_iso, expires_iso, lease_id),
        )
        return expires_iso


# ==============================================================================
# Transactional Outbox Pattern
# ==============================================================================
class TransactionalOutboxManager:
    """Commits stage completion and writes outbox record in the exact same SQLite transaction."""

    @staticmethod
    def commit_stage_completion(
        conn: sqlite3.Connection,
        experiment_id: str,
        stage_id: str,
        candidate_id: str,
        fidelity_level: int,
        worker_id: str,
        fencing_token: int,
        run_id: str,
        run_dir: str,
        exit_code: int,
        readout_data: Dict[str, Any],
        certificate_data: Optional[Dict[str, Any]],
        manifest_row: Dict[str, Any],
    ) -> None:
        """Atomically notarizes stage, releases lease, and enqueues outbox record."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cert_json = json.dumps(certificate_data, sort_keys=True) if certificate_data else None
        cert_sha256 = hashlib.sha256(cert_json.encode("utf-8")).hexdigest() if cert_json else None
        readout_json = json.dumps(readout_data, sort_keys=True)
        is_verified = bool(exit_code == 0 and certificate_data and certificate_data.get("verified"))
        final_state = "NOTARIZED" if is_verified else "FAILED"

        if fencing_token <= 0:
            raise FencingTokenRevokedError("Commit rejected: fencing_token must be positive.")

        # Guarded atomic update: asserts current_fencing_token matches exactly and state == 'RUNNING'
        cursor = conn.execute(
            """
            UPDATE stages
            SET state = ?,
                run_id = ?,
                run_dir = ?,
                exit_code = ?,
                readout_json = ?,
                certificate_json = ?,
                certificate_sha256 = ?,
                finished_utc = ?
            WHERE experiment_id = ? AND stage_id = ? AND candidate_id = ? AND fidelity_level = ?
              AND current_fencing_token = ? AND state = 'RUNNING'
            """,
            (final_state, run_id, str(run_dir), exit_code, readout_json, cert_json, cert_sha256, now_iso,
             experiment_id, stage_id, candidate_id, fidelity_level, fencing_token),
        )

        if cursor.rowcount == 0:
            raise FencingTokenRevokedError(
                f"Stage '{stage_id}' commit rejected: stale fencing token {fencing_token} or stage not in RUNNING state."
            )

        # Release worker lease
        conn.execute(
            "UPDATE worker_leases SET status = 'RELEASED' WHERE fencing_token = ?",
            (fencing_token,),
        )

        # Insert into ledger_outbox (guaranteed unique by run_id)
        event_type = "STAGE_NOTARIZED" if final_state == "NOTARIZED" else "STAGE_FAILED"
        payload_str = json.dumps(manifest_row, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO ledger_outbox (
                run_id, experiment_id, stage_id, candidate_id, fidelity_level,
                event_type, payload_json, status, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            ON CONFLICT(run_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                event_type = excluded.event_type
            """,
            (run_id, experiment_id, stage_id, candidate_id, fidelity_level, event_type, payload_str, now_iso),
        )

        # Audit log
        conn.execute(
            """
            INSERT INTO stage_audit_log (
                experiment_id, stage_id, candidate_id, fidelity_level,
                from_state, to_state, fencing_token, worker_id, reason, details_json, timestamp_utc
            ) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, ?, 'Stage completion committed', ?, ?)
            """,
            (experiment_id, stage_id, candidate_id, fidelity_level, final_state,
             fencing_token, worker_id, json.dumps({"exit_code": exit_code, "run_id": run_id}), now_iso),
        )


# ==============================================================================
# Deterministic Ledger Projector
# ==============================================================================
class LedgerProjector:
    """Deterministically projects outbox rows to runs/ledger.jsonl with file locking & deduplication."""

    def __init__(self, db: ExperimentDatabase, ledger_path: Path | str):
        self.db = db
        self.ledger_path = Path(ledger_path).resolve()
        self.lock_path = self.ledger_path.parent / ".ledger.lock"

    def project_all(self, batch_size: int = 100) -> int:
        """Sweeps pending outbox entries and appends to ledger.jsonl idempotently."""
        conn = self.db.get_connection()
        projected_count = 0
        try:
            conn.execute("BEGIN IMMEDIATE;")
            pending = conn.execute(
                """
                SELECT outbox_id, run_id, payload_json
                FROM ledger_outbox
                WHERE status = 'PENDING'
                ORDER BY outbox_id ASC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()

            if not pending:
                conn.commit()
                return 0

            # Acquire cross-process file lock on ledger
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "a", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    # Read existing run_ids from runs/ledger.jsonl
                    existing_run_ids = set()
                    if self.ledger_path.exists():
                        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                            line_str = line.strip()
                            if line_str:
                                try:
                                    existing_run_ids.add(json.loads(line_str).get("run_id"))
                                except Exception:
                                    pass

                    lines_to_append = []
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    for row in pending:
                        outbox_id = row["outbox_id"]
                        run_id = row["run_id"]
                        payload_json = row["payload_json"]

                        if run_id not in existing_run_ids:
                            lines_to_append.append(payload_json)
                            existing_run_ids.add(run_id)

                        conn.execute(
                            "UPDATE ledger_outbox SET status = 'PROJECTED', projected_utc = ? WHERE outbox_id = ?",
                            (now_iso, outbox_id),
                        )
                        projected_count += 1

                    if lines_to_append:
                        with self.ledger_path.open("a", encoding="utf-8") as f:
                            for line in lines_to_append:
                                f.write(line + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                finally:
                    if fcntl is not None:
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass

            conn.commit()
            return projected_count
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ==============================================================================
# Crash Recovery Protocol
# ==============================================================================
class CrashRecoveryProtocol:
    """Evaluates stage state and determines execution action: SKIP, RESUME, or FRESH."""

    @staticmethod
    def prepare_stage_execution(
        conn: sqlite3.Connection,
        experiment_id: str,
        stage_id: str,
        candidate_id: str,
        fidelity_level: int,
    ) -> Dict[str, Any]:
        """Evaluates stage state and determines execution action."""
        stage = conn.execute(
            """
            SELECT state, journal_path, restart_step, executed_steps, total_steps,
                   certificate_json, readout_json
            FROM stages
            WHERE experiment_id = ? AND stage_id = ? AND candidate_id = ? AND fidelity_level = ?
            """,
            (experiment_id, stage_id, candidate_id, fidelity_level),
        ).fetchone()

        if not stage:
            raise ExperimentEngineError(f"Stage does not exist: {experiment_id}:{stage_id}")

        # 1. Zero-Duplicate Recompute Rule for Notarized Stages
        if stage["state"] == "NOTARIZED":
            return {
                "action": "SKIP_NOTARIZED",
                "message": "Stage already completed and notarized; 0 duplicate compute required.",
                "certificate": json.loads(stage["certificate_json"]) if stage["certificate_json"] else None,
                "readout": json.loads(stage["readout_json"]) if stage["readout_json"] else None,
            }

        # 2. Check for Checkpoint Resumption from Interrupted Stage
        journal_path_str = stage["journal_path"]
        if journal_path_str and Path(journal_path_str).is_dir():
            journal_path = Path(journal_path_str)
            recovery_verdict = CrashRecoveryProtocol._verify_journal_integrity(journal_path)
            if recovery_verdict["can_resume"]:
                return {
                    "action": "RESUME_CHECKPOINT",
                    "resume_step": recovery_verdict["step"],
                    "slot": recovery_verdict["slot"],
                    "journal_path": str(journal_path),
                    "message": f"Resuming from verified checkpoint at step {recovery_verdict['step']} via {recovery_verdict['slot']}.",
                }
            else:
                return {
                    "action": "RESTART_FROM_STEP_0",
                    "resume_step": 0,
                    "reason": recovery_verdict["reason"],
                    "journal_path": str(journal_path),
                    "message": f"Checkpoint unrecoverable ({recovery_verdict['reason']}); restarting from step 0.",
                }

        # 3. Fresh Stage Start
        return {
            "action": "START_STEP_0",
            "resume_step": 0,
            "message": "Fresh stage start from step 0.",
        }

    @staticmethod
    def _verify_journal_integrity(journal_dir: Path) -> Dict[str, Any]:
        """Inspects dual ping-pong slots and verifies SHA-256 tensor digests."""
        pointer_file = journal_dir / "latest-forward.json"
        if not pointer_file.is_file():
            return {"can_resume": False, "reason": "Missing latest-forward.json pointer"}

        try:
            ptr = json.loads(pointer_file.read_text(encoding="utf-8"))
        except Exception as e:
            return {"can_resume": False, "reason": f"Corrupted latest-forward.json: {e}"}

        primary_slot = ptr.get("directory")
        fallback_slot = ptr.get("previous")

        # Validate primary slot
        if primary_slot and (journal_dir / primary_slot).is_dir():
            valid, step, _ = CrashRecoveryProtocol._validate_slot_sha256(journal_dir / primary_slot)
            if valid:
                return {"can_resume": True, "step": step, "slot": primary_slot}

        # Fallback to secondary slot
        if fallback_slot and (journal_dir / fallback_slot).is_dir():
            valid, step, _ = CrashRecoveryProtocol._validate_slot_sha256(journal_dir / fallback_slot)
            if valid:
                return {"can_resume": True, "step": step, "slot": fallback_slot, "fallback_used": True}

        return {"can_resume": False, "reason": "Both slots corrupted or missing"}

    @staticmethod
    def _validate_slot_sha256(slot_dir: Path) -> Tuple[bool, int, Optional[str]]:
        """Verifies meta.json descriptors and array SHA-256 digests in slot."""
        meta_path = slot_dir / "meta.json"
        if not meta_path.is_file():
            return False, 0, "Missing meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            step = int(meta.get("step", 0))
            arrays = meta.get("arrays", [])
            for desc in arrays:
                fpath = slot_dir / desc["file"]
                if not fpath.is_file():
                    return False, 0, f"Missing array file {desc['file']}"
                if fpath.stat().st_size != desc["bytes"]:
                    return False, 0, f"Size mismatch for {desc['file']}"
                actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                if actual_sha != desc["sha256"]:
                    return False, 0, f"SHA256 mismatch for {desc['file']}"
            return True, step, None
        except Exception as e:
            return False, 0, str(e)


# ==============================================================================
# Top-Level Experiment Engine Facade
# ==============================================================================
class ExperimentEngine:
    """High-level facade orchestrating SQLite WAL transactions, outbox projection, and recovery."""

    def __init__(self, root: Optional[Path | str] = None, db_path: Optional[Path | str] = None):
        self.root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
        self.db_path = Path(db_path).resolve() if db_path else self.root / "runs" / "experiments.db"
        self.ledger_path = self.root / "runs" / "ledger.jsonl"
        self.db = ExperimentDatabase(self.db_path)
        self.projector = LedgerProjector(self.db, self.ledger_path)

    def submit(self, experiment_data: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Submits an experiment and its stages into the authoritative database."""
        exp_id = experiment_data.get("experiment_id") or f"exp-{uuid.uuid4().hex[:8]}"
        name = experiment_data.get("name", "experiment")
        desc = experiment_data.get("description", "")
        ir_json = json.dumps(experiment_data.get("ir_spec", {}), sort_keys=True)
        ir_sha = hashlib.sha256(ir_json.encode("utf-8")).hexdigest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        stages_spec = experiment_data.get("stages", [])
        if not stages_spec:
            # Create a default stage
            stages_spec = [{
                "stage_id": "stage_0",
                "backend": "TORCHFDTD",
                "fidelity_level": 2,
                "candidate_id": "default",
            }]

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "experiment_id": exp_id,
                "name": name,
                "stage_count": len(stages_spec),
                "ir_spec_sha256": ir_sha,
                "stages": stages_spec,
            }

        conn = self.db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, ir_spec_sha256, ir_spec_json,
                    status, target_claim, created_utc, updated_utc
                ) VALUES (?, ?, ?, ?, ?, 'CREATED', ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    updated_utc = excluded.updated_utc
                """,
                (exp_id, name, desc, ir_sha, ir_json,
                 experiment_data.get("target_claim"), now_iso, now_iso),
            )

            for order, st in enumerate(stages_spec):
                stage_id = st.get("stage_id", f"stage_{order}")
                cand_id = st.get("candidate_id", "default")
                fid = int(st.get("fidelity_level", 1))
                backend = st.get("backend", "TORCHFDTD").upper()
                cfg_hash = hashlib.sha256(json.dumps(st, sort_keys=True).encode("utf-8")).hexdigest()

                conn.execute(
                    """
                    INSERT INTO stages (
                        experiment_id, stage_id, candidate_id, fidelity_level,
                        stage_order, backend, state, config_hash, created_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                    ON CONFLICT(experiment_id, stage_id, candidate_id, fidelity_level) DO NOTHING
                    """,
                    (exp_id, stage_id, cand_id, fid, order, backend, cfg_hash, now_iso),
                )

            conn.commit()
            return {
                "ok": True,
                "dry_run": False,
                "experiment_id": exp_id,
                "name": name,
                "stage_count": len(stages_spec),
                "status": "CREATED",
            }
        except Exception as e:
            conn.rollback()
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()

    def query(self, experiment_id: str, stage_id: Optional[str] = None) -> Dict[str, Any]:
        """Queries experiment and stage status."""
        conn = self.db.get_connection()
        try:
            exp_row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()

            if not exp_row:
                return {"ok": False, "error": f"Experiment not found: {experiment_id}"}

            if stage_id:
                stage_rows = conn.execute(
                    "SELECT * FROM stages WHERE experiment_id = ? AND stage_id = ?",
                    (experiment_id, stage_id),
                ).fetchall()
            else:
                stage_rows = conn.execute(
                    "SELECT * FROM stages WHERE experiment_id = ? ORDER BY stage_order ASC",
                    (experiment_id,),
                ).fetchall()

            stages = [dict(s) for s in stage_rows]

            # Get latest lease
            lease_row = conn.execute(
                """
                SELECT * FROM worker_leases
                WHERE experiment_id = ? ORDER BY fencing_token DESC LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()

            # Get pending outbox count
            outbox_pending = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM ledger_outbox
                WHERE experiment_id = ? AND status = 'PENDING'
                """,
                (experiment_id,),
            ).fetchone()

            return {
                "ok": True,
                "experiment_id": exp_row["experiment_id"],
                "name": exp_row["name"],
                "status": exp_row["status"],
                "target_claim": exp_row["target_claim"],
                "stages": stages,
                "worker_lease": dict(lease_row) if lease_row else None,
                "fencing_token": lease_row["fencing_token"] if lease_row else 0,
                "pending_outbox_records": outbox_pending["cnt"] if outbox_pending else 0,
            }
        finally:
            conn.close()

    def get_run_dir_for_experiment(self, experiment_id: str) -> Optional[Path]:
        """Finds run directory for a completed or active experiment."""
        conn = self.db.get_connection()
        try:
            row = conn.execute(
                "SELECT run_dir FROM stages WHERE experiment_id = ? AND run_dir IS NOT NULL LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if row and row["run_dir"]:
                p = Path(row["run_dir"])
                if p.is_dir():
                    return p
            return None
        finally:
            conn.close()

    def acquire_worker_lease(
        self,
        experiment_id: str,
        stage_id: str,
        candidate_id: str = "default",
        fidelity_level: int = 1,
        worker_id: str = "worker-0",
        ttl_seconds: int = 300,
    ) -> Dict[str, Any]:
        conn = self.db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            lease = WorkerFencingManager.acquire_lease(
                conn, experiment_id, stage_id, candidate_id, fidelity_level, worker_id, ttl_seconds
            )
            conn.commit()
            return lease
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def renew_worker_lease(
        self,
        lease_id: str,
        fencing_token: int,
        ttl_seconds: int = 300,
    ) -> str:
        conn = self.db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            exp = WorkerFencingManager.renew_heartbeat(conn, lease_id, fencing_token, ttl_seconds)
            conn.commit()
            return exp
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def commit_stage_run(
        self,
        experiment_id: str,
        stage_id: str,
        candidate_id: str,
        fidelity_level: int,
        worker_id: str,
        fencing_token: int,
        run_id: str,
        run_dir: str,
        exit_code: int,
        readout_data: Dict[str, Any],
        certificate_data: Optional[Dict[str, Any]],
        manifest_row: Dict[str, Any],
    ) -> None:
        conn = self.db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            TransactionalOutboxManager.commit_stage_completion(
                conn, experiment_id, stage_id, candidate_id, fidelity_level,
                worker_id, fencing_token, run_id, run_dir, exit_code,
                readout_data, certificate_data, manifest_row
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def project_outbox_to_ledger(self, batch_size: int = 100) -> int:
        return self.projector.project_all(batch_size=batch_size)

    def prepare_execution(
        self,
        experiment_id: str,
        stage_id: str,
        candidate_id: str = "default",
        fidelity_level: int = 1,
    ) -> Dict[str, Any]:
        conn = self.db.get_connection()
        try:
            return CrashRecoveryProtocol.prepare_stage_execution(
                conn, experiment_id, stage_id, candidate_id, fidelity_level
            )
        finally:
            conn.close()
