"""The eleven MCP tools' platform logic.

Only ``submit_run`` (and ``resume_run``) starts a process, and the only program
it starts is ``tools/run.sh``.  The exit status of that process is returned
unchanged; ``ok`` is derived from it, never from the presence of a result file.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Optional

from . import exits, paths, validation
from .plan import criteria_for


_RUN_ID = re.compile(r"\[run\.sh\] run_id=(\S+)")
_TAIL_LINES = 20
_TAIL_CHARS = 4000


def _ledger_path(root):
    return root / "runs" / "ledger.jsonl"


def _ledger_rows(root):
    path = _ledger_path(root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def _purpose(config):
    task = config.get("purpose") or config.get("task") or ""
    if not isinstance(task, str):
        return ""
    one_line = " ".join(task.split())
    sentence = re.split(r"(?<=[.!?])\s+", one_line, maxsplit=1)[0]
    return sentence[:240]


def _config_for(root, name):
    path, error = paths.resolve_config(name, root)
    if error:
        return None, None, error
    try:
        data = paths.read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return path, None, f"config is not readable JSON: {exc}"
    if not isinstance(data, dict):
        return path, None, "config must be a JSON object"
    return path, data, None


def solver_status(root=None):
    root = (root or paths.project_root()).resolve()
    role = os.environ.get("PARITY_SOLVER_ROLE", "fork")
    try:
        pin = paths.read_json(root / "configs" / "baseline-pin.json")
    except (OSError, json.JSONDecodeError):
        pin = {}
    if role == "upstream":
        declared = (pin.get("upstream", {}) or {}).get("solver_root", "src/torchfdtd_upstream")
    else:
        declared = pin.get("solver_root", "src/torchfdtd")
    solver = root / declared
    if (
        role == "fork"
        and not (solver / "torchfdtd").is_dir()
        and (root / "vendor" / "torchfdtd" / "torchfdtd" / "__init__.py").is_file()
        and (root / "tools" / "setup_agent.sh").is_file()
    ):
        subprocess.run(
            ["bash", str(root / "tools" / "setup_agent.sh"), "--skip-pip", "--quiet"],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    return {
        "role": role,
        "root": declared,
        "available": (solver / "torchfdtd").is_dir(),
        "python_package_seen": (solver / "torchfdtd" / "__init__.py").is_file(),
    }


def _driver_from_config(root, config):
    for key in ("driver", "script"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            candidate = (root / value).resolve()
            return {
                "driver": value,
                "driver_source": f"config field `{key}`",
                "driver_exists": paths.inside_root(candidate, root) and candidate.is_file(),
            }
    return None


def _driver_from_ledger(root, config_path):
    target = config_path.resolve()
    found = None
    for row in _ledger_rows(root):
        recorded = row.get("config")
        script = row.get("script")
        if not isinstance(recorded, str) or not isinstance(script, str):
            continue
        try:
            candidate = (root / recorded).resolve()
        except OSError:
            continue
        if candidate == target and paths.inside_root(root / script, root):
            found = script
    if found is None:
        return None
    return {
        "driver": found,
        "driver_source": "runs/ledger.jsonl",
        "driver_exists": (root / found).is_file(),
    }


def driver_for(root, config_path, config):
    return _driver_from_config(root, config) or _driver_from_ledger(root, config_path)


def _command(root, config_path, driver):
    config_arg = paths.relative_to_root(config_path, root)
    argv = [(root / "tools" / "run.sh").as_posix(), config_arg, driver]
    return argv, shlex.join(argv)


def _known_command(root):
    return (root / "tools" / "run.sh").as_posix()


def list_configs():
    root = paths.project_root().resolve()
    pairs = {}
    for row in _ledger_rows(root):
        config, script = row.get("config"), row.get("script")
        if isinstance(config, str) and isinstance(script, str):
            pairs[(config, script)] = True

    configs = []
    for config, script in sorted(pairs):
        path = root / config
        if not path.is_file():
            continue
        try:
            data = paths.read_json(path)
        except (OSError, json.JSONDecodeError):
            data = {}
        configs.append(
            {
                "config": config,
                "driver": script,
                "driver_exists": (root / script).is_file(),
                "purpose": _purpose(data) if isinstance(data, dict) else "",
            }
        )

    paired = {config for config, _ in pairs}
    unpaired = []
    for path in sorted((root / "configs").glob("*.json")):
        rel = paths.relative_to_root(path, root)
        if rel in paired:
            continue
        try:
            data = paths.read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data.get("task") or not data.get("kind"):
            continue
        unpaired.append(
            {
                "config": rel,
                "purpose": _purpose(data),
                "reason": "no ledger row records a driver for this config; none is invented",
            }
        )

    return {
        "ok": True,
        "configs": configs,
        "unpaired_configs": unpaired,
        "source": "runs/ledger.jsonl (config -> driver), configs/*.json (purpose)",
    }


def describe_run(config):
    root = paths.project_root().resolve()
    config_path, data, error = _config_for(root, config)
    if error:
        return {"ok": False, "kind": "config_unknown", "errors": [error], "config": str(config)}

    driver = driver_for(root, config_path, data)
    validation_result = validation.validate_config(config_path, root)
    criteria = criteria_for(config_path, root / "plan.md")
    solver = solver_status(root)
    issues = []
    if driver is None:
        issues.append("no driver is recorded for this config; none was invented")
    elif not driver["driver_exists"]:
        issues.append(f"recorded driver does not exist: {driver['driver']}")
    if not validation_result["ok"]:
        issues.extend(validation_result["errors"])

    argv = None
    command = None
    if driver and driver["driver_exists"]:
        argv, command = _command(root, config_path, driver["driver"])

    result = {
        "ok": not issues,
        "config": paths.relative_to_root(config_path, root),
        "driver": driver["driver"] if driver else None,
        "driver_source": driver["driver_source"] if driver else None,
        "driver_exists": driver["driver_exists"] if driver else False,
        "solver": solver,
        "criteria": criteria["criteria"],
        "criteria_text": criteria["criteria_text"],
        "criteria_source": criteria["criteria_source"],
        "criteria_note": criteria["criteria_note"],
        "validation": validation_result,
        "command": command,
        "argv": argv,
    }
    if issues:
        result["kind"] = "unusable_config"
        result["errors"] = issues
    return result


def _tail(text):
    if not text:
        return ""
    lines = text.strip().splitlines()
    tail = "\n".join(lines[-_TAIL_LINES:])
    return tail[-_TAIL_CHARS:]


def _ledger_row_for(root, run_id):
    if not run_id:
        return None
    for row in reversed(_ledger_rows(root)):
        if row.get("run_id") == run_id:
            return row
    return None


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _validate_run_id(run_id: Any) -> Optional[dict]:
    if not isinstance(run_id, str) or not run_id.strip():
        return {
            "ok": False,
            "kind": "bad_request",
            "errors": ["run_id must be a non-empty string"],
            "run_id": run_id,
        }
    if ".." in run_id or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "errors": [f"invalid characters in run_id: {run_id}"],
            "run_id": run_id,
        }
    return None


def _inspect_journal(root: Path, target_str: str) -> dict:
    """Inspect and validate a ResidentRestartJournal directory or run_id."""
    if "\x00" in str(target_str):
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "error": f"Invalid run_id (null character prohibited): {target_str}",
        }
    runs_dir = (root / "runs").resolve()
    try:
        cand_runs = (runs_dir / target_str).resolve()
    except (OSError, ValueError):
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "error": f"Invalid run_id: {target_str}",
        }
    if not paths.inside_root(cand_runs, runs_dir):
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "error": f"Invalid run_id (path traversal prohibited): {target_str}",
        }
    if not cand_runs.is_dir():
        return {
            "ok": False,
            "kind": "run_not_found",
            "error": f"Run directory or journal target not found: {target_str}",
        }

    journal_dir = (cand_runs / "journal") if (cand_runs / "journal").is_dir() else cand_runs
    run_dir = cand_runs

    contract_path = journal_dir / "contract.json"
    pointer_path = journal_dir / "latest-forward.json"
    owner_path = journal_dir / "owner.json"
    complete_path = journal_dir / "complete.json"

    if not contract_path.is_file():
        return {
            "ok": False,
            "kind": "no_checkpoint",
            "error": f"No contract.json in journal {journal_dir}",
            "journal_dir": str(journal_dir),
        }
    if not pointer_path.is_file():
        return {
            "ok": False,
            "kind": "no_checkpoint",
            "error": f"No checkpoint pointer (latest-forward.json) in journal {journal_dir}",
            "journal_dir": str(journal_dir),
        }

    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "kind": "corrupted_checkpoint",
            "error": f"Unreadable journal metadata: {exc}",
        }

    # Lock & Liveness Check
    owner_data = None
    is_locked = False
    is_stale_lock = False
    if owner_path.is_file():
        try:
            owner_data = json.loads(owner_path.read_text(encoding="utf-8"))
            pid = owner_data.get("pid")
            if isinstance(pid, int) and pid > 0:
                if _pid_alive(pid):
                    is_locked = True
                else:
                    is_stale_lock = True
        except Exception:
            pass

    is_completed = complete_path.is_file()
    step = pointer.get("step", 0)
    total_steps = contract.get("steps", 0)

    # Ping-Pong Slot Health
    slot_health = {}
    for slot_name in ("slot_a", "slot_b"):
        slot_dir = journal_dir / slot_name
        meta_file = slot_dir / "meta.json"
        if slot_dir.is_dir() and meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                arrays = meta.get("arrays", [])
                missing = [a["file"] for a in arrays if not (slot_dir / a["file"]).is_file()]
                slot_health[slot_name] = {
                    "exists": True,
                    "step": meta.get("step"),
                    "valid": len(missing) == 0,
                    "missing_files": missing,
                }
            except Exception:
                slot_health[slot_name] = {"exists": True, "valid": False}
        else:
            slot_health[slot_name] = {"exists": False, "valid": False}

    can_resume = (not is_completed) and (not is_locked)
    refusal_reason = None
    if is_completed:
        refusal_reason = f"Simulation already completed (reached step {step}/{total_steps})"
    elif is_locked:
        refusal_reason = f"Journal actively locked by PID {owner_data.get('pid')} (run {owner_data.get('run_id')})"

    return {
        "ok": True,
        "run_id": run_dir.name if run_dir else None,
        "run_dir": str(run_dir) if run_dir else None,
        "journal_dir": str(journal_dir),
        "contract": contract,
        "checkpoint": {
            "step": step,
            "total_steps": total_steps,
            "progress_percent": round(100.0 * step / total_steps, 2) if total_steps else None,
            "block": pointer.get("block"),
            "peak_energy": pointer.get("peak_energy"),
            "current_slot": pointer.get("directory"),
            "fallback_slot": pointer.get("previous"),
            "timestamp": pointer.get("written"),
        },
        "is_completed": is_completed,
        "is_locked": is_locked,
        "is_stale_lock": is_stale_lock,
        "owner": owner_data,
        "slot_health": slot_health,
        "can_resume": can_resume,
        "refusal_reason": refusal_reason,
    }


def inspect_breakpoint(run_id: str, root=None) -> dict:
    err = _validate_run_id(run_id)
    if err:
        return err
    root = (root or paths.project_root()).resolve()
    info = _inspect_journal(root, run_id)
    if not info["ok"]:
        return {
            "ok": False,
            "kind": info.get("kind", "inspect_failed"),
            "run_id": run_id,
            "errors": [info.get("error", "Journal inspection failed")],
        }
    return {
        "ok": True,
        "run_id": info["run_id"],
        "can_resume": info["can_resume"],
        "refusal_reason": info["refusal_reason"],
        "checkpoint": info["checkpoint"],
        "is_completed": info["is_completed"],
        "is_locked": info["is_locked"],
        "is_stale_lock": info["is_stale_lock"],
        "contract": {
            "dt": info["contract"].get("dt"),
            "steps": info["contract"].get("steps"),
            "grid_shape": info["contract"].get("grid_shape"),
            "epsilon_sha256": info["contract"].get("epsilon_sha256"),
            "block_size": info["contract"].get("block_size"),
        },
        "slot_health": info["slot_health"],
    }


def submit_run(config, dry_run=True, resume_from=None, root=None):
    root = (root or paths.project_root()).resolve()
    config_path, data, error = _config_for(root, config)
    if error:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "invalid_config",
            "errors": [error],
            "config": str(config),
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    validation_result = validation.validate_config(config_path, root)
    if not validation_result["ok"]:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "invalid_config",
            "errors": validation_result["errors"],
            "config": paths.relative_to_root(config_path, root),
            "validation": validation_result,
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    driver = driver_for(root, config_path, data)
    if driver is None:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "driver_unknown",
            "errors": ["no driver is recorded for this config; none was invented"],
            "config": paths.relative_to_root(config_path, root),
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }
    if not driver["driver_exists"]:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "driver_missing",
            "errors": [f"recorded driver does not exist: {driver['driver']}"],
            "config": paths.relative_to_root(config_path, root),
            "driver": driver["driver"],
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    # Pre-validate checkpoint if resume_from is requested
    resume_info = None
    if resume_from:
        resume_info = _inspect_journal(root, resume_from)
        if not resume_info["ok"]:
            return {
                "ok": False,
                "dry_run": bool(dry_run),
                "executed": False,
                "kind": resume_info["kind"],
                "errors": [resume_info["error"]],
                "config": paths.relative_to_root(config_path, root),
                "resume_from": resume_from,
                "exit_status": None,
                "run_id": None,
                "ledger_row": None,
                "stderr_tail": "",
            }
        if not resume_info["can_resume"]:
            return {
                "ok": False,
                "dry_run": bool(dry_run),
                "executed": False,
                "kind": "unresumable_run",
                "errors": [resume_info["refusal_reason"]],
                "config": paths.relative_to_root(config_path, root),
                "resume_from": resume_from,
                "checkpoint": resume_info["checkpoint"],
                "exit_status": None,
                "run_id": None,
                "ledger_row": None,
                "stderr_tail": "",
            }
        contract = resume_info["contract"]
        cfg_dt = data.get("dt") or data.get("time_step") if data else None
        if cfg_dt is not None and "dt" in contract:
            scale = max(abs(float(cfg_dt)), abs(float(contract["dt"])), 1e-30)
            if abs(float(cfg_dt) - float(contract["dt"])) / scale > 1e-9:
                return {
                    "ok": False,
                    "dry_run": bool(dry_run),
                    "executed": False,
                    "kind": "contract_mismatch",
                    "errors": [f"Config dt ({cfg_dt}) does not match checkpoint contract ({contract['dt']})"],
                    "config": paths.relative_to_root(config_path, root),
                    "resume_from": resume_from,
                    "exit_status": None,
                    "run_id": None,
                    "ledger_row": None,
                    "stderr_tail": "",
                }

    solver = solver_status(root)
    argv, command = _command(root, config_path, driver["driver"])
    if resume_from:
        argv.extend(["--resume", str(resume_from)])
        command = shlex.join(argv)

    launcher_exists = (root / "tools" / "run.sh").is_file()
    preflight_errors = []
    if not launcher_exists:
        preflight_errors.append(f"launcher does not exist: {root / 'tools' / 'run.sh'}")
    if not solver["available"]:
        preflight_errors.append(
            f"solver is unavailable at {solver['root']}; run tools/bootstrap_solver.sh first"
        )
    preflight = {
        "ok": not preflight_errors,
        "errors": preflight_errors,
        "validation": validation_result,
        "solver": solver,
        "launcher_exists": launcher_exists,
        "note": (
            "this is the config-level verdict; tools/run.sh runs its own "
            "checks/pre_environment.py when a real run starts"
        ),
    }

    if dry_run:
        return {
            "ok": preflight["ok"],
            "dry_run": True,
            "executed": False,
            "kind": (
                "dry_run"
                if preflight["ok"]
                else "launcher_missing"
                if not launcher_exists
                else "solver_unavailable"
            ),
            "config": paths.relative_to_root(config_path, root),
            "driver": driver["driver"],
            "command": command,
            "argv": argv,
            "resume_from": resume_from,
            "resumed_from_step": resume_info["checkpoint"]["step"] if resume_info else None,
            "preflight": preflight,
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    if not solver["available"]:
        return {
            "ok": False,
            "dry_run": False,
            "executed": False,
            "kind": "solver_unavailable",
            "errors": preflight["errors"],
            "config": paths.relative_to_root(config_path, root),
            "driver": driver["driver"],
            "command": command,
            "argv": argv,
            "preflight": preflight,
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    run_sh = _known_command(root)
    if not Path(run_sh).is_file():
        return {
            "ok": False,
            "dry_run": False,
            "executed": False,
            "kind": "launcher_missing",
            "errors": [f"launcher does not exist: {run_sh}"],
            "config": paths.relative_to_root(config_path, root),
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    exec_argv = argv
    if os.name == "nt":
        import sys
        launch_py = root / "tools" / "launch.py"
        solver_dst = root / "src" / "torchfdtd"
        solver_vendor = root / "vendor" / "torchfdtd" / "torchfdtd"
        if not (solver_dst / "torchfdtd").is_dir() and solver_vendor.is_dir():
            try:
                os.symlink(solver_vendor.parent.resolve(), solver_dst, target_is_directory=True)
            except OSError:
                try:
                    import _winapi
                    _winapi.CreateJunction(str(solver_vendor.parent.resolve()), str(solver_dst))
                except OSError:
                    pass
        py_exe = os.environ.get("PY") or sys.executable
        exec_argv = [py_exe, "-B", "-u", str(launch_py), *argv[1:]]

    try:
        completed = subprocess.run(
            exec_argv,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {
            "ok": False,
            "dry_run": False,
            "executed": False,
            "kind": "launcher_error",
            "errors": [f"could not start tools/run.sh: {exc}"],
            "config": paths.relative_to_root(config_path, root),
            "driver": driver["driver"],
            "command": command,
            "exit_status": None,
            "run_id": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    exit_status = completed.returncode
    match = _RUN_ID.search(completed.stdout or "")
    run_id = match.group(1) if match else None
    ledger_row = _ledger_row_for(root, run_id)
    has_stderr = bool((completed.stderr or "").strip())
    return {
        "ok": exit_status == 0,
        "dry_run": False,
        "process_started": True,
        "executed": exit_status not in exits.NOT_EXECUTED,
        "kind": exits.kind(exit_status),
        "exit_status": exit_status,
        "run_id": run_id,
        "resumed_from_run_id": resume_from,
        "resumed_from_step": resume_info["checkpoint"]["step"] if resume_info else None,
        "ledger_row": ledger_row,
        "config": paths.relative_to_root(config_path, root),
        "driver": driver["driver"],
        "command": command,
        "stderr_tail": _tail(completed.stderr) or _tail(completed.stdout),
        "tail_source": "stderr" if has_stderr else "stdout",
        "stdout_tail": _tail(completed.stdout),
        "meaning": exits.explain(exit_status)["meaning"],
    }


def resume_run(run_id: str, dry_run: bool = True, additional_steps: int | None = None, root=None, **kwargs) -> dict:
    err = _validate_run_id(run_id)
    if err:
        err["dry_run"] = bool(dry_run)
        err["executed"] = False
        err["new_run_id"] = None
        err["exit_status"] = None
        err["ledger_row"] = None
        err["stderr_tail"] = ""
        return err
    root = (root or paths.project_root()).resolve()
    info = _inspect_journal(root, run_id)
    if not info["ok"]:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": info.get("kind", "run_not_found"),
            "errors": [info.get("error", f"Run {run_id} not found or unresumable")],
            "run_id": run_id,
            "new_run_id": None,
            "exit_status": None,
            "ledger_row": None,
            "stderr_tail": "",
        }
    if not info["can_resume"]:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "unresumable_run",
            "errors": [info["refusal_reason"]],
            "run_id": run_id,
            "new_run_id": None,
            "checkpoint": info["checkpoint"],
            "exit_status": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    manifest_path = Path(info["run_dir"]) / "manifest.json"
    config_rel = None
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_val = m.get("config")
            if config_val:
                try:
                    config_rel = Path(config_val).resolve().relative_to(root).as_posix()
                except ValueError:
                    config_rel = str(config_val).replace("\\", "/")
        except Exception:
            pass

    if not config_rel:
        row = _ledger_row_for(root, run_id)
        if row and row.get("config"):
            config_rel = row["config"]

    if not config_rel:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "executed": False,
            "kind": "manifest_missing",
            "errors": [f"Cannot determine original configuration for run {run_id}"],
            "run_id": run_id,
            "new_run_id": None,
            "exit_status": None,
            "ledger_row": None,
            "stderr_tail": "",
        }

    res = submit_run(config=config_rel, dry_run=dry_run, resume_from=run_id, root=root)
    res["resumed_from_run_id"] = run_id
    res["resumed_from_step"] = info["checkpoint"]["step"]
    res["new_run_id"] = res.get("run_id")
    return res


def get_certificate(run_id: str, root=None) -> dict:
    root = (root or paths.project_root()).resolve()
    if not isinstance(run_id, str) or not run_id.strip():
        return {
            "ok": False,
            "kind": "bad_request",
            "errors": ["run_id must be a non-empty string"],
            "run_id": run_id,
        }
    if ".." in run_id or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "errors": [f"invalid characters in run_id: {run_id}"],
            "run_id": run_id,
        }
    try:
        run_dir = (root / "runs" / run_id).resolve()
    except (OSError, ValueError):
        return {
            "ok": False,
            "kind": "invalid_run_id",
            "errors": [f"invalid run_id: {run_id}"],
            "run_id": run_id,
        }
    if not run_dir.is_dir() or not paths.inside_root(run_dir, root):
        return {
            "ok": False,
            "kind": "run_not_found",
            "errors": [f"run_id '{run_id}' does not exist under runs/"],
            "run_id": run_id,
        }

    cert_path = run_dir / "verification_certificate.json"
    manifest_path = run_dir / "manifest.json"

    manifest_data = {}
    if manifest_path.is_file():
        try:
            manifest_data = paths.read_json(manifest_path)
        except Exception:
            manifest_data = {}

    provenance = {
        "project_git_head": manifest_data.get("project_git_head"),
        "solver_commit": manifest_data.get("solver_commit"),
        "solver_role": manifest_data.get("solver_role"),
        "solver_dirty_lines": manifest_data.get("solver_dirty_lines"),
        "gpu": manifest_data.get("gpu"),
        "cpu_platform": manifest_data.get("cpu_platform"),
        "script": manifest_data.get("script"),
        "config": manifest_data.get("config"),
        "started_utc": manifest_data.get("started_utc"),
        "finished_utc": manifest_data.get("finished_utc"),
        "resumed_from_journal": manifest_data.get("resumed_from_journal"),
        "resumed_from_run_id": manifest_data.get("resumed_from_run_id"),
    }

    if not cert_path.is_file():
        status = manifest_data.get("status", "unknown")
        exit_code = manifest_data.get("exit_code")
        return {
            "ok": False,
            "kind": "certificate_missing",
            "errors": [f"no verification_certificate.json found for run '{run_id}'"],
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "provenance": provenance,
        }

    try:
        cert_data = paths.read_json(cert_path)
    except Exception as e:
        return {
            "ok": False,
            "kind": "certificate_corrupted",
            "errors": [f"failed to parse verification_certificate.json: {e}"],
            "run_id": run_id,
            "provenance": provenance,
        }

    # Verify tampering / SHA256 integrity
    tamper_issues = []
    telemetry_file = run_dir / "solver_telemetry.json"
    readout_file = run_dir / "readout.json"
    if "telemetry_sha256" in cert_data:
        if not telemetry_file.is_file():
            tamper_issues.append("solver_telemetry.json is missing on disk (deletion detected)")
        else:
            actual_tel_sha = hashlib.sha256(telemetry_file.read_bytes()).hexdigest()
            if actual_tel_sha != cert_data["telemetry_sha256"]:
                tamper_issues.append("solver_telemetry.json hash does not match certificate")

    if "readout_sha256" in cert_data:
        if not readout_file.is_file():
            tamper_issues.append("readout.json is missing on disk (deletion detected)")
        else:
            actual_rdo_sha = hashlib.sha256(readout_file.read_bytes()).hexdigest()
            if actual_rdo_sha != cert_data["readout_sha256"]:
                tamper_issues.append("readout.json hash does not match certificate")

    if "certificate_payload_sha256" in cert_data:
        payload_copy = {
            k: v
            for k, v in cert_data.items()
            if k not in ("certificate_payload_sha256", "hmac_sha256")
        }
        payload_bytes = json.dumps(payload_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_payload_sha = hashlib.sha256(payload_bytes).hexdigest()
        if expected_payload_sha != cert_data["certificate_payload_sha256"]:
            tamper_issues.append("verification_certificate.json payload hash mismatch")
        notary_secret = os.environ.get("TORCHFDTD_NOTARY_SECRET")
        if notary_secret and "hmac_sha256" in cert_data:
            expected_hmac = hmac.new(
                notary_secret.encode("utf-8"), payload_bytes, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected_hmac, str(cert_data["hmac_sha256"])):
                tamper_issues.append("verification_certificate.json HMAC signature mismatch")

    verified = bool(cert_data.get("verified", False))
    exit_code = manifest_data.get("exit_code")

    if exit_code is not None and exit_code != 0:
        verified = False

    if tamper_issues:
        verified = False

    status = manifest_data.get("status", "completed" if verified else cert_data.get("status", "failed"))

    metrics = {
        "executed_steps": cert_data.get("executed_steps"),
        "resumed_from_step": cert_data.get("resumed_from_step", 0),
        "total_power_sum": cert_data.get("total_power_sum"),
        "passivity_passed": cert_data.get("passivity_passed"),
        "energy_decay_ratio": cert_data.get("energy_decay_ratio", cert_data.get("decay_ratio")),
        "spectral_decay_passed": cert_data.get("spectral_decay_passed"),
        "box_closure_residual": cert_data.get("box_closure_residual"),
        "box_closure_passed": cert_data.get("box_closure_passed"),
        "auto_shutoff_triggered": cert_data.get("auto_shutoff_triggered", False),
        "max_flux_diff": cert_data.get("max_flux_diff"),
        "compared_monitors": cert_data.get("compared_monitors"),
    }

    extensions = {}
    for ext_key in ("mesh_convergence", "graded_accuracy", "cross_solver_reconciliation", "inverse_design"):
        if ext_key in cert_data:
            extensions[ext_key] = cert_data[ext_key]

    return {
        "ok": True,
        "run_id": run_id,
        "verified": verified,
        "certified_utc": cert_data.get("certified_utc"),
        "status": status,
        "exit_code": exit_code,
        "tags": cert_data.get("tags", []),
        "failure_cause": cert_data.get("status") if not verified else None,
        "error": cert_data.get("error") if not verified else None,
        "tamper_detected": bool(tamper_issues),
        "tamper_errors": tamper_issues,
        "metrics": metrics,
        "extensions": extensions,
        "provenance": provenance,
        "raw_certificate": cert_data,
    }


def create_project(project_spec=None, project_dict=None, root=None, **kwargs):
    from . import declarative
    spec = project_spec if project_spec is not None else project_dict
    return declarative.create_project(spec, root=root, **kwargs)


def validate_project(project=None, config=None, config_path=None, check_environment=True, root=None, **kwargs):
    from . import declarative
    return declarative.validate_project(
        project=project,
        config=config,
        config_path=config_path,
        check_environment=check_environment,
        root=root,
        **kwargs,
    )


def read_ledger(limit=20):
    root = paths.project_root().resolve()
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "kind": "bad_request", "errors": ["limit must be an integer"], "rows": []}
    if limit < 0:
        return {"ok": False, "kind": "bad_request", "errors": ["limit must be >= 0"], "rows": []}
    rows = _ledger_rows(root)
    selected = rows[-limit:] if limit else []
    return {
        "ok": True,
        "rows": selected,
        "count": len(selected),
        "total": len(rows),
        "ledger": paths.relative_to_root(_ledger_path(root), root),
    }


def explain_exit(code):
    return exits.explain(code)
