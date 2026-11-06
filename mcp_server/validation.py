"""Config validation that calls the project's own gate.

There is one validator in this project for a run config: the pre-flight gate in
``checks/pre_environment.py``.  This module executes that file and reports what
it says.  It does not restate the rules, so the MCP tool and the launcher cannot
drift apart.

The gate also probes the machine (git commits, CUDA availability).  A validation
is about the config, and must not read the machine, so those probes are supplied
with inert answers here.  The config assertions themselves are the real ones.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest import mock

from . import paths


_LOCK = threading.RLock()


def validate_config(name, root=None):
    """Run ``checks/pre_environment.py`` against ``name`` and return its answer.

    Returns ``{"ok": bool, "errors": [str], ...}``.  It runs no simulation and
    starts no process.
    """
    root = (root or paths.project_root()).resolve()
    config_path, error = paths.resolve_config(name, root)
    if error:
        return _refusal(error, name, root)

    gate = root / "checks" / "pre_environment.py"
    if not gate.is_file():
        return _refusal(f"existing validator is missing: {gate}", name, root)

    try:
        config = paths.read_json(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _refusal(f"config is not readable JSON: {exc}", name, root)
    if not isinstance(config, dict):
        return _refusal("config must be a JSON object", name, root)

    try:
        pin = paths.read_json(root / "configs" / "baseline-pin.json")
    except (OSError, json.JSONDecodeError):
        pin = {}
    upstream = (pin.get("upstream", {}) or {}).get("solver_commit", "")

    stdout = io.StringIO()
    stderr = io.StringIO()
    with _LOCK:
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "PARITY_CONFIG": str(config_path),
                    "PARITY_SOLVER_ROLE": "fork",
                    "PARITY_SOLVER_ROOT": "",
                    "PARITY_SOLVER_UPSTREAM": "",
                },
                clear=False,
            ), mock.patch.object(
                subprocess, "check_output", side_effect=_inert_git(upstream)
            ), mock.patch.object(
                subprocess,
                "run",
                side_effect=RuntimeError("validation must not start a process"),
            ), mock.patch.object(
                subprocess,
                "Popen",
                side_effect=RuntimeError("validation must not start a process"),
            ), mock.patch.object(
                subprocess,
                "check_call",
                side_effect=RuntimeError("validation must not start a process"),
            ), mock.patch.dict(
                sys.modules, {"torch": _inert_torch()}
            ), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(
                stderr
            ):
                import runpy

                runpy.run_path(str(gate), run_name="mcp_validation")
            return {
                "ok": True,
                "errors": [],
                "config": paths.relative_to_root(config_path, root),
                "validator": "checks/pre_environment.py",
                "environment_stubbed": True,
                "note": (
                    "the config gate ran with its machine probes (git, CUDA) stubbed; "
                    "config assertions are the project's own"
                ),
            }
        except SystemExit as exc:
            message = _message(exc.code, stderr.getvalue())
            return _failure(message, config_path, root)
        except AssertionError as exc:
            return _failure(str(exc), config_path, root)
        except Exception as exc:  # no traceback escapes a tool
            return _failure(f"{type(exc).__name__}: {exc}", config_path, root)


def _failure(message, config_path, root):
    return {
        "ok": False,
        "errors": [message],
        "config": paths.relative_to_root(config_path, root),
        "validator": "checks/pre_environment.py",
        "environment_stubbed": True,
        "note": (
            "the config gate ran with its machine probes (git, CUDA) stubbed; "
            "config assertions are the project's own"
        ),
    }


def _refusal(message, name, root):
    try:
        label = paths.relative_to_root(paths.resolve_config(name, root)[0], root)
    except Exception:
        label = str(name)
    return {
        "ok": False,
        "errors": [message],
        "config": label,
        "validator": "checks/pre_environment.py",
        "environment_stubbed": True,
    }


def _message(code, stderr):
    if isinstance(code, str) and code.strip():
        return code.strip()
    text = (stderr or "").strip()
    if text:
        return text
    return f"pre-flight refused the config (exit {code})"


def _inert_git(upstream_commit):
    """Answer the gate's git probes without running git.

    The gate asks for the upstream commit, the upstream dirty state and the fork
    commit/dirty state.  Validation does not need the machine, so the upstream
    answers are the pinned values and the fork answer is a local label.
    """

    def check_output(args, **kwargs):
        path = Path(args[2]) if len(args) > 2 else Path("")
        if len(args) >= 5 and args[3:] == ["rev-parse", "--show-toplevel"]:
            return str(path.resolve()) + "\n"
        if len(args) >= 5 and args[3:] == ["rev-parse", "HEAD"]:
            return (upstream_commit or "stub-upstream") + "\n"
        if len(args) >= 5 and args[3:] == ["status", "--porcelain"]:
            return ""
        if "upstream" in path.name:
            return (upstream_commit or "stub-upstream") + "\n"
        return "stub-fork\n"

    return check_output


def _inert_torch():
    torch = mock.MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_name.return_value = "validation stub (no GPU read)"
    return torch
