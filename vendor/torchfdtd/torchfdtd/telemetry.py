"""Solver kernel telemetry emitter for verifiable zero-trust simulation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

TELEMETRY_SCHEMA_VERSION = "1.0"


def get_run_dir() -> Optional[Path]:
    """Return Path to run directory if RUN_DIR_ABS is set and exists, else None."""
    val = os.environ.get("RUN_DIR_ABS")
    if not val or not val.strip():
        return None
    p = Path(val.strip())
    return p if p.is_dir() else None


def hash_array(array: Any) -> str:
    """Return SHA-256 hex digest of contiguous array or tensor bytes."""
    if isinstance(array, torch.Tensor):
        array = array.detach().contiguous().cpu().numpy()
    else:
        array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def emit_telemetry(
    run_dir: Path,
    *,
    project_canonical_sha256: str,
    epsilon_sha256: str,
    is_lossless: bool,
    executed_steps: int,
    resumed_from_step: int = 0,
    energy_decay_ratio: float = 0.0,
    monitors: Dict[str, Dict[str, Any]],
    auto_shutoff_triggered: bool = False,
    shutoff_step: Optional[int] = None,
    shutoff_energy_ratio: float = 0.0,
    shutoff_flux_delta: float = 0.0,
    execution_profile: Optional[str] = None,
) -> Path:
    """Emit unforgeable solver_telemetry.json atomically to run_dir."""
    telemetry = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "project_canonical_sha256": str(project_canonical_sha256),
        "epsilon_sha256": str(epsilon_sha256),
        "is_lossless": bool(is_lossless),
        "executed_steps": int(executed_steps),
        "resumed_from_step": int(resumed_from_step),
        "energy_decay_ratio": float(energy_decay_ratio),
        "monitors": monitors,
        "auto_shutoff_triggered": bool(auto_shutoff_triggered),
        "shutoff_step": int(shutoff_step) if shutoff_step is not None else None,
        "shutoff_energy_ratio": float(shutoff_energy_ratio),
        "shutoff_flux_delta": float(shutoff_flux_delta),
        "execution_profile": str(execution_profile) if execution_profile is not None else "reference",
    }
    target = run_dir / "solver_telemetry.json"
    tmp_target = run_dir / "solver_telemetry.json.tmp"
    tmp_target.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp_target.replace(target)
    return target
