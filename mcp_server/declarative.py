"""Declarative scene builder and preflight validator for MCP Server v2 (LEG 50).

Provides autonomous optical agents with high-level declarative scene generation,
automated Poynting box synthesis for V2-box verification, and multi-tier
preflight validation without writing Python driver code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import paths

C0 = 299792458.0
MAX_RESIDENT_CELLS = 8_000_000


def synthesize_poynting_box(
    dimension: str,
    center: List[float],
    size: List[float],
    wavelength_um: float = 1.55,
) -> List[Dict[str, Any]]:
    """Synthesize 4-sided (2D) or 6-sided (3D) Poynting enclosure monitors.

    These planes are calibrated for automated postflight notary V2-box closure verification.
    """
    cx, cy, cz = center[0], center[1], center[2] if len(center) > 2 else 0.0
    sx, sy, sz = size[0], size[1], size[2] if len(size) > 2 else 0.04

    spec = {
        "sampling": "wavelength",
        "wavelength_start": round(wavelength_um * 0.95, 4),
        "wavelength_stop": round(wavelength_um * 1.05, 4),
        "frequency_points": 3,
        "apodization": "none",
    }

    # 4-surface 2D enclosure (x-left, x-right, y-top, y-bottom)
    monitors = [
        {
            "id": "box-right",
            "name": "box-right",
            "normal": "x",
            "center": [round(cx + sx / 2.0, 6), round(cy, 6), round(cz, 6)],
            "size": [0.0, round(sy, 6), round(sz, 6)],
            "record_flux": True,
            "spectrum": spec,
        },
        {
            "id": "box-left",
            "name": "box-left",
            "normal": "x",
            "center": [round(cx - sx / 2.0, 6), round(cy, 6), round(cz, 6)],
            "size": [0.0, round(sy, 6), round(sz, 6)],
            "record_flux": True,
            "spectrum": spec,
        },
        {
            "id": "box-top",
            "name": "box-top",
            "normal": "y",
            "center": [round(cx, 6), round(cy + sy / 2.0, 6), round(cz, 6)],
            "size": [round(sx, 6), 0.0, round(sz, 6)],
            "record_flux": True,
            "spectrum": spec,
        },
        {
            "id": "box-bottom",
            "name": "box-bottom",
            "normal": "y",
            "center": [round(cx, 6), round(cy - sy / 2.0, 6), round(cz, 6)],
            "size": [round(sx, 6), 0.0, round(sz, 6)],
            "record_flux": True,
            "spectrum": spec,
        },
    ]

    if dimension.lower() == "3d":
        monitors.extend([
            {
                "id": "box-front",
                "name": "box-front",
                "normal": "z",
                "center": [round(cx, 6), round(cy, 6), round(cz + sz / 2.0, 6)],
                "size": [round(sx, 6), round(sy, 6), 0.0],
                "record_flux": True,
                "spectrum": spec,
            },
            {
                "id": "box-back",
                "name": "box-back",
                "normal": "z",
                "center": [round(cx, 6), round(cy, 6), round(cz - sz / 2.0, 6)],
                "size": [round(sx, 6), round(sy, 6), 0.0],
                "record_flux": True,
                "spectrum": spec,
            },
        ])

    return monitors


def parse_and_build_project(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """Parse flexible declarative specifications into a canonical Project model dictionary.

    Returns (project_id, project_dict, canonical_sha256).
    Raises ValueError on invalid specifications.
    """
    if "project_id" in spec:
        val = spec["project_id"]
        if val is None or not str(val).strip():
            raise ValueError("project_id cannot be empty or None")
        project_id = str(val).strip()
    elif "name" in spec:
        val = spec["name"]
        if val is None or not str(val).strip():
            raise ValueError("name cannot be empty or None")
        project_id = str(val).strip()
    else:
        project_id = "declarative_project"

    if ".." in project_id or "/" in project_id or "\\" in project_id:
        raise ValueError(f"Invalid characters in project_id (path traversal prohibited): {project_id}")

    dimension = str(spec.get("dimension", "2d")).lower()
    if dimension not in ("2d", "3d"):
        raise ValueError(f"Invalid dimension: {dimension}; must be '2d' or '3d'")

    # Resolve grid / mesh spacing
    grid_cfg = spec.get("grid")
    if isinstance(grid_cfg, dict):
        dx = float(grid_cfg.get("dx_um", 0.02))
        dy = float(grid_cfg.get("dy_um", 0.02))
        dz = float(grid_cfg.get("dz_um", dx))
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError(f"Grid spacing must be strictly positive, got dx={dx}, dy={dy}, dz={dz}")
        mesh = min(dx, dy) if dimension == "2d" else min(dx, dy, dz)
    else:
        mesh = float(spec.get("mesh", 0.02))
        if mesh <= 0:
            raise ValueError(f"Mesh spacing must be strictly positive, got mesh={mesh}")
        dx = dy = dz = mesh

    # Resolve domain physical dimensions
    domain_cfg = spec.get("domain")
    if isinstance(domain_cfg, dict):
        x_span = domain_cfg.get("x_um")
        y_span = domain_cfg.get("y_um")
        z_span = domain_cfg.get("z_um")
        if x_span is None or y_span is None:
            raise ValueError("domain must specify x_um and y_um ranges, e.g. [-1.0, 1.0]")
        sx = abs(float(x_span[1]) - float(x_span[0]))
        sy = abs(float(y_span[1]) - float(y_span[0]))
        sz = abs(float(z_span[1]) - float(z_span[0])) if z_span else (dx if dimension == "2d" else 1.0)
        cx = (float(x_span[1]) + float(x_span[0])) / 2.0
        cy = (float(y_span[1]) + float(y_span[0])) / 2.0
        cz = (float(z_span[1]) + float(z_span[0])) / 2.0 if z_span else 0.0
    elif "size" in spec:
        raw_size = spec["size"]
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
            raise ValueError("size must be a list/tuple of at least 2 numbers [span_x, span_y]")
        sx = float(raw_size[0])
        sy = float(raw_size[1])
        sz = float(raw_size[2]) if len(raw_size) > 2 else (dx if dimension == "2d" else 1.0)
        cx = cy = cz = 0.0
    else:
        sx, sy, sz = 4.0, 2.0, (dx if dimension == "2d" else 1.0)
        cx = cy = cz = 0.0

    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"Physical spans must be strictly positive, got size=({sx}, {sy}, {sz})")

    # Time steps / duration
    steps = spec.get("steps")
    if steps is not None:
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"Steps must be >= 1, got {steps}")
    elif "duration_fs" in spec:
        duration_fs = float(spec["duration_fs"])
        if duration_fs <= 0:
            raise ValueError("duration_fs must be strictly positive")
        cfl_dt = (mesh * 1e-6) / (C0 * math.sqrt(2 if dimension == "2d" else 3))
        steps = max(10, math.ceil(duration_fs * 1e-15 / cfl_dt))
    else:
        steps = 200

    wavelength_um = float(spec.get("wavelength_um") or spec.get("wavelength") or 1.55)
    if wavelength_um <= 0:
        raise ValueError(f"Wavelength must be strictly positive, got {wavelength_um}")

    pml_cells = int(spec.get("pml_cells", 16 if max(sx, sy) < 3.0 else 32))
    if pml_cells < 2:
        raise ValueError("pml_cells must be at least 2")

    # Boundaries
    if dimension == "2d":
        boundaries = spec.get("boundaries") or {
            "x_min": {"kind": "pml"},
            "x_max": {"kind": "pml"},
            "y_min": {"kind": "pml"},
            "y_max": {"kind": "pml"},
        }
    else:
        boundaries = spec.get("boundaries") or {
            "x_min": {"kind": "pml"},
            "x_max": {"kind": "pml"},
            "y_min": {"kind": "pml"},
            "y_max": {"kind": "pml"},
            "z_min": {"kind": "pml"},
            "z_max": {"kind": "pml"},
        }

    region = {
        "dimension": dimension,
        "size": [round(sx, 6), round(sy, 6), round(sz, 6)],
        "mesh": round(mesh, 6),
        "mesh_steps": [round(dx, 6), round(dy, 6), round(sz if dimension == "2d" else dz, 6)],
        "material_sampling": "yee",
        "interface_method": "staircase",
        "pml_cells": pml_cells,
        "precision": spec.get("precision", "float64"),
        "backend": spec.get("backend", "cuda"),
        "cuda_kernel": "fused",
        "boundaries": boundaries,
        "steps": steps,
    }

    # Materials
    materials = list(spec.get("materials") or [
        {"name": "si", "index": 3.477, "model": "dielectric"},
        {"name": "sio2", "index": 1.444, "model": "dielectric"},
    ])

    # Structures
    structures = list(spec.get("structures") or [])
    if "grating_teeth" in spec and isinstance(spec["grating_teeth"], dict):
        gt = spec["grating_teeth"]
        periods = int(gt.get("periods", 8))
        period_um = float(gt.get("period_um", 0.9))
        tooth_w = float(gt.get("tooth_width_um", 0.387))
        tooth_h = float(gt.get("tooth_height_um", 0.22))
        start_x = float(gt.get("start_x_um", 0.0))
        base_y = float(gt.get("base_y_um", 0.0))
        mat = str(gt.get("material", "si"))
        for i in range(periods):
            tx = start_x + i * period_um + tooth_w / 2.0
            ty = base_y + tooth_h / 2.0
            structures.append({
                "name": f"tooth_{i+1}",
                "kind": "rectangle",
                "center": [round(tx, 6), round(ty, 6), 0.0],
                "size": [round(tooth_w, 6), round(tooth_h, 6), round(sz, 6)],
                "material": mat,
            })

    # Sources
    sources = []
    raw_sources = spec.get("sources") or (spec.get("source") and [spec["source"]]) or []
    if raw_sources:
        for s in raw_sources:
            src_entry = dict(s)
            src_entry.setdefault("name", "source")
            src_entry.setdefault("kind", "plane")
            src_entry.setdefault("normal", "x")
            src_entry.setdefault("direction", "+")
            src_entry.setdefault("component", "Ey")
            src_entry.setdefault("wavelength", wavelength_um)
            src_entry.setdefault("pulse_cycles", 2)
            sources.append(src_entry)
    else:
        # Default interior source plane inside Poynting enclosure
        pml_pad_x = pml_cells * dx
        pml_pad_y = pml_cells * dy
        b_sx = max(0.2, sx - 2 * pml_pad_x - 0.4)
        b_sy = max(0.2, sy - 2 * pml_pad_y - 0.4)
        src_x = cx - b_sx / 4.0
        src_w = b_sy * 0.8
        sources.append({
            "name": "source",
            "kind": "plane",
            "normal": "x",
            "direction": "+",
            "component": "Ey",
            "center": [round(src_x, 6), round(cy, 6), round(cz, 6)],
            "size": [0.0, round(max(0.04, src_w), 6), round(sz, 6)],
            "wavelength": wavelength_um,
            "time_definition": "standard",
            "pulse_length": 2.0e-15,
            "pulse_offset": 5.0e-15,
        })

    # Monitors
    monitors = list(spec.get("monitors") or [])
    has_box = any("box" in m.get("name", "") for m in monitors)
    if "poynting_box" in spec and isinstance(spec["poynting_box"], dict):
        pb = spec["poynting_box"]
        b_center = pb.get("center", [cx, cy, cz])
        b_size = pb.get("size", [sx * 0.7, sy * 0.7, sz])
        monitors.extend(synthesize_poynting_box(dimension, b_center, b_size, wavelength_um))
    elif not has_box:
        # Auto-synthesize default 4-surface Poynting box for automated V2 conservation
        pml_pad_x = pml_cells * dx
        pml_pad_y = pml_cells * dy
        b_sx = max(0.2, sx - 2 * pml_pad_x - 0.4)
        b_sy = max(0.2, sy - 2 * pml_pad_y - 0.4)
        monitors.extend(synthesize_poynting_box(dimension, [cx, cy, cz], [b_sx, b_sy, sz], wavelength_um))

    for m in monitors:
        if "id" not in m:
            m["id"] = m.get("name", "mon")

    project_dict = {
        "schema_version": 1,
        "name": project_id,
        "region": region,
        "materials": materials,
        "structures": structures,
        "sources": sources,
        "monitors": monitors,
    }

    # Deterministic canonical hash
    canonical_repr = json.dumps(project_dict, sort_keys=True, separators=(",", ":"))
    canonical_sha256 = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()

    return project_id, project_dict, canonical_sha256


def create_project(project_spec: Optional[Dict[str, Any]] = None, root: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
    """Declaratively build and serialize an optical simulation project.

    Writes to configs/projects/<project_id>.json and returns project manifest.
    """
    root = (root or paths.project_root()).resolve()
    merged = dict(project_spec or {})
    merged.update(kwargs)

    # Check for empty or None project_id when explicitly provided
    if "project_id" in merged:
        val = merged["project_id"]
        if val is None or not str(val).strip():
            return {
                "ok": False,
                "kind": "invalid_scene_specification",
                "errors": ["project_id cannot be empty or None"],
            }

    try:
        project_id, project_dict, canonical_sha256 = parse_and_build_project(merged)
    except Exception as exc:
        return {
            "ok": False,
            "kind": "invalid_scene_specification",
            "errors": [str(exc)],
        }

    if ".." in project_id or "/" in project_id or "\\" in project_id:
        return {
            "ok": False,
            "kind": "invalid_scene_specification",
            "errors": [f"Invalid characters in project_id (path traversal prohibited): {project_id}"],
        }

    projects_dir = (root / "configs" / "projects").resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)
    config_file = (projects_dir / f"{project_id}.json").resolve()

    if not paths.inside_root(config_file, projects_dir):
        return {
            "ok": False,
            "kind": "invalid_scene_specification",
            "errors": [f"Project config path escapes configs/projects directory: {project_id}"],
        }

    task_desc = merged.get("task") or merged.get("title") or f"Declarative optical simulation: {project_id}"
    raw_kind = str(merged.get("kind", "physics")).lower()
    if raw_kind not in ("canary", "benchmark", "alignment", "audit", "physics", "speed", "survey"):
        raw_kind = "physics"

    decay_floor = float(merged.get("decay_floor", 0.1))
    tau_box = float(merged.get("tau_box", 1e-5))
    tau_pml = float(merged.get("tau_pml", 1e-5))
    poynting_box_enabled = bool(merged.get("poynting_box", True))

    config_content: Dict[str, Any] = {
        "kind": raw_kind,
        "task": task_desc,
        "policy": "declarative-scene-v2",
        "driver": "src/declarative_runner.py",
        "project": project_dict,
        "poynting_box": poynting_box_enabled,
        "decay_floor": decay_floor,
        "tau_box": tau_box,
        "tau_pml": tau_pml,
        "post_notary": {
            "tau_box": tau_box,
            "tau_pml": tau_pml,
            "decay_floor": decay_floor,
        },
    }

    if "incident_flux" in merged:
        config_content["incident_flux"] = float(merged["incident_flux"])
    elif "incident_power" in merged:
        config_content["incident_power"] = float(merged["incident_power"])

    if "stop_at_step" in merged:
        config_content["stop_at_step"] = int(merged["stop_at_step"])
    if "max_steps" in merged:
        config_content["max_steps"] = int(merged["max_steps"])
    if "resume_stop_at_step" in merged:
        config_content["resume_stop_at_step"] = int(merged["resume_stop_at_step"])

    config_rel = paths.relative_to_root(config_file, root)
    try:
        config_file.write_text(json.dumps(config_content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        return {
            "ok": False,
            "kind": "io_error",
            "errors": [f"Failed to write project config {config_rel}: {exc}"],
        }

    summary = {
        "dimension": project_dict["region"]["dimension"],
        "size": project_dict["region"]["size"],
        "mesh": project_dict["region"]["mesh"],
        "steps": project_dict["region"]["steps"],
        "structures_count": len(project_dict["structures"]),
        "sources_count": len(project_dict["sources"]),
        "monitors_count": len(project_dict["monitors"]),
    }

    return {
        "ok": True,
        "project_id": project_id,
        "config_path": config_rel,
        "project_canonical_sha256": canonical_sha256,
        "summary": summary,
    }


def validate_project(
    project: Optional[Dict[str, Any]] = None,
    config: Optional[str] = None,
    config_path: Optional[str] = None,
    check_environment: bool = True,
    root: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Validate proposed declarative project structure against physics gates and schema."""
    root = (root or paths.project_root()).resolve()
    target_config = config or config_path or kwargs.get("config_file")

    project_dict = None
    if target_config:
        resolved_path, err = paths.resolve_config(target_config, root)
        if err:
            return {
                "ok": False,
                "kind": "invalid_project",
                "errors": [f"Cannot resolve config file: {err}"],
            }
        try:
            raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
            project_dict = raw_data.get("project", raw_data)
        except Exception as exc:
            return {
                "ok": False,
                "kind": "invalid_project",
                "errors": [f"Failed to read project config: {exc}"],
            }
    elif isinstance(project, dict):
        project_dict = project.get("project", project)
    elif kwargs:
        # Attempt building from keyword arguments
        try:
            _, project_dict, _ = parse_and_build_project(kwargs)
        except Exception as exc:
            return {
                "ok": False,
                "kind": "validation_failed",
                "errors": [str(exc)],
            }

    if not isinstance(project_dict, dict):
        return {
            "ok": False,
            "kind": "invalid_project",
            "errors": ["No valid project dictionary or configuration supplied"],
        }

    region = project_dict.get("region", {})
    size = region.get("size")
    mesh = region.get("mesh")
    dimension = region.get("dimension", "2d")

    errors = []
    warnings = []

    if not size or not isinstance(size, (list, tuple)) or len(size) < 2:
        errors.append("region.size must be a list/tuple of at least 2 dimensions")
    elif any(s <= 0 for s in size):
        errors.append(f"All region.size dimensions must be > 0, got {size}")

    if not mesh or float(mesh) <= 0:
        errors.append(f"region.mesh must be > 0, got {mesh}")

    if errors:
        return {
            "ok": False,
            "kind": "validation_failed",
            "errors": errors,
            "warnings": warnings,
        }

    sx, sy = float(size[0]), float(size[1])
    sz = float(size[2]) if len(size) > 2 else 0.04
    m = float(mesh)

    nx = math.ceil(sx / m)
    ny = math.ceil(sy / m)
    nz = 1 if dimension == "2d" else math.ceil(sz / m)
    total_cells = nx * ny * nz

    if total_cells > MAX_RESIDENT_CELLS:
        errors.append(
            f"Total cells {total_cells} exceeds resident GPU ceiling ({MAX_RESIDENT_CELLS}). "
            "Please increase mesh spacing or reduce simulation physical spans."
        )

    # CFL stability limit
    cfl_dim = 2 if dimension == "2d" else 3
    dt_s = (m * 1e-6) / (C0 * math.sqrt(cfl_dim))
    dt_fs = round(dt_s * 1e15, 4)

    steps = int(region.get("steps", 200))

    # Memory estimate (MB): 6 field components * 8 bytes * 2 buffers
    mem_mb = (total_cells * 6 * 8 * 2) / (1024 * 1024)

    # Check Poynting box enclosure closure
    monitors = project_dict.get("monitors", [])
    box_mon_names = {m.get("name") for m in monitors if isinstance(m, dict)}
    required_box = {"box-right", "box-left", "box-top", "box-bottom"}
    poynting_box_closed = required_box.issubset(box_mon_names) or {"box_right", "box_left", "box_top", "box_bottom"}.issubset(box_mon_names)

    canonical_repr = json.dumps(project_dict, sort_keys=True, separators=(",", ":"))
    canonical_sha256 = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()

    # Optional Pydantic model validation if solver fork is present in path
    try:
        from torchfdtd.models import Project
        Project.model_validate(project_dict)
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"Pydantic model validation failure: {exc}")

    if errors:
        return {
            "ok": False,
            "kind": "validation_failed",
            "errors": errors,
            "warnings": warnings,
        }

    return {
        "ok": True,
        "errors": [],
        "warnings": warnings,
        "project_canonical_sha256": canonical_sha256,
        "grid_shape": [nx, ny, nz],
        "total_cells": total_cells,
        "dt_fs": dt_fs,
        "total_steps": steps,
        "poynting_box_closed": poynting_box_closed,
        "estimated_memory_mb": round(mem_mb, 2),
    }
