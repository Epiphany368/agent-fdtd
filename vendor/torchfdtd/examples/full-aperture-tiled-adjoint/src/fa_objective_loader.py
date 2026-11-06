"""Objective hook of the FA drivers (2026-09-30): obj = fa_objectives.make_objective(name, cfg).

Search order: FA_OBJECTIVES_PY, then fa_objectives.py next to this file.
Its canonical targets are read from FA_TARGETS_DIR (default ../targets relative to this file).
A stub module (fa_objectives_stub.py, not shipped here) is used only with FA_ALLOW_STUB=1 and recorded as stub=True.
cfg passed: lam_nm (ascending), pitch_um, radius_um, and focal_um for e1_achromat; everything else is the module's DEFAULTS.
obj(T) -> 0-dim tensor to maximize, T = dict of complex (Nl, m, m) flux-normalized pupil Jones maps xx, yx[, xy, yy].
obj.metrics(T) -> {'scalars': JSON-safe dict, 'arrays': numpy dict} (or a flat dict), obj.info -> dict.
"""
import hashlib, importlib.util, os
from pathlib import Path

_FILE = {}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, *, lam_nm, pitch_um, aperture_um, focal_um=None, **_ignored):
    here = Path(__file__).resolve().parent
    path = Path(os.environ.get("FA_OBJECTIVES_PY") or here / "fa_objectives.py"); stub = False
    if not path.exists():
        if os.environ.get("FA_ALLOW_STUB", "0") != "1":
            raise SystemExit(f"FA objective module {path} not found (production refuses the stub; FA_ALLOW_STUB=1 for wiring tests only)")
        path = here / "fa_objectives_stub.py"; stub = True
    spec = importlib.util.spec_from_file_location("fa_objectives", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tdir = Path(os.environ.get("FA_TARGETS_DIR", here.parent / "targets"))
    if hasattr(mod, "TARGETS") and tdir.is_dir(): mod.TARGETS = tdir                   # the module reads TARGETS at call time
    cfg = dict(lam_nm=[int(v) for v in lam_nm], pitch_um=float(pitch_um), radius_um=float(aperture_um) / 2)
    if name == "e1_achromat" and focal_um is not None: cfg["focal_um"] = float(focal_um)
    if os.environ.get("FA_OBJ_CFG"):                                     # 2026-09-30: extra objective cfg (JSON), e.g. {"z_um": 300, "aperture": "square"}
        import json; cfg.update(json.loads(os.environ["FA_OBJ_CFG"]))
    if stub: cfg.update(pupil_n=None, aperture_um=float(aperture_um), focal_um=float(focal_um or aperture_um / 0.6))
    obj = mod.make_objective(name, cfg)
    _FILE[id(obj)] = dict(objective_name=name, objective_file=str(path), objective_file_sha256=_sha256(path), stub=stub,
                          targets_dir=str(tdir) if tdir.is_dir() else None, cfg_passed=cfg)
    print(f"[objective] {name} from {path}{' (STUB)' if stub else ''}, targets {tdir}", flush=True)
    return obj


def info(obj):
    """Objective record for the manifest: the module's own info (target sha256, cfg) plus the file that was loaded."""
    d = dict(getattr(obj, "info", None) or {}); d.update(_FILE.get(id(obj), {}))
    return d
