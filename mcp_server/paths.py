"""Project-root and path resolution for the MCP server.

The server is normally installed from the project root (``pip install -e .``).
Tests and isolated deployments may point it at another checkout with
``TORCHFDTD_PLATFORM_ROOT``; the override changes only where the tools read and
where ``tools/run.sh`` is executed, never the shape of the tools.
"""
import json
import os
from pathlib import Path


ROOT_ENV = "TORCHFDTD_PLATFORM_ROOT"


def project_root():
    """Return the platform root, without importing the platform or the solver."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [here.parents[1], Path.cwd()]
    for candidate in candidates:
        if (candidate / "tools" / "run.sh").is_file() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return here.parents[1].resolve()


def inside_root(path, root=None):
    root = (root or project_root()).resolve()
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def resolve_config(name, root=None):
    """Resolve a config name or path to an existing file inside the project.

    ``name`` may be ``configs/foo.json``, ``foo.json`` or an absolute path.
    Returns ``(path, error)``; a missing file is a structured error, not an
    attempt to guess a location.
    """
    root = (root or project_root()).resolve()
    if isinstance(name, os.PathLike):
        name = os.fspath(name)
    if not isinstance(name, str) or not name.strip():
        return None, "config must be a non-empty path or file name"
    supplied = Path(name).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [root / supplied, root / "configs" / supplied]
    for candidate in candidates:
        if candidate.is_file():
            path = candidate.resolve()
            if not inside_root(path, root):
                return None, f"config is outside the project root: {name}"
            return path, None
    return None, f"config not found under the project root: {name}"


def relative_to_root(path, root=None):
    root = (root or project_root()).resolve()
    return Path(path).resolve().relative_to(root).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
