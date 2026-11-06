#!/usr/bin/env bash
# One-command Agent & Developer bootstrap for torchfdtd-platform.
#
# Sets up the Python environment, initializes/links the TorchFDTD solver into
# `src/torchfdtd` (from bundled `vendor/torchfdtd`, an explicit path argument,
# `SOLVER_FORK`, or by cloning upstream), installs the `torchfdtd-platform-mcp`
# MCP server entry point, and runs the spine verification check.
#
# USAGE
#   ./tools/setup_agent.sh                        # auto-detect bundled vendor/torchfdtd or existing link
#   ./tools/setup_agent.sh /path/to/torchfdtd     # link an external TorchFDTD checkout
#   ./tools/setup_agent.sh --skip-pip             # link solver only (reuse existing Python environment)

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 65

SKIP_PIP=0
QUIET=0
SOLVER_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-pip) SKIP_PIP=1; shift ;;
        --quiet|-q) QUIET=1; shift ;;
        -h|--help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) SOLVER_ARG="$1"; shift ;;
    esac
done

log() {
    [ "$QUIET" -eq 0 ] && echo "[setup_agent] $*"
}

log "bootstrapping torchfdtd-platform in $ROOT"

# 1. Locate or create Python environment (.venv)
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
    log "using existing virtual environment: $PY"
elif [ "$SKIP_PIP" -eq 1 ]; then
    PY=$(command -v python3 || command -v python || true)
    [ -n "$PY" ] || { echo "[setup_agent] ERROR: python3 not found" >&2; exit 69; }
else
    SYS_PY=$(command -v python3 || command -v python || true)
    [ -n "$SYS_PY" ] || { echo "[setup_agent] ERROR: python3 not found" >&2; exit 69; }
    log "creating .venv with $SYS_PY"
    "$SYS_PY" -m venv "$ROOT/.venv" || exit 65
    PY="$ROOT/.venv/bin/python"
fi

if [ "$SKIP_PIP" -eq 0 ]; then
    log "installing dependencies and torchfdtd-platform-mcp entry point..."
    "$PY" -m pip install --quiet -r "$ROOT/requirements.txt" || exit 65
    "$PY" -m pip install --quiet --no-build-isolation --no-deps -e "$ROOT" 2>/dev/null \
        || "$PY" -m pip install --quiet -e "$ROOT" || true
fi

# 2. Resolve and initialize the solver at src/torchfdtd
ensure_independent_git_repo() {
    local target="$1"
    local real_target
    real_target=$(cd "$target" && pwd -P)
    local top
    top=$(git -C "$real_target" rev-parse --show-toplevel 2>/dev/null || true)
    if [ "$top" != "$real_target" ]; then
        log "initializing independent git repository in $real_target"
        git -C "$real_target" init -q -b main
        git -C "$real_target" add -A
        GIT_AUTHOR_NAME="TorchFDTD Bundled" GIT_AUTHOR_EMAIL="noreply@example.com" \
        GIT_COMMITTER_NAME="TorchFDTD Bundled" GIT_COMMITTER_EMAIL="noreply@example.com" \
            git -C "$real_target" commit -q -m "Bundled TorchFDTD solver (v1.1.7 + platform fork)"
    fi
}

if [ -n "$SOLVER_ARG" ]; then
    ensure_independent_git_repo "$SOLVER_ARG"
    PY="$PY" bash "$ROOT/tools/bootstrap_solver.sh" "$SOLVER_ARG" >/dev/null || exit 65
elif [ -d "$ROOT/src/torchfdtd/torchfdtd" ]; then
    ensure_independent_git_repo "$ROOT/src/torchfdtd"
    log "solver already linked at src/torchfdtd"
elif [ -d "$ROOT/vendor/torchfdtd/torchfdtd" ]; then
    SOLVER_RUNTIME="$ROOT/.solver/torchfdtd"
    if [ ! -d "$SOLVER_RUNTIME/torchfdtd" ]; then
        log "materializing bundled vendor/torchfdtd into .solver/torchfdtd..."
        rm -rf "$SOLVER_RUNTIME"
        mkdir -p "$ROOT/.solver"
        cp -a "$ROOT/vendor/torchfdtd" "$SOLVER_RUNTIME"
    fi
    ensure_independent_git_repo "$SOLVER_RUNTIME"
    PY="$PY" bash "$ROOT/tools/bootstrap_solver.sh" "$SOLVER_RUNTIME" >/dev/null || exit 65
elif [ -n "${SOLVER_FORK:-}" ] && [ -d "${SOLVER_FORK}/torchfdtd" ]; then
    ensure_independent_git_repo "$SOLVER_FORK"
    PY="$PY" bash "$ROOT/tools/bootstrap_solver.sh" "$SOLVER_FORK" >/dev/null || exit 65
else
    log "no local solver found; cloning upstream hyoseokp/TorchFDTD into .solver/torchfdtd..."
    mkdir -p "$ROOT/.solver"
    git clone --depth 1 https://github.com/hyoseokp/TorchFDTD.git "$ROOT/.solver/torchfdtd" || {
        echo "[setup_agent] ERROR: failed to clone TorchFDTD; pass a local checkout path to ./tools/setup_agent.sh <path>" >&2
        exit 65
    }
    PY="$PY" bash "$ROOT/tools/bootstrap_solver.sh" "$ROOT/.solver/torchfdtd" >/dev/null || exit 65
fi

# 3. Verify spine self-test (unless running in quiet auto-bootstrap mode)
if [ "$QUIET" -eq 0 ]; then
    log "running spine verification (configs/selftest-spine.json)..."
    PY="$PY" bash "$ROOT/tools/run.sh" configs/selftest-spine.json src/selftest_spine.py || exit $?
    log "READY: torchfdtd-platform and MCP server are configured and verified."
fi
