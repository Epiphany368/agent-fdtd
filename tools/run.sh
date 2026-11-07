#!/usr/bin/env bash
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   tools/run.sh
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = f1feb078fae665f07af4a692db3524da7300c14d677c9b471d15713946cfd12a
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
#!/usr/bin/env bash
# Single simulation entry: preflight, lock, snapshot, effective config, status.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 2 ]; then echo 'usage: run.sh <config> <python-script> [args...]' >&2; exit 64; fi
if [ -z "${PY:-}" ]; then
    if [ -x "$PWD/.venv/bin/python" ]; then
        PY="$PWD/.venv/bin/python"
    elif [ -x "$PWD/.venv/Scripts/python.exe" ]; then
        PY="$PWD/.venv/Scripts/python.exe"
    else
        PY=$(command -v python3 || command -v python || echo "$PWD/.venv/bin/python")
    fi
fi
[ -x "$PY" ] || { echo "Python unavailable: $PY" >&2; exit 69; }
if [ ! -e "$PWD/src/torchfdtd" ] && [ -d "$PWD/vendor/torchfdtd/torchfdtd" ] && [ -f "$PWD/tools/setup_agent.sh" ]; then
    PY="$PY" bash "$PWD/tools/setup_agent.sh" --skip-pip --quiet || true
fi
exec "$PY" -B -u tools/launch.py "$@"
