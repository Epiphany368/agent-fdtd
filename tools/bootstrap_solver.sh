#!/usr/bin/env bash
# Point this project at a TorchFDTD checkout.
#
# WHY THIS EXISTS. The solver is a separate, MIT-licensed project and is NOT vendored here.
# `tools/launch.py` looks for `src/torchfdtd` (or `src/torchfdtd_upstream`), and in the
# working repository those are symlinks to a machine-specific path. A fresh clone therefore
# has no engine and every run refuses at pre-flight. This script creates the link from a path
# you supply, so the dependency is explicit and reproducible instead of assumed.
#
# USAGE
#   tools/bootstrap_solver.sh /path/to/torchfdtd          # fork role (the default)
#   tools/bootstrap_solver.sh /path/to/upstream --upstream
#   SOLVER_FORK=/path/to/torchfdtd tools/bootstrap_solver.sh
#
# EXIT STATUS: 0 = link created and importable; 64 = usage; 65 = pre-flight refusal
# (path missing, or not a TorchFDTD checkout, or the interpreter cannot import it).

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC="${1:-${SOLVER_FORK:-}}"
ROLE="fork"
LINK="$ROOT/src/torchfdtd"

if [ "${2:-}" = "--upstream" ] || [ "${1:-}" = "--upstream" ]; then
    ROLE="upstream"
    LINK="$ROOT/src/torchfdtd_upstream"
fi

if [ -z "$SRC" ] || [ "$SRC" = "--upstream" ]; then
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 64
fi

[ -d "$SRC" ] || { echo "pre-flight: no such directory: $SRC" >&2; exit 65; }
[ -f "$SRC/torchfdtd/__init__.py" ] || {
    echo "pre-flight: $SRC does not look like a TorchFDTD checkout (no torchfdtd/__init__.py)" >&2
    exit 65
}

REAL=$(cd "$SRC" && pwd -P)
ln -sfn "$REAL" "$LINK"
echo "solver ($ROLE): $LINK -> $REAL"

if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "pre-flight: no python available to check the import" >&2
    exit 65
fi

"$PY" - <<'PY' || exit 65
import sys
sys.path.insert(0, 'src')
try:
    import torchfdtd
except Exception as error:                                  # noqa: BLE001 - reported, not hidden
    raise SystemExit(f'pre-flight: torchfdtd did not import from the link: {error}')
print(f'import check: torchfdtd {getattr(torchfdtd, "__version__", "(no __version__)")} from {torchfdtd.__file__}')
PY
