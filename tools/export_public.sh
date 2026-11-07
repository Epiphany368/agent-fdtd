#!/usr/bin/env bash
# Build a public, publishable tree for this project.
#
# WHY THIS EXISTS. The working repository tracks things that must not be published:
# an internal dispatch ledger with agent transcripts, a steward's control files, review
# records, scratch output, and machine-specific absolute paths. "Which files are public"
# must be one re-runnable command rather than a hand decision made once, because a hand
# decision is re-made wrong later.
#
# WHAT IT PRODUCES. A directory (default `dist/public`) containing only the allowlisted
# paths, with machine-specific absolute paths rewritten to their symbolic equivalents, and
# — with `--git` — a fresh single-commit git repository, so the published history does not
# carry the internal one.
#
# USAGE
#   tools/export_public.sh                         # build dist/public, leave it as plain files
#   tools/export_public.sh --git                   # also initialise a fresh repo with one commit
#   tools/export_public.sh --bundle-solver --git   # bundle merged TorchFDTD in vendor/torchfdtd for zero-config Agent use
#   tools/export_public.sh --dest /tmp/x           # build somewhere else
#
# EXIT STATUS: 0 = export built and verified clean; 1 = a leak was found (build is kept for
# inspection); 64 = usage; 65 = pre-flight refusal (not a git checkout, dirty tree, or the
# criteria index could not be extracted); 69 = python unavailable.

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT/dist/public"
WITH_GIT=0
BUNDLE_SOLVER=0

while [ $# -gt 0 ]; do
    case "$1" in
        --git) WITH_GIT=1; shift ;;
        --bundle-solver) BUNDLE_SOLVER=1; shift ;;
        --dest) DEST="${2:?--dest needs a path}"; shift 2 ;;
        -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

cd "$ROOT" || exit 65
git rev-parse --git-dir >/dev/null 2>&1 || { echo "pre-flight: not a git checkout" >&2; exit 65; }
[ -z "$(git status --porcelain)" ] || { echo "pre-flight: working tree is dirty; export from a clean commit" >&2; exit 65; }

COMMIT=$(git rev-parse --short HEAD)

# The allowlist, and the reason each entry is in it.
#   LICENSE, README.md      required to be publishable at all
#   STATUS.md               the derived index; this is the project's argument
#   PROVENANCE.md           what came from where, including the pinned commits
#   src/ checks/ configs/   the code and its tests
#   tools/                  the launcher, the exporter, the solver bootstrap
#   docs/                   one long-form evidence document per leg
#   output/                 the owner-facing maps and the one-pager
#   runs/ledger.jsonl       the record of what actually ran; no run payloads
INCLUDE=(
    LICENSE
    README.md
    README_zh.md
    SKILL.md
    .mcp.json
    requirements.txt
    pyproject.toml
    STATUS.md
    PROVENANCE.md
    src
    checks
    configs
    tools
    mcp_server
    mcptests
    docs
    output
    runs/ledger.jsonl
)

# Paths deliberately NOT published, with the reason. Printed so the exclusion is auditable.
EXCLUDE_NOTE=(
    "AGENTS.md   engineering-management rules for the agents that build this project"
    "plan.md     the internal contract; its criteria are published as the extracted docs/leg-criteria.json"
    "control/    steward lease, dispatch ledger, wave tables -- internal coordination"
    "dispatch/   agent prompts and replies; the raw record of who asked what"
    "reviews/    internal review records"
    "scratch/    raw machine output"
    "runs/*      run payloads (field arrays, snapshots); only the ledger is published"
    "data/       machine-local lock directory"
    "env/        machine-specific symbol values (already untracked)"
)

# Machine-specific path roots -> their symbolic equivalents.
#
# The table lives in the UNTRACKED file `env/scrub-paths.txt`, one `<prefix><TAB><replacement>`
# per line, most specific first. It is not in this script on purpose: a scrub table written
# here would publish the very paths it exists to remove -- which is exactly what the first
# dry run of this exporter caught, in this file.
SCRUB=()
ROOTS=()
add_rule() { [ -n "${1:-}" ] && { SCRUB+=("s#${1}#${2}#g"); ROOTS+=("$1"); }; }
SCRUB_FILE="$ROOT/env/scrub-paths.txt"
if [ -f "$SCRUB_FILE" ]; then
    while IFS=$'\t' read -r prefix replacement; do
        case "$prefix" in ''|'#'*) continue ;; esac
        add_rule "${prefix%/}" "$replacement"
    done < "$SCRUB_FILE"
else
    echo "export_public: WARNING: no $SCRUB_FILE; only \$HOME will be scrubbed" >&2
fi
# Last, and least specific: the current user's home directory.
add_rule "${HOME%/}" '$HOME_LITERAL'

echo "export_public: from $COMMIT -> $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"

copied=0
for entry in "${INCLUDE[@]}"; do
    if [ ! -e "$entry" ]; then
        echo "  skip (absent): $entry"
        continue
    fi
    mkdir -p "$DEST/$(dirname "$entry")"
    cp -a "$entry" "$DEST/$entry"
    copied=$((copied + 1))
done

# The criteria index. plan.md is private, but a published tree must still answer
# "what was this leg's pre-declared criterion?"  The index is extracted, never
# hand-written; a failed extraction makes the export unpublishable.
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=$(command -v python3)
[ -x "$PY" ] || { echo "export_public: python unavailable; cannot extract docs/leg-criteria.json" >&2; exit 69; }
if ! "$PY" -B "$ROOT/mcp_server/plan.py" "$ROOT/plan.md" "$DEST/docs/leg-criteria.json"; then
    echo "export_public: criteria index extraction failed; the export is not publishable" >&2
    exit 65
fi

# The solver links are absolute symlinks into a machine-specific directory. Copy preserves
# them and `grep -I` skips them, so the leak check cannot see them: remove them explicitly.
SOLVER_REAL=""
if [ -d "$ROOT/src/torchfdtd/torchfdtd" ]; then
    SOLVER_REAL=$(cd "$ROOT/src/torchfdtd" && pwd -P)
fi
rm -f "$DEST/src/torchfdtd" "$DEST/src/torchfdtd_upstream"

# Optionally bundle the MIT-licensed merged TorchFDTD solver into vendor/torchfdtd
# so an AI Agent cloning the exported repo has everything needed out-of-the-box.
if [ "$BUNDLE_SOLVER" -eq 1 ]; then
    [ -n "$SOLVER_REAL" ] && [ -d "$SOLVER_REAL/torchfdtd" ] || {
        echo "export_public: --bundle-solver requested, but src/torchfdtd is unavailable" >&2
        exit 65
    }
    echo "export_public: bundling merged solver from $SOLVER_REAL -> $DEST/vendor/torchfdtd"
    mkdir -p "$DEST/vendor/torchfdtd"
    for item in LICENSE THIRD_PARTY_NOTICES.txt CITATION.cff README.md pyproject.toml torchfdtd examples tests; do
        if [ -e "$SOLVER_REAL/$item" ]; then
            cp -a "$SOLVER_REAL/$item" "$DEST/vendor/torchfdtd/$item"
        fi
    done
fi

cat > "$DEST/.gitignore" <<'EOF'
.venv/
__pycache__/
*.pyc
/.solver/
/src/torchfdtd
/src/torchfdtd_upstream
/runs/*/
/runs/.lock
EOF

# Nothing generated, nothing cached.
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$DEST" -name '*.pyc' -delete 2>/dev/null
find "$DEST" -name '*.egg-info' -type d -prune -exec rm -rf {} + 2>/dev/null
rm -rf "$DEST/.venv" 2>/dev/null

# Scrub. Text files only; a binary that contains a home directory is a finding, not a rewrite.
find_leaks() {   # print files under $DEST containing any scrubbed root
    local root
    for root in "${ROOTS[@]}"; do
        grep -rIl -- "$root" "$DEST" 2>/dev/null
    done | sort -u
}
mapfile -t textfiles < <(find_leaks)
echo "export_public: scrubbing ${#textfiles[@]} text file(s)"
for f in "${textfiles[@]}"; do
    for rule in "${SCRUB[@]}"; do
        sed -i -e "$rule" "$f"
    done
done

# Verify. Any remaining absolute home path is a leak and fails the export.
leaks=$(find_leaks | wc -l)

{
    echo "public export of $COMMIT"
    echo
    echo "included:"
    for entry in "${INCLUDE[@]}"; do echo "  $entry"; done
    echo
    echo "excluded, and why:"
    for note in "${EXCLUDE_NOTE[@]}"; do echo "  $note"; done
} > "$DEST/EXPORT-MANIFEST.txt"

if [ "$WITH_GIT" -eq 1 ]; then
    git -C "$DEST" init -q -b main
    git -C "$DEST" add -A
    GIT_AUTHOR_NAME="torchfdtd-platform" GIT_AUTHOR_EMAIL="noreply@example.com" \
    GIT_COMMITTER_NAME="torchfdtd-platform" GIT_COMMITTER_EMAIL="noreply@example.com" \
        git -C "$DEST" commit -q -m "AgentFDTD 1.0.1: one command, verifiable acceptance

Public tree exported from the working repository at $COMMIT.
The solver is a dependency (MIT-licensed torchfdtd), not bundled."
    echo "export_public: fresh single-commit repo at $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
fi

echo "export_public: $copied top-level entries copied"
if [ "$leaks" -ne 0 ]; then
    echo "export_public: LEAK -- $leaks file(s) still contain a machine-specific root" >&2
    find_leaks | head -20 >&2
    exit 1
fi
echo "export_public: clean -- no machine-specific root remains (${#ROOTS[@]} root(s) checked)"
