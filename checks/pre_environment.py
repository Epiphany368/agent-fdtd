# ---------------------------------------------------------------------------
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   checks/pre_environment.py
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = dbe73f79f6afd1ed3d12424ca82bd5d394e1956d893beb5325fc73e8ebf8213a
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
"""Pin and launch-config gate, before acquiring the resource lock.

Two roles, two rules (plan.md v2.0, 2026-10-12).

UPSTREAM is FROZEN PROVENANCE. It must stay clean and must stay at the commit every
result registered before plan v2.0 resolves to. If it moves, that is a hard failure:
the history it anchors stops being checkable.

THE FORK is OUR WORKING SOLVER. It is allowed -- expected -- to move, so it is
*recorded* rather than *forbidden*. The reproducibility that demanding a clean pin
used to provide is provided instead by the run snapshot: launch.py copies the solver
source into runs/<id>/snapshot/ and writes the commit and dirty state into
effective.json. A dirty fork is therefore fine; an *unrecorded* one is not.

Reproducing a result registered before v2.0 is still possible and is what
PARITY_SOLVER_ROLE=upstream selects.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM_COMMIT_EXPECTED = 'ebcc0f7d1361c29cb478dbd2a478fb4c58654223'

ROOT = Path(__file__).resolve().parents[1]


# CHANGED from the copy: solver roots in `configs/baseline-pin.json` are
# repository-relative, so no machine-specific absolute path is tracked here.
# Set PARITY_SOLVER_ROOT (fork) or PARITY_SOLVER_UPSTREAM to override.
def resolve(value, env_name):
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def git(solver, *args):
    return subprocess.check_output(['git', '-C', str(solver), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def check_solver_git(solver_path):
    """Verify that solver_path is an independent git repository root.

    Returns dict(commit=..., dirty_lines=..., toplevel=...).
    Raises RuntimeError if solver_path lacks an independent git repository root
    (e.g. upward traversal to parent repository, missing .git, or invalid path).
    """
    solver = Path(solver_path).resolve()
    try:
        top = git(solver, 'rev-parse', '--show-toplevel')
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f'{solver} is not a git repository ({error})') from error
    if Path(top).resolve() != solver:
        raise RuntimeError(f'Git upward traversal detected: {solver} is inside {top}')
    commit = git(solver, 'rev-parse', 'HEAD')
    dirty = git(solver, 'status', '--porcelain')
    changed = len([line for line in dirty.splitlines() if line.strip()])
    return {'commit': commit, 'dirty_lines': changed, 'toplevel': str(Path(top).resolve())}


def check_upstream(pin, required=True):
    """The frozen provenance invariant. Never weakened when the baseline is present.

    A clone that has only the fork link has no frozen baseline. That is absence,
    not movement: a read-only audit does not read the baseline and may continue,
    while a run that measures something refuses here -- the same refusal a genuine
    missing baseline has always deserved, but as a declared message and exit 65
    rather than a ``CalledProcessError`` traceback.
    """
    up = resolve(pin['upstream']['solver_root'], 'PARITY_SOLVER_UPSTREAM')
    unavailable = None
    try:
        info = check_solver_git(up)
        commit = info['commit']
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        # Do not pre-check the path: the CPU gate contracts drive this function with
        # a mocked ``git`` and a path that need not exist on disk.
        unavailable = f'{up} is not a valid git checkout ({error})'
    else:
        assert commit == up_expected(pin), (
            f'FROZEN UPSTREAM MOVED: {commit[:12]} != {up_expected(pin)[:12]}. '
            'This anchors every result registered before plan v2.0. Restore it; '
            'do not update the pin to match.')
        assert info['dirty_lines'] == 0, 'Frozen upstream is dirty; restore it before running anything'
        return commit
    if required:
        print(f'frozen upstream baseline is unavailable: {unavailable}. This run needs the '
              f'pinned provenance {up_expected(pin)[:12]} from configs/baseline-pin.json '
              'upstream.solver_root (or PARITY_SOLVER_UPSTREAM); link it with '
              'tools/bootstrap_solver.sh <upstream checkout> --upstream; no simulation was run',
              file=sys.stderr)
        raise SystemExit(65)
    print(f'preflight: frozen upstream baseline unavailable ({unavailable}); '
          'read-only audit does not read it', file=sys.stderr)
    return None


def up_expected(pin):
    # The pin may carry the commit under either key depending on how it was written.
    return (pin.get('upstream', {}).get('solver_commit')
            or pin.get('solver_commit_at_split')
            or UPSTREAM_COMMIT_EXPECTED)


def check_fork(pin):
    """Record, do not forbid. Returns the commit plus a dirty summary."""
    fork = resolve(pin['solver_root'], 'PARITY_SOLVER_ROOT')
    try:
        info = check_solver_git(fork)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'solver fork is unavailable: {fork} is not a valid git checkout ({error}); '
              'link it with tools/bootstrap_solver.sh <torchfdtd checkout>; '
              'no simulation was run', file=sys.stderr)
        raise SystemExit(65)
    return info['commit'], info['dirty_lines']


if 'PARITY_CONFIG' not in os.environ:
    if __name__ == '__main__':
        print('pre_environment: PARITY_CONFIG environment variable required', file=sys.stderr)
        raise SystemExit(64)
else:
    root = ROOT
    pin = json.loads((root / 'configs/baseline-pin.json').read_text())
    config = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())

    assert config.get('task'), 'Every run needs a step number in task'
    assert config.get('kind') in ('canary', 'benchmark', 'alignment', 'audit',
                                  'physics', 'speed', 'survey'), 'Unknown task kind'

    # The reference-contract gate answers from the config alone, so it runs before any
    # provenance or solver lookup. A clone without the frozen baseline must still reach
    # this refusal -- the declared answer to "this config needs a contract that is not
    # here" -- instead of dying on a missing machine-local solver link on the way.
    if config['kind'] == 'alignment' or config.get('reference'):
        required_by = []
        if config['kind'] == 'alignment':
            required_by.append('kind: alignment')
        if config.get('reference'):
            required_by.append('config field `reference`')
        required_by = ' and '.join(required_by)
        # A missing contract module is its own refusal. It must not escape as a bare
        # ModuleNotFoundError; it must name the module, the field that required it and
        # state that nothing ran, and it must exit 65. A ModuleNotFoundError raised
        # *inside* an existing reference_contract is a genuine contract error: re-raise
        # it rather than masking it as a missing module.
        try:
            from reference_contract import load_reference
        except ModuleNotFoundError as error:
            if error.name != 'reference_contract':
                raise
            print(f'reference contract module is missing: cannot import reference_contract '
                  f'(required by {required_by}); no simulation was run', file=sys.stderr)
            raise SystemExit(65)
        family = config.get('artifact_family', 'staircase')
        try:
            load_reference(root / config['reference'], family)
        except (FileNotFoundError, ValueError, KeyError, TypeError, IndexError) as error:
            raise SystemExit(f'{family} reference unavailable/invalid: {error}; '
                             'no simulation allowed; see output/lumerical-reference-contract.md')

    role = os.environ.get('PARITY_SOLVER_ROLE', 'fork')

    # The frozen provenance is checked on EVERY run that can observe something: if it has
    # moved, no measurement made now is comparable to anything registered before v2.0.
    # A read-only audit, or a standalone fork clone without the upstream directory on disk,
    # may proceed when role != 'upstream' because it does not read the baseline; that
    # absence is printed, not hidden. When the upstream checkout is present (or explicitly
    # configured via PARITY_SOLVER_UPSTREAM, or selected via role == 'upstream'), any
    # movement, dirtiness, or invalid git root still refuses.
    upstream_present = bool(os.environ.get('PARITY_SOLVER_UPSTREAM')) or (
        resolve(pin['upstream']['solver_root'], 'PARITY_SOLVER_UPSTREAM').exists()
    )
    upstream_commit = check_upstream(
        pin, required=(config['kind'] != 'audit' and upstream_present) or role == 'upstream')

    if role == 'upstream':
        solver_commit, solver_dirty = upstream_commit, 0
        solver_label = 'upstream (frozen, reproducing pre-v2.0 evidence)'
    else:
        solver_commit, solver_dirty = check_fork(pin)
        solver_label = f'fork (dirty files: {solver_dirty})'

    upstream_label = upstream_commit[:12] if upstream_commit else 'absent'

    import torch
    if config['kind'] != 'audit':
        allow_cpu = (
            config.get('policy') == 'declarative-scene-v2'
            or os.environ.get('TORCHFDTD_ALLOW_CPU') == '1'
        )
        if not allow_cpu:
            assert torch.cuda.is_available(), 'Pinned GPU chain unavailable'
        gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (fallback)'
        print(f'preflight: role={role} solver={solver_commit[:12]} upstream={upstream_label} '
              f'GPU={gpu_label}')
        print(f'preflight: {solver_label}')
    else:
        print(f'preflight: role={role} solver={solver_commit[:12]} '
              f'upstream={upstream_label} read-only audit (no simulation)')
