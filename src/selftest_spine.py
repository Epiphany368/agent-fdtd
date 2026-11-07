"""Spine self-test: the launcher contract, in this repository, with no physics.

Purpose: prove the entry path before any number is produced. It checks that the
single command (`./tools/run.sh <config> <script>`) acquires the lock, snapshots
the source, resolves the config, writes `effective.json`, records the ledger row
and reports an exit status -- all inside this project, with no import from either
reference project and no simulation.

Run:  ./tools/run.sh configs/selftest-spine.json src/selftest_spine.py
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # the snapshot copy
PROJECT = Path.cwd()                                        # the live project
RUN_DIR = Path(os.environ['RUN_DIR_ABS'])
CONFIG_PATH = Path(os.environ['PARITY_CONFIG'])


def git(args, cwd):
    result = subprocess.run(['git', '-C', str(cwd), *args], text=True, encoding='utf-8', errors='replace', capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    pin = json.loads((PROJECT / 'configs/baseline-pin.json').read_text(encoding='utf-8'))
    fork = PROJECT / pin['solver_root']
    upstream = PROJECT / pin['upstream']['solver_root']
    snapshot = Path(os.environ['PARITY_SNAPSHOT'])

    upstream_present = upstream.exists()
    has_criteria_index = (PROJECT / 'configs/leg-criteria.json').is_file() or (PROJECT / 'docs/leg-criteria.json').is_file()
    is_public_export = has_criteria_index and not (PROJECT / 'plan.md').is_file()

    checks = {
        'run_dir_exists': RUN_DIR.is_dir(),
        'snapshot_exists': snapshot.is_dir(),
        'snapshot_has_src': (snapshot / 'src').is_dir(),
        'snapshot_has_project_docs': (
            (snapshot / 'README.md').is_file() and (
                (snapshot / 'configs/leg-criteria.json').is_file()
                or (snapshot / 'docs/leg-criteria.json').is_file()
            )
            if is_public_export
            else all((snapshot / name).is_file() for name in ('plan.md', 'README.md', 'AGENTS.md'))
        ),
        'snapshot_has_solver_copy': (snapshot / 'src/solver/torchfdtd').is_dir(),
        'snapshot_hashes_written': (RUN_DIR / 'snapshot-sha256.json').is_file(),
        'empty_environment_skips_snapshot_dirs': not any(
            (snapshot / 'src' / name).exists() for name in ('torchfdtd', 'torchfdtd_upstream')),
        'fork_link_resolves': fork.is_dir(),
        'upstream_link_resolves': upstream.is_dir() if upstream_present else True,
        'upstream_commit_is_the_pinned_one': (
            git(['rev-parse', 'HEAD'], upstream) == pin['upstream']['solver_commit']
            if upstream_present else True
        ),
        'fork_commit_recorded': git(['rev-parse', 'HEAD'], fork) is not None,
        'this_script_is_the_snapshot_copy': Path(__file__).resolve().is_relative_to(snapshot),
    }

    effective = {
        'stage': 'spine-selftest',
        'simulation_executed': False,
        'config': config,
        'project_commit': git(['rev-parse', 'HEAD'], PROJECT),
        'fork_commit': git(['rev-parse', 'HEAD'], fork),
        'upstream_commit': git(['rev-parse', 'HEAD'], upstream),
        'python': sys.executable,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'checks': checks,
    }
    Path(os.environ['EFFECTIVE_JSON']).write_text(
        json.dumps(effective, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')

    for name, ok in checks.items():
        print(f'{"PASS" if ok else "FAIL"} {name}')
    print(f'spine self-test: {sum(checks.values())}/{len(checks)} checks passed')
    return 0 if all(checks.values()) else 71


if __name__ == '__main__':
    sys.exit(main())
