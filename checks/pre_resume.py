"""Preflight gate validating resume journal contract and fingerprints (LEG 45).

Zero-Trust Rule: Agents possess propose rights, never dispose rights. A resumed
simulation must adhere to the exact immutable contract (epsilon, dt, geometry,
time step) recorded in the checkpoint journal. Carrying forward electromagnetic
fields into a mutated structure is strictly forbidden and results in preflight
rejection with exit code 65.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main() -> int:
    journal_dir_str = os.environ.get('TORCHFDTD_RESUME_JOURNAL')
    if not journal_dir_str or not journal_dir_str.strip():
        # Normal fresh run, no resume contract to validate
        return 0

    journal_dir = Path(journal_dir_str.strip()).resolve()
    if not journal_dir.is_dir():
        print(f"[pre_resume] FAILED: resume journal directory not found: {journal_dir}", file=sys.stderr)
        return 1

    contract_file = journal_dir / 'contract.json'
    if not contract_file.is_file():
        print(f"[pre_resume] FAILED: contract.json missing in journal {journal_dir}", file=sys.stderr)
        return 1

    pointer_file = journal_dir / 'latest-forward.json'
    if not pointer_file.is_file():
        print(f"[pre_resume] FAILED: latest-forward.json missing in journal {journal_dir}", file=sys.stderr)
        return 1

    try:
        contract = json.loads(contract_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[pre_resume] FAILED: unreadable contract.json ({e})", file=sys.stderr)
        return 1

    try:
        pointer = json.loads(pointer_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[pre_resume] FAILED: unreadable latest-forward.json ({e})", file=sys.stderr)
        return 1

    # Check current run config
    run_dir_str = os.environ.get('RUN_DIR_ABS')
    run_dir = Path(run_dir_str) if run_dir_str else Path.cwd()
    config_file = run_dir / 'config.input.json'
    if not config_file.is_file() and 'PARITY_CONFIG' in os.environ:
        config_file = Path(os.environ['PARITY_CONFIG'])

    if config_file.is_file():
        try:
            config = json.loads(config_file.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[pre_resume] FAILED: unreadable config ({e})", file=sys.stderr)
            return 1

        # Check dt / time_step
        cfg_dt = config.get('dt') or config.get('time_step')
        if cfg_dt is not None and 'dt' in contract:
            scale = max(abs(float(cfg_dt)), abs(float(contract['dt'])), 1e-30)
            if abs(float(cfg_dt) - float(contract['dt'])) / scale > 1e-9:
                print(
                    f"[pre_resume] FAILED: dt mismatch: contract={contract['dt']} != config={cfg_dt}",
                    file=sys.stderr,
                )
                return 1

        # Check epsilon_sha256 if present in config
        cfg_eps = config.get('epsilon_sha256')
        if cfg_eps is not None and 'epsilon_sha256' in contract:
            if str(cfg_eps) != str(contract['epsilon_sha256']):
                print(
                    f"[pre_resume] FAILED: epsilon_sha256 mismatch: contract={contract['epsilon_sha256']} != config={cfg_eps}",
                    file=sys.stderr,
                )
                return 1

        # Check grid shape / dimensions if present
        cfg_shape = config.get('grid_shape') or config.get('shape')
        if cfg_shape is not None and 'grid_shape' in contract:
            if list(cfg_shape) != list(contract['grid_shape']):
                print(
                    f"[pre_resume] FAILED: grid_shape mismatch: contract={contract['grid_shape']} != config={cfg_shape}",
                    file=sys.stderr,
                )
                return 1

        # Check project canonical hash if present
        cfg_plan = config.get('project_canonical_sha256')
        if cfg_plan is not None and contract.get('project_canonical_sha256') is not None:
            if str(cfg_plan) != str(contract['project_canonical_sha256']):
                print(
                    f"[pre_resume] FAILED: project_canonical_sha256 mismatch: contract={contract['project_canonical_sha256']} != config={cfg_plan}",
                    file=sys.stderr,
                )
                return 1

    print(f"[pre_resume] OK: journal verified at {journal_dir.name} (step={pointer.get('step')}, block={pointer.get('block')})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
