"""Deterministic mock driver for LEG 45 adversarial challenger tests.

Avoids filesystem churn and DrvFs race conditions by providing a single,
immutable driver script controllable via CHALLENGER_ACTION environment variable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

action = os.environ.get('CHALLENGER_ACTION', 'exit_0')
run_dir_str = os.environ.get('RUN_DIR_ABS')
run_dir = Path(run_dir_str) if run_dir_str else Path.cwd()

base_eps_sha256 = os.environ.get('CHALLENGER_EPS_SHA256', 'a' * 64)

if action == 'exit_0':
    sys.exit(0)

elif action == 'exit_42':
    sys.exit(42)

elif action == 'exit_53':
    telemetry = {
        'schema_version': '1.0',
        'project_canonical_sha256': 'a' * 64,
        'epsilon_sha256': base_eps_sha256,
        'is_lossless': True,
        'executed_steps': 500,
        'resumed_from_step': 1984,
        'stop_reason': 'stopped-non-monotone',
    }
    (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry), encoding='utf-8')
    sys.exit(53)

elif action == 'tamper_claim_0':
    # Kernel accurately telemetry reports resumed from 1984
    telemetry = {
        'schema_version': '1.0',
        'project_canonical_sha256': 'a' * 64,
        'epsilon_sha256': base_eps_sha256,
        'is_lossless': True,
        'executed_steps': 2016,
        'resumed_from_step': 1984,
        'energy_decay_ratio': 1e-7,
        'monitors': {'box-right': {'flux': 0.6}, 'box-left': {'flux': 0.1}},
    }
    (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry), encoding='utf-8')

    # Malicious readout claiming fresh run from 0
    readout = {
        'simulation_executed': True,
        'executed_steps': 2016,
        'resumed_from_step': 0,
        'epsilon_sha256': base_eps_sha256,
        'is_lossless': True,
        'cases': [{'id': 'c1', 'measured': {'T': 0.6, 'R': 0.1}}],
    }
    (run_dir / 'readout.json').write_text(json.dumps(readout), encoding='utf-8')
    sys.exit(0)

elif action == 'tamper_executed_steps':
    # Kernel reports 2016 steps
    telemetry = {
        'schema_version': '1.0',
        'project_canonical_sha256': 'a' * 64,
        'epsilon_sha256': base_eps_sha256,
        'is_lossless': True,
        'executed_steps': 2016,
        'resumed_from_step': 1984,
        'energy_decay_ratio': 1e-7,
        'monitors': {'box-right': {'flux': 0.6}, 'box-left': {'flux': 0.1}},
    }
    (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry), encoding='utf-8')

    # Malicious readout claiming fake 4000 steps executed
    readout = {
        'simulation_executed': True,
        'executed_steps': 4000,
        'resumed_from_step': 1984,
        'epsilon_sha256': base_eps_sha256,
        'is_lossless': True,
        'cases': [{'id': 'c1', 'measured': {'T': 0.6, 'R': 0.1}}],
    }
    (run_dir / 'readout.json').write_text(json.dumps(readout), encoding='utf-8')
    sys.exit(0)

else:
    sys.exit(0)
