"""Adversarial stress-testing suite for LEG 45: ResidentRestartJournal Preflight & Postflight Gates.

Author: challenger_leg45_2 (empirical challenger / critic, specialist)
Adversarial vectors tested:
1. Preflight Epsilon Mutation:
   - Mutate epsilon by 1e-7 at single grid cell -> exit 65 (pre_resume.py)
   - Mutate epsilon by 1e-12 at single grid cell -> exit 65 (pre_resume.py)
2. Preflight dt Mutation:
   - Mutate dt by +0.1% -> exit 65 (pre_resume.py)
   - Mutate dt by -0.1% -> exit 65 (pre_resume.py)
3. Preflight Grid Shape Mutation:
   - Alter grid dimensions / PML thickness -> exit 65 (pre_resume.py)
4. Postflight Tamper Detection:
   - Fabricate driver claiming resumed_from_step: 0 on resume from 1984 -> exit 66 (tamper_detected)
   - Fabricate driver claiming fake executed_steps -> exit 66 (tamper_detected)
5. Non-Zero Driver Exit Preservation:
   - Resumed run with driver exit 53 -> exit 53 preserved (not overwritten by notary)
   - Resumed run with driver exit 42 -> exit 42 preserved (not overwritten by notary)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))

import torchfdtd
from torchfdtd.resident_restart import ResidentRestartJournal, MARKER
from torchfdtd.telemetry import hash_array

FIXTURE_DRIVER = 'checks/fixtures/mock_challenger_leg45_driver.py'


class ChallengerLeg45GateAdversarialTests(unittest.TestCase):
    """Adversarial stress-tests for preflight contract gates and postflight notary."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='challenger_leg45_gate_')
        self.journal_dir = Path(self.temp_dir) / 'journal'
        self.created_run_dirs = []

        # Base simulation parameters
        self.grid_shape = [100, 50, 1]
        self.dt = 1.5e-16
        self.steps = 4000
        self.block_size = 32

        # Base epsilon array: background SiO2 with Si core
        self.base_eps = np.ones(self.grid_shape, dtype=np.float64) * 2.25
        self.base_eps[30:70, 20:30, 0] = 12.11
        self.base_eps_sha256 = hash_array(self.base_eps)

        contract = {
            'marker': MARKER,
            'epsilon_sha256': self.base_eps_sha256,
            'epsilon_shape': self.grid_shape,
            'epsilon_dtype': 'float64',
            'dt': self.dt,
            'steps': self.steps,
            'grid_shape': self.grid_shape,
            'block_size': self.block_size,
        }

        # Initialize journal and commit checkpoint at step 1984 (block 62)
        journal = ResidentRestartJournal(self.journal_dir, contract=contract, every_blocks=1)
        t_state = [torch.ones(self.grid_shape + [3], dtype=torch.float64)]
        t_signals = torch.ones(4, 7, dtype=torch.complex128)
        journal.save_checkpoint(1920, 60, t_state, t_signals, peak_energy=0.85)
        journal.save_checkpoint(1984, 62, t_state, t_signals, peak_energy=1.0)
        journal.close()

    def tearDown(self):
        for run_dir in self.created_run_dirs:
            shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _track_run_dir(self, output: str):
        for line in output.splitlines():
            if 'run_dir=' in line:
                part = line.split('run_dir=')[1].split()[0]
                p = Path(part)
                if p.is_dir() and p not in self.created_run_dirs:
                    self.created_run_dirs.append(p)

    def _run_launch(self, action: str, config_dict: dict, extra_args: list = None) -> subprocess.CompletedProcess:
        config_file = Path(self.temp_dir) / f'config_{time.time_ns()}.json'
        config_file.write_text(json.dumps(config_dict), encoding='utf-8')

        cmd = [sys.executable, '-B', str(ROOT / 'tools/launch.py'), str(config_file), FIXTURE_DRIVER]
        if extra_args:
            cmd.extend(extra_args)

        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE='1',
            CHALLENGER_ACTION=action,
            CHALLENGER_EPS_SHA256=self.base_eps_sha256,
        )

        # Retry loop to gracefully handle concurrent project locks
        for attempt in range(60):
            res = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            if "Another run holds the project lock" in res.stdout:
                time.sleep(1.5)
                continue
            self._track_run_dir(res.stdout)
            return res

        self._track_run_dir(res.stdout)
        return res

    # -------------------------------------------------------------------------
    # MANDATE 1: Preflight Epsilon Mutation (1e-7 and 1e-12 at single cell)
    # -------------------------------------------------------------------------
    def test_preflight_epsilon_mutation_1e_7_single_cell(self):
        """Mutating epsilon by 1e-7 at cell (50, 25, 0) triggers preflight exit 65."""
        mutated_eps = self.base_eps.copy()
        mutated_eps[50, 25, 0] += 1e-7
        mutated_sha256 = hash_array(mutated_eps)
        self.assertNotEqual(mutated_sha256, self.base_eps_sha256)

        config = {
            'task': 'Challenger Epsilon Mutation 1e-7',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': mutated_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('epsilon_sha256 mismatch', res.stdout)

    def test_preflight_epsilon_mutation_1e_12_single_cell(self):
        """Mutating epsilon by 1e-12 at cell (45, 22, 0) triggers preflight exit 65."""
        mutated_eps = self.base_eps.copy()
        mutated_eps[45, 22, 0] += 1e-12
        mutated_sha256 = hash_array(mutated_eps)
        self.assertNotEqual(mutated_sha256, self.base_eps_sha256)

        config = {
            'task': 'Challenger Epsilon Mutation 1e-12',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': mutated_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('epsilon_sha256 mismatch', res.stdout)

    # -------------------------------------------------------------------------
    # MANDATE 2: Preflight dt Mutation (+/- 0.1%)
    # -------------------------------------------------------------------------
    def test_preflight_dt_mutation_0_1_percent_increase(self):
        """Increasing dt by 0.1% (+0.001 * dt) triggers preflight exit 65."""
        mutated_dt = self.dt * 1.001  # +0.1%
        config = {
            'task': 'Challenger dt Mutation +0.1%',
            'kind': 'canary',
            'dt': mutated_dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('dt mismatch', res.stdout)

    def test_preflight_dt_mutation_0_1_percent_decrease(self):
        """Decreasing dt by 0.1% (-0.001 * dt) triggers preflight exit 65."""
        mutated_dt = self.dt * 0.999  # -0.1%
        config = {
            'task': 'Challenger dt Mutation -0.1%',
            'kind': 'canary',
            'dt': mutated_dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('dt mismatch', res.stdout)

    # -------------------------------------------------------------------------
    # MANDATE 3: Preflight Grid Shape / PML Mutation
    # -------------------------------------------------------------------------
    def test_preflight_grid_shape_dimension_mutation(self):
        """Altering grid shape dimensions triggers preflight exit 65."""
        mutated_shape = [105, 50, 1]  # Expanded X dimension
        config = {
            'task': 'Challenger Grid Shape Mutation',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': mutated_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('grid_shape mismatch', res.stdout)

    def test_preflight_pml_thickness_grid_mutation(self):
        """Altering PML thickness causing grid shape change triggers preflight exit 65."""
        mutated_shape = [100, 60, 1]  # Y dimension changed due to thicker PML
        config = {
            'task': 'Challenger PML Thickness Mutation',
            'kind': 'canary',
            'dt': self.dt,
            'shape': mutated_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, f"Expected exit 65, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('grid_shape mismatch', res.stdout)

    # -------------------------------------------------------------------------
    # MANDATE 4: Postflight Tamper Detection
    # -------------------------------------------------------------------------
    def test_postflight_tamper_resumed_step_claim_0(self):
        """Resumed from step 1984 but readout claims resumed_from_step: 0 -> exit 66."""
        config = {
            'task': 'Challenger Tamper Claim 0',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('tamper_claim_0', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 66, f"Expected exit 66, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:postflight=tamper_detected', res.stdout)
        self.assertIn('resumed_from_step mismatch', res.stdout)

    def test_postflight_tamper_fake_executed_steps(self):
        """Driver claiming fake executed_steps (e.g. 4000 vs kernel 2016) -> exit 66."""
        config = {
            'task': 'Challenger Tamper Executed Steps',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('tamper_executed_steps', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 66, f"Expected exit 66, got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:postflight=tamper_detected', res.stdout)
        self.assertIn('step count mismatch', res.stdout)

    # -------------------------------------------------------------------------
    # MANDATE 5: Non-Zero Driver Exit Preservation
    # -------------------------------------------------------------------------
    def test_non_zero_driver_exit_preservation_exit_53(self):
        """Driver encountering stop condition (exit 53) must preserve exit 53, not override."""
        config = {
            'task': 'Challenger Non-Zero Stop Condition Exit 53',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_53', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 53, f"Launcher must preserve driver exit code 53! Got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:exit=53', res.stdout)
        self.assertNotIn('tamper_detected', res.stdout)

        # Inspect manifest in created run dir
        self.assertTrue(self.created_run_dirs)
        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 53)
        self.assertEqual(manifest['status'], 'failed:exit=53')
        self.assertEqual((run_dir / 'status').read_text().strip(), 'failed:exit=53')

    def test_non_zero_driver_exit_preservation_exit_42(self):
        """Driver encountering arbitrary error (exit 42) must preserve exit 42."""
        config = {
            'task': 'Challenger Non-Zero Exit 42',
            'kind': 'canary',
            'dt': self.dt,
            'grid_shape': self.grid_shape,
            'epsilon_sha256': self.base_eps_sha256,
        }
        res = self._run_launch('exit_42', config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 42, f"Launcher must preserve driver exit code 42! Got {res.returncode}. Output:\n{res.stdout}")
        self.assertIn('failed:exit=42', res.stdout)

        self.assertTrue(self.created_run_dirs)
        run_dir = self.created_run_dirs[-1]
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['exit_code'], 42)
        self.assertEqual(manifest['status'], 'failed:exit=42')


if __name__ == '__main__':
    unittest.main()
