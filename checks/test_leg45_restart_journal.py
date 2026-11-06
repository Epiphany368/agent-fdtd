"""Acceptance test suite for LEG 45: ResidentRestartJournal Checkpoint & Bit-Identical Resume.

Mandatory Verification:
1. Unit tests for ResidentRestartJournal:
   - Atomic dual-slot ping-pong swapping (slot_a <-> slot_b)
   - Bit-identical tensor serialization and in-place restoration
   - Contract validation on journal initialization
   - Fault-injection & rollback: corrupted latest slot falls back cleanly to previous valid slot
2. Positive Control:
   - 4000 steps simulation with block_size=32 (125 blocks)
   - Run A: uninterrupted baseline (steps 0 -> 4000)
   - Run B: interrupted at step 1984 (62 blocks), resumed to 4000
   - Assert |T_A - T_B| == 0.0, |R_A - R_B| == 0.0, and monitor digests match bit-for-bit
   - Telemetry verification: resumed_from_step=1984, executed_steps=2016
3. Negative Controls via tools/launch.py:
   - NC 1: Mutated epsilon -> preflight exit 65 (pre_resume.py)
   - NC 2: Mutated dt -> preflight exit 65 (pre_resume.py)
   - NC 3: Damaged latest slot -> rollback to previous slot, resumes cleanly (exit 0)
   - NC 4: Tampered readout resumed_from_step -> post-flight exit 66 (tamper_detected)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
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

import torchfdtd
from torchfdtd.resident_restart import ResidentRestartJournal, build_resident_contract, MARKER
from torchfdtd.streamed_restart import RestartRecordDamaged
from torchfdtd.telemetry import hash_array
from torchfdtd import (
    Boundaries,
    BoundaryFace,
    FieldMonitor,
    Project,
    Region,
    Source,
    SpectrumSettings,
)
from torchfdtd.differentiable import AdjointOptions, _FDTD, _System, _slice

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))


class Leg45ResidentRestartJournalUnitTests(unittest.TestCase):
    """Hermetic unit tests for ResidentRestartJournal durability and rollback primitives."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='test_resident_journal_')
        self.journal_dir = Path(self.temp_dir) / 'journal'

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_write_and_read_bit_identical(self):
        """State tensors and signals are saved and restored with 0.0 bitwise difference."""
        contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1e-16,
            'steps': 4000,
            'grid_shape': [100, 50, 1],
            'block_size': 32,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract, every_blocks=1)

        t1 = torch.randn(100, 50, 1, 3, dtype=torch.float64)
        t2 = torch.randn(100, 50, 1, 3, dtype=torch.float64)
        t_psi = torch.randn(10, 50, 1, 3, dtype=torch.float64)
        state = [t1, t2, t_psi]
        signals = torch.randn(4, 7, dtype=torch.complex128)

        # Save checkpoint at step 32
        journal.save_checkpoint(32, 1, state, signals, peak_energy=1.2345)
        self.assertTrue(journal.has_checkpoint())

        # Load into empty target tensors
        r1 = torch.zeros_like(t1)
        r2 = torch.zeros_like(t2)
        r_psi = torch.zeros_like(t_psi)
        r_state = [r1, r2, r_psi]
        r_signals = torch.zeros_like(signals)

        step, blk, peak, is_complete = journal.load_latest(r_state, r_signals)
        self.assertEqual(step, 32)
        self.assertEqual(blk, 1)
        self.assertAlmostEqual(peak, 1.2345, places=10)
        self.assertFalse(is_complete)

        # Exact bitwise equality
        self.assertTrue(torch.equal(t1, r1))
        self.assertTrue(torch.equal(t2, r2))
        self.assertTrue(torch.equal(t_psi, r_psi))
        self.assertTrue(torch.equal(signals, r_signals))

        journal.close()

    def test_ping_pong_dual_slot_rotation(self):
        """Checkpoints alternate ping-pong between slot_a and slot_b with previous pointer."""
        contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1e-16,
            'steps': 4000,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract, every_blocks=1)
        state = [torch.randn(10, 10, dtype=torch.float64)]
        signals = torch.randn(2, 2, dtype=torch.complex128)

        # Save 1 -> slot_a
        p1 = journal.save_checkpoint(32, 1, state, signals)
        self.assertEqual(p1['directory'], 'slot_a')
        self.assertIsNone(p1.get('previous'))
        self.assertTrue((self.journal_dir / 'slot_a').is_dir())

        # Save 2 -> slot_b
        p2 = journal.save_checkpoint(64, 2, state, signals)
        self.assertEqual(p2['directory'], 'slot_b')
        self.assertEqual(p2.get('previous'), 'slot_a')
        self.assertTrue((self.journal_dir / 'slot_b').is_dir())
        self.assertTrue((self.journal_dir / 'slot_a').is_dir())

        # Save 3 -> slot_a
        p3 = journal.save_checkpoint(96, 3, state, signals)
        self.assertEqual(p3['directory'], 'slot_a')
        self.assertEqual(p3.get('previous'), 'slot_b')

        journal.close()

    def test_negative_control_contract_epsilon_mutation(self):
        """Opening an existing journal with mutated epsilon_sha256 raises ValueError."""
        contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1e-16,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract)
        journal.close()

        mutated_contract = {
            'marker': MARKER,
            'epsilon_sha256': 'f' * 64,  # Mutated
            'dt': 1e-16,
        }
        with self.assertRaises(ValueError) as ctx:
            ResidentRestartJournal(self.journal_dir, contract=mutated_contract)
        self.assertIn('epsilon_sha256', str(ctx.exception))

    def test_negative_control_contract_dt_mutation(self):
        """Opening an existing journal with mutated dt raises ValueError."""
        contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1.5e-16,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract)
        journal.close()

        mutated_contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1.6e-16,  # Mutated
        }
        with self.assertRaises(ValueError) as ctx:
            ResidentRestartJournal(self.journal_dir, contract=mutated_contract)
        self.assertIn('dt', str(ctx.exception))

    def test_damaged_slot_rollback(self):
        """Corrupted latest slot triggers RestartRecordDamaged and rolls back to fallback slot."""
        contract = {
            'marker': MARKER,
            'epsilon_sha256': 'a' * 64,
            'dt': 1e-16,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract, every_blocks=1)
        t_a = torch.full((10, 10), 1.0, dtype=torch.float64)
        t_b = torch.full((10, 10), 2.0, dtype=torch.float64)
        signals = torch.zeros(2, 2, dtype=torch.complex128)

        # Slot A at step 32
        journal.save_checkpoint(32, 1, [t_a], signals)
        # Slot B at step 64
        journal.save_checkpoint(64, 2, [t_b], signals)

        # Corrupt Slot B: truncate state-0.bin
        corrupted_file = self.journal_dir / 'slot_b' / 'state-0.bin'
        corrupted_file.write_bytes(b'corrupted_data_truncate')

        # Load latest should rollback to Slot A
        target = [torch.zeros(10, 10, dtype=torch.float64)]
        target_sig = torch.zeros(2, 2, dtype=torch.complex128)

        step, blk, peak, is_complete = journal.load_latest(target, target_sig)
        self.assertEqual(step, 32, "Should roll back to step 32")
        self.assertEqual(blk, 1)
        self.assertTrue(torch.equal(target[0], t_a), "Target tensor must match Slot A")
        self.assertEqual(len(journal.rollbacks), 1)
        self.assertEqual(journal.rollbacks[0]['slot'], 'slot_b')

        journal.close()


class Leg45BitIdentitySimulationTests(unittest.TestCase):
    """Positive Control: 100% Bit-Identity (0.0 error) between uninterrupted and resumed run."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='test_bit_identity_')
        self.journal_dir = Path(self.test_dir) / 'journal'

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_test_simulation(self, steps=4000, block_size=32, device='cpu'):
        """Construct deterministic 2D Yee grid with frequency plane monitors."""
        dx = 0.05
        boundaries = Boundaries(
            x_min=BoundaryFace(kind='pml', layers=8),
            x_max=BoundaryFace(kind='pml', layers=8),
            y_min=BoundaryFace(kind='pml', layers=8),
            y_max=BoundaryFace(kind='pml', layers=8),
        )
        region = Region(
            dimension='2d',
            size=(5.0, 3.0, 0.05),
            mesh=dx,
            precision='float64',
            backend='auto',
            boundaries=boundaries,
        )
        region.steps = steps

        spec = SpectrumSettings(
            sampling='frequency',
            custom_frequencies_hz=[3e8 / (1.55e-6)],
        )

        monitors = [
            FieldMonitor(id='near', center=(-1.0, 0, 0), size=(0, 2.0, 0.05), normal='x', spectrum=spec),
            FieldMonitor(id='far', center=(1.0, 0, 0), size=(0, 2.0, 0.05), normal='x', spectrum=spec),
            FieldMonitor(id='box-left', center=(-1.5, 0, 0), size=(0, 2.0, 0.05), normal='x', spectrum=spec),
            FieldMonitor(id='box-right', center=(1.5, 0, 0), size=(0, 2.0, 0.05), normal='x', spectrum=spec),
        ]

        # Dielectric slab in center
        p = Project(
            name='bit_identity_test',
            region=region,
            sources=[
                Source(
                    id='src',
                    kind='plane',
                    normal='x',
                    direction='+',
                    component='Ey',
                    center=(-1.8, 0, 0),
                    size=(0, 2.0, 0.05),
                    wavelength=1.55,
                    pulse_cycles=4,
                )
            ],
            monitors=monitors,
        )
        return p

    def test_positive_control_n8_bit_identity_0_error(self):
        """Uninterrupted 4000 steps vs Interrupted at step 1984 and Resumed to 4000 -> 0.0 difference."""
        from torchfdtd import Monitor
        from torchfdtd.adjoint_spectrum import SpectralObservation
        from torchfdtd.telemetry import hash_array

        steps = 4000
        block_size = 32
        interrupt_step = 1984  # Exactly 62 blocks completed

        dx = 0.05
        boundaries = Boundaries(
            x_min=BoundaryFace(kind='pml', layers=8),
            x_max=BoundaryFace(kind='pml', layers=8),
            y_min=BoundaryFace(kind='pml', layers=8),
            y_max=BoundaryFace(kind='pml', layers=8),
        )
        region = Region(
            dimension='2d',
            size=(5.0, 3.0, 0.05),
            mesh=dx,
            precision='float64',
            backend='auto',
            boundaries=boundaries,
            steps=steps,
        )
        sources = [
            Source(
                id='src',
                kind='plane',
                normal='x',
                direction='+',
                component='Ey',
                center=(-1.8, 0, 0),
                size=(0, 2.0, 0.05),
                wavelength=1.55,
                pulse_cycles=4,
            )
        ]
        monitors = [
            Monitor(id='near', center=(-0.8, 0, 0), component='Ey'),
            Monitor(id='far', center=(0.8, 0, 0), component='Ey'),
            Monitor(id='box_left', center=(-1.2, 0, 0), component='Ey'),
            Monitor(id='box_right', center=(1.2, 0, 0), component='Ey'),
        ]
        project = Project(name='bit_identity_test', region=region, sources=sources, monitors=monitors)

        eps_a = torch.ones(*project.region.shape, 3, dtype=torch.float64)
        eps_a[40:60, :, :, :] = 3.48**2  # Dielectric slab in center

        freq_hz = [3e8 / 1.55e-6]
        spectral = SpectralObservation(
            eps_a,
            project.region,
            [m.component for m in project.monitors],
            freq_hz,
            block_size=block_size,
        )

        # Baseline: Uninterrupted Run A (steps 0 to 4000)
        sys_a = _System(project, eps_a)
        rep_a = {}
        opt_a = AdjointOptions()
        sig_a = _FDTD.apply(eps_a, sys_a, opt_a, rep_a, spectral)

        # Interrupted Run B1: steps 0 to 1984 with restart journal saving and simulated interrupt
        journal_path = self.journal_dir
        eps_b = eps_a.clone()
        sys_b1 = _System(project, eps_b)
        opt_b1 = AdjointOptions(restart_directory=journal_path, restart_every_blocks=1, stop_at_step=interrupt_step)
        rep_b1 = {}

        sig_b1 = _FDTD.apply(eps_b, sys_b1, opt_b1, rep_b1, spectral)

        # Checkpoint at step 1984 is now published in journal
        journal = ResidentRestartJournal(journal_path)
        self.assertTrue(journal.has_checkpoint())
        ptr = journal._read_pointer()
        self.assertEqual(ptr['step'], interrupt_step)
        self.assertEqual(ptr['block'], interrupt_step // block_size)

        # Resumed Run B2: steps 1984 to 4000 from journal
        sys_b2 = _System(project, eps_b)  # Full 4000 steps project
        opt_b2 = AdjointOptions(restart_directory=journal_path, restart_every_blocks=1)
        rep_b2 = {}
        # Environment variable propagation
        os.environ['TORCHFDTD_RESUME_JOURNAL'] = str(journal_path)
        try:
            sig_b2 = _FDTD.apply(eps_b, sys_b2, opt_b2, rep_b2, spectral)
        finally:
            os.environ.pop('TORCHFDTD_RESUME_JOURNAL', None)

        # Bit-Identity Assertions
        diff = (sig_a - sig_b2).abs().max().item()
        self.assertEqual(diff, 0.0, f"Signals difference {diff} must be exactly 0.0 (bit-identical)")

        # Verify Transmission & Reflection: |T_A - T_B| == 0.0, |R_A - R_B| == 0.0
        # Monitor 0: near, Monitor 1: far (Transmission), Monitor 2: box_left (Reflection)
        t_a = (sig_a[:, 1].abs()**2).sum().item()
        t_b = (sig_b2[:, 1].abs()**2).sum().item()
        self.assertEqual(abs(t_a - t_b), 0.0, f"|T_A - T_B| difference {abs(t_a - t_b)} must be exactly 0.0")

        r_a = (sig_a[:, 2].abs()**2).sum().item()
        r_b = (sig_b2[:, 2].abs()**2).sum().item()
        self.assertEqual(abs(r_a - r_b), 0.0, f"|R_A - R_B| difference {abs(r_a - r_b)} must be exactly 0.0")

        # Verify monitor field digests match bit-for-bit
        digest_a = hash_array(sig_a.detach().cpu().numpy())
        digest_b = hash_array(sig_b2.detach().cpu().numpy())
        self.assertEqual(digest_a, digest_b, "Field digests must match bit-for-bit")

        # Verify telemetry
        self.assertIn('telemetry_kernel', rep_b2)
        tk = rep_b2['telemetry_kernel']
        self.assertEqual(tk['resumed_from_step'], interrupt_step)
        self.assertEqual(tk['executed_steps'], steps - interrupt_step)


class Leg45LauncherIntegrationTests(unittest.TestCase):
    """Subprocess integration tests validating tools/launch.py and pre/post-flight gates."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='test_launch_resume_')
        self.journal_dir = Path(self.temp_dir) / 'journal'
        self.created_run_dirs = []
        self.created_scripts = []

        # Create valid initial journal with 2 checkpoints
        contract = {
            'marker': MARKER,
            'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
            'dt': 1.5e-16,
            'steps': 4000,
            'grid_shape': [100, 50, 1],
            'block_size': 32,
        }
        journal = ResidentRestartJournal(self.journal_dir, contract=contract, every_blocks=1)
        t_a = torch.ones(100, 50, 1, 3, dtype=torch.float64)
        signals = torch.ones(4, 7, dtype=torch.complex128)
        journal.save_checkpoint(1920, 60, [t_a], signals, peak_energy=0.8)
        journal.save_checkpoint(1984, 62, [t_a], signals, peak_energy=1.0)
        journal.close()

    def tearDown(self):
        for script_file in self.created_scripts:
            script_file.unlink(missing_ok=True)
        for run_dir in self.created_run_dirs:
            shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        lock_file = ROOT / 'runs' / '.lock'
        if lock_file.is_file():
            try:
                # If no process holds lock, ensure file handle can be unlinked
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _track_run_dir(self, output: str):
        for line in output.splitlines():
            if 'run_dir=' in line:
                part = line.split('run_dir=')[1].split()[0]
                p = Path(part)
                if p.is_dir() and p not in self.created_run_dirs:
                    self.created_run_dirs.append(p)

    def _run_launch(self, script_code: str, config_dict: dict, extra_args: list = None) -> subprocess.CompletedProcess:
        # Pre-cleanup any stale lock before launch
        lock_file = ROOT / 'runs' / '.lock'
        if lock_file.is_file():
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

        script_name = f"_test_mock_driver_leg45_{os.getpid()}_{len(self.created_scripts)}.py"
        script_file = ROOT / 'src' / script_name
        script_file.write_text(script_code, encoding='utf-8')
        self.created_scripts.append(script_file)

        config_file = Path(self.temp_dir) / 'config.json'
        config_file.write_text(json.dumps(config_dict), encoding='utf-8')

        cmd = [sys.executable, '-B', str(ROOT / 'tools/launch.py'), str(config_file), f"src/{script_name}"]
        if extra_args:
            cmd.extend(extra_args)

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        try:
            res = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            self._track_run_dir(exc.stdout or '')
            raise
        self._track_run_dir(res.stdout)
        return res

    def test_negative_control_1_launch_rejects_epsilon_mutation_exit_65(self):
        """Preflight gate pre_resume.py rejects mutated epsilon with exit 65."""
        driver = "import sys; sys.exit(0)\n"
        # Config has mutated epsilon
        config = {
            'task': 'NC1 Epsilon Mutation',
            'kind': 'canary',
            'dt': 1.5e-16,
            'epsilon_sha256': 'mutated_f' * 6 + '0000',
        }
        res = self._run_launch(driver, config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, res.stdout)
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('epsilon_sha256 mismatch', res.stdout)

    def test_negative_control_2_launch_rejects_dt_mutation_exit_65(self):
        """Preflight gate pre_resume.py rejects mutated dt with exit 65."""
        driver = "import sys; sys.exit(0)\n"
        # Config has mutated dt
        config = {
            'task': 'NC2 dt Mutation',
            'kind': 'canary',
            'dt': 1.6e-16,  # Contract has 1.5e-16
            'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
        }
        res = self._run_launch(driver, config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 65, res.stdout)
        self.assertIn('failed:preflight=pre_resume.py', res.stdout)
        self.assertIn('dt mismatch', res.stdout)

    def test_negative_control_3_launch_recovers_from_corrupted_slot(self):
        """Corrupted slot_b falls back to slot_a (step 1920) and completes with exit 0."""
        # Corrupt slot_b
        (self.journal_dir / 'slot_b' / 'state-0.bin').write_bytes(b'truncated')

        driver = """import json, os, sys
from pathlib import Path
run_dir = Path(os.environ['RUN_DIR_ABS'])
journal_path = os.environ.get('TORCHFDTD_RESUME_JOURNAL')
assert journal_path, 'TORCHFDTD_RESUME_JOURNAL must be propagated'

from torchfdtd.resident_restart import ResidentRestartJournal
import torch
journal = ResidentRestartJournal(journal_path)
t = [torch.zeros(100, 50, 1, 3, dtype=torch.float64)]
sig = torch.zeros(4, 7, dtype=torch.complex128)
step, blk, peak, is_comp = journal.load_latest(t, sig)
assert step == 1920, f'Expected fallback step 1920, got {step}'

telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
    'is_lossless': True,
    'executed_steps': 2080,
    'resumed_from_step': 1920,
    'energy_decay_ratio': 1e-7,
    'monitors': {'box-right': {'flux': 0.5}, 'box-left': {'flux': 0.1}},
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'simulation_executed': True,
    'executed_steps': 2080,
    'resumed_from_step': 1920,
    'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
    'is_lossless': True,
    'cases': [{'id': 'c1', 'measured': {'T': 0.5, 'R': 0.1}}],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {
            'task': 'NC3 Corrupted Slot Fallback',
            'kind': 'canary',
            'dt': 1.5e-16,
            'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
        }
        res = self._run_launch(driver, config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertIn('exit=0', res.stdout)

        # Verify certificate issued with resumed_from_step
        self.assertTrue(self.created_run_dirs)
        run_dir = self.created_run_dirs[-1]
        cert = json.loads((run_dir / 'verification_certificate.json').read_text(encoding='utf-8'))
        self.assertTrue(cert['verified'])
        self.assertEqual(cert['resumed_from_step'], 1920)
        self.assertIn('verified:resumed_run', cert['tags'])

    def test_negative_control_4_notary_rejects_tampered_resume_steps_exit_66(self):
        """Driver claiming resumed_from_step=0 when telemetry has 1984 -> postflight exit 66."""
        driver = """import json, os, sys
from pathlib import Path
run_dir = Path(os.environ['RUN_DIR_ABS'])

telemetry = {
    'schema_version': '1.0',
    'project_canonical_sha256': 'a' * 64,
    'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
    'is_lossless': True,
    'executed_steps': 2016,
    'resumed_from_step': 1984,  # Kernel resumed from 1984
    'energy_decay_ratio': 1e-7,
    'monitors': {'box-right': {'flux': 0.5}, 'box-left': {'flux': 0.1}},
}
(run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry))

readout = {
    'simulation_executed': True,
    'executed_steps': 2016,
    'resumed_from_step': 0,     # Tampered claim: says fresh run
    'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
    'is_lossless': True,
    'cases': [{'id': 'c1', 'measured': {'T': 0.5, 'R': 0.1}}],
}
(run_dir / 'readout.json').write_text(json.dumps(readout))
sys.exit(0)
"""
        config = {
            'task': 'NC4 Tamper Resumed Steps',
            'kind': 'canary',
            'dt': 1.5e-16,
            'epsilon_sha256': '379475a1bd5b1f1e8a55c30b2d1ab408ae35ef0a8577cc62aa04a303fda52835',
        }
        res = self._run_launch(driver, config, extra_args=['--resume', str(self.journal_dir)])
        self.assertEqual(res.returncode, 66, res.stdout)
        self.assertIn('failed:postflight=tamper_detected', res.stdout)
        self.assertIn('resumed_from_step mismatch', res.stdout)


if __name__ == '__main__':
    unittest.main()
