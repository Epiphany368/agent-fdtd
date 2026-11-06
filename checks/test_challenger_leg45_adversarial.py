"""Adversarial stress-testing suite for LEG 45: ResidentRestartJournal Checkpoint & Bit-Identical Resume.

Author: challenger_leg45_1 (empirical challenger / critic, specialist)
Mandate:
1. Adversarial Bit-Identity Stress Tests:
   - Multiple wavelengths (1.31 um, 1.55 um, 1.65 um) with monitor field digests and flux conservation.
   - Non-standard block boundary alignments (step 960 = 30 blocks).
   - Early interruption during active source injection (step 64 = 2 blocks, while plane wave pulse is actively emitting).
   - Multi-stage cascading restart (0 -> 320 -> 960 -> 1600 -> 1920) matching uninterrupted run.
   - Assert |T_uninterrupted - T_resumed| == 0.0 and |R_uninterrupted - R_resumed| == 0.0, and field monitor SHA-256 digests match bitwise.
2. Corrupted Slot Recovery Stress Tests:
   - Fault A: Truncated state tensor file in slot_b.
   - Fault B: Corrupted/broken meta.json / invalid descriptor in slot_b.
   - Fault C: Zero-byte signals.bin in slot_b.
   - Fault D: Unrecoverable corruption (both slot_a and slot_b corrupted) -> verify strict failure (RestartRecordDamaged raised, zero-trust refusal to fabricate state).
   - Assert that ResidentRestartJournal.load_latest() transparently falls back to slot_a without crashing and resumes computation accurately.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

import torchfdtd
from torchfdtd import (
    Boundaries,
    BoundaryFace,
    Monitor,
    Project,
    Region,
    Source,
)
from torchfdtd.adjoint_spectrum import SpectralObservation
from torchfdtd.differentiable import AdjointOptions, _FDTD, _System
from torchfdtd.resident_restart import ResidentRestartJournal, build_resident_contract, MARKER
from torchfdtd.streamed_restart import RestartRecordDamaged
from torchfdtd.telemetry import hash_array

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))


class ChallengerLeg45BitIdentityAdversarialTests(unittest.TestCase):
    """Adversarial Bit-Identity Verification under edge conditions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='challenger_leg45_bit_')
        self.journal_dir = Path(self.temp_dir) / 'journal'

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_2d_simulation(self, steps=2560, wavelengths=(1.31, 1.55, 1.65), block_size=32):
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
        project = Project(name='challenger_adv_sim', region=region, sources=sources, monitors=monitors)

        eps = torch.ones(*project.region.shape, 3, dtype=torch.float64)
        eps[40:60, :, :, :] = 3.48**2  # High-index slab in center

        freq_hz = [3e8 / (wl * 1e-6) for wl in wavelengths]
        spectral = SpectralObservation(
            eps,
            project.region,
            [m.component for m in project.monitors],
            freq_hz,
            block_size=block_size,
        )
        return project, eps, spectral

    def test_adversarial_bit_identity_multi_wavelength_and_step_960(self):
        """Multi-wavelength (1.31, 1.55, 1.65 um) interrupted at step 960 (30 blocks) -> exactly 0.0 error."""
        steps = 2560
        interrupt_step = 960
        wavelengths = (1.31, 1.55, 1.65)
        project, eps_a, spectral_a = self._create_2d_simulation(steps=steps, wavelengths=wavelengths)

        # Baseline: Uninterrupted Run A (0 -> 2560)
        sys_a = _System(project, eps_a)
        rep_a = {}
        opt_a = AdjointOptions()
        sig_a = _FDTD.apply(eps_a, sys_a, opt_a, rep_a, spectral_a)

        # Interrupted Run B1: (0 -> 960) with checkpoint saving
        eps_b = eps_a.clone()
        spectral_b = SpectralObservation(
            eps_b,
            project.region,
            [m.component for m in project.monitors],
            [3e8 / (wl * 1e-6) for wl in wavelengths],
            block_size=32,
        )
        sys_b1 = _System(project, eps_b)
        rep_b1 = {}
        opt_b1 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1, stop_at_step=interrupt_step)
        sig_b1 = _FDTD.apply(eps_b, sys_b1, opt_b1, rep_b1, spectral_b)

        journal = ResidentRestartJournal(self.journal_dir)
        self.assertTrue(journal.has_checkpoint())
        ptr = journal._read_pointer()
        self.assertEqual(ptr['step'], interrupt_step)
        self.assertEqual(ptr['block'], 30)
        journal.close()

        # Resumed Run B2: (960 -> 2560)
        sys_b2 = _System(project, eps_b)
        rep_b2 = {}
        opt_b2 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1)
        sig_b2 = _FDTD.apply(eps_b, sys_b2, opt_b2, rep_b2, spectral_b)

        # Absolute Error Checks
        diff = (sig_a - sig_b2).abs().max().item()
        self.assertEqual(diff, 0.0, f"Signals difference {diff} must be strictly 0.0 across all 3 wavelengths")

        # Spectral Transmittance T and Reflectance R across all 3 wavelengths
        # Monitor index 1: far (T), Monitor index 2: box_left (R)
        for w_idx, wl in enumerate(wavelengths):
            t_a = (sig_a[w_idx, 1].abs()**2).item()
            t_b = (sig_b2[w_idx, 1].abs()**2).item()
            self.assertEqual(abs(t_a - t_b), 0.0, f"|T_A - T_B| at {wl} um must be 0.0")

            r_a = (sig_a[w_idx, 2].abs()**2).item()
            r_b = (sig_b2[w_idx, 2].abs()**2).item()
            self.assertEqual(abs(r_a - r_b), 0.0, f"|R_A - R_B| at {wl} um must be 0.0")

        # Field SHA-256 Digest Bit-Identity
        digest_a = hash_array(sig_a.detach().cpu().numpy())
        digest_b = hash_array(sig_b2.detach().cpu().numpy())
        self.assertEqual(digest_a, digest_b, "Multi-wavelength spectral digests must match bit-for-bit")

        # Telemetry verification
        self.assertIn('telemetry_kernel', rep_b2)
        tk = rep_b2['telemetry_kernel']
        self.assertEqual(tk['resumed_from_step'], interrupt_step)
        self.assertEqual(tk['executed_steps'], steps - interrupt_step)

    def test_adversarial_bit_identity_early_interruption_during_pulse(self):
        """Interruption at step 64 (block 2) while plane pulse is actively injecting fields -> 0.0 error."""
        steps = 1280
        interrupt_step = 64  # Pulse is actively mid-injection
        project, eps_a, spectral_a = self._create_2d_simulation(steps=steps, wavelengths=(1.55,))

        # Uninterrupted Baseline
        sys_a = _System(project, eps_a)
        rep_a = {}
        opt_a = AdjointOptions()
        sig_a = _FDTD.apply(eps_a, sys_a, opt_a, rep_a, spectral_a)

        # Interrupted Run B1
        eps_b = eps_a.clone()
        spectral_b = SpectralObservation(
            eps_b,
            project.region,
            [m.component for m in project.monitors],
            [3e8 / 1.55e-6],
            block_size=32,
        )
        sys_b1 = _System(project, eps_b)
        rep_b1 = {}
        opt_b1 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1, stop_at_step=interrupt_step)
        sig_b1 = _FDTD.apply(eps_b, sys_b1, opt_b1, rep_b1, spectral_b)

        # Resumed Run B2
        sys_b2 = _System(project, eps_b)
        rep_b2 = {}
        opt_b2 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1)
        sig_b2 = _FDTD.apply(eps_b, sys_b2, opt_b2, rep_b2, spectral_b)

        # Bit-identity assertion
        diff = (sig_a - sig_b2).abs().max().item()
        self.assertEqual(diff, 0.0, f"Pulse mid-injection resume error {diff} must be 0.0")
        self.assertEqual(
            hash_array(sig_a.detach().cpu().numpy()),
            hash_array(sig_b2.detach().cpu().numpy()),
            "Field digests must match bit-for-bit even when interrupted mid-pulse",
        )

    def test_adversarial_multi_restart_cascade(self):
        """Cascading restarts (0 -> 320 -> 960 -> 1600 -> 1920) across alternating ping-pong slots -> 0.0 error."""
        steps = 1920
        project, eps_a, spectral_a = self._create_2d_simulation(steps=steps, wavelengths=(1.55,))

        # Baseline
        sys_a = _System(project, eps_a)
        rep_a = {}
        opt_a = AdjointOptions()
        sig_a = _FDTD.apply(eps_a, sys_a, opt_a, rep_a, spectral_a)

        # Cascade Run B
        eps_b = eps_a.clone()
        spectral_b = SpectralObservation(
            eps_b,
            project.region,
            [m.component for m in project.monitors],
            [3e8 / 1.55e-6],
            block_size=32,
        )

        # Stage 1: 0 -> 320
        opt_1 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1, stop_at_step=320)
        _FDTD.apply(eps_b, _System(project, eps_b), opt_1, {}, spectral_b)

        # Stage 2: 320 -> 960
        opt_2 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1, stop_at_step=960)
        _FDTD.apply(eps_b, _System(project, eps_b), opt_2, {}, spectral_b)

        # Stage 3: 960 -> 1600
        opt_3 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1, stop_at_step=1600)
        _FDTD.apply(eps_b, _System(project, eps_b), opt_3, {}, spectral_b)

        # Stage 4: 1600 -> 1920 (Completion)
        opt_4 = AdjointOptions(restart_directory=self.journal_dir, restart_every_blocks=1)
        rep_4 = {}
        sig_b = _FDTD.apply(eps_b, _System(project, eps_b), opt_4, rep_4, spectral_b)

        diff = (sig_a - sig_b).abs().max().item()
        self.assertEqual(diff, 0.0, f"Cascaded 4-stage resume error {diff} must be 0.0")
        self.assertEqual(
            hash_array(sig_a.detach().cpu().numpy()),
            hash_array(sig_b.detach().cpu().numpy()),
        )
        self.assertEqual(rep_4['telemetry_kernel']['resumed_from_step'], 1600)
        self.assertEqual(rep_4['telemetry_kernel']['executed_steps'], 320)


class ChallengerLeg45CorruptedSlotRecoveryTests(unittest.TestCase):
    """Adversarial fault-injection stress testing on dual-slot ping-pong recovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='challenger_leg45_corrupt_')
        self.journal_dir = Path(self.temp_dir) / 'journal'

        self.contract = {
            'marker': MARKER,
            'epsilon_sha256': 'e' * 64,
            'dt': 1e-16,
            'steps': 4000,
            'grid_shape': [100, 50, 1],
            'block_size': 32,
        }
        # Initialize journal and write 2 slots
        journal = ResidentRestartJournal(self.journal_dir, contract=self.contract, every_blocks=1)
        self.t_a = torch.full((100, 50, 1, 3), 10.0, dtype=torch.float64)
        self.sig_a = torch.full((2, 4), 1.0 + 2.0j, dtype=torch.complex128)
        journal.save_checkpoint(32, 1, [self.t_a], self.sig_a, peak_energy=0.5)

        self.t_b = torch.full((100, 50, 1, 3), 20.0, dtype=torch.float64)
        self.sig_b = torch.full((2, 4), 3.0 + 4.0j, dtype=torch.complex128)
        journal.save_checkpoint(64, 2, [self.t_b], self.sig_b, peak_energy=1.0)
        journal.close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fault_a_truncated_state_tensor_file_in_slot_b(self):
        """Truncated state tensor in slot_b transparently falls back to slot_a."""
        state_file = self.journal_dir / 'slot_b' / 'state-0.bin'
        self.assertTrue(state_file.is_file())
        state_file.write_bytes(b'truncated_partial_bytes')

        journal = ResidentRestartJournal(self.journal_dir)
        target_t = [torch.zeros_like(self.t_a)]
        target_sig = torch.zeros_like(self.sig_a)

        step, blk, peak, is_comp = journal.load_latest(target_t, target_sig)
        self.assertEqual(step, 32, "Must roll back to slot_a step 32")
        self.assertEqual(blk, 1)
        self.assertAlmostEqual(peak, 0.5)
        self.assertFalse(is_comp)
        self.assertTrue(torch.equal(target_t[0], self.t_a), "Must match slot_a tensor data")
        self.assertTrue(torch.equal(target_sig, self.sig_a), "Must match slot_a signals")
        self.assertEqual(len(journal.rollbacks), 1)
        self.assertEqual(journal.rollbacks[0]['slot'], 'slot_b')
        journal.close()

    def test_fault_b_corrupted_meta_json_in_slot_b(self):
        """Malformed / corrupted meta.json in slot_b transparently falls back to slot_a."""
        meta_file = self.journal_dir / 'slot_b' / 'meta.json'
        self.assertTrue(meta_file.is_file())
        meta_file.write_text('{broken_json: true, unterminated...', encoding='utf-8')

        journal = ResidentRestartJournal(self.journal_dir)
        target_t = [torch.zeros_like(self.t_a)]
        target_sig = torch.zeros_like(self.sig_a)

        step, blk, peak, is_comp = journal.load_latest(target_t, target_sig)
        self.assertEqual(step, 32, "Must roll back to slot_a step 32")
        self.assertTrue(torch.equal(target_t[0], self.t_a))
        self.assertTrue(torch.equal(target_sig, self.sig_a))
        self.assertEqual(len(journal.rollbacks), 1)
        journal.close()

    def test_fault_c_zero_byte_signals_file_in_slot_b(self):
        """Zero-byte signals.bin file in slot_b triggers size mismatch and falls back to slot_a."""
        sig_file = self.journal_dir / 'slot_b' / 'signals.bin'
        self.assertTrue(sig_file.is_file())
        sig_file.write_bytes(b'')  # 0 bytes

        journal = ResidentRestartJournal(self.journal_dir)
        target_t = [torch.zeros_like(self.t_a)]
        target_sig = torch.zeros_like(self.sig_a)

        step, blk, peak, is_comp = journal.load_latest(target_t, target_sig)
        self.assertEqual(step, 32, "Must roll back to slot_a step 32")
        self.assertTrue(torch.equal(target_t[0], self.t_a))
        self.assertTrue(torch.equal(target_sig, self.sig_a))
        self.assertEqual(len(journal.rollbacks), 1)
        journal.close()

    def test_fault_d_unrecoverable_both_slots_corrupted_strict_zero_trust(self):
        """When BOTH slot_b and slot_a are corrupted, load_latest raises RestartRecordDamaged."""
        # Corrupt slot_b
        (self.journal_dir / 'slot_b' / 'state-0.bin').write_bytes(b'bad_b')
        # Corrupt slot_a
        (self.journal_dir / 'slot_a' / 'state-0.bin').write_bytes(b'bad_a')

        journal = ResidentRestartJournal(self.journal_dir)
        target_t = [torch.zeros_like(self.t_a)]
        target_sig = torch.zeros_like(self.sig_a)

        with self.assertRaises(RestartRecordDamaged):
            journal.load_latest(target_t, target_sig)
        journal.close()


if __name__ == '__main__':
    unittest.main()
