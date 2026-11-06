"""Adversarial stress-testing suite for LEG 46 Modal Block-Fused Injection Kernel.

Author: challenger_leg46_3 (empirical challenger / critic, specialist)
Mandate:
1. Verify in WSL2 that _ModalSystem and ModeInjectedPlaneSimulation execute cleanly with both
   execution_profile='block_fused' and execution_profile='reference'.
2. Empirically verify that |T_fused - T_legacy| == 0.0 and |R_fused - R_legacy| == 0.0 hold under
   varying modal waveguide profiles and step counts.
3. Test edge case: ensure that setting os.environ['TORCHFDTD_BLOCK_FUSED'] = '0' cleanly falls back
   to reference mode without errors.
4. Stress test higher-order guided modes (mode_index > 0).
5. Verify zero dynamic allocation invariants in block_fused kernel.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

import torchfdtd
from torchfdtd import (
    Boundaries,
    BoundaryFace,
    FieldMonitor,
    Monitor,
    Project,
    Region,
    Source,
)
from torchfdtd.adjoint_spectrum import SpectralObservation
from torchfdtd.differentiable import AdjointOptions, _FDTD, _System
from torchfdtd.mode_ports import C0
from torchfdtd.mode_injection import (
    prepare_modal_launch,
    ModeInjectedPlaneSimulation,
    _profile,
    _ModalSystem,
)
from torchfdtd.telemetry import hash_array

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))


class ChallengerLeg46ModalStressTests(unittest.TestCase):
    """Empirical challenge suite for LEG 46 block-fused modal injection kernel."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='challenger_leg46_modal_')
        # Ensure environment clean at start of each test
        os.environ.pop('TORCHFDTD_BLOCK_FUSED', None)

    def tearDown(self):
        os.environ.pop('TORCHFDTD_BLOCK_FUSED', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_modal_project(
        self,
        normal='x',
        width=4.0,
        height=1.2,
        length=8.0,
        mesh=0.1,
        steps=150,
        wavelength=1.55,
        pulse_cycles=2,
    ):
        """Constructs a 3D modal injection project with transverse periodic boundaries."""
        w = 'xyz'.index(normal)
        axes = (w, (w + 1) % 3, (w + 2) % 3)

        def xyz(values):
            return tuple(values[axes.index(i)] for i in range(3))

        boundaries = Boundaries(**{
            axis + '_' + side: BoundaryFace(kind='periodic')
            for axis in 'xyz' if axis != normal
            for side in ('min', 'max')
        })
        region = Region(
            dimension='3d',
            size=xyz((length, width, height)),
            mesh=mesh,
            pml_cells=8,
            steps=steps,
            precision='float64',
            material_sampling='yee',
            boundaries=boundaries,
        )
        sources = [
            Source(
                kind='plane',
                normal=normal,
                direction='+',
                center=xyz((-2.0, 0, 0)),
                size=xyz((0, width, height)),
                wavelength=wavelength,
                pulse_cycles=pulse_cycles,
            )
        ]
        monitors = [
            FieldMonitor(id='outside', center=xyz((-2.8, 0, 0)), size=xyz((0, width, height)), normal=normal),
            FieldMonitor(id='near', center=xyz((-1.0, 0, 0)), size=xyz((0, width, height)), normal=normal),
            FieldMonitor(id='far', center=xyz((1.0, 0, 0)), size=xyz((0, width, height)), normal=normal),
        ]
        return Project(name='challenger_modal', region=region, sources=sources, monitors=monitors)

    def _run_modal_comparison(self, project, eps_fn, steps=150, mode_index=0, num_modes=1):
        """Helper to run both block_fused and reference profiles and assert 0.0 bit-identity."""
        launch = prepare_modal_launch(project, eps_fn, mode_index=mode_index, num_modes=num_modes)
        cross_section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
        base = torch.tensor(np.broadcast_to(cross_section, project.region.shape + (3,)).copy())
        freq_hz = [C0 / (project.sources[0].wavelength * 1e-6)]

        # Spectral observers for T and R
        spec_monitors = [
            Monitor(id='refl', center=(-2.5, 0, 0), component='Ey'),
            Monitor(id='trans', center=(1.0, 0, 0), component='Ey'),
        ]
        p_modal = project.model_copy(deep=True)
        p_modal.monitors = spec_monitors

        # 1. Path A: execution_profile='block_fused'
        opts_fused = AdjointOptions(execution_profile='block_fused')
        spec_fused = SpectralObservation(base, p_modal.region, [m.component for m in spec_monitors], freq_hz, block_size=32)
        sys_fused = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_fused, observation_monitors=spec_fused.observers)
        rep_fused = {}
        res_fused = _FDTD.apply(base, sys_fused, opts_fused, rep_fused, spec_fused)

        # 2. Path B: execution_profile='reference'
        opts_ref = AdjointOptions(execution_profile='reference')
        spec_ref = SpectralObservation(base, p_modal.region, [m.component for m in spec_monitors], freq_hz, block_size=32)
        sys_ref = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_ref, observation_monitors=spec_ref.observers)
        rep_ref = {}
        res_ref = _FDTD.apply(base, sys_ref, opts_ref, rep_ref, spec_ref)

        # 3. Assert IEEE 754 bit-identity on spectral signals
        T_fused = float((res_fused[:, 1].real**2 + res_fused[:, 1].imag**2).sum().item())
        R_fused = float((res_fused[:, 0].real**2 + res_fused[:, 0].imag**2).sum().item())

        T_ref = float((res_ref[:, 1].real**2 + res_ref[:, 1].imag**2).sum().item())
        R_ref = float((res_ref[:, 0].real**2 + res_ref[:, 0].imag**2).sum().item())

        diff_T = abs(T_fused - T_ref)
        diff_R = abs(R_fused - R_ref)

        self.assertEqual(diff_T, 0.0, f"|T_fused - T_ref| must be exactly 0.0 (got {diff_T})")
        self.assertEqual(diff_R, 0.0, f"|R_fused - R_ref| must be exactly 0.0 (got {diff_R})")
        self.assertTrue(torch.equal(res_fused, res_ref), "Signals tensor must be bitwise identical")
        self.assertEqual(hash_array(res_fused), hash_array(res_ref), "SHA-256 hash must be bitwise identical")

        # 4. ModeInjectedPlaneSimulation comparison
        sim_fused = ModeInjectedPlaneSimulation(project, launch, opts_fused)
        sim_ref = ModeInjectedPlaneSimulation(project, launch, opts_ref)
        with torch.no_grad():
            out_fused = sim_fused(base, freq_hz)
            out_ref = sim_ref(base, freq_hz)

        max_field_diff = 0.0
        for key in out_fused:
            f_diff = (out_fused[key].fields - out_ref[key].fields).abs().max().item()
            if f_diff > max_field_diff:
                max_field_diff = f_diff
            self.assertEqual(f_diff, 0.0, f"Monitor '{key}' field difference must be 0.0 (got {f_diff})")

        return {
            'T_fused': T_fused,
            'R_fused': R_fused,
            'diff_T': diff_T,
            'diff_R': diff_R,
            'max_field_diff': max_field_diff,
            'hash': hash_array(res_fused),
        }

    def test_1_execution_profiles_clean_runs(self):
        """Verify _ModalSystem and ModeInjectedPlaneSimulation execute cleanly with 'block_fused', 'reference', and 'auto'."""
        project = self._build_modal_project(steps=100)
        eps_fn = lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 4.0, 2.25)
        launch = prepare_modal_launch(project, eps_fn)
        cross_section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
        base = torch.tensor(np.broadcast_to(cross_section, project.region.shape + (3,)).copy())
        freq_hz = [C0 / 1.55e-6]

        outputs = {}
        for profile in ('block_fused', 'reference', 'auto'):
            opts = AdjointOptions(execution_profile=profile)
            sim = ModeInjectedPlaneSimulation(project, launch, opts)
            with torch.no_grad():
                out = sim(base, freq_hz)
            outputs[profile] = out
            self.assertIn('outside', out)
            self.assertIn('near', out)
            self.assertIn('far', out)

        # Cross-compare all pairs: 'block_fused', 'reference', 'auto' must be mutually bit-identical
        for p1 in ('block_fused', 'auto'):
            for key in ('outside', 'near', 'far'):
                f_diff = (outputs[p1][key].fields - outputs['reference'][key].fields).abs().max().item()
                self.assertEqual(f_diff, 0.0, f"Profile {p1} vs reference on monitor {key} diff={f_diff}")

    def test_2_varying_waveguide_profiles_bit_identity(self):
        """Empirically test |T_fused - T_legacy| == 0.0 and |R_fused - R_legacy| == 0.0 across varying waveguide profiles."""
        profiles = [
            # Profile A: Standard high-contrast Si slab (eps_core = 3.48^2 ~ 12.11, eps_clad = 1.444^2 ~ 2.085)
            (
                "Profile A (Si-SiO2 high contrast)",
                lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 3.48**2, 1.444**2),
            ),
            # Profile B: Asymmetric dielectric slab (air upper clad eps=1.0, Si3N4 core eps=4.0, SiO2 sub eps=2.1)
            (
                "Profile B (Asymmetric air-Si3N4-SiO2)",
                lambda u, v: np.where(u > 0.3, 1.0, np.where(u < -0.3, 2.1, 4.0)),
            ),
            # Profile C: Low index contrast slab (eps_core = 3.45^2, eps_clad = 3.40^2)
            (
                "Profile C (Low contrast 3.45 / 3.40)",
                lambda u, v: np.where(abs(u) < 0.5 - 1e-9, 3.45**2, 3.40**2),
            ),
            # Profile D: Smooth continuous Gaussian profile
            (
                "Profile D (Smooth Gaussian profile)",
                lambda u, v: 2.25 + (4.0 - 2.25) * np.exp(-((u / 0.4)**2)),
            ),
        ]

        project = self._build_modal_project(steps=120)
        for name, eps_fn in profiles:
            with self.subTest(profile=name):
                res = self._run_modal_comparison(project, eps_fn, steps=120)
                self.assertEqual(res['diff_T'], 0.0, f"Failed bit-identity T on {name}")
                self.assertEqual(res['diff_R'], 0.0, f"Failed bit-identity R on {name}")
                self.assertEqual(res['max_field_diff'], 0.0, f"Failed field bit-identity on {name}")

    def test_3_varying_step_counts_and_block_boundaries(self):
        """Empirically test bit-identity across short, prime, and block-aligned step counts."""
        eps_fn = lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 4.0, 2.25)
        step_counts = [
            48,   # Short step count during active Gaussian envelope
            96,   # Exact 3 blocks of 32
            115,  # Non-block-aligned (115 % 32 = 19)
            160,  # Exact 5 blocks of 32
            227,  # Non-block-aligned prime
            320,  # Extended stepping (10 blocks)
        ]

        for steps in step_counts:
            with self.subTest(steps=steps):
                project = self._build_modal_project(steps=steps)
                res = self._run_modal_comparison(project, eps_fn, steps=steps)
                self.assertEqual(res['diff_T'], 0.0, f"Failed bit-identity T at steps={steps}")
                self.assertEqual(res['diff_R'], 0.0, f"Failed bit-identity R at steps={steps}")
                self.assertEqual(res['max_field_diff'], 0.0, f"Failed field bit-identity at steps={steps}")

    def test_4_environ_block_fused_zero_fallback(self):
        """Test edge case: TORCHFDTD_BLOCK_FUSED='0' forces clean fallback to reference mode."""
        project = self._build_modal_project(steps=120)
        eps_fn = lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 4.0, 2.25)
        launch = prepare_modal_launch(project, eps_fn)
        cross_section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
        base = torch.tensor(np.broadcast_to(cross_section, project.region.shape + (3,)).copy())
        freq_hz = [C0 / 1.55e-6]

        spec_monitors = [
            Monitor(id='refl', center=(-2.5, 0, 0), component='Ey'),
            Monitor(id='trans', center=(1.0, 0, 0), component='Ey'),
        ]
        p_modal = project.model_copy(deep=True)
        p_modal.monitors = spec_monitors

        # 1. First, establish reference ground truth
        os.environ.pop('TORCHFDTD_BLOCK_FUSED', None)
        opts_ref = AdjointOptions(execution_profile='reference')
        spec_ref = SpectralObservation(base, p_modal.region, [m.component for m in spec_monitors], freq_hz, block_size=32)
        sys_ref = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_ref, observation_monitors=spec_ref.observers)
        rep_ref = {}
        res_ref = _FDTD.apply(base, sys_ref, opts_ref, rep_ref, spec_ref)

        # 2. Now set TORCHFDTD_BLOCK_FUSED='0' and pass options with execution_profile='block_fused'
        os.environ['TORCHFDTD_BLOCK_FUSED'] = '0'
        opts_fallback = AdjointOptions(execution_profile='block_fused')
        spec_fallback = SpectralObservation(base, p_modal.region, [m.component for m in spec_monitors], freq_hz, block_size=32)
        sys_fallback = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_fallback, observation_monitors=spec_fallback.observers)

        # Invariant: _prepare_block_fused was skipped, so _fused_modal['E'] is None
        self.assertIsNone(sys_fallback._fused_modal['E'], "Fallback MUST ensure _fused_modal['E'] is None")
        self.assertIsNone(sys_fallback._fused_modal['H'], "Fallback MUST ensure _fused_modal['H'] is None")

        rep_fallback = {}
        res_fallback = _FDTD.apply(base, sys_fallback, opts_fallback, rep_fallback, spec_fallback)

        # Must match reference exactly
        self.assertTrue(torch.equal(res_fallback, res_ref), "Fallback must yield bit-identical signals to reference")
        self.assertEqual(hash_array(res_fallback), hash_array(res_ref), "Fallback SHA-256 must match reference")

        # Also test ModeInjectedPlaneSimulation with TORCHFDTD_BLOCK_FUSED='0'
        sim_fallback = ModeInjectedPlaneSimulation(project, launch, opts_fallback)
        sim_ref = ModeInjectedPlaneSimulation(project, launch, opts_ref)
        with torch.no_grad():
            out_fallback = sim_fallback(base, freq_hz)
            out_ref = sim_ref(base, freq_hz)

        for key in out_fallback:
            f_diff = (out_fallback[key].fields - out_ref[key].fields).abs().max().item()
            self.assertEqual(f_diff, 0.0, f"Fallback monitor {key} fields must match reference (diff={f_diff})")

        # 3. Setting TORCHFDTD_BLOCK_FUSED='1' should re-enable block-fused
        os.environ['TORCHFDTD_BLOCK_FUSED'] = '1'
        sys_fused_again = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_fallback, observation_monitors=spec_fallback.observers)
        self.assertIsNotNone(sys_fused_again._fused_modal['E'], "TORCHFDTD_BLOCK_FUSED='1' must re-enable fused modal injection")

    def test_5_higher_order_guided_mode(self):
        """Stress-test higher-order guided mode (mode_index=1) in a wide multimode waveguide."""
        # Wide core supporting multiple guided modes
        project = self._build_modal_project(width=5.0, height=1.6, steps=120)
        eps_fn = lambda u, v: np.where(abs(u) < 0.9 - 1e-9, 4.0, 2.25)

        # Test mode_index = 1
        res = self._run_modal_comparison(project, eps_fn, steps=120, mode_index=1, num_modes=2)
        self.assertEqual(res['diff_T'], 0.0, "Higher-order mode T diff must be 0.0")
        self.assertEqual(res['diff_R'], 0.0, "Higher-order mode R diff must be 0.0")
        self.assertEqual(res['max_field_diff'], 0.0, "Higher-order mode field diff must be 0.0")

    def test_6_block_fused_zero_dynamic_allocation_structure(self):
        """Verify that _ModalSystem preallocates scratch and target views, eliminating dynamic allocs."""
        project = self._build_modal_project(steps=50)
        eps_fn = lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 4.0, 2.25)
        launch = prepare_modal_launch(project, eps_fn)
        cross_section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
        base = torch.tensor(np.broadcast_to(cross_section, project.region.shape + (3,)).copy())

        opts = AdjointOptions(execution_profile='block_fused')
        sys = _ModalSystem(project.model_copy(deep=True), base, launch=launch, options=opts)

        # Check internal structure
        self.assertIn('E', sys._fused_modal)
        self.assertIn('H', sys._fused_modal)
        self.assertIsNotNone(sys._fused_modal['E'])
        self.assertIsNotNone(sys._fused_modal['H'])

        target_u, target_v, profiles, waveforms = sys._fused_modal['E']
        scratch = sys._modal_scratch['E']

        self.assertIsNotNone(scratch)
        # Scratch shape must match profile spatial dimensions
        self.assertEqual(scratch.shape, profiles.shape[:-1])
        # Scratch storage pointer must remain fixed across injections
        scratch_ptr_before = scratch.data_ptr()

        # Step injection manually and verify scratch reuse
        sys.inject(sys.grid.E, 'E', 0)
        sys.inject(sys.grid.E, 'E', 1)
        sys.inject(sys.grid.E, 'E', 2)

        scratch_ptr_after = sys._modal_scratch['E'].data_ptr()
        self.assertEqual(scratch_ptr_before, scratch_ptr_after, "Scratch buffer storage must be reused in-place")


if __name__ == '__main__':
    unittest.main()
