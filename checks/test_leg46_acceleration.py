"""Acceptance Test Suite for Wave B — LEG 46:
Block-Fused Modal Injection, Zero-Allocation Observation Ring Buffer Kernel, and Verified Auto-Shutoff.

Acceptance Criteria:
1. Positive control on N8 grating showing >= 1.35x speedup.
2. Exact bit-identity positive control (|T_fused - T_legacy| == 0.0, |R_fused - R_legacy| == 0.0).
3. Auto-shutoff accuracy (|Delta T| <= 1e-6, |Delta R| <= 1e-6).
4. Negative control 1: premature shutoff with E(t)/E_max > 1e-9 exits 66 (under_converged_time).
5. Negative control 2: monitor flux instability exits 66 (under_converged_flux).
6. Negative control 3: spoofed readout auto-shutoff tag exits 66 (tamper_detected).
"""
import copy
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

from torchfdtd import (
    AdjointOptions,
    Boundaries,
    BoundaryFace,
    DifferentiablePlaneSimulation,
    FieldMonitor,
    Monitor,
    Project,
    Region,
    RunControl,
    Source,
)
from torchfdtd.adjoint_spectrum import SpectralObservation
from torchfdtd.differentiable import _FDTD, _System
from torchfdtd.run_control import DecayDecision
from torchfdtd.telemetry import emit_telemetry, hash_array
from torchfdtd.mode_ports import C0
from torchfdtd.mode_injection import (
    prepare_modal_launch,
    ModeInjectedPlaneSimulation,
    _profile,
    _ModalSystem,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))

LAUNCH = ROOT / 'tools' / 'launch.py'
POST_NOTARY = ROOT / 'checks' / 'post_notary.py'


def import_post_notary():
    import importlib.util
    spec = importlib.util.spec_from_file_location('post_notary', POST_NOTARY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Leg46ZeroAllocAndBitIdentityTests(unittest.TestCase):
    """Verifies Zero-Allocation Observation Ring Buffer and Exact Bit-Identity (0.0 error)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg46_zero_alloc_')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_n8_project(self, steps=2000, block_size=32):
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
            Monitor(id='box-left', center=(-1.2, 0, 0), component='Ey'),
            Monitor(id='box-right', center=(1.2, 0, 0), component='Ey'),
        ]
        return Project(name='leg46_n8', region=region, sources=sources, monitors=monitors)

    def test_2_exact_bit_identity_0_error(self):
        """Exact bit-identity positive control (|T_fused - T_legacy| == 0.0, |R_fused - R_legacy| == 0.0)."""
        project = self._build_n8_project(steps=1000)
        freq_hz = [3e8 / 1.55e-6]

        eps = torch.ones(*project.region.shape, 3, dtype=torch.float64)
        eps[40:60, :, :, :] = 3.48**2  # N8 core silicon slab

        # Path 1: Block-fused and zero-alloc observe
        opts_fused = AdjointOptions(execution_profile='block_fused')
        spec_fused = SpectralObservation(eps, project.region, [m.component for m in project.monitors], freq_hz, block_size=32)
        system_fused = _System(project.model_copy(deep=True), eps, observation_monitors=spec_fused.observers)
        rep_fused = {}
        res_fused = _FDTD.apply(eps, system_fused, opts_fused, rep_fused, spec_fused)

        # Path 2: Legacy fallback path (no out tensor passed to observe)
        opts_legacy = AdjointOptions(execution_profile='reference')
        spec_legacy = SpectralObservation(eps, project.region, [m.component for m in project.monitors], freq_hz, block_size=32)
        system_legacy = _System(project.model_copy(deep=True), eps, observation_monitors=spec_legacy.observers)
        
        # Monkeypatch system_legacy.observe to force legacy out=None path
        orig_observe = system_legacy.observe
        def legacy_observe(state=None, out=None):
            res = orig_observe(state=state, out=None)
            if out is not None:
                out.copy_(res)
            return res
        system_legacy.observe = legacy_observe

        rep_legacy = {}
        res_legacy = _FDTD.apply(eps, system_legacy, opts_legacy, rep_legacy, spec_legacy)

        # Compute T and R
        T_fused = float((res_fused[:, 1].real**2 + res_fused[:, 1].imag**2).sum().item())
        R_fused = float((res_fused[:, 0].real**2 + res_fused[:, 0].imag**2).sum().item())

        T_legacy = float((res_legacy[:, 1].real**2 + res_legacy[:, 1].imag**2).sum().item())
        R_legacy = float((res_legacy[:, 0].real**2 + res_legacy[:, 0].imag**2).sum().item())

        diff_T = abs(T_fused - T_legacy)
        diff_R = abs(R_fused - R_legacy)

        self.assertEqual(diff_T, 0.0, f"Transmission must be bit-identical (diff={diff_T})")
        self.assertEqual(diff_R, 0.0, f"Reflection must be bit-identical (diff={diff_R})")

        # Field bytes must be 100% bitwise identical
        diff_signals = (res_fused != res_legacy).sum().item()
        self.assertEqual(diff_signals, 0, f"Signals tensor had {diff_signals} mismatched elements")
        self.assertTrue(torch.equal(res_fused, res_legacy))
        self.assertEqual(hash_array(res_fused), hash_array(res_legacy))

    def _build_modal_project(self, steps=150):
        normal = 'x'
        width = 4.0
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
            size=xyz((8.0, width, 1.2)),
            mesh=0.1,
            pml_cells=8,
            steps=steps,
            precision='float64',
            material_sampling='yee',
            boundaries=boundaries,
        )
        project = Project(
            region=region,
            sources=[
                Source(
                    kind='plane',
                    normal=normal,
                    direction='+',
                    center=xyz((-2.0, 0, 0)),
                    size=xyz((0, width, 1.2)),
                    pulse_cycles=2,
                )
            ],
            monitors=[
                FieldMonitor(id=key, center=xyz((x, 0, 0)), size=xyz((0, width, 1.2)), normal=normal)
                for key, x in [('outside', -2.8), ('near', -1.0), ('far', 1.0)]
            ],
        )
        epsilon = lambda u, v: np.where(abs(u) < 0.25 - 1e-9, 4.0, np.where(abs(abs(u) - 0.25) < 1e-9, 3.125, 2.25))
        return project, epsilon

    def test_modal_injection_block_fused_exact_bit_identity(self):
        """Positive control exercising _ModalSystem and modal injection block-fused kernel vs reference."""
        project, eps_spec = self._build_modal_project(steps=150)
        launch = prepare_modal_launch(project, eps_spec)
        cross_section = np.stack([_profile(launch.epsilon[..., c], 'x') for c in range(3)], axis=-1)
        base = torch.tensor(np.broadcast_to(cross_section, project.region.shape + (3,)).copy())
        freq_hz = [C0 / 1.55e-6]

        monitors = [
            Monitor(id='refl', center=(-2.5, 0, 0), component='Ey'),
            Monitor(id='trans', center=(1.0, 0, 0), component='Ey'),
        ]
        p_modal = project.model_copy(deep=True)
        p_modal.monitors = monitors

        # 1. Direct _ModalSystem execution: block_fused vs reference
        opts_fused = AdjointOptions(execution_profile='block_fused')
        spec_fused = SpectralObservation(base, p_modal.region, [m.component for m in monitors], freq_hz, block_size=32)
        sys_fused = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_fused, observation_monitors=spec_fused.observers)
        rep_fused = {}
        res_fused = _FDTD.apply(base, sys_fused, opts_fused, rep_fused, spec_fused)

        opts_ref = AdjointOptions(execution_profile='reference')
        spec_ref = SpectralObservation(base, p_modal.region, [m.component for m in monitors], freq_hz, block_size=32)
        sys_ref = _ModalSystem(p_modal.model_copy(deep=True), base, launch=launch, options=opts_ref, observation_monitors=spec_ref.observers)
        rep_ref = {}
        res_ref = _FDTD.apply(base, sys_ref, opts_ref, rep_ref, spec_ref)

        T_fused = float((res_fused[:, 1].real**2 + res_fused[:, 1].imag**2).sum().item())
        R_fused = float((res_fused[:, 0].real**2 + res_fused[:, 0].imag**2).sum().item())

        T_ref = float((res_ref[:, 1].real**2 + res_ref[:, 1].imag**2).sum().item())
        R_ref = float((res_ref[:, 0].real**2 + res_ref[:, 0].imag**2).sum().item())

        diff_T = abs(T_fused - T_ref)
        diff_R = abs(R_fused - R_ref)

        self.assertEqual(diff_T, 0.0, f"Modal transmission must be bit-identical (diff={diff_T})")
        self.assertEqual(diff_R, 0.0, f"Modal reflection must be bit-identical (diff={diff_R})")
        self.assertTrue(torch.equal(res_fused, res_ref), "Signals tensor must be bitwise identical")
        self.assertEqual(hash_array(res_fused), hash_array(res_ref))

        # 2. ModeInjectedPlaneSimulation execution: block_fused vs reference
        sim_fused = ModeInjectedPlaneSimulation(project, launch, opts_fused)
        sim_ref = ModeInjectedPlaneSimulation(project, launch, opts_ref)
        with torch.no_grad():
            out_fused = sim_fused(base, freq_hz)
            out_ref = sim_ref(base, freq_hz)

        for key in out_fused:
            f_diff = (out_fused[key].fields - out_ref[key].fields).abs().max().item()
            self.assertEqual(f_diff, 0.0, f"ModeInjectedPlaneSimulation monitor {key} fields must be bit-identical (diff={f_diff})")


class Leg46AutoShutoffAndSpeedupTests(unittest.TestCase):
    """Verifies Verified Auto-Shutoff Accuracy, Tail Savings, and >= 1.35x End-to-End Speedup."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg46_speedup_')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_grating_project(self, steps=4000, block_size=32):
        dx = 0.05
        boundaries = Boundaries(
            x_min=BoundaryFace(kind='pml', layers=8),
            x_max=BoundaryFace(kind='pml', layers=8),
            y_min=BoundaryFace(kind='periodic'),
            y_max=BoundaryFace(kind='periodic'),
        )
        control = RunControl(auto_shutoff=True, min_steps=300, after_source_s=0.0, source_tail_amplitude=1e-5)
        region = Region(
            dimension='2d',
            size=(5.0, 3.0, 0.05),
            mesh=dx,
            precision='float64',
            backend='auto',
            boundaries=boundaries,
            steps=steps,
            run_control=control,
        )
        sources = [
            Source(
                id='src',
                kind='plane',
                normal='x',
                direction='+',
                injection='oneway',
                component='Ey',
                center=(-1.8, 0, 0),
                size=(0, 3.0, 0.05),
                wavelength=1.55,
                pulse_cycles=4,
            )
        ]
        monitors = [
            Monitor(id='box-left', center=(-1.2, 0, 0), component='Ey'),
            Monitor(id='box-right', center=(1.2, 0, 0), component='Ey'),
        ]
        return Project(name='leg46_grating', region=region, sources=sources, monitors=monitors)

    def test_3_auto_shutoff_accuracy_and_1_speedup(self):
        """Positive controls: >= 1.35x speedup and |Delta T| <= 1e-6, |Delta R| <= 1e-6."""
        project = self._build_grating_project(steps=4500)
        freq_hz = [3e8 / 1.55e-6]

        eps = torch.ones(*project.region.shape, 3, dtype=torch.float64)
        eps[40:60, :, :, :] = 3.48**2  # grating slab

        # 1. Full unoptimized baseline:
        # Runs full 4500 steps without auto-shutoff, using legacy observe
        opts_full = AdjointOptions(auto_shutoff=False, allow_auto_shutoff=False)
        spec_full = SpectralObservation(eps, project.region, [m.component for m in project.monitors], freq_hz, block_size=32)
        system_full = _System(project.model_copy(deep=True), eps, observation_monitors=spec_full.observers)
        
        orig_obs = system_full.observe
        def legacy_obs(state=None, out=None):
            res = orig_obs(state=state, out=None)
            if out is not None:
                out.copy_(res)
            return res
        system_full.observe = legacy_obs
        
        rep_full = {}
        t0 = time.perf_counter()
        res_full = _FDTD.apply(eps, system_full, opts_full, rep_full, spec_full)
        t_baseline = time.perf_counter() - t0

        # 2. Optimized run: Block-fused, zero-alloc observe, and verified auto-shutoff
        opts_opt = AdjointOptions(auto_shutoff=True, allow_auto_shutoff=True, execution_profile='block_fused')
        spec_opt = SpectralObservation(eps, project.region, [m.component for m in project.monitors], freq_hz, block_size=32)
        system_opt = _System(project.model_copy(deep=True), eps, observation_monitors=spec_opt.observers)
        
        rep_opt = {}
        t0 = time.perf_counter()
        res_opt = _FDTD.apply(eps, system_opt, opts_opt, rep_opt, spec_opt)
        t_opt = time.perf_counter() - t0

        speedup = t_baseline / max(t_opt, 1e-6)
        print(f"\n[LEG 46 BENCHMARK] Baseline: {t_baseline:.4f}s, Optimized: {t_opt:.4f}s -> Speedup: {speedup:.2f}x")

        # Acceptance Criterion 1: Speedup >= 1.35x
        self.assertGreaterEqual(speedup, 1.35, f"Expected speedup >= 1.35x, got {speedup:.2f}x")

        # Check telemetry facts
        telem = rep_opt.get('telemetry_kernel', {})
        self.assertTrue(telem.get('auto_shutoff_triggered'), "Auto-shutoff must be triggered")
        shutoff_step = telem.get('shutoff_step')
        self.assertIsNotNone(shutoff_step)
        self.assertLess(shutoff_step, 4500, "Shutoff must terminate before full step count")
        self.assertLessEqual(telem.get('shutoff_energy_ratio', 1.0), 1e-9)
        self.assertLessEqual(telem.get('shutoff_flux_delta', 1.0), 1e-6)

        # Acceptance Criterion 3: Accuracy |Delta T| <= 1e-6, |Delta R| <= 1e-6
        T_full = float((res_full[:, 1].real**2 + res_full[:, 1].imag**2).sum().item())
        R_full = float((res_full[:, 0].real**2 + res_full[:, 0].imag**2).sum().item())

        T_opt = float((res_opt[:, 1].real**2 + res_opt[:, 1].imag**2).sum().item())
        R_opt = float((res_opt[:, 0].real**2 + res_opt[:, 0].imag**2).sum().item())

        delta_T = abs(T_opt - T_full)
        delta_R = abs(R_opt - R_full)
        print(f"[LEG 46 ACCURACY] |Delta T|: {delta_T:.3e}, |Delta R|: {delta_R:.3e}")

        self.assertLessEqual(delta_T, 1e-6, f"Transmission delta {delta_T} exceeds 1e-6")
        self.assertLessEqual(delta_R, 1e-6, f"Reflection delta {delta_R} exceeds 1e-6")


class Leg46NegativeControlNotaryGateTests(unittest.TestCase):
    """Verifies Post-Flight Notarization Gates for Auto-Shutoff:

    Negative Control 1: Premature shutoff with E(t)/E_max > 1e-9 -> exit 66 (under_converged_time).
    Negative Control 2: Monitor flux instability delta_Phi > 1e-6 -> exit 66 (under_converged_flux).
    Negative Control 3: Spoofed readout auto-shutoff -> exit 66 (tamper_detected).
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg46_notary_')
        self.post_notary = import_post_notary()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_run_dir(self, telemetry_data, readout_data, config_data=None):
        run_dir = Path(self.test_dir) / 'run_001'
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'solver_telemetry.json').write_text(json.dumps(telemetry_data, indent=2), encoding='utf-8')
        (run_dir / 'readout.json').write_text(json.dumps(readout_data, indent=2), encoding='utf-8')
        if config_data:
            (run_dir / 'config.json').write_text(json.dumps(config_data, indent=2), encoding='utf-8')
        return run_dir

    def test_4_negative_control_1_premature_energy_exit_66(self):
        """Negative Control 1: Premature shutoff with E(t)/E_max > 1e-9 exits 66 (under_converged_time)."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'energy_decay_ratio': 5e-8,
            'auto_shutoff_triggered': True,
            'shutoff_step': 1500,
            'shutoff_energy_ratio': 5e-8,  # > 1e-9!
            'shutoff_flux_delta': 1e-7,
            'monitors': {
                'box-right': {'flux': 0.8},
                'box-left': {'flux': 0.1},
                'box-top': {'flux': 0.05},
                'box-bottom': {'flux': 0.05},
            },
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'auto_shutoff_triggered': True,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'cases': [{'id': 'c1', 'measured': {'T': 0.8, 'R': 0.1}}],
        }
        config = {'decay_floor': 1e-6}
        run_dir = self._setup_run_dir(telemetry, readout, config)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('under_converged_time', str(ctx.exception))
        self.assertIn('auto_shutoff premature energy', str(ctx.exception))

        # Test CLI execution returns exit code 66
        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('under_converged_time', proc.stderr)

    def test_5_negative_control_2_unstable_flux_exit_66(self):
        """Negative Control 2: Monitor flux instability delta_Phi > 1e-6 exits 66 (under_converged_flux)."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'energy_decay_ratio': 1e-10,
            'auto_shutoff_triggered': True,
            'shutoff_step': 1500,
            'shutoff_energy_ratio': 1e-10,  # <= 1e-9 passes
            'shutoff_flux_delta': 2e-5,     # > 1e-6 fails!
            'monitors': {
                'box-right': {'flux': 0.8},
                'box-left': {'flux': 0.1},
                'box-top': {'flux': 0.05},
                'box-bottom': {'flux': 0.05},
            },
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'auto_shutoff_triggered': True,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'cases': [{'id': 'c1', 'measured': {'T': 0.8, 'R': 0.1}}],
        }
        run_dir = self._setup_run_dir(telemetry, readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('under_converged_flux', str(ctx.exception))
        self.assertIn('auto_shutoff unstable flux', str(ctx.exception))

        # Test CLI execution returns exit code 66
        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('under_converged_flux', proc.stderr)

    def test_6_negative_control_3_spoofed_readout_exit_66(self):
        """Negative Control 3: Spoofed readout auto-shutoff tag exits 66 (tamper_detected)."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'energy_decay_ratio': 1e-7,
            'auto_shutoff_triggered': False,  # Kernel says FALSE!
            'monitors': {
                'box-right': {'flux': 0.8},
                'box-left': {'flux': 0.1},
                'box-top': {'flux': 0.05},
                'box-bottom': {'flux': 0.05},
            },
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'auto_shutoff_triggered': True,   # Malicious driver claims TRUE!
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'cases': [{'id': 'c1', 'measured': {'T': 0.8, 'R': 0.1}}],
        }
        run_dir = self._setup_run_dir(telemetry, readout)

        with self.assertRaises(RuntimeError) as ctx:
            self.post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))
        self.assertIn('auto_shutoff contradiction', str(ctx.exception))

        # Test CLI execution returns exit code 66
        cmd = [sys.executable, str(POST_NOTARY), str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 66)
        self.assertIn('tamper_detected', proc.stderr)

    def test_auto_shutoff_successful_notarization_tag(self):
        """Valid auto-shutoff run receives verified:auto_shutoff and convergence tags."""
        telemetry = {
            'schema_version': '1.0',
            'project_canonical_sha256': 'a' * 64,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'energy_decay_ratio': 5e-10,
            'auto_shutoff_triggered': True,
            'shutoff_step': 1500,
            'shutoff_energy_ratio': 5e-10,
            'shutoff_flux_delta': 5e-7,
            'monitors': {
                'box-right': {'flux': 0.8},
                'box-left': {'flux': 0.1},
                'box-top': {'flux': 0.05},
                'box-bottom': {'flux': 0.05},
            },
        }
        readout = {
            'simulation_executed': True,
            'executed_steps': 1500,
            'resumed_from_step': 0,
            'auto_shutoff_triggered': True,
            'epsilon_sha256': 'b' * 64,
            'is_lossless': True,
            'cases': [{'id': 'c1', 'measured': {'T': 0.8, 'R': 0.1}}],
        }
        run_dir = self._setup_run_dir(telemetry, readout)

        cert = self.post_notary.verify_post_flight(run_dir)
        self.assertTrue(cert['verified'])
        self.assertTrue(cert['auto_shutoff_triggered'])
        self.assertIn('verified:auto_shutoff', cert['tags'])
        self.assertIn('verified:energy_decay_1e-9', cert['tags'])
        self.assertIn('verified:flux_stable_1e-6', cert['tags'])


if __name__ == '__main__':
    unittest.main()
