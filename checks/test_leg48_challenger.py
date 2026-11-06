"""Adversarial Empirical Challenge Suite for Wave C — LEG 48:
V4 Cross-Solver Reconciliation Gate & Normalization Invariance.

Role: EMPIRICAL CHALLENGER (challenger_leg48_1)
Mission: Stress-test Fourier decomposition, power normalization invariance,
Parseval energy completeness, high-contrast dielectric stress, and discrepancy
metric edge boundary enforcement.

Attacks Implemented:
- Attack 1: Source Amplitude Scaling Invariance (real, complex, extreme scales).
- Attack 2: Spatial Shift Invariance (discrete cyclic roll and continuous coordinate shift).
- Attack 3: Parseval / Energy Completeness (spatial integrated Poynting flux vs mode powers).
- Attack 4: High-Contrast Geometric Stress (extreme index contrasts, Rayleigh cutoff boundary).
- Attack 5: Discrepancy Bound Edge Testing (tight tolerance boundary step-function & tamper gates).
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
import unittest

import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'checks')):
    if p not in sys.path:
        sys.path.insert(0, p)

import leg48_cross_reconciliation as driver
import post_notary


def create_mock_run_directory(
    base_dir: Path,
    name: str,
    telemetry: dict,
    readout: dict,
    config: dict
) -> Path:
    """Helper to set up a mock run directory satisfying pre-requisite notary checks."""
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(config)
    cfg.setdefault('kind', 'reconciliation')
    cfg.setdefault('reconciliation_tolerance', 0.015)
    cfg.setdefault('tau_box', 0.05)

    default_monitors = {
        'box-right': {'flux': 0.8876},
        'box-left': {'flux': 0.1112},
        'top': {'flux': 0.0006},
        'bottom': {'flux': 0.0006}
    }

    tel = copy.deepcopy(telemetry)
    tel.setdefault('schema_version', '1.0')
    tel.setdefault('executed_steps', 4000)
    tel.setdefault('is_lossless', True)
    if 'monitors' not in tel:
        tel['monitors'] = copy.deepcopy(default_monitors)

    ro = copy.deepcopy(readout)
    ro.setdefault('simulation_executed', True)
    ro.setdefault('executed_steps', 4000)
    ro.setdefault('is_lossless', True)
    if 'monitors' not in ro:
        ro['monitors'] = copy.deepcopy(tel['monitors'])
    else:
        for k in ('box-right', 'box-left', 'top', 'bottom'):
            if k not in ro['monitors']:
                ro['monitors'][k] = default_monitors[k]
            if k not in tel['monitors']:
                tel['monitors'][k] = ro['monitors'][k]

    (run_dir / 'solver_telemetry.json').write_text(json.dumps(tel, indent=2), encoding='utf-8')
    (run_dir / 'readout.json').write_text(json.dumps(ro, indent=2), encoding='utf-8')
    (run_dir / 'config.input.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return run_dir


class Attack1SourceAmplitudeScalingInvariance(unittest.TestCase):
    """Attack 1: Verify source amplitude scaling invariance across real and complex scales."""

    def setUp(self):
        self.wavelength = 1.55
        self.period = 2.0
        self.medium_index = 1.444
        self.orders = [-1, 0, 1]
        self.nx = 200
        self.x = np.linspace(-self.period / 2, self.period / 2, self.nx, endpoint=False)
        self.k0 = 2.0 * math.pi / self.wavelength

        # Base physical fields with multi-order propagating modes
        # m = 0
        ky0 = self.medium_index * self.k0
        ez0 = 1.0 * np.ones_like(self.x, dtype=np.complex128)
        hx0 = (ky0 / self.k0) * ez0

        # m = +1
        kx1 = 2.0 * math.pi * 1 / self.period
        ky1 = math.sqrt((self.medium_index * self.k0) ** 2 - kx1 ** 2)
        ez1 = 0.5 * np.exp(1j * kx1 * self.x)
        hx1 = 0.5 * (ky1 / self.k0) * np.exp(1j * kx1 * self.x)

        # m = -1
        kx_m1 = 2.0 * math.pi * (-1) / self.period
        ky_m1 = math.sqrt((self.medium_index * self.k0) ** 2 - kx_m1 ** 2)
        ez_m1 = 0.3 * np.exp(1j * kx_m1 * self.x)
        hx_m1 = 0.3 * (ky_m1 / self.k0) * np.exp(1j * kx_m1 * self.x)

        self.ez_base = ez0 + ez1 + ez_m1
        self.hx_base = hx0 + hx1 + hx_m1
        self.p_inc_base = 1.5

        # Base decomposition and efficiencies
        self.f_base, self.b_base, _ = driver.fourier_order_decomposition(
            self.x, self.ez_base, self.hx_base, self.wavelength, self.period, self.medium_index, self.orders
        )
        self.eff_base = driver.compute_normalized_efficiencies(self.f_base, self.b_base, self.p_inc_base)

    def test_real_amplitude_scaling_invariance(self):
        """Test scaling factors across 11 orders of magnitude (1e-6 to 1e5)."""
        scale_factors = [1e-6, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e5]

        for alpha in scale_factors:
            ez_scaled = alpha * self.ez_base
            hx_scaled = alpha * self.hx_base
            p_inc_scaled = (alpha ** 2) * self.p_inc_base

            f_scaled, b_scaled, _ = driver.fourier_order_decomposition(
                self.x, ez_scaled, hx_scaled, self.wavelength, self.period, self.medium_index, self.orders
            )

            # Unnormalized power must scale strictly as alpha^2
            for m in self.orders:
                label = f"{m:+d}"
                expected_p = (alpha ** 2) * self.f_base[label]
                rel_diff = abs(f_scaled[label] - expected_p) / max(expected_p, 1e-15)
                self.assertLess(rel_diff, 1e-13, f"Power scaling failure for m={label} at alpha={alpha}")

            # Normalized efficiencies must remain exactly invariant (< 1e-12)
            eff_scaled = driver.compute_normalized_efficiencies(f_scaled, b_scaled, p_inc_scaled)
            for key, val in self.eff_base.items():
                abs_diff = abs(eff_scaled[key] - val)
                self.assertLess(abs_diff, 1e-13, f"Efficiency invariance violated for {key} at alpha={alpha}: diff={abs_diff}")

    def test_complex_phase_and_scaling_invariance(self):
        """Test scaling combined with arbitrary complex phase rotations e^(i*theta)."""
        thetas = [0.0, math.pi / 6, math.pi / 4, math.pi / 2, 2.345, math.pi]
        alphas = [0.05, 1.0, 10.0]

        for theta in thetas:
            for alpha in alphas:
                phasor = alpha * np.exp(1j * theta)
                ez_scaled = phasor * self.ez_base
                hx_scaled = phasor * self.hx_base
                p_inc_scaled = (abs(phasor) ** 2) * self.p_inc_base

                f_scaled, b_scaled, _ = driver.fourier_order_decomposition(
                    self.x, ez_scaled, hx_scaled, self.wavelength, self.period, self.medium_index, self.orders
                )
                eff_scaled = driver.compute_normalized_efficiencies(f_scaled, b_scaled, p_inc_scaled)

                for key, val in self.eff_base.items():
                    abs_diff = abs(eff_scaled[key] - val)
                    self.assertLess(abs_diff, 1e-13, f"Phase invariance violated for {key} at theta={theta}, alpha={alpha}")

    def test_cross_solver_discrepancy_invariance_under_scaling(self):
        """Verify that discrepancy against external reference is invariant under amplitude scaling."""
        mock_ext_eff = {k: v + 0.001 * (idx + 1) for idx, (k, v) in enumerate(self.eff_base.items())}
        base_disc, _, _ = driver.calculate_cross_solver_discrepancy(self.eff_base, mock_ext_eff)

        for alpha in [0.05, 1.0, 10.0, 100.0]:
            ez_scaled = alpha * self.ez_base
            hx_scaled = alpha * self.hx_base
            p_inc_scaled = (alpha ** 2) * self.p_inc_base

            f_scaled, b_scaled, _ = driver.fourier_order_decomposition(
                self.x, ez_scaled, hx_scaled, self.wavelength, self.period, self.medium_index, self.orders
            )
            eff_scaled = driver.compute_normalized_efficiencies(f_scaled, b_scaled, p_inc_scaled)
            disc_scaled, _, _ = driver.calculate_cross_solver_discrepancy(eff_scaled, mock_ext_eff)

            self.assertAlmostEqual(base_disc, disc_scaled, places=13)


class Attack2SpatialShiftInvariance(unittest.TestCase):
    """Attack 2: Verify spatial shift invariance across full grating period Lambda = 2.0 um."""

    def setUp(self):
        self.period = 2.0
        self.wavelength = 1.55
        self.medium_index = 1.0  # Air monitor line
        self.k0 = 2.0 * math.pi / self.wavelength
        self.nx = 200
        self.x = np.linspace(-self.period / 2, self.period / 2, self.nx, endpoint=False)
        self.orders = [-1, 0, 1]

        # Multi-harmonic superposition
        # Orders -1, 0, +1
        self.c_plus = {
            -1: 0.35 * np.exp(1j * 0.4),
            0: 0.80 * np.exp(-1j * 0.2),
            1: 0.50 * np.exp(1j * 1.1)
        }

        ez = np.zeros(self.nx, dtype=np.complex128)
        hx = np.zeros(self.nx, dtype=np.complex128)

        for m, amp in self.c_plus.items():
            kx = 2.0 * math.pi * m / self.period
            ky = math.sqrt((self.medium_index * self.k0) ** 2 - kx ** 2)
            spatial_phase = np.exp(1j * kx * self.x)
            ez += amp * spatial_phase
            hx += amp * (ky / self.k0) * spatial_phase

        self.ez = ez
        self.hx = hx

    def test_discrete_cyclic_roll_invariance(self):
        """Cyclically roll periodic field arrays by integer grid steps."""
        roll_steps = [0, 1, 10, 25, 50, 100, 150, 199]

        f_base, b_base, amps_base = driver.fourier_order_decomposition(
            self.x, self.ez, self.hx, self.wavelength, self.period, self.medium_index, self.orders
        )

        for roll in roll_steps:
            ez_rolled = np.roll(self.ez, roll)
            hx_rolled = np.roll(self.hx, roll)

            f_rolled, b_rolled, amps_rolled = driver.fourier_order_decomposition(
                self.x, ez_rolled, hx_rolled, self.wavelength, self.period, self.medium_index, self.orders
            )

            # Order powers |c_m|^2 must remain strictly invariant
            for m in self.orders:
                label = f"{m:+d}"
                base_power = f_base[label]
                rolled_power = f_rolled[label]
                abs_diff = abs(base_power - rolled_power)
                self.assertLess(
                    abs_diff, 1e-13,
                    f"Power invariance violated for order {label} at roll={roll}: diff={abs_diff}"
                )

    def test_continuous_spatial_shift_analytical(self):
        """Analytically shift coordinates by arbitrary Delta x across period [0, Lambda]."""
        shifts_um = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

        f_base, b_base, amps_base = driver.fourier_order_decomposition(
            self.x, self.ez, self.hx, self.wavelength, self.period, self.medium_index, self.orders
        )

        for dx in shifts_um:
            # Shifted field: Ez(x + dx), Hx(x + dx)
            ez_shifted = np.zeros(self.nx, dtype=np.complex128)
            hx_shifted = np.zeros(self.nx, dtype=np.complex128)

            for m, amp in self.c_plus.items():
                kx = 2.0 * math.pi * m / self.period
                ky = math.sqrt((self.medium_index * self.k0) ** 2 - kx ** 2)
                spatial_phase = np.exp(1j * kx * (self.x + dx))
                ez_shifted += amp * spatial_phase
                hx_shifted += amp * (ky / self.k0) * spatial_phase

            f_shifted, b_shifted, amps_shifted = driver.fourier_order_decomposition(
                self.x, ez_shifted, hx_shifted, self.wavelength, self.period, self.medium_index, self.orders
            )

            for m in self.orders:
                label = f"{m:+d}"
                # Power must be strictly invariant (< 1e-13)
                abs_p_diff = abs(f_shifted[label] - f_base[label])
                self.assertLess(abs_p_diff, 1e-13, f"Power diff {abs_p_diff} at dx={dx} for order {label}")

                # Complex amplitude must have exact phase rotation exp(+i * kx * dx)
                kx = 2.0 * math.pi * m / self.period
                expected_e_plus = amps_base[label][0] * np.exp(1j * kx * dx)
                actual_e_plus = amps_shifted[label][0]
                amp_diff = abs(actual_e_plus - expected_e_plus)
                self.assertLess(amp_diff, 1e-13, f"Complex amplitude phase mismatch {amp_diff} at dx={dx} for {label}")


class Attack3ParsevalEnergyCompleteness(unittest.TestCase):
    """Attack 3: Verify Parseval energy completeness: sum of mode powers = spatial integrated Poynting flux."""

    def setUp(self):
        self.period = 2.0
        self.wavelength = 1.55
        self.medium_index = 1.444
        self.k0 = 2.0 * math.pi / self.wavelength
        self.nx = 400
        self.x = np.linspace(-self.period / 2, self.period / 2, self.nx, endpoint=False)

        # Propagating orders for n = 1.444, period = 2.0, lambda = 1.55:
        # k0 = 4.05367. n*k0 = 5.8535.
        # kx(m) = 2*pi*m / 2.0 = pi*m.
        # m = 0: kx = 0 <= 5.85 (prop)
        # m = +/-1: kx = 3.1416 <= 5.85 (prop)
        # m = +/-2: kx = 6.283 > 5.85 (evanescent)
        self.prop_orders = [-1, 0, 1]
        self.all_orders = [-2, -1, 0, 1, 2]

    def test_parseval_theorem_bidirectional_propagating_modes(self):
        """Verify modal power sum equals spatial integrated Poynting flux for arbitrary bidirectional waves."""
        # Forward (+y) and backward (-y) modal amplitudes
        c_forward = {
            -1: 0.25 * np.exp(1j * 0.3),
            0: 0.70 * np.exp(-1j * 0.5),
            1: 0.40 * np.exp(1j * 0.9)
        }
        c_backward = {
            -1: 0.10 * np.exp(-1j * 0.7),
            0: 0.15 * np.exp(1j * 0.2),
            1: 0.05 * np.exp(1j * 1.4)
        }

        ez = np.zeros(self.nx, dtype=np.complex128)
        hx = np.zeros(self.nx, dtype=np.complex128)

        for m in self.prop_orders:
            kx = 2.0 * math.pi * m / self.period
            ky = math.sqrt((self.medium_index * self.k0) ** 2 - kx ** 2)
            phase = np.exp(1j * kx * self.x)

            # E_z = E_+ + E_-
            # H_x = (ky/k0) * (E_+ - E_-)
            ez += (c_forward[m] + c_backward[m]) * phase
            hx += (ky / self.k0) * (c_forward[m] - c_backward[m]) * phase

        # Physical time-averaged spatial Poynting flux: <Sy> = 0.5 * Re(Ez * Hx*)
        sy_spatial = 0.5 * np.real(ez * np.conj(hx))
        mean_spatial_flux = float(np.mean(sy_spatial))

        # Modal decomposition
        f_powers, b_powers, _ = driver.fourier_order_decomposition(
            self.x, ez, hx, self.wavelength, self.period, self.medium_index, self.prop_orders
        )

        modal_net_flux = sum(f_powers[f"{m:+d}"] - b_powers[f"{m:+d}"] for m in self.prop_orders)

        abs_diff = abs(mean_spatial_flux - modal_net_flux)
        self.assertLess(
            abs_diff, 1e-13,
            f"Parseval identity violated: spatial={mean_spatial_flux:.12e}, modal={modal_net_flux:.12e}, diff={abs_diff:.12e}"
        )

    def test_evanescent_modes_zero_real_poynting_flux(self):
        """Verify that evanescent modes carry zero net time-averaged Poynting flux and are safely decoupled."""
        # Add an evanescent order (m = +2, where kx > n*k0)
        m_evan = 2
        kx_evan = 2.0 * math.pi * m_evan / self.period
        ky2 = (self.medium_index * self.k0) ** 2 - kx_evan ** 2
        self.assertLess(ky2, 0.0, "m=2 must be evanescent")
        alpha_y = math.sqrt(-ky2)

        # For an evanescent wave decaying in +y: Ez ~ e^(-alpha_y * y) e^(i kx x)
        # Hx = (1 / (i k0)) dEz/dy = (-alpha_y / (i k0)) Ez = i (alpha_y / k0) Ez
        amp_evan = 0.8 * np.exp(1j * 0.4)
        phase_evan = np.exp(1j * kx_evan * self.x)
        ez_evan = amp_evan * phase_evan
        hx_evan = 1j * (alpha_y / self.k0) * ez_evan

        # Poynting flux of pure evanescent wave: 0.5 * Re(Ez * Hx*)
        # Since Hx is 90 degrees out of phase with Ez, Re(Ez * Hx*) == 0 identically!
        sy_evan = 0.5 * np.real(ez_evan * np.conj(hx_evan))
        self.assertLess(np.max(np.abs(sy_evan)), 1e-14, "Evanescent wave has nonzero time-averaged Poynting flux!")

        # Superpose propagating and evanescent wave
        m_prop = 0
        ky_prop = self.medium_index * self.k0
        amp_prop = 1.0 + 0.0j
        ez_prop = amp_prop * np.ones_like(self.x, dtype=np.complex128)
        hx_prop = (ky_prop / self.k0) * ez_prop

        ez_total = ez_prop + ez_evan
        hx_total = hx_prop + hx_evan

        # Spatial average of total Poynting flux
        sy_total = 0.5 * np.real(ez_total * np.conj(hx_total))
        mean_total_flux = float(np.mean(sy_total))

        # Expected propagating flux: 0.5 * (ky/k0) * |amp_prop|^2
        expected_flux = 0.5 * (ky_prop / self.k0) * (abs(amp_prop) ** 2)
        self.assertAlmostEqual(mean_total_flux, expected_flux, places=13)

        # In decomposition, evanescent order should report zero power
        f_powers, b_powers, _ = driver.fourier_order_decomposition(
            self.x, ez_total, hx_total, self.wavelength, self.period, self.medium_index, orders=[0, 2]
        )
        self.assertEqual(f_powers['+2'], 0.0)
        self.assertEqual(b_powers['+2'], 0.0)
        self.assertAlmostEqual(f_powers['+0'], expected_flux, places=13)


class Attack4HighContrastGeometricStress(unittest.TestCase):
    """Attack 4: High-contrast dielectric stress and Rayleigh cutoff boundary handling."""

    def test_rayleigh_cutoff_boundary_stability(self):
        """Verify behavior right at the Rayleigh cutoff (ky^2 = 0) where kx = n*k0."""
        wavelength = 1.55
        k0 = 2.0 * math.pi / wavelength
        # Choose period so that order m = 1 has kx exactly equal to n*k0
        medium_index = 2.0
        # kx = 2*pi*m / period = 2*pi / period = n * 2*pi / wavelength => period = wavelength / n
        period = wavelength / medium_index

        nx = 200
        x = np.linspace(-period / 2, period / 2, nx, endpoint=False)
        ez = np.exp(1j * (2.0 * math.pi / period) * x)
        hx = np.ones_like(ez)

        # Exact cutoff: ky2 = (n*k0)^2 - kx^2 == 0.0
        # Must not raise ZeroDivisionError and safely mark order as non-propagating
        try:
            f_powers, b_powers, amps = driver.fourier_order_decomposition(
                x, ez, hx, wavelength, period, medium_index, orders=[1]
            )
            self.assertEqual(f_powers['+1'], 0.0)
            self.assertEqual(b_powers['+1'], 0.0)
        except ZeroDivisionError:
            self.fail("fourier_order_decomposition raised ZeroDivisionError at exact Rayleigh cutoff ky^2 = 0!")

    def test_extreme_dielectric_contrasts(self):
        """Test metagrating benchmark simulation with extreme permittivity contrasts (e.g. n=5.0, eps=25.0)."""
        extreme_cfg = {
            'period_um': 2.0,
            'wavelength_um': 1.55,
            'substrate_index': 1.0,   # Air substrate
            'ridge_index': 5.0,       # Extreme contrast dielectric (eps = 25.0)
            'ridge_height_um': 0.50,
            'ridges': [
                {'center_x_um': -0.93, 'width_um': 0.08},
                {'center_x_um': -0.32, 'width_um': 0.22}
            ],
            'reconciliation_tolerance': 0.015
        }

        # Simulation should produce deterministic non-NaN outputs
        efficiencies, energy_balance, eps_sha = driver.simulate_metagrating_benchmark(extreme_cfg)
        self.assertTrue(math.isfinite(energy_balance))
        self.assertEqual(len(eps_sha), 64)
        for k, v in efficiencies.items():
            self.assertTrue(math.isfinite(v), f"Order {k} has non-finite efficiency: {v}")

        # Check export functions with extreme contrast
        meep_exp = driver.export_geometry_to_meep(extreme_cfg)
        self.assertEqual(meep_exp['geometry_blocks'][1]['material_epsilon'], 25.0)

        rcwa_exp = driver.export_geometry_to_rcwa(extreme_cfg)
        self.assertEqual(rcwa_exp['patterned_layer']['ridge_epsilon'], 25.0)


class Attack5DiscrepancyBoundEdgeTesting(unittest.TestCase):
    """Attack 5: Strict step-function boundary enforcement of discrepancy tolerance and tamper rejection."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='leg48_challenger_attack5_')
        self.cfg = {
            'kind': 'reconciliation',
            'reconciliation_tolerance': 0.015,
            'tau_box': 0.05
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tolerance_boundary_step_function(self):
        """Verify strict step-function behavior at tolerance = 0.015.

        - 0.01499: PASS
        - 0.0149999999: PASS
        - 0.0150000000: PASS (<= tolerance)
        - 0.0150000001: FAIL (> tolerance, raises cross_solver_discrepancy_exceeded)
        - 0.01501: FAIL
        """
        passing_values = [0.01499, 0.0149999999, 0.0150000000]
        failing_values = [0.0150000001, 0.01501, 0.0200]

        for disc in passing_values:
            readout = {
                'cross_solver': {
                    'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                    'max_discrepancy': disc,
                    'tolerance': 0.015,
                }
            }
            telemetry = {
                'executed_steps': 4000,
                'is_lossless': True,
                'cross_solver': {'max_discrepancy': disc}
            }
            run_dir = create_mock_run_directory(
                Path(self.test_dir), f'pass_{disc:.8f}', telemetry, readout, self.cfg
            )
            cert = post_notary.verify_post_flight(run_dir)
            self.assertTrue(cert['verified'], f"Discrepancy {disc} should have passed")
            self.assertTrue(cert['cross_solver_reconciliation']['reconciled'])

        for disc in failing_values:
            readout = {
                'cross_solver': {
                    'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                    'max_discrepancy': disc,
                    'tolerance': 0.015,
                }
            }
            telemetry = {
                'executed_steps': 4000,
                'is_lossless': True,
                'cross_solver': {'max_discrepancy': disc}
            }
            run_dir = create_mock_run_directory(
                Path(self.test_dir), f'fail_{disc:.8f}', telemetry, readout, self.cfg
            )
            with self.assertRaises(RuntimeError) as ctx:
                post_notary.verify_post_flight(run_dir)
            self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))

    def test_covert_order_efficiency_tampering_caught(self):
        """Attacker reports max_discrepancy = 0.010, but individual order discrepancy is 0.025.

        Check 10 must intercept the discrepancy and raise cross_solver_discrepancy_exceeded.
        """
        readout = {
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.010,  # Falsified small max discrepancy
                'tolerance': 0.015,
                'torchfdtd_efficiencies': {
                    'T-1': 0.040, 'T0': 0.071, 'T+1': 0.776,
                    'R-1': 0.045, 'R0': 0.016, 'R+1': 0.051
                },
                'external_efficiencies': {
                    'T-1': 0.040, 'T0': 0.071,
                    'T+1': 0.751,  # Diff is 0.025 > 0.015!
                    'R-1': 0.045, 'R0': 0.016, 'R+1': 0.051
                }
            }
        }
        telemetry = {
            'executed_steps': 4000,
            'is_lossless': True,
            'cross_solver': {'max_discrepancy': 0.010}
        }
        run_dir = create_mock_run_directory(
            Path(self.test_dir), 'covert_tamper', telemetry, readout, self.cfg
        )
        with self.assertRaises(RuntimeError) as ctx:
            post_notary.verify_post_flight(run_dir)
        self.assertIn('cross_solver_discrepancy_exceeded', str(ctx.exception))

    def test_telemetry_discrepancy_mismatch_caught(self):
        """Discrepancy reported in kernel telemetry differs from readout by > FLUX_TOLERANCE."""
        readout = {
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.010,
                'tolerance': 0.015,
            }
        }
        telemetry = {
            'executed_steps': 4000,
            'is_lossless': True,
            'cross_solver': {'max_discrepancy': 0.012}  # Mismatch 0.002 > 1e-5
        }
        run_dir = create_mock_run_directory(
            Path(self.test_dir), 'tel_mismatch', telemetry, readout, self.cfg
        )
        with self.assertRaises(RuntimeError) as ctx:
            post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))

    def test_external_solver_abnormal_termination_codes(self):
        """External solver exit code != 0 (e.g. exit 1, exit 137 OOM) must be rejected."""
        error_exit_codes = [1, 2, 127, 137, 255]

        for code in error_exit_codes:
            readout = {
                'cross_solver': {
                    'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': code},
                    'max_discrepancy': 0.005,
                    'tolerance': 0.015,
                }
            }
            telemetry = {
                'executed_steps': 4000,
                'is_lossless': True,
                'cross_solver': {'max_discrepancy': 0.005}
            }
            run_dir = create_mock_run_directory(
                Path(self.test_dir), f'ext_exit_{code}', telemetry, readout, self.cfg
            )
            with self.assertRaises(RuntimeError) as ctx:
                post_notary.verify_post_flight(run_dir)
            self.assertIn('tamper_detected', str(ctx.exception))

    def test_zero_executed_steps_fraud_rejected(self):
        """Simulation reporting 0 executed steps in kernel telemetry must be rejected."""
        readout = {
            'cross_solver': {
                'external_solver': {'name': 'meep', 'version': '1.34.0', 'exit_code': 0},
                'max_discrepancy': 0.005,
                'tolerance': 0.015,
            }
        }
        telemetry = {
            'executed_steps': 0,  # Zero steps executed
            'is_lossless': True,
            'cross_solver': {'max_discrepancy': 0.005}
        }
        run_dir = create_mock_run_directory(
            Path(self.test_dir), 'zero_steps_fraud', telemetry, readout, self.cfg
        )
        with self.assertRaises(RuntimeError) as ctx:
            post_notary.verify_post_flight(run_dir)
        self.assertIn('tamper_detected', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
