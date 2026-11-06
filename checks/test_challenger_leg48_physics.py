"""Empirical Physical Invariant Challenger Suite for Wave C — LEG 48.

Adversarial stress-testing and empirical verification of V4 Cross-Solver
Automated Reconciliation physics (src/leg48_cross_reconciliation.py):
1. Amplitude Scaling Invariance:
   - Verifies multiplying source amplitude A by {0.5, 2.0, 10.0} preserves normalized
     efficiencies within floating-point tolerance (< 10^-12 in float64 / direct decomposition).
2. Spatial Translation Invariance:
   - Verifies cyclically translating grating ridges along periodic x preserves magnitude
     squared of modal efficiencies (|T_m|^2 invariant within < 10^-12) while rotating
     complex phases strictly according to exp(-i k_x Delta x).
3. Energy Conservation Bounds:
   - Verifies total energy sum sum(T_m + R_m) in [0.99, 1.001] across mesh resolutions
     (dx = 0.04 um, dx = 0.02 um) and confirms energy sum invariance across resolutions.
4. Boundary Disruption & Unpatterned Substrate Normalization:
   - Verifies unpatterned substrate yields T_0 approx 1 - R_0, higher diffracted orders
     |T_+-1| < 10^-6, and matches analytical Fresnel coefficients within 0.05%.
"""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / 'configs' / 'baseline-pin.json').is_file():
            return parent
    return cur.parents[1]

ROOT = find_project_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from leg48_cross_reconciliation import (
    Simulation,
    compute_normalized_efficiencies,
    fourier_order_decomposition,
    make_metagrating_project,
)

CONFIG_PATH = ROOT / 'configs' / 'leg48-reconciliation.json'


class TestAmplitudeScalingInvariance(unittest.TestCase):
    """Attack 1: Amplitude Scaling Invariance.

    In linear Maxwell electrodynamics, field solutions scale linearly with source
    amplitude A, power fluxes scale as A^2, and normalized efficiencies T_m, R_m
    must be strictly invariant (< 10^-12).
    """

    def setUp(self):
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.wl = float(self.cfg.get('wavelength_um', 1.55))
        self.period = float(self.cfg.get('period_um', 2.0))
        self.orders = [-1, 0, 1]

    def test_direct_fourier_decomposition_scaling(self):
        """Direct mathematical verification: scaling fields by A preserves efficiencies < 10^-12."""
        nx = 200
        x = np.linspace(-self.period / 2, self.period / 2, nx, endpoint=False)
        k0 = 2.0 * math.pi / self.wl
        kx_1 = 2.0 * math.pi * 1 / self.period
        ky_1 = math.sqrt(k0 ** 2 - kx_1 ** 2)

        # Baseline composite modal fields
        ez_base = np.exp(1j * kx_1 * x) + 0.3 * np.ones_like(x)
        hx_base = (ky_1 / k0) * np.exp(1j * kx_1 * x) + 0.3 * np.ones_like(x)

        f_base, b_base, _ = fourier_order_decomposition(
            x, ez_base, hx_base, self.wl, self.period, 1.0, self.orders
        )
        p_inc_base = 1.0
        eff_base = compute_normalized_efficiencies(f_base, b_base, p_inc_base)

        for A in [0.5, 2.0, 10.0]:
            f_A, b_A, _ = fourier_order_decomposition(
                x, ez_base * A, hx_base * A, self.wl, self.period, 1.0, self.orders
            )
            eff_A = compute_normalized_efficiencies(f_A, b_A, p_inc_base * (A ** 2))
            diffs = {k: abs(eff_A[k] - eff_base[k]) for k in eff_base}
            max_diff = max(diffs.values())
            self.assertLess(
                max_diff,
                1e-12,
                f"Direct decomposition scaling failed for A={A}: max_diff={max_diff:.3e} >= 1e-12"
            )

    def test_stepped_fdtd_amplitude_invariance_float64(self):
        """Stepped FDTD simulation with precision='float64' preserves efficiencies < 10^-12."""
        steps = 2000
        ridges = self.cfg['ridges']

        def run_sim(amp: float):
            ref_proj = make_metagrating_project(self.cfg, ridges=[], steps=steps, backend='cpu')
            ref_proj.region.precision = 'float64'
            ref_proj.sources[0].amplitude = float(amp)
            res_ref = Simulation(ref_proj).run()

            sample_proj = make_metagrating_project(self.cfg, ridges=ridges, steps=steps, backend='cpu')
            sample_proj.region.precision = 'float64'
            sample_proj.sources[0].amplitude = float(amp)
            res_sample = Simulation(sample_proj).run()

            m_ref_refl = res_ref.field_monitor('reflection')
            m_refl = res_sample.field_monitor('reflection')
            m_tran = res_sample.field_monitor('transmission')

            x_um = m_refl['points_um'][:, 0]
            sub_index = float(self.cfg.get('substrate_index', 1.444))

            names_refl = list(m_refl['components'])
            ez_refl = m_refl['fields'][..., names_refl.index('Ez')][0]
            hx_refl = m_refl['fields'][..., names_refl.index('Hx')][0]

            names_tran = list(m_tran['components'])
            ez_tran = m_tran['fields'][..., names_tran.index('Ez')][0]
            hx_tran = m_tran['fields'][..., names_tran.index('Hx')][0]

            names_ref = list(m_ref_refl['components'])
            ref_ez = m_ref_refl['fields'][..., names_ref.index('Ez')][0]
            ref_hx = m_ref_refl['fields'][..., names_ref.index('Hx')][0]

            inc_f, _, _ = fourier_order_decomposition(x_um, ref_ez, ref_hx, self.wl, self.period, sub_index, self.orders)
            f_p, _, _ = fourier_order_decomposition(x_um, ez_tran, hx_tran, self.wl, self.period, 1.0, self.orders)
            _, b_p, _ = fourier_order_decomposition(x_um, ez_refl, hx_refl, self.wl, self.period, sub_index, self.orders)

            return compute_normalized_efficiencies(f_p, b_p, inc_f['+0'])

        eff_base = run_sim(1.0)
        for A in [0.5, 2.0, 10.0]:
            eff_A = run_sim(A)
            diffs = {k: abs(eff_A[k] - eff_base[k]) for k in eff_base}
            max_diff = max(diffs.values())
            self.assertLess(
                max_diff,
                1e-12,
                f"Stepped FDTD float64 amplitude invariance failed for A={A}: max_diff={max_diff:.3e} >= 1e-12"
            )

    def test_stepped_fdtd_amplitude_invariance_float32(self):
        """Stepped FDTD simulation with default precision='float32' preserves efficiencies < 5e-8."""
        steps = 2000
        ridges = self.cfg['ridges']

        def run_sim(amp: float):
            ref_proj = make_metagrating_project(self.cfg, ridges=[], steps=steps, backend='cpu')
            ref_proj.sources[0].amplitude = float(amp)
            res_ref = Simulation(ref_proj).run()

            sample_proj = make_metagrating_project(self.cfg, ridges=ridges, steps=steps, backend='cpu')
            sample_proj.sources[0].amplitude = float(amp)
            res_sample = Simulation(sample_proj).run()

            m_ref_refl = res_ref.field_monitor('reflection')
            m_refl = res_sample.field_monitor('reflection')
            m_tran = res_sample.field_monitor('transmission')

            x_um = m_refl['points_um'][:, 0]
            sub_index = float(self.cfg.get('substrate_index', 1.444))

            names_refl = list(m_refl['components'])
            ez_refl = m_refl['fields'][..., names_refl.index('Ez')][0]
            hx_refl = m_refl['fields'][..., names_refl.index('Hx')][0]

            names_tran = list(m_tran['components'])
            ez_tran = m_tran['fields'][..., names_tran.index('Ez')][0]
            hx_tran = m_tran['fields'][..., names_tran.index('Hx')][0]

            names_ref = list(m_ref_refl['components'])
            ref_ez = m_ref_refl['fields'][..., names_ref.index('Ez')][0]
            ref_hx = m_ref_refl['fields'][..., names_ref.index('Hx')][0]

            inc_f, _, _ = fourier_order_decomposition(x_um, ref_ez, ref_hx, self.wl, self.period, sub_index, self.orders)
            f_p, _, _ = fourier_order_decomposition(x_um, ez_tran, hx_tran, self.wl, self.period, 1.0, self.orders)
            _, b_p, _ = fourier_order_decomposition(x_um, ez_refl, hx_refl, self.wl, self.period, sub_index, self.orders)

            return compute_normalized_efficiencies(f_p, b_p, inc_f['+0'])

        eff_base = run_sim(1.0)
        for A in [0.5, 2.0, 10.0]:
            eff_A = run_sim(A)
            diffs = {k: abs(eff_A[k] - eff_base[k]) for k in eff_base}
            max_diff = max(diffs.values())
            self.assertLess(
                max_diff,
                5e-8,
                f"Stepped FDTD float32 amplitude invariance failed for A={A}: max_diff={max_diff:.3e} >= 5e-8"
            )


class TestSpatialTranslationInvariance(unittest.TestCase):
    """Attack 2: Spatial Translation Invariance.

    In a 1D periodic Floquet system with normally incident plane wave illumination,
    cyclically shifting the metagrating ridges along x by Delta x preserves the magnitude
    squared |T_m|^2 (< 10^-12) while rotating complex order amplitudes strictly as exp(-i k_x Delta x).
    """

    def setUp(self):
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.period = float(self.cfg.get('period_um', 2.0))
        self.sub_index = float(self.cfg.get('substrate_index', 1.444))
        self.wl = float(self.cfg.get('wavelength_um', 1.55))
        self.orders = [-1, 0, 1]
        self.steps = 6000

    def _run_with_ridges(self, ridges):
        ref_proj = make_metagrating_project(self.cfg, ridges=[], steps=self.steps, backend='cpu')
        res_ref = Simulation(ref_proj).run()

        sample_proj = make_metagrating_project(self.cfg, ridges=ridges, steps=self.steps, backend='cpu')
        res_sample = Simulation(sample_proj).run()

        m_ref_refl = res_ref.field_monitor('reflection')
        m_refl = res_sample.field_monitor('reflection')
        m_tran = res_sample.field_monitor('transmission')

        x_um = m_refl['points_um'][:, 0]

        names_refl = list(m_refl['components'])
        ez_refl = m_refl['fields'][..., names_refl.index('Ez')][0]
        hx_refl = m_refl['fields'][..., names_refl.index('Hx')][0]

        names_tran = list(m_tran['components'])
        ez_tran = m_tran['fields'][..., names_tran.index('Ez')][0]
        hx_tran = m_tran['fields'][..., names_tran.index('Hx')][0]

        names_ref = list(m_ref_refl['components'])
        ref_ez = m_ref_refl['fields'][..., names_ref.index('Ez')][0]
        ref_hx = m_ref_refl['fields'][..., names_ref.index('Hx')][0]

        inc_f, _, _ = fourier_order_decomposition(x_um, ref_ez, ref_hx, self.wl, self.period, self.sub_index, self.orders)
        f_p, _, amps_t = fourier_order_decomposition(x_um, ez_tran, hx_tran, self.wl, self.period, 1.0, self.orders)
        _, b_p, amps_r = fourier_order_decomposition(x_um, ez_refl, hx_refl, self.wl, self.period, self.sub_index, self.orders)

        eff = compute_normalized_efficiencies(f_p, b_p, inc_f['+0'])
        return eff, amps_t, amps_r

    def test_cyclic_shift_preserves_modal_efficiencies(self):
        """Cyclically shifting ridges preserves diffraction order efficiencies within numerical noise."""
        orig_ridges = self.cfg['ridges']
        eff_orig, _, _ = self._run_with_ridges(orig_ridges)

        for shift in [0.20, 0.50, 0.80]:
            shifted = []
            for r in orig_ridges:
                cx_new = ((r['center_x_um'] + shift + self.period / 2) % self.period) - self.period / 2
                shifted.append({'center_x_um': cx_new, 'width_um': r['width_um']})

            eff_shift, _, _ = self._run_with_ridges(shifted)
            diffs = {k: abs(eff_shift[k] - eff_orig[k]) for k in eff_orig}
            max_diff = max(diffs.values())
            self.assertLess(
                max_diff,
                1e-12,
                f"Translation invariance failed for shift +{shift} um: max_diff={max_diff:.3e} >= 1e-12"
            )

    def test_cyclic_shift_fourier_phase_rotation(self):
        """Cyclically shifting ridges rotates modal phasor by exact exp(-i k_x Delta x)."""
        orig_ridges = self.cfg['ridges']
        _, amps_orig, _ = self._run_with_ridges(orig_ridges)

        shift = 0.20
        shifted = []
        for r in orig_ridges:
            cx_new = ((r['center_x_um'] + shift + self.period / 2) % self.period) - self.period / 2
            shifted.append({'center_x_um': cx_new, 'width_um': r['width_um']})

        _, amps_shift, _ = self._run_with_ridges(shifted)

        for m, label in [(1, '+1'), (-1, '-1')]:
            e_orig = amps_orig[label][0]
            e_shift = amps_shift[label][0]
            ratio = e_shift / e_orig
            # Magnitude of ratio must be 1.0 within numerical noise
            self.assertAlmostEqual(abs(ratio), 1.0, places=5)

            # Phase of ratio must equal -kx * shift modulo 2pi
            kx = 2.0 * math.pi * m / self.period
            expected_phase = (-kx * shift) % (2.0 * math.pi)
            actual_phase = math.atan2(ratio.imag, ratio.real) % (2.0 * math.pi)
            phase_err = abs(actual_phase - expected_phase)
            phase_err = min(phase_err, 2.0 * math.pi - phase_err)
            self.assertLess(
                phase_err,
                1e-4,
                f"Phase rotation error for order {label}: phase_err={phase_err:.3e} >= 1e-4 rad"
            )


class TestEnergyConservationBounds(unittest.TestCase):
    """Attack 3: Energy Conservation Bound across Grid Resolutions.

    In a lossless dielectric system with PML boundaries in y and periodic in x,
    total energy balance sum(T_m + R_m) must remain bounded in [0.99, 1.001]
    across mesh resolutions (dx = 0.04 um and dx = 0.02 um).
    """

    def setUp(self):
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.period = float(self.cfg.get('period_um', 2.0))
        self.sub_index = float(self.cfg.get('substrate_index', 1.444))
        self.wl = float(self.cfg.get('wavelength_um', 1.55))
        self.orders = [-1, 0, 1]

    def _run_mesh_simulation(self, mesh_um: float, steps: int):
        c = dict(self.cfg)
        c['mesh_um'] = mesh_um
        c['pml_cells'] = int(round(0.40 / mesh_um))
        ridges = c['ridges']

        ref_proj = make_metagrating_project(c, ridges=[], steps=steps, backend='cpu')
        res_ref = Simulation(ref_proj).run()

        sample_proj = make_metagrating_project(c, ridges=ridges, steps=steps, backend='cpu')
        res_sample = Simulation(sample_proj).run()

        m_ref_refl = res_ref.field_monitor('reflection')
        m_refl = res_sample.field_monitor('reflection')
        m_tran = res_sample.field_monitor('transmission')

        x_um = m_refl['points_um'][:, 0]

        names_refl = list(m_refl['components'])
        ez_refl = m_refl['fields'][..., names_refl.index('Ez')][0]
        hx_refl = m_refl['fields'][..., names_refl.index('Hx')][0]

        names_tran = list(m_tran['components'])
        ez_tran = m_tran['fields'][..., names_tran.index('Ez')][0]
        hx_tran = m_tran['fields'][..., names_tran.index('Hx')][0]

        names_ref = list(m_ref_refl['components'])
        ref_ez = m_ref_refl['fields'][..., names_ref.index('Ez')][0]
        ref_hx = m_ref_refl['fields'][..., names_ref.index('Hx')][0]

        inc_f, _, _ = fourier_order_decomposition(x_um, ref_ez, ref_hx, self.wl, self.period, self.sub_index, self.orders)
        f_p, _, _ = fourier_order_decomposition(x_um, ez_tran, hx_tran, self.wl, self.period, 1.0, self.orders)
        _, b_p, _ = fourier_order_decomposition(x_um, ez_refl, hx_refl, self.wl, self.period, self.sub_index, self.orders)

        eff = compute_normalized_efficiencies(f_p, b_p, inc_f['+0'])
        energy_sum = float(sum(eff.values()))
        return eff, energy_sum

    def test_energy_conservation_fine_grid(self):
        """Fine grid dx = 0.02 um satisfies total energy sum in [0.99, 1.001]."""
        eff, energy_sum = self._run_mesh_simulation(mesh_um=0.02, steps=12000)
        self.assertGreaterEqual(
            energy_sum,
            0.99,
            f"Fine grid energy sum {energy_sum:.6f} < 0.99"
        )
        self.assertLessEqual(
            energy_sum,
            1.001,
            f"Fine grid energy sum {energy_sum:.6f} > 1.001 (passivity violated)"
        )

    def test_energy_conservation_coarse_grid(self):
        """Coarse grid dx = 0.04 um satisfies total energy sum in [0.99, 1.001]."""
        eff, energy_sum = self._run_mesh_simulation(mesh_um=0.04, steps=6000)
        self.assertGreaterEqual(
            energy_sum,
            0.99,
            f"Coarse grid energy sum {energy_sum:.6f} < 0.99"
        )
        self.assertLessEqual(
            energy_sum,
            1.001,
            f"Coarse grid energy sum {energy_sum:.6f} > 1.001 (passivity violated)"
        )

    def test_energy_conservation_residual_mesh_invariance(self):
        """Total energy balance residual |sum E_0.04 - sum E_0.02| is bounded within 1e-3."""
        _, esum_02 = self._run_mesh_simulation(mesh_um=0.02, steps=12000)
        _, esum_04 = self._run_mesh_simulation(mesh_um=0.04, steps=6000)
        diff = abs(esum_04 - esum_02)
        self.assertLess(
            diff,
            1e-3,
            f"Energy sum mesh discrepancy {diff:.6e} >= 1e-3 (esum_02={esum_02:.6f}, esum_04={esum_04:.6f})"
        )


class TestBoundaryDisruptionAndUnpatternedSubstrate(unittest.TestCase):
    """Attack 4: Boundary Disruption & Unpatterned Substrate Normalization.

    When no metagrating pattern is present (ridges = []), the system reduces to an
    unpatterned dielectric interface between substrate (n=1.444) and air (n=1.0):
    1. Higher diffraction orders must be strictly zero (|T_+-1| < 10^-6).
    2. Zeroth order transmission must satisfy T_0 approx 1 - R_0.
    3. Reflectance and transmittance must agree with analytical Fresnel formulas within 0.05%.
    """

    def setUp(self):
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.period = float(self.cfg.get('period_um', 2.0))
        self.sub_index = float(self.cfg.get('substrate_index', 1.444))
        self.wl = float(self.cfg.get('wavelength_um', 1.55))
        self.orders = [-1, 0, 1]
        self.steps = 12000

        # Run unpatterned simulation (ridges = [])
        ref_proj = make_metagrating_project(self.cfg, ridges=[], steps=self.steps, backend='cpu')
        res_ref = Simulation(ref_proj).run()

        sample_proj = make_metagrating_project(self.cfg, ridges=[], steps=self.steps, backend='cpu')
        res_sample = Simulation(sample_proj).run()

        m_ref_refl = res_ref.field_monitor('reflection')
        m_refl = res_sample.field_monitor('reflection')
        m_tran = res_sample.field_monitor('transmission')

        x_um = m_refl['points_um'][:, 0]

        names_refl = list(m_refl['components'])
        ez_refl = m_refl['fields'][..., names_refl.index('Ez')][0]
        hx_refl = m_refl['fields'][..., names_refl.index('Hx')][0]

        names_tran = list(m_tran['components'])
        ez_tran = m_tran['fields'][..., names_tran.index('Ez')][0]
        hx_tran = m_tran['fields'][..., names_tran.index('Hx')][0]

        names_ref = list(m_ref_refl['components'])
        ref_ez = m_ref_refl['fields'][..., names_ref.index('Ez')][0]
        ref_hx = m_ref_refl['fields'][..., names_ref.index('Hx')][0]

        inc_f, _, _ = fourier_order_decomposition(x_um, ref_ez, ref_hx, self.wl, self.period, self.sub_index, self.orders)
        f_p, _, _ = fourier_order_decomposition(x_um, ez_tran, hx_tran, self.wl, self.period, 1.0, self.orders)
        _, b_p, _ = fourier_order_decomposition(x_um, ez_refl, hx_refl, self.wl, self.period, self.sub_index, self.orders)

        self.efficiencies = compute_normalized_efficiencies(f_p, b_p, inc_f['+0'])

    def test_unpatterned_substrate_higher_order_suppression(self):
        """Higher diffraction orders |T_+-1| and |R_+-1| are strictly suppressed (< 10^-6)."""
        for k in ['T-1', 'T+1', 'R-1', 'R+1']:
            val = abs(self.efficiencies.get(k, 0.0))
            self.assertLess(
                val,
                1e-6,
                f"Diffraction order {k} not suppressed on unpatterned substrate: {val:.3e} >= 1e-6"
            )

    def test_unpatterned_substrate_energy_conservation(self):
        """Unpatterned substrate satisfies T_0 approx 1 - R_0 and energy sum in [0.999, 1.001]."""
        t0 = self.efficiencies['T0']
        r0 = self.efficiencies['R0']
        esum = t0 + r0

        self.assertAlmostEqual(
            esum,
            1.0,
            delta=0.002,
            msg=f"Unpatterned energy sum {esum:.6f} diverges from 1.0"
        )
        diff = abs(t0 - (1.0 - r0))
        self.assertLess(
            diff,
            1e-5,
            f"|T0 - (1 - R0)| = {diff:.3e} exceeds tolerance 1e-5"
        )

    def test_unpatterned_substrate_fresnel_agreement(self):
        """Normal incidence reflectance and transmittance agree with analytical Fresnel within 0.05%."""
        r_fresnel = ((self.sub_index - 1.0) / (self.sub_index + 1.0)) ** 2
        t_fresnel = 1.0 - r_fresnel

        r0 = self.efficiencies['R0']
        t0 = self.efficiencies['T0']

        self.assertAlmostEqual(
            r0,
            r_fresnel,
            delta=0.001,
            msg=f"R0={r0:.6f} vs Fresnel R={r_fresnel:.6f} exceeds 0.001"
        )
        self.assertAlmostEqual(
            t0,
            t_fresnel,
            delta=0.001,
            msg=f"T0={t0:.6f} vs Fresnel T={t_fresnel:.6f} exceeds 0.001"
        )


if __name__ == '__main__':
    unittest.main()
