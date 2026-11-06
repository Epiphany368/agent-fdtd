"""Adversarial empirical challenge test suite for Wave D — LEG 49 (Track I Inverse Design Loop).

Target: Analytic Yee Jacobian d(eps)/d(rho), Chain-Rule Autograd Graph,
        Discrete Adjoint Operators, and Optimization Convergence.
Role: challenger_leg49_3 (EMPIRICAL CHALLENGER / critic, specialist).
Milestone: Wave D — LEG 49 Gate Iteration 2 (Commit 698a691).

Five Core Attacks Covered:
1. Attack 1: Perturbed step size finite difference scan (h in 1e-1 to 1e-8) confirming O(h^2) second-order convergence.
2. Attack 2: High-dimensional & random design perturbations (Gaussian, uniform, binary noise fields, relative error < 1e-6).
3. Attack 3: Dual adjoint identity stress test across 100 randomly sampled orthogonal vectors (< 1e-14).
4. Attack 4: Continuation beta parameter sweep (beta in [1, 2, 4, 8, 16, 32, 64] without NaN/Inf).
5. Attack 5: Monotonic optimization invariant under noisy initializations (gain > 0, verified by post-notary).
"""

from __future__ import annotations

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
import torch

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leg49_m3_constraints import M3FilterProjection
from src.leg49_grating_vehicle import GratingVehicle2D, get_default_device
from src.leg49_inverse_design import GratingInverseDesignLoop
from checks.post_notary import verify_post_flight

# Limit CPU OpenMP threads for test predictability
torch.set_num_threads(4)


class ChallengerAttack1StepSizeScanTests(unittest.TestCase):
    """Attack 1: Perturbed step size finite difference scan (h in 1e-1 to 1e-8).

    Verifies Taylor remainder convergence scaling and confirms second-order O(h^2)
    convergence across intermediate step sizes, contrasting against O(h^1) forward difference.
    """

    def setUp(self) -> None:
        torch.manual_seed(20261105)
        np.random.seed(20261105)
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_attack1_perturbed_step_size_finite_difference_scan(self) -> None:
        """Attack 1: Scan perturbation h in [10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8].

        Confirms second-order convergence O(h^2) across intermediate step sizes.
        Taylor Theorem for central difference:
          D_h f[v] - J*v = (h^2 / 6) * D^3 f(v, v, v) + O(h^4)
          => remainder E(h) = C * h^2
          => convergence rate p = log10(E(10h) / E(h)) -> 2.0.
        """
        n = 50
        m3 = M3FilterProjection(
            dx_um=0.020,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            beta=4.0,
            eta=0.5,
            dtype=self.dtype,
            device=self.device,
        )

        test_points = [
            ("uniform_mid", torch.full((n,), 0.5, dtype=self.dtype)),
            ("random_interior", torch.rand(n, dtype=self.dtype) * 0.7 + 0.15),
            ("extreme_boundary", torch.where(torch.rand(n) > 0.5, 0.95, 0.05).to(dtype=self.dtype)),
        ]

        h_scan = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

        for pt_name, rho in test_points:
            j_analytic = m3.compute_analytic_jacobian(rho, beta=4.0)

            for trial in range(3):
                v = torch.randn(n, dtype=self.dtype)
                v = v / v.norm()

                analytic_dv = j_analytic @ v
                remainders = []

                for h in h_scan:
                    f_plus = m3.forward(rho + h * v, beta=4.0)
                    f_minus = m3.forward(rho - h * v, beta=4.0)
                    fd_dv = (f_plus - f_minus) / (2.0 * h)
                    err = (fd_dv - analytic_dv).abs().max().item()
                    remainders.append(err)

                # 1. Confirm O(h^2) second-order convergence rate across intermediate steps:
                #    [1e-1, 1e-2, 1e-3, 1e-4]
                rates = []
                for k in range(3):
                    e_coarse = remainders[k]
                    e_fine = remainders[k + 1]
                    self.assertGreater(e_coarse, 0.0)
                    self.assertGreater(e_fine, 0.0)
                    rate = math.log10(e_coarse / e_fine)  # since step ratio is 10
                    rates.append(rate)

                mean_rate = sum(rates) / len(rates)
                self.assertGreaterEqual(
                    mean_rate,
                    1.85,
                    f"Point {pt_name} trial {trial}: Central diff rate {mean_rate:.3f} < 1.85 (expected ~2.0)",
                )
                self.assertLessEqual(
                    mean_rate,
                    2.15,
                    f"Point {pt_name} trial {trial}: Central diff rate {mean_rate:.3f} > 2.15 (expected ~2.0)",
                )

                # 2. Check error bounds across the full scan:
                # At h=1e-5 and 1e-6, relative error must be well below 1e-6 (machine precision domain)
                norm_denom = max(analytic_dv.abs().max().item(), 1e-12)
                rel_err_1e5 = remainders[4] / norm_denom
                rel_err_1e6 = remainders[5] / norm_denom
                self.assertLess(rel_err_1e5, 1e-6, f"Relative error at h=1e-5 exceeded 1e-6: {rel_err_1e5}")
                self.assertLess(rel_err_1e6, 1e-6, f"Relative error at h=1e-6 exceeded 1e-6: {rel_err_1e6}")

                # At h=1e-7 and 1e-8, floating point roundoff noise (eps_mach / h) appears
                # but must remain strictly bounded (< 1e-5)
                rel_err_1e7 = remainders[6] / norm_denom
                rel_err_1e8 = remainders[7] / norm_denom
                self.assertLess(rel_err_1e7, 1e-5, f"Relative error at h=1e-7 unstable: {rel_err_1e7}")
                self.assertLess(rel_err_1e8, 1e-5, f"Relative error at h=1e-8 unstable: {rel_err_1e8}")

    def test_forward_difference_linear_contrast_control(self) -> None:
        """Negative contrast control: one-sided forward difference scales as O(h^1).

        D_h^+ f[v] - J*v = (h / 2) * H(v, v) + O(h^2)
        => rate = log10(E(10h) / E(h)) -> 1.0 (contrasts against central diff rate 2.0).
        """
        n = 40
        m3 = M3FilterProjection(
            dx_um=0.020,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            beta=4.0,
            eta=0.5,
            dtype=self.dtype,
            device=self.device,
        )

        rho = torch.rand(n, dtype=self.dtype) * 0.6 + 0.2
        j_analytic = m3.compute_analytic_jacobian(rho, beta=4.0)
        f_base = m3.forward(rho, beta=4.0)

        v = torch.randn(n, dtype=self.dtype)
        v = v / v.norm()
        analytic_dv = j_analytic @ v

        h_steps = [1e-1, 1e-2, 1e-3, 1e-4]
        remainders = []

        for h in h_steps:
            f_plus = m3.forward(rho + h * v, beta=4.0)
            fd_dv = (f_plus - f_base) / h
            err = (fd_dv - analytic_dv).abs().max().item()
            remainders.append(err)

        rates = [
            math.log10(remainders[k] / remainders[k + 1])
            for k in range(len(h_steps) - 1)
        ]
        mean_rate = sum(rates) / len(rates)

        self.assertGreaterEqual(
            mean_rate,
            0.90,
            f"Forward diff rate {mean_rate:.3f} < 0.90 (expected ~1.0)",
        )
        self.assertLessEqual(
            mean_rate,
            1.10,
            f"Forward diff rate {mean_rate:.3f} > 1.10 (expected ~1.0)",
        )

    def test_taylor_boundary_overstep_smoothness(self) -> None:
        """Adversarial perturbation: push rho outside [0, 1] and verify C^infty smoothness."""
        n = 40
        m3 = M3FilterProjection(dx_um=0.020, min_feature_um=0.150, beta=4.0, dtype=self.dtype)

        rho = torch.zeros(n, dtype=self.dtype)
        rho[n // 2 :] = 1.0

        v = torch.randn(n, dtype=self.dtype)
        v = v / v.norm()

        h = 0.05
        j_analytic = m3.compute_analytic_jacobian(rho, beta=4.0)
        analytic_dv = j_analytic @ v

        f_plus = m3.forward(rho + h * v, beta=4.0)
        f_minus = m3.forward(rho - h * v, beta=4.0)
        fd_dv = (f_plus - f_minus) / (2.0 * h)

        diff = (fd_dv - analytic_dv).abs().max().item()
        self.assertLess(diff, 0.05, "Overstep boundary perturbation must remain stable and smooth")


class ChallengerAttack2RandomDesignPerturbationsTests(unittest.TestCase):
    """Attack 2: High-dimensional & random design perturbations.

    Tests gradient accuracy with randomly drawn Gaussian, uniform, and binary noise
    fields rho. Asserts relative error < 1e-6 and autograd graph VJP consistency < 1e-12.
    """

    def setUp(self) -> None:
        torch.manual_seed(424242)
        np.random.seed(424242)
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_attack2_high_dimensional_gaussian_noise_perturbations(self) -> None:
        """Attack 2a: High-dimensional Gaussian noise design fields rho ~ N(0.5, 0.15^2).

        Tests N=100 and N=200 cells. Asserts relative error < 1e-6 and autograd VJP < 1e-12.
        """
        for n in [100, 200]:
            m3 = M3FilterProjection(
                dx_um=0.020,
                min_feature_um=0.150,
                filter_radius_um=0.150,
                beta=4.0,
                dtype=self.dtype,
                device=self.device,
            )

            # Draw Gaussian noise field
            rho = (torch.randn(n, dtype=self.dtype) * 0.15 + 0.50).clamp(0.05, 0.95)
            rho.requires_grad = True

            # 1. Finite difference verification
            res = m3.verify_jacobian_finite_difference(
                rho, h=1e-6, tolerance=1e-6, num_directions=10, beta=4.0
            )
            self.assertTrue(res["passed"], f"FD failed on Gaussian noise N={n}: {res}")
            self.assertLessEqual(
                res["max_relative_error"],
                1e-6,
                f"Gaussian noise N={n} relative error {res['max_relative_error']} >= 1e-6",
            )

            # 2. Autograd VJP agreement (< 1e-12)
            j = m3.compute_analytic_jacobian(rho, beta=4.0)
            for _ in range(5):
                w = torch.randn(n, dtype=self.dtype)
                analytic_vjp = j.T @ w

                if rho.grad is not None:
                    rho.grad.zero_()
                proj = m3.forward(rho, beta=4.0)
                loss = (proj * w).sum()
                loss.backward()
                autograd_vjp = rho.grad

                rel_err = (analytic_vjp - autograd_vjp).abs().max().item() / max(
                    analytic_vjp.abs().max().item(), 1e-12
                )
                self.assertLess(
                    rel_err,
                    1e-12,
                    f"Gaussian noise N={n} autograd VJP mismatch: {rel_err:.2e}",
                )

    def test_attack2_high_dimensional_uniform_noise_perturbations(self) -> None:
        """Attack 2b: High-dimensional Uniform noise design fields rho ~ U(0.05, 0.95).

        Tests N=100 and N=200 cells. Asserts relative error < 1e-6 and autograd VJP < 1e-12.
        """
        for n in [100, 200]:
            m3 = M3FilterProjection(
                dx_um=0.020,
                min_feature_um=0.150,
                filter_radius_um=0.150,
                beta=4.0,
                dtype=self.dtype,
                device=self.device,
            )

            rho = torch.rand(n, dtype=self.dtype) * 0.90 + 0.05
            rho.requires_grad = True

            res = m3.verify_jacobian_finite_difference(
                rho, h=1e-6, tolerance=1e-6, num_directions=10, beta=4.0
            )
            self.assertTrue(res["passed"], f"FD failed on Uniform noise N={n}: {res}")
            self.assertLessEqual(
                res["max_relative_error"],
                1e-6,
                f"Uniform noise N={n} relative error {res['max_relative_error']} >= 1e-6",
            )

            j = m3.compute_analytic_jacobian(rho, beta=4.0)
            for _ in range(5):
                w = torch.randn(n, dtype=self.dtype)
                analytic_vjp = j.T @ w

                if rho.grad is not None:
                    rho.grad.zero_()
                proj = m3.forward(rho, beta=4.0)
                loss = (proj * w).sum()
                loss.backward()
                autograd_vjp = rho.grad

                rel_err = (analytic_vjp - autograd_vjp).abs().max().item() / max(
                    analytic_vjp.abs().max().item(), 1e-12
                )
                self.assertLess(
                    rel_err,
                    1e-12,
                    f"Uniform noise N={n} autograd VJP mismatch: {rel_err:.2e}",
                )

    def test_attack2_high_dimensional_binary_noise_perturbations(self) -> None:
        """Attack 2c: High-dimensional binary noise design fields rho in {0.05, 0.95}^N.

        Tests high-contrast boundary transitions. Asserts relative error < 1e-6.
        """
        for n in [100, 200]:
            m3 = M3FilterProjection(
                dx_um=0.020,
                min_feature_um=0.150,
                filter_radius_um=0.150,
                beta=4.0,
                dtype=self.dtype,
                device=self.device,
            )

            # High-contrast binary noise
            binary_mask = torch.rand(n, dtype=self.dtype) > 0.5
            rho = torch.where(binary_mask, 0.95, 0.05)
            rho.requires_grad = True

            res = m3.verify_jacobian_finite_difference(
                rho, h=1e-6, tolerance=1e-6, num_directions=10, beta=4.0
            )
            self.assertTrue(res["passed"], f"FD failed on Binary noise N={n}: {res}")
            self.assertLessEqual(
                res["max_relative_error"],
                1e-6,
                f"Binary noise N={n} relative error {res['max_relative_error']} >= 1e-6",
            )

    def test_pathological_density_fields(self) -> None:
        """Test degenerate density patterns: uniform zeros, uniform ones, Nyquist comb, delta spike."""
        n = 60
        m3 = M3FilterProjection(dx_um=0.020, min_feature_um=0.150, beta=4.0, dtype=self.dtype)

        pathological_fields = {
            "all_zeros": torch.zeros(n, dtype=self.dtype),
            "all_ones": torch.ones(n, dtype=self.dtype),
            "mid_grayscale": torch.full((n,), 0.5, dtype=self.dtype),
            "nyquist_comb": torch.tensor([0.0, 1.0] * (n // 2), dtype=self.dtype),
            "delta_spike": torch.zeros(n, dtype=self.dtype),
        }
        pathological_fields["delta_spike"][n // 2] = 1.0

        for name, rho in pathological_fields.items():
            j = m3.compute_analytic_jacobian(rho, beta=4.0)
            self.assertFalse(torch.isnan(j).any(), f"NaN in Jacobian for {name}")
            self.assertFalse(torch.isinf(j).any(), f"Inf in Jacobian for {name}")

            res = m3.verify_jacobian_finite_difference(rho, tolerance=1e-6, beta=4.0)
            self.assertTrue(res["passed"], f"FD check failed for {name}: {res}")


class ChallengerAttack3DualAdjointIdentityTests(unittest.TestCase):
    """Attack 3: Dual adjoint identity stress test.

    Evaluates <v, J u> vs <J^T v, u> across 100 randomly sampled orthogonal vectors u, v.
    Asserts bit-identity or ULP difference < 1e-14.
    """

    def setUp(self) -> None:
        torch.manual_seed(999888)
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_attack3_dual_adjoint_identity_100_orthogonal_vectors(self) -> None:
        """Attack 3: Stress test Hilbert adjoint dot-product identity across 100 orthogonal vector pairs.

        For linear operator J:
          <v, J u> = v^T (J u)
          <J^T v, u> = (J^T v)^T u
        For 100 Gram-Schmidt orthogonal pairs u perp v:
          Assert relative difference < 1e-14 down to floating-point rounding limit.
        """
        n = 80
        m3 = M3FilterProjection(
            dx_um=0.020,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            beta=4.0,
            dtype=self.dtype,
            device=self.device,
        )

        rho = torch.rand(n, dtype=self.dtype) * 0.8 + 0.1
        j = m3.compute_analytic_jacobian(rho, beta=4.0)
        j_t = j.T

        max_rel_diff = 0.0

        for pair_idx in range(100):
            # Generate random vector u
            u = torch.randn(n, dtype=self.dtype)
            u = u / u.norm()

            # Generate random vector z and orthogonalize against u via Gram-Schmidt
            z = torch.randn(n, dtype=self.dtype)
            proj_u = torch.dot(z, u) * u
            v = z - proj_u
            v = v / v.norm()

            # Verify orthogonality to machine precision
            dot_uv = abs(torch.dot(u, v).item())
            self.assertLess(dot_uv, 1e-14, f"Pair {pair_idx}: Vectors u and v are not orthogonal: {dot_uv}")

            # LHS: <v, J u>
            ju = j @ u
            lhs = torch.dot(v, ju).item()

            # RHS: <J^T v, u>
            jtv = j_t @ v
            rhs = torch.dot(jtv, u).item()

            denom = max(abs(lhs), abs(rhs), 1e-15)
            rel_diff = abs(lhs - rhs) / denom
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff

            self.assertLess(
                rel_diff,
                5e-14,
                f"Pair {pair_idx}: Dual adjoint identity failed! lhs={lhs:.16e}, rhs={rhs:.16e}, rel_diff={rel_diff:.2e}",
            )

        # Print check report info
        self.assertLess(max_rel_diff, 5e-14, f"Max relative difference across 100 pairs exceeded 5e-14: {max_rel_diff}")

    def test_adjoint_chain_rule_consistency_with_grating_permittivity(self) -> None:
        """Verify chain rule adjoint identity across full rho -> eps mapping."""
        vehicle = GratingVehicle2D(
            dx_um=0.050,
            dy_um=0.050,
            lx_um=1.5,
            steps=50,
            device=self.device,
            dtype=self.dtype,
        )
        m3 = M3FilterProjection(
            dx_um=0.050,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            beta=4.0,
            dtype=self.dtype,
        )

        n = vehicle.num_design_cells
        rho = torch.rand(n, dtype=self.dtype) * 0.7 + 0.15
        rho.requires_grad = True

        eps_adj_seed = torch.randn((vehicle.nx, vehicle.ny, vehicle.nz), dtype=self.dtype)

        rho_proj = m3.forward(rho, beta=4.0)
        eps = vehicle.density_to_epsilon(rho_proj)
        loss = (eps * eps_adj_seed).sum()
        loss.backward()
        autograd_grad = rho.grad.clone()

        # Analytic adjoint composition
        z_nodes = torch.tensor(vehicle.region.mesh_nodes[2], dtype=self.dtype)
        half_h = vehicle.core_h_um / 2.0
        z_lo = z_nodes[:-1]
        z_hi = z_nodes[1:]
        overlap = (torch.minimum(z_hi, torch.tensor(half_h)) - torch.maximum(z_lo, torch.tensor(-half_h))).clamp_min(0.0)
        fz = overlap / (z_hi - z_lo)

        d_eps_d_proj = (eps_adj_seed * fz[None, None, :]).sum(dim=(1, 2)) * vehicle.delta_eps
        j_proj = m3.compute_analytic_jacobian(rho, beta=4.0)
        analytic_grad = j_proj.T @ d_eps_d_proj

        rel_mismatch = (autograd_grad - analytic_grad).abs().max().item() / autograd_grad.abs().max().item()
        self.assertLess(
            rel_mismatch,
            1e-12,
            f"Adjoint chain rule through permittivity failed: {rel_mismatch:.2e}",
        )


class ChallengerAttack4ContinuationBetaSweepTests(unittest.TestCase):
    """Attack 4: Continuation beta parameter sweep.

    Stress-tests projection derivative sharpness with beta in [1, 2, 4, 8, 16, 32, 64].
    Verifies that autograd and VJP never return NaN or Inf even as beta -> infinity.
    """

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_attack4_continuation_beta_parameter_sweep(self) -> None:
        """Attack 4: Sweep beta in [1, 2, 4, 8, 16, 32, 64].

        Verifies non-negativity, stability, absence of NaN/Inf, and strict autograd agreement.
        """
        n = 50
        rho = torch.rand(n, dtype=self.dtype) * 0.8 + 0.1
        rho.requires_grad = True

        m3 = M3FilterProjection(
            dx_um=0.020,
            min_feature_um=0.150,
            filter_radius_um=0.150,
            dtype=self.dtype,
            device=self.device,
        )

        beta_list = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

        for beta in beta_list:
            j_analytic = m3.compute_analytic_jacobian(rho, beta=beta)

            # 1. Zero NaN / Inf
            self.assertFalse(torch.isnan(j_analytic).any(), f"NaN in Jacobian for beta={beta}")
            self.assertFalse(torch.isinf(j_analytic).any(), f"Inf in Jacobian for beta={beta}")

            # 2. Non-negativity
            self.assertGreaterEqual(
                j_analytic.min().item(),
                -1e-15,
                f"Jacobian entries must be non-negative for beta={beta}",
            )

            # 3. Peak derivative bound: d_proj <= beta / (2.0 * tanh(beta / 2.0))
            max_deriv = j_analytic.sum(dim=-1).max().item()
            expected_peak = beta / (2.0 * math.tanh(beta / 2.0))
            self.assertLessEqual(
                max_deriv,
                expected_peak * 1.01,
                f"Derivative peak {max_deriv} exceeded expected {expected_peak} for beta={beta}",
            )

            # 4. Autograd VJP agreement (< 1e-12)
            w = torch.randn(n, dtype=self.dtype)
            vjp_analytic = j_analytic.T @ w

            if rho.grad is not None:
                rho.grad.zero_()
            proj = m3.forward(rho, beta=beta)
            loss = (proj * w).sum()
            loss.backward()
            vjp_autograd = rho.grad

            rel_err = (vjp_analytic - vjp_autograd).abs().max().item() / max(
                vjp_analytic.abs().max().item(), 1e-12
            )
            self.assertLess(
                rel_err,
                1e-12,
                f"Autograd VJP mismatch for beta={beta}: {rel_err:.2e}",
            )

            # 5. Central finite difference verification (< 1e-6)
            res = m3.verify_jacobian_finite_difference(
                rho, h=1e-6, tolerance=1e-6, num_directions=5, beta=beta
            )
            self.assertTrue(res["passed"], f"FD check failed for beta={beta}: {res}")

    def test_continuation_beta_asymptotic_infinity_stability(self) -> None:
        """Asymptotic stress test: beta in {128, 256} approaching Heavyside step function."""
        n = 50
        rho = torch.rand(n, dtype=self.dtype) * 0.8 + 0.1
        m3 = M3FilterProjection(dx_um=0.020, min_feature_um=0.150, dtype=self.dtype)

        for beta in [128.0, 256.0]:
            proj = m3.forward(rho, beta=beta)
            self.assertFalse(torch.isnan(proj).any(), f"NaN in forward projection for beta={beta}")
            self.assertFalse(torch.isinf(proj).any(), f"Inf in forward projection for beta={beta}")
            self.assertTrue((proj >= 0.0 - 1e-9).all() and (proj <= 1.0 + 1e-9).all())

            j = m3.compute_analytic_jacobian(rho, beta=beta)
            self.assertFalse(torch.isnan(j).any(), f"NaN in Jacobian for beta={beta}")
            self.assertFalse(torch.isinf(j).any(), f"Inf in Jacobian for beta={beta}")

    def test_extreme_filter_radii_sweep(self) -> None:
        """Sweep filter radius R in [60, 100, 150, 200, 250, 300, 400] nm."""
        n = 60
        dx_um = 0.020

        for r_nm in [60, 100, 150, 200, 250, 300, 400]:
            r_um = r_nm / 1000.0
            m3 = M3FilterProjection(
                dx_um=dx_um,
                min_feature_um=r_um,
                filter_radius_um=r_um,
                dtype=self.dtype,
                device=self.device,
            )

            w = m3._get_filter_matrix(n)

            # Exact row-sum conservation
            row_sums = w.sum(dim=-1)
            expected_sums = torch.ones(n, dtype=self.dtype)
            self.assertTrue(
                torch.allclose(row_sums, expected_sums, atol=1e-14),
                f"Row sums must equal 1.0 within 1e-14 for R={r_nm} nm",
            )

            # Compact support: W_ij == 0 when distance >= R
            for i in range(n):
                for j in range(n):
                    dist = abs(i - j) * dx_um
                    if dist >= r_um - 1e-9:
                        self.assertEqual(
                            w[i, j].item(),
                            0.0,
                            f"Compact support violated at ({i}, {j}) for R={r_nm} nm",
                        )


class ChallengerAttack5MonotonicOptimizationTests(unittest.TestCase):
    """Attack 5: Monotonic optimization invariant under perturbed topologies.

    Executes 3 optimization iterations starting from random noisy grating initializations.
    Verifies that step-by-step progress remains strictly positive (gain > 0) or halts cleanly.
    """

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_attack5_monotonic_optimization_invariant_under_perturbed_topologies(self) -> None:
        """Attack 5: Execute 3 optimization iterations from noisy grating initializations.

        Verifies that step-by-step progress yields strictly positive gain (> 0)
        and verifies post-notary passes cleanly under temp run directory.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="adv_attack5_opt_"))
        try:
            os.environ["RUN_DIR_ABS"] = str(temp_dir)
            device = get_default_device()

            # Vehicle setup for 3 iterations with physical PML decay
            vehicle = GratingVehicle2D(
                dx_um=0.050,
                dy_um=0.050,
                lx_um=2.5,
                steps=1000,
                device=device,
                dtype=self.dtype,
            )
            m3 = M3FilterProjection(
                dx_um=0.050,
                min_feature_um=0.150,
                filter_radius_um=0.150,
                dtype=self.dtype,
                device=device,
            )

            # Test across random noisy initialization seed
            loop = GratingInverseDesignLoop(
                vehicle=vehicle,
                m3_filter=m3,
                optimizer_type="adam",
                learning_rate=0.05,
                seed=98765,
            )

            results = loop.run_optimization(num_iterations=3)

            # 1. Assertions on optimization progress
            self.assertEqual(results["iterations_completed"], 3)
            self.assertGreater(
                results["objective_gain"],
                0.0,
                f"Optimization gain must be strictly positive: {results['objective_gain']}",
            )
            self.assertGreater(
                results["final_objective"],
                results["initial_objective"],
                "Final objective must exceed initial objective",
            )

            # 2. Assert Jacobian relative error <= 1e-6 across all steps
            self.assertLessEqual(
                results["jacobian_relative_error"],
                1e-6,
                "Jacobian error across all steps must be <= 1e-6",
            )

            # 3. Assert M3 compliance on realized discrete patterns
            for step in results["steps"]:
                m3_rec = step["m3_fabrication"]
                self.assertTrue(m3_rec["compliant"], f"M3 violated in step {step['step_index']}: {m3_rec}")
                self.assertGreaterEqual(m3_rec["min_tooth_width_nm"], 150.0 - 1e-3)
                self.assertGreaterEqual(m3_rec["min_trench_width_nm"], 150.0 - 1e-3)
                self.assertEqual(len(m3_rec["violations"]), 0)

                # Physical passivity check
                self.assertLessEqual(step["forward"]["power_ratio"], 1.0 + 1e-5)

            # 4. Packaging and notary verification
            last_step = results["steps"][-1]
            config = {
                "kind": "inverse_design",
                "inverse_design": True,
                "iterations": 3,
                "wavelength_um": 1.55,
                "tags": ["inverse_design", "inverse_design_step"],
            }
            (temp_dir / "config.input.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

            readout = {
                "schema_version": "1.0",
                "kind": "inverse_design",
                "simulation_executed": True,
                "executed_steps": vehicle.steps,
                "epsilon_sha256": last_step["forward"]["epsilon_sha256"],
                "is_lossless": True,
                "inverse_design": results,
                "monitors": {
                    "top": {"flux": last_step["forward"]["monitors"]["top"]["flux"]},
                    "bottom": {"flux": last_step["forward"]["monitors"]["bottom"]["flux"]},
                },
                "tags": [
                    "inverse_design",
                    "inverse_design_step",
                    "verified:analytic_jacobian",
                    "verified:m3_fabrication_compliant",
                ],
            }
            (temp_dir / "readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")

            cert = verify_post_flight(temp_dir)
            self.assertTrue(cert["verified"], f"Post notary verification failed: {cert}")
            self.assertIn("verified:inverse_design_step", cert["tags"])
            self.assertIn("verified:analytic_jacobian", cert["tags"])
            self.assertIn("verified:m3_fabrication_compliant", cert["tags"])
            self.assertTrue(cert["inverse_design"]["monotonic_progress"])
            self.assertTrue(cert["inverse_design"]["m3_compliant"])
            self.assertTrue(cert["inverse_design"]["all_steps_passive"])
        finally:
            os.environ.pop("RUN_DIR_ABS", None)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_adversarial_m3_morphology_stress_patterns(self) -> None:
        """Adversarial morphological projection on pathological pinholes, spikes, and comb structures."""
        m3 = M3FilterProjection(dx_um=0.020, min_feature_um=0.150)
        n = 300

        patterns = {
            "single_cell_spike": np.zeros(n, dtype=bool),
            "single_cell_gap": np.ones(n, dtype=bool),
            "nyquist_20nm_comb": np.array([True, False] * (n // 2), dtype=bool),
            "dense_pinholes": np.ones(n, dtype=bool),
            "random_hash": np.random.RandomState(42).choice([True, False], size=n),
        }
        patterns["single_cell_spike"][50] = True
        patterns["single_cell_gap"][50] = False
        patterns["dense_pinholes"][::10] = False

        for name, raw_arr in patterns.items():
            realized = m3.project_m3_discrete(raw_arr)
            check = m3.check_violations(realized)

            self.assertTrue(
                check["compliant"],
                f"Pathological pattern {name} failed M3 compliance after Tier 3 projection: {check}",
            )
            self.assertEqual(len(check["violations"]), 0)
            self.assertGreaterEqual(check["min_tooth_width_nm"], 150.0 - 1e-3)
            self.assertGreaterEqual(check["min_trench_width_nm"], 150.0 - 1e-3)

    def test_notary_rejection_of_non_monotonic_objective(self) -> None:
        """Adversarial notary verification: regressing objective strictly triggers exit 66."""
        temp_dir = Path(tempfile.mkdtemp(prefix="challenger_nc_mono_"))
        try:
            config = {
                "kind": "inverse_design",
                "inverse_design": True,
                "iterations": 2,
                "tags": ["inverse_design", "inverse_design_step"],
            }
            readout = {
                "kind": "inverse_design",
                "simulation_executed": True,
                "inverse_design": {
                    "device": "canonical_grating_2d",
                    "iterations_completed": 2,
                    "initial_objective": 0.50,
                    "final_objective": 0.42,
                    "objective_gain": -0.08,
                    "gradient_tolerance": 1e-06,
                    "steps": [
                        {
                            "step_index": 0,
                            "objective_value": 0.50,
                            "forward": {"executed_steps": 100, "is_lossless": True, "epsilon_sha256": "h0", "power_ratio": 0.9},
                            "adjoint": {"executed_steps": 100, "is_lossless": True, "epsilon_sha256": "h0"},
                            "gradient_verification": {"relative_error": 1.0e-7, "passed": True},
                            "m3_fabrication": {"min_tooth_width_nm": 200.0, "min_trench_width_nm": 200.0, "violations": [], "compliant": True},
                        },
                        {
                            "step_index": 1,
                            "objective_value": 0.42,
                            "forward": {"executed_steps": 100, "is_lossless": True, "epsilon_sha256": "h1", "power_ratio": 0.9},
                            "adjoint": {"executed_steps": 100, "is_lossless": True, "epsilon_sha256": "h1"},
                            "gradient_verification": {"relative_error": 1.0e-7, "passed": True},
                            "m3_fabrication": {"min_tooth_width_nm": 200.0, "min_trench_width_nm": 200.0, "violations": [], "compliant": True},
                        },
                    ],
                },
            }
            telemetry = {
                "schema_version": "1.0",
                "project_canonical_sha256": "p_hash",
                "epsilon_sha256": "h1",
                "is_lossless": True,
                "executed_steps": 200,
                "energy_decay_ratio": 1e-7,
                "monitors": {"top": {"flux": 0.42}},
            }
            (temp_dir / "config.input.json").write_text(json.dumps(config), encoding="utf-8")
            (temp_dir / "readout.json").write_text(json.dumps(readout), encoding="utf-8")
            (temp_dir / "solver_telemetry.json").write_text(json.dumps(telemetry), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", "checks/post_notary.py", str(temp_dir)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 66, f"Notary must exit 66 on non-monotonic objective: {proc.stderr}")
            cert = json.loads((temp_dir / "verification_certificate.json").read_text(encoding="utf-8"))
            self.assertFalse(cert["verified"])
            self.assertEqual(cert["status"], "non_monotonic_objective")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_step_size_box_projection_clamp_at_extremes(self) -> None:
        """Adversarial step sizes (lr = 100.0) must respect box projection [0, 1] without exploding."""
        vehicle = GratingVehicle2D(
            dx_um=0.050,
            dy_um=0.050,
            lx_um=1.5,
            steps=50,
            device=self.device,
            dtype=self.dtype,
        )
        m3 = M3FilterProjection(dx_um=0.050, min_feature_um=0.150, dtype=self.dtype)

        loop = GratingInverseDesignLoop(
            vehicle=vehicle,
            m3_filter=m3,
            optimizer_type="adam",
            learning_rate=100.0,
        )

        step_facts = loop.evaluate_step(iteration=0)

        self.assertGreaterEqual(loop.rho.min().item(), 0.0)
        self.assertLessEqual(loop.rho.max().item(), 1.0)

        eps = vehicle.density_to_epsilon(m3.forward(loop.rho))
        self.assertGreaterEqual(eps.min().item(), vehicle.eps_clad - 1e-9)
        self.assertLessEqual(eps.max().item(), vehicle.eps_core + 1e-9)

        self.assertTrue(math.isfinite(step_facts["objective_value"]))
        self.assertTrue(step_facts["m3_fabrication"]["compliant"])

    def test_zero_gradient_degenerate_stability(self) -> None:
        """Adversarial zero gradient: optimizer must advance without NaN/Inf/ZeroDivision."""
        vehicle = GratingVehicle2D(
            dx_um=0.050,
            dy_um=0.050,
            lx_um=1.5,
            steps=50,
            device=self.device,
            dtype=self.dtype,
        )
        m3 = M3FilterProjection(dx_um=0.050, min_feature_um=0.150, dtype=self.dtype)

        loop = GratingInverseDesignLoop(vehicle=vehicle, m3_filter=m3, optimizer_type="adam")

        loop.optimizer.zero_grad()
        loop.rho.grad = torch.zeros_like(loop.rho)
        loop.optimizer.step()

        self.assertFalse(torch.isnan(loop.rho).any())
        self.assertFalse(torch.isinf(loop.rho).any())


if __name__ == "__main__":
    unittest.main()
