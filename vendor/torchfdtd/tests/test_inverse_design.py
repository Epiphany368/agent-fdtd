from __future__ import annotations

import unittest
import numpy as np
import torch

from torchfdtd.inverse_design import (
    DesignRegionDFTAccumulator,
    assess_forward_step,
    escape_local_minimum_perturbation,
    propose_minmax_lp_step,
    screen_waveguide_seeds_2d,
)
from torchfdtd.subpixel_waveguide import conformal_variant0_cross_section


class TestInverseDesign(unittest.TestCase):
    def test_minmax_lp_step_and_forward_assessment(self) -> None:
        n = 39
        J = np.ones((n, 5)) / n * 0.001
        p = np.zeros(n)
        eta = np.array([0.90, 0.95, 0.96, 0.95, 0.91])
        res = propose_minmax_lp_step(
            p,
            eta,
            J,
            scale=np.ones(n),
            bounds=np.tile([-20.0, 20.0], (n, 1)),
            boundary_J=np.eye(n),
            radius_nm=5.0,
            seed_eta=eta,
        )
        self.assertEqual(res["status"], "proposal_ready")
        delta = np.asarray(res["delta"], float)
        self.assertLessEqual(float(np.max(np.abs(delta))), 5.0 + 1e-9)
        self.assertGreater(float(np.linalg.norm(delta)), 25.0)

        assessed = assess_forward_step(
            eta,
            np.asarray(res["predicted_eta"], float),
            np.asarray(res["predicted_eta"], float),
            eta,
            geometry_ok=True,
        )
        self.assertTrue(assessed["accepted_forward"])

        bad = eta + np.array([0.01, 0.01, -0.008, 0.01, 0.01])
        self.assertFalse(assess_forward_step(eta, bad, bad, eta, geometry_ok=True)["accepted_forward"])

    def test_escape_local_minimum_nullspace_perturbation(self) -> None:
        n = 12
        p = np.zeros(n)
        J = np.zeros((n, 3))
        J[0, :] = 0.01
        J[1, :] = -0.005
        B = np.eye(n)
        bounds = np.tile([-15.0, 15.0], (n, 1))
        delta_escape = escape_local_minimum_perturbation(p, J, B, bounds, radius_nm=6.0)
        self.assertEqual(delta_escape.shape, (n,))
        self.assertTrue(np.isclose(np.max(np.abs(B @ delta_escape)), 6.0, atol=1e-6))
        self.assertLess(float(np.max(np.abs(J.T @ delta_escape))), 1e-10)

    def test_design_region_dft_accumulator_zero_replay(self) -> None:
        acc = DesignRegionDFTAccumulator(
            ix_slice=slice(2, 6),
            iy_slice=slice(1, 5),
            iz_slice=slice(1, 4),
            wavelengths_um=(1.26, 1.31, 1.36),
            dt_s=2.0e-17,
            device="cpu",
        )
        E = torch.ones((8, 6, 5, 3), dtype=torch.float32)
        for step in range(10):
            acc.accumulate_step(E, step, is_adjoint=False)
            acc.accumulate_step(E * 0.5, step, is_adjoint=True)
        grad = acc.permittivity_gradient_per_wavelength()
        self.assertEqual(grad.shape, (3, 4, 4, 3, 3))
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().max()), 0.0)

    def test_screen_waveguide_seeds_2d_orders_smooth_over_abrupt(self) -> None:
        u = np.linspace(-1.2, 1.2, 31)
        v = np.linspace(-0.6, 0.6, 21)
        eps_core = 3.4757 ** 2
        eps_clad = 1.444 ** 2

        def smooth_taper_seed(x_um: float, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
            frac = np.clip(x_um / 10.0, 0.0, 1.0)
            w = 0.38 + (0.71 - 0.38) * frac
            return conformal_variant0_cross_section(
                uu, vv, comp="all", slabs=[(-0.5 * w, 0.5 * w, -0.1075, 0.1075, eps_core)], eps_bg=eps_clad
            )

        def abrupt_jump_seed(x_um: float, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
            w = 0.38 if x_um < 5.0 else 0.71
            return conformal_variant0_cross_section(
                uu, vv, comp="all", slabs=[(-0.5 * w, 0.5 * w, -0.1075, 0.1075, eps_core)], eps_bg=eps_clad
            )

        scores = screen_waveguide_seeds_2d(
            candidates=[("abrupt_seed", abrupt_jump_seed), ("smooth_seed", smooth_taper_seed)],
            x_stations_um=[0.0, 4.9, 5.1, 10.0],
            u_coords_um=u,
            v_coords_um=v,
            wavelength_um=1.31,
            num_modes=2,
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0].seed_id, "smooth_seed")
        self.assertLess(scores[0].adiabaticity_metric, scores[1].adiabaticity_metric)


if __name__ == "__main__":
    unittest.main()
