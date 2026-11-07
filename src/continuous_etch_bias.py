r"""Continuous-Geometry Subpixel Etch-Bias & 21-Point Edge-Crossing Sweep (Wave E — LEG 53).

Provides piecewise-differentiable subpixel boundary parameterization:

    f_i(x_L, x_R) = (1 / \Delta x) * max(0, min(x_{i+1}, x_R) - max(x_i, x_L))

And evaluates continuous etch-bias shifts:
    x_L(b) = x_L - b
    x_R(b) = x_R + b
For nominal and +/- 10 nm shifts.

Includes the mandatory 21-point edge-crossing sweep test (1 nm steps across 20 nm Yee cell):
- Verifies adjoint gradient \partial \eta / \partial x_R matches central finite differences
  within relative error <= 1e-5 inside the cell (k=1..19).
- Quantifies the piecewise-linear derivative kink at cell boundaries (k=0, k=20)
  while verifying that \eta(x_R) remains continuous without jumps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def subpixel_fill_fraction_1d(
    x_nodes: torch.Tensor,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
) -> torch.Tensor:
    r"""Compute 1D piecewise-differentiable cell volume fraction for a tooth [x_left, x_right].

    x_nodes: 1D tensor of cell edge coordinates [x_0, x_1, ..., x_N], shape (N+1,).
    x_left: scalar or 1D tensor of left boundary.
    x_right: scalar or 1D tensor of right boundary.

    Returns:
    f: tensor of shape (N,), where f_i = (1 / dx_i) * max(0, min(x_{i+1}, x_R) - max(x_i, x_L)).
    """
    x_lo = x_nodes[:-1]
    x_hi = x_nodes[1:]
    dx = x_hi - x_lo

    # Piecewise-differentiable interval intersection
    overlap = torch.clamp(
        torch.minimum(x_hi, x_right) - torch.maximum(x_lo, x_left),
        min=0.0,
    )
    return overlap / dx


def multi_tooth_fill_fraction_1d(
    x_nodes: torch.Tensor,
    teeth: Union[List[Tuple[float, float]], torch.Tensor],
    etch_bias_um: float = 0.0,
) -> torch.Tensor:
    r"""Compute total subpixel volume fraction for multiple teeth under etch bias.

    Each tooth is defined by nominal [x_L, x_R].
    Under etch bias b:
        x_L(b) = x_L - b
        x_R(b) = x_R + b
    Total fill fraction is clamped to [0, 1].
    """
    n_cells = x_nodes.shape[0] - 1
    total_f = torch.zeros(n_cells, dtype=x_nodes.dtype, device=x_nodes.device)

    if isinstance(teeth, list):
        for tooth in teeth:
            x_l_nom, x_r_nom = tooth
            x_l = torch.as_tensor(x_l_nom - etch_bias_um, dtype=x_nodes.dtype, device=x_nodes.device)
            x_r = torch.as_tensor(x_r_nom + etch_bias_um, dtype=x_nodes.dtype, device=x_nodes.device)
            f_k = subpixel_fill_fraction_1d(x_nodes, x_l, x_r)
            total_f = total_f + f_k
    else:
        # teeth is tensor of shape (K, 2)
        for k in range(teeth.shape[0]):
            x_l = teeth[k, 0] - etch_bias_um
            x_r = teeth[k, 1] + etch_bias_um
            f_k = subpixel_fill_fraction_1d(x_nodes, x_l, x_r)
            total_f = total_f + f_k

    return torch.clamp(total_f, min=0.0, max=1.0)


class ContinuousEtchBiasGeometry:
    """Parametric subpixel geometry with continuous etch-bias and autograd support."""

    def __init__(
        self,
        lx_um: float = 2.5,
        dx_um: float = 0.020,
        core_h_um: float = 0.220,
        n_core: float = 3.477,
        n_clad: float = 1.444,
        nominal_teeth_um: Optional[List[Tuple[float, float]]] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.lx_um = float(lx_um)
        self.dx_um = float(dx_um)
        self.core_h_um = float(core_h_um)
        self.n_core = float(n_core)
        self.n_clad = float(n_clad)
        self.eps_core = self.n_core**2
        self.eps_clad = self.n_clad**2
        self.delta_eps = self.eps_core - self.eps_clad
        self.device = torch.device(device)
        self.dtype = dtype

        self.nx = int(round(self.lx_um / self.dx_um))
        self.x_nodes = torch.linspace(
            0.0, self.lx_um, self.nx + 1, dtype=self.dtype, device=self.device
        )
        self.x_centers = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])

        if nominal_teeth_um is None:
            # Standard canonical grating teeth: e.g. period 0.630 um, duty cycle 0.5
            period = 0.630
            tooth_w = 0.315
            teeth = []
            x_curr = 0.300
            while x_curr + tooth_w <= self.lx_um - 0.300:
                teeth.append((x_curr, x_curr + tooth_w))
                x_curr += period
            self.nominal_teeth_um = teeth
        else:
            self.nominal_teeth_um = nominal_teeth_um

    def compute_density(self, etch_bias_um: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
        """Compute 1D continuous density rho(x) in [0, 1] under etch bias."""
        if isinstance(etch_bias_um, torch.Tensor):
            total_f = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
            for x_l_nom, x_r_nom in self.nominal_teeth_um:
                x_l = x_l_nom - etch_bias_um
                x_r = x_r_nom + etch_bias_um
                f_k = subpixel_fill_fraction_1d(self.x_nodes, x_l, x_r)
                total_f = total_f + f_k
            return torch.clamp(total_f, min=0.0, max=1.0)
        else:
            return multi_tooth_fill_fraction_1d(
                self.x_nodes, self.nominal_teeth_um, etch_bias_um=float(etch_bias_um)
            )

    def density_to_epsilon_1d(self, density_1d: torch.Tensor) -> torch.Tensor:
        """Convert 1D density to 1D core permittivity: eps(x) = eps_clad + delta_eps * rho(x)."""
        return self.eps_clad + self.delta_eps * density_1d

    def density_to_epsilon_3d(
        self,
        density_1d: torch.Tensor,
        ny: int = 15,
        nz: int = 90,
        z_span_um: float = 1.80,
    ) -> torch.Tensor:
        """Expand 1D density into 3D Yee permittivity tensor (Nx, Ny, Nz)."""
        z_nodes = torch.linspace(-z_span_um, z_span_um, nz + 1, dtype=self.dtype, device=self.device)
        z_lo = z_nodes[:-1]
        z_hi = z_nodes[1:]
        half_h = self.core_h_um / 2.0
        z_overlap = torch.clamp(
            torch.minimum(z_hi, torch.tensor(half_h, device=self.device))
            - torch.maximum(z_lo, torch.tensor(-half_h, device=self.device)),
            min=0.0,
        )
        fz = z_overlap / (z_hi - z_lo)  # shape (nz,)

        # eps base: eps_clad everywhere
        eps = torch.full((self.nx, ny, nz), self.eps_clad, dtype=self.dtype, device=self.device)
        grating_delta = density_1d[:, None, None] * fz[None, None, :] * self.delta_eps
        return eps + grating_delta


def run_21point_edge_crossing_sweep(
    dx_nm: float = 20.0,
    cell_index: int = 25,
    n_points: int = 21,
    h_fd_nm: float = 0.05,
    device: str = "cpu",
) -> Dict[str, Any]:
    r"""Execute the mandatory 21-point edge-crossing sweep test across one Yee cell.

    Sweep:
        x_R moves across cell `cell_index` in steps of 1 nm (for 20 nm cell -> 21 points: 0 nm to 20 nm).
    At each position:
        - Target function \eta(x_R) is computed via piecewise-differentiable subpixel projection.
        - Adjoint gradient d\eta / dx_R is obtained via PyTorch backward().
        - Central finite difference (d\eta / dx_R)_FD = (\eta(x_R + h) - \eta(x_R - h)) / (2h) is computed.
    Verification:
        - For interior points (k = 1..19): relative error between adjoint and FD is <= 1e-5.
        - At boundary points (k = 0, k = 20): characterizes the kink step transition and one-sided consistency.
        - Verifies \eta(x_R) is strictly continuous across cell boundaries.
    """
    dx_um = dx_nm * 1e-3
    h_um = h_fd_nm * 1e-3
    dev = torch.device(device)
    dtype = torch.float64

    # Build 1D mesh with 60 cells
    total_cells = 60
    x_nodes = torch.linspace(0.0, total_cells * dx_um, total_cells + 1, dtype=dtype, device=dev)
    x_lo_cell = float(x_nodes[cell_index].item())
    x_hi_cell = float(x_nodes[cell_index + 1].item())

    # Fixed left tooth edge well to the left
    x_left_fixed = float(x_nodes[cell_index - 5].item())

    # Analytical test objective kernel:
    # A smooth field weight w(x) mimicking mode field profile interaction
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    field_weight = torch.exp(-(((x_centers - x_lo_cell) / (10.0 * dx_um)) ** 2))

    r"""Differentiable objective: \eta(x_R) = (1 / W) \sum_i w_i * f_i(x_L, x_R)^2."""
    def evaluate_objective(x_right_val: torch.Tensor) -> torch.Tensor:
        f = subpixel_fill_fraction_1d(x_nodes, torch.as_tensor(x_left_fixed, dtype=dtype, device=dev), x_right_val)
        # Physical quadratic power/overlap objective |I|^2
        weighted = torch.sum(field_weight * (f ** 2))
        return weighted / torch.sum(field_weight)

    points_data = []
    max_interior_rel_err = 0.0
    interior_passed = True

    # Sweep k from 0 to 20 (step = 1 nm)
    for k in range(n_points):
        # x_R position
        offset_nm = k * (dx_nm / (n_points - 1))
        x_r_val = x_lo_cell + offset_nm * 1e-3

        # 1. Adjoint derivative via autograd
        x_r_t = torch.tensor(x_r_val, dtype=dtype, device=dev, requires_grad=True)
        obj = evaluate_objective(x_r_t)
        obj.backward()
        adj_grad = float(x_r_t.grad.item())
        obj_val = float(obj.item())

        # 2. Central finite difference
        with torch.no_grad():
            obj_plus = float(evaluate_objective(torch.tensor(x_r_val + h_um, dtype=dtype, device=dev)).item())
            obj_minus = float(evaluate_objective(torch.tensor(x_r_val - h_um, dtype=dtype, device=dev)).item())
            cfd_grad = (obj_plus - obj_minus) / (2.0 * h_um)

            # One-sided forward and backward differences (for boundary kink characterization)
            fwd_grad = (obj_plus - obj_val) / h_um
            bwd_grad = (obj_val - obj_minus) / h_um

        is_interior = (0 < k < n_points - 1)
        rel_err = abs(adj_grad - cfd_grad) / max(abs(adj_grad), 1e-12)

        if is_interior:
            max_interior_rel_err = max(max_interior_rel_err, rel_err)
            if rel_err > 1e-5:
                interior_passed = False

        points_data.append({
            "k": k,
            "offset_nm": offset_nm,
            "x_r_um": x_r_val,
            "objective": obj_val,
            "adjoint_grad": adj_grad,
            "cfd_grad": cfd_grad,
            "fwd_grad": fwd_grad,
            "bwd_grad": bwd_grad,
            "relative_error": rel_err,
            "is_interior": is_interior,
        })

    # Characterize kink at boundary k=0 (entry to cell) and k=20 (exit from cell)
    # At k=0: left derivative comes from outside cell (smaller or zero fill slope in cell_index)
    # At k=20: right derivative reaches full cell fill (slope transitions)
    k0_fwd = points_data[0]["fwd_grad"]
    k0_bwd = points_data[0]["bwd_grad"]
    kink_entry_jump = abs(k0_fwd - k0_bwd)

    k20_fwd = points_data[-1]["fwd_grad"]
    k20_bwd = points_data[-1]["bwd_grad"]
    kink_exit_jump = abs(k20_fwd - k20_bwd)

    # Continuity check: maximum adjacent objective jump should be smooth: O(dx_step)
    obj_jumps = [
        abs(points_data[i + 1]["objective"] - points_data[i]["objective"])
        for i in range(len(points_data) - 1)
    ]
    max_obj_jump = max(obj_jumps)
    continuity_verified = max_obj_jump < 0.05  # Smooth continuous transition

    return {
        "sweep_passed": interior_passed and continuity_verified,
        "max_interior_rel_err": max_interior_rel_err,
        "tolerance": 1e-5,
        "continuity_verified": continuity_verified,
        "max_adjacent_obj_jump": max_obj_jump,
        "kink_entry_jump": kink_entry_jump,
        "kink_exit_jump": kink_exit_jump,
        "num_points": n_points,
        "cell_dx_nm": dx_nm,
        "points": points_data,
    }


def evaluate_nominal_and_corner_etch_bias(
    geom: ContinuousEtchBiasGeometry,
    shifts_nm: Tuple[float, float, float] = (-10.0, 0.0, 10.0),
) -> Dict[str, Any]:
    """Evaluate subpixel geometry across nominal and +/- 10 nm etch-bias process shifts."""
    results = {}
    for shift in shifts_nm:
        shift_um = shift * 1e-3
        rho = geom.compute_density(shift_um)
        eps = geom.density_to_epsilon_1d(rho)
        mean_density = float(torch.mean(rho).item())
        mean_eps = float(torch.mean(eps).item())
        results[f"shift_{int(shift):+d}nm"] = {
            "shift_nm": shift,
            "shift_um": shift_um,
            "mean_density": mean_density,
            "mean_epsilon": mean_eps,
            "min_epsilon": float(torch.min(eps).item()),
            "max_epsilon": float(torch.max(eps).item()),
            "density_sha256": hex(hash(tuple(rho.cpu().numpy().round(8))))[2:],
        }
    return results


if __name__ == "__main__":
    print("Testing continuous subpixel etch bias and 21-point edge sweep...")
    sweep_res = run_21point_edge_crossing_sweep()
    print(f"Sweep passed: {sweep_res['sweep_passed']}, max interior rel err: {sweep_res['max_interior_rel_err']:.3e}")
    assert sweep_res["sweep_passed"], f"Sweep failed: {sweep_res}"

    geom = ContinuousEtchBiasGeometry()
    corners = evaluate_nominal_and_corner_etch_bias(geom)
    for name, c in corners.items():
        print(f"Corner {name}: mean_density={c['mean_density']:.4f}, mean_eps={c['mean_epsilon']:.4f}")
    print("SUCCESS: Continuous etch-bias and 21-point edge sweep verified!")
