"""M3 constraint enforcement and analytic Jacobian for inverse design (Wave D — LEG 49).

This module implements:
1. Continuous spatial filtering (conic kernel R >= 150 nm, normalized sum K = 1).
2. Tanh threshold projection with beta-continuation.
3. Explicit analytic Jacobian d(eps)/d(rho) and central finite difference verification (<= 1e-6).
4. Three-tier M3 constraint enforcement:
   - Tier 1: Conic filter + projection in forward pass.
   - Tier 2: Differentiable binarization penalty P_M3(rho) = 4/N * sum(rho * (1 - rho)).
   - Tier 3: Discrete morphological projection close(open(b, m), m) with m = ceil(150 nm / dx).
   - Zero-violation verification via fabrication.measure_feature_sizes.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch


class M3FilterProjection:
    """Mathematical spatial filtering, projection, and M3 fabrication constraint enforcement."""

    def __init__(
        self,
        dx_um: float = 0.020,
        min_feature_um: float = 0.150,
        filter_radius_um: float = 0.150,
        beta: float = 1.0,
        eta: float = 0.5,
        length: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dx_um = float(dx_um)
        self.min_feature_um = float(min_feature_um)
        self.filter_radius_um = float(filter_radius_um)
        self.beta = float(beta)
        self.eta = float(eta)
        self.dtype = dtype
        self.device = torch.device(device)
        self._filter_matrices: dict[int, torch.Tensor] = {}

        if length is not None:
            self._get_filter_matrix(length)

    @property
    def structuring_element_cells(self) -> int:
        """Structuring element size m in cells for M3 discrete morphological projection."""
        return int(math.ceil(self.min_feature_um / self.dx_um - 1e-9))

    def _get_filter_matrix(self, n: int) -> torch.Tensor:
        """Get or build boundary-normalized 1D conic convolution matrix W of shape (n, n)."""
        if n in self._filter_matrices:
            return self._filter_matrices[n]

        rad_cells = int(math.ceil(self.filter_radius_um / self.dx_um - 1e-9))
        w = torch.zeros((n, n), dtype=self.dtype, device=self.device)

        # Distances and conic weights: max(0, 1 - d/R)
        for i in range(n):
            for j in range(max(0, i - rad_cells), min(n, i + rad_cells + 1)):
                dist = abs(i - j) * self.dx_um
                if dist < self.filter_radius_um:
                    weight = 1.0 - dist / self.filter_radius_um
                    w[i, j] = weight
            row_sum = w[i].sum()
            if row_sum > 0:
                w[i] /= row_sum

        self._filter_matrices[n] = w
        return w

    def filter(self, rho: torch.Tensor) -> torch.Tensor:
        """Conic spatial filter: rho_bar = W @ rho."""
        orig_shape = rho.shape
        rho_flat = rho.view(-1)
        n = rho_flat.shape[0]
        w = self._get_filter_matrix(n).to(device=rho.device, dtype=rho.dtype)
        filtered = w @ rho_flat
        return filtered.view(orig_shape)

    def project(self, rho_filt: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """Modified hyperbolic tangent threshold projection with continuation sharpness beta."""
        b = self.beta if beta is None else float(beta)
        if b <= 0.0:
            return rho_filt

        lower = math.tanh(b * self.eta)
        denom = lower + math.tanh(b * (1.0 - self.eta))
        if abs(denom) < 1e-12:
            return rho_filt

        proj = (lower + torch.tanh(b * (rho_filt - self.eta))) / denom
        return proj

    def forward(self, rho: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """Composite mapping: rho -> rho_filt -> rho_proj."""
        return self.project(self.filter(rho), beta=beta)

    def compute_analytic_jacobian(
        self, rho: torch.Tensor, beta: float | None = None
    ) -> torch.Tensor:
        """Compute exact analytic Jacobian matrix J[i, j] = d(rho_proj_i) / d(rho_j).

        Returns:
            Tensor of shape (N, N) where N = rho.numel().
        """
        b = self.beta if beta is None else float(beta)
        rho_flat = rho.view(-1)
        n = rho_flat.shape[0]
        w = self._get_filter_matrix(n).to(device=rho.device, dtype=rho.dtype)

        rho_filt = w @ rho_flat

        if b <= 0.0:
            return w.clone()

        lower = math.tanh(b * self.eta)
        denom = lower + math.tanh(b * (1.0 - self.eta))
        tanh_term = torch.tanh(b * (rho_filt - self.eta))
        # d/du tanh(u) = 1 - tanh^2(u) = sech^2(u)
        d_proj = (b * (1.0 - tanh_term.square())) / denom

        # J = diag(d_proj) @ W
        jacobian = d_proj[:, None] * w
        return jacobian

    def verify_jacobian_finite_difference(
        self,
        rho: torch.Tensor | None = None,
        h: float = 1e-6,
        tolerance: float = 1e-6,
        num_directions: int = 5,
        seed: int = 42,
        beta: float | None = None,
    ) -> dict[str, Any]:
        """Validate analytic Jacobian against central finite differences.

        Computes directional derivatives along random unit vectors in double precision
        and verifies relative error <= tolerance (default 1e-6).
        """
        generator = torch.Generator(device="cpu").manual_seed(seed)
        b = self.beta if beta is None else float(beta)

        if rho is None:
            n = 50
            # Interior values [0.1, 0.9] to avoid boundary clamp discontinuity
            test_rho = (
                torch.rand(n, dtype=torch.float64, generator=generator) * 0.8 + 0.1
            ).to(device=self.device)
        else:
            test_rho = rho.detach().to(dtype=torch.float64, device=self.device).clone()
            n = test_rho.numel()

        jacobian = self.compute_analytic_jacobian(test_rho, beta=b)

        max_rel_error = 0.0
        directional_errors = []

        with torch.no_grad():
            for _ in range(num_directions):
                v = torch.randn(n, dtype=torch.float64, generator=generator).to(
                    device=self.device
                )
                v_norm = v.norm()
                if v_norm < 1e-12:
                    continue
                v = v / v_norm

                analytic_dv = jacobian @ v

                rho_plus = test_rho + h * v
                rho_minus = test_rho - h * v

                f_plus = self.forward(rho_plus, beta=b)
                f_minus = self.forward(rho_minus, beta=b)

                fd_dv = (f_plus - f_minus) / (2.0 * h)

                diff = (fd_dv - analytic_dv).abs().max().item()
                norm_denom = analytic_dv.abs().max().item()
                if norm_denom < 1e-12:
                    rel_err = diff
                else:
                    rel_err = diff / norm_denom

                directional_errors.append(rel_err)
                if rel_err > max_rel_error:
                    max_rel_error = rel_err

        passed = max_rel_error <= tolerance
        return {
            "max_relative_error": float(max_rel_error),
            "directional_errors": directional_errors,
            "tolerance": float(tolerance),
            "passed": bool(passed),
            "num_directions": num_directions,
            "h": float(h),
            "beta": float(b),
        }

    def penalty(self, rho_proj: torch.Tensor) -> torch.Tensor:
        """Tier 2 differentiable penalty: P_M3(rho) = 4/N * sum(rho * (1 - rho)).

        Equals 0 for pure binary {0, 1} and maximum 1.0 for intermediate 0.5 grayscale.
        """
        rho_flat = rho_proj.view(-1)
        return (4.0 / float(rho_flat.shape[0])) * torch.sum(
            rho_flat * (1.0 - rho_flat)
        )

    def project_m3_discrete(
        self, binary_design: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Tier 3 discrete morphological projection.

        Applies alternating sequential filter close(open(b, m), m) with
        structuring element m = ceil(min_feature_um / dx_um) cells.
        Guarantees that all teeth (1s) and trenches (0s) have width >= min_feature_um.
        """
        if isinstance(binary_design, torch.Tensor):
            arr = binary_design.detach().cpu().numpy().astype(bool)
        else:
            arr = np.asarray(binary_design, dtype=bool)

        orig_ndim = arr.ndim
        if orig_ndim == 1:
            arr = arr[:, None]

        m = self.structuring_element_cells

        # Morphological opening removes small solid teeth (< m cells)
        # Morphological closing removes small void trenches (< m cells)
        try:
            from torchfdtd.fabrication import morphological_open

            opened = morphological_open(arr, side=m, boundary="extend")
            # Closing is dual of opening on inverted design: close(x) = ~open(~x)
            closed = ~morphological_open(~opened, side=m, boundary="extend")
            result = closed
        except Exception:
            # Standalone fallback 1D morphology
            result = self._morphology_1d_close_open(arr, m)

        if orig_ndim == 1:
            return result[:, 0]
        return result

    @staticmethod
    def _morphology_1d_close_open(arr: np.ndarray, m: int) -> np.ndarray:
        """Standalone 1D morphology for close(open(arr, m), m)."""
        out = arr.copy()
        for col in range(out.shape[1]):
            col_data = out[:, col].copy()
            # 1. 1D Open: Erode then Dilate
            eroded = np.zeros_like(col_data)
            for i in range(len(col_data)):
                lo = max(0, i - m // 2)
                hi = min(len(col_data), i + (m + 1) // 2)
                if np.all(col_data[lo:hi]):
                    eroded[i] = True
            opened = np.zeros_like(col_data)
            for i in range(len(col_data)):
                lo = max(0, i - m // 2)
                hi = min(len(col_data), i + (m + 1) // 2)
                if np.any(eroded[lo:hi]):
                    opened[i] = True

            # 2. 1D Close: Dilate then Erode on opened
            dilated = np.zeros_like(opened)
            for i in range(len(opened)):
                lo = max(0, i - m // 2)
                hi = min(len(opened), i + (m + 1) // 2)
                if np.any(opened[lo:hi]):
                    dilated[i] = True
            closed = np.zeros_like(opened)
            for i in range(len(opened)):
                lo = max(0, i - m // 2)
                hi = min(len(opened), i + (m + 1) // 2)
                if np.all(dilated[lo:hi]):
                    closed[i] = True

            out[:, col] = closed
        return out

    def check_violations(
        self, binary_design: np.ndarray | torch.Tensor
    ) -> dict[str, Any]:
        """Evaluate minimum tooth width and minimum trench width.

        Returns:
            Dictionary with min_tooth_width_nm, min_trench_width_nm, violations list, and compliance bool.
        """
        if isinstance(binary_design, torch.Tensor):
            arr = binary_design.detach().cpu().numpy().astype(bool)
        else:
            arr = np.asarray(binary_design, dtype=bool)

        if arr.ndim == 1:
            arr_2d = arr[:, None]
        else:
            arr_2d = arr

        try:
            from torchfdtd.fabrication import measure_feature_sizes

            fs = measure_feature_sizes(arr_2d, self.dx_um, boundary="extend")
            min_tooth_um = float(fs.min_linewidth_um)
            min_trench_um = float(fs.min_gap_um)
            viols = list(
                fs.violations(
                    min_linewidth_um=self.min_feature_um,
                    min_gap_um=self.min_feature_um,
                )
            )
        except Exception:
            # Direct run-length calculation fallback
            min_tooth_um, min_trench_um, viols = self._measure_run_lengths(arr_2d)

        min_tooth_nm = min_tooth_um * 1000.0
        min_trench_nm = min_trench_um * 1000.0

        req_nm = self.min_feature_um * 1000.0
        compliant = (
            len(viols) == 0
            and min_tooth_nm >= req_nm - 1e-3
            and min_trench_nm >= req_nm - 1e-3
        )

        return {
            "min_tooth_width_nm": float(min_tooth_nm),
            "min_trench_width_nm": float(min_trench_nm),
            "violations": viols,
            "compliant": bool(compliant),
        }

    def _measure_run_lengths(
        self, arr: np.ndarray
    ) -> tuple[float, float, list[str]]:
        """Measure minimum solid run and void run in um."""
        min_solid_cells = 1000000
        min_void_cells = 1000000
        violations = []

        flat = arr[:, 0]
        n = len(flat)
        if n == 0:
            return 0.0, 0.0, ["empty_design"]

        runs = []
        curr_val = flat[0]
        curr_len = 1
        for i in range(1, n):
            if flat[i] == curr_val:
                curr_len += 1
            else:
                runs.append((curr_val, curr_len))
                curr_val = flat[i]
                curr_len = 1
        runs.append((curr_val, curr_len))

        for val, length in runs:
            if val:
                if length < min_solid_cells:
                    min_solid_cells = length
            else:
                if length < min_void_cells:
                    min_void_cells = length

        min_tooth_um = min_solid_cells * self.dx_um if min_solid_cells < 1000000 else 10.0
        min_trench_um = min_void_cells * self.dx_um if min_void_cells < 1000000 else 10.0

        if min_tooth_um < self.min_feature_um - 1e-6:
            violations.append("min_linewidth")
        if min_trench_um < self.min_feature_um - 1e-6:
            violations.append("min_gap")

        return min_tooth_um, min_trench_um, violations
