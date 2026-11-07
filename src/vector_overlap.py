r"""Calibrated Full-Vector Mode Overlap with 16 P_in P_fib Denominator (Wave E — LEG 53).

Provides unit-power-calibrated Lorentz reciprocity mode overlap for 2D and 3D geometries,
strictly enforcing the 16 P_in P_fib normalization denominator:

    eta_3D(omega) = | \iint_S (E_t x H_{t,fib}^* - H_t x E_{t,fib}^*) . n_hat dS |^2 / (16 P_in P_fib)

And for 2D TE cross-section (propagation normal n_hat = +z_hat, tangential fields Ey, Hx):

    eta_2D(omega) = | \int_{x1}^{x2} (Ey(x) Hx,fib^*(x) + Hx(x) Ey,fib^*(x)) dx |^2 / (16 P_in P_fib)

Where:
    P = 1/2 Re \iint_S (E_t x H_t^*) . n_hat dS

Guarantees 3 analytic self-checks:
1. Identity Test: identical forward mode yields eta = 1.000000000000 +/- 1e-15.
2. Orthogonality Test: orthogonal higher-order mode (e.g. TE1 / HG1) yields eta < 1e-14.
3. Directional Isolation Test: reverse-propagating mode is strictly nulled (eta = 0.0 < 1e-15),
   and forward/backward superposition isolates forward power without reflection crosstalk.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

# Standard vacuum impedance
ETA_0: float = 376.73031366857


def integrate_simpson_or_trapezoid(
    y: Union[torch.Tensor, np.ndarray], dx: float
) -> Union[torch.Tensor, complex, float]:
    """Integrate 1D array over uniform spacing dx using trapezoidal rule."""
    if isinstance(y, torch.Tensor):
        # trapezoid integration along last dimension
        return torch.trapezoid(y, dx=dx, dim=-1)
    return float(np.trapezoid(y, dx=dx, axis=-1)) if y.dtype != complex else complex(np.trapezoid(y, dx=dx, axis=-1))


def compute_poynting_power_2d(
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    dx: float,
) -> Union[torch.Tensor, float]:
    r"""Calculate forward time-averaged Poynting power across 2D TE line monitor (propagation +z).

    P = 1/2 * |Re \int (-Ey * Hx^*) dx|
    """
    if isinstance(ey, torch.Tensor):
        integrand = -ey * torch.conj(hx)
        p_val = 0.5 * torch.real(torch.trapezoid(integrand, dx=dx, dim=-1))
        return torch.abs(p_val)
    integrand = -ey * np.conj(hx)
    p_val = 0.5 * np.real(np.trapezoid(integrand, dx=dx, axis=-1))
    return float(abs(p_val))


def compute_poynting_power_3d(
    ex: Union[torch.Tensor, np.ndarray],
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    hy: Union[torch.Tensor, np.ndarray],
    dx: float,
    dy: float,
) -> Union[torch.Tensor, float]:
    r"""Calculate forward Poynting power across 3D planar monitor with normal +z.

    P = 1/2 * |Re \iint (Ex * Hy^* - Ey * Hx^*) dx dy|
    """
    if isinstance(ex, torch.Tensor):
        integrand = ex * torch.conj(hy) - ey * torch.conj(hx)
        p_val = 0.5 * torch.real(torch.trapezoid(torch.trapezoid(integrand, dx=dy, dim=-1), dx=dx, dim=-1))
        return torch.abs(p_val)
    integrand = ex * np.conj(hy) - ey * np.conj(hx)
    p_val = 0.5 * np.real(np.trapezoid(np.trapezoid(integrand, dx=dy, axis=-1), dx=dx, axis=-1))
    return float(abs(p_val))


def compute_lorentz_reciprocity_integral_2d(
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    ey_fib: Union[torch.Tensor, np.ndarray],
    hx_fib: Union[torch.Tensor, np.ndarray],
    dx: float,
) -> Union[torch.Tensor, complex]:
    r"""Calculate 2D TE Lorentz reciprocity projection integral:

    I_rec = -\int (Ey * Hx,fib^* + Hx * Ey,fib^*) dx
    """
    if isinstance(ey, torch.Tensor):
        integrand = -(ey * torch.conj(hx_fib) + hx * torch.conj(ey_fib))
        return torch.trapezoid(integrand, dx=dx, dim=-1)
    integrand = -(ey * np.conj(hx_fib) + hx * np.conj(ey_fib))
    return complex(np.trapezoid(integrand, dx=dx, axis=-1))


def compute_lorentz_reciprocity_integral_3d(
    ex: Union[torch.Tensor, np.ndarray],
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    hy: Union[torch.Tensor, np.ndarray],
    ex_fib: Union[torch.Tensor, np.ndarray],
    ey_fib: Union[torch.Tensor, np.ndarray],
    hx_fib: Union[torch.Tensor, np.ndarray],
    hy_fib: Union[torch.Tensor, np.ndarray],
    dx: float,
    dy: float,
) -> Union[torch.Tensor, complex]:
    r"""Calculate 3D Lorentz reciprocity projection integral for normal +z:

    I_rec = \iint ( (Et x H_fib^* - Ht x E_fib^*) . z_hat ) dx dy
          = \iint ( Ex * Hy_fib^* - Ey * Hx_fib^* - Hx * Ey_fib^* + Hy * Ex_fib^* ) dx dy
    """
    if isinstance(ex, torch.Tensor):
        integrand = (
            ex * torch.conj(hy_fib)
            - ey * torch.conj(hx_fib)
            - hx * torch.conj(ey_fib)
            + hy * torch.conj(ex_fib)
        )
        return torch.trapezoid(torch.trapezoid(integrand, dx=dy, dim=-1), dx=dx, dim=-1)
    integrand = (
        ex * np.conj(hy_fib)
        - ey * np.conj(hx_fib)
        - hx * np.conj(ey_fib)
        + hy * np.conj(ex_fib)
    )
    return complex(np.trapezoid(np.trapezoid(integrand, dx=dy, axis=-1), dx=dx, axis=-1))


def vector_mode_overlap_2d(
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    ey_fib: Union[torch.Tensor, np.ndarray],
    hx_fib: Union[torch.Tensor, np.ndarray],
    dx: float,
    p_in: Union[torch.Tensor, float, None] = None,
    p_fib: Union[torch.Tensor, float, None] = None,
    eps_guard: float = 1e-300,
) -> Union[torch.Tensor, float]:
    """Compute unit-power-calibrated 2D TE mode overlap efficiency eta in [0, 1].

    Formula:
        eta = |I_rec|^2 / (16 * P_in * P_fib)
    """
    is_torch = isinstance(ey, torch.Tensor)

    i_rec = compute_lorentz_reciprocity_integral_2d(ey, hx, ey_fib, hx_fib, dx)

    if p_fib is None:
        p_fib = compute_poynting_power_2d(ey_fib, hx_fib, dx)
    if p_in is None:
        p_in = compute_poynting_power_2d(ey, hx, dx)

    if is_torch:
        num = torch.abs(i_rec) ** 2
        p_fib_t = p_fib if isinstance(p_fib, torch.Tensor) else torch.tensor(p_fib, dtype=num.dtype, device=num.device)
        p_in_t = p_in if isinstance(p_in, torch.Tensor) else torch.tensor(p_in, dtype=num.dtype, device=num.device)
        denom = 16.0 * p_in_t * p_fib_t
        safe_denom = torch.clamp(denom, min=eps_guard)
        eta = torch.where(denom > eps_guard, num / safe_denom, torch.zeros_like(num))
        return eta
    else:
        num = float(abs(i_rec) ** 2)
        denom = 16.0 * float(p_in) * float(p_fib)
        if denom <= eps_guard:
            return 0.0
        return float(num / denom)


def vector_mode_overlap_3d(
    ex: Union[torch.Tensor, np.ndarray],
    ey: Union[torch.Tensor, np.ndarray],
    hx: Union[torch.Tensor, np.ndarray],
    hy: Union[torch.Tensor, np.ndarray],
    ex_fib: Union[torch.Tensor, np.ndarray],
    ey_fib: Union[torch.Tensor, np.ndarray],
    hx_fib: Union[torch.Tensor, np.ndarray],
    hy_fib: Union[torch.Tensor, np.ndarray],
    dx: float,
    dy: float,
    p_in: Union[torch.Tensor, float, None] = None,
    p_fib: Union[torch.Tensor, float, None] = None,
    eps_guard: float = 1e-300,
) -> Union[torch.Tensor, float]:
    """Compute unit-power-calibrated 3D mode overlap efficiency eta in [0, 1].

    Formula:
        eta = |I_rec|^2 / (16 * P_in * P_fib)
    """
    is_torch = isinstance(ex, torch.Tensor)
    i_rec = compute_lorentz_reciprocity_integral_3d(
        ex, ey, hx, hy, ex_fib, ey_fib, hx_fib, hy_fib, dx, dy
    )
    if p_fib is None:
        p_fib = compute_poynting_power_3d(ex_fib, ey_fib, hx_fib, hy_fib, dx, dy)
    if p_in is None:
        p_in = compute_poynting_power_3d(ex, ey, hx, hy, dx, dy)

    if is_torch:
        num = torch.abs(i_rec) ** 2
        p_fib_t = p_fib if isinstance(p_fib, torch.Tensor) else torch.tensor(p_fib, dtype=num.dtype, device=num.device)
        p_in_t = p_in if isinstance(p_in, torch.Tensor) else torch.tensor(p_in, dtype=num.dtype, device=num.device)
        denom = 16.0 * p_in_t * p_fib_t
        safe_denom = torch.clamp(denom, min=eps_guard)
        return torch.where(denom > eps_guard, num / safe_denom, torch.zeros_like(num))
    else:
        num = float(abs(i_rec) ** 2)
        denom = 16.0 * float(p_in) * float(p_fib)
        if denom <= eps_guard:
            return 0.0
        return float(num / denom)


def vector_mode_overlap_collocated(
    fields: torch.Tensor,
    fields_fib: torch.Tensor,
    weights: torch.Tensor,
    p_in: Union[torch.Tensor, float, None] = None,
    p_fib: Union[torch.Tensor, float, None] = None,
    normal: str = "z",
    eps_guard: float = 1e-300,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r"""Compute calibrated Lorentz reciprocity mode overlap directly from collocated monitor fields.

    fields: shape (..., points, 6), components (Ex, Ey, Ez, Hx, Hy, Hz).
    fields_fib: shape (..., points, 6), target fiber mode fields.
    weights: shape (points,) or broadcastable, surface integration weights.
    """
    if normal == "z":
        ex, ey = fields[..., 0], fields[..., 1]
        hx, hy = fields[..., 3], fields[..., 4]
        ex_f, ey_f = fields_fib[..., 0], fields_fib[..., 1]
        hx_f, hy_f = fields_fib[..., 3], fields_fib[..., 4]

        # Lorentz reciprocity integrand: (Et x H_fib^* - Ht x E_fib^*) . z_hat
        integrand_rec = (
            ex * torch.conj(hy_f)
            - ey * torch.conj(hx_f)
            - hx * torch.conj(ey_f)
            + hy * torch.conj(ex_f)
        )
        # Forward Poynting power integrand: (Et x Ht*) . z_hat
        integrand_p_fib = 0.5 * torch.real(ex_f * torch.conj(hy_f) - ey_f * torch.conj(hx_f))
        integrand_p_fwd = 0.5 * torch.real(ex * torch.conj(hy) - ey * torch.conj(hx))
    else:
        raise ValueError(f"Unsupported normal: {normal}")

    i_rec = torch.sum(integrand_rec * weights, dim=-1)
    if p_fib is None:
        p_fib = torch.abs(torch.sum(integrand_p_fib * weights, dim=-1))
    if p_in is None:
        p_in = torch.abs(torch.sum(integrand_p_fwd * weights, dim=-1))

    num = torch.abs(i_rec) ** 2
    p_fib_t = p_fib if isinstance(p_fib, torch.Tensor) else torch.tensor(p_fib, dtype=num.dtype, device=num.device)
    p_in_t = p_in if isinstance(p_in, torch.Tensor) else torch.tensor(p_in, dtype=num.dtype, device=num.device)
    denom = 16.0 * p_in_t * p_fib_t
    safe_denom = torch.clamp(denom, min=eps_guard)
    eta = torch.where(denom > eps_guard, num / safe_denom, torch.zeros_like(num))

    metrics = {
        "i_rec": complex(i_rec.item()) if i_rec.numel() == 1 else i_rec,
        "numerator": float(num.item()) if num.numel() == 1 else num,
        "denominator": float(denom.item()) if isinstance(denom, torch.Tensor) and denom.numel() == 1 else denom,
        "p_fib": float(p_fib_t.item()) if p_fib_t.numel() == 1 else p_fib_t,
        "p_in": float(p_in_t.item()) if p_in_t.numel() == 1 else p_in_t,
        "eta": float(eta.item()) if eta.numel() == 1 else eta,
    }
    return eta, metrics


def gaussian_fiber_mode_2d(
    x: Union[torch.Tensor, np.ndarray],
    waist_um: float = 5.2,
    wavelength_um: float = 1.55,
    theta_deg: float = 10.0,
    n_clad: float = 1.444,
    x0_um: float = 0.0,
    z_imp: Union[float, None] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Synthesize analytical 2D TE fundamental Gaussian fiber mode (Ey, Hx).

    Parameters:
    - x: coordinate grid in um.
    - waist_um: 1/e field radius w_0 in um (MFD/2).
    - wavelength_um: vacuum wavelength in um.
    - theta_deg: tilt angle from surface normal (+z) in degrees.
    - n_clad: refractive index of background cladding.
    - x0_um: beam center position in um.
    - z_imp: optical wave impedance Z = eta_0 / n_clad (in Ohms or reduced units).
    """
    theta_rad = math.radians(theta_deg)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # Effective projected waist along monitor line
    w_eff = waist_um / max(cos_theta, 1e-6)

    # Wavevector components in cladding
    k0 = 2.0 * math.pi / wavelength_um
    kx = k0 * n_clad * sin_theta

    eff_z = (ETA_0 / n_clad) if z_imp is None else float(z_imp)

    if isinstance(x, torch.Tensor):
        x_shifted = x - x0_um
        envelope = torch.exp(-((x_shifted / w_eff) ** 2))
        phase = torch.exp(1j * (kx * x_shifted))
        ey = envelope * phase
        hx = -(cos_theta / eff_z) * ey
        return ey, hx
    else:
        x_shifted = x - x0_um
        envelope = np.exp(-((x_shifted / w_eff) ** 2))
        phase = np.exp(1j * (kx * x_shifted))
        ey = envelope * phase
        hx = -(cos_theta / eff_z) * ey
        return ey, hx


def hermite_gaussian_mode_2d(
    x: Union[torch.Tensor, np.ndarray],
    waist_um: float = 5.2,
    order: int = 1,
    wavelength_um: float = 1.55,
    theta_deg: float = 10.0,
    n_clad: float = 1.444,
    x0_um: float = 0.0,
    z_imp: Union[float, None] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Synthesize higher-order Hermite-Gaussian mode (e.g. HG1 / TE1 odd mode).

    For order=1 (odd mode): H_1(u) = 2*u, orthogonal to HG0 across symmetric window.
    """
    theta_rad = math.radians(theta_deg)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    w_eff = waist_um / max(cos_theta, 1e-6)
    k0 = 2.0 * math.pi / wavelength_um
    kx = k0 * n_clad * sin_theta
    eff_z = (ETA_0 / n_clad) if z_imp is None else float(z_imp)

    if isinstance(x, torch.Tensor):
        u = (x - x0_um) / w_eff
        if order == 1:
            poly = 2.0 * u
        elif order == 2:
            poly = 4.0 * (u**2) - 2.0
        else:
            poly = torch.ones_like(u)
        envelope = poly * torch.exp(-(u**2))
        phase = torch.exp(1j * (kx * (x - x0_um)))
        ey = envelope * phase
        hx = -(cos_theta / eff_z) * ey
        return ey, hx
    else:
        u = (x - x0_um) / w_eff
        if order == 1:
            poly = 2.0 * u
        elif order == 2:
            poly = 4.0 * (u**2) - 2.0
        else:
            poly = np.ones_like(u)
        envelope = poly * np.exp(-(u**2))
        phase = np.exp(1j * (kx * (x - x0_um)))
        ey = envelope * phase
        hx = -(cos_theta / eff_z) * ey
        return ey, hx


def run_analytic_self_checks(
    n_points: int = 4001,
    x_span_um: float = 60.0,
    waist_um: float = 5.2,
    wavelength_um: float = 1.55,
    theta_deg: float = 10.0,
    use_torch: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute the three mandatory pre-FDTD analytic field self-checks.

    1. Identity Test:
       Target forward mode vs identical mode with scaling/phase -> eta = 1.0 +/- 1e-15.
    2. Orthogonality Test:
       Fundamental HG0 mode vs odd HG1 mode -> eta < 1e-14.
    3. Directional Isolation Test:
       Superposition of forward (amplitude alpha) and backward (amplitude beta) waves ->
       verifies complete backward component nulling and exact forward power extraction.
    """
    half_x = x_span_um / 2.0
    dx = x_span_um / (n_points - 1)

    if use_torch:
        x_grid = torch.linspace(
            -half_x, half_x, n_points, dtype=torch.float64, device=device
        )
    else:
        x_grid = np.linspace(-half_x, half_x, n_points, dtype=np.float64)

    # 1. Identity Check
    ey_fib, hx_fib = gaussian_fiber_mode_2d(
        x_grid, waist_um=waist_um, wavelength_um=wavelength_um, theta_deg=theta_deg
    )
    p_fib = compute_poynting_power_2d(ey_fib, hx_fib, dx)

    # Test with scaling amplitude A and phase phi
    scale_amp = 2.45
    phase_shift = 0.7853981633974483  # pi/4
    complex_scalar = scale_amp * (math.cos(phase_shift) + 1j * math.sin(phase_shift))

    ey_scaled = ey_fib * complex_scalar
    hx_scaled = hx_fib * complex_scalar
    p_scaled = compute_poynting_power_2d(ey_scaled, hx_scaled, dx)

    eta_identity_raw = vector_mode_overlap_2d(
        ey_fib, hx_fib, ey_fib, hx_fib, dx, p_in=p_fib, p_fib=p_fib
    )
    eta_identity_scaled = vector_mode_overlap_2d(
        ey_scaled, hx_scaled, ey_fib, hx_fib, dx, p_in=p_scaled, p_fib=p_fib
    )

    eta_id_val = float(eta_identity_raw.item() if isinstance(eta_identity_raw, torch.Tensor) else eta_identity_raw)
    eta_id_scaled_val = float(
        eta_identity_scaled.item() if isinstance(eta_identity_scaled, torch.Tensor) else eta_identity_scaled
    )

    identity_passed = abs(eta_id_val - 1.0) <= 1e-15 and abs(eta_id_scaled_val - 1.0) <= 1e-15

    # 2. Orthogonality Check (HG0 vs HG1 odd mode)
    ey_hg1, hx_hg1 = hermite_gaussian_mode_2d(
        x_grid, waist_um=waist_um, order=1, wavelength_um=wavelength_um, theta_deg=theta_deg
    )
    p_hg1 = compute_poynting_power_2d(ey_hg1, hx_hg1, dx)
    eta_ortho = vector_mode_overlap_2d(
        ey_hg1, hx_hg1, ey_fib, hx_fib, dx, p_in=p_hg1, p_fib=p_fib
    )
    eta_ortho_val = float(eta_ortho.item() if isinstance(eta_ortho, torch.Tensor) else eta_ortho)
    ortho_passed = eta_ortho_val < 1e-14

    # 3. Directional Isolation Check
    # Forward wave: Ey_fwd = ey_fib, Hx_fwd = hx_fib
    # Backward wave: Ey_bwd = ey_fib, Hx_bwd = -hx_fib
    alpha = 0.8
    beta = 0.6
    ey_super = (alpha + beta) * ey_fib
    hx_super = (alpha - beta) * hx_fib

    p_super = compute_poynting_power_2d(ey_super, hx_super, dx)
    p_fwd_pure = (alpha**2) * float(p_fib.item() if isinstance(p_fib, torch.Tensor) else p_fib)

    # Pure reverse wave (alpha=0, beta=1)
    ey_rev = ey_fib
    hx_rev = -hx_fib
    p_rev = compute_poynting_power_2d(ey_rev, hx_rev, dx)
    eta_rev = vector_mode_overlap_2d(
        ey_rev, hx_rev, ey_fib, hx_fib, dx, p_in=p_rev, p_fib=p_fib
    )
    eta_rev_val = float(eta_rev.item() if isinstance(eta_rev, torch.Tensor) else eta_rev)

    # Mixed superposition wave
    i_rec_super = compute_lorentz_reciprocity_integral_2d(
        ey_super, hx_super, ey_fib, hx_fib, dx
    )
    i_rec_fwd = compute_lorentz_reciprocity_integral_2d(
        alpha * ey_fib, alpha * hx_fib, ey_fib, hx_fib, dx
    )

    c_super = complex(i_rec_super.item() if isinstance(i_rec_super, torch.Tensor) else i_rec_super)
    c_fwd = complex(i_rec_fwd.item() if isinstance(i_rec_fwd, torch.Tensor) else i_rec_fwd)
    diff_super_fwd = float(abs(c_super - c_fwd))

    # Overlap efficiency with respect to forward incident power
    eta_super_vs_fwd = vector_mode_overlap_2d(
        ey_super, hx_super, ey_fib, hx_fib, dx, p_in=p_fwd_pure, p_fib=p_fib
    )
    eta_super_vs_fwd_val = float(
        eta_super_vs_fwd.item() if isinstance(eta_super_vs_fwd, torch.Tensor) else eta_super_vs_fwd
    )

    directional_passed = (
        eta_rev_val <= 1e-15
        and diff_super_fwd <= 1e-14
        and abs(eta_super_vs_fwd_val - 1.0) <= 1e-14
    )

    all_passed = identity_passed and ortho_passed and directional_passed

    return {
        "all_passed": all_passed,
        "identity_test": {
            "passed": identity_passed,
            "eta_raw": eta_id_val,
            "eta_raw_err": abs(eta_id_val - 1.0),
            "eta_scaled": eta_id_scaled_val,
            "eta_scaled_err": abs(eta_id_scaled_val - 1.0),
            "tolerance": 1e-15,
        },
        "orthogonality_test": {
            "passed": ortho_passed,
            "eta_ortho": eta_ortho_val,
            "tolerance": 1e-14,
        },
        "directional_isolation_test": {
            "passed": directional_passed,
            "eta_pure_backward": eta_rev_val,
            "backward_null_tolerance": 1e-15,
            "reciprocity_integral_crosstalk_diff": diff_super_fwd,
            "eta_forward_extracted": eta_super_vs_fwd_val,
        },
    }



if __name__ == "__main__":
    res = run_analytic_self_checks()
    print("Analytic Self Checks Result:")
    import json
    print(json.dumps(res, indent=2))
    assert res["all_passed"], f"Checks failed: {res}"
    print("SUCCESS: All 3 analytic self-checks passed!")
