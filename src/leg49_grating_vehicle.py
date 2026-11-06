"""Canonical 2D SOI grating coupler test vehicle for inverse design (Wave D — LEG 49).

Specifications:
- Operating wavelength: 1.55 um (1550 nm).
- Material stack: 220 nm Silicon core (n = 3.477, eps = 12.0895) on SiO2 substrate (n = 1.444, eps = 2.0851).
- Yee cell spacing: dx = dy = 0.020 um (20 nm standard).
- Differentiable mapping: 1D design density rho(x) -> 3D Yee permittivity tensor epsilon(x, y, z).
- Mode overlap and upward Poynting flux evaluation.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch

from torchfdtd import (
    AdjointOptions,
    Boundaries,
    BoundaryFace,
    DifferentiablePlaneSimulation,
    FieldMonitor,
    Project,
    Region,
    Source,
)
from torchfdtd.solver import C0


def get_default_device() -> str:
    """Return 'cuda' if GPU kernel execution is actually supported, else 'cpu'."""
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


class GratingVehicle2D:
    """Canonical 2D SOI grating coupler test vehicle."""

    def __init__(
        self,
        dx_um: float = 0.020,
        dy_um: float = 0.020,
        lx_um: float = 2.5,
        core_h_um: float = 0.220,
        etch_depth_um: float = 0.220,
        wavelength_um: float = 1.550,
        fiber_angle_deg: float = 10.0,
        fiber_waist_um: float = 5.2,
        steps: int = 1000,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dx_um = float(dx_um)
        self.dy_um = float(dy_um)
        self.lx_um = float(lx_um)
        self.core_h_um = float(core_h_um)
        self.etch_depth_um = float(etch_depth_um)
        self.wavelength_um = float(wavelength_um)
        self.fiber_angle_deg = float(fiber_angle_deg)
        self.fiber_waist_um = float(fiber_waist_um)
        self.steps = int(steps)
        eff_dev = device if device is not None else get_default_device()
        self.device = torch.device(eff_dev)
        self.dtype = dtype
        self._reference_flux: float | None = None

        # Material indices & permittivities
        self.n_core = 3.477
        self.n_clad = 1.444
        self.eps_core = self.n_core**2
        self.eps_clad = self.n_clad**2
        self.delta_eps = self.eps_core - self.eps_clad

        # Layout geometry
        self.slab_y_um = 0.30  # Transverse slab in y (>= 5 cells even at 0.05 um mesh)
        self.half_z_um = 1.80  # Vertical z span from -1.8 to +1.8 um
        self.period_x_um = self.lx_um  # Propagation along x (or design region)
        self.frequency_hz = C0 / (self.wavelength_um * 1e-6)

        # Monitor and source positions along vertical z
        self.source_z_um = -0.50
        self.monitor_top_z_um = 1.10
        self.monitor_bottom_z_um = -1.10
        self.pml_um = 0.40

        # Project and model
        self.project = self._build_project()
        self.model = DifferentiablePlaneSimulation(
            self.project, AdjointOptions(checkpoints=2)
        )

        # Coordinate axes
        self.region = self.project.region
        self.nx = self.region.shape[0]
        self.ny = self.region.shape[1]
        self.nz = self.region.shape[2]

        # Design region in x (e.g. interior cells)
        self.num_design_cells = self.nx

    @property
    def plan_hash(self) -> str:
        """The canonical plan hash SHA-256 string for the project."""
        return self.model.plan.plan_hash

    def get_reference_input_flux(self) -> float:
        """Compute or return cached forward reference flux in homogeneous cladding."""
        if self._reference_flux is not None:
            return self._reference_flux

        with torch.no_grad():
            bg_eps = torch.full(
                (self.nx, self.ny, self.nz),
                self.eps_clad,
                dtype=self.dtype,
                device=self.device,
            )
            res = self.model(bg_eps, [self.frequency_hz])
            ref_top = abs(float(res["top"].flux().item()))
            self._reference_flux = max(ref_top, 1e-45)
            return self._reference_flux

    def _build_project(self) -> Project:
        """Construct FDTD Project with periodic transverse boundaries and PML in z."""
        mesh = self.dx_um
        pml_cells = int(math.ceil(self.pml_um / mesh))

        # Boundaries: PML on x and z, periodic on y (2D invariant slab)
        boundaries = Boundaries(
            x_min=BoundaryFace(kind="pml", layers=pml_cells),
            x_max=BoundaryFace(kind="pml", layers=pml_cells),
            y_min=BoundaryFace(kind="periodic"),
            y_max=BoundaryFace(kind="periodic"),
            z_min=BoundaryFace(kind="pml", layers=pml_cells),
            z_max=BoundaryFace(kind="pml", layers=pml_cells),
        )

        region = Region(
            dimension="3d",
            size=(self.period_x_um, self.slab_y_um, 2 * self.half_z_um),
            mesh=mesh,
            mesh_steps=(self.dx_um, self.dy_um, self.dx_um),
            pml_cells=pml_cells,
            material_sampling="yee",
            boundaries=boundaries,
            precision="float64" if self.dtype == torch.float64 else "float32",
            steps=self.steps,
        )

        source_lx = max(0.2, self.period_x_um - 2 * self.pml_um - 2 * mesh)

        sources = [
            Source(
                id="wave_source",
                kind="plane",
                normal="z",
                direction="+",
                size=(source_lx, self.slab_y_um, 0),
                center=(0, 0, self.source_z_um),
                component="Ex",
                wavelength=self.wavelength_um,
                pulse_cycles=1,
            )
        ]

        monitors = [
            FieldMonitor(
                id="top",
                normal="z",
                size=(source_lx, self.slab_y_um, 0),
                center=(0, 0, self.monitor_top_z_um),
            ),
            FieldMonitor(
                id="bottom",
                normal="z",
                size=(source_lx, self.slab_y_um, 0),
                center=(0, 0, self.monitor_bottom_z_um),
            ),
        ]

        return Project(
            name="canonical_grating_2d",
            region=region,
            sources=sources,
            monitors=monitors,
        )

    def density_to_epsilon(self, density_1d: torch.Tensor) -> torch.Tensor:
        """Differentiable mapping: 1D density rho(x) -> 3D Yee permittivity tensor epsilon(x, y, z).

        The grating layer occupies z in [-core_h_um/2, core_h_um/2].
        Substrate (z < -core_h_um/2) has eps_clad.
        Superstrate (z > core_h_um/2) has eps_clad.
        Inside grating layer, eps = eps_clad + delta_eps * rho(x).
        """
        device = density_1d.device
        dtype = self.dtype

        # Ensure density is correctly shaped (nx,)
        rho_flat = density_1d.view(-1).to(dtype=dtype, device=device)
        if rho_flat.shape[0] != self.nx:
            # Resample or interpolate to self.nx
            rho_flat = torch.nn.functional.interpolate(
                rho_flat[None, None, :], size=self.nx, mode="linear", align_corners=False
            )[0, 0]

        # Base background: eps_clad everywhere
        eps = torch.full(
            (self.nx, self.ny, self.nz),
            self.eps_clad,
            dtype=dtype,
            device=device,
        )

        # Identify z cells inside grating core layer
        z_nodes = torch.tensor(
            self.region.mesh_nodes[2], dtype=dtype, device=device
        )
        z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])
        half_h = self.core_h_um / 2.0

        # Fractional vertical cell overlap
        z_lo = z_nodes[:-1]
        z_hi = z_nodes[1:]
        overlap = (torch.minimum(z_hi, torch.tensor(half_h, device=device)) -
                   torch.maximum(z_lo, torch.tensor(-half_h, device=device))).clamp_min(0.0)
        fz = overlap / (z_hi - z_lo)  # shape (nz,)

        # Broadcast: rho_flat (nx,) x ones(ny) x fz (nz,)
        grating_delta = (
            rho_flat[:, None, None] * fz[None, None, :] * self.delta_eps
        )
        eps = eps + grating_delta
        return eps

    def evaluate_upward_flux(
        self, top_plane: Any, input_flux: torch.Tensor | float | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Calculate total upward-radiated Poynting flux ratio Phi_up / P_in."""
        if input_flux is None:
            input_flux = self.get_reference_input_flux()
        flux_val = top_plane.flux()
        if isinstance(input_flux, torch.Tensor):
            ratio = flux_val / (input_flux.abs() + 1e-45)
        else:
            ratio = flux_val / max(abs(input_flux), 1e-45)

        return ratio, {
            "raw_flux": float(flux_val.item()) if flux_val.numel() == 1 else float(flux_val[0].item()),
            "power_ratio": float(ratio.item()) if ratio.numel() == 1 else float(ratio[0].item()),
        }

    def evaluate_target_overlap(
        self, top_plane: Any, input_flux: torch.Tensor | float = 1.0
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Calculate complex mode overlap with tilted Gaussian fiber beam (SMF-28)."""
        # top_plane.fields: shape (frequency, points, 6)
        # points are collocated field sample points across the monitor plane
        fields = top_plane.fields
        ex = fields[0, :, 0]  # Ex component
        weights = top_plane.weights  # integration weights per point
        points_um = top_plane.points_um  # (points, 3)

        x_coords = points_um[:, 0]
        xc = 0.0  # beam center
        theta_rad = math.radians(self.fiber_angle_deg)
        k0 = 2.0 * math.pi / self.wavelength_um
        wx = self.fiber_waist_um / math.cos(theta_rad)

        # Gaussian target beam profile
        phase = -k0 * self.n_clad * (x_coords - xc) * math.sin(theta_rad)
        gaussian_env = torch.exp(-((x_coords - xc) ** 2) / (wx**2))
        fib_beam = gaussian_env * torch.complex(torch.cos(phase), torch.sin(phase))

        # Complex overlap integral: c_fib = int(E * fib*) / sqrt(int |E|^2 * int |fib|^2)
        overlap_num = (ex * fib_beam.conj() * weights).sum()
        norm_e = (ex.abs().square() * weights).sum().clamp_min(1e-12)
        norm_fib = (fib_beam.abs().square() * weights).sum().clamp_min(1e-12)

        overlap_eff = overlap_num.abs().square() / (norm_e * norm_fib)
        raw_flux = top_plane.flux()
        coupling_eff = overlap_eff * (raw_flux / max(float(input_flux), 1e-12))

        return coupling_eff, {
            "mode_overlap": float(overlap_eff.item()),
            "coupling_efficiency": float(coupling_eff.item()),
            "raw_flux": float(raw_flux.item()) if raw_flux.numel() == 1 else float(raw_flux[0].item()),
        }
