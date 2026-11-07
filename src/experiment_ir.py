"""PhotonicExperimentIR: Solver-Agnostic Scene Representation & Compilation Pipeline (LEG 55).

Decouples physical nanophotonic scene intent (stratified stacks, 2D/3D geometries,
multi-spectral sources, Poynting/modal monitors, and execution constraints) from single-solver
internals. Enforces strict Refusal Contracts with UnsupportedBackendError("UNSUPPORTED_BY_BACKEND: ...").
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# ==============================================================================
# Refusal Exception Definition
# ==============================================================================
class UnsupportedBackendError(ValueError):
    """Raised when an Experiment IR specification cannot be mapped losslessly
    or physically to the requested compilation backend.

    All messages MUST be prefixed with 'UNSUPPORTED_BY_BACKEND: '.
    """

    def __init__(self, backend: str, reason: str):
        message = f"UNSUPPORTED_BY_BACKEND: Backend '{backend}' refuses IR: {reason}"
        super().__init__(message)
        self.backend = backend
        self.reason = reason


# ==============================================================================
# Metadata & Governance
# ==============================================================================
class ExperimentMetadata(IRModel):
    ir_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    experiment_id: str = Field(default="exp-001", min_length=1, max_length=128)
    name: str = Field(default="photonic_experiment", min_length=1, max_length=128)
    description: str = Field(default="")
    tags: List[str] = Field(default_factory=list)
    created_utc: str = Field(default="")
    provenance: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Materials
# ==============================================================================
class LorentzPoleSpec(IRModel):
    resonance_rad_s: float = Field(ge=0.0)
    strength_rad_s_squared: float = Field(gt=0.0)
    damping_rad_s: float = Field(ge=0.0)


class MaterialDef(IRModel):
    name: str = Field(min_length=1, max_length=100)
    index: float = Field(default=1.0, ge=1.0, le=20.0)
    loss_k: float = Field(default=0.0, ge=0.0)
    epsilon_tensor: Optional[Tuple[float, float, float, float, float, float]] = None  # (xx, yy, zz, xy, xz, yz)
    poles: List[LorentzPoleSpec] = Field(default_factory=list)

    @property
    def permittivity_real(self) -> float:
        return self.index ** 2 - self.loss_k ** 2

    @property
    def permittivity_imag(self) -> float:
        return 2.0 * self.index * self.loss_k

    @property
    def is_isotropic_dielectric(self) -> bool:
        return self.epsilon_tensor is None and len(self.poles) == 0 and self.loss_k == 0.0


# ==============================================================================
# Geometries
# ==============================================================================
class GeometryPrimitive(IRModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(default="geom", min_length=1, max_length=100)
    material: str = Field(min_length=1, max_length=100)
    mesh_order: int = Field(default=2, ge=1, le=100)
    enabled: bool = True
    tilted_out_of_plane: bool = False


class BoxPrimitive(GeometryPrimitive):
    kind: Literal["box"] = "box"
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]

    @model_validator(mode="after")
    def _positive_size(self):
        if any(s <= 0.0 for s in self.size):
            raise ValueError(f"Box size spans must be strictly positive, got {self.size}")
        return self


class CylinderPrimitive(GeometryPrimitive):
    kind: Literal["cylinder"] = "cylinder"
    center: Tuple[float, float, float]
    radius: float = Field(gt=0.0)
    height: float = Field(gt=0.0)
    axis: Literal["x", "y", "z"] = "z"


class PolygonExtrusion(GeometryPrimitive):
    kind: Literal["polygon"] = "polygon"
    vertices_xy: List[Tuple[float, float]] = Field(min_length=3)
    holes: List[List[Tuple[float, float]]] = Field(default_factory=list)
    z_min: float
    z_max: float
    sidewall_angle_deg: float = Field(default=0.0, ge=-89.0, le=89.0)
    rotation_deg: float = 0.0

    @model_validator(mode="after")
    def _valid_z_span(self):
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be strictly less than z_max ({self.z_max})")
        return self


class RidgeSpec(IRModel):
    center_x_um: float
    width_um: float = Field(gt=0.0)
    material: Optional[str] = None


class PeriodicGrating1D(GeometryPrimitive):
    kind: Literal["periodic_grating_1d"] = "periodic_grating_1d"
    period_um: float = Field(gt=0.0)
    ridge_height_um: float = Field(gt=0.0)
    substrate_material: str = Field(min_length=1)
    cladding_material: str = Field(default="Air", min_length=1)
    ridge_material: str = Field(min_length=1)
    ridges: List[RidgeSpec] = Field(min_length=1)
    z_center_um: float = 0.0


# ==============================================================================
# Boundaries & Domain Execution Constraints
# ==============================================================================
class BoundaryFaceSpec(IRModel):
    kind: Literal["pml", "periodic", "bloch", "pec", "pmc"] = "pml"
    pml_layers: Optional[int] = Field(default=None, ge=4, le=128)
    bloch_phase_rad: float = 0.0


class BoundaryConditionsSpec(IRModel):
    x_min: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))
    x_max: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))
    y_min: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))
    y_max: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))
    z_min: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))
    z_max: BoundaryFaceSpec = Field(default_factory=lambda: BoundaryFaceSpec(kind="pml"))

    @property
    def is_x_periodic(self) -> bool:
        return self.x_min.kind in ("periodic", "bloch") and self.x_max.kind in ("periodic", "bloch")

    @property
    def is_y_periodic(self) -> bool:
        return self.y_min.kind in ("periodic", "bloch") and self.y_max.kind in ("periodic", "bloch")


class DomainSpec(IRModel):
    dimension: Literal["2d", "3d"] = "3d"
    size_um: Tuple[float, float, float]
    boundaries: BoundaryConditionsSpec = Field(default_factory=BoundaryConditionsSpec)
    background_material: str = Field(default="Air")


class ExecutionConstraints(IRModel):
    mesh_um: float = Field(default=0.02, gt=0.0, le=1.0)
    courant_factor: float = Field(default=0.7, gt=0.0, le=0.99)
    max_steps: int = Field(default=10000, ge=10, le=1000000)
    decay_threshold: float = Field(default=1e-6, gt=0.0, lt=1.0)
    precision: Literal["float32", "float64"] = "float32"
    subpixel_smoothing: bool = True


# ==============================================================================
# Sources
# ==============================================================================
class SourceSpec(IRModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(default="source")
    wavelength_um: float = Field(gt=0.0)
    pulse_cycles: float = Field(default=4.0, ge=1.0, le=50.0)


class PlaneWaveSource(SourceSpec):
    kind: Literal["plane_wave"] = "plane_wave"
    normal: Literal["x", "y", "z"] = "z"
    direction: Literal["+", "-"] = "+"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    polarization: Literal["Ex", "Ey", "Ez", "TE", "TM"] = "Ex"
    incident_angle_deg: float = 0.0


class GaussianBeamSource(SourceSpec):
    kind: Literal["gaussian_beam"] = "gaussian_beam"
    normal: Literal["x", "y", "z"] = "z"
    direction: Literal["+", "-"] = "+"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    waist_radius_um: float = Field(default=1.0, gt=0.0)
    waist_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_deg: float = Field(default=0.0, ge=-89.0, le=89.0)
    refractive_index: float = Field(default=1.0, ge=1.0)


class PointDipoleSource(SourceSpec):
    kind: Literal["point_dipole"] = "point_dipole"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    component: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"] = "Ez"


# ==============================================================================
# Monitors
# ==============================================================================
class MonitorSpec(IRModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(default="monitor")


class SpectralFluxMonitor(MonitorSpec):
    kind: Literal["spectral_flux"] = "spectral_flux"
    normal: Literal["x", "y", "z"] = "z"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wavelengths_um: List[float] = Field(min_length=1)


class VolumeFieldMonitor(MonitorSpec):
    kind: Literal["volume_field"] = "volume_field"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    record_components: List[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = Field(default_factory=lambda: ["Ez"])
    downsample_factor: int = Field(default=1, ge=1, le=16)


class DiffractionOrderMonitor(MonitorSpec):
    kind: Literal["diffraction_order"] = "diffraction_order"
    normal: Literal["x", "y", "z"] = "z"
    position_z_um: float = 0.0
    orders: List[Union[int, Tuple[int, int]]] = Field(default_factory=lambda: [-2, -1, 0, 1, 2])
    wavelengths_um: List[float] = Field(min_length=1)


class ModeExpansionMonitor(MonitorSpec):
    kind: Literal["mode_expansion"] = "mode_expansion"
    normal: Literal["x", "y", "z"] = "x"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mode_indices: List[int] = Field(default_factory=lambda: [0])
    normalization: Literal["lorentz_16_pin_pfib", "standard"] = "lorentz_16_pin_pfib"


class TimeSeriesMonitor(MonitorSpec):
    kind: Literal["time_series"] = "time_series"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    component: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"] = "Ez"


# ==============================================================================
# Root PhotonicExperimentIR
# ==============================================================================
GeometryUnion = Union[BoxPrimitive, CylinderPrimitive, PolygonExtrusion, PeriodicGrating1D]
SourceUnion = Union[PlaneWaveSource, GaussianBeamSource, PointDipoleSource]
MonitorUnion = Union[SpectralFluxMonitor, VolumeFieldMonitor, DiffractionOrderMonitor, ModeExpansionMonitor, TimeSeriesMonitor]


class PhotonicExperimentIR(IRModel):
    metadata: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    domain: DomainSpec
    materials: List[MaterialDef] = Field(min_length=1)
    geometries: List[GeometryUnion] = Field(default_factory=list)
    sources: List[SourceUnion] = Field(min_length=1)
    monitors: List[MonitorUnion] = Field(default_factory=list)
    constraints: ExecutionConstraints = Field(default_factory=ExecutionConstraints)
    target_claims: List[Dict[str, Any]] = Field(default_factory=list)

    def get_material(self, name: str) -> MaterialDef:
        for mat in self.materials:
            if mat.name == name:
                return mat
        raise ValueError(f"Material '{name}' referenced in geometry/domain not found in IR.materials")

    def content_hash(self) -> str:
        dumped = self.model_dump(mode="json")
        canonical = json.dumps(dumped, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhotonicExperimentIR:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str) -> PhotonicExperimentIR:
        return cls.model_validate_json(text)

    def compile_to(self, backend: str, **kwargs) -> Any:
        b = backend.lower().strip()
        if b in ("torchfdtd", "fdtd"):
            return compile_to_torchfdtd(self, **kwargs)
        elif b == "rcwa":
            return compile_to_rcwa(self, **kwargs)
        elif b == "gds":
            return compile_to_gds(self, **kwargs)
        else:
            raise UnsupportedBackendError(backend, f"Unknown or unsupported solver backend '{backend}'.")


# ==============================================================================
# Compilation Backends
# ==============================================================================

C0 = 299792458.0  # Speed of light in m/s


def compile_to_torchfdtd(ir: PhotonicExperimentIR) -> Any:
    """Compiles PhotonicExperimentIR into a validated torchfdtd.models.Project instance."""
    from torchfdtd.models import (
        Project,
        Region,
        Boundaries,
        BoundaryFace,
        Material,
        Structure,
        Source,
        FieldMonitor,
        SpectrumSettings,
    )

    # Refusal Check: Signed 32-bit field index limit
    nx = int(math.ceil(ir.domain.size_um[0] / ir.constraints.mesh_um))
    ny = int(math.ceil(ir.domain.size_um[1] / ir.constraints.mesh_um))
    nz = int(math.ceil(ir.domain.size_um[2] / ir.constraints.mesh_um)) if ir.domain.dimension == "3d" else 1
    total_cells = nx * ny * nz
    if total_cells * 3 >= 2 ** 31:
        raise UnsupportedBackendError(
            "TorchFDTD",
            f"Grid size {total_cells} cells exceeds signed 32-bit field index limit (2**31)."
        )

    # 1. Map Region & Boundaries
    b_spec = ir.domain.boundaries
    boundaries = Boundaries(
        x_min=BoundaryFace(kind=b_spec.x_min.kind, layers=b_spec.x_min.pml_layers),
        x_max=BoundaryFace(kind=b_spec.x_max.kind, layers=b_spec.x_max.pml_layers),
        y_min=BoundaryFace(kind=b_spec.y_min.kind, layers=b_spec.y_min.pml_layers),
        y_max=BoundaryFace(kind=b_spec.y_max.kind, layers=b_spec.y_max.pml_layers),
        z_min=BoundaryFace(kind=b_spec.z_min.kind, layers=b_spec.z_min.pml_layers),
        z_max=BoundaryFace(kind=b_spec.z_max.kind, layers=b_spec.z_max.pml_layers),
    )

    region = Region(
        dimension=ir.domain.dimension,
        size=ir.domain.size_um,
        mesh=ir.constraints.mesh_um,
        courant_factor=ir.constraints.courant_factor,
        steps=ir.constraints.max_steps,
        precision=ir.constraints.precision,
        material_sampling="yee" if ir.constraints.subpixel_smoothing else "cell",
        interface_method="subpixel" if ir.constraints.subpixel_smoothing else "staircase",
        boundaries=boundaries,
        bloch_phase=(b_spec.x_min.bloch_phase_rad, b_spec.y_min.bloch_phase_rad, b_spec.z_min.bloch_phase_rad),
    )

    # 2. Map Materials
    materials: List[Material] = []
    for m in ir.materials:
        materials.append(
            Material(
                name=m.name,
                index=m.index,
                model="dielectric",
            )
        )

    # 3. Map Geometries
    structures: List[Structure] = []
    for g in ir.geometries:
        if not g.enabled:
            continue
        if isinstance(g, BoxPrimitive):
            structures.append(
                Structure(
                    name=g.name,
                    kind="rectangle",
                    center=g.center,
                    size=g.size,
                    material=g.material,
                    mesh_order=g.mesh_order,
                )
            )
        elif isinstance(g, CylinderPrimitive):
            structures.append(
                Structure(
                    name=g.name,
                    kind="circle",
                    center=g.center,
                    radius=g.radius,
                    size=(2 * g.radius, 2 * g.radius, g.height),
                    material=g.material,
                    mesh_order=g.mesh_order,
                )
            )
        elif isinstance(g, PolygonExtrusion):
            xs = [p[0] for p in g.vertices_xy]
            ys = [p[1] for p in g.vertices_xy]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            cz = (g.z_min + g.z_max) / 2.0
            sx = max(xs) - min(xs)
            sy = max(ys) - min(ys)
            sz = g.z_max - g.z_min
            structures.append(
                Structure(
                    name=g.name,
                    kind="polygon",
                    center=(cx, cy, cz),
                    size=(sx, sy, sz),
                    vertices=g.vertices_xy,
                    holes=g.holes if g.holes else None,
                    material=g.material,
                    mesh_order=g.mesh_order,
                )
            )
        elif isinstance(g, PeriodicGrating1D):
            lx, ly, lz = ir.domain.size_um
            sub_h = lz / 2.0
            # Substrate
            structures.append(
                Structure(
                    name=f"{g.name}_substrate",
                    kind="rectangle",
                    center=(0.0, 0.0, g.z_center_um - g.ridge_height_um / 2.0 - sub_h / 2.0),
                    size=(lx, ly, sub_h),
                    material=g.substrate_material,
                    mesh_order=1,
                )
            )
            # Grating Teeth
            for idx, r in enumerate(g.ridges):
                mat = r.material or g.ridge_material
                structures.append(
                    Structure(
                        name=f"{g.name}_tooth_{idx}",
                        kind="rectangle",
                        center=(r.center_x_um, 0.0, g.z_center_um),
                        size=(r.width_um, ly, g.ridge_height_um),
                        material=mat,
                        mesh_order=g.mesh_order,
                    )
                )

    # 4. Map Sources
    sources: List[Source] = []
    for s in ir.sources:
        if isinstance(s, PlaneWaveSource):
            sources.append(
                Source(
                    name=s.name,
                    kind="plane",
                    normal=s.normal,
                    direction=s.direction,
                    center=s.center,
                    size=s.size,
                    component=s.polarization,
                    wavelength=s.wavelength_um,
                    pulse_cycles=s.pulse_cycles,
                )
            )
        elif isinstance(s, GaussianBeamSource):
            sources.append(
                Source(
                    name=s.name,
                    kind="plane",
                    normal=s.normal,
                    direction=s.direction,
                    center=s.center,
                    size=(s.waist_radius_um * 4, s.waist_radius_um * 4, 0.0),
                    wavelength=s.wavelength_um,
                    pulse_cycles=s.pulse_cycles,
                )
            )
        elif isinstance(s, PointDipoleSource):
            sources.append(
                Source(
                    name=s.name,
                    kind="point",
                    center=s.position,
                    component=s.component,
                    wavelength=s.wavelength_um,
                    pulse_cycles=s.pulse_cycles,
                )
            )

    # 5. Map Monitors
    monitors: List[FieldMonitor] = []
    for m in ir.monitors:
        if isinstance(m, SpectralFluxMonitor):
            custom_freqs = [C0 / (wl * 1e-6) for wl in m.wavelengths_um]
            monitors.append(
                FieldMonitor(
                    name=m.name,
                    normal=m.normal,
                    center=m.center,
                    size=m.size,
                    record_flux=True,
                    spectrum=SpectrumSettings(sampling="custom", custom_frequencies_hz=custom_freqs, apodization="none"),
                )
            )
        elif isinstance(m, DiffractionOrderMonitor):
            custom_freqs = [C0 / (wl * 1e-6) for wl in m.wavelengths_um]
            lx, ly, _ = ir.domain.size_um
            monitors.append(
                FieldMonitor(
                    name=m.name,
                    normal=m.normal,
                    center=(0.0, 0.0, m.position_z_um),
                    size=(lx, ly, 0.0),
                    record_fields=True,
                    record_flux=True,
                    spectrum=SpectrumSettings(sampling="custom", custom_frequencies_hz=custom_freqs, apodization="none"),
                )
            )
        elif isinstance(m, VolumeFieldMonitor):
            monitors.append(
                FieldMonitor(
                    name=m.name,
                    center=m.center,
                    size=m.size,
                    record_fields=True,
                )
            )
        elif isinstance(m, ModeExpansionMonitor):
            monitors.append(
                FieldMonitor(
                    name=m.name,
                    normal=m.normal,
                    center=m.center,
                    size=m.size,
                    record_fields=True,
                    record_flux=True,
                )
            )

    project = Project(
        region=region,
        materials=materials,
        structures=structures,
        sources=sources,
        monitors=monitors,
    )
    return project


def compile_to_rcwa(ir: PhotonicExperimentIR, n_harmonics: int = 120) -> Dict[str, Any]:
    """Compiles PhotonicExperimentIR into a Fourier modal expansion configuration for RCWA solvers.

    Enforces strict Refusal Contracts against non-periodic structures, point dipoles, and time monitors.
    """
    # Refusal Check 1: Non-Periodic Transverse Boundaries
    if not ir.domain.boundaries.is_x_periodic:
        raise UnsupportedBackendError(
            "RCWA",
            f"Non-periodic transverse boundaries (x_min={ir.domain.boundaries.x_min.kind}, "
            f"x_max={ir.domain.boundaries.x_max.kind}). RCWA strictly requires periodic or Bloch boundary conditions along transverse axes."
        )

    # Refusal Check 2: Localized or Unsupported Sources
    for src in ir.sources:
        if isinstance(src, PointDipoleSource):
            raise UnsupportedBackendError(
                "RCWA",
                "Source kind 'point_dipole' is not supported. RCWA natively models infinite plane waves or Bloch modal excitation."
            )
        if isinstance(src, GaussianBeamSource):
            raise UnsupportedBackendError(
                "RCWA",
                "Source kind 'gaussian_beam' in open space is not supported by standard RCWA."
            )

    # Refusal Check 3: Time-Domain Monitors
    for mon in ir.monitors:
        if isinstance(mon, TimeSeriesMonitor):
            raise UnsupportedBackendError(
                "RCWA",
                "RCWA is an intrinsic frequency-domain solver and cannot record time-domain field traces."
            )

    # Refusal Check 4: Periodic Grating Geometry Exclusivity & Presence
    grating: Optional[PeriodicGrating1D] = None
    for g in ir.geometries:
        if not g.enabled:
            continue
        if isinstance(g, PeriodicGrating1D):
            if grating is None:
                grating = g
        else:
            raise UnsupportedBackendError(
                "RCWA",
                f"Non-periodic geometry '{g.id}' ({type(g).__name__}) is not supported by the RCWA backend. No periodic layer stack or PeriodicGrating1D geometry found for RCWA modal compilation.",
            )

    if grating is None:
        raise UnsupportedBackendError(
            "RCWA",
            "No periodic layer stack or PeriodicGrating1D geometry found for RCWA modal compilation.",
        )

    period_um = grating.period_um
    sub_mat = ir.get_material(grating.substrate_material)
    clad_mat = ir.get_material(grating.cladding_material)
    ridge_mat = ir.get_material(grating.ridge_material)

    source = ir.sources[0]
    wavelength_um = source.wavelength_um
    theta_deg = getattr(source, "incident_angle_deg", 0.0)

    ridges_data = []
    for r in grating.ridges:
        r_mat = ir.get_material(r.material) if r.material else ridge_mat
        ridges_data.append({
            "center_x_um": r.center_x_um,
            "width_um": r.width_um,
            "x_min": r.center_x_um - r.width_um / 2.0,
            "x_max": r.center_x_um + r.width_um / 2.0,
            "epsilon": r_mat.permittivity_real,
        })

    order_count = 2 * n_harmonics + 1

    rcwa_payload = {
        "solver": "rcwa",
        "provenance": "PhotonicExperimentIR.compile_to_rcwa",
        "period_um": period_um,
        "wavelength_um": wavelength_um,
        "theta_deg": theta_deg,
        "harmonics": n_harmonics,
        "order_count": order_count,
        "input_layer": {
            "name": sub_mat.name,
            "epsilon": sub_mat.permittivity_real,
        },
        "output_layer": {
            "name": clad_mat.name,
            "epsilon": clad_mat.permittivity_real,
        },
        "patterned_layers": [
            {
                "thickness_um": grating.ridge_height_um,
                "background_epsilon": clad_mat.permittivity_real,
                "ridges": ridges_data,
            }
        ],
    }
    return rcwa_payload


def compile_to_gds(
    ir: PhotonicExperimentIR,
    pdk_layer_map: Dict[str, Tuple[int, int]],
    cell_name: str = "TOP",
    dbu_um: float = 0.001,
) -> Dict[str, Any]:
    """Compiles 2D planar polygon projections onto calibrated GDSII layers with integer DBU.

    Enforces Refusal Contracts against tilted 3D objects and unmapped PDK materials.
    """
    polygons_by_layer: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
    z_manifest = []

    for geom in ir.geometries:
        if not geom.enabled:
            continue

        # Refusal Check 1: Tilted Out-of-Plane 3D Geometry
        if geom.tilted_out_of_plane:
            raise UnsupportedBackendError(
                "GDS",
                f"Tilted 3D geometry '{geom.id}' has non-zero out-of-plane rotation."
            )

        # Refusal Check 2: Unmapped Material in PDK Stack
        mat_name = getattr(geom, "ridge_material", getattr(geom, "material", ""))
        if mat_name not in pdk_layer_map:
            raise UnsupportedBackendError(
                "GDS",
                f"Material '{mat_name}' on geometry '{geom.id}' is not mapped in PDK layer map."
            )
        layer_pair = pdk_layer_map[mat_name]

        if isinstance(geom, BoxPrimitive):
            cx, cy, cz = geom.center
            sx, sy, sz = geom.size
            pts = [
                (cx - sx / 2.0, cy - sy / 2.0),
                (cx + sx / 2.0, cy - sy / 2.0),
                (cx + sx / 2.0, cy + sy / 2.0),
                (cx - sx / 2.0, cy + sy / 2.0),
            ]
            polygons_by_layer.setdefault(layer_pair, []).append(pts)
            z_manifest.append({
                "geometry_id": geom.id,
                "layer": layer_pair[0],
                "datatype": layer_pair[1],
                "z_min_um": cz - sz / 2.0,
                "z_max_um": cz + sz / 2.0,
                "material": mat_name,
            })

        elif isinstance(geom, PolygonExtrusion):
            polygons_by_layer.setdefault(layer_pair, []).append(geom.vertices_xy)
            z_manifest.append({
                "geometry_id": geom.id,
                "layer": layer_pair[0],
                "datatype": layer_pair[1],
                "z_min_um": geom.z_min,
                "z_max_um": geom.z_max,
                "material": mat_name,
            })

        elif isinstance(geom, PeriodicGrating1D):
            ly = ir.domain.size_um[1]
            for idx, r in enumerate(geom.ridges):
                ridge_mat = r.material or geom.ridge_material
                if ridge_mat not in pdk_layer_map:
                    raise UnsupportedBackendError(
                        "GDS",
                        f"Material '{ridge_mat}' has no layer mapping in pdk_layer_map",
                    )
                tooth_layer_pair = pdk_layer_map[ridge_mat]
                pts = [
                    (r.center_x_um - r.width_um / 2.0, -ly / 2.0),
                    (r.center_x_um + r.width_um / 2.0, -ly / 2.0),
                    (r.center_x_um + r.width_um / 2.0, ly / 2.0),
                    (r.center_x_um - r.width_um / 2.0, ly / 2.0),
                ]
                polygons_by_layer.setdefault(tooth_layer_pair, []).append(pts)
                z_manifest.append({
                    "geometry_id": f"{geom.id}_tooth_{idx}",
                    "layer": tooth_layer_pair[0],
                    "datatype": tooth_layer_pair[1],
                    "z_min_um": geom.z_center_um - geom.ridge_height_um / 2.0,
                    "z_max_um": geom.z_center_um + geom.ridge_height_um / 2.0,
                    "material": ridge_mat,
                })

        else:
            raise UnsupportedBackendError(
                "GDS",
                f"Unsupported geometry primitive '{geom.kind}' for planar GDS compilation."
            )

    # Quantize to integer database units (1 nm resolution)
    quantized_records: Dict[str, List[List[Tuple[int, int]]]] = {}
    for l_pair, poly_list in polygons_by_layer.items():
        q_polys = []
        for poly in poly_list:
            q_pts = [(int(round(x / dbu_um)), int(round(y / dbu_um))) for x, y in poly]
            q_polys.append(q_pts)
        quantized_records[f"{l_pair[0]}/{l_pair[1]}"] = q_polys

    return {
        "format": "gdsii",
        "cell": cell_name,
        "dbu_um": dbu_um,
        "polygon_count": sum(len(p) for p in polygons_by_layer.values()),
        "layers": quantized_records,
        "layer_stack_manifest": z_manifest,
    }
