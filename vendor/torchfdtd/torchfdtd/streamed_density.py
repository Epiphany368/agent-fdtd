"""Bounded periodic density-layer material production and exact linear VJP."""
from dataclasses import dataclass
from copy import deepcopy
import math

import torch

from .adjoint_batch import AdjointBatchOptions, RecomputedAdjointBatch, _PreparedCase
from .density_layer import _overlap
from .execution_tuning import _streamed_options
from .memory_profile import host_memory
from .solver import field_axes
from .streamed import _reservation
from .streamed_geometry import (StreamedGeometrySimulation, StreamedGeometryPlaneSimulation,
    _GeometryExecution, _GeometrySlabOperator, _SlabMaterial, _plane_construction_preflight)


@dataclass(frozen=True)
class StreamedDensityLayer:
    parameters: torch.Tensor
    region: object
    bottom_um: float
    top_um: float
    background_epsilon: float
    design_epsilon: float
    pixel_origin: str
    axes: tuple

    @property
    def shape(self):
        return self.region.shape+(3,)


@dataclass(frozen=True)
class _DensitySnapshot:
    shape: tuple
    steps: tuple
    period: tuple
    bottom_um: float
    top_um: float
    background_epsilon: float
    design_epsilon: float
    pixel_origin: str
    axes: tuple


def streamed_density_layer(density, region, *, bottom_um, top_um, background_epsilon,
                           design_epsilon, pixel_origin='cell_edges'):
    """Prepare a CPU density layer without constructing a 3D material map.

    Transfer conventions are identical to periodic_density_layer. Only density
    differentiates, while interface heights and permittivities remain fixed.
    """
    if (not isinstance(density, torch.Tensor) or density.device.type != 'cpu'
            or density.dtype != getattr(torch, region.precision) or density.ndim != 2
            or min(density.shape) < 1):
        raise ValueError('Density must be a nonempty CPU xy tensor matching region precision.')
    if not bool(torch.isfinite(density).all()) or bool(((density < 0)|(density > 1)).any()):
        raise ValueError('Density must be finite in [0,1].')
    if pixel_origin not in ('cell_edges', 'sample_centers'):
        raise ValueError('Unknown density pixel origin.')
    if region.dimension != '3d' or region.mesh_type != 'uniform' or region.material_sampling != 'yee':
        raise ValueError('Density transfer requires a uniform 3D Yee-material grid.')
    if any(face.kind not in ('periodic', 'bloch') for axis in (0, 1) for face in region.boundaries.pair(axis)):
        raise ValueError('Density transfer requires periodic/Bloch x and y boundaries.')
    values = (bottom_um, top_um, background_epsilon, design_epsilon)
    if any(isinstance(v, torch.Tensor) or not math.isfinite(v) for v in values):
        raise ValueError('Layer interfaces and material values must be fixed finite scalars.')
    if bottom_um >= top_um or min(background_epsilon, design_epsilon) < 1:
        raise ValueError('Layer thickness must be positive and permittivities at least one.')
    lower, upper = region.interior_bounds(2)
    if bottom_um < lower or top_um > upper:
        raise ValueError('Patterned layer must lie inside the non-PML region.')
    axes = tuple(torch.tensor(axis, dtype=density.dtype) for c in ('Ex', 'Ey', 'Ez') for axis in field_axes(region, c))
    return StreamedDensityLayer(density, region.model_copy(deep=True), *map(float, values), pixel_origin, axes)


class _DensityMaterial(_SlabMaterial):
    def weights(self, indices, component):
        geometry = self.geometry
        x, y, z = geometry.axes[3*component:3*component+3]
        # _overlap owns its coordinate conversion. NumPy views avoid a second
        # tensor-construction copy warning and contain only slab/axis entries.
        wx = _overlap(x[indices].numpy(), geometry.steps[0], self.parameters.shape[0],
                      geometry.period[0], self.parameters, geometry.pixel_origin)
        wy = _overlap(y.numpy(), geometry.steps[1], self.parameters.shape[1],
                      geometry.period[1], self.parameters, geometry.pixel_origin)
        dz = geometry.steps[2]
        fraction = (torch.minimum(z+dz/2, z.new_tensor(geometry.top_um))
                    -torch.maximum(z-dz/2, z.new_tensor(geometry.bottom_um))).clamp_min(0)/dz
        return wx, wy, fraction

    def index_select(self, axis, indices):
        if axis != 0:
            raise ValueError('Only x slabs are supported.')
        output = self.parameters.new_empty((len(indices), *self.shape[1:]))
        with torch.no_grad():
            for c in range(3):
                wx, wy, fraction = self.weights(indices, c)
                xy = wx@self.parameters@wy.T
                output[..., c] = (self.geometry.background_epsilon
                    +(self.geometry.design_epsilon-self.geometry.background_epsilon)*xy[:, :, None]*fraction)
        return output

    def accumulate_vjp(self, gradient, seed, indices):
        # Sum every replicated halo row. Repeated periodic indices remain
        # repeated rows in wx, so the transpose accumulates them exactly.
        contrast = self.geometry.design_epsilon-self.geometry.background_epsilon
        for c in range(3):
            wx, wy, fraction = self.weights(indices, c)
            xy_bar = (seed[..., c]*fraction).sum(-1)*contrast
            gradient.add_(wx.T@xy_bar@wy)


class _DensitySlabOperator(_GeometrySlabOperator):
    def _material_epsilon(self, material):
        self.material = _DensityMaterial(self.geometry, material)
        return self.material


class _DensityExecution(_GeometryExecution):
    def snapshot(self):
        g = self.geometry
        self.geometry = _DensitySnapshot(tuple(g.shape), tuple(g.region.axis_steps), tuple(g.region.actual_size),
            g.bottom_um, g.top_um, g.background_epsilon, g.design_epsilon, g.pixel_origin,
            tuple(axis.detach().clone() for axis in g.axes))

    def operator(self, host, options, store):
        return _DensitySlabOperator(host, options.slab_width, options.device, geometry=self.geometry,
            cuda_binding=options.cuda_binding, reuse_buffers=options.reuse_tile_buffers,
            tile_transfers=options.tile_transfers, tile_buffers=options.tile_buffers,
            local_checkpoints=options.local_checkpoints,
            state_factory=store.new_state if store is not None else None)

    def reservation(self, project, value, options, spectral):
        reservation = _reservation(project, value, options, spectral)
        item = value.element_size()
        nx, ny, nz = project.region.shape
        cells = reservation['max_extended_tile_cells']
        width = cells//(ny*nz)
        dx, dy = value.shape
        axes = sum(axis.numel()*axis.element_size() for axis in self.geometry.axes)
        # Weights are regenerated component-by-component. Include overlap
        # temporaries, matrix-product intermediates, tile material and real seed
        # reduction, plus parameter-sized gradients. No dense xy cache is kept.
        workspace = (16*(width*dx+ny*dy+width*dy+width*ny+dx*ny)
                     +12*cells+8*dx*dy+8*(width+ny+nz+dx+dy))*item
        extra = workspace+2*axes+4096+self.fixed_host_bytes
        total = reservation['host_reservation_bytes']+extra
        available = host_memory()['available_bytes']
        limit = min(options.host_budget_bytes, int(available*.8)) if available is not None else options.host_budget_bytes
        if total > limit:
            raise ValueError(f'Streamed density reservation exceeds the host budget: required={total}, admissible={limit}.')
        reservation.update(host_reservation_bytes=total, density_workspace_bytes=workspace,
            geometry_axes_bytes=axes, geometry_snapshot_axes_bytes=axes, geometry_snapshot_metadata_bytes=4096,
            geometry_parameter_bytes=value.numel()*item, dense_material_storage_bytes=0, dense_material_vjp_bytes=0)
        return reservation


class StreamedDensitySimulation(StreamedGeometrySimulation):
    _execution_type = _DensityExecution

    def _validate_geometry(self, geometry):
        if not isinstance(geometry, StreamedDensityLayer):
            raise ValueError('Use streamed_density_layer to prepare density materials.')
        if geometry.region.model_dump() != self.project.region.model_dump():
            raise ValueError('Density layer must use the simulation region.')
        density = geometry.parameters
        if (density.device.type != 'cpu' or density.dtype != getattr(torch, self.project.region.precision)
                or density.ndim != 2 or min(density.shape) < 1
                or not bool(torch.isfinite(density).all()) or bool(((density < 0)|(density > 1)).any())):
            raise ValueError('Density must match the CPU region precision and be finite in [0,1].')
        if any(s.enabled and s.injection == 'oneway' for s in self.project.sources):
            raise ValueError('Streamed density does not yet support one-way source material validation.')


class StreamedDensityPlaneSimulation(StreamedGeometryPlaneSimulation):
    _streamed_model_type = StreamedDensitySimulation


class _DensityPlaneCaseModel(StreamedDensityPlaneSimulation):
    def __init__(self, project, options, *, layer, quadrature_counts):
        super().__init__(project, options, quadrature_counts=quadrature_counts)
        self.layer = dict(layer)

    def prepare(self, density):
        return streamed_density_layer(density, self.project.region, **self.layer)

    def forward(self, density, frequency_hz, *, block_size=32):
        return super().forward(self.prepare(density), frequency_hz, block_size=block_size)


class _DensityPreparedCase(_PreparedCase):
    def __init__(self, spec, layer):
        if spec.policy.streamed is None or spec.parameter_indices != (0,):
            raise ValueError('Density cases require one density input and an explicit streamed policy.')
        self.spec, self.dispersive, self.planes = spec, False, True
        self.configuration = deepcopy((spec.policy, spec.fixed_background_epsilon,
            spec.parameter_indices, spec.block_size, spec.quadrature_counts))
        self.model = _DensityPlaneCaseModel(spec.project, _streamed_options(spec.policy),
            layer=layer, quadrature_counts=spec.quadrature_counts)
        self.project = self.model.project
        self.snapshot = self.project.model_dump()
        self.layer_snapshot = dict(layer)
        scalar = torch.empty((), dtype=getattr(torch, self.project.region.precision))
        self.spectral = self.model._spectral(scalar, spec.frequency_hz, spec.block_size)
        self.output_shapes = [(self.spectral.frequency.numel(), len(plan['weights']), 6)
                              for _, _, plan, _ in self.model.plans]
        self.layout_bytes = self.model.layout_reservation_bytes
        points = sum(shape[1] for shape in self.output_shapes)
        self.metadata_bytes = (4*points+len(self.output_shapes)*self.spectral.frequency.numel())*scalar.element_size()
        self.output_dtype = torch.complex128 if scalar.dtype == torch.float64 else torch.complex64
        self.output_bytes = sum(math.prod(shape)*2*scalar.element_size() for shape in self.output_shapes)

    def check_fixed(self):
        super().check_fixed()
        if self.model.layer != self.layer_snapshot:
            raise RuntimeError('Density layer settings changed. Rebuild the batch.')

    def reservation(self, parameters):
        self.check_fixed()
        density, = self.inputs(parameters)
        geometry = self.model.prepare(density)
        execution = _DensityExecution(geometry, self.layout_bytes)
        return execution.reservation(self.model.model.project, density,
                                     self.model.model.streaming_options, self.spectral)


class RecomputedDensityBatch(RecomputedAdjointBatch):
    """Reuse sequential case replay with small 2D density inputs and VJPs."""
    def __init__(self, cases, options=None, *, layer):
        torch.nn.Module.__init__(self)
        self.options = options or AdjointBatchOptions()
        cases = tuple(cases)
        construction = 0
        for case in cases:
            construction += _plane_construction_preflight(case.project,
                _streamed_options(case.policy), case.quadrature_counts)
        available = host_memory()['available_bytes']
        limit = min(self.options.host_budget_bytes, int(available*.8)) if available is not None else self.options.host_budget_bytes
        if construction > limit:
            raise ValueError('Density batch plane construction exceeds the shared host budget.')
        self._cases = tuple(_DensityPreparedCase(case, layer) for case in cases)
        if not self._cases:
            raise ValueError('Provide at least one density case.')
