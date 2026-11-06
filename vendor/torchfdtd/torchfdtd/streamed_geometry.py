"""Parameter-to-signal streaming without a dense epsilon or epsilon VJP.

Geometry is evaluated on CPU, one slab and one bounded differentiation chunk
at a time. Ordered overlap and Yee sampling match smooth_geometry_epsilon.
Periodic slab indices repeat the primary-domain material, without Bloch phase.
"""
from dataclasses import dataclass
import math

import torch

from .differentiable import DifferentiableResult, _System
from .differentiable_geometry import DifferentiableSolid, _as_tensor, _evaluate, _tensors
from .memory_profile import host_memory
from .solver import field_axes
from .spacetime import SlabBlockOperator
from .streamed import StreamedSimulation, _Streamed, _StreamedExecution, _reservation
from .adjoint_planes import DifferentiablePlaneSimulation


@dataclass(frozen=True)
class StreamedGeometry:
    """Small differentiable parameter packet and fixed geometry sampling metadata."""
    parameters: torch.Tensor
    region: object
    kinds: tuple
    width: float
    chunk_cells: int
    axes: tuple

    @property
    def shape(self):
        return self.region.shape + ((3,) if len(self.axes) == 9 else ())


@dataclass(frozen=True)
class _GeometrySnapshot:
    shape: tuple
    kinds: tuple
    width: float
    chunk_cells: int
    axes: tuple


def streamed_geometry(region, solids, *, background=1., width=None, chunk_cells=65536):
    """Prepare CPU analytic geometry without rasterizing the full domain.

    Parameters use the region precision (FP32 by default). The fixed mesh,
    transition width and explicit periodic copies follow smooth_geometry_epsilon.
    This nondispersive, first-order API currently excludes one-way sources.
    """
    solids = tuple(solids)
    if region.interface_method != 'staircase':
        raise ValueError('Regularized geometry requires staircase coefficients.')
    if isinstance(chunk_cells, bool) or not isinstance(chunk_cells, int) or chunk_cells < 1:
        raise ValueError('chunk_cells must be a positive integer.')
    width = region.reference_step if width is None else width
    if isinstance(width, torch.Tensor) or not math.isfinite(width) or width <= 0:
        raise ValueError('Transition width must be a fixed positive finite scalar.')
    if any(not isinstance(s, DifferentiableSolid) or s.kind not in ('box', 'ellipsoid', 'cylinder') for s in solids):
        raise ValueError('Use DifferentiableSolid constructors.')
    inputs = [background, *[v for s in solids for v in (s.center, s.extent, s.rotation, s.epsilon)]]
    like = torch.empty((), dtype=getattr(torch, region.precision), device='cpu')
    if any(t.dtype != like.dtype or t.device.type != 'cpu' for t in _tensors(inputs)):
        raise ValueError('Streamed geometry tensors must be on CPU with the region precision.')
    rows = []
    for solid in solids:
        values = [_as_tensor(v, like) for v in (solid.center, solid.extent, solid.rotation, solid.epsilon)]
        if any(v.shape != (3,) for v in values[:3]) or values[3].ndim != 0:
            raise ValueError('Center, extent and rotation need three values, and epsilon must be scalar.')
        rows.append(torch.cat((*values[:3], values[3].reshape(1))))
    parameters = torch.stack(rows) if rows else like.new_empty((0, 10))
    background = _as_tensor(background, like)
    if background.ndim != 0 or not bool(torch.isfinite(background)) or not bool(background >= 1):
        raise ValueError('Background epsilon must be a finite scalar at least one.')
    if not bool(torch.isfinite(parameters).all()) or bool((parameters[:, 3:6] <= 0).any()):
        raise ValueError('Solid parameters must be finite and extents positive.')
    if bool((parameters[:, 9] < 1).any()):
        raise ValueError('Solid epsilon must be at least one for the conservative CFL contract.')
    coordinates = ([axis for c in ('Ex', 'Ey', 'Ez') for axis in field_axes(region, c)]
                   if region.material_sampling == 'yee' else
                   [(nodes[:-1]+nodes[1:])/2 for nodes in region.mesh_nodes])
    axes = tuple(torch.tensor(axis, dtype=like.dtype) for axis in coordinates)
    packet = torch.cat((background.reshape(1), parameters.reshape(-1)))
    return StreamedGeometry(packet, region.model_copy(deep=True), tuple(s.kind for s in solids),
                            float(width), chunk_cells, axes)


class _SlabMaterial:
    """Only the row-access operations used by SlabBlockOperator are supported."""
    def __init__(self, geometry, parameters):
        self.geometry, self.parameters = geometry, parameters
        self.shape, self.dtype, self.device = geometry.shape, parameters.dtype, parameters.device
        self.ndim = len(self.shape)

    def __getitem__(self, selection):
        start, stop, step = selection.indices(self.shape[0])
        return self.index_select(0, torch.arange(start, stop, step))

    def chunks(self, indices):
        ny, nz = self.shape[1:3]
        count = len(indices)*ny*nz
        for begin in range(0, count, self.geometry.chunk_cells):
            end = min(count, begin+self.geometry.chunk_cells)
            flat = torch.arange(begin, end)
            yield begin, end, (indices[flat//(ny*nz)], (flat//nz)%ny, flat%nz)

    def points(self, index, component):
        axes = self.geometry.axes[3*component:3*component+3]
        return torch.stack([axis[i] for axis, i in zip(axes, index)], -1)

    def evaluate(self, points, packet):
        # Streamed solids keep the ten-value rows; polygons are not streamed.
        layout = tuple((kind, 10*i, 10) for i, kind in enumerate(self.geometry.kinds))
        return _evaluate(points, packet[1:], packet[0], layout, self.geometry.width)

    def index_select(self, axis, indices):
        if axis != 0:
            raise ValueError('Only x slabs are supported.')
        components = len(self.geometry.axes)//3
        output = self.parameters.new_empty((len(indices)*self.shape[1]*self.shape[2], components))
        with torch.no_grad():
            for begin, end, index in self.chunks(indices):
                for c in range(components):
                    output[begin:end, c] = self.evaluate(self.points(index, c), self.parameters)
        return output.reshape((len(indices), *self.shape[1:]))

    def accumulate_vjp(self, gradient, seed, indices):
        components = len(self.geometry.axes)//3
        weights = seed.reshape(-1, components)
        for begin, end, index in self.chunks(indices):
            for c in range(components):
                with torch.enable_grad():
                    packet = self.parameters.detach().requires_grad_(True)
                    value = self.evaluate(self.points(index, c), packet)
                    contribution, = torch.autograd.grad(value, packet, weights[begin:end, c])
                gradient.add_(contribution)


class _GeometrySlabOperator(SlabBlockOperator):
    def __init__(self, *args, geometry, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.material = None

    def _material_epsilon(self, material):
        self.material = _SlabMaterial(self.geometry, material)
        return self.material

    def _accumulate_gradient(self, gradient, contribution, indices):
        self.material.accumulate_vjp(gradient, contribution, indices)


class _GeometryExecution(_StreamedExecution):
    def __init__(self, geometry, fixed_host_bytes=0):
        self.geometry = geometry
        self.fixed_host_bytes = fixed_host_bytes

    def snapshot(self):
        # Freeze replay metadata independently of caller-owned tensor/model
        # objects. The autograd input itself remains version-checked by _Streamed.
        geometry = self.geometry
        self.geometry = _GeometrySnapshot(tuple(geometry.shape), tuple(geometry.kinds),
            float(geometry.width), int(geometry.chunk_cells),
            tuple(axis.detach().clone() for axis in geometry.axes))

    def host(self, project, value, spectral):
        # This view is metadata only. It must never be used to prepare updates
        # or validate source materials. Tile synthesis replaces every row read.
        placeholder = value.new_ones(()).expand(self.geometry.shape)
        return _System(project, placeholder, prepare_updates=False,
                       observation_monitors=None if spectral is None else spectral.observers)

    def operator(self, host, options, store):
        return _GeometrySlabOperator(host, options.slab_width, options.device,
            geometry=self.geometry, cuda_binding=options.cuda_binding,
            reuse_buffers=options.reuse_tile_buffers, tile_transfers=options.tile_transfers,
            tile_buffers=options.tile_buffers, local_checkpoints=options.local_checkpoints,
            state_factory=store.new_state if store is not None else None)

    def reservation(self, project, value, options, spectral):
        reservation = _reservation(project, value, options, spectral)
        item = value.element_size()
        # Differentiation replays one component, retaining intermediate points,
        # rotations, occupancy and their VJPs for each ordered solid. Include
        # transient packet gradients and one real material/seed tile explicitly.
        cells = min(self.geometry.chunk_cells, reservation['max_extended_tile_cells'])
        graph = cells*(256+512*len(self.geometry.kinds))*item
        axes = sum(a.numel()*a.element_size() for a in self.geometry.axes)
        material = 6*reservation['max_extended_tile_cells']*item
        snapshot_metadata = 4096+128*len(self.geometry.kinds)
        extra = graph+2*axes+snapshot_metadata+material+4*value.numel()*item+self.fixed_host_bytes
        total = reservation['host_reservation_bytes']+extra
        available = host_memory()['available_bytes']
        limit = min(options.host_budget_bytes, int(available*.8)) if available is not None else options.host_budget_bytes
        if total > limit:
            raise ValueError(f'Streamed geometry reservation exceeds the host budget: required={total}, admissible={limit}.')
        reservation.update(host_reservation_bytes=total, geometry_workspace_bytes=graph,
                           geometry_axes_bytes=axes, geometry_snapshot_axes_bytes=axes,
                           geometry_snapshot_metadata_bytes=snapshot_metadata,
                           geometry_material_tile_bytes=material,
                           geometry_parameter_bytes=value.numel()*item,
                           dense_material_storage_bytes=0, dense_material_vjp_bytes=0)
        return reservation


class StreamedGeometrySimulation(StreamedSimulation):
    """Stream analytic shape parameters directly through the FDTD transpose.

    No global epsilon or epsilon-gradient array is allocated. Field-bank storage
    still follows StreamedAdjointOptions, including optional disk-backed banks.
    One-way injection is excluded until bounded material validation is provided.
    """
    _execution_type = _GeometryExecution

    def __init__(self, project, options=None):
        super().__init__(project, options)
        from .boundaries import reject_pmc_faces
        reject_pmc_faces(self.project.region, 'StreamedGeometrySimulation')

    def spectrum(self, geometry, frequency_hz, *, window=None, block_size=32):
        from .adjoint_spectrum import SpectralObservation
        self._validate_geometry(geometry)
        spectral = SpectralObservation(geometry.parameters, self.project.region,
            [m.component for m in self.project.monitors if m.enabled], frequency_hz, window, block_size)
        return self._run(geometry, spectral)

    def _validate_geometry(self, geometry):
        if not isinstance(geometry, StreamedGeometry):
            raise ValueError('Use streamed_geometry to prepare parameterized materials.')
        if geometry.region.model_dump() != self.project.region.model_dump():
            raise ValueError('Geometry must use the simulation region.')
        packet = geometry.parameters
        if (packet.device.type != 'cpu' or packet.dtype != getattr(torch, self.project.region.precision)
                or packet.shape != (1+10*len(geometry.kinds),)):
            raise ValueError('Geometry parameter packet must match the CPU region precision and solid count.')
        rows = packet[1:].reshape(-1, 10)
        if (not bool(torch.isfinite(packet).all()) or bool(packet[0] < 1)
                or bool((rows[:, 9] < 1).any()) or bool((rows[:, 3:6] <= 0).any())):
            raise ValueError('Geometry requires finite parameters, positive extents and epsilon >= 1.')
        if any(s.enabled and s.injection == 'oneway' for s in self.project.sources):
            raise ValueError('Streamed geometry does not yet support one-way source material validation.')

    def _run(self, geometry, spectral):
        self._validate_geometry(geometry)
        options, region = self.streaming_options, self.project.region
        execution = self._execution_type(geometry, getattr(self, '_geometry_fixed_host_bytes', 0))
        reservation = execution.reservation(self.project, geometry.parameters, options, spectral)
        execution.snapshot()
        report = dict(experimental=True, spatial_streaming=True, parameterized_geometry=True,
            full_time_autograd=False, higher_order=False, precision=str(geometry.parameters.dtype),
            complex_fields=region.complex_fields, slab_width=options.slab_width,
            temporal_depth=options.temporal_depth, checkpoint_capacity=options.checkpoints,
            local_checkpoint_capacity=options.local_checkpoints, state_storage=options.state_storage,
            execution_device=options.device, tile_transfers=options.tile_transfers,
            tile_buffers=options.tile_buffers if options.tile_transfers == 'async' else 1,
            cuda_binding=options.cuda_binding, reuse_tile_buffers=options.reuse_tile_buffers,
            policy='manual', observation_storage='time_history' if spectral is None else 'online_spectrum',
            **reservation)
        if spectral is not None:
            report.update(spectral.reservation(min(options.temporal_depth, region.steps)))
        signals = _Streamed.apply(geometry.parameters, self.project, options, report, spectral, execution)
        if spectral is not None:
            return spectral.result(signals, report)
        return DifferentiableResult(signals, region.time_step,
            tuple(m.component for m in self.project.monitors if m.enabled), report)


class StreamedGeometryPlaneSimulation(DifferentiablePlaneSimulation):
    """Collocated spectral planes and normalized flux from bounded geometry.

    Uses the existing plane interpolation and real flux VJP. Fixed plane layout
    storage is charged in addition to material and solver workspace admission.
    """
    _streamed_model_type = StreamedGeometrySimulation

    def __init__(self, project, options=None, *, quadrature_counts=None):
        from .streamed import StreamedAdjointOptions
        options = options or StreamedAdjointOptions()
        if not isinstance(options, StreamedAdjointOptions):
            raise ValueError('Geometry planes require StreamedAdjointOptions.')
        construction = _plane_construction_preflight(project, options, quadrature_counts)
        super().__init__(project, options, quadrature_counts=quadrature_counts)
        self.layout_construction_reservation_bytes = construction
        self.model._geometry_fixed_host_bytes = self.layout_reservation_bytes

    def forward(self, geometry, frequency_hz, *, block_size=32):
        self.model._validate_geometry(geometry)
        return self._planes(geometry.parameters, frequency_hz, block_size,
                            lambda spectral: self.model._run(geometry, spectral))


def _plane_construction_preflight(project, options, quadrature_counts):
    """Bound plane layout construction before allocating point/map arrays.

    Only existing one-dimensional mesh axes are inspected. No Cartesian plane,
    interpolation map, lookup dictionary or observer list is materialized here.
    Observer deduplication can make the actual layout considerably smaller.
    """
    active = [m for m in project.monitors if m.enabled]
    if not active or any(m.kind != 'field' for m in active):
        raise ValueError('Streamed geometry planes require enabled field monitors only.')
    counts = {} if quadrature_counts is None else dict(quadrature_counts)
    if set(counts)-{m.id for m in active}:
        raise ValueError('Quadrature counts must refer to enabled monitor IDs.')
    available = host_memory()['available_bytes']
    limit = min(options.host_budget_bytes, int(available*.8)) if available is not None else options.host_budget_bytes
    reserved = 4096*len(active)
    if reserved > limit:
        raise ValueError('Plane layout construction exceeds the host budget before allocation.')
    region = project.region
    nodes = region.mesh_nodes
    # Include fixed-axis inspection/copy headroom. It is linear in axis lengths.
    reserved += 64*sum(len(axis) for axis in nodes)
    for raw in active:
        monitor = project.resolved_monitor(raw)
        normal = 'xyz'.index(monitor.normal)
        explicit = counts.get(monitor.id)
        if explicit is not None:
            if (region.dimension != '3d' or monitor.spatial_interpolation == 'nearest'
                    or len(explicit) != 2
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in explicit)):
                raise ValueError('Quadrature counts require two positive integers and a 3D interpolated plane.')
            points = math.prod(explicit)
        else:
            points = 1
            for axis, coordinates in enumerate(nodes):
                if axis == normal or (region.dimension == '2d' and axis == 2):
                    continue
                lo = monitor.center[axis]-monitor.size[axis]/2
                hi = monitor.center[axis]+monitor.size[axis]/2
                # Two endpoints plus strictly interior nodes form cell edges.
                interior = max(0, int(coordinates.searchsorted(hi-1e-12, side='left'))
                               -int(coordinates.searchsorted(lo+1e-12, side='right')))
                stride = monitor.downsample_xyz[axis] if monitor.downsample_xyz is not None else monitor.downsample
                points *= math.ceil((interior+1)/stride)
        neighbors = math.prod(1 if len(axis) == 2 else 2 for axis in nodes)
        samples = 6*neighbors*points
        # At most one observer per interpolation entry. Cover retained arrays,
        # Python lookup/list construction, temporary interpolation products and
        # copies while plans coexist. Complex weights take sixteen bytes.
        reserved += 4096+128*points+samples*(2048+96)
        if reserved > limit:
            raise ValueError(f'Plane layout construction exceeds the host budget before allocation: required={reserved}, admissible={limit}.')
    return reserved
