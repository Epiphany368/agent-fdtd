"""Overlapping-tile decomposition of a planar device with near-field stitching.

A large planar device (a millimetre-scale metasurface) is cut laterally into
tiles of a chosen core size plus an overlap margin on every lateral side. Each
tile is an independent Project on the global uniform mesh with CPML on every
face, the same normal-incidence sheet source restricted to the tile and one
output plane spanning the tile. The output planes are stitched from the tile
cores. The relative L2 difference between neighbouring tiles inside their
shared overlap is the error indicator. This is the approximate overlapping
domain method for problems that fit no memory tier; the exact streamed X-slab
mode in streamed.py stitches halos exactly for one problem that fits DRAM or
disk and is a different method.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import itertools
import json
import math
import time
import warnings

import numpy as np
import torch

from .adjoint_planes import COMPONENTS, DifferentiablePlaneResult, DifferentiablePlaneSimulation
from .geometry import object_bounds
from .models import FieldMonitor, Project, Source, Structure
from .radiation import project_farfield



def suggest_overlap(distance_um, max_angle_deg, wavelength_um, *, absorber_um=None):
    """Overlap margin for light diffracted at up to ``max_angle_deg`` over ``distance_um``.

    The geometric spread ``distance * tan(angle)`` is the lateral excursion of
    the steepest ray between the device and the output plane. The tile CPML
    sits outside the extended tile, but the fields within about one absorber
    thickness of it carry its residual reflection and the edge wave of the
    truncated sheet source, so that thickness is added. Without an explicit
    absorber thickness half a wavelength is used, the thickness of a ten-cell
    CPML at twenty cells per wavelength. Returns the margin and its two parts.
    """
    for name, value in (('distance_um', distance_um), ('max_angle_deg', max_angle_deg), ('wavelength_um', wavelength_um)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number.')
    if distance_um < 0:
        raise ValueError('distance_um must be nonnegative.')
    if not 0 <= max_angle_deg < 90:
        raise ValueError('max_angle_deg must lie in [0, 90): a grazing ray has no finite lateral excursion.')
    if wavelength_um <= 0:
        raise ValueError('wavelength_um must be positive.')
    absorber = wavelength_um / 2 if absorber_um is None else absorber_um
    if isinstance(absorber, bool) or not isinstance(absorber, (int, float)) or not math.isfinite(absorber) or absorber < 0:
        raise ValueError('absorber_um must be a finite nonnegative thickness.')
    spread = distance_um * math.tan(math.radians(max_angle_deg))
    return dict(overlap_um=spread + absorber, spread_um=spread, absorber_um=absorber,
                distance_um=distance_um, max_angle_deg=max_angle_deg, wavelength_um=wavelength_um)


@dataclass
class TileSpec:
    """One tile. Cell ranges are half-open global lateral cell indices."""
    id: str
    index: tuple            # tile position along each tiled axis
    core: tuple             # owned cells, cropped into the stitched plane
    extended: tuple         # core plus overlap, clipped at the device edge: the non-PML tile region
    full: tuple             # extended plus the tile's own CPML: the tile grid
    center_um: tuple        # global coordinates of the tile region centre
    project: Project

    @property
    def core_in_extended(self):
        return tuple((c0 - e0, c1 - e0) for (c0, c1), (e0, _) in zip(self.core, self.extended))


@dataclass
class TilePlan:
    project: Project        # validated copy of the global project
    normal: str             # propagation axis
    lateral: tuple          # tiled axis indices
    interior: tuple         # global non-PML cell ranges along the tiled axes: the tiled extent
    counts: tuple           # tiles along each tiled axis
    tile_cells: int
    overlap_cells: int
    tile_um: float
    overlap_um: float
    spacing_um: float
    offset_um: float        # output plane position along the normal
    direction: int          # +1 when light leaves the device toward +normal
    monitor_id: str
    tiles: tuple
    suggestion: dict | None
    report: dict = field(default_factory=dict)

    @property
    def axis(self):
        return 'xyz'.index(self.normal)

    @property
    def lateral_axes(self):
        return tuple('xyz'[a] for a in self.lateral)

    @property
    def dimension(self):
        return self.project.region.dimension

    def tile(self, identifier):
        matches = [t for t in self.tiles if t.id == identifier]
        if len(matches) != 1:
            raise KeyError(identifier)
        return matches[0]


def _lateral_axes(region, normal):
    if normal not in ('x', 'y', 'z'):
        raise ValueError('normal must be x, y or z.')
    axis = 'xyz'.index(normal)
    if region.dimension == '2d' and axis == 2:
        raise ValueError('A 2D device propagates along x or y; z is the invariant axis.')
    return axis, tuple(a for a in range(3) if a != axis and (region.dimension == '3d' or a != 2))


def plan_tiles(project, tile_um, overlap_um, *, normal=None, max_angle_deg=None):
    """Partition the lateral non-PML extent into overlapping tiles on the global mesh.

    Cores are ``round(tile_um / mesh)`` cells and the margin is
    ``ceil(overlap_um / mesh)`` cells, both clipped at the device edge. Every
    tile Project keeps the global mesh, precision, duration, materials and CPML
    settings, so each tile node coincides with a global node. With
    ``max_angle_deg`` the plan computes ``suggest_overlap`` from the output
    plane distance and the source wavelength and warns when the margin is
    below it.
    """
    p = Project.model_validate(project.model_dump())
    r = p.region
    if r.mesh_type != 'uniform' or r.mesh_steps is not None:
        raise ValueError('Tiling requires a uniform mesh with one spacing on every axis.')
    if r.complex_fields:
        raise ValueError('Tiling requires real fields; a Bloch phase describes a periodic cell, not a finite device.')
    normal = normal or ('z' if r.dimension == '3d' else 'y')
    axis, lateral = _lateral_axes(r, normal)
    for a in (axis,) + lateral:
        if any(face.kind != 'pml' for face in r.boundaries.pair(a)):
            raise ValueError('Tiling requires CPML on the lateral and normal faces of the global project; each tile adds its own CPML at the cuts.')
    for name, value in (('tile_um', tile_um), ('overlap_um', overlap_um)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive finite length in micrometres.')
    h = r.mesh
    nodes = r.mesh_nodes
    interior = tuple((r.pml_layers(a, 0), r.shape[a] - r.pml_layers(a, 1)) for a in lateral)
    sources = [p.resolved_source(s) for s in p.sources if s.enabled]
    if not sources:
        raise ValueError('Tiling requires an enabled sheet source.')
    for s in sources:
        if s.kind != 'plane' or s.injection != 'soft' or s.normal != normal:
            raise ValueError(f'{s.name}: tiling requires soft plane sources normal to {normal}; point, one-way and TFSF sources are not partitioned.')
        for a, (lo, hi) in zip(lateral, interior):
            if s.center[a] - s.size[a] / 2 > nodes[a][lo] + 1e-9 or s.center[a] + s.size[a] / 2 < nodes[a][hi] - 1e-9:
                raise ValueError(f'{s.name}: the sheet must span the whole non-PML lateral extent so that every tile sees the same incidence.')
    monitors = [m for m in p.monitors if m.enabled]
    if len(monitors) != 1 or monitors[0].kind != 'field' or monitors[0].normal != normal:
        raise ValueError(f'Tiling requires exactly one enabled field monitor normal to {normal}: the output plane.')
    monitor = p.resolved_monitor(monitors[0])
    offset = monitor.center[axis]
    sides = {int(math.copysign(1, offset - s.center[axis])) for s in sources}
    if len(sides) != 1 or any(s.center[axis] == offset for s in sources):
        raise ValueError('Every sheet source must lie on the same side of the output plane.')
    direction = sides.pop()
    tile_cells = max(1, round(tile_um / h))
    overlap_cells = max(1, math.ceil(overlap_um / h - 1e-9))
    counts = tuple(math.ceil((hi - lo) / tile_cells) for lo, hi in interior)
    layers = tuple((r.pml_layers(a, 0), r.pml_layers(a, 1)) for a in lateral)
    region_size = list(r.size)
    tiles = []
    for index in itertools.product(*(range(c) for c in counts)):
        core, extended, full, center, size = [], [], [], [0., 0., 0.], list(region_size)
        for i, (a, (lo, hi), k) in enumerate(zip(lateral, interior, index)):
            c0 = lo + k * tile_cells
            c1 = min(hi, c0 + tile_cells)
            e0, e1 = max(lo, c0 - overlap_cells), min(hi, c1 + overlap_cells)
            f0, f1 = e0 - layers[i][0], e1 + layers[i][1]
            core.append((c0, c1)); extended.append((e0, e1)); full.append((f0, f1))
            center[a] = float((nodes[a][f0] + nodes[a][f1]) / 2)
            size[a] = (f1 - f0) * h
        identifier = 'tile-' + '-'.join(str(k) for k in index)
        tiles.append(TileSpec(identifier, tuple(index), tuple(core), tuple(extended), tuple(full), tuple(center),
                              _tile_project(p, identifier, lateral, extended, full, center, size, sources, monitor)))
    suggestion = None
    if max_angle_deg is not None:
        # The device point farthest from the output plane against the propagation direction.
        bounds = [object_bounds(o) for o in p.structures if o.enabled]
        if bounds:
            far = min(c[axis] - s[axis] / 2 for c, s in bounds) if direction > 0 else max(c[axis] + s[axis] / 2 for c, s in bounds)
        else:
            far = min(s.center[axis] for s in sources) if direction > 0 else max(s.center[axis] for s in sources)
        distance = max(0., direction * (offset - far))
        suggestion = suggest_overlap(distance, max_angle_deg, sources[0].wavelength, absorber_um=max(l for pair in layers for l in pair) * h)
        if overlap_cells * h < suggestion['overlap_um']:
            warnings.warn(f'Tile overlap {overlap_cells * h:.4g} um is below the suggested {suggestion["overlap_um"]:.4g} um '
                          f'({suggestion["spread_um"]:.4g} um spread over {distance:.4g} um at {max_angle_deg:g} degrees plus '
                          f'{suggestion["absorber_um"]:.4g} um absorber).', stacklevel=2)
    largest = max(math.prod(t.project.region.shape) for t in tiles)
    report = dict(tiles=len(tiles), counts=list(counts), tile_cells=tile_cells, overlap_cells=overlap_cells,
                  tile_um=tile_cells * h, overlap_um=overlap_cells * h, requested_tile_um=tile_um, requested_overlap_um=overlap_um,
                  global_cells=math.prod(r.shape), largest_tile_cells=largest,
                  total_tile_cells=sum(math.prod(t.project.region.shape) for t in tiles),
                  interior_cells=[hi - lo for lo, hi in interior], suggestion=suggestion)
    return TilePlan(p, normal, lateral, interior, counts, tile_cells, overlap_cells, tile_um, overlap_um, h,
                    offset, direction, monitor.id, tuple(tiles), suggestion, report)


def _tile_project(p, identifier, lateral, extended, full, center, size, sources, monitor):
    r = p.region
    h = r.mesh
    nodes = r.mesh_nodes
    low = [c - s / 2 for c, s in zip(center, size)]
    high = [c + s / 2 for c, s in zip(center, size)]
    structures = []
    for s in p.structures:
        if not s.enabled:
            continue
        c, extent = object_bounds(s)
        # Keep every solid that reaches the tile grid whole; the Yee rasteriser
        # samples the analytic solid only at the tile's own nodes, which clips it
        # exactly at the tile boundary including the CPML. Nothing is truncated at
        # the extended bounds, so no artificial interface precedes the absorber.
        if any(c[a] + extent[a] / 2 < low[a] - h or c[a] - extent[a] / 2 > high[a] + h for a in lateral):
            continue
        data = s.model_dump()
        data['center'] = [v - o for v, o in zip(s.center, center)]
        structures.append(Structure.model_validate(data))
    def restricted(item, cls, ranges, **extra):
        data = item.model_dump()
        position, span = list(item.center), list(item.size)
        for a, (e0, e1) in zip(lateral, ranges):
            position[a] = float((nodes[a][e0] + nodes[a][e1]) / 2 - center[a])
            span[a] = (e1 - e0) * h
        data.update(center=position, size=span, **extra)
        return cls.model_validate(data)
    region = r.model_dump()
    # Tiles are resident; the explicit mode keeps the resident cell guard at plan time.
    region.update(size=tuple(size), memory_mode='resident', execution_mode='resident', mesh_refinements=[], slice_position=0.)
    monitor_data = dict(record_fields=COMPONENTS, downsample=1, downsample_xyz=None, use_global_monitor=False,
                        spatial_interpolation='specified')
    return Project.model_validate(dict(
        name=f'{p.name} {identifier}', region=region, limits=p.limits.model_dump(), materials=[m.model_dump() for m in p.materials],
        structures=[s.model_dump() for s in structures],
        # A sheet flagged extend_through_pml keeps spanning the tile's own CPML.
        sources=[restricted(s, Source, full if s.extend_through_pml else extended).model_dump() for s in sources],
        global_source=None if p.global_source is None else p.global_source.model_dump(),
        monitors=[restricted(monitor, FieldMonitor, extended, **monitor_data).model_dump()],
        global_monitor=p.global_monitor.model_dump()))


@dataclass
class StitchedPlane:
    """Six-component spectra on the global lateral cell-centre grid of the tiled extent."""
    fields: torch.Tensor    # frequency, u, v, component (Ex,Ey,Ez,Hx,Hy,Hz)
    frequency_hz: torch.Tensor
    u_um: torch.Tensor      # global cell centres along the first transverse axis
    v_um: torch.Tensor      # second transverse axis; one zero sample for a 2D device
    normal: str
    offset_um: float
    direction: int
    spacing_um: tuple
    dimension: str
    region_span_um: tuple   # global region bounds along the normal
    run_signature: str
    report: dict
    blend: str = 'hard'
    tile_planes: list | None = None   # the per-tile planes, so another blend needs no rerun
    components: tuple = COMPONENTS
    field_units: str = 'reduced field * s'
    fourier_convention: str = 'exp(+2 pi i f t)'

    @property
    def axes(self):
        """Transverse axis names in increasing axis order; the second is z for a 2D device."""
        return tuple(c for c in 'xyz' if c != self.normal)

    def intensity(self):
        """|E|^2 per grid point, (frequency, u, v)."""
        return self.fields[..., :3].abs().square().sum(-1)

    def poynting(self):
        """Time-averaged Poynting component along the positive normal, (frequency, u, v)."""
        a = 'xyz'.index(self.normal)
        b, c = (a + 1) % 3, (a + 2) % 3
        return .5 * (self.fields[..., b] * self.fields[..., 3 + c].conj() - self.fields[..., c] * self.fields[..., 3 + b].conj()).real

    def as_plane(self):
        """The same data as a DifferentiablePlaneResult with full midpoint quadrature."""
        count, nu, nv, _ = self.fields.shape
        dtype, device = self.fields.real.dtype, self.fields.device
        ug, vg = torch.meshgrid(self.u_um.to(device=device, dtype=dtype), self.v_um.to(device=device, dtype=dtype), indexing='ij')
        a = 'xyz'.index(self.normal)
        b, c = [i for i in range(3) if i != a]
        points = torch.empty((nu * nv, 3), dtype=dtype, device=device)
        points[:, a] = self.offset_um
        points[:, b] = ug.reshape(-1)
        points[:, c] = vg.reshape(-1)
        hu, hv = self.spacing_um
        area = hu * 1e-6 * (hv * 1e-6 if self.dimension == '3d' else 1.)
        shape = [1, 1, 1]
        shape[b], shape[c] = nu, nv
        return DifferentiablePlaneResult(
            self.fields.reshape(count, nu * nv, 6), self.frequency_hz.to(device=device, dtype=dtype), points,
            torch.full((nu * nv,), area, dtype=dtype, device=device), tuple(shape), self.normal, self.run_signature,
            dict(self.report), flux_units='reduced E*H * s^2 * ' + ('m^2' if self.dimension == '3d' else 'm per invariant length'))

    def radiation_surface(self, *, depth_um=None):
        """One named open face and its box for the radiation transforms.

        The face is the plane itself, named by its outward normal, and the box
        extends from it back through the device (to the global region boundary
        unless ``depth_um`` is given). Feed the pair to ``project_farfield``,
        ``project_nearzone`` or ``farfield_at_points`` with ``open_surface=True``.
        """
        if self.dimension != '3d':
            raise ValueError('The radiation adapter needs a 3D stitched plane.')
        a = 'xyz'.index(self.normal)
        lo, hi = self.region_span_um
        if depth_um is None:
            depth_um = self.offset_um - lo if self.direction > 0 else hi - self.offset_um
        if isinstance(depth_um, bool) or not isinstance(depth_um, (int, float)) or not math.isfinite(depth_um) or depth_um <= 0:
            raise ValueError('depth_um must be a positive finite depth behind the plane.')
        hu, hv = self.spacing_um
        bounds = [None] * 3
        b, c = [i for i in range(3) if i != a]
        bounds[b] = (float(self.u_um[0]) - hu / 2, float(self.u_um[-1]) + hu / 2)
        bounds[c] = (float(self.v_um[0]) - hv / 2, float(self.v_um[-1]) + hv / 2)
        bounds[a] = (self.offset_um - depth_um, self.offset_um) if self.direction > 0 else (self.offset_um, self.offset_um + depth_um)
        name = self.normal + ('_max' if self.direction > 0 else '_min')
        return {name: self.as_plane()}, tuple(bounds)


def farfield_from_stitched(stitched, directions, *, depth_um=None, **kwargs):
    """Open-surface far field of the stitched plane through ``project_farfield``."""
    faces, bounds = stitched.radiation_surface(depth_um=depth_um)
    return project_farfield(faces, directions, bounds_um=bounds, open_surface=True, **kwargs)


def _tile_tensor(plane, tile, plan):
    """Return (F, Nu_ext, Nv_ext, 6) fields and the frequencies of one tile plane."""
    if isinstance(plane, DifferentiablePlaneResult):
        fields, shape, components, frequency = plane.fields, plane.shape, plane.components, plane.frequency_hz
    elif isinstance(plane, dict):
        fields = torch.as_tensor(np.asarray(plane['fields']))
        shape, components = tuple(plane['shape']), tuple(plane['components'])
        frequency = torch.as_tensor(np.asarray(plane['frequency_hz'], dtype=np.float64))
    elif isinstance(plane, torch.Tensor):
        fields, shape, components, frequency = plane, None, COMPONENTS, None
    else:
        raise TypeError('Each tile plane must be a native field-monitor dict, a DifferentiablePlaneResult or a tensor.')
    if tuple(components) != COMPONENTS:
        if set(components) != set(COMPONENTS):
            raise ValueError('Tile planes must carry all six components.')
        fields = fields[..., [components.index(c) for c in COMPONENTS]]
    if shape is not None:
        if fields.ndim != 3 or fields.shape[1] != math.prod(shape) or fields.shape[2] != 6:
            raise ValueError(f'{tile.id}: plane fields {tuple(fields.shape)} do not match the quadrature shape {tuple(shape)}.')
        fields = fields.reshape(fields.shape[0], *shape, 6).squeeze(plan.axis + 1)
    expected = tuple(e1 - e0 for e0, e1 in tile.extended) + ((1,) if len(tile.extended) == 1 else ())
    if fields.ndim != 4 or tuple(fields.shape[1:3]) != expected or fields.shape[3] != 6 or not fields.is_complex():
        raise ValueError(f'{tile.id}: expected complex fields shaped (F, {expected[0]}, {expected[1]}, 6), got {tuple(fields.shape)}.')
    return fields, frequency


def _weights(plan, tile, blend):
    axes = []
    for i, ((c0, c1), (e0, e1)) in enumerate(zip(tile.core, tile.extended)):
        cells = np.arange(e0, e1) + .5
        if blend == 'hard':
            w = ((cells > c0) & (cells < c1)).astype(np.float64)
        else:
            n = plan.overlap_cells
            w = np.ones(len(cells))
            if tile.index[i] > 0:
                w = w * np.clip((cells - (c0 - n)) / (2 * n), 0, 1)
            if tile.index[i] < plan.counts[i] - 1:
                w = w * np.clip(((c1 + n) - cells) / (2 * n), 0, 1)
        axes.append(w)
    if len(axes) == 1:
        axes.append(np.ones(1))
    return np.outer(axes[0], axes[1])


def _signature(plan):
    region = plan.project.region.model_dump(mode='json')
    for key in ('memory_mode', 'execution_mode', 'tiling', 'backend', 'cuda_kernel', 'cuda_monitor_kernel'):
        region.pop(key, None)
    payload = dict(region=region, sources=[plan.project.resolved_source(s).model_dump(mode='json') for s in plan.project.sources],
                   normal=plan.normal, offset=plan.offset_um, tiles=[(t.id, t.core, t.extended) for t in plan.tiles])
    # Tile absorber faces follow the structures a tile cuts, so a device and its air reference can differ there.
    from .boundaries import absorber_faces
    faces = [[list(face) for face in absorber_faces(t.project)] for t in plan.tiles]
    if any(faces):
        payload['absorber_faces'] = faces
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stitch_planes(plan, planes, *, blend='hard'):
    """Assemble tile output planes into one StitchedPlane on the global grid.

    ``planes`` is aligned with ``plan.tiles``: native ``result.field_monitor``
    dicts, ``DifferentiablePlaneResult`` objects or complex tensors shaped
    ``(F, Nu_ext, Nv_ext, 6)`` on each tile's extended grid. ``blend='hard'``
    crops each tile to its core; ``'linear'`` ramps neighbouring tiles across
    their shared overlap band, which is ``2 * overlap_cells`` wide. The report
    holds the relative L2 mismatch of every neighbouring pair in that band.
    Torch operations only, so a graph on the tile fields survives stitching.
    """
    if blend not in ('hard', 'linear'):
        raise ValueError("blend must be 'hard' or 'linear'.")
    planes = list(planes)
    if len(planes) != len(plan.tiles):
        raise ValueError(f'Expected {len(plan.tiles)} tile planes, got {len(planes)}.')
    tensors, frequency = [], None
    for tile, plane in zip(plan.tiles, planes):
        fields, f = _tile_tensor(plane, tile, plan)
        if f is not None:
            f = f.to(device=fields.device, dtype=torch.float64)
            if frequency is None:
                frequency = f
            elif f.shape != frequency.shape or not torch.allclose(f, frequency, rtol=1e-12, atol=0):
                raise ValueError('Every tile plane must carry the same frequencies.')
        tensors.append(fields)
    if frequency is None:
        raise ValueError('Tensor planes carry no frequencies; pass DifferentiablePlaneResult or native dicts.')
    first = tensors[0]
    if any(t.dtype != first.dtype or t.device != first.device or t.shape[0] != first.shape[0] for t in tensors):
        raise ValueError('Tile planes must share dtype, device and frequency count.')
    r = plan.project.region
    nodes = r.mesh_nodes
    origin = tuple(lo for lo, _ in plan.interior)
    extent = tuple(hi - lo for lo, hi in plan.interior) + ((1,) if len(plan.lateral) == 1 else ())
    rdtype = first.real.dtype
    out = torch.zeros((first.shape[0], *extent, 6), dtype=first.dtype, device=first.device)
    total = torch.zeros(extent, dtype=rdtype, device=first.device)
    for tile, fields in zip(plan.tiles, tensors):
        w = torch.as_tensor(_weights(plan, tile, blend), dtype=rdtype, device=first.device)
        window = tuple(slice(e0 - o, e1 - o) for (e0, e1), o in zip(tile.extended, origin))
        if len(window) == 1:
            window += (slice(0, 1),)
        out[(slice(None),) + window] += fields * w[None, :, :, None]
        total[window] += w
    if bool((total <= 0).any()):
        raise RuntimeError('A global cell received no tile weight; the plan does not cover the tiled extent.')
    fields = out / total[None, :, :, None]
    pairs = []
    for i in range(len(plan.lateral)):
        for tile, fields_a in zip(plan.tiles, tensors):
            if tile.index[i] == plan.counts[i] - 1:
                continue
            index = list(tile.index)
            index[i] += 1
            position = [t.index for t in plan.tiles].index(tuple(index))
            other, fields_b = plan.tiles[position], tensors[position]
            band = [(max(ea[0], eb[0]), min(ea[1], eb[1])) for ea, eb in zip(tile.extended, other.extended)]
            local_a = tuple(slice(b0 - e0, b1 - e0) for (b0, b1), (e0, _) in zip(band, tile.extended))
            local_b = tuple(slice(b0 - e0, b1 - e0) for (b0, b1), (e0, _) in zip(band, other.extended))
            if len(local_a) == 1:
                local_a += (slice(None),); local_b += (slice(None),)
            def relative(window):
                a_band = fields_a[(slice(None),) + local_a][window].detach()
                b_band = fields_b[(slice(None),) + local_b][window].detach()
                scale = .5 * float(a_band.norm() + b_band.norm())
                return float((a_band - b_band).norm()) / scale if scale > 0 else float('nan')
            # The full band reaches the cells next to each neighbour's absorber;
            # the central half, within overlap/2 of the cut, is what the hard
            # crop actually joins.
            half = max(1, plan.overlap_cells // 2)
            cut = tile.core[i][1]
            center = [slice(None)] * 4
            center[i + 1] = slice(max(band[i][0], cut - half) - band[i][0], min(band[i][1], cut + half) - band[i][0])
            pairs.append(dict(tiles=[tile.id, other.id], axis='xyz'[plan.lateral[i]], band_cells=[list(b) for b in band],
                              band_um=(band[i][1] - band[i][0]) * plan.spacing_um, mismatch=relative((slice(None),) * 4),
                              mismatch_center=relative(tuple(center))))
    summary = {}
    for key in ('mismatch', 'mismatch_center'):
        values = [q[key] for q in pairs if math.isfinite(q[key])]
        summary['max_' + key] = max(values) if values else None
        summary['mean_' + key] = float(np.mean(values)) if values else None
    report = dict(method='overlapping tiles, stitched output plane', blend=blend, tiles=len(plan.tiles),
                  tile_cells=plan.tile_cells, overlap_cells=plan.overlap_cells, overlap_um=plan.overlap_cells * plan.spacing_um,
                  pairs=pairs, **summary,
                  approximation='overlapping domain decomposition: each tile misses the scatterers beyond its extended region and carries its own absorber at the cuts')
    u = torch.as_tensor((nodes[plan.lateral[0]][origin[0]:origin[0] + extent[0]] + nodes[plan.lateral[0]][origin[0] + 1:origin[0] + extent[0] + 1]) / 2,
                        dtype=torch.float64, device=first.device)
    if len(plan.lateral) == 2:
        v = torch.as_tensor((nodes[plan.lateral[1]][origin[1]:origin[1] + extent[1]] + nodes[plan.lateral[1]][origin[1] + 1:origin[1] + extent[1] + 1]) / 2,
                            dtype=torch.float64, device=first.device)
        spacing = (plan.spacing_um, plan.spacing_um)
    else:
        v = torch.zeros(1, dtype=torch.float64, device=first.device)
        spacing = (plan.spacing_um, float(r.size[2]))
    span = (-r.actual_size[plan.axis] / 2, r.actual_size[plan.axis] / 2)
    return StitchedPlane(fields, frequency, u, v, plan.normal, plan.offset_um, plan.direction, spacing, r.dimension,
                         span, _signature(plan), report, blend, planes)


def run_tiled(project, plan, *, backend, options=None, executor='sequential', blend='hard'):
    """Run the plan's tiles natively and stitch their output planes.

    ``executor='sequential'`` calls ``Simulation.run`` per tile (``options`` are
    its keyword arguments), ``'batch'`` uses ``run_batch`` across processes
    (``options`` are ``BatchRunner`` arguments) and ``'tensor'`` uses
    ``run_grouped_batch`` so that tiles of equal shape share fused CUDA cohorts
    (``options`` are its arguments, ``cohort_size`` among them). The stitched
    plane report records the executor, its plan and the per-tile summaries.
    """
    if project.model_dump() != plan.project.model_dump():
        raise ValueError('The plan was made for a different project. Call plan_tiles again.')
    if backend not in ('auto', 'cpu', 'cuda'):
        raise ValueError('backend must be auto, cpu or cuda.')
    if executor not in ('sequential', 'batch', 'tensor'):
        raise ValueError("executor must be 'sequential', 'batch' or 'tensor'.")
    options = dict(options or {})
    projects = [Project.model_validate({**t.project.model_dump(), 'region': {**t.project.region.model_dump(), 'backend': backend}})
                for t in plan.tiles]
    # A sheet extended across a tile's absorber face is refused before the first tile runs (boundaries.absorber_faces).
    from .boundaries import absorber_faces
    for q in projects:
        absorber_faces(q)
    started = time.perf_counter()
    if executor == 'sequential':
        from .solver import Simulation
        results = [Simulation(q).run(**options) for q in projects]
        execution = dict(method='sequential Simulation.run', options=options)
    else:
        from .batch import BatchCase
        cases = [BatchCase(t.id, q) for t, q in zip(plan.tiles, projects)]
        if executor == 'batch':
            from .batch import run_batch
            report = run_batch(cases, backend=backend, keep_results=True, **options)
        else:
            from .grouped_batch import run_grouped_batch
            report = run_grouped_batch(cases, keep_results=True, **options)
        report.raise_for_errors()
        results = [item.load() for item in report.items]
        execution = dict(report.plan, seconds=report.seconds)
    planes = [result.field_monitor(plan.monitor_id) for result in results]
    stitched = stitch_planes(plan, planes, blend=blend)
    keys = ('seconds', 'setup_seconds', 'steps', 'shape', 'cells', 'backend', 'cuda_kernel', 'mcells_per_second', 'timing_scope')
    stitched.report.update(executor=executor, backend=backend, seconds=time.perf_counter() - started, execution=execution,
                           tiles_run=[dict(id=t.id, **{k: result.summary[k] for k in keys if k in result.summary})
                                      for t, result in zip(plan.tiles, results)])
    return stitched


def propagate_plane(stitched, distance_um, index, *, pad=2):
    """Angular-spectrum propagation of every component to a parallel plane.

    One inverse FFT of ``angular_spectrum.propagate_volume`` at one distance:
    each Cartesian component of E and H satisfies the scalar Helmholtz
    equation in the homogeneous medium of index ``index``, so the plane is
    zero padded ``pad`` times its size, transformed, multiplied by
    ``exp(i k_n d)`` and cropped back. ``distance_um`` is measured away from
    the device along the plane's propagation direction and must be
    nonnegative; the plane holds only outgoing waves. Rays leaving the padded
    window wrap around, so choose ``pad`` with
    ``(pad - 1) * span / 2 >= distance * tan(max angle)``.
    """
    from .angular_spectrum import propagate_volume
    if isinstance(distance_um, bool) or not isinstance(distance_um, (int, float)) or not math.isfinite(distance_um) or distance_um < 0:
        raise ValueError('distance_um must be a finite nonnegative distance away from the device.')
    volume = propagate_volume(stitched, [float(distance_um)], index=index, components=stitched.components, pad=pad, chunk=1)
    report = dict(stitched.report, propagation=dict(distance_um=distance_um, index=index, pad=pad,
                  method='angular spectrum per Cartesian component, zero padded'))
    return replace(stitched, fields=volume.fields[:, 0], offset_um=stitched.offset_um + stitched.direction * distance_um, report=report,
                   tile_planes=None)


class TiledPlaneSimulation(torch.nn.Module):
    """One DifferentiablePlaneSimulation per tile, stitched on the autograd graph.

    ``epsilon`` covers the global grid, ``(Nx, Ny, Nz)`` or with three Yee
    components. Each tile receives the crop over its full grid (extended region
    plus its own CPML); the crops are views, so the gradient of a cell that
    several tiles contain is the sum of those tiles' gradients. The stitched
    plane is built from each tile's core fields, which depend on that tile's
    crop alone, so the summed gradient is the exact derivative of the stitched
    forward model. Tiles run sequentially; each keeps its checkpoint state
    until backward, so many tiles need host or disk checkpoint storage.
    """
    def __init__(self, project, plan, options=None):
        super().__init__()
        if project.model_dump() != plan.project.model_dump():
            raise ValueError('The plan was made for a different project. Call plan_tiles again.')
        self.plan = plan
        self.models = [DifferentiablePlaneSimulation(t.project, options) for t in plan.tiles]

    def forward(self, epsilon, frequency_hz, *, blend='hard', block_size=32):
        r = self.plan.project.region
        if not isinstance(epsilon, torch.Tensor) or epsilon.ndim not in (3, 4) or tuple(epsilon.shape[:3]) != r.shape or (epsilon.ndim == 4 and epsilon.shape[3] != 3):
            raise ValueError(f'epsilon must cover the global grid {r.shape}, optionally with three Yee components.')
        planes = []
        for tile, model in zip(self.plan.tiles, self.models):
            crop = epsilon
            for a, (f0, f1) in zip(self.plan.lateral, tile.full):
                crop = crop.narrow(a, f0, f1 - f0)
            planes.append(model(crop, frequency_hz, block_size=block_size)[self.plan.monitor_id])
        return stitch_planes(self.plan, planes, blend=blend)
