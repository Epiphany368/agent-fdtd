"""Regularized analytic solids with a bounded, recomputed geometry VJP."""
from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable

from .geometry import validate_polygon
from .solver import field_axes


def _scale(value, factor):
    if isinstance(value, torch.Tensor):
        return value * factor
    return tuple(v * factor for v in value)


@dataclass(frozen=True)
class DifferentiableSolid:
    """Tensor-valued analytic solid, in micrometres and degrees.

    Use the named constructors. Later solids overwrite earlier ones through
    their regularized occupancy. This is not the coupled CAD subpixel operator.
    """
    kind: str
    extent: object
    epsilon: object
    center: object = (0., 0., 0.)
    rotation: object = (0., 0., 0.)
    vertices: object = None
    outline: object = None
    holes: object = ()

    @classmethod
    def box(cls, size, *, epsilon, center=(0., 0., 0.), rotation=(0., 0., 0.)):
        return cls('box', _scale(size, .5), epsilon, center, rotation)

    @classmethod
    def ellipsoid(cls, radii, *, epsilon, center=(0., 0., 0.), rotation=(0., 0., 0.)):
        return cls('ellipsoid', radii, epsilon, center, rotation)

    @classmethod
    def sphere(cls, radius, *, epsilon, center=(0., 0., 0.)):
        return cls.ellipsoid((radius, radius, radius), epsilon=epsilon, center=center)

    @classmethod
    def cylinder(cls, radius, height, *, epsilon, radius_y=None,
                 center=(0., 0., 0.), rotation=(0., 0., 0.)):
        return cls('cylinder', (radius, radius if radius_y is None else radius_y, height / 2),
                   epsilon, center, rotation)

    @classmethod
    def polygon(cls, vertices, z_limits, *, epsilon, holes=(), center=(0., 0., 0.), rotation=(0., 0., 0.)):
        """Simple contour of local (x, y) vertices extruded between local z limits.

        ``holes`` mirrors the native polygon structure but is rejected when
        nonempty: overlay a background-permittivity solid instead.
        """
        return cls('polygon', z_limits, epsilon, center, rotation, vertices, None, tuple(holes))

    @classmethod
    def spline(cls, control_points, z_limits, *, epsilon, kind='catmull_rom', samples_per_segment=8,
               center=(0., 0., 0.), rotation=(0., 0., 0.)):
        """Closed cubic outline through/near local control points, sampled as a polygon."""
        return cls('spline', z_limits, epsilon, center, rotation, control_points, (kind, samples_per_segment))


_SPLINE_BASES = {
    'bspline': (((-1., 3., -3., 1.), (3., -6., 3., 0.), (-3., 0., 3., 0.), (1., 4., 1., 0.)), 6.),
    'catmull_rom': (((-1., 3., -3., 1.), (2., -5., 4., -1.), (-1., 0., 1., 0.), (0., 2., 0., 0.)), 2.),
}


def spline_outline(control_points, *, kind='catmull_rom', samples_per_segment=8):
    """Sample a closed uniform cubic spline as polygon vertices.

    ``control_points`` is an (N, 2) floating tensor whose gradient flows through
    the sampled vertices. Segment i runs from control i towards control i+1
    with periodic indexing, so the outline orientation follows the controls.
    Catmull-Rom passes through every control at its segment start; the
    B-spline stays inside the local control hull. No self-intersection check
    happens here; smooth_geometry_epsilon validates the sampled contour.
    """
    if kind not in _SPLINE_BASES:
        raise ValueError('Spline kind must be catmull_rom or bspline.')
    if isinstance(samples_per_segment, bool) or not isinstance(samples_per_segment, int) or samples_per_segment < 1:
        raise ValueError('samples_per_segment must be a positive integer.')
    points = control_points
    if not isinstance(points, torch.Tensor) or not points.is_floating_point() or points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError('Control points must be a floating (N, 2) tensor with at least three rows.')
    count = points.shape[0]
    u = torch.arange(samples_per_segment, dtype=points.dtype, device=points.device) / samples_per_segment
    powers = torch.stack((u*u*u, u*u, u, torch.ones_like(u)), -1)
    rows, divisor = _SPLINE_BASES[kind]
    weights = powers @ (points.new_tensor(rows) / divisor)
    index = (torch.arange(count, device=points.device)[:, None] + torch.arange(-1, 3, device=points.device)) % count
    return torch.einsum('sk,nkd->nsd', weights, points[index]).reshape(count*samples_per_segment, 2)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _as_tensor(value, like):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return torch.stack([_as_tensor(item, like) for item in value])
    return like.new_tensor(value)


def _rotation(angles):
    x, y, z = angles * (math.pi / 180)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    # Local -> world, fixed world axes x then y then z: Rz @ Ry @ Rx.
    return torch.stack((
        torch.stack((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx)),
        torch.stack((sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx)),
        torch.stack((-sy, cy*sx, cy*cx)),
    ))


def _step(distance, width):
    fraction = (distance / width + .5).clamp(0, 1)
    return fraction.square() * (3 - 2*fraction)


_EDGE_BLOCK = 64


def _polygon_distance(xy, vertices):
    """Signed Euclidean distance to a simple polygon, positive inside.

    The nearest edge and the even-odd parity are found without gradient in
    edge blocks, so transient memory is O(points * block). The distance is then
    recomputed from that edge alone, which keeps the graph at O(points) and
    gives the envelope derivative with respect to vertices and sample points.
    """
    count = vertices.shape[0]
    with torch.no_grad():
        point = xy.detach()
        a = vertices.detach()
        b = a.roll(-1, 0)
        x, y = point[:, :1], point[:, 1:]
        crossings = torch.zeros(point.shape[0], dtype=torch.int64, device=point.device)
        best_value = nearest = None
        for start in range(0, count, _EDGE_BLOCK):
            ea, eb = a[start:start+_EDGE_BLOCK], b[start:start+_EDGE_BLOCK]
            edge = eb - ea
            offset = point[:, None, :] - ea[None]
            fraction = ((offset*edge).sum(-1) / (edge*edge).sum(-1)).clamp(0, 1)
            value, index = (offset - fraction[..., None]*edge[None]).square().sum(-1).min(1)
            index = index + start
            if best_value is None:
                best_value, nearest = value, index
            else:
                closer = value < best_value
                best_value = torch.where(closer, value, best_value)
                nearest = torch.where(closer, index, nearest)
            straddle = (ea[None, :, 1] > y) != (eb[None, :, 1] > y)
            # A straddling edge never has equal endpoint y, so the division is safe.
            dy = torch.where(straddle, edge[None, :, 1].expand_as(straddle), torch.ones_like(value)[:, None])
            crossing_x = ea[None, :, 0] + edge[None, :, 0]*(y - ea[None, :, 1])/dy
            crossings += (straddle & (x < crossing_x)).sum(1)
        sign = (crossings % 2 == 1).to(xy.dtype)*2 - 1
    start = vertices[nearest]
    edge = vertices[(nearest+1) % count] - start
    offset = xy - start
    fraction = ((offset*edge).sum(-1) / (edge*edge).sum(-1)).clamp(0, 1)
    square = (offset - fraction[:, None]*edge).square().sum(-1)
    # Avoid the undefined sqrt derivative exactly on the contour.
    return sign * torch.sqrt(square + torch.finfo(square.dtype).tiny)


def _evaluate(points, parameters, background, layout, width):
    epsilon = background.expand(points.shape[0])
    for kind, offset, count in layout:
        row = parameters[offset:offset+count]
        if kind == 'polygon':
            # center(3), z limits(2), rotation(3), epsilon(1), vertices(2N).
            local = (points - row[:3]) @ _rotation(row[5:8])
            occupancy = _step(_polygon_distance(local[:, :2], row[9:].reshape(-1, 2)), width)
            occupancy = occupancy * _step(torch.minimum(local[:, 2]-row[3], row[4]-local[:, 2]), width)
            epsilon = epsilon * (1-occupancy) + row[8] * occupancy
            continue
        # center(3), extent(3), rotation(3), epsilon(1).
        local = (points - row[:3]) @ _rotation(row[6:9])
        extent = row[3:6]
        if kind == 'box':
            occupancy = _step(extent-local.abs(), width).prod(-1)
        else:
            dimensions = 3 if kind == 'ellipsoid' else 2
            scaled = local[:, :dimensions] / extent[:dimensions]
            level = 1 - torch.linalg.vector_norm(scaled, dim=-1)
            occupancy = _step(level * extent[:dimensions].amin(), width)
            if kind == 'cylinder':
                occupancy = occupancy * _step(extent[2]-local[:, 2].abs(), width)
        epsilon = epsilon * (1-occupancy) + row[9] * occupancy
    return epsilon


def _points(axes, begin, end, shape):
    indices = torch.arange(begin, end, device=axes[0].device)
    x = indices // (shape[1]*shape[2])
    y = (indices // shape[2]) % shape[1]
    z = indices % shape[2]
    return torch.stack((axes[0][x], axes[1][y], axes[2][z]), -1)


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameters, background, layout, shape, width, chunk_cells, *axes):
        components = len(axes) // 3
        ctx.save_for_backward(parameters, background, *axes)
        ctx.spec = layout, shape, width, chunk_cells, components
        output = parameters.new_empty((math.prod(shape), components))
        for begin in range(0, output.shape[0], chunk_cells):
            end = min(begin+chunk_cells, output.shape[0])
            for component in range(components):
                points = _points(axes[3*component:3*component+3], begin, end, shape)
                output[begin:end, component] = _evaluate(points, parameters, background, layout, width)
        return output.reshape(shape + ((3,) if components == 3 else ()))

    @staticmethod
    @once_differentiable
    def backward(ctx, seed):
        parameters, background, *axes = ctx.saved_tensors
        layout, shape, width, chunk_cells, components = ctx.spec
        dp, db = torch.zeros_like(parameters), torch.zeros_like(background)
        # Do not flatten a noncontiguous caller VJP into a full-domain copy.
        for begin in range(0, math.prod(shape), chunk_cells):
            end = min(begin+chunk_cells, math.prod(shape))
            indices = torch.arange(begin, end, device=seed.device)
            index = (indices // (shape[1]*shape[2]), (indices // shape[2]) % shape[1], indices % shape[2])
            for component in range(components):
                weights = seed[index + ((component,) if components == 3 else ())]
                with torch.enable_grad():
                    local_parameters = parameters.detach().requires_grad_(True)
                    local_background = background.detach().requires_grad_(True)
                    points = _points(axes[3*component:3*component+3], begin, end, shape)
                    value = _evaluate(points, local_parameters, local_background, layout, width)
                    gradients = torch.autograd.grad(value, (local_parameters, local_background), weights,
                                                    allow_unused=True)
                if gradients[0] is not None:
                    dp.add_(gradients[0])
                db.add_(gradients[1])
                del value, gradients, points, local_parameters, local_background
        return (dp, db, None, None, None, None) + (None,)*len(axes)


def _polygon_row(solid, values, like):
    if values[0].shape != (3,) or values[1].shape != (2,) or values[2].shape != (3,) or values[3].ndim != 0:
        raise ValueError('Polygon center and rotation need three values, z limits two values, and epsilon must be scalar.')
    if len(solid.holes):
        raise ValueError('Differentiable polygons do not take holes; overlay a solid with the background permittivity.')
    vertices = _as_tensor(solid.vertices, like)
    if solid.kind == 'spline':
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError('Spline control points need at least three (x, y) pairs.')
        vertices = spline_outline(vertices, kind=solid.outline[0], samples_per_segment=solid.outline[1])
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError('Polygon vertices need (x, y) pairs.')
    if not bool(torch.isfinite(vertices).all()) or not bool(values[1][1] > values[1][0]):
        raise ValueError('Polygon vertices must be finite and the upper z limit must exceed the lower one.')
    # The same simple-contour validator as the native polygon structure.
    validate_polygon(vertices.detach().cpu().numpy())
    return torch.cat((values[0], values[1], values[2], values[3].reshape(1), vertices.reshape(-1)))


def smooth_geometry_epsilon(region, solids, *, background=1., width=None, chunk_cells=65536,
                            device=None):
    """Rasterize and differentiate ordered solids without a full geometry graph.

    The mesh and transition width are fixed. A custom first-order backward
    replays one chunk and electric component at a time. The dense epsilon
    output and its incoming VJP still occupy full-grid storage. Local graph
    memory scales with chunk_cells and solid count, not global cell count.
    Polygon and spline solids add a transient nearest-edge search of
    chunk_cells by 64 edges per block; the retained graph stays O(chunk_cells).

    FP32 is the default for an FP32 region. Tensor inputs must share the
    region's precision and their device. For streamed FDTD keep them on CPU.
    Periodic copies are explicit solids, never inferred from boundary types.
    """
    solids = tuple(solids)
    if region.interface_method != 'staircase':
        raise ValueError('Regularized geometry supplies diagonal epsilon, not coupled subpixel coefficients.')
    if isinstance(chunk_cells, bool) or not isinstance(chunk_cells, int) or chunk_cells < 1:
        raise ValueError('chunk_cells must be a positive integer.')
    width = region.reference_step if width is None else width
    if isinstance(width, torch.Tensor) or not math.isfinite(width) or width <= 0:
        raise ValueError('Transition width must be a fixed positive finite scalar.')
    if any(not isinstance(s, DifferentiableSolid) or s.kind not in ('box', 'ellipsoid', 'cylinder', 'polygon', 'spline') for s in solids):
        raise ValueError('Use DifferentiableSolid box, sphere, ellipsoid, cylinder, polygon or spline constructors.')
    inputs = [background, *[v for s in solids for v in (s.center, s.extent, s.rotation, s.epsilon, s.vertices)]]
    tensors = list(_tensors(inputs))
    dtype = getattr(torch, region.precision)
    chosen_device = torch.device(device) if device is not None else tensors[0].device if tensors else torch.device('cpu')
    if chosen_device.type not in ('cpu', 'cuda'):
        raise ValueError('Geometry supports CPU or CUDA.')
    like = torch.empty((), dtype=dtype, device=chosen_device)
    if any(t.dtype != dtype or t.device != like.device for t in tensors):
        raise ValueError('Tensor geometry inputs must share the region precision and selected device.')
    rows, layout, offset = [], [], 0
    for solid in solids:
        values = [_as_tensor(v, like) for v in (solid.center, solid.extent, solid.rotation, solid.epsilon)]
        if solid.kind in ('polygon', 'spline'):
            row = _polygon_row(solid, values, like)
            layout.append(('polygon', offset, row.numel()))
        else:
            if any(v.shape != (3,) for v in values[:3]) or values[3].ndim != 0:
                raise ValueError('Center, extent and rotation need three values, and epsilon must be scalar.')
            if bool((values[1] <= 0).any()):
                raise ValueError('Solid parameters must be finite and extents positive.')
            row = torch.cat((*values[:3], values[3].reshape(1)))
            layout.append((solid.kind, offset, 10))
        rows.append(row)
        offset += row.numel()
    parameters = torch.cat(rows) if rows else like.new_empty(0)
    background = _as_tensor(background, like)
    if background.ndim != 0 or not bool(torch.isfinite(background)) or not bool(background >= 1):
        raise ValueError('Background epsilon must be a finite scalar at least one.')
    if not bool(torch.isfinite(parameters).all()):
        raise ValueError('Solid parameters must be finite and extents positive.')
    if any(bool(parameters[offset + (8 if kind == 'polygon' else 9)] < 1) for kind, offset, _ in layout):
        raise ValueError('Solid epsilon must be at least one for the conservative CFL contract.')
    if region.material_sampling == 'yee':
        coordinates = [axis for c in ('Ex', 'Ey', 'Ez') for axis in field_axes(region, c)]
    else:
        coordinates = [(nodes[:-1]+nodes[1:])/2 for nodes in region.mesh_nodes]
    axes = tuple(torch.tensor(axis, dtype=dtype, device=like.device) for axis in coordinates)
    return _Geometry.apply(parameters, background, tuple(layout), region.shape,
                           width, chunk_cells, *axes)
