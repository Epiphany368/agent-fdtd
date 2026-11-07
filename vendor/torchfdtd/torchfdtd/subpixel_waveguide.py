"""Conformal Variant 0 anisotropic subpixel permittivity for 2D cross-sections and 3D waveguides.

Provides Lumerical Conformal Variant 0 constitutive averaging on the staggered Yee
grid for dielectric strip waveguides, rib waveguides, strip-to-rib tapers, and
multi-branch photonic devices:
- Longitudinal electric component Ew (tangential to both transverse interfaces):
  2D arithmetic area average <eps(u, v)>_{u, v} over the staggered control volume.
- Primary transverse electric component Eu (normal to vertical u-sidewalls,
  tangential to horizontal v-interfaces):
  1D harmonic average along u followed by 1D arithmetic average along v:
  < (<eps(u, v)^{-1}>_u)^{-1} >_v.
- Secondary transverse electric component Ev (normal to horizontal v-interfaces,
  tangential to vertical u-sidewalls):
  1D harmonic average along v followed by 1D arithmetic average along u:
  < (<eps(u, v)^{-1}>_v)^{-1} >_u.

For axis-aligned rectangular slabs/ridges (including tapers with x-dependent widths),
the 2D subcell integrals are evaluated analytically via exact interval partitioning
with zero quadrature discretization noise. General 2D polygons are supported via
tensor-product Gauss-Legendre or midpoint subgrid quadrature.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import numpy as np

from .geometry import polygon_contains


def _infer_step(coords: np.ndarray, axis: int) -> np.ndarray:
    """Infer local cell spacing along `axis` from 1D or 2D coordinate arrays."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[axis if arr.ndim > 1 else 0] < 2:
        raise ValueError('Explicit grid spacing du_um/dv_um (or dy_um/dz_um) is required for singleton axes.')
    target_axis = axis if arr.ndim > 1 else 0
    return np.abs(np.gradient(arr, axis=target_axis))


def _normalize_slabs(polygons_yz_or_slabs, slabs, eps_core=None):
    """Normalize slab/polygon descriptors into ('rect', u_min, u_max, v_min, v_max, eps) or ('poly', vertices, holes, eps)."""
    raw = polygons_yz_or_slabs if polygons_yz_or_slabs is not None else slabs
    if raw is None:
        raise ValueError('Provide polygons_yz_or_slabs or slabs describing the cross-section.')
    items = []
    for entry in raw:
        if isinstance(entry, dict):
            u_min = entry.get('u_min', entry.get('y_min'))
            u_max = entry.get('u_max', entry.get('y_max'))
            v_min = entry.get('v_min', entry.get('z_min'))
            v_max = entry.get('v_max', entry.get('z_max'))
            eps = entry.get('eps', entry.get('epsilon', eps_core))
            if u_min is not None and u_max is not None and v_min is not None and v_max is not None:
                items.append(('rect', float(u_min), float(u_max), float(v_min), float(v_max), float(eps)))
            elif 'vertices' in entry:
                holes = entry.get('holes', ())
                items.append(('poly', np.asarray(entry['vertices'], dtype=np.float64), holes, float(eps)))
            else:
                raise ValueError('Dictionary slab entry requires (y_min, y_max, z_min, z_max, eps) or (vertices, eps).')
        elif isinstance(entry, (tuple, list)):
            if len(entry) >= 2 and isinstance(entry[0], str) and entry[0] in ('rect', 'poly'):
                items.append(tuple(entry))
            elif len(entry) == 5 and np.isscalar(entry[0]):
                u_min, u_max, v_min, v_max, eps = entry
                items.append(('rect', float(u_min), float(u_max), float(v_min), float(v_max), float(eps)))
            elif len(entry) == 4 and np.isscalar(entry[0]):
                if eps_core is None:
                    raise ValueError('4-tuple rectangle (u_min, u_max, v_min, v_max) requires eps_core.')
                u_min, u_max, v_min, v_max = entry
                items.append(('rect', float(u_min), float(u_max), float(v_min), float(v_max), float(eps_core)))
            elif len(entry) == 2:
                verts, eps = entry
                items.append(('poly', np.asarray(verts, dtype=np.float64), (), float(eps)))
            elif len(entry) == 3 and not np.isscalar(entry[0]):
                verts, holes, eps = entry
                items.append(('poly', np.asarray(verts, dtype=np.float64), holes, float(eps)))
            else:
                raise ValueError(f'Unsupported slab/polygon descriptor: {entry!r}')
        else:
            raise ValueError(f'Unsupported slab/polygon descriptor: {entry!r}')
    return items


def _analytical_rect_conformal_variant0(
    u: np.ndarray,
    v: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    rects: Sequence[tuple[float, float, float, float, float]],
    eps_bg: float,
    role: str,
) -> np.ndarray:
    """Exact analytical 2D Conformal Variant 0 integral for arbitrary axis-aligned rectangles.

    Partitions each cell [u - du/2, u + du/2] x [v - dv/2, v + dv/2] along all
    rectangle u-boundaries and v-boundaries so that overlaps, ribs, and multi-layer
    stacks are integrated with zero quadrature error. Later rectangles override
    earlier rectangles where they overlap.
    """
    u_lo = u - 0.5 * du
    u_hi = u + 0.5 * du
    v_lo = v - 0.5 * dv
    v_hi = v + 0.5 * dv

    if not rects:
        return np.full(u.shape, float(eps_bg), dtype=np.float64)

    u_cuts = sorted({float(r[1]) for r in rects} | {float(r[2]) for r in rects})
    v_cuts = sorted({float(r[3]) for r in rects} | {float(r[4]) for r in rects})

    # Elementary intervals (-inf, c_0), (c_0, c_1), ..., (c_{m-1}, +inf)
    u_edges = [-np.inf] + u_cuts + [np.inf]
    v_edges = [-np.inf] + v_cuts + [np.inf]
    n_u_int = len(u_edges) - 1
    n_v_int = len(v_edges) - 1

    # Material epsilon on each elementary rectangle (iu, iv)
    patch_eps = np.full((n_u_int, n_v_int), float(eps_bg), dtype=np.float64)
    u_mids = np.array([
        (u_edges[i] - 1.0 if not np.isfinite(u_edges[i]) else
         (u_edges[i + 1] + 1.0 if not np.isfinite(u_edges[i + 1]) else 0.5 * (u_edges[i] + u_edges[i + 1])))
        for i in range(n_u_int)
    ])
    v_mids = np.array([
        (v_edges[j] - 1.0 if not np.isfinite(v_edges[j]) else
         (v_edges[j + 1] + 1.0 if not np.isfinite(v_edges[j + 1]) else 0.5 * (v_edges[j] + v_edges[j + 1])))
        for j in range(n_v_int)
    ])
    for _, umin, umax, vmin, vmax, eps_val in rects:
        u_mask = (u_mids >= umin) & (u_mids <= umax)
        v_mask = (v_mids >= vmin) & (v_mids <= vmax)
        patch_eps[np.ix_(u_mask, v_mask)] = float(eps_val)

    # Fractional overlap of each cell with each elementary u-interval and v-interval
    # Shape: (n_u_int, *u.shape) and (n_v_int, *v.shape)
    fu = np.empty((n_u_int,) + u.shape, dtype=np.float64)
    for i in range(n_u_int):
        a, b = u_edges[i], u_edges[i + 1]
        lo = u_lo if not np.isfinite(a) else np.maximum(u_lo, a)
        hi = u_hi if not np.isfinite(b) else np.minimum(u_hi, b)
        fu[i] = np.maximum(0.0, hi - lo) / du

    fv = np.empty((n_v_int,) + v.shape, dtype=np.float64)
    for j in range(n_v_int):
        a, b = v_edges[j], v_edges[j + 1]
        lo = v_lo if not np.isfinite(a) else np.maximum(v_lo, a)
        hi = v_hi if not np.isfinite(b) else np.minimum(v_hi, b)
        fv[j] = np.maximum(0.0, hi - lo) / dv

    if role == 'w':
        # 2D arithmetic average: sum_{i,j} fu[i] * fv[j] * patch_eps[i, j]
        return np.einsum('i...,ij,j...->...', fu, patch_eps, fv)
    if role == 'u':
        # Harmonic along u for each v-interval j, then arithmetic along v
        inv_patch = 1.0 / patch_eps  # (n_u_int, n_v_int)
        # For each j: <eps^{-1}>_u = sum_i fu[i] * inv_patch[i, j]
        inv_u_per_v = np.einsum('i...,ij->j...', fu, inv_patch)
        harm_u_per_v = 1.0 / np.maximum(inv_u_per_v, 1e-30)
        return np.einsum('j...,j...->...', fv, harm_u_per_v)
    if role == 'v':
        # Harmonic along v for each u-interval i, then arithmetic along u
        inv_patch = 1.0 / patch_eps  # (n_u_int, n_v_int)
        inv_v_per_u = np.einsum('ij,j...->i...', inv_patch, fv)
        harm_v_per_u = 1.0 / np.maximum(inv_v_per_u, 1e-30)
        return np.einsum('i...,i...->...', fu, harm_v_per_u)
    raise ValueError(f'Unknown component role: {role!r}')


_GL4_NODES = np.array(
    [-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526],
    dtype=np.float64,
)
_GL4_WEIGHTS = np.array(
    [0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385],
    dtype=np.float64,
)


def _extract_all_edges(items: Sequence[tuple]) -> np.ndarray:
    """Return array of edges (ua, va, ub, vb) for all rectangles and polygons."""
    edge_list = []
    for item in items:
        if item[0] == 'rect':
            _, umin, umax, vmin, vmax, _ = item
            ring = np.array(
                [[umin, vmin], [umax, vmin], [umax, vmax], [umin, vmax]], dtype=np.float64
            )
            rings = [ring]
        else:
            _, verts, holes, _ = item
            rings = [verts] + (list(holes) if holes else [])
        for ring in rings:
            nxt = np.roll(ring, -1, axis=0)
            edges = np.column_stack([ring[:, 0], ring[:, 1], nxt[:, 0], nxt[:, 1]])
            edge_list.append(edges)
    if not edge_list:
        return np.empty((0, 4), dtype=np.float64)
    return np.vstack(edge_list)


def _eval_point_permittivity(
    u_pts: np.ndarray,
    v_pts: np.ndarray,
    items: Sequence[tuple],
    eps_bg: float,
) -> np.ndarray:
    """Evaluate pointwise permittivity at (u_pts, v_pts) using painter's order."""
    out = np.full(u_pts.shape, float(eps_bg), dtype=np.float64)
    for item in items:
        if item[0] == 'rect':
            _, umin, umax, vmin, vmax, eps_val = item
            mask = (u_pts >= umin) & (u_pts <= umax) & (v_pts >= vmin) & (v_pts <= vmax)
        else:
            _, verts, holes, eps_val = item
            mask = polygon_contains(u_pts, v_pts, verts, holes)
        out[mask] = eps_val
    return out


def _integrate_cell_conformal(
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    cell_edges: np.ndarray,
    items: Sequence[tuple],
    eps_bg: float,
    role: str,
    slab_edges: np.ndarray | None = None,
) -> float:
    """Piecewise 4-point Gauss-Legendre outer + exact analytical 1D scanline inner integral."""
    # If role == 'v' (harmonic along v, arithmetic along u), outer axis is u and inner is v.
    # Otherwise (role == 'u' or 'w'), outer axis is v and inner is u.
    outer_is_v = role != 'v'
    if outer_is_v:
        o0, o1 = v0, v1
        i0, i1 = u0, u1
        ea_i, ea_o, eb_i, eb_o = (
            cell_edges[:, 0],
            cell_edges[:, 1],
            cell_edges[:, 2],
            cell_edges[:, 3],
        )
    else:
        o0, o1 = u0, u1
        i0, i1 = v0, v1
        ea_i, ea_o, eb_i, eb_o = (
            cell_edges[:, 1],
            cell_edges[:, 0],
            cell_edges[:, 3],
            cell_edges[:, 2],
        )

    # Critical outer-axis cuts where vertices lie or edges cross i0 / i1
    cuts = [o0, o1]
    for val in np.concatenate([ea_o, eb_o]):
        if o0 + 1e-13 < val < o1 - 1e-13:
            cuts.append(float(val))
    di_edge = eb_i - ea_i
    nonzero_i = np.abs(di_edge) > 1e-14
    if np.any(nonzero_i):
        for bound_i in (i0, i1):
            t = (bound_i - ea_i[nonzero_i]) / di_edge[nonzero_i]
            valid = (t >= 0.0) & (t <= 1.0)
            if np.any(valid):
                o_cross = ea_o[nonzero_i][valid] + t[valid] * (
                    eb_o[nonzero_i][valid] - ea_o[nonzero_i][valid]
                )
                for val in o_cross:
                    if o0 + 1e-13 < val < o1 - 1e-13:
                        cuts.append(float(val))

    o_cuts = np.unique(np.round(cuts, decimals=13))
    o_cuts[0] = o0
    o_cuts[-1] = o1
    do_Len = o1 - o0
    di_Len = i1 - i0
    if do_Len <= 0.0 or di_Len <= 0.0:
        return float(eps_bg)

    if len(items) == 1 and slab_edges is not None:
        eps_core = float(items[0][-1])
        if outer_is_v:
            s_ea_i, s_ea_o, s_eb_i, s_eb_o = (
                slab_edges[:, 0],
                slab_edges[:, 1],
                slab_edges[:, 2],
                slab_edges[:, 3],
            )
        else:
            s_ea_i, s_ea_o, s_eb_i, s_eb_o = (
                slab_edges[:, 1],
                slab_edges[:, 0],
                slab_edges[:, 3],
                slab_edges[:, 2],
            )
        s_di = s_eb_i - s_ea_i
        s_do = s_eb_o - s_ea_o
        total_outer = 0.0
        for k in range(len(o_cuts) - 1):
            oa, ob = o_cuts[k], o_cuts[k + 1]
            half_do = 0.5 * (ob - oa)
            if half_do <= 1e-14:
                continue
            mid_o = 0.5 * (oa + ob)
            for q_node, q_weight in zip(_GL4_NODES, _GL4_WEIGHTS):
                o_s = mid_o + half_do * float(q_node)
                cross = ((s_ea_o <= o_s) & (o_s < s_eb_o)) | ((s_eb_o <= o_s) & (o_s < s_ea_o))
                f_core = 0.0
                if np.any(cross):
                    ic = np.sort(
                        s_ea_i[cross] + (o_s - s_ea_o[cross]) * s_di[cross] / s_do[cross]
                    )
                    for m in range(0, len(ic) - 1, 2):
                        lo = max(i0, float(ic[m]))
                        hi = min(i1, float(ic[m + 1]))
                        if hi > lo:
                            f_core += (hi - lo) / di_Len
                f_core = min(max(f_core, 0.0), 1.0)
                if role == 'w':
                    g_val = f_core * eps_core + (1.0 - f_core) * float(eps_bg)
                else:
                    g_val = 1.0 / (f_core / eps_core + (1.0 - f_core) / float(eps_bg))
                total_outer += half_do * float(q_weight) * g_val
        return total_outer / do_Len

    do_edge = eb_o - ea_o
    total_outer = 0.0
    for k in range(len(o_cuts) - 1):
        oa, ob = o_cuts[k], o_cuts[k + 1]
        half_do = 0.5 * (ob - oa)
        if half_do <= 1e-14:
            continue
        mid_o = 0.5 * (oa + ob)
        for q_node, q_weight in zip(_GL4_NODES, _GL4_WEIGHTS):
            o_s = mid_o + half_do * float(q_node)
            # Analytical 1D crossings along inner axis at outer coordinate o_s
            cross_mask = ((ea_o <= o_s) & (o_s < eb_o)) | ((eb_o <= o_s) & (o_s < ea_o))
            if np.any(cross_mask):
                i_cross = ea_i[cross_mask] + (o_s - ea_o[cross_mask]) * (
                    eb_i[cross_mask] - ea_i[cross_mask]
                ) / do_edge[cross_mask]
                i_clipped = np.clip(i_cross, i0, i1)
                i_pts = np.unique(np.concatenate([[i0, i1], i_clipped]))
            else:
                i_pts = np.array([i0, i1], dtype=np.float64)

            seg_len = np.diff(i_pts)
            keep = seg_len > 1e-14
            if not np.any(keep):
                continue
            seg_len = seg_len[keep]
            i_mid = 0.5 * (i_pts[:-1][keep] + i_pts[1:][keep])
            o_arr = np.full_like(i_mid, o_s)
            if outer_is_v:
                eps_seg = _eval_point_permittivity(i_mid, o_arr, items, eps_bg)
            else:
                eps_seg = _eval_point_permittivity(o_arr, i_mid, items, eps_bg)

            if role == 'w':
                g_val = float(np.sum(seg_len * eps_seg) / di_Len)
            else:
                # Harmonic average along inner axis (u for role=='u', v for role=='v')
                g_val = float(di_Len / np.sum(seg_len / eps_seg))

            total_outer += half_do * float(q_weight) * g_val

    return total_outer / do_Len


def _subgrid_conformal_variant0(
    u: np.ndarray,
    v: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    items: Sequence[tuple],
    eps_bg: float,
    role: str,
    n_sub: int,
) -> np.ndarray:
    """Exact 1D scanline + piecewise Gauss-Legendre Conformal Variant 0 for 2D polygons."""
    del n_sub  # Retained for API compatibility; integration is piecewise analytical + GL4
    flat_u = u.ravel()
    flat_v = v.ravel()
    flat_du = np.broadcast_to(du, u.shape).ravel()
    flat_dv = np.broadcast_to(dv, v.shape).ravel()
    n_pts = len(flat_u)

    # First evaluate all cell centers in one vectorized pass (exact for non-boundary cells)
    out = _eval_point_permittivity(flat_u, flat_v, items, eps_bg)

    all_edges = _extract_all_edges(items)
    if len(all_edges) == 0:
        return out.reshape(u.shape)

    edge_umin = np.minimum(all_edges[:, 0], all_edges[:, 2])
    edge_umax = np.maximum(all_edges[:, 0], all_edges[:, 2])
    edge_vmin = np.minimum(all_edges[:, 1], all_edges[:, 3])
    edge_vmax = np.maximum(all_edges[:, 1], all_edges[:, 3])

    u0_all = flat_u - 0.5 * flat_du
    u1_all = flat_u + 0.5 * flat_du
    v0_all = flat_v - 0.5 * flat_dv
    v1_all = flat_v + 0.5 * flat_dv

    # Find cells whose box overlaps at least one edge bounding box
    poly_umin = float(edge_umin.min())
    poly_umax = float(edge_umax.max())
    poly_vmin = float(edge_vmin.min())
    poly_vmax = float(edge_vmax.max())
    cand_ids = np.flatnonzero(
        (u1_all >= poly_umin - 1e-12)
        & (u0_all <= poly_umax + 1e-12)
        & (v1_all >= poly_vmin - 1e-12)
        & (v0_all <= poly_vmax + 1e-12)
    )

    outer_is_v = role != 'v'
    for idx in cand_ids:
        u0, u1 = float(u0_all[idx]), float(u1_all[idx])
        v0, v1 = float(v0_all[idx]), float(v1_all[idx])
        slab_hit = (
            (edge_vmax >= v0 - 1e-12) & (edge_vmin <= v1 + 1e-12)
            if outer_is_v
            else (edge_umax >= u0 - 1e-12) & (edge_umin <= u1 + 1e-12)
        )
        hit = (
            slab_hit
            & ((edge_umax >= u0 - 1e-12) & (edge_umin <= u1 + 1e-12))
            if outer_is_v
            else slab_hit & ((edge_vmax >= v0 - 1e-12) & (edge_vmin <= v1 + 1e-12))
        )
        if not np.any(hit):
            continue
        out[idx] = _integrate_cell_conformal(
            u0, u1, v0, v1, all_edges[hit], items, eps_bg, role, slab_edges=all_edges[slab_hit]
        )

    return out.reshape(u.shape)


def conformal_variant0_cross_section(
    u_um: np.ndarray,
    v_um: np.ndarray,
    comp: str | int | None = 'Ey',
    *,
    polygons_yz_or_slabs=None,
    slabs=None,
    eps_core: float | None = None,
    du_um: float | np.ndarray | None = None,
    dv_um: float | np.ndarray | None = None,
    dy_um: float | np.ndarray | None = None,
    dz_um: float | np.ndarray | None = None,
    eps_bg: float = 1.0,
    normal: str = 'x',
    n_sub: int = 64,
) -> np.ndarray:
    """Compute Conformal Variant 0 diagonal permittivity on a 2D waveguide cross-section.

    Parameters
    ----------
    u_um, v_um : array-like
        Cyclic transverse coordinates (for `normal='x'`, `u_um = y_um` and `v_um = z_um`),
        matching the coordinates passed by `solve_waveguide_modes` to `permittivity(u, v, comp)`.
    comp : {'Ex', 'Ey', 'Ez', 0, 1, 2, 'all', None}
        Electric field component being sampled. When `'all'` or `None`, returns a
        `(..., 3)` array in global `(Ex, Ey, Ez)` order.
    polygons_yz_or_slabs, slabs : sequence
        Sequence of rectangular slab descriptors `(u_min, u_max, v_min, v_max, eps)`
        (or dicts) or 2D polygon descriptors `(vertices_uv, eps)`.
    du_um, dv_um, dy_um, dz_um : float or array-like, optional
        Transverse cell sizes in micrometres. If omitted, inferred from `u_um, v_um`.
    eps_bg : float
        Background cladding permittivity.
    normal : {'x', 'y', 'z'}
        Waveguide propagation normal axis.
    n_sub : int
        Subcell resolution per axis when general polygon contours are present.
    """
    if normal not in ('x', 'y', 'z'):
        raise ValueError("normal must be 'x', 'y', or 'z'.")
    w_axis = 'xyz'.index(normal)
    u_axis, v_axis = (w_axis + 1) % 3, (w_axis + 2) % 3

    u_arr, v_arr = np.broadcast_arrays(np.asarray(u_um, dtype=np.float64), np.asarray(v_um, dtype=np.float64))
    step_u = du_um if du_um is not None else dy_um
    step_v = dv_um if dv_um is not None else dz_um
    if step_u is None:
        step_u = _infer_step(u_arr, 0)
    if step_v is None:
        step_v = _infer_step(v_arr, 1 if v_arr.ndim > 1 else 0)
    du = np.broadcast_to(np.asarray(step_u, dtype=np.float64), u_arr.shape)
    dv = np.broadcast_to(np.asarray(step_v, dtype=np.float64), v_arr.shape)

    items = _normalize_slabs(polygons_yz_or_slabs, slabs, eps_core=eps_core)
    only_rects = all(item[0] == 'rect' for item in items)

    def _eval_axis(global_axis: int) -> np.ndarray:
        if global_axis == w_axis:
            role = 'w'
        elif global_axis == u_axis:
            role = 'u'
        elif global_axis == v_axis:
            role = 'v'
        else:
            raise ValueError(f'Invalid global axis: {global_axis}')
        if only_rects:
            return _analytical_rect_conformal_variant0(u_arr, v_arr, du, dv, items, float(eps_bg), role)
        return _subgrid_conformal_variant0(u_arr, v_arr, du, dv, items, float(eps_bg), role, int(n_sub))

    if comp is None or comp == 'all':
        return np.stack([_eval_axis(0), _eval_axis(1), _eval_axis(2)], axis=-1)
    if isinstance(comp, int):
        if comp not in (0, 1, 2):
            raise ValueError('Integer component index must be 0, 1, or 2.')
        return _eval_axis(comp)
    if comp in ('Ex', 'Ey', 'Ez'):
        return _eval_axis('xyz'.index(comp[1].lower()))
    if comp in ('x', 'y', 'z'):
        return _eval_axis('xyz'.index(comp))
    raise ValueError(f'Unknown electric component {comp!r}; expected Ex, Ey, Ez, or all.')


def make_conformal_variant0_section(
    *,
    polygons_yz_or_slabs=None,
    slabs=None,
    eps_core: float | None = None,
    du_um: float | np.ndarray | None = None,
    dv_um: float | np.ndarray | None = None,
    dy_um: float | np.ndarray | None = None,
    dz_um: float | np.ndarray | None = None,
    eps_bg: float = 1.0,
    normal: str = 'x',
    n_sub: int = 64,
) -> Callable:
    """Return a 3-argument callable `section(u_um, v_um, comp='Ey')` for mode solvers and launches."""
    items = _normalize_slabs(polygons_yz_or_slabs, slabs, eps_core=eps_core)

    def section(u_um, v_um, comp='Ey'):
        return conformal_variant0_cross_section(
            u_um,
            v_um,
            comp,
            slabs=items,
            du_um=du_um,
            dv_um=dv_um,
            dy_um=dy_um,
            dz_um=dz_um,
            eps_bg=eps_bg,
            normal=normal,
            n_sub=n_sub,
        )

    return section


def _staggered_control_widths(nodes: np.ndarray, offset: float, n: int) -> np.ndarray:
    """Control-volume widths along one axis for staggered Yee samples (offset 0.5 or 0.0)."""
    cell_widths = np.diff(nodes)
    if offset == 0.5:
        return cell_widths[:n]
    # Nodal samples: dual cell width is average of adjacent primary cells
    dual = np.empty(len(nodes), dtype=np.float64)
    dual[0] = cell_widths[0]
    dual[-1] = cell_widths[-1]
    if len(nodes) > 2:
        dual[1:-1] = 0.5 * (cell_widths[:-1] + cell_widths[1:])
    return dual[:n]


def conformal_variant0_strip_rib_3d(
    region,
    *,
    eps_core: float,
    eps_bg: float,
    slabs_fn: Callable | None = None,
    width_fn: Callable | float | None = None,
    z_bounds: tuple[float, float] | None = None,
    slab_width_fn: Callable | float | None = None,
    slab_z_bounds: tuple[float, float] | None = None,
    center_y_fn: Callable | float = 0.0,
    x_bounds: tuple[float, float] | None = None,
    normal: str = 'x',
    n_sub_w: int = 4,
    dtype=None,
) -> np.ndarray:
    """Build a 3D `(nx, ny, nz, 3)` Conformal Variant 0 diagonal permittivity array on `region`.

    Supports both uniform and graded 3D rectilinear meshes via `field_axes(region, comp)`
    and exact local Yee control volumes.

    Parameters
    ----------
    region : Region
        3D simulation region (uniform or graded mesh).
    eps_core : float
        Waveguide core permittivity (e.g. `n_Si**2`).
    eps_bg : float
        Background cladding permittivity (e.g. `n_SiO2**2`).
    slabs_fn : callable, optional
        Function `slabs_fn(w_um)` returning a list of `(u_min, u_max, v_min, v_max, eps)`
        at longitudinal coordinate `w_um` (scalar or 1D array).
    width_fn : float or callable, optional
        Strip/ridge width in micrometres (constant float or `width_fn(w_um)`).
    z_bounds : (float, float), optional
        Vertical `(z_min, z_max)` bounds of the strip/ridge core in micrometres.
    slab_width_fn : float or callable, optional
        For rib waveguides or strip-to-rib tapers, the slab width (float or `slab_width_fn(w_um)`).
    slab_z_bounds : (float, float), optional
        Vertical `(z_slab_min, z_slab_max)` bounds of the partial-etch slab.
    center_y_fn : float or callable, default 0.0
        Transverse center coordinate `y_center(w_um)` of the waveguide.
    x_bounds : (float, float), optional
        Optional longitudinal interval `(w_min, w_max)` outside which only background exists.
    normal : {'x', 'y', 'z'}, default 'x'
        Longitudinal propagation axis.
    n_sub_w : int, default 4
        Number of Gauss-Legendre quadrature nodes along the longitudinal control cell
        to resolve smoothly varying tapers.
    dtype : data-type, optional
        Output NumPy dtype; defaults to `region.precision` (`float32` or `float64`).
    """
    from .solver import field_axes

    if normal not in ('x', 'y', 'z'):
        raise ValueError("normal must be 'x', 'y', or 'z'.")
    w_axis = 'xyz'.index(normal)
    u_axis, v_axis = (w_axis + 1) % 3, (w_axis + 2) % 3

    out_dtype = np.dtype(dtype if dtype is not None else (np.float64 if region.precision == 'float64' else np.float32))
    shape = tuple(region.shape)
    eps_3d = np.full(shape + (3,), float(eps_bg), dtype=out_dtype)

    if slabs_fn is None:
        if width_fn is None or z_bounds is None:
            raise ValueError('Provide either slabs_fn(w) or both width_fn and z_bounds.')

        def _default_slabs_fn(w_val: float):
            yc = float(center_y_fn(w_val) if callable(center_y_fn) else center_y_fn)
            w_core = float(width_fn(w_val) if callable(width_fn) else width_fn)
            rects = []
            if slab_width_fn is not None and slab_z_bounds is not None:
                w_slab = float(slab_width_fn(w_val) if callable(slab_width_fn) else slab_width_fn)
                if w_slab > 0.0 and slab_z_bounds[1] > slab_z_bounds[0]:
                    rects.append((yc - 0.5 * w_slab, yc + 0.5 * w_slab,
                                  float(slab_z_bounds[0]), float(slab_z_bounds[1]), float(eps_core)))
            if w_core > 0.0 and z_bounds[1] > z_bounds[0]:
                rects.append((yc - 0.5 * w_core, yc + 0.5 * w_core,
                              float(z_bounds[0]), float(z_bounds[1]), float(eps_core)))
            return rects

        slabs_fn = _default_slabs_fn

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(max(1, int(n_sub_w)))
    gl_nodes = 0.5 * gl_nodes
    gl_weights = 0.5 * gl_weights

    for comp_idx, comp_name in enumerate(('Ex', 'Ey', 'Ez')):
        axes_c = field_axes(region, comp_name)
        offsets = [0.5 if i == comp_idx else 0.0 for i in range(3)]
        widths_c = [
            _staggered_control_widths(region.mesh_nodes[i], offsets[i], shape[i])
            for i in range(3)
        ]

        w_coords = axes_c[w_axis][:shape[w_axis]]
        u_coords = axes_c[u_axis][:shape[u_axis]]
        v_coords = axes_c[v_axis][:shape[v_axis]]
        dw_arr = widths_c[w_axis]
        du_arr = widths_c[u_axis]
        dv_arr = widths_c[v_axis]

        uu, vv = np.meshgrid(u_coords, v_coords, indexing='ij')
        du_grid, dv_grid = np.meshgrid(du_arr, dv_arr, indexing='ij')

        if comp_idx == w_axis:
            role = 'w'
        elif comp_idx == u_axis:
            role = 'u'
        else:
            role = 'v'

        # Cache identical cross-sections along w (e.g., straight lead sections)
        cache: dict[tuple, np.ndarray] = {}
        comp_vol = np.empty((shape[w_axis], shape[u_axis], shape[v_axis]), dtype=np.float64)

        for iw, (w0, dw) in enumerate(zip(w_coords, dw_arr)):
            if x_bounds is not None and (w0 + 0.5 * dw < x_bounds[0] or w0 - 0.5 * dw > x_bounds[1]):
                comp_vol[iw] = float(eps_bg)
                continue
            slice_acc = np.zeros((shape[u_axis], shape[v_axis]), dtype=np.float64)
            for node, weight in zip(gl_nodes, gl_weights):
                wq = float(w0 + node * dw)
                if x_bounds is not None and not (x_bounds[0] <= wq <= x_bounds[1]):
                    slice_acc += weight * float(eps_bg)
                    continue
                raw_rects = slabs_fn(wq)
                norm_rects = tuple(_normalize_slabs(raw_rects, None, eps_core=eps_core))
                if norm_rects not in cache:
                    if all(r[0] == 'rect' for r in norm_rects):
                        cache[norm_rects] = _analytical_rect_conformal_variant0(
                            uu, vv, du_grid, dv_grid, norm_rects, float(eps_bg), role
                        )
                    else:
                        cache[norm_rects] = _subgrid_conformal_variant0(
                            uu, vv, du_grid, dv_grid, norm_rects, float(eps_bg), role, 64
                        )
                slice_acc += weight * cache[norm_rects]
            comp_vol[iw] = slice_acc

        # Permute (w, u, v) back to global (x, y, z)
        inv_perm = np.argsort((w_axis, u_axis, v_axis))
        eps_3d[..., comp_idx] = comp_vol.transpose(inv_perm).astype(out_dtype)

    return eps_3d


def build_safe_evanescent_graded_axis(
    domain_min: float,
    domain_max: float,
    si_min: float,
    si_max: float,
    d_fine: float,
    d_coarse: float,
    evanescent_buffer_um: float = 0.85,
    power_exponent: float = 3.0,
    max_stretch_ratio: float = 1.08,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a 1D non-uniform FDTD grid node array with a strictly uniform
    silicon + evanescent buffer core and a tangent-continuous (C^2 slow-start)
    power-law expansion in the outer cladding.

    Engineering Safety Guarantees:
      1. Uniform Fine Core + Evanescent Halo:
         Every point inside [si_min - evanescent_buffer_um, si_max + evanescent_buffer_um]
         is discretized with strictly uniform step ``d_fine`` (zero step-jump truncation
         error where >99.99% of the guided/evanescent modal energy resides).
      2. Zero-Slope Slow-Start Non-Linear Expansion:
         Outside the uniform buffer, cell width grows following a smooth power-law
         ``d(xi) = d_fine + (d_coarse - d_fine) * xi**p`` (default ``p = 3.0``),
         so both d(delta)/dy = 0 and d^2(delta)/dy^2 = 0 at the buffer boundary:
         expansion starts imperceptibly slowly near the buffer edge and accelerates
         rapidly past ~1 um toward ``d_coarse``, while strictly clamping adjacent cell
         ratios ``delta_{i+1} / delta_i <= max_stretch_ratio``.
    """
    if domain_max <= domain_min:
        raise ValueError("domain_max must exceed domain_min")
    if d_fine <= 0.0 or d_coarse < d_fine:
        raise ValueError("Require 0 < d_fine <= d_coarse")
    p = max(1.5, float(power_exponent))
    r_max = max(1.01, float(max_stretch_ratio))

    fine_lo = max(float(domain_min), float(si_min) - float(evanescent_buffer_um))
    fine_hi = min(float(domain_max), float(si_max) + float(evanescent_buffer_um))
    if fine_hi <= fine_lo:
        fine_lo = 0.5 * (domain_min + domain_max) - d_fine
        fine_hi = 0.5 * (domain_min + domain_max) + d_fine

    n_fine = max(2, int(round((fine_hi - fine_lo) / d_fine)))
    df_exact = (fine_hi - fine_lo) / n_fine
    core_nodes = np.linspace(fine_lo, fine_hi, n_fine + 1, dtype=np.float64)

    def _march_side(start_coord: float, target_coord: float, direction: float) -> list[float]:
        span = abs(target_coord - start_coord)
        if span <= 1e-12:
            return []
        pts: list[float] = []
        pos = 0.0
        prev_step = df_exact
        while pos < span - 1e-9:
            xi = min(1.0, max(0.0, pos / span))
            desired = df_exact + (d_coarse - df_exact) * (xi ** p)
            step = min(desired, prev_step * r_max, float(d_coarse))
            rem = span - pos
            if rem <= step * 1.35:
                pts.append(float(target_coord))
                break
            pos += step
            prev_step = step
            pts.append(float(start_coord + direction * pos))
        return pts

    left_pts = _march_side(fine_lo, domain_min, -1.0)[::-1]
    right_pts = _march_side(fine_hi, domain_max, +1.0)
    nodes = np.concatenate([
        np.asarray(left_pts, dtype=np.float64),
        core_nodes,
        np.asarray(right_pts, dtype=np.float64),
    ])
    diffs = np.diff(nodes)
    ratios = np.maximum(diffs[1:] / diffs[:-1], diffs[:-1] / diffs[1:]) if diffs.size > 1 else np.ones(1)
    n_uniform_equiv = int(round((domain_max - domain_min) / d_fine))
    meta = {
        "n_cells": int(diffs.size),
        "n_uniform_equiv": int(n_uniform_equiv),
        "cell_reduction_ratio": float(1.0 - diffs.size / max(1, n_uniform_equiv)),
        "fine_zone_min_um": float(fine_lo),
        "fine_zone_max_um": float(fine_hi),
        "min_step_um": float(diffs.min()),
        "max_step_um": float(diffs.max()),
        "max_adjacent_ratio": float(ratios.max()),
    }
    return nodes, meta

