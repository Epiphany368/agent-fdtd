"""Morphological fabrication checks and binary export of a pixel design.

The measurements act on the binarized design, never on the filter radius:
a filter bounds the smoothness of the continuous density, but projection
and thresholding can still leave a one-pixel gap or line, so the minimum
linewidth and gap are measured here by morphological openings with digital
squares of every integer side. Lengths are micrometres; pixels are square.
"""
from dataclasses import dataclass
import math

import numpy as np

from .models import Structure

EDGE_TOLERANCE_UM = 1e-9   # widening of exported rectangles, 1e-6 of a nanometre


def _binary(design):
    import torch
    if isinstance(design, torch.Tensor):
        design = design.detach().cpu().numpy()
    array = np.asarray(design)
    if array.ndim != 2 or array.size == 0:
        raise ValueError('The design must be a nonempty two-dimensional array.')
    if array.dtype != bool:
        if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
            raise ValueError('The design must be binary: boolean, or exactly 0 and 1 values.')
        array = array.astype(bool)
    return array


def _spacing(spacing_um):
    spacing = (float(spacing_um),)*2 if isinstance(spacing_um, (int, float)) else tuple(float(v) for v in spacing_um)
    if len(spacing) != 2 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError('spacing_um must be one positive length or two positive lengths.')
    if not math.isclose(spacing[0], spacing[1], rel_tol=1e-9):
        raise ValueError('Morphological measurements use digital squares and require square pixels.')
    return spacing


def _boundary(boundary):
    modes = (boundary,)*2 if isinstance(boundary, str) else tuple(boundary)
    if len(modes) != 2 or any(mode not in ('periodic', 'extend') for mode in modes):
        raise ValueError("boundary must be 'periodic' or 'extend' per axis.")
    return modes


def square_offsets(side):
    """Integer offsets of the digital square of one pixel side.

    Odd sides are centred on a pixel, even sides on a pixel corner, so every
    integer side is a distinct structuring element and a rectangle of W by H
    pixels survives the opening with side s exactly when min(W, H) >= s. A
    digital disk would report right-angle corners as narrower than the
    rectangle they belong to, so squares are the measuring element here.
    """
    if isinstance(side, bool) or not isinstance(side, int) or side < 1:
        raise ValueError('side must be a positive integer number of pixels.')
    span = range(-(side//2), side//2+1) if side % 2 else range(-(side//2), side//2)
    return tuple((i, j) for i in span for j in span)


def _padded(array, reach, modes):
    width = ((reach, reach), (reach, reach))
    for axis, mode in enumerate(modes):
        pad = [(0, 0), (0, 0)]
        pad[axis] = width[axis]
        array = np.pad(array, pad, mode='wrap' if mode == 'periodic' else 'edge')
    return array


def _erode(array, offsets, modes):
    reach = max(max(abs(i), abs(j)) for i, j in offsets)
    padded = _padded(array, reach, modes)
    n, m = array.shape
    result = np.ones_like(array)
    for i, j in offsets:
        result &= padded[reach+i:reach+i+n, reach+j:reach+j+m]
    return result


def _dilate(array, offsets, modes):
    reach = max(max(abs(i), abs(j)) for i, j in offsets)
    padded = _padded(array, reach, modes)
    n, m = array.shape
    result = np.zeros_like(array)
    for i, j in offsets:
        result |= padded[reach-i:reach-i+n, reach-j:reach-j+m]
    return result


def morphological_open(binary, side, *, boundary='extend'):
    """Opening (erosion then dilation) by the digital square of the given pixel side."""
    array, modes, offsets = _binary(binary), _boundary(boundary), square_offsets(side)
    return _dilate(_erode(array, offsets, modes), offsets, modes)


@dataclass(frozen=True)
class FeatureSizes:
    """Measured minimum linewidth (solid) and gap (void) of a binary design.

    Each value is the largest square side whose opening leaves the solid
    (or the void) unchanged, in micrometres of pixel spacing; a feature of L
    pixels therefore measures exactly L pixels. `saturated` names the
    measurements that reached maximum_side_pixels without a change,
    which happens for a uniform design or one whose features exceed the
    search range; those values are lower bounds.
    """
    min_linewidth_um: float
    min_gap_um: float
    linewidth_pixels: int
    gap_pixels: int
    spacing_um: float
    maximum_side_pixels: int
    saturated: tuple
    solid_fraction: float

    def violations(self, *, min_linewidth_um=None, min_gap_um=None):
        """Names of the declared constraints the measured design violates."""
        found = []
        if min_linewidth_um is not None and self.min_linewidth_um < min_linewidth_um-1e-9:
            found.append('min_linewidth')
        if min_gap_um is not None and self.min_gap_um < min_gap_um-1e-9:
            found.append('min_gap')
        return tuple(found)

    def report(self):
        return dict(min_linewidth_um=self.min_linewidth_um, min_gap_um=self.min_gap_um,
                    linewidth_pixels=self.linewidth_pixels, gap_pixels=self.gap_pixels,
                    spacing_um=self.spacing_um, maximum_side_pixels=self.maximum_side_pixels,
                    saturated=list(self.saturated), solid_fraction=self.solid_fraction,
                    definition='largest digital-square side whose morphological opening leaves the solid (linewidth) or the void (gap) unchanged')


def _largest_unchanged_side(array, modes, maximum):
    for side in range(2, maximum+1):
        offsets = square_offsets(side)
        if not np.array_equal(_dilate(_erode(array, offsets, modes), offsets, modes), array):
            return side-1, False
    return maximum, True


def measure_feature_sizes(design, spacing_um, *, boundary='extend', maximum_side_pixels=None):
    """Minimum linewidth and gap of the binarized design by morphological opening.

    `design` is a boolean or 0/1 array on square pixels of `spacing_um`.
    `boundary` is 'periodic' or 'extend' per axis: a periodic axis wraps, an
    extended axis continues the edge pixel outward so features touching the
    edge of a bounded design region are not counted as narrow. A uniform
    axis of one pixel with 'extend' reduces the square to a line, so a
    one-dimensional design measures its run lengths exactly.
    """
    array, modes = _binary(design), _boundary(boundary)
    spacing = _spacing(spacing_um)[0]
    maximum = max(array.shape) if maximum_side_pixels is None else maximum_side_pixels
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('maximum_side_pixels must be a positive integer.')
    saturated = []
    if array.any():
        line, line_saturated = _largest_unchanged_side(array, modes, maximum)
    else:
        line, line_saturated = maximum, True
    if (~array).any():
        gap, gap_saturated = _largest_unchanged_side(~array, modes, maximum)
    else:
        gap, gap_saturated = maximum, True
    if line_saturated:
        saturated.append('min_linewidth')
    if gap_saturated:
        saturated.append('min_gap')
    return FeatureSizes(line*spacing, gap*spacing, line, gap, spacing, maximum, tuple(saturated),
                        float(array.mean()))


def fabrication_perturbation(design, amount_um, spacing_um, *, boundary='extend'):
    """Eroded and dilated copies of the binary design by a declared amount.

    The amount must be a whole number of pixels; erosion removes every solid
    pixel within that many pixels (Chebyshev distance) of the void and
    dilation adds every void pixel within it of the solid, both by the
    digital square of side 2 amount + 1. Returns (eroded, dilated, realized_um).
    """
    array, modes = _binary(design), _boundary(boundary)
    spacing = _spacing(spacing_um)[0]
    if not math.isfinite(amount_um) or amount_um <= 0:
        raise ValueError('amount_um must be positive and finite.')
    pixels = amount_um/spacing
    if not math.isclose(pixels, round(pixels), abs_tol=1e-6, rel_tol=0):
        raise ValueError('The perturbation amount must be a whole number of pixels.')
    radius = int(round(pixels))
    offsets = square_offsets(2*radius+1)
    return _erode(array, offsets, modes), _dilate(array, offsets, modes), radius*spacing


def binary_structures(design, *, origin_um, spacing_um, z_min_um, z_max_um, material, id_prefix='design'):
    """Native rectangles covering every solid pixel of the binary design.

    Pixel (i, j) spans [origin_um[0] + i h_x, origin_um[0] + (i+1) h_x] by the
    same in y. Solid runs along x are merged into one rectangle per run and
    vertically adjacent runs with equal x extent are merged, so the list is
    compact. Every rectangle is widened by EDGE_TOLERANCE_UM on each side so
    that a Yee node lying on a pixel edge is material whichever side it
    rounds to, as the importer's polygon boundary tolerance makes it; the
    widening is far below the GDS precision and vanishes on export.
    """
    array = _binary(design)
    spacing = (float(spacing_um),)*2 if isinstance(spacing_um, (int, float)) else tuple(float(v) for v in spacing_um)
    origin = tuple(float(v) for v in origin_um)
    if len(spacing) != 2 or any(not math.isfinite(v) or v <= 0 for v in spacing) or len(origin) != 2 or any(not math.isfinite(v) for v in origin):
        raise ValueError('origin_um and spacing_um need two finite values, spacings positive.')
    if not math.isfinite(z_min_um) or not math.isfinite(z_max_um) or z_min_um >= z_max_um:
        raise ValueError('z bounds must be finite with z_min_um < z_max_um.')
    if not isinstance(material, str) or not material:
        raise ValueError('Provide the material name of the solid pixels.')
    runs = {}  # (i0, i1) -> list of j
    for j in range(array.shape[1]):
        column = array[:, j]
        i = 0
        while i < len(column):
            if not column[i]:
                i += 1
                continue
            start = i
            while i < len(column) and column[i]:
                i += 1
            runs.setdefault((start, i), []).append(j)
    rectangles = []
    for (i0, i1), columns in sorted(runs.items()):
        j = 0
        while j < len(columns):
            start = j
            while j+1 < len(columns) and columns[j+1] == columns[j]+1:
                j += 1
            j0, j1 = columns[start], columns[j]+1
            j += 1
            rectangles.append((i0, i1, j0, j1))
    structures = []
    for index, (i0, i1, j0, j1) in enumerate(rectangles):
        x0, x1 = origin[0]+i0*spacing[0]-EDGE_TOLERANCE_UM, origin[0]+i1*spacing[0]+EDGE_TOLERANCE_UM
        y0, y1 = origin[1]+j0*spacing[1]-EDGE_TOLERANCE_UM, origin[1]+j1*spacing[1]+EDGE_TOLERANCE_UM
        structures.append(Structure(id=f'{id_prefix}-{index:04d}', name=f'{id_prefix} {index}', kind='rectangle',
                                    center=((x0+x1)/2, (y0+y1)/2, (z_min_um+z_max_um)/2),
                                    size=(x1-x0, y1-y0, z_max_um-z_min_um), material=material))
    return tuple(structures)
