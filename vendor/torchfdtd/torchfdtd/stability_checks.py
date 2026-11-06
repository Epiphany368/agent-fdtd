"""Validation-time warnings for configurations known to be unstable in long runs.

These checks read the project only; they change nothing and reject nothing.
/api/validate appends them to the estimate's warnings and `torchfdtd run`
prints them with the run summary.
"""
from __future__ import annotations

import numpy as np


def _layer_samples(region, axis, side):
    """(innermost, outermost) position of the material samples in the PML rows of a face, the rows 'frozen' freezes.

    Yee sampling puts E on the nodes and half nodes of the axis, so the node on the
    inner edge of an upper layer is in its rows and the one of a lower layer is
    not; cell sampling puts every component at the cell centre.
    """
    nodes = np.asarray(region.mesh_nodes[axis], dtype=np.float64)
    n, layers, half = region.shape[axis], region.pml_layers(axis, side), (nodes[:-1]+nodes[1:])/2
    yee = region.material_sampling == 'yee'
    return (half[layers-1], nodes[0] if yee else half[0]) if side == 0 else (nodes[n-layers] if yee else half[n-layers], half[n-1])


def _pml_contacts(project):
    """(structure, [(face, ends inside the layer)]) of every enabled dispersive structure with a material sample in a PML layer.

    The test is the support bounds against the sample positions of the layer rows,
    within 1e-6 of a cell. The structure ends inside the layer when either of its
    ends along the face normal lies there, so that it misses the innermost or the
    outermost row of samples; otherwise it crosses the layer.
    """
    from .mesh import object_bounds
    r = project.region
    dispersive = {m.name for m in project.materials if m.oscillators}
    active = 2 if r.dimension == '2d' else 3
    tolerance = 1e-6*r.reference_step
    rows = {(axis, side): _layer_samples(r, axis, side) for axis in range(active) for side in (0, 1) if r.pml_layers(axis, side)}
    found = []
    for s in project.structures:
        if not s.enabled or s.material not in dispersive:
            continue
        center, size = object_bounds(s)
        faces = []
        for (axis, side), (inner, outer) in rows.items():
            lo, hi = center[axis]-size[axis]/2, center[axis]+size[axis]/2
            first, last = sorted((inner, outer))
            if lo <= last+tolerance and hi >= first-tolerance:
                faces.append(('xyz'[axis]+('_max' if side else '_min'), lo > first+tolerance or hi < last-tolerance))
        if faces:
            found.append((s, faces))
    return found


def dispersive_structures_in_pml(project):
    """(structure name, PML faces) of every enabled dispersive structure with a material sample in a PML layer."""
    return [(s.name, [face for face, _ in faces]) for s, faces in _pml_contacts(project)]


def negative_permittivity_in_band(material, dt):
    """Whether the discrete (bilinear) Re eps of the material turns negative anywhere below the Nyquist frequency."""
    from .materials import permittivity
    frequencies = np.logspace(-6, np.log10(.4999), 4001)/dt
    return bool(np.any(permittivity(material, frequencies, dt).real < 0))


def stability_warnings(project):
    """Warnings about configurations the long-run stability evidence marks as unsafe (docs/BOUNDARIES.md)."""
    warnings = []
    r = project.region
    contacts = _pml_contacts(project) if r.pml_dispersion == 'ade' else []
    # One band test per material, not per structure: a metalens tile names thousands of posts.
    used = {s.material for s, _ in contacts}
    negative = {m.name: negative_permittivity_in_band(m, r.time_step) for m in project.materials if m.name in used}
    ending, crossing, positive = [], [], []
    for s, faces in contacts:
        named = f'{s.name} ({", ".join(face for face, _ in faces)})'
        (positive if not negative[s.material] else ending if any(ends for _, ends in faces) else crossing).append(named)
    if ending:
        warnings.append(f'Dispersive material inside PML layers, ending there: {"; ".join(ending)}. A pole whose Re eps turns '
                        'negative in the band of the grid, in a structure with an end inside a CPML layer, is the configuration that '
                        'diverged in the records (a post in the outer cells of a layer corner grows by e^0.057 per step, a post '
                        'entering the corner from the interior by e^0.10, docs/BOUNDARIES.md); set region.pml_dispersion="absorber", '
                        'whose crossed faces reflect about 1e-2 to 1 at oblique incidence and across transverse interfaces, or end the '
                        'structure before the PML.')
    if crossing:
        warnings.append(f'Dispersive material inside PML layers, crossing them: {"; ".join(crossing)}. Re eps of these materials '
                        'turns negative in the band of the grid, and the CPML is not known to be stable around them: a Drude bar '
                        'crossing a layer grew by e^0.0011 per step in the records (docs/BOUNDARIES.md), slowly enough to pass the '
                        'run-control growth limit for thousands of steps. pml_dispersion="absorber" stays stable there but reflects '
                        '0.016 to 1.2 per monitor in the G3-07 half space, against 4e-9 to 7e-6 for the CPML; choose it where that '
                        'reflection is acceptable, keep "ade" and watch the state norm to the last step, or end the structure '
                        'before the PML.')
    if positive:
        warnings.append(f'Dispersive material inside PML layers: {"; ".join(positive)}. Re eps of these materials stays positive on '
                        'the whole band of the grid, so the negative-permittivity growth of the records has no band here, and the '
                        'CPML reflects far less than pml_dispersion="absorber" (4e-9 to 7e-6 per monitor in the G3-07 half space, '
                        'against 0.016 to 1.2); keep pml_dispersion="ade" with the run-control divergence check, or end the '
                        'structure before the PML.')
    return warnings
