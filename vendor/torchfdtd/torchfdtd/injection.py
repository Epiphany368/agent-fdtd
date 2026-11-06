"""Prepared soft-source terms and discrete normal-incidence Huygens planes.

The incident E/H pair comes from a scalar Yee line, not a continuum E/H
impedance guess. All work here occurs before the captured simulation loop.
"""
from functools import lru_cache
import math
import numpy as np

from .waveforms import source_quadrature_signal, source_time_signal
from .mesh import local_uniform_step


def oneway_plan(source, region):
    """Validate the supported plane geometry and return its staggered cut."""
    r, s = region, source
    axis = 'xyz'.index(s.normal)
    active = 2 if r.dimension == '2d' else 3
    if axis >= active:
        raise ValueError(f'{s.name}: one-way propagation must be along an active axis.')
    if any(face.kind != 'pml' for face in r.boundaries.pair(axis)):
        raise ValueError(f'{s.name}: the propagation axis requires PML at both ends.')
    if s.size[axis] != 0:
        raise ValueError(f'{s.name}: the one-way plane must have zero span along its normal.')
    for a in range(active):
        if a == axis: continue
        if any(face.kind not in ('periodic', 'bloch') for face in r.boundaries.pair(a)) or r.bloch_phase[a] != 0:
            raise ValueError(f'{s.name}: normal-incidence planes require periodic transverse boundaries (zero Bloch phase).')
        if s.center[a] != 0 or not math.isclose(s.size[a], r.actual_size[a], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{s.name}: the plane must cover the complete transverse unit cell and be centered on it.')
    nodes = r.mesh_nodes[axis]
    k = int(np.argmin(abs(nodes[:-1]-s.center[axis])))
    lo, hi = r.interior_bounds(axis)
    if k < 2 or k+2 >= len(nodes) or nodes[k-1] < lo or nodes[k+2] > hi:
        raise ValueError(f'{s.name}: leave at least two interior cells between the injection plane and PML.')
    local_uniform_step(r,axis,k-1,k+2)
    return axis, k, 1 if s.direction == '+' else -1


def validate_oneway_materials(project, epsilon, ownership):
    """Do not inject homogeneous incident fields through a different medium."""
    for s in project.sources:
        if not s.enabled or s.injection != 'oneway' or s.kind=='tfsf': continue
        axis, k, _ = oneway_plan(s, project.region)
        sl = [slice(None)]*3
        sl[axis] = slice(k-1, k+2)
        sl = tuple(sl)
        expected = project.region.background_index**2
        if not np.allclose(epsilon[sl], expected, rtol=1e-7 if epsilon.dtype == np.float32 else 1e-13, atol=0):
            raise ValueError(f'{s.name}: the injection neighborhood must match the homogeneous background index.')
        occupied = set(np.unique(ownership[sl]))
        if any(i in occupied and m.oscillators for i, m in enumerate(project.materials)):
            raise ValueError(f'{s.name}: dispersive materials cannot intersect the injection neighborhood.')


def oneway_metadata(source, region):
    axis,k,d=oneway_plan(source,region)
    h=k-1 if d==1 else k
    nodes=region.mesh_nodes[axis]
    return dict(source_id=source.id,normal=source.normal,direction=source.direction,
                electric_index=k,magnetic_index=h,electric_coordinate_um=float(nodes[k]),
                magnetic_coordinate_um=float((nodes[h]+nodes[h+1])/2),
                incident_drive_offset_cells=8,incident_pml_cells=source.incident_pml_cells,
                total_field_side='greater coordinate' if d==1 else 'lesser coordinate')


@lru_cache(maxsize=8)
def _incident_line(drive_bytes, courant, index, layers):
    """Finite scalar Yee line with thick CPML, in float64 on the host.

    Scalar h points along +normal cross electric polarization. The probe is
    eight cells downstream of the soft electric drive. Its delay and numerical
    dispersion are retained, including in the public preview.
    """
    drive = np.frombuffer(drive_bytes, dtype=np.float64)
    count, drive_at, probe = 2*layers+64, layers+24, layers+32
    e, h, pe, ph = (np.zeros(count) for _ in range(4))
    coefficients = []
    for forward in (False, True):
        j = np.arange(count, dtype=float)
        depth = np.maximum(layers-j-(1 if forward else .5), j-(count-layers)+(1 if forward else .5))
        rho = np.maximum(depth, 0)/(layers+1)
        sigma = 40*rho**3/(layers+1)
        # Match attenuation to the background wave speed. The two staggered
        # profiles are evaluated at their own derivative locations.
        decay = np.exp(-(sigma+1e-8)*courant/index)
        coupling = np.divide((decay-1)*sigma, sigma+1e-8)
        coefficients.append((decay, coupling))
    electric, magnetic = np.zeros(len(drive)), np.zeros(len(drive))
    for step, value in enumerate(drive):
        # E correction needs the old H sample. H correction needs the new E.
        electric[step] = courant/index**2*h[probe-1]
        diff = np.zeros(count)
        diff[1:] = h[1:]-h[:-1]
        pe *= coefficients[0][0]
        pe += coefficients[0][1]*diff
        e -= courant/index**2*(diff+pe)
        e[drive_at] += value
        magnetic[step] = courant*e[probe]
        diff[:-1] = e[1:]-e[:-1]
        diff[-1] = 0
        ph *= coefficients[1][0]
        ph += coefficients[1][1]*diff
        h -= courant*(diff+ph)
    electric.setflags(write=False)
    magnetic.setflags(write=False)
    return electric, magnetic


def oneway_tables(source, region):
    samples = source_time_signal(source, np.arange(1, region.steps+1)*region.time_step)
    axis,k,_=oneway_plan(source,region)
    courant = region.rectangular_courant*region.reference_step/local_uniform_step(region,axis,k-1,k+2)
    return _incident_line(np.asarray(samples, dtype=np.float64).tobytes(), courant, region.background_index, source.incident_pml_cells)


def source_terms(project, source):
    """Yield (field name, spatial slice, sampled increment, spatial profile)."""
    from .solver import source_slice, source_profile
    r, s = project.region, project.resolved_source(source)
    if not s.enabled: return
    if s.kind=='tfsf':return  # Live incident-line state, prepared by tfsf.py.
    if s.injection == 'oneway':
        axis, k, direction = oneway_plan(s, r)
        e_values, h_values = oneway_tables(s, r)
        e_loc, h_loc = [slice(None)]*3, [slice(None)]*3
        e_loc[axis] = slice(k, k+1)
        h_at = k-1 if direction == 1 else k
        h_loc[axis] = slice(h_at, h_at+1)
        for component, weight in s.polarization_components:
            transverse = 'xyz'.index(component[1].lower())
            magnetic = 3-axis-transverse
            cross_sign = 1 if (axis+1)%3 == transverse else -1
            yield component, tuple(e_loc), e_values*weight, None
            yield 'H'+'xyz'[magnetic], tuple(h_loc), h_values*(weight*direction*cross_sign), None
        return
    times = np.arange(1, r.steps+1)*r.time_step+s.time_offset_steps*r.time_step
    waveform = source_time_signal(s, times)
    # A Gaussian drive is (s+i*q)(t)*profile, split into two real-waveform
    # terms: s*profile and q*(i*profile). See source_quadrature_signal.
    quadrature = source_quadrature_signal(s, times) if s.gaussian is not None else None
    for component, weight in s.polarization_components:
        scalar = s.model_copy(update={'component':component, 'theta':None})
        loc = source_slice(scalar, r)
        profile = source_profile(scalar, loc, r)
        yield component, loc, waveform*weight, profile
        if quadrature is not None:
            yield component, loc, quadrature*weight, 1j*profile
