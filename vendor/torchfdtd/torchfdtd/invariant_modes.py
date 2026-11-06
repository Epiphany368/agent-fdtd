"""True 1D transverse Yee eigenproblem for z-invariant XY propagation.

This is the kz=0 restriction of Maxwell's equations, not a thin 3D box.
There is one spatial difference matrix D; the invariant derivative is the
exact zero operator. The two uncoupled polarizations form a 2N by 2N
electric eigenproblem. Singleton axes below are field-storage conventions
only. Power is normalized with dy (or dx) in metres, never dy*dz.
"""
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from .mode_ports import WaveguideMode, _difference, _power_pair, _canonical_degenerate_basis


def validate_invariant_launch(region, source):
    """The native 2D domain is XY: no propagation/aperture cut along z."""
    if source.normal not in ('x', 'y'):
        raise ValueError('A 2D modal launch must propagate along x or y, never invariant z.')
    if source.center[2] != 0 or not math.isclose(source.size[2], region.actual_size[2],
                                               rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('The 2D source must be centered at z=0 and span the invariant storage axis.')


def solve_invariant_waveguide_modes(permittivity, *, shape, spacing_um, wavelength_um,
                                    origin_um, normal='x', num_modes=2,
                                    precision='float32', tolerance=None, maxiter=3000):
    """Solve only invariant-z modes; epsilon(u,v) must be independent of z.

    The cyclic transverse storage shape has exactly one invariant entry.
    Material invariance is checked at several z coordinates at each active
    Yee sample. Callers remain responsible for invariance outside these probes.
    No invariant spacing enters the operator or its power normalization.
    """
    if normal not in ('x', 'y') or precision not in ('float32', 'float64'):
        raise ValueError('Invariant-z modes require normal x/y and precision float32/float64.')
    w = 'xyz'.index(normal)
    u, v = (w+1)%3, (w+2)%3
    invariant = (u, v).index(2)
    active = 1-invariant
    if (len(shape) != 2 or any(type(n) is not int for n in shape)
            or shape[invariant] != 1 or shape[active] < 3):
        raise ValueError('A 1D transverse problem needs one invariant entry and at least three active cells.')
    if len(spacing_um) != 2 or not np.isfinite(spacing_um).all() or min(spacing_um) <= 0:
        raise ValueError('Two positive finite storage spacings required.')
    if len(origin_um) != 2 or not np.isfinite(origin_um).all():
        raise ValueError('Two finite origins required.')
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError('Positive finite wavelength required.')
    n = shape[active]
    if type(num_modes) is not int or not 1 <= num_modes < 2*n-1:
        raise ValueError('Mode count must be below the 1D electric eigensystem size minus one.')
    dtype = np.dtype(precision)
    cdtype = np.complex64 if precision == 'float32' else np.complex128
    tol = (2e-6 if precision == 'float32' else 1e-10) if tolerance is None else tolerance
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('Positive finite eigensolver tolerance required.')
    sampled = []
    for offsets in ((.5, 0.), (0., .5), (0., 0.)):
        x = origin_um[active]+(np.arange(n)+offsets[active])*spacing_um[active]
        def values_at(z):
            coords = (x, np.full(n, z)) if active == 0 else (np.full(n, z), x)
            raw = permittivity(*coords) if callable(permittivity) else permittivity
            if np.iscomplexobj(raw):
                raise ValueError('Only real positive isotropic permittivity supported.')
            values = np.broadcast_to(np.asarray(raw, dtype=dtype), (n,)).copy()
            if not np.isfinite(values).all() or values.min() <= 0:
                raise ValueError('Permittivity must be finite and positive.')
            return values
        values = values_at(0.)
        for z in (origin_um[invariant], -origin_um[invariant], spacing_um[invariant]/2):
            if not np.array_equal(values_at(z), values):
                raise ValueError('The 1D mode solve requires z-invariant permittivity.')
        sampled.append(values)
    eu, ev, ew = [sparse.diags(a, format='csr') for a in sampled]
    inverse_ew = sparse.diags(1/sampled[2], format='csr')
    dp, dm = _difference(n, spacing_um[active], dtype)
    zero = sparse.csr_matrix((n, n), dtype=dtype)
    up, um = (dp, dm) if active == 0 else (zero, zero)
    vp, vm = (dp, dm) if active == 1 else (zero, zero)
    ident = sparse.eye(n, dtype=dtype, format='csr')
    k = dtype.type(2*math.pi/wavelength_um)
    # Maxwell restriction: beta Et=P Ht, beta Ht=Q Et, d/dz=0 exactly.
    # For normal x (active u=y): PQ is diag(
    # (k I + D+ eps_w^-1 D-/k) k eps_u, k^2 eps_v + D- D+).
    p = sparse.bmat([[-up@inverse_ew@vm/k, k*ident+up@inverse_ew@um/k],
                     [-k*ident-vp@inverse_ew@vm/k, vp@inverse_ew@um/k]], format='csr')
    q = sparse.bmat([[um@vp/k, -k*ev-um@up/k],
                     [k*eu+vm@vp/k, -vm@up/k]], format='csr')
    operator = (p@q).astype(dtype)
    target = math.sqrt(max(float(a.max()) for a in sampled))*1.001
    sigma = dtype.type((target*float(k))**2)
    candidates = min(2*n-2, max(num_modes+4, 2*num_modes))
    start = np.random.default_rng(918).normal(size=2*n).astype(dtype)
    eigenvalues, vectors = eigs(operator, k=candidates, sigma=sigma, which='LM',
        tol=tol, maxiter=maxiter, v0=start, ncv=min(2*n, max(4*candidates+1, 40)))
    nearest = np.argsort(abs(eigenvalues-sigma))[:num_modes]
    order = nearest[np.argsort(-eigenvalues[nearest].real)]
    length_weight = spacing_um[active]*1e-6
    previous = []
    for index in order:
        value = eigenvalues[index]
        if value.real <= 0 or abs(value.imag) > max(20*tol, 1e-5)*abs(value.real):
            raise ValueError('Requested 1D modes contain evanescent/nonreal beta.')
        beta = math.sqrt(float(value.real))
        electric = vectors[:, index].astype(cdtype)
        transverse = np.r_[electric, q@electric/beta]
        for old_beta, old in previous:
            if abs(beta-old_beta) < max(10*tol, 1e-8)*max(beta, old_beta):
                transverse -= _power_pair(transverse, old, length_weight)*old
        power = float(_power_pair(transverse, transverse, length_weight).real)
        if not np.isfinite(power) or power <= np.finfo(dtype).tiny:
            raise ValueError('Mode has zero/negative power or dependent eigenvector.')
        transverse /= math.sqrt(power)
        pivot = transverse[:2*n][np.argmax(abs(transverse[:2*n]))]
        transverse *= np.exp(-1j*np.angle(pivot))
        previous.append((beta, transverse.copy()))
    modes = []
    for beta, transverse in _canonical_degenerate_basis(previous, n, length_weight, tol):
        elu, elv, hlu, hlv = transverse.reshape(4, n)
        elw = 1j*(inverse_ew@(um@hlv-vm@hlu))/k
        hlw = -1j*(up@elv-vp@elu)/k
        electric = np.r_[elu, elv]
        all_e, all_h = np.r_[elu, elv, elw], np.r_[hlu, hlv, hlw]
        curl_e = np.r_[vp@elw-1j*beta*elv, 1j*beta*elu-up@elw, up@elv-vp@elu]
        curl_h = np.r_[vm@hlw-1j*beta*hlv, 1j*beta*hlu-um@hlw, um@hlv-vm@hlu]
        epsilon_e = np.r_[sampled[0]*elu, sampled[1]*elv, sampled[2]*elw]
        residual = np.linalg.norm(operator@electric-beta**2*electric)/(beta**2*np.linalg.norm(electric))
        maxwell = max(np.linalg.norm(curl_e-1j*k*all_h)/(k*np.linalg.norm(all_h)),
                      np.linalg.norm(curl_h+1j*k*epsilon_e)/(k*np.linalg.norm(epsilon_e)))
        if max(residual, maxwell) > max(100*tol, 2e-4):
            raise RuntimeError('1D mode fails independent eigenpair/Maxwell residual check.')
        fields = np.empty((*shape, 6), dtype=cdtype)
        for local, axis in enumerate((u, v, w)):
            fields[..., axis] = all_e.reshape(3, *shape)[local]
            fields[..., 3+axis] = all_h.reshape(3, *shape)[local]
        fields.setflags(write=False)
        modes.append(WaveguideMode(beta, wavelength_um, normal, tuple(spacing_um), tuple(origin_um),
            fields, float(residual), float(maxwell), precision, invariant_axis='z'))
    return tuple(modes)


def validate_line_quadrature(plane, mode):
    """Admit only a complete uniform line with SI length weights, z=0."""
    expected_units = 'reduced E*H * s^2 * m per invariant length'
    if plane.flux_units != expected_units:
        raise ValueError('2D modal quadrature requires m per invariant length flux units.')
    points = plane.points_um.detach().cpu().numpy()
    weights = plane.weights.detach().cpu().numpy()
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),):
        raise ValueError('Mode line needs points (N,3) and weights (N,).')
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Mode line must have finite positive quadrature weights.')
    if not np.all(points[:, 2] == 0):
        raise ValueError('Invariant-z line quadrature must use z=0.')
    axis = next(a for a in mode.transverse_axes if a != 2)
    slot = mode.transverse_axes.index(axis)
    coordinates = np.unique(points[:, axis])
    count = len(coordinates)
    if count < 2 or count != len(points):
        raise ValueError('Mode line must be complete with no duplicate points.')
    length_um = mode.fields.shape[slot]*mode.spacing_um[slot]
    expected = mode.origin_um[slot]+(np.arange(count)+.5)*length_um/count
    if not np.allclose(coordinates, expected, rtol=0, atol=2e-6*length_um):
        raise ValueError('Mode line must cover its full aperture with uniform midpoints.')
    if not np.allclose(weights, length_um*1e-6/count, rtol=2e-5, atol=0):
        raise ValueError('2D modal quadrature needs SI length weights, never area weights.')
