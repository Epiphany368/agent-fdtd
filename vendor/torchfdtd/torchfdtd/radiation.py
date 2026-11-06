"""Differentiable continuum radiation transforms of collocated spectral planes.

These transforms use exp(-i omega t) phasors (the native positive-time DFT),
reduced E/H units, a homogeneous lossless isotropic exterior and mu_r=1.
They do not undo Yee dispersion or interpolation error. Surface/mesh convergence
is required when applying them to discrete FDTD data.
"""
from dataclasses import dataclass, replace
import math
import torch
from .adjoint_planes import DifferentiablePlaneResult, COMPONENTS

C0 = 299792458.0


def _real_metadata(value, like, name):
    raw = torch.as_tensor(value, device=like.device)
    if raw.is_complex():
        raise ValueError(f'{name} must be real for the lossless radiation contract.')
    # Preserve Python/NumPy double precision before casting to the requested
    # field precision. An existing tensor keeps its original precision.
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _plane(plane):
    if not isinstance(plane, DifferentiablePlaneResult):
        raise TypeError('Expected a DifferentiablePlaneResult.')
    f, p, w, e = plane.frequency_hz, plane.points_um, plane.weights, plane.fields
    if (plane.normal not in 'xyz' or len(plane.normal) != 1 or
            plane.components != COMPONENTS or plane.fourier_convention != 'exp(+2 pi i f t)' or
            plane.flux_units != 'reduced E*H * s^2 * m^2'):
        raise ValueError('Radiation requires native six-component 3D positive-time spectral planes.')
    if not e.is_complex() or e.dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Plane fields must be complex64 or complex128.')
    if f.ndim != 1 or not len(f) or p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError('Invalid frequencies or quadrature points.')
    if w.shape != (len(p),) or e.shape != (len(f), len(p), 6):
        raise ValueError('Plane field and quadrature shapes do not match.')
    if any(x.device != e.device or x.dtype != e.real.dtype for x in (f, p, w)):
        raise ValueError('Plane metadata dtype/device must match the real field dtype/device.')
    if any(x.requires_grad for x in (f, p, w)):
        raise ValueError('Frequency and quadrature geometry are fixed in radiation transforms.')
    if not all(bool(torch.isfinite(x).all()) for x in (f, p, w, e)) or bool((f <= 0).any()) or bool((w <= 0).any()):
        raise ValueError('Plane data must be finite with positive frequencies and area weights.')
    return 'xyz'.index(plane.normal)


def _rectangle(plane, bounds_um):
    """Validate full midpoint tensor-product quadrature, independent of order."""
    a = _plane(plane)
    bounds = _real_metadata(bounds_um, plane.points_um, 'Bounds')
    if bounds.requires_grad or bounds.shape != (3, 2) or not bool(torch.isfinite(bounds).all()):
        raise ValueError('Bounds must be a fixed finite (3, 2) array in micrometres.')
    p = plane.points_um
    tolerance = 32 * torch.finfo(p.dtype).eps * max(1., float(bounds.abs().max()))
    if float((p[:, a] - p[0, a]).abs().max()) > tolerance:
        raise ValueError('All quadrature points must lie in one plane.')
    lengths, counts = [], []
    for d in range(3):
        if d == a:
            continue
        lo, hi = bounds[d]
        length = float(hi - lo)
        values = torch.unique(p[:, d], sorted=True)
        if length <= 0 or len(values) < 2:
            raise ValueError('A radiation face needs positive transverse spans and at least two samples per axis.')
        expected = lo + (torch.arange(len(values), device=p.device, dtype=p.dtype) + .5) * (hi - lo) / len(values)
        if not torch.allclose(values, expected, rtol=0, atol=tolerance):
            raise ValueError('Quadrature must cover the full declared rectangle at uniform cell midpoints.')
        lengths.append(length)
        counts.append(len(values))
    if math.prod(counts) != len(p) or len(torch.unique(p, dim=0)) != len(p):
        raise ValueError('Quadrature must contain each tensor-product point exactly once.')
    area = math.prod(lengths) * 1e-12
    expected_weights = torch.full_like(plane.weights, area / len(p))
    if not torch.allclose(plane.weights, expected_weights, rtol=2e-5, atol=0):
        raise ValueError('Quadrature weights must be the full rectangle area in square metres.')
    return bounds, area


_GROWING = ('has a negative imaginary part: with exp(-i omega t) phasors that is a growing '
            'exterior, not a passive lossy one. Use a non-negative imaginary part.')


def _passive(value, name, count=None):
    """A fixed passive scalar, or with ``count`` a fixed (count,) per-frequency array.

    With exp(-i omega t) phasors a passive exterior has Im(n) >= 0 and
    Im(mu_r) >= 0, so exp(ikr) decays. A negative imaginary part describes a
    growing exterior and is rejected, elementwise for an array. A real scalar
    or one-element tensor is returned as float so the lossless scalar path is
    unchanged; an array is returned as a CPU float64 or complex128 tensor.
    """
    if isinstance(value, torch.Tensor) and value.requires_grad:
        raise ValueError(f'{name} is fixed metadata, not a design variable.')
    if isinstance(value, bool) or (isinstance(value, torch.Tensor) and value.dtype == torch.bool):
        raise ValueError(f'{name} must be a number.')
    if isinstance(value, (list, tuple)):
        # Python doubles stay exact; the default float32 tensor dtype would round them.
        array = torch.as_tensor(value, dtype=torch.complex128)
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().to(torch.complex128 if value.is_complex() else torch.float64)
    else:
        array = None
    if array is not None and array.numel() != 1:
        if count is None:
            raise ValueError(f'{name} must be one scalar.')
        if array.shape != (count,):
            raise ValueError(f'{name} must be one scalar or one value per plane frequency, shape ({count},).')
        z = array.real if array.is_complex() and not bool((array.imag != 0).any()) else array
        if not bool(torch.isfinite(z).all()) or bool((z.real <= 0).any()):
            raise ValueError(f'{name} must be finite with a positive real part.')
        if z.is_complex() and bool((z.imag < 0).any()):
            raise ValueError(f'{name} {_GROWING}')
        return z
    if array is not None:
        value = array.item()
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag) or z.real <= 0:
        raise ValueError(f'{name} must be finite with a positive real part.')
    if z.imag < 0:
        raise ValueError(f'{name} {_GROWING}')
    return z.real if z.imag == 0 else z


def _index(value):
    return _passive(value, 'Exterior refractive index')


def _exterior(refractive_index, relative_permeability, count=None):
    """Return (n, mu_r, admittance n/mu_r); the lossless scalar path keeps Python floats."""
    n = _passive(refractive_index, 'Exterior refractive index', count)
    mu = _passive(relative_permeability, 'Exterior relative permeability', count)
    epsilon = n**2 / mu
    if isinstance(epsilon, torch.Tensor):
        growing = epsilon.is_complex() and bool((epsilon.imag < 0).any())
    else:
        growing = isinstance(epsilon, complex) and epsilon.imag < 0
    if growing:
        raise ValueError('Exterior permittivity n^2/mu_r has a negative imaginary part; the exterior is not passive.')
    return n, mu, n / mu


def _broadcast(value, like, ndim):
    """Scalars pass through; a per-frequency tensor is cast to the field precision and shaped (F, 1, ...)."""
    if not isinstance(value, torch.Tensor):
        return value
    dtype = like.dtype if value.is_complex() else like.real.dtype
    return value.to(device=like.device, dtype=dtype).reshape((-1,) + (1,) * (ndim - 1))


def _reference_scale(reference, threshold):
    _plane(reference)
    if not 0 < threshold < 1:
        raise ValueError('Reference threshold must be between zero and one.')
    scale = reference.fields.detach().abs().amax()
    if not bool(scale > 0):
        raise ValueError('Reference power is zero or too weak.')
    normalized = replace(reference, fields=reference.fields / scale)
    power = normalized.flux().abs()
    if not bool(torch.isfinite(power).all()) or bool((power <= threshold * power.max()).any()):
        raise ValueError('Reference power is zero or too weak in a requested frequency band.')
    return scale, power


@dataclass
class DiffractionResult:
    orders: torch.Tensor
    wavevector_per_um: torch.Tensor  # F, order, xyz (positive normal branch)
    forward_fields: torch.Tensor  # F, order, 6, Fourier mean amplitudes
    backward_fields: torch.Tensor
    forward_power: torch.Tensor  # unsigned directional reduced power * s^2 * m^2
    backward_power: torch.Tensor
    propagating: torch.Tensor
    phase_origin_um: torch.Tensor


def diffraction_orders(plane, orders, *, period_um, refractive_index=1.,
                       bloch_wavevector_per_um=(0., 0.), cutoff_tolerance=1e-5):
    """Rayleigh orders, separated using both E and H, not E intensity alone.

    ``period_um`` and Bloch components follow cyclic transverse axes: y,z for
    x-normal, z,x for y-normal, and x,y for z-normal. The phase origin is the
    plane centre. Evanescent amplitudes are returned but carry zero directional
    real power. Orders at grazing cutoff are rejected because separation is
    singular. The caller must supply a full periodic unit cell in homogeneous
    material, outside PML. Sampled fields alone cannot verify that assumption.
    """
    a = _plane(plane)
    b, c = (a + 1) % 3, (a + 2) % 3
    n = _index(refractive_index)
    if isinstance(n, complex):
        raise ValueError('Diffraction orders require a real lossless exterior index.')
    period = _real_metadata(period_um, plane.points_um, 'Period')
    bloch = _real_metadata(bloch_wavevector_per_um, period, 'Bloch wavevector')
    raw_orders = torch.as_tensor(orders, device=period.device)
    if (period.shape != (2,) or bloch.shape != (2,) or period.requires_grad or bloch.requires_grad or
            not bool(torch.isfinite(period).all()) or not bool(torch.isfinite(bloch).all()) or bool((period <= 0).any())):
        raise ValueError('Period and Bloch wavevector must be fixed finite pairs, with positive periods.')
    if (raw_orders.ndim != 2 or raw_orders.shape[1] != 2 or not len(raw_orders) or raw_orders.is_complex() or raw_orders.dtype == torch.bool or
            raw_orders.requires_grad or not bool(torch.isfinite(raw_orders).all()) or
            not torch.equal(raw_orders, raw_orders.round()) or len(torch.unique(raw_orders, dim=0)) != len(raw_orders)):
        raise ValueError('Orders must be distinct integer pairs.')
    if not 0 < cutoff_tolerance < .01:
        raise ValueError('Cutoff tolerance must be between zero and .01.')
    orders = raw_orders.to(torch.int64)
    origin = plane.points_um.mean(0)
    bounds = torch.stack((origin.clone(), origin.clone()), dim=1)
    for d, length in zip((b, c), period):
        bounds[d, 0] -= length / 2
        bounds[d, 1] += length / 2
    _, area = _rectangle(plane, bounds)
    counts = torch.tensor([len(torch.unique(plane.points_um[:, d])) for d in (b, c)], device=period.device)
    if bool((2 * orders.abs() >= counts).any()):
        raise ValueError('Requested orders must lie strictly inside quadrature Nyquist limits.')
    transverse = bloch[None, :] + 2 * math.pi * orders.to(period.dtype) / period
    k0 = 2 * math.pi * plane.frequency_hz / C0 * 1e-6
    k2 = (n * k0).square()[:, None]
    kz2 = k2 - transverse.square().sum(-1)[None, :]
    # k_z^2 subtracts two comparable values. Its cutoff guard must also cover
    # roundoff in the supplied frequency and transverse phase metadata.
    cutoff_squared = max(cutoff_tolerance ** 2, 32 * torch.finfo(period.dtype).eps)
    if bool((kz2.abs() <= cutoff_squared * k2).any()):
        raise ValueError('A requested diffraction order is at grazing cutoff.')
    kz = kz2.to(plane.fields.dtype).sqrt()
    phase = torch.exp(-1j * (transverse @ (plane.points_um[:, (b, c)] - origin[(b, c),]).T))
    coefficients = torch.einsum('on,fnc,n->foc', phase, plane.fields, (plane.weights / area).to(plane.fields.dtype))
    e = coefficients[..., (b, c)]
    q = torch.stack((coefficients[..., 3 + c], -coefficients[..., 3 + b]), dim=-1)
    kt = transverse.to(plane.fields.dtype)[None, :, :]
    inverse_y_q = k0[:, None, None] / kz[..., None] * (q - kt * (kt * q).sum(-1, keepdim=True) / k2[..., None])
    forward_e = (e + inverse_y_q) / 2
    backward_e = (e - inverse_y_q) / 2
    propagating = kz2 > 0

    def reconstruct(et, sign):
        en = -(kt * et).sum(-1) / (sign * kz)
        components = [None] * 3
        components[a], components[b], components[c] = en, et[..., 0], et[..., 1]
        electric = torch.stack(components, -1)
        components[a] = sign * kz
        components[b] = kt[..., 0].expand_as(kz)
        components[c] = kt[..., 1].expand_as(kz)
        wavevector = torch.stack(components, -1)
        magnetic = torch.linalg.cross(wavevector, electric) / k0[:, None, None]
        power = sign * .5 * (electric[..., b] * magnetic[..., c].conj() - electric[..., c] * magnetic[..., b].conj()).real * area
        power = torch.where(propagating, power, torch.zeros_like(power))
        return torch.cat((electric, magnetic), -1), power, wavevector

    forward, pf, k = reconstruct(forward_e, 1)
    backward, pb, _ = reconstruct(backward_e, -1)
    return DiffractionResult(orders, k, forward, backward, pf, pb, propagating, origin)


def diffraction_efficiency(plane, reference, orders, *, direction='forward',
                           subtract_incident=False, min_reference_fraction=.01, **kwargs):
    """Order power / matched incident plane power, preserving both field graphs."""
    if direction not in ('forward', 'backward'):
        raise ValueError('Direction must be forward or backward.')
    if plane.run_signature != reference.run_signature or plane.normal != reference.normal:
        raise ValueError('Reference mesh, source, duration and normal must match.')
    for name in ('frequency_hz', 'points_um', 'weights'):
        x, y = getattr(plane, name), getattr(reference, name)
        if x.device != y.device or x.dtype != y.dtype or not torch.equal(x, y):
            raise ValueError(f'Reference {name} must match exactly.')
    scale, denominator = _reference_scale(reference, min_reference_fraction)
    field = plane.fields / scale
    if subtract_incident:
        field = field - reference.fields / scale
    result = diffraction_orders(replace(plane, fields=field), orders, **kwargs)
    return getattr(result, direction + '_power') / denominator[:, None]


_FACES = tuple(d + side for d in 'xyz' for side in ('_min', '_max'))


@dataclass
class FarFieldResult:
    electric_amplitude: torch.Tensor  # F, direction, xyz, E_spectral * metres
    directions: torch.Tensor
    frequency_hz: torch.Tensor
    refractive_index: float  # complex for a lossy exterior, (F,) tensor per frequency
    phase_origin_um: torch.Tensor
    relative_permeability: float = 1.
    approximation: str = None  # 'open surface' when faces were omitted
    surfaces: tuple = _FACES

    def intensity(self):
        """Reduced spectral power per steradian, not normalized efficiency.

        For a lossy exterior this is the source-referred pattern
        `.5 Re(n/mu_r) |A|^2`, without the exp(-2 Im(k) r) attenuation.
        """
        ratio = _broadcast(self.refractive_index / self.relative_permeability, self.electric_amplitude, 2)
        return .5 * ratio.real * self.electric_amplitude.abs().square().sum(-1)

    def fields_at_radius(self, radius_m):
        """Leading 1/r term only. Radius is measured from the phase origin.

        ``radius_m`` is one scalar or one fixed radius per direction, in metres.
        """
        device = self.electric_amplitude.device
        r = torch.as_tensor(radius_m, dtype=torch.float64, device=device)
        if (r.requires_grad or r.ndim > 1 or (r.ndim == 1 and r.shape != (len(self.directions),)) or
                not bool(torch.isfinite(r).all()) or bool((r <= 0).any())):
            raise ValueError('Radius must be finite and positive, one scalar or one fixed value per direction.')
        # A macroscopic radius spans millions of optical radians. Evaluate
        # this fixed phase in double precision, then retain the field
        # dtype. FP32 phase rounding would corrupt coherent fields at .5 m.
        k = 2 * math.pi * self.frequency_hz.double() / C0 * self.refractive_index
        propagation = (torch.exp(1j * k[:, None] * r) / r).to(self.electric_amplitude.dtype)
        e = self.electric_amplitude * propagation[:, :, None]
        admittance = _broadcast(self.refractive_index / self.relative_permeability, e, 3)
        h = admittance * torch.linalg.cross(self.directions.to(e.dtype)[None].expand_as(e), e)
        return torch.cat((e, h), -1)


def _closed_box(faces, bounds_um, refractive_index, relative_permeability=1., open_surface=False):
    """Validate named faces against one declared box and exterior; return the retained face names.

    With ``open_surface`` one to five named faces are admitted. Each name still
    fixes the outward normal and the face must coincide with that face of the
    declared box, so a single plane is ``{'z_max': plane}`` with outward +z.
    """
    if open_surface:
        names = tuple(name for name in _FACES if name in faces)
        if not names or len(names) == 6 or set(faces) - set(_FACES):
            raise ValueError('Open-surface projection takes one to five named box faces; use the closed box for all six.')
    else:
        names = _FACES
        if set(faces) != set(names):
            raise ValueError('Far-field projection requires all six named closed-box faces.')
    first = faces[names[0]]
    _plane(first)
    exterior = tuple(v.to(first.fields.device) if isinstance(v, torch.Tensor) else v
                     for v in _exterior(refractive_index, relative_permeability, len(first.frequency_hz)))
    device, dtype = first.fields.device, first.fields.real.dtype
    bound = _real_metadata(bounds_um, first.points_um, 'Bounds')
    if bound.shape != (3, 2) or not bool(torch.isfinite(bound).all()) or bool((bound[:, 1] <= bound[:, 0]).any()):
        raise ValueError('Closed-box bounds must have three positive finite spans.')
    for name in names:
        face = faces[name]
        a = _plane(face)
        if 'xyz'[a] != name[0] or face.run_signature != first.run_signature:
            raise ValueError('Surface face normal or run signature does not match.')
        if face.fields.dtype != first.fields.dtype or face.fields.device != device or not torch.equal(face.frequency_hz, first.frequency_hz):
            raise ValueError('All surface frequencies, dtypes and devices must match.')
        _rectangle(face, bound)
        side = 0 if name.endswith('min') else 1
        tolerance = 32 * torch.finfo(dtype).eps * max(1., float(bound.abs().max()))
        if not torch.allclose(face.points_um[:, a], bound[a, side].expand(len(face.points_um)), rtol=0, atol=tolerance):
            raise ValueError('Surface plane does not coincide with its declared closed-box face.')
    return names, first, exterior, bound


_EDGE_TAPER_DECAY = 15.  # aperture-edge amplitude exp(-.5 * 15) ~ 5e-4


def _face_weights(face, name, bound, edge_window):
    """Quadrature weights, optionally tapered by a Gaussian edge window on one open plane.

    ``edge_window`` holds the fractional width of the tapered region over both
    edges of each cyclic transverse axis. The centre keeps unit weight.
    """
    window = torch.as_tensor(edge_window, dtype=torch.float64)
    if window.shape != (2,) or not bool(torch.isfinite(window).all()) or bool((window < 0).any()) or bool((window > 1).any()):
        raise ValueError('edge_window must hold two fractions in [0, 1].')
    if not bool((window > 0).any()):
        return face.weights
    weights = face.weights
    a = 'xyz'.index(name[0])
    for fraction, d in zip(window.tolist(), ((a + 1) % 3, (a + 2) % 3)):
        if fraction <= 0:
            continue
        lo, hi = bound[d].double()
        transition = fraction * float(hi - lo) / 2
        x = face.points_um[:, d].double()
        lower, upper = lo + transition, hi - transition
        taper = torch.ones_like(x)
        taper = torch.where(x < lower, torch.exp(-.5 * _EDGE_TAPER_DECAY * ((x - lower) / transition).square()), taper)
        taper = torch.where(x > upper, torch.exp(-.5 * _EDGE_TAPER_DECAY * ((x - upper) / transition).square()), taper)
        weights = weights * taper.to(weights.dtype)
    return weights


def _window(edge_window, names):
    if len(names) > 1 and bool((torch.as_tensor(edge_window, dtype=torch.float64) != 0).any()):
        raise ValueError('An edge window is only defined for a single open plane, not for several faces.')


def _observation_points(points_um, like, bound):
    """Fixed finite (P, 3) micrometre points strictly outside the closed box."""
    points = _real_metadata(points_um, like.points_um, 'Observation points')
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or points.requires_grad or
            not bool(torch.isfinite(points).all())):
        raise ValueError('Observation points must be a fixed finite (P, 3) array in micrometres.')
    tolerance = 32 * torch.finfo(points.dtype).eps * max(1., float(bound.abs().max()))
    inside = ((points >= bound[:, 0] - tolerance) & (points <= bound[:, 1] + tolerance)).all(-1)
    if bool(inside.any()):
        raise ValueError('Observation points must lie strictly outside the closed box.')
    return points


def _chunks(*sizes):
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in sizes):
        raise ValueError('Projection chunk sizes must be positive integers.')


def spherical_directions(theta_rad, phi_rad):
    """Unit vectors on the tensor-product theta x phi grid, theta varying slowest.

    Theta is measured from +z in [0, pi] and phi from +x toward +y, in radians.
    """
    theta = torch.as_tensor(theta_rad, dtype=torch.float64).reshape(-1)
    phi = torch.as_tensor(phi_rad, dtype=torch.float64).reshape(-1)
    # Single-precision pi exceeds math.pi by one FP32 ulp; admit that roundoff.
    slack = 4 * torch.finfo(torch.float32).eps
    if (not len(theta) or not len(phi) or not bool(torch.isfinite(theta).all()) or not bool(torch.isfinite(phi).all()) or
            bool((theta < -slack).any()) or bool((theta > math.pi + slack).any())):
        raise ValueError('Theta must be finite radians in [0, pi] and phi finite radians, both nonempty.')
    t, p = torch.meshgrid(theta.clamp(0., math.pi), phi, indexing='ij')
    return torch.stack((torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)), -1).reshape(-1, 3)


def kspace_directions(ux, uy, axis='z'):
    """Unit directions on the tensor-product grid of direction cosines, ``ux`` varying slowest.

    ``axis`` names the hemisphere: 'z' (or '+z') has a non-negative z
    component and '-z' a non-positive one, likewise for x and y. ``ux`` and
    ``uy`` are the cosines along the cyclic transverse axes of that axis
    (y,z for x; z,x for y; x,y for z). Points with ``ux^2 + uy^2 > 1`` are
    evanescent, not directions, and are rejected.
    """
    sign = -1. if str(axis).startswith('-') else 1.
    name = str(axis).lstrip('+-')
    if name not in ('x', 'y', 'z') or len(str(axis)) > 2:
        raise ValueError("axis must be one of x, y, z, optionally signed, such as '-z'.")
    a = 'xyz'.index(name)
    u = torch.as_tensor(ux, dtype=torch.float64).reshape(-1)
    v = torch.as_tensor(uy, dtype=torch.float64).reshape(-1)
    if not len(u) or not len(v) or not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError('Direction cosines must be finite and nonempty.')
    ug, vg = torch.meshgrid(u, v, indexing='ij')
    transverse = ug.square() + vg.square()
    if bool((transverse > 1).any()):
        raise ValueError('ux^2 + uy^2 must not exceed 1: such points are evanescent, not propagating directions.')
    directions = torch.empty(ug.shape + (3,), dtype=torch.float64)
    directions[..., a] = sign * torch.sqrt((1 - transverse).clamp(min=0.))
    directions[..., (a + 1) % 3] = ug
    directions[..., (a + 2) % 3] = vg
    return directions.reshape(-1, 3)


def _origin(origin_um):
    origin = torch.as_tensor(origin_um, dtype=torch.float64)
    if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
        raise ValueError('Origin must be a finite point in micrometres.')
    return origin


def spherical_points(theta_rad, phi_rad, radius_um, origin_um=(0., 0., 0.)):
    """Cartesian micrometre points at one radius on the spherical grid around ``origin_um``."""
    r = float(radius_um)
    if not math.isfinite(r) or r <= 0:
        raise ValueError('Radius must be finite and positive.')
    return _origin(origin_um) + r * spherical_directions(theta_rad, phi_rad)


def cartesian_plane_points(normal, offset_um, u_um, v_um, origin_um=(0., 0., 0.)):
    """Points of a u x v grid on the plane ``offset_um`` along ``normal`` from ``origin_um``.

    ``u`` and ``v`` follow the cyclic transverse axes used by the plane APIs:
    y,z for an x-normal plane, z,x for y-normal and x,y for z-normal. ``u``
    varies slowest. Coordinates are micrometres relative to ``origin_um``.
    """
    if normal not in ('x', 'y', 'z'):
        raise ValueError('Plane normal must be x, y or z.')
    a = 'xyz'.index(normal)
    origin = _origin(origin_um)
    offset = float(offset_um)
    u = torch.as_tensor(u_um, dtype=torch.float64).reshape(-1)
    v = torch.as_tensor(v_um, dtype=torch.float64).reshape(-1)
    if (not math.isfinite(offset) or not len(u) or not len(v) or
            not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(v).all())):
        raise ValueError('Plane offset and transverse coordinates must be finite and nonempty.')
    ug, vg = torch.meshgrid(u, v, indexing='ij')
    points = torch.empty((ug.numel(), 3), dtype=torch.float64)
    points[:, a] = origin[a] + offset
    points[:, (a + 1) % 3] = origin[(a + 1) % 3] + ug.reshape(-1)
    points[:, (a + 2) % 3] = origin[(a + 2) % 3] + vg.reshape(-1)
    return points


def farfield_at_points(faces, points_um, *, bounds_um, refractive_index=1., relative_permeability=1.,
                       phase_origin_um=(0., 0., 0.), open_surface=False, **kwargs):
    """Far-field model at Cartesian points: leading 1/r term along each point's direction.

    Directions and radii are measured from ``phase_origin_um``. Returns (F, P, 6)
    E/H in the plane field units. Points must lie strictly outside the box.
    """
    names, first, _, bound = _closed_box(faces, bounds_um, refractive_index, relative_permeability, open_surface)
    points = _observation_points(points_um, first, bound)
    origin = _real_metadata(phase_origin_um, first.points_um, 'Phase origin')
    if origin.shape != (3,) or origin.requires_grad or not bool(torch.isfinite(origin).all()):
        raise ValueError('Phase origin must be a fixed finite point in micrometres.')
    offset = (points - origin).double()
    radius = offset.norm(dim=-1)
    if bool((radius <= 0).any()):
        raise ValueError('Far-field observation points must not coincide with the phase origin.')
    far = project_farfield(faces, offset / radius[:, None], bounds_um=bounds_um, refractive_index=refractive_index,
                           relative_permeability=relative_permeability, phase_origin_um=phase_origin_um,
                           open_surface=open_surface, **kwargs)
    return far.fields_at_radius(radius * 1e-6)


@dataclass
class NearZoneResult:
    fields: torch.Tensor  # F, point, Ex..Hz in the plane field units (reduced field * s)
    points_um: torch.Tensor
    frequency_hz: torch.Tensor
    refractive_index: float  # complex for a lossy exterior, (F,) tensor per frequency
    relative_permeability: float = 1.
    approximation: str = None  # 'open surface' when faces were omitted
    surfaces: tuple = _FACES

    def poynting(self):
        """Time-averaged Poynting vector .5 Re(E x H*), reduced E*H * s^2 per point."""
        e, h = self.fields[..., :3], self.fields[..., 3:]
        return .5 * torch.linalg.cross(e, h.conj()).real


def project_nearzone(faces, points_um, *, bounds_um, refractive_index=1., relative_permeability=1.,
                     open_surface=False, edge_window=(0., 0.), point_chunk=2048, observation_chunk=64):
    """Exact homogeneous Green-function fields of the box currents at finite distance.

    With ``A = int J g dS``, ``F = int M g dS``, ``g = exp(ikR)/(4 pi R)``,
    ``J = outward x H`` and ``M = -outward x E`` in reduced units:
    ``E = i k0 mu_r [A + grad div A / k^2] - curl F`` and
    ``H = i k0 eps_r [F + grad div F / k^2] + curl A`` with ``k = k0 n`` and
    ``eps_r = n^2 / mu_r``. The gradient and curl act on the Green function,
    so near-, intermediate- and far-zone terms are all retained and E gains a
    radial component. A complex passive ``n`` or ``mu_r`` gives a complex ``k``.
    Points must lie strictly outside the box: inside, the same integral
    returns the negative field of exterior sources, not the interior field.
    The Green function and its derivatives are evaluated in double precision
    before casting to the field dtype, so macroscopic radii keep a coherent
    phase. ``open_surface`` admits one to five faces as an approximation.
    """
    names, first, (n, mu, _), bound = _closed_box(faces, bounds_um, refractive_index, relative_permeability, open_surface)
    _window(edge_window, names)
    points = _observation_points(points_um, first, bound)
    _chunks(point_chunk, observation_chunk)
    device, cdtype, rdtype = first.fields.device, first.fields.dtype, first.fields.real.dtype
    k0 = 2 * math.pi * first.frequency_hz.double() / C0
    k = n * k0
    epsilon = _broadcast(n**2 / mu, first.fields, 4)
    permeability = _broadcast(mu, first.fields, 4)
    ik0 = (1j * k0).to(cdtype)[:, None, None, None]
    scale = 1 / k.square()
    inverse_k2 = scale.to(cdtype if scale.is_complex() else rdtype)[:, None, None, None]
    output = []
    for start in range(0, len(points), observation_chunk):
        observed = points[start:start + observation_chunk].double() * 1e-6
        e = torch.zeros((len(k), len(observed), 3), device=device, dtype=cdtype)
        h = torch.zeros_like(e)
        for name in names:
            face = faces[name]
            normal = torch.zeros(3, device=device, dtype=rdtype)
            normal['xyz'.index(name[0])] = -1 if name.endswith('min') else 1
            weights = _face_weights(face, name, bound, edge_window)
            for offset in range(0, len(face.points_um), point_chunk):
                section = slice(offset, offset + point_chunk)
                fields = face.fields[:, section]
                outward = normal.to(cdtype).expand_as(fields[..., :3])
                weight = weights[section][None, :, None]
                j = (torch.linalg.cross(outward, fields[..., 3:]) * weight)[:, None]
                m = (-torch.linalg.cross(outward, fields[..., :3]) * weight)[:, None]
                displacement = observed[:, None, :] - (face.points_um[section].double() * 1e-6)[None]
                r = displacement.norm(dim=-1)
                rhat = (displacement / r[..., None]).to(rdtype)[None]
                ikr = 1j * k[:, None, None] * r[None]
                g = torch.exp(ikr) / (4 * math.pi * r[None])
                g1 = g * (ikr - 1) / r[None]
                g2 = g1 * (ikr - 1) / r[None] + g / r[None].square()
                g, g1, g1r, g2 = [x.to(cdtype)[..., None] for x in (g, g1, g1 / r[None], g2)]
                rj = (rhat * j).sum(-1, keepdim=True)
                rm = (rhat * m).sum(-1, keepdim=True)
                potential_a = g * j + (g2 * rhat * rj + g1r * (j - rhat * rj)) * inverse_k2
                potential_f = g * m + (g2 * rhat * rm + g1r * (m - rhat * rm)) * inverse_k2
                complex_rhat = rhat.to(cdtype)
                curl_a = g1 * torch.linalg.cross(*torch.broadcast_tensors(complex_rhat, j))
                curl_f = g1 * torch.linalg.cross(*torch.broadcast_tensors(complex_rhat, m))
                e = e + (ik0 * permeability * potential_a - curl_f).sum(2)
                h = h + (ik0 * epsilon * potential_f + curl_a).sum(2)
        output.append(torch.cat((e, h), -1))
    return NearZoneResult(torch.cat(output, dim=1), points, first.frequency_hz, n, mu,
                          'open surface' if open_surface else None, names)


def project_farfield(faces, directions, *, bounds_um, refractive_index=1., relative_permeability=1.,
                     phase_origin_um=(0., 0., 0.), open_surface=False, edge_window=(0., 0.),
                     direction_chunk=16, point_chunk=2048):
    """Closed-box vector equivalence-current integral into arbitrary directions.

    ``faces`` maps x_min/x_max/y_min/y_max/z_min/z_max to collocated spectral
    planes with positive-axis component conventions. Outward signs are supplied
    here. The box must enclose all scatterers/sources of the supplied fields and
    lie in the declared homogeneous exterior, outside PML. For scattering,
    subtract matched incident fields on ALL six faces before projection.
    A periodic cell or a box intersecting a substrate is not a valid
    isolated-object closed surface for this API. ``open_surface=True`` admits a
    subset of the named faces (a single plane is one named face) as the
    documented open-surface approximation, flagged in the result. A complex
    passive ``refractive_index`` or ``relative_permeability`` gives a lossy
    exterior with complex wavenumber and impedance.
    """
    names, first, (n, mu, admittance), bound = _closed_box(faces, bounds_um, refractive_index, relative_permeability, open_surface)
    _window(edge_window, names)
    device, dtype = first.fields.device, first.fields.real.dtype
    direction = _real_metadata(directions, first.points_um, 'Directions')
    origin = _real_metadata(phase_origin_um, first.points_um, 'Phase origin')
    if (direction.ndim != 2 or direction.shape[1] != 3 or not len(direction) or direction.requires_grad or
            not bool(torch.isfinite(direction).all()) or
            not torch.allclose(direction.norm(dim=-1), torch.ones(len(direction), device=device, dtype=dtype), rtol=2e-5, atol=2e-6)):
        raise ValueError('Directions must be fixed finite unit vectors with shape (N, 3).')
    if origin.shape != (3,) or origin.requires_grad or not bool(torch.isfinite(origin).all()):
        raise ValueError('Phase origin must be a fixed finite point in micrometres.')
    _chunks(direction_chunk, point_chunk)
    k = 2 * math.pi * first.frequency_hz / C0 * n
    if isinstance(n, torch.Tensor):
        k = k.to(first.fields.dtype if k.is_complex() else dtype)
    admittance = _broadcast(admittance, first.fields, 3)
    output = []
    for start in range(0, len(direction), direction_chunk):
        s = direction[start:start + direction_chunk]
        integral = torch.zeros((len(k), len(s), 3), device=device, dtype=first.fields.dtype)
        for name in names:
            face = faces[name]
            normal = torch.zeros(3, device=device, dtype=dtype)
            normal['xyz'.index(name[0])] = -1 if name.endswith('min') else 1
            weights = _face_weights(face, name, bound, edge_window)
            for offset in range(0, len(face.points_um), point_chunk):
                section = slice(offset, offset + point_chunk)
                fields = face.fields[:, section]
                outward = normal.to(fields.dtype).expand_as(fields[..., :3])
                j = torch.linalg.cross(outward, fields[..., 3:])
                m = -torch.linalg.cross(outward, fields[..., :3])
                phase = torch.exp(-1j * k[:, None, None] * (s @ ((face.points_um[section] - origin) * 1e-6).T)[None])
                factor = phase * weights[section]
                ji = torch.einsum('fdp,fpc->fdc', factor, j)
                mi = torch.einsum('fdp,fpc->fdc', factor, m)
                sc = s.to(fields.dtype)[None].expand_as(ji)
                integral = integral + (ji - sc * (sc * ji).sum(-1, keepdim=True)) / admittance - torch.linalg.cross(sc, mi)
        output.append(1j * k[:, None, None] / (4 * math.pi) * integral)
    return FarFieldResult(torch.cat(output, dim=1), direction, first.frequency_hz, n, origin, mu,
                          'open surface' if open_surface else None, names)


def normalized_farfield_intensity(faces, reference, directions, *, min_reference_fraction=.01, **kwargs):
    """Closed-surface differential power / incident plane power, per steradian."""
    scale, denominator = _reference_scale(reference, min_reference_fraction)
    for face in faces.values():
        if (face.run_signature != reference.run_signature or face.fields.device != reference.fields.device or
                face.fields.dtype != reference.fields.dtype or not torch.equal(face.frequency_hz, reference.frequency_hz)):
            raise ValueError('Incident reference and surface run signatures/frequencies/dtypes/devices must match.')
    normalized = {name: replace(face, fields=face.fields / scale) for name, face in faces.items()}
    result = project_farfield(normalized, directions, **kwargs)
    return result.intensity() / denominator[:, None]
