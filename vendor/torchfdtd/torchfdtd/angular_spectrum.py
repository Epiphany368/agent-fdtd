"""Angular-spectrum propagation of a uniform-mesh output plane through a homogeneous exterior.

The plane is zero padded, transformed once per component and frequency, and
every Cartesian component of E and H is multiplied by the free-space transfer
function ``exp(i k_n z)`` with ``k_n = sqrt(k^2 - ku^2 - kv^2)`` (imaginary for
evanescent waves, which decay as ``exp(-|k_n| z)``). Sections, volumes and
arbitrary points are extracted from the same spectrum. Only outgoing waves on
the device side are represented: the exterior beyond the plane must be
homogeneous, lossless and source free. The plane spacing sets the largest
propagating angle it can hold (``sin(theta) = wavelength / (2 spacing)``) and
the zero padding keeps light that leaves the aperture from wrapping around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .adjoint_planes import DifferentiablePlaneResult

C0 = 299792458.0


@dataclass
class PlaneGrid:
    """A uniform-mesh plane as ``(F, Nu, Nv, C)`` fields on its two transverse axes."""
    fields: torch.Tensor
    u_um: torch.Tensor
    v_um: torch.Tensor
    spacing_um: tuple
    normal: str
    offset_um: float
    direction: int
    frequency_hz: torch.Tensor
    components: tuple
    dimension: str

    @property
    def axes(self):
        return tuple(c for c in 'xyz' if c != self.normal)


def _uniform(coordinates, name):
    values = (coordinates.detach().to(torch.float64) if isinstance(coordinates, torch.Tensor)
              else torch.as_tensor(np.asarray(coordinates, dtype=np.float64))).reshape(-1)
    if len(values) < 2:
        return values, None
    steps = values[1:] - values[:-1]
    spacing = float(steps.mean())
    # A DifferentiablePlaneResult stores its points in the field precision, so
    # float32 coordinates tens of micrometres from the origin carry rounding of
    # a few 1e-6 um; accept that as uniform rather than only 1e-6 of the spacing.
    tolerance = max(1e-6 * spacing, 2. ** -20 * float(values.abs().max()))
    if spacing <= 0 or float((steps - spacing).abs().max()) > tolerance:
        raise ValueError(f'The plane must be sampled uniformly along {name}.')
    return values, spacing


def plane_grid(plane, *, direction=None):
    """Recognise a StitchedPlane, a DifferentiablePlaneResult or a native field-monitor dict."""
    if hasattr(plane, 'u_um') and hasattr(plane, 'direction'):
        u, hu = _uniform(plane.u_um, plane.axes[0])
        v, hv = _uniform(plane.v_um, plane.axes[1])
        spacing = (hu if hu is not None else plane.spacing_um[0], hv if hv is not None else plane.spacing_um[1])
        return PlaneGrid(plane.fields, u, v, spacing, plane.normal, float(plane.offset_um),
                         int(plane.direction) if direction is None else int(direction),
                         torch.as_tensor(plane.frequency_hz, dtype=torch.float64).reshape(-1), tuple(plane.components), plane.dimension)
    if isinstance(plane, DifferentiablePlaneResult):
        fields, points, shape, components, normal, frequency = plane.fields, plane.points_um, plane.shape, plane.components, plane.normal, plane.frequency_hz
        dimension = '2d' if plane.flux_units.endswith('invariant length') else '3d'
    elif isinstance(plane, dict):
        fields = plane['fields'] if isinstance(plane['fields'], torch.Tensor) else torch.as_tensor(np.asarray(plane['fields']))
        points, shape, components, normal = np.asarray(plane['points_um']), tuple(plane['shape']), tuple(plane['components']), plane['normal_axis']
        frequency = np.asarray(plane['frequency_hz'], dtype=np.float64)
        dimension = '2d' if str(plane.get('flux_units', '')).endswith('invariant length') else '3d'
    else:
        raise TypeError('Expected a StitchedPlane, a DifferentiablePlaneResult or a native field-monitor dict.')
    if not fields.is_complex() or fields.ndim != 3 or len(shape) != 3 or fields.shape[1] != math.prod(shape):
        raise ValueError('Plane fields must be complex (F, points, components) with a three-axis quadrature shape.')
    axis = 'xyz'.index(normal)
    if shape[axis] != 1:
        raise ValueError('The quadrature must have one sample along the plane normal.')
    b, c = [i for i in range(3) if i != axis]
    grid = fields.reshape(fields.shape[0], *shape, fields.shape[2]).squeeze(axis + 1)
    coordinates = (points.detach().to(torch.float64) if isinstance(points, torch.Tensor) else torch.as_tensor(np.asarray(points, dtype=np.float64))).reshape(*shape, 3)
    transverse = coordinates.select(axis, 0)   # (Nb, Nc, 3) with b < c
    u, hu = _uniform(transverse[:, 0, b], 'xyz'[b])
    v, hv = _uniform(transverse[0, :, c], 'xyz'[c])
    expected = torch.stack(torch.meshgrid(u, v, indexing='ij'), -1)
    if not torch.allclose(transverse[..., [b, c]], expected, rtol=0, atol=1e-9 * max(1., float(coordinates.abs().max()))):
        raise ValueError('The quadrature must be a tensor-product grid of the two transverse axes.')
    offset = float(coordinates.reshape(-1, 3)[0, axis])
    if float((coordinates.reshape(-1, 3)[:, axis] - offset).abs().max()) > 1e-9 * max(1., abs(offset)):
        raise ValueError('All quadrature points must lie in one plane.')
    hv = 1. if hv is None else hv
    if hu is None:
        raise ValueError('The plane needs at least two samples along its first transverse axis.')
    return PlaneGrid(grid, u, v, (hu, hv), normal, offset, 1 if direction is None else int(direction),
                     torch.as_tensor(frequency, dtype=torch.float64).reshape(-1), tuple(components), dimension)


class PlaneSpectrum:
    """Zero-padded transverse wavenumbers and lazily computed spectra of one plane.

    ``spectrum(f, c)`` returns the padded 2D FFT of component ``c`` at frequency
    ``f``; nothing is cached, so one padded spectrum is resident at a time.
    ``normal_wavenumber(f)`` returns the real and imaginary parts of ``k_n`` in
    double precision. The transfer phase is reduced modulo 2 pi in double
    precision before it is cast to the field precision.
    """
    def __init__(self, plane, pad=2, *, index=1., direction=None):
        if isinstance(index, bool) or not isinstance(index, (int, float)) or not math.isfinite(index) or index <= 0:
            raise ValueError('index must be a positive real refractive index of the homogeneous exterior.')
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 1:
            raise ValueError('pad must be an integer of at least one.')
        self.grid = plane if isinstance(plane, PlaneGrid) else plane_grid(plane, direction=direction)
        self.pad, self.index = pad, float(index)
        fields = self.grid.fields
        self.count, self.nu, self.nv = fields.shape[0], fields.shape[1], fields.shape[2]
        self.pu, self.pv = pad * self.nu, (pad * self.nv if self.nv > 1 else 1)
        self.ou, self.ov = (self.pu - self.nu) // 2, (self.pv - self.nv) // 2
        hu, hv = self.grid.spacing_um
        device = fields.device
        self.ku = 2 * math.pi * torch.fft.fftfreq(self.pu, hu, dtype=torch.float64, device=device)
        self.kv = 2 * math.pi * torch.fft.fftfreq(self.pv, hv, dtype=torch.float64, device=device)
        self.k = 2 * math.pi * self.index * self.grid.frequency_hz.to(device) / C0 * 1e-6
        self.wavelength_um = (C0 / self.grid.frequency_hz * 1e6 / self.index).tolist()
        # The transverse origin of the padded grid, so that exp(i ku x) is measured from it.
        self.u_origin = float(self.grid.u_um[0]) - self.ou * hu
        self.v_origin = float(self.grid.v_um[0]) - self.ov * hv
        limit = [min(1., w / (2 * hu)) for w in self.wavelength_um]
        self.report = dict(pad=pad, index=self.index, padded_shape=(self.pu, self.pv), spacing_um=(hu, hv),
                           wavelength_um=self.wavelength_um, max_angle_deg=[math.degrees(math.asin(v)) for v in limit],
                           evanescent_fraction=[float(((self.ku[:, None]**2 + self.kv[None, :]**2) > kk**2).double().mean()) for kk in self.k.tolist()],
                           note='sin(max angle) = wavelength / (2 spacing); light beyond (pad - 1) * span / 2 wraps around')

    def component(self, name):
        if name not in self.grid.components:
            raise ValueError(f'Component {name} is not stored on the plane.')
        return self.grid.components.index(name)

    def padded(self, f, c):
        return torch.nn.functional.pad(self.grid.fields[f, :, :, c], (self.ov, self.pv - self.nv - self.ov, self.ou, self.pu - self.nu - self.ou))

    def spectrum(self, f, c):
        """Padded 2D FFT of one component at one frequency, on the autograd graph."""
        return torch.fft.fft2(self.padded(f, c))

    def normal_wavenumber(self, f):
        """(Re k_n, Im k_n) as (Pu, Pv) float64 tensors; Im k_n >= 0 decays away from the plane."""
        argument = self.k[f]**2 - (self.ku[:, None]**2 + self.kv[None, :]**2)
        return torch.sqrt(argument.clamp(min=0)), torch.sqrt((-argument).clamp(min=0))

    def transfer(self, f, z_um, dtype, *, conjugate=False, wavenumber=None):
        """exp(i k_n z) (or its conjugate) for one distance, phase reduced modulo 2 pi in double precision."""
        real, imaginary = self.normal_wavenumber(f) if wavenumber is None else wavenumber
        rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        phase = torch.remainder(real * z_um, 2 * math.pi)
        return torch.polar(torch.exp(-imaginary * z_um).to(rdtype), (-phase if conjugate else phase).to(rdtype))


def _distances(z_um, like):
    """Distances as a float64 tensor; a nearly uniform grid (within 1e-6 of its largest value) is snapped to a uniform one."""
    z = (z_um.detach().to(torch.float64) if isinstance(z_um, torch.Tensor) else torch.as_tensor(np.asarray(z_um, dtype=np.float64))).reshape(-1).to(like.device)
    if not len(z) or not bool(torch.isfinite(z).all()) or bool((z < 0).any()):
        raise ValueError('Distances must be finite and nonnegative, measured from the plane along the propagation direction.')
    if len(z) > 1:
        step = float((z[-1] - z[0]) / (len(z) - 1))
        uniform = z[0] + step * torch.arange(len(z), dtype=torch.float64, device=z.device)
        if step > 0 and float((z - uniform).abs().max()) <= 1e-6 * max(1., float(z.abs().max())):
            return uniform
    return z


def _uniform_step(z):
    if len(z) < 2:
        return None
    steps = z[1:] - z[:-1]
    step = float(steps.mean())
    return step if float((steps - step).abs().max()) <= 1e-9 * max(1., abs(step)) else None


def _components(spectrum, components):
    names = tuple(components)
    if not names or len(set(names)) != len(names):
        raise ValueError('Select at least one distinct component.')
    return names, [spectrum.component(name) for name in names]


def _blocks(spectrum, f, z, chunk, dtype, conjugate=False):
    """Yield (start, distances, exact T at the start, ratio per step or None) for every z chunk.

    The transfer function is exact at each chunk start; inside a uniformly
    spaced chunk the caller multiplies by the ratio ``exp(i k_n dz)`` in place,
    so the phase error grows at most ``chunk`` times the field precision.
    """
    wavenumber = spectrum.normal_wavenumber(f)
    exact = lambda distance: spectrum.transfer(f, distance, dtype, conjugate=conjugate, wavenumber=wavenumber)
    step = _uniform_step(z)
    ratio = exact(step) if step is not None else None
    for start in range(0, len(z), chunk):
        block = z[start:start + chunk].tolist()
        yield start, block, exact(block[0]), ratio, exact


class _Section(torch.autograd.Function):
    """G[z, a] = sum_b Y[a, b] T[z, a, b] with a running product; the adjoint recomputes conj(T)."""
    @staticmethod
    def forward(ctx, y, spectrum, f, z, chunk, axis):
        ctx.spectrum, ctx.f, ctx.z, ctx.chunk, ctx.axis = spectrum, f, z, chunk, axis
        with torch.no_grad():
            out = torch.empty((len(z), y.shape[1 - axis]), dtype=y.dtype, device=y.device)
            for start, block, transfer, ratio, exact in _blocks(spectrum, f, z, chunk, y.dtype):
                running = y * transfer
                for j, distance in enumerate(block):
                    if j:
                        if ratio is not None:
                            running.mul_(ratio)
                        else:
                            running = y * exact(distance)
                    out[start + j] = running.sum(axis)
        return out

    @staticmethod
    def backward(ctx, grad):
        spectrum, axis = ctx.spectrum, ctx.axis
        with torch.no_grad():
            result = torch.zeros((spectrum.pu, spectrum.pv), dtype=grad.dtype, device=grad.device)
            for start, block, transfer, ratio, exact in _blocks(spectrum, ctx.f, ctx.z, ctx.chunk, grad.dtype, conjugate=True):
                for j, distance in enumerate(block):
                    if j:
                        if ratio is not None:
                            transfer.mul_(ratio)
                        else:
                            transfer = exact(distance)
                    seed = grad[start + j]
                    result.addcmul_(transfer, seed[:, None] if axis == 1 else seed[None, :])
        return result, None, None, None, None, None


class _Volume(torch.autograd.Function):
    """V[z] = crop(ifft2(S T_z)); the adjoint is sum_z conj(T_z) fft2(pad(grad_z)) / (Pu Pv)."""
    @staticmethod
    def forward(ctx, s, spectrum, f, z, chunk):
        ctx.spectrum, ctx.f, ctx.z, ctx.chunk = spectrum, f, z, chunk
        crop = (slice(spectrum.ou, spectrum.ou + spectrum.nu), slice(spectrum.ov, spectrum.ov + spectrum.nv))
        with torch.no_grad():
            out = torch.empty((len(z), spectrum.nu, spectrum.nv), dtype=s.dtype, device=s.device)
            for start, block, transfer, ratio, exact in _blocks(spectrum, f, z, chunk, s.dtype):
                for j, distance in enumerate(block):
                    if j:
                        if ratio is not None:
                            transfer.mul_(ratio)
                        else:
                            transfer = exact(distance)
                    out[start + j] = torch.fft.ifft2(s * transfer)[crop]
        return out

    @staticmethod
    def backward(ctx, grad):
        spectrum = ctx.spectrum
        padding = (spectrum.ov, spectrum.pv - spectrum.nv - spectrum.ov, spectrum.ou, spectrum.pu - spectrum.nu - spectrum.ou)
        with torch.no_grad():
            result = torch.zeros((spectrum.pu, spectrum.pv), dtype=grad.dtype, device=grad.device)
            scale = 1. / (spectrum.pu * spectrum.pv)
            for start, block, transfer, ratio, exact in _blocks(spectrum, ctx.f, ctx.z, ctx.chunk, grad.dtype, conjugate=True):
                for j, distance in enumerate(block):
                    if j:
                        if ratio is not None:
                            transfer.mul_(ratio)
                        else:
                            transfer = exact(distance)
                    result.addcmul_(transfer, torch.fft.fft2(torch.nn.functional.pad(grad[start + j], padding)), value=scale)
        return result, None, None, None, None


def plane_spectrum(plane, pad=2, *, index=1., direction=None):
    """The zero-padded spectrum object of a plane; see PlaneSpectrum."""
    return PlaneSpectrum(plane, pad, index=index, direction=direction)


def _spectrum(plane, pad, index, direction):
    return plane if isinstance(plane, PlaneSpectrum) else PlaneSpectrum(plane, pad, index=index, direction=direction)


@dataclass
class SectionResult:
    """Fields on a section through the propagation axis, ``(F, Nz, Na, C)``."""
    fields: torch.Tensor
    z_um: torch.Tensor            # distance from the plane along the propagation direction
    a_um: torch.Tensor            # transverse coordinate along the section
    normal_um: torch.Tensor       # absolute coordinate along the normal
    section: str
    axes: tuple                   # (normal name, transverse name)
    offset_um: float              # the fixed transverse coordinate
    frequency_hz: torch.Tensor
    components: tuple
    index: float
    report: dict = field(default_factory=dict)

    def intensity(self):
        """Sum of |E|^2 over the electric components present, ``(F, Nz, Na)``."""
        electric = [i for i, c in enumerate(self.components) if c.startswith('E')]
        if not electric:
            raise ValueError('The section holds no electric component.')
        return self.fields[..., electric].abs().square().sum(-1)

    def focus(self, frequency_index=0):
        """Peak intensity position and the transverse FWHM through it, in micrometres."""
        intensity = self.intensity()[frequency_index].detach()
        iz, ia = divmod(int(intensity.argmax()), intensity.shape[1])
        profile = intensity[iz]
        return dict(z_um=float(self.z_um[iz]), normal_um=float(self.normal_um[iz]), a_um=float(self.a_um[ia]),
                    peak_intensity=float(profile[ia]), fwhm_um=_fwhm(profile, ia, self.a_um))


def _fwhm(profile, peak, coordinates):
    half = float(profile[peak]) / 2
    values = profile.cpu().double().numpy()
    x = coordinates.cpu().double().numpy()
    left = right = None
    for i in range(peak, 0, -1):
        if values[i - 1] < half <= values[i]:
            left = x[i - 1] + (half - values[i - 1]) / (values[i] - values[i - 1]) * (x[i] - x[i - 1])
            break
    for i in range(peak, len(values) - 1):
        if values[i + 1] < half <= values[i]:
            right = x[i] + (values[i] - half) / (values[i] - values[i + 1]) * (x[i + 1] - x[i])
            break
    return float(right - left) if left is not None and right is not None else float('nan')


def propagate_section(plane, z_um, *, index=1., section='xz', offset_um=0., components=('Ex', 'Ey', 'Ez'), pad=2, chunk=64, direction=None):
    """Fields on a section through the normal at one fixed transverse coordinate, for all z at once.

    ``section`` names the two axes of the section, the first transverse and the
    second the normal ('xz' for a z-normal plane); ``offset_um`` is the fixed
    coordinate of the other transverse axis. The spectrum is multiplied by
    ``exp(i kb offset)`` and summed over that axis for every z (a matrix product
    per chunk), then one inverse FFT along the remaining axis per z chunk. The
    transfer function is exact at each chunk start and follows a recurrence
    inside a uniformly spaced chunk. Gradients flow back to the plane fields.
    """
    spectrum = _spectrum(plane, pad, index, direction)
    grid = spectrum.grid
    z = _distances(z_um, grid.fields)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('chunk must be a positive integer.')
    if not isinstance(section, str) or len(section) != 2 or section[1] != grid.normal or section[0] not in grid.axes:
        raise ValueError(f'section must pair a transverse axis with the normal, for example {grid.axes[0]}{grid.normal}.')
    if not isinstance(offset_um, (int, float)) or isinstance(offset_um, bool) or not math.isfinite(offset_um):
        raise ValueError('offset_um must be a finite coordinate.')
    along = 'u' if section[0] == grid.axes[0] else 'v'
    names, indices = _components(spectrum, components)
    if along == 'u':
        phase = torch.exp(1j * spectrum.kv * (offset_um - spectrum.v_origin)) / spectrum.pv
        coordinates, crop = grid.u_um, slice(spectrum.ou, spectrum.ou + spectrum.nu)
    else:
        phase = torch.exp(1j * spectrum.ku * (offset_um - spectrum.u_origin)) / spectrum.pu
        coordinates, crop = grid.v_um, slice(spectrum.ov, spectrum.ov + spectrum.nv)
    output = []
    for f in range(spectrum.count):
        per_component = []
        for c in indices:
            s = spectrum.spectrum(f, c)
            y = s * (phase[None, :] if along == 'u' else phase[:, None]).to(s.dtype)
            summed = _Section.apply(y, spectrum, f, z, chunk, 1 if along == 'u' else 0)
            per_component.append(torch.fft.ifft(summed, dim=-1)[:, crop])
        output.append(torch.stack(per_component, -1))
    fields = torch.stack(output, 0)
    return SectionResult(fields, z, coordinates, grid.offset_um + grid.direction * z, section, (grid.normal, section[0]), float(offset_um),
                         grid.frequency_hz, names, spectrum.index, dict(spectrum.report, chunk=chunk, method='angular spectrum section'))


@dataclass
class VolumeResult:
    """Fields on the full transverse grid for every z, ``(F, Nz, Nu, Nv, C)``, or the NPZ they were written to."""
    fields: torch.Tensor | None
    z_um: torch.Tensor
    u_um: torch.Tensor
    v_um: torch.Tensor
    normal_um: torch.Tensor
    normal: str
    axes: tuple
    frequency_hz: torch.Tensor
    components: tuple
    index: float
    bytes: dict
    path: str | None = None
    report: dict = field(default_factory=dict)

    def intensity(self):
        if self.fields is None:
            raise ValueError('The volume was written to disk; load the NPZ chunks instead.')
        electric = [i for i, c in enumerate(self.components) if c.startswith('E')]
        return self.fields[..., electric].abs().square().sum(-1)


def volume_bytes(plane, z_um, *, components=('Ex', 'Ey', 'Ez'), pad=2, chunk=8, index=1., direction=None):
    """Output and working-set byte estimates of ``propagate_volume``."""
    spectrum = _spectrum(plane, pad, index, direction)
    z = _distances(z_um, spectrum.grid.fields)
    names, _ = _components(spectrum, components)
    item = spectrum.grid.fields.element_size()
    output = spectrum.count * len(z) * spectrum.nu * spectrum.nv * len(names) * item
    working = 4 * spectrum.pu * spectrum.pv * item + 2 * spectrum.pu * spectrum.pv * 8 + chunk * spectrum.nu * spectrum.nv * item
    return dict(output_bytes=output, working_bytes=working, planes=len(z), padded_shape=(spectrum.pu, spectrum.pv))


def propagate_volume(plane, z_um, *, index=1., components=('Ex', 'Ey', 'Ez'), pad=2, chunk=8, budget_bytes=None, output=None, direction=None):
    """Fields on the full transverse grid at every z by one inverse FFT per z, chunked.

    ``budget_bytes`` rejects a call whose output plus working set exceeds it
    (the output alone when ``output`` names an NPZ path, in which case each z
    chunk is written as ``fields_f{f}_z{start}`` with the coordinates and
    nothing is retained). The NPZ path detaches from the graph.
    """
    spectrum = _spectrum(plane, pad, index, direction)
    grid = spectrum.grid
    z = _distances(z_um, grid.fields)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('chunk must be a positive integer.')
    names, indices = _components(spectrum, components)
    estimate = volume_bytes(spectrum, z, components=names, chunk=chunk)
    required = estimate['working_bytes'] + (0 if output is not None else estimate['output_bytes'])
    if budget_bytes is not None and (isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes <= 0):
        raise ValueError('budget_bytes must be a positive integer.')
    if budget_bytes is not None and required > budget_bytes:
        raise ValueError(f'Volume needs about {required:,} bytes ({estimate["output_bytes"]:,} output), above the budget of {budget_bytes:,}. '
                         'Use propagate_section, fewer planes or components, or write chunks to an NPZ.')
    normal_um = grid.offset_um + grid.direction * z
    common = dict(z_um=z, u_um=grid.u_um, v_um=grid.v_um, normal_um=normal_um, normal=grid.normal, axes=grid.axes,
                  frequency_hz=grid.frequency_hz, components=names, index=spectrum.index, bytes=estimate,
                  report=dict(spectrum.report, chunk=chunk, method='angular spectrum volume'))
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError(f'{path} exists; choose a new NPZ path.')
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
            def write(name, array):
                with archive.open(name + '.npy', 'w') as stream:
                    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
            write('z_um', z.cpu().numpy())
            write('normal_um', normal_um.cpu().numpy())
            write(grid.axes[0] + '_um', grid.u_um.cpu().numpy())
            write(grid.axes[1] + '_um', grid.v_um.cpu().numpy())
            write('frequency_hz', grid.frequency_hz.cpu().numpy())
            write('components', np.array(names))
            with torch.no_grad():
                for f in range(spectrum.count):
                    for start in range(0, len(z), chunk):
                        block = torch.stack([_Volume.apply(spectrum.spectrum(f, c), spectrum, f, z[start:start + chunk], chunk) for c in indices], -1)
                        write(f'fields_f{f}_z{start}', block.cpu().numpy())
        return VolumeResult(None, path=str(path), **common)
    output_fields = torch.stack([torch.stack([_Volume.apply(spectrum.spectrum(f, c), spectrum, f, z, chunk) for c in indices], -1)
                                 for f in range(spectrum.count)], 0)
    return VolumeResult(output_fields, **common)


def propagate_points(plane, points_um, *, index=1., components=('Ex', 'Ey', 'Ez'), pad=2, point_chunk=16, direction=None):
    """Fields at arbitrary points beyond the plane by direct evaluation of the spectrum sum, ``(F, P, C)``.

    Points are global coordinates; each must lie on the propagation side of the
    plane. The cost is ``points x padded samples``, so use it for a few points.
    """
    spectrum = _spectrum(plane, pad, index, direction)
    grid = spectrum.grid
    points = (points_um.detach().to(torch.float64) if isinstance(points_um, torch.Tensor)
              else torch.as_tensor(np.asarray(points_um, dtype=np.float64))).reshape(-1, 3).to(grid.fields.device)
    if not len(points) or not bool(torch.isfinite(points).all()):
        raise ValueError('Points must be a finite (P, 3) array in micrometres.')
    axis = 'xyz'.index(grid.normal)
    b, c = [i for i in range(3) if i != axis]
    z = grid.direction * (points[:, axis] - grid.offset_um)
    if bool((z < -1e-12).any()):
        raise ValueError('Every point must lie on the propagation side of the plane.')
    z = z.clamp(min=0)
    if isinstance(point_chunk, bool) or not isinstance(point_chunk, int) or point_chunk < 1:
        raise ValueError('point_chunk must be a positive integer.')
    names, indices = _components(spectrum, components)
    x = points[:, b] - spectrum.u_origin
    y = points[:, c] - spectrum.v_origin
    rows = []
    for f in range(spectrum.count):
        real, imaginary = spectrum.normal_wavenumber(f)
        spectra = [spectrum.spectrum(f, ci) for ci in indices]
        per_frequency = []
        for start in range(0, len(points), point_chunk):
            sl = slice(start, start + point_chunk)
            # exp(i(ku x + kv y + kn z)) per point, phases reduced in double precision.
            phase = (spectrum.ku[None, :, None] * x[sl, None, None] + spectrum.kv[None, None, :] * y[sl, None, None] + real[None] * z[sl, None, None])
            magnitude = torch.exp(-imaginary[None] * z[sl, None, None])
            kernel = torch.polar(magnitude, torch.remainder(phase, 2 * math.pi)).to(grid.fields.dtype) / (spectrum.pu * spectrum.pv)
            per_frequency.append(torch.stack([(s[None] * kernel).sum((1, 2)) for s in spectra], -1))
        rows.append(torch.cat(per_frequency, 0))
    return PointsResult(torch.stack(rows, 0), points, grid.frequency_hz, names, spectrum.index, dict(spectrum.report, method='angular spectrum points'))


@dataclass
class PointsResult:
    """Fields at arbitrary points, ``(F, P, C)``."""
    fields: torch.Tensor
    points_um: torch.Tensor
    frequency_hz: torch.Tensor
    components: tuple
    index: float
    report: dict = field(default_factory=dict)

    def intensity(self):
        electric = [i for i, c in enumerate(self.components) if c.startswith('E')]
        return self.fields[..., electric].abs().square().sum(-1)
