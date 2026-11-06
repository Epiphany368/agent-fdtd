"""G3-04, G3-05, G3-08 and G3-13: Mie cylinder/sphere, Drude sphere, Bloch grating against RCWA, curved-interface convergence.

The pre-declared cases live in docs/validation/cases/G3-04_*.json, G3-05_*.json,
G3-08_*.json and G3-13_*.json. Every limit asserted here is copied from those
files; nothing is tuned after a run. The fast suite (no environment variable)
runs the coarse meshes only. TORCHFDTD_G3_FULL=1 adds the judged meshes and the
recorded-only rows; TORCHFDTD_G3_RECORD=<dir> writes one JSON record per task
into that directory (docs/validation/g3 for the recorded run).

Independent references: an infinite-cylinder Mie series and a complex-index
sphere Mie series written here with SciPy Bessel functions, the repository's
real-index sphere series (examples/tfsf_sphere.py, cross-checked here), and
TORCWA (Kim and Lee, Comput. Phys. Commun. 282, 108552, 2023) run in a separate
interpreter by benchmarks/g3_torcwa_grating.py into
docs/validation/g3/G3-08_torcwa_reference.json.
"""
import sys
from pathlib import Path

# benchmarks/ and examples/ ship with the repository, not with the wheel, so a run against an
# installed package still reads them from the checkout that holds this test.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import math
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from scipy.special import jv, jvp, hankel1, h1vp, spherical_jn, spherical_yn

from torchfdtd import (Project, Region, Source, Structure, Material, FieldMonitor, SpectrumSettings, Simulation,
                       RunControl, BoundaryFace, Boundaries, DifferentiableSolid, smooth_geometry_epsilon)
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.radiation import diffraction_orders, C0
import torchfdtd.solver as solver_module

ROOT = Path(__file__).resolve().parents[1]
FULL = os.environ.get('TORCHFDTD_G3_FULL') == '1'
RECORD_DIR = os.environ.get('TORCHFDTD_G3_RECORD')
CUDA = torch.cuda.is_available()
CASES = {task: json.loads((ROOT/'docs'/'validation'/'cases'/name).read_text(encoding='utf-8')) for task, name in (
    ('G3-04', 'G3-04_mie_cylinder_sphere.json'), ('G3-05', 'G3-05_drude_sphere.json'),
    ('G3-08', 'G3-08_bloch_grating_rcwa.json'), ('G3-13', 'G3-13_curved_interface_convergence.json'))}
LAYER_A_RTOL = CASES['G3-04']['acceptance']['layer_a_cuda_fp32_vs_cpu_fp64']['rtol']


def full_only(reason):
    if not FULL:
        pytest.skip(reason+' (TORCHFDTD_G3_FULL=1 runs it)')


def needs_cuda():
    if not CUDA:
        pytest.skip('CUDA unavailable')


# ----------------------------------------------------------------------------
# Records: one JSON per task, rows keyed by a stable id, written only on request.
# ----------------------------------------------------------------------------
def record(task, row_id, payload):
    if not RECORD_DIR:
        return
    target = Path(RECORD_DIR)/f'{task}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(target.read_text(encoding='utf-8')) if target.exists() else dict(
        task=task, case_id=CASES[task]['case_id'], fixture_path=f'docs/validation/cases/{CASES[task]["case_id"]}.json',
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         os=platform.platform(), gpu=torch.cuda.get_device_name() if CUDA else None), rows={})
    data['generated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    data['full_mode'] = FULL
    data['rows'][row_id] = payload
    target.write_text(json.dumps(data, indent=1, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def listed(values):
    return [float(v) for v in np.asarray(values).ravel()]


# ----------------------------------------------------------------------------
# Independent analytic references
# ----------------------------------------------------------------------------
def cylinder_coefficients(x, m, polarization, orders=None):
    """Infinite nonmagnetic cylinder in vacuum at normal incidence.

    'TM': E parallel to the axis (TorchFDTD 2D Ez); 'TE': H parallel to the axis
    (TorchFDTD Ey for propagation along x). From continuity of the axial field
    and of its radial derivative divided by mu (TM) or by epsilon (TE):
      TM  b_n = [J_n(mx) J_n'(x) - m J_n'(mx) J_n(x)] / [J_n(mx) H_n'(x) - m J_n'(mx) H_n(x)]
      TE  a_n = [m J_n(mx) J_n'(x) - J_n'(mx) J_n(x)] / [m J_n(mx) H_n'(x) - J_n'(mx) H_n(x)]
    """
    if orders is None:
        orders = math.ceil(abs(x)+4*abs(x)**(1/3)+10)
    n = np.arange(0, orders+1)
    jx, djx, hx, dhx = jv(n, x), jvp(n, x), hankel1(n, x), h1vp(n, x)
    jm, djm = jv(n, m*x), jvp(n, m*x)
    if polarization == 'TM':
        return (jm*djx-m*djm*jx)/(jm*dhx-m*djm*hx)
    if polarization == 'TE':
        return (m*jm*djx-djm*jx)/(m*jm*dhx-djm*hx)
    raise ValueError(polarization)


def cylinder_cross_sections(wavelength_um, radius_um, index, polarization):
    """Scattering, extinction and absorption widths (um per unit length) of an infinite cylinder.

    The far field of the scattered wave gives C = (4/k) sum_n |c_n|^2 over all
    integer n, which is the standard Q_sca = (2/x)[|c_0|^2 + 2 sum |c_n|^2] times 2a.
    """
    sca, ext = [], []
    for wavelength in np.atleast_1d(wavelength_um):
        k = 2*np.pi/wavelength
        c = cylinder_coefficients(k*radius_um, index, polarization)
        weights = np.r_[1., 2.*np.ones(len(c)-1)]
        sca.append(4/k*np.sum(weights*abs(c)**2))
        ext.append(4/k*np.sum(weights*c.real))
    sca, ext = np.asarray(sca), np.asarray(ext)
    return dict(scattering=sca, extinction=ext, absorption=ext-sca)


def sphere_cross_sections(wavelength_um, radius_um, index):
    """Bohren-Huffman sphere series with a real or complex index (exp(-i omega t): Im > 0 absorbs)."""
    wavelengths = np.atleast_1d(np.asarray(wavelength_um, dtype=float))
    indices = np.broadcast_to(np.asarray(index), wavelengths.shape)
    sca, ext = [], []
    for wavelength, m in zip(wavelengths, indices):
        k = 2*np.pi/wavelength
        x = k*radius_um
        m = complex(m)
        n = np.arange(1, math.ceil(x+4*x**(1/3)+10)+1)
        mx = m*x
        jx, djx = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
        yx, dyx = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
        jm, djm = spherical_jn(n, mx), spherical_jn(n, mx, derivative=True)
        psi_x, dpsi_x = x*jx, jx+x*djx
        psi_m, dpsi_m = mx*jm, jm+mx*djm
        xi_x, dxi_x = x*(jx+1j*yx), (jx+1j*yx)+x*(djx+1j*dyx)
        a = (m*psi_m*dpsi_x-psi_x*dpsi_m)/(m*psi_m*dxi_x-xi_x*dpsi_m)
        b = (psi_m*dpsi_x-m*psi_x*dpsi_m)/(psi_m*dxi_x-m*xi_x*dpsi_m)
        sca.append(2*np.pi/k**2*np.sum((2*n+1)*(abs(a)**2+abs(b)**2)))
        ext.append(2*np.pi/k**2*np.sum((2*n+1)*(a+b).real))
    sca, ext = np.asarray(sca), np.asarray(ext)
    return dict(scattering=sca, extinction=ext, absorption=ext-sca)


def drude_permittivity(wavelength_um, epsilon_inf, plasma_rad_s, collision_rad_s):
    omega = 2*np.pi*C0/(np.asarray(wavelength_um, dtype=float)*1e-6)
    return epsilon_inf-plasma_rad_s**2/(omega**2+1j*collision_rad_s*omega)


# ----------------------------------------------------------------------------
# Fixture builders. Physical domain, PML thickness, sources, monitors and
# duration are fixed by the case files; only the mesh, backend and precision vary.
# ----------------------------------------------------------------------------
def pml_boundaries(layers, axes, extra=None):
    faces = dict(extra or {})
    for axis in axes:
        for side in ('min', 'max'):
            faces[f'{axis}_{side}'] = BoundaryFace(layers=layers)
    return Boundaries(**faces)


def region(dimension, size, mesh, pml_um, backend, precision, duration_fs, interface='staircase', bloch=None):
    layers = round(pml_um/mesh)
    if bloch is None:
        boundaries = pml_boundaries(layers, 'xy' if dimension == '2d' else 'xyz')
        phase = (0., 0., 0.)
    else:
        boundaries = pml_boundaries(layers, 'y', {'x_min': BoundaryFace(kind='bloch'), 'x_max': BoundaryFace(kind='bloch')})
        phase = (bloch, 0., 0.)
    r = Region(dimension=dimension, size=size, mesh=mesh, pml_cells=min(layers, 50), boundaries=boundaries, bloch_phase=phase,
               backend=backend, cuda_kernel='torch' if bloch is not None or backend == 'cpu' else 'fused',
               precision=precision, material_sampling='yee', interface_method=interface,
               snapshot_interval=10000, run_control=RunControl(divergence_check=True))
    r.steps = math.ceil(duration_fs*1e-15/r.time_step)
    return r


def box_monitors(half, axes, spectrum, prefix=''):
    monitors = []
    for axis in axes:
        for side, sign in (('min', -1), ('max', 1)):
            center = tuple(sign*half if a == axis else 0. for a in 'xyz')
            size = tuple(0. if a == axis else (2*half if a in axes else 1.) for a in 'xyz')
            monitors.append(FieldMonitor(id=f'{prefix}{axis}_{side}', name=f'{prefix}{axis}_{side}', normal=axis, center=center, size=size, spectrum=spectrum))
    return monitors


def wavelength_spectrum(start, stop, points):
    return SpectrumSettings(sampling='wavelength', wavelength_start=start, wavelength_stop=stop, frequency_points=points, apodization='none')


def cylinder_project(mesh, polarization, *, backend='cpu', precision='float64', index=1.5, radius=.3, center=(0., 0., 0.),
                     interface='staircase', band=(1.3, 1.8, 9), pulse=(8e-15, 30e-15), wavelength=1.55, duration_fs=120.):
    f = CASES['G3-04']['fixture']['cylinder']
    r = region('2d', (f['domain_um'], f['domain_um'], 1.), mesh, f['pml_um'], backend, precision, duration_fs, interface)
    spectrum = wavelength_spectrum(*band)
    monitors = box_monitors(f['monitor_half_um'], 'xy', spectrum)
    monitors.append(FieldMonitor(id='incident', name='incident', normal='x', center=(0., 0., 0.), size=(0., .4, 1.), spectrum=spectrum))
    component = {'TM': 'Ez', 'TE': 'Ey'}[polarization]
    return Project(name='G3 cylinder', region=r, materials=[Material(name='cylinder', index=index)],
                   structures=[Structure(kind='circle', radius=radius, material='cylinder', center=center)],
                   sources=[Source(kind='tfsf', size=(f['tfsf_um'], f['tfsf_um'], 1.), component=component, normal='x', direction='+',
                                   wavelength=wavelength, time_definition='standard', pulse_length=pulse[0], pulse_offset=pulse[1])],
                   monitors=monitors)


def sphere_project(mesh, *, backend='cpu', precision='float64'):
    f = CASES['G3-04']['fixture']['sphere']
    r = region('3d', (f['domain_um'],)*3, mesh, f['pml_um'], backend, precision, f['duration_fs'])
    spectrum = wavelength_spectrum(*f['band'])
    monitors = box_monitors(f['monitor_half_um'], 'xyz', spectrum)
    monitors.append(FieldMonitor(id='incident', name='incident', center=(0., 0., 0.), size=(0., .4, .4), spectrum=spectrum))
    return Project(name='G3 sphere', region=r, materials=[Material(name='sphere', index=f['index'])],
                   structures=[Structure(kind='sphere', radius=f['radius_um'], material='sphere')],
                   sources=[Source(kind='tfsf', size=(f['tfsf_um'],)*3, component='Ez', normal='x', direction='+', wavelength=1.55,
                                   time_definition='standard', pulse_length=8e-15, pulse_offset=30e-15)],
                   monitors=monitors)


def drude_project(mesh, radius, *, backend='cpu', precision='float64'):
    f = CASES['G3-05']['fixture']
    r = region('3d', (f['domain_um'],)*3, mesh, f['pml_um'], backend, precision, f['duration_fs'])
    spectrum = wavelength_spectrum(*f['band'])
    monitors = box_monitors(f['inner_monitor_half_um'], 'xyz', spectrum, 'in_')+box_monitors(f['outer_monitor_half_um'], 'xyz', spectrum, 'out_')
    monitors.append(FieldMonitor(id='incident', name='incident', center=(0., 0., 0.), size=(0., .06, .06), spectrum=spectrum))
    d = f['drude']
    return Project(name='G3 drude sphere', region=r,
                   materials=[Material(name='drude', model='drude', epsilon_inf=d['epsilon_inf'], plasma_rad_s=d['plasma_rad_s'], collision_rad_s=d['collision_rad_s'])],
                   structures=[Structure(kind='sphere', radius=radius, material='drude')] if radius else [],
                   sources=[Source(kind='tfsf', size=(f['tfsf_um'],)*3, component='Ez', normal='x', direction='+', wavelength=f['pulse_wavelength_um'],
                                   time_definition='standard', pulse_length=f['pulse_length_s'], pulse_offset=f['pulse_offset_s'])],
                   monitors=monitors)


def grating_project(mesh, polarization, angle_deg, wavelength, *, backend='cpu', precision='float64', grating=True,
                    interface='subpixel', duration_fs=None):
    f = CASES['G3-08']['fixture']
    period, height = f['period_um'], f['height_um']
    kx = 2*math.pi/wavelength*math.sin(math.radians(angle_deg))
    r = region('2d', (period, f['domain_y_um'], 1.), mesh, f['pml_um'], backend, precision,
               f['duration_fs'] if duration_fs is None else duration_fs, interface, bloch=kx*period)
    spectrum = SpectrumSettings(sampling='custom', custom_frequencies_hz=[C0/(w*1e-6) for w in sorted(f['wavelengths_um'], reverse=True)], apodization='none')
    monitors = [FieldMonitor(id=name, name=name, normal='y', center=(0., f[name+'_y_um'], 0.), size=(period, 0., 1.), spectrum=spectrum)
                for name in ('incident', 'reflection', 'transmission')]
    component = {'TE': 'Ez', 'TM': 'Ex'}[polarization]
    return Project(name='G3 grating', region=r, materials=[Material(name='grating', index=f['index'])],
                   structures=[Structure(kind='rectangle', size=(period*f['fill'], height, 1.), center=(0., f['grating_center_y_um'], 0.), material='grating')] if grating else [],
                   sources=[Source(kind='plane', normal='y', direction='+', size=(period, 0., 1.), center=(0., f['source_y_um'], 0.), component=component,
                                   wavelength=wavelength, time_definition='standard', pulse_length=f['pulse_length_s'], pulse_offset=f['pulse_offset_s'])],
                   monitors=monitors)


# ----------------------------------------------------------------------------
# Runs and measurements
# ----------------------------------------------------------------------------
_RUNS = {}


def run(project, key=None):
    """Run once per key (the fast suite shares runs between tests); return (result, wall seconds)."""
    if key is not None and key in _RUNS:
        return _RUNS[key]
    started = time.perf_counter()
    result = Simulation(project).run()
    value = (result, time.perf_counter()-started)
    if key is not None:
        _RUNS[key] = value
    return value


@contextmanager
def permittivity_override(epsilon):
    """Run the standard TFSF solver on a supplied permittivity image (the differentiable solid's output)."""
    original = solver_module.voxelize

    def patched(project, *, with_ownership=False, interface_plan=None):
        out = original(project, with_ownership=with_ownership, interface_plan=interface_plan)
        eps = np.asarray(epsilon, dtype=out[0].dtype)
        if eps.shape != out[0].shape:
            raise ValueError(f'Permittivity override shape {eps.shape} differs from the solver image {out[0].shape}.')
        return (eps, *out[1:])
    with mock.patch.object(solver_module, 'voxelize', patched):
        yield


def scattered_power(sample, reference, axes, prefix=''):
    """Outward scattered power through the monitor box after matched-reference subtraction."""
    power = None
    for axis in axes:
        for side, sign in (('min', -1), ('max', 1)):
            a = sample.field_monitor(f'{prefix}{axis}_{side}')
            b = reference.field_monitor(f'{prefix}{axis}_{side}')
            f = a['fields']-b['fields']
            flux = .5*np.real(np.cross(f[..., :3], f[..., 3:].conj()))[..., 'xyz'.index(axis)] @ a['weights']
            power = sign*flux if power is None else power+sign*flux
    return power


def total_inward_power(sample, axes, prefix=''):
    power = None
    for axis in axes:
        for side, sign in (('min', -1), ('max', 1)):
            flux = sample.field_monitor(f'{prefix}{axis}_{side}')['flux']
            power = -sign*flux if power is None else power-sign*flux
    return power


def incident_intensity(reference):
    incident = reference.field_monitor('incident')
    intensity = abs(incident['flux'])/sum(incident['weights'])
    assert np.all(intensity > .01*intensity.max()), 'incident spectrum too weak at a requested wavelength'
    return C0/incident['frequency_hz']*1e6, intensity


def cylinder_width(sample, reference):
    wavelengths, intensity = incident_intensity(reference)
    return wavelengths, scattered_power(sample, reference, 'xy')/intensity*1e6


def sphere_sections(sample, reference, prefix='', axes='xyz'):
    wavelengths, intensity = incident_intensity(reference)
    return wavelengths, scattered_power(sample, reference, axes, prefix)/intensity*1e12


def peak_and_width(wavelengths, values):
    """Parabolic peak refinement and linear-interpolated full width at half of the refined peak."""
    order = np.argsort(wavelengths)
    wl, s = np.asarray(wavelengths)[order], np.asarray(values)[order]
    i = min(max(int(np.argmax(s)), 1), len(s)-2)
    y0, y1, y2 = s[i-1], s[i], s[i+1]
    d = (y0-y2)/(2*(y0-2*y1+y2))
    step = wl[1]-wl[0]
    position, height = wl[i]+d*step, y1-(y0-y2)*d/4
    half = height/2
    lo = i
    while lo > 0 and s[lo] > half:
        lo -= 1
    hi = i
    while hi < len(s)-1 and s[hi] > half:
        hi += 1
    low = wl[lo]+(half-s[lo])/(s[lo+1]-s[lo])*step if lo < i else wl[0]
    high = wl[hi-1]+(half-s[hi-1])/(s[hi]-s[hi-1])*step if hi > i else wl[-1]
    return float(position), float(height), float(high-low)


def lift_line(monitor):
    """Lift a 2D y-normal line monitor to a two-sample z-invariant plane for the native order decomposition."""
    fields = torch.as_tensor(monitor['fields'], dtype=torch.complex128)
    points = torch.as_tensor(monitor['points_um'], dtype=torch.float64)
    n = len(points)
    pts = torch.cat([torch.cat([points[:, :2], torch.full((n, 1), z, dtype=torch.float64)], 1) for z in (-.25, .25)])
    area = float(points[:, 0].max()-points[:, 0].min()+(points[1, 0]-points[0, 0]))*1e-6*1e-6
    return DifferentiablePlaneResult(torch.cat([fields, fields], 1), torch.as_tensor(monitor['frequency_hz'], dtype=torch.float64), pts,
                                     torch.full((2*n,), area/(2*n), dtype=torch.float64), (n, 1, 2), 'y', 'same-run', {})


def grating_orders(sample, reference, angle_deg, wavelength, orders=(-2, -1, 0, 1, 2)):
    """Efficiencies and face-referenced complex amplitudes of the propagating orders on both sides."""
    f = CASES['G3-08']['fixture']
    period = f['period_um']
    kx = 2*math.pi/wavelength*math.sin(math.radians(angle_deg))
    incident = lift_line(reference.field_monitor('incident'))
    inc = diffraction_orders(incident, [(0, 0)], period_um=(1., period), bloch_wavevector_per_um=(0., kx))
    p_inc = inc.forward_power[:, 0]
    y_top, y_bottom = f['grating_center_y_um']-f['height_um']/2, f['grating_center_y_um']+f['height_um']/2
    out = {}
    for name, direction, y_m in (('transmission', 'forward', f['transmission_y_um']), ('reflection', 'backward', f['reflection_y_um'])):
        plane = lift_line(sample.field_monitor(name))
        if name == 'reflection':
            plane = replace(plane, fields=plane.fields-lift_line(reference.field_monitor(name)).fields)
        res = diffraction_orders(plane, [(0, m) for m in orders], period_um=(1., period), bloch_wavevector_per_um=(0., kx))
        efficiency = getattr(res, direction+'_power')/p_inc[:, None]
        other = getattr(res, ('backward' if direction == 'forward' else 'forward')+'_power')/p_inc[:, None]
        ky = res.wavevector_per_um[..., 1]
        for component in ('Ez', 'Ex'):
            c = 2 if component == 'Ez' else 0
            amplitude = getattr(res, direction+'_fields')[..., c]
            face = amplitude*torch.exp(-1j*ky*(y_m-y_bottom)) if direction == 'forward' else amplitude*torch.exp(-1j*ky*(y_top-y_m))
            e_top = inc.forward_fields[:, 0, c]*torch.exp(1j*inc.wavevector_per_um[:, 0, 1]*(y_top-f['incident_y_um']))
            out[(name, component)] = face/e_top[:, None]
        out[(name, 'efficiency')] = efficiency
        out[(name, 'counter_power')] = other
        out[(name, 'propagating')] = res.propagating
    out['wavelengths_um'] = C0/incident.frequency_hz.numpy()*1e6
    out['orders'] = list(orders)
    return out


# ----------------------------------------------------------------------------
# G3-04: Mie cylinder (2D) and sphere (3D)
# ----------------------------------------------------------------------------
def test_g3_04_reference_series_limits():
    """Optical theorem, small-size limits and agreement with the repository's real-index sphere series."""
    from examples.tfsf_sphere import mie_cross_section
    wl = np.linspace(1.3, 1.8, 9)
    for polarization in ('TM', 'TE'):
        c = cylinder_cross_sections(wl, .3, 1.5, polarization)
        assert abs(c['absorption']).max() < 1e-12
    k = 2*np.pi/wl
    a = 1e-3
    tm = cylinder_cross_sections(wl, a, 1.5, 'TM')['scattering']
    np.testing.assert_allclose(tm, np.pi**2/4*k**3*a**4*(1.5**2-1)**2, rtol=2e-4)
    te = cylinder_cross_sections(wl, a, 1.5, 'TE')['scattering']
    np.testing.assert_allclose(te, np.pi**2/2*k**3*a**4*((1.5**2-1)/(1.5**2+1))**2, rtol=2e-4)
    assert cylinder_cross_sections(wl, .3, 1.5+.1j, 'TM')['absorption'].min() > 0
    mine = sphere_cross_sections(wl, .3, 1.5)
    np.testing.assert_allclose(mine['scattering'], mie_cross_section(wl, .3, 1.5), rtol=1e-12)
    assert abs(mine['absorption']).max() < 1e-12
    d = CASES['G3-05']['fixture']['drude']
    wl = np.linspace(.33, .45, 9)
    eps = drude_permittivity(wl, d['epsilon_inf'], d['plasma_rad_s'], d['collision_rad_s'])
    assert np.all(eps.imag > 0)
    c = sphere_cross_sections(wl, a, np.sqrt(eps))
    alpha = 4*np.pi*a**3*(eps-1)/(eps+2)
    np.testing.assert_allclose(c['absorption'], 2*np.pi/wl*alpha.imag, rtol=4e-3)
    np.testing.assert_allclose(c['scattering'], (2*np.pi/wl)**4/(6*np.pi)*abs(alpha)**2, rtol=4e-3)


def cylinder_rows(polarization, meshes, backend, precision, interface='staircase', center=(0., 0., 0.)):
    f = CASES['G3-04']['fixture']['cylinder']
    rows = []
    for mesh in meshes:
        project = cylinder_project(mesh, polarization, backend=backend, precision=precision, interface=interface, center=center)
        key = ('cylinder', polarization, mesh, backend, precision, interface, center)
        sample, wall = run(project, key)
        empty = project.model_copy(deep=True)
        empty.structures = []
        reference, wall_reference = run(empty, key+('empty',))
        wavelengths, width = cylinder_width(sample, reference)
        mie = cylinder_cross_sections(wavelengths, f['radius_um'], f['index'], polarization)['scattering']
        error = width/mie-1
        rows.append(dict(mesh_um=mesh, grid=list(project.region.shape), steps=project.region.steps, backend=backend, precision=precision,
                         interface=interface, center_um=list(center), wavelengths_um=listed(wavelengths), width_um=listed(width),
                         mie_um=listed(mie), relative_error=listed(error), max_relative_error=float(abs(error).max()),
                         wall_s=wall, wall_reference_s=wall_reference))
    return rows


@pytest.mark.parametrize('polarization', ['TM', 'TE'])
def test_g3_04_cylinder_scattering(polarization):
    case = CASES['G3-04']
    meshes = case['fixture']['cylinder']['mesh_sequence_um']
    judged = case['fixture']['cylinder']['judged_mesh_um']
    rows = cylinder_rows(polarization, meshes if FULL else [m for m in meshes if m > judged], 'cpu', 'float64')
    for row in rows:
        record('G3-04', f'cylinder/{polarization}/staircase/h={row["mesh_um"]}/cpu-float64', row)
    assert rows[0]['max_relative_error'] < .2, 'coarse-mesh smoke limit'
    full_only('judged mesh of the cylinder fixture')
    final = [r for r in rows if r['mesh_um'] == judged][0]
    limit = case['acceptance']['integrated_cross_section_relative_error']
    assert final['max_relative_error'] <= limit, f'{polarization} cylinder at h={judged}: {final["max_relative_error"]:.4f} > {limit}'


@pytest.mark.parametrize('polarization', ['TM', 'TE'])
def test_g3_04_cylinder_cuda_layer_a(polarization):
    needs_cuda()
    case = CASES['G3-04']
    judged = case['fixture']['cylinder']['judged_mesh_um']
    meshes = [2*judged]+([judged] if FULL else [])
    cpu = {r['mesh_um']: r for r in cylinder_rows(polarization, meshes, 'cpu', 'float64')}
    cuda = {r['mesh_um']: r for r in cylinder_rows(polarization, meshes, 'cuda', 'float32')}
    for mesh in meshes:
        difference = np.asarray(cuda[mesh]['width_um'])/np.asarray(cpu[mesh]['width_um'])-1
        row = dict(cuda[mesh], layer_a_relative_difference=listed(difference), layer_a_max_relative_difference=float(abs(difference).max()), layer_a_rtol=LAYER_A_RTOL)
        record('G3-04', f'cylinder/{polarization}/staircase/h={mesh}/cuda-float32', row)
        assert abs(difference).max() <= LAYER_A_RTOL, f'layer A {polarization} h={mesh}: {abs(difference).max():.2e}'


def test_g3_04_cylinder_resonance():
    case = CASES['G3-04']
    f = case['fixture']['resonance']
    limits = case['acceptance']['resonance']
    meshes = f['mesh_sequence_um'] if FULL else f['mesh_sequence_um'][:1]
    rows = []
    for mesh in meshes:
        backend, precision = ('cuda', 'float32') if (CUDA and mesh < f['mesh_sequence_um'][0]) else ('cpu', 'float64')
        project = cylinder_project(mesh, 'TE', backend=backend, precision=precision, index=f['index'], radius=f['radius_um'],
                                   band=tuple(f['band']), pulse=(f['pulse_length_s'], f['pulse_offset_s']), wavelength=f['pulse_wavelength_um'])
        sample, wall = run(project)
        empty = project.model_copy(deep=True)
        empty.structures = []
        reference, _ = run(empty)
        wavelengths, width = cylinder_width(sample, reference)
        mie = cylinder_cross_sections(wavelengths, f['radius_um'], f['index'], 'TE')['scattering']
        measured, analytic = peak_and_width(wavelengths, width), peak_and_width(wavelengths, mie)
        row = dict(mesh_um=mesh, grid=list(project.region.shape), steps=project.region.steps, backend=backend, precision=precision,
                   wavelengths_um=listed(wavelengths), width_um=listed(width), mie_um=listed(mie),
                   peak_um=measured[0], mie_peak_um=analytic[0], peak_relative_error=measured[0]/analytic[0]-1,
                   peak_height_um=measured[1], mie_peak_height_um=analytic[1], fwhm_um=measured[2], mie_fwhm_um=analytic[2],
                   fwhm_relative_error=measured[2]/analytic[2]-1, wall_s=wall)
        rows.append(row)
        record('G3-04', f'resonance/TE/h={mesh}/{backend}-{precision}', row)
    full_only('judged mesh of the resonance fixture')
    final = rows[-1]
    assert abs(final['peak_relative_error']) <= limits['peak_position_relative_error'], final['peak_relative_error']
    assert abs(final['fwhm_relative_error']) <= limits['fwhm_relative_error'], final['fwhm_relative_error']


def sphere_rows(meshes, backend, precision):
    f = CASES['G3-04']['fixture']['sphere']
    rows = []
    for mesh in meshes:
        project = sphere_project(mesh, backend=backend, precision=precision)
        sample, wall = run(project, ('sphere', mesh, backend, precision))
        empty = project.model_copy(deep=True)
        empty.structures = []
        reference, _ = run(empty, ('sphere', mesh, backend, precision, 'empty'))
        wavelengths, section = sphere_sections(sample, reference)
        mie = sphere_cross_sections(wavelengths, f['radius_um'], f['index'])['scattering']
        error = section/mie-1
        rows.append(dict(mesh_um=mesh, grid=list(project.region.shape), steps=project.region.steps, backend=backend, precision=precision,
                         wavelengths_um=listed(wavelengths), cross_section_um2=listed(section), mie_um2=listed(mie),
                         relative_error=listed(error), max_relative_error=float(abs(error).max()), wall_s=wall))
    return rows


def test_g3_04_sphere_scattering():
    case = CASES['G3-04']
    f = case['fixture']['sphere']
    meshes = f['mesh_sequence_um'] if FULL else [m for m in f['mesh_sequence_um'] if m >= f['judged_mesh_um']]
    rows = sphere_rows(meshes, 'cpu', 'float64')
    for row in rows:
        record('G3-04', f'sphere/h={row["mesh_um"]}/cpu-float64', row)
    judged = [r for r in rows if r['mesh_um'] == f['judged_mesh_um']][0]
    limit = case['acceptance']['integrated_cross_section_relative_error']
    assert judged['max_relative_error'] <= limit, f'sphere at h={f["judged_mesh_um"]}: {judged["max_relative_error"]:.4f} > {limit}'


def test_g3_04_sphere_cuda_layer_a():
    needs_cuda()
    f = CASES['G3-04']['fixture']['sphere']
    meshes = [m for m in f['mesh_sequence_um'] if m >= f['judged_mesh_um']]
    cpu = {r['mesh_um']: r for r in sphere_rows(meshes, 'cpu', 'float64')}
    cuda = {r['mesh_um']: r for r in sphere_rows(meshes, 'cuda', 'float32')}
    for mesh in meshes:
        difference = np.asarray(cuda[mesh]['cross_section_um2'])/np.asarray(cpu[mesh]['cross_section_um2'])-1
        row = dict(cuda[mesh], layer_a_relative_difference=listed(difference), layer_a_max_relative_difference=float(abs(difference).max()), layer_a_rtol=LAYER_A_RTOL)
        record('G3-04', f'sphere/h={mesh}/cuda-float32', row)
        assert abs(difference).max() <= LAYER_A_RTOL, f'layer A sphere h={mesh}: {abs(difference).max():.2e}'


# ----------------------------------------------------------------------------
# G3-05: Drude sphere scattering and absorption
# ----------------------------------------------------------------------------
def drude_rows(mesh, backend, precision):
    f = CASES['G3-05']['fixture']
    d = f['drude']
    empty = drude_project(mesh, 0., backend=backend, precision=precision)
    reference, wall_reference = run(empty, ('drude', mesh, backend, precision, 'empty'))
    rows = {}
    for radius in f['radii_um']:
        project = drude_project(mesh, radius, backend=backend, precision=precision)
        sample, wall = run(project, ('drude', mesh, backend, precision, radius))
        wavelengths, intensity = incident_intensity(reference)
        inner = scattered_power(sample, reference, 'xyz', 'in_')/intensity*1e12
        outer = scattered_power(sample, reference, 'xyz', 'out_')/intensity*1e12
        absorption = total_inward_power(sample, 'xyz', 'in_')/intensity*1e12
        eps = drude_permittivity(wavelengths, d['epsilon_inf'], d['plasma_rad_s'], d['collision_rad_s'])
        mie = sphere_cross_sections(wavelengths, radius, np.sqrt(eps))
        # The trapezoidal ADE samples the Drude response at (2/dt) tan(omega dt/2); the Mie
        # series with that permittivity separates the ADE discretisation from the grid error.
        from torchfdtd.materials import permittivity
        eps_ade = permittivity(project.materials[0], C0/(wavelengths*1e-6), project.region.time_step)
        mie_ade = sphere_cross_sections(wavelengths, radius, np.sqrt(eps_ade))
        sca_error, abs_error = outer/mie['scattering']-1, absorption/mie['absorption']-1
        rows[radius] = dict(mesh_um=mesh, grid=list(project.region.shape), steps=project.region.steps, time_step_s=project.region.time_step,
                            backend=backend, precision=precision, radius_um=radius, cells_per_radius=radius/mesh,
                            wavelengths_um=listed(wavelengths), scattering_um2=listed(outer), inner_scattering_um2=listed(inner),
                            absorption_um2=listed(absorption), mie_scattering_um2=listed(mie['scattering']), mie_absorption_um2=listed(mie['absorption']),
                            mie_ade_scattering_um2=listed(mie_ade['scattering']), mie_ade_absorption_um2=listed(mie_ade['absorption']),
                            scattering_relative_error=listed(sca_error), absorption_relative_error=listed(abs_error),
                            max_scattering_relative_error=float(abs(sca_error).max()), max_absorption_relative_error=float(abs(abs_error).max()),
                            inner_outer_scattering_max_difference_over_band_maximum=float(abs(inner-outer).max()/abs(outer).max()),
                            peak_scattering_um=peak_and_width(wavelengths, outer)[0], mie_peak_scattering_um=peak_and_width(wavelengths, mie['scattering'])[0],
                            peak_absorption_um=peak_and_width(wavelengths, absorption)[0], mie_peak_absorption_um=peak_and_width(wavelengths, mie['absorption'])[0],
                            wall_s=wall, wall_reference_s=wall_reference)
    return rows


@pytest.mark.parametrize('radius', CASES['G3-05']['fixture']['radii_um'])
def test_g3_05_drude_sphere(radius):
    case = CASES['G3-05']
    f = case['fixture']
    meshes = f['mesh_sequence_um'] if FULL else f['mesh_sequence_um'][:1]
    rows = []
    for mesh in meshes:
        row = drude_rows(mesh, 'cpu', 'float64')[radius]
        rows.append(row)
        record('G3-05', f'r={radius}/h={mesh}/cpu-float64', row)
        assert row['inner_outer_scattering_max_difference_over_band_maximum'] < CASES['G3-05']['acceptance']['self_consistency']['inner_outer_scattering_max_difference_over_band_maximum'], 'inner and outer scattered-power surfaces disagree'
    full_only('judged mesh h/4 of the Drude fixture')
    final = rows[-1]
    limits = case['acceptance']['per_radius'][str(radius)]
    failures = []
    if final['max_scattering_relative_error'] > limits['scattering_relative_error']:
        failures.append(f'scattering {final["max_scattering_relative_error"]:.3f} > {limits["scattering_relative_error"]}')
    if final['max_absorption_relative_error'] > limits['absorption_relative_error']:
        failures.append(f'absorption {final["max_absorption_relative_error"]:.3f} > {limits["absorption_relative_error"]}')
    assert not failures, f'r={radius} um at h={final["mesh_um"]} um: '+'; '.join(failures)


def test_g3_05_drude_cuda_layer_a():
    needs_cuda()
    f = CASES['G3-05']['fixture']
    meshes = f['mesh_sequence_um'][:2] if FULL else f['mesh_sequence_um'][:1]
    for mesh in meshes:
        cpu, cuda = drude_rows(mesh, 'cpu', 'float64'), drude_rows(mesh, 'cuda', 'float32')
        for radius in f['radii_um']:
            differences = {name: np.asarray(cuda[radius][name])/np.asarray(cpu[radius][name])-1 for name in ('scattering_um2', 'absorption_um2')}
            worst = max(float(abs(v).max()) for v in differences.values())
            row = dict(cuda[radius], layer_a_relative_difference={k: listed(v) for k, v in differences.items()},
                       layer_a_max_relative_difference=worst, layer_a_rtol=LAYER_A_RTOL)
            record('G3-05', f'r={radius}/h={mesh}/cuda-float32', row)
            assert worst <= LAYER_A_RTOL, f'layer A Drude r={radius} h={mesh}: {worst:.2e}'


# ----------------------------------------------------------------------------
# G3-08: Bloch grating diffraction orders against TORCWA
# ----------------------------------------------------------------------------
TORCWA = json.loads((ROOT/'docs'/'validation'/'g3'/'G3-08_torcwa_reference.json').read_text(encoding='utf-8'))


def torcwa_configuration(angle_deg, wavelength):
    for c in TORCWA['configurations']:
        if abs(c['angle_deg']-angle_deg) < 1e-9 and abs(c['wavelength_um']-wavelength) < 1e-9:
            return c
    raise KeyError((angle_deg, wavelength))


def grating_configurations():
    f = CASES['G3-08']['fixture']
    return [(pol, angle, wl) for pol in ('TE', 'TM') for angle in f['angles_deg'] for wl in f['wavelengths_um']]


def grating_row(polarization, angle, wavelength, mesh, backend, precision, interface, duration_fs=None):
    f = CASES['G3-08']['fixture']
    limits = CASES['G3-08']['acceptance']
    key = ('grating', polarization, angle, wavelength, mesh, backend, precision, interface, duration_fs)
    sample, wall = run(grating_project(mesh, polarization, angle, wavelength, backend=backend, precision=precision, interface=interface, duration_fs=duration_fs), key)
    reference, _ = run(grating_project(mesh, polarization, angle, wavelength, backend=backend, precision=precision, grating=False, interface=interface, duration_fs=duration_fs), key+('empty',))
    result = grating_orders(sample, reference, angle, wavelength)
    i = int(np.argmin(abs(result['wavelengths_um']-wavelength)))
    oracle = torcwa_configuration(angle, wavelength)
    component = 'Ez' if polarization == 'TE' else 'Ex'
    orders, balance = {}, 0.
    for j, m in enumerate(result['orders']):
        propagating = bool(result[('transmission', 'propagating')][i, j])
        if not propagating:
            continue
        ref = oracle['oracle'][str(m)][polarization]
        entry = {}
        for name, key_prefix in (('transmission', 't'), ('reflection', 'r')):
            efficiency = float(result[(name, 'efficiency')][i, j])
            phase = float(torch.angle(result[(name, component)][i, j]))
            reference_phase = ref[key_prefix+'_phase_rad']
            dominant = ref[key_prefix+'_efficiency'] >= limits['dominant_order_efficiency']
            entry[name] = dict(efficiency=efficiency, torcwa_efficiency=ref[key_prefix+'_efficiency'], efficiency_error=efficiency-ref[key_prefix+'_efficiency'],
                               phase_rad=phase, torcwa_phase_rad=reference_phase, phase_error_rad=math.remainder(phase-reference_phase, 2*math.pi),
                               dominant=dominant, counter_direction_power=float(result[(name, 'counter_power')][i, j]))
            balance += efficiency
        orders[str(m)] = entry
    project = grating_project(mesh, polarization, angle, wavelength, backend=backend, precision=precision, interface=interface, duration_fs=duration_fs)
    return dict(polarization=polarization, angle_deg=angle, wavelength_um=wavelength, mesh_um=mesh, grid=list(project.region.shape), steps=project.region.steps,
                duration_fs=f['duration_fs'] if duration_fs is None else duration_fs, backend=backend, precision=precision, interface=interface,
                orders=orders, efficiency_sum=balance, torcwa_harmonics=oracle['oracle_harmonics'],
                max_efficiency_error=max(abs(v[n]['efficiency_error']) for v in orders.values() for n in ('transmission', 'reflection')),
                max_dominant_phase_error_rad=max([abs(v[n]['phase_error_rad']) for v in orders.values() for n in ('transmission', 'reflection') if v[n]['dominant']] or [0.]),
                wall_s=wall)


def check_grating_row(row):
    limits = CASES['G3-08']['acceptance']
    failures = []
    if row['max_efficiency_error'] > limits['efficiency_absolute_error']:
        failures.append(f'efficiency error {row["max_efficiency_error"]:.4f} > {limits["efficiency_absolute_error"]}')
    if row['max_dominant_phase_error_rad'] > limits['dominant_order_phase_error_rad']:
        failures.append(f'dominant phase error {row["max_dominant_phase_error_rad"]:.4f} > {limits["dominant_order_phase_error_rad"]}')
    if abs(row['efficiency_sum']-1) > limits['lossless_balance_absolute_error']:
        failures.append(f'efficiency sum {row["efficiency_sum"]:.4f} off unity by more than {limits["lossless_balance_absolute_error"]}')
    return failures


def grating_row_id(row):
    return f'{row["polarization"]}/{row["angle_deg"]:g}deg/{row["wavelength_um"]}um/{row["interface"]}/h={row["mesh_um"]}/{row["duration_fs"]:g}fs/{row["backend"]}-{row["precision"]}'


def test_g3_08_torcwa_reference_is_converged():
    """The oracle file records the harmonic sequence; TE must be converged geometrically, TM within the Laurent-rule 1/N tail."""
    limits = CASES['G3-08']['acceptance']['oracle_convergence']
    assert TORCWA['empty_layer_check']['max_abs_error'] < 1e-8
    assert TORCWA['reference_method'].startswith('TORCWA')
    for c in TORCWA['configurations']:
        for m, row in c['convergence'].items():
            for pol in ('TE', 'TM'):
                for key in ('t_efficiency', 'r_efficiency'):
                    assert abs(row[pol][key+'_change_from_previous']) <= limits['efficiency_change_at_last_doubling'][pol], (c['angle_deg'], c['wavelength_um'], m, pol, key)
                for key in ('t_phase_rad', 'r_phase_rad'):
                    if c['oracle'][m][pol][key[0]+'_efficiency'] >= CASES['G3-08']['acceptance']['dominant_order_efficiency']:
                        assert abs(row[pol][key+'_change_from_previous']) <= limits['phase_change_at_last_doubling'][pol], (c['angle_deg'], c['wavelength_um'], m, pol, key)


def test_g3_08_empty_cell():
    """No grating: unit forward zero-order transmission, no other orders, no backward power; FP64 at the declared limit, FP32 at the layer-A tolerance."""
    f = CASES['G3-08']['fixture']
    limits = CASES['G3-08']['acceptance']['empty_cell']
    mesh = f['mesh_sequence_um'][0]
    executions = [('cpu', 'float64', None)]+([('cuda', 'float32', LAYER_A_RTOL)] if CUDA else [])
    for backend, precision, fp32_limit in executions:
        for polarization, angle in (('TE', 0.), ('TM', 20.)):
            wavelength = f['wavelengths_um'][1]
            declared = limits['normal_incidence'] if angle == 0. else limits['oblique_incidence']
            zero_limit = other_limit = declared if fp32_limit is None else max(declared, fp32_limit)
            reference, _ = run(grating_project(mesh, polarization, angle, wavelength, backend=backend, precision=precision, grating=False, duration_fs=f['fast_duration_fs']),
                               ('grating', polarization, angle, wavelength, mesh, backend, precision, 'subpixel', f['fast_duration_fs'], 'empty'))
            result = grating_orders(reference, reference, angle, wavelength)
            i = int(np.argmin(abs(result['wavelengths_um']-wavelength)))
            zero = result['orders'].index(0)
            t = result[('transmission', 'efficiency')][i]
            others = [float(t[j]) for j in range(len(t)) if j != zero]
            backward = float(result[('transmission', 'counter_power')][i, zero])
            record('G3-08', f'empty/{polarization}/{angle:g}deg/{wavelength}um/h={mesh}/{backend}-{precision}',
                   dict(polarization=polarization, angle_deg=angle, wavelength_um=wavelength, mesh_um=mesh, backend=backend, precision=precision,
                        duration_fs=f['fast_duration_fs'], zero_order_transmission=float(t[zero]), other_orders=others, backward_zero_order=backward,
                        zero_order_limit=zero_limit, other_order_limit=other_limit))
            assert abs(float(t[zero])-1) <= zero_limit, (backend, precision, float(t[zero]))
            assert max(others+[0.]) <= other_limit, (backend, precision, others)
            assert backward <= other_limit, (backend, precision, backward)


@pytest.mark.parametrize('polarization,angle,wavelength', grating_configurations())
def test_g3_08_grating_orders(polarization, angle, wavelength):
    f = CASES['G3-08']['fixture']
    coarse, judged = f['mesh_sequence_um'][0], f['judged_mesh_um']
    if FULL:
        needs_cuda()
        row = grating_row(polarization, angle, wavelength, judged, 'cuda', 'float32', 'subpixel')
    else:
        backend, precision = ('cuda', 'float32') if CUDA else ('cpu', 'float64')
        row = grating_row(polarization, angle, wavelength, coarse, backend, precision, 'subpixel', duration_fs=f['fast_duration_fs'])
    record('G3-08', grating_row_id(row), row)
    failures = check_grating_row(row)
    if not FULL:
        # The fast suite runs the coarse mesh with the shorter control duration; it is a smoke check only.
        assert row['max_efficiency_error'] < 5*CASES['G3-08']['acceptance']['efficiency_absolute_error'], failures
        return
    assert not failures, f'{polarization} {angle:g} deg {wavelength} um at h={judged}: '+'; '.join(failures)


@pytest.mark.parametrize('polarization,angle,wavelength', grating_configurations())
def test_g3_08_grating_cuda_layer_a(polarization, angle, wavelength):
    """The same discrete problem at the coarse mesh in CPU FP64 and CUDA FP32."""
    needs_cuda()
    full_only('CPU FP64 grating runs at the full duration')
    mesh = CASES['G3-08']['fixture']['mesh_sequence_um'][0]
    cpu = grating_row(polarization, angle, wavelength, mesh, 'cpu', 'float64', 'subpixel')
    cuda = grating_row(polarization, angle, wavelength, mesh, 'cuda', 'float32', 'subpixel')
    worst = 0.
    for m, entry in cpu['orders'].items():
        for name in ('transmission', 'reflection'):
            a, b = entry[name]['efficiency'], cuda['orders'][m][name]['efficiency']
            worst = max(worst, abs(b-a)/max(abs(a), 1e-6))
    record('G3-08', grating_row_id(cpu), dict(cpu, judged_against_torcwa=False, failures=check_grating_row(cpu)))
    record('G3-08', grating_row_id(cuda), dict(cuda, layer_a_max_relative_difference=worst, layer_a_rtol=LAYER_A_RTOL))
    assert worst <= LAYER_A_RTOL, f'layer A grating {polarization} {angle:g} deg {wavelength} um: {worst:.2e}'


@pytest.mark.parametrize('polarization', ['TE', 'TM'])
def test_g3_08_staircase_and_duration_controls(polarization):
    """Recorded-only context rows: the staircase interface at both meshes and the shorter duration, normal incidence."""
    needs_cuda()
    full_only('recorded control rows')
    f = CASES['G3-08']['fixture']
    wavelength = f['wavelengths_um'][1]
    for mesh in f['mesh_sequence_um']:
        row = grating_row(polarization, 0., wavelength, mesh, 'cuda', 'float32', 'staircase')
        record('G3-08', grating_row_id(row), dict(row, judged_against_torcwa=False, failures=check_grating_row(row)))
    row = grating_row(polarization, 0., wavelength, f['mesh_sequence_um'][0], 'cuda', 'float32', 'subpixel', duration_fs=f['fast_duration_fs'])
    record('G3-08', grating_row_id(row), dict(row, judged_against_torcwa=False, failures=check_grating_row(row)))


# ----------------------------------------------------------------------------
# G3-13: curved-interface convergence on the G3-04 cylinder
# ----------------------------------------------------------------------------
def convergence_orders(rows):
    errors = [r['max_relative_error'] for r in rows]
    return [math.log2(errors[i]/errors[i+1]) if errors[i+1] > 0 else None for i in range(len(errors)-1)]


@pytest.mark.parametrize('polarization', ['TM', 'TE'])
def test_g3_13_mesh_and_interface_convergence(polarization):
    f = CASES['G3-13']['fixture']
    meshes = f['mesh_sequence_um'] if FULL else f['mesh_sequence_um'][:2]
    base = f['mesh_sequence_um'][0]
    rows = {}
    for interface in ('staircase', 'subpixel'):
        rows[interface] = cylinder_rows(polarization, meshes, 'cpu', 'float64', interface)
        for row in rows[interface]:
            record('G3-13', f'mesh/{polarization}/{interface}/h={row["mesh_um"]}', dict(row, convergence_order_estimates=convergence_orders(rows[interface])))
    staircase, subpixel = rows['staircase'][0], rows['subpixel'][0]
    verdict = dict(polarization=polarization, base_mesh_um=base, staircase_max_relative_error=staircase['max_relative_error'],
                   subpixel_max_relative_error=subpixel['max_relative_error'],
                   staircase_order_estimates=convergence_orders(rows['staircase']), subpixel_order_estimates=convergence_orders(rows['subpixel']),
                   staircase_wall_s=[r['wall_s'] for r in rows['staircase']], subpixel_wall_s=[r['wall_s'] for r in rows['subpixel']],
                   subpixel_at_h_beats_staircase_at_h=subpixel['max_relative_error'] < staircase['max_relative_error'],
                   subpixel_at_h_error_below_staircase_at_half_h=subpixel['max_relative_error'] < rows['staircase'][1]['max_relative_error'],
                   subpixel_at_h_wall_below_staircase_at_half_h=subpixel['wall_s'] < rows['staircase'][1]['wall_s'])
    record('G3-13', f'verdict/{polarization}', verdict)
    assert verdict['subpixel_at_h_beats_staircase_at_h'], verdict


@pytest.mark.parametrize('polarization', ['TM', 'TE'])
def test_g3_13_subcell_shift(polarization):
    f = CASES['G3-13']['fixture']
    mesh = f['shift_mesh_um']
    for interface in ('staircase', 'subpixel'):
        rows = []
        for fraction in f['shift_fractions']:
            center = (fraction*mesh, 0., 0.)
            row = cylinder_rows(polarization, [mesh], 'cpu', 'float64', interface, center)[0]
            rows.append(row)
            record('G3-13', f'shift/{polarization}/{interface}/h={mesh}/shift={fraction:g}h', row)
        spread = max(r['max_relative_error'] for r in rows)-min(r['max_relative_error'] for r in rows)
        widths = np.array([r['width_um'] for r in rows])
        record('G3-13', f'shift/{polarization}/{interface}/summary', dict(polarization=polarization, interface=interface, mesh_um=mesh,
               max_relative_error_by_shift=[r['max_relative_error'] for r in rows], max_relative_error_spread=spread,
               width_variation_relative=float((widths.max(0)-widths.min(0)).max()/widths.mean())))


def test_g3_13_smoothing_width_sweep():
    """Differentiable-solid transition width at fixed mesh, run through the standard TFSF solver."""
    f = CASES['G3-13']['fixture']
    g = CASES['G3-04']['fixture']['cylinder']
    mesh = f['smoothing_mesh_um']
    for polarization in ('TM', 'TE'):
        project = cylinder_project(mesh, polarization)
        empty = project.model_copy(deep=True)
        empty.structures = []
        reference, _ = run(empty, ('cylinder', polarization, mesh, 'cpu', 'float64', 'staircase', (0., 0., 0.), 'empty'))
        staircase, _ = run(project, ('cylinder', polarization, mesh, 'cpu', 'float64', 'staircase', (0., 0., 0.)))
        wavelengths, sharp = cylinder_width(staircase, reference)
        mie = cylinder_cross_sections(wavelengths, g['radius_um'], g['index'], polarization)['scattering']
        image = solver_module.voxelize(project)[0]
        solid = DifferentiableSolid.cylinder(torch.tensor(g['radius_um'], dtype=torch.float64), torch.tensor(4., dtype=torch.float64),
                                             epsilon=torch.tensor(g['index']**2, dtype=torch.float64))
        rows = []
        for fraction in f['smoothing_width_fractions']:
            width = fraction*mesh
            epsilon = smooth_geometry_epsilon(project.region, [solid], width=width).numpy()
            with permittivity_override(epsilon):
                sample, wall = run(project)
            _, measured = cylinder_width(sample, reference)
            error = measured/mie-1
            row = dict(polarization=polarization, mesh_um=mesh, width_fraction=fraction, width_um=width, width_result_um=listed(measured),
                       mie_um=listed(mie), relative_error=listed(error), max_relative_error=float(abs(error).max()),
                       max_relative_difference_from_staircase=float(abs(measured/sharp-1).max()),
                       samples_differing_from_staircase=int((abs(epsilon-image) > 1e-9).sum()), wall_s=wall)
            if fraction <= 1e-3:
                # Off the contour the vanishing width must reproduce the staircase image exactly; samples on the
                # contour take the documented half value and are counted, and their effect on the width is recorded.
                on_contour = np.zeros(image.shape, dtype=bool)
                for c, component in enumerate(('Ex', 'Ey', 'Ez')):
                    axes = solver_module.field_axes(project.region, component)
                    radius = np.hypot(axes[0][:, None, None], axes[1][None, :, None])
                    on_contour[..., c] = abs(radius-g['radius_um']) < 1e-9
                differing = abs(epsilon-image) > 1e-9
                row['samples_on_contour'] = int(on_contour.sum())
                assert not (differing & ~on_contour).any(), 'the vanishing-width image differs from the staircase image off the contour'
                assert np.allclose(epsilon[on_contour], (g['index']**2+1)/2), 'contour samples must take the half value'
            rows.append(row)
            record('G3-13', f'smoothing/{polarization}/h={mesh}/w={fraction:g}h', row)
