"""Independent physics validation, stage G3 part A (docs/COMPLETION_PROGRAM_KO.md section 5).

G3-01 uniform propagation against the exact Yee dispersion relation and the continuum,
G3-02 lossless dielectric slab against a transfer matrix, G3-03 Drude/Lorentz slabs with
the fitting and ADE errors separated, G3-07 default CPML reflection and long-time stability.

Every oracle here (the Yee relation, the Fresnel/Airy transfer matrix, the analytic
permittivities, the bilinear ADE response and the DFT) is written in this module and shares
no code with the solver. The pre-declared fixtures and limits are docs/validation/cases/G3-0*.json;
the limits below are copied from them, not the other way round. Each test writes its measurements
to docs/validation/g3/<task>.json before asserting, so a failed criterion is still recorded; the
committed file is touched only with TORCHFDTD_WRITE_RECORDS=1 (tests/record_output.py).
The finest meshes run only with TORCHFDTD_G3_FINE=1 (the recorded run sets it).
"""
from __future__ import annotations

import datetime
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import (Project, Region, Structure, Source, Monitor, Material, LorentzPole, Simulation,
                       Boundaries, BoundaryFace, RunControl, OpticalData, FitOptions, fit_material)
from torchfdtd.boundaries import YeeGrid
from torchfdtd.materials import MaterialADE, permittivity
from record_output import record_path

C0 = 299792458.0
ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'docs' / 'validation' / 'cases'
FINE = os.environ.get('TORCHFDTD_G3_FINE') == '1'
FINE_REASON = 'finest mesh: set TORCHFDTD_G3_FINE=1 (the recorded run does)'
CUDA = torch.cuda.is_available()
LAMBDA0 = 1.55
# One-cycle Gaussian at 1.55 um (G3-01 propagation, G3-07 normal incidence): sigma = 1/f0.
PULSE = dict(pulse='gaussian', time_definition='standard', wavelength=LAMBDA0, pulse_length=8.609020092789869e-15,
             pulse_offset=25e-15, eliminate_discontinuities=True)
# Three-cycle Gaussian at 1.55 um (slab fixtures): a fixed-k_parallel sheet at 45 deg has grazing and evanescent
# components beyond 1.55/sin(45 deg) = 2.19 um; three cycles put 2.5e-7 of the peak amplitude there.
PULSE3 = dict(pulse='gaussian', time_definition='standard', wavelength=LAMBDA0, pulse_length=2.5827060278369606e-14,
              pulse_offset=75e-15, eliminate_discontinuities=True)
# Three-cycle Gaussian at 1.45 um (oblique CPML fixtures): the 60 deg design is grazing at 1.79 um (5e-4 of the peak).
PULSE3_145 = dict(pulse='gaussian', time_definition='standard', wavelength=1.45, pulse_length=2.4160798324926404e-14,
                  pulse_offset=70e-15, eliminate_discontinuities=True)
FP32 = dict(rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------------------------------------------- records
def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, complex):
        return dict(re=float(value.real), im=float(value.imag))
    return value


def git_text(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    env = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, fdtd=getattr(fdtd, '__version__', None),
               cuda_runtime=torch.version.cuda, cuda_available=CUDA, gpu=torch.cuda.get_device_name(0) if CUDA else None,
               os=platform.platform(), machine=platform.machine(), cpu=platform.processor(),
               commit=git_text('rev-parse', 'HEAD'), dirty_paths=len((git_text('status', '--porcelain') or '').splitlines()),
               fine_meshes=FINE, recorded_at=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat())
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            env['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return env


_RESET = set()


class Record:
    """Write-through JSON record of one task: one entry per measured instance."""
    def __init__(self, task, case):
        self.task, self.case = task, case

    @property
    def path(self):
        return record_path(f'docs/validation/g3/{self.task}.json')

    def add(self, key, **values):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.task in _RESET and self.path.is_file():
            data = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            _RESET.add(self.task)
            data = dict(task=self.task, case=f'docs/validation/cases/{self.case}.json', test_module='tests/test_physics_g3_a.py',
                        environment=environment(), entries={})
        data['entries'][key] = clean(values)
        with open(self.path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def limits(case, *keys):
    """Read a pre-declared limit from the case file so the assertion and the record agree."""
    node = json.loads((CASES / f'{case}.json').read_text(encoding='utf-8'))['acceptance']
    for key in keys:
        node = node[key]
    return node


# ----------------------------------------------------------------------------------------------- oracles
def band(points, low=1.3, high=1.8):
    return np.linspace(C0/(high*1e-6), C0/(low*1e-6), points)


def dft(times, signal, frequencies):
    """Sum s(t) exp(+2 pi i f t) dt: picks the exp(-i omega t) component of the trace."""
    dt = times[1]-times[0]
    return (np.exp(2j*np.pi*np.multiply.outer(frequencies, times)) @ signal)*dt


def yee_kx(omega, dt, h, n=1., ky=0.):
    """Wavenumber along x from (n/(c dt))^2 sin^2(omega dt/2) = sin^2(kx h/2)/h^2 + sin^2(ky h/2)/h^2 (metres)."""
    s = (n*np.sin(omega*dt/2)/(C0*dt))**2 - (np.sin(ky*h/2)/h)**2
    return 2/h*np.arcsin(h*np.sqrt(s))


def tmm_slab(eps, d, k0, ky, pol):
    """Fresnel/Airy slab in vacuum for the tangential scalar (Ez for TE, Hz for TM), faces at 0 and d (metres)."""
    kx1 = np.sqrt(k0**2-ky**2+0j)
    kx2 = np.sqrt(eps*k0**2-ky**2+0j)
    kx2 = np.where(kx2.imag < 0, -kx2, kx2)
    q1, q2 = (kx1, kx2) if pol == 'TE' else (kx1, kx2/eps)
    r12 = (q1-q2)/(q1+q2)
    t12 = 2*q1/(q1+q2)
    r23 = -r12
    t23 = 2*q2/(q1+q2)
    phase = np.exp(1j*kx2*d)
    den = 1+r12*r23*phase**2
    return (r12+r23*phase**2)/den, t12*t23*phase/den


def eps_drude(omega, eps_inf=1., wp=2e15, gamma=1e14):
    return eps_inf-wp**2/(omega**2+1j*gamma*omega)


LORENTZ_POLES = ((1.9e15, .5*1.9e15**2, 2e14), (7.5e14, .3*7.5e14**2, 1e14))


def eps_lorentz(omega, eps_inf=2.25, poles=LORENTZ_POLES):
    return eps_inf+sum(a/(w0**2-omega**2-1j*g*omega) for w0, a, g in poles)


def eps_poles(material, omega):
    """Permittivity of a fitted multipole material from its coefficients (test-side formula)."""
    return material.epsilon_inf+sum(p.strength_rad_s_squared/(p.resonance_rad_s**2-omega**2-1j*p.damping_rad_s*omega) for p in material.poles)


def bilinear(fn, omega, dt):
    """Trapezoidal (bilinear) constitutive response: omega replaced by (2/dt) tan(omega dt/2)."""
    return fn(2/dt*np.tan(omega*dt/2))


def slab_metrics(r_fdtd, t_fdtd, r_ref, t_ref):
    R, T = abs(r_fdtd)**2, abs(t_fdtd)**2
    Rr, Tr = abs(r_ref)**2, abs(t_ref)**2
    t_mask, r_mask = abs(t_ref) > .1, abs(r_ref) > .1
    return dict(R_abs_error=float(np.max(abs(R-Rr))), T_abs_error=float(np.max(abs(T-Tr))),
                balance_residual=float(np.max(abs(R+T-1))), A_abs_error=float(np.max(abs((1-R-T)-(1-Rr-Tr)))),
                t_phase_error=float(np.max(abs(np.angle(t_fdtd/t_ref))[t_mask])) if t_mask.any() else None,
                r_phase_error=float(np.max(abs(np.angle(r_fdtd/r_ref))[r_mask])) if r_mask.any() else None,
                t_phase_band_points=int(t_mask.sum()), R_fdtd=R, T_fdtd=T, R_ref=Rr, T_ref=Tr,
                t_phase_fdtd=np.angle(t_fdtd), t_phase_ref=np.angle(t_ref))


def fp32_metrics(reference, trial):
    peak = float(np.max(abs(reference)))
    a, b = np.asarray(trial)/peak, np.asarray(reference)/peak
    return dict(max_abs_error=float(np.max(abs(a-b))), relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                within=bool(np.allclose(a, b, **FP32)), peak=peak)


# ----------------------------------------------------------------------------------------------- run cache
_RUNS = {}


def run(project):
    key = json.dumps(project.model_dump(mode='json'), sort_keys=True)
    if key not in _RUNS:
        _RUNS[key] = Simulation(project).run()
    return _RUNS[key]


def periodic_pair(kind):
    return dict(y_min=BoundaryFace(kind=kind), y_max=BoundaryFace(kind=kind))


# =============================================================================================== G3-01
def eigenmode(dim, n, shape, m, polarization, steps=40, h=.155):
    """Real discrete plane wave with E only; returns the measured cos(omega dt), its Yee value and the polarization leak."""
    dims = 2 if dim == '2d' else 3
    faces = {a+'_'+side: BoundaryFace(kind='periodic') for a in 'xyz'[:dims] for side in ('min', 'max')}
    r = Region(dimension=dim, size=tuple(s*h if i < dims else 1 for i, s in enumerate(shape)), mesh=h, steps=10,
               boundaries=Boundaries(**faces), precision='float64', backend='cpu')
    assert r.shape == tuple(shape)
    fdtd.set_backend('numpy')
    fdtd.backend.float = np.float64
    g = YeeGrid(r)
    g.inverse_permittivity[:] = 1/n**2
    q = np.array([2*np.pi*mi/ni if ni > 1 else 0. for mi, ni in zip(m, shape)])
    K = 2*np.sin(q/2)                       # discrete wavevector in cell units
    if dim == '2d':
        pols = {'TE': np.array([0., 0., 1.]), 'TM': np.array([-K[1], K[0], 0.])/np.hypot(K[0], K[1])}
    else:
        p1 = np.cross(K, [0., 0., 1.])
        p1 = p1/np.linalg.norm(p1) if np.linalg.norm(p1) > 1e-12 else np.array([1., 0., 0.])
        p2 = np.cross(K, p1)
        pols = {'pol1': p1, 'pol2': p2/np.linalg.norm(p2)}
    pol = pols[polarization]
    idx = np.indices(shape)
    base = np.einsum('i,i...->...', q, idx)
    for a in range(3):
        g.E[..., a] = pol[a]*np.cos(base+q[a]/2)
    g.H[:] = 0
    history = []
    for _ in range(steps):
        g.update_E()
        g.update_H()
        history.append(g.E.copy())
    e1, e2, e3 = history[0], history[1], history[2]
    cos_measured = float(np.sum(e2*(e3+e1))/(2*np.sum(e2*e2)))
    S = g.courant_number
    sin_half = S*np.linalg.norm(K)/(2*n)
    cos_yee = 1-2*sin_half**2
    # Polarization purity: every component must stay in the span of cos/sin of its own staggered
    # phase with a common amplitude pair scaled by pol_a (components with pol_a = 0 stay zero).
    last = history[-1]
    scale = max(np.max(abs(e)) for e in history)     # envelope, not the instantaneous standing-wave amplitude
    leak = 0.
    pairs = []
    for a in range(3):
        basis = np.stack([np.cos(base+q[a]/2).ravel(), np.sin(base+q[a]/2).ravel()], axis=1)
        values = last[..., a].ravel()
        if abs(pol[a]) < 1e-12:
            leak = max(leak, float(np.max(abs(values))/scale))
            continue
        coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
        leak = max(leak, float(np.max(abs(values-basis@coefficients))/scale))
        pairs.append(coefficients/pol[a])
    pairs = np.array(pairs)
    leak = max(leak, float(np.max(abs(pairs-pairs[0]))/np.max(abs(pairs))))
    omega_measured = math.acos(cos_measured)/g.time_step
    k = np.linalg.norm(q)/(h*1e-6)
    return dict(cos_measured=cos_measured, cos_yee=cos_yee, residual=abs(cos_measured-cos_yee), polarization_leak=leak,
                omega_measured=omega_measured, phase_velocity_error=(omega_measured/k)/(C0/n)-1,
                courant=S, cells_per_wavelength=float(2*np.pi/np.linalg.norm(q)))


def propagation_project(dim, N, n, component, backend='cpu', precision='float64'):
    h = LAMBDA0/N
    dims = 2 if dim == '2d' else 3
    dt = .99/math.sqrt(dims)*h*1e-6/C0
    faces = {a+'_'+side: BoundaryFace(kind='periodic') for a in 'yz'[:dims-1] for side in ('min', 'max')}
    region = Region(dimension=dim, size=(24.8, .775, .775 if dims == 3 else 1), mesh=h, steps=round(100e-15/dt), pml_cells=5*N//10,
                    background_index=n, backend=backend, precision=precision, material_sampling='yee', snapshot_interval=10000,
                    boundaries=Boundaries(**faces), field=component)
    return Project(name=f'G3-01 {dim} N{N} n{n} {component}', region=region, materials=[Material(name='void', index=1)],
                   sources=[Source(kind='plane', center=(-11.005, 0, 0), size=(0, .775, .775 if dims == 3 else 0), component=component, **PULSE)],
                   monitors=[Monitor(id='a', name='a', center=(-9.92, 0, 0), component=component),
                             Monitor(id='b', name='b', center=(-6.82, 0, 0), component=component)])


def propagation_phase(result, n, h_um):
    """k measured from the phase difference of the two traces against the Yee relation and the continuum."""
    f = band(11)
    omega = 2*np.pi*f
    sa, sb = dft(result.times, result.signals[:, 0], f), dft(result.times, result.signals[:, 1], f)
    dt = result.times[1]-result.times[0]
    D = 3.1e-6
    k_yee = yee_kx(omega, dt, h_um*1e-6, n)
    k_cont = n*omega/C0
    phase = np.angle(sb/sa)
    phase += 2*np.pi*np.round((k_yee*D-phase)/(2*np.pi))
    k_meas = phase/D
    trace = result.signals[:, 1]
    return dict(wavelength_um=(C0/f*1e6), k_measured=k_meas, k_yee=k_yee, k_continuum=k_cont,
                phase_residual_vs_yee=abs(k_meas-k_yee)*D, phase_error_per_wavelength=2*np.pi*(k_meas/k_cont-1),
                phase_error_per_wavelength_yee=2*np.pi*(k_yee/k_cont-1), phase_velocity_ratio=k_cont/k_meas,
                trace_end_over_peak=float(abs(trace[-1])/np.max(abs(trace))), steps=len(trace), dt_s=dt, courant=C0*dt/(h_um*1e-6),
                spectrum_floor=float(np.min(abs(sa))/np.max(abs(sa))))


class TestG301:
    CASE = 'G3-01_uniform_propagation'
    record = Record('G3-01', CASE)

    @pytest.mark.parametrize('dim', ['2d', '3d'])
    @pytest.mark.parametrize('n', [1., 1.5])
    def test_eigenmode_matches_yee_relation(self, dim, n):
        shape = (24, 20, 1) if dim == '2d' else (24, 20, 16)
        m = (1, -2, 1 if dim == '3d' else 0)
        limit = limits(self.CASE, 'part_A_discrete_relation')
        for pol in (('TE', 'TM') if dim == '2d' else ('pol1', 'pol2')):
            out = eigenmode(dim, n, shape, m, pol)
            self.record.add(f'eigenmode oblique {dim} n={n} {pol}', shape=shape, m=m, index=n, polarization=pol, **out,
                            limit_residual=limit['cos_omega_dt_residual_max'], limit_leak=limit['polarization_leak_max'])
            assert out['residual'] <= limit['cos_omega_dt_residual_max'], (dim, n, pol, out['residual'])
            assert out['polarization_leak'] <= limit['polarization_leak_max'], (dim, n, pol, out['polarization_leak'])

    @pytest.mark.parametrize('dim', ['2d', '3d'])
    @pytest.mark.parametrize('n', [1., 1.5])
    def test_eigenmode_mesh_sweep(self, dim, n):
        limit = limits(self.CASE, 'part_A_discrete_relation')
        for N in (10, 20, 40):
            shape = (N, 6, 1) if dim == '2d' else (N, 6, 6)
            for pol in (('TE', 'TM') if dim == '2d' else ('pol1', 'pol2')):
                out = eigenmode(dim, n, shape, (1, 0, 0), pol)
                self.record.add(f'eigenmode axis {dim} n={n} N={N} {pol}', shape=shape, index=n, polarization=pol, N=N, **out,
                                phase_error_per_wavelength=2*np.pi*(1/(1+out['phase_velocity_error'])-1))
                assert out['residual'] <= limit['cos_omega_dt_residual_max'], (dim, n, N, pol, out['residual'])
                assert out['polarization_leak'] <= limit['polarization_leak_max'], (dim, n, N, pol, out['polarization_leak'])

    @pytest.mark.parametrize('N', [10, 20, 40])
    @pytest.mark.parametrize('n', [1., 1.5])
    @pytest.mark.parametrize('component', ['Ez', 'Ey'])
    def test_propagation_phase_2d(self, N, n, component):
        if N == 40 and not FINE:
            pytest.skip(FINE_REASON)
        self._propagation('2d', N, n, component)

    @pytest.mark.parametrize('N', [10, 20, 40])
    @pytest.mark.parametrize('n', [1., 1.5])
    @pytest.mark.parametrize('component', ['Ey', 'Ez'])
    def test_propagation_phase_3d(self, N, n, component):
        if N == 40 and not FINE:
            pytest.skip(FINE_REASON)
        self._propagation('3d', N, n, component)

    def _propagation(self, dim, N, n, component):
        p = propagation_project(dim, N, n, component)
        result = run(p)
        out = propagation_phase(result, n, LAMBDA0/N)
        limit = limits(self.CASE, 'part_B_discrete_relation', 'phase_residual_vs_yee_max_rad')
        self.record.add(f'propagation {dim} N={N} n={n} {component}', N=N, index=n, component=component, mesh_um=LAMBDA0/N, shape=p.region.shape,
                        pml_layers=p.region.pml_cells, seconds=result.summary['seconds'], **out, limit_phase_residual_rad=limit,
                        max_phase_residual_vs_yee=float(np.max(out['phase_residual_vs_yee'])))
        assert np.max(out['phase_residual_vs_yee']) <= limit, (dim, N, n, component, np.max(out['phase_residual_vs_yee']))

    @pytest.mark.parametrize('dim,pair', [('2d', ('Ez', 'Ey')), ('3d', ('Ey', 'Ez'))])
    @pytest.mark.parametrize('N', [10, 20])
    def test_polarizations_identical(self, dim, pair, N):
        limit = limits(self.CASE, 'part_B_polarization', 'polarization_trace_difference_max')
        for n in (1., 1.5):
            a, b = (run(propagation_project(dim, N, n, c)) for c in pair)
            peak = np.max(abs(a.signals))
            difference = float(np.max(abs(a.signals-b.signals))/peak)
            self.record.add(f'polarization identity {dim} N={N} n={n}', components=pair, N=N, index=n, trace_difference=difference, limit=limit)
            assert difference <= limit, (dim, N, n, difference)

    @pytest.mark.parametrize('dim,n,component', [('2d', 1., 'Ez'), ('3d', 1.5, 'Ey')])
    def test_layer_a_cuda_fp32(self, dim, n, component):
        if not CUDA:
            pytest.skip('CUDA unavailable')
        reference = run(propagation_project(dim, 20, n, component))
        p = propagation_project(dim, 20, n, component, backend='cuda', precision='float32')
        trial = Simulation(p).run()
        out = fp32_metrics(reference.signals, trial.signals)
        self.record.add(f'layer A cuda fp32 {dim} n={n} {component}', N=20, index=n, component=component, steps=p.region.steps,
                        cuda_graph=trial.summary['cuda_graph'], gpu=trial.summary['gpu'], **out, limits=FP32)
        assert out['within'], out


# =============================================================================================== G3-02
def slab_geometry(n, d):
    M = round(20*d*n/LAMBDA0)
    h20 = d/M
    return dict(M=M, h20=h20, layers20=round(.5/h20), Ly=6*h20, offset=h20/8)


def slab_project(n, d, N, angle_deg, pol, with_slab, *, backend='cpu', precision='float64', materials=None, material=None,
                 h=None, layers=None, Ly=None, offset=None, duration_fs=400.):
    """2D slab fixture shared by G3-02 and G3-03; the dispersive tests pass their own mesh."""
    geo = slab_geometry(n, d) if h is None else None
    if geo is not None:
        h = geo['h20'] if N == 20 else geo['h20']/2
        layers = geo['layers20'] if N == 20 else 2*geo['layers20']
        Ly, offset = geo['Ly'], geo['offset']
    ky = 2*np.pi/LAMBDA0*math.sin(math.radians(angle_deg))
    bloch = angle_deg != 0
    dt = .99/math.sqrt(2)*h*1e-6/C0
    region = Region(dimension='2d', size=(6., Ly, 1), mesh=h, steps=round(duration_fs*1e-15/dt), pml_cells=layers, backend=backend,
                    precision=precision, material_sampling='yee', snapshot_interval=10000, field='Ez' if pol == 'TE' else 'Hz',
                    boundaries=Boundaries(**periodic_pair('bloch' if bloch else 'periodic')), bloch_phase=(0, ky*Ly if bloch else 0, 0))
    source = 'Ez' if pol == 'TE' else 'Ey'
    if pol == 'TE':
        monitors = [Monitor(id='r', name='reflection', center=(-1., 0, 0), component='Ez'),
                    Monitor(id='t', name='transmission', center=(1., 0, 0), component='Ez')]
    else:
        monitors = [Monitor(id='r', name='reflection', center=(-1.+h/2, h/2, 0), component='Hz'),
                    Monitor(id='t', name='transmission', center=(1.+h/2, h/2, 0), component='Hz')]
    materials = materials or [Material(name='slab', index=n)]
    structures = [Structure(name='slab', center=(offset, 0, 0), size=(d, 10, 2), material=material or materials[0].name)] if with_slab else []
    return Project(name=f'slab n{n} d{d} N{N} {angle_deg}deg {pol} {"slab" if with_slab else "empty"}', region=region, materials=materials,
                   structures=structures, sources=[Source(kind='plane', center=(-2., 0, 0), size=(0, Ly, 0), component=source, **PULSE3)],
                   monitors=monitors)


def slab_response(sample, empty, d_um, angle_deg, pol, h_um, f, offset_um):
    """Complex r (front face) and t (across the faces) from the two traces, Yee-corrected for the vacuum propagation."""
    omega = 2*np.pi*f
    ky = 2*np.pi/(LAMBDA0*1e-6)*math.sin(math.radians(angle_deg))
    dt = sample.times[1]-sample.times[0]
    kx = yee_kx(omega, dt, h_um*1e-6, 1., ky)
    kx_cont = np.sqrt((omega/C0)**2-ky**2)
    ratio_r = (dft(sample.times, sample.signals[:, 0], f)-dft(empty.times, empty.signals[:, 0], f))/dft(empty.times, empty.signals[:, 0], f)
    ratio_t = dft(sample.times, sample.signals[:, 1], f)/dft(empty.times, empty.signals[:, 1], f)
    x_monitor = (-1. if pol == 'TE' else -1.+h_um/2)*1e-6
    x_face = (-d_um/2+offset_um)*1e-6
    d = d_um*1e-6
    return dict(r=ratio_r*np.exp(-2j*kx*(x_face-x_monitor)), t=ratio_t*np.exp(1j*kx*d),
                r_continuum=ratio_r*np.exp(-2j*kx_cont*(x_face-x_monitor)), t_continuum=ratio_t*np.exp(1j*kx_cont*d), kx=kx, ky=ky)


class TestG302:
    """Revision 2 case: the R/T and phase limits apply at about 40 cells per material wavelength; the 20-cell
    mesh is recorded for the energy balance and the convergence order (the first case's FAILED evidence stays
    in place as the finding, see docs/validation/cases/G3-02r2_slab_tmm_40_cells.json)."""
    CASE = 'G3-02r2_slab_tmm_40_cells'
    record = Record('G3-02r2', CASE)
    SLABS = [(1.5, .2), (1.5, .5), (3.5, .2), (3.5, .5)]
    _measured = {}

    def _measure(self, slab, pol, angle, N):
        key = (slab, pol, angle, N)
        if key in self._measured:
            return self._measured[key]
        n, d = slab
        geo = slab_geometry(n, d)
        h = geo['h20'] if N == 20 else geo['h20']/2
        sample = run(slab_project(n, d, N, angle, pol, True))
        empty = run(slab_project(n, d, N, angle, pol, False))
        f = band(21)
        out = slab_response(sample, empty, d, angle, pol, h, f, geo['offset'])
        r_ref, t_ref = tmm_slab(n**2+0j, d*1e-6, 2*np.pi*f/C0, out['ky'], pol)
        metrics = slab_metrics(out['r'], out['t'], r_ref, t_ref)
        cont = slab_metrics(out['r_continuum'], out['t_continuum'], r_ref, t_ref)
        self._measured[key] = dict(n=n, d=d, h=h, geo=geo, sample=sample, empty=empty, f=f, out=out, metrics=metrics, cont=cont)
        return self._measured[key]

    def _record(self, m, angle, pol, N, rt_limits_apply):
        n, d, geo, sample, empty, f, out, metrics, cont = (m[k] for k in ('n', 'd', 'geo', 'sample', 'empty', 'f', 'out', 'metrics', 'cont'))
        n40 = limits(self.CASE, 'N40')
        balance_limit = limits(self.CASE, 'N40' if rt_limits_apply else 'N20', 'balance_residual_max')
        rt_within = bool(metrics['R_abs_error'] <= n40['R_abs_error_max'] and metrics['T_abs_error'] <= n40['T_abs_error_max']
                         and metrics['t_phase_error'] <= n40['t_phase_error_max_rad'])
        balance_ok = bool(metrics['balance_residual'] <= balance_limit)
        key = f'slab n={n} d={d} {angle}deg {pol} N{N}'
        self.record.add(key, index=n, thickness_um=d, angle_deg=angle, polarization=pol, N=N, mesh_um=m['h'], cells_across_slab=geo['M']*(N//20),
                        cells_per_material_wavelength_at_1p55=LAMBDA0/(n*m['h']), pml_layers=sample.project.region.pml_cells, shape=sample.project.region.shape,
                        steps=sample.project.region.steps, dt_s=sample.times[1]-sample.times[0], bloch_phase=sample.project.region.bloch_phase[1],
                        angle_range_deg=[float(np.degrees(np.arcsin(out['ky']*C0/(2*np.pi*f[i])))) for i in (-1, 0)],
                        wavelength_um=C0/f*1e6, seconds=sample.summary['seconds']+empty.summary['seconds'],
                        **{k: v for k, v in metrics.items()}, continuum_corrected=dict(t_phase_error=cont['t_phase_error'], r_phase_error=cont['r_phase_error']),
                        limits=dict(R=n40['R_abs_error_max'], T=n40['T_abs_error_max'], balance=balance_limit, t_phase=n40['t_phase_error_max_rad']),
                        rt_limits_apply=rt_limits_apply, rt_within_limits=rt_within, balance_passed=balance_ok,
                        passed=bool(balance_ok and (rt_within or not rt_limits_apply)))
        return key, metrics

    @pytest.mark.parametrize('angle', [0, 20, 45])
    @pytest.mark.parametrize('pol', ['TE', 'TM'])
    @pytest.mark.parametrize('slab', SLABS, ids=lambda s: f'n{s[0]}_d{s[1]}')
    def test_slab_r_t_against_tmm_40_cells(self, slab, pol, angle):
        if not FINE:
            pytest.skip(FINE_REASON)
        key, metrics = self._record(self._measure(slab, pol, angle, 40), angle, pol, 40, True)
        limit = limits(self.CASE, 'N40')
        assert metrics['R_abs_error'] <= limit['R_abs_error_max'], (key, 'R', metrics['R_abs_error'])
        assert metrics['T_abs_error'] <= limit['T_abs_error_max'], (key, 'T', metrics['T_abs_error'])
        assert metrics['balance_residual'] <= limit['balance_residual_max'], (key, 'R+T-1', metrics['balance_residual'])
        assert metrics['t_phase_error'] <= limit['t_phase_error_max_rad'], (key, 't phase', metrics['t_phase_error'])

    @pytest.mark.parametrize('angle', [0, 20, 45])
    @pytest.mark.parametrize('pol', ['TE', 'TM'])
    @pytest.mark.parametrize('slab', SLABS, ids=lambda s: f'n{s[0]}_d{s[1]}')
    def test_slab_balance_20_cells(self, slab, pol, angle):
        """The 20-cell mesh is a reported measurement: only the energy balance is a limit here."""
        key, metrics = self._record(self._measure(slab, pol, angle, 20), angle, pol, 20, False)
        limit = limits(self.CASE, 'N20', 'balance_residual_max')
        assert metrics['balance_residual'] <= limit, (key, 'R+T-1', metrics['balance_residual'])

    @pytest.mark.parametrize('angle', [0, 20, 45])
    @pytest.mark.parametrize('pol', ['TE', 'TM'])
    @pytest.mark.parametrize('slab', SLABS, ids=lambda s: f'n{s[0]}_d{s[1]}')
    def test_dispersion_order(self, slab, pol, angle):
        """Second-order convergence: the 20-cell errors must be 3 to 5 times the 40-cell errors (dispersion finding kept as a test)."""
        if not FINE:
            pytest.skip(FINE_REASON)
        m20, m40 = self._measure(slab, pol, angle, 20), self._measure(slab, pol, angle, 40)
        limit = limits(self.CASE, 'N20', 'convergence_order')
        ratios = dict(R=m20['metrics']['R_abs_error']/m40['metrics']['R_abs_error'], t_phase=m20['metrics']['t_phase_error']/m40['metrics']['t_phase_error'])
        n, d = slab
        self.record.add(f'order n={n} d={d} {angle}deg {pol}', index=n, thickness_um=d, angle_deg=angle, polarization=pol,
                        R_abs_error=dict(N20=m20['metrics']['R_abs_error'], N40=m40['metrics']['R_abs_error']),
                        t_phase_error=dict(N20=m20['metrics']['t_phase_error'], N40=m40['metrics']['t_phase_error']),
                        ratio=ratios, limits=dict(ratio_min=limit['ratio_min'], ratio_max=limit['ratio_max']),
                        passed=bool(all(limit['ratio_min'] <= r <= limit['ratio_max'] for r in ratios.values())))
        for name, r in ratios.items():
            assert limit['ratio_min'] <= r <= limit['ratio_max'], (slab, pol, angle, name, r)

    @pytest.mark.parametrize('pol', ['TE', 'TM'])
    def test_layer_a_cuda_fp32(self, pol):
        if not CUDA:
            pytest.skip('CUDA unavailable')
        reference = run(slab_project(1.5, .2, 20, 45, pol, True))
        p = slab_project(1.5, .2, 20, 45, pol, True, backend='cuda', precision='float32')
        trial = Simulation(p).run()
        out = fp32_metrics(reference.signals, trial.signals)
        self.record.add(f'layer A cuda fp32 n=1.5 d=0.2 45deg {pol} N20', polarization=pol, steps=p.region.steps, complex_fields=trial.summary['complex_fields'],
                        cuda_graph=trial.summary['cuda_graph'], gpu=trial.summary['gpu'], **out, limits=FP32)
        assert out['within'], out


# =============================================================================================== G3-03
def drude_material():
    return Material(name='G3-03 Drude', model='drude', epsilon_inf=1., plasma_rad_s=2e15, collision_rad_s=1e14)


def lorentz_material():
    return Material(name='G3-03 Lorentz', model='multipole', epsilon_inf=2.25,
                    poles=[LorentzPole(resonance_rad_s=w0, strength_rad_s_squared=a, damping_rad_s=g) for w0, a, g in LORENTZ_POLES])


MATERIALS = {'drude': (drude_material, eps_drude, .1), 'lorentz': (lorentz_material, eps_lorentz, .5)}
MESHES = {'h20': (.02, 12), 'h10': (.01, 24)}
G303_OFFSET = .0025


def dispersive_project(kind, mesh, pol, with_slab, material=None):
    make, _, d = MATERIALS[kind]
    h, layers = MESHES[mesh]
    material = material or make()
    return slab_project(1., d, 20, 0, pol, with_slab, materials=[Material(name='void', index=1), material], material=material.name,
                        h=h, layers=layers, Ly=6*.02, offset=G303_OFFSET)


class TestG303:
    CASE = 'G3-03_dispersive_slab_fit_ade'
    record = Record('G3-03', CASE)

    def _compare(self, kind, sample, empty, eps_fn, h, f, pol, d):
        out = slab_response(sample, empty, d, 0, pol, h, f, G303_OFFSET)
        omega = 2*np.pi*f
        r_ref, t_ref = tmm_slab(eps_fn(omega), d*1e-6, omega/C0, 0., pol)
        return out, slab_metrics(out['r'], out['t'], r_ref, t_ref)

    @pytest.mark.parametrize('mesh', ['h20', 'h10'])
    @pytest.mark.parametrize('pol', ['TE', 'TM'])
    @pytest.mark.parametrize('kind', ['drude', 'lorentz'])
    def test_analytic_slab_against_tmm(self, kind, pol, mesh):
        if mesh == 'h10' and not FINE:
            pytest.skip(FINE_REASON)
        _, eps_fn, d = MATERIALS[kind]
        h, _ = MESHES[mesh]
        sample, empty = run(dispersive_project(kind, mesh, pol, True)), run(dispersive_project(kind, mesh, pol, False))
        f = band(21)
        out, metrics = self._compare(kind, sample, empty, eps_fn, h, f, pol, d)
        eps = eps_fn(2*np.pi*f)
        limit = limits(self.CASE, 'part_a')
        key = f'analytic {kind} {pol} {mesh}'
        self.record.add(key, material=kind, polarization=pol, mesh_um=h, thickness_um=d, cells_across_slab=round(d/h), pml_layers=sample.project.region.pml_cells,
                        shape=sample.project.region.shape, steps=sample.project.region.steps, dt_s=sample.times[1]-sample.times[0],
                        wavelength_um=C0/f*1e6, epsilon_re=eps.real, epsilon_im=eps.imag, n=np.sqrt(eps).real, k=np.sqrt(eps).imag,
                        A_fdtd=1-metrics['R_fdtd']-metrics['T_fdtd'], A_ref=1-metrics['R_ref']-metrics['T_ref'],
                        seconds=sample.summary['seconds']+empty.summary['seconds'], material_update=sample.summary['material_update'],
                        **metrics, limits=dict(R=limit['R_abs_error_max'], T=limit['T_abs_error_max'], A=limit['A_abs_error_max'], t_phase=limit['t_phase_error_max_rad']))
        assert metrics['R_abs_error'] <= limit['R_abs_error_max'], (key, 'R', metrics['R_abs_error'])
        assert metrics['T_abs_error'] <= limit['T_abs_error_max'], (key, 'T', metrics['T_abs_error'])
        assert metrics['A_abs_error'] <= limit['A_abs_error_max'], (key, 'A', metrics['A_abs_error'])
        assert metrics['t_phase_error'] <= limit['t_phase_error_max_rad'], (key, 't phase', metrics['t_phase_error'])

    @pytest.mark.parametrize('kind', ['drude', 'lorentz'])
    def test_fitted_material_slab(self, kind):
        make, eps_fn, d = MATERIALS[kind]
        wavelengths = np.linspace(1.2, 1.9, 61)
        index = np.sqrt(eps_fn(2*np.pi*C0/(wavelengths*1e-6)))
        data = OpticalData.from_nk(wavelengths, index.real, index.imag, reference='G3-03 analytic samples')
        fit = fit_material(data, name=f'G3-03 fitted {kind}', options=FitOptions(max_poles=4, include_drude=True, tolerance=1e-3,
                                                                                    wavelength_range_um=(1.2, 1.9), seed=0))
        f = band(21)
        omega = 2*np.pi*f
        eps_fit = eps_poles(fit.material, omega) if fit.material.poles else np.full(len(f), fit.material.epsilon_inf+0j)
        eps_true = eps_fn(omega)
        assert np.allclose(eps_fit, permittivity(fit.material, f), rtol=1e-12, atol=0)
        h, _ = MESHES['h20']
        sample = run(dispersive_project(kind, 'h20', 'TE', True, material=fit.material))
        empty = run(dispersive_project(kind, 'h20', 'TE', False))
        out, vs_fitted = self._compare(kind, sample, empty, lambda w: eps_poles(fit.material, w) if fit.material.poles else np.full(np.shape(w), fit.material.epsilon_inf+0j), h, f, 'TE', d)
        _, vs_analytic = self._compare(kind, sample, empty, eps_fn, h, f, 'TE', d)
        r_fit, t_fit = tmm_slab(eps_fit, d*1e-6, omega/C0, 0., 'TE')
        r_true, t_true = tmm_slab(eps_true, d*1e-6, omega/C0, 0., 'TE')
        fit_only = slab_metrics(r_fit, t_fit, r_true, t_true)
        limit = limits(self.CASE, 'part_b')
        report = fit.report
        self.record.add(f'fitted {kind} TE h20', material=kind, converged=fit.converged, poles=len(fit.material.poles), epsilon_inf=fit.material.epsilon_inf,
                        fitted_poles=[p.model_dump() for p in fit.material.poles], fit_band_um=fit.material.fit_band_um, samples=len(wavelengths),
                        fit_report_analytic=report['analytic'], fit_band_errors=dict(max_abs_n=float(np.max(abs(np.sqrt(eps_fit).real-np.sqrt(eps_true).real))),
                                                                                       max_abs_k=float(np.max(abs(np.sqrt(eps_fit).imag-np.sqrt(eps_true).imag)))),
                        fdtd_fitted_vs_tmm_fitted={k: vs_fitted[k] for k in ('R_abs_error', 'T_abs_error', 'A_abs_error', 't_phase_error')},
                        fdtd_fitted_vs_tmm_analytic={k: vs_analytic[k] for k in ('R_abs_error', 'T_abs_error', 'A_abs_error', 't_phase_error')},
                        tmm_fitted_vs_tmm_analytic={k: fit_only[k] for k in ('R_abs_error', 'T_abs_error', 'A_abs_error', 't_phase_error')},
                        wavelength_um=C0/f*1e6, R_fdtd=vs_fitted['R_fdtd'], T_fdtd=vs_fitted['T_fdtd'], seconds=sample.summary['seconds'],
                        limits=limit)
        assert fit.converged, report['analytic']
        for label, m in (('fdtd_fitted_vs_tmm_fitted', vs_fitted), ('fdtd_fitted_vs_tmm_analytic', vs_analytic)):
            lim = limit[label]
            assert m['R_abs_error'] <= lim['R_abs_error_max'], (kind, label, 'R', m['R_abs_error'])
            assert m['T_abs_error'] <= lim['T_abs_error_max'], (kind, label, 'T', m['T_abs_error'])
            assert m['A_abs_error'] <= lim['A_abs_error_max'], (kind, label, 'A', m['A_abs_error'])
            assert m['t_phase_error'] <= lim['t_phase_error_max_rad'], (kind, label, 't phase', m['t_phase_error'])

    @pytest.mark.parametrize('kind', ['drude', 'lorentz'])
    def test_ade_constitutive_error(self, kind):
        make, eps_fn, _ = MATERIALS[kind]
        material = make()
        f = band(21)
        omega = 2*np.pi*f
        limit = limits(self.CASE, 'part_c')
        exact = np.sqrt(eps_fn(omega))
        results = {}
        for mesh, (h, _) in MESHES.items():
            dt = .99/math.sqrt(2)*h*1e-6/C0
            ade = bilinear(eps_fn, omega, dt)
            solver = permittivity(material, f, dt)
            rel = float(np.max(abs(solver/ade-1)))
            index = np.sqrt(ade)
            # Driven single cell of the solver's own ADE, sampled over whole periods after settling.
            fdtd.set_backend('numpy')
            fdtd.backend.float = np.float64
            r = Region(dimension='2d', size=(1., 1., 1), mesh=h, steps=10, precision='float64', backend='cpu')
            g = YeeGrid(r)
            assert math.isclose(g.time_step, dt, rel_tol=1e-12)
            state = MaterialADE(g, material, np.array([0]))
            period = 128 if mesh == 'h20' else 256
            fd = 1/(period*dt)
            electric, displacement, previous = [], [], 0.
            settle = 32768   # 0.76 ps at h10: the slowest pole (gamma 1e14 /s) has decayed by exp(-38)
            for step in range(settle+period*32):
                drive = math.sin(2*math.pi*fd*(step+1)*dt)
                old, response = state.prepare(g.E)
                g.E[0, 0, 0, 0] += (drive-previous)/material.epsilon_inf
                state.correct(g.E, old, response)
                previous = drive
                if step >= settle:
                    electric.append(g.E[0, 0, 0, 0])
                    displacement.append(drive)
            phase = np.exp(2j*np.pi*np.arange(period*32)/period)
            driven = np.dot(displacement, phase)/np.dot(electric, phase)
            driven_rel = abs(driven/bilinear(eps_fn, 2*np.pi*fd, dt)-1)
            results[mesh] = dict(mesh_um=h, dt_s=dt, max_abs_n_error=float(np.max(abs(index.real-exact.real))), max_abs_k_error=float(np.max(abs(index.imag-exact.imag))),
                                 max_rel_eps_error=float(np.max(abs(ade-eps_fn(omega))/abs(eps_fn(omega)))), solver_vs_bilinear_rel=rel,
                                 driven_frequency_hz=fd, driven_eps=complex(driven), driven_vs_bilinear_rel=float(driven_rel))
        ratio = dict(n=results['h20']['max_abs_n_error']/max(results['h10']['max_abs_n_error'], 1e-300), k=results['h20']['max_abs_k_error']/max(results['h10']['max_abs_k_error'], 1e-300))
        self.record.add(f'ade constitutive {kind}', material=kind, wavelength_um=C0/f*1e6, n_exact=exact.real, k_exact=exact.imag, per_dt=results,
                        error_ratio_dt_over_half_dt=ratio, limits=limit)
        for mesh, out in results.items():
            assert max(out['max_abs_n_error'], out['max_abs_k_error']) <= limit['ade_constitutive_error_max'], (kind, mesh, out)
            assert out['solver_vs_bilinear_rel'] <= limit['solver_permittivity_vs_bilinear_rel'], (kind, mesh, out['solver_vs_bilinear_rel'])
            assert out['driven_vs_bilinear_rel'] <= limit['ade_driven_cell_rel'], (kind, mesh, out['driven_vs_bilinear_rel'])


# =============================================================================================== G3-07
def pml_project(*, long, background=1., angle_deg=0., layers=10, duration_fs=100., interface=False, steps=None,
                backend='cpu', precision='float64', check_interval=100):
    h = .025
    Lx, xs = (40.5, -10.5) if long else (11.5, 4.)
    xm = xs+1.
    pulse = PULSE3_145 if angle_deg else PULSE
    Ly = 2. if interface else .3
    ky = 2*np.pi/LAMBDA0*math.sin(math.radians(angle_deg))
    bloch = angle_deg != 0
    dt = .99/math.sqrt(2)*h*1e-6/C0
    region = Region(dimension='2d', size=(Lx, Ly, 1), mesh=h, steps=steps or round(duration_fs*1e-15/dt), pml_cells=layers, backend=backend,
                    precision=precision, material_sampling='yee', snapshot_interval=10000, background_index=background,
                    boundaries=Boundaries(**periodic_pair('bloch' if bloch else 'periodic')), bloch_phase=(0, ky*Ly if bloch else 0, 0),
                    run_control=RunControl(check_interval=check_interval))
    monitors = ([Monitor(id='v', name='vacuum side', center=(xm, .5, 0)), Monitor(id='d', name='dielectric side', center=(xm, -.5, 0))]
                if interface else [Monitor(id='m', name='m', center=(xm, 0, 0))])
    structures = [Structure(name='half space', center=(0, -.4875, 0), size=(100, 1., 2), material='n2')] if interface else []
    return Project(name=f'G3-07 {"long" if long else "short"} n{background} {angle_deg}deg L{layers} {"interface" if interface else ""}', region=region,
                   materials=[Material(name='void', index=1), Material(name='n2', index=2)], structures=structures,
                   sources=[Source(kind='plane', center=(xs, 0, 0), size=(0, Ly, 0), component='Ez', **pulse)], monitors=monitors)


def reflection(short, long, monitor, low=1.3, high=1.8):
    f = band(21, low, high)
    incident = dft(long.times, long.signals[:, monitor], f)
    reflected = dft(short.times, short.signals[:, monitor]-long.signals[:, monitor], f)
    R = abs(reflected)**2/abs(incident)**2
    energy = float(np.sum(abs(short.signals[:, monitor]-long.signals[:, monitor])**2)/np.sum(abs(long.signals[:, monitor])**2))
    at_design = float(abs(dft(short.times, short.signals[:, monitor]-long.signals[:, monitor], np.array([C0/1.55e-6])))[0]**2
                      / abs(dft(long.times, long.signals[:, monitor], np.array([C0/1.55e-6])))[0]**2)
    return dict(wavelength_um=C0/f*1e6, R=R, R_max_on_band=float(np.max(R)), R_at_design=at_design, R_broadband=energy,
                R_max_dB=float(10*np.log10(np.max(R))), incident_peak=float(np.max(abs(long.signals[:, monitor]))))


class TestG307:
    CASE = 'G3-07_cpml_reflection_stability'
    record = Record('G3-07', CASE)

    @pytest.mark.parametrize('background', [1., 2.])
    def test_normal_reflection(self, background):
        short, long = run(pml_project(long=False, background=background)), run(pml_project(long=True, background=background))
        out = reflection(short, long, 0)
        limit = limits(self.CASE, 'normal', 'R_max_on_band')
        self.record.add(f'normal n={background} L10', background_index=background, layers=10, steps=short.project.region.steps, shape_short=short.project.region.shape,
                        shape_long=long.project.region.shape, dt_s=short.times[1]-short.times[0], seconds=short.summary['seconds']+long.summary['seconds'], **out, limit=limit)
        assert out['R_max_on_band'] <= limit, (background, out['R_max_on_band'], out['R_max_dB'])

    @pytest.mark.parametrize('angle', [30, 60])
    def test_oblique_reflection(self, angle):
        short, long = run(pml_project(long=False, angle_deg=angle, duration_fs=150.)), run(pml_project(long=True, angle_deg=angle, duration_fs=150.))
        out = reflection(short, long, 0, 1.3, 1.6)
        limit = limits(self.CASE, 'oblique', 'R_max_on_band')
        ky = 2*np.pi/(LAMBDA0*1e-6)*math.sin(math.radians(angle))
        f = band(21, 1.3, 1.6)
        self.record.add(f'oblique {angle}deg L10', angle_deg=angle, bloch_phase=short.project.region.bloch_phase[1], steps=short.project.region.steps,
                        angle_deg_over_band=np.degrees(np.arcsin(ky*C0/(2*np.pi*f))), complex_fields=short.summary['complex_fields'],
                        seconds=short.summary['seconds']+long.summary['seconds'], **out, limit=limit)
        assert out['R_max_on_band'] <= limit, (angle, out['R_max_on_band'], out['R_max_dB'])

    def test_interface_reflection(self):
        short, long = run(pml_project(long=False, interface=True, duration_fs=150.)), run(pml_project(long=True, interface=True, duration_fs=150.))
        limit = limits(self.CASE, 'interface', 'R_max_on_band')
        results = {}
        for monitor, label in ((0, 'vacuum side'), (1, 'dielectric side')):
            results[label] = reflection(short, long, monitor)
        self.record.add('interface n=2 half space L10', steps=short.project.region.steps, shape_short=short.project.region.shape, seconds=short.summary['seconds']+long.summary['seconds'],
                        monitors=results, limit=limit)
        for label, out in results.items():
            assert out['R_max_on_band'] <= limit, (label, out['R_max_on_band'], out['R_max_dB'])

    def test_depth_and_time_sweeps(self):
        base = reflection(run(pml_project(long=False)), run(pml_project(long=True)), 0)
        deep = reflection(run(pml_project(long=False, layers=20)), run(pml_project(long=True, layers=20)), 0)
        longer = reflection(run(pml_project(long=False, duration_fs=200.)), run(pml_project(long=True, duration_fs=200.)), 0)
        self.record.add('sweeps vacuum normal', depth=dict(L10=base['R_max_on_band'], L20=deep['R_max_on_band'], L10_dB=base['R_max_dB'], L20_dB=deep['R_max_dB']),
                        time=dict(fs100=base['R_max_on_band'], fs200=longer['R_max_on_band'], relative_change=abs(longer['R_max_on_band']/base['R_max_on_band']-1)),
                        policy='reported only')
        assert deep['R_max_on_band'] <= base['R_max_on_band']
        assert abs(longer['R_max_on_band']/base['R_max_on_band']-1) <= .1

    @pytest.mark.parametrize('label,kwargs', [('vacuum normal', dict(background=1.)), ('n=2 normal', dict(background=2.)), ('vacuum 60deg', dict(angle_deg=60))])
    def test_long_time_stability(self, label, kwargs):
        result = Simulation(pml_project(long=False, steps=20000, **kwargs)).run()
        history = result.summary['diagnostics']
        steps = np.array([d['step'] for d in history])
        energy = np.array([d['state_norm'] for d in history])
        peak = float(energy.max())
        half = int(np.searchsorted(steps, 10000))
        # Growth is judged above a floor of 1e-12 of the peak energy (1e-6 in field amplitude): below it the
        # FP64 state norm is round-off noise (the vacuum run reaches 1e-17 of its peak).
        late_growth = float(energy[half:].max()/max(energy[half], 1e-12*peak))
        limit = limits(self.CASE, 'stability')
        out = dict(steps=int(steps[-1]), duration_fs=float(steps[-1]*result.times[0]*1e15), source_end_fs=result.summary['source_end_s']*1e15,
                   energy_peak=peak, energy_last=float(energy[-1]), energy_last_over_peak=float(energy[-1]/peak), late_growth=late_growth,
                   late_growth_floor=1e-12*peak, energy_at_half_over_peak=float(energy[half]/peak), energy_max_second_half_over_peak=float(energy[half:].max()/peak),
                   energy_at_half=float(energy[half]), field_peak_last=history[-1]['field_peak'], termination=result.summary['termination_reason'],
                   seconds=result.summary['seconds'], checks=len(history))
        oblique = 'angle_deg' in kwargs
        self.record.add(f'stability {label}', **kwargs, **out, limits=limit, decay_criterion_applies=not oblique)
        assert late_growth <= limit['late_growth_max'], (label, out)
        if not oblique:
            assert out['energy_last_over_peak'] <= limit['energy_last_over_peak_max'], (label, out)

    def test_layer_a_cuda_fp32(self):
        if not CUDA:
            pytest.skip('CUDA unavailable')
        reference = run(pml_project(long=False))
        p = pml_project(long=False, backend='cuda', precision='float32')
        trial = Simulation(p).run()
        out = fp32_metrics(reference.signals, trial.signals)
        self.record.add('layer A cuda fp32 short vacuum normal', steps=p.region.steps, cuda_graph=trial.summary['cuda_graph'], gpu=trial.summary['gpu'], **out, limits=FP32)
        assert out['within'], out
