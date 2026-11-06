"""Solver-independent helpers for the metalens comparison (numpy only; both solver scripts import this).

Geometry files, the shared source waveform, the frequency samples, environment facts,
host-load notes and the record writer. Nothing here runs a simulation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

C0 = 299792458.0
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'
FIGURES = ROOT / 'docs' / 'figures' / 'meep_comparison'
GEOMETRY_FILES = {'2d': HERE / 'geometry.json', '3d': HERE / 'geometry_3d.json'}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_geometry(part):
    path = GEOMETRY_FILES[part]
    spec = json.loads(path.read_text(encoding='utf-8'))
    spec['_sha256'] = sha256_file(path)
    spec['_path'] = str(path)
    return spec


def waveform(wave, times_s):
    """The shared pulse: TorchFDTD's legacy Gaussian with sigma = cycles / f0, offset 4 sigma, sin(omega t) carrier."""
    times_s = np.asarray(times_s, dtype=np.float64)
    frequency = C0 / (wave['wavelength_um'] * 1e-6)
    sigma = wave['pulse_cycles'] / frequency
    u = (times_s - 4 * sigma) / sigma
    return wave.get('amplitude', 1.0) * np.exp(-.5 * u * u) * np.sin(2 * math.pi * frequency * times_s)


def frequencies_hz(spec):
    freqs = [C0 / (w * 1e-6) for w in spec['wavelengths_um']]
    assert all(a < b for a, b in zip(freqs, freqs[1:])), 'wavelengths must be listed in frequency-increasing order'
    return freqs


def time_step_s(spec):
    return spec['courant_number'] * spec['mesh_um'] * 1e-6 / C0


def cells(spec):
    return [int(round(v / spec['mesh_um'])) for v in spec['cell_um']]


def package_versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def nvidia_smi(fields=('name', 'driver_version', 'memory.total', 'memory.used', 'utilization.gpu')):
    try:
        out = subprocess.run(['nvidia-smi', f'--query-gpu={",".join(fields)}', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(fields, [v.strip() for v in out.split(',')]))


def environment(extra=None):
    info = dict(platform=platform.platform(), python=sys.version.split()[0], cpu=cpu_model(), logical_cpus=os.cpu_count(),
                wsl_distribution=os.environ.get('WSL_DISTRO_NAME'), gpu=nvidia_smi())
    if extra:
        info.update(extra)
    return info


def host_load_note(timing_mode):
    """Load average and GPU use at the moment of the call; the maintainer's --timing runs add their own quiet-host note."""
    try:
        load = list(os.getloadavg())
    except (OSError, AttributeError):
        load = None
    smi = nvidia_smi()
    gpu = None if smi is None else dict(memory_used_mb=float(smi['memory.used']), utilization_percent=float(smi['utilization.gpu']))
    text = ('development run on a shared host: other agents used the CPU and the GPU at the same time; the numbers bound the '
            'solver time from above' if timing_mode == 'development' else 'timing run: one warm-up, three timed solves, medians reported')
    return dict(note=text, load_average_1_5_15=load, gpu=gpu, sampled_at=time.strftime('%Y-%m-%dT%H:%M:%S'))


def timing_summary(samples):
    """Medians of the timed samples (a single development sample is its own median)."""
    keys = ('setup_seconds', 'stepping_seconds', 'full_seconds')
    return {k: float(np.median([s[k] for s in samples])) for k in keys}


def write_record(name, record):
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f'{name}.json'
    path.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    return path


def as_list(a, digits=7):
    """JSON list with `digits` significant figures (the records stay small; both solvers use the same rounding)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim > 1:
        return [as_list(v, digits) for v in a]
    return [float(f'{v:.{digits}g}') for v in a.ravel()]


def uniform_weight(weights):
    """The one quadrature weight of a plane whose points all carry the same weight."""
    weights = np.asarray(weights, dtype=np.float64)
    assert np.allclose(weights, weights[0], rtol=1e-12, atol=0), 'non-uniform monitor weights'
    return float(weights[0])


# ----------------------------------------------------------------------------
# Observables from collocated DFT fields. Both solver scripts call these on their
# own arrays so that the reductions are identical.
# ----------------------------------------------------------------------------
def intensity(fields):
    """|E|^2 summed over the electric components of a (freq, point, comp) array."""
    return np.sum(np.abs(fields) ** 2, axis=-1)


def poynting_2d(ez, hx):
    """Reduced y-directed Poynting density of a 2D TM (Ez, Hx, Hy) field: S_y = Re(Ez Hx*)/2."""
    return .5 * np.real(ez * np.conj(hx))


def poynting_3d_z(ex, ey, hx, hy):
    """Reduced z-directed Poynting density: S_z = Re(Ex Hy* - Ey Hx*)/2."""
    return .5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))


def peak_parabolic(coord, values):
    """Peak position by a three-point parabola through the largest sample and its neighbours."""
    coord = np.asarray(coord, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    i = int(np.argmax(values))
    if i == 0 or i == len(values) - 1:
        return float(coord[i]), float(values[i])
    y0, y1, y2 = values[i - 1], values[i], values[i + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0. if denom == 0 else .5 * (y0 - y2) / denom
    h = coord[i + 1] - coord[i]
    return float(coord[i] + shift * h), float(y1 - .25 * (y0 - y2) * shift)


def fwhm(coord, values):
    """Full width at half maximum of a sampled profile, half-maximum crossings by linear interpolation."""
    coord = np.asarray(coord, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    i = int(np.argmax(values))
    half = values[i] / 2
    left = i
    while left > 0 and values[left] > half:
        left -= 1
    right = i
    while right < len(values) - 1 and values[right] > half:
        right += 1
    if values[left] > half or values[right] > half:
        return math.nan, math.nan, math.nan
    xl = coord[left] + (half - values[left]) / (values[left + 1] - values[left]) * (coord[left + 1] - coord[left])
    xr = coord[right - 1] + (half - values[right - 1]) / (values[right] - values[right - 1]) * (coord[right] - coord[right - 1])
    return float(xr - xl), float(xl), float(xr)


def windowed_power_1d(coord, density, weight, centre, half_width):
    """Power inside |x - centre| <= half_width of a line density sampled with one quadrature weight per point."""
    coord = np.asarray(coord, dtype=np.float64)
    mask = np.abs(coord - centre) <= half_width
    return float(np.sum(np.asarray(density)[mask]) * weight)


def windowed_power_2d(u, v, density, weight, centre_u, centre_v, half_u, half_v):
    """Power inside the rectangle |u-cu| <= half_u, |v-cv| <= half_v of a flattened (point) density with one weight per point."""
    mask = (np.abs(np.asarray(u) - centre_u) <= half_u) & (np.abs(np.asarray(v) - centre_v) <= half_v)
    return float(np.sum(np.asarray(density)[mask]) * weight)


# ----------------------------------------------------------------------------
# Per-wavelength metrics of one solver's record (the scripts fill the summary rows with these; compare.py
# recomputes them from the stored arrays at the centre wavelength).
# ----------------------------------------------------------------------------
def metrics_2d(axis_y, axis_intensity, x, profile, poynting, weight, ridge_top):
    y_peak, i_peak = peak_parabolic(axis_y, axis_intensity)
    width, xl, xr = fwhm(x, profile)
    x_peak, p_peak = peak_parabolic(x, profile)
    power = windowed_power_1d(x, poynting, weight, x_peak, 1.5 * width)
    return dict(axis_peak_y_um=y_peak, axis_peak_from_ridge_top_um=y_peak - ridge_top, axis_peak_intensity=i_peak, focal_plane_peak_x_um=x_peak,
                focal_plane_peak_intensity=p_peak, fwhm_um=width, fwhm_left_um=xl, fwhm_right_um=xr, efficiency=power, efficiency_window_um=3 * width)


def metrics_3d(z, axis_intensity, x, y, intensity_map, poynting_map, weight, pillar_top):
    """intensity_map and poynting_map are flattened x-major (nx, ny) planes."""
    z_peak, i_peak = peak_parabolic(z, axis_intensity)
    nx, ny = len(x), len(y)
    prof = np.asarray(intensity_map, dtype=np.float64).reshape(nx, ny)
    ix, iy = np.unravel_index(int(np.argmax(prof)), prof.shape)
    x_peak, _ = peak_parabolic(x, prof[:, iy])
    y_peak, p_peak = peak_parabolic(y, prof[ix, :])
    fx, xl, xr = fwhm(x, prof[:, iy])
    fy, yl, yr = fwhm(y, prof[ix, :])
    u, v = np.repeat(np.asarray(x), ny), np.tile(np.asarray(y), nx)
    power = windowed_power_2d(u, v, poynting_map, weight, x_peak, y_peak, 1.5 * fx, 1.5 * fy)
    return dict(axis_peak_z_um=z_peak, axis_peak_from_pillar_top_um=z_peak - pillar_top, axis_peak_intensity=i_peak, focal_plane_peak_x_um=x_peak,
                focal_plane_peak_y_um=y_peak, focal_plane_peak_intensity=p_peak, fwhm_x_um=fx, fwhm_y_um=fy, efficiency=power,
                efficiency_window_um=[3 * fx, 3 * fy], peak_row_index=int(ix), peak_column_index=int(iy))


def summary_2d(obs, spec):
    rows = []
    for k, wl in enumerate(obs['wavelengths_um']):
        row = metrics_2d(obs['axis']['y_um'], np.asarray(obs['axis']['intensity'])[k], obs['focal']['x_um'], np.asarray(obs['focal']['intensity'])[k],
                         np.asarray(obs['focal']['poynting_y'])[k], obs['focal']['weight_m'], spec['ridge_top_um'])
        rows.append(dict(wavelength_um=wl, **row, transmission=obs['incident']['transmission'][k]))
    return dict(summary=rows)


def axis_row(section):
    """Index of the transverse sample nearest the optical axis in a section record."""
    return int(np.argmin(np.abs(np.asarray(section['transverse_um']))))


def summary_3d(obs, spec):
    rows = []
    i0 = axis_row(obs['xz'])
    for k, wl in enumerate(obs['wavelengths_um']):
        row = metrics_3d(obs['xz']['z_um'], np.asarray(obs['xz']['intensity'][k])[i0], obs['focal']['x_um'], obs['focal']['y_um'],
                         np.asarray(obs['focal']['intensity'])[k], np.asarray(obs['focal']['poynting_z'])[k], obs['focal']['weight_m2'], spec['pillar_top_um'])
        rows.append(dict(wavelength_um=wl, **row, transmission=obs['incident']['transmission'][k]))
    return dict(summary=rows)


def trim_3d(obs, wavelength_um=1.55):
    """Keep the focal-plane maps and the sections at the centre wavelength only; the on-axis rows stay for every wavelength."""
    k = obs['wavelengths_um'].index(wavelength_um)
    obs['focal']['intensity'] = obs['focal']['intensity'][k]
    obs['focal']['poynting_z'] = obs['focal']['poynting_z'][k]
    obs['focal']['stored_wavelength_um'] = wavelength_um
    for name in ('xz', 'yz'):
        i0 = axis_row(obs[name])
        obs[name]['axis_intensity'] = [obs[name]['intensity'][j][i0] for j in range(len(obs['wavelengths_um']))]
        obs[name]['intensity'] = obs[name]['intensity'][k]
        obs[name]['stored_wavelength_um'] = wavelength_um
    return obs
