"""TorchFDTD side of the metagrating comparison (see BRIEF.md and README.md).

Run inside the WSL venv from the worktree root:
    PYTHONPATH=<worktree> python examples/meep_comparison/metagrating/torchfdtd_metagrating.py \
        --out docs/validation/meep_comparison/metagrating_torchfdtd.json [--timing]

The script records the raw complex Ez/Hx lines of the reflection and transmission
monitors for the grating run and for the bare-substrate reference run. No diffraction
efficiency is computed here; compare.py applies the same Fourier decomposition to both
solvers' records.
"""
from __future__ import annotations

import argparse
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
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Import torchfdtd from the checkout that holds this example, as the microring script does.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torchfdtd  # noqa: E402
from torchfdtd import (Boundaries, BoundaryFace, FieldMonitor, Material, Project, Region, Simulation, Source, SpectrumSettings,  # noqa: E402
                       Structure)
from torchfdtd.solver import field_axes, source_slice, voxelize  # noqa: E402
from torchfdtd.waveforms import source_time_signal  # noqa: E402

GEOMETRY = HERE / 'geometry.json'
C0 = 299792458.0


def load_geometry(path=GEOMETRY):
    raw = Path(path).read_bytes()
    geometry = json.loads(raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(raw).hexdigest()
    return geometry


def waveform(wave, times_s):
    """The shared pulse: TorchFDTD 'cycles' Gaussian, sigma = cycles/f, offset 4 sigma, sin(omega t)."""
    times_s = np.asarray(times_s, dtype=np.float64)
    frequency = C0 / (wave['wavelength_um'] * 1e-6)
    sigma = wave['pulse_cycles'] / frequency
    u = (times_s - 4 * sigma) / sigma
    return wave['amplitude'] * np.exp(-.5 * u * u) * np.sin(2 * math.pi * frequency * times_s)


def make_project(g, ridges=None, *, steps=None, spectrum=None, backend='cuda'):
    """The shared scene. ridges=None uses geometry.json; ridges=[] is the bare-substrate reference."""
    ridges = g['ridges'] if ridges is None else ridges
    lx, ly = g['cell_size_um']
    spectrum = spectrum or g['spectrum']
    region = Region(dimension='2d', size=(lx, ly, 1), mesh=g['mesh_um'], steps=steps or g['steps'], pml_cells=g['pml_cells'],
                    courant_factor=0.99, backend=backend, precision='float32', material_sampling='yee', cuda_kernel='fused' if backend == 'cuda' else 'torch',
                    cuda_monitor_kernel='fused' if backend == 'cuda' else 'torch', snapshot_interval=min(steps or g['steps'], 10000),
                    boundaries=Boundaries(x_min=BoundaryFace(kind='periodic'), x_max=BoundaryFace(kind='periodic')))
    top = g['substrate_top_y_um']
    structures = [Structure(id='substrate', name='substrate', center=(0, (top - ly / 2) / 2, 0), size=(2 * lx, top + ly / 2, 1),
                            material='substrate')]
    for k, ridge in enumerate(ridges):
        structures.append(Structure(id=f'ridge{k}', name=f'ridge{k}', center=(ridge['center_x_um'], top + g['ridge_height_um'] / 2, 0),
                                    size=(ridge['width_um'], g['ridge_height_um'], 1), material='ridge'))
    wave = g['source']['waveform']
    spec = SpectrumSettings(sampling=spectrum['sampling'], wavelength_start=spectrum['wavelength_start_um'],
                            wavelength_stop=spectrum['wavelength_stop_um'], frequency_points=spectrum['points'], apodization='none')
    monitors = [FieldMonitor(id=name, name=name, normal='y', center=(0, g['monitors'][name + '_y_um'], 0), size=(lx, 0, 1),
                             record_fields=('Ez', 'Hx'), record_poynting=(), record_flux=True, dft_precision='float64', spectrum=spec)
                for name in ('reflection', 'transmission')]
    return Project(name='metagrating', region=region,
                   materials=[Material(name='substrate', index=g['substrate_index']), Material(name='ridge', index=g['ridge_index'])],
                   structures=structures,
                   sources=[Source(id='source', kind='plane', component='Ez', normal='y', center=(0, g['source']['y_um'], 0),
                                   size=(lx, 0, 0), wavelength=wave['wavelength_um'], pulse_cycles=wave['pulse_cycles'],
                                   amplitude=wave['amplitude'])],
                   monitors=monitors)


def ez_nodes(region):
    """Physical Ez Yee node coordinates (x nodes, y nodes) in micrometres."""
    axes = field_axes(region, 'Ez')
    return np.asarray(axes[0]), np.asarray(axes[1])


def staircase(project, g):
    """Columns of Ez nodes inside silicon on the row through the ridge middle, and the substrate row count."""
    eps, _ = voxelize(project)
    ez = np.asarray(eps)[..., 2][:, :, 0]
    xs, ys = ez_nodes(project.region)
    row = int(np.argmin(abs(ys - (g['substrate_top_y_um'] + g['ridge_height_um'] / 2))))
    silicon = np.flatnonzero(ez[:, row] > (g['ridge_index'] ** 2 + g['substrate_index'] ** 2) / 2)
    substrate_rows = np.flatnonzero(abs(ez[0, :] - g['substrate_index'] ** 2) < 1e-6)
    ridge_rows = np.flatnonzero(ez[silicon[0], :] > (g['ridge_index'] ** 2 + g['substrate_index'] ** 2) / 2) if len(silicon) else np.array([], dtype=int)
    return dict(row_y_um=float(ys[row]), silicon_columns=silicon.tolist(), silicon_rows=ridge_rows.tolist(),
                substrate_rows=[int(substrate_rows[0]), int(substrate_rows[-1])] if len(substrate_rows) else [],
                ez_x_nodes_um=xs.tolist(), ez_y_nodes_um=ys.tolist())


def check_grid(project, g):
    r = project.region
    assert list(r.shape[:2]) == list(g['cells']), (r.shape, g['cells'])
    dt = g['courant_number'] * g['mesh_um'] * 1e-6 / C0
    assert math.isclose(r.time_step, dt, rel_tol=1e-9), (r.time_step, dt)
    assert math.isclose(r.rectangular_courant, g['courant_number'], rel_tol=1e-9)
    assert r.pml_layers(1, 0) == g['pml_cells'] and r.pml_layers(1, 1) == g['pml_cells'] and r.pml_layers(0, 0) == 0
    xs, ys = ez_nodes(r)
    src = source_slice(project.sources[0], r)
    assert src[0] == slice(0, g['cells'][0]) and src[1].stop - src[1].start == 1, src
    assert math.isclose(ys[src[1].start], g['source']['y_um'], abs_tol=1e-9), (ys[src[1].start], g['source']['y_um'])
    times = np.arange(1, r.steps + 1) * r.time_step
    native = source_time_signal(project.sources[0], times)
    shared = waveform(g['source']['waveform'], times)
    assert np.allclose(native, shared, rtol=0, atol=1e-15), 'shared waveform differs from the torchfdtd waveform'
    return dict(source_row=int(src[1].start), source_columns=[int(src[0].start), int(src[0].stop)],
                waveform_max_abs_difference=float(np.max(abs(native - shared))))


def monitor_record(result, name):
    m = result.field_monitor(name)
    names = list(m['components'])
    ez = m['fields'][..., names.index('Ez')]
    hx = m['fields'][..., names.index('Hx')]
    return dict(y_um=float(m['points_um'][0, 1]), x_um=m['points_um'][:, 0].tolist(), weights_m=np.asarray(m['weights']).tolist(),
                frequency_hz=np.asarray(m['frequency_hz']).tolist(), wavelength_um=(C0 / np.asarray(m['frequency_hz']) * 1e6).tolist(),
                ez_real=ez.real.tolist(), ez_imag=ez.imag.tolist(), hx_real=hx.real.tolist(), hx_imag=hx.imag.tolist(),
                flux=np.asarray(m['flux']).tolist(), flux_units=m['flux_units'], field_units=m['field_units'])


def nvidia_smi():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    keys = ('name', 'driver_version', 'memory_total_mb', 'memory_used_mb', 'utilization_percent')
    return dict(zip(keys, [v.strip() for v in out.split(',')]))


def host_cpu_load_percent():
    try:
        out = subprocess.run(['powershell.exe', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).LoadPercentage'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def timed_run(project):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = Simulation(project).run()
    torch.cuda.synchronize()
    full = time.perf_counter() - started
    return result, dict(setup_seconds=result.summary['setup_seconds'], stepping_seconds=result.summary['seconds'], full_seconds=full)


def require_checkout_import():
    """The records name the commit of the checkout that holds this example; an installed wheel would misattribute them."""
    imported = Path(torchfdtd.__file__).resolve()
    if not str(imported).startswith(str(REPO)):
        raise SystemExit(f'torchfdtd was imported from {imported}, not from the checkout {REPO} that holds this example. '
                         f'Run the script from that checkout: cd {REPO} && python {HERE.relative_to(REPO).as_posix()}/{Path(__file__).name} ...')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--timing', action='store_true', help='one warm-up, then three timed grating solves')
    parser.add_argument('--geometry', default=str(GEOMETRY))
    args = parser.parse_args()
    require_checkout_import()
    g = load_geometry(args.geometry)
    assert g['ridges'], 'geometry.json carries no ridges; run design.py first'
    torch.backends.cudnn.benchmark = False
    gpu_before = nvidia_smi()
    cpu_before = host_cpu_load_percent()

    sample = make_project(g)
    reference = make_project(g, ridges=[])
    grid = check_grid(sample, g)
    check_grid(reference, g)
    stairs = staircase(sample, g)
    reference_stairs = staircase(reference, g)
    assert not reference_stairs['silicon_columns']

    ref_result, ref_timing = timed_run(reference)
    repeats = 3 if args.timing else 1
    samples = []
    if args.timing:
        timed_run(sample)  # warm-up
    for _ in range(repeats):
        result, timing = timed_run(sample)
        samples.append(timing)
    r = sample.region
    record = dict(
        schema='torchfdtd-meep-comparison-v1', example='metagrating', solver='torchfdtd', geometry_sha256=g['_sha256'],
        date=time.strftime('%Y-%m-%d'),
        method='2D TMz Yee grid (Ez, Hx, Hy), periodic in x (real fields), CPML in y, soft additive Ez sheet across the full period '
               'with the shared Gaussian pulse, staircase Yee material sampling, DFT lines of Ez and Hx at the reflection and '
               'transmission rows (trilinear interpolation to the pixel centres in x, H carries its half-step phase), '
               'bare-substrate reference run with the same settings. Fields are the solver\'s reduced units; compare.py forms ratios.',
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], wsl_distribution=os.environ.get('WSL_DISTRO_NAME'),
                         packages=versions(('torchfdtd', 'torch', 'cupy-cuda12x', 'numpy')), torch_cuda=torch.version.cuda,
                         gpu=gpu_before, torchfdtd_file=torchfdtd.__file__, device=torch.cuda.get_device_name(0)),
        grid=dict(cells=list(r.shape[:2]), mesh_um=r.mesh, dt_s=r.time_step, steps=r.steps, courant_number=r.rectangular_courant,
                  pml_cells=g['pml_cells'], pml_axis='y', periodic_axis='x', precision='float32', dft_precision='float64',
                  backend=result.summary['backend'], cuda_kernel='fused', cuda_graph=result.summary['cuda_graph'],
                  boundary='CPML, cubic sigma profile sigma=40*rho^3/(L+1) in Courant units, kappa=1, alpha=1e-8',
                  **grid),
        staircase=stairs,
        phasor_convention='DFT = dt*sum(f(t) exp(+2 pi i f t)); a +y travelling wave is exp(+i k_y y); H sampled at t+dt/2 with its phase applied',
        phasor_time_sign=1,
        monitors={name: monitor_record(result, name) for name in ('reflection', 'transmission')},
        reference_monitors={name: monitor_record(ref_result, name) for name in ('reflection', 'transmission')},
        field_peak=dict(sample=result.summary['field_peak'], reference=ref_result.summary['field_peak']),
        timing=dict(timing_mode='timing' if args.timing else 'development', repeats=repeats, samples=samples,
                    setup_seconds=float(np.median([s['setup_seconds'] for s in samples])),
                    stepping_seconds=float(np.median([s['stepping_seconds'] for s in samples])),
                    full_seconds=float(np.median([s['full_seconds'] for s in samples])),
                    reference=ref_timing, gpu_after=nvidia_smi(), host_cpu_percent_before=cpu_before,
                    host_cpu_percent_after=host_cpu_load_percent(),
                    host_load_note=('development run on a shared host: GPU utilisation before the run '
                                    f'{(gpu_before or {}).get("utilization_percent")}%, host CPU {cpu_before}%; not a timing measurement')
                    if not args.timing else f'timing run: GPU utilisation before {(gpu_before or {}).get("utilization_percent")}%, host CPU {cpu_before}%'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out)
    print(json.dumps(dict(timing=record['timing']['samples'], field_peak=record['field_peak'], silicon_columns=stairs['silicon_columns'])))


if __name__ == '__main__':
    main()
