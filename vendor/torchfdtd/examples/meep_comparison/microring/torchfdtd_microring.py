"""TorchFDTD side of the microring comparison with Meep.

Reads geometry.json, runs the ring-plus-bus scene and the straight-bus normalisation
scene on the GPU, and writes the record docs/validation/meep_comparison/microring_torchfdtd.json.

    PYTHONPATH=<worktree> python examples/meep_comparison/microring/torchfdtd_microring.py [--out PATH] [--timing]

The script asserts that the imported torchfdtd package is the one inside this worktree.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torchfdtd  # noqa: E402

TORCHFDTD_FILE = Path(torchfdtd.__file__).resolve()
assert str(TORCHFDTD_FILE).startswith(str(REPO)), f'torchfdtd was imported from {TORCHFDTD_FILE}, not from {REPO}'

import torch  # noqa: E402
from torchfdtd import FieldMonitor, Material, Project, Region, Simulation, Source, SpectrumSettings, Structure  # noqa: E402
from torchfdtd.solver import field_axes, source_slice, voxelize  # noqa: E402
from torchfdtd.waveforms import source_time_signal  # noqa: E402

C0 = 299792458.0
DEFAULT_OUT = REPO / 'docs' / 'validation' / 'meep_comparison' / 'microring_torchfdtd.json'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def waveform(source, times):
    """Shared pulse: amplitude*exp(-0.5*((t-4*sigma)/sigma)^2)*sin(2*pi*f0*t), sigma = pulse_cycles/f0."""
    f0 = C0 / (source['wavelength_um'] * 1e-6)
    sigma = source['pulse_cycles'] / f0
    u = (np.asarray(times, dtype=np.float64) - 4 * sigma) / sigma
    return source['amplitude'] * np.exp(-.5 * u * u) * np.sin(2 * math.pi * f0 * times)


def waveform_fingerprint(values):
    """Portable fingerprint of the sampled pulse: values below 1e-30 set to +0, then float32.

    The Meep process runs with flush-to-zero, so the Gaussian's tail (float64 or float32 denormals, signed
    zeros) differs from the TorchFDTD process at the byte level while every value above 1e-30 is identical.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.where(abs(values) < 1e-30, 0.0, values).astype(np.float32)


def wavelengths_um(spectrum):
    return np.linspace(spectrum['wavelength_start_um'], spectrum['wavelength_stop_um'], spectrum['points'])


def nvidia_smi():
    fields = ('name', 'driver_version', 'memory.total', 'memory.used', 'utilization.gpu')
    try:
        out = subprocess.run(['nvidia-smi', f'--query-gpu={",".join(fields)}', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(fields, [v.strip() for v in out.split(',')]))


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_snapshot():
    smi = nvidia_smi()
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return dict(load_average=load, gpu_memory_used_mib=None if smi is None else float(smi['memory.used']),
                gpu_utilization_percent=None if smi is None else float(smi['utilization.gpu']))


def build_project(g, with_ring):
    r = Region(dimension='2d', size=tuple(g['size_um']), mesh=g['mesh_um'], steps=g['steps'], pml_cells=g['pml_cells'],
               courant_factor=g['courant_factor'], background_index=g['background_index'], material_sampling='yee',
               interface_method='staircase', backend='cuda', precision='float32', cuda_kernel='fused', cuda_monitor_kernel='fused',
               snapshot_interval=10000)
    bus, src, mon, ring = g['bus'], g['source'], g['monitors'], g['ring']
    structures = [Structure(id='bus', name='bus', kind='rectangle', center=(0, bus['center_y_um'], 0),
                            size=(g['size_um'][0], bus['width_um'], 1), material='core')]
    if with_ring:
        structures.append(Structure(id='ring', name='ring', kind='ring', center=(ring['center_um'][0], ring['center_um'][1], 0),
                                    radius=ring['outer_radius_um'], inner_radius=ring['inner_radius_um'], size=(1, 1, 1), material='core'))
    spectrum = SpectrumSettings(sampling='wavelength', wavelength_start=g['spectrum']['wavelength_start_um'],
                                wavelength_stop=g['spectrum']['wavelength_stop_um'], frequency_points=g['spectrum']['points'],
                                apodization='none')
    monitors = [FieldMonitor(id=label, name=label, normal='x', center=(x, mon['center_y_um'], 0), size=(0, mon['width_um'], 1),
                             spectrum=spectrum, dft_precision='float64')
                for label, x in (('in', mon['in_x_um']), ('out', mon['out_x_um']))]
    return Project(name='microring' if with_ring else 'straight bus', region=r,
                   materials=[Material(name='core', index=g['core_index']), Material(name='clad', index=g['background_index'])],
                   structures=structures,
                   sources=sources(g),
                   monitors=monitors)


def sources(g):
    """Meep's restriction of a line source whose ends lie on nodes: weight 1 inside, 0.5 on the two end rows."""
    src = g['source']
    dx = g['mesh_um']
    full, half = src['rows']['full_weight'], src['rows']['half_weight']
    pulse = dict(wavelength=src['wavelength_um'], pulse_cycles=src['pulse_cycles'], component=src['component'], injection='soft')
    inner = (full[1] - 1 - full[0]) * dx
    x, y0 = src['x_um'], src['center_y_um']
    out = [Source(id='line', name='line', kind='plane', center=(x, y0, 0), size=(0, inner, 0), amplitude=src['amplitude'], **pulse)]
    for label, row in (('end_low', half[0]), ('end_high', half[1])):
        y = row * dx - g['size_um'][1] / 2
        out.append(Source(id=label, name=label, kind='point', center=(x, y, 0), amplitude=.5 * src['amplitude'], **pulse))
    return out


def check_grid(g, p):
    r = p.region
    assert list(r.shape) == g['cells'], (r.shape, g['cells'])
    assert math.isclose(r.time_step, g['dt_s'], rel_tol=1e-12), (r.time_step, g['dt_s'])
    assert r.steps == g['steps']
    assert all(r.pml_layers(a, s) == g['pml_cells'] for a in range(2) for s in range(2))
    rows = g['source']['rows']
    loc = source_slice(p.sources[0], r)
    assert loc[0] == slice(g['source']['x_cell'], g['source']['x_cell'] + 1), loc
    assert loc[1] == slice(*rows['full_weight']), loc
    ends = [source_slice(s, r) for s in p.sources[1:]]
    assert [e[:2] for e in ends] == [(g['source']['x_cell'], row) for row in rows['half_weight']], ends
    assert p.sources[1].amplitude == p.sources[2].amplitude == .5 * p.sources[0].amplitude
    assert [rows['half_weight'][0], rows['half_weight'][1] + 1] == g['source']['y_cells'] == [rows['full_weight'][0] - 1, rows['full_weight'][1] + 1]
    times = np.arange(1, r.steps + 1) * r.time_step
    native = source_time_signal(p.sources[0], times)
    shared = waveform(g['source'], times)
    assert np.allclose(native, shared, rtol=0, atol=1e-15), 'shared waveform differs from the torchfdtd waveform'
    assert np.allclose(source_time_signal(p.sources[1], times), .5 * shared, rtol=0, atol=1e-15)
    axes = field_axes(r, 'Ez')
    for label in ('in', 'out'):
        m = next(m for m in p.monitors if m.id == label)
        assert int(np.argmin(abs(axes[0] - m.center[0]))) == g['monitors'][f'{label}_x_cell']
        assert abs(axes[0][g['monitors'][f'{label}_x_cell']] - m.center[0]) < 1e-9
        inside = np.flatnonzero((axes[1] > m.center[1] - m.size[1] / 2) & (axes[1] < m.center[1] + m.size[1] / 2))
        assert [int(inside[0]), int(inside[-1]) + 1] == g['monitors']['y_cells'], inside
    return dict(shared_waveform_sha256=sha256_array(waveform_fingerprint(shared)), line_cells=[[loc[0].start, loc[0].stop], [loc[1].start, loc[1].stop]],
                end_cells=[list(e[:2]) for e in ends], end_weight=.5)


def staircase(g, p):
    """Ez-node permittivity of the interior (nodes strictly inside the PML), the same nodes the Meep script hashes."""
    eps, _ = voxelize(p)
    ez = np.asarray(eps[..., 2][:, :, 0], dtype=np.float64)
    n = g['pml_cells']
    interior = ez[n + 1:g['cells'][0] - n, n + 1:g['cells'][1] - n]
    axes = field_axes(p.region, 'Ez')
    x, y = np.meshgrid(axes[0], axes[1], indexing='ij')
    radius = np.hypot(x - g['ring']['center_um'][0], y - g['ring']['center_um'][1])
    return dict(interior_cells=[[n + 1, g['cells'][0] - n], [n + 1, g['cells'][1] - n]],
                interior_ez_epsilon_sha256=sha256_array(np.round(interior, 4)),
                interior_core_nodes=int(np.count_nonzero(interior > g['background_index']**2 + 1e-6)),
                min_node_distance_to_circles_um=float(min(np.min(abs(radius - g['ring']['outer_radius_um'])),
                                                          np.min(abs(radius - g['ring']['inner_radius_um'])))))


def solve(p):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = Simulation(p).run()
    full = time.perf_counter() - start
    s = result.summary
    assert s['steps'] == p.region.steps and s['termination_reason'] not in ('cancelled', 'decayed'), s['termination_reason']
    return result, dict(setup_seconds=s['setup_seconds'], stepping_seconds=s['seconds'], full_seconds=full,
                        cuda_graph=s['cuda_graph'], field_peak=s['field_peak'], steps=s['steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--geometry', default=str(HERE / 'geometry.json'))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--timing', action='store_true', help='one warm-up ring solve, then three timed ring solves')
    args = parser.parse_args()
    geometry_path = Path(args.geometry)
    g = json.loads(geometry_path.read_text(encoding='utf-8'))
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), 'the recorded TorchFDTD run uses the GPU'

    ring_project = build_project(g, True)
    straight_project = build_project(g, False)
    checks = check_grid(g, ring_project)
    check_grid(g, straight_project)
    stairs = staircase(g, ring_project)
    print(json.dumps(dict(checks=checks, staircase=stairs)), flush=True)

    before = load_snapshot()
    repeats = 3 if args.timing else 1
    samples = []
    if args.timing:
        _, warm = solve(ring_project)
        print(json.dumps(dict(warm_up=warm)), flush=True)
    for repeat in range(repeats):
        ring, timing = solve(ring_project)
        samples.append(timing)
        print(json.dumps(dict(repeat=repeat, **timing)), flush=True)
    straight, straight_timing = solve(straight_project)
    print(json.dumps(dict(straight=straight_timing)), flush=True)
    after = load_snapshot()

    flux = {}
    for label in ('in', 'out'):
        a, b = ring.field_monitor(label), straight.field_monitor(label)
        assert np.array_equal(a['frequency_hz'], b['frequency_hz'])
        flux[label] = dict(ring=a['flux'].astype(np.float64), straight=b['flux'].astype(np.float64))
    frequency = ring.field_monitor('out')['frequency_hz']
    wavelength = C0 / frequency * 1e6
    assert np.allclose(wavelength, wavelengths_um(g['spectrum']), rtol=1e-12, atol=0)
    T = flux['out']['ring'] / flux['out']['straight']
    median = {k: float(np.median([s[k] for s in samples])) for k in ('setup_seconds', 'stepping_seconds', 'full_seconds')}
    smi = nvidia_smi()
    record = dict(
        schema='torchfdtd-meep-comparison-v1', example='microring', solver='torchfdtd', date=time.strftime('%Y-%m-%d'),
        geometry_file=str(geometry_path.name), geometry_sha256=sha256_file(geometry_path), script_sha256=sha256_file(__file__),
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], cpu=cpu_model(), logical_cpus=os.cpu_count(),
                         wsl_distribution=os.environ.get('WSL_DISTRO_NAME'), gpu=smi, torch=torch.__version__, torch_cuda=torch.version.cuda,
                         cupy=package_version('cupy-cuda12x') or package_version('cupy'), torchfdtd=package_version('torchfdtd'),
                         torchfdtd_file=str(TORCHFDTD_FILE), device='cuda', precision='float32 fields, float64 plane DFT accumulators'),
        method='2D Ez (TM) scene, staircase permittivity at the Yee Ez nodes (material_sampling="yee"), fused CUDA Yee/CPML kernel, '
               'fused CUDA plane DFT monitors with complex128 accumulators over the whole run, soft Ez line source with the shared '
               'sampled pulse, CPML of the declared thickness on all four sides, T = out flux of the ring run / out flux of the '
               'straight-bus run.',
        grid=dict(cells=g['cells'], mesh_um=ring_project.region.mesh, dt_s=ring_project.region.time_step, steps=ring_project.region.steps,
                  courant_number=ring_project.region.rectangular_courant, pml_cells=g['pml_cells'], run_time_ps=ring_project.region.steps * ring_project.region.time_step * 1e12,
                  precision='float32', cuda_kernel='fused', cuda_monitor_kernel='fused', cuda_graph=samples[-1]['cuda_graph']),
        pml=dict(formulation='stretched-coordinate CPML, cubic sigma profile, kappa=1, alpha=1e-8', cells=g['pml_cells'],
                 thickness_um=g['pml_cells'] * g['mesh_um']),
        staircase=stairs, source=dict(**g['source'], **checks),
        monitors=dict(in_x_cell=g['monitors']['in_x_cell'], out_x_cell=g['monitors']['out_x_cell'], y_cells=g['monitors']['y_cells'],
                      quadrature='trilinear Yee interpolation at the midpoints of the clipped cells spanning the plane (102 intervals for 101 nodes), '
                                 'H phase advanced by half a step', quadrature_points=int(len(ring.field_monitor('out')['weights'])),
                      span_um=float(np.sum(ring.field_monitor('out')['weights']) * 1e6)),
        wavelength_um=wavelength.tolist(), frequency_hz=frequency.tolist(),
        flux_out_ring=flux['out']['ring'].tolist(), flux_out_straight=flux['out']['straight'].tolist(),
        flux_in_ring=flux['in']['ring'].tolist(), flux_in_straight=flux['in']['straight'].tolist(),
        T=T.tolist(), T_in=(flux['in']['ring'] / flux['in']['straight']).tolist(),
        timing=dict(timing_mode='timing' if args.timing else 'development', repeats=repeats, samples=samples, **median,
                    straight_run=straight_timing, load_before=before, load_after=after,
                    host_load_note=('development run on a shared workstation: other GPU and CPU jobs may have been running; '
                                    if not args.timing else 'timed run: one warm-up solve, then three timed solves; ')
                                   + f'WSL load average before {before["load_average"]}, after {after["load_average"]}; GPU memory used before the run '
                                   + f'{before["gpu_memory_used_mib"]} MiB, GPU utilisation before the run {before["gpu_utilization_percent"]} %.'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out)
    print(json.dumps(dict(T_min=float(T.min()), T_max=float(T.max()), **median)))


if __name__ == '__main__':
    main()
