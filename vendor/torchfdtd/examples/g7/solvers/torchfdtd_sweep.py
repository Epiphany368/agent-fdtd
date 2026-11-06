"""G7-04 TorchFDTD sweep: the metagrating at meshes 0.04, 0.02, 0.01 and 0.005 um, one series and precision per call.

From the repository root on a CUDA GPU; the timed runs need an otherwise idle GPU:
    python examples/g7/solvers/torchfdtd_sweep.py --series staircase --precision float32 --max-host-cpu-percent 30

Each mesh uses the derived geometry of common.derive_geometry (same physical time, 0.4 um absorber,
Courant number of the base fixture, half-cell shift in the staircase series where an edge would sit
on a node). The scene is the one of examples/meep_comparison/metagrating/torchfdtd_metagrating.py
with the precision, the interface method (staircase, or the experimental subpixel interfaces for the
smoothed series) and the absorber thickness of the point. Per point: one bare-substrate reference run,
one warm-up and three timed grating runs (fused CUDA kernels); with --max-host-cpu-percent every run
first waits until the Windows host load is below that value. The record keeps the per-order Fourier
means of the DFT lines, the efficiencies from the compare.py decomposition and every timing sample.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

# The repository root goes to the END of sys.path, as in tests/conftest.py: an installed torchfdtd package
# (the release wheel) is imported when one exists, otherwise the checkout that holds this script. It is
# imported before torchfdtd_metagrating, which would put the checkout first.
if str(common.ROOT) not in sys.path:
    sys.path.append(str(common.ROOT))
import torchfdtd  # noqa: E402
from torchfdtd import Project  # noqa: E402

spec = importlib.util.spec_from_file_location('torchfdtd_metagrating', common.METAGRATING / 'torchfdtd_metagrating.py')
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

REGION_PML_CAP = 50  # Region.pml_cells is capped at 50; BoundaryFace.layers carries thicker absorbers


def make_project(g, ridges=None, *, precision, series):
    """The scene of torchfdtd_metagrating.make_project with this point's precision, interface method and absorber."""
    data = tm.make_project(dict(g, pml_cells=min(g['pml_cells'], REGION_PML_CAP)), ridges).model_dump()
    region = data['region']
    region.update(precision=precision, interface_method='subpixel' if series == 'smoothed' else 'staircase')
    for face in ('y_min', 'y_max'):
        region['boundaries'][face].update(kind='pml', layers=g['pml_cells'])
    return Project.model_validate(data)


def package_state():
    """Which torchfdtd ran: its file, version and whether it is this checkout or an installed package."""
    path = Path(torchfdtd.__file__).resolve()
    checkout = str(path).startswith(str(common.ROOT.resolve()))
    pyproject = common.ROOT / 'pyproject.toml'
    version = None
    if checkout and pyproject.exists():
        version = next((line.split('=', 1)[1].strip().strip('"') for line in pyproject.read_text(encoding='utf-8').splitlines()
                        if line.startswith('version =')), None)
    return dict(torchfdtd_file=str(path), torchfdtd_version=getattr(torchfdtd, '__version__', None) or version or tm.versions(('torchfdtd',))['torchfdtd'],
                torchfdtd_import='checkout' if checkout else 'installed package',
                installed_distribution_version=tm.versions(('torchfdtd',))['torchfdtd'])


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=common.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return dict(commit=run('rev-parse', 'HEAD'), dirty=bool(run('status', '--porcelain', '--untracked-files=no')))


def wait_for_host(max_cpu_percent, max_wait_seconds):
    """With max_cpu_percent, wait until the Windows host load is at most that value (at most max_wait_seconds)."""
    started = time.perf_counter()
    while True:
        cpu = tm.host_cpu_load_percent()
        waited = time.perf_counter() - started
        busy = max_cpu_percent is not None and cpu is not None and cpu > max_cpu_percent
        if not busy or waited >= max_wait_seconds:
            return dict(waited_seconds=waited, host_cpu_percent_before=cpu, gate_timed_out=bool(busy))
        time.sleep(15)


def full_record(result, ref_result):
    return dict(phasor_time_sign=1, monitors={n: tm.monitor_record(result, n) for n in ('reflection', 'transmission')},
                reference_monitors={n: tm.monitor_record(ref_result, n) for n in ('reflection', 'transmission')})


def run_point(base, mesh, series, precision, repeats, compare, gate):
    g = common.derive_geometry(base, mesh, series)
    sample = make_project(g, precision=precision, series=series)
    reference = make_project(g, ridges=[], precision=precision, series=series)
    grid = tm.check_grid(sample, g)
    tm.check_grid(reference, g)
    stairs = tm.staircase(sample, g) if series == 'staircase' else None
    if stairs is not None:
        stairs = {k: v for k, v in stairs.items() if k not in ('ez_x_nodes_um', 'ez_y_nodes_um')}
    gpu_before, cpu_before = tm.nvidia_smi(), tm.host_cpu_load_percent()
    host = wait_for_host(*gate)
    ref_result, ref_timing = tm.timed_run(reference)
    ref_timing.update(host)
    host = wait_for_host(*gate)
    _, warmup = tm.timed_run(sample)
    warmup.update(host)
    samples, t1_runs = [], []
    for _ in range(repeats):
        host = wait_for_host(*gate)
        result, timing = tm.timed_run(sample)
        samples.append(dict(timing, **host))
        amps = common.point_amplitudes(full_record(result, ref_result))
        T, _ = common.efficiencies_from_amplitudes(amps, g['substrate_index'])
        t1_runs.append(T[:, common.ORDERS.index(1)])
    record_full = full_record(result, ref_result)
    common.check_lines(record_full, g)
    amplitudes = common.point_amplitudes(record_full)
    T, R = common.efficiencies_from_amplitudes(amplitudes, g['substrate_index'])
    wavelength = np.asarray(record_full['monitors']['transmission']['wavelength_um'])
    _, T_c, R_c, diagnostics = compare.solver_efficiencies(record_full, g, list(common.ORDERS))
    assert np.max(abs(T - T_c)) < 1e-12 and np.max(abs(R - R_c)) < 1e-12, 'amplitude path differs from compare.py'
    full = [s['full_seconds'] for s in samples]
    r = sample.region
    summary = result.summary
    record = dict(
        schema='g7-04-point-v1', case='G7-04', solver='torchfdtd', series=series, precision=precision, mesh_um=mesh,
        resolution_per_um=round(1 / mesh), date=time.strftime('%Y-%m-%d %H:%M:%S'), geometry=g, geometry_sha256=g['_sha256'],
        grid=dict(cells=list(r.shape[:2]), cell_count=int(np.prod(r.shape[:2])), steps=r.steps, dt_s=r.time_step,
                  courant_number=r.rectangular_courant, pml_cells=g['pml_cells'], pml_um=g['pml_cells'] * mesh,
                  physical_time_s=r.steps * r.time_step, precision=precision, dft_precision='float64', backend=summary['backend'],
                  cuda_kernel='fused', cuda_graph=summary.get('cuda_graph'), interface_method=r.interface_method,
                  subpixel_quadrature=r.subpixel_quadrature if series == 'smoothed' else None, material_sampling=r.material_sampling,
                  boundary='CPML, cubic sigma profile sigma=40*rho^3/(L+1) in Courant units, kappa=1, alpha=1e-8', **grid),
        staircase=stairs, subpixel=summary.get('subpixel'),
        amplitudes=amplitudes, wavelength_um=wavelength.tolist(),
        efficiencies=dict(T={str(m): T[:, j].tolist() for j, m in enumerate(common.ORDERS)},
                          R={str(m): R[:, j].tolist() for j, m in enumerate(common.ORDERS)}, total=(T.sum(axis=1) + R.sum(axis=1)).tolist()),
        observables={k: v for k, v in common.observables(T, wavelength).items() if k != 'design_index'},
        repeat_max_abs_t1_difference=float(max(np.max(abs(t - t1_runs[-1])) for t in t1_runs)),
        diagnostics=diagnostics, field_peak=dict(sample=summary.get('field_peak'), reference=ref_result.summary.get('field_peak')),
        timing=dict(cost='full_seconds: Simulation(project).run() wall time of one grating solve (setup and stepping), '
                         'CUDA-synchronised; the bare-substrate reference run is recorded separately and not counted',
                    repeats=repeats, warmup=warmup, samples=samples, median_full_seconds=float(np.median(full)),
                    min_full_seconds=float(min(full)), max_full_seconds=float(max(full)),
                    median_stepping_seconds=float(np.median([s['stepping_seconds'] for s in samples])),
                    median_setup_seconds=float(np.median([s['setup_seconds'] for s in samples])), reference=ref_timing,
                    device=torch.cuda.get_device_name(0)),
        host=dict(shared=True, gpu_before=gpu_before, gpu_after=tm.nvidia_smi(), host_cpu_percent_before=cpu_before,
                  host_cpu_percent_after=tm.host_cpu_load_percent(), max_host_cpu_percent=gate[0], max_wait_seconds=gate[1],
                  note='the timed runs need an otherwise idle GPU; the Windows host CPU load is sampled before every run and recorded, '
                       'and with max_host_cpu_percent every run waits until the load is below it (at most max_wait_seconds)'),
        environment=dict(platform=platform.platform(), python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda,
                         packages=tm.versions(('torch', 'cupy-cuda12x', 'numpy')), git=git_state(), **package_state()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', choices=common.SERIES, required=True)
    parser.add_argument('--precision', choices=('float32', 'float64'), required=True)
    parser.add_argument('--meshes', type=float, nargs='+', default=list(common.TORCHFDTD_MESHES_UM))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out-dir', default=str(common.RECORDS))
    parser.add_argument('--max-host-cpu-percent', type=float, default=None, help='wait before each run until the host load is at most this')
    parser.add_argument('--max-wait-seconds', type=float, default=600, help='longest wait for a quiet host before a run starts anyway')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    base = common.load_geometry()
    compare = common.load_compare()
    for mesh in args.meshes:
        assert mesh in common.TORCHFDTD_MESHES_UM, mesh
        record = run_point(base, mesh, args.series, args.precision, args.repeats, compare, (args.max_host_cpu_percent, args.max_wait_seconds))
        out = Path(args.out_dir) / f'torchfdtd-{args.series}-{args.precision}-{mesh:g}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
        t = record['timing']
        print(f"{out.name}: T+1(1.55)={record['observables']['t1_design']:.6f} band={record['observables']['t1_band_mean']:.6f} "
              f"median {t['median_full_seconds']:.3f} s [{t['min_full_seconds']:.3f}, {t['max_full_seconds']:.3f}] "
              f"cells {record['grid']['cell_count']} steps {record['grid']['steps']} repeat spread {record['repeat_max_abs_t1_difference']:.2e}",
              flush=True)


if __name__ == '__main__':
    main()
