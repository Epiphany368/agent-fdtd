"""Environment diagnosis for `torchfdtd doctor`.

The report lists Python, torch, CuPy, the CUDA runtime and driver, the device,
whether the fused kernels compile and launch, and the backend a default project
would select. Every unsupported situation gets one specific message. `notice`
entries describe a supported fallback (CPU execution, the torch kernels on the
GPU); `error` entries describe an installation that cannot run as configured.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import torch


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi():
    """Return (driver version, CUDA version the driver supports) from nvidia-smi, or (None, None)."""
    try:
        completed = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if completed.returncode != 0:
        return None, None
    driver = re.search(r'Driver Version:\s*([0-9.]+)', completed.stdout)
    cuda = re.search(r'CUDA Version:\s*([0-9.]+)', completed.stdout)
    return (driver.group(1) if driver else None), (cuda.group(1) if cuda else None)


def _version_tuple(text):
    return tuple(int(part) for part in text.split('.')[:2])


def _cupy_distributions():
    """Every installed distribution named cupy or cupy-* (cupy-cuda11x/12x/13x, cupy-rocm-*), name -> version."""
    found = {}
    for distribution in metadata.distributions():
        name = (distribution.metadata['Name'] or '').lower()
        if name.startswith('cupy'):
            found[name] = distribution.version
    return dict(sorted(found.items()))


def probe_cupy():
    """Import CuPy the way the solver does; report the version or the failure text."""
    distributions = _cupy_distributions()
    record = dict(version=next(iter(distributions.values()), None), distributions=distributions, importable=False, error=None)
    try:
        from .cuda_bootstrap import prepare_cuda_kernels
        module = prepare_cuda_kernels()
        record.update(importable=True, version=module.__version__)
    except RuntimeError as exc:
        record['error'] = str(exc)
    return record


def probe_fused_kernels():
    """Compile and launch the fused Yee/CPML kernels on a 20 x 20 cell, 20 step 2D grid."""
    from .models import Monitor, Project, Region, Source
    from .solver import Simulation
    project = Project(name='doctor', region=Region(dimension='2d', size=(2, 2, .1), mesh=.1, pml_cells=3, steps=20,
                                                   backend='cuda', cuda_kernel='fused', cuda_monitor_kernel='fused'),
                      sources=[Source(wavelength=1, pulse_cycles=1)], monitors=[Monitor(center=(.4, 0, 0))])
    result = Simulation(project).run()
    if not np.isfinite(result.electric).all():
        raise RuntimeError('The fused kernel launch produced non-finite fields.')
    return dict(cells=int(result.electric[..., 0].size), steps=project.region.steps)


def diagnose(probe=probe_fused_kernels):
    """Assemble the report without printing. `probe` is the fused-kernel launch, replaceable in tests."""
    checks = []

    def check(name, status, message):
        checks.append(dict(name=name, status=status, message=message))

    report = dict(torchfdtd=_package_version('torchfdtd'), python=platform.python_version(),
                  platform=platform.platform(), machine=platform.machine())
    cuda_runtime = torch.version.cuda
    cuda = bool(torch.cuda.is_available())
    report['torch'] = dict(version=torch.__version__, cuda_runtime=cuda_runtime, cuda_available=cuda)
    driver, driver_cuda = _nvidia_smi()
    report['driver'] = dict(version=driver, cuda_version=driver_cuda)
    cupy = report['cupy'] = probe_cupy()
    report['device'] = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        report['device'] = dict(name=properties.name, capability=f'{properties.major}.{properties.minor}',
                                memory_total_bytes=int(total), memory_free_bytes=int(free), count=torch.cuda.device_count())
        check('cuda', 'ok', f'CUDA device {properties.name} (compute capability {properties.major}.{properties.minor}) '
                            f'with torch {torch.__version__} on CUDA runtime {cuda_runtime}.')
    elif cuda_runtime is None:
        check('cuda', 'notice', f'This torch build ({torch.__version__}) has no CUDA support, so every simulation runs '
                                'on the CPU. For GPU execution install a CUDA-enabled torch wheel from '
                                'https://pytorch.org/get-started/locally/ and an NVIDIA driver for that CUDA version.')
    elif driver is None:
        check('cuda', 'notice', f'torch {torch.__version__} was built for CUDA {cuda_runtime} but no NVIDIA driver was '
                                'found (nvidia-smi is absent or failed), so every simulation runs on the CPU. Install '
                                'the NVIDIA driver for this GPU, or use a CPU torch build to avoid the unused download.')
    else:
        check('cuda', 'error', f'torch {torch.__version__} was built for CUDA {cuda_runtime} and NVIDIA driver {driver} '
                               f'is installed, but torch reports no usable CUDA device. Check that the GPU is visible '
                               '(nvidia-smi lists it), that CUDA_VISIBLE_DEVICES does not hide it, and that the driver '
                               'supports the CUDA runtime listed above.')
    if cuda_runtime is not None and driver_cuda is not None:
        # A driver runs every runtime of its own major CUDA version (minor-version
        # compatibility); an older major fails at context creation.
        needed = _version_tuple(cuda_runtime)[0]
        if _version_tuple(driver_cuda)[0] < needed:
            check('driver', 'error', f'NVIDIA driver {driver} supports CUDA {driver_cuda} at most, but torch '
                                     f'{torch.__version__} needs a driver for CUDA {needed}.x (it was built for '
                                     f'CUDA {cuda_runtime}). Update the driver or install a torch wheel built for '
                                     f'CUDA {driver_cuda}.')
        else:
            check('driver', 'ok', f'NVIDIA driver {driver} supports CUDA {driver_cuda}; torch needs CUDA {cuda_runtime}.')
    if cupy['importable']:
        check('cupy', 'ok', f'CuPy {cupy["version"]} imports; the fused CUDA kernels and streamed GPU tiles are available.')
    elif cupy['version'] is None:
        check('cupy', 'notice', 'CuPy is not installed, so the fused CUDA kernels and streamed GPU tiles are unavailable '
                                'and CUDA projects run the torch kernels. Install the extra: pip install "torchfdtd[cuda-kernels]".')
    else:
        installed = ', '.join(cupy.get('distributions') or ()) or 'cupy'
        check('cupy', 'error', f'CuPy {cupy["version"]} ({installed}) is installed but cannot be imported: {cupy["error"]} '
                               'Install a cupy-cuda12x build that matches the CUDA runtime of this torch and the driver.')
    fused = report['fused_kernels'] = dict(status='not_applicable', detail=None, seconds=None)
    if cuda and cupy['importable']:
        started = time.perf_counter()
        try:
            detail = probe()
            fused.update(status='ok', detail=detail, seconds=round(time.perf_counter() - started, 3))
            check('fused_kernels', 'ok', f'The fused Yee/CPML kernels compiled and ran {detail["steps"]} steps on '
                                         f'{detail["cells"]} cells in {fused["seconds"]} s.')
        except Exception as exc:  # noqa: BLE001 - any compile or launch failure is the finding to report
            fused.update(status='error', detail=f'{type(exc).__name__}: {exc}', seconds=round(time.perf_counter() - started, 3))
            check('fused_kernels', 'error', f'The fused CUDA kernels failed to compile or launch: {type(exc).__name__}: {exc} '
                                            'CUDA projects still run with cuda_kernel="torch". Check that the CuPy build '
                                            'matches the CUDA runtime and that NVRTC is available.')
    else:
        fused['detail'] = 'skipped: no CUDA device' if not cuda else 'skipped: CuPy is not importable'
    if cuda and cupy['importable'] and fused['status'] == 'ok':
        backend = dict(backend='cuda', cuda_kernel='fused', reason='CUDA device and CuPy available')
    elif cuda:
        backend = dict(backend='cuda', cuda_kernel='torch', reason='CUDA device available; fused kernels ' +
                       ('failed' if fused['status'] == 'error' else 'need CuPy'))
    else:
        backend = dict(backend='cpu', cuda_kernel='torch', reason='no CUDA device')
    report['backend'] = backend
    check('backend', 'ok', f'A project with backend="auto" runs on {backend["backend"]} with the {backend["cuda_kernel"]} '
                           f'kernels ({backend["reason"]}).')
    report['checks'] = checks
    report['ok'] = not any(item['status'] == 'error' for item in checks)
    return report


def format_report(report):
    lines = [f'torchfdtd {report["torchfdtd"] or "(not installed as a distribution)"} on Python {report["python"]}, '
             f'{report["platform"]}',
             f'torch {report["torch"]["version"]}: CUDA runtime {report["torch"]["cuda_runtime"] or "none"}, '
             f'device available: {report["torch"]["cuda_available"]}',
             f'NVIDIA driver {report["driver"]["version"] or "none found"}'
             + (f' (supports CUDA {report["driver"]["cuda_version"]})' if report['driver']['cuda_version'] else ''),
             f'CuPy {report["cupy"]["version"] or "not installed"}: importable {report["cupy"]["importable"]}']
    if report['device']:
        device = report['device']
        lines.append(f'device: {device["name"]}, compute capability {device["capability"]}, '
                     f'{device["memory_free_bytes"] / 2**30:.2f} of {device["memory_total_bytes"] / 2**30:.2f} GiB free')
    for item in report['checks']:
        lines.append(f'[{item["status"].upper()}] {item["name"]}: {item["message"]}')
    lines.append('result: ' + ('OK' if report['ok'] else 'ERROR (see the entries above)'))
    return '\n'.join(lines)


def main(as_json=False, stream=None):
    report = diagnose()
    stream = stream or sys.stdout
    print(json.dumps(report, indent=2) if as_json else format_report(report), file=stream)
    return 0 if report['ok'] else 1
