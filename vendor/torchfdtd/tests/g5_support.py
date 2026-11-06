"""Shared fixture, limits and record writer of the G5 memory-tier tests.

One public synthetic scatterer serves G5-01 to G5-04: a smooth sphere of
permittivity 4 in air, a Gaussian point source and either two point probes or
one x-normal spectral plane behind the sphere. Limits are read from the case
files so the assertions and the evidence agree; records are written through to
docs/validation/g5/<task>.json, one entry per measured instance (the committed
file only with TORCHFDTD_WRITE_RECORDS=1, see tests/record_output.py).
"""
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from torchfdtd import (BoundaryFace, FieldMonitor, Monitor, Project, Region, Source, StreamedAdjointOptions,
                       smooth_sphere_epsilon)
from record_output import record_path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'docs' / 'validation' / 'cases'
CUDA = torch.cuda.is_available()

MESH = .1
SIZE = (2.4, 1.6, 1.6)
STEPS = 40
PML_CELLS = 3
FREQUENCY_HZ = [3e14]
SOURCE = dict(center=(-.7, 0, 0), wavelength=.8, pulse='gaussian', time_definition='standard',
              pulse_length=1.5e-15, pulse_offset=2e-15, amplitude=100)
SPHERE = dict(radius=.3, width=.08)
DRUDE = dict(epsilon_inf=2., strength_rad_s_squared=4e30, damping_rad_s=1e14)


def scene(precision='float32', observation='point', faces='cpml', steps=STEPS):
    """The G5 scatterer: 24 x 16 x 16 cells at 0.1 um, 3 PML cells, 40 steps of the standard Gaussian pulse."""
    region = Region(dimension='3d', size=SIZE, mesh=MESH, pml_cells=PML_CELLS, steps=steps, precision=precision,
                    backend='cpu', material_sampling='yee')
    if faces == 'pmc':
        region.boundaries.y_min = BoundaryFace(kind='pmc')
        region.boundaries.y_max = BoundaryFace(kind='pmc')
        region.boundaries.z_min = BoundaryFace(kind='symmetric')
        region.boundaries.z_max = BoundaryFace(kind='pec')
    if observation == 'plane':
        monitors = [FieldMonitor(center=(.7, 0, 0), size=(0, .8, .8), normal='x', downsample=2)]
    else:
        monitors = [Monitor(component='Ez', center=(.7, 0, 0)), Monitor(component='Hy', center=(.5, .2, 0))]
    project = Project(region=region, sources=[Source(**SOURCE)], monitors=monitors)
    assert tuple(project.region.shape) == (24, 16, 16), project.region.shape
    return project


def sphere(project, *, inside, outside, diagonal=False):
    dtype = torch.float64 if project.region.precision == 'float64' else torch.float32
    value = smooth_sphere_epsilon(project.region, torch.tensor(SPHERE['radius'], dtype=dtype), inside=inside,
                                  outside=outside, width=SPHERE['width'], yee=diagonal)
    return value.detach().contiguous()


def streamed_options(path, scratch, **overrides):
    """host_sync, disk_sync and host_async: the three streamed tiers of G5-01 on CUDA."""
    storage, transfers = path.split('_')
    values = dict(device='cuda', slab_width=6, temporal_depth=4, checkpoints=2, state_storage=storage,
                  state_directory=scratch if storage == 'disk' else None,
                  disk_budget_bytes=256*2**20 if storage == 'disk' else None,
                  tile_transfers=transfers, tile_buffers=2)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


def limits(case, *keys):
    node = json.loads((CASES / f'{case}.json').read_text(encoding='utf-8'))['acceptance']
    for key in keys:
        node = node[key]
    return node


def tolerance(case, precision):
    return dict(limits(case, 'tolerances', precision))


# ----------------------------------------------------------------------------------------------- records
def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
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
    env = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
               cuda_runtime=torch.version.cuda, cuda_available=CUDA, gpu=torch.cuda.get_device_name(0) if CUDA else None,
               os=platform.platform(), machine=platform.machine(), cpu=platform.processor(),
               commit=git_text('rev-parse', 'HEAD'), dirty_paths=len((git_text('status', '--porcelain') or '').splitlines()),
               recorded_at=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat())
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
    def __init__(self, task, case, module):
        self.task, self.case, self.module = task, case, module

    @property
    def path(self):
        return record_path(f'docs/validation/g5/{self.task}.json')

    def add(self, key, **values):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.task in _RESET and self.path.is_file():
            data = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            _RESET.add(self.task)
            data = dict(task=self.task, case=f'docs/validation/cases/{self.case}.json', test_module=self.module,
                        environment=environment(), entries={})
        data['entries'][key] = clean(values)
        with open(self.path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
