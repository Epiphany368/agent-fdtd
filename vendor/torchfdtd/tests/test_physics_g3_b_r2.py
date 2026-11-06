"""Revision-2 items of the G3-08 fixture that must not change tests/test_physics_g3_b.py.

The first G3-08 case declared the layer-A tolerance as rtol 1e-4 with atol 0 and one
instance failed it (TM, 20 degrees, 0.92 um: 1.15e-4 on an efficiency of 0.022, an
absolute 2.6e-6); its FAILED run stays on record. The revision-2 case
docs/validation/cases/G3-08r2_bloch_grating_rcwa_layer_a.json restates only that
tolerance as the program's pair, rtol 1e-4 and atol 1e-6, and this module carries the
test that applies it. The physics criteria and their tests are unchanged and are run
together with this module for the revision-2 record:

  TORCHFDTD_G3_FULL=1 TORCHFDTD_G3_RECORD=docs/validation/g3/r2 python -m pytest -q -p no:cacheprovider \
      tests/test_physics_g3_b.py tests/test_physics_g3_b_r2.py -k "g3_08 and not g3_08_grating_cuda_layer_a"

The G3-05 item that recorded the rejection of dispersive materials by the subpixel
operator was removed when dispersive subpixel interfaces were added
(docs/SUBPIXEL_INTERFACES.md).
"""
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from test_physics_g3_b import (CASES, CUDA, full_only, grating_configurations, grating_row, grating_row_id, needs_cuda,
                               check_grating_row)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = os.environ.get('TORCHFDTD_G3_RECORD')
R2_CASE = json.loads((ROOT/'docs'/'validation'/'cases'/'G3-08r2_bloch_grating_rcwa_layer_a.json').read_text(encoding='utf-8'))
LAYER_A_R2 = R2_CASE['acceptance']['layer_a_cuda_fp32_vs_cpu_fp64']


def record_file(name, case_id, row_id, payload, directory=None):
    """Write one row into <record dir>/<name>.json; the revision-2 rows never touch the first records."""
    directory = directory or RECORD_DIR
    if not directory:
        return
    target = Path(directory)/f'{name}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(target.read_text(encoding='utf-8')) if target.exists() else dict(
        task=name.split('r2')[0].split('_')[0], case_id=case_id, fixture_path=f'docs/validation/cases/{case_id}.json',
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         os=platform.platform(), gpu=torch.cuda.get_device_name() if CUDA else None), rows={})
    data['generated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    data['rows'][row_id] = payload
    target.write_text(json.dumps(data, indent=1, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


@pytest.mark.parametrize('polarization,angle,wavelength', grating_configurations())
def test_g3_08r2_grating_cuda_layer_a(polarization, angle, wavelength):
    """The same discrete problem at h = 0.01 um in CPU FP64 and CUDA FP32 against rtol 1e-4 and atol 1e-6 (revision 2)."""
    needs_cuda()
    full_only('CPU FP64 grating runs at the full duration')
    mesh = CASES['G3-08']['fixture']['mesh_sequence_um'][0]
    cpu = grating_row(polarization, angle, wavelength, mesh, 'cpu', 'float64', 'subpixel')
    cuda = grating_row(polarization, angle, wavelength, mesh, 'cuda', 'float32', 'subpixel')
    rtol, atol = LAYER_A_R2['rtol'], LAYER_A_R2['atol']
    excess, worst_relative, instances = -1., 0., {}
    for m, entry in cpu['orders'].items():
        for name in ('transmission', 'reflection'):
            a, b = entry[name]['efficiency'], cuda['orders'][m][name]['efficiency']
            difference = abs(b-a)
            excess = max(excess, difference-(rtol*abs(a)+atol))
            worst_relative = max(worst_relative, difference/max(abs(a), 1e-6))
            instances[f'{name} m={m}'] = dict(cpu=a, cuda=b, absolute_difference=difference, allowance=rtol*abs(a)+atol)
    verdict = dict(layer_a_r2_rtol=rtol, layer_a_r2_atol=atol, layer_a_r2_max_excess=excess, layer_a_r2_pass=excess <= 0.,
                   layer_a_max_relative_difference=worst_relative, instances=instances)
    record_file('G3-08r2', R2_CASE['case_id'], grating_row_id(cpu), dict(cpu, judged_against_torcwa=False, failures=check_grating_row(cpu)))
    record_file('G3-08r2', R2_CASE['case_id'], grating_row_id(cuda), dict(cuda, **verdict))
    assert excess <= 0., f'layer A r2 {polarization} {angle:g} deg {wavelength} um: excess {excess:.2e} over rtol {rtol} atol {atol}'

