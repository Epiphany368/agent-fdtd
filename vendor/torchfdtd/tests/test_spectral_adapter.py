import numpy as np
import pytest
import torch

from benchmarks.open_source import upstream_run
from benchmarks.spectral_ensemble import projects_for, error


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_external_fused_observer_preserves_upstream_fields_and_spectra():
    pytest.importorskip('cupy')
    p=projects_for('sphere',32,800,1)[0]
    _,reference=upstream_run(p,with_planes=True,plane_kernel='torch')
    _,fused=upstream_run(p,with_planes=True,plane_kernel='fused')
    for a,b in zip(fused[:3],reference[:3]):
        np.testing.assert_array_equal(a,b)
    assert max(error(a,b)['relative_l2'] for a,b in zip(fused[3:],reference[3:])) < 3e-6
