"""Scalar paths reject active tensors and tolerate unused library entries."""
import numpy as np
from fastapi.testclient import TestClient
from torchfdtd.models import Material
from torchfdtd.server import create_app
from torchfdtd.solver import voxelize, estimate
from test_solver import small


def test_unused_tensor_does_not_change_scalar_voxelization():
    project = small()
    before = voxelize(project)[0]
    project.materials.append(Material(name='Unused tensor', model='tensor'))
    np.testing.assert_array_equal(voxelize(project)[0], before)
    assert estimate(project)['cells'] == np.prod(project.region.shape)


def test_tensor_has_no_misleading_scalar_optical_preview(tmp_path):
    client = TestClient(create_app(tmp_path))
    result = client.post('/api/materials/preview', json=Material(name='Tensor', model='tensor').model_dump())
    assert result.status_code == 422
    assert 'tensor' in result.json()['detail'].lower()
