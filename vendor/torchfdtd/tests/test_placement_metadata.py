"""Float32 field precision must not round placement metadata.

Regression for the `float32` blocker recorded in the sibling project's plan step 23:
a modal detector coordinate stored in the field dtype cannot satisfy the E-node
placement rule at device-scale x. Float32 resolution at ~275 um is ~1.6e-5 um,
against a where-in-the-cell tolerance of 2e-5 (q units, i.e. 8e-7 um at 40 nm).

The fix keeps the coordinate in ``points_um_geometry`` (float64) while
``points_um`` stays in the field dtype for field arithmetic
(``detector_allocation`` requires field-dtype coordinates).
"""
import numpy as np
import torch

from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.field_monitors import plane_plan
from torchfdtd.models import BoundaryFace, Boundaries, FieldMonitor, Region


def _region(precision='float32'):
    boundaries = Boundaries(
        x_min=BoundaryFace(kind='pml', layers=16), x_max=BoundaryFace(kind='pml', layers=16),
        y_min=BoundaryFace(kind='pml', layers=20), y_max=BoundaryFace(kind='pml', layers=20),
        z_min=BoundaryFace(kind='periodic'), z_max=BoundaryFace(kind='periodic'))
    return Region(dimension='3d', size=(600., 6., 0.2), mesh=0.04, mesh_steps=(0.04, 0.02, 0.04),
                  pml_cells=16, steps=10, precision=precision, material_sampling='yee',
                  mesh_type='uniform', boundaries=boundaries)


def _snapped_monitor(region):
    span = region.size[0]
    dx = 0.04
    x = -span / 2 + round((275.54 + span / 2) / dx) * dx
    return FieldMonitor(id='o', name='o', center=(x, 0, 0), size=(0, 5., 0.2), normal='x')


def test_plane_plan_returns_float64_coordinates_even_at_float32_precision():
    plan = plane_plan(_region('float32'), _snapped_monitor(_region('float32')))
    assert plan['points_um'].dtype == np.float64


def test_float32_field_dtype_cannot_hold_a_device_scale_e_node():
    region = _region('float32')
    plan = plane_plan(region, _snapped_monitor(region))
    origin, step = float(region.mesh_nodes[0][0]), 0.04
    x64 = torch.as_tensor(plan['points_um'], dtype=torch.float64)
    x32 = x64.to(torch.float32)
    q = lambda value: abs((float(value) - origin) / step - round((float(value) - origin) / step))
    assert q(x64[0, 0]) <= 2e-5          # float64 metadata satisfies the E-node rule
    assert q(x32[0, 0]) > 2e-5           # the leak this regression guards against
    assert x32[0, 0].item() != x64[0, 0].item()


def test_plane_result_carries_float64_placement_metadata():
    result = DifferentiablePlaneResult(
        fields=torch.zeros((1, 2, 6), dtype=torch.complex64),
        frequency_hz=torch.zeros(1), points_um=torch.zeros((2, 3), dtype=torch.float32),
        weights=torch.ones(2), shape=(1, 2, 1), normal='x', run_signature='s', report={},
        points_um_geometry=torch.zeros((2, 3), dtype=torch.float64))
    assert result.points_um.dtype == torch.float32
    assert result.points_um_geometry.dtype == torch.float64
    assert DifferentiablePlaneResult(
        fields=torch.zeros((1, 2, 6), dtype=torch.complex64), frequency_hz=torch.zeros(1),
        points_um=torch.zeros((2, 3)), weights=torch.ones(2), shape=(1, 2, 1), normal='x',
        run_signature='s', report={}).points_um_geometry is None
