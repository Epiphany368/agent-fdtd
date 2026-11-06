"""CPU evidence for bounded material production and exact parameter reduction."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchfdtd import (DifferentiableSolid, StreamedAdjointOptions, StreamedSimulation,
                      smooth_geometry_epsilon)
from torchfdtd.streamed_geometry import (StreamedGeometrySimulation, streamed_geometry,
                                         _SlabMaterial, _GeometryExecution)
from test_differentiable import project


def solids(p):
    return [DifferentiableSolid.box((.53, .41, .37), epsilon=p[0], center=(p[1], .02, -.03),
                                   rotation=(13., -8., 17.)),
            DifferentiableSolid.sphere(p[2], epsilon=2.6, center=(.12, -.06, .04))]


@pytest.mark.parametrize('sampling', ['yee', 'cell'])
def test_producer_and_vjp_match_dense_with_repeated_periodic_indices(sampling):
    scene = project('3d', precision='float64', steps=10)
    scene.region.material_sampling = sampling
    p = torch.tensor([3.1, -.04, .24], dtype=torch.float64, requires_grad=True)
    bg = torch.tensor(1.1, dtype=p.dtype, requires_grad=True)
    material = streamed_geometry(scene.region, solids(p), background=bg, chunk_cells=37)
    producer = _SlabMaterial(material, material.parameters)
    n = scene.region.shape[0]
    indices = torch.arange(-n-2, n+3).remainder(n)
    value = producer.index_select(0, indices)
    dense = smooth_geometry_epsilon(scene.region, solids(p), background=bg, chunk_cells=41)
    torch.testing.assert_close(value, dense.index_select(0, indices), rtol=1e-14, atol=1e-14)
    seed = torch.randn_like(value)
    expected = torch.autograd.grad(dense.index_select(0, indices), (p, bg), seed)
    gradient = torch.zeros_like(material.parameters)
    producer.accumulate_vjp(gradient, seed, indices)
    actual = torch.autograd.grad(material.parameters, (p, bg), gradient)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('periodic,depth', [(False, 2), (True, 9)])
@pytest.mark.parametrize('storage', ['host', 'disk'])
def test_geometry_fields_loss_and_shape_vjp_match_dense(periodic, depth, storage, tmp_path):
    scene = project('3d', precision='float32', steps=10, periodic=periodic)
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    settings = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=depth,
        checkpoints=1, state_storage=storage, state_directory=tmp_path/'banks',
        disk_budget_bytes=128*1024**2)
    material = streamed_geometry(scene.region, solids(p), chunk_cells=73)
    result = StreamedGeometrySimulation(scene, settings)(material)
    actual, = torch.autograd.grad(result.signals.square().sum(), p)
    dense = StreamedSimulation(scene, settings)(smooth_geometry_epsilon(scene.region, solids(p), chunk_cells=73))
    expected, = torch.autograd.grad(dense.signals.square().sum(), p)
    torch.testing.assert_close(result.signals, dense.signals, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-8)
    assert actual.abs().sum() > 0
    assert result.report['dense_material_storage_bytes'] == 0
    assert result.report['dense_material_vjp_bytes'] == 0


class AllocationTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.allocations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        def visit(value):
            if isinstance(value, torch.Tensor):
                self.allocations.append((tuple(value.shape), value.untyped_storage().nbytes()))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
        visit(result)
        return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_parameter_streaming_with_pec_async_cuda_and_disk_banks(tmp_path):
    from torchfdtd import BoundaryFace
    scene = project('3d', precision='float32', steps=36)
    scene.region.material_sampling = 'yee'
    scene.region.boundaries.x_min = BoundaryFace(kind='pec')
    scene.region.boundaries.x_max = BoundaryFace(kind='antisymmetric')
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    options = StreamedAdjointOptions(device='cuda', slab_width=3, temporal_depth=3,
        checkpoints=1, tile_transfers='async', state_storage='disk',
        state_directory=tmp_path/'banks', disk_budget_bytes=128*1024**2)
    actual = StreamedGeometrySimulation(scene, options)(
        streamed_geometry(scene.region, solids(p), background=1.1, chunk_cells=211))
    gradient, = torch.autograd.grad(actual.signals.square().sum(), p)
    dense = StreamedSimulation(scene, options)(smooth_geometry_epsilon(
        scene.region, solids(p), background=1.1, chunk_cells=211))
    expected, = torch.autograd.grad(dense.signals.square().sum(), p)
    torch.testing.assert_close(actual.signals, dense.signals, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gradient, expected, rtol=6e-5, atol=1e-8)
    assert gradient.abs().sum() > 0


def test_metadata_never_materializes_expanded_full_volume_and_geometry_is_small():
    scene = project('3d', precision='float32', steps=10)
    scene.region.size = (12.8, 1.6, 1.6)
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    with AllocationTrace() as trace:
        material = streamed_geometry(scene.region, solids(p), chunk_cells=37)
        execution = _GeometryExecution(material)
        host = execution.host(scene, material.parameters, None)
    cells = scene.region.shape[0]*scene.region.shape[1]*scene.region.shape[2]
    assert host.epsilon.untyped_storage().nbytes() == 4
    assert max(size for _, size in trace.allocations) < cells*4
    assert material.parameters.numel() == 21
    assert all(axis.ndim == 1 for axis in material.axes)
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=2)
    report = execution.reservation(scene, material.parameters, options, None)
    assert report['geometry_workspace_bytes'] > 0
    with pytest.raises(ValueError, match='geometry reservation'):
        from dataclasses import replace
        execution.reservation(scene, material.parameters,
            replace(options, host_budget_bytes=report['host_reservation_bytes']-1), None)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bloch_spectrum_and_parameter_vjp(device):
    from torchfdtd import BoundaryFace
    from test_differentiable import gpu
    if device == 'cuda':
        gpu()
    scene = project('3d', precision='float32', steps=10)
    scene.region.material_sampling = 'yee'
    scene.region.boundaries.x_min = scene.region.boundaries.x_max = BoundaryFace(kind='bloch')
    scene.region.bloch_phase = (.43, 0., 0.)
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    options = StreamedAdjointOptions(device=device, slab_width=3, temporal_depth=2,
        tile_transfers='async' if device == 'cuda' else 'sync', checkpoints=1)
    frequencies = [.04/scene.region.time_step]
    result = StreamedGeometrySimulation(scene, options).spectrum(
        streamed_geometry(scene.region, solids(p), chunk_cells=211), frequencies)
    actual, = torch.autograd.grad((result.fields.abs()/scene.region.time_step).square().sum(), p)
    dense = StreamedSimulation(scene, options).spectrum(
        smooth_geometry_epsilon(scene.region, solids(p), chunk_cells=211), frequencies)
    expected, = torch.autograd.grad((dense.fields.abs()/scene.region.time_step).square().sum(), p)
    torch.testing.assert_close(result.fields/scene.region.time_step, dense.fields/scene.region.time_step,
                               rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=1e-8)


def test_planes_flux_and_normalized_flux_vjp():
    from torchfdtd import (FieldMonitor, DifferentiablePlaneSimulation,
                          StreamedGeometryPlaneSimulation)
    scene = project('3d', precision='float32', steps=10, periodic=True)
    scene.monitors = [FieldMonitor(center=(.12, 0, 0), size=(0, .45, .45), normal='x', downsample=3)]
    identifier = scene.monitors[0].id
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=2, checkpoints=1)
    frequencies = [.04/scene.region.time_step]
    streamed = StreamedGeometryPlaneSimulation(scene, options)
    dense = DifferentiablePlaneSimulation(scene, options)
    reference = streamed(streamed_geometry(scene.region, [], background=1.1), frequencies)[identifier]
    actual = streamed(streamed_geometry(scene.region, solids(p), background=1.1, chunk_cells=211), frequencies)[identifier]
    wanted = dense(smooth_geometry_epsilon(scene.region, solids(p), background=1.1, chunk_cells=211), frequencies)[identifier]
    torch.testing.assert_close(actual.fields, wanted.fields, rtol=1e-5, atol=1e-25)
    loss = actual.normalized_flux(reference).sum()
    expected_loss = wanted.normalized_flux(reference).sum()
    grad, = torch.autograd.grad(loss, p)
    expected, = torch.autograd.grad(expected_loss, p)
    torch.testing.assert_close(grad, expected, rtol=5e-5, atol=1e-7)
    assert grad.abs().sum() > 0


def test_long_disk_run_allocates_no_full_domain_material_or_vjp(tmp_path):
    scene = project('3d', precision='float32', steps=10)
    scene.region.size = (6.4, 1.5, 1.4)
    scene.region.material_sampling = 'yee'
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=2,
        state_storage='disk', state_directory=tmp_path/'banks', disk_budget_bytes=256*1024**2,
        checkpoints=0)
    with AllocationTrace() as trace:
        result = StreamedGeometrySimulation(scene, options)(
            streamed_geometry(scene.region, solids(p), chunk_cells=257))
        torch.autograd.grad(result.signals.square().sum(), p)
    full_shapes = (scene.region.shape, scene.region.shape+(3,), scene.region.shape+(1,))
    assert not [(shape, size) for shape, size in trace.allocations if shape in full_shapes and size > 4]
    assert result.report['max_extended_tile_cells']*8 < torch.tensor(scene.region.shape).prod()
    assert result.report['forward_backing_store']['closed']
    assert result.report['backward_backing_store']['closed']


def test_replay_snapshots_caller_geometry_axes_and_region():
    scene = project('3d', precision='float32', steps=10)
    p = torch.tensor([3.1, -.04, .24], requires_grad=True)
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=2, checkpoints=1)
    model = StreamedGeometrySimulation(scene, options)
    geometry = streamed_geometry(scene.region, solids(p), chunk_cells=211)
    result = model(geometry)
    # Caller edits must not change the material replayed for the completed run.
    for axis in geometry.axes[::3]:
        axis.add_(.3)
    geometry.region.size = (3.2, 1.5, 1.4)
    actual, = torch.autograd.grad(result.signals.square().sum(), p)
    fresh = model(streamed_geometry(scene.region, solids(p), chunk_cells=211))
    expected, = torch.autograd.grad(fresh.signals.square().sum(), p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result.report['geometry_snapshot_axes_bytes'] == result.report['geometry_axes_bytes']


@pytest.mark.parametrize('budget,counts', [(1, None), (1024**2, (100000, 100000))])
def test_plane_layout_preflight_rejects_before_point_or_map_allocation(monkeypatch, budget, counts):
    from torchfdtd import FieldMonitor, StreamedGeometryPlaneSimulation
    import torchfdtd.adjoint_planes as planes
    scene = project('3d', precision='float32', steps=10)
    scene.monitors = [FieldMonitor(center=(.12, 0, 0), size=(0, .45, .45), normal='x')]
    def forbidden(*args, **kwargs):
        pytest.fail('Plane layout allocation happened before admission.')
    monkeypatch.setattr(planes, 'plane_plan', forbidden)
    monkeypatch.setattr(planes, 'interpolation_map', forbidden)
    quadrature = None if counts is None else {scene.monitors[0].id: counts}
    with pytest.raises(ValueError, match='Plane layout construction exceeds'):
        StreamedGeometryPlaneSimulation(scene,
            StreamedAdjointOptions(device='cpu', host_budget_bytes=budget), quadrature_counts=quadrature)
