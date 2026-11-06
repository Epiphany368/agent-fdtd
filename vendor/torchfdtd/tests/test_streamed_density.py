"""Periodic density slabs and the production response chain without dense maps."""
import pytest
import torch

from torchfdtd import (AdjointBatchOptions, AdjointExecutionPolicy, AdjointOptions,
    PeriodicLayerResponse, PlaneReferenceCache, StreamedAdjointOptions,
    periodic_density_layer, periodic_layer_response, streamed_density_layer)
from torchfdtd.periodic_response import _periodic_project
from torchfdtd.streamed_density import _DensityExecution, _DensityMaterial
from test_streamed_geometry import AllocationTrace


SPEC = dict(wavelength_um=.5, background_index=1.4, design_index=1.8, period_um=(1.2, 1.2),
    height_um=.3, detector_offset_um=.5, theta_inside_rad=.1, phi_rad=.3)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('origin', ['cell_edges', 'sample_centers'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_density_slabs_and_repeated_seam_vjp_match_dense(origin, dtype):
    project, _ = _periodic_project(SPEC, dtype=dtype, mesh=.1, steps=10,
                                  pml_cells=3, forward_kernel='torch', memory_mode='streamed')
    density = torch.linspace(.1, .9, 15, dtype=dtype).reshape(3, 5).requires_grad_(True)
    settings = dict(bottom_um=-.135, top_um=.174, background_epsilon=1.96,
                    design_epsilon=3.24, pixel_origin=origin)
    geometry = streamed_density_layer(density, project.region, **settings)
    execution = _DensityExecution(geometry)
    execution.snapshot()
    producer = _DensityMaterial(execution.geometry, density)
    n = project.region.shape[0]
    indices = torch.arange(-n-3, n+5).remainder(n)
    slab = producer.index_select(0, indices)
    dense = periodic_density_layer(density, project.region, **settings)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(slab, dense.index_select(0, indices), rtol=tolerance, atol=tolerance)
    seed = torch.randn((3, *slab.shape[:-1]), dtype=dtype,
                       generator=torch.Generator().manual_seed(81)).movedim(0, -1)
    assert not seed.is_contiguous()
    expected, = torch.autograd.grad(dense.index_select(0, indices), density, seed)
    actual = torch.zeros_like(density)
    producer.accumulate_vjp(actual, seed, indices)
    torch.testing.assert_close(actual, expected, rtol=4*tolerance, atol=4*tolerance)
    assert actual.norm() > 0


def make_model(tmp_path, *, storage='host', spec=SPEC, steps=60, device='cpu', budget=512*1024**2):
    settings = dict(mesh=.2, steps=steps, pml_cells=3, quadrature_counts=(3, 3),
                    forward_kernel='torch')
    options = StreamedAdjointOptions(device=device, slab_width=2, temporal_depth=2,
        checkpoints=1, state_storage=storage, state_directory=tmp_path/'banks',
        disk_budget_bytes=budget, host_budget_bytes=budget, gpu_budget_bytes=budget,
        tile_transfers='async' if device == 'cuda' else 'sync')
    policy = AdjointExecutionPolicy(device=device, streamed=options, host_budget_bytes=budget)
    model = PeriodicLayerResponse(spec, density_shape=(3, 2), dtype=torch.float32,
        policy=policy, batch_options=AdjointBatchOptions(host_budget_bytes=budget,
            gpu_budget_bytes=budget, disk_budget_bytes=budget),
        reference_cache=PlaneReferenceCache(1024**2), **settings)
    return model, settings


def objective(response):
    return (response*response.new_tensor([[.1, .3, -.2, .7], [.4, -.1, .5, .2]])).sum()


def test_density_prepared_case_snapshots_mutable_quadrature_configuration(tmp_path, monkeypatch):
    model, _ = make_model(tmp_path, steps=10)
    case = model._batch._cases[0]
    case.check_fixed()
    original = dict(case.spec.quadrature_counts)
    identifier = next(iter(original))
    case.spec.quadrature_counts[identifier] = (5, 5)
    monkeypatch.setattr(case.model, 'prepare',
        lambda *args: pytest.fail('Prepared material after configuration mutation.'))
    with pytest.raises(RuntimeError, match='Batch case configuration changed'):
        case.reservation((torch.full((3, 2), .4),))
    case.spec.quadrature_counts.clear()
    case.spec.quadrature_counts.update(original)
    case.check_fixed()


@pytest.mark.parametrize('storage', ['host', 'disk'])
def test_periodic_response_routes_density_without_dense_material_and_matches_reference(tmp_path, monkeypatch, storage):
    import torchfdtd.periodic_adjoint as implementation
    model, settings = make_model(tmp_path, storage=storage)
    density = torch.tensor([[.2, .4], [.5, .3], [.7, .6]], requires_grad=True)
    reference = periodic_layer_response(density, SPEC, **settings, options=AdjointOptions(checkpoints=1))
    expected, = torch.autograd.grad(objective(reference), density)
    monkeypatch.setattr(implementation, 'periodic_density_layer',
        lambda *args, **kwargs: pytest.fail('Streamed periodic response allocated a dense material map.'))
    actual = model(density)
    gradient, = torch.autograd.grad(objective(actual), density)
    torch.testing.assert_close(actual, reference, rtol=8e-5, atol=3e-6)
    torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=5e-6)
    assert gradient.norm() > 1e-5
    assert model.plan()['material_input'] == 'streamed_density'
    assert model.last_report['batch']['parameter_bytes'] == density.numel()*density.element_size()
    assert model.last_report['batch']['replayed_cases'] == 2
    assert model.last_report['reference_cache_misses'] == 1
    with torch.no_grad():
        model(density+.01)
    assert model.last_report['reference_cache_hits'] == 1


def test_density_snapshot_survives_mutable_axes_and_region():
    project, _ = _periodic_project(SPEC, dtype=torch.float32, mesh=.2, steps=10,
        pml_cells=3, forward_kernel='torch', memory_mode='streamed')
    density = torch.ones((3, 2))*.3
    geometry = streamed_density_layer(density, project.region, bottom_um=-.15, top_um=.15,
                                     background_epsilon=1.96, design_epsilon=3.24)
    execution = _DensityExecution(geometry)
    execution.snapshot()
    producer = _DensityMaterial(execution.geometry, density)
    indices = torch.tensor([3, 0, 1])
    expected = producer.index_select(0, indices)
    for axis in geometry.axes:
        axis.add_(.7)
    geometry.region.size = (1.6, .8, 12.)
    torch.testing.assert_close(producer.index_select(0, indices), expected, rtol=0, atol=0)


def test_periodic_density_full_response_trace_has_no_global_epsilon_or_vjp(tmp_path):
    spec = dict(SPEC, period_um=(3.2, 1.2))
    model, _ = make_model(tmp_path, storage='disk', spec=spec, steps=45)
    shape = model.project.region.shape
    density = torch.tensor([[.2, .4], [.5, .3], [.7, .6]], requires_grad=True)
    with AllocationTrace() as trace:
        response = model(density)
        gradient, = torch.autograd.grad(objective(response), density)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-6
    full = (shape, shape+(3,), shape+(1,))
    assert not [(s, size) for s, size in trace.allocations if s in full and size > 8]
    assert model.plan()['dense_material_storage_bytes'] == 0


def test_density_plan_never_synthesizes_material_and_tiny_budget_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(_DensityMaterial, 'index_select',
        lambda *args, **kwargs: pytest.fail('Admission synthesized material.'))
    with pytest.raises(ValueError, match='host budget'):
        make_model(tmp_path, budget=1)
    model, _ = make_model(tmp_path)
    shape = model.project.region.shape
    with AllocationTrace() as trace:
        plan = model.plan()
    assert plan['material_input'] == 'streamed_density'
    assert not [(s, size) for s, size in trace.allocations if s in (shape, shape+(3,)) and size > 4]


def test_async_cuda_periodic_density_response_and_gradient_match_dense_cpu(tmp_path):
    from test_differentiable import gpu
    gpu()
    model, settings = make_model(tmp_path, device='cuda')
    density = torch.tensor([[.2, .4], [.5, .3], [.7, .6]], requires_grad=True)
    reference = periodic_layer_response(density, SPEC, **settings, options=AdjointOptions(checkpoints=1))
    expected, = torch.autograd.grad(objective(reference), density)
    response = model(density)
    actual, = torch.autograd.grad(objective(response), density)
    torch.testing.assert_close(response, reference, rtol=1e-4, atol=4e-6)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=7e-6)
    assert actual.norm() > 1e-5
    assert model.last_report['batch']['parameter_bytes'] == density.numel()*density.element_size()
    reports = model.last_report['batch']['replay_reports']
    assert all(report['tile_transfers'] == 'async' for report in reports)
