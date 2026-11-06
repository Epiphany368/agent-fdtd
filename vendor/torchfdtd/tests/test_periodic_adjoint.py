from dataclasses import replace

import pytest
import torch

from torchfdtd import (AdjointBatchOptions, AdjointExecutionPolicy, AdjointOptions,
    PeriodicLayerResponse, PlaneReferenceCache, StreamedAdjointOptions,
    periodic_layer_response, spectral_pupil_response)


SPEC = dict(wavelength_um=.5, background_index=1.4, design_index=1.8, period_um=(.8, .8),
            height_um=.2, detector_offset_um=.5, theta_inside_rad=.1, phi_rad=.3)
SETTINGS = dict(mesh=.1, steps=160, pml_cells=6, quadrature_counts=(4, 4))
BUDGET = 1024**3


def policy(device, execution, tmp_path):
    if execution == 'resident':
        return AdjointExecutionPolicy(device=device, host_budget_bytes=BUDGET,
            resident=AdjointOptions(checkpoints=1, resident_budget_bytes=BUDGET,
                gpu_budget_bytes=BUDGET, host_budget_bytes=BUDGET,
                backward_kernel='fused' if device == 'cuda' else 'torch'))
    options = StreamedAdjointOptions(device=device, slab_width=4, temporal_depth=4,
        checkpoints=1, gpu_budget_bytes=BUDGET, host_budget_bytes=BUDGET,
        tile_transfers='async' if device == 'cuda' else 'sync')
    if execution == 'file':
        options = replace(options, state_storage='disk', state_directory=str(tmp_path), disk_budget_bytes=BUDGET)
    return AdjointExecutionPolicy(device=device, streamed=options, host_budget_bytes=BUDGET)


def model(device, execution, tmp_path, **kwargs):
    settings = dict(SETTINGS, density_shape=(2, 2), dtype=torch.float64,
        policy=policy(device, execution, tmp_path),
        batch_options=AdjointBatchOptions(host_budget_bytes=BUDGET, gpu_budget_bytes=BUDGET,
            disk_budget_bytes=BUDGET, replay_rtol=1e-9), reference_cache=PlaneReferenceCache(1024**2))
    settings.update(kwargs)
    return PeriodicLayerResponse(SPEC, **settings)


def objective(response):
    weights = response.new_tensor([[.1, .3, -.2, .7], [.4, -.1, .5, .2]])
    return (response*weights).sum() + .2*response.mean(0).square().sum()


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('execution', ['resident', 'dram', 'file'])
def test_coupled_polarization_response_and_density_gradient_match_legacy(device, execution, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    d = torch.tensor([[.2, .4], [.5, .3]], dtype=torch.float64, requires_grad=True)
    expected = periodic_layer_response(d, SPEC, **SETTINGS, options=AdjointOptions(checkpoints=1))
    expected_grad, = torch.autograd.grad(objective(expected), d)
    module = model(device, execution, tmp_path)
    plan = module.plan()
    assert plan['host_reservation_bytes'] > plan['batch']['host_reservation_bytes']
    actual = module(d)
    grad, = torch.autograd.grad(objective(actual), d)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(grad, expected_grad, rtol=2e-7, atol=1e-9)
    assert grad.norm() > 1e-4
    assert module.last_report['batch']['replayed_cases'] == 2
    assert not module.last_report['batch']['full_case_graph_retention']
    assert module.last_report['reference_cache_misses'] == 1
    with torch.no_grad():
        changed = module(d+.03)
    assert module.last_report['reference_cache_hits'] == 1
    assert not torch.allclose(changed, actual)
    if execution == 'file':
        assert not list(tmp_path.iterdir())


def test_density_direction_optimizer_and_spectral_schedule(tmp_path):
    module = model('cpu', 'resident', tmp_path)
    logits = torch.tensor([[-.9, -.2], [.5, .2]], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=.03)
    before = logits.detach().clone()
    response = spectral_pupil_response([[module]], logits.sigmoid(), [.7], replay_rtol=1e-9, replay_atol=0.)
    loss = response.square().sum()
    loss.backward()
    g = logits.grad.detach().clone()
    direction = logits.new_tensor([[.3, -.2], [.1, .4]])
    def score(v):
        return spectral_pupil_response([[module]], v.sigmoid(), [.7]).square().sum()
    with torch.no_grad():
        h = 1e-5
        fd = (score(logits+h*direction)-score(logits-h*direction))/(2*h)
    torch.testing.assert_close((g*direction).sum(), fd, rtol=2e-5, atol=1e-9)
    optimizer.step()
    assert not torch.equal(before, logits)
    assert module.last_report['plan']['geometry_allowance_bytes'] > 0


def test_denied_or_changed_budget_precedes_dense_geometry_and_reference(tmp_path, monkeypatch):
    import torchfdtd.periodic_adjoint as implementation
    with pytest.raises(ValueError, match='host budget'):
        model('cpu', 'resident', tmp_path,
            batch_options=AdjointBatchOptions(host_budget_bytes=1, gpu_budget_bytes=BUDGET))
    module = model('cpu', 'resident', tmp_path)
    monkeypatch.setattr(implementation, 'host_memory', lambda: dict(available_bytes=1))
    monkeypatch.setattr(implementation, 'periodic_density_layer', lambda *a, **kw: pytest.fail('Allocated denied geometry'))
    monkeypatch.setattr(module, '_references', lambda: pytest.fail('Executed denied references'))
    with pytest.raises(ValueError, match='host budget'):
        module(torch.full((2, 2), .3, dtype=torch.float64))
    monkeypatch.undo()
    module._cache.budget_bytes += 1
    with pytest.raises(ValueError, match='budget changed'):
        module.plan()


def test_plan_allocates_no_full_material_map_and_returns_an_inspection_copy(tmp_path, monkeypatch):
    import torchfdtd.periodic_adjoint as implementation
    spec = dict(SPEC)
    module = model('cpu', 'resident', tmp_path)
    monkeypatch.setattr(implementation, 'periodic_density_layer', lambda *a, **kw: pytest.fail('Allocated plan geometry'))
    monkeypatch.setattr(module, '_references', lambda: pytest.fail('Ran plan references'))
    report = module.plan()
    assert report['epsilon_shape'] == module.project.region.shape+(3,)
    copied = module.project
    copied.sources[0].wavelength = 9.
    assert module.project.sources[0].wavelength == spec['wavelength_um']
    with pytest.raises(ValueError, match='shape and dtype'):
        module(torch.zeros((3, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match='finite'):
        module(torch.full((2, 2), float('nan'), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cache_keys_separate_execution_policies(tmp_path):
    cache = PlaneReferenceCache(1024**2)
    a = model('cuda', 'resident', tmp_path, reference_cache=cache)
    b = model('cuda', 'dram', tmp_path, reference_cache=cache)
    with torch.no_grad():
        d = torch.full((2, 2), .3, dtype=torch.float64)
        x, y = a(d), b(d)
    assert cache.misses == 2
    torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fp32_resident_and_streamed_polarization_gradient(tmp_path):
    a = model('cuda', 'resident', tmp_path, dtype=torch.float32)
    b = model('cuda', 'dram', tmp_path, dtype=torch.float32)
    d = torch.tensor([[.2, .4], [.5, .3]], dtype=torch.float32, requires_grad=True)
    x, y = a(d), b(d)
    gx, = torch.autograd.grad(objective(x), d)
    gy, = torch.autograd.grad(objective(y), d)
    assert torch.isfinite(gx).all() and gx.norm() > 1e-4
    torch.testing.assert_close(x, y, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(gx, gy, rtol=1e-3, atol=1e-5)


def test_cr_runner_policy_contract_and_explicit_file_limits(tmp_path):
    from types import SimpleNamespace
    from benchmarks.cr_spectral_objective import execution_settings
    args = SimpleNamespace(execution_policy='legacy', gpu_budget_gib=2., host_budget_gib=4.,
        backward_kernel='auto', slab_width=8, temporal_depth=4,
        state_directory=None, disk_budget_gib=None)
    assert execution_settings(args) is None
    args.execution_policy = 'resident'
    result = execution_settings(args)
    assert result['policy'].resident.resident_budget_bytes == 2*1024**3
    args.execution_policy = 'dram'
    result = execution_settings(args)
    assert result['policy'].streamed.tile_transfers == 'async'
    args.backward_kernel = 'torch'
    with pytest.raises(ValueError, match='fused backward'):
        execution_settings(args)
    args.backward_kernel = 'auto'
    args.execution_policy = 'file'
    with pytest.raises(ValueError, match='explicit directory'):
        execution_settings(args)
    args.state_directory, args.disk_budget_gib = str(tmp_path), 8.
    result = execution_settings(args)
    assert result['policy'].streamed.disk_free_reserve_bytes == 100*1024**3
    assert result['batch_options'].disk_budget_bytes == 8*1024**3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_explicit_cr_runner_preserves_objective_and_full_gradient(tmp_path, monkeypatch):
    """Synthetic files exercise real runner parsing, preflight and nested VJPs."""
    import hashlib
    import json
    import numpy as np
    from benchmarks.cr_spectral_objective import main
    density = tmp_path/'density.npy'
    np.save(density, np.array([[0., 1.], [1., 0.]]))
    wavelengths = [450., 540., 650.]
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps(dict(density_sha256=hashlib.sha256(density.read_bytes()).hexdigest(),
        wavelengths_nm=wavelengths, ray_weights=[.7],
        cases=[[dict(SPEC, wavelength_um=w/1000)] for w in wavelengths])))
    eye = torch.eye(3, dtype=torch.float64)
    context = tmp_path/'context.pt'
    torch.save(dict(context=dict(wavelengths_nm=wavelengths, sampled_qe=[.5, .8, .6],
        source_spectrum=[1., 2., 1.], scene_spectral_basis=eye, electron_calibration=1e-10,
        scene_covariance=eye, scene_target_cross_covariance_xyz=.6*eye, target_covariance_xyz=eye,
        exposure_weighted_cfa_green_e=[20., 80., 320.], reference_weighted_cfa_green_e=100.,
        exposure_probabilities=[.2, .5, .3], read_noise_e_rms=1.5)), context)
    reference = None
    for mode in ('legacy', 'resident', 'dram'):
        output = tmp_path/f'{mode}.json'
        monkeypatch.setattr('sys.argv', ['cr_spectral_objective', '--schedule', str(schedule),
            '--density', str(density), '--context', str(context), '--output', str(output),
            '--mesh', '.1', '--steps', '160', '--pml-cells', '6', '--forward-kernel', 'fused',
            '--backward-kernel', 'fused', '--cpu-threads', '1', '--reference-cache-mib', '1', '--precision', 'float64',
            '--execution-policy', mode, '--gpu-budget-gib', '1', '--host-budget-gib', '1',
            '--slab-width', '4', '--temporal-depth', '4'])
        main()
        record = json.loads(output.read_text())
        gradient = torch.from_numpy(np.load(output.with_suffix('.gradient.npy')))
        assert record['gradient_l2'] > 0 and record['restart_session']['computed_cases'] == 3
        assert record['restart_contract']['cpu_threads'] == 1
        values = (torch.tensor(record['response']), torch.tensor(record['weighted_bits_per_pixel']), gradient)
        if reference is None:
            reference = values
        else:
            for actual, expected in zip(values, reference):
                torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-10)
            assert record['execution_preflight']['cases'] == 3
            assert json.loads(output.with_suffix('.plan.json').read_text())['stage'] == 'admitted_not_executed'
            assert 'execution' in record['restart_contract']


def test_default_fp32_model_accepts_fp32_density_and_backpropagates(tmp_path):
    module = PeriodicLayerResponse(SPEC, **SETTINGS, density_shape=(2, 2),
        policy=policy('cpu', 'resident', tmp_path),
        batch_options=AdjointBatchOptions(host_budget_bytes=BUDGET, gpu_budget_bytes=BUDGET))
    density = torch.tensor([[.2, .4], [.5, .3]], dtype=torch.float32, requires_grad=True)
    response = module(density)
    gradient, = torch.autograd.grad(objective(response), density)
    assert response.dtype == gradient.dtype == torch.float32
    assert bool(torch.isfinite(gradient).all()) and gradient.norm() > 1e-4
