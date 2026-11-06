"""Metadata-only reversible accounting and final admission, no GPU tensors."""
from types import SimpleNamespace

import pytest
import torch

from torchfdtd.reversible_memory import _reversible_reservation


def fixture(monkeypatch, shape=(4, 3, 2), sampled=False):
    import torchfdtd.adjoint_memory as base_module
    import torchfdtd.memory_profile as host_module
    import torchfdtd.cuda_memory as cuda_module
    source = SimpleNamespace(enabled=True, polarization_components=(('Ex', 1.), ('Ey', .2)),
        pulse='sampled' if sampled else 'gaussian', signal=SimpleNamespace(time_s=[0., 1., 2.]))
    project = SimpleNamespace(region=SimpleNamespace(shape=shape, steps=10), sources=[source],
                              resolved_source=lambda value: value)
    calls = []
    def base(project, options, device):
        assert options.checkpoints == 1 and options.storage == 'device'
        assert options.backward_kernel == ('fused' if device.type == 'cuda' else 'torch')
        assert options.gpu_budget_bytes is options.host_budget_bytes is None
        calls.append(device.type)
        return dict(memory_reservation_bytes=1000, host_reservation_bytes=1000 if device.type == 'cpu' else 100,
            workspace_reservation_bytes=400, workspace_components_bytes={'base': 400},
            workspace_model='test', device_checkpoint_reservation_bytes=576,
            output_history_bytes=80, source_history_bytes=80, history_reservation_bytes=240,
            host_checkpoint_reservation_bytes=0, disk_checkpoint_reservation_bytes=0,
            device_staging_reservation_bytes=0)
    monkeypatch.setattr(base_module, '_resident_reservation', base)
    monkeypatch.setattr(host_module, 'host_memory', lambda: {'available_bytes': 10**12})
    capacity = []
    def limit(device, required, budget):
        capacity.append((str(device), required, budget))
        return min(10**12, budget) if budget is not None else 10**12
    monkeypatch.setattr(cuda_module, 'cuda_budget_limit', limit)
    return project, calls, capacity


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_terminal_reclassification_complete_totals_and_metadata_only(monkeypatch, device):
    p, calls, capacity = fixture(monkeypatch, sampled=True)
    # Neither planner nor source-preparation accounting may construct tensors.
    monkeypatch.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('tensor allocation'))
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('tensor allocation'))
    result = _reversible_reservation(p, SimpleNamespace(), device)
    diagnostic = 2*72*8+4096
    preparation = 10*(128+32*2)+64*3
    assert result['diagnostic_reservation_bytes'] == diagnostic
    assert result['source_preparation_host_bytes'] == preparation
    assert result['terminal_state_bytes'] == 576
    assert result['checkpoint_states'] == result['device_checkpoint_reservation_bytes'] == 0
    assert result['host_checkpoint_reservation_bytes'] == result['disk_checkpoint_reservation_bytes'] == 0
    assert result['contiguous_seed_reservation_bytes'] == 80
    assert result['inverse_extra_coefficient_tensor_bytes'] == 0
    assert result['retained_caller_epsilon_bytes'] == 96
    assert calls == [device]
    if device == 'cpu':
        assert result['memory_reservation_bytes'] == result['host_reservation_bytes'] == 1000+diagnostic+preparation
        assert result['gpu_reservation_bytes'] == 0 and not capacity
    else:
        assert result['memory_reservation_bytes'] == result['gpu_reservation_bytes'] == 1000+diagnostic
        assert result['host_reservation_bytes'] == 100+preparation
        assert capacity == [('cuda', 1000+diagnostic, None)]


@pytest.mark.parametrize('device,budget', [('cpu', 'host_budget_bytes'), ('cpu', 'resident_budget_bytes'),
                                         ('cuda', 'gpu_budget_bytes'), ('cuda', 'resident_budget_bytes'),
                                         ('cuda', 'host_budget_bytes')])
def test_final_extra_costs_cannot_bypass_explicit_budget(monkeypatch, device, budget):
    p, _, _ = fixture(monkeypatch)
    # The mocked base fits 1000, but diagnostics/preparation do not.
    with pytest.raises(ValueError, match='budget'):
        _reversible_reservation(p, SimpleNamespace(**{budget: 1000}), device)


def test_chunk_is_bounded_and_live_host_capacity_includes_extras(monkeypatch):
    p, _, _ = fixture(monkeypatch, shape=(1000, 1000, 1000))
    result = _reversible_reservation(p, SimpleNamespace(), 'cpu')
    assert result['diagnostic_chunk_elements'] == 65536
    assert result['diagnostic_buffer_bytes'] == 2*65536*8
    import torchfdtd.memory_profile as host_module
    monkeypatch.setattr(host_module, 'host_memory', lambda: {'available_bytes': 2000})
    with pytest.raises(ValueError, match='available host memory'):
        _reversible_reservation(p, SimpleNamespace(), 'cpu')


def test_no_source_preparation_without_sources_and_invalid_budget(monkeypatch):
    p, _, _ = fixture(monkeypatch)
    p.sources = []
    assert _reversible_reservation(p, SimpleNamespace(), 'cpu')['source_preparation_host_bytes'] == 0
    with pytest.raises(ValueError, match='positive integer'):
        _reversible_reservation(p, SimpleNamespace(host_budget_bytes=True), 'cpu')


def test_real_budgeted_project_preserves_explicit_resident_admission():
    from torchfdtd.models import Monitor, Project, Region
    project = Project(region=Region(dimension='3d', size=(.6, .6, .6), mesh=.1,
        steps=10, backend='cpu', memory_mode='budgeted', material_sampling='yee',
        boundaries={a+'_'+side: {'kind': 'periodic'}
                    for a in 'xyz' for side in ('min', 'max')}),
        monitors=[Monitor(component='Ex', center=(0, 0, 0))])
    budget = 64*1024**2
    result = _reversible_reservation(project,
        SimpleNamespace(resident_budget_bytes=budget), 'cpu')
    assert result['budgeted_resident']
    assert 0 < result['memory_reservation_bytes'] < budget
    # Admission still includes extras even when the base subtotal fits.
    base_only = result['memory_reservation_bytes'] - result['diagnostic_reservation_bytes'] - result['source_preparation_host_bytes']
    with pytest.raises(ValueError, match='total resident reservation'):
        _reversible_reservation(project,
            SimpleNamespace(resident_budget_bytes=base_only), 'cpu')
