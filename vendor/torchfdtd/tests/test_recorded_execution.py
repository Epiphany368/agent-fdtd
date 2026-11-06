"""CPU graph bridge and complete admission, without CUDA execution."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from torchfdtd import ReversibleCPMLOptions
from torchfdtd.recorded_execution import _RecordedPlanesFromHost, _recorded_reservation
from test_reversible_cpml_planes import fixture


def policy(device='cpu', **options):
    return SimpleNamespace(recorded=ReversibleCPMLOptions(**options), resident=None,
                           streamed=None, device=device, host_budget_bytes=256*1024**2)


def prepared(complex_fields=False):
    project, epsilon, background = fixture(complex_fields)
    counts = {'incident': (2, 2), 'detector': (2, 2)}
    settings = policy()
    adapter = _RecordedPlanesFromHost(project, settings, quadrature_counts=counts,
                                     fixed_background_epsilon=1.3)
    frequencies = [.04/project.region.time_step]
    return adapter, epsilon, background, frequencies


def test_cpu_diagonal_bloch_bridge_fields_and_material_gradient():
    torch.set_num_threads(1)
    adapter, epsilon, background, frequencies = prepared(True)
    parameter = epsilon.clone().requires_grad_()
    reference_parameter = epsilon.clone().requires_grad_()
    actual = adapter(parameter, frequency_hz=frequencies, block_size=7)
    reference = adapter.model(reference_parameter, frequencies, fixed_epsilon=background, block_size=7)
    def fields(planes):
        return torch.stack([p.fields for p in planes.values()])/adapter.project.region.time_step
    a, b = fields(actual), fields(reference)
    assert a.device.type == 'cpu' and torch.equal(a, b)
    grad, = torch.autograd.grad((a.real+.3*a.imag).square().sum(), parameter)
    expected, = torch.autograd.grad((b.real+.3*b.imag).square().sum(), reference_parameter)
    assert torch.equal(grad, expected) and torch.count_nonzero(grad) > 0
    lo, hi = adapter.model.interior_z
    assert torch.count_nonzero(grad[:, :, :lo]) == torch.count_nonzero(grad[:, :, hi+1:]) == 0
    report = next(iter(actual.values())).report
    reservation = report['execution_reservation']
    assert report['unified_execution'] == 'recorded_cpml'
    assert reservation['generated_background_host_bytes'] == epsilon.numel()*4
    assert reservation['host_transfer_reservation_bytes'] == 0
    assert not reservation['generated_background_cached']
    assert not any(isinstance(v, torch.Tensor) for v in vars(adapter).values())


def test_background_and_plane_contract_before_construction(monkeypatch):
    import torchfdtd.recorded_execution as module
    project, _, _ = fixture()
    monkeypatch.setattr(module, '_make_plane', lambda *args: pytest.fail('layout constructed before validation'))
    for background in (None, 1, True, float('inf'), .9, torch.tensor(1.3)):
        with pytest.raises(ValueError, match='background epsilon'):
            _RecordedPlanesFromHost(project, policy(), fixed_background_epsilon=background)
    with pytest.raises(ValueError, match='nondispersive'):
        _RecordedPlanesFromHost(project, policy(), dispersive=True, fixed_background_epsilon=1.3)
    project.region.precision = 'float64'
    with pytest.raises(ValueError, match='FP32'):
        _RecordedPlanesFromHost(project, policy(), fixed_background_epsilon=1.3)


def test_frozen_configuration_and_input_contract():
    adapter, epsilon, _, frequencies = prepared()
    adapter.fixed_background_epsilon = 1.4
    with pytest.raises(ValueError, match='configuration changed'):
        adapter.check_fixed()
    adapter.fixed_background_epsilon = 1.3
    adapter.check_fixed()
    for value in (epsilon.double(), epsilon.transpose(0, 1), epsilon[..., None]):
        with pytest.raises(ValueError, match='contiguous real CPU FP32'):
            adapter(value, frequency_hz=frequencies)
    adapter.project.sources[0].amplitude += 1
    with pytest.raises(ValueError, match='configuration changed'):
        adapter(epsilon, frequency_hz=frequencies)


def test_layout_budget_rejected_before_plane_construction(monkeypatch):
    import torchfdtd.recorded_execution as module
    project, _, _ = fixture()
    settings = policy()
    settings.host_budget_bytes = 1
    monkeypatch.setattr(module, 'ReversibleCPMLPlaneSimulation', lambda *a, **k: pytest.fail('layout allocation'))
    with pytest.raises(ValueError, match='before allocation'):
        _RecordedPlanesFromHost(project, settings, fixed_background_epsilon=1.3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bridge_additions_and_complete_budgets(monkeypatch, device):
    adapter, epsilon, _, frequencies = prepared()
    model = adapter.model
    settings = policy(device)
    points = sum(len(plan['weights']) for _, _, plan, _ in model.plans)
    output = 6*8*points
    base = dict(host_reservation_bytes=1000, gpu_reservation_bytes=2000 if device == 'cuda' else 0,
                memory_reservation_bytes=2000 if device == 'cuda' else 1000,
                allocation_headroom_bytes=40, plane_output_bytes=output)
    monkeypatch.setattr(model, 'plan', lambda *a, **k: dict(base))
    import torchfdtd.memory_profile as host
    import torchfdtd.cuda_memory as cuda
    monkeypatch.setattr(host, 'host_memory', lambda: {'available_bytes': 10**12})
    monkeypatch.setattr(cuda, 'cuda_budget_limit', lambda d, n, b: b or 10**12)
    def reserve():
        return _recorded_reservation(adapter.project, (tuple(epsilon.shape),), settings,
            frequencies, plane=model, fixed_background_epsilon=1.3)
    result = reserve()
    payload = epsilon.numel()*4
    metadata = (4*points+len(model.plans))*4
    if device == 'cuda':
        assert result['gpu_reservation_bytes'] == 2000+3*payload+(3*payload+19)//20+4096
        assert result['host_reservation_bytes'] == 1000+2*payload+2*output+metadata
    else:
        assert result['host_reservation_bytes'] == result['memory_reservation_bytes'] == 1000+payload
    for budget in ('host', 'option_host', 'resident', 'gpu'):
        if budget == 'gpu' and device == 'cpu':
            continue
        settings.host_budget_bytes = 256*1024**2
        settings.recorded = policy().recorded
        if budget == 'host':
            settings.host_budget_bytes = result['host_reservation_bytes']-1
        else:
            key = {'option_host': 'host_budget_bytes', 'resident': 'resident_budget_bytes', 'gpu': 'gpu_budget_bytes'}[budget]
            total = result['host_reservation_bytes'] if budget == 'option_host' else result['memory_reservation_bytes']
            settings.recorded = replace(settings.recorded, **{key: total-1})
        model.model.options = settings.recorded
        model.project.region.memory_mode = 'budgeted' if settings.recorded.resident_budget_bytes is not None else 'resident'
        with pytest.raises(ValueError, match='budget'):
            reserve()


def test_admission_precedes_background_and_device_copy(monkeypatch):
    import torchfdtd.recorded_execution as module
    adapter, epsilon, _, frequencies = prepared()
    def reject(*args, **kwargs):
        raise ValueError('admission rejected')
    monkeypatch.setattr(module, '_recorded_reservation', reject)
    monkeypatch.setattr(torch, 'full_like', lambda *a, **k: pytest.fail('background allocated'))
    monkeypatch.setattr(torch.Tensor, 'to', lambda *a, **k: pytest.fail('input copied'))
    with pytest.raises(ValueError, match='admission rejected'):
        adapter(epsilon, frequency_hz=frequencies)
