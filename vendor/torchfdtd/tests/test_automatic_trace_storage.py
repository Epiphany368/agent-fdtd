"""Complete admission and unchanged derivatives for automatic boundary history."""
import gc
import weakref

import pytest
import torch

from torchfdtd import ReversibleCPMLOptions, ReversibleCPMLSimulation
from torchfdtd.reversible_cpml_memory import _cpml_reversible_reservation
from test_reversible_cpml import fixture
from test_reversible_cpml_memory import mock_base, options


def auto_options(**kwargs):
    return options('auto', host_budget_bytes=10**9, **kwargs)


def test_auto_metadata_device_first_and_exact_complete_budget(monkeypatch):
    project, _ = mock_base(monkeypatch)
    project.region.steps = 200
    direct = _cpml_reversible_reservation(project, options(), 'cuda', (2, 5))
    host = _cpml_reversible_reservation(project,
        options('cpu', trace_transfers='async'), 'cuda', (2, 5))
    assert host['memory_reservation_bytes'] < direct['memory_reservation_bytes']
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('field allocation during plan'))
    monkeypatch.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('tensor allocation during plan'))
    exact = _cpml_reversible_reservation(project,
        auto_options(gpu_budget_bytes=direct['memory_reservation_bytes']), 'cuda', (2, 5))
    assert exact['trace_storage_requested'] == 'auto'
    assert exact['trace_storage'] == 'device' and exact['trace_transfers'] == 'sync'
    assert exact['trace_placement_attempts'] == [dict(storage='device', transfers='sync', admitted=True)]
    fallback = _cpml_reversible_reservation(project,
        auto_options(gpu_budget_bytes=direct['memory_reservation_bytes']-1), 'cuda', (2, 5))
    assert fallback['trace_storage'] == 'cpu' and fallback['trace_transfers'] == 'async'
    assert fallback['trace_device_bytes'] == 0
    assert fallback['trace_transfer_device_bytes'] == 2*32*fallback['trace_frame_bytes']
    assert fallback['trace_host_bytes'] == fallback['trace_bytes']
    assert fallback['trace_pinned_host_bytes'] == fallback['trace_transfer_host_bytes']
    assert fallback['trace_placement_attempts'][0]['admitted'] is False
    assert fallback['trace_placement_attempts'][1]['admitted'] is True
    for name in ('memory_reservation_bytes', 'host_reservation_bytes', 'workspace_reservation_bytes'):
        assert fallback[name] == host[name]


@pytest.mark.parametrize('complex_fields,components', [(False, 1), (True, 3)])
def test_auto_requeries_live_capacity_and_counts_complex_trace(monkeypatch, complex_fields, components):
    import torchfdtd.cuda_memory as cuda
    project, _ = mock_base(monkeypatch)
    project.region.steps = 200
    project.region.complex_fields = complex_fields
    project.region.cuda_kernel = 'fused'
    def reserve(opts):
        return _cpml_reversible_reservation(project, opts, 'cuda', (2, 5), material_components=components)
    direct = reserve(options())
    host = reserve(options('cpu', trace_transfers='async'))
    capacity = [direct['memory_reservation_bytes']]
    monkeypatch.setattr(cuda, 'cuda_budget_limit', lambda device, required, budget: capacity[0])
    opts = auto_options()
    first = reserve(opts)
    capacity[0] = host['memory_reservation_bytes']
    second = reserve(opts)
    assert first['trace_storage'] == 'device' and second['trace_storage'] == 'cpu'
    assert opts.trace_storage == 'auto'  # No remembered device decision on the model/options.
    assert second['trace_item_bytes'] == (8 if complex_fields else 4)
    assert second['material_components'] == components
    capacity[0] -= 1
    with pytest.raises(ValueError, match='cannot admit.*CUDA budget'):
        reserve(opts)


def test_auto_host_budget_and_live_host_memory_are_not_bypassed(monkeypatch):
    import torchfdtd.memory_profile as memory
    project, _ = mock_base(monkeypatch)
    project.region.steps = 200
    host = _cpml_reversible_reservation(project,
        options('cpu', trace_transfers='async'), 'cuda', (2, 5))
    def reserve(host_budget):
        return _cpml_reversible_reservation(project,
            options('auto', host_budget_bytes=host_budget,
                gpu_budget_bytes=host['memory_reservation_bytes']), 'cuda', (2, 5))
    assert reserve(host['host_reservation_bytes'])['trace_storage'] == 'cpu'
    with pytest.raises(ValueError, match='cannot admit.*host budget'):
        reserve(host['host_reservation_bytes']-1)
    monkeypatch.setattr(memory, 'host_memory', lambda: dict(available_bytes=host['host_reservation_bytes']))
    with pytest.raises(ValueError, match='cannot admit.*available host memory'):
        reserve(10**9)


def test_auto_resident_budget_and_invalid_metadata_fail_before_allocation(monkeypatch):
    project, _ = mock_base(monkeypatch)
    project.region.steps = 200
    host = _cpml_reversible_reservation(project,
        options('cpu', trace_transfers='async'), 'cuda', (2, 5))
    with pytest.raises(ValueError, match='cannot admit.*resident budget'):
        _cpml_reversible_reservation(project,
            auto_options(resident_budget_bytes=host['memory_reservation_bytes']-1), 'cuda', (2, 5))
    for kwargs, match in [({'material_components': 2}, 'material_components'),
                          ({'interval': (-1, 5)}, 'interval')]:
        interval = kwargs.pop('interval', (2, 5))
        with pytest.raises(ValueError, match=match):
            _cpml_reversible_reservation(project, auto_options(), 'cuda', interval, **kwargs)
    with pytest.raises(ValueError, match='trace_transfers'):
        _cpml_reversible_reservation(project, auto_options(trace_transfers='invalid'), 'cuda', (2, 5))


def test_auto_requires_explicit_host_budget_and_keeps_existing_default():
    assert ReversibleCPMLOptions().trace_storage == 'device'
    with pytest.raises(ValueError, match='explicit host_budget_bytes'):
        ReversibleCPMLOptions(trace_storage='auto')
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            ReversibleCPMLOptions(trace_storage='auto', host_budget_bytes=invalid)
    assert ReversibleCPMLOptions(trace_storage='auto', host_budget_bytes=2**30,
                               trace_transfers='async').trace_storage == 'auto'


def test_cpu_auto_matches_static_signals_and_retained_vjps():
    torch.set_num_threads(1)
    project, base, fixed = fixture(48)
    automatic = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        trace_storage='auto', host_budget_bytes=64*1024**2))
    plan = automatic.plan()
    assert plan['trace_storage'] == 'cpu' and plan['trace_transfers'] == 'sync'
    parameter = base.clone().requires_grad_()
    baseline_parameter = base.clone().requires_grad_()
    result = automatic(parameter, fixed_epsilon=fixed)
    baseline = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(trace_storage='cpu'))(
        baseline_parameter, fixed_epsilon=fixed)
    assert torch.equal(result.signals, baseline.signals)
    generator = torch.Generator().manual_seed(722)
    for _ in range(2):
        seed = torch.randn((4, 48), generator=generator).T
        actual, = torch.autograd.grad(result.signals, parameter, seed, retain_graph=True)
        expected, = torch.autograd.grad(baseline.signals, baseline_parameter, seed, retain_graph=True)
        assert torch.equal(actual, expected)
    with torch.no_grad():
        inference = automatic(parameter, fixed_epsilon=fixed)
    assert torch.equal(inference.signals, result.signals)
    assert inference.report['forward_only'] and inference.report['terminal_copies'] == 0


@pytest.mark.cuda
@pytest.mark.parametrize('complex_diagonal', [False, True])
@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_cuda_auto_archive_vjps_stream_contract_and_release(complex_diagonal, storage):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    from test_reversible_cpml_async_cuda import cuda_fixture
    project, base, fixed = cuda_fixture(complex_diagonal)
    host_options = ReversibleCPMLOptions(trace_storage='cpu', trace_transfers='async', trace_chunk_steps=7)
    host_plan = ReversibleCPMLSimulation(project, host_options).plan(
        device=base.device, material_components=3 if complex_diagonal else 1)
    opts = ReversibleCPMLOptions(trace_storage='auto', host_budget_bytes=128*1024**2,
        gpu_budget_bytes=host_plan['gpu_reservation_bytes'] if storage == 'cpu' else None,
        trace_chunk_steps=7)
    automatic = ReversibleCPMLSimulation(project, opts)
    parameter = base.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = automatic(parameter, fixed_epsilon=fixed)
    assert result.report['trace_storage_requested'] == 'auto'
    assert result.report['trace_storage'] == storage
    context = result.signals.grad_fn
    saved_archive = context.saved_tensors[1].clone()
    refs = [weakref.ref(context.system), *[weakref.ref(t) for t in context.saved_tensors]]
    if storage == 'cpu':
        assert context.transport.state == 'complete'
        assert context.transport.archive.device.type == 'cpu'
        refs.append(weakref.ref(context.transport))
        refs.extend(weakref.ref(t) for slot in context.transport.slots for t in (slot.cpu, slot.gpu))
    else:
        assert context.transport is None
        assert context.saved_tensors[1].device.type == 'cuda'
    del context
    dtype = torch.complex64 if complex_diagonal else torch.float32
    generator = torch.Generator().manual_seed(419)
    seeds = [torch.randn((4, 48), dtype=dtype, generator=generator).cuda().T for _ in range(2)]
    gradients = [torch.autograd.grad(result.signals, parameter, s, retain_graph=True)[0] for s in seeds]
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated()-before <= result.report['gpu_reservation_bytes']
    baseline_parameter = base.clone().requires_grad_()
    baseline = ReversibleCPMLSimulation(project, ReversibleCPMLOptions())(
        baseline_parameter, fixed_epsilon=fixed)
    assert torch.equal(result.signals, baseline.signals)
    assert torch.equal(saved_archive.to(base.device), baseline.signals.grad_fn.saved_tensors[1])
    for seed, actual in zip(seeds, gradients):
        expected, = torch.autograd.grad(baseline.signals, baseline_parameter, seed, retain_graph=True)
        assert torch.equal(actual, expected)
    if storage == 'cpu':
        transport = result.signals.grad_fn.transport
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match='captured compute stream'):
                transport.reverse()
            # Autograd runs this node on its recorded forward stream, even
            # when the caller requests backward from a different stream.
            other_stream, = torch.autograd.grad(result.signals, parameter, seeds[0], retain_graph=True)
        torch.cuda.synchronize()
        assert torch.equal(other_stream, gradients[0])
        recovered, = torch.autograd.grad(result.signals, parameter, seeds[0], retain_graph=True)
        assert torch.equal(recovered, gradients[0])
        del transport
    enabled = gc.isenabled()
    gc.disable()
    try:
        del result
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.cuda
def test_cuda_auto_allocation_error_does_not_retry_or_replay(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    from test_reversible_cpml_async_cuda import cuda_fixture
    import torchfdtd.reversible_trace as transport
    project, base, fixed = cuda_fixture(False)
    host = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        trace_storage='cpu', trace_transfers='async', trace_chunk_steps=7)).plan(device=base.device)
    automatic = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        trace_storage='auto', host_budget_bytes=128*1024**2,
        gpu_budget_bytes=host['gpu_reservation_bytes'], trace_chunk_steps=7))
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(args)
        raise torch.cuda.OutOfMemoryError('injected transport allocation failure')
    monkeypatch.setattr(transport, 'AsyncBoundaryTrace', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='injected transport allocation'):
        automatic(base.clone().requires_grad_(), fixed_epsilon=fixed)
    assert len(attempts) == 1
