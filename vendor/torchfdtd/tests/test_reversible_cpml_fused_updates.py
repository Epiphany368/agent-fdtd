"""Independent split-path comparisons for optional recorded CPML launches."""
import copy
import gc
import weakref

import pytest
import torch

from torchfdtd import ReversibleCPMLOptions, ReversibleCPMLSimulation
from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
from torchfdtd.reversible_cpml import _advance_recorded, _recorded_system
from torchfdtd.recorded_observations import recorder
from test_reversible_cpml import fixture, effective, relative


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    prepare_cuda_kernels()


def case(diagonal=False, steps=33):
    project, base, fixed = fixture(steps)
    project.region.backend = 'cuda'
    if diagonal:
        base = torch.stack((base, base+0.11, base+0.27), -1)
        fixed = torch.stack((fixed, fixed+0.11, fixed+0.27), -1)
    return project, base.to('cuda'), fixed.to('cuda')


@pytest.mark.parametrize('kwargs', [dict(block_size=True), dict(block_size=128.0),
    dict(block_size=64), dict(block_size='fast'), dict(forward_kernel='fast'),
    dict(adjoint_kernel='fused')])
def test_option_validation(kwargs):
    with pytest.raises(ValueError):
        ReversibleCPMLOptions(**kwargs)


def test_cpu_fallback_preserves_results():
    project, base, fixed = fixture(20)
    results = []
    for options in (ReversibleCPMLOptions(), ReversibleCPMLOptions(
            forward_kernel='fused_eh', adjoint_kernel='one_pass', block_size='auto')):
        x = base.clone().requires_grad_()
        result = ReversibleCPMLSimulation(project, options)(x, fixed_epsilon=fixed)
        grad, = torch.autograd.grad(result.signals.square().sum(), x)
        results.append((result, grad))
    assert torch.equal(results[0][0].signals, results[1][0].signals)
    assert torch.equal(results[0][1], results[1][1])
    report = results[1][0].report
    assert report['forward_kernel_used'] == report['adjoint_kernel_used'] == 'split'
    assert set(report['fallback_reason']) == {'forward', 'adjoint', 'block_size'}
    assert all('CUDA' in reason for reason in report['fallback_reason'].values())
    assert report['cuda_tuning_scratch_bytes'] == 0


def test_tile_benchmark_checks_budget_before_material_allocation(monkeypatch):
    from benchmarks.reversible_cpml_fused_updates import build_tile
    original = torch.ones

    def reject_volume(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) >= 3:
            raise AssertionError('Material allocation occurred before admission')
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(torch, 'ones', reject_volume)
    with pytest.raises(ValueError, match='budget'):
        build_tile(core=.6, over=.4, steps=10, trace='cpu', device='cpu',
            gpu_gib=1e-9, launch_modes=['split', 'all'])


@pytest.mark.cuda
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('block', [128, 256, 512, 1024, 'auto'])
def test_split_launch_sizes_preserve_every_field_and_psi(diagonal, block):
    require_cuda()
    project, base, _ = case(diagonal)
    systems, reports = [], []
    for requested in (None, block):
        report = {}
        systems.append(_recorded_system(base, project, None,
            ReversibleCPMLOptions(block_size=requested), report))
        reports.append(report)
    for n in range(project.region.steps):
        for system in systems:
            _advance_recorded(system, n, None, 4, 14)
        for left, right in zip(systems[0].state(), systems[1].state()):
            assert torch.equal(left, right)
    assert set(reports[1]['cuda_launches']) == {'yee_E', 'yee_H'}


@pytest.mark.cuda
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('variant', ['marching', 'gather'])
def test_fused_forward_trace_fields_psi_and_recorder(diagonal, variant):
    require_cuda()
    from torchfdtd.reversible_cuda_fused import FusedEHMarching, FusedEHGather
    project, base, _ = case(diagonal)
    baseline = _recorded_system(base, project, None)
    candidate = _recorded_system(base, project, None)
    candidate.fused_eh = (FusedEHMarching(candidate) if variant == 'marching' else FusedEHGather(candidate))
    samples = [torch.empty((project.region.steps, len(project.monitors)), device='cuda') for _ in range(2)]
    recorders = [recorder(system, data) for system, data in zip((baseline, candidate), samples)]
    for n in range(project.region.steps):
        frames = [torch.empty((2, *project.region.shape[:2], 2), device='cuda') for _ in range(2)]
        for system, record, frame in zip((baseline, candidate), recorders, frames):
            _advance_recorded(system, n, frame, 4, 14)
            record(n)
        assert torch.equal(frames[0], frames[1])
        assert torch.equal(samples[0][n], samples[1][n])
        for left, right in zip(baseline.state(), candidate.state()):
            assert torch.equal(left, right)
    assert torch.count_nonzero(samples[0]) > 0


@pytest.mark.cuda
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('block', [128, 256, 512, 1024, 'auto'])
@pytest.mark.parametrize('forward,adjoint', [('fused_eh', 'split'), ('split', 'one_pass'), ('fused_eh', 'one_pass')])
def test_public_signals_gradient_drift_and_retained_backward(diagonal, block, forward, adjoint):
    require_cuda()
    project, base, fixed = case(diagonal)
    reference = None
    for options in (ReversibleCPMLOptions(), ReversibleCPMLOptions(
            block_size=block, forward_kernel=forward, adjoint_kernel=adjoint)):
        x = base.clone().requires_grad_()
        model = ReversibleCPMLSimulation(project, options)
        result = model(x, fixed_epsilon=fixed)
        loss = result.signals.square().sum()
        grad, = torch.autograd.grad(loss, x, retain_graph=True)
        report = result.report
        drift = {name: report['last_backward'][name] for name in (
            'initial_max_abs', 'initial_l2', 'initial_relative_peak', 'initial_relative_l2')}
        if reference is None:
            reference = (result.signals.detach().clone(), grad.clone(), drift)
        else:
            assert torch.equal(reference[0], result.signals)
            assert torch.equal(reference[1], grad)
            assert reference[2] == drift
            again, = torch.autograd.grad(loss, x)
            assert torch.equal(again, grad)
            assert report['forward_kernel_used'] == forward
            assert report['adjoint_kernel_used'] == adjoint
            assert not report['fallback_reason']
        with torch.no_grad():
            no_grad = model(base, fixed_epsilon=fixed)
        assert torch.equal(reference[0], no_grad.signals)


@pytest.mark.cuda
def test_autotuning_does_not_keep_states_alive_or_mutate_input():
    require_cuda()
    from torchfdtd.reversible_cuda_tuning import _CACHE
    project, base, _ = case()
    before = base.clone()
    refs = []
    reports = []
    for _ in range(2):
        report = {}
        system = _recorded_system(base, project, None, ReversibleCPMLOptions(block_size='auto'), report)
        assert torch.count_nonzero(system.grid.E) == 0
        assert torch.count_nonzero(system.grid.H) == 0
        refs.extend([weakref.ref(system), weakref.ref(system.grid.E)])
        reports.append(copy.deepcopy(report))
        del system
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert torch.equal(base, before)
    assert all(entry['cached'] for entry in reports[1]['cuda_launches'].values())
    assert _CACHE


@pytest.mark.cuda
@pytest.mark.parametrize('complex_fields', [False, True])
@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_plane_spectra_interval_sources_and_fallback(complex_fields, storage):
    require_cuda()
    from torchfdtd import ReversibleCPMLPlaneSimulation, DifferentiablePlaneSimulation, AdjointOptions
    from test_reversible_cpml_planes import fixture as plane_fixture
    from test_reversible_cpml_interval import material_maps
    project, _, _ = plane_fixture(complex_fields, 'Ey')
    project.region.backend = 'cuda'
    base, fixed = material_maps(project, True, 'cuda')
    frequencies = torch.tensor([.033, .057], device='cuda') / project.region.time_step
    counts = {'incident': (3, 4), 'detector': (3, 4)}
    common = dict(interior_z=(8, 11), trace_storage=storage,
                  trace_transfers='async' if storage == 'cpu' else 'sync', trace_chunk_steps=7)
    results = []
    for mode in ('split', 'all', 'oracle'):
        options = ReversibleCPMLOptions(**common, **(dict(forward_kernel='fused_eh',
            adjoint_kernel='one_pass', block_size='auto') if mode == 'all' else {}))
        model = (DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=3), quadrature_counts=counts)
                 if mode == 'oracle' else ReversibleCPMLPlaneSimulation(project, options, quadrature_counts=counts))
        x = base.clone().requires_grad_()
        planes = model(x, frequencies, block_size=7, **({} if mode == 'oracle' else dict(fixed_epsilon=fixed)))
        fields = torch.stack([value.fields for value in planes.values()])/project.region.time_step
        grad, = torch.autograd.grad((fields.real+.37*fields.imag).square().mean(), x)
        results.append((fields.detach(), grad))
        if mode == 'all':
            report = next(iter(planes.values())).report
            assert report['forward_kernel_used'] == ('split' if complex_fields else 'fused_eh')
            assert report['adjoint_kernel_used'] == ('split' if complex_fields else 'one_pass')
            assert bool(report['fallback_reason']) is complex_fields
        if mode != 'oracle':
            with torch.no_grad():
                again = model(base, frequencies, fixed_epsilon=fixed, block_size=7)
            assert all(torch.equal(planes[key].fields, again[key].fields) for key in planes)
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    # The checkpoint oracle differentiates exterior material too. Compare only
    # the declared design interval, including monitors on both sides of it.
    assert relative(results[1][1][:, :, 8:12], results[2][1][:, :, 8:12]) < 3e-5


@pytest.mark.cuda
def test_fused_memory_admission_peak_and_lifetime():
    require_cuda()
    project, base, fixed = case(True, 64)
    options = ReversibleCPMLOptions(block_size='auto', forward_kernel='fused_eh', adjoint_kernel='one_pass')
    model = ReversibleCPMLSimulation(project, options)
    x = base.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = model(x, fixed_epsilon=fixed)
    system = result.signals.grad_fn.system
    refs = [weakref.ref(system), weakref.ref(system.fused_eh),
            *[weakref.ref(t) for t in system.fused_eh.E+system.fused_eh.H]]
    del system
    torch.autograd.grad(result.signals.square().mean(), x, retain_graph=True)
    torch.cuda.synchronize()
    report = result.report
    assert torch.cuda.max_memory_allocated()-before <= report['gpu_reservation_bytes']
    cells = base.numel()//3
    assert report['fused_forward_buffer_bytes'] > 24*cells
    assert report['fused_adjoint_buffer_bytes'] == 24*cells
    assert report['cuda_tuning_scratch_bytes'] > 24*cells
    enabled = gc.isenabled()
    gc.disable()
    try:
        del result
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
    small = ReversibleCPMLSimulation(project, ReversibleCPMLOptions())
    original = small.plan(device='cuda', material_components=3)['gpu_reservation_bytes']
    limited = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        forward_kernel='fused_eh', adjoint_kernel='one_pass', block_size='auto', gpu_budget_bytes=original))
    with pytest.raises(ValueError, match='CUDA budget'):
        limited(base, fixed_epsilon=fixed)


@pytest.mark.cuda
def test_compile_failure_falls_back_without_changing_results(monkeypatch):
    require_cuda()
    import torchfdtd.reversible_cuda_fused as fused
    project, base, fixed = case()
    with torch.no_grad():
        baseline = ReversibleCPMLSimulation(project)(base, fixed_epsilon=fixed)
    def fail(*args, **kwargs):
        raise RuntimeError('CUDA kernel compilation failed: injected test failure')
    monkeypatch.setattr(fused, '_compiled', fail)
    with torch.no_grad():
        candidate = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
            forward_kernel='fused_eh'))(base, fixed_epsilon=fixed)
    assert torch.equal(baseline.signals, candidate.signals)
    assert 'compilation failed' in candidate.report['fallback_reason']['forward']
    assert candidate.report['forward_kernel_used'] == 'split'


@pytest.mark.cuda
def test_adjoint_compile_failure_keeps_split_gradient(monkeypatch):
    require_cuda()
    import torchfdtd.reversible_cuda_fused as fused
    project, base, fixed = case(True)
    x = base.clone().requires_grad_()
    baseline = ReversibleCPMLSimulation(project)(x, fixed_epsilon=fixed)
    expected, = torch.autograd.grad(baseline.signals.square().sum(), x)

    def fail(*args, **kwargs):
        raise RuntimeError('CUDA kernel compilation failed: injected adjoint failure')

    monkeypatch.setattr(fused, 'OnePassFieldAdjoint', fail)
    candidate = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
        adjoint_kernel='one_pass'))(x, fixed_epsilon=fixed)
    actual, = torch.autograd.grad(candidate.signals.square().sum(), x)
    assert torch.equal(baseline.signals, candidate.signals)
    assert torch.equal(expected, actual)
    assert candidate.report['adjoint_kernel_used'] == 'split'
    assert 'compilation failed' in candidate.report['fallback_reason']['adjoint']


@pytest.mark.cuda
@pytest.mark.parametrize('requested', [1024, 'auto', 'cached_auto'])
def test_block_specialization_failure_reports_original_launch(monkeypatch, requested):
    require_cuda()
    from collections import OrderedDict
    import torchfdtd.reversible_cuda_tuning as tuning
    project, base, fixed = case()
    original = tuning._compile

    def fail_specialization(source, *args):
        if '__launch_bounds__' in source:
            raise RuntimeError('CUDA kernel compilation failed: injected block failure')
        return original(source, *args)

    monkeypatch.setattr(tuning, '_CACHE', OrderedDict())
    cached = requested == 'cached_auto'
    if cached:
        requested = 'auto'
        x = base.clone().requires_grad_()
        warm = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
            block_size='auto'))(x, fixed_epsilon=fixed)
        torch.autograd.grad(warm.signals.square().sum(), x)
        assert tuning._CACHE
        cached_sources = {key[2] for key in tuning._CACHE}
        del warm, x
    monkeypatch.setattr(tuning, '_compile', fail_specialization)
    results = []
    for block in (None, requested):
        x = base.clone().requires_grad_()
        result = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
            block_size=block))(x, fixed_epsilon=fixed)
        grad, = torch.autograd.grad(result.signals.square().sum(), x)
        results.append((result, grad))
    assert torch.equal(results[0][0].signals, results[1][0].signals)
    assert torch.equal(results[0][1], results[1][1])
    report = results[1][0].report
    for label, launch in report['cuda_launches'].items():
        assert launch['fallback_reason']
        assert report['fallback_reason']['block_size:'+label] == launch['fallback_reason']
        assert launch['block_size'] == (128 if label.startswith('interior_') else 256)
        if launch['cached']:
            assert launch['cache_invalidated'], (label, launch)
            assert 'median_ms' not in launch
    if cached:
        # Both adjoint phases share a program. The first invalidates its cached
        # choice; the second then takes the cold-cache fallback for that source.
        invalidated = {launch['source_sha256'] for launch in report['cuda_launches'].values()
                       if launch.get('cache_invalidated')}
        assert invalidated == cached_sources
    assert not tuning._CACHE


@pytest.mark.cuda
@pytest.mark.parametrize('requested', [1024, 'auto', 'cached_auto'])
def test_tuning_does_not_retry_non_compile_errors(monkeypatch, requested):
    require_cuda()
    from collections import OrderedDict
    import torchfdtd.reversible_cuda_tuning as tuning
    project, base, fixed = case()
    monkeypatch.setattr(tuning, '_CACHE', OrderedDict())
    if requested == 'cached_auto':
        requested = 'auto'
        with torch.no_grad():
            ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
                block_size='auto'))(base, fixed_epsilon=fixed)
        assert tuning._CACHE
    calls = []
    failure = RuntimeError('CUDA out of memory: injected module-load failure')

    def fail(*args, **kwargs):
        calls.append(args)
        raise failure

    monkeypatch.setattr(tuning, '_compile', fail)
    with torch.no_grad(), pytest.raises(RuntimeError) as raised:
        ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
            block_size=requested))(base, fixed_epsilon=fixed)
    assert raised.value is failure
    assert len(calls) == 1


@pytest.mark.cuda
def test_non_compile_cuda_error_propagates(monkeypatch):
    require_cuda()
    import torchfdtd.reversible_cuda_fused as fused
    project, base, fixed = case()

    def fail(*args, **kwargs):
        raise RuntimeError('CUDA out of memory: injected allocation failure')

    monkeypatch.setattr(fused, '_compiled', fail)
    with torch.no_grad(), pytest.raises(RuntimeError, match='out of memory'):
        ReversibleCPMLSimulation(project, ReversibleCPMLOptions(
            forward_kernel='fused_eh'))(base, fixed_epsilon=fixed)


@pytest.mark.cuda
def test_fused_adjoint_directional_finite_difference():
    require_cuda()
    project, base, fixed = case(True, 96)
    from test_reversible_cpml_interval import material_maps
    base, fixed = material_maps(project, True, 'cuda')
    model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(interior_z=(8, 11),
        forward_kernel='fused_eh', adjoint_kernel='one_pass', block_size='auto'))
    x = base.clone().requires_grad_()
    value = model(x, fixed_epsilon=fixed).signals.square().mean()
    grad, = torch.autograd.grad(value, x)
    direction = torch.zeros_like(base)
    generator = torch.Generator().manual_seed(753)
    direction[:, :, 8:12] = torch.randn(direction[:, :, 8:12].shape, generator=generator).to('cuda')
    h = .005
    with torch.no_grad():
        plus = model(base+h*direction, fixed_epsilon=fixed).signals.square().mean()
        minus = model(base-h*direction, fixed_epsilon=fixed).signals.square().mean()
    fd = (plus-minus)/(2*h)
    derivative = (grad*direction).sum()
    assert derivative.abs() > 1e-8
    assert float((fd-derivative).abs()/derivative.abs()) < 2e-3


@pytest.mark.cuda
@pytest.mark.parametrize('shape', [(33, 9, 65), (7, 5, 170)])
def test_marching_chunks_and_deep_grid_fallback(shape):
    require_cuda()
    project, _, _ = case(steps=10)
    project.region.size = tuple(n*.1 for n in shape)
    material = torch.full(project.region.shape, 1.7, device='cuda')
    assert tuple(project.region.shape) == shape
    reference = _recorded_system(material, project, None)
    report = {}
    candidate = _recorded_system(material, project, None,
        ReversibleCPMLOptions(forward_kernel='fused_eh'), report)
    assert report['forward_kernel_variant'] == ('gather' if shape[2] == 170 else 'marching')
    # Populate every cell, including the periodic seams and PML, so this checks
    # all chunks even when the short source pulse cannot reach the far edges.
    generator = torch.Generator().manual_seed(187)
    for left, right in zip(reference.state(), candidate.state()):
        left.copy_(torch.randn(left.shape, generator=generator).to('cuda')*.01)
        right.copy_(left)
    for n in range(project.region.steps):
        _advance_recorded(reference, n, None, 4, shape[2]-6)
        _advance_recorded(candidate, n, None, 4, shape[2]-6)
        assert all(torch.equal(a, b) for a, b in zip(reference.state(), candidate.state()))


@pytest.mark.cuda
def test_one_pass_all_cotangents_and_cpml_memories():
    require_cuda()
    from torchfdtd.reversible_cpml_kernels import InteriorReconstruction
    project, material, _ = case(True, 17)
    helpers, traces = [], []
    seed = torch.randn((project.region.steps, len(project.monitors)), generator=torch.Generator().manual_seed(812)).cuda()
    for mode in ('split', 'one_pass'):
        system = _recorded_system(material, project, None)
        tape = []
        for n in range(project.region.steps):
            frame = torch.empty((2, *project.region.shape[:2], 2), device='cuda')
            _advance_recorded(system, n, frame, 4, 14)
            tape.append(frame)
        helpers.append(InteriorReconstruction(system, 4, 14, torch.zeros_like(material), seed,
            options=ReversibleCPMLOptions(adjoint_kernel=mode, block_size='auto'),
            report=dict(fallback_reason={})))
        traces.append(tape)
    for n in reversed(range(project.region.steps)):
        for helper, tape in zip(helpers, traces):
            helper.step(n, tape[n])
        for name in ('e_bar', 'h_bar', 'gradient'):
            assert torch.equal(getattr(helpers[0], name), getattr(helpers[1], name))
        assert all(torch.equal(a, b) for a, b in zip(helpers[0].psi_bars, helpers[1].psi_bars))
