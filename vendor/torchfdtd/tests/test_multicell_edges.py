"""Public launch ownership, graph capture and optional compilation failures."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from torchfdtd import ReversibleCPMLOptions, ReversibleCPMLSimulation
from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
from torchfdtd.cuda_kernels import FusedYeeCUDA
from torchfdtd.cuda_complex import FusedComplexYeeCUDA
from torchfdtd.differentiable import _System
from test_multicell_cuda import cuda_case


def test_new_option_preserves_existing_positional_field_order():
    from dataclasses import fields
    names = [field.name for field in fields(ReversibleCPMLOptions)]
    assert names[-3:] == ['forward_kernel', 'adjoint_kernel', 'cells_per_thread']


@pytest.mark.parametrize('complex_fields', [False, True])
def test_public_constructor_rejects_field_offset_before_compilation(complex_fields):
    lanes = 6 if complex_fields else 3
    field = SimpleNamespace(shape=(1, 1, (2**31+lanes-1)//lanes, 3),
        is_cuda=True, is_complex=lambda: complex_fields,
        dtype=torch.complex64 if complex_fields else torch.float32)
    grid = SimpleNamespace(is_torch=True, E=field, material_states=[], subpixel=None)
    constructor = FusedComplexYeeCUDA if complex_fields else FusedYeeCUDA
    with pytest.raises(ValueError, match='signed 32-bit'):
        constructor(grid, cells_per_thread=4)


@pytest.mark.cuda
@pytest.mark.parametrize('physics', ['cpml', 'bloch', 'pmc'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_public_constructor_and_graph_replay_preserve_all_states(physics, precision):
    prepare_cuda_kernels()
    if physics == 'pmc':
        from test_pmc_general import scene, random_epsilon
        p = scene([('pmc', 'pmc'), ('pec', 'pmc'), ('pml', 'pml')],
            size=(.9, .7, 2.1), backend='cuda', precision=precision)
        base = random_epsilon(p.region, 417, device='cuda')
    else:
        p, base, _ = cuda_case(True, physics == 'bloch')
        p.region.precision = precision
        base = base.to(torch.float64 if precision == 'float64' else torch.float32)
    p.region.cuda_kernel = 'fused'
    baseline, selected = _System(p, base), _System(p, base)
    constructor = FusedComplexYeeCUDA if physics == 'bloch' else FusedYeeCUDA
    report = {}
    launch = constructor(selected.grid, cells_per_thread=4, block_size=512, report=report)
    generator = torch.Generator(device='cuda').manual_seed(273)
    for a, b in zip(baseline.state(), selected.state()):
        values = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)*.01
        a.copy_(values)
        b.copy_(values)
    # Compile all specializations and create views before capture. Replays own
    # the same field/CPML bindings throughout the graph's lifetime.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.update_E()
        launch.update_H()
    for _ in range(8):
        baseline.kernel.update_E()
        baseline.kernel.update_H()
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(baseline.state(), selected.state()))
    assert all(row['cells_per_thread'] == 4 and row['block_size'] == 512
               for row in report['cuda_launches'].values())


@pytest.mark.cuda
@pytest.mark.parametrize('setting', ['fixed', 'auto', 'cached_auto'])
def test_multicell_compile_failure_restores_original_launch_and_vjp(setting, monkeypatch):
    import torchfdtd.reversible_cuda_tuning as tuning
    p, base, fixed = cuda_case(True, True)
    monkeypatch.setattr(tuning, '_CACHE', OrderedDict())
    options = ReversibleCPMLOptions(cells_per_thread=4 if setting == 'fixed' else 'auto',
        block_size=512 if setting == 'fixed' else 'auto')
    if setting == 'cached_auto':
        warm = base.clone().requires_grad_()
        result = ReversibleCPMLSimulation(p, options)(warm, fixed_epsilon=fixed)
        torch.autograd.grad(result.signals.abs().square().sum(), warm)
        assert tuning._CACHE
    original = tuning._compile

    def fail_specialization(source, *args):
        if '__launch_bounds__' in source:
            raise RuntimeError('CUDA kernel compilation failed: injected launch failure')
        return original(source, *args)

    monkeypatch.setattr(tuning, '_compile', fail_specialization)
    outcomes = []
    for settings in (ReversibleCPMLOptions(), options):
        value = base.clone().requires_grad_()
        result = ReversibleCPMLSimulation(p, settings)(value, fixed_epsilon=fixed)
        gradient, = torch.autograd.grad(result.signals.abs().square().sum(), value)
        outcomes.append((result, gradient))
    assert torch.equal(outcomes[0][0].signals, outcomes[1][0].signals)
    assert torch.equal(outcomes[0][1], outcomes[1][1])
    for row in outcomes[1][0].report['cuda_launches'].values():
        assert row['cells_per_thread'] == 1 and row['block_size'] == 256
        assert row['fallback_reason']
        if row['cached']:
            assert row['cache_invalidated'] and 'median_ms' not in row
    assert not tuning._CACHE


@pytest.mark.cuda
def test_block_only_tuning_retains_original_report_keys():
    p, base, fixed = cuda_case(False, False)
    with torch.no_grad():
        result = ReversibleCPMLSimulation(p, ReversibleCPMLOptions(block_size='auto'))(
            base, fixed_epsilon=fixed)
    for row in result.report['cuda_launches'].values():
        assert row['cells_per_thread'] == 1
        assert set(row['median_ms']) == {128, 256, 512, 1024}
