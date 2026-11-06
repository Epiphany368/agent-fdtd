"""Forward-only runs and single-read validity checks of recorded CPML.

Each native path is compared bitwise (``torch.equal``) with the computation it
replaces: the recorded forward (``forward_only='never'``), the per-chunk host
reads of the 0.16.0 checks (``legacy_require_finite``,
``legacy_require_material``) and the list-indexed DFT blocks.
"""
from collections import Counter
import inspect
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import torchfdtd.reversible_cpml as core
from torchfdtd import ReversibleCPMLOptions, ReversibleCPMLSimulation
from torchfdtd.reversible_cpml_planes import ReversibleCPMLPlaneSimulation
from test_adjoint_spectrum_index import use_legacy_blocks
from test_reversible_cpml import fixture
from test_reversible_cpml_extended import extended_fixture
from test_reversible_cpml_planes import fixture as plane_fixture


def legacy_require_finite(value, message, chunk):
    flat = value.reshape(-1)
    chunk = max(1, chunk // 2) if value.is_complex() else chunk
    for start in range(0, flat.numel(), chunk):
        if not bool(torch.isfinite(flat[start:start + chunk]).all()):
            raise RuntimeError(message)


def legacy_require_material(value, chunk):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), chunk):
        block = flat[start:start + chunk]
        if not bool(torch.isfinite(block).all()) or bool((block < 1).any()):
            raise ValueError('The recorded CPML CFL contract requires finite epsilon >= 1 in both maps.')


class ScalarReads(TorchDispatchMode):
    """Count scalar reads (a host synchronization on CUDA) by solver function."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            frame = inspect.currentframe().f_back
            while frame is not None and frame.f_code.co_name not in (
                    '_require_finite', '_require_material', 'legacy_require_finite', 'legacy_require_material'):
                frame = frame.f_back
            self.counts['checks' if frame is not None else 'other'] += 1
            del frame
        return func(*args, **(kwargs or {}))


def cuda_or_cpu(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cuda':
        pytest.importorskip('cupy')


CASES = {
    'point-scalar': lambda: fixture(96),
    'point-diagonal': lambda: extended_fixture(False),
    'point-diagonal-bloch': lambda: extended_fixture(True),
    'plane-scalar': lambda: plane_fixture(False),
    'plane-diagonal-bloch': lambda: plane_fixture(True),
}


def build(case, device, options):
    project, base, fixed = CASES[case]()
    if device == 'cuda':
        project.region.cuda_kernel = 'fused'
    if case.startswith('plane'):
        model = ReversibleCPMLPlaneSimulation(project, options, quadrature_counts={'incident': (3, 4), 'detector': (3, 4)})
        frequencies = torch.tensor([.033, .057], dtype=torch.float32, device=device)/project.region.time_step
        def call(parameter):
            planes = model(parameter, frequencies, fixed_epsilon=fixed.to(device), block_size=7)
            return torch.stack([plane.fields for plane in planes.values()]), next(iter(planes.values())).report
    else:
        model = ReversibleCPMLSimulation(project, options)
        def call(parameter):
            result = model(parameter, fixed_epsilon=fixed.to(device))
            return result.signals, result.report
    return base.to(device), call


def execute(case, device, options):
    """No-grad output, recorded output and a seeded material VJP."""
    base, call = build(case, device, options)
    with torch.no_grad():
        free, free_report = call(base.clone())
    parameter = base.clone().requires_grad_()
    recorded, recorded_report = call(parameter)
    seed = torch.randn(recorded.shape, dtype=recorded.dtype, generator=torch.Generator().manual_seed(611)).to(device)
    gradient, = torch.autograd.grad(recorded, parameter, seed)
    return free, recorded.detach(), gradient, free_report, recorded_report


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('case', list(CASES))
def test_native_paths_equal_the_recorded_computation(monkeypatch, device, case):
    cuda_or_cpu(device)
    torch.set_num_threads(1)
    options = dict(trace_storage='cpu', trace_transfers='async') if device == 'cuda' and case == 'plane-diagonal-bloch' else {}
    with monkeypatch.context() as patch:
        use_legacy_blocks(patch)
        patch.setattr(core, '_require_finite', legacy_require_finite)
        patch.setattr(core, '_require_material', legacy_require_material)
        old = execute(case, device, ReversibleCPMLOptions(forward_only='never', **options))
    new = execute(case, device, ReversibleCPMLOptions(**options))
    free, recorded, gradient, free_report, recorded_report = new
    assert free_report['forward_only'] is True and recorded_report['forward_only'] is False
    assert old[3]['forward_only'] is False and old[3]['sampled_forward_peak'] > 0
    assert free_report['sampled_forward_peak'] is None and free_report['terminal_copies'] == 0
    assert free.dtype == old[0].dtype and not free.requires_grad
    assert torch.equal(free, old[0])
    assert torch.equal(free, recorded)
    assert torch.equal(recorded, old[1])
    assert torch.equal(gradient, old[2])
    assert free.abs().max() > 0 and gradient.abs().max() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_forward_only_selection_and_no_tape(monkeypatch, device):
    cuda_or_cpu(device)
    project, base, fixed = fixture(20)
    base, fixed = base.to(device), fixed.to(device)
    model = ReversibleCPMLSimulation(project)
    recorded = core._RecordedCPML.apply
    calls = []
    def tracked(*args):
        calls.append(args[0].requires_grad)
        return recorded(*args)
    monkeypatch.setattr(core._RecordedCPML, 'apply', tracked)
    # Grad enabled but nothing to differentiate, grad disabled, inference mode.
    constant = model(base, fixed_epsilon=fixed)
    with torch.no_grad():
        detached = model(base.clone().requires_grad_(), fixed_epsilon=fixed)
    with torch.inference_mode():
        inference = model(base, fixed_epsilon=fixed)
    assert calls == []
    for result in (constant, detached, inference):
        assert result.report['forward_only'] is True and result.signals.grad_fn is None
    differentiable = model(base.clone().requires_grad_(), fixed_epsilon=fixed)
    assert calls == [True] and differentiable.report['forward_only'] is False
    assert differentiable.signals.grad_fn is not None
    never = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(forward_only='never'))
    with torch.no_grad():
        kept = never(base, fixed_epsilon=fixed)
    assert calls == [True, False] and kept.report['forward_only'] is False
    assert kept.report['sampled_forward_peak'] > 0
    assert torch.equal(constant.signals, kept.signals)
    assert torch.equal(inference.signals, differentiable.signals.detach())


@pytest.mark.parametrize('value', ['always', 'Auto', True, None])
def test_forward_only_option_is_validated(value):
    with pytest.raises(ValueError, match='forward_only'):
        ReversibleCPMLOptions(forward_only=value)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('diagonal', [False, True])
def test_checks_read_once_per_tensor(monkeypatch, device, diagonal):
    cuda_or_cpu(device)
    if diagonal:
        project, base, fixed = extended_fixture(False)
    else:
        project, base, fixed = fixture(96)
    base, fixed = base.to(device), fixed.to(device)
    model = ReversibleCPMLSimulation(project)
    chunk = model.plan(device=device, material_components=3 if diagonal else 1)['diagnostic_chunk_elements']
    reads = {}
    for label, patched in (('legacy', True), ('native', False)):
        with monkeypatch.context() as patch:
            if patched:
                patch.setattr(core, '_require_finite', legacy_require_finite)
                patch.setattr(core, '_require_material', legacy_require_material)
            for grad in (True, False):
                parameter = base.clone().requires_grad_(grad)
                counter = ScalarReads()
                with counter:
                    result = model(parameter, fixed_epsilon=fixed)
                reads[label, grad] = dict(counter.counts)
                assert result.report['forward_only'] is not grad
    blocks = lambda numel: -(-numel//chunk)
    material = 2*2*blocks(base.numel())
    trace = blocks(result.report['trace_bytes']//4)
    signals = blocks(project.region.steps*len(project.monitors))
    fields = blocks(3*math.prod(project.region.shape))
    assert trace > 1 and blocks(base.numel()) >= (2 if diagonal else 1)
    # Recorded: two material maps, observations and the boundary trace.
    assert reads['legacy', True]['checks'] == material+signals+trace
    assert reads['native', True]['checks'] == 2+2
    # Forward only: two material maps, observations and the final E and H.
    assert reads['legacy', False]['checks'] == material+signals+2*fields
    assert reads['native', False]['checks'] == 2+3
    assert reads['native', True]['other'] == reads['legacy', True]['other']
    print(dict(device=device, diagonal=diagonal, chunk=chunk,
               **{f'{label}_{"recorded" if grad else "forward_only"}': value for (label, grad), value in reads.items()}))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('diagonal', [False, True])
def test_checks_still_reject_every_chunk(device, diagonal):
    cuda_or_cpu(device)
    if diagonal:
        project, base, fixed = extended_fixture(False)
    else:
        project, base, fixed = fixture(20)
    base, fixed = base.to(device), fixed.to(device)
    model = ReversibleCPMLSimulation(project)
    for target, position, value in ((base, -1, float('nan')), (fixed, -1, .5), (base, 0, float('inf')),
                                    (fixed, base.numel()//2, -float('inf'))):
        original = target.view(-1)[position].clone()
        target.view(-1)[position] = value
        for grad in (True, False):
            with pytest.raises(ValueError, match='finite epsilon >= 1'):
                model(base.clone().requires_grad_(grad), fixed_epsilon=fixed)
        target.view(-1)[position] = original
    chunk = model.plan(device=device)['diagnostic_chunk_elements']
    for dtype in (torch.float32, torch.complex64):
        value = torch.zeros(3*chunk+5, dtype=dtype, device=device)
        core._require_finite(value, 'finite', chunk)
        value[-1] = float('nan')
        with pytest.raises(RuntimeError, match='finite'):
            core._require_finite(value, 'finite', chunk)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_forward_only_rejects_nonfinite_fields(monkeypatch, device):
    cuda_or_cpu(device)
    project, base, fixed = fixture(20)
    base, fixed = base.to(device), fixed.to(device)
    model = ReversibleCPMLSimulation(project)
    advance = core._advance_recorded
    def poisoned(system, step, frame, a, b):
        advance(system, step, frame, a, b)
        if step + 1 == project.region.steps:
            # A cell far from every monitor, after its last observation.
            system.state()[1][0, 0, 1, 0] = float('nan')
    monkeypatch.setattr(core, '_advance_recorded', poisoned)
    with torch.no_grad(), pytest.raises(RuntimeError, match='CPML fields became nonfinite'):
        model(base, fixed_epsilon=fixed)


def legacy_interior_scale(system, a, b, chunk=None):
    """The 0.16.0 drift scale: fixed blocks of 65536 real lanes."""
    square = torch.zeros((), dtype=torch.float64, device=system.device)
    peak = torch.zeros((), dtype=torch.float32, device=system.device)
    for field in system.state()[:2]:
        capacity = 32768 if field.is_complex() else 65536
        rows = field.view(-1, field.shape[2], 3)
        for z in range(a, b + 1, capacity // 3):
            end = min(b + 1, z + capacity // 3)
            row_count = max(1, capacity // (3 * (end - z)))
            for row in range(0, rows.shape[0], row_count):
                block = rows[row:row + row_count, z:end].reshape(-1)
                peak = torch.maximum(peak, block.abs().amax())
                lanes = torch.view_as_real(block) if block.is_complex() else block
                square.add_(lanes.to(torch.float64).square().sum())
    maximum, norm = float(peak), math.sqrt(float(square))
    if not math.isfinite(maximum) or not math.isfinite(norm):
        raise RuntimeError('Reversible interior fields became nonfinite.')
    return maximum, norm


def wide(case, device):
    """48 x 49 transverse cells: the 11-plane interior spans several default chunks."""
    project = extended_fixture(True)[0] if case == 'diagonal-bloch' else fixture(64)[0]
    project.region.size = (4.8, 4.9, 2)
    if device == 'cuda':
        project.region.cuda_kernel = 'fused'
    shape = project.region.shape
    base = torch.full(shape, 1.4)
    base[:, :, 9:11] = 2.25
    base[20:30, 10:40, 6:12] = 3.1
    fixed = torch.full(shape, 1.3)
    fixed[:, :, 15:] = 1.7
    if case == 'diagonal-bloch':
        offsets = torch.tensor([0., .2, .4])
        base, fixed = base[..., None]+offsets, fixed[..., None]+offsets
    return project, base.contiguous().to(device), fixed.contiguous().to(device)


def timeless(report):
    report = dict(report, last_backward=dict(report['last_backward']))
    del report['forward_seconds'], report['last_backward']['seconds']
    return report


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('case', ['scalar', 'diagonal-bloch'])
def test_diagnostic_chunk_default_is_unchanged_and_large_chunk_moves_only_the_drift_l2(monkeypatch, device, case):
    cuda_or_cpu(device)
    torch.set_num_threads(1)
    project, base, fixed = wide(case, device)

    def run(**options):
        model = ReversibleCPMLSimulation(project, ReversibleCPMLOptions(**options))
        parameter = base.clone().requires_grad_()
        result = model(parameter, fixed_epsilon=fixed)
        seed = torch.randn(result.signals.shape, dtype=result.signals.dtype,
                           generator=torch.Generator().manual_seed(907)).to(device)
        gradient, = torch.autograd.grad(result.signals, parameter, seed)
        return result.signals.detach(), gradient, result.report

    a, b = ReversibleCPMLSimulation(project).interior_z
    lanes = 3*project.region.shape[0]*project.region.shape[1]*(b-a+1)*(2 if project.region.complex_fields else 1)
    assert lanes > 65536
    with monkeypatch.context() as patch:
        patch.setattr(core, '_interior_scale', legacy_interior_scale)
        old = run()
    default, large = run(), run(diagnostic_chunk_elements=1 << 20)
    assert torch.equal(default[0], old[0]) and torch.equal(default[1], old[1])
    assert timeless(default[2]) == timeless(old[2])
    assert default[2]['diagnostic_chunk_elements'] == 65536
    assert torch.equal(large[0], default[0]) and torch.equal(large[1], default[1])
    assert large[2]['diagnostic_chunk_elements'] == lanes
    assert large[2]['sampled_forward_peak'] == default[2]['sampled_forward_peak']
    assert large[2]['last_backward']['initial_max_abs'] == default[2]['last_backward']['initial_max_abs']
    for value, reference in ((large[2]['sampled_forward_l2'], default[2]['sampled_forward_l2']),
                             (large[2]['last_backward']['initial_l2'], default[2]['last_backward']['initial_l2'])):
        assert reference > 0 and abs(value-reference) <= 1e-12*reference
        print(dict(case=case, device=device, drift_l2_relative_difference=abs(value-reference)/reference))
    assert default[1].abs().max() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_diagnostic_chunk_reservation_grows_by_24_bytes_per_lane(device):
    cuda_or_cpu(device)
    project, _, _ = wide('scalar', device)
    a, b = ReversibleCPMLSimulation(project).interior_z
    lanes = 3*project.region.shape[0]*project.region.shape[1]*(b-a+1)
    plans = [ReversibleCPMLSimulation(project, ReversibleCPMLOptions(diagnostic_chunk_elements=n)).plan(device=device)
             for n in (65536, 1 << 20)]
    assert [p['diagnostic_chunk_elements'] for p in plans] == [65536, lanes]
    growth = 24*(lanes-65536)
    assert plans[1]['diagnostic_reservation_bytes']-plans[0]['diagnostic_reservation_bytes'] == growth
    assert plans[1]['workspace_reservation_bytes']-plans[0]['workspace_reservation_bytes'] == growth
    extra = plans[1]['memory_reservation_bytes']-plans[0]['memory_reservation_bytes']
    # CUDA adds its 5% allocation headroom on top of the scratch.
    assert extra == growth if device == 'cpu' else growth <= extra <= growth*21//20+1
    # A chunk larger than the interior is bounded by the interior.
    small, _, _ = fixture(20)
    bounded = [ReversibleCPMLSimulation(small, ReversibleCPMLOptions(diagnostic_chunk_elements=n)).plan(device=device)
               for n in (65536, 1 << 26)]
    for key in ('diagnostic_chunk_elements', 'diagnostic_reservation_bytes', 'memory_reservation_bytes'):
        assert bounded[0][key] == bounded[1][key]
    assert bounded[0]['diagnostic_chunk_elements'] == 3*6*7*11


@pytest.mark.parametrize('value', [65535, (1 << 26)+1, True, 65536.])
def test_diagnostic_chunk_option_is_validated(value):
    with pytest.raises(ValueError, match='diagnostic_chunk_elements'):
        ReversibleCPMLOptions(diagnostic_chunk_elements=value)
