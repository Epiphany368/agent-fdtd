"""CPU oracle for complex slab extension and its Hermitian transpose."""
import pytest
import torch
from torchfdtd.differentiable import _System
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd import Source, Monitor
from test_bloch_adjoint import scene


def test_interior_bloch_tile_needs_no_phase_materialization():
    p = scene()
    host = _System(p, torch.ones(p.region.shape, dtype=torch.float64), prepare_updates=False)
    operator = SlabBlockOperator(host, 4, 'cpu')
    descriptors = list(operator.tiles(1))
    assert operator._halo_phase(descriptors[0]) is not None
    assert operator._halo_phase(descriptors[1]) is None
    assert operator._halo_phase(descriptors[-1]) is not None
    value = torch.randn(4,3,dtype=torch.complex128)
    assert operator._phase_value(value, None) is value


def test_packed_interior_tile_does_not_alias_restart_or_epsilon():
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    host = _System(p, eps, prepare_updates=False)
    state = tuple(torch.randn_like(s) for s in host.state())
    original = tuple(s.clone() for s in state)
    operator = SlabBlockOperator(host, 4, 'cpu')
    descriptor = list(operator.tiles(1))[1]
    local, _, _ = operator._tile(eps, state, descriptor, 0, 1)
    for s in local.state():s.zero_()
    local.epsilon.zero_()
    for actual, expected in zip(state, original):torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(eps, torch.full_like(eps, 1.7), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('depth,nonuniform,diagonal', [(3, False, False), (18, True, True)])
@pytest.mark.parametrize('checkpoints', [0, 2])
@pytest.mark.parametrize('pml_axis', [None, 'x', 'z'])
def test_complex_slab_against_resident_autograd(dtype, depth, nonuniform, diagonal, checkpoints, pml_axis):
    p = scene(nonuniform)
    p.region.steps=max(p.region.steps,depth+2)
    if pml_axis is not None:
        p.region.dimension = '3d'
        p.region.mesh_type = 'uniform'
        p.region.mesh_coordinates = None
        for axis in 'xyz':
            for side in ('min', 'max'):
                getattr(p.region.boundaries, axis+'_'+side).kind = 'pml' if axis == pml_axis else 'bloch'
        p.region.bloch_phase = tuple(0 if a == pml_axis else (-.47 if a == 'x' else .82) for a in 'xyz')
        center = [0., 0., 0.]
        center['xyz'.index(pml_axis)] = -.3
        size = list(p.region.size)
        size['xyz'.index(pml_axis)] = 0
        component = 'Ey' if pml_axis == 'x' else 'Ex'
        p.sources = [Source(kind='plane', normal=pml_axis, size=tuple(size), center=tuple(center),
                            component=component, pulse='continuous')]
        p.monitors = [Monitor(component=component, center=(.1,.1,.1)),
                      Monitor(component='Hz', center=(-.6,.1,.1))]
    p.region.precision = 'float64' if dtype == torch.float64 else 'float32'
    torch.manual_seed(729)
    epsilon = 1.4+torch.rand(p.region.shape+((3,) if diagonal else ()), dtype=dtype)
    host = _System(p, epsilon, prepare_updates=False)
    state = tuple(torch.randn_like(s)*.03 for s in host.state())
    endpoint = tuple(torch.randn_like(s)*.02 for s in state)
    weights = torch.randn(depth, len(host.monitors), dtype=host.field_dtype)
    old = tuple(s.clone() for s in state)
    operator = SlabBlockOperator(host, 5, 'cpu', local_checkpoints=checkpoints)
    result, signals = operator.forward(epsilon, state, 1, depth)
    bars, gradient = operator.transpose(epsilon, state, 1, depth, endpoint, weights)
    eps = epsilon.clone().requires_grad_()
    inputs = tuple(s.clone().requires_grad_() for s in state)
    current, observed = inputs, []
    for j in range(depth):
        current = host.reference_step(current, j+1, eps)
        observed.append(host.observe(current))
    observed = torch.stack(observed)
    loss = (observed.conj()*weights).real.sum()
    loss = loss+sum((s.conj()*b).real.sum() for s,b in zip(current, endpoint))
    expected = torch.autograd.grad(loss, (*inputs, eps))
    tolerance = dict(rtol=8e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=3e-12)
    for actual, want in zip(result, current):torch.testing.assert_close(actual, want, **tolerance)
    torch.testing.assert_close(signals, observed, **tolerance)
    for actual, want in zip(bars, expected[:-1]):torch.testing.assert_close(actual, want, **tolerance)
    torch.testing.assert_close(gradient, expected[-1], **tolerance)
    assert not gradient.is_complex()
    for actual, want in zip(state, old):torch.testing.assert_close(actual, want, rtol=0, atol=0)
    if depth == 18 and pml_axis != 'x':
        assert any(len(d[2]) > 2*p.region.shape[0] for d in operator.tiles(depth))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_complex_file_banks_match_host_blocks(tmp_path, dtype):
    from torchfdtd.state_store import StateStore
    p = scene(True)
    p.region.precision = 'float64' if dtype == torch.float64 else 'float32'
    eps = torch.full(p.region.shape+(3,), 1.7, dtype=dtype)
    host = _System(p, eps, prepare_updates=False)
    torch.manual_seed(973)
    state = tuple(torch.randn_like(s)*.02 for s in host.state())
    endpoint = tuple(torch.randn_like(s)*.03 for s in host.state())
    weights = torch.randn(10, len(host.monitors), dtype=host.field_dtype)
    reference = SlabBlockOperator(host, 5, 'cpu', local_checkpoints=2)
    expected, signals = reference.forward(eps, state, 1, 10)
    expected_bar, gradient = reference.transpose(eps, state, 1, 10, endpoint, weights)
    state_bytes = sum(s.numel()*s.element_size() for s in state)
    sentinel = tmp_path/'keep.txt'
    sentinel.write_text('user data')
    with StateStore(tmp_path, 4*state_bytes) as store:
        stored = store.new_state(host.state())
        stored_endpoint = store.new_state(host.state())
        for bank, values in ((stored, state), (stored_endpoint, endpoint)):
            for target, value in zip(bank, values):
                target.index_copy_(0, torch.arange(value.shape[0]), value)
        operator = SlabBlockOperator(host, 5, 'cpu', local_checkpoints=2, state_factory=store.new_state)
        actual, observed = operator.forward(eps, stored, 1, 10)
        bars, actual_gradient = operator.transpose(eps, stored, 1, 10, stored_endpoint, weights)
        for a,b in zip(actual, expected):torch.testing.assert_close(a[:], b, rtol=0, atol=0)
        for a,b in zip(bars, expected_bar):torch.testing.assert_close(a[:], b, rtol=0, atol=0)
        torch.testing.assert_close(observed, signals, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, gradient, rtol=0, atol=0)
        assert store.live_bytes == store.peak_bytes == 4*state_bytes
        with pytest.raises(MemoryError, match='disk budget'):store.new_state(host.state())
    assert store.live_bytes == 0
    assert list(tmp_path.iterdir()) == [sentinel]


def test_complex_reservation_counts_field_bytes_and_rejects_small_budget(tmp_path):
    from dataclasses import replace
    from torchfdtd import StreamedAdjointOptions, StreamedSimulation
    from torchfdtd.streamed import _reservation
    p = scene()
    eps = torch.ones(p.region.shape, dtype=torch.float64)
    host = _System(p, eps, prepare_updates=False)
    options = StreamedAdjointOptions(device='cpu', slab_width=5, temporal_depth=3,
        state_storage='disk', state_directory=tmp_path/'scratch', disk_budget_bytes=128*1024**2)
    record = _reservation(p, eps, options)
    actual_bytes = sum(s.numel()*s.element_size() for s in host.state())
    assert record['state_bytes'] == actual_bytes
    assert record['disk_reservation_bytes'] == actual_bytes*record['state_bank_capacity']
    assert record['state_bank_capacity'] == options.checkpoints+3
    with pytest.raises(ValueError, match='disk budget'):
        _reservation(p, eps, replace(options, disk_budget_bytes=record['disk_reservation_bytes']-1))
    with pytest.raises(ValueError, match='host budget'):
        _reservation(p, eps, replace(options, host_budget_bytes=record['host_reservation_bytes']-1))
    with pytest.raises(ValueError, match='disk budget'):
        StreamedSimulation(p, replace(options, disk_budget_bytes=1))(eps)
    assert not (tmp_path/'scratch').exists()
