"""Native complex slab kernels, mixed packets and asynchronous slot ownership."""
import pytest
import torch
from torchfdtd.differentiable import _System
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd.state_store import StateStore
from torchfdtd import Source, Monitor, StreamedAdjointOptions
from test_bloch_adjoint import scene
from test_differentiable import gpu


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('transfers,binding,storage', [('sync','dlpack','host'), ('async','direct','host'),
                                                     ('async','direct','disk')])
@pytest.mark.parametrize('pml_axis', [None, 'x', 'z'])
def test_cuda_complex_block_and_transpose(tmp_path, dtype, transfers, binding, storage, pml_axis):
    gpu()
    p = scene(True)
    if pml_axis is not None:
        p.region.dimension = '3d'
        p.region.mesh_type = 'uniform'
        p.region.mesh_coordinates = None
        for axis in 'xyz':
            for side in ('min', 'max'):
                getattr(p.region.boundaries, axis+'_'+side).kind = 'pml' if axis == pml_axis else 'bloch'
        p.region.bloch_phase = tuple(0 if a == pml_axis else (-.47 if a == 'x' else .82) for a in 'xyz')
        center = [0.,0.,0.]
        center['xyz'.index(pml_axis)] = -.3
        size = list(p.region.size)
        size['xyz'.index(pml_axis)] = 0
        component = 'Ey' if pml_axis == 'x' else 'Ex'
        p.sources = [Source(kind='plane', normal=pml_axis, size=tuple(size), center=tuple(center),
                            component=component, pulse='continuous')]
        p.monitors = [Monitor(component=component, center=(.1,.1,.1)), Monitor(component='Hz', center=(-.6,.1,.1))]
    p.region.precision = 'float64' if dtype == torch.float64 else 'float32'
    torch.manual_seed(421)
    eps = 1.5+torch.rand(p.region.shape+(3,), dtype=dtype)
    host = _System(p, eps, prepare_updates=False)
    initial = tuple(torch.randn_like(s)*.03 for s in host.state())
    endpoint = tuple(torch.randn_like(s)*.02 for s in host.state())
    weights = torch.randn(10, len(host.monitors), dtype=host.field_dtype)
    oracle = SlabBlockOperator(host, 5, 'cpu', local_checkpoints=2)
    expected, signals = oracle.forward(eps, initial, 1, 10)
    wanted, gradient = oracle.transpose(eps, initial, 1, 10, endpoint, weights)
    from torchfdtd.streamed import _reservation
    reservation = _reservation(p, eps, StreamedAdjointOptions(device='cuda', slab_width=5,
        temporal_depth=10, local_checkpoints=2, tile_transfers=transfers, tile_buffers=3))
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    store = StateStore(tmp_path, 128*1024**2) if storage == 'disk' else None
    try:
        operator = SlabBlockOperator(host, 5, 'cuda', local_checkpoints=2,
            tile_transfers=transfers, tile_buffers=3, cuda_binding=binding,
            state_factory=None if store is None else store.new_state)
        if store is not None:
            states = []
            for state in (initial, endpoint):
                bank = store.new_state(host.state())
                for target, value in zip(bank, state):target.index_copy_(0, torch.arange(value.shape[0]), value)
                states.append(bank)
            initial, endpoint = states
        tolerance = dict(rtol=8e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=3e-12)
        # Reusing every slot also checks cached argument bindings and old seeds.
        for repeat in range(2):
            result, observed = operator.forward(eps, initial, 1, 10)
            bars, actual_gradient = operator.transpose(eps, initial, 1, 10, endpoint, weights)
            for a,b in zip(result, expected):torch.testing.assert_close(a[:] if store else a, b, **tolerance)
            for a,b in zip(bars, wanted):torch.testing.assert_close(a[:] if store else a, b, **tolerance)
            torch.testing.assert_close(observed, signals, **tolerance)
            torch.testing.assert_close(actual_gradient, gradient, **tolerance)
        report = operator.workspace_report()
        assert report['binding_hits'] > 0
        if transfers == 'async':assert report['pinned_bytes'] > 0
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated()-baseline <= reservation['gpu_reservation_bytes']
    finally:
        if store is not None:store.close()
    assert not list(tmp_path.iterdir())
