import gc

import numpy as np
import pytest
import torch

from torchfdtd.cuda_kernels import _compile, _direct_cuda_view
from torchfdtd.tile_workspace import TileWorkspace
from test_differentiable import gpu


def test_workspace_reuses_storage_and_invalidates_old_pointer_bindings():
    workspace = TileWorkspace('cpu', cache_entries=2)
    value = workspace.array('payload', (20,), torch.float64)
    pointer = value.data_ptr()
    assert workspace.array('payload', (10,), torch.float64).data_ptr() == pointer
    calls = []
    def bind(tensor):
        calls.append(tensor.data_ptr())
        return tensor.detach()
    first = workspace.cuda_arguments('source', [value], bind)
    assert workspace.cuda_arguments('source', [value], bind) is first
    assert len(calls) == 1
    larger = workspace.array('payload', (40,), torch.float64)
    assert not workspace.bindings
    assert larger.data_ptr() != pointer
    for key in ('a', 'b', 'c'):workspace.cuda_arguments(key, [larger], bind)
    assert len(workspace.bindings) == 2
    assert workspace.allocated_bytes == 320


@pytest.mark.parametrize('dtype',[torch.float64,torch.complex64,torch.complex128])
def test_direct_view_owns_offset_allocation_on_nondefault_stream(dtype):
    gpu()
    import cupy
    producer = torch.arange(128, device='cuda', dtype=torch.float64).to(dtype)
    if producer.is_complex():producer = producer+1j*(producer+1)
    selected = producer[8:120]
    expected = selected.cpu().numpy().copy()*2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), cupy.cuda.Device(torch.cuda.current_device()), cupy.cuda.ExternalStream(stream.cuda_stream):
        array = _direct_cuda_view(cupy, selected)
        assert array.data.ptr == selected.data_ptr()
        if selected.is_complex():
            with pytest.raises(ValueError, match='resolved'):_direct_cuda_view(cupy, selected.conj())
        del selected, producer
        gc.collect()
        # Use the solver's header-free compiler. CuPy's unrelated elementwise
        # cache can require toolkit headers under a non-ASCII Windows path.
        real = 'float' if dtype == torch.complex64 else 'double'
        count = array.size*(2 if dtype.is_complex else 1)
        source = ('extern "C" __global__ void scale('+real+'* p, int n) {'
                  'int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)p[i]*=2;}')
        kernel, module = _compile(source, torch.cuda.current_device(),
                                  cupy.cuda.Device().compute_capability, 'scale')
        kernel(((count+127)//128,), (128,), (array, np.int32(count)))
        actual = cupy.asnumpy(array)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='contiguous CUDA'):
        _direct_cuda_view(cupy, torch.ones(4))


@pytest.mark.parametrize('binding', ['direct', 'dlpack'])
@pytest.mark.parametrize('local_checkpoints', [0,2])
@pytest.mark.parametrize('buffers', [0, 1, 2, 3])
def test_reused_streamed_workspaces_match_unreused_on_nondefault_stream(binding, buffers, local_checkpoints):
    gpu()
    from torchfdtd import StreamedAdjointOptions, StreamedSimulation
    from test_differentiable import project
    from dataclasses import replace
    p = project(steps=13, periodic=False)
    options = StreamedAdjointOptions(slab_width=3, temporal_depth=4, cuda_binding=binding,local_checkpoints=local_checkpoints,
                                     tile_transfers='async' if buffers else 'sync', tile_buffers=buffers or 1)
    epsilon = torch.full(p.region.shape, 1.8, dtype=torch.float64, requires_grad=True)
    expected = StreamedSimulation(p, replace(options, reuse_tile_buffers=False, tile_transfers='sync',local_checkpoints=0))(epsilon)
    wanted, = torch.autograd.grad(expected.signals.square().sum(), epsilon)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        independent = epsilon.detach().clone().requires_grad_()
        result = StreamedSimulation(p, options)(independent)
        got, = torch.autograd.grad(result.signals.square().sum(), independent)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result.signals, expected.signals, rtol=0, atol=0)
    torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    assert result.report['backward_workspace']['binding_hits'] > 0
    report = result.report['backward_workspace']
    assert report['buffer_bytes'] == sum(report['buffers_by_name'].values())
    assert report['pinned_bytes'] == sum(report['pinned_by_name'].values())
    assert report['host_staging_bytes'] == sum(report['host_staging_by_name'].values())
    assert report['buffers_by_name']['output_packet'] > 0
    if buffers:
        assert result.report['backward_workspace']['h2d_bytes'] > 0
        assert result.report['backward_workspace']['d2h_bytes'] > 0
        assert result.report['backward_workspace']['pinned_bytes'] > 0


@pytest.mark.parametrize('asynchronous', [False, True])
def test_cuda_packets_reuse_staging_on_nondefault_stream(asynchronous):
    gpu()
    workspace = TileWorkspace('cuda', asynchronous=asynchronous)
    stream = torch.cuda.Stream()
    values = (torch.arange(120, dtype=torch.float64).reshape(4,5,6).transpose(0,1),
              torch.arange(41, dtype=torch.float64).to(torch.complex128).conj())
    pointers = None
    with torch.cuda.stream(stream):
        for size in (41,23,41,17):
            workspace.drain()
            inputs = (values[0] + size, values[1][:size])
            packed, layout = workspace.copy_packet('payload', inputs)
            uploaded = layout.unpack(packed)
            # Exercise actual compute between the H2D and D2H streams.
            for value in uploaded:value.mul_(2)
            returned = workspace.to_host(uploaded).wait()
            for actual, expected in zip(returned, inputs):
                torch.testing.assert_close(actual, 2*expected, rtol=0, atol=0)
            addresses = tuple(value.data_ptr() for pool in
                              (workspace.buffers, workspace.pinned, workspace.host_staging)
                              for value in pool.values())
            if pointers is None:pointers = addresses
            else:assert addresses == pointers
        workspace.drain()
    assert workspace.allocations == 2
    assert (workspace.pinned_allocations if asynchronous else workspace.host_allocations) == 2
    assert not (workspace.host_staging if asynchronous else workspace.pinned)
    assert workspace.h2d_bytes == workspace.d2h_bytes > 0
    assert not workspace.events


@pytest.mark.parametrize('failure_point', ['_tile', '_return'])
def test_async_pipeline_drains_after_partial_failure_and_can_retry(monkeypatch, failure_point):
    gpu()
    from torchfdtd.differentiable import _System
    from torchfdtd.spacetime import SlabBlockOperator
    from test_differentiable import project
    p = project(steps=10, periodic=True)
    epsilon = torch.full(p.region.shape, 1.6, dtype=torch.float64)
    host = _System(p, epsilon)
    operation = SlabBlockOperator(host, 3, tile_transfers='async', tile_buffers=2)
    original = getattr(operation, failure_point)
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:raise RuntimeError('injected tile preparation failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(operation, failure_point, fail_second)
    with pytest.raises(RuntimeError, match='injected'):operation.forward(epsilon, host.state(), 0, 2)
    assert all(not w.events for w in operation.workspaces)
    monkeypatch.setattr(operation, failure_point, original)
    got, signals = operation.forward(epsilon, host.state(), 0, 2)
    expected, observed = SlabBlockOperator(host, 3).forward(epsilon, host.state(), 0, 2)
    for actual, want in zip(got, expected):torch.testing.assert_close(actual, want, rtol=0, atol=0)
    torch.testing.assert_close(signals, observed, rtol=0, atol=0)
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_output_growth_does_not_hold_expired_cuda_allocation():
    from torchfdtd.tile_workspace import TileWorkspace
    workspace=TileWorkspace('cuda')
    count=1024**2
    baseline=torch.cuda.memory_allocated()
    workspace.array('output_packet',(count,),torch.uint8,invalidate_bindings=False)
    torch.cuda.reset_peak_memory_stats()
    value=workspace.array('output_packet',(2*count,),torch.uint8,invalidate_bindings=False)
    assert torch.cuda.max_memory_allocated()-baseline==2*count
    assert value.numel()==2*count
    # A caller retaining an old view still owns it. Growth must not invalidate
    # such a tensor merely to reduce the peak.
    retained=value
    grown=workspace.array('output_packet',(3*count,),torch.uint8,invalidate_bindings=False)
    retained.fill_(7);grown.fill_(9)
    assert bool((retained==7).all()) and bool((grown==9).all())
