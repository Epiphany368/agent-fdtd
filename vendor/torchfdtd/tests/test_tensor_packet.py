import pytest
import torch
from torchfdtd.tensor_packet import TensorLayout, pack_tensors
from torchfdtd.spacetime import _host_copies


def test_mixed_precision_complex_integer_and_alignment():
    values=(torch.tensor([2**60+3],dtype=torch.int64),
            torch.tensor([1.,-0.,float('inf')],dtype=torch.float32),
            torch.tensor([[1+2j,-3+4j]],dtype=torch.complex128).conj(),
            torch.arange(6,dtype=torch.float64).reshape(2,3).T,
            torch.tensor([True,False]),torch.empty((0,3),dtype=torch.float64),
            torch.tensor([0x7ff8000000000015],dtype=torch.int64).view(torch.float64))
    packet,layout=pack_tensors(values)
    assert packet.dtype==torch.uint8
    recovered=layout.unpack(packet.clone())
    for a,b in zip(recovered,values):
        assert a.dtype==b.dtype and a.shape==b.shape
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
        assert a.reshape(-1).view(torch.uint8).tolist()==b.resolve_conj().contiguous().reshape(-1).view(torch.uint8).tolist()
    for (_,dtype,offset,_) in layout.entries:
        assert offset%torch.empty((),dtype=dtype).element_size()==0
    for a,b in zip(_host_copies(values),recovered):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


def test_homogeneous_fast_path_and_layout_guard():
    values=(torch.arange(3,dtype=torch.float64,requires_grad=True),torch.ones(2,dtype=torch.float64))
    packet,layout=pack_tensors(values)
    assert packet.dtype==torch.float64 and packet.numel()==5 and not packet.requires_grad
    result=layout.unpack(packet)
    result[0][0]=7
    assert packet[0]==7 and values[0][0]==0
    with pytest.raises(ValueError):layout.unpack(packet[:-1])
    with pytest.raises(ValueError):pack_tensors([])


def test_cpu_workspace_mixed_roundtrip():
    from torchfdtd.tile_workspace import TileWorkspace
    values=(torch.tensor([1+2j],dtype=torch.complex64),torch.tensor([.25],dtype=torch.float64))
    workspace=TileWorkspace('cpu')
    packet,layout=pack_tensors(values)
    copied=workspace.copy('payload',packet)
    staged=layout.unpack(copied)
    recovered=workspace.to_host(staged).wait()
    for a,b in zip(recovered,values):
        assert a.dtype==b.dtype
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mixed', [False, True])
def test_direct_packet_has_identical_bytes_without_contiguous_temporaries(monkeypatch, mixed):
    values = [torch.arange(120, dtype=torch.float64).reshape(4,5,6)[:,::2,:].requires_grad_(),
              torch.tensor([1.,-0.,float('inf')], dtype=torch.float64),
              torch.tensor([0x7ff8000000000015], dtype=torch.int64).view(torch.float64),
              torch.empty((2,0), dtype=torch.float64),
              torch.tensor(3., dtype=torch.float64).expand(7,8)]
    if mixed:
        values += [torch.tensor([1+2j,-3+4j], dtype=torch.complex128).conj(),
                   torch.tensor([2**60+3], dtype=torch.int64), torch.tensor([True,False])]
    expected, layout = pack_tensors(values)
    assert TensorLayout.from_tensors(values) == layout
    storage = torch.empty_like(expected)
    def disallow(*args, **kwargs):raise AssertionError('Intermediate contiguous packet allocation')
    monkeypatch.setattr(torch, 'cat', disallow)
    monkeypatch.setattr(torch.Tensor, 'contiguous', disallow)
    packet, actual_layout = pack_tensors(values, out=storage)
    assert packet is storage and not packet.requires_grad and packet.grad_fn is None
    assert actual_layout == layout
    assert torch.equal(packet.view(torch.uint8), expected.view(torch.uint8))


def test_packet_destination_rejects_alias_grad_and_strided_storage():
    values = (torch.arange(8, dtype=torch.float64),)
    with pytest.raises(ValueError, match='share storage'):
        pack_tensors(values, out=values[0])
    with pytest.raises(ValueError, match='detached'):
        pack_tensors(values, out=torch.empty(8, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='layout'):
        pack_tensors(values, out=torch.empty(16, dtype=torch.float64)[::2])
    with pytest.raises(ValueError, match='layout'):
        pack_tensors(values, out=torch.empty(7, dtype=torch.float64))


def test_packet_slots_reuse_input_and_output_allocations_without_aliasing_inputs():
    from torchfdtd.tile_workspace import TileWorkspace
    workspace = TileWorkspace('cpu')
    values = (torch.arange(60, dtype=torch.float64).reshape(5,4,3)[:,::2],
              torch.arange(20, dtype=torch.float64).to(torch.complex128).conj())
    pointers = None
    for scale in (1,2,3):
        inputs = tuple(value*scale for value in values)
        packed, layout = workspace.copy_packet('payload', inputs)
        uploaded = layout.unpack(packed)
        returned = workspace.to_host(uploaded).wait()
        for actual, expected in zip(returned, inputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()
        addresses = tuple(value.data_ptr() for value in workspace.buffers.values())
        if pointers is None:pointers = addresses
        else:assert addresses == pointers
    assert workspace.allocations == 2
    assert not workspace.pinned and not workspace.host_staging


def test_output_layout_changes_do_not_flush_input_kernel_bindings():
    from torchfdtd.tile_workspace import TileWorkspace
    workspace = TileWorkspace('cpu')
    value = workspace.array('field', (100,), torch.complex128)
    value.fill_(1+2j)
    cached = workspace.cuda_arguments('forward', [value], lambda t:t.detach())
    for values in ((value, value), (value, value.real.clone()), (value[:3],)):
        result = workspace.to_host(values).wait()
        for actual, expected in zip(result, values):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert workspace.cuda_arguments('forward', [value], lambda t:t.detach()) is cached
    # Byte storage does not change dtype between complex forward output and
    # mixed complex/real backward output, including shorter boundary slabs.
    assert workspace.allocations == 2
