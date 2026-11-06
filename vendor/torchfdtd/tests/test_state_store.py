import gc

import pytest
import torch

from torchfdtd.state_store import StateStore
from torchfdtd import StreamedAdjointOptions,StreamedSimulation,DifferentiableSimulation
from test_differentiable import project,gpu


def test_first_adjoint_write_skips_zero_reads_and_preserves_overlap(tmp_path):
    template=torch.zeros(8,2,dtype=torch.complex128)
    expected=torch.zeros_like(template)
    with StateStore(tmp_path,1024) as store:
        array,=store.new_state([template])
        first=torch.tensor([2,3,4])
        values=torch.full((3,2),1+2j,dtype=template.dtype)
        array.index_add_(0,first,values);expected.index_add_(0,first,values)
        assert store.read_bytes==0
        # Existing prefix, new suffix and a repeated row in the same request.
        second=torch.tensor([3,4,5,6,3])
        values=torch.full((5,2),3-1j,dtype=template.dtype).conj()
        array.index_add_(0,second,values);expected.index_add_(0,second,values)
        assert store.read_bytes==3*array.row_bytes
        torch.testing.assert_close(array[:],expected,rtol=0,atol=0)


def test_reserved_disk_headroom_rechecked_before_each_bank(tmp_path,monkeypatch):
    import torchfdtd.state_store as storage
    template=torch.zeros(8,dtype=torch.float64)
    remaining=[1024]
    monkeypatch.setattr(storage,'disk_free',lambda _:remaining[0])
    with StateStore(tmp_path,512,free_reserve_bytes=256) as store:
        first,=store.new_state([template])
        remaining[0]=319  # Another process consumed space after admission.
        with pytest.raises(OSError,match='headroom'):store.new_state([template])
        assert store.created_banks==1 and store.live_bytes==64
        assert first.bank.path.exists()
        assert store.report()['free_reserve_bytes']==256
    assert not list(tmp_path.iterdir())


def test_streamed_headroom_rejected_before_directory_creation(tmp_path,monkeypatch):
    import torchfdtd.state_store as storage
    monkeypatch.setattr(storage,'disk_free',lambda _:1024)
    p=project(steps=10)
    directory=tmp_path/'scratch'
    options=StreamedAdjointOptions(device='cpu',state_storage='disk',state_directory=directory,
        disk_budget_bytes=10**9,disk_free_reserve_bytes=1024)
    with pytest.raises(ValueError,match='disk budget'):
        StreamedSimulation(p,options)(torch.ones(p.region.shape,dtype=torch.float64))
    assert not directory.exists()


@pytest.mark.parametrize('dtype',[torch.float64,torch.complex64,torch.complex128])
def test_file_slabs_preserve_duplicate_reductions_and_release_banks(tmp_path,dtype):
    sentinel=tmp_path/'user.txt';sentinel.write_text('keep')
    template=torch.zeros((),dtype=dtype).expand(7,3,2)
    size=template.numel()*template.element_size()
    with StateStore(tmp_path,2*size) as store:
        array,=store.new_state([template])
        expected=torch.zeros_like(template)
        indices=torch.tensor([5,6,0,1,0,1,2])
        values=torch.arange(42,dtype=torch.float64).reshape(7,3,2).to(dtype)
        if values.is_complex():values=(values+1j*(values+1)).conj()
        array.index_add_(0,indices,values);expected.index_add_(0,indices,values)
        torch.testing.assert_close(array[:],expected,rtol=0,atol=0)
        torch.testing.assert_close(array.index_select(0,indices),expected.index_select(0,indices),rtol=0,atol=0)
        empty=torch.empty(0,dtype=torch.int64)
        assert array.index_select(0,empty).shape == (0,3,2)
        array.index_copy_(0,empty,torch.empty((0,3,2),dtype=dtype))
        array.index_copy_(0,torch.arange(7),values)
        torch.testing.assert_close(array[:],values,rtol=0,atol=0)
        second,=store.new_state([template])
        with pytest.raises(MemoryError,match='disk budget'):store.new_state([template])
        del second;gc.collect()
        assert store.live_bytes == size
        assert store.max_read_bytes <= size
    assert store.live_bytes == 0 and store.report()['closed']
    with pytest.raises(RuntimeError,match='closed'):array[:]
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('periodic,diagonal',[(False,False),(True,True)])
def test_disk_streamed_gradient_matches_full_autograd_and_cleans_up(tmp_path,device,periodic,diagonal):
    if device=='cuda':gpu()
    p=project('3d',steps=13,periodic=periodic)
    epsilon=torch.full(p.region.shape+((3,) if diagonal else ()),1.7,dtype=torch.float64,requires_grad=True)
    reference=DifferentiableSimulation(p).reference(epsilon)
    wanted,=torch.autograd.grad(reference.square().sum(),epsilon)
    options=StreamedAdjointOptions(device=device,slab_width=4,temporal_depth=3,checkpoints=2,local_checkpoints=1,
                                  state_storage='disk',state_directory=tmp_path,disk_budget_bytes=32*1024**2,
                                  tile_transfers='async' if device=='cuda' else 'sync')
    result=StreamedSimulation(p,options)(epsilon)
    assert not list(tmp_path.iterdir())
    actual,=torch.autograd.grad(result.signals.square().sum(),epsilon,retain_graph=True)
    torch.testing.assert_close(result.signals,reference,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(actual,wanted,rtol=2e-10,atol=1e-12)
    again,=torch.autograd.grad(result.signals.square().sum(),epsilon)
    torch.testing.assert_close(again,actual,rtol=0,atol=0)
    assert not list(tmp_path.iterdir())
    for phase in ('forward','backward'):
        record=result.report[phase+'_backing_store']
        assert record['closed'] and record['live_logical_file_bytes']==0
        assert 0 < record['peak_logical_file_bytes'] <= result.report['disk_reservation_bytes']
        assert record['logical_read_bytes']>0 and record['logical_written_bytes']>0


def test_disk_budget_rejected_before_directory_creation(tmp_path):
    p=project(steps=10)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64)
    directory=tmp_path/'scratch'
    options=StreamedAdjointOptions(device='cpu',state_storage='disk',state_directory=directory,disk_budget_bytes=1)
    with pytest.raises(ValueError,match='disk budget'):StreamedSimulation(p,options)(epsilon)
    assert not directory.exists()


def test_failed_tile_read_cleans_scratch_without_touching_user_files(tmp_path,monkeypatch):
    from torchfdtd.state_store import DiskArray
    p=project(steps=10)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64,requires_grad=True)
    sentinel=tmp_path/'keep.bin';sentinel.write_bytes(b'user data')
    def fail(*args):raise OSError('injected read failure')
    monkeypatch.setattr(DiskArray,'index_select',fail)
    options=StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2,state_storage='disk',
                                  state_directory=tmp_path,disk_budget_bytes=8*1024**2)
    with pytest.raises(OSError,match='injected'):StreamedSimulation(p,options)(epsilon)
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('periodic',[False,True])
def test_disk_banks_preserve_random_cpml_endpoint_and_periodic_halo_derivatives(tmp_path,device,periodic):
    if device=='cuda':gpu()
    from torchfdtd.differentiable import _System
    from torchfdtd.spacetime import SlabBlockOperator
    p=project('3d',steps=12,periodic=periodic)
    epsilon=torch.full(p.region.shape+(3,),1.7,dtype=torch.float64)
    host=_System(p,epsilon,prepare_updates=False)
    torch.manual_seed(705)
    initial=tuple(torch.randn_like(s)*.01 for s in host.state())
    endpoint=tuple(torch.randn_like(s)*.02 for s in host.state())
    weights=torch.randn(9,len(host.monitors),dtype=epsilon.dtype)
    reference=SlabBlockOperator(host,5,device,local_checkpoints=1)
    expected,signal=reference.forward(epsilon,initial,1,9)
    wanted,gradient=reference.transpose(epsilon,initial,1,9,endpoint,weights)
    with StateStore(tmp_path,32*1024**2) as store:
        stored_initial=store.new_state(host.state());stored_endpoint=store.new_state(host.state())
        for stored,values in ((stored_initial,initial),(stored_endpoint,endpoint)):
            for target,value in zip(stored,values):target.index_copy_(0,torch.arange(value.shape[0]),value)
        operation=SlabBlockOperator(host,5,device,local_checkpoints=1,state_factory=store.new_state)
        actual,observed=operation.forward(epsilon,stored_initial,1,9)
        bars,got_gradient=operation.transpose(epsilon,stored_initial,1,9,stored_endpoint,weights)
        for actual,want in zip(actual,expected):torch.testing.assert_close(actual[:],want,rtol=0,atol=0)
        for actual,want in zip(bars,wanted):torch.testing.assert_close(actual[:],want,rtol=0,atol=0)
        torch.testing.assert_close(observed,signal,rtol=0,atol=0)
        torch.testing.assert_close(got_gradient,gradient,rtol=0,atol=0)
    assert not list(tmp_path.iterdir())


def test_failed_backward_reduction_cleans_scratch(tmp_path,monkeypatch):
    from torchfdtd.state_store import DiskArray
    p=project(steps=10)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64,requires_grad=True)
    options=StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2,state_storage='disk',
                                  state_directory=tmp_path,disk_budget_bytes=8*1024**2)
    result=StreamedSimulation(p,options)(epsilon)
    def fail(*args):raise OSError('injected reduction failure')
    monkeypatch.setattr(DiskArray,'index_add_',fail)
    with pytest.raises(OSError,match='injected'):result.signals.square().sum().backward()
    assert not list(tmp_path.iterdir())


def test_partial_io_and_truncated_file_are_handled(tmp_path):
    template=torch.zeros((5,3),dtype=torch.float64)
    class PartialIO:
        def __init__(self,file):self.file=file
        def seek(self,*args):return self.file.seek(*args)
        def readinto(self,value):return self.file.readinto(value[:7])
        def write(self,value):return self.file.write(value[:7])
        def close(self):return self.file.close()
    with StateStore(tmp_path,1024) as store:
        array,=store.new_state([template])
        file=array.bank.file
        array.bank.file=PartialIO(file)
        values=torch.arange(15,dtype=torch.float64).reshape(5,3)
        array.index_copy_(0,torch.arange(5),values)
        torch.testing.assert_close(array[:],values,rtol=0,atol=0)
        file.truncate(3)
        with pytest.raises(OSError,match='Truncated'):array[:]
    assert not list(tmp_path.iterdir())


def test_file_policy_admission_releases_dense_host_banks(tmp_path):
    from dataclasses import replace
    from torchfdtd.streamed import _reservation
    p=project(steps=10)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64)
    host=StreamedAdjointOptions(device='cpu',slab_width=1,temporal_depth=1)
    disk=replace(host,state_storage='disk',state_directory=tmp_path,disk_budget_bytes=8*1024**2)
    reservation=_reservation(p,epsilon,disk)
    tight=reservation['host_reservation_bytes']
    with pytest.raises(ValueError,match='host budget'):_reservation(p,epsilon,replace(host,host_budget_bytes=tight))
    assert _reservation(p,epsilon,replace(disk,host_budget_bytes=tight)) == reservation
    assert not list(tmp_path.iterdir())


def test_tuning_can_compare_host_and_disk_policies(tmp_path):
    from dataclasses import replace
    from torchfdtd import tune_streamed
    p=project(steps=10)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64,requires_grad=True)
    host=StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=3)
    disk=replace(host,state_storage='disk',state_directory=tmp_path,disk_budget_bytes=8*1024**2)
    tuned=tune_streamed(p,epsilon,candidates=[host,disk],probe_steps=10,repeats=1)
    assert all(row['status']=='measured' for row in tuned.report['candidates'])
    assert epsilon.grad is None and not list(tmp_path.iterdir())


def test_unavailable_volume_is_rejected_and_io_is_always_cpu(tmp_path,monkeypatch):
    from pathlib import Path
    from torchfdtd.state_store import disk_free
    with monkeypatch.context() as patch:
        patch.setattr(Path,'exists',lambda self:False)
        with pytest.raises(ValueError,match='unavailable'):disk_free(tmp_path)
    template=torch.zeros((3,2),dtype=torch.float64)
    values=torch.ones_like(template)
    indices=torch.arange(3)
    with StateStore(tmp_path,1024) as store,torch.device('meta'):
        array,=store.new_state([template])
        array.index_add_(0,indices,values)
        result=array[:]
        assert result.device.type=='cpu'
        torch.testing.assert_close(result,values,rtol=0,atol=0)
