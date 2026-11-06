import time
import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions,DifferentiableSimulation
from torchfdtd.staging import AsyncStateStaging
from test_differentiable import project,gpu


@pytest.mark.parametrize('storage',['host','disk','hierarchical'])
@pytest.mark.parametrize('slots',[1,2])
def test_async_tiers_match_sync_on_nondefault_stream(tmp_path,storage,slots):
    gpu()
    p=project('3d',steps=37,periodic=True)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,device='cuda',requires_grad=True)
    base=dict(checkpoints=3,storage=storage,checkpoint_directory=tmp_path,disk_budget_bytes=64*1024**2,
              host_budget_bytes=64*1024**2,device_checkpoints=1 if storage=='hierarchical' else 0,
              host_checkpoints=1 if storage=='hierarchical' else 0)
    expected=DifferentiableSimulation(p,AdjointOptions(**base))(eps)
    want,=torch.autograd.grad(expected.signals.square().sum(),eps)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        independent_eps=eps.detach().clone().requires_grad_()
        actual=DifferentiableSimulation(p,AdjointOptions(**base,checkpoint_transfers='async',staging_slots=slots))(independent_eps)
        got,=torch.autograd.grad(actual.signals.square().sum(),independent_eps)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(got,want,rtol=0,atol=0)
    assert actual.report['async_saves']>0 and actual.report['async_loads']>0
    assert actual.report['peak_staging_leases']<=slots
    assert list(tmp_path.iterdir())==[]


def test_delayed_disk_failure_propagates_and_cleans(tmp_path,monkeypatch):
    gpu()
    p=project(steps=20)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,device='cuda',requires_grad=True)
    options=AdjointOptions(checkpoints=2,storage='disk',checkpoint_directory=tmp_path,
                           disk_budget_bytes=32*1024**2,checkpoint_transfers='async')
    result=DifferentiableSimulation(p,options)(eps)
    def fail(path,**arrays):
        time.sleep(.01)
        path.write_bytes(b'partial')
        raise OSError('delayed disk failure')
    monkeypatch.setattr(np,'savez',fail)
    with pytest.raises(OSError,match='delayed disk failure'):result.signals.sum().backward()
    assert list(tmp_path.iterdir())==[]


def test_staging_snapshot_is_not_live_state_and_completed_futures_are_released():
    gpu()
    state=(torch.arange(10000,device='cuda',dtype=torch.float32),)
    report={};pool=AsyncStateStaging(state,1,report)
    try:
        future=pool.save(state)
        state[0].fill_(-1)
        value=future.result()
        torch.testing.assert_close(value[0],torch.arange(10000,dtype=torch.float32),rtol=0,atol=0)
        pool.forget(future)
        assert all(slot.future is None for slot in pool.slots)
        ticket=pool.load(value)
        pool.consume(ticket,state)
        torch.testing.assert_close(state[0].cpu(),value[0],rtol=0,atol=0)
        with pytest.raises(RuntimeError,match='once'):pool.consume(ticket,state)
    finally:pool.close()
    assert not pool.tasks
