import gc
import weakref
import pytest
import torch
from torchfdtd import DifferentiableSimulation,AdjointOptions
from test_bloch_adjoint import scene


@pytest.mark.parametrize('backend',['cpu',pytest.param('torch',marks=pytest.mark.cuda),pytest.param('fused',marks=pytest.mark.cuda)])
@pytest.mark.parametrize('complex_fields',[False,True])
def test_completed_adjoint_does_not_retain_system_until_gc(backend,complex_fields):
    device='cpu' if backend=='cpu' else 'cuda'
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_differentiable import project
    p=scene() if complex_fields else project(steps=14)
    if backend=='fused':p.region.cuda_kernel='fused'
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,backward_kernel='fused' if backend=='fused' else 'torch'))
    eps=torch.full(p.region.shape,1.7,device=device,dtype=torch.float64,requires_grad=True)
    def iteration():
        result=model(eps)
        reference=weakref.ref(result.signals.grad_fn.system)
        torch.autograd.grad(result.signals.abs().square().sum(),eps)
        return reference
    enabled=gc.isenabled();gc.disable()
    try:
        reference=iteration()
        assert reference() is None, 'Completed solver remains alive without cyclic garbage collection.'
    finally:
        if enabled:gc.enable()
        gc.collect()


@pytest.mark.parametrize('storage',['host','disk'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_streamed_replay_releases_operators_and_tiles_without_gc(storage,device,tmp_path,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import torchfdtd.streamed as streamed
    import torchfdtd.spacetime as spacetime
    from torchfdtd import StreamedSimulation,StreamedAdjointOptions
    from test_differentiable import project
    references=[]
    class Operator(streamed.SlabBlockOperator):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);references.append(weakref.ref(self))
    class System(spacetime._System):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);references.append(weakref.ref(self))
    monkeypatch.setattr(streamed,'SlabBlockOperator',Operator)
    monkeypatch.setattr(spacetime,'_System',System)
    p=project(steps=10)
    model=StreamedSimulation(p,StreamedAdjointOptions(device=device,slab_width=6,temporal_depth=2,tile_transfers='async' if device=='cuda' else 'sync',
        checkpoints=2,local_checkpoints=1,state_storage=storage,state_directory=tmp_path,disk_budget_bytes=64*1024**2))
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    def iteration():
        result=model(eps)
        torch.autograd.grad(result.signals.square().sum(),eps)
    enabled=gc.isenabled();gc.disable()
    try:
        iteration()
        assert references and all(ref() is None for ref in references)
        assert not list(tmp_path.iterdir())
    finally:
        if enabled:gc.enable()
        gc.collect()
