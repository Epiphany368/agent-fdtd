"""Exercise actual file-bank lifetimes through global replay recursion."""
import gc
import pytest
import torch
from torchfdtd import StreamedSimulation, StreamedAdjointOptions
from test_bloch_adjoint import scene


@pytest.mark.parametrize('steps,depth', [(10,1),(13,3),(29,4),(41,7)])
@pytest.mark.parametrize('checkpoints', [0,1,2,4])
def test_file_bank_bound_without_cyclic_gc(tmp_path, steps, depth, checkpoints):
    p=scene()
    p.region.steps=steps
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    options=StreamedAdjointOptions(device='cpu',state_storage='disk',state_directory=tmp_path,
        disk_budget_bytes=32*1024**2,slab_width=8,temporal_depth=depth,checkpoints=checkpoints,
        local_checkpoints=1)
    was_enabled=gc.isenabled()
    gc.disable()
    try:
        result=StreamedSimulation(p,options)(eps)
        for _ in range(2):
            gradient,=torch.autograd.grad(result.signals.abs().square().sum(),eps,retain_graph=True)
            assert torch.isfinite(gradient).all() and gradient.norm()>0
            for phase,bound in [('forward',2),('backward',checkpoints+3)]:
                store=result.report[phase+'_backing_store']
                assert store['peak_logical_file_bytes'] <= bound*result.report['state_bytes']
                assert store['closed'] and store['live_logical_file_bytes']==0
            assert not list(tmp_path.iterdir())
        assert result.report['disk_reservation_bytes']==(checkpoints+3)*result.report['state_bytes']
    finally:
        if was_enabled:gc.enable()
