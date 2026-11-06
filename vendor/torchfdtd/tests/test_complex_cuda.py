import pytest
import torch
from torchfdtd import DifferentiableSimulation,AdjointOptions
from torchfdtd.differentiable import _System
from test_bloch_adjoint import scene
from test_differentiable import gpu


@pytest.mark.parametrize('nonuniform',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_complex_cuda_state_cpml_and_density_vjp(dtype,nonuniform):
    gpu()
    p=scene(nonuniform);p.region.precision='float64' if dtype==torch.float64 else 'float32'
    eps=torch.full(p.region.shape+(3,),1.7,device='cuda',dtype=dtype,requires_grad=True)
    with torch.no_grad():
        oracle=_System(p,eps)
        q=p.model_copy(deep=True);q.region.cuda_kernel='fused'
        candidate=_System(q,eps)
        # Exercise all field components and CPML recurrence with nonzero data.
        generator=torch.Generator(device='cuda').manual_seed(173)
        for a,b in zip(oracle.state(),candidate.state()):
            values=torch.randn(a.shape,device=a.device,dtype=a.dtype,generator=generator)*.01
            a.copy_(values);b.copy_(values)
        for step in range(11):
            oracle.advance(step,step+1);candidate.advance(step,step+1)
        tolerance=3e-6 if dtype==torch.float32 else 1e-12
        for a,b in zip(oracle.state(),candidate.state()):
            torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance*.01)
    reference=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    fused=DifferentiableSimulation(q,AdjointOptions(checkpoints=2))
    def loss(x):return (x.real+.4*x.imag).square().sum()+x.abs().square().mean()
    expected=reference(eps);eg,=torch.autograd.grad(loss(expected.signals),eps)
    actual=fused(eps);ag,=torch.autograd.grad(loss(actual.signals),eps)
    torch.testing.assert_close(actual.signals,expected.signals,rtol=tolerance,atol=tolerance*.01)
    torch.testing.assert_close(ag,eg,rtol=10*tolerance,atol=tolerance*.01)
    assert actual.report['forward_backend']=='fused CUDA'
    assert actual.report['backward_backend']=='torch explicit transpose'
    assert ag.norm()>1e-5


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_three_dimensional_negative_seams_and_disk_replay(dtype,tmp_path):
    from torchfdtd import Source,Monitor
    gpu()
    p=scene();p.region.dimension='3d'
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
    p.region.bloch_phase=(-.47,.82,0)
    p.region.precision='float64' if dtype==torch.float64 else 'float32'
    p.sources=[Source(kind='plane',normal='z',size=(1.6,1.5,0),center=(0,0,-.3),component='Ex',pulse='continuous')]
    p.monitors=[Monitor(component='Ex',center=(.4,-.4,.1)),Monitor(component='Hy',center=(-.4,.4,.1))]
    q=p.model_copy(deep=True);q.region.cuda_kernel='fused'
    eps=torch.full(p.region.shape+(3,),1.7,device='cuda',dtype=dtype,requires_grad=True)
    expected=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))(eps)
    eg,=torch.autograd.grad(expected.signals.abs().square().sum(),eps)
    actual=DifferentiableSimulation(q,AdjointOptions(checkpoints=2,storage='disk',checkpoint_directory=tmp_path,
        disk_budget_bytes=32*1024**2,host_budget_bytes=32*1024**2,checkpoint_transfers='async'))(eps)
    ag,=torch.autograd.grad(actual.signals.abs().square().sum(),eps)
    tol=3e-6 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(actual.signals,expected.signals,rtol=tol,atol=tol*.01)
    torch.testing.assert_close(ag,eg,rtol=10*tol,atol=tol*.01)
    assert ag.norm()>1e-5
    assert not list(tmp_path.iterdir())
