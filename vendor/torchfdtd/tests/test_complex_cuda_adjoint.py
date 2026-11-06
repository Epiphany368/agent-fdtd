import pytest
import torch
from torchfdtd import DifferentiableSimulation,AdjointOptions
from test_bloch_adjoint import scene
from test_differentiable import gpu


@pytest.mark.parametrize('nonuniform',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fused_complex_backward_against_full_autograd(dtype,nonuniform):
    gpu();p=scene(nonuniform)
    p.region.precision='float64' if dtype==torch.float64 else 'float32'
    p.region.cuda_kernel='fused'
    eps=torch.full(p.region.shape+(3,),1.7,device='cuda',dtype=dtype,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,backward_kernel='fused'))
    def loss(x):return (x.real+.4*x.imag).square().sum()+x.abs().square().mean()
    oracle=model.reference(eps);expected,=torch.autograd.grad(loss(oracle),eps)
    result=model(eps);actual,=torch.autograd.grad(loss(result.signals),eps)
    tol=3e-5 if dtype==torch.float32 else 2e-10
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol*.01)
    assert actual.norm()>1e-5
    assert result.report['backward_backend']=='fused CUDA complex transpose'
    frequencies=[.025/p.region.time_step,.07/p.region.time_step]
    history=model(eps).spectrum(frequencies)/p.region.time_step
    hg,=torch.autograd.grad(loss(history),eps)
    online=model.spectrum(eps,frequencies,block_size=5)
    og,=torch.autograd.grad(loss(online.fields/p.region.time_step),eps)
    torch.testing.assert_close(og,hg,rtol=tol,atol=tol*.01)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_random_state_and_cpml_adjoint_buffers(dtype):
    from torchfdtd.differentiable import _System
    from torchfdtd.cuda_complex_adjoint import FusedComplexAdjointCUDA
    gpu();p=scene(True);p.region.precision='float64' if dtype==torch.float64 else 'float32'
    eps=torch.full(p.region.shape,1.7,device='cuda',dtype=dtype)
    system=_System(p,eps)
    generator=torch.Generator(device='cuda').manual_seed(174)
    with torch.no_grad():
        for value in system.state():value.copy_(torch.randn(value.shape,device='cuda',dtype=value.dtype,generator=generator)*.01)
        bars=tuple(torch.randn(v.shape,device='cuda',dtype=v.dtype,generator=generator) for v in system.state())
        seed=torch.randn((1,len(system.monitors)),device='cuda',dtype=system.field_dtype,generator=generator)
        gradient=torch.zeros_like(eps)
        fused=FusedComplexAdjointCUDA(system,gradient,seed)
        for target,value in zip((fused.e_bar,fused.h_bar,*fused.psi_bars[0]),bars):target.copy_(value)
        expected,eg=system.transpose_step(system.state(),tuple(v.clone() for v in bars),seed[0])
        fused.step(0)
        actual=(fused.e_bar,fused.h_bar,*fused.psi_bars[fused.phase])
        tol=3e-5 if dtype==torch.float32 else 2e-12
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tol,atol=tol*.01)
        torch.testing.assert_close(gradient,eg,rtol=tol,atol=tol*.01)


@pytest.mark.parametrize('pml_axis',['x','z'])
def test_three_dimensional_seams_and_async_disk_replay(pml_axis,tmp_path):
    from torchfdtd import Source,Monitor
    gpu();p=scene();p.region.dimension='3d';p.region.cuda_kernel='fused'
    for axis in 'xyz':
        for side in ('min','max'):getattr(p.region.boundaries,axis+'_'+side).kind='pml' if axis==pml_axis else 'bloch'
    p.region.bloch_phase=tuple(0 if axis==pml_axis else (-.47 if axis=='x' else .82) for axis in 'xyz')
    center=[0.,0.,0.];center['xyz'.index(pml_axis)]=-.3
    size=list(p.region.size);size['xyz'.index(pml_axis)]=0
    component='Ey' if pml_axis=='x' else 'Ex'
    p.sources=[Source(kind='plane',normal=pml_axis,size=tuple(size),center=tuple(center),component=component,pulse='continuous')]
    p.monitors=[Monitor(component=component,center=(.1,.1,.1)),Monitor(component='Hz',center=(.2,.1,.1))]
    eps=torch.full(p.region.shape+(3,),1.7,device='cuda',dtype=torch.float64,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,backward_kernel='fused',storage='disk',
        checkpoint_directory=tmp_path,disk_budget_bytes=128*1024**2,host_budget_bytes=128*1024**2,checkpoint_transfers='async'))
    oracle=model.reference(eps);eg,=torch.autograd.grad(oracle.abs().square().sum(),eps)
    actual=model(eps);ag,=torch.autograd.grad(actual.signals.abs().square().sum(),eps)
    torch.testing.assert_close(ag,eg,rtol=2e-10,atol=1e-11)
    assert ag.norm()>1e-5
    assert not list(tmp_path.iterdir())


def test_longer_float32_replay_with_torch_forward():
    gpu();p=scene();p.region.dimension='3d';p.region.steps=96;p.region.precision='float32'
    # The backward choice is independent of the forward choice.
    p.region.cuda_kernel='torch'
    eps=torch.full(p.region.shape,1.7,device='cuda',dtype=torch.float32,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=3,backward_kernel='fused'))
    def loss(v):return v.real.square().sum()+.7*v.imag.square().sum()
    oracle=model.reference(eps);eg,=torch.autograd.grad(loss(oracle),eps)
    actual=model(eps);ag,=torch.autograd.grad(loss(actual.signals),eps)
    torch.testing.assert_close(ag,eg,rtol=3e-5,atol=3e-7)
    assert ag.norm()>1e-5
    assert actual.report['forward_backend']=='torch CUDA'
