import pytest
import torch

from torchfdtd import AdjointOptions,DifferentiableSimulation
from torchfdtd.cuda_adjoint import FusedAdjointCUDA
from torchfdtd.differentiable import _System
from test_differentiable import project,gpu


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('periodic,diagonal',[(False,False),(True,True)])
def test_random_physical_and_adjoint_state_transpose(precision,periodic,diagonal):
    gpu()
    p=project('3d',precision=precision,steps=12,periodic=periodic)
    dtype=getattr(torch,precision)
    generator=torch.Generator(device='cuda').manual_seed(82)
    eps=1.4+torch.rand(p.region.shape+((3,) if diagonal else ()),device='cuda',dtype=dtype,generator=generator)
    system=_System(p,eps)
    for state in system.state():state.copy_(torch.randn(state.shape,device='cuda',dtype=dtype,generator=generator))
    adjoint=tuple(torch.randn(x.shape,device='cuda',dtype=dtype,generator=generator) for x in system.state())
    signals=torch.randn((len(system.monitors),3),device='cuda',dtype=dtype,generator=generator).T
    gradient=torch.zeros_like(eps)
    fused=FusedAdjointCUDA(system,gradient,signals)
    fused.e_bar.copy_(adjoint[0]);fused.h_bar.copy_(adjoint[1])
    for target,value in zip(fused.psi_bars[0],adjoint[2:]):target.copy_(value)
    expected=tuple(x.clone() for x in adjoint)
    total=torch.zeros_like(eps)
    tolerance=2e-6 if precision=='float32' else 2e-13
    for step in (2,1,0):
        expected,part=system.transpose_step(system.state(),expected,signals[step])
        total.add_(part)
        fused.step(step)
        actual=(fused.e_bar,fused.h_bar,*fused.psi_bars[fused.phase])
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
        torch.testing.assert_close(gradient,total,rtol=tolerance,atol=tolerance)


def test_fused_rejects_cpu_and_unknown_selection():
    with pytest.raises(ValueError,match='backward_kernel'):AdjointOptions(backward_kernel='invalid')
    p=project()
    with pytest.raises(ValueError,match='CUDA'):
        DifferentiableSimulation(p,AdjointOptions(backward_kernel='fused'))(torch.ones(p.region.shape,dtype=torch.float64))


@pytest.mark.parametrize('precision',['float32','float64'])
def test_dense_observer_code_is_bounded_and_duplicate_additions_are_ordered(monkeypatch,precision):
    import numpy as np
    import torchfdtd.cuda_adjoint as module
    from torchfdtd.tile_workspace import TileWorkspace
    gpu()
    p=project('3d',precision=precision,steps=12)
    dtype=getattr(torch,precision)
    epsilon=torch.full(p.region.shape,1.7,device='cuda',dtype=dtype)
    codes=[]
    compile_original=module._compile
    def capture(code,*args):
        if args[-1]=='add_observations':codes.append(code)
        return compile_original(code,*args)
    monkeypatch.setattr(module,'_compile',capture)
    generator=torch.Generator(device='cuda').manual_seed(719)
    for count in (7,75600):
        observations=[]
        for j in range(count):
            family='H' if j%2 else 'E'
            component=j%3
            # Deliberate repeated field locations with different seed values.
            loc=tuple(int(v) for v in np.unravel_index((j//6)%91,p.region.shape))
            observations.append((family+'xyz'[component],loc,component))
        system=_System(p,epsilon,observation_monitors=observations)
        seed=torch.randn((2,count),dtype=dtype,device='cuda',generator=generator)
        workspace=TileWorkspace('cuda',asynchronous=count>7)
        fused=FusedAdjointCUDA(system,torch.zeros_like(epsilon),seed,
            buffers=workspace,direct_views=True)
        for field in (fused.e_bar,fused.h_bar):
            field.copy_(torch.randn(field.shape,dtype=dtype,device='cuda',generator=generator))
        expected=[field.cpu().numpy().copy() for field in (fused.e_bar,fused.h_bar)]
        seeds=seed[1].cpu().numpy()
        for j,(name,loc,component) in enumerate(observations):
            expected[int(name[0]=='H')][(*loc,component)]+=seeds[j]
        fn,arrays,_,groups=fused.observer
        with fused.cp.cuda.Device(fused.device),fused.stream():
            fn(((groups+127)//128,),(128,),(*arrays,np.int32(1)))
        for actual,want in zip((fused.e_bar,fused.h_bar),expected):
            torch.testing.assert_close(actual.cpu(),torch.from_numpy(want),rtol=0,atol=0)
        assert fused.observer_layout.numel()*8<=32*count+8
        workspace.drain()
    assert codes[0]==codes[1] and len(codes[1])<1500
