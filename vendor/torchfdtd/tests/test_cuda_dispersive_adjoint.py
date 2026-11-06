"""Native ADE values and VJPs against the independent full-time Torch graph."""
import gc
import math
import weakref

import pytest
import torch

from torchfdtd import AdjointOptions, BoundaryFace, DispersiveSimulation
from torchfdtd.dispersive_adjoint import _DispersiveSystem
from torchfdtd.cuda_dispersive_adjoint import FusedDispersiveAdjointCUDA
from test_differentiable import gpu, project


def inputs(p, layout, dtype):
    shape = p.region.shape
    generator = torch.Generator(device='cuda').manual_seed(621)
    def variable(s):
        return (.3+.5*torch.rand(s,device='cuda',dtype=dtype,generator=generator)).requires_grad_()
    eps = variable(shape+((3,) if layout=='diagonal' else ()))
    sizes = {
        'shared': [(2,), (), (2,)],
        'strength': [(2,*shape), (2,), ()],
        'rates': [(2,), (2,*shape), (2,*shape,3)],
        'diagonal': [(2,*shape,3), (2,*shape,3), ()],
        'fully_spatial': [(2,*shape,3), (2,*shape,3), (2,*shape,3)],
    }[layout]
    variables = (eps, *(variable(s) for s in sizes))
    return variables, (1+eps, variables[1]*1e30, variables[2]*1e15, variables[3]*1e15)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('bloch', [False,True])
@pytest.mark.parametrize('layout', ['shared','strength','rates','diagonal','fully_spatial'])
def test_native_ade_forward_and_parameter_vjps(dtype,bloch,layout):
    gpu()
    p = project(dimension='3d' if layout=='rates' else '2d',steps=13,
                precision='float64' if dtype==torch.float64 else 'float32')
    if bloch:
        p.region.boundaries.x_min=BoundaryFace(kind='bloch')
        p.region.boundaries.x_max=BoundaryFace(kind='bloch')
        p.region.bloch_phase=(.61,0,0)
    p.region.cuda_kernel='fused'
    variables,args = inputs(p,layout,dtype)
    model = DispersiveSimulation(p,AdjointOptions(checkpoints=2,backward_kernel='fused'))
    reference = model.reference(*args)
    expected = torch.autograd.grad(reference.abs().square().sum(),variables,retain_graph=True)
    result = model(*args)
    actual = torch.autograd.grad(result.signals.abs().square().sum(),variables)
    tolerance=dict(rtol=2e-4,atol=3e-6) if dtype==torch.float32 else dict(rtol=2e-10,atol=2e-12)
    torch.testing.assert_close(result.signals,reference,**tolerance)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,**tolerance)
    assert result.report['forward_backend']=='fused CUDA ADE'
    assert result.report['backward_backend']=='fused CUDA ADE transpose'
    shared=sum(math.prod(s) for s in (v.shape for v in variables[1:]) if len(s)<=1)
    assert result.report['material_gradient_reduction_bytes']==shared*math.ceil(math.prod(p.region.shape)/256)*dtype.itemsize


@pytest.mark.parametrize('bloch', [False,True])
def test_complete_one_step_state_transpose_and_immutable_replay(bloch):
    gpu()
    p=project(steps=10)
    if bloch:
        p.region.boundaries.x_min=BoundaryFace(kind='bloch')
        p.region.boundaries.x_max=BoundaryFace(kind='bloch')
        p.region.bloch_phase=(-.37,0,0)
    _,args=inputs(p,'shared',torch.float64)
    model=DispersiveSimulation(p)
    parameters,layout=model._pack(*args)
    parameters=parameters.detach().requires_grad_()
    system=_DispersiveSystem(p,args[0],parameters,layout,fused_backward=True)
    torch.manual_seed(883)
    state=tuple((torch.randn_like(t)*.02).requires_grad_() for t in system.state())
    endpoints=tuple(torch.randn_like(t)*.03 for t in state)
    signals=torch.randn((p.region.steps,len(system.monitors)),device='cuda',dtype=system.field_dtype)
    evolved=system.reference_step(state,3,parameters)
    loss=sum((a.conj()*b).real.sum() for a,b in zip(evolved,endpoints))
    loss+=(system.observe(evolved).conj()*signals[3]).real.sum()
    expected=torch.autograd.grad(loss,(*state,parameters))
    with torch.no_grad():
        for target,value in zip(system.state(),state):target.copy_(value)
        gradient=torch.zeros_like(parameters)
        kernel=FusedDispersiveAdjointCUDA(system,gradient,signals)
        bars=(kernel.curl.e_bar,kernel.curl.h_bar,*kernel.curl.psi_bars[0],kernel.p_bar,kernel.q_bar)
        for target,value in zip(bars,endpoints):target.copy_(value)
        kernel.step(3);kernel.finalize({})
        actual=(kernel.curl.e_bar,kernel.curl.h_bar,*kernel.curl.psi_bars[kernel.curl.phase],kernel.p_bar,kernel.q_bar,gradient)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=4e-11,atol=2e-12)
    for a,b in zip(system.state(),state):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('forward', ['torch','fused'])
def test_spectral_replay_mixed_tiers_and_geometry_taylor(tmp_path,forward):
    gpu()
    p=project(steps=19,periodic=True)
    p.region.cuda_kernel=forward
    options=AdjointOptions(checkpoints=3,storage='hierarchical',device_checkpoints=1,host_checkpoints=1,
        checkpoint_directory=tmp_path,host_budget_bytes=32*1024**2,disk_budget_bytes=32*1024**2,
        checkpoint_transfers='async',staging_slots=2,backward_kernel='fused')
    model=DispersiveSimulation(p,options)
    design=torch.tensor([.4,.7,.2],device='cuda',dtype=torch.float64,requires_grad=True)
    mask=torch.linspace(-1,1,math.prod(p.region.shape),device='cuda',dtype=torch.float64).reshape(p.region.shape)
    frequency=[.02/p.region.time_step,.055/p.region.time_step]
    def loss(x):
        density=torch.sigmoid(2*(mask+x[0]))
        result=model.spectrum(1.3+.4*density,torch.stack((density*x[1],.2*density))*1e30,
                              torch.tensor([0.,1.5e15],device='cuda',dtype=x.dtype),x[2]*1e15,
                              frequency,block_size=4)
        return (result.fields.abs().square()/p.region.time_step**2).sum()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        base=loss(design)
        gradient,=torch.autograd.grad(base,design)
        direction=design.new_tensor([.2,-.3,.1])
        errors=[float((loss(design.detach()+h*direction)-base.detach()-h*(gradient@direction)).abs())
                for h in (.004,.002,.001)]
    torch.cuda.current_stream().wait_stream(stream)
    assert all(3.7<a/b<4.3 for a,b in zip(errors,errors[1:]))
    assert not list(tmp_path.iterdir())


def test_native_ade_releases_system_without_cyclic_gc():
    gpu()
    p=project(steps=10);p.region.cuda_kernel='fused'
    model=DispersiveSimulation(p,AdjointOptions(checkpoints=1,backward_kernel='fused'))
    def iteration():
        _,args=inputs(p,'shared',torch.float64)
        result=model(*args)
        reference=weakref.ref(result.signals.grad_fn.system)
        result.signals.square().sum().backward()
        return reference
    enabled=gc.isenabled();gc.disable()
    try:
        reference=iteration()
        assert reference() is None
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('backward', ['torch','fused'])
def test_fused_plane_flux_zero_pole_and_material_finite_difference(backward):
    gpu()
    from torchfdtd import DifferentiablePlaneSimulation, DispersivePlaneSimulation
    from test_adjoint_planes import scene
    p=scene();p.region.steps=23;p.region.cuda_kernel='fused'
    eps=torch.full(p.region.shape,1.7,device='cuda',dtype=torch.float64)
    options=AdjointOptions(checkpoints=2,backward_kernel=backward)
    model=DispersivePlaneSimulation(p,options)
    frequencies=[.025/p.region.time_step,.06/p.region.time_step]
    zero=model(eps,[0.],1.5e15,2e14,frequencies)
    reference=DifferentiablePlaneSimulation(p,options)(eps,frequencies)
    for key in zero:torch.testing.assert_close(zero[key].fields,reference[key].fields,rtol=3e-11,atol=1e-27)
    strength=torch.tensor(.8,device='cuda',dtype=torch.float64,requires_grad=True)
    def loss(s):
        planes=model(eps,s[None]*1e30,1.5e15,2e14,frequencies,block_size=5)
        return sum(plane.normalized_flux(zero[key]).sum() for key,plane in planes.items())
    result=loss(strength)
    gradient,=torch.autograd.grad(result,strength,retain_graph=True)
    repeated,=torch.autograd.grad(result,strength)
    torch.testing.assert_close(gradient,repeated,rtol=0,atol=0)
    h=1e-4
    finite=(loss(strength.detach()+h)-loss(strength.detach()-h))/(2*h)
    torch.testing.assert_close(gradient,finite,rtol=2e-6,atol=1e-8)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_sixty_four_poles_shared_rates_and_drude_limit(dtype):
    gpu()
    p=project(steps=10,precision='float64' if dtype==torch.float64 else 'float32');p.region.cuda_kernel='fused'
    eps=torch.full(p.region.shape,1.6,device='cuda',dtype=dtype,requires_grad=True)
    strengths=torch.linspace(.001,.01,64,device='cuda',dtype=dtype,requires_grad=True)
    gamma=torch.tensor(.2,device='cuda',dtype=dtype,requires_grad=True)
    model=DispersiveSimulation(p,AdjointOptions(checkpoints=1,backward_kernel='fused'))
    args=(eps,strengths*1e30,0.,gamma*1e15)
    ref=model.reference(*args)
    wanted=torch.autograd.grad(ref.square().sum(),(eps,strengths,gamma),retain_graph=True)
    got=model(*args)
    actual=torch.autograd.grad(got.signals.square().sum(),(eps,strengths,gamma))
    tolerance=dict(rtol=2e-4,atol=3e-6) if dtype==torch.float32 else dict(rtol=3e-10,atol=2e-12)
    for a,b in zip(actual,wanted):torch.testing.assert_close(a,b,**tolerance)
