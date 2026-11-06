import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, DifferentiableSolid,
    Monitor, Project, Region, Source, StreamedAdjointOptions, StreamedSimulation,
    smooth_geometry_epsilon)
from torchfdtd.solver import field_axes


def region(precision='float32', sampling='yee'):
    return Region(dimension='3d', size=(1.4, 1.3, 1.2), mesh=.1, pml_cells=3,
                  steps=18, precision=precision, material_sampling=sampling)


def scene(parameters):
    return [
        DifferentiableSolid.box((.43, .31, .29), center=parameters[:3],
            rotation=parameters[3:6], epsilon=parameters[6]),
        DifferentiableSolid.cylinder(parameters[7], parameters[8],
            center=(.16, -.08, .03), rotation=(13., -17., 8.), epsilon=3.1, radius_y=.17),
        DifferentiableSolid.ellipsoid((.18, .13, .11), center=(-.2, .12, .04),
            rotation=(19., 5., -11.), epsilon=2.2),
    ]


def test_independent_rotated_box_membership_yee_locations_and_overwrite():
    r=region()
    solids=[DifferentiableSolid.box((.8,.6,.5),epsilon=2.),
            DifferentiableSolid.box((.21,.41,.31),epsilon=4.,rotation=(0.,0.,90.))]
    result=smooth_geometry_epsilon(r,solids,width=.001,chunk_cells=77)
    expected=torch.ones_like(result)
    for c,component in enumerate(('Ex','Ey','Ez')):
        x,y,z=torch.meshgrid(*(torch.tensor(a,dtype=result.dtype) for a in field_axes(r,component)),indexing='ij')
        expected[...,c][(x.abs()<.4)&(y.abs()<.3)&(z.abs()<.25)]=2.
        # Ninety-degree z rotation swaps the x and y extents.
        expected[...,c][(x.abs()<.205)&(y.abs()<.105)&(z.abs()<.155)]=4.
    # Avoid a sharp test surface exactly on a staggered sample.
    for c,component in enumerate(('Ex','Ey','Ez')):
        x,y,z=torch.meshgrid(*(torch.tensor(a,dtype=result.dtype) for a in field_axes(r,component)),indexing='ij')
        away=((x.abs()-.4).abs()>.001)&((y.abs()-.3).abs()>.001)&((z.abs()-.25).abs()>.001)
        torch.testing.assert_close(result[...,c][away],expected[...,c][away],rtol=0,atol=0)


def test_geometry_vjp_directional_derivative_all_parameters_and_noncontiguous_seed():
    r=region('float64')
    p=torch.tensor([.02,-.03,.04,11.,-9.,17.,3.4,.21,.38],dtype=torch.float64,requires_grad=True)
    background=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    epsilon=smooth_geometry_epsilon(r,scene(p),background=background,width=.09,chunk_cells=113)
    seed=torch.randn((3,*r.shape),dtype=p.dtype,generator=torch.Generator().manual_seed(21)).movedim(0,-1)
    assert not seed.is_contiguous()
    dp,db=torch.autograd.grad(epsilon,(p,background),seed)
    assert bool((dp.abs()>1e-5).all()) and abs(float(db))>1e-5
    direction=torch.tensor([.3,-.2,.1,4.,-3.,2.,.4,.2,-.1],dtype=p.dtype)
    def scalar(t):
        value=smooth_geometry_epsilon(r,scene(p.detach()+t*direction),
            background=background.detach()+t*.3,width=.09,chunk_cells=10000)
        return (value*seed).sum()
    h=1e-5
    finite=(scalar(h)-scalar(-h))/(2*h)
    torch.testing.assert_close((dp*direction).sum()+.3*db,finite,rtol=2e-6,atol=1e-7)


def test_geometry_keeps_only_parameters_and_axes_between_forward_and_backward():
    p=torch.tensor(.23,dtype=torch.float32,requires_grad=True)
    def retained(r):
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t.numel()),t)[1],lambda t:t):
            value=smooth_geometry_epsilon(r,[DifferentiableSolid.sphere(p,epsilon=3.)],chunk_cells=41)
        return value,sum(saved),max(saved)
    small=region();large=small.model_copy(update={'size':(2.8,2.6,2.4)})
    a,small_saved,_=retained(small)
    b,large_saved,max_saved=retained(large)
    assert b.numel()>=7*a.numel() and large_saved<3*small_saved
    assert max_saved<=max(large.shape)
    torch.autograd.grad(a.sum(),p)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_geometry_to_fdtd_adjoint_and_streamed_chain(device):
    if device=='cuda':
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
        pytest.importorskip('cupy')
    r=region()
    project=Project(region=r,sources=[Source(center=(-.2,0,0),pulse='continuous')],
                    monitors=[Monitor(center=(.1,0,0)),Monitor(center=(0,.1,0),component='Hy')])
    p=torch.tensor([.02,-.03,.04,11.,-9.,17.,3.4,.21,.38],device=device,dtype=torch.float32,requires_grad=True)
    model=DifferentiableSimulation(project,AdjointOptions(checkpoints=2))
    epsilon=smooth_geometry_epsilon(r,scene(p),width=.1,chunk_cells=257)
    reference=model.reference(epsilon)
    expected,=torch.autograd.grad(reference.square().sum(),p)
    result=model(smooth_geometry_epsilon(r,scene(p),width=.1,chunk_cells=257))
    actual,=torch.autograd.grad(result.signals.square().sum(),p)
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=1e-8)
    assert bool(torch.isfinite(actual).all()) and torch.linalg.vector_norm(actual)>0
    # The same CPU geometry can feed host-backed CUDA or CPU spatial replay.
    host=p.detach().cpu().requires_grad_(True)
    streamed=StreamedSimulation(project,StreamedAdjointOptions(device=device,slab_width=4,
        temporal_depth=2,checkpoints=1,gpu_budget_bytes=128*1024**2,host_budget_bytes=256*1024**2))
    result=streamed(smooth_geometry_epsilon(r,scene(host),width=.1,chunk_cells=257))
    gradient,=torch.autograd.grad(result.signals.square().sum(),host)
    torch.testing.assert_close(gradient,expected.cpu(),rtol=4e-4,atol=1e-8)


def test_empty_geometry_material_gradient_and_invalid_input_contracts():
    r=region(sampling='cell')
    background=torch.tensor(1.5,dtype=torch.float32,requires_grad=True)
    field=smooth_geometry_epsilon(r,[],background=background,chunk_cells=137)
    field.mean().backward()
    torch.testing.assert_close(background.grad,torch.tensor(1.,dtype=torch.float32))
    with pytest.raises(ValueError,match='precision'):
        smooth_geometry_epsilon(r,[DifferentiableSolid.sphere(torch.tensor(.2,dtype=torch.float64),epsilon=3.)])
    with pytest.raises(ValueError,match='extents'):
        smooth_geometry_epsilon(r,[DifferentiableSolid.cylinder(.2,-.3,epsilon=3.)])
    with pytest.raises(ValueError,match='finite'):
        smooth_geometry_epsilon(r,[],background=float('nan'))
    with pytest.raises(ValueError,match='chunk_cells'):
        smooth_geometry_epsilon(r,[],chunk_cells=0)
