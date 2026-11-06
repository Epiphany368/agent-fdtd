"""Complex seam conjugation, real-epsilon VJPs and lossless complex replay."""
import math
import numpy as np
import pytest
import torch
from torchfdtd import (Source,Monitor,FieldMonitor,BoundaryFace,Simulation,AdjointOptions,
    DifferentiableSimulation,DifferentiablePlaneSimulation,StreamedSimulation)
from test_differentiable import project,gpu


def scene(nonuniform=False):
    p=project(steps=14,periodic=True)
    p.region.boundaries.x_min.kind=p.region.boundaries.x_max.kind='bloch'
    p.region.bloch_phase=(.63,0,0)
    p.region.cuda_kernel='torch'
    p.region.material_sampling='yee'
    p.sources=[Source(kind='plane',normal='y',center=(0,-.3,0),size=(1.6,0,0),component='Ez',pulse='continuous',wavelength=1.1)]
    p.monitors=[Monitor(component='Ez',center=(-.7,.1,0)),Monitor(component='Hx',center=(.6,.1,0)),Monitor(component='Ez',center=(-.7,.1,0))]
    if nonuniform:
        p.region.mesh_type='explicit'
        nodes=[]
        for span,count in ((1.6,16),(1.5,15)):
            u=np.linspace(-1,1,count+1)
            nodes.append(tuple(span/2*(.8*u+.2*u**3)))
        p.region.mesh_coordinates=(*nodes,(-.7,.7))
    return p


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('nonuniform',[False,True])
def test_complex_transpose_and_spectral_seed_against_full_autograd(device,nonuniform):
    if device=='cuda':gpu()
    p=scene(nonuniform)
    eps=torch.full(p.region.shape+(3,),1.7,dtype=torch.float64,device=device,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    expected=model.reference(eps)
    def loss(v):return (v.real+.6*v.imag).square().sum()+.3*v.abs().square().sum()
    eg,=torch.autograd.grad(loss(expected),eps)
    actual=model(eps)
    ag,=torch.autograd.grad(loss(actual.signals),eps)
    torch.testing.assert_close(actual.signals,expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(ag,eg,rtol=2e-10,atol=1e-11)
    assert actual.signals.imag.abs().max()>1e-4
    frequency=[.025/p.region.time_step,.07/p.region.time_step]
    history=model(eps).spectrum(frequency)/p.region.time_step
    hg,=torch.autograd.grad(loss(history),eps)
    online=model.spectrum(eps,frequency,block_size=5)
    sg,=torch.autograd.grad(loss(online.fields/p.region.time_step),eps)
    torch.testing.assert_close(online.fields/p.region.time_step,history,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(sg,hg,rtol=2e-10,atol=1e-11)
    assert actual.report['backward_backend']=='torch explicit transpose'


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complex_native_forward_and_disk_replay(device,tmp_path):
    if device=='cuda':gpu()
    p=scene();p.region.background_index=math.sqrt(1.7)
    p.region.backend=device
    expected=Simulation(p).run(cuda_graph=False).signals
    eps=torch.full(p.region.shape,1.7,device=device,dtype=torch.float64,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,storage='disk',checkpoint_directory=tmp_path,
        disk_budget_bytes=32*1024**2,host_budget_bytes=32*1024**2,
        checkpoint_transfers='async' if device=='cuda' else 'sync'))
    result=model(eps)
    np.testing.assert_allclose(result.signals.detach().cpu().numpy(),expected,rtol=1e-11,atol=1e-12)
    grad,=torch.autograd.grad(result.signals.abs().square().sum(),eps)
    oracle=model.reference(eps)
    eg,=torch.autograd.grad(oracle.abs().square().sum(),eps)
    torch.testing.assert_close(grad,eg,rtol=2e-10,atol=1e-11)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('execution',['resident','cpu','cuda','cuda_disk'])
def test_bloch_plane_seam_interpolation_and_gradient(execution,tmp_path):
    from torchfdtd import StreamedAdjointOptions
    if execution.startswith('cuda'):gpu()
    from test_adjoint_planes import reference
    p=scene()
    p.monitors=[FieldMonitor(normal='y',center=(0,.2,0),size=(1.6,0,1),downsample=2)]
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    freq=torch.tensor([.03/p.region.time_step],dtype=eps.dtype)
    # Oracle must preserve complex interpolation weights at the Bloch seam.
    from torchfdtd.differentiable import _System
    from torchfdtd.field_monitors import plane_plan,interpolation_map
    from torchfdtd.adjoint_planes import COMPONENTS
    sys=_System(p,eps);state=sys.state();history=[]
    plan=plane_plan(p.region,p.monitors[0])
    for step in range(p.region.steps):
        state=sys.reference_step(state,step,eps)
        values=[]
        for c in COMPONENTS:
            ix,w=interpolation_map(p.region,c,plan['points_um'])
            v=(state[int(c[0]=='H')].reshape(-1)[torch.tensor(ix)]*torch.tensor(w)).sum(0)
            t=(step+1+(.5 if c[0]=='H' else 0))*p.region.time_step
            values.append(torch.exp(2j*torch.pi*freq[:,None]*t)*v[None,:])
        history.append(torch.stack(values,-1))
    expected=torch.stack(history).sum(0)
    eg,=torch.autograd.grad((expected.real+.4*expected.imag).square().sum(),eps)
    options=None if execution=='resident' else StreamedAdjointOptions(
        device='cuda' if execution.startswith('cuda') else 'cpu',slab_width=5,temporal_depth=3,
        checkpoints=2,local_checkpoints=1,tile_transfers='async' if execution.startswith('cuda') else 'sync',
        state_storage='disk' if execution=='cuda_disk' else 'host',
        state_directory=tmp_path,disk_budget_bytes=128*1024**2)
    actual=DifferentiablePlaneSimulation(p,options)(eps,freq)[p.monitors[0].id].fields/p.region.time_step
    ag,=torch.autograd.grad((actual.real+.4*actual.imag).square().sum(),eps)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(ag,eg,rtol=2e-10,atol=1e-11)
    assert not list(tmp_path.iterdir())


def test_unsupported_complex_execution_fails_explicitly():
    p=scene()
    with pytest.raises(ValueError,match='CUDA tensor'):
        DifferentiableSimulation(p,AdjointOptions(backward_kernel='fused'))(torch.ones(p.region.shape,dtype=torch.float64))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_three_dimensional_two_phase_float32(device):
    if device=='cuda':gpu()
    from torchfdtd import Project,Region,Boundaries
    p=Project(region=Region(dimension='3d',size=(1.2,1.2,1.6),mesh=.1,steps=11,pml_cells=3,
                            precision='float32',material_sampling='yee',cuda_kernel='torch',
                            boundaries=Boundaries(x_min=BoundaryFace(kind='bloch'),x_max=BoundaryFace(kind='bloch'),
                                                  y_min=BoundaryFace(kind='bloch'),y_max=BoundaryFace(kind='bloch'))),
              sources=[Source(kind='plane',normal='z',center=(0,0,-.3),size=(1.2,1.2,0),component='Ex',theta=70,phi=30,pulse='continuous')],
              monitors=[Monitor(component='Ex',center=(.4,-.4,.1)),Monitor(component='Hy',center=(-.4,.4,.1))])
    p.region.boundaries.x_min.kind=p.region.boundaries.x_max.kind='bloch'
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
    p.region.bloch_phase=(.42,-.37,0)
    eps=torch.full(p.region.shape+(3,),1.7,dtype=torch.float32,device=device,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    expected=model.reference(eps)
    eg,=torch.autograd.grad((expected.real+.7*expected.imag).square().sum(),eps)
    result=model(eps)
    ag,=torch.autograd.grad((result.signals.real+.7*result.signals.imag).square().sum(),eps)
    torch.testing.assert_close(result.signals,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(ag,eg,rtol=3e-5,atol=2e-6)


def test_complex_geometry_central_difference():
    p=scene();p.region.steps=18
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    mask=torch.zeros(p.region.shape,dtype=torch.float64);mask[6:10,6:10]=1
    value=torch.tensor(.5,dtype=mask.dtype,requires_grad=True)
    window=torch.linspace(.1,1,p.region.steps,dtype=mask.dtype)
    def objective(v):
        f=model.spectrum(1+v*mask,[.03/p.region.time_step],window=window,block_size=5).fields/p.region.time_step
        return (f.real+.3*f.imag).square().sum()
    derivative,=torch.autograd.grad(objective(value),value)
    with torch.no_grad():finite=(objective(value+1e-5)-objective(value-1e-5))/(2e-5)
    torch.testing.assert_close(derivative,finite,rtol=2e-7,atol=1e-9)
    assert abs(derivative)>1e-6


def test_later_ray_configuration_cannot_change_existing_replay():
    p=scene();eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    reference=model.reference(eps)
    expected,=torch.autograd.grad(reference.abs().square().sum(),eps)
    result=model(eps)
    model.project.region.bloch_phase=(-.2,0,0)
    model.project.region.steps=21
    model.project.sources[0].amplitude=2
    actual,=torch.autograd.grad(result.signals.abs().square().sum(),eps)
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=1e-11)
