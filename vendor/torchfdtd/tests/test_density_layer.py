import pytest
import torch
from torchfdtd import periodic_density_layer,BoundaryFace
from test_differentiable import project


def region():
    r=project(dimension='3d',periodic=True).region;r.material_sampling='yee'
    r.boundaries.y_min=r.boundaries.y_max=BoundaryFace(kind='periodic')
    return r


def transfer(density):
    return periodic_density_layer(density,region(),bottom_um=-.2,top_um=.2,background_epsilon=2.,design_epsilon=4.)


def test_volume_and_periodic_translation():
    d=torch.arange(12,dtype=torch.float64).reshape(4,3)/12
    eps=transfer(d);r=region()
    volume=(eps-2).sum((0,1,2))*torch.tensor(r.axis_steps).prod()/2
    expected=d.mean()*r.actual_size[0]*r.actual_size[1]*.4
    torch.testing.assert_close(volume,expected.expand(3),rtol=1e-6,atol=1e-12)
    torch.testing.assert_close(transfer(d.roll(1,0)),eps.roll(4,0))
    torch.testing.assert_close(transfer(d.roll(1,1)),eps.roll(5,1))
    assert eps.shape==r.shape+(3,)
    torch.testing.assert_close(transfer(torch.zeros_like(d)),torch.full_like(eps,2.))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_density_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    d=torch.full((4,3),.4,dtype=torch.float64,device=device,requires_grad=True)
    assert torch.autograd.gradcheck(lambda d:transfer(d).square().mean(),(d,))


def test_invalid_transfer():
    with pytest.raises(ValueError,match=r'\[0,1\]'):transfer(torch.ones(4,3)*2)
    r=region();r.boundaries.y_min=BoundaryFace(kind='pml')
    with pytest.raises(ValueError,match='periodic'):periodic_density_layer(torch.ones(4,3),r,bottom_um=-.2,top_um=.2,background_epsilon=2,design_epsilon=4)


def test_fdtd_density_directional_gradient():
    from torchfdtd import DifferentiableSimulation,AdjointOptions
    p=project(dimension='3d',steps=24);p.region=region()
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    d=torch.full((4,3),.4,dtype=torch.float64,requires_grad=True)
    direction=torch.arange(12,dtype=d.dtype).reshape(4,3)/12-.4
    def loss(d):return model(transfer(d)).signals.square().sum()
    gradient,=torch.autograd.grad(loss(d),d)
    with torch.no_grad():finite=(loss(d+1e-5*direction)-loss(d-1e-5*direction))/2e-5
    torch.testing.assert_close((gradient*direction).sum(),finite,rtol=2e-6,atol=1e-9)
    assert gradient.norm()>1e-5


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sample_center_origin_is_half_pixel_translation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    r=region();r.size=(1.6,1.6,1.4)
    d=torch.arange(16,dtype=torch.float64,device=device).reshape(4,4)/16
    kwargs=dict(bottom_um=-.2,top_um=.2,background_epsilon=2.,design_epsilon=4.)
    edges=periodic_density_layer(d,r,**kwargs)
    centers=periodic_density_layer(d,r,**kwargs,pixel_origin='sample_centers')
    torch.testing.assert_close(centers,edges.roll((-2,-2),(0,1)),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(centers.sum(),edges.sum())
    d=(d*.8+.1).requires_grad_()
    assert torch.autograd.gradcheck(lambda d:periodic_density_layer(d,r,**kwargs,pixel_origin='sample_centers').square().mean(),(d,))
