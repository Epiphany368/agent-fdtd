"""Independent covariance identities and gradients for real target information."""
import math
import numpy as np
import pytest
import torch
from torchfdtd import gaussian_target_information,shot_read_covariance


def inputs(device='cpu'):
    g=torch.Generator().manual_seed(275)
    b=torch.randn((5,3),generator=g,dtype=torch.float64)
    x=torch.randn((5,5),generator=g,dtype=torch.float64)
    xx=x@x.T+torch.eye(5,dtype=x.dtype)
    xz=xx@b;zz=b.T@xx@b+.4*torch.eye(3,dtype=x.dtype)
    a=torch.randn((4,5),generator=g,dtype=x.dtype)
    n=torch.diag(torch.tensor([.7,1.,1.4,2.],dtype=x.dtype))
    return tuple(v.to(device) for v in (a,xx,xz,zz,n))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_schur_complement_and_decoder(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    data=inputs(device)
    result=gaussian_target_information(*data)
    a,xx,xz,zz,n=[v.cpu().numpy() for v in data]
    yy=a@xx@a.T+n;yz=a@xz
    decoder=np.linalg.solve(yy,yz).T
    posterior=zz-decoder@yz
    bits=(np.linalg.slogdet(zz)[1]-np.linalg.slogdet(posterior)[1])/(2*np.log(2))
    np.testing.assert_allclose(result.posterior_target_covariance.cpu(),posterior,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(result.decoder.cpu(),decoder,rtol=2e-13,atol=2e-13)
    assert float(result.information_bits)==pytest.approx(bits,rel=2e-13)
    assert float(result.recovered_trace_fraction)==pytest.approx(1-np.trace(posterior)/np.trace(zz))
    assert float(result.recovered_whitened_fraction)==pytest.approx(1-np.trace(np.linalg.solve(zz,posterior))/3)


def test_residual_target_uncertainty_and_high_snr_ceiling():
    a=torch.tensor([[1e6]],dtype=torch.float64)
    xx=torch.tensor([[2.]],dtype=a.dtype);xz=xx.clone();zz=xx+.5;n=torch.ones_like(xx)
    result=gaussian_target_information(a,xx,xz,zz,n)
    expected_post=.5+2/(1+2e12)
    assert float(result.posterior_target_covariance)==pytest.approx(expected_post,rel=1e-13)
    assert float(result.information_bits)==pytest.approx(.5*math.log2(2.5/expected_post),rel=1e-13)
    assert float(result.target_given_scene_covariance)==pytest.approx(.5)
    zero=gaussian_target_information(a*0,xx,xz,zz,n)
    assert abs(float(zero.information_bits))<1e-14


def test_batch_axes_and_target_basis_invariance():
    a,xx,xz,zz,n=inputs()
    a=torch.stack((a,a*.4,a*2)).reshape(3,1,4,5)
    result=gaussian_target_information(a,xx,xz,zz,n)
    change=torch.tensor([[1.,.2,.3],[.1,1.2,0.],[0.,.1,.8]],dtype=a.dtype)
    transformed=gaussian_target_information(a,xx,xz@change.T,change@zz@change.T,n)
    torch.testing.assert_close(result.information_bits,transformed.information_bits,rtol=1e-12,atol=1e-12)
    for i in range(3):
        direct=gaussian_target_information(a[i,0],xx,xz,zz,n)
        torch.testing.assert_close(result.information_bits[i,0],direct.information_bits)
    with pytest.raises(ValueError,match='Partial broadcasting'):
        gaussian_target_information(a,xx.expand(1,1,5,5),xz,zz,n)


def test_response_and_shot_noise_gradcheck():
    a,xx,xz,zz,_=inputs()
    a=(a.abs()+.2).requires_grad_()
    read=torch.tensor(1.5,dtype=a.dtype,requires_grad=True)
    def objective(response,sigma):
        mean=response@torch.arange(1,6,dtype=response.dtype)
        return gaussian_target_information(response,xx,xz,zz,shot_read_covariance(mean,sigma)).information_bits
    assert torch.autograd.gradcheck(objective,(a,read),eps=1e-6,atol=1e-6,rtol=1e-5)


def test_float32_and_mean_noise_gradient():
    data=inputs()
    reference=gaussian_target_information(*data)
    actual=gaussian_target_information(*(v.float() for v in data))
    torch.testing.assert_close(actual.information_bits.double(),reference.information_bits,rtol=1e-5,atol=1e-5)
    mean=torch.tensor([0.,2.,7.],dtype=torch.float64,requires_grad=True)
    cov=shot_read_covariance(mean,1.5)
    grad,=torch.autograd.grad(cov.sum(),mean)
    torch.testing.assert_close(grad,torch.ones_like(mean))
    with pytest.raises(ValueError,match='nonnegative'):shot_read_covariance(-torch.ones(3),1.5)
    with pytest.raises(ValueError,match='real'):shot_read_covariance(mean,torch.tensor(1+2j))
    with pytest.raises(ValueError,match='channel'):shot_read_covariance(torch.empty(0),1.5)


@pytest.mark.parametrize('kind',['asymmetric','noise','singular_scene','invalid_joint','complex','nan','shape'])
def test_invalid_covariances_fail_without_repair(kind):
    a,xx,xz,zz,n=inputs()
    if kind=='asymmetric':xx[0,1]+=.1
    elif kind=='noise':n[0,0]=-1
    elif kind=='singular_scene':xx.zero_()
    elif kind=='invalid_joint':xz*=100
    elif kind=='complex':a=a.to(torch.complex128)
    elif kind=='nan':a[0,0]=float('nan')
    elif kind=='shape':xz=xz[:,:2]
    with pytest.raises(ValueError):gaussian_target_information(a,xx,xz,zz,n)


def test_fdtd_plane_geometry_to_information_gradient():
    from test_adjoint_planes import scene
    from torchfdtd import DifferentiablePlaneSimulation,AdjointOptions
    p=scene();model=DifferentiablePlaneSimulation(p,AdjointOptions(checkpoints=2))
    mask=torch.zeros(p.region.shape,dtype=torch.float64);mask[6:9,5:8]=1
    xx=torch.eye(2,dtype=mask.dtype);xz=.7*xx;zz=xx.clone()
    def loss(parameter):
        planes=model(1+parameter*mask,[.022/p.region.time_step,.051/p.region.time_step])
        # Synthetic positive optical channel, not the locked CR detector model.
        response=torch.stack([1000*(v.fields[...,:3]/p.region.time_step).abs().square().mean((1,2)) for v in planes.values()])
        noise=shot_read_covariance(response.sum(-1),1.5)
        return gaussian_target_information(response,xx,xz,zz,noise).information_bits
    parameter=torch.tensor(.5,dtype=mask.dtype,requires_grad=True)
    gradient,=torch.autograd.grad(loss(parameter),parameter)
    with torch.no_grad():finite=(loss(parameter+1e-5)-loss(parameter-1e-5))/(2e-5)
    torch.testing.assert_close(gradient,finite,rtol=1e-5,atol=1e-9)
    assert abs(gradient)>1e-8
