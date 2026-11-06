import numpy as np
import pytest
import torch
from torchfdtd import (spectral_interpolate,spectral_electron_model,exposure_target_information,
                        gaussian_target_information,shot_read_covariance,recompute_cases)


def fixture(device='cpu'):
    wavelength=torch.tensor([420.,540.,670.],dtype=torch.float64,device=device)
    response=torch.arange(1,13,dtype=wavelength.dtype,device=device).reshape(4,3)/20
    qe=torch.tensor([.4,.8,.5],dtype=wavelength.dtype,device=device)
    mean=torch.tensor([1.,2.,1.5],dtype=wavelength.dtype,device=device)
    return response,wavelength,qe,mean,torch.diag(1/mean)


def test_quadrature_and_mean_against_numpy():
    r,w,q,m,b=fixture();result=spectral_electron_model(r,w,q,m,b,calibration=1e-10)
    weight=m.numpy()*np.array([60,125,65])*1e-9*(w.numpy()*1e-9)/(6.62607015e-34*299792458)*q.numpy()*1e-10
    expected=r.numpy()*weight
    np.testing.assert_allclose(result.mean_electrons,expected.sum(-1),rtol=1e-14)
    np.testing.assert_allclose(result.measurement_matrix,expected@b.numpy().T,rtol=1e-14)
    explicit=spectral_electron_model(r,w,q,m,b,calibration=1e-10,quadrature_nm=[5,5,5])
    assert not torch.allclose(explicit.mean_electrons,result.mean_electrons)


def test_interpolation_endpoints_and_derivative():
    v=torch.tensor([[1.,4.,2.],[3.,2.,1.]],dtype=torch.float64,requires_grad=True)
    actual=spectral_interpolate(v,[420,500,670],[420,460,500,585,670])
    expected=np.stack([np.interp([420,460,500,585,670],[420,500,670],row) for row in v.detach().numpy()])
    np.testing.assert_allclose(actual.detach(),expected)
    assert torch.autograd.gradcheck(lambda v:spectral_interpolate(v,[420,500,670],[440,600]),(v,))
    with pytest.raises(ValueError,match='extrapolation'):spectral_interpolate(v,[420,500,670],[700])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exposures_against_explicit_loop_and_gradcheck(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    r,w,q,m,b=fixture(device);r.requires_grad_();eye=torch.eye(3,device=device,dtype=r.dtype)
    def score(response):
        model=spectral_electron_model(response,w,q,m,b,calibration=1e-10)
        return exposure_target_information(model,eye,.6*eye,eye,exposure_scales=[.02,.04,.08,.16,.32],
            probabilities=[.2]*5,read_noise_e_rms=1.5,raw_pixels=4)
    actual=score(r)
    model=spectral_electron_model(r,w,q,m,b,calibration=1e-10)
    expected=torch.stack([gaussian_target_information(model.measurement_matrix*s,eye,.6*eye,eye,
        shot_read_covariance(model.mean_electrons*s,1.5)).information_bits/4 for s in (.02,.04,.08,.16,.32)])
    torch.testing.assert_close(actual.bits_per_pixel,expected)
    assert torch.autograd.gradcheck(lambda r:score(r).weighted_bits_per_pixel,(r,),rtol=1e-5,atol=1e-8)


def test_case_recompute_through_electron_objective():
    r,w,q,m,b=fixture();eye=torch.eye(3,dtype=r.dtype)
    x=torch.tensor(.3,dtype=r.dtype,requires_grad=True)
    cases=[lambda x,row=row:row*(1+x.square()) for row in r]
    def score(response):
        model=spectral_electron_model(response,w,q,m,b,calibration=1e-10)
        return exposure_target_information(model,eye,.6*eye,eye,exposure_scales=[.1,.3],
            probabilities=[.4,.6],read_noise_e_rms=1.5,raw_pixels=4).weighted_bits_per_pixel
    expected=torch.stack([f(x) for f in cases]);eg,=torch.autograd.grad(score(expected),x)
    actual=recompute_cases(cases,x);ag,=torch.autograd.grad(score(actual),x)
    torch.testing.assert_close(ag,eg,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('kind',['calibration','negative','grid','quadrature','qe'])
def test_invalid_inputs(kind):
    r,w,q,m,b=fixture();kwargs={'calibration':1e-10}
    if kind=='calibration':kwargs['calibration']=torch.tensor(1e-10,requires_grad=True)
    if kind=='negative':r[0,0]=-1
    if kind=='grid':w[1]=w[0]
    if kind=='quadrature':kwargs['quadrature_nm']=[1,0,1]
    if kind=='qe':q[0]=float('nan')
    with pytest.raises(ValueError):spectral_electron_model(r,w,q,m,b,**kwargs)
