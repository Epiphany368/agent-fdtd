import pytest
import torch
from torchfdtd import spectral_pupil_response,spectral_electron_model,exposure_target_information


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_coupled_information_and_exact_subset_weights(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    d=torch.tensor([.2,.4],dtype=torch.float64,device=device,requires_grad=True)
    base=torch.arange(1,9,dtype=d.dtype,device=device).reshape(2,4)/20
    cases=[[lambda d,w=w,r=r:base*(1+(d[0]*(w+1)+d[1]*(r+1)).square())
            for r in range(3)] for w in range(3)]
    weights=[.11,.27,.43]
    expected=torch.stack([sum(f(d).mean(0)*weight for f,weight in zip(row,weights)) for row in cases],1)
    actual=spectral_pupil_response(cases,d,weights)
    torch.testing.assert_close(actual,expected.cpu(),rtol=1e-14,atol=1e-14)
    subsets=sum(spectral_pupil_response([[row[i]] for row in cases],d,[weights[i]]) for i in range(3))
    torch.testing.assert_close(subsets,actual,rtol=1e-14,atol=1e-14)
    def score(response):
        eye=torch.eye(3,dtype=d.dtype,device=response.device)
        model=spectral_electron_model(response,[420,540,670],[.4,.8,.5],[1,2,1.5],eye,calibration=1e-10)
        return exposure_target_information(model,eye,.6*eye,eye,exposure_scales=[.02,.08,.32],
            probabilities=[.2,.5,.3],read_noise_e_rms=1.5,raw_pixels=4).weighted_bits_per_pixel
    ag,=torch.autograd.grad(score(actual),d)
    eg,=torch.autograd.grad(score(expected),d)
    torch.testing.assert_close(ag,eg,rtol=1e-11,atol=1e-12)
    direction=d.new_tensor([.3,-.2]);h=1e-5
    with torch.no_grad():
        fd=(score(spectral_pupil_response(cases,d+h*direction,weights))-score(spectral_pupil_response(cases,d-h*direction,weights)))/(2*h)
    torch.testing.assert_close((ag*direction).sum().cpu(),fd,rtol=1e-6,atol=1e-10)
    assert ag.norm()>1e-5


@pytest.mark.parametrize('weights',[[1,-1],[0,0],[1,float('nan')],[1],torch.tensor([.5,.5],requires_grad=True)])
def test_invalid_weights(weights):
    d=torch.tensor(.2)
    with pytest.raises(ValueError):spectral_pupil_response([[lambda d:d.expand(2,4)]*2],d,weights)


def test_invalid_outputs_and_schedule():
    d=torch.tensor(.2)
    for value in (d.expand(4),-d.expand(2,4),d.expand(2,4).to(torch.complex64)):
        with pytest.raises(ValueError):spectral_pupil_response([[lambda d:value]],d,[1])
    with pytest.raises(ValueError):spectral_pupil_response([[],[]],d,[1])


def test_periodic_response_uniform_medium_and_gradient():
    from torchfdtd import periodic_layer_response
    spec=dict(wavelength_um=.5,background_index=1.4,design_index=1.8,period_um=(.8,.8),
        height_um=.2,detector_offset_um=.5,theta_inside_rad=.1,phi_rad=.3)
    d=torch.full((2,2),.3,dtype=torch.float64,requires_grad=True)
    kwargs=dict(mesh=.1,steps=320,pml_cells=6,quadrature_counts=(4,4))
    with torch.no_grad():
        uniform=periodic_layer_response(torch.zeros_like(d),spec,**kwargs)
    torch.testing.assert_close(uniform,torch.full_like(uniform,.25),rtol=0,atol=1e-9)
    channel=d.new_tensor([.1,.2,.3,.4])
    def loss(d):return (periodic_layer_response(d,spec,**kwargs)*channel).sum()
    g,=torch.autograd.grad(loss(d),d)
    direction=d.new_tensor([[.2,-.3],[.4,.1]]);h=1e-5
    with torch.no_grad():fd=(loss(d+h*direction)-loss(d-h*direction))/(2*h)
    torch.testing.assert_close((g*direction).sum(),fd,rtol=2e-5,atol=1e-8)
    assert g.norm()>1e-5
