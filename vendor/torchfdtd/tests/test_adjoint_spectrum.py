"""Online DFT values and VJPs against full-history autograd and analytic sums."""
from dataclasses import replace

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation, DifferentiableResult, StreamedSimulation, StreamedAdjointOptions
from torchfdtd.adjoint_spectrum import SpectralObservation
from torchfdtd.streamed import _reservation
from test_differentiable import project, gpu


@pytest.mark.parametrize('mode', ['resident', 'host', 'disk'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('diagonal', [False, True])
def test_online_spectrum_and_backward_match_full_history(tmp_path, mode, device, diagonal):
    if device == 'cuda':gpu()
    p = project(steps=13, periodic=diagonal)
    geometry_device = device if mode == 'resident' else 'cpu'
    generator = torch.Generator(device=geometry_device).manual_seed(372)
    epsilon = (1.4+.2*torch.rand(p.region.shape+((3,) if diagonal else ()),
                               dtype=torch.float64, device=geometry_device, generator=generator)).requires_grad_()
    reference = DifferentiableSimulation(p)
    history = reference.reference(epsilon)
    frequency = torch.tensor([.011,.037,.092],dtype=epsilon.dtype,device=epsilon.device)/p.region.time_step
    window = torch.linspace(.1,1,p.region.steps,dtype=epsilon.dtype,device=epsilon.device)
    expected = DifferentiableResult(history,p.region.time_step,tuple(m.component for m in p.monitors)).spectrum(frequency,window=window)
    def loss(values):
        scaled = values/p.region.time_step
        return (scaled.real+.7*scaled.imag).square().sum()+.1*scaled.abs().pow(2).sum()
    expected_grad, = torch.autograd.grad(loss(expected),epsilon)
    if mode == 'resident':
        model = DifferentiableSimulation(p,AdjointOptions(checkpoints=2,storage='host'))
    else:
        model = StreamedSimulation(p,StreamedAdjointOptions(device=device,slab_width=4,temporal_depth=3,
                                   state_storage=mode,state_directory=tmp_path if mode=='disk' else None,
                                   disk_budget_bytes=32*1024**2 if mode=='disk' else None,
                                   tile_transfers='async' if device=='cuda' else 'sync'))
    result = model.spectrum(epsilon,frequency,window=window,**({'block_size':5} if mode=='resident' else {}))
    assert result.report['observation_storage']=='online_spectrum'
    torch.testing.assert_close(result.fields/p.region.time_step,expected/p.region.time_step,rtol=2e-11,atol=1e-12)
    # User mutations must not modify the adjoint's fixed observation settings.
    frequency.fill_(1)
    window.fill_(0)
    result.frequency_hz.fill_(2)
    actual_grad, = torch.autograd.grad(loss(result.fields),epsilon,retain_graph=True)
    again, = torch.autograd.grad(loss(result.fields),epsilon)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=3e-10,atol=1e-11)
    torch.testing.assert_close(again,actual_grad,rtol=0,atol=0)
    assert torch.linalg.vector_norm(actual_grad)>1e-4
    if mode=='disk':assert list(tmp_path.iterdir())==[]


def test_linear_transform_matches_explicit_complex_sum_and_real_transpose():
    p=project(steps=17)
    epsilon=torch.ones(p.region.shape,dtype=torch.float64)
    components=('Ex','Hy','Ez')
    frequencies=torch.tensor([.02,.08],dtype=epsilon.dtype)/p.region.time_step
    samples=torch.randn((17,3),dtype=epsilon.dtype,requires_grad=True)
    observer=SpectralObservation(epsilon,p.region,components,frequencies)
    terms=[]
    for f in frequencies:
        terms.append(torch.stack([sum(samples[t,m]*torch.exp(-2j*torch.pi*f*p.region.time_step*(t+1+(.5 if c[0]=='H' else 0)))
                                      for t in range(17))*p.region.time_step for m,c in enumerate(components)]))
    expected=torch.stack(terms)
    actual=observer.zeros()
    for start,stop in ((0,4),(4,11),(11,17)):
        observer.accumulate(actual,samples.detach()[start:stop],start)
    torch.testing.assert_close(actual/p.region.time_step,expected/p.region.time_step,rtol=1e-12,atol=1e-12)
    seed=torch.randn_like(expected)/p.region.time_step
    reference,=torch.autograd.grad(expected,samples,grad_outputs=seed)
    transpose=torch.cat([observer.transpose(seed,a,b) for a,b in ((0,6),(6,17))])
    torch.testing.assert_close(transpose,reference,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('frequency,window,match', [
    ([],None,'nonempty'),([float('nan')],None,'finite'),([0],None,'positive'),
    ([1e30],None,'Nyquist'),([1e12],[1],'finite weight'),
    ([1e12],torch.full((13,),float('nan')),'finite weight'),
    (torch.tensor([1e12],requires_grad=True),None,'fixed observation'),
    ([1e12],torch.ones(13,requires_grad=True),'fixed observation'),
    ([1e12+1j],None,'real'),
])
def test_invalid_settings(frequency,window,match):
    p=project(steps=13)
    with pytest.raises(ValueError,match=match):
        DifferentiableSimulation(p).spectrum(torch.ones(p.region.shape,dtype=torch.float64),frequency,window=window)


def test_spectral_admission_removes_full_monitor_history():
    p=project(steps=10000)
    p.monitors=p.monitors*20
    eps=torch.empty(p.region.shape,dtype=torch.float64,device='meta')
    # Construct only the small fixed settings on CPU, without physical states.
    metadata=torch.ones((),dtype=torch.float64)
    observer=SpectralObservation(metadata,p.region,[m.component for m in p.monitors],[1e12,2e12])
    options=StreamedAdjointOptions(device='cpu',temporal_depth=4,slab_width=4)
    online=_reservation(p,eps,options,observer)
    full=_reservation(p,eps,options)
    assert full['host_reservation_bytes']-online['host_reservation_bytes']>9_000_000
    tight=replace(options,host_budget_bytes=online['host_reservation_bytes'])
    _reservation(p,eps,tight,observer)
    with pytest.raises(ValueError,match='host budget'):_reservation(p,eps,tight)


def test_spectral_higher_order_is_explicitly_rejected():
    p=project(steps=10)
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    output=DifferentiableSimulation(p).spectrum(eps,[1e12])
    with pytest.raises(RuntimeError,match='Higher-order'):
        torch.autograd.grad(output.fields.real.sum(),eps,create_graph=True)


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_spectral_block_size(size):
    p=project(steps=10)
    with pytest.raises(ValueError,match='block_size'):
        DifferentiableSimulation(p).spectrum(torch.ones(p.region.shape,dtype=torch.float64),[1e12],block_size=size)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_float32_3d_spectrum_matches_history(device):
    if device=='cuda':gpu()
    p=project(dimension='3d',precision='float32',steps=11)
    eps=torch.full(p.region.shape,1.8,dtype=torch.float32,device=device,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=1))
    frequencies=[.01/p.region.time_step,.071/p.region.time_step]
    history=model(eps).spectrum(frequencies)/p.region.time_step
    expected,=torch.autograd.grad(history.abs().square().sum(),eps)
    online=model.spectrum(eps,frequencies).fields/p.region.time_step
    actual,=torch.autograd.grad(online.abs().square().sum(),eps)
    torch.testing.assert_close(online,history,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual,expected,rtol=4e-5,atol=2e-6)


def test_geometry_central_difference_with_disk_spectrum(tmp_path):
    from torchfdtd import smooth_sphere_epsilon
    p=project(steps=19)
    model=StreamedSimulation(p,StreamedAdjointOptions(device='cpu',slab_width=5,temporal_depth=4,
                              state_storage='disk',state_directory=tmp_path,disk_budget_bytes=32*1024**2))
    def objective(radius):
        eps=smooth_sphere_epsilon(p.region,radius,width=.08,inside=3.2)
        fields=model.spectrum(eps,[.019/p.region.time_step,.068/p.region.time_step]).fields/p.region.time_step
        return (fields.real+.4*fields.imag).square().sum()
    radius=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    derivative,=torch.autograd.grad(objective(radius),radius)
    h=1e-5
    with torch.no_grad():difference=(objective(radius+h)-objective(radius-h))/(2*h)
    torch.testing.assert_close(derivative,difference,rtol=2e-7,atol=1e-9)
    assert abs(derivative)>1e-4
    assert list(tmp_path.iterdir())==[]
