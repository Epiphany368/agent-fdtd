from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torchfdtd import (quadrant_intensity_allocation,DifferentiablePlaneSimulation,
                        FieldMonitor,AdjointOptions)
from torchfdtd.field_monitors import plane_plan
from test_differentiable import project


def plane(fields=None):
    axis=torch.tensor([-.75,-.25,.25,.75],dtype=torch.float64)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    points=torch.stack((x.flatten(),y.flatten(),torch.zeros(16,dtype=x.dtype)),1)
    if fields is None:
        rng=torch.Generator().manual_seed(12)
        fields=torch.randn(2,16,6,generator=rng,dtype=torch.complex128)
    return SimpleNamespace(fields=fields,points_um=points,weights=torch.full((16,),.25e-12,dtype=axis.dtype),
                           normal='z',components=('Ex','Ey','Ez','Hx','Hy','Hz'))


def test_matches_independent_quadrant_integrals_and_sum():
    p=plane();transmission=torch.tensor([.7,1.1],dtype=torch.float64)
    actual=quadrant_intensity_allocation(p,transmission)
    intensity=np.sum(abs(p.fields.numpy()[...,:3])**2,axis=-1).reshape(2,4,4)
    q=np.stack([intensity[:,:2,:2].sum((1,2)),intensity[:,2:,:2].sum((1,2)),
                intensity[:,:2,2:].sum((1,2)),intensity[:,2:,2:].sum((1,2))],-1)
    expected=q/q.sum(-1,keepdims=True)*transmission.numpy()[:,None]
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-14)
    torch.testing.assert_close(actual.sum(-1),transmission)
    shifted=plane(p.fields);shifted.points_um[:,:2]+=1
    torch.testing.assert_close(quadrant_intensity_allocation(shifted,transmission,split_um=(1,1)),actual)
    # Magnetic fields cannot alter this electric allocation proxy.
    modified=plane(p.fields.clone());modified.fields[...,3:]*=100
    torch.testing.assert_close(quadrant_intensity_allocation(modified,transmission),actual)


def test_field_and_total_transmission_gradcheck():
    p=plane();fields=p.fields.requires_grad_();t=torch.tensor([.6,.8],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda e,t:quadrant_intensity_allocation(plane(e),t),(fields,t))


def quadrant_plane(ex,weights):
    """xy plane with only E_x set; points cycle through R, G2, G1, B, one group per quadrant."""
    per=ex.shape[1]//4
    corners=torch.tensor([[-1.,-1.,0.],[1.,-1.,0.],[-1.,1.,0.],[1.,1.,0.]],dtype=ex.real.dtype,device=ex.device)
    fields=torch.stack((ex,)+(torch.zeros_like(ex),)*5,-1)
    return SimpleNamespace(fields=fields,points_um=corners.repeat_interleave(per,0),weights=weights,
                           normal='z',components=('Ex','Ey','Ez','Hx','Hy','Hz'))


def dft_scale_plane(amplitude,a,dtype,device):
    """One sample per quadrant, E_x = amplitude*[a,1,1,1], areas of 1e-14 m^2."""
    ex=(amplitude*torch.cat((a[None],torch.ones(3,dtype=a.dtype,device=device)))).to(dtype)[None]
    return quadrant_plane(ex,torch.full((4,),1e-14,dtype=a.dtype,device=device))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('amplitude',[1.,1e-8,1e-14,1e-16])
def test_fp32_dft_scale_ratio_and_gradient(device,amplitude):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # Time-integrated DFT fields (~1e-14) times m^2 areas (~1e-14) fall below the
    # float32 normal range. The unscaled ratio returned a NaN gradient at 1e-14
    # and raised on a zero total at 1e-16. Analytic first quadrant: a^2/(a^2+3).
    a=torch.tensor(1.5,dtype=torch.float32,device=device,requires_grad=True)
    output=quadrant_intensity_allocation(dft_scale_plane(amplitude,a,torch.complex64,device),
                                         torch.ones(1,dtype=torch.float32,device=device))
    gradient,=torch.autograd.grad(output[0,0],a)
    assert abs(output[0,0].item()-1.5**2/(1.5**2+3))<1e-6
    assert abs(gradient.item()-6*1.5/(1.5**2+3)**2)<1e-5


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_dft_scale_field_and_total_transmission_gradcheck(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    unit=plane().fields.to(device).requires_grad_();t=torch.tensor([.6,.8],dtype=torch.float64,device=device,requires_grad=True)
    def response(e,t):
        # Unit-scale inputs keep gradcheck's perturbation meaningful at 1e-14 fields.
        p=plane(e*1e-14);p.points_um=p.points_um.to(device);p.weights=torch.full((16,),1e-14,dtype=torch.float64,device=device)
        return quadrant_intensity_allocation(p,t)
    assert torch.autograd.gradcheck(response,(unit,t))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_scaled_ratio_matches_unscaled_formula_on_unit_fields(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    fields=plane().fields.to(device).requires_grad_();t=torch.tensor([.7,1.1],dtype=torch.float64,device=device,requires_grad=True)
    p=plane(fields);p.points_um=p.points_um.to(device);p.weights=p.weights.to(device)
    actual=quadrant_intensity_allocation(p,t)
    low_x=p.points_um[:,0]<0;low_y=p.points_um[:,1]<0
    masks=torch.stack((low_x&low_y,~low_x&low_y,low_x&~low_y,~low_x&~low_y)).to(torch.float64)
    integrals=(fields[...,:3].abs().square().sum(-1)*p.weights)@masks.T
    expected=integrals/integrals.sum(-1,keepdim=True)*t[:,None]
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-7)
    seed=torch.arange(1.,9.,dtype=torch.float64,device=device).reshape(2,4)
    torch.testing.assert_close(torch.autograd.grad((actual*seed).sum(),(fields,t)),
                               torch.autograd.grad((expected*seed).sum(),(fields,t)),rtol=1e-7,atol=1e-7)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_nonuniform_areas_weight_the_ratio(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # Two samples per quadrant with areas 1e-14 and 1e-13 m^2. E_x is 1e-14*[a,1]
    # in R and 1e-14*[1,1] elsewhere, so R integrates a^2*w1+w2 against 3*(w1+w2).
    a=torch.tensor(1.5,dtype=torch.float32,device=device,requires_grad=True)
    ex=(1e-14*torch.cat((a[None],torch.ones(7,dtype=a.dtype,device=device)))).to(torch.complex64)[None]
    weights=torch.tensor([1e-14,1e-13]*4,dtype=torch.float32,device=device)
    output=quadrant_intensity_allocation(quadrant_plane(ex,weights),torch.ones(1,dtype=torch.float32,device=device))
    gradient,=torch.autograd.grad(output[0,0],a)
    w1,w2=1.,10.;first=1.5**2*w1+w2;others=3*(w1+w2);total=first+others
    expected=torch.tensor([first,w1+w2,w1+w2,w1+w2],device=device)/total
    torch.testing.assert_close(output[0],expected,rtol=0,atol=1e-6)
    assert abs(gradient.item()-2*1.5*w1*others/total**2)<1e-5


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_per_frequency_scales_stay_independent(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # Three frequencies at field scales 1, 1e-8 and 1e-14 in one call, each with
    # its own parameter and total transmission; no frequency sees another's gradient.
    a=torch.tensor([1.5,.5,2.],dtype=torch.float32,device=device,requires_grad=True)
    amplitude=torch.tensor([1.,1e-8,1e-14],dtype=torch.float32,device=device)
    t=torch.tensor([1.,.5,2.],dtype=torch.float32,device=device)
    ex=(amplitude[:,None]*torch.cat((a[:,None],torch.ones(3,3,dtype=a.dtype,device=device)),1)).to(torch.complex64)
    output=quadrant_intensity_allocation(quadrant_plane(ex,torch.full((4,),1e-14,dtype=torch.float32,device=device)),t)
    ratio=torch.stack((a.detach().square(),)+(torch.ones_like(a),)*3,1)/(a.detach().square()+3)[:,None]
    torch.testing.assert_close(output,ratio*t[:,None],rtol=0,atol=1e-6)
    for f in range(3):
        gradient,=torch.autograd.grad(output[f,0],a,retain_graph=True)
        assert abs(gradient[f].item()-t[f].item()*6*a[f].item()/(a[f].item()**2+3)**2)<1e-5
        assert bool((gradient[[g for g in range(3) if g!=f]]==0).all())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_point_phases_leave_ratio_and_gradient_unchanged(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=torch.tensor(1.5,dtype=torch.float32,device=device,requires_grad=True)
    t=torch.ones(1,dtype=torch.float32,device=device)
    plain=quadrant_intensity_allocation(dft_scale_plane(1e-14,a,torch.complex64,device),t)
    phase=torch.polar(torch.ones(4,dtype=torch.float32,device=device),torch.tensor([.3,-1.2,2.5,4.],dtype=torch.float32,device=device))
    ex=(1e-14*torch.cat((a[None],torch.ones(3,dtype=a.dtype,device=device)))).to(torch.complex64)[None]*phase
    phased=quadrant_intensity_allocation(quadrant_plane(ex,torch.full((4,),1e-14,dtype=torch.float32,device=device)),t)
    torch.testing.assert_close(phased,plain,rtol=0,atol=1e-6)
    seed=torch.tensor([[1.,2.,3.,4.]],device=device)
    torch.testing.assert_close(torch.autograd.grad((phased*seed).sum(),a)[0],
                               torch.autograd.grad((plain*seed).sum(),a)[0],rtol=0,atol=1e-5)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_common_source_scale_only_enters_through_total_transmission(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # A common complex factor on every field leaves the ratio, the gradient of a
    # geometry-like gain and the gradient of total transmission unchanged, and
    # receives no gradient itself. Only the caller's total transmission carries it.
    p=plane();base=p.fields.to(device);points=p.points_um.to(device);weights=p.weights.to(device)
    mask=((points[:,0]<0)&(points[:,1]<0)).to(torch.complex128)[None,:,None]
    gain=torch.tensor(.4,dtype=torch.float64,device=device,requires_grad=True)
    t=torch.tensor([.7,1.1],dtype=torch.float64,device=device,requires_grad=True)
    scale=torch.polar(torch.tensor(1e-14,dtype=torch.float64,device=device),torch.tensor(.7,dtype=torch.float64,device=device)).requires_grad_()
    def response(factor):
        q=plane(base*(1+gain*mask)*factor);q.points_um=points;q.weights=weights
        return quadrant_intensity_allocation(q,t)
    plain=response(torch.ones((),dtype=torch.complex128,device=device));scaled=response(scale)
    torch.testing.assert_close(scaled,plain,rtol=1e-12,atol=0)
    seed=torch.arange(1.,9.,dtype=torch.float64,device=device).reshape(2,4)
    plain_gradients=torch.autograd.grad((plain*seed).sum(),(gain,t))
    *scaled_gradients,scale_gradient=torch.autograd.grad((scaled*seed).sum(),(gain,t,scale))
    torch.testing.assert_close(tuple(scaled_gradients),plain_gradients,rtol=1e-12,atol=0)
    assert (scale_gradient*scale).abs().item()<1e-12


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_zero_intensity_raises_and_zero_transmission_keeps_finite_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=torch.tensor(1.5,dtype=torch.float32,device=device,requires_grad=True)
    with pytest.raises(ValueError,match='finite and positive'):
        quadrant_intensity_allocation(dft_scale_plane(0.,a,torch.complex64,device),torch.ones(1,dtype=torch.float32,device=device))
    t=torch.zeros(1,dtype=torch.float32,device=device,requires_grad=True)
    output=quadrant_intensity_allocation(dft_scale_plane(1e-14,a,torch.complex64,device),t)
    assert bool((output==0).all())
    seed=torch.tensor([[1.,2.,3.,4.]],device=device)
    gradient_a,gradient_t=torch.autograd.grad((output*seed).sum(),(a,t))
    assert bool(torch.isfinite(gradient_a)) and gradient_a.item()==0
    assert abs(gradient_t.item()-(1.5**2+2+3+4)/(1.5**2+3))<1e-5


@pytest.mark.parametrize('kind',['zero','negative_t','weights','missing_quadrant','normal','nan'])
def test_invalid_contract(kind):
    p=plane();t=torch.ones(2,dtype=torch.float64)
    if kind=='zero':p.fields.zero_()
    if kind=='negative_t':t[0]=-1
    if kind=='weights':p.weights[0]=0
    if kind=='missing_quadrant':p.points_um[:,0]=1
    if kind=='normal':p.normal='x'
    if kind=='nan':p.fields[0,0,0]=float('nan')
    with pytest.raises(ValueError):quadrant_intensity_allocation(p,t)


@pytest.mark.parametrize('normal',['x','y','z'])
def test_fixed_midpoint_quadrature(normal):
    size=[.8,.6,.4];size['xyz'.index(normal)]=0
    p=project(dimension='3d');m=FieldMonitor(normal=normal,center=(0,0,0),size=tuple(size),downsample=4)
    plan=plane_plan(p.region,m,(24,24));a='xyz'.index(normal)
    transverse=[i for i in range(3) if i!=a]
    assert len(plan['weights'])==576
    assert np.sum(plan['weights'])==pytest.approx(np.prod([m.size[i] for i in transverse])*1e-12)
    for i in transverse:
        np.testing.assert_allclose(np.unique(plan['points_um'][:,i]),(-.5+(np.arange(24)+.5)/24)*m.size[i])
    assert np.all(plan['points_um'][:,a]==0)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bloch',[False,True])
def test_fdtd_geometry_and_allocation_directional_derivative(device,bloch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=project(dimension='3d',steps=14)
    if bloch:
        p.region.boundaries.x_min.kind=p.region.boundaries.x_max.kind='bloch'
        p.region.bloch_phase=(.4,0,0)
    p.monitors=[FieldMonitor(id='detector',normal='z',center=(0,0,.1),size=(.8,.8,0))]
    model=DifferentiablePlaneSimulation(p,AdjointOptions(checkpoints=2),quadrature_counts={'detector':(6,6)})
    mask=torch.zeros(p.region.shape,dtype=torch.float64,device=device);mask[5:9,5:8,5:9]=1
    def objective(x):
        output=model(1.5+x*mask,[.03/p.region.time_step])['detector']
        assert output.fields.shape==(1,36,6)
        # Synthetic total transmission isolates the joint chain rule, not optics.
        allocation=quadrant_intensity_allocation(output,(.5+x.square())[None])
        return (allocation*torch.tensor([1.,2.,3.,4.],device=device)).sum()
    x=torch.tensor(.2,dtype=torch.float64,device=device,requires_grad=True)
    gradient,=torch.autograd.grad(objective(x),x)
    with torch.no_grad():finite=(objective(x+1e-5)-objective(x-1e-5))/(2e-5)
    torch.testing.assert_close(gradient,finite,rtol=2e-6,atol=1e-8)


def test_bad_quadrature_configuration():
    p=project(dimension='3d');p.monitors=[FieldMonitor(id='p',normal='z',size=(.8,.8,0))]
    with pytest.raises(ValueError,match='IDs'):DifferentiablePlaneSimulation(p,quadrature_counts={'absent':(24,24)})
    with pytest.raises(ValueError,match='positive'):DifferentiablePlaneSimulation(p,quadrature_counts={'p':(0,24)})
