"""Discrete derivatives, complete CPML restart, and bounded checkpoint schedules."""
import math

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, BoundaryFace, DifferentiableSimulation,
                        Monitor, Project, Region, Simulation, Source, smooth_sphere_epsilon)


def project(dimension='2d',precision='float64',steps=24,periodic=False):
    r=Region(dimension=dimension,size=(1.6,1.5,1.4),mesh=.1,pml_cells=3,
             steps=steps,precision=precision,backend='cpu')
    if periodic:
        r.boundaries.x_min=BoundaryFace(kind='periodic')
        r.boundaries.x_max=BoundaryFace(kind='periodic')
    else:
        r.boundaries.x_min=BoundaryFace(layers=4,kappa=2,alpha=.03,alpha_polynomial=1)
    return Project(region=r,sources=[Source(center=(-.2,0,0),pulse='continuous',wavelength=1.1)],
                   monitors=[Monitor(component='Ez',center=(.1,0,0)),
                             Monitor(component='Hy',center=(0,.1,0)),
                             Monitor(component='Ez',center=(.1,0,0))])


def gpu():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


@pytest.mark.parametrize('dimension,periodic,diagonal',[('2d',False,False),('2d',True,True),('3d',False,True)])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_discrete_adjoint_matches_full_autograd(dimension,periodic,diagonal,device):
    if device=='cuda':gpu()
    p=project(dimension=dimension,periodic=periodic)
    shape=p.region.shape+((3,) if diagonal else ())
    generator=torch.Generator(device=device).manual_seed(82)
    epsilon=(1.4+.1*torch.rand(shape,device=device,dtype=torch.float64,generator=generator)).requires_grad_()
    sim=DifferentiableSimulation(p,AdjointOptions(checkpoints=3))
    expected=sim.reference(epsilon)
    weights=torch.randn(expected.shape,device=device,dtype=epsilon.dtype,generator=generator)
    expected_grad,=torch.autograd.grad((expected*weights).sum(),epsilon)
    actual=sim(epsilon)
    actual_grad,=torch.autograd.grad((actual.signals*weights).sum(),epsilon)
    torch.testing.assert_close(actual.signals,expected,rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-10,atol=1e-12)
    assert torch.linalg.vector_norm(actual_grad)>1e-5
    assert actual.report['peak_checkpoints']<=3
    assert actual.report['replayed_steps']>0


@pytest.mark.parametrize('storage',['device','host','disk'])
@pytest.mark.parametrize('checkpoints',[0,1,4])
def test_checkpoint_storage_and_schedule_equivalence(tmp_path,storage,checkpoints):
    p=project(steps=11)
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    options=AdjointOptions(checkpoints=checkpoints,storage=storage,
                           checkpoint_directory=tmp_path,disk_budget_bytes=16*1024**2,
                           host_budget_bytes=16*1024**2)
    sim=DifferentiableSimulation(p,options)
    reference=sim.reference(eps)
    expected,=torch.autograd.grad(reference.square().sum(),eps)
    result=sim(eps)
    got,=torch.autograd.grad(result.signals.square().sum(),eps)
    torch.testing.assert_close(got,expected,rtol=1e-11,atol=1e-14)
    assert result.report['peak_checkpoints']<=checkpoints
    assert list(tmp_path.iterdir())==[]


def test_directional_taylor_and_geometry_chain():
    p=project(steps=35)
    sim=DifferentiableSimulation(p,AdjointOptions(checkpoints=3))
    radius=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    center=torch.tensor([.03,-.02,0.],dtype=torch.float64,requires_grad=True)
    def loss(rad,position):
        eps=smooth_sphere_epsilon(p.region,rad,center=position,width=.08,inside=3.1,yee=True)
        return sim(eps).signals[:,0].square().mean()
    value=loss(radius,center)
    dr,dc=torch.autograd.grad(value,(radius,center))
    direction=torch.tensor([.4,-.3,0.],dtype=torch.float64)
    directional=dr+(dc*direction).sum()
    errors=[]
    for h in (1e-3,5e-4,2.5e-4):
        difference=loss(radius.detach()+h,center.detach()+h*direction)-value.detach()
        errors.append(abs((difference-h*directional).item()))
    assert errors[0]/errors[1]>3.8 and errors[1]/errors[2]>3.8
    h=1e-5
    central=(loss(radius.detach()+h,center.detach())-loss(radius.detach()-h,center.detach()))/(2*h)
    torch.testing.assert_close(dr,central,rtol=1e-6,atol=1e-9)


def test_matches_existing_native_cpu_forward_and_spectrum():
    p=project(steps=40)
    p.region.background_index=math.sqrt(1.6)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,requires_grad=True)
    actual=DifferentiableSimulation(p)(eps)
    expected=Simulation(p).run(cuda_graph=False)
    np.testing.assert_allclose(actual.signals.detach().numpy(),expected.signals,rtol=1e-12,atol=1e-14)
    f=np.array([1.7e14,2e14])
    got=actual.spectrum(f)
    for i,component in enumerate(actual.monitor_components):
        t=expected.times+(p.region.time_step/2 if component[0]=='H' else 0)
        reference=np.exp(-2j*np.pi*f[:,None]*t)@expected.signals[:,i]*p.region.time_step
        np.testing.assert_allclose(got[:,i].detach().numpy(),reference,rtol=1e-12,atol=1e-28)
    torch.autograd.grad(got.abs().square().sum(),eps)


def test_multiple_calls_accumulate_gradients_and_do_not_reuse_state():
    p=project(steps=13)
    sim=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,storage='host'))
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    a,b=sim(eps),sim(eps)
    actual,=torch.autograd.grad(a.signals.square().sum()+2*b.signals.square().sum(),eps)
    c=sim(eps)
    expected,=torch.autograd.grad(3*c.signals.square().sum(),eps)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)


def test_adam_updates_geometry_parameter():
    p=project(steps=30)
    sim=DifferentiableSimulation(p,AdjointOptions(checkpoints=3))
    radius=torch.nn.Parameter(torch.tensor(.25,dtype=torch.float64))
    optimizer=torch.optim.Adam([radius],lr=.005)
    losses=[]
    for _ in range(4):
        optimizer.zero_grad()
        epsilon=smooth_sphere_epsilon(p.region,radius,width=.09,inside=3.)
        loss=sim(epsilon).signals[:,0].square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        assert radius.grad is not None and torch.isfinite(radius.grad)
        optimizer.step()
    assert losses[-1]<losses[0]


def test_invalid_feature_and_memory_contracts(tmp_path):
    p=project()
    eps=torch.ones(p.region.shape,dtype=torch.float64)
    with pytest.raises(ValueError,match='checkpoint budget'):
        DifferentiableSimulation(p,AdjointOptions(storage='host',host_budget_bytes=1))(eps)
    with pytest.raises(ValueError,match='explicit disk'):
        AdjointOptions(storage='disk')
    p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(kind='bloch')
    # Complex fused backward is supported on CUDA. Construction is valid,
    # but attempting that backend with a CPU epsilon must still be rejected.
    model=DifferentiableSimulation(p,AdjointOptions(backward_kernel='fused'))
    with pytest.raises(ValueError,match='CUDA tensor'):
        model(eps)


def test_higher_order_is_explicitly_rejected():
    p=project(steps=10)
    eps=torch.full(p.region.shape,1.5,dtype=torch.float64,requires_grad=True)
    value=DifferentiableSimulation(p)(eps).signals.sum()
    with pytest.raises(RuntimeError,match='Higher-order'):
        torch.autograd.grad(value,eps,create_graph=True)


def test_long_history_budget_rejected_before_system_allocation(monkeypatch):
    gpu()
    import torchfdtd.differentiable as implementation
    p=project(steps=100000)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,device='cuda')
    fields_only=102*math.prod(p.region.shape)*eps.element_size()
    def unexpected_allocation(*args):
        raise AssertionError('A rejected budget must not allocate native state.')
    monkeypatch.setattr(implementation,'_System',unexpected_allocation)
    with pytest.raises(ValueError,match='GPU budget'):
        DifferentiableSimulation(p,AdjointOptions(checkpoints=0,gpu_budget_bytes=fields_only+10000))(eps)


def test_one_checkpoint_reduces_replay_without_changing_gradient():
    p=project(steps=30)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,requires_grad=True)
    results=[]
    for count in (0,1):
        result=DifferentiableSimulation(p,AdjointOptions(checkpoints=count))(eps)
        gradient,=torch.autograd.grad(result.signals.square().sum(),eps)
        results.append((result.report,gradient))
    torch.testing.assert_close(results[0][1],results[1][1],rtol=0,atol=0)
    assert results[0][0]['replayed_steps']==30*29//2
    assert results[1][0]['peak_checkpoints']==1
    assert results[1][0]['replayed_steps']<results[0][0]['replayed_steps']/2


@pytest.mark.parametrize('precision',['float32','float64'])
def test_three_tier_cuda_checkpoint_equivalence(tmp_path,precision):
    gpu()
    p=project(precision=precision,steps=30)
    dtype=torch.float32 if precision=='float32' else torch.float64
    eps=torch.full(p.region.shape,1.6,dtype=dtype,device='cuda',requires_grad=True)
    expected=DifferentiableSimulation(p,AdjointOptions(checkpoints=3))(eps)
    target,=torch.autograd.grad(expected.signals.square().mean(),eps)
    options=AdjointOptions(checkpoints=3,storage='hierarchical',device_checkpoints=1,host_checkpoints=1,
                           checkpoint_directory=tmp_path,disk_budget_bytes=16*1024**2,host_budget_bytes=16*1024**2)
    actual=DifferentiableSimulation(p,options)(eps)
    got,=torch.autograd.grad(actual.signals.square().mean(),eps)
    torch.testing.assert_close(got,target,rtol=0,atol=0)
    assert actual.report['checkpoint_tiers']==['disk','host','device']
    assert actual.report['peak_checkpoints_by_tier']==dict(device=1,host=1,disk=1)
    assert list(tmp_path.iterdir())==[]


def test_disk_failure_cleans_owned_files(tmp_path,monkeypatch):
    p=project(steps=12)
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,requires_grad=True)
    options=AdjointOptions(checkpoints=2,storage='disk',checkpoint_directory=tmp_path,disk_budget_bytes=16*1024**2)
    result=DifferentiableSimulation(p,options)(eps)
    def broken(path,**arrays):
        path.write_bytes(b'partial checkpoint')
        raise OSError('simulated disk write failure')
    monkeypatch.setattr(np,'savez',broken)
    with pytest.raises(OSError,match='simulated disk'):
        result.signals.sum().backward()
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('scene',['nonuniform','oneway_forward','oneway_backward'])
def test_mesh_metrics_and_fixed_plane_source_gradients(scene,device):
    if device=='cuda':gpu()
    if scene=='nonuniform':
        p=project(steps=32,periodic=True)
        nodes=[]
        for span,count in ((1.6,16),(1.5,15)):
            u=np.linspace(-1,1,count+1)
            nodes.append(tuple(span/2*(.8*u+.2*u**3)))
        p.region.mesh_type='explicit'
        p.region.mesh_coordinates=(*nodes,(-.7,.7))
        p.region.material_sampling='yee'
        p.sources[0].component='Hz'
        p.monitors[0].component='Ex'
        p.monitors[1].component='Hz'
        p.region.background_index=math.sqrt(1.6)
    else:
        from benchmarks.oneway_sources import plane_project
        p=plane_project(dimension='2d',index=1.5,
                        direction='+' if scene=='oneway_forward' else '-')
        p.region.size=(3.2,.4,.4)
        p.region.pml_cells=6
        p.region.steps=48
        p.sources[0].theta=43
        p.sources[0].phi=90
        p.monitors.append(Monitor(center=(.5,0,0),component='Hy'))
    p=Project.model_validate(p.model_dump())
    sim=DifferentiableSimulation(p,AdjointOptions(checkpoints=3))
    baseline=torch.full(p.region.shape,p.region.background_index**2,device=device,dtype=torch.float64)
    # Independent native forward, with matching homogeneous material and fixed drive.
    expected=Simulation(p).run(cuda_graph=False)
    np.testing.assert_allclose(sim(baseline).signals.detach().cpu().numpy(),expected.signals,
                               rtol=2e-11,atol=2e-13)
    design=torch.tensor(.1,device=device,dtype=torch.float64,requires_grad=True)
    mask=torch.zeros_like(baseline)
    if scene=='nonuniform':mask[5:11,5:10]=1
    else:
        from torchfdtd.solver import field_axes
        x=torch.tensor(field_axes(p.region,'Ez')[0],device=device)
        sign=1 if scene=='oneway_forward' else -1
        mask[(x*sign>.25)&(x*sign<.8)]=1
    def objective(parameter,reference=False):
        epsilon=baseline+parameter*mask
        signals=sim.reference(epsilon) if reference else sim(epsilon).signals
        return signals.square().sum()
    want,=torch.autograd.grad(objective(design,True),design)
    got,=torch.autograd.grad(objective(design),design)
    torch.testing.assert_close(got,want,rtol=2e-10,atol=1e-12)
    assert got.abs()>1e-8
    h=1e-5
    central=(objective(design.detach()+h)-objective(design.detach()-h))/(2*h)
    torch.testing.assert_close(got,central,rtol=1e-6,atol=1e-9)
