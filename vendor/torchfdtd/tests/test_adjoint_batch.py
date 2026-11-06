"""Joint budgets and nonseparable inverse objectives across FDTD cases."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from torchfdtd import (AdjointCase, AdjointBatchOptions, RecomputedAdjointBatch,
    smooth_sphere_epsilon)
from torchfdtd.adjoint_batch import _PreparedCase, _BatchRun
from test_plane_execution import planes, policies
from test_streamed_dispersive import scene, inputs


def direct(case,parameters):
    model=case.policy.simulation(case.project,dispersive=len(case.parameter_indices)==4,
        quadrature_counts=case.quadrature_counts)
    args=tuple(parameters[i] for i in case.parameter_indices)
    if case.project.monitors[0].kind=='field':
        return model(*args,frequency_hz=case.frequency_hz,block_size=case.block_size)
    if case.frequency_hz is not None:return model.spectrum(*args,frequency_hz=case.frequency_hz,block_size=case.block_size)
    return model(*args)


def values(result,project):
    if isinstance(result,dict):return torch.cat([plane.fields.flatten() for plane in result.values()])/(project.region.steps*project.region.time_step)
    if hasattr(result,'fields'):return result.fields.flatten()/(project.region.steps*project.region.time_step)
    return result.signals.flatten()


def objective(results,specs):
    # Different result layouts feed one coupled scalar. Cross-case terms make
    # independent scalar losses an incorrect substitute for its full VJP.
    summaries=torch.stack([torch.stack((v.abs().square().sum(),v.real.sum()))
        for v in (values(r,s.project) for r,s in zip(results,specs))])
    return summaries.sum(0).square().sum()+(summaries[0]*summaries[-1]).sum()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dispersive',[False,True])
@pytest.mark.parametrize('observation',['history','spectrum','planes'])
def test_coupled_case_vjp_with_shared_materials(device,dispersive,observation):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    specs=[]
    for i,policy in enumerate(policies(device)):
        p=planes(True) if observation=='planes' else scene(True,steps=11)
        p.region.bloch_phase=(.2+.3*i,0,0)
        specs.append(AdjointCase(p,policy,(0,1,2,3) if dispersive else (0,),
            None if observation=='history' else [1e14,2e14]))
    parameters=inputs(specs[0].project,'shared',True)
    if not dispersive:parameters=parameters[:1]
    expected=tuple(direct(s,parameters) for s in specs)
    expected_grad=torch.autograd.grad(objective(expected,specs),parameters)
    batch=RecomputedAdjointBatch(specs,AdjointBatchOptions(host_budget_bytes=256*1024**2,gpu_budget_bytes=256*1024**2))
    report=batch.plan(*parameters)
    assert report['forward_cases']==0
    actual=batch(*parameters)
    for a,e,s in zip(actual.cases,expected,specs):
        torch.testing.assert_close(values(a,s.project),values(e,s.project),rtol=1e-11,atol=1e-12)
    gradients=torch.autograd.grad(objective(actual.cases,specs),parameters,retain_graph=True)
    repeated=torch.autograd.grad(objective(actual.cases,specs),parameters)
    scales=(1.,specs[0].project.region.time_step**-2,specs[0].project.region.time_step**-1,specs[0].project.region.time_step**-1)
    for a,b,e,scale in zip(gradients,repeated,expected_grad,scales):
        torch.testing.assert_close(a*scale,e*scale,rtol=1e-8,atol=1e-9)
        torch.testing.assert_close(b,a,rtol=0,atol=0)
    assert actual.report['replayed_cases']==2
    assert actual.report['host_reservation_bytes']==actual.report['batch_overhead_bytes']+max(
        r['host_reservation_bytes'] for r in actual.report['case_reservations'])


def test_heterogeneous_case_bindings_geometry_and_unused_parameter(tmp_path):
    radius=torch.tensor(.24,dtype=torch.float64,requires_grad=True)
    material=torch.tensor([.6,1.3,.2],dtype=torch.float64,requires_grad=True)
    unused=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    first=planes(True,'3d')
    second=scene(False,steps=13)
    specs=[AdjointCase(first,policies()[0],(0,),[1e14],{'first':(2,3),'second':(3,2)}),
           AdjointCase(second,policies(directory=tmp_path)[1],(1,2,3,4),[1.5e14,2e14])]
    def parameters():
        return (smooth_sphere_epsilon(first.region,radius,width=.1,inside=2.),
            smooth_sphere_epsilon(second.region,radius,width=.1,inside=2.),
            material[:1]*1e30,material[1]*1e15,material[2]*1e15,unused)
    expected=tuple(direct(s,parameters()) for s in specs)
    want=torch.autograd.grad(objective(expected,specs),(radius,material,unused),allow_unused=True)
    result=RecomputedAdjointBatch(specs,AdjointBatchOptions(disk_budget_bytes=128*1024**2))(*parameters())
    got=torch.autograd.grad(objective(result.cases,specs),(radius,material,unused),allow_unused=True)
    for a,b in zip(got[:2],want[:2]):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    assert got[2] is want[2] is None
    assert result.cases[0]['first'].shape==(1,2,3)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('limitation',['output','host','gpu','disk','bad_later_shape'])
def test_all_case_admission_before_any_solver_allocation(tmp_path,monkeypatch,limitation):
    p=planes()
    first,second=policies('cuda' if limitation=='gpu' else 'cpu',tmp_path if limitation=='disk' else None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1024**3,2*1024**3))
    specs=[AdjointCase(p,first,frequency_hz=[1e14]),AdjointCase(p,second,frequency_hz=[2e14])]
    parameters=(inputs(p,'shared')[0],)
    options=AdjointBatchOptions()
    if limitation=='bad_later_shape':
        specs[1]=replace(specs[1],parameter_indices=(1,))
        parameters+=(torch.ones(2,dtype=torch.float64),)
    else:
        name={'output':'output_budget_bytes','host':'host_budget_bytes','gpu':'gpu_budget_bytes','disk':'disk_budget_bytes'}[limitation]
        options=replace(options,**{name:1})
    batch=RecomputedAdjointBatch(specs,options)
    def forbidden(*a,**kw):pytest.fail('Started an earlier case before admitting the complete batch')
    monkeypatch.setattr(_PreparedCase,'evaluate',forbidden)
    with pytest.raises(ValueError):batch(*parameters)
    assert list(tmp_path.iterdir())==[]


def test_joint_host_budget_includes_outputs_and_gradient_accumulators():
    p=planes()
    spec=AdjointCase(p,policies()[0],frequency_hz=[1e14])
    epsilon=inputs(p,'shared')[0]
    batch=RecomputedAdjointBatch([spec]*3)
    plan=batch.plan(epsilon)
    max_case=max(r['host_reservation_bytes'] for r in plan['case_reservations'])
    assert plan['host_reservation_bytes']>max_case+3*plan['output_bytes']
    limited=RecomputedAdjointBatch([spec]*3,AdjointBatchOptions(host_budget_bytes=max_case))
    with pytest.raises(ValueError,match='shared host budget'):limited(epsilon)


def test_replay_rechecks_live_shared_memory(monkeypatch):
    p=scene(steps=10)
    batch=RecomputedAdjointBatch([AdjointCase(p,policies()[0])])
    epsilon=inputs(p,'shared',True)[0]
    result=batch(epsilon)
    monkeypatch.setattr('torchfdtd.adjoint_batch.host_memory',lambda:{'available_bytes':1})
    with pytest.raises(ValueError,match='shared host budget'):result.cases[0].signals.sum().backward()


def test_result_shape_dtype_and_unused_case_replay():
    p=scene(steps=10)
    q=p.model_copy(deep=True);q.region.precision='float32'
    specs=[AdjointCase(p,policies()[0]),AdjointCase(q,policies()[1],(1,),[1e14])]
    a=inputs(p,'shared',True)[0]
    b=inputs(q,'shared',True)[0]
    result=RecomputedAdjointBatch(specs)(a,b)
    assert result.cases[0].signals.dtype==torch.float64
    assert result.cases[1].fields.dtype==torch.complex64
    got=torch.autograd.grad(result.cases[0].signals.square().sum(),(a,b),allow_unused=True)
    assert got[0].norm()>0 and got[1] is None
    assert result.report['replayed_cases']==1


def test_fixed_configuration_and_first_order_guards():
    p=scene(steps=10)
    batch=RecomputedAdjointBatch([AdjointCase(p,policies()[0])])
    epsilon=inputs(p,'shared',True)[0]
    result=batch(epsilon)
    with pytest.raises(RuntimeError,match='first-order'):
        torch.autograd.grad(result.cases[0].signals.sum(),epsilon,create_graph=True)
    result=batch(epsilon)
    batch._cases[0].project.sources[0].amplitude*=2
    with pytest.raises(RuntimeError,match='configuration changed'):result.cases[0].signals.sum().backward()


def test_no_overlapping_case_systems_with_cyclic_gc_disabled(monkeypatch):
    from torchfdtd.differentiable import _System
    live=[]
    original=_System.__init__
    def track(self,*args,**kwargs):
        original(self,*args,**kwargs)
        live.append(weakref.ref(self))
    original_admit=_BatchRun.admit
    def guard(self,*args):
        assert not any(ref() is not None for ref in live)
        return original_admit(self,*args)
    monkeypatch.setattr(_System,'__init__',track)
    monkeypatch.setattr(_BatchRun,'admit',guard)
    p=planes(True)
    batch=RecomputedAdjointBatch([AdjointCase(p,policy,frequency_hz=[1e14]) for policy in policies()])
    epsilon=inputs(p,'shared',True)[0]
    gc.collect()
    enabled=gc.isenabled();gc.disable()
    try:
        result=batch(epsilon)
        assert not any(ref() is not None for ref in live)
        sum(plane.flux().sum() for case in result.cases for plane in case.values()).backward()
        assert not any(ref() is not None for ref in live)
    finally:
        if enabled:gc.enable()


def test_replay_detects_drift_in_tiny_si_spectra(monkeypatch):
    p=scene(steps=11)
    batch=RecomputedAdjointBatch([AdjointCase(p,policies()[0],frequency_hz=[1e14])])
    original=_PreparedCase.evaluate
    def corrupted(self,parameters):
        result=original(self,parameters)
        return replace(result,fields=result.fields*1.1)
    monkeypatch.setattr(_PreparedCase,'evaluate',corrupted)
    epsilon=inputs(p,'shared',True)[0]
    result=batch(epsilon)
    assert result.cases[0].fields.abs().max()<1e-9
    with pytest.raises(RuntimeError,match='drifted'):
        (result.cases[0].fields.real/p.region.time_step).sum().backward()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_coupled_normalized_flux_preserves_geometry_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=planes()
    p.region.precision='float32'
    specs=[AdjointCase(p,policy,frequency_hz=[1e14,2e14]) for policy in policies(device)]
    batch=RecomputedAdjointBatch(specs)
    mask=torch.zeros(p.region.shape,dtype=torch.float32);mask[6:9,5:8]=1
    with torch.no_grad():reference=batch(torch.ones_like(mask))
    def loss(result):
        ratios=torch.stack([case['first'].normalized_flux(ref['first']) for case,ref in zip(result,reference.cases)])
        return ratios.mean()+.1*ratios.sum(0).square().sum()
    variable=torch.tensor(.5,dtype=torch.float32,requires_grad=True)
    expected=tuple(direct(s,(1+variable*mask,)) for s in specs)
    want,=torch.autograd.grad(loss(expected),variable)
    result=batch(1+variable*mask)
    got,=torch.autograd.grad(loss(result.cases),variable)
    torch.testing.assert_close(got,want,rtol=1e-4,atol=1e-6)
    assert torch.isfinite(got) and abs(got)>1e-6
