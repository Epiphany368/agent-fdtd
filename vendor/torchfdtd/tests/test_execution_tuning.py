"""Unified selection must preserve geometry gradients across device/tier moves."""
from dataclasses import replace

import pytest
import torch

from torchfdtd import (AdjointExecutionPolicy, AdjointOptions, StreamedAdjointOptions,
    DifferentiableSimulation, DispersiveSimulation, tune_adjoint_execution)
from torchfdtd.execution_tuning import _resident_reservation
from test_streamed_dispersive import scene,inputs


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dispersive',[False,True])
@pytest.mark.parametrize('spectral',[False,True])
def test_selection_and_cpu_design_chain_match_resident_oracle(device,dispersive,spectral):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=scene(True,steps=11)
    parameters=inputs(p,'shared',True) if dispersive else (inputs(p,'shared',True)[0],)
    for value in parameters:value.grad=torch.full_like(value,3.)
    original=tuple(v.detach().clone() for v in parameters)
    streamed=StreamedAdjointOptions(device=device,slab_width=3,temporal_depth=2,
                                   host_budget_bytes=64*1024**2,gpu_budget_bytes=64*1024**2)
    resident=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=2,
        gpu_budget_bytes=64*1024**2,backward_kernel='fused' if device=='cuda' else 'torch'),
        device=device,host_budget_bytes=64*1024**2)
    candidates=[replace(resident,host_budget_bytes=1),resident,
                AdjointExecutionPolicy(streamed=streamed,device=device,host_budget_bytes=64*1024**2)]
    settings=dict(frequency_hz=[1e14,2e14],window=torch.linspace(.1,1.,11,dtype=torch.float64)) if spectral else {}
    with torch.no_grad():
        selection=tune_adjoint_execution(p,*parameters,candidates=candidates,
            probe_steps=10,repeats=1,reference_cache_bytes=0,**settings)
    assert selection.report['candidates'][0]['status']=='rejected'
    assert all(r['status']=='measured' for r in selection.report['candidates'][1:])
    for value,before in zip(parameters,original):
        torch.testing.assert_close(value,before,rtol=0,atol=0)
        torch.testing.assert_close(value.grad,torch.full_like(value,3.),rtol=0,atol=0)
    expected_index=min((row['selection_score_seconds'],row['index'])
        for row in selection.report['candidates'] if row['status']=='measured')[1]
    assert selection.report['selected_index']==expected_index
    # Exercise each admitted execution wrapper, not only the timing winner.
    reference=(DispersiveSimulation if dispersive else DifferentiableSimulation)(p)
    expected=reference.spectrum(*parameters,**settings) if spectral else reference(*parameters)
    values=expected.fields/(11*p.region.time_step) if spectral else expected.signals
    expected_grad=torch.autograd.grad(values.abs().square().sum(),parameters)
    for policy in candidates[1:]:
        model=policy.simulation(p,dispersive=dispersive)
        result=model.spectrum(*parameters,**settings) if spectral else model(*parameters)
        actual=result.fields/(11*p.region.time_step) if spectral else result.signals
        assert actual.device.type=='cpu'
        torch.testing.assert_close(actual,values,rtol=1e-10,atol=1e-12)
        gradients=torch.autograd.grad(actual.abs().square().sum(),parameters)
        scales=(1.,p.region.time_step**-2,p.region.time_step**-1,p.region.time_step**-1)
        for a,b,scale in zip(gradients,expected_grad,scales):
            torch.testing.assert_close(a*scale,b*scale,rtol=1e-8,atol=1e-10)


def test_default_search_includes_both_execution_families():
    p=scene(steps=10)
    epsilon=inputs(p,'shared')[0]
    selection=tune_adjoint_execution(p,epsilon,options=StreamedAdjointOptions(
        device='cpu',slab_width=3,temporal_depth=2,host_budget_bytes=64*1024**2),
        probe_steps=10,repeats=1)
    measured=[row['policy'] for row in selection.report['candidates'] if row['status']=='measured']
    assert any(row['resident'] is not None for row in measured)
    assert any(row['streamed'] is not None for row in measured)
    assert selection.report['default_candidate_planning']
    assert selection.report['reference_cache_peak_bytes']<=64*1024**2


def test_unified_host_budget_covers_cpu_solver_not_only_checkpoints():
    p=scene()
    policy=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=0,host_budget_bytes=1),device='cpu',host_budget_bytes=1)
    with pytest.raises(ValueError,match='unified host budget'):
        _resident_reservation(p,(p.region.shape,),policy)


def test_cuda_copy_reservation_denies_before_material_copy(monkeypatch):
    p=scene(True)
    policy=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=0,gpu_budget_bytes=64*1024**2),
        device='cuda',host_budget_bytes=64*1024**2)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1024**3,2*1024**3))
    report=_resident_reservation(p,(p.region.shape,),policy)
    assert report['gpu_transfer_reservation_bytes']==2*torch.tensor(p.region.shape).prod().item()*8
    limited=replace(policy,resident=replace(policy.resident,gpu_budget_bytes=report['gpu_reservation_bytes']-1))
    model=limited.simulation(p)
    def forbidden(*a,**kw):pytest.fail('Copied a CPU material before complete admission')
    monkeypatch.setattr(torch.Tensor,'to',forbidden)
    with pytest.raises(ValueError,match='transfer reservation'):
        model(torch.ones(p.region.shape,dtype=torch.float64))


def test_large_streamed_project_is_not_admitted_as_resident():
    p=scene()
    p.region.memory_mode='streamed'
    p.region.dimension='3d'
    p.region.size=(100.,100.,100.)
    policy=AdjointExecutionPolicy(resident=AdjointOptions(),device='cpu')
    # No cell cap applies on the Python API; one billion cells exceed the 32-bit field indices.
    with pytest.raises(ValueError,match='signed 32-bit'):
        policy.simulation(p)


def test_uncached_reference_does_not_keep_previous_resident_system(monkeypatch):
    import gc
    import weakref
    from torchfdtd.differentiable import _System
    p=scene(steps=11)
    epsilon=inputs(p,'shared')[0]
    live=[]
    original=_System.__init__
    def guarded(self,*args,**kwargs):
        assert not any(reference() is not None for reference in live)
        original(self,*args,**kwargs)
        live.append(weakref.ref(self))
    monkeypatch.setattr(_System,'__init__',guarded)
    first=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=1),device='cpu')
    second=replace(first,resident=replace(first.resident,checkpoints=2))
    gc.collect()
    enabled=gc.isenabled()
    gc.disable()
    try:
        tuned=tune_adjoint_execution(p,epsilon,candidates=[first,second],
            probe_steps=10,repeats=1,reference_cache_bytes=0)
        assert tuned.report['extra_reference_evaluations']>0
        assert not any(reference() is not None for reference in live)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('setting,value',[('probe_steps',0),('repeats',False),
    ('max_calibration_steps','10'),('reference_cache_bytes',-1)])
def test_bad_calibration_settings_rejected_before_policy_generation(monkeypatch,setting,value):
    p=scene()
    def forbidden(*a,**kw):pytest.fail('Generated policies before validating calibration settings')
    monkeypatch.setattr('torchfdtd.execution_tuning._generated_candidates',forbidden)
    with pytest.raises(ValueError,match=setting):
        tune_adjoint_execution(p,inputs(p,'shared')[0],**{setting:value})
