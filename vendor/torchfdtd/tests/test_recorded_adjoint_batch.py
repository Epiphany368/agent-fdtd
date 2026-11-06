"""Sequential public recorded-plane cases retain one active solver graph."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from torchfdtd.adjoint_batch import AdjointCase, AdjointBatchOptions, RecomputedAdjointBatch, _PreparedCase
from torchfdtd.execution_tuning import AdjointExecutionPolicy
from torchfdtd import AdjointOptions, ReversibleCPMLOptions
from test_reversible_cpml_planes import fixture, relative


def setup_cases(recorded=True):
    cases=[]
    for component in ('Ex','Ey'):
        project,base,_=fixture(True,component)
        policy=AdjointExecutionPolicy(device='cpu',**(
            dict(recorded=ReversibleCPMLOptions(trace_storage='cpu')) if recorded
            else dict(resident=AdjointOptions(checkpoints=2))))
        cases.append(AdjointCase(project,policy,frequency_hz=[.033/project.region.time_step],
            quadrature_counts={'incident':(2,2),'detector':(2,2)},block_size=7,
            fixed_background_epsilon=1.3 if recorded else None))
    base[:,:,:4]=1.3
    base[:,:,15:]=1.3
    return cases,base


def fields(result,dt):
    return torch.stack([torch.stack([p.fields for p in case.values()])
                        for case in result.cases])/dt


def coupled(value):
    summaries=(value.real+.37*value.imag).flatten(1).sum(1)
    return value.abs().square().mean()+.01*summaries.prod()


def test_recorded_two_case_coupled_vjp_retained_seeds_and_owner_release(monkeypatch):
    torch.set_num_threads(1)
    recorded,base=setup_cases()
    checkpoint,_=setup_cases(False)
    import torchfdtd.reversible_cpml as core
    original=core._System
    owners=[]
    def tracked(*args,**kwargs):
        assert all(ref() is None for ref in owners), 'Previous case retained its live system'
        system=original(*args,**kwargs)
        owners.append(weakref.ref(system))
        return system
    monkeypatch.setattr(core,'_System',tracked)
    rows=[]
    for specs in (checkpoint,recorded):
        parameter=base.clone().requires_grad_()
        mask=torch.zeros_like(parameter,dtype=torch.bool)
        mask[:,:,4:15]=True
        effective=torch.where(mask,parameter,torch.full_like(parameter,1.3))
        result=RecomputedAdjointBatch(specs)(effective)
        value=fields(result,specs[0].project.region.time_step)
        gradient=torch.autograd.grad(coupled(value),parameter,retain_graph=True)[0]
        generator=torch.Generator().manual_seed(125)
        seed=torch.randn((*value.shape[:-1],12),dtype=torch.complex64,generator=generator)[...,::2]
        assert seed.shape==value.shape and not seed.is_contiguous()
        seeded=torch.autograd.grad(value,parameter,seed,retain_graph=True)[0]
        repeated=torch.autograd.grad(value,parameter,seed)[0]
        assert torch.equal(seeded,repeated)
        assert result.report['forward_cases']==result.report['replayed_cases']==2
        assert result.report['full_case_graph_retention'] is False
        assert torch.count_nonzero(gradient[~mask])==0
        assert gradient.norm()>0
        rows.append((value.detach(),gradient,seeded))
        gc.collect()
        assert all(ref() is None for ref in owners)
    assert all(relative(a,b)<1e-4 for a,b in zip(rows[1],rows[0]))
    batch=RecomputedAdjointBatch(recorded)
    with torch.no_grad(): batch(base)
    gc.collect()
    assert all(ref() is None for ref in owners)
    assert len(owners)==10  # two forward + three two-case replays + two no-grad cases


def test_recorded_shared_budget_and_configuration_guards(monkeypatch):
    specs,base=setup_cases()
    batch=RecomputedAdjointBatch(specs)
    plan=batch.plan(base)
    assert plan['host_reservation_bytes']==plan['batch_overhead_bytes']+max(
        r['host_reservation_bytes'] for r in plan['case_reservations'])
    tight=RecomputedAdjointBatch(specs,AdjointBatchOptions(host_budget_bytes=plan['host_reservation_bytes']-1))
    def forbidden(*args,**kwargs): raise AssertionError('Allocated before complete batch admission')
    monkeypatch.setattr(_PreparedCase,'evaluate',forbidden)
    with pytest.raises(ValueError,match='shared host budget'): tight(base)
    original=specs[0].policy
    object.__setattr__(specs[0],'policy',replace(original,host_budget_bytes=original.host_budget_bytes-1))
    with pytest.raises(RuntimeError,match='configuration changed'): batch.plan(base)
    object.__setattr__(specs[0],'policy',original)
    object.__setattr__(specs[0],'fixed_background_epsilon',1.4)
    with pytest.raises(RuntimeError,match='configuration changed'): batch.plan(base)


def test_recorded_case_scope_and_saved_parameter_version():
    specs,base=setup_cases()
    for value in (None,1,True,float('nan'),float('inf'),.9):
        with pytest.raises(ValueError,match='Python float'):
            replace(specs[0],fixed_background_epsilon=value)
    with pytest.raises(ValueError,match='nondispersive'):
        replace(specs[0],parameter_indices=(0,1,2,3))
    ordinary,_=setup_cases(False)
    with pytest.raises(ValueError,match='requires a recorded'):
        replace(ordinary[0],fixed_background_epsilon=1.3)
    parameter=base.clone().requires_grad_()
    result=RecomputedAdjointBatch(specs[:1])(parameter)
    with torch.no_grad(): parameter.add_(.01)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        fields(result,specs[0].project.region.time_step).abs().square().sum().backward()
