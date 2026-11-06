"""Recorded CPML routing through the actual calibrated periodic response."""
from dataclasses import asdict
import gc
import hashlib
import json
import weakref

import pytest
import torch
from torchfdtd import (AdjointBatchOptions, AdjointExecutionPolicy, AdjointOptions,
                       PeriodicLayerResponse, ReversibleCPMLOptions)

SPEC = dict(wavelength_um=.5,background_index=1.4,design_index=1.8,period_um=(.8,.8),
            height_um=.2,detector_offset_um=.5,theta_inside_rad=.1,phi_rad=.3)
BUDGET = 1024**3


def make(recorded=True, *, collar=1, dtype=torch.float32, **kwargs):
    options = dict(device='cpu',host_budget_bytes=BUDGET)
    if recorded:
        options['recorded'] = ReversibleCPMLOptions(trace_storage='cpu',collar_cells=collar,
            host_budget_bytes=BUDGET,resident_budget_bytes=BUDGET)
    else:
        options['resident'] = AdjointOptions(checkpoints=1,host_budget_bytes=BUDGET,
            resident_budget_bytes=BUDGET,backward_kernel='torch')
    return PeriodicLayerResponse(SPEC,density_shape=(2,2),dtype=dtype,
        policy=AdjointExecutionPolicy(**options),
        batch_options=AdjointBatchOptions(host_budget_bytes=BUDGET,gpu_budget_bytes=BUDGET),
        mesh=.1,steps=160,pml_cells=6,quadrature_counts=(4,4),forward_kernel='torch',**kwargs)


def objective(response):
    weights = response.new_tensor([[.1,.3,-.2,.7],[.4,-.1,.5,.2]])
    return (response*weights).sum()+.2*response.mean(0).square().sum()


def test_oblique_response_full_density_vjp_fd_and_case_lifetime(monkeypatch):
    torch.set_num_threads(1)
    import torchfdtd.reversible_cpml as core
    original = core._RecordedCPML.forward
    owners = []
    def tracked(ctx,*args):
        assert not any(owner() is not None for owner in owners), 'Previous case retained its solver graph'
        result = original(ctx,*args)
        owners.append(weakref.ref(ctx.system))
        return result
    monkeypatch.setattr(core._RecordedCPML,'forward',staticmethod(tracked))
    modules = [make(False),make(True)]
    density = torch.tensor([[.2,.4],[.5,.3]],dtype=torch.float32)
    rows = []
    for model in modules:
        value = density.clone().requires_grad_()
        response = model(value)
        loss = objective(response)
        gradient, = torch.autograd.grad(loss,value)
        assert torch.isfinite(response).all() and torch.isfinite(gradient).all()
        rows.append((response.detach(),gradient.detach()))
        del response,loss,gradient,value
        gc.collect()
    for baseline,actual in zip(rows[0],rows[1]):
        delta = (actual-baseline).double()
        assert delta.norm()/baseline.double().norm() < 1e-4
        assert delta.abs().max()/baseline.double().abs().max() < 1e-4
    report = modules[1].last_report['batch']
    assert report['forward_cases']==2 and report['replayed_cases']==2
    assert report['full_case_graph_retention'] is False
    direction = density.new_tensor([[.3,-.2],[.1,.4]])
    with torch.no_grad():
        fd = (objective(modules[1](density+.003*direction))
              -objective(modules[1](density-.003*direction)))/.006
    dot = (rows[1][1].double()*direction.double()).sum()
    assert abs(float(fd)-float(dot))/abs(float(dot)) < .01
    assert modules[1].last_report['reference_cache_hits']==1
    gc.collect()
    assert owners and not any(owner() is not None for owner in owners)
    # A populated cache must not bypass live admission or allocate fresh geometry.
    import torchfdtd.periodic_adjoint as implementation
    monkeypatch.setattr(implementation,'host_memory',lambda: dict(available_bytes=1))
    monkeypatch.setattr(implementation,'periodic_density_layer',lambda *a,**k: pytest.fail('Allocated material after rejected admission'))
    with pytest.raises(ValueError,match='budget|memory'):
        modules[1](density)


def test_component_support_rejection_precedes_material_and_case_allocation(monkeypatch):
    import torchfdtd.periodic_adjoint as implementation
    monkeypatch.setattr(implementation,'periodic_density_layer',lambda *a,**k: pytest.fail('Allocated material'))
    monkeypatch.setattr(implementation,'RecomputedAdjointBatch',lambda *a,**k: pytest.fail('Prepared cases before support rejection'))
    with pytest.raises(ValueError,match='layer support.*reconstruction interval'):
        make(collar=22)
    with pytest.raises(ValueError,match='FP32'):
        make(dtype=torch.float64)


def test_recorded_cache_namespace_and_legacy_identity():
    baseline,recorded = make(False),make(True)
    cases = baseline._batch._cases
    policy = asdict(cases[0].spec.policy)
    policy.pop('recorded',None)
    payload = dict(version='budgeted-periodic-response-1',
        projects=[case.spec.project.model_dump(mode='json') for case in cases],
        policy=policy,frequency=[299792458.0/(SPEC['wavelength_um']*1e-6)],
        quadrature_counts=(4,4),dtype=str(torch.float32))
    previous_key = hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    assert baseline._reference_key == previous_key
    assert recorded._reference_key != baseline._reference_key
    assert recorded._response_key != baseline._response_key
    assert all(case.spec.fixed_background_epsilon == SPEC['background_index']**2
               for case in recorded._batch._cases)
