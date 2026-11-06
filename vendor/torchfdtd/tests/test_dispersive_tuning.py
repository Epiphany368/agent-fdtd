"""Material-aware policy selection must preserve gradients and bound references."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from torchfdtd import (StreamedAdjointOptions, StreamedDispersiveSimulation,
    DispersiveSimulation, tune_streamed, tune_streamed_dispersive)
from torchfdtd.streamed_tuning import _ReferenceCache, _DispersiveTuningWorkload, _fit_default_policy
from test_streamed_dispersive import scene, inputs


@pytest.mark.parametrize('spectral', [False,True])
@pytest.mark.parametrize('cache_bytes', [0,64*1024**2])
def test_material_and_spectral_tuning_preserves_inputs(tmp_path,spectral,cache_bytes):
    p = scene(True,steps=13)
    values = inputs(p,'spatial',True)
    copies = [t.detach().clone() for t in values]
    for t in values:t.grad=torch.full_like(t,.3)
    base = StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2,checkpoints=1,
        state_directory=tmp_path,disk_budget_bytes=64*1024**2)
    candidates = [replace(base,host_budget_bytes=1),base,
                  replace(base,state_storage='disk',slab_width=6,temporal_depth=3,local_checkpoints=1)]
    settings = dict(frequency_hz=[.03/p.region.time_step,.06/p.region.time_step],
                    window=torch.linspace(.3,1.,p.region.steps,dtype=torch.float64)) if spectral else {}
    with torch.no_grad():
        selected = tune_streamed_dispersive(p,*values,candidates=candidates,probe_steps=10,repeats=1,
            reference_cache_bytes=cache_bytes,**settings)
    assert selected.options in candidates[1:]
    assert selected.report['candidates'][0]['status']=='rejected'
    assert selected.report['reference_cache_peak_bytes']<=cache_bytes
    assert selected.report['reference_policy_index']==1
    assert selected.report['calibration_steps']==[10,12,13]
    assert selected.report['gradient_comparison_coordinates']==('epsilon_inf','strength * dt^2','omega0 * dt','gamma * dt')
    assert selected.report['observation']==('online_spectrum' if spectral else 'time_history')
    if cache_bytes==0:assert selected.report['extra_reference_evaluations']>=4
    for actual,original in zip(values,copies):
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        torch.testing.assert_close(actual.grad,torch.full_like(actual,.3),rtol=0,atol=0)
    assert p.region.steps==13 and not list(tmp_path.iterdir())
    result = StreamedDispersiveSimulation(p,selected.options)(*values)
    reference = DispersiveSimulation(p).reference(*values)
    got = torch.autograd.grad(result.signals.abs().square().sum(),values)
    expected = torch.autograd.grad(reference.abs().square().sum(),values)
    dt=p.region.time_step
    for a,b,scale in zip(got,expected,(1.,dt*dt,dt,dt)):
        torch.testing.assert_close(a/scale,b/scale,rtol=1e-9,atol=1e-11)


@pytest.mark.parametrize('precision', ['float32','float64'])
def test_tuning_detects_material_vjp_error_hidden_by_si_units(monkeypatch,precision):
    import torchfdtd.streamed_dispersive as module
    p=scene(steps=13,precision=precision)
    values=inputs(p,'shared')
    base=StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2)
    def corrupt(project,options):
        model=StreamedDispersiveSimulation(project,options)
        def run(epsilon,strength,omega,gamma):
            result=model(epsilon,strength,omega,gamma)
            if options.temporal_depth==3:
                # Forward is exactly unchanged, and the SI derivative error is
                # below ordinary absolute tolerances. Dimensionless VJP is wrong.
                result.signals=result.signals+(gamma.sum()-gamma.detach().sum())*1e-16
            return result
        return run
    monkeypatch.setattr(module,'StreamedDispersiveSimulation',corrupt)
    with pytest.raises(AssertionError):
        tune_streamed_dispersive(p,*values,candidates=[base,replace(base,temporal_depth=3)],probe_steps=10,repeats=1)


def test_reference_cache_owns_compact_storage_and_releases_evictions():
    base=torch.arange(1000,dtype=torch.float64)
    gradient=base[:4]
    cache=_ReferenceCache(48)
    value=torch.ones(2,dtype=torch.float64)
    cache.put(1,(value,(gradient,)))
    ref=cache.get(1)
    assert ref[1][0].untyped_storage().nbytes()==32
    assert ref[1][0].data_ptr()!=gradient.data_ptr()
    owner=weakref.ref(ref[1][0]);del ref
    enabled=gc.isenabled();gc.disable()
    try:
        cache.put(2,(value,(gradient+1,)))
        assert owner() is None
        assert cache.get(1) is None and cache.bytes==cache.peak_bytes==48
        cache.put(3,(value,(base,)))
        assert cache.get(3) is None and cache.bytes==48
    finally:
        if enabled:gc.enable()


def test_budget_rejection_precedes_material_packing(monkeypatch):
    p=scene()
    values=inputs(p,'diagonal')
    monkeypatch.setattr(torch,'cat',lambda *a,**k:pytest.fail('Packed before full-workload admission'))
    with pytest.raises(ValueError,match='No streamed'):
        tune_streamed_dispersive(p,*values,candidates=[StreamedAdjointOptions(device='cpu',host_budget_bytes=1)])


def test_capacity_fit_shrinks_width_and_uses_only_configured_disk(tmp_path):
    p=scene(True)
    values=inputs(p,'shared')
    workload=_DispersiveTuningWorkload(p,*values)
    base=StreamedAdjointOptions(device='cpu',slab_width=16,temporal_depth=2)
    narrow=workload.reservation(replace(base,slab_width=3))
    limited=replace(base,host_budget_bytes=narrow['host_reservation_bytes'])
    fitted,detail=_fit_default_policy(workload,limited,0)
    assert fitted.slab_width==3 and fitted.temporal_depth==2 and detail['admission_probes']>1
    assert fitted.state_storage=='host'
    # Make state-bank lifetime dominate so reducing halos cannot rescue DRAM.
    base=replace(base,checkpoints=20,state_directory=tmp_path/'absent',disk_budget_bytes=64*1024**2)
    disk=replace(base,state_storage='disk',slab_width=1,temporal_depth=1)
    need=workload.reservation(disk)['host_reservation_bytes']
    fitted,_=_fit_default_policy(workload,replace(base,host_budget_bytes=need),0)
    assert fitted is not None and fitted.state_storage=='disk'
    missing,_=_fit_default_policy(workload,replace(base,host_budget_bytes=need,state_directory=None,disk_budget_bytes=None),0)
    assert missing is None and not (tmp_path/'absent').exists()


def test_nondispersive_spectrum_tuning_uses_full_window_and_budget():
    p=scene(True,steps=13)
    epsilon=inputs(p,'shared')[0]
    base=StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=3)
    result=tune_streamed(p,epsilon,candidates=[base],probe_steps=10,repeats=1,
        frequency_hz=[.04/p.region.time_step],window=torch.linspace(0,1,13,dtype=torch.float64))
    assert result.report['observation']=='online_spectrum'
    assert result.report['candidates'][0]['reservation']['source_and_output_history_bytes']>0
    with pytest.raises(ValueError,match='Window'):
        tune_streamed(p,epsilon,candidates=[base],frequency_hz=[1e14],window=[1.])


def test_default_search_fits_small_budget_and_varies_global_checkpoints():
    p=scene(True,steps=10)
    values=inputs(p,'shared')
    base=StreamedAdjointOptions(device='cpu',slab_width=16,temporal_depth=4,
        checkpoints=2)
    workload=_DispersiveTuningWorkload(p,*values)
    limit=workload.reservation(replace(base,slab_width=3,temporal_depth=1,checkpoints=4))['host_reservation_bytes']+128*1024
    base=replace(base,host_budget_bytes=limit)
    tuned=tune_streamed_dispersive(p,*values,options=base,probe_steps=10,repeats=1)
    plans=tuned.report['default_candidate_planning']
    assert any(row['fitted'] and row['fitted']['slab_width']<row['proposed']['slab_width'] for row in plans)
    assert {row['policy']['checkpoints'] for row in tuned.report['candidates'] if row['status']=='measured'}=={0,2,4}
    assert all(row['reservation']['host_reservation_bytes']+tuned.report['tuning_reference_reservation_bytes']<=limit
               for row in tuned.report['candidates'] if row['status']=='measured')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_policy_comparison_checks_async_material_gradient():
    p=scene(True,steps=10)
    values=inputs(p,'shared',True)
    base=StreamedAdjointOptions(device='cuda',slab_width=3,temporal_depth=2,checkpoints=1)
    selected=tune_streamed_dispersive(p,*values,candidates=[base,replace(base,tile_transfers='async',tile_buffers=2)],
        probe_steps=10,repeats=1,frequency_hz=[.03/p.region.time_step],reference_cache_bytes=0)
    assert all(row['status']=='measured' for row in selected.report['candidates'])
    assert selected.report['reference_cache_peak_bytes']==0
    assert all(t.grad is None for t in values)
