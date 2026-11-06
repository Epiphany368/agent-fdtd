"""Resident admission must happen before field allocation and ADE packing."""
from dataclasses import replace

import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, DispersiveSimulation, FieldMonitor,
                         estimate_adjoint_memory)
from test_differentiable import project
from test_streamed_dispersive import scene, inputs


@pytest.mark.parametrize('storage',['device','host','disk','hierarchical'])
@pytest.mark.parametrize('dispersive',[False,True])
@pytest.mark.parametrize('spectral',[False,True])
def test_resident_plan_matches_runtime_and_exact_restart(tmp_path,storage,dispersive,spectral):
    p=scene(True,steps=12)
    design=inputs(p,'diagonal',True) if dispersive else (inputs(p,'shared',True)[0],)
    options=AdjointOptions(checkpoints=3,storage=storage,checkpoint_directory=tmp_path/'banks',
        host_budget_bytes=64*1024**2,disk_budget_bytes=64*1024**2,
        device_checkpoints=1 if storage=='hierarchical' else 0,host_checkpoints=1 if storage=='hierarchical' else 0)
    settings=dict(frequency_hz=[1e14,2e14],window=torch.linspace(.2,1.,12,dtype=torch.float64),block_size=4) if spectral else {}
    shape_args=dict(parameter_shapes=tuple(tuple(v.shape) for v in design)) if dispersive else {}
    estimate=estimate_adjoint_memory(p,options,**shape_args,**settings)
    assert not (tmp_path/'banks').exists()
    model=(DispersiveSimulation if dispersive else DifferentiableSimulation)(p,options)
    result=model.spectrum(*design,**settings) if spectral else model(*design)
    assert all(result.report[key]==value for key,value in estimate.items())
    assert estimate['restart_state_bytes']==result.report['restart_bytes']
    values=result.fields if spectral else result.signals
    gradients=torch.autograd.grad(values.abs().square().sum(),design)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert estimate['host_reservation_bytes']>=estimate['host_checkpoint_reservation_bytes']
    assert not (tmp_path/'banks').exists() or not list((tmp_path/'banks').iterdir())


@pytest.mark.parametrize('failure',['host_checkpoint','host_available','disk_budget','disk_available'])
@pytest.mark.parametrize('dispersive',[False,True])
def test_denial_precedes_fields_material_carrier_and_scratch(tmp_path,monkeypatch,failure,dispersive):
    p=scene(steps=12)
    values=inputs(p,'shared')
    options=AdjointOptions(storage='host',host_budget_bytes=64*1024**2)
    if failure=='host_checkpoint':options=replace(options,host_budget_bytes=1)
    if failure=='host_available':
        monkeypatch.setattr('torchfdtd.adjoint_memory.host_memory',lambda:dict(available_bytes=1))
    if failure.startswith('disk'):
        options=replace(options,storage='disk',checkpoint_directory=tmp_path/'absent',
                        disk_budget_bytes=1 if failure=='disk_budget' else 64*1024**2)
    if failure=='disk_available':monkeypatch.setattr('torchfdtd.adjoint_memory.disk_free',lambda _:1)
    def forbidden(*args,**kwargs):pytest.fail('Allocated a field or packed material before admission')
    monkeypatch.setattr('torchfdtd.differentiable._System',forbidden)
    monkeypatch.setattr('torchfdtd.dispersive_adjoint._DispersiveSystem',forbidden)
    monkeypatch.setattr(torch,'cat',forbidden)
    model=(DispersiveSimulation if dispersive else DifferentiableSimulation)(p,options)
    with pytest.raises(ValueError,match='budget|available'):
        model(*values) if dispersive else model(values[0])
    assert not (tmp_path/'absent').exists()


def test_metadata_cuda_plan_queries_capacity_without_cuda_tensors(monkeypatch):
    p=scene(True)
    options=AdjointOptions(checkpoints=4,storage='host',checkpoint_transfers='async',staging_slots=2,
        gpu_budget_bytes=64*1024**2,host_budget_bytes=64*1024**2)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1024**3,2*1024**3))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,9))
    def forbidden(*args,**kwargs):pytest.fail('Allocated a resident system while planning')
    monkeypatch.setattr('torchfdtd.differentiable._System',forbidden)
    estimate=estimate_adjoint_memory(p,options,device='cuda',parameter_shapes=(p.region.shape,(2,),(),()),frequency_hz=[1e14])
    assert estimate['host_checkpoint_reservation_bytes']==6*estimate['restart_state_bytes']
    assert estimate['gpu_reservation_bytes']==estimate['memory_reservation_bytes']
    with pytest.raises(ValueError,match='GPU budget'):
        estimate_adjoint_memory(p,replace(options,gpu_budget_bytes=1),device='cuda')


def test_planner_preserves_checkpoint_budget_semantics_and_rechecks_ram(monkeypatch):
    p=project(steps=10)
    options=AdjointOptions(storage='device',host_budget_bytes=1)
    plan=estimate_adjoint_memory(p,options)
    assert plan['host_checkpoint_reservation_bytes']==0
    assert plan['host_reservation_bytes']>options.host_budget_bytes
    monkeypatch.setattr('torchfdtd.adjoint_memory.host_memory',lambda:dict(available_bytes=plan['host_reservation_bytes']))
    with pytest.raises(ValueError,match='available host memory'):estimate_adjoint_memory(p,options)


@pytest.mark.parametrize('dispersive',[False,True])
@pytest.mark.parametrize('unsupported',['plane','auto_shutoff'])
def test_metadata_plan_rejects_unsupported_workflows(dispersive,unsupported):
    p=scene()
    if unsupported=='plane':p.monitors=[FieldMonitor(size=(0,.2,.2))]
    else:p.region.run_control.auto_shutoff=True
    shapes=dict(parameter_shapes=(p.region.shape,(1,),(),())) if dispersive else {}
    with pytest.raises(ValueError,match='point monitors|fixed number'):
        estimate_adjoint_memory(p,**shapes)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_ade_denial_precedes_packing_and_native_fields(monkeypatch):
    p=scene(True)
    values=tuple(t.to('cuda') for t in inputs(p,'diagonal'))
    model=DispersiveSimulation(p,AdjointOptions(gpu_budget_bytes=1))
    monkeypatch.setattr(torch,'cat',lambda *a,**kw:pytest.fail('Packed before GPU admission'))
    with pytest.raises(ValueError,match='GPU budget'):model(*values)
