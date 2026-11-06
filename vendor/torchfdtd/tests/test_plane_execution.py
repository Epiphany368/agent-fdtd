"""Plane/flux execution policies preserve material and geometry gradients."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from torchfdtd import (AdjointOptions, AdjointExecutionPolicy, FieldMonitor,
    Monitor, StreamedAdjointOptions, tune_adjoint_execution)
from torchfdtd.adjoint_planes import DifferentiablePlaneSimulation
from torchfdtd.dispersive_adjoint import DispersivePlaneSimulation
from torchfdtd.execution_tuning import _resident_reservation, _StreamedPlanesFromHost
from test_streamed_dispersive import scene, inputs


def planes(bloch=False,dimension='2d'):
    p=scene(bloch,steps=11)
    p.region.dimension=dimension
    p.region.cuda_kernel='fused'
    p.monitors=[FieldMonitor(id='first',center=(.12,0,0),size=(0,.65,.55),normal='x',downsample=2),
                FieldMonitor(id='second',center=(.21,.03,0),size=(0,.55,.45),normal='x',downsample=2)]
    return p


def policies(device='cpu',directory=None):
    budget=128*1024**2
    resident=AdjointExecutionPolicy(resident=AdjointOptions(checkpoints=2,
        resident_budget_bytes=budget,gpu_budget_bytes=budget,
        backward_kernel='fused' if device=='cuda' else 'torch'),device=device,host_budget_bytes=budget)
    streamed=StreamedAdjointOptions(device=device,slab_width=4,temporal_depth=3,
        host_budget_bytes=budget,gpu_budget_bytes=budget,
        tile_transfers='async' if device=='cuda' else 'sync',
        state_storage='disk' if directory else 'host',state_directory=directory,
        disk_budget_bytes=budget if directory else None)
    return resident,AdjointExecutionPolicy(streamed=streamed,device=device,host_budget_bytes=budget)


def objective(results,dt):
    value=0
    for plane in results.values():
        field=plane.fields/dt
        value=value+(field.real+.3*field.imag).square().sum()
        value=value+plane.flux().sum()/plane.weights.sum()/dt**2
    return value


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dispersive',[False,True])
@pytest.mark.parametrize('bloch,dimension',[(False,'2d'),(True,'2d'),(True,'3d')])
def test_plane_policy_selection_and_full_material_vjp(device,dispersive,bloch,dimension):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=planes(bloch,dimension)
    values=inputs(p,'shared',True)
    if not dispersive:values=values[:1]
    for value in values:value.grad=torch.full_like(value,3.)
    original=tuple(v.detach().clone() for v in values)
    candidates=policies(device)
    frequency=[.025/p.region.time_step,.06/p.region.time_step]
    with torch.no_grad():
        selected=tune_adjoint_execution(p,*values,candidates=[replace(candidates[0],host_budget_bytes=1),*candidates],
            frequency_hz=frequency,probe_steps=10,repeats=1,reference_cache_bytes=0)
    assert selected.report['observation']=='fixed_plane_spectrum_and_flux'
    assert [r['status'] for r in selected.report['candidates']]==['rejected','measured','measured']
    for before,value in zip(original,values):
        torch.testing.assert_close(before,value,rtol=0,atol=0)
        torch.testing.assert_close(value.grad,torch.full_like(value,3.),rtol=0,atol=0)
    native=(DispersivePlaneSimulation if dispersive else DifferentiablePlaneSimulation)(p)
    expected=native(*values,frequency_hz=frequency)
    expected_gradient=torch.autograd.grad(objective(expected,p.region.time_step),values)
    # Both execution modes, including CUDA transfers, must preserve the graph.
    for policy in candidates:
        actual=policy.simulation(p,dispersive=dispersive)(*values,frequency_hz=frequency)
        for key,plane in actual.items():
            assert plane.fields.device.type=='cpu'
            torch.testing.assert_close(plane.fields/p.region.time_step,expected[key].fields/p.region.time_step,rtol=1e-9,atol=1e-11)
            torch.testing.assert_close(plane.normalized_flux(expected[key]),
                expected[key].normalized_flux(expected[key]),rtol=1e-9,atol=1e-11)
            assert plane.report['execution_reservation']['plane_layout_reservation_bytes']>0
        gradient=torch.autograd.grad(objective(actual,p.region.time_step),values)
        scales=(1.,p.region.time_step**-2,p.region.time_step**-1,p.region.time_step**-1)
        for a,b,scale in zip(gradient,expected_gradient,scales):
            torch.testing.assert_close(a*scale,b*scale,rtol=1e-8,atol=1e-9)
    assert selected.simulation(p)(*values,frequency_hz=frequency).keys()==expected.keys()


def test_quadrature_selection_survives_file_policy_and_geometry_graph(tmp_path):
    p=planes(True,'3d')
    counts={'first':(2,3),'second':(3,2)}
    radius=torch.tensor(.24,dtype=torch.float64,requires_grad=True)
    from torchfdtd import smooth_sphere_epsilon
    epsilon=smooth_sphere_epsilon(p.region,radius,width=.1,inside=2.)
    frequency=[.035/p.region.time_step]
    candidates=policies(directory=tmp_path)
    selected=tune_adjoint_execution(p,epsilon,candidates=candidates,frequency_hz=frequency,
        quadrature_counts=counts,probe_steps=10,repeats=1)
    assert all(r['status']=='measured' for r in selected.report['candidates'])
    counts['first']=(5,5)
    model=selected.simulation(p)
    result=model(epsilon,frequency_hz=frequency)
    assert result['first'].fields.shape==(1,6,6)
    got,=torch.autograd.grad(objective(result,p.region.time_step),radius)
    expected=DifferentiablePlaneSimulation(p,quadrature_counts=selected.quadrature_counts)(
        smooth_sphere_epsilon(p.region,radius,width=.1,inside=2.),frequency)
    want,=torch.autograd.grad(objective(expected,p.region.time_step),radius)
    torch.testing.assert_close(got,want,rtol=1e-8,atol=1e-10)
    assert abs(got)>1e-6
    assert list(tmp_path.iterdir())==[]


def test_generated_plane_candidates_include_both_families():
    p=planes()
    selected=tune_adjoint_execution(p,inputs(p,'shared')[0],
        options=policies()[1].streamed,frequency_hz=[1e14],probe_steps=10,repeats=1)
    measured=[r['policy'] for r in selected.report['candidates'] if r['status']=='measured']
    assert any(r['resident'] is not None for r in measured)
    assert any(r['streamed'] is not None for r in measured)
    assert selected.report['default_candidate_planning']


@pytest.mark.parametrize('kind',['mixed','point_quadrature','missing_frequency','window'])
def test_invalid_plane_contract_rejected_before_tuning(monkeypatch,kind):
    p=planes()
    settings=dict(frequency_hz=[1e14])
    if kind=='mixed':p.monitors.append(Monitor())
    elif kind=='point_quadrature':p.monitors=[Monitor()];settings['quadrature_counts']={}
    elif kind=='missing_frequency':settings={}
    else:settings['window']=torch.ones(p.region.steps)
    def forbidden(*a,**kw):pytest.fail('Generated candidates for an unsupported observation contract')
    monkeypatch.setattr('torchfdtd.execution_tuning._generated_candidates',forbidden)
    with pytest.raises(ValueError):tune_adjoint_execution(p,inputs(p,'shared')[0],**settings)


@pytest.mark.parametrize('mode',['resident','streamed'])
def test_plane_layout_denied_before_state_or_material_packing(monkeypatch,mode):
    from torchfdtd.plane_execution import plane_reservation
    p=planes()
    values=inputs(p,'shared',True)
    policy=policies()[mode=='streamed']
    model=policy.simulation(p,dispersive=True)
    shapes=tuple(tuple(v.shape) for v in values)
    if mode=='resident':reservation=_resident_reservation(p,shapes,policy,[1e14])
    else:reservation=plane_reservation(model.model,shapes,policy.streamed,[1e14])
    limited=replace(policy,host_budget_bytes=reservation['host_reservation_bytes']-1)
    model=limited.simulation(p,dispersive=True)
    def forbidden(*a,**kw):pytest.fail('Allocated state or packed materials before plane layout admission')
    monkeypatch.setattr(torch,'cat',forbidden)
    monkeypatch.setattr('torchfdtd.differentiable._System',forbidden)
    monkeypatch.setattr('torchfdtd.streamed_dispersive._SlabDispersiveSystem',forbidden)
    with pytest.raises(ValueError,match='host budget'):model(*values,frequency_hz=[1e14])


def test_plane_result_cpu_copy_budget_denies_before_cuda_transfer(monkeypatch):
    p=planes()
    policy=policies('cuda')[0]
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1024**3,2*1024**3))
    reservation=_resident_reservation(p,(p.region.shape,),policy,[1e14,2e14])
    parameter_bytes=torch.ones(p.region.shape,dtype=torch.float64).numel()*8
    assert reservation['host_transfer_reservation_bytes']==2*parameter_bytes+2*reservation['plane_output_bytes']+reservation['plane_result_metadata_bytes']
    policy=replace(policy,host_budget_bytes=reservation['host_reservation_bytes']-1)
    model=policy.simulation(p)
    def forbidden(*a,**kw):pytest.fail('Transferred material before CPU plane-result admission')
    monkeypatch.setattr(torch.Tensor,'to',forbidden)
    with pytest.raises(ValueError,match='host budget'):
        model(torch.ones(p.region.shape,dtype=torch.float64),frequency_hz=[1e14,2e14])


def test_calibration_rejects_corrupted_plane_vjp(monkeypatch):
    p=planes()
    original=_StreamedPlanesFromHost.forward
    def corrupted(self,*values,**kwargs):
        result=original(self,*values,**kwargs)
        for plane in result.values():plane.fields.register_hook(lambda grad:1.1*grad)
        return result
    monkeypatch.setattr(_StreamedPlanesFromHost,'forward',corrupted)
    with pytest.raises(AssertionError):
        tune_adjoint_execution(p,inputs(p,'shared')[0],candidates=policies(),
            frequency_hz=[1e14],probe_steps=10,repeats=1)


def test_plane_tuning_releases_system_before_uncached_reference(monkeypatch):
    from torchfdtd.differentiable import _System
    p=planes()
    live=[]
    original=_System.__init__
    def guarded(self,*args,**kwargs):
        assert not any(ref() is not None for ref in live)
        original(self,*args,**kwargs)
        live.append(weakref.ref(self))
    monkeypatch.setattr(_System,'__init__',guarded)
    first=policies()[0]
    second=replace(first,resident=replace(first.resident,checkpoints=1))
    gc.collect()
    enabled=gc.isenabled()
    gc.disable()
    try:
        tune_adjoint_execution(p,inputs(p,'shared')[0],candidates=[first,second],
            frequency_hz=[1e14],probe_steps=10,repeats=1,reference_cache_bytes=0)
        assert not any(ref() is not None for ref in live)
    finally:
        if enabled:gc.enable()


def test_reference_signature_tracks_resolved_global_source():
    from torchfdtd.adjoint_planes import _plane_signature
    p=planes()
    p.sources[0].use_global_source=True
    q=p.model_copy(deep=True)
    q.global_source.wavelength=p.global_source.wavelength*1.1
    assert _plane_signature(p)!=_plane_signature(q)
    q=p.model_copy(deep=True)
    q.region.memory_mode='budgeted'
    q.region.backend='cuda'
    q.region.cuda_kernel='torch'
    assert _plane_signature(p)==_plane_signature(q)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fp32_normalized_plane_gradient_matches_fp64_and_finite_difference(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=planes()
    frequency=[.035/p.region.time_step]
    def setup(dtype,policy):
        project=p.model_copy(deep=True)
        project.region.precision='float32' if dtype==torch.float32 else 'float64'
        model=policy.simulation(project)
        mask=torch.zeros(project.region.shape,dtype=dtype)
        mask[6:9,5:8]=1
        with torch.no_grad():ref=model(torch.ones_like(mask),frequency_hz=frequency)['first']
        def loss(value):
            return model(1+value*mask,frequency_hz=frequency)['first'].normalized_flux(ref).sum()
        return loss
    reference=setup(torch.float64,policies('cpu')[0])
    x=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    expected=reference(x)
    wanted,=torch.autograd.grad(expected,x)
    with torch.no_grad():fd=(reference(x+1e-5)-reference(x-1e-5))/(2e-5)
    torch.testing.assert_close(wanted,fd,rtol=1e-7,atol=1e-9)
    for policy in policies(device):
        loss=setup(torch.float32,policy)
        parameter=torch.tensor(.5,dtype=torch.float32,requires_grad=True)
        actual=loss(parameter)
        gradient,=torch.autograd.grad(actual,parameter)
        torch.testing.assert_close(actual.double(),expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(gradient.double(),wanted,rtol=1e-4,atol=1e-6)
        assert torch.isfinite(gradient) and abs(gradient)>1e-6
