"""Allocation-derived admission must bound executed fused operators and tiers."""
from dataclasses import replace
import gc

import pytest
import torch

from torchfdtd import (AdjointOptions, BoundaryFace, DifferentiableSimulation,
    DifferentiablePlaneSimulation, DispersiveSimulation, DispersivePlaneSimulation,
    FieldMonitor, Source, estimate_adjoint_memory)
from torchfdtd.adjoint_memory import _source_spatial_budget,_spectral_library_reservation
from torchfdtd.injection import source_terms
from test_budgeted_resident import large_scene
from test_differentiable import project
from test_streamed_dispersive import inputs


@pytest.mark.parametrize('dispersive',[False,True])
def test_512_cube_admission_distinguishes_native_and_tensor_workspaces(monkeypatch,dispersive):
    p=large_scene(51.2)
    p.region.cuda_kernel='fused'
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(48*1024**3,48*1024**3))
    monkeypatch.setattr('torchfdtd.adjoint_memory.host_memory',lambda:dict(available_bytes=128*1024**3))
    shapes=dict(parameter_shapes=(p.region.shape,(1,),(),())) if dispersive else {}
    options=AdjointOptions(checkpoints=0 if dispersive else 2,backward_kernel='fused',
        resident_budget_bytes=32*1024**3)
    result=estimate_adjoint_memory(p,options,device='cuda',**shapes)
    assert result['workspace_model']=='fused_cuda_allocations'
    assert result['gpu_reservation_bytes']<32*1024**3
    assert result['device_checkpoint_reservation_bytes']==options.checkpoints*result['restart_state_bytes']
    # Keep the conservative bound for the explicit Torch transpose. The CUDA
    # allocation inventory must not admit a different operator by mistake.
    with pytest.raises(ValueError,match='resident byte budget'):
        estimate_adjoint_memory(p,replace(options,backward_kernel='torch'),device='cuda',**shapes)


def test_mixed_complex_forward_does_not_use_native_workspace(monkeypatch):
    p=project()
    p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(kind='bloch')
    p.region.bloch_phase=(.4,0,0)
    p.region.cuda_kernel='torch'
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1024**3,2*1024**3))
    result=estimate_adjoint_memory(p,AdjointOptions(backward_kernel='fused'),device='cuda')
    assert result['workspace_model']=='conservative_tensor_bound'


def test_many_bloch_profiles_are_counted_without_building_them(monkeypatch):
    p=project(dimension='3d',steps=10)
    p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(kind='bloch')
    p.region.bloch_phase=(.4,0,0)
    p.sources=[Source(kind='plane',normal='y',size=(1.4,0,.6),theta=45,phi=30) for _ in range(20)]
    prepared=[term for source in p.sources for term in source_terms(p,source)]
    actual=sum(profile.size*16 for _,_,_,profile in prepared if profile is not None)
    monkeypatch.setattr('torchfdtd.solver.source_profile',lambda *a:pytest.fail('Built profiles during admission'))
    profiles,injection=_source_spatial_budget(p,16)
    assert profiles>=actual and profiles>0 and injection>0
    p.sources=p.sources[:1]
    one_profiles,one_injection=_source_spatial_budget(p,16)
    assert profiles==20*one_profiles and injection==one_injection


def test_cold_spectral_library_reservation_honors_hardware_and_env(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,9))
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.delenv('CUBLASLT_WORKSPACE_SIZE',raising=False)
    ada=_spectral_library_reservation(torch.device('cuda'))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(9,0))
    assert _spectral_library_reservation(torch.device('cuda'))>ada
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':65536:2')
    monkeypatch.setenv('CUBLASLT_WORKSPACE_SIZE','4096')
    assert _spectral_library_reservation(torch.device('cuda'))==2*(128+4)*1024**2
    monkeypatch.setenv('CUBLASLT_WORKSPACE_SIZE','-1')
    with pytest.raises(ValueError,match='must not be negative'):_spectral_library_reservation(torch.device('cuda'))


@pytest.mark.parametrize('precision,bloch,dispersive,layout',[
    ('float32',False,False,'shared'),('float64',False,False,'diagonal'),
    ('float32',True,False,'diagonal'),('float64',True,False,'shared'),
    ('float32',False,True,'spatial'),('float64',False,True,'shared'),
    ('float32',True,True,'shared'),('float64',True,True,'diagonal'),
])
@pytest.mark.parametrize('storage',['device','host','hierarchical'])
def test_fused_peak_and_vjp_with_noncontiguous_inputs(tmp_path,precision,bloch,dispersive,layout,storage):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p=project(dimension='3d',precision=precision,steps=11)
    p.region.cuda_kernel='fused'
    if bloch:
        p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(kind='bloch')
        p.region.bloch_phase=(.4,0,0)
        p.sources=[Source(kind='plane',normal='y',size=(1.4,0,.6),component='Ez',pulse='continuous',wavelength=1.1)]
    host=inputs(p,layout,True)
    if not dispersive:host=host[:1]
    model_type=DispersiveSimulation if dispersive else DifferentiableSimulation
    oracle=model_type(p).reference(*host)
    expected=torch.autograd.grad(oracle.abs().square().sum(),host)
    # Every non-scalar material is a view with stride two. Include the resulting
    # CUDA contiguous-copy/packing allocations in the measured solver peak.
    values=tuple(torch.empty((*v.shape,2),device='cuda',dtype=v.dtype)[...,0].copy_(v.detach()).requires_grad_() for v in host)
    options=AdjointOptions(checkpoints=3,storage=storage,backward_kernel='fused',
        checkpoint_transfers='sync' if storage=='device' else 'async',staging_slots=2,
        device_checkpoints=1 if storage=='hierarchical' else 0,
        host_checkpoints=1 if storage=='hierarchical' else 0,
        host_budget_bytes=64*1024**2,disk_budget_bytes=64*1024**2,checkpoint_directory=tmp_path)
    shapes=dict(parameter_shapes=tuple(tuple(v.shape) for v in values)) if dispersive else {}
    plan=estimate_adjoint_memory(p,options,device='cuda',**shapes)
    gc.collect();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result=model_type(p,options)(*values)
    actual=torch.autograd.grad(result.signals.abs().square().sum(),values)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-baseline
    assert peak<=plan['gpu_reservation_bytes']
    assert result.report['workspace_model']=='fused_cuda_allocations'
    assert result.report['restart_bytes']==plan['restart_state_bytes']
    assert plan['workspace_reservation_bytes']==sum(plan['workspace_components_bytes'].values())
    if storage!='device':
        assert result.report['device_staging_bytes']==plan['device_staging_reservation_bytes']
    tolerance=2e-4 if precision=='float32' else 2e-10
    torch.testing.assert_close(result.signals.cpu(),oracle.detach(),rtol=tolerance,atol=tolerance*1e-2)
    for a,b,scale in zip(actual,expected,(1.,1e30,1e15,1e15)):
        torch.testing.assert_close(a.cpu()*scale,b*scale,rtol=tolerance,atol=tolerance*1e-2)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('dispersive,plane',[(False,False),(True,False),(True,True)])
def test_spectral_and_plane_peak_include_library_and_observer_workspace(dispersive,plane):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p=project(dimension='3d',steps=11)
    p.region.cuda_kernel='fused'
    p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(kind='bloch')
    p.region.bloch_phase=(.4,0,0)
    if plane:p.monitors=[FieldMonitor(size=(0,.4,.4))]
    values=tuple(v.detach().to('cuda').requires_grad_() for v in inputs(p,'shared'))
    if not dispersive:values=values[:1]
    options=AdjointOptions(checkpoints=2,backward_kernel='fused')
    kind=(DispersivePlaneSimulation if dispersive else DifferentiablePlaneSimulation) if plane else (
        DispersiveSimulation if dispersive else DifferentiableSimulation)
    model=kind(p,options)
    frequency=[1e14,2e14]
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    if plane:
        result=next(iter(model(*values,frequency,block_size=4).values()))
    else:result=model.spectrum(*values,frequency,block_size=4)
    gradients=torch.autograd.grad((result.fields/p.region.time_step).abs().square().sum(),values)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated()-baseline<=result.report['gpu_reservation_bytes']
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
