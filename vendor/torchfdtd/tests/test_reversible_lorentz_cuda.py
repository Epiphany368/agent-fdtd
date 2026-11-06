from dataclasses import replace

import pytest
import torch

from torchfdtd import (ReversibleSimulation, ReversibleOptions, ReversibleCPMLSimulation,
                      ReversibleCPMLOptions, ReversibleCPMLPlaneSimulation)
from tests.test_reversible_lorentz_api import configuration, declaration

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize('cpml', [False, True])
@pytest.mark.parametrize('poles', [1, 2])
def test_packed_cuda_fields_density_vjp_and_retained_calls_match_cpu(cpml, poles):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = configuration(cpml)
    material = declaration(poles)
    values = []
    for device in ('cpu', 'cuda'):
        model = ReversibleCPMLSimulation(p, material=material) if cpml else ReversibleSimulation(p, material=material)
        rho = torch.full(p.region.shape, .43, device=device, requires_grad=True)
        background = torch.linspace(1, 2, p.region.shape[2], device=device)
        result = model(rho, fixed_epsilon=background)
        first = torch.autograd.grad(result.signals.square().sum(), rho, retain_graph=True)[0]
        second = torch.autograd.grad(result.signals.square().sum(), rho)[0]
        assert torch.equal(first, second)
        values.append((result.signals.detach().cpu(), first.cpu()))
        with torch.no_grad():
            direct = model(rho, fixed_epsilon=background)
        assert torch.equal(result.signals, direct.signals)
    torch.testing.assert_close(values[1][0], values[0][0], rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(values[1][1], values[0][1], rtol=5e-4, atol=2e-6)


@pytest.mark.parametrize('storage,transfers,offload', [('device', 'sync', False),
    ('cpu', 'sync', False), ('cpu', 'async', False), ('cpu', 'async', True)])
def test_trace_and_terminal_pole_transfers_preserve_cuda_results(storage, transfers, offload):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = configuration(True)
    material = declaration(2)
    values = []
    for options in (ReversibleCPMLOptions(), ReversibleCPMLOptions(trace_storage=storage,
            trace_transfers=transfers, trace_chunk_steps=7, offload_terminal=offload)):
        model = ReversibleCPMLSimulation(p, options, material=material)
        rho = torch.full((*p.region.shape[:2], 1), .37, device='cuda', requires_grad=True)
        background = torch.linspace(1, 2, p.region.shape[2], device='cuda')
        result = model(rho, fixed_epsilon=background)
        first = torch.autograd.grad(result.signals.square().sum(), rho, retain_graph=True)[0]
        second = torch.autograd.grad(result.signals.square().sum(), rho)[0]
        assert torch.equal(first, second)
        assert result.report['pole_state_bytes'] > 0
        values.append((result.signals.detach(), first))
    assert torch.equal(values[0][0], values[1][0])
    assert torch.equal(values[0][1], values[1][1])


@pytest.mark.parametrize('poles,rho_value,offload', [(1,0.,False),(2,1.,True),(3,.4,True)])
def test_periodic_offload_extremes_and_pole_padding_match_cpu(poles,rho_value,offload):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = configuration(False)
    material = declaration(poles)
    for pole in material.poles:
        pole.damping_rad_s = 1e12
    outputs = []
    for device in ('cpu','cuda'):
        model = ReversibleSimulation(p,ReversibleOptions(offload_terminal=offload),material=material)
        rho = torch.full(p.region.shape,rho_value,device=device,requires_grad=True)
        result = model(rho,fixed_epsilon=torch.linspace(1.,2.,p.region.shape[2]))
        gradient = torch.autograd.grad(result.signals.square().sum(),rho)[0]
        assert bool(torch.isfinite(gradient).all())
        outputs.append((result.signals.detach().cpu(),gradient.cpu()))
        assert result.report['terminal_offload_used'] == (offload and device=='cuda')
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=2e-5,atol=2e-7)
    torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=5e-4,atol=2e-6)


@pytest.mark.parametrize('count,block', [(2,128),(4,256),('auto',128)])
def test_plane_launch_counts_tails_and_offload_are_bit_invariant(count,block):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = configuration(True,planes=True)
    p.region.size = (.7,.5,2.1)
    rho = torch.full((*p.region.shape[:2],1),.39,device='cuda',requires_grad=True)
    background = torch.linspace(1,2,p.region.shape[2],device='cuda')
    frequency = rho.new_tensor([299792458/(.6e-6)])
    outputs = []
    for options in (ReversibleCPMLOptions(),ReversibleCPMLOptions(cells_per_thread=count,
            block_size=block,trace_storage='cpu',trace_transfers='async',trace_chunk_steps=7,
            offload_terminal=True)):
        model = ReversibleCPMLPlaneSimulation(p,options,material=declaration(2),
            quadrature_counts={'plane':(2,2)})
        result = model(rho,frequency,fixed_epsilon=background)['plane']
        gradient = torch.autograd.grad((result.fields/p.region.time_step).abs().square().sum(),rho)[0]
        assert float(gradient.abs().max()) > 1e-6
        outputs.append((result.fields.detach(),gradient))
    assert torch.equal(outputs[0][0],outputs[1][0])
    assert torch.equal(outputs[0][1],outputs[1][1])


def test_cuda_inverse_rejects_bad_seed_rows_and_frames_before_touching_fields():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torchfdtd.reversible_lorentz import DensityMaterial,_system
    from torchfdtd.reversible_lorentz_cuda import LorentzReconstructionCUDA
    p = configuration(True)
    model = ReversibleCPMLSimulation(p,material=declaration(2))
    interval = model.interior_z
    rho = torch.full((*p.region.shape[:2],1),.4,device='cuda')
    mat = DensityMaterial(rho,torch.ones(p.region.shape[2]),p,interval,declaration(2))
    system = _system(p,rho,mat,None)
    grad = rho.new_zeros((*p.region.shape[:2],mat.depth,3))
    inverse = LorentzReconstructionCUDA(system,mat,grad,rho.new_zeros((1,len(system.monitors))))
    before = system.grid.E.clone()
    frame = rho.new_zeros((2,*p.region.shape[:2],2))
    with pytest.raises(ValueError,match='seed block'):
        inverse.step(p.region.steps-1,frame,observation_index=1)
    with pytest.raises(ValueError,match='boundary frame'):
        inverse.step(p.region.steps-1,frame[:,:1,:1],observation_index=0)
    assert torch.equal(before,system.grid.E)


def test_complete_automatic_cuda_admission_uses_async_host_trace_with_poles():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = configuration(True)
    material = declaration(2)
    explicit = ReversibleCPMLSimulation(p,ReversibleCPMLOptions(trace_storage='cpu',
        trace_transfers='async',trace_chunk_steps=7),material=material)
    budget = explicit.plan(device='cuda',density_layers=1)['gpu_reservation_bytes']
    automatic = ReversibleCPMLSimulation(p,ReversibleCPMLOptions(trace_storage='auto',
        host_budget_bytes=1<<30,gpu_budget_bytes=budget,trace_chunk_steps=7),material=material)
    outputs = []
    for model in (explicit,automatic):
        rho = torch.full((*p.region.shape[:2],1),.4,device='cuda',requires_grad=True)
        result = model(rho,fixed_epsilon=torch.linspace(1,2,p.region.shape[2],device='cuda'))
        gradient = torch.autograd.grad(result.signals.square().sum(),rho)[0]
        outputs.append((result.signals.detach(),gradient))
    assert result.report['trace_storage'] == 'cpu'
    assert result.report['trace_transfers'] == 'async'
    assert not result.report['trace_placement_attempts'][0]['admitted']
    assert torch.equal(outputs[0][0],outputs[1][0])
    assert torch.equal(outputs[0][1],outputs[1][1])
