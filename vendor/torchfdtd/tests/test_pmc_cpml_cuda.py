"""One bounded direct-CUDA mixed endpoint/CPML pulse and adjoint case."""
import numpy as np
import pytest
import torch

from torchfdtd.pmc_cpml import EndpointCPMLSimulation
from torchfdtd.pmc_simulation import C_UM_S


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_direct_cuda_pulse_material_source_vjp_and_compact_construction(monkeypatch):
    pytest.importorskip('cupy')
    arguments=dict(nodes_um=[np.arange(9)*.1,np.arange(5)*.1,np.arange(8)*.1],
        faces=[('pml','pml'),('pmc','pmc'),('pml','pmc')],dt_seconds=.03/C_UM_S,
        pml_cells=2,background_epsilon=2.,sources=[(0,(4,4,7))],
        observations=[('E',0,(4,4,7)),('E',0,(4,4,7)),('H',1,(4,4,4))],
        checkpoints=2,tensor_budget_bytes=64_000_000)
    cpu=EndpointCPMLSimulation(**arguments)
    # CUDA preparation cannot silently construct the CPU sparse reference.
    import torchfdtd.pmc_simulation as module
    monkeypatch.setattr(module,'EndpointTopology',lambda *a,**kw:pytest.fail('CPU sparse topology on CUDA path'))
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    gpu=EndpointCPMLSimulation(**arguments,device='cuda')
    assert gpu._psi_count==cpu._psi_count
    assert all(sum(gpu.backend.psi_lengths[k] for k in keys)>0
               for keys in gpu.backend.forward_groups.values())
    assert gpu.backend.metadata_bytes<4096
    assert not hasattr(gpu.backend.topology,'coordinates')
    torch.testing.assert_close(gpu.collar.cpu(),cpu.collar)
    torch.manual_seed(419)
    waveform=.1*torch.randn(17,1)
    # Independent ordinary-autograd CPU reference, previously checked against
    # the independently assembled split recurrence in test_pmc_cpml.
    parameter=torch.tensor(.2,requires_grad=True)
    drive=waveform.clone().requires_grad_()
    eps=2+parameter*(~cpu.collar)
    state=cpu._zero();samples=[]
    for kick in drive:
        state=cpu._step(state,eps,kick)
        samples.append(torch.stack([(state.electric if f=='E' else state.magnetic)[i]
                                    for f,i in cpu.observation_ids]))
    expected=torch.stack(samples)
    reference=torch.autograd.grad(expected.square().sum(),(parameter,drive))
    cuda_parameter=parameter.detach().cuda().requires_grad_()
    cuda_drive=waveform.cuda().requires_grad_()
    result=gpu(2+cuda_parameter*(~gpu.collar),cuda_drive)
    actual=torch.autograd.grad(result.square().sum(),(cuda_parameter,cuda_drive))
    torch.testing.assert_close(result.cpu(),expected,rtol=3e-5,atol=4e-7)
    for a,b in zip(actual,reference):torch.testing.assert_close(a.cpu(),b,rtol=1e-4,atol=5e-7)
    assert gpu.last_report['peak_checkpoints']<=2
    assert gpu.last_report['reverse_steps']==17
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-base
    assert peak<=gpu.memory_plan(17)['tensor_upper_bound_bytes']
    print(dict(trace_error=float((result.cpu()-expected).abs().max().detach()),
        material_vjp=float(actual[0]),material_vjp_error=float((actual[0].cpu()-reference[0]).abs()),
        source_vjp_error=float((actual[1].cpu()-reference[1]).abs().max()),
        metadata_bytes=gpu.backend.metadata_bytes,peak_torch_allocated_bytes=peak,
        tensor_plan_bytes=gpu.memory_plan(17)['tensor_upper_bound_bytes']))
