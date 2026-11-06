"""One native CPU/CUDA endpoint CPML integration gate, run only with a GPU slot."""
import json
import numpy as np
import pytest
import torch

from test_endpoint_native_cpml import scene
from torchfdtd.endpoint_project import endpoint_from_project
from torchfdtd.solver import Simulation, Result, estimate


def test_native_cpml_cuda_results_npz_material_and_waveform_vjp(tmp_path):
    # Keep capability probing inside the test: collection does not touch CUDA.
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for native endpoint CPML integration')
    pytest.importorskip('cupy')
    project = scene()
    cpu_result = Simulation(project).run()
    gpu_project = project.model_copy(deep=True)
    gpu_project.region.backend = 'cuda'
    gpu_result = Simulation(gpu_project).run()
    for name in ('signals', 'electric', 'magnetic', 'frames'):
        np.testing.assert_allclose(getattr(gpu_result, name), getattr(cpu_result, name),
                                   rtol=4e-5, atol=5e-7, err_msg=name)
    for key in ('E_upper', 'H_upper'):
        np.testing.assert_allclose(gpu_result.endpoint_fields[key], cpu_result.endpoint_fields[key],
                                   rtol=4e-5, atol=5e-7, err_msg=key)
    assert cpu_result.endpoint_fields['E_upper'].size > 0
    assert gpu_result.summary['engine'].endswith('/CPML')
    assert gpu_result.summary['backend'] == 'cuda'
    native_plan = gpu_result.summary['endpoint_plan']['memory']
    assert native_plan['psi_state_bytes'] > 0
    assert native_plan['tensor_upper_bound_bytes'] <= estimate(gpu_project)['endpoint_tensor_bytes']
    path = tmp_path/'native-cpml-cuda.npz'
    gpu_result.save(path)
    loaded = Result.load(path)
    for name in ('signals', 'electric', 'magnetic', 'frames'):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(gpu_result, name))
    for key in ('E_upper', 'H_upper'):
        np.testing.assert_array_equal(loaded.endpoint_fields[key], gpu_result.endpoint_fields[key])
    assert loaded.summary['endpoint_plan']['cpml'] == json.loads(json.dumps(gpu_result.summary['endpoint_plan']['cpml']))

    cpu = endpoint_from_project(project, checkpoints=3)
    cpu_epsilon = cpu.rasterize()
    # Vary only the interior, preserving exact fixed CPML/collar material.
    cpu_epsilon[~cpu.simulation.collar] += .2
    cpu_epsilon.requires_grad_()
    cpu_waveforms = cpu.waveforms().requires_grad_()
    reference = cpu(cpu_epsilon, cpu_waveforms)
    cpu_gradients = torch.autograd.grad(reference.signals.square().sum(),
                                        (cpu_epsilon, cpu_waveforms))
    assert torch.count_nonzero(cpu_gradients[0][cpu.simulation.collar]) == 0
    assert cpu_gradients[0][~cpu.simulation.collar].abs().max() > 0
    assert cpu_gradients[1].abs().max() > 0

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gpu = endpoint_from_project(gpu_project, device='cuda', checkpoints=3)
    torch.testing.assert_close(gpu.simulation.collar.cpu(), cpu.simulation.collar)
    gpu_epsilon = cpu_epsilon.detach().cuda().requires_grad_()
    gpu_waveforms = cpu_waveforms.detach().cuda().requires_grad_()
    actual = gpu(gpu_epsilon, gpu_waveforms)
    gpu_gradients = torch.autograd.grad(actual.signals.square().sum(),
                                        (gpu_epsilon, gpu_waveforms))
    torch.testing.assert_close(actual.signals.cpu(), reference.signals, rtol=4e-5, atol=5e-7)
    for candidate, expected in zip(gpu_gradients, cpu_gradients):
        torch.testing.assert_close(candidate.cpu(), expected, rtol=2e-4, atol=1e-6)
    assert torch.count_nonzero(gpu_gradients[0][gpu.simulation.collar]) == 0
    assert gpu_gradients[0][~gpu.simulation.collar].abs().max() > 0
    report = gpu.simulation.last_report
    assert report['peak_checkpoints'] <= 3
    assert report['reverse_steps'] == project.region.steps
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated()-baseline_bytes
    plan = gpu.simulation.memory_plan(project.region.steps)
    # The default budget is derived at each admission for the planned bytes.
    from torchfdtd.pmc_simulation import derived_budget_bytes
    assert gpu.simulation.tensor_budget_bytes is None
    assert plan['tensor_upper_bound_bytes'] <= derived_budget_bytes(gpu.simulation.device, plan['tensor_upper_bound_bytes'])
    assert peak_delta <= plan['tensor_upper_bound_bytes']
    print(dict(trace_max_error=float((actual.signals.cpu()-reference.signals).abs().max().detach()),
               material_vjp_max_error=float((gpu_gradients[0].cpu()-cpu_gradients[0]).abs().max()),
               waveform_vjp_max_error=float((gpu_gradients[1].cpu()-cpu_gradients[1]).abs().max()),
               torch_peak_delta_bytes=peak_delta, tensor_plan_bytes=plan['tensor_upper_bound_bytes']))
