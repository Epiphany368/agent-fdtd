"""Single bounded native tensor CUDA integration gate, not a performance run."""
import numpy as np
import pytest
import torch

from test_tensor_native import scene
from torchfdtd.solver import Simulation, estimate
from torchfdtd.tensor_project import tensor_from_project


def test_native_tensor_cuda_periodic_bloch_cpml_and_material_table_vjp():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for native tensor integration')
    records = []
    for boundary in ('periodic', 'bloch', 'cpml'):
        project = scene(boundary == 'cpml')
        if boundary == 'bloch':
            for axis in range(3):
                for face in project.region.boundaries.pair(axis):
                    face.kind = 'bloch'
            project.region.bloch_phase = (.31, -.23, .19)
        cpu_result = Simulation(project).run()
        gpu_project = project.model_copy(deep=True)
        gpu_project.region.backend = 'cuda'
        planned = estimate(gpu_project)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gpu_result = Simulation(gpu_project).run()
        torch.cuda.synchronize()
        native_peak = torch.cuda.max_memory_allocated()-baseline
        for name in ('signals', 'electric', 'magnetic', 'frames', 'epsilon'):
            np.testing.assert_allclose(getattr(gpu_result, name), getattr(cpu_result, name),
                                       rtol=8e-5, atol=8e-7, err_msg=boundary+'/'+name)
        assert gpu_result.summary['backend'] == 'cuda'
        assert gpu_result.summary['cuda_kernel'] == 'torch'
        assert gpu_result.summary['cuda_graph'] is False
        assert native_peak <= planned['tensor_solver_reservation']['memory_reservation_bytes']

        cpu = tensor_from_project(project, checkpoints=2)
        table = torch.tensor([project.materials[0].epsilon_tensor], dtype=torch.float32, requires_grad=True)
        expected = cpu(table).signals
        weights = torch.linspace(.2, 1.1, expected.numel()).reshape(expected.shape)
        reference, = torch.autograd.grad((weights*expected.abs().square()).sum(), table)
        assert reference.abs().max() > 0
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gpu = tensor_from_project(gpu_project, device='cuda', checkpoints=2)
        cuda_table = table.detach().cuda().requires_grad_()
        actual = gpu(cuda_table).signals
        gradient, = torch.autograd.grad((weights.cuda()*actual.abs().square()).sum(), cuda_table)
        torch.testing.assert_close(actual.cpu(), expected, rtol=8e-5, atol=8e-7)
        torch.testing.assert_close(gradient.cpu(), reference, rtol=3e-4, atol=2e-7)
        torch.cuda.synchronize()
        adjoint_peak = torch.cuda.max_memory_allocated()-baseline
        reservation = gpu.plan()['memory']['memory_reservation_bytes']
        assert adjoint_peak <= reservation
        records.append(dict(boundary=boundary,
            trace_max_error=float(np.max(abs(gpu_result.signals-cpu_result.signals))),
            table_vjp_max_error=float((gradient.cpu()-reference).abs().max()),
            native_torch_peak_delta_bytes=native_peak,
            native_reservation_bytes=planned['tensor_solver_reservation']['memory_reservation_bytes'],
            adjoint_torch_peak_delta_bytes=adjoint_peak, adjoint_reservation_bytes=reservation))
        del actual, expected, gradient, reference, cuda_table, gpu, cpu
    print(records)
