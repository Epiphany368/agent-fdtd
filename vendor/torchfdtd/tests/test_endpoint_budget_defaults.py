"""Endpoint tensor and host preparation budgets derived from memory when left as None."""
import numpy as np
import pytest
import torch

GIB = 1024**3


def host(monkeypatch, available):
    """Fix the available host memory seen by the derived budgets."""
    import torchfdtd.memory_profile as profile
    monkeypatch.setattr(profile, 'host_memory', lambda: {'total_bytes': 2*available, 'available_bytes': available})


def endpoint(**options):
    from torchfdtd.pmc_simulation import C_UM_S, EndpointSimulation
    return EndpointSimulation([np.array([0., .2, .55, 1.])]*3, [('pmc', 'pmc'), ('pec', 'pmc'), ('pmc', 'pec')],
                              dt_seconds=.04/C_UM_S, sources=[(2, (3, 3, 1))], observations=[('E', 2, (3, 3, 1))], **options)


def test_endpoint_tensor_budget_is_derived_at_each_admission(monkeypatch):
    host(monkeypatch, 10*GIB)
    sim = endpoint()
    assert sim.tensor_budget_bytes is None
    # A duration whose payload exceeds the old 256 MB default is admitted.
    planned = sim.memory_plan(20_000_000)['tensor_upper_bound_bytes']
    assert 256_000_000 < planned <= int(.8*10*GIB)
    sim.admit_tensors(planned, 'Requested simulation exceeds endpoint tensor budget')
    # The budget follows the memory at the admission, not at construction, and the refusal names it.
    host(monkeypatch, GIB//4)
    with pytest.raises(ValueError) as error:
        sim.admit_tensors(planned, 'Requested simulation exceeds endpoint tensor budget')
    message = str(error.value)
    assert f'{planned:,} bytes needed, {int(.8*(GIB//4)):,} bytes allowed' in message
    assert 'derived budget, 80% of available host memory; pass tensor_budget_bytes' in message


def test_explicit_endpoint_tensor_budget_overrides(monkeypatch):
    host(monkeypatch, 10*GIB)
    sim = endpoint(tensor_budget_bytes=1_000_000)
    assert sim.tensor_budget_bytes == 1_000_000
    with pytest.raises(ValueError, match='tensor_budget_bytes=1,000,000'):
        sim(torch.ones(sim.topology.counts['E']), torch.zeros(10**5, 1))
    with pytest.raises(ValueError, match='too small'):
        endpoint(tensor_budget_bytes=16)
    with pytest.raises(ValueError, match='positive integer'):
        endpoint(tensor_budget_bytes=0)


def test_cuda_budget_is_derived_for_the_admitted_bytes(monkeypatch):
    # The derived CUDA budget is asked for the real requirement, so unused cache is released when that makes it fit.
    import torchfdtd.cuda_memory as memory
    from torchfdtd.pmc_simulation import derived_budget_bytes
    calls = []
    monkeypatch.setattr(memory, 'cuda_budget_limit', lambda device, required, budget=None: (calls.append((device.type, required)), 7*GIB)[1])
    assert derived_budget_bytes(torch.device('cuda', 0), 123_456) == 7*GIB
    assert calls == [('cuda', 123_456)]


def test_endpoint_cpml_and_project_budgets_are_derived(monkeypatch):
    from test_endpoint_project import project as endpoint_project
    from torchfdtd.endpoint_project import endpoint_from_project
    from torchfdtd.pmc_cpml import EndpointCPMLSimulation
    from torchfdtd.pmc_simulation import C_UM_S
    host(monkeypatch, 10*GIB)
    cpml = EndpointCPMLSimulation([np.arange(7)*.1, np.arange(5)*.1, np.arange(4)*.1],
                                  [('pml', 'pml'), ('pmc', 'pmc'), ('pec', 'pmc')], dt_seconds=.03/C_UM_S,
                                  pml_cells=1, background_epsilon=2., sources=[(0, (3, 4, 3))],
                                  observations=[('E', 0, (3, 4, 3))])
    assert cpml.tensor_budget_bytes is None
    adapter = endpoint_from_project(endpoint_project(), boundary_faces=[('pmc', 'pmc')]*3)
    assert adapter.simulation.tensor_budget_bytes is None and adapter.host_preparation_budget_bytes is None
    explicit = endpoint_from_project(endpoint_project(), boundary_faces=[('pmc', 'pmc')]*3,
                                     tensor_budget_bytes=64_000_000, host_preparation_budget_bytes=8_000_000)
    assert (explicit.simulation.tensor_budget_bytes, explicit.host_preparation_budget_bytes) == (64_000_000, 8_000_000)
    assert explicit().signals.shape == adapter().signals.shape == (10, 2)
    # Host preparation is derived at each admission too, and its refusal names the keyword.
    tensors_only = endpoint_from_project(endpoint_project(), boundary_faces=[('pmc', 'pmc')]*3, tensor_budget_bytes=64_000_000)
    host(monkeypatch, 1024)
    with pytest.raises(ValueError, match='host preparation byte budget: .* 80% of available host memory; pass host_preparation_budget_bytes'):
        tensors_only.plan()


def test_endpoint_cuda_budget_is_derived_from_free_device_memory(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    from torchfdtd.pmc_simulation import derived_budget_bytes
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (5*GIB, 12*GIB))
    sim = endpoint(device='cuda')
    assert sim.tensor_budget_bytes is None and derived_budget_bytes(sim.device, 0) == int(.8*5*GIB)
    with pytest.raises(ValueError, match='80% of free CUDA memory; pass tensor_budget_bytes'):
        sim.admit_tensors(5*GIB, 'Requested simulation exceeds endpoint tensor budget')
    assert endpoint(device='cuda', tensor_budget_bytes=100_000_000).tensor_budget_bytes == 100_000_000


def test_recorded_derived_defaults_keep_run_times_and_stay_within_memory():
    import json
    from pathlib import Path
    record = json.loads((Path(__file__).resolve().parents[1]/'docs/validation/endpoint_budget_defaults.json').read_text())
    assert {c['device'] for c in record['cases']} == {'cpu', 'cuda'}
    # The budgets only admit: every old- and new-default run executes the same steps.
    assert record['identical_work'] and 'only compared with the planned payload' in record['budget_role']
    for case in record['cases']:
        if case['device'] == 'cuda':
            assert all('utilization_percent' in run['gpu_load_before'] and 'utilization_percent' in run['gpu_load_after']
                       for runs in case['runs'].values() for run in runs)
            # Free memory is the admission reading: the smaller of the runtime and the device-wide NVML reading.
            readings = [case['cuda_runtime_free_bytes']]+([case['cuda_nvml_free_bytes']] if case['cuda_nvml_free_bytes'] is not None else [])
            assert case['cuda_free_bytes'] == min(readings)
            assert case['peak_over_available'] == case['cuda_peak_allocated_growth_bytes']/case['cuda_free_bytes']
        new = case['new_default']['total_seconds']
        if case['old_default_admits']:
            old = case['old_default']['total_seconds']
            # Alternating repeats: the new median lies within 15% of the old one.
            assert .85 < new['median']/old['median'] < 1.15, (case['device'], case['kind'], case['cells_per_axis'])
        else:
            assert case['planned_tensor_bytes'] > 0 and len(new['values']) == 3
        assert case['planned_tensor_bytes'] <= case['derived_tensor_budget_bytes']
        assert case['peak_over_available'] < .8
