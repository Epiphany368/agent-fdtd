"""Memory admission of the full-autograd reference oracles."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from torchfdtd import BoundaryFace, DifferentiableSimulation, DispersiveSimulation
from torchfdtd import oracle_memory
from torchfdtd.differentiable import _System
from torchfdtd.dispersive_adjoint import _DispersiveSystem
from torchfdtd.oracle_memory import (admit_oracle, oracle_graph_bytes, oracle_requirements, oracle_source_terms,
                                     oracle_state_bytes)
from test_differentiable import project

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


def host(monkeypatch, available):
    """Fix the available host memory seen by every admission."""
    import torchfdtd.memory_profile as profile
    monkeypatch.setattr(profile, 'host_memory', lambda: {'total_bytes': 2*available, 'available_bytes': available})
    monkeypatch.setattr(oracle_memory, 'host_memory', profile.host_memory)


def captured(monkeypatch):
    """Record every estimate the oracles admit, and admit it."""
    seen = []
    admit = oracle_memory.admit_oracle
    monkeypatch.setattr(oracle_memory, 'admit_oracle', lambda estimate, *a, **k: (seen.append(estimate), admit(estimate, *a, **k))[1])
    return seen


def state_bytes(system):
    return sum(x.numel()*x.element_size() for x in system.state())


def saved_bytes(run):
    """Unique storage bytes autograd saves while run() builds its graph."""
    seen, keep = {}, []
    def pack(t):
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        keep.append(t)
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        run()
    return sum(seen.values())


def benchmark():
    spec = importlib.util.spec_from_file_location('oracle_graph_memory', ROOT/'benchmarks'/'oracle_graph_memory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dimension,precision,boundary', [('2d', 'float64', 'pml'), ('3d', 'float32', 'periodic'),
                                                          ('3d', 'float64', 'bloch')])
def test_state_bytes_count_every_restart_array(dimension, precision, boundary):
    p = project(dimension, precision, steps=10, periodic=boundary == 'periodic')
    if boundary == 'bloch':
        p.region.boundaries.x_min = p.region.boundaries.x_max = BoundaryFace(kind='bloch')
        p.region.bloch_phase = (.3, 0, 0)
    dtype = torch.float64 if precision == 'float64' else torch.float32
    epsilon = torch.full(p.region.shape, 1.5, dtype=dtype)
    assert oracle_state_bytes(p.region) == state_bytes(_System(p, epsilon))
    model = DispersiveSimulation(p)
    parameters, layout = model._pack(epsilon, torch.tensor([1e30, 2e30], dtype=dtype), 1e15, 1e14)
    system = _DispersiveSystem(p, epsilon, parameters, layout)
    assert oracle_state_bytes(p.region, pole_count=2) == state_bytes(system)


def test_state_bytes_include_stored_pmc_faces_and_their_pole_banks():
    from test_pmc_general import gradient_scene, random_epsilon
    p = gradient_scene('float64')
    epsilon = random_epsilon(p.region, 3)
    assert oracle_state_bytes(p.region) == state_bytes(_System(p, epsilon))
    parameters, layout = DispersiveSimulation(p)._pack(epsilon, torch.tensor([1e30], dtype=torch.float64), 1e15, 1e14)
    assert oracle_state_bytes(p.region, pole_count=1) == state_bytes(_DispersiveSystem(p, epsilon, parameters, layout))


def test_graph_estimate_formula():
    p = project(steps=100)
    r = p.region
    steps = r.steps+oracle_memory.GRAPH_EXTRA_STEPS
    n = math.prod(r.shape)
    small, factors = oracle_memory.GRAPH_SMALL_TENSOR_BYTES, oracle_memory.GRAPH_TENSOR_FACTORS
    node, per_term = oracle_memory.GRAPH_NODE_BYTES, oracle_memory.GRAPH_SOURCE_TERM_BYTES
    # Yee: state plus the reciprocal and scaled permittivity of every material element.
    yee = oracle_graph_bytes(r, 'yee', material_elements=n, source_terms=3)
    assert yee['tensor_bytes'] == math.ceil(steps*(factors['yee']*(oracle_state_bytes(r)+2*n*8)+small))
    assert yee['node_bytes'] == steps*(node['yee']+3*per_term)
    # Three components per material sample when the caller does not say.
    assert oracle_graph_bytes(r, 'yee')['tensor_bytes'] == oracle_graph_bytes(r, 'yee', material_elements=3*n)['tensor_bytes']
    fixed = oracle_memory.GRAPH_FIXED_HOST_BYTES['cpu']
    total = yee['tensor_bytes']+yee['node_bytes']+fixed
    assert oracle_requirements(yee, 'cpu') == dict(graph=total, host=total)
    # ADE: pole banks in the state and four coefficient arrays per oscillator element.
    ade = oracle_graph_bytes(r, 'ade', pole_count=3, material_elements=3*n, parameter_elements=3*3*n)
    unit = oracle_state_bytes(r, 3)+2*3*n*8+4*9*n*8
    assert ade['tensor_bytes'] == math.ceil(steps*(factors['ade']*unit+small))
    # Tensor ADE: 2P+1 operator applications plus 1+P per Neumann term, three components each.
    applies = 2*2+1+5*(1+2)
    tensor = oracle_graph_bytes(r, 'tensor_ade', pole_count=2, iterations=5)
    tensor_unit = oracle_state_bytes(r, 2)+3*n*8*applies
    assert tensor['tensor_bytes'] == math.ceil(steps*(factors['tensor_ade']*tensor_unit+small))
    assert tensor['node_bytes'] == steps*(node['tensor_ade']*applies+per_term)
    r.steps = 200
    longer = oracle_graph_bytes(r, 'yee', material_elements=n, source_terms=3)
    assert longer['tensor_bytes']-yee['tensor_bytes'] == pytest.approx(100*(factors['yee']*(oracle_state_bytes(r)+2*n*8)+small), abs=1)


def test_mixed_dtype_is_refused_before_admission(monkeypatch):
    # A float32 project with float64 permittivity would run a float64 graph admitted at float32 sizes.
    seen = captured(monkeypatch)
    p = project('3d', 'float32', steps=10, periodic=True)
    epsilon = torch.full(p.region.shape+(3,), 1.5, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='dtype must match the project precision'):
        DifferentiableSimulation(p).reference(epsilon)
    with pytest.raises(ValueError, match='float32 or float64'):
        DifferentiableSimulation(p).reference(epsilon.to(torch.float16))
    assert seen == []
    assert DifferentiableSimulation(p).reference(epsilon.detach().float().requires_grad_()).dtype == torch.float32


def test_modal_oracle_refuses_mixed_dtype_before_admission(monkeypatch):
    from test_mode_adjoint_oracle import FREQUENCY, scene
    from torchfdtd import AdjointOptions
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation
    seen = captured(monkeypatch)
    p, launch, base, _ = scene('cpu', 'float32', steps=12)
    model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=1))
    with pytest.raises(ValueError, match='dtype must match the project precision'):
        model.reference(base.double(), FREQUENCY)
    assert seen == []
    model.reference(base, FREQUENCY)
    assert seen[-1] == oracle_graph_bytes(model.model.project.region, 'yee', material_elements=base.numel(),
                                          source_terms=len(launch.terms))


def test_per_cell_ade_parameters_are_counted(monkeypatch):
    # Per-cell and per-component oscillator parameters make every step's a, d, 4d and k per cell.
    seen = captured(monkeypatch)
    p = project('3d', 'float64', steps=10, periodic=True)
    for a in 'yz':
        setattr(p.region.boundaries, a+'_min', BoundaryFace(kind='periodic'))
        setattr(p.region.boundaries, a+'_max', BoundaryFace(kind='periodic'))
    poles = 4
    epsilon = torch.full(p.region.shape+(3,), 1.5, dtype=torch.float64, requires_grad=True)
    shape = (poles, *p.region.shape, 3)
    s, w, g = (torch.full(shape, v, dtype=torch.float64, requires_grad=True) for v in (.5e30, 1e15, 2e14))
    model = DispersiveSimulation(p)
    retained = saved_bytes(lambda: model.reference(epsilon, s, w, g))
    estimate = seen[-1]
    assert estimate == oracle_graph_bytes(model.project.region, 'ade', pole_count=poles, material_elements=epsilon.numel(),
                                          parameter_elements=s.numel(), source_terms=1)
    # Per step, the tensor estimate covers the saved tensors with at least half again for the allocator.
    steps = p.region.steps
    assert estimate['tensor_bytes']/(steps+oracle_memory.GRAPH_EXTRA_STEPS) >= 1.5*retained/steps


def test_source_terms_add_host_node_bytes(monkeypatch):
    from torchfdtd import Monitor, Project, Region, Source
    seen = captured(monkeypatch)
    r = Region(dimension='3d', size=(1.6, 1.5, 1.4), mesh=.1, pml_cells=3, steps=10, precision='float32', backend='cpu')
    centres = [(-.3+.1*(i % 7), -.3+.1*((i//7) % 7), -.2) for i in range(40)]
    p = Project(region=r, sources=[Source(center=c, pulse='continuous', wavelength=1.1) for c in centres]+
                [Source(center=(0, 0, .1), pulse='continuous', wavelength=1.1, theta=90, phi=45)],
                monitors=[Monitor(component='Ez', center=(.1, 0, 0))])
    p = Project.model_validate(p.model_dump())
    # Forty single-component sources and one with two polarization components.
    assert oracle_source_terms(p) == 42
    epsilon = torch.full(p.region.shape, 1.5, requires_grad=True)
    DifferentiableSimulation(p).reference(epsilon)
    single = oracle_graph_bytes(p.region, 'yee', material_elements=epsilon.numel(), source_terms=1)
    assert seen[-1]['node_bytes']-single['node_bytes'] == 41*oracle_memory.GRAPH_SOURCE_TERM_BYTES*(p.region.steps+oracle_memory.GRAPH_EXTRA_STEPS)


def test_calibration_record_is_recomputed_and_bounds_every_peak():
    """Rebuild every case, recompute its estimate from the current code and compare it with the measured peak."""
    record = json.loads((ROOT/'docs/validation/oracle_graph_memory.json').read_text())
    constants = record['estimate']
    assert constants['tensor_factors'] == oracle_memory.GRAPH_TENSOR_FACTORS
    assert constants['node_bytes'] == oracle_memory.GRAPH_NODE_BYTES
    assert constants['source_term_bytes'] == oracle_memory.GRAPH_SOURCE_TERM_BYTES
    assert constants['small_tensor_bytes'] == oracle_memory.GRAPH_SMALL_TENSOR_BYTES
    assert constants['extra_steps'] == oracle_memory.GRAPH_EXTRA_STEPS
    assert constants['fixed_host_bytes'] == oracle_memory.GRAPH_FIXED_HOST_BYTES
    module = benchmark()
    windows = next(e for e in record['environments'] if e['device'] == 'cuda')['os'].startswith('Windows')
    assert {c['device'] for c in record['cases']} == {'cpu', 'cuda'}
    assert {c['case'] for c in record['cases']} == set(module.CASES)
    held_out = {name for name, spec in module.CASES.items() if spec['held_out']}
    assert {'heldout_yee_periodic3_diag_f64_30', 'heldout_yee_periodic3_diag_f64_40', 'heldout_ade_p4_pn3_f64_periodic3',
            'heldout_yee_sources160_f32'} <= held_out
    for case in record['cases']:
        region, extra = module.build(case['case'], case['steps'], 'cpu')[:2]
        value = oracle_graph_bytes(region, case['kind'], **extra)
        if case['device'] == 'cpu':
            graph = oracle_requirements(value, 'cpu')['graph']
        else:
            graph = value['tensor_bytes']+constants['cuda_library_allowance_bytes']
            # Under Windows (WDDM) the device graph also commits host memory.
            host_bytes = value['node_bytes']+oracle_memory.GRAPH_FIXED_HOST_BYTES['cuda']+(graph if windows else 0)
            assert host_bytes == case['host_estimate_bytes']
            if case['new_process_peak']:
                assert host_bytes >= (1.1 if case['held_out'] else 1)*case['host_peak_bytes'], case['case']
        assert graph == case['estimate_bytes'], case['case']
        assert graph >= case['peak_bytes'], (case['device'], case['case'], case['steps'])
        if case['held_out']:
            # Held-out cases keep a ten percent margin.
            assert graph >= 1.1*case['peak_bytes'], (case['device'], case['case'], case['steps'])


def test_old_cell_step_cap_no_longer_refuses_below_memory(monkeypatch):
    # 3360 cells x 600 steps passed the old two-million cell-step cap by 16,000.
    host(monkeypatch, 64*GIB)
    p = project('3d', 'float32', steps=600, periodic=True)
    assert math.prod(p.region.shape)*p.region.steps > 2_000_000
    shape = p.region.shape+(3,)
    epsilon = torch.full(shape, 1.5, requires_grad=True)
    signals = DifferentiableSimulation(p).reference(epsilon)
    gradient, = torch.autograd.grad(signals.square().sum(), epsilon)
    assert signals.shape == (600, 3) and torch.isfinite(gradient).all()
    # The ADE product also clears its old cap of two million pole-cell-steps by memory alone.
    region = project('2d', steps=10**4).region
    assert admit_oracle(oracle_graph_bytes(region, 'ade', pole_count=4), 'cpu')['oracle_graph_bytes'] < .8*64*GIB


def test_tensor_ade_refuses_before_packing(monkeypatch):
    # The tensor ADE oracle admits before its validation scratch and before packing allocates.
    from test_anisotropy_dispersive import parameters, scale
    from test_anisotropy_walls import scene
    from torchfdtd.anisotropy_dispersive import TensorDispersiveSimulation, _TensorPoleLayout
    host(monkeypatch, 8*GIB)
    p = scene(dict(x='periodic', y='periodic', z='periodic'), 'float64', steps=12)
    inputs = scale(parameters(p, torch.Generator().manual_seed(11), torch.float64))
    model = TensorDispersiveSimulation(p)
    region = model.project.region
    region.steps = 10**7
    monkeypatch.setattr(torch.linalg, 'eigvalsh', lambda *a, **k: pytest.fail('Oracle validated before admission'))
    with pytest.raises(ValueError, match='estimated .* bytes of retained graph'):
        model.reference(*inputs)
    monkeypatch.undo()
    host(monkeypatch, 64*GIB)
    region.steps = 12
    _, layout, _ = model._pack(*inputs)
    assert layout.iterations > 0
    # A cap between the zero-term and the exact Neumann length refuses at packing, before the parameters are packed.
    terms = oracle_source_terms(model.project)
    cap = oracle_requirements(oracle_graph_bytes(region, 'tensor_ade', pole_count=layout.pole_count, source_terms=terms), 'cpu')['graph']
    exact = oracle_requirements(oracle_graph_bytes(region, 'tensor_ade', pole_count=layout.pole_count, iterations=layout.iterations,
                                                   source_terms=terms), 'cpu')['graph']
    assert cap < exact
    monkeypatch.setattr(_TensorPoleLayout, 'flatten', lambda *a: pytest.fail('Oracle packed before admission'))
    with pytest.raises(ValueError, match=f'graph_budget_bytes={cap:,}'):
        model.reference(*inputs, graph_budget_bytes=cap)


def test_over_large_graph_is_refused_with_estimate_memory_and_cap_option(monkeypatch):
    host(monkeypatch, 8*GIB)
    model = DifferentiableSimulation(project(steps=10))
    model.project.region.steps = 10**7
    epsilon = torch.full(model.project.region.shape, 1.5, dtype=torch.float64)
    monkeypatch.setattr('torchfdtd.differentiable._System', lambda *a, **k: pytest.fail('Oracle allocated fields'))
    with pytest.raises(ValueError) as error:
        model.reference(epsilon)
    message = str(error.value)
    estimate = oracle_graph_bytes(model.project.region, 'yee', material_elements=epsilon.numel(),
                                  source_terms=oracle_source_terms(model.project))
    required = oracle_requirements(estimate, 'cpu')['graph']
    assert f'{required:,} bytes of retained graph' in message
    assert f'80% of {8*GIB:,} bytes of available host memory' in message
    assert 'graph_budget_bytes' in message


def test_user_cap_refuses_and_admits(monkeypatch):
    host(monkeypatch, 64*GIB)
    p = project(steps=12)
    epsilon = torch.full(p.region.shape, 1.5, dtype=torch.float64, requires_grad=True)
    model = DifferentiableSimulation(p)
    required = oracle_requirements(oracle_graph_bytes(p.region, 'yee', material_elements=epsilon.numel(),
                                                      source_terms=oracle_source_terms(p)), 'cpu')['graph']
    with pytest.raises(ValueError, match=f'graph_budget_bytes={required-1:,}'):
        model.reference(epsilon, graph_budget_bytes=required-1)
    assert model.reference(epsilon, graph_budget_bytes=required).shape == (12, 3)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            model.reference(epsilon, graph_budget_bytes=bad)
    dispersive = DispersiveSimulation(p)
    ade = oracle_requirements(oracle_graph_bytes(p.region, 'ade', pole_count=1, material_elements=epsilon.numel(),
                                                 parameter_elements=1, source_terms=oracle_source_terms(p)), 'cpu')['graph']
    with pytest.raises(ValueError, match=f'graph_budget_bytes={ade-1:,}'):
        dispersive.reference(epsilon, [1e30], 1e15, 2e14, graph_budget_bytes=ade-1)


def test_cuda_graph_uses_free_device_memory_and_host_nodes(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    host(monkeypatch, 64*GIB)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (GIB, 12*GIB))
    region = project('3d', steps=10).region
    region.steps = 10**5
    estimate = oracle_graph_bytes(region, 'yee')
    assert estimate['tensor_bytes'] > .8*GIB
    with pytest.raises(ValueError, match=f'80% of {GIB:,} bytes of free CUDA memory'):
        admit_oracle(estimate, 'cuda')
    # Graph nodes stay in host memory when the tensors are on the device.
    with pytest.raises(ValueError, match='host memory for the graph nodes'):
        admit_oracle(dict(tensor_bytes=1, node_bytes=60*GIB), 'cuda')
    import sys
    from torchfdtd.adjoint_memory import _spectral_library_reservation
    library = _spectral_library_reservation(torch.device('cuda'))
    committed = 1+library if sys.platform == 'win32' else 0
    assert admit_oracle(dict(tensor_bytes=1, node_bytes=1), 'cuda') == dict(
        oracle_graph_bytes=1+library, oracle_host_bytes=1+oracle_memory.GRAPH_FIXED_HOST_BYTES['cuda']+committed)
