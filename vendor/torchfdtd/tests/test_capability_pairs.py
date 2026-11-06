"""Risk-based pairwise checks of the capability registry against the code.

Every combination of the registry axes is one small Project (the recipe in
torchfdtd/capabilities.py) executed through the entry point of its execution
mode. The covering array below exercises every value pair of every axis pair
at least once, plus one case per registry rule and lane and six hand-picked
high-risk combinations. An admitted case runs and must produce finite output;
a rejected case must raise the registry's exception with the registry's
message prefix from the registry's code path before any field allocation.
The test fails when the registry says admitted but the code rejects, or the
registry says rejected but the code runs.
"""
from __future__ import annotations

import gc
import inspect
import json
import math
import random
import re
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd import capabilities as C
from torchfdtd import (AdjointOptions, DifferentiablePlaneSimulation, DifferentiableSimulation,
                       DispersivePlaneSimulation, DispersiveSimulation, ModeInjectedPlaneSimulation,
                       ModeNetwork, OpenPortOptions, ReversibleCPMLOptions, ReversibleCPMLPlaneSimulation,
                       ReversibleCPMLSimulation, ReversibleOptions, ReversibleSimulation, Simulation,
                       StreamedAdjointOptions, StreamedDispersiveSimulation, StreamedSimulation, native_radiation_box,
                       plan_tiles, prepare_modal_launch, prepare_open_modal_launch, project_farfield, run_tensor_batch,
                       run_tiled, tensor_from_project)
from torchfdtd.anisotropy_dispersive import TensorDispersiveSimulation
from torchfdtd.streamed_tensor import StreamedTensorSimulation
from torchfdtd.solver import voxelize

CUDA = torch.cuda.is_available()
FREQUENCY_HZ = [299792458.0/(C.WAVELENGTH_UM*1e-6)]
DIRECTIONS = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
INPUTS = []   # tensors the runner allocates as inputs of the current case


def track(*tensors):
    INPUTS.extend(tensors)
    return tensors[0] if len(tensors) == 1 else tensors


def input_bytes():
    return sum(t.numel()*t.element_size() for t in INPUTS if t.is_cuda)


class Unavailable(Exception):
    """No entry point can express the combination; carries the schema field or signature that lacks it."""
    def __init__(self, code_path, reason):
        super().__init__(reason)
        self.code_path = code_path


# ----------------------------------------------------------------------------
# Inputs derived from the recipe project
# ----------------------------------------------------------------------------

def device_of(config):
    return 'cpu' if config['backend'] == 'cpu' else 'cuda'


def dtype_of(config):
    return torch.float64 if config['precision'] == 'float64' else torch.float32


def adjoint_options(config):
    kernel = {'cpu': 'auto', 'cuda_torch': 'torch', 'cuda_fused': 'fused'}[config['backend']]
    return AdjointOptions(checkpoints=2, backward_kernel=kernel)


def streamed_options(config):
    return StreamedAdjointOptions(device=device_of(config), slab_width=8, temporal_depth=4, checkpoints=1,
                                  gpu_budget_bytes=2*1024**3, host_budget_bytes=4*1024**3)


def scalar_epsilon(project, config, *, grad):
    eps, _ = voxelize(project)
    return track(torch.as_tensor(np.ascontiguousarray(eps), dtype=dtype_of(config),
                                 device='cpu' if config['execution'] in C.STREAMED_EXECUTIONS else device_of(config)).requires_grad_(grad))


def dispersive_inputs(project, config, *, grad):
    """epsilon_inf from the voxelized scene, the material's poles confined to its cells."""
    eps, _, ownership = voxelize(project, with_ownership=True)
    device = 'cpu' if config['execution'] in C.STREAMED_EXECUTIONS else device_of(config)
    dtype = dtype_of(config)
    index = [m.name for m in project.materials].index('block')
    mask = torch.as_tensor(ownership == index, dtype=dtype, device=device)
    poles = project.materials[index].oscillators
    epsilon = torch.as_tensor(np.ascontiguousarray(eps), dtype=dtype, device=device).requires_grad_(grad)
    strength = torch.stack([mask*a for _, a, _ in poles]).requires_grad_(grad)
    omega0 = torch.tensor([w for w, _, _ in poles], dtype=dtype, device=device)
    gamma = torch.tensor([g for _, _, g in poles], dtype=dtype, device=device)
    track(epsilon, strength, omega0, gamma, mask)
    return epsilon, strength, omega0, gamma


def tensor_inputs(project, config, *, grad):
    """Node tensors rasterized by the project adapter; the pole for tensor_ade confined to the block nodes."""
    device = 'cpu' if config['execution'] in C.STREAMED_EXECUTIONS else device_of(config)
    adapter = tensor_from_project(project, device=device, checkpoints=2)
    epsilon = track(adapter.rasterize().detach().clone().requires_grad_(grad))
    track(adapter._owners())
    if config['material'] != 'tensor_ade':
        return adapter, (epsilon,)
    background = project.region.background_index**2
    mask = (epsilon.detach()[..., 0, 0] != background).to(epsilon.dtype)
    eye = torch.eye(3, dtype=epsilon.dtype, device=epsilon.device)
    strength = (C.POLE['strength_rad_s_squared']*mask[..., None, None]*eye)[None].requires_grad_(grad)
    omega0 = torch.tensor([C.POLE['resonance_rad_s']], dtype=epsilon.dtype, device=epsilon.device)
    gamma = torch.tensor([C.POLE['damping_rad_s']], dtype=epsilon.dtype, device=epsilon.device)
    track(strength, omega0, gamma, mask, eye)
    return adapter, (epsilon, strength, omega0, gamma)


def guide_section(project):
    """Fixed modal cross-section: the rod's permittivity as a function of the transverse coordinates."""
    rod = project.structures[0]
    d = 'xyz'.index(project.sources[0].normal)
    u, v = (d+1) % 3, (d+2) % 3
    inside = project.materials[[m.name for m in project.materials].index(rod.material)].index**2
    def section(a, b):
        return np.where((np.abs(a-rod.center[u]) <= rod.size[u]/2) & (np.abs(b-rod.center[v]) <= rod.size[v]/2), inside, 1.)
    return section


def finite_signals(result):
    signals = result.signals if hasattr(result, 'signals') else result
    assert bool(torch.isfinite(signals.detach()).all())
    return signals


def backward_finite(value, *inputs):
    grads = torch.autograd.grad(value, [x for x in inputs if x.requires_grad], allow_unused=True)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    return grads


# ----------------------------------------------------------------------------
# Lane dispatch: the entry point of every execution mode
# ----------------------------------------------------------------------------

def execute(config, scratch):
    """Run one combination through its entry point; returns a report dict or raises."""
    execution = config['execution']
    if config['source'] == 'mode' and execution in ('reversible_adjoint', 'tensor_batch', 'tiled'):
        raise Unavailable('torchfdtd/models.py::Source.kind',
                          'The Project schema offers point, plane and TFSF sources; fixed-eigenmode launches exist only as '
                          'inputs of ModeInjectedPlaneSimulation and ModeNetwork.')
    if config['monitor'] == 'mode_port' and execution in ('reversible_adjoint', 'tensor_batch', 'tiled'):
        raise Unavailable('torchfdtd/models.py::Monitor.kind',
                          'The Project schema offers point and field monitors; mode ports exist only as ModeNetwork inputs.')
    if config['material'] == 'tensor_ade' and execution in ('reversible_adjoint', 'tensor_batch', 'tiled'):
        raise Unavailable('torchfdtd/models.py::Material.model',
                          'The Project schema offers no tensor ADE material; tensor poles exist only as '
                          'TensorDispersiveSimulation inputs.')
    if config['material'] == 'tensor_ade' and execution in ('streamed', 'streamed_adjoint'):
        raise Unavailable('torchfdtd/streamed_tensor.py::StreamedTensorSimulation.forward',
                          'The streamed tensor path takes node tensors only; it has no pole inputs.')
    project = C.example_project(config)
    if execution == 'forward':
        return run_forward(config, project, scratch)
    if execution == 'checkpointed_adjoint':
        return run_differentiable(config, project, adjoint_options(config), grad=True)
    if execution == 'streamed':
        with torch.no_grad():
            return run_differentiable(config, project, streamed_options(config), grad=False)
    if execution == 'streamed_adjoint':
        return run_differentiable(config, project, streamed_options(config), grad=True)
    if execution == 'reversible_adjoint':
        return run_reversible(config, project)
    if execution == 'tensor_batch':
        return run_batch(config, project, scratch)
    if execution == 'tiled':
        return run_tiles(config, project)
    raise AssertionError(execution)


def run_forward(config, project, scratch):
    if config['monitor'] == 'mode_port' or config['source'] == 'mode' or config['material'] == 'tensor_ade':
        with torch.no_grad():
            return run_differentiable(config, project, adjoint_options(config), grad=False)
    result = Simulation(project).run()
    report = dict(result.summary)
    assert np.isfinite(result.signals).all() if result.signals.size else True
    if config['monitor'] == 'radiation_box':
        box = native_radiation_box(result, C.radiation_box_ids(), bounds_um=C.radiation_box_bounds(),
                                   refractive_index=project.region.background_index)
        far = box.project(torch.tensor(DIRECTIONS, dtype=torch.float32))
        assert bool(torch.isfinite(far.electric_amplitude).all())
        report['farfield'] = 'native_radiation_box'
    return report


def run_differentiable(config, project, options, *, grad):
    monitor, material, source = config['monitor'], config['material'], config['source']
    streamed = isinstance(options, StreamedAdjointOptions)
    if monitor == 'mode_port':
        return run_mode_network(config, project, options, grad=grad)
    if source == 'mode':
        return run_modal_planes(config, project, options, grad=grad)
    if material == 'tensor_ade':
        model = TensorDispersiveSimulation(project, options)
        _, inputs = tensor_inputs(project, config, grad=grad)
        result = model(*inputs)
        finite_signals(result)
        if grad:
            backward_finite(result.signals.abs().square().sum(), *inputs)
        return dict(result.report)
    if material == 'anisotropic_tensor':
        if monitor == 'point':
            if streamed:
                model = StreamedTensorSimulation(project, options)
                _, (epsilon,) = tensor_inputs(project, config, grad=grad)
            else:
                adapter, (epsilon,) = tensor_inputs(project, config, grad=grad)
                model = adapter.simulation
            result = model(epsilon)
            finite_signals(result)
            if grad:
                backward_finite(result.signals.abs().square().sum(), epsilon)
            return dict(result.report)
        raise AssertionError('tensor media with plane monitors are rejected by the Project schema')
    if material == 'dispersive_ade':
        inputs = dispersive_inputs(project, config, grad=grad)
        if monitor == 'point':
            model = (StreamedDispersiveSimulation if streamed else DispersiveSimulation)(project, options)
            result = model(*inputs)
            finite_signals(result)
            if grad:
                backward_finite(result.signals.abs().square().sum(), *inputs)
            return dict(result.report)
        model = DispersivePlaneSimulation(project, options)
        planes = model(*inputs, FREQUENCY_HZ)
        return finish_planes(config, project, planes, inputs, grad)
    epsilon = scalar_epsilon(project, config, grad=grad)
    if monitor == 'point':
        model = (StreamedSimulation if streamed else DifferentiableSimulation)(project, options)
        result = model(epsilon)
        finite_signals(result)
        if grad:
            backward_finite(result.signals.abs().square().sum(), epsilon)
        return dict(result.report)
    model = DifferentiablePlaneSimulation(project, options)
    planes = model(epsilon, FREQUENCY_HZ)
    return finish_planes(config, project, planes, (epsilon,), grad)


def finish_planes(config, project, planes, inputs, grad):
    values = sum(p.fields.abs().square().sum() for p in planes.values())
    assert bool(torch.isfinite(values))
    report = dict(next(iter(planes.values())).report)
    if config['monitor'] == 'radiation_box':
        far = project_farfield(planes, torch.tensor(DIRECTIONS, dtype=inputs[0].real.dtype, device=inputs[0].device),
                               bounds_um=C.radiation_box_bounds(), refractive_index=project.region.background_index)
        values = values+far.electric_amplitude.abs().square().sum()
        assert bool(torch.isfinite(values))
        report['farfield'] = 'project_farfield'
    if grad:
        backward_finite(values, *inputs)
    return report


def run_modal_planes(config, project, options, *, grad):
    launch = modal_launch(config, project)
    model = ModeInjectedPlaneSimulation(project, launch, options)
    epsilon = scalar_epsilon(project, config, grad=grad)
    planes = model(epsilon, FREQUENCY_HZ)
    return finish_planes(config, project, planes, (epsilon,), grad)


def modal_launch(config, project):
    section = guide_section(project)
    if C.transverse_kind(config) == 'pml':
        return prepare_open_modal_launch(project, section, options=OpenPortOptions(cladding_epsilon=1.), num_modes=1)
    return prepare_modal_launch(project, section, num_modes=1)


def run_mode_network(config, project, options, *, grad):
    open_ports = OpenPortOptions(cladding_epsilon=1.) if C.transverse_kind(config) == 'pml' else None
    network = ModeNetwork(project, C.mode_ports(config), guide_section(project), options, num_modes=1, open_ports=open_ports)
    epsilon = track(network.reference_epsilon(device=device_of(config)).requires_grad_(grad))
    result = network(epsilon)
    assert bool(torch.isfinite(result.s).all())
    if grad:
        backward_finite(result.s.abs().square().sum(), epsilon)
    return dict(result.report)


def run_reversible(config, project):
    def input_map():
        if config['material'] == 'dispersive_ade':
            _, _, ownership = voxelize(project,with_ownership=True)
            index = [m.name for m in project.materials].index('block')
            mask = ownership==index
            # Reversible density is scalar per cell. The scene rasterizer
            # reports separate Yee-component owners, so take their union.
            if mask.ndim == 4:
                mask = mask.any(axis=-1)
            return track(torch.as_tensor(mask,dtype=dtype_of(config),
                device=device_of(config)).contiguous().requires_grad_(True))
        return scalar_epsilon(project,config,grad=True)
    def fixed_map(epsilon):
        return track(torch.ones_like(epsilon) if config['material']=='dispersive_ade' else epsilon.detach().clone())
    active_pml = any(kind == 'pml' for name, kind in C.face_kinds(config).items()
                     if 'xyz'.index(name[0]) in C.active_axes(config))
    if config['monitor'] == 'point' and not active_pml:
        model = ReversibleSimulation(project, ReversibleOptions())
        epsilon = input_map() if config['material']=='dispersive_ade' else track(scalar_epsilon(project, config, grad=True)[..., 0].contiguous().detach().requires_grad_(True))
        result = model(epsilon)
        finite_signals(result)
        backward_finite(result.signals.abs().square().sum(), epsilon)
        return dict(result.report)
    options = ReversibleCPMLOptions()
    if config['monitor'] == 'point':
        model = ReversibleCPMLSimulation(project, options)
        epsilon = input_map()
        result = model(epsilon, fixed_epsilon=fixed_map(epsilon))
        finite_signals(result)
        backward_finite(result.signals.abs().square().sum(), epsilon)
        return dict(result.report)
    model = ReversibleCPMLPlaneSimulation(project, options)
    epsilon = input_map()
    planes = model(epsilon, FREQUENCY_HZ, fixed_epsilon=fixed_map(epsilon))
    return finish_planes(config, project, planes, (epsilon,), True)


def run_batch(config, project, scratch):
    report = run_tensor_batch([project], output_dir=scratch/'batch', cuda_graph=False)
    report.raise_for_errors()
    item = report.items[0]
    result = item.result
    summary = dict(result.summary)
    if config['monitor'] == 'radiation_box':
        box = native_radiation_box(result, C.radiation_box_ids(), bounds_um=C.radiation_box_bounds(),
                                   refractive_index=project.region.background_index)
        far = box.project(torch.tensor(DIRECTIONS, dtype=torch.float32))
        assert bool(torch.isfinite(far.electric_amplitude).all())
        summary['farfield'] = 'native_radiation_box'
    return summary


def run_tiles(config, project):
    plan = plan_tiles(project, C.TILE_UM, C.TILE_OVERLAP_UM)
    stitched = run_tiled(project, plan, backend=device_of(config))
    assert bool(torch.isfinite(stitched.fields).all())
    return dict(tiles=len(plan.tiles), report=plan.report)


# ----------------------------------------------------------------------------
# Case selection: pairwise covering array, one case per rule and lane, high-risk cases
# ----------------------------------------------------------------------------

NAMES = list(C.AXES)
SEED = 20260922
HIGH_RISK = {
    'ade_pmc_streamed_cuda': dict(dimension='3d', mesh='uniform', material='dispersive_ade', boundary='pmc', source='point',
                                  monitor='point', execution='streamed', precision='float32', backend='cuda_torch'),
    'tensor_cpml_checkpointed_fp32': dict(dimension='3d', mesh='uniform', material='anisotropic_tensor', boundary='cpml',
                                          source='point', monitor='point', execution='checkpointed_adjoint', precision='float32',
                                          backend='cpu'),
    'bloch_plane_tensor_batch': dict(dimension='3d', mesh='uniform', material='dielectric', boundary='bloch', source='point',
                                     monitor='plane_dft', execution='tensor_batch', precision='float32', backend='cuda_fused'),
    'graded_mode_port_reversible': dict(dimension='3d', mesh='graded', material='dielectric', boundary='periodic', source='sheet',
                                        monitor='mode_port', execution='reversible_adjoint', precision='float32', backend='cpu'),
    # run_tensor_batch checks the tensor material before the complex fields.
    'tensor_bloch_tensor_batch': dict(dimension='3d', mesh='uniform', material='anisotropic_tensor', boundary='bloch', source='point',
                                      monitor='point', execution='tensor_batch', precision='float32', backend='cuda_torch'),
    # Simulation.run dispatches tensor materials to run_tensor before the fused complex refusal.
    'tensor_bloch_forward_fused': dict(dimension='3d', mesh='uniform', material='anisotropic_tensor', boundary='bloch', source='point',
                                       monitor='point', execution='forward', precision='float32', backend='cuda_fused'),
}


def all_pairs():
    pairs = set()
    for i, a in enumerate(NAMES):
        for b in NAMES[i+1:]:
            for va in C.AXES[a]:
                for vb in C.AXES[b]:
                    pairs.add(((a, va), (b, vb)))
    return pairs


def pairs_of(config):
    return {((a, config[a]), (b, config[b])) for i, a in enumerate(NAMES) for b in NAMES[i+1:]}


def covering_array(seed=SEED, candidates=60):
    """Greedy pairwise covering array: each candidate fixes the axes in random order, choosing the
    value that covers the most uncovered pairs; the best of `candidates` per step is kept."""
    rng = random.Random(seed)
    uncovered = all_pairs()
    configs = []
    while uncovered:
        best, best_gain = None, -1
        for _ in range(candidates):
            order = NAMES[:]
            rng.shuffle(order)
            config = {}
            for axis in order:
                values = list(C.AXES[axis])
                rng.shuffle(values)
                def gain(value):
                    return sum(1 for other, chosen in config.items()
                               if (((axis, value), (other, chosen)) if NAMES.index(axis) < NAMES.index(other)
                                   else ((other, chosen), (axis, value))) in uncovered)
                config[axis] = max(values, key=gain)
            covered = pairs_of(config) & uncovered
            if len(covered) > best_gain:
                best, best_gain = config, len(covered)
        configs.append(dict((n, best[n]) for n in NAMES))
        uncovered -= pairs_of(best)
    return configs


def coverage_cases(base, seed=SEED):
    """Add one combination per registry rule and lane not yet exercised by `base`: the axes the
    entry names are fixed to its values, the others enumerate in a seeded shuffled order."""
    from itertools import product
    rng = random.Random(seed)
    hit = {C.verdict(c).name for c in base}
    extra = []
    for item in [*C.RULES, *C.LANES]:
        if item.name in hit:
            continue
        choices = []
        for axis in NAMES:
            values = sorted(item.when.get(axis, C.AXES[axis]))
            if isinstance(item, C.Rule) and item.executions and axis == 'execution':
                values = sorted(item.executions)
            rng.shuffle(values)
            choices.append(values)
        found = None
        for values in product(*choices):
            config = dict(zip(NAMES, values))
            if C.verdict(config).name == item.name:
                found = config
                break
        if found is None:
            raise AssertionError(f'No combination reaches registry entry {item.name}; remove or fix it.')
        extra.append(found)
        hit.add(item.name)
    return extra


PAIRWISE = covering_array()
COVERAGE = coverage_cases(PAIRWISE)
CASES = [('pair', c) for c in PAIRWISE]+[('cover', c) for c in COVERAGE]


def case_id(config):
    return '-'.join(config[n] for n in NAMES)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

def error_text(exc):
    """The message the registry prefix applies to: the inner value error of a pydantic ValidationError."""
    import pydantic
    if isinstance(exc, pydantic.ValidationError):
        error = exc.errors()[0]
        message = error['msg']
        return message[len('Value error, '):] if message.startswith('Value error, ') else message
    return str(exc)


def innermost_frame(exc):
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = frame.filename.replace('\\', '/')
        # The package may run from the checkout or from an installed wheel; both spell the module the same way.
        if '/torchfdtd/' in path and not path.endswith('capabilities.py'):
            return 'torchfdtd/'+path.rsplit('/torchfdtd/', 1)[1], frame.name
    return None, None


def resolve_code_path(code_path):
    """Import the object a registry code path names: torchfdtd/<module>.py::<Class.method|function|Class>."""
    import importlib
    module_path, name = code_path.split('::')
    module = importlib.import_module(module_path[:-3].replace('/', '.'))
    obj = module
    for part in name.split('.'):
        obj = getattr(obj, part)
    return module, obj


def assert_code_path_carries_message(rule):
    """Static check: the named function's source contains every literal fragment of the message."""
    import typing
    module, obj = resolve_code_path(rule.code_path)
    if rule.stage == 'unavailable':
        return
    if rule.exception == 'ValidationError' and rule.message.startswith('Input should be'):
        field = {'torchfdtd/models.py::Material': 'model'}[rule.code_path]
        literals = typing.get_args(obj.model_fields[field].annotation)
        assert rule.message == 'Input should be '+', '.join(repr(v) for v in literals[:-1])+f' or {literals[-1]!r}'
        return
    source = inspect.getsource(obj)
    for fragment in re.split(r'\{\w+\}', rule.message):
        fragment = fragment.strip(' :.')
        if len(fragment) >= 12:
            assert fragment in source, f'{rule.name}: {fragment!r} is not raised by {rule.code_path}'


def assert_unavailable_claim(rule):
    """The named schema field or signature really lacks the option the combination needs."""
    import typing
    _, obj = resolve_code_path(rule.code_path.rsplit('.', 1)[0] if rule.code_path.endswith(('.kind', '.model')) else rule.code_path)
    if rule.code_path.endswith('Source.kind'):
        assert 'mode' not in typing.get_args(obj.model_fields['kind'].annotation)
    elif rule.code_path.endswith('Monitor.kind'):
        from torchfdtd.models import FieldMonitor
        kinds = set(typing.get_args(obj.model_fields['kind'].annotation))|set(typing.get_args(FieldMonitor.model_fields['kind'].annotation))
        assert kinds == {'point', 'field'}
    elif rule.code_path.endswith('Material.model'):
        assert not any('tensor' in v and v != 'tensor' for v in typing.get_args(obj.model_fields['model'].annotation))
        assert 'tensor_ade' not in typing.get_args(obj.model_fields['model'].annotation)
    else:
        parameters = inspect.signature(obj).parameters
        assert 'strength' not in parameters and 'omega0' not in parameters, rule.code_path


def check_case(config, scratch):
    expected = C.verdict(config)
    if config['backend'] != 'cpu' and not CUDA:
        pytest.skip('CUDA unavailable')
    INPUTS.clear()
    gc.collect()
    allocated = torch.cuda.memory_allocated() if CUDA else 0
    try:
        report = execute(config, scratch)
    except Unavailable as exc:
        assert expected.status == 'rejected' and expected.stage == 'unavailable', \
            f'{case_id(config)}: registry says {expected.status} {expected.name} but no entry point can express it: {exc}'
        assert exc.code_path == expected.code_path and str(exc).startswith(expected.message)
        assert_unavailable_claim(next(r for r in C.RULES if r.name == expected.name))
        return
    except Exception as exc:
        assert expected.status == 'rejected', \
            f'{case_id(config)}: registry admits through {expected.name} but the code raised {type(exc).__name__}: {error_text(exc)[:300]}'
        rule = next(r for r in C.RULES if r.name == expected.name)
        assert type(exc).__name__ == rule.exception, f'{case_id(config)}: {rule.name} expects {rule.exception}, got {type(exc).__name__}: {exc}'
        text = error_text(exc)
        prefix = rule.expected_message(config)
        assert text.startswith(prefix), f'{case_id(config)}: {rule.name} expects {prefix!r}, got {text[:300]!r}'
        assert_code_path_carries_message(rule)
        if rule.stage != 'schema':
            module, function = innermost_frame(exc)
            assert module == rule.code_path.split('::')[0] and function == rule.code_path.split('::')[1].split('.')[-1], \
                f'{case_id(config)}: {rule.name} names {rule.code_path}, raised from {module}::{function}'
        if CUDA and config['backend'] != 'cpu' and rule.stage != 'post':
            growth = torch.cuda.memory_allocated()-allocated-input_bytes()
            fields = 6*math.prod(C.example_project(config).region.shape)*(8 if config['precision'] == 'float64' else 4) \
                if rule.stage != 'schema' else 0
            assert growth < max(fields, 1), f'{case_id(config)}: {rule.name} raised after allocating {growth} bytes on CUDA'
        return
    assert expected.status == 'admitted', \
        f'{case_id(config)}: registry rejects through {expected.name} ({expected.message}) but the code ran'
    lane = next(l for l in C.LANES if l.name == expected.name)
    resolve_code_path(lane.code_path)
    if lane.name == 'forward.tensor':
        assert any('Torch operations, not fused kernels' in w for w in report['warnings'])
    if lane.name == 'tensor_batch':
        assert report['cuda_kernel'] == 'fused_batch'


@pytest.mark.parametrize('kind,config', CASES, ids=[case_id(c) for _, c in CASES])
def test_pairwise(kind, config, tmp_path):
    check_case(config, tmp_path)


@pytest.mark.parametrize('name', list(HIGH_RISK), ids=list(HIGH_RISK))
def test_high_risk_combinations(name, tmp_path):
    config = HIGH_RISK[name]
    check_case(config, tmp_path)


def test_covering_array_covers_every_pair_rule_and_lane():
    covered = set().union(*(pairs_of(c) for c in PAIRWISE))
    assert covered == all_pairs()
    names = {C.verdict(c).name for _, c in CASES}
    assert names >= {r.name for r in C.RULES}|{l.name for l in C.LANES}
    assert len({C.config_id(c) for _, c in CASES}) == len(CASES)


def test_registry_is_complete_and_every_rule_names_real_code():
    summary = C.registry_summary()
    assert summary['unused'] == []
    assert summary['admitted'] > 0 and summary['admitted']+summary['rejected'] == summary['total'] == C.combination_count()
    for rule in C.RULES:
        if rule.stage == 'unavailable':
            assert_unavailable_claim(rule)
        else:
            assert_code_path_carries_message(rule)
    for lane in C.LANES:
        resolve_code_path(lane.code_path)
    for config in C.combinations():
        C.face_kinds(config)   # the recipe assigns every face


def test_rendered_tables_and_json_match_the_registry():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import build_capability_tables as build
    markdown, payload = build.build()
    assert build.MARKDOWN.read_text(encoding='utf-8') == markdown, 'run scripts/build_capability_tables.py'
    assert build.JSON.read_text(encoding='utf-8') == payload, 'run scripts/build_capability_tables.py'
    data = json.loads(payload)
    assert data['summary']['total'] == C.combination_count()
    assert set(data['pairs']) == {f'{a}|{b}' for i, a in enumerate(NAMES) for b in NAMES[i+1:]}


def test_api_capabilities_serves_the_registry():
    from fastapi.testclient import TestClient
    from torchfdtd.server import create_app
    with TestClient(create_app()) as client:
        payload = client.get('/api/capabilities').json()
    assert payload['features'] and payload['combinations']['kind'] == 'capability_registry'
    assert payload['combinations']['summary']['rules'] == len(C.RULES)
