"""G5-01: resident CUDA, streamed host banks, streamed disk banks and asynchronous streamed tiles agree.

One scatterer (tests/g5_support.py) is run through the resident CUDA adjoint,
which is the reference, and through the three streamed tiers with the
time-history and online-DFT point observations and the spectral plane
observation, for scalar, diagonal, Drude (ADE) and stored-face (PMC/symmetric)
material states, plus the node-tensor material on point observations. The
spectral fields per time step, the scalar objective on the spectrum and the
gradient of every differentiable input must agree with the resident result at
the layer-A tolerance declared in docs/validation/cases/G5-01.json. Unsupported
combinations must be refused by the registry rule before any field is
allocated.
"""
import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiablePlaneSimulation, DifferentiableSimulation, DispersivePlaneSimulation,
                       DispersiveSimulation, ModeNetwork, StreamedAdjointOptions, StreamedDispersiveSimulation,
                       StreamedSimulation)
from torchfdtd import capabilities as C
from torchfdtd.anisotropy import TensorDielectricSimulation
from torchfdtd.streamed_tensor import StreamedTensorSimulation
from g5_support import CUDA, DRUDE, FREQUENCY_HZ, Record, scene, sphere, streamed_options, tolerance
from test_anisotropy_walls import tensor_field

CASE = 'G5-01'
RECORD = Record('G5-01', CASE, 'tests/test_memory_paths_g5.py')
PATHS = ('host_sync', 'disk_sync', 'host_async')
OBSERVATIONS = ('history', 'online', 'plane')
cuda = pytest.mark.skipif(not CUDA, reason='CUDA unavailable: the resident reference and the streamed tiles run on the GPU')
_references = {}


# ----------------------------------------------------------------------------------------------- inputs
def inputs(material, project, device):
    """Differentiable inputs of one material state on one device; the Drude strength is s_hat * 4e30 (rad/s)^2."""
    dtype = torch.float64 if project.region.precision == 'float64' else torch.float32
    if material in ('scalar', 'pmc'):
        return (sphere(project, inside=4., outside=1.).to(device).requires_grad_(),)
    if material == 'diagonal':
        return (sphere(project, inside=4., outside=1., diagonal=True).to(device).requires_grad_(),)
    if material == 'drude':
        epsilon_inf = sphere(project, inside=DRUDE['epsilon_inf'], outside=1.).to(device).requires_grad_()
        s_hat = sphere(project, inside=1., outside=0.)[None].to(device).requires_grad_()
        return (epsilon_inf, s_hat)
    if material == 'tensor':
        return (tensor_field(project.region, torch.Generator().manual_seed(29), dtype).contiguous().to(device).requires_grad_(),)
    raise AssertionError(material)


def drude_arguments(s_hat):
    dtype = s_hat.dtype
    return (s_hat*DRUDE['strength_rad_s_squared'], torch.tensor([0.], dtype=dtype, device=s_hat.device),
            torch.tensor([DRUDE['damping_rad_s']], dtype=dtype, device=s_hat.device))


def spectral_fields(material, observation, project, options, values):
    """The spectral fields of one run: (F, monitors) for points, (F, points, 6) for the plane."""
    streamed = isinstance(options, StreamedAdjointOptions)
    if material == 'drude':
        epsilon_inf, s_hat = values
        arguments = (epsilon_inf, *drude_arguments(s_hat))
        if observation == 'plane':
            planes = DispersivePlaneSimulation(project, options)(*arguments, FREQUENCY_HZ)
            return next(iter(planes.values())), 'plane'
        model = (StreamedDispersiveSimulation if streamed else DispersiveSimulation)(project, options)
        if observation == 'online':
            return model.spectrum(*arguments, FREQUENCY_HZ), 'online'
        return model(*arguments), 'history'
    epsilon, = values
    if material == 'tensor':
        model = StreamedTensorSimulation(project, options) if streamed else TensorDielectricSimulation(
            project, options, cpml_material='tensor')
    elif observation == 'plane':
        planes = DifferentiablePlaneSimulation(project, options)(epsilon, FREQUENCY_HZ)
        return next(iter(planes.values())), 'plane'
    else:
        model = (StreamedSimulation if streamed else DifferentiableSimulation)(project, options)
    if observation == 'online':
        return model.spectrum(epsilon, FREQUENCY_HZ), 'online'
    return model(epsilon), 'history'


def evaluate(material, observation, project, options, values):
    """Spectral amplitude per time step, the scalar objective and the gradients of every input."""
    result, kind = spectral_fields(material, observation, project, options, values)
    fields = result.spectrum(FREQUENCY_HZ) if kind == 'history' else result.fields
    amplitude = fields/project.region.time_step
    objective = amplitude.abs().square().sum()
    gradients = torch.autograd.grad(objective, values)
    return amplitude.detach().cpu(), objective.detach().cpu(), tuple(g.detach().cpu() for g in gradients), result.report


def reference(material, precision, observation):
    key = (material, precision, observation)
    if key not in _references:
        project = scene(precision, observation, 'pmc' if material == 'pmc' else 'cpml')
        values = inputs(material, project, 'cuda')
        _references[key] = evaluate(material, observation, project, AdjointOptions(checkpoints=2), values)
    return _references[key]


# ----------------------------------------------------------------------------------------------- equivalence
def instances():
    for precision in ('float32', 'float64'):
        for material in ('scalar', 'diagonal', 'drude', 'pmc'):
            for observation in OBSERVATIONS:
                if material == 'pmc' and observation == 'plane':
                    continue  # planes_pmc: refused by the registry, checked below
                for path in PATHS:
                    yield precision, material, observation, path
    for path in PATHS:
        yield 'float32', 'tensor', 'online', path


@cuda
@pytest.mark.parametrize('precision,material,observation,path', list(instances()),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_streamed_tier_matches_resident_cuda(precision, material, observation, path, tmp_path):
    tol = tolerance(CASE, precision)
    expected_fields, expected_objective, expected_gradients, _ = reference(material, precision, observation)
    project = scene(precision, observation, 'pmc' if material == 'pmc' else 'cpml')
    values = inputs(material, project, 'cpu')
    options = streamed_options(path, tmp_path/'banks')
    fields, objective, gradients, report = evaluate(material, observation, project, options, values)
    assert expected_fields.abs().max() > 1e-3 and all(g.abs().max() > 1e-6 for g in expected_gradients)
    torch.testing.assert_close(fields, expected_fields, **tol)
    torch.testing.assert_close(objective, expected_objective, **tol)
    for actual, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted, **tol)
    assert report['state_storage'] == path.split('_')[0] and report['tile_transfers'] == path.split('_')[1]
    assert report['observation_storage'] == ('online_spectrum' if observation != 'history' else 'time_history')
    if path.startswith('disk'):
        assert report['forward_backing_store']['closed'] and report['backward_backing_store']['closed']
        assert not list((tmp_path/'banks').iterdir())
    RECORD.add(f'{precision}/{material}/{observation}/{path}',
               precision=precision, material=material, observation=observation, path=path,
               tolerance=tol, objective=float(objective), objective_reference=float(expected_objective),
               max_abs_field_error=float((fields-expected_fields).abs().max()),
               max_abs_field=float(expected_fields.abs().max()),
               max_abs_gradient_error=[float((a-b).abs().max()) for a, b in zip(gradients, expected_gradients)],
               max_abs_gradient=[float(b.abs().max()) for b in expected_gradients],
               peak_block_checkpoints=report['peak_block_checkpoints'], replayed_blocks=report['replayed_blocks'],
               tile_buffers=report['tile_buffers'])


# ----------------------------------------------------------------------------------------------- refusals
def registry_config(**overrides):
    config = dict(dimension='3d', mesh='uniform', material='dielectric', boundary='cpml', source='point',
                  monitor='point', execution='streamed_adjoint', precision='float32', backend='cuda_fused')
    config.update(overrides)
    return config


@pytest.mark.parametrize('name,overrides,build', [
    ('planes_pmc', dict(boundary='pmc', monitor='plane_dft'),
     lambda project, options: DifferentiablePlaneSimulation(project, options)),
    ('streamed_dispersive_oneway', dict(material='dispersive_ade', source='plane_oneway', boundary='periodic'),
     lambda project, options: StreamedDispersiveSimulation(project, options)),
    ('streamed_dispersive_planes_oneway', dict(material='dispersive_ade', source='plane_oneway', monitor='plane_dft', boundary='periodic'),
     lambda project, options: DispersivePlaneSimulation(project, options)),
    ('mode_network_streamed', dict(monitor='mode_port'),
     lambda project, options: ModeNetwork(project, C.mode_ports(registry_config(monitor='mode_port')), None, options)),
])
def test_unsupported_combinations_are_refused_by_the_registry_rule(name, overrides, build, monkeypatch):
    config = registry_config(**overrides)
    verdict = C.verdict(config)
    assert verdict.status == 'rejected' and verdict.name == name
    rule = next(r for r in C.RULES if r.name == name)
    project = C.example_project(config)
    options = StreamedAdjointOptions(device='cuda' if CUDA else 'cpu', slab_width=8, temporal_depth=4, checkpoints=1)
    # The refusal comes from the constructor: no field bank or tile is created.
    monkeypatch.setattr('torchfdtd.streamed._Streamed.apply',
                        lambda *args, **kwargs: pytest.fail('A refused combination reached the streamed operator.'))
    with pytest.raises(ValueError) as caught:
        build(project, options)
    assert str(caught.value).startswith(rule.expected_message(config))


def test_admitted_combinations_name_the_streamed_lanes():
    assert C.verdict(registry_config()).name == 'streamed.point'
    assert C.verdict(registry_config(monitor='plane_dft')).name == 'streamed.planes'
    assert C.verdict(registry_config(material='dispersive_ade')).name == 'streamed.dispersive'
    assert C.verdict(registry_config(material='dispersive_ade', monitor='plane_dft')).name == 'streamed.dispersive_planes'
    assert C.verdict(registry_config(material='anisotropic_tensor')).name == 'streamed.tensor'
    assert C.verdict(registry_config(boundary='pmc')).name == 'streamed.point'
