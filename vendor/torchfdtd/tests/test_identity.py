"""Identity conditions on the resolved plan and deterministic replay (G2-06, docs/IDENTITY_CONDITIONS.md)."""
import numpy as np
import pytest
import torch

from torchfdtd import (Project, Structure, Monitor, FieldMonitor, BoundaryFace, SpectrumSettings,
                       Simulation, DifferentiableSimulation, StreamedAdjointOptions)
from torchfdtd.models import RunControl
from torchfdtd.identity import (reference_key, cache_key, restart_key, identity, invalidated,
                                REFERENCE_SECTIONS, CACHE_SECTIONS, KERNEL_SCHEME_FIELDS)
from torchfdtd.plan import resolve_plan
from torchfdtd.adjoint_planes import _plane_signature, _run_fingerprint
from test_plan import scene
from test_differentiable import gpu

# The fp32 discrete tolerance of docs/validation/completion_gates.json proposed_thresholds.
CUDA_REPLAY_RTOL, CUDA_REPLAY_ATOL = 1e-4, 1e-6


def base():
    p = scene(precision='float32')
    p.monitors = [Monitor(id='ez', component='Ez', center=(.1, 0, 0)),
                  FieldMonitor(id='plane', center=(.12, 0, 0), size=(0, .65, .55), normal='x', downsample=2)]
    return Project.model_validate(p.model_dump())


def edited(transform):
    p = base()
    transform(p)
    return Project.model_validate(p.model_dump())


def relabel(p):
    p.name = 'renamed'
    p.materials[2].color = '#123456'
    for item in p.structures + p.sources + p.monitors:
        item.id = 'new-' + item.id
        item.name = 'renamed'


def place(p):
    p.region.backend = 'cuda'; p.region.memory_mode = 'streamed'; p.region.execution_mode = 'streamed_host'
    p.region.tiling.size_um = 12; p.region.field = 'Hy'; p.region.slice_position = .3; p.region.snapshot_interval = 3


def bloch(p):
    p.region.boundaries.y_min = p.region.boundaries.y_max = BoundaryFace(kind='bloch'); p.region.bloch_phase = (0, .3, 0)


def apodize(p):
    p.monitors[1].spectrum = SpectrumSettings(sampling='frequency', apodization='start', apodization_center=3e-15, apodization_time_width=2e-15)


# variant -> (conditions the specification invalidates, legacy _plane_signature changes, legacy _run_fingerprint changes)
MATRIX = {
    'labels': (relabel, (), False, False),
    'placement': (place, (), False, False),
    'precision': (lambda p: setattr(p.region, 'precision', 'float64'), ('cache', 'restart'), False, True),
    'kernel_scheme': (lambda p: setattr(p.region, 'cuda_kernel', 'fused'), ('restart',), False, False),
    'monitor_kernel': (lambda p: setattr(p.region, 'cuda_monitor_kernel', 'fused'), ('restart',), False, False),
    'scatterer_size': (lambda p: setattr(p.structures[0], 'size', (.4, .4, .2)), ('cache', 'restart'), False, True),
    'scatterer_removed': (lambda p: setattr(p, 'structures', []), ('cache', 'restart'), False, True),
    'material_index': (lambda p: setattr(p.materials[2], 'index', 2.1), ('cache', 'restart'), False, True),
    'interface_method': (lambda p: setattr(p.region, 'interface_method', 'subpixel'), ('cache', 'restart'), False, True),
    'waveform_phase': (lambda p: setattr(p.sources[0], 'phase', 30), ('reference', 'cache', 'restart'), True, True),
    'waveform_amplitude': (lambda p: setattr(p.sources[0], 'amplitude', 2), ('reference', 'cache', 'restart'), True, True),
    'source_moved': (lambda p: setattr(p.sources[0], 'center', (-.3, 0, 0)), ('reference', 'cache', 'restart'), True, True),
    'mesh': (lambda p: setattr(p.region, 'mesh', .08), ('reference', 'cache', 'restart'), True, True),
    'time_step': (lambda p: setattr(p.region, 'courant_factor', .8), ('reference', 'cache', 'restart'), True, True),
    'steps': (lambda p: setattr(p.region, 'steps', 30), ('reference', 'cache', 'restart'), True, True),
    'auto_shutoff': (lambda p: setattr(p.region, 'run_control', RunControl(auto_shutoff=True)), ('reference', 'cache', 'restart'), False, False),
    'inert_divergence_setting': (lambda p: setattr(p.region, 'run_control', RunControl(field_limit=1e3, growth_limit=10)), (), False, False),
    'background': (lambda p: setattr(p.region, 'background_index', 1.2), ('reference', 'cache', 'restart'), True, True),
    'pml': (lambda p: setattr(p.region.boundaries, 'x_max', BoundaryFace(kappa=3)), ('reference', 'cache', 'restart'), True, True),
    'pml_dispersion': (lambda p: setattr(p.region, 'pml_dispersion', 'frozen'), ('reference', 'cache', 'restart'), False, False),
    'bloch': (bloch, ('reference', 'cache', 'restart'), True, True),
    'plane_geometry': (lambda p: setattr(p.monitors[1], 'size', (0, .45, .55)), ('reference', 'cache', 'restart'), True, True),
    'plane_apodization': (apodize, ('reference', 'cache', 'restart'), True, True),
    'plane_downsample': (lambda p: setattr(p.monitors[1], 'downsample', 1), ('reference', 'cache', 'restart'), True, True),
    # The plan hashes what the run reads; the legacy keys hash raw settings or leave observation samples to the caller.
    'inert_source_setting': (lambda p: setattr(p.sources[0], 'wavelength_stop', 1.9), (), True, True),
    'plane_frequencies': (lambda p: setattr(p.monitors[1].spectrum, 'wavelength_stop', 1.7), ('cache', 'restart'), False, False),
    'point_monitor_moved': (lambda p: setattr(p.monitors[0], 'center', (.2, 0, 0)), ('cache', 'restart'), False, False),
}


@pytest.mark.parametrize('variant', sorted(MATRIX))
def test_each_change_invalidates_exactly_the_declared_conditions(variant):
    transform, expected, signature_changes, fingerprint_changes = MATRIX[variant]
    before, after = resolve_plan(base()), resolve_plan(edited(transform))
    assert invalidated(before, after) == expected
    keys = identity(before)
    assert len({keys['reference'], keys['cache'], keys['restart']}) == 3 and all(len(k) == 64 for k in keys.values())
    assert (_plane_signature(edited(transform)) != _plane_signature(base())) == signature_changes
    assert (_run_fingerprint(edited(transform)) != _run_fingerprint(base())) == fingerprint_changes


def test_scatterer_and_its_air_reference_share_the_reference_key_on_a_graded_mesh():
    from torchfdtd import freeze_refinements
    from test_adjoint_planes import graded_scene
    p = graded_scene(); p.structures = [Structure(center=(.9, 0, 0), size=(.2, .2, .2))]
    p = Project.model_validate(p.model_dump())
    frozen = freeze_refinements(p)
    air = frozen.model_copy(deep=True); air.structures = []
    air = Project.model_validate(air.model_dump())
    plans = [resolve_plan(q) for q in (p, frozen, air)]
    assert len({reference_key(q) for q in plans}) == 1
    assert cache_key(plans[0]) == cache_key(plans[1]) != cache_key(plans[2])
    assert invalidated(plans[1], plans[2]) == ('cache', 'restart')
    # An unfrozen scatterer of a different size re-meshes and is no longer a matched reference.
    other = graded_scene(); other.structures = [Structure(center=(.9, 0, 0), size=(.2, 2, .2))]
    assert invalidated(plans[0], resolve_plan(Project.model_validate(other.model_dump()))) == ('reference', 'cache', 'restart')


def test_restart_contract_covers_epsilon_options_and_kernel_scheme_only():
    plan = resolve_plan(base())
    epsilon = torch.full(plan.shape + (3,), 1.7, dtype=torch.float32)
    options = StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2, restart_directory='D:/journal', restart_every_blocks=3)
    key = restart_key(plan, epsilon=epsilon, options=options)
    assert key == restart_key(plan, epsilon=epsilon.clone(), options=StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2))
    assert key != restart_key(plan, epsilon=epsilon*1.0001, options=options)
    assert key != restart_key(plan, epsilon=epsilon.double(), options=options)
    assert key != restart_key(plan, epsilon=epsilon, options=StreamedAdjointOptions(device='cpu', slab_width=8, temporal_depth=2))
    assert key != restart_key(plan, epsilon=epsilon, options=StreamedAdjointOptions(device='cuda', slab_width=4, temporal_depth=2))
    fused = resolve_plan(edited(lambda p: setattr(p.region, 'cuda_kernel', 'fused')))
    assert cache_key(fused) == cache_key(plan) and restart_key(fused, epsilon=epsilon, options=options) != key
    assert invalidated(plan, plan, epsilon=(epsilon, epsilon*1.0001)) == ('restart',)
    with pytest.raises(TypeError):
        restart_key(plan, epsilon=np.ones(3))
    assert set(REFERENCE_SECTIONS) < set(CACHE_SECTIONS) and KERNEL_SCHEME_FIELDS == ('cuda_kernel', 'cuda_monitor_kernel')


def replay(p):
    result = Simulation(p).run()
    plane = result.frequency_fields[0]
    return [result.signals, result.electric, result.magnetic, result.frames, plane['fields'], plane['flux']], result.summary['plan_hash']


def test_two_cpu_runs_of_one_plan_are_bitwise_equal():
    first, first_hash = replay(base())
    second, second_hash = replay(base())
    assert first_hash == second_hash
    for a, b in zip(first, second):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    model = DifferentiableSimulation(scene())
    epsilon = torch.full(model.plan.shape + (3,), 1.7, dtype=torch.float32)
    assert torch.equal(model(epsilon).signals, DifferentiableSimulation(scene())(epsilon).signals)


@pytest.mark.parametrize('kernel', ['torch', 'fused'])
def test_two_cuda_runs_of_one_plan_agree_within_the_declared_tolerance(kernel):
    gpu()
    p = base(); p.region.backend = 'cuda'; p.region.cuda_kernel = kernel; p.region.cuda_monitor_kernel = kernel
    p = Project.model_validate(p.model_dump())
    first, first_hash = replay(p)
    second, second_hash = replay(p)
    assert first_hash == second_hash == resolve_plan(base()).plan_hash
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, rtol=CUDA_REPLAY_RTOL, atol=CUDA_REPLAY_ATOL)
