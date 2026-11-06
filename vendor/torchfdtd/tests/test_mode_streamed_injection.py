"""Streamed X-slab execution of fixed modal sheets and plane observers.

These are discrete consistency checks against resident execution on a short
run that truncates the pulse, not physical transmission values."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from test_mode_branches import tiny_project, yee_epsilon, guide
from torchfdtd import AdjointOptions, FieldMonitor, StreamedAdjointOptions
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.mode_branches import prepare_aperture_modal_launch
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_s_parameters

FREQUENCY = [299792458/1.55e-6]


def scene(normal, steps=300):
    p = tiny_project('float32', steps=steps, normal=normal)
    if normal == 'x':
        p.monitors = [FieldMonitor(id='out', normal='x', center=(.3, 0., 0.), size=(0., 1.2, .5))]
        section = guide
        fn = lambda x, y, z: guide(y, z)
        mask = (slice(9, 13), slice(4, 10))
    else:
        p.monitors = [FieldMonitor(id='out', normal='y', center=(0., .3, 0.), size=(1.2, 0., .5))]
        section = lambda z, x: guide(x, z)
        fn = lambda x, y, z: guide(x, z)
        mask = (slice(4, 10), slice(9, 13))
    launch = prepare_aperture_modal_launch(p, section)
    base = yee_epsilon(p.region, fn)
    design = torch.zeros_like(base)
    design[mask] = 1
    return p, launch, base, design


def objective(plane, reference, launch):
    t = modal_s_parameters(plane, reference, launch)['transmission'][0]
    return t.real+.3*t.imag+t.abs().square()


@pytest.mark.parametrize('device,normal', [('cpu', 'x'), ('cuda', 'y')])
def test_streamed_modal_planes_and_material_gradient_match_resident(device, normal):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is unavailable')
        pytest.importorskip('cupy')
    p, launch, base, design = scene(normal)
    resident = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=3))
    # The y-normal sheet spans every X slab, so each tile injects only its rows.
    streamed = ModeInjectedPlaneSimulation(p, launch, StreamedAdjointOptions(device=device, slab_width=4,
        temporal_depth=3, checkpoints=2, tile_transfers='async' if device == 'cuda' else 'sync'))
    with torch.no_grad():
        reference = resident(base, FREQUENCY)['out']
    parameter = torch.tensor(.4, requires_grad=True)
    dense = resident(base+parameter*design, FREQUENCY)['out']
    tiled = streamed(base+parameter*design, FREQUENCY)['out']
    assert tiled.report['spatial_streaming'] and tiled.report['modal_source']
    assert tiled.report['modal_aperture_cells'] == [list(a) for a in launch.aperture]
    scale = float(dense.fields.detach().abs().max())
    torch.testing.assert_close(tiled.fields/scale, dense.fields.detach()/scale, rtol=1e-5, atol=1e-6)
    expected, = torch.autograd.grad(objective(dense, reference, launch), parameter)
    actual, = torch.autograd.grad(objective(tiled, reference, launch), parameter)
    assert expected.abs() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-8)
    print({'device': device, 'normal': normal, 'field_difference': float((tiled.fields-dense.fields).detach().abs().max()/scale),
           'gradient': [float(expected), float(actual)]})


def test_streamed_modal_admission_and_signature_guards_precede_fields(monkeypatch):
    p, launch, base, _ = scene('x', steps=12)
    def forbidden(*args, **kwargs):
        raise AssertionError('fields were allocated before admission')
    monkeypatch.setattr('torchfdtd.mode_injection._ModalSystem', forbidden)
    monkeypatch.setattr('torchfdtd.streamed._System', forbidden)
    model = ModeInjectedPlaneSimulation(p, launch, StreamedAdjointOptions(device='cpu', host_budget_bytes=1))
    with pytest.raises(ValueError, match='host budget'):
        model(base, FREQUENCY)
    model = ModeInjectedPlaneSimulation(p, launch, StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2))
    with pytest.raises(ValueError, match='CPU tensor'):
        model.model._run(base.to('meta'), None)
    changed = base.clone()
    changed[launch.electric_index, 6, 2, 0] += .1
    with pytest.raises(ValueError, match='Injection-neighborhood'):
        model(changed, FREQUENCY)
    model.model.project.region.steps += 1
    with pytest.raises(ValueError, match='configuration changed'):
        model(base, FREQUENCY)
    with pytest.raises(ValueError, match='Modal launch region changed'):
        model.model._run(base, None)
