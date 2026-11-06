"""Physical timing aperture and full computational modal packet contracts."""
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, FieldMonitor
from torchfdtd.open_mode_injection import (OpenPortOptions, _open_geometry,
                                          prepare_open_modal_launch)


def open_project(steps=12, normal='x'):
    w = 'xyz'.index(normal)
    axes = (w, (w+1)%3, (w+2)%3)
    def xyz(values):
        return tuple(values[axes.index(i)] for i in range(3))
    region = Region(dimension='3d', size=xyz((8., 4., 4.)), mesh=.1,
                    pml_cells=8, steps=steps, precision='float32', material_sampling='yee')
    sizes = xyz((0., 2.4, 2.4))
    project = Project(region=region, sources=[Source(kind='plane', normal=normal,
        center=xyz((-2., 0., 0.)), size=sizes, pulse_cycles=2)], monitors=[
        FieldMonitor(id=key, normal=normal, center=xyz((x, 0., 0.)), size=sizes)
        for key, x in (('near', -1.), ('far', 1.))])
    return project


@pytest.mark.parametrize('normal', ['x', 'y', 'z'])
def test_open_template_keeps_physical_bounds_and_fixed_cladding_collar(normal):
    project = open_project(normal=normal)
    copy, source, axis, index, direction, fixed = _open_geometry(project)
    assert axis == 'xyz'.index(normal) and direction == 1
    assert index == 20 and len(fixed) == 4
    assert copy.model_dump() == project.model_dump()
    for a, lo, hi in fixed:
        assert a != axis and hi-lo == 9
    for a in range(3):
        if a != axis:
            assert np.isclose(source.size[a], 2.4)
    project.sources[0].size = tuple(0 if a == axis else 1 for a in range(3))
    with pytest.raises(ValueError, match='complete physical aperture'):
        _open_geometry(project)


def test_open_source_admission_before_eigensolve(monkeypatch):
    import torchfdtd.open_mode_ports as module
    monkeypatch.setattr(module, 'solve_open_waveguide_modes', lambda *a, **k: pytest.fail('eigensolve before admission'))
    with pytest.raises(ValueError, match='source byte budget'):
        prepare_open_modal_launch(open_project(), 2.25,
            options=OpenPortOptions(1.4**2), source_budget_bytes=1)
    with pytest.raises(ValueError, match='OpenPortOptions'):
        prepare_open_modal_launch(open_project(), 2.25, options=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native open-port acceptance requires CUDA')
def test_native_open_fiber_network_material_adjoint():
    pytest.importorskip('cupy')
    from benchmarks.open_mode_network import run
    assert run('cuda')['accepted']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native open-port acceptance requires CUDA')
def test_native_open_fiber_degenerate_four_channel_network():
    pytest.importorskip('cupy')
    from benchmarks.open_mode_network import run_four_channels
    assert run_four_channels('cuda')['accepted']
