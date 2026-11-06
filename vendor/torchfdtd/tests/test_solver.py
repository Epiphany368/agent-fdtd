import threading

import numpy as np
import pytest

from torchfdtd import Project, Region, Structure, Source, Monitor, Simulation
from torchfdtd.models import demo_project
from torchfdtd.solver import voxelize


def small(backend='cpu', dimension='2d', precision='float64'):
    return Project(region=Region(size=(4, 4, 4), dimension=dimension, mesh=.1, pml_cells=5,
                                 backend=backend, precision=precision, steps=180, snapshot_interval=60),
                   structures=[Structure(kind='sphere', radius=.5)],
                   sources=[Source(center=(-.8, 0, 0), wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(.8,0,0))])


def test_schema_rejects_silent_geometry_errors():
    with pytest.raises(ValueError, match='non-PML'):
        Project(sources=[Source(center=(4,0,0))])
    with pytest.raises(ValueError, match='positive'):
        Structure(size=(-1,1,1))
    with pytest.raises(ValueError, match='inner radius'):
        Structure(kind='ring',inner_radius=1,radius=.5)
    with pytest.raises(ValueError):
        Region(mesh=float('nan'))


def test_voxel_priority_and_rotation():
    p=small()
    p.structures=[Structure(id='a',size=(2,.3,1),rotation=90,material='Si (constant n)'),
                  Structure(id='b',size=(.2,.2,1),material='Air',mesh_order=1)]
    eps,_=voxelize(p)
    assert eps[20,25,0] == pytest.approx(3.48**2)
    assert eps[25,20,0] == 1
    assert eps[20,20,0] == 1


def test_cpu_fields_propagate_and_save_roundtrip(tmp_path):
    p=small();r=Simulation(p).run()
    assert np.max(abs(r.signals))>1e-3
    assert np.isfinite(r.electric).all()
    path=tmp_path/'project.json';p.save(path)
    assert Project.load(path)==p
    r.save(tmp_path/'result.npz')
    data=np.load(tmp_path/'result.npz')
    np.testing.assert_array_equal(data['E'],r.electric)
    assert r.summary['steps']==180


def test_zero_sources_and_cancellation():
    p=small();p.sources=[]
    r=Simulation(p).run();assert np.count_nonzero(r.electric)==0
    event=threading.Event();event.set()
    r=Simulation(p).run(cancel=event)
    assert r.summary['cancelled'] and r.summary['steps']==0


@pytest.mark.parametrize('dimension', ['2d','3d'])
@pytest.mark.parametrize('precision', ['float32','float64'])
def test_cuda_matches_cpu_and_graph_matches_eager(dimension, precision):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU not available')
    p=small(dimension=dimension, precision=precision)
    cpu=Simulation(p).run()
    p.region.backend='cuda'
    eager=Simulation(p).run(cuda_graph=False)
    graph=Simulation(p).run(cuda_graph=True)
    assert graph.summary['cuda_graph']
    atol=2e-6 if precision=='float32' else 1e-12
    np.testing.assert_allclose(graph.electric,eager.electric,atol=atol,rtol=atol)
    np.testing.assert_allclose(graph.signals,cpu.signals,atol=atol,rtol=atol)
    np.testing.assert_allclose(graph.electric,cpu.electric,atol=atol,rtol=atol)


def test_pml_absorbs_pulse():
    p=small();p.structures=[];p.sources[0].center=(0,0,0);p.region.steps=650
    r=Simulation(p).run()
    peaks=np.max(abs(r.frames),axis=(1,2))
    assert peaks[-1] < peaks.max()*.025

