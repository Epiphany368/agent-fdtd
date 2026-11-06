"""Focused CPU native tensor dispatch, adapter parity and bounded outputs."""
import threading

import numpy as np
import pytest
import torch

from torchfdtd.models import Project, Region, Source, Monitor, Material, Structure
from torchfdtd.tensor_project import tensor_from_project
from torchfdtd.tensor_native import estimate_tensor, run_tensor
from torchfdtd.solver import Simulation, Result


def scene(cpml=False):
    faces = {a+'_'+side: {'kind': 'pml' if cpml and a == 'x' else 'periodic'}
             for a in 'xyz' for side in ('min', 'max')}
    return Project(region=Region(dimension='3d', size=(1.6, .6, .6), mesh=.1,
        steps=12, pml_cells=3, material_sampling='yee', backend='cpu',
        snapshot_interval=3, boundaries=faces),
        materials=[Material(name='rotated', model='tensor', epsilon_tensor=(2., 2.5, 3., .2, .1, .15))],
        structures=[Structure(material='rotated', center=(0, 0, 0), size=(.3, .3, .3))],
        sources=[Source(component='Ez', center=(0, 0, 0), pulse='continuous')],
        monitors=[Monitor(component='Ez', center=(.1, 0, 0)), Monitor(component='Hy', center=(.1, 0, 0))])


@pytest.mark.parametrize('cpml', [False, True])
def test_native_tensor_dispatch_trace_fields_and_npz(cpml, tmp_path):
    p = scene(cpml)
    updates = []
    result = Simulation(p).run(progress=updates.append)
    adapter = tensor_from_project(p, checkpoints=0)
    expected = adapter()
    np.testing.assert_allclose(result.signals, expected.signals.numpy(), rtol=2e-6, atol=1e-8)
    from torchfdtd.anisotropy import _TensorSystem
    with torch.no_grad():
        epsilon = adapter.rasterize()
        system = _TensorSystem(p, epsilon)
        system.advance(0, p.region.steps)
    np.testing.assert_array_equal(result.electric, system.grid.E.numpy())
    np.testing.assert_array_equal(result.magnetic, system.grid.H.numpy())
    assert result.summary['engine'] == 'TorchFDTD full-tensor dielectric'
    assert result.summary['cuda_kernel'] == 'torch' and not result.summary['cuda_graph']
    assert updates[-1]['step'] == p.region.steps
    assert len(updates) == len(result.frames) == 4
    assert result.summary['tensor_plan']['memory']['adapter_preparation_bytes'] > 0
    path = tmp_path/'tensor.npz'; result.save(path)
    loaded = Result.load(path)
    np.testing.assert_array_equal(loaded.signals, result.signals)
    np.testing.assert_array_equal(loaded.electric, result.electric)
    assert loaded.project.materials[0].model == 'tensor'
    assert 'epsilon_zz' in loaded.summary['epsilon_preview']


def test_native_tensor_cancel_diagnostics_and_preallocation(monkeypatch):
    p = scene()
    # Planning must not call material rasterization or allocate the tensor system.
    import torchfdtd.tensor_project as module
    rasterize = module.TensorProject.rasterize
    monkeypatch.setattr(module.TensorProject, 'rasterize', lambda *args, **kwargs: pytest.fail('planning rasterized'))
    plan = estimate_tensor(p)
    assert plan['tensor_native_output_bytes'] > 0
    assert plan['tensor_native_host_reservation_bytes'] > plan['tensor_solver_reservation']['host_reservation_bytes']
    import torchfdtd.memory_profile as memory
    original = memory.host_memory
    monkeypatch.setattr(memory, 'host_memory', lambda: {'available_bytes': 1})
    with pytest.raises(ValueError, match='memory'):
        estimate_tensor(p)
    monkeypatch.setattr(memory, 'host_memory', original)
    monkeypatch.setattr(module.TensorProject, 'rasterize', rasterize)
    event = threading.Event(); event.set()
    cancelled = run_tensor(p, cancel=event)
    assert cancelled.summary['cancelled'] and cancelled.signals.shape == (0, 2)
    assert cancelled.frames.shape[0] == 0
    p.region.run_control.field_limit = 1e-20
    with pytest.raises(FloatingPointError, match='limit'):
        run_tensor(p)


@pytest.mark.parametrize('bloch', [False, True])
def test_native_uniform_rotated_tensor_independent_fourier_recurrence(bloch):
    """NumPy FFT symbols, no native tensor operator/curl/update calls in oracle."""
    from torchfdtd.solver import index_at
    from torchfdtd.waveforms import source_time_signal
    p = scene()
    p.structures[0].size = (2., 1., 1.)  # One constant tensor over the complete cell.
    if bloch:
        for axis in range(3):
            for face in p.region.boundaries.pair(axis):
                face.kind = 'bloch'
        p.region.bloch_phase = (.31, -.23, .19)
    native = Simulation(p).run()
    shape = p.region.shape
    phase = np.array(p.region.bloch_phase)
    q = np.stack(np.meshgrid(*(2*np.pi*np.fft.fftfreq(n)+phase[a]/n
                    for a, n in enumerate(shape)), indexing='ij'), -1)
    coordinates = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), -1)
    twist = np.exp(1j*(coordinates*(phase/np.array(shape))).sum(-1))[..., None]
    inverse = np.linalg.inv(np.array([[2., .2, .1], [.2, 2.5, .15], [.1, .15, 3.]]))
    # Incident-triplet average in the unshifted array-coordinate Fourier basis.
    left, right = (1+np.exp(1j*q))/2, (1+np.exp(-1j*q))/2
    symbol = inverse*left[..., :, None]*right[..., None, :]
    for a in range(3):
        symbol[..., a, a] = inverse[a, a]
    scale = p.region.reference_step/np.array(p.region.axis_steps)
    forward = (np.exp(1j*q)-1)*scale
    backward = (1-np.exp(-1j*q))*scale
    def transform(value):
        return np.fft.fftn(value/twist, axes=(0, 1, 2))
    def physical(value):
        return np.fft.ifftn(value, axes=(0, 1, 2))*twist
    source = p.resolved_source(p.sources[0])
    drives = source_time_signal(source, np.arange(1, p.region.steps+1)*p.region.time_step)
    source_index = index_at(source.center, p.region, source.component)+('xyz'.index(source.component[1].lower()),)
    injection = np.zeros(shape+(3,), dtype=np.complex128)
    injection[source_index] = 1
    source_symbol = transform(injection)
    e = np.zeros_like(injection); h = np.zeros_like(injection)
    samples = []
    for drive in drives:
        e += p.region.rectangular_courant*np.einsum('...ab,...b->...a', symbol, np.cross(backward, h))
        e += drive*source_symbol
        h -= p.region.rectangular_courant*np.cross(forward, e)
        electric, magnetic = physical(e), physical(h)
        samples.append([(electric if m.component[0] == 'E' else magnetic)[
            index_at(m.center, p.region, m.component)+('xyz'.index(m.component[1].lower()),)] for m in p.monitors])
    np.testing.assert_allclose(native.signals, np.array(samples), rtol=5e-5, atol=2e-7)
    np.testing.assert_allclose(native.electric, physical(e), rtol=5e-5, atol=2e-7)
    np.testing.assert_allclose(native.magnetic, physical(h), rtol=5e-5, atol=2e-7)
    assert np.max(abs(native.signals)) > 1e-6
