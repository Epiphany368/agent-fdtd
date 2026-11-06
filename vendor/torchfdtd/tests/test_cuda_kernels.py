"""Fused updates must reproduce complete Yee/CPML states, not only a timing."""
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Monitor, Structure, Simulation
from torchfdtd.boundaries import YeeGrid
from torchfdtd.cuda_kernels import FusedYeeCUDA
from torchfdtd.models import Boundaries, BoundaryFace, MeshRefinement, Material


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    pytest.importorskip('cupy')


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('layout', ['pml2d', 'pml3d', 'mixed', 'graded'])
def test_fused_random_fields_and_all_cpml_states(precision, layout):
    require_cuda()
    region = Region(dimension='2d' if layout == 'pml2d' else '3d',
                    size=(3.2, 2.8, 2.4), mesh=.1, pml_cells=4, precision=precision)
    region.boundaries.x_min = BoundaryFace(layers=5, kappa=3, alpha=.03, alpha_polynomial=1)
    if layout == 'mixed':
        region.boundaries.y_min = BoundaryFace(kind='periodic')
        region.boundaries.y_max = BoundaryFace(kind='periodic')
    if layout == 'graded':
        region.material_sampling = 'yee'
        region.mesh_type = 'graded'
        region.mesh_max = .2
        region.mesh_auto_refine = False
        region.mesh_refinements = [MeshRefinement(size=(.4, .4, .4))]
    old_dtype = torch.get_default_dtype()
    try:
        fdtd.set_backend('torch.cuda.'+precision)
        fdtd.backend.float = torch.float32 if precision == 'float32' else torch.float64
        reference, candidate = YeeGrid(region), YeeGrid(region)
        rng = np.random.default_rng(615)
        for name in ('E', 'H', 'inverse_permittivity'):
            dst = getattr(reference, name)
            values = rng.normal(size=tuple(dst.shape))*.02 if name != 'inverse_permittivity' else rng.uniform(.25, 1, tuple(dst.shape))
            dst.copy_(torch.as_tensor(values, dtype=dst.dtype, device=dst.device))
            getattr(candidate, name).copy_(dst)
        fused = FusedYeeCUDA(candidate)
        for _ in range(23):
            reference.update_E()
            reference.update_H()
            fused.update_E()
            fused.update_H()
        tol = 3e-6 if precision == 'float32' else 2e-14
        for got, expected in zip([candidate.E, candidate.H]+candidate.memory_states,
                                 [reference.E, reference.H]+reference.memory_states):
            torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    finally:
        fdtd.set_backend('numpy')
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize('material', ['dielectric', 'lorentz'])
def test_complete_fused_graph_matches_reference_and_eager(material):
    require_cuda()
    p = Project(region=Region(size=(4, 3, 1), mesh=.1, pml_cells=5, steps=180, backend='cuda', precision='float64'),
                materials=[Material(name='body', model=material, index=1.5)],
                structures=[Structure(material='body', radius=.4, kind='circle')],
                sources=[Source(center=(-.8, 0, 0), pulse_cycles=1)],
                monitors=[Monitor(center=(.7, 0, 0))])
    reference = Simulation(p).run()
    p.region.cuda_kernel = 'fused'
    eager = Simulation(p).run(cuda_graph=False)
    graph = Simulation(p).run()
    assert graph.summary['cuda_kernel'] == 'fused'
    assert graph.summary['cuda_graph']
    assert np.max(abs(graph.electric)) > 1e-6
    for result in [eager, graph]:
        np.testing.assert_allclose(result.electric, reference.electric, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result.magnetic, reference.magnetic, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result.signals, reference.signals, rtol=1e-11, atol=1e-12)


def test_fused_rejects_cpu_and_complex_fields():
    p = Project(region=Region(backend='cpu', cuda_kernel='fused', size=(3, 3, 1), mesh=.1, pml_cells=4))
    with pytest.raises(ValueError, match='CUDA GPU'):
        Simulation(p).run()
    require_cuda()
    p.region.backend = 'cuda'
    p.region.boundaries.y_min = BoundaryFace(kind='bloch')
    p.region.boundaries.y_max = BoundaryFace(kind='bloch')
    torch.cuda.synchronize();allocated = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match='Bloch'):
        Simulation(p).run()
    assert torch.cuda.memory_allocated() == allocated   # refused before the grid is built


def test_fused_preserves_subnormal_fields():
    require_cuda()
    old_dtype = torch.get_default_dtype()
    try:
        fdtd.set_backend('torch.cuda.float32')
        fdtd.backend.float = torch.float32
        r = Region(size=(2, 2, 1), mesh=.1, pml_cells=4)
        a, b = YeeGrid(r), YeeGrid(r)
        a.E[10, 10, 0, 2] = 1e-38
        b.E.copy_(a.E)
        fused = FusedYeeCUDA(b)
        for _ in range(20):
            a.update_E()
            a.update_H()
            fused.update_E()
            fused.update_H()
        assert torch.count_nonzero(a.H).item() > 0
        torch.testing.assert_close(b.E, a.E, rtol=0, atol=0)
        torch.testing.assert_close(b.H, a.H, rtol=0, atol=0)
    finally:
        fdtd.set_backend('numpy')
        torch.set_default_dtype(old_dtype)


def test_fused_frequency_plane_matches_cpu_flux():
    require_cuda()
    from test_field_monitors import slab_project
    from torchfdtd import normalize_flux
    p = slab_project()
    p.materials[1].index = 1.5
    cpu = Simulation(p).run()
    p.region.backend = 'cuda'
    p.region.cuda_kernel = 'fused'
    gpu = Simulation(p).run()
    for got, expected in zip(gpu.frequency_fields, cpu.frequency_fields):
        scale = np.max(abs(expected['fields']))
        np.testing.assert_allclose(got['fields']/scale, expected['fields']/scale, rtol=1e-11, atol=1e-11)
        # Backend choice must not invalidate physically matching normalization.
        normalized = normalize_flux(got, expected)
        np.testing.assert_allclose(normalized['ratio'][normalized['valid']], 1, rtol=1e-10)
