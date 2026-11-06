"""G4-04: boundary and seeded random fixtures on the CUDA kernels against the CPU path.

Case file: docs/validation/cases/G4-04_cuda_edge_fixtures.json. Named fixtures
cover the minimum admitted grid, odd sizes on every axis, index boundaries
(a structure crossing the PML interface, a source and a monitor on the last
interior cell, a plane spanning the interior), strong material contrast with a
PEC wall, and ADE plus per-face CPML memory with odd sizes. Twenty random
fixtures seeded from the case file draw sizes, CPML depths, PEC walls,
materials, structures, source components and positions. Streamed execution
covers a partial X slab and temporal tiles that do not divide the step count.
Every CUDA result is judged against the CPU float64 run of the same discrete
problem, and the fused graph path is run twice for bitwise determinism.

Out-of-bounds checking: run this file with PYTORCH_NO_CUDA_MEMORY_CACHING=1 and
CUDA_LAUNCH_BLOCKING=1 (see the case file); compute-sanitizer cannot instrument
the CUDA 12.6 runtime on this host. CUDA graphs cannot be captured under a
non-caching allocator (cudaMalloc is not permitted during capture), so that run
launches the fused kernels eagerly; the ordinary run captures them in a graph.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, BoundaryFace, DifferentiableSimulation, FieldMonitor, Material, Monitor,
                       Project, Region, Simulation, Source, SpectrumSettings, StreamedAdjointOptions,
                       StreamedSimulation, Structure)

CASE = json.loads((Path(__file__).resolve().parents[1] / 'docs' / 'validation' / 'cases' / 'G4-04_cuda_edge_fixtures.json').read_text(encoding='utf-8'))
F = CASE['fixture']
MESH = F['mesh_um']
LIMITS = {precision: (CASE['acceptance'][precision]['rtol'], CASE['acceptance'][precision]['atol']) for precision in ('float32', 'float64')}
SEEDS = F['random']['seeds']
LORENTZ = Material(name='resonant', model='lorentz', **F['lorentz'])
GRAPH = os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1'
PATHS = [('torch', 'torch', False), ('fused', 'fused', GRAPH)]

pytestmark = pytest.mark.cuda


def gpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


def region(cells, pml, steps, precision='float64', **faces):
    r = Region(dimension='3d', size=tuple(n * MESH for n in cells), mesh=MESH, pml_cells=pml, steps=steps,
               precision=precision, backend='cpu', snapshot_interval=10000)
    for name, face in faces.items():
        setattr(r.boundaries, name, face)
    return r


def spectrum():
    return SpectrumSettings(sampling='frequency', frequency_points=2, apodization='none')


def plane(r, x, shrink=()):
    """x-normal plane spanning the interior, one cell short of any PEC wall in ``shrink``."""
    lo = [r.interior_bounds(a)[0] + (MESH if (a, 'min') in shrink else 0) for a in range(3)]
    hi = [r.interior_bounds(a)[1] - (MESH if (a, 'max') in shrink else 0) for a in range(3)]
    return FieldMonitor(id='plane', normal='x', center=(x, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
                        size=(0, hi[1] - lo[1], hi[2] - lo[2]), spectrum=spectrum())


def minimum_grid(precision):
    r = region((11, 11, 11), 3, 30, precision)
    return Project(region=r, sources=[Source(center=(0, 0, 0), component='Ez', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(0, 0, 0), component='Ez'), plane(r, 0.0)])


def odd_sizes(precision):
    r = region((13, 15, 17), 3, 32, precision, x_min=BoundaryFace(layers=5, kappa=2.5), z_max=BoundaryFace(layers=4, kappa=1.5))
    return Project(region=r, materials=[Material(name='glass', index=1.5)],
                   structures=[Structure(kind='sphere', center=(.05, -.05, .15), radius=.25, material='glass')],
                   sources=[Source(center=(-.05, .05, -.15), component='Hy', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(.15, .05, .05), component='Hy'), plane(r, .1)])


def index_boundaries(precision):
    r = region((14, 13, 12), 3, 34, precision)
    upper = [r.interior_bounds(a)[1] for a in range(3)]
    lower = [r.interior_bounds(a)[0] for a in range(3)]
    return Project(region=r, materials=[Material(name='glass', index=2.0)],
                   # The box's +x face lies one cell beyond the PML interface and its -y face on the interface.
                   structures=[Structure(kind='rectangle', center=(upper[0] - .2 + MESH, lower[1] + .2, 0), size=(.4, .4, .5), material='glass')],
                   sources=[Source(center=(upper[0] - MESH / 2, 0, upper[2] - MESH / 2), component='Ex', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(lower[0] + MESH / 2, upper[1] - MESH / 2, lower[2] + MESH / 2), component='Ex'), plane(r, upper[0] - MESH / 2)])


def strong_contrast(precision):
    r = region((15, 13, 13), 4, 36, precision, x_max=BoundaryFace(kind='pec'))
    return Project(region=r, materials=[Material(name='high', index=4.0)],
                   structures=[Structure(kind='sphere', center=(.1, 0, 0), radius=.2, material='high')],
                   sources=[Source(center=(-.2, .05, -.05), component='Ey', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(.25, -.05, .05), component='Ey'), plane(r, .15)])


def ade_odd(precision):
    r = region((15, 13, 17), 3, 30, precision, y_min=BoundaryFace(layers=5, kappa=2), z_max=BoundaryFace(layers=4))
    return Project(region=r, materials=[LORENTZ],
                   structures=[Structure(kind='rectangle', center=(0, .05, -.05), size=(.3, .3, .5), material='resonant')],
                   sources=[Source(center=(-.2, 0, .1), component='Ez', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(.2, 0, -.1), component='Ez'), plane(r, .15)])


NAMED = dict(minimum_grid=minimum_grid, odd_sizes=odd_sizes, index_boundaries=index_boundaries,
             strong_contrast=strong_contrast, ade_odd=ade_odd)


def random_project(seed):
    """One random 3D fixture; every draw is recorded in the returned description."""
    rng = np.random.default_rng(seed)
    pml = int(rng.integers(3, 6))
    face, side, layers = int(rng.integers(0, 3)), 'min' if rng.random() < .5 else 'max', int(rng.integers(3, 6))
    cells = tuple(int(rng.integers(2 * pml + 5 + (max(layers - pml, 0) if a == face else 0), 2 * pml + 12)) for a in range(3))
    faces = {'xyz'[face] + '_' + side: BoundaryFace(layers=layers, kappa=float(rng.uniform(1, 3)))}
    pec = None
    if rng.random() < .35:
        pec = (int(rng.integers(0, 3)), 'max' if rng.random() < .5 else 'min')
        faces['xyz'[pec[0]] + '_' + pec[1]] = BoundaryFace(kind='pec')
    r = region(cells, pml, int(rng.integers(20, 41)), **faces)
    bounds = [r.interior_bounds(a) for a in range(3)]
    epsilon = float(rng.uniform(1, 16))
    dispersive = rng.random() < .3
    material = LORENTZ if dispersive else Material(name='contrast', index=float(np.sqrt(epsilon)))
    kind = 'rectangle' if rng.random() < .5 else 'sphere'
    center = [float(rng.uniform(lo + .05, hi - .05)) for lo, hi in bounds]
    size = [float(rng.uniform(.2, hi - lo)) for lo, hi in bounds]
    touch = rng.random() < .4
    if touch:  # one box face on the PML interface or one cell beyond it
        a = int(rng.integers(0, 3))
        lo, hi = bounds[a]
        size[a] = float(rng.uniform(.2, hi - lo))
        center[a] = hi - size[a] / 2 + (MESH if rng.random() < .5 else 0)
    if kind == 'sphere':
        structure = Structure(kind='sphere', center=tuple(center), radius=float(min(size) / 2), material=material.name)
    else:
        structure = Structure(kind='rectangle', center=tuple(center), size=tuple(size), material=material.name)
    component = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'][int(rng.integers(0, 6))]
    source = [float(rng.uniform(lo + MESH / 2, hi - MESH / 2)) for lo, hi in bounds]
    last_cell = rng.random() < .4
    if last_cell:
        a = int(rng.integers(0, 3))
        source[a] = bounds[a][1] - MESH / 2
    monitor = [float(rng.uniform(lo + MESH / 2, hi - MESH / 2)) for lo, hi in bounds]
    shrink = {(pec[0], pec[1])} if pec else set()
    p = Project(region=r, materials=[material], structures=[structure],
                sources=[Source(center=tuple(source), component=component, wavelength=float(rng.uniform(.8, 1.2)), pulse_cycles=1)],
                monitors=[Monitor(center=tuple(monitor), component=component), plane(r, float(rng.uniform(*bounds[0])), shrink)])
    description = dict(seed=seed, cells=cells, pml=pml, extra_face=('xyz'[face] + '_' + side, layers), pec=pec, steps=r.steps,
                       material='lorentz' if dispersive else f'epsilon {epsilon:.3f}', structure=kind, touches_pml=touch,
                       component=component, source_on_last_cell=last_cell)
    return p, description


def assert_normalized_close(got, expected, precision, label):
    rtol, atol = LIMITS[precision]
    got, expected = np.asarray(got, dtype=np.complex128), np.asarray(expected, dtype=np.complex128)
    scale = np.max(np.abs(expected))
    assert scale > 0, f'{label}: the reference is identically zero'
    np.testing.assert_allclose(got / scale, expected / scale, rtol=rtol, atol=atol, err_msg=label)


def on_cuda(project, kernel, monitor):
    q = project.model_copy(deep=True)
    q.region.backend, q.region.cuda_kernel, q.region.cuda_monitor_kernel = 'cuda', kernel, monitor
    return q


def compare_with_cpu(project, precision):
    reference = Simulation(project).run()
    assert reference.summary['field_peak'] > 0
    for kernel, monitor, graph in PATHS:
        got = Simulation(on_cuda(project, kernel, monitor)).run(cuda_graph=graph)
        assert got.summary['cuda_kernel'] == kernel and got.summary['cuda_graph'] is graph and got.summary['steps'] == project.region.steps
        for key in ('electric', 'magnetic', 'signals'):
            assert_normalized_close(getattr(got, key), getattr(reference, key), precision, f'{kernel} {key}')
        assert_normalized_close(got.field_monitor('plane')['fields'], reference.field_monitor('plane')['fields'], precision, f'{kernel} plane')
        if kernel == 'fused':
            again = Simulation(on_cuda(project, kernel, monitor)).run(cuda_graph=graph)
            for key in ('electric', 'magnetic', 'signals'):
                np.testing.assert_array_equal(getattr(again, key), getattr(got, key), err_msg=f'fused run is not deterministic: {key}')
            np.testing.assert_array_equal(again.field_monitor('plane')['fields'], got.field_monitor('plane')['fields'])


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('name', sorted(NAMED))
def test_named_boundary_fixture(name, precision):
    gpu()
    project = NAMED[name](precision)
    assert max(project.region.shape) <= 32
    compare_with_cpu(project, precision)


def test_random_fixture_seeds_are_the_declared_twenty():
    assert len(SEEDS) == 20 and len(set(SEEDS)) == 20
    assert SEEDS == [F['random']['master_seed'] * 100 + i for i in range(20)]


@pytest.mark.parametrize('seed', SEEDS)
def test_random_fixture(seed):
    gpu()
    project, description = random_project(seed)
    assert max(project.region.shape) <= 32, description
    compare_with_cpu(project, 'float64')


def streamed_scene():
    r = region(tuple(F['streamed']['cells']), 3, F['streamed']['steps'], x_min=BoundaryFace(layers=4))
    return Project(region=r, sources=[Source(center=(.05, 0, -.05), component='Ez', wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(center=(.25, .05, .05), component='Ez'), Monitor(center=(-.15, -.05, .15), component='Hy')])


@pytest.mark.parametrize('slab,depth,transfers,buffers', [tuple(row) for row in F['streamed']['configurations']])
def test_streamed_partial_slabs_and_unaligned_tiles(slab, depth, transfers, buffers):
    gpu()
    p = streamed_scene()
    cells, steps = p.region.shape, p.region.steps
    assert cells[0] % slab or steps % depth, 'this configuration has no remainder to test'
    generator = torch.Generator().manual_seed(F['streamed']['epsilon_seed'])
    epsilon = (1 + 3 * torch.rand(cells, generator=generator, dtype=torch.float64)).requires_grad_(True)
    reference = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))(epsilon)
    expected, = torch.autograd.grad(reference.signals.square().sum(), epsilon)
    assert reference.report['forward_backend'] == 'torch CPU'
    streamed = epsilon.detach().clone().requires_grad_(True)
    options = StreamedAdjointOptions(slab_width=slab, temporal_depth=depth, checkpoints=2, device='cuda', tile_transfers=transfers,
                                     tile_buffers=buffers, gpu_budget_bytes=256 * 1024**2, host_budget_bytes=1024**3)
    result = StreamedSimulation(p, options)(streamed)
    gradient, = torch.autograd.grad(result.signals.square().sum(), streamed)
    assert result.report['slab_width'] == slab and result.report['tile_transfers'] == transfers
    assert_normalized_close(result.signals.detach().numpy(), reference.signals.detach().numpy(), 'float64', 'streamed signals')
    assert_normalized_close(gradient.numpy(), expected.numpy(), 'float64', 'streamed gradient')
    assert float(expected.norm()) > 0
