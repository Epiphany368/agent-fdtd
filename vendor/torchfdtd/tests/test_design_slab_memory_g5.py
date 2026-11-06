"""G5-04: geometry and density design slabs never materialize the full epsilon or its VJP, and the host budget counts the design tensor and optimizer moments.

The public synthetic structure is a rotated box and a sphere (geometry) or a
16 x 8 periodic density layer (density) on an 800 x 32 x 32 grid whose full
diagonal FP32 epsilon (9,830,400 bytes) exceeds the declared GPU budget
(9,600,000 bytes). The streamed adjoint runs on CUDA with disk banks; the peak
Torch allocated delta must stay below the full epsilon array and within the
budget, and no host tensor of the full material shape may be allocated. The
reservation of every streamed path must grow by exactly the design tensor size
per declared optimizer moment (docs/validation/cases/G5-04.json).
"""
from dataclasses import replace

import pytest
import torch

from torchfdtd import (BoundaryFace, DifferentiableSolid, Monitor, Project, Region, Source, StreamedAdjointOptions,
                       StreamedSimulation, estimate_streamed_memory, plan_streamed_work, streamed_density_layer)
from torchfdtd.models import Boundaries
from torchfdtd.streamed import _reservation
from torchfdtd.streamed_density import StreamedDensitySimulation, _DensityExecution
from torchfdtd.streamed_geometry import StreamedGeometrySimulation, _GeometryExecution, streamed_geometry
from g5_support import CUDA, Record, SOURCE, limits, scene, sphere
from test_streamed_geometry import AllocationTrace

CASE = 'G5-04'
RECORD = Record('G5-04', CASE, 'tests/test_design_slab_memory_g5.py')
cuda = pytest.mark.skipif(not CUDA, reason='CUDA unavailable')
SHAPE = (800, 32, 32)
MESH = .05
GPU_BUDGET = 9_600_000
DESIGN_SHAPE = (16, 8)


def design_project(periodic):
    nx, ny, nz = SHAPE
    faces = {}
    if periodic:
        faces = dict(x_min=BoundaryFace(kind='periodic'), x_max=BoundaryFace(kind='periodic'),
                     y_min=BoundaryFace(kind='periodic'), y_max=BoundaryFace(kind='periodic'))
    region = Region(dimension='3d', size=(nx*MESH, ny*MESH, nz*MESH), mesh=MESH, pml_cells=4, steps=10, precision='float32',
                    backend='cpu', material_sampling='yee', memory_mode='streamed', boundaries=Boundaries(**faces))
    source = dict(SOURCE, center=(0, 0, -.1), pulse_length=.5e-15, pulse_offset=.5e-15)
    project = Project(region=region, sources=[Source(**source)], monitors=[Monitor(component='Ez', center=(0, 0, .1))])
    assert tuple(project.region.shape) == SHAPE, project.region.shape
    return project


def design_options(scratch, **overrides):
    values = dict(device='cuda', slab_width=8, temporal_depth=5, checkpoints=0, gpu_budget_bytes=GPU_BUDGET,
                  host_budget_bytes=8*2**30, state_storage='disk', state_directory=scratch, disk_budget_bytes=2*2**30)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


def geometry_inputs(project):
    params = torch.tensor([3.1, .1, .4], requires_grad=True)
    solids = [DifferentiableSolid.box((.9, .9, .5), epsilon=params[0], center=(params[1], 0, 0), rotation=(0., 0., 20.)),
              DifferentiableSolid.sphere(params[2], epsilon=2.4, center=(-.6, .2, .1))]
    return params, streamed_geometry(project.region, solids, chunk_cells=65536)


def density_inputs(project):
    density = torch.rand(DESIGN_SHAPE, generator=torch.Generator().manual_seed(3)).requires_grad_()
    layer = streamed_density_layer(density, project.region, bottom_um=-.2, top_um=.2, background_epsilon=1., design_epsilon=4.)
    return density, layer


@cuda
@pytest.mark.parametrize('path', ['geometry', 'density'])
def test_design_slabs_never_materialize_the_full_epsilon_or_vjp(path, tmp_path):
    full_epsilon_bytes = SHAPE[0]*SHAPE[1]*SHAPE[2]*3*4
    assert GPU_BUDGET < full_epsilon_bytes == limits(CASE, 'full_epsilon_bytes')
    project = design_project(periodic=path == 'density')
    parameters, material = (geometry_inputs if path == 'geometry' else density_inputs)(project)
    model = (StreamedGeometrySimulation if path == 'geometry' else StreamedDensitySimulation)(project, design_options(tmp_path/'banks'))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with AllocationTrace() as trace:
        result = model(material)
        gradient, = torch.autograd.grad(result.signals.square().sum(), parameters)
    report = result.report
    peak = max(report[phase]['peak_torch_allocated_bytes']['delta_bytes'] for phase in ('forward_memory', 'backward_memory'))
    assert all(report[phase]['peak_torch_allocated_bytes']['exact'] for phase in ('forward_memory', 'backward_memory'))
    assert peak < full_epsilon_bytes and peak <= GPU_BUDGET and report['gpu_reservation_bytes'] <= GPU_BUDGET
    full_shapes = (SHAPE, SHAPE+(3,), SHAPE+(1,))
    full = [(shape, size) for shape, size in trace.allocations if shape in full_shapes and size > 4]
    assert not full, full[:5]
    assert report['dense_material_storage_bytes'] == 0 and report['dense_material_vjp_bytes'] == 0
    assert report['geometry_parameter_bytes'] == material.parameters.numel()*4
    assert report['design_tensor_bytes'] == material.parameters.numel()*4
    assert bool(torch.isfinite(gradient).all()) and gradient.abs().max() > 0
    assert report['forward_backing_store']['closed'] and report['backward_backing_store']['closed']
    RECORD.add(path, path=path, shape=list(SHAPE), cells=SHAPE[0]*SHAPE[1]*SHAPE[2], full_epsilon_bytes=full_epsilon_bytes,
               gpu_budget_bytes=GPU_BUDGET, gpu_reservation_bytes=report['gpu_reservation_bytes'],
               peak_torch_allocated_delta_bytes=peak, peak_over_full_epsilon=peak/full_epsilon_bytes,
               max_host_allocation_bytes=max(size for _, size in trace.allocations),
               full_shape_host_allocations=len(full), design_parameters=material.parameters.numel(),
               gradient_max_abs=float(gradient.abs().max()), forward_seconds=report['forward_seconds'],
               backward_seconds=report['backward_seconds'], max_extended_tile_cells=report['max_extended_tile_cells'])


# ----------------------------------------------------------------------------------------------- budget
def test_dense_reservation_grows_by_the_design_tensor_and_optimizer_moments():
    project = scene('float32', 'point')
    epsilon = sphere(project, inside=4., outside=1.)
    design = epsilon.numel()*epsilon.element_size()
    base = StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4)
    plain = _reservation(project, epsilon, base)
    assert plain['design_tensor_bytes'] == design and plain['optimizer_moments'] == 0
    assert plain['optimizer_state_reservation_bytes'] == 0
    assert plain['dense_parameter_reservation_bytes'] == plain['dense_parameter_multiplier']*design >= design
    for moments in (1, 2):
        with_moments = _reservation(project, epsilon, replace(base, optimizer_moments=moments))
        assert with_moments['optimizer_state_reservation_bytes'] == moments*design
        assert with_moments['host_reservation_bytes']-plain['host_reservation_bytes'] == moments*design
        assert with_moments['gpu_reservation_bytes'] == plain['gpu_reservation_bytes']
        estimate = estimate_streamed_memory(project, replace(base, optimizer_moments=moments))
        assert estimate['host_reservation_bytes'] == with_moments['host_reservation_bytes']
    plan = plan_streamed_work(project, replace(base, optimizer_moments=2), candidates=[replace(base, optimizer_moments=2)])
    assert plan.report['candidates'][0]['memory']['optimizer_state_reservation_bytes'] == 2*design
    result = StreamedSimulation(project, replace(base, optimizer_moments=2))(epsilon)
    assert result.report['optimizer_state_reservation_bytes'] == 2*design and result.report['design_tensor_bytes'] == design


def test_design_path_reservations_grow_by_the_design_tensor_and_optimizer_moments(tmp_path):
    project = scene('float32', 'point')
    _, geometry = geometry_inputs(project)
    periodic = design_project(True)
    density, layer = density_inputs(periodic)
    base = StreamedAdjointOptions(device='cpu', slab_width=8, temporal_depth=5)
    for name, execution, value, scene_project in (('geometry', _GeometryExecution(geometry), geometry.parameters, project),
                                                  ('density', _DensityExecution(layer), density, periodic)):
        design = value.numel()*value.element_size()
        plain = execution.reservation(scene_project, value, base, None)
        assert plain['design_tensor_bytes'] == design and plain['optimizer_state_reservation_bytes'] == 0
        for moments in (1, 2):
            grown = execution.reservation(scene_project, value, replace(base, optimizer_moments=moments), None)
            assert grown['optimizer_state_reservation_bytes'] == moments*design, name
            assert grown['host_reservation_bytes']-plain['host_reservation_bytes'] == moments*design, name
        # A budget that admits the design without moments refuses it with them.
        tight = replace(base, host_budget_bytes=plain['host_reservation_bytes'])
        execution.reservation(scene_project, value, tight, None)
        with pytest.raises(ValueError, match='host budget'):
            execution.reservation(scene_project, value, replace(tight, optimizer_moments=2), None)


def test_adam_moments_match_the_declared_optimizer_state():
    project = scene('float32', 'point')
    _, geometry = geometry_inputs(project)
    leaf = geometry.parameters.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([leaf], lr=1e-2)
    leaf.sum().backward()
    optimizer.step()
    state = optimizer.state[leaf]
    moments = [v for v in state.values() if isinstance(v, torch.Tensor) and v.shape == leaf.shape]
    assert len(moments) == 2
    reservation = _GeometryExecution(geometry).reservation(project, geometry.parameters,
        StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4, optimizer_moments=len(moments)), None)
    assert reservation['optimizer_state_reservation_bytes'] == sum(m.numel()*m.element_size() for m in moments)


def test_optimizer_moments_option_is_validated():
    for invalid in (-1, 9, True, 1.5):
        with pytest.raises(ValueError, match='optimizer_moments'):
            StreamedAdjointOptions(optimizer_moments=invalid)
