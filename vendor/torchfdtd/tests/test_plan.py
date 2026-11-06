"""The resolved simulation plan: one immutable record every public entry point consumes (G2-01, G2-02)."""
import sys
from pathlib import Path

# benchmarks/ and examples/ ship with the repository, not with the wheel, so a run against an
# installed package still reads them from the checkout that holds this test.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import json

import numpy as np
import pytest
import torch

from torchfdtd import (Project, Region, Structure, Source, Monitor, FieldMonitor, Material, BoundaryFace,
                       Simulation, DifferentiableSimulation, StreamedSimulation, DifferentiablePlaneSimulation,
                       StreamedAdjointOptions, run_tensor_batch)
from torchfdtd.plan import SimulationPlan, PlanInvalidated, resolve_plan, SAMPLE_TIME_STEPS, PLACEMENT_FIELDS
from test_differentiable import gpu


def scene(monitors='point', **region):
    settings = dict(dimension='2d', size=(1.6, 1.5, 1.4), mesh=.1, pml_cells=3, steps=24, backend='cpu',
                    material_sampling='yee')
    settings.update(region)
    r = Region(**settings)
    r.boundaries.x_min = BoundaryFace(layers=4, kappa=2, alpha=.03, alpha_polynomial=1)
    observers = ([Monitor(id='ez', component='Ez', center=(.1, 0, 0)), Monitor(id='hy', component='Hy', center=(0, .1, 0))]
                 if monitors == 'point' else [FieldMonitor(id='plane', center=(.12, 0, 0), size=(0, .65, .55), normal='x', downsample=2)])
    return Project(region=r, structures=[Structure(id='block', center=(-.3, .1, 0), size=(.3, .4, .2))],
                   sources=[Source(id='src', center=(-.2, 0, 0), pulse='continuous', wavelength=1.1)], monitors=observers)


def assert_same_plan(a, b):
    """Array-level agreement, not only the hash."""
    assert a.plan_hash == b.plan_hash and a.diff(b) == [k for k in a.diff(b) if k.startswith('placement.')]
    assert a.time_step == b.time_step and a.steps == b.steps and a.shape == b.shape
    for x, y in zip(a.nodes, b.nodes):
        np.testing.assert_array_equal(x, y)
    for component in a.axes:
        for x, y in zip(a.axes[component], b.axes[component]):
            np.testing.assert_array_equal(x, y)
    assert a.boundaries.wrap == b.boundaries.wrap and len(a.boundaries.cpml) == len(b.boundaries.cpml)
    for x, y in zip(a.boundaries.cpml, b.boundaries.cpml):
        assert (x.forward, x.axis, x.component, x.side, x.start, x.stop) == (y.forward, y.axis, y.component, y.side, y.start, y.stop)
        for name in ('kappa', 'sigma', 'alpha', 'b', 'c'):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
    assert len(a.source_terms) == len(b.source_terms)
    for (fa, la, sa, pa), (fb, lb, sb, pb) in zip(a.source_terms, b.source_terms):
        assert fa == fb and la == lb
        np.testing.assert_array_equal(sa, sb)
        assert (pa is None) == (pb is None)
        if pa is not None:
            np.testing.assert_array_equal(pa, pb)
    for x, y in zip(a.ade, b.ade):
        for name in ('a', 'd', 'k'):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
    assert [m.kind for m in a.monitors] == [m.kind for m in b.monitors]
    for x, y in zip(a.monitors, b.monitors):
        assert x.components == y.components and x.index == y.index
        for name in ('points_um', 'weights', 'frequency_hz'):
            if getattr(x, name) is not None:
                np.testing.assert_array_equal(getattr(x, name), getattr(y, name))


def test_resolved_plan_is_immutable_and_placement_stays_outside_the_hash():
    p = scene()
    plan = resolve_plan(p)
    assert isinstance(plan, SimulationPlan) and len(plan.plan_hash) == 64
    with pytest.raises((AttributeError, TypeError)):
        plan.time_step = 1.
    for array in (*plan.nodes, *plan.axes['Ex'], plan.source_terms[0][2], plan.boundaries.cpml[0].b, plan.material.epsilon):
        assert not array.flags.writeable
    assert plan.sample_time_steps == SAMPLE_TIME_STEPS == {'E': 1., 'H': 1.5}
    assert plan.fourier_convention == 'exp(+2 pi i f t)'
    assert plan.plan_hash == resolve_plan(p).plan_hash
    placed = p.model_copy(deep=True)
    placed.region.backend = 'cuda'; placed.region.cuda_kernel = 'fused'; placed.region.cuda_monitor_kernel = 'fused'
    placed.region.memory_mode = 'streamed'; placed.region.precision = 'float64'; placed.region.execution_mode = 'streamed_host'
    placed.region.field = 'Hy'; placed.region.snapshot_interval = 7; placed.name = 'renamed'
    for item in placed.structures + placed.sources + placed.monitors:
        item.id = 'new-' + item.id; item.name = 'renamed'
    placed = Project.model_validate(placed.model_dump())
    other = resolve_plan(placed)
    assert other.plan_hash == plan.plan_hash
    assert set(other.diff(plan)) == {'placement.' + k for k in ('backend', 'cuda_kernel', 'cuda_monitor_kernel', 'memory_mode',
                                                                 'precision', 'execution_mode')}
    assert set(plan.placement) == set(PLACEMENT_FIELDS)
    # The sampled material is a deterministic function of the hashed inputs.
    np.testing.assert_array_equal(plan.material.epsilon, resolve_plan(p).material.epsilon)
    np.testing.assert_array_equal(plan.material.ownership, resolve_plan(p).material.ownership)


def test_plan_json_and_diff_name_the_changed_keys():
    p = scene()
    plan = resolve_plan(p)
    payload = json.loads(json.dumps(plan.to_json(), allow_nan=False))
    assert payload['plan_hash'] == plan.plan_hash and payload['steps'] == 24
    assert len(payload['nodes'][0]) == plan.shape[0] + 1 and len(payload['sources'][0]['terms'][0]['samples']) == 24
    assert payload['boundaries']['cpml'][0]['b'] and payload['placement']['backend'] == 'cpu'
    assert payload['resources']['cells'] == int(np.prod(plan.shape))
    moved = p.model_copy(deep=True); moved.structures[0].center = (-.2, .1, 0)
    moved = Project.model_validate(moved.model_dump())
    assert resolve_plan(moved).diff(plan) == ['material.structures[0].center[0]']
    refined = p.model_copy(deep=True); refined.region.mesh = .08
    refined = Project.model_validate(refined.model_dump())
    keys = resolve_plan(refined).diff(plan)
    assert any(k.startswith('mesh.nodes') for k in keys) and any(k.startswith('time.time_step') for k in keys)
    assert any(k.startswith('exterior.boundaries.cpml') for k in keys) and any(k.startswith('sources[0].terms[0].samples') for k in keys)
    pulsed = p.model_copy(deep=True); pulsed.sources[0].phase = 30
    pulsed = Project.model_validate(pulsed.model_dump())
    keys = resolve_plan(pulsed).diff(plan)
    assert keys == ['sources[0].terms[0].samples']
    # A setting the sampled waveform does not read is carried but not hashed.
    inert = p.model_copy(deep=True); inert.sources[0].wavelength_stop = 1.9
    inert = Project.model_validate(inert.model_dump())
    assert resolve_plan(inert).diff(plan) == [] and resolve_plan(inert).sources[0].settings['wavelength_stop'] == 1.9
    with pytest.raises(TypeError):
        plan.diff(object())


def test_pml_dispersion_enters_the_exterior_section_and_the_hash():
    plan = resolve_plan(scene())
    frozen = resolve_plan(scene(pml_dispersion='frozen'))
    assert plan.pml_dispersion == 'ade' and frozen.pml_dispersion == 'frozen'
    assert frozen.plan_hash != plan.plan_hash
    assert frozen.diff(plan) == ['exterior.pml_dispersion']
    assert frozen.sections['exterior']['pml_dispersion'] == 'frozen'
    assert frozen.to_json()['pml_dispersion'] == 'frozen'


def test_run_control_enters_the_time_section_through_the_shutoff_decision():
    """The settings the automatic shutoff reads change the completed step count; the divergence checks never change a result."""
    from torchfdtd.models import RunControl
    plan = resolve_plan(scene())
    assert plan.run_control == {'auto_shutoff': False} and plan.sections['time']['run_control'] == {'auto_shutoff': False}
    inert = resolve_plan(scene(run_control=RunControl(min_steps=20, check_interval=5, field_limit=1e3, growth_limit=10)))
    assert inert.plan_hash == plan.plan_hash and inert.diff(plan) == []
    on = resolve_plan(scene(run_control=RunControl(auto_shutoff=True)))
    keys = on.diff(plan)
    assert on.plan_hash != plan.plan_hash and 'time.run_control.auto_shutoff' in keys and all(k.startswith('time.run_control.') for k in keys)
    assert on.to_json()['run_control']['auto_shutoff'] is True and on.to_json()['run_control']['min_steps'] == 100
    tuned = resolve_plan(scene(run_control=RunControl(auto_shutoff=True, min_steps=20)))
    assert tuned.diff(on) == ['time.run_control.min_steps']
    divergence = resolve_plan(scene(run_control=RunControl(auto_shutoff=True, field_limit=1e3, divergence_check=False)))
    assert divergence.plan_hash == on.plan_hash and divergence.diff(on) == []


def test_entry_points_share_the_plan_for_point_monitors(tmp_path):
    from fastapi.testclient import TestClient
    from torchfdtd.server import create_app
    p = scene()
    resident = Simulation(p)
    differentiable = DifferentiableSimulation(p)
    streamed = StreamedSimulation(p, StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2))
    plans = [resident.plan, differentiable.plan, streamed.plan]
    hashes = {resident.plan_hash, differentiable.plan_hash, streamed.plan_hash}
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/validate', json=p.model_dump())
        assert response.status_code == 200
        hashes.add(response.json()['plan_hash'])
    app.state.pool.shutdown()
    result = resident.run()
    hashes.add(result.summary['plan_hash'])
    assert len(hashes) == 1 and hashes == {resolve_plan(p).plan_hash}
    for other in plans[1:]:
        assert_same_plan(plans[0], other)
    # The run consumed the plan's material and sources: the epsilon image is the plan's sampled material.
    np.testing.assert_array_equal(result.epsilon, resident.plan.material.epsilon[:, :, 0, 2])
    epsilon = torch.as_tensor(np.array(resident.plan.material.epsilon), dtype=torch.float32)
    signals = differentiable(epsilon).signals.detach().numpy()
    np.testing.assert_allclose(signals, result.signals, rtol=1e-5, atol=1e-6)


def test_entry_points_share_the_plan_for_field_monitors(tmp_path):
    from fastapi.testclient import TestClient
    from torchfdtd.server import create_app
    p = scene(monitors='field', precision='float64')
    resident = Simulation(p)
    planes = DifferentiablePlaneSimulation(p)
    hashes = {resident.plan_hash, planes.plan_hash}
    app = create_app(tmp_path)
    with TestClient(app) as client:
        hashes.add(client.post('/api/validate', json=p.model_dump()).json()['plan_hash'])
    app.state.pool.shutdown()
    result = resident.run()
    hashes.add(result.summary['plan_hash'])
    assert len(hashes) == 1
    assert_same_plan(resident.plan, planes.plan)
    # The internal solver of the plane wrapper carries the same mesh, time base, boundaries and sources.
    inner = planes.model.plan
    for x, y in zip(inner.nodes, planes.plan.nodes):
        np.testing.assert_array_equal(x, y)
    assert inner.time_step == planes.plan.time_step
    keys = inner.diff(planes.plan)
    assert keys and all(k.startswith('monitors[') for k in keys)
    monitor = planes.plan.monitors[0]
    plane = result.frequency_fields[0]
    np.testing.assert_array_equal(plane['points_um'], monitor.points_um)
    np.testing.assert_array_equal(plane['weights'], monitor.weights)
    np.testing.assert_array_equal(plane['frequency_hz'], monitor.frequency_hz)


def test_tensor_batch_reports_the_shared_plan_hash():
    gpu()
    for monitors in ('point', 'field'):
        p = scene(monitors=monitors, backend='auto')
        expected = Simulation(p).plan
        report = run_tensor_batch([p, p], cuda_graph=False)
        assert report.items[0].status == 'completed'
        for item in report.items:
            assert item.summary['plan_hash'] == expected.plan_hash
            assert_same_plan(resolve_plan(item.result.project), expected)


def test_runtime_grid_and_system_arrays_equal_the_plan():
    from torchfdtd.boundaries import YeeGrid
    from torchfdtd.differentiable import _System
    p = scene(precision='float32')
    plan = resolve_plan(p)
    grid = YeeGrid(Simulation(p).project.region)
    plan.verify_grid(grid)
    assert grid.time_step == plan.time_step
    for segment in plan.boundaries.cpml:
        actual = grid.cpml[segment.forward, segment.axis, segment.component][segment.side if segment.side < len(grid.cpml[segment.forward, segment.axis, segment.component]) else -1]
        assert actual['slice'][segment.axis] == slice(segment.start, segment.stop)
        for name, planned in (('b', segment.b), ('c', segment.c), ('inv_k', 1/segment.kappa)):
            value = np.asarray(actual[name]).reshape(-1)
            np.testing.assert_array_equal(value, planned.astype(value.dtype))
    epsilon = torch.ones(plan.shape + (3,), dtype=torch.float32)
    system = _System(DifferentiableSimulation(p).project, epsilon)
    plan.verify_grid(system.grid)
    terms = [(family, loc, comp, waveform) for family in ('E', 'H') for loc, comp, waveform, _ in system.sources[family]]
    assert len(terms) == len(plan.source_terms)
    for (family, loc, comp, waveform), (component, location, samples, _) in zip(terms, plan.source_terms):
        assert component == family + 'xyz'[comp] and loc == location
        np.testing.assert_array_equal(waveform.numpy(), samples.astype(np.float32))
    # A grid prepared for a different scene is rejected by name.
    other = scene()
    other.region.boundaries.x_min = BoundaryFace(kappa=3)
    other = Project.model_validate(other.model_dump())
    with pytest.raises(PlanInvalidated, match='CPML'):
        plan.verify_grid(YeeGrid(Simulation(other).project.region))
    with pytest.raises(PlanInvalidated, match='time step'):
        resolve_plan(scene(steps=30, courant_factor=.8)).verify_grid(grid)


@pytest.mark.parametrize('wrapper', ['resident', 'streamed', 'planes'])
def test_project_mutation_after_construction_is_rejected(wrapper):
    p = scene(monitors='field' if wrapper == 'planes' else 'point', precision='float64', size=(4.8, 4.8, 1.4),
              mesh_type='graded', mesh_max=.3, mesh_ppw=6, mesh_auto_refine=True, steps=10)
    p.sources[0].wavelength = 3
    p = Project.model_validate(p.model_dump())
    assert p.region.shape != p.region.base_shape
    if wrapper == 'resident':
        model = DifferentiableSimulation(p)
    elif wrapper == 'streamed':
        model = StreamedSimulation(p, StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2))
    else:
        model = DifferentiablePlaneSimulation(p)
    nodes = model.plan.nodes
    epsilon = torch.ones(model.project.region.shape + (3,), dtype=torch.float64)
    def run():
        return model(epsilon, [1e14]) if wrapper == 'planes' else model(epsilon)
    run()
    # Moving the refined structure changes the geometry-dependent mesh on re-validation.
    model.project.structures[0].center = (1.4, .1, 0)
    with pytest.raises(PlanInvalidated, match='changed'):
        run()
    rebuilt = Project.model_validate(model.project.model_dump())
    assert any(not np.array_equal(a, b) for a, b in zip(rebuilt.region.mesh_nodes, nodes))
    assert resolve_plan(rebuilt).plan_hash != model.plan_hash


def test_plan_covers_ade_oneway_tfsf_bloch_and_endpoint_faces():
    from torchfdtd.boundaries import YeeGrid
    from torchfdtd.injection import oneway_tables
    from torchfdtd.materials import MaterialADE
    from torchfdtd.models import demo_project
    from torchfdtd.waveforms import source_time_signal
    r = Region(dimension='2d', size=(3, 1, 1), mesh=.05, steps=40, pml_cells=6, backend='cpu', material_sampling='yee',
               precision='float64')
    r.boundaries.y_min = r.boundaries.y_max = BoundaryFace(kind='periodic')
    materials = [Material(name='Air', index=1),
                 Material(name='drude', model='drude', plasma_rad_s=1.3e16, collision_rad_s=1e14, epsilon_inf=1.5),
                 Material(name='multi', model='multipole', epsilon_inf=2, poles=[
                     dict(resonance_rad_s=2e15, strength_rad_s_squared=4e30, damping_rad_s=1e14),
                     dict(resonance_rad_s=0, strength_rad_s_squared=1e30, damping_rad_s=1e13)])]
    p = Project(region=r, materials=materials,
                structures=[Structure(center=(.8, 0, 0), size=(.2, 3, 1), material='drude'),
                            Structure(center=(.3, 0, 0), size=(.1, 3, 1), material='multi')],
                sources=[Source(id='sheet', kind='plane', injection='oneway', normal='x', center=(-.9, 0, 0), size=(0, 1, 0), component='Ez')],
                monitors=[FieldMonitor(center=(.5, 0, 0), size=(0, 1, 1), normal='x'), Monitor(center=(-.5, 0, 0))])
    plan = resolve_plan(p)
    grid = YeeGrid(Simulation(p).project.region)
    for ade in plan.ade:
        material = next(m for m in p.materials if m.name == ade.material)
        live = MaterialADE(grid, material, np.zeros(1, dtype=np.int64), True)
        for name in ('a', 'd', 'k'):
            np.testing.assert_array_equal(getattr(ade, name), np.asarray(getattr(live, name), dtype=np.float64).reshape(-1))
    source = plan.sources[0]
    assert source.injection == 'oneway' and source.oneway['electric_index'] == source.terms[0].location[0].start
    e_table, h_table = oneway_tables(p.resolved_source(p.sources[0]), p.region)
    np.testing.assert_array_equal(source.terms[0].samples, e_table)
    np.testing.assert_array_equal(source.terms[1].samples, -h_table)
    np.testing.assert_array_equal(source.terms[0].sample_times, np.arange(1, 41)*r.time_step)
    np.testing.assert_array_equal(source.terms[1].sample_times, np.arange(1, 41)*r.time_step+.5*r.time_step)
    from torchfdtd.solver import index_at
    assert plan.monitors[0].kind == 'field' and len(plan.monitors[0].maps) == 6
    assert plan.monitors[1].index == tuple(index_at((-.5, 0, 0), p.region, 'Ez')) == (20, 10, 0)
    # A TFSF box keeps its drive and box metadata; a Bloch axis keeps its wrap phase and sheet profile.
    from benchmarks.tfsf_sources import box_project
    box = box_project()
    box.region.steps = 30
    box = Project.model_validate(box.model_dump())
    tfsf = resolve_plan(box).sources[0]
    assert tfsf.kind == 'tfsf' and tfsf.terms == () and tfsf.tfsf['incident_line_cells'] > 0
    # One-way and TFSF metadata carry no labels: renaming the sources leaves both hashes unchanged.
    for original in (p, box):
        renamed = original.model_copy(deep=True)
        for item in renamed.sources + renamed.monitors + renamed.structures:
            item.id = 'new-' + item.id; item.name = 'renamed'
        assert resolve_plan(Project.model_validate(renamed.model_dump())).plan_hash == resolve_plan(original).plan_hash
    np.testing.assert_array_equal(tfsf.tfsf_drive, source_time_signal(box.sources[0], np.arange(1, 31)*box.region.time_step))
    bloch = scene(precision='float64')
    bloch.region.boundaries.y_min = bloch.region.boundaries.y_max = BoundaryFace(kind='bloch')
    bloch.region.bloch_phase = (0, .3, 0)
    bloch.sources = [Source(kind='plane', normal='x', center=(-.2, 0, 0), size=(0, 1.5, 0), component='Ez')]
    bloch = Project.model_validate(bloch.model_dump())
    wrapped = resolve_plan(bloch)
    assert wrapped.boundaries.wrap == {1: np.exp(.3j)} and wrapped.boundaries.bloch_phase == (0., .3, 0.)
    assert np.iscomplexobj(wrapped.sources[0].terms[0].profile)
    # Stored PMC/symmetric endpoint faces are part of the boundary plan.
    endpoint = resolve_plan(demo_project('pmc'))
    assert set(endpoint.boundaries.pmc_lower) == set(endpoint.boundaries.pmc_upper) == {0, 1, 2}
    np.testing.assert_allclose(list(endpoint.boundaries.pmc_upper.values()), 1., rtol=1e-12)
    assert endpoint.material.epsilon.shape == (17, 17, 17, 3)


def test_tensor_material_projects_resolve_but_defer_their_node_tensor():
    p = Project(region=Region(dimension='3d', size=(.6, .6, .6), mesh=.1, steps=12, backend='cpu', material_sampling='yee',
                              boundaries={a+'_'+s: {'kind': 'periodic'} for a in 'xyz' for s in ('min', 'max')}),
                materials=[Material(name='aniso', model='tensor', epsilon_tensor=(2.3, 2.5, 2.7, .13, .11, .09)), Material(name='glass', index=1.4)],
                structures=[Structure(material='aniso', size=(.3, .3, .3))],
                sources=[Source(component='Ex', center=(0, 0, 0), pulse='continuous')], monitors=[Monitor(component='Ex', center=(.1, 0, 0))])
    plan = resolve_plan(p)
    assert len(plan.plan_hash) == 64 and plan.materials[0]['model'] == 'tensor'
    with pytest.raises(ValueError, match='tensor_from_project'):
        plan.material
