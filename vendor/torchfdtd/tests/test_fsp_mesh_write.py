"""Authored FSP grids, independent coordinate checks and native EM roundtrips."""
import math

import numpy as np
import pytest

from torchfdtd import Project, Simulation, run_tensor_batch
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_geometry import write_fsp_scene, write_fsp_geometry
from torchfdtd.fsp_native import FDTD, convert_fsp
from test_fsp_native import fixture
from test_fsp_mesh import settings
from test_fsp_settings_write import imported, roundtrip, settings_fixture


def remeshed(project, **changes):
    data=project.model_dump()
    data['region'].update(changes)
    return Project.model_validate(data)


def authored_2d():
    h=.1e-6;dt=.8*h/(299792458*math.sqrt(2))
    return fixture(region_overrides=dict(dimension=0,dt=dt,MaxSimTime=20*dt))


@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('axis_steps',[None,(.12,.16,.2)])
@pytest.mark.parametrize('periodic',[False,True])
def test_uniform_input_controls_nodes_pml_and_dt(dimension,axis_steps,periodic,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Commercial API loaded'))
    doc,p=imported(authored_2d() if dimension=='2d' else fixture())
    r=p.region;r.boundaries.x_min.layers=3;r.boundaries.x_max.layers=5
    if periodic:r.boundaries.y_min.kind=r.boundaries.y_max.kind='periodic'
    p=remeshed(p,mesh=.12,mesh_steps=axis_steps,material_sampling='yee' if axis_steps else 'cell',
               size=(4.45,4.75,4.31 if dimension=='3d' else .1),steps=37,courant_factor=.71)
    q,report=roundtrip(doc,p)
    out,_=write_fsp_scene(doc,p);props=next(n for n in out.root.children if n.uid==FDTD).properties
    assert props['customGrid'].value==2
    assert report['mesh_export']['nodes_changed']
    assert report['mesh_export']['shape_after']==q.region.shape
    assert report['mesh_export']['external_remeshing_verified'] is False
    assert q.region.material_sampling==p.region.material_sampling
    active=2 if dimension=='2d' else 3
    for a in range(active):
        axis='xyz'[a];h=(axis_steps or (.12,)*3)[a]
        n=math.ceil(p.region.size[a]/h)
        want=(np.arange(n+1)-n/2)*h*1e-6
        # Verify the serialized lattice independently of convert_fsp and
        # Region.mesh_nodes, including the saved duplicate periodic planes.
        stored=np.r_[want[0]-h*1e-6,want,want[-1]+h*1e-6] if periodic and a==1 else want
        np.testing.assert_allclose(props[axis+'Grid'].value.reshape(-1),stored,rtol=0,atol=3e-21)
        assert props['d'+axis].value==pytest.approx(h*1e-6)
        lo=want[p.region.pml_layers(a,0)];hi=want[-1-p.region.pml_layers(a,1)]
        if a<2:
            assert props['GUI'+axis].value==pytest.approx((lo+hi)/2,abs=1e-21)
            assert props['GUI'+('width' if a==0 else 'height')].value==pytest.approx(hi-lo)
        else:
            assert props['GUIz1'].value==pytest.approx(lo)
            assert props['GUIz2'].value==pytest.approx(hi)
    want_dt=.71*1e-6/(299792458*math.sqrt(sum(1/h**2 for h in (axis_steps or (.12,)*3)[:active])))
    assert props['dt'].value==pytest.approx(want_dt,rel=1e-14)
    assert props['MaxSimTime'].value==pytest.approx(37*want_dt,rel=1e-14)
    assert q.region.time_step==pytest.approx(want_dt,rel=1e-14)
    np.testing.assert_allclose(q.import_provenance.origin_m,p.import_provenance.origin_m,atol=1e-21)


@pytest.mark.parametrize('axis_steps',[None,(.12,.16,.2)])
def test_smaller_fixed_dt_is_encoded_by_controlling_cfl(axis_steps):
    doc,p=imported(fixture())
    p=remeshed(p,mesh=.12,mesh_steps=axis_steps,material_sampling='yee' if axis_steps else 'cell')
    p.region.time_step_override=p.region.cfl_time_step*.6
    q,report=roundtrip(doc,p)
    assert q.region.courant_factor==pytest.approx(p.region.courant_factor*.6)
    assert q.region.time_step==pytest.approx(p.region.time_step,rel=1e-14)
    assert any('effective dt encoded' in s for s in report['native_only_settings'])
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,rtol=3e-6,atol=2e-7)


def test_unchanged_uniform_grid_can_encode_smaller_fixed_dt():
    doc,p=imported(fixture());p.region.time_step_override=p.region.cfl_time_step*.73
    q,report=roundtrip(doc,p)
    assert not report['mesh_export']['nodes_changed']
    assert q.region.time_step==pytest.approx(p.region.time_step,rel=1e-14)


@pytest.mark.parametrize('mode',[0,1])
def test_saved_nonuniform_to_uniform_and_repeat_edit(mode):
    doc,p=imported(fixture(region_overrides=settings(mode)))
    p=remeshed(p,mesh_type='uniform',mesh=.13,mesh_steps=(.13,.16,.2),mesh_coordinates=None,
               time_step_override=None,mesh_auto_refine=False)
    q,report=roundtrip(doc,p)
    assert report['mesh_export']['nodes_changed'] and q.region.material_sampling=='yee'
    out,_=write_fsp_scene(doc,p)
    q=remeshed(q,mesh_type='uniform',mesh=.16,mesh_steps=(.16,.2,.24),mesh_coordinates=None,time_step_override=None)
    r,_=roundtrip(out,q)
    assert r.region.shape==q.region.shape


def test_geometry_only_api_still_rejects_remeshing():
    doc,p=imported(fixture());p.region.mesh=.12
    with pytest.raises(ValueError,match='region.mesh'):write_fsp_geometry(doc,p)


@pytest.mark.parametrize('changes,match',[
    (dict(mesh_type='graded',material_sampling='yee'),'uniform target'),
    (dict(material_sampling='yee'),'material sampling'),
])
def test_unmapped_generator_or_sampling_cannot_silently_change(changes,match):
    doc,p=imported(fixture());p=remeshed(p,**changes);before=doc.data
    with pytest.raises(ValueError,match=match):write_fsp_scene(doc,p)
    assert doc.data==before


def test_explicit_node_edit_is_rejected_until_input_generator_is_mapped():
    doc,p=imported(fixture(region_overrides=settings()))
    coords=[list(a) for a in p.region.mesh_coordinates];coords[0][12]+=.003
    p=remeshed(p,mesh_coordinates=coords)
    with pytest.raises(ValueError,match='uniform target'):write_fsp_scene(doc,p)


@pytest.mark.parametrize('axis_steps',[None,(.12,.16,.2)])
def test_remeshed_spectral_scene_cpu_cuda_and_shared_batch(axis_steps):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    doc,p=imported(settings_fixture())
    p=remeshed(p,mesh=.12,mesh_steps=axis_steps,material_sampling='yee' if axis_steps else 'cell',steps=80)
    p.structures[0].radius=.347;p.sources[0].center=(-.53,.017,.023)
    p.monitors[0].center=(.57,.011,.037);p.monitors[0].spatial_interpolation='specified'
    q,_=roundtrip(doc,p)
    a,b=Simulation(p).run(),Simulation(q).run()
    # This tests the edited scene against its reimported scene. It does not
    # claim optical equivalence across two different mesh resolutions.
    np.testing.assert_allclose(a.electric,b.electric,rtol=3e-6,atol=2e-7)
    np.testing.assert_allclose(a.magnetic,b.magnetic,rtol=3e-6,atol=2e-7)
    q.region.backend='cuda';q.region.cuda_kernel='fused';q.region.cuda_monitor_kernel='fused'
    gpu=Simulation(q).run();batch=run_tensor_batch([q,q.model_copy(deep=True)])
    batch.raise_for_errors()
    np.testing.assert_allclose(a.electric,gpu.electric,rtol=3e-5,atol=2e-6)
    scale=np.max(abs(a.frequency_fields[0]['fields']));assert scale>0
    np.testing.assert_allclose(a.frequency_fields[0]['fields']/scale,gpu.frequency_fields[0]['fields']/scale,rtol=4e-5,atol=4e-5)
    for item in batch.items:
        np.testing.assert_array_equal(item.result.electric,gpu.electric)
        np.testing.assert_array_equal(item.result.magnetic,gpu.magnetic)
        np.testing.assert_array_equal(item.result.frequency_fields[0]['fields'],gpu.frequency_fields[0]['fields'])


@pytest.mark.parametrize('monitor',['point','plane'])
def test_two_dimensional_invariant_span_has_no_native_field_or_flux_effect(monitor):
    from test_fsp_monitors import spectral_fixture
    dt=.8*.1e-6/(299792458*math.sqrt(2))
    raw=spectral_fixture('x',dict(monitorShape=2,z1=0.,z2=0.),dict(dimension=0,dt=dt,MaxSimTime=20*dt))
    if monitor=='point':raw=authored_2d()
    doc,p=imported(raw);p=remeshed(p,mesh=.13,size=(4.8,4.8,1.7),steps=80)
    q,report=roundtrip(doc,p)
    assert q.region.size[2]==pytest.approx(.13)
    assert any('invariant 2D' in s for s in report['native_only_settings'])
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,rtol=1e-6,atol=1e-8)
    if monitor=='point':
        assert np.max(abs(a.signals))>0
        np.testing.assert_allclose(a.signals,b.signals,rtol=1e-6,atol=1e-8)
        return
    scale=np.max(abs(a.frequency_fields[0]['fields']));assert scale>0
    np.testing.assert_allclose(a.frequency_fields[0]['fields']/scale,b.frequency_fields[0]['fields']/scale,rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(a.frequency_fields[0]['flux'],b.frequency_fields[0]['flux'],rtol=1e-6,atol=1e-35)


def test_translated_original_keeps_global_geometry_and_domain_origin():
    offsets=(.23,-.11,.37)
    changes=dict(GUIx=offsets[0]*1e-6,GUIy=offsets[1]*1e-6,
                 GUIz1=(-2+offsets[2])*1e-6,GUIz2=(2+offsets[2])*1e-6)
    changes.update({a+'Grid':(np.linspace(-2.4,2.4,49)+offsets[i])*1e-6 for i,a in enumerate('xyz')})
    doc,p=imported(fixture(region_overrides=changes));p=remeshed(p,mesh=.14,size=(4.7,4.5,4.8))
    q,_=roundtrip(doc,p)
    np.testing.assert_allclose(q.import_provenance.origin_m,np.array(offsets)*1e-6,rtol=0,atol=1e-21)
    np.testing.assert_allclose(q.structures[0].center,p.structures[0].center,atol=1e-15)
    np.testing.assert_allclose(q.sources[0].center,p.sources[0].center,atol=1e-15)


def test_dimension_change_remains_explicitly_unmapped():
    doc,p=imported(fixture());p=remeshed(p,dimension='2d')
    with pytest.raises(ValueError,match='region.dimension'):write_fsp_scene(doc,p)
