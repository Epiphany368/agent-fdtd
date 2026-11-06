import json
import numpy as np
import pytest
import torch
from torchfdtd.models import Project,Region,Material,Structure,Source,Monitor
from torchfdtd.endpoint_project import endpoint_from_project
from torchfdtd.pmc_reference import EndpointTopology
from torchfdtd.solver import voxelize
from torchfdtd.waveforms import source_time_signal


def project():
    return Project(region=Region(dimension='3d',size=(.6,.6,.6),mesh=.1,steps=10,
        backend='cpu',material_sampling='yee',boundaries={a+'_'+side:{'kind':'periodic'} for a in 'xyz' for side in ('min','max')}),materials=[Material(name='core',index=2)],
        structures=[Structure(material='core',center=(.3,.3,0),size=(.1,.1,.6))],
        sources=[Source(component='Ez',center=(.3,.3,0),pulse='continuous')],
        monitors=[Monitor(component='Ez',center=(.3,.3,0)),Monitor(component='Hx',center=(.3,0,0))])


def test_project_endpoint_material_waveforms_and_native_result():
    p=project();adapter=endpoint_from_project(p.model_dump(),boundary_faces=[('pmc','pmc')]*3)
    epsilon=adapter.rasterize(chunk_size=7);source=adapter.simulation.source_ids[0]
    assert epsilon[source]==4
    # Native base-volume Yee map agrees, while endpoint material requires extra samples.
    native,_=voxelize(p)
    torch.testing.assert_close(epsilon[:native.size],torch.from_numpy(native.ravel()))
    reference=EndpointTopology(adapter.simulation.topology.nodes,adapter.simulation.topology.faces)
    xyz=reference.coordinates['E'];expected=np.where((abs(xyz[:,0]-.3)<=.05)&(abs(xyz[:,1]-.3)<=.05)&(abs(xyz[:,2])<=.3),4.,1.)
    np.testing.assert_array_equal(epsilon.numpy(),expected)
    drive=adapter.waveforms()
    np.testing.assert_allclose(drive[:,0].numpy(),source_time_signal(p.resolved_source(p.sources[0]),np.arange(1,11)*p.region.time_step),rtol=1e-6,atol=1e-9)
    result=adapter();assert result.signals.shape==(10,2) and result.signals.abs().max()>0
    assert result.monitor_components==('Ez','Hx')
    assert adapter.plan()['source_terms'][0]['index'][:2]==[6,6]
    json.dumps(adapter.plan())


def test_parameterized_epsilon_and_waveform_gradients():
    adapter=endpoint_from_project(project(),boundary_faces=[('symmetric','symmetric')]*3,checkpoints=2)
    parameter=torch.tensor(2.,requires_grad=True)
    epsilon=adapter.simulation.sample_epsilon(lambda xyz,c:parameter.expand(len(xyz)))
    drive=adapter.waveforms().requires_grad_()
    result=adapter(epsilon,drive)
    gp,gw=torch.autograd.grad(result.signals.square().sum(),(parameter,drive))
    assert torch.isfinite(gp) and gp.abs()>0 and gw.abs().max()>0
    assert result.report['execution']['reverse_steps']==10


def test_reject_disabled_wall_and_unsupported_contracts():
    p=project();p.sources[0].enabled=False
    with pytest.raises(ValueError,match='disabled'):endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    with pytest.raises(ValueError,match='outside'):endpoint_from_project(project(),boundary_faces=[('pec','pec')]*3)
    p=project();p.monitors[0].time_downsample=2
    with pytest.raises(ValueError,match='downsample'):endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    p=project();p.sources[0].kind='plane';p.sources[0].center=(0,0,0);p.sources[0].size=(0,0,0)
    with pytest.raises(ValueError,match='point soft'):endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    adapter=endpoint_from_project(project(),boundary_faces=[('pmc','pmc')]*3)
    adapter.project.sources[0].amplitude=2
    with pytest.raises(ValueError,match='changed'):adapter.waveforms()


def test_lower_wall_nearest_sample_rejects_not_moves():
    p=project();p.sources[0].center=(-.3,-.3,0)
    with pytest.raises(ValueError,match='constrained'):endpoint_from_project(p,boundary_faces=[('pec','pmc')]*3)


def test_preparation_admission_before_raster_or_waveforms():
    with pytest.raises(ValueError,match='preparation'):
        endpoint_from_project(project(),boundary_faces=[('pmc','pmc')]*3,host_preparation_budget_bytes=1)
    p=project();p.sources[0].theta=90;p.sources[0].phi=45;p.sources[0].center=(0,0,0)
    adapter=endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    assert len(adapter.plan()['source_terms'])==2
    assert adapter.waveforms().shape==(10,2)


def test_fixed_explicit_nodes_and_graded_mesh_rejection():
    p=project();coordinates=(-.3,-.22,-.1,0.,.05,.2,.3)
    p.region.mesh_type='explicit';p.region.mesh_coordinates=(coordinates,)*3
    for item in p.sources+p.monitors:item.center=tuple(.299 if x==.3 else x for x in item.center)
    adapter=endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    assert adapter.plan()['nodes_um']==[list(coordinates)]*3
    assert torch.isfinite(adapter().signals).all()
    p=project();p.region.mesh_type='graded'
    with pytest.raises(ValueError,match='graded'):
        endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)


def test_coincident_source_terms_are_not_silently_merged():
    p=project();p.sources.append(Source(component='Ez',center=p.sources[0].center))
    with pytest.raises(ValueError,match='unique'):
        endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
