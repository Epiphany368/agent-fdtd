"""Plane plans and solver meshes agree across dimensions, mesh kinds, paths, devices and execution."""
from functools import lru_cache

import numpy as np
import pytest
import torch

from torchfdtd import (Project,Region,Source,Structure,FieldMonitor,SpectrumSettings,Simulation,
                        AdjointOptions,StreamedAdjointOptions,DifferentiablePlaneSimulation)
from torchfdtd.adjoint_planes import COMPONENTS
from torchfdtd.field_monitors import plane_plan
from torchfdtd.solver import voxelize,field_axes

FREQUENCIES=[2.5e14,3e14]
DIMENSIONS=['2d','3d']
MESHES=['uniform','graded','explicit']
DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]
MONITORS=('transmission','reflection','side')


def scene(dimension,mesh,device):
    """3.2 um square or cube with a dielectric block, one point source and three field planes."""
    def spectrum():return SpectrumSettings(sampling='custom',custom_frequencies_hz=FREQUENCIES,apodization='none')
    r=Region(dimension=dimension,size=(3.2,3.2,3.2),mesh=.1,mesh_type='graded' if mesh=='graded' else 'uniform',
             material_sampling='yee',mesh_max=.3,mesh_ppw=6,pml_cells=3,steps=72,backend=device,precision='float64',
             snapshot_interval=72)
    p=Project(region=r,structures=[Structure(id='block',center=(.9,0,0),size=(.2,.6,.6))],
        sources=[Source(id='source',center=(-.4,0,0),wavelength=1,pulse_cycles=1)],
        monitors=[FieldMonitor(id='transmission',center=(.4,0,0),size=(0,1,1),spectrum=spectrum()),
                  FieldMonitor(id='reflection',center=(-.9,0,0),size=(0,1,1),spectrum=spectrum()),
                  FieldMonitor(id='side',normal='y',center=(0,.9,0),size=(1.2,0,1),spectrum=spectrum())])
    p=Project.model_validate(p.model_dump())
    if mesh=='explicit':
        # The realized graded nodes become fixed coordinates without automatic meshing.
        graded=scene(dimension,'graded',device).region
        p.region.mesh_type='explicit';p.region.mesh_coordinates=tuple(tuple(map(float,a)) for a in graded.mesh_nodes)
        p=Project.model_validate(p.model_dump())
        assert p.region.shape==graded.shape
    assert (p.region.shape!=p.region.base_shape)==(mesh!='uniform')
    assert max(p.region.shape)<=32
    return p


def assert_same_nodes(actual,expected):
    assert tuple(len(a)-1 for a in actual)==tuple(len(b)-1 for b in expected)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def assert_same_sampling(actual,expected):
    assert actual.time_step==expected.time_step
    for c in COMPONENTS:
        for a,b in zip(field_axes(actual,c),field_axes(expected,c)):np.testing.assert_array_equal(a,b)


@lru_cache(maxsize=None)
def ordinary(dimension,mesh,device):
    """Ordinary Simulation plane monitors on the project's own mesh."""
    p=scene(dimension,mesh,device)
    result=Simulation(p).run()
    assert result.summary['backend']==device and result.summary['steps']==p.region.steps
    assert_same_nodes(result.project.region.mesh_nodes,p.region.mesh_nodes)
    assert_same_sampling(result.project.region,p.region)
    records={}
    for monitor,record in zip(p.monitors,result.frequency_fields):
        plan=plane_plan(p.region,p.resolved_monitor(monitor))
        assert record['id']==monitor.id
        np.testing.assert_array_equal(record['points_um'],plan['points_um'])
        np.testing.assert_array_equal(record['weights'],plan['weights'])
        assert record['shape']==plan['shape'] and record['fields'].shape==(len(FREQUENCIES),len(plan['weights']),6)
        assert np.abs(record['fields']).max()>0 and np.isfinite(record['fields']).all()
        records[monitor.id]=record
    assert tuple(records)==MONITORS
    return p,records


@pytest.mark.parametrize('dimension',DIMENSIONS)
@pytest.mark.parametrize('mesh',MESHES)
@pytest.mark.parametrize('device',DEVICES)
def test_ordinary_plane_monitors_use_the_project_mesh(dimension,mesh,device):
    p,records=ordinary(dimension,mesh,device)
    assert len({r['run_signature'] for r in records.values()})==1 and records['side']['run_signature']
    assert_same_nodes(p.region.mesh_nodes,scene(dimension,mesh,device).region.mesh_nodes)


@pytest.mark.parametrize('dimension',DIMENSIONS)
@pytest.mark.parametrize('mesh',MESHES)
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('execution',['resident','streamed'])
def test_differentiable_planes_share_the_solver_mesh_and_spectra(dimension,mesh,device,execution):
    if device=='cuda':pytest.importorskip('cupy')
    p,records=ordinary(dimension,mesh,device)
    if execution=='resident':options=AdjointOptions(checkpoints=2)
    else:options=StreamedAdjointOptions(device=device,slab_width=8,temporal_depth=4,checkpoints=2,
                                         tile_transfers='async' if device=='cuda' else 'sync')
    model=DifferentiablePlaneSimulation(p,options)
    for region in (model.project.region,model.model.project.region):
        assert_same_nodes(region.mesh_nodes,p.region.mesh_nodes)
        assert_same_sampling(region,p.region)
    assert model.model.project.region.shape==p.region.shape
    epsilon,_=voxelize(p)
    epsilon=torch.as_tensor(epsilon,dtype=torch.float64,device=device if execution=='resident' else 'cpu')
    with torch.no_grad():planes=model(epsilon,FREQUENCIES)
    assert tuple(planes)==MONITORS
    for identifier,record in records.items():
        plane=planes[identifier]
        np.testing.assert_array_equal(plane.points_um.cpu().numpy(),record['points_um'])
        np.testing.assert_array_equal(plane.weights.cpu().numpy(),record['weights'])
        assert plane.shape==record['shape'] and plane.normal==record['normal_axis']
        assert (plane.run_signature,plane.run_fingerprint)==(model.signature,model.fingerprint)
        fields=plane.fields.cpu().numpy();scale=np.abs(record['fields']).max()
        np.testing.assert_allclose(fields/scale,record['fields']/scale,rtol=1e-10,atol=1e-11)
        flux=plane.flux().cpu().numpy();flux_scale=np.abs(record['flux']).max()
        np.testing.assert_allclose(flux/flux_scale,record['flux']/flux_scale,rtol=1e-10,atol=1e-11)


def test_material_gradient_on_the_graded_mesh_matches_finite_differences():
    p,_=ordinary('3d','graded','cpu')
    model=DifferentiablePlaneSimulation(p,AdjointOptions(checkpoints=2))
    epsilon,_=voxelize(p)
    mask=torch.as_tensor(epsilon>1,dtype=torch.float64)
    assert mask.sum()>0
    with torch.no_grad():reference=model(torch.ones_like(mask),FREQUENCIES)
    assert reference['transmission'].run_signature==model.signature
    def loss(value):
        planes=model(1+value*mask,FREQUENCIES)
        return sum(planes[k].normalized_flux(reference[k]).sum() for k in MONITORS)
    value=torch.tensor(1.5,dtype=torch.float64,requires_grad=True)
    gradient,=torch.autograd.grad(loss(value),value)
    with torch.no_grad():finite=(loss(value+1e-4)-loss(value-1e-4))/2e-4
    torch.testing.assert_close(gradient,finite,rtol=1e-6,atol=1e-9)
    assert abs(gradient)>1e-4
