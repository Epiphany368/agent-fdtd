"""Nonuniform metric conservation, physical propagation, and backend parity."""
import fdtd
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torchfdtd import Project,Region,Structure,Source,Monitor,MeshRefinement,Simulation,freeze_refinements
from torchfdtd.models import Boundaries,BoundaryFace
from torchfdtd.boundaries import YeeGrid
from torchfdtd.mesh import mesh_summary
from torchfdtd.solver import field_axes,C0


def project():
    return Project(region=Region(size=(8,6,2),mesh=.05,mesh_type='graded',material_sampling='yee',mesh_max=.12,
                          mesh_ppw=20,pml_cells=8,backend='cpu',precision='float64',steps=400),
                   structures=[Structure(kind='sphere',radius=.4)],
                   sources=[Source(center=(-1.5,0,0),wavelength=1.5)],monitors=[Monitor(center=(1.5,0,0))])


def test_planner_bounds_geometry_changes_and_freeze():
    p=project();r=p.region;s=mesh_summary(p)
    assert s['cell_reduction_percent']>25
    for i,nodes in enumerate(r.mesh_nodes[:2]):
        widths=np.diff(nodes)
        assert nodes[-1]-nodes[0]==pytest.approx(r.actual_size[i])
        assert widths.min()>=r.mesh*(1-1e-10)
        assert widths.max()<=r.mesh_max*(1+1e-10)
        np.testing.assert_allclose(widths[:r.pml_cells+1],r.mesh)
        np.testing.assert_allclose(widths[-r.pml_cells-1:],r.mesh)
        assert max(np.max(widths[1:]/widths[:-1]),np.max(widths[:-1]/widths[1:]))<=r.mesh_grading*(1+1e-10)
    frozen=freeze_refinements(p);old=frozen.region.mesh_nodes
    frozen.structures=[];frozen=Project.model_validate(frozen.model_dump())
    for a,b in zip(old,frozen.region.mesh_nodes):np.testing.assert_array_equal(a,b)
    p.structures[0].size=(5,5,1);p.structures[0].kind='rectangle'
    updated=Project.model_validate(p.model_dump())
    assert np.prod(updated.region.shape)>np.prod(r.shape)
    with pytest.raises(ValueError,match='Yee'):Region(mesh_type='graded')
    with pytest.raises(ValueError,match='positive'):MeshRefinement(size=(0,1,1))


@pytest.mark.parametrize('bloch',[False,True])
def test_nonuniform_curl_adjoint_and_discrete_energy(bloch):
    kind='bloch' if bloch else 'periodic'
    r=Region(dimension='3d',size=(2.4,2,1.6),mesh=.05,mesh_type='graded',material_sampling='yee',mesh_max=.13,
             boundaries=Boundaries(**{f'{a}_{s}':BoundaryFace(kind=kind) for a in 'xyz' for s in ('min','max')}),
             bloch_phase=(.3,-.5,.2) if bloch else (0,0,0),precision='float64')
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64;g=YeeGrid(r)
    rng=np.random.default_rng(237);g.E[:]=rng.normal(size=g.E.shape);g.H[:]=rng.normal(size=g.H.shape)
    if bloch:g.E[:]+=1j*rng.normal(size=g.E.shape);g.H[:]+=1j*rng.normal(size=g.H.shape)
    widths=[np.diff(a) for a in r.mesh_nodes];dual=[(d+np.roll(d,1))/2 for d in widths]
    def weight(electric):
        return np.stack([np.prod(np.array(np.broadcast_arrays(*[(widths[a] if (a==c)==electric else dual[a]).reshape(tuple(-1 if i==a else 1 for i in range(3))) for a in range(3)])),axis=0) for c in range(3)],axis=-1)
    we,wh=weight(True),weight(False)
    left=np.sum(g.E.conj()*g.curl(g.H,False)*we);right=np.sum(g.curl(g.E,True).conj()*g.H*wh)
    assert left==pytest.approx(right,abs=1e-12)
    def energy():return float(np.sum(abs(g.E)**2*we)+np.sum(abs(g.H)**2*wh)+g.courant_number*np.sum(g.E.conj()*g.curl(g.H,False)*we).real)
    initial=energy()
    for _ in range(60):g.update_E();g.update_H()
    assert energy()==pytest.approx(initial,rel=2e-13)


def test_graded_vacuum_wave_speed_and_cpml():
    p=Project(region=Region(size=(10,.6,1),mesh=.025,mesh_type='graded',material_sampling='yee',mesh_max=.075,
              mesh_ppw=20,pml_cells=16,steps=1500,backend='cpu',precision='float64',snapshot_interval=1500,
              boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'),y_max=BoundaryFace(kind='periodic'))),
              sources=[Source(kind='plane',center=(-2,0,0),size=(0,.6,0),wavelength=1.5,pulse_cycles=1)],
              monitors=[Monitor(center=(-1,0,0)),Monitor(center=(1,0,0))])
    graded=Simulation(p).run();p.region.mesh_type='uniform';uniform=Simulation(p).run()
    assert graded.summary['cells']<uniform.summary['cells']*.65
    t=graded.times;use=t<35e-15
    near,far=graded.signals[use,0],graded.signals[use,1]
    delay=(np.argmax(np.correlate(far,near,mode='full'))-len(near)+1)*(t[1]-t[0])
    assert delay==pytest.approx(2e-6/C0,rel=.02)
    assert np.max(abs(graded.signals[t>65e-15]))/np.max(abs(graded.signals))<.006
    f=np.linspace(C0/1.8e-6,C0/1.3e-6,41);kernel=np.exp(2j*np.pi*f[:,None]*t)
    ug=kernel@graded.signals;uu=kernel@uniform.signals
    coarse_error=np.max(abs(ug[:,1]/ug[:,0]-uu[:,1]/uu[:,0]))
    assert coarse_error<.04
    p.region.mesh_type='graded';p.region.mesh_ppw=30
    finer=Simulation(p).run();uf=kernel@finer.signals
    assert np.max(abs(uf[:,1]/uf[:,0]-uu[:,1]/uu[:,0]))<coarse_error*.6


@pytest.mark.parametrize('precision',['float32','float64'])
def test_graded_dispersive_bloch_gpu_and_physical_output(precision,tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=project();p.region.steps=80;p.region.precision=precision
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch';p.region.bloch_phase=(0,.4,0)
    p.materials[2].model='drude';p.materials[2].epsilon_inf=2
    cpu=Simulation(p).run();p.region.backend='cuda';gpu=Simulation(p).run()
    assert gpu.summary['cuda_graph'] and gpu.summary['cell_reduction_percent']>25
    tol=3e-5 if precision=='float32' else 3e-12
    np.testing.assert_allclose(cpu.electric,gpu.electric,atol=tol,rtol=tol)
    np.testing.assert_allclose(cpu.signals,gpu.signals,atol=tol,rtol=tol)
    assert gpu.frames.shape[1:]==p.region.base_shape[:2]
    gpu.save(tmp_path/'mesh.npz');d=np.load(tmp_path/'mesh.npz')
    for i,a in enumerate('xyz'):np.testing.assert_array_equal(d[f'mesh_{a}_um'],gpu.project.region.mesh_nodes[i])
    ax=field_axes(gpu.project.region,'Ex');nodes=gpu.project.region.mesh_nodes
    np.testing.assert_array_equal(ax[0],(nodes[0][:-1]+nodes[0][1:])/2)
    np.testing.assert_array_equal(ax[1],nodes[1][:-1])


def test_mesh_api_preview_and_freeze(tmp_path):
    from torchfdtd.server import create_app
    c=TestClient(create_app(tmp_path));p=project().model_dump()
    response=c.post('/api/mesh/preview',json=p);assert response.status_code==200
    data=response.json();assert data['summary']['cell_reduction_percent']>25 and data['refinements']
    response=c.post('/api/mesh/freeze',json=p);assert response.status_code==200
    assert not response.json()['region']['mesh_auto_refine']


def test_graded_slab_transmission_against_fresnel():
    p=Project(region=Region(size=(8,.5,1),mesh=.025,mesh_type='graded',material_sampling='yee',mesh_ppw=24,
              pml_cells=16,steps=1500,backend='cpu',precision='float64',snapshot_interval=1500,
              boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'),y_max=BoundaryFace(kind='periodic'))),
              structures=[Structure(center=(.0125,0,0),size=(.2,.5,1),material='SiO2 (constant n)')],
              sources=[Source(kind='plane',center=(-1.5,0,0),size=(0,.5,0),wavelength=1.55,pulse_cycles=2)],
              monitors=[Monitor(center=(1.5,0,0))])
    p.materials[1].index=1.5;p=freeze_refinements(p)
    slab=Simulation(p).run();p.structures=[];air=Simulation(p).run()
    wavelength=np.linspace(1.3,1.8,41);kernel=np.exp(2j*np.pi*(C0/(wavelength*1e-6))[:,None]*slab.times)
    transmission=abs((kernel@slab.signals[:,0])/(kernel@air.signals[:,0]))**2
    expected=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    coarse_error=np.max(abs(transmission-expected))
    assert coarse_error<.02
    p.region.mesh_ppw=48
    air_fine=Simulation(p).run();p.structures=slab.project.structures
    slab_fine=Simulation(p).run()
    transmission=abs((kernel@slab_fine.signals[:,0])/(kernel@air_fine.signals[:,0]))**2
    assert np.max(abs(transmission-expected))<min(.005,coarse_error*.5)


def test_mesh_facade_and_axis_sampling():
    from torchfdtd import FDTD
    fd=FDTD();fd.set('mesh type','graded');fd.set('maximum mesh step',.1e-6);fd.set('mesh ppw',30)
    assert fd.project.region.mesh_type=='graded' and fd.project.region.material_sampling=='yee'
    assert fd.project.region.mesh_max==pytest.approx(.1) and fd.project.region.mesh_ppw==30
    r=Project.model_validate(fd.project.model_dump()).region
    axes=field_axes(r,'Hz')
    assert axes[2].tolist()==[0.]
    np.testing.assert_array_equal(axes[0],(r.mesh_nodes[0][:-1]+r.mesh_nodes[0][1:])/2)
