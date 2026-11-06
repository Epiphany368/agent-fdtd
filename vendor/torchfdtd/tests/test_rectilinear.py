"""Rectangular CFL, independent Fourier waves and nonuniform energy identities."""
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Project,Region,Simulation,FieldMonitor,SpectrumSettings,run_tensor_batch,Result
from torchfdtd.models import Boundaries,BoundaryFace
from torchfdtd.boundaries import YeeGrid
from torchfdtd.solver import field_axes
from torchfdtd.cuda_kernels import FusedYeeCUDA
from torchfdtd.tuning import _result_digest


def periodic(kind='periodic'):
    return Boundaries(**{a+'_'+side:BoundaryFace(kind=kind) for a in 'xyz' for side in ('min','max')})


def coordinates(widths):
    points=np.r_[0,np.cumsum(widths)];return tuple(points-points[-1]/2)


@pytest.mark.parametrize('backend,precision',[('numpy','float64'),('torch.cuda','float32'),('torch.cuda','float64')])
@pytest.mark.parametrize('explicit',[False,True])
def test_three_dimensional_discrete_fourier_wave_on_rectangular_cells(backend,precision,explicit):
    if backend!='numpy' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    shape=np.array([12,14,16]);steps=np.array([.09,.13,.18]);spans=shape*steps
    r=Region(dimension='3d',size=tuple(spans),mesh_steps=tuple(steps),material_sampling='yee',
             boundaries=periodic(),precision=precision,courant_factor=.85)
    if explicit:
        r.mesh_coordinates=tuple(tuple(x) for x in r.mesh_nodes);r.mesh_type='explicit'
    expected_dt=.85*1e-6/(299792458*np.sqrt(np.sum(1/steps**2)))
    assert r.time_step==pytest.approx(expected_dt,rel=1e-13)
    # Derive the plane wave using the discrete dispersion relation, independent
    # of the production curl and metric arrays. H starts half a step after E.
    k=2*np.pi*np.array([1,2,1])/spans;kt=2*np.sin(k*steps/2)/steps
    electric=np.cross(kt,[0,0,1]);electric/=np.linalg.norm(electric)
    magnetic=np.cross(kt,electric)/np.linalg.norm(kt)
    phase_step=2*np.arcsin(299792458*r.time_step*1e6*np.linalg.norm(kt)/2)
    old=torch.get_default_dtype()
    try:
        fdtd.set_backend(backend if backend=='numpy' else backend+'.'+precision)
        fdtd.backend.float=np.float64 if backend=='numpy' else getattr(torch,precision)
        g=YeeGrid(r);assert g.time_step==r.time_step
        initial=[]
        for family,amplitudes,offset in (('E',electric,0),('H',magnetic,.5)):
            array=np.empty((*r.shape,3),complex)
            for c,a in enumerate('xyz'):
                grid=np.meshgrid(*field_axes(r,family+a),indexing='ij',sparse=True)
                phase=sum(v*w for v,w in zip(grid,k))-offset*phase_step
                array[...,c]=amplitudes[c]*np.exp(1j*phase)
            initial.append(array)
            field=getattr(g,family)
            if g.is_torch:field.copy_(torch.as_tensor(array.real,dtype=field.dtype,device=field.device))
            else:field[:]=array.real
        fused=FusedYeeCUDA(g) if g.is_torch else None
        for q in range(81):
            if fused:fused.update_E();fused.update_H()
            else:g.update_E();g.update_H()
        for actual,base in zip((g.E,g.H),initial):
            values=actual.detach().cpu().numpy() if g.is_torch else actual
            expected=(base*np.exp(-81j*phase_step)).real
            tol=2e-5 if precision=='float32' else 8e-14
            np.testing.assert_allclose(values,expected,atol=tol,rtol=tol)
    finally:
        fdtd.set_backend('numpy');fdtd.backend.float=np.float64;torch.set_default_dtype(old)


@pytest.mark.parametrize('bloch',[False,True])
def test_arbitrary_positive_nodes_preserve_weighted_adjoint_and_discrete_energy(bloch):
    rng=np.random.default_rng(24817)
    nodes=tuple(coordinates(rng.uniform(.06,.2,n)) for n in (12,13,14))
    r=Region(dimension='3d',mesh_type='explicit',mesh_coordinates=nodes,
             size=tuple(a[-1]-a[0] for a in nodes),material_sampling='yee',precision='float64',
             boundaries=periodic('bloch' if bloch else 'periodic'),bloch_phase=(.2,-.3,.1) if bloch else (0,0,0))
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64;g=YeeGrid(r)
    for field in (g.E,g.H):
        field[:]=rng.normal(size=field.shape)
        if bloch:field[:]+=1j*rng.normal(size=field.shape)
    primal=[np.diff(v) for v in nodes];dual=[(v+np.roll(v,1))/2 for v in primal]
    def weight(electric):
        return np.stack([np.prod(np.stack(np.broadcast_arrays(*[
            (primal[a] if (a==c)==electric else dual[a]).reshape(tuple(-1 if i==a else 1 for i in range(3)))
            for a in range(3)])),axis=0) for c in range(3)],axis=-1)
    we,wh=weight(True),weight(False)
    left=np.sum(g.E.conj()*g.curl(g.H,False)*we)
    right=np.sum(g.curl(g.E,True).conj()*g.H*wh)
    np.testing.assert_allclose(left,right,atol=1e-13)
    def energy():
        return np.sum(abs(g.E)**2*we)+np.sum(abs(g.H)**2*wh)+g.courant_number*np.sum(g.E.conj()*g.curl(g.H,False)*we).real
    initial=energy();assert initial>0
    for _ in range(240):g.update_E();g.update_H()
    assert energy()==pytest.approx(initial,rel=5e-13)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_explicit_pml_dispersive_single_batch_and_npz(precision,tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_solver import small
    from test_multipole import multi_material
    p=small('cpu','3d',precision);p.region.steps=80;p.region.material_sampling='yee'
    nodes=tuple(coordinates(np.r_[np.full(6,.12),np.full(16,.1),np.full(6,.12)]*scale) for scale in (1,1.15,1.3))
    p.region.mesh_coordinates=nodes;p.region.mesh_type='explicit';p.region.size=tuple(v[-1]-v[0] for v in nodes)
    p.materials.append(multi_material());p.structures[0].material='three-pole'
    p.monitors.append(FieldMonitor(center=(.7,0,0),size=(0,1,1),
        spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')))
    cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel='fused';p.region.cuda_monitor_kernel='fused'
    single=Simulation(p).run();batch=run_tensor_batch([p,p.model_copy(deep=True)],cuda_graph_steps=8);batch.raise_for_errors()
    tol=3e-6 if precision=='float32' else 3e-13
    np.testing.assert_allclose(single.electric,cpu.electric,atol=tol,rtol=tol)
    for item in batch.items:
        assert _result_digest(item.load())==_result_digest(single)
    single.save(tmp_path/'explicit.npz');loaded=Result.load(tmp_path/'explicit.npz')
    assert _result_digest(loaded)==_result_digest(single)
    for a,b in zip(loaded.project.region.mesh_nodes,nodes):np.testing.assert_array_equal(a,b)
    # Identical nodes are insufficient if the actual time step differs.
    changed=p.model_copy(deep=True);changed.region.time_step_override=p.region.time_step*.9
    with pytest.raises(ValueError,match='identical'):run_tensor_batch([p,changed]).raise_for_errors()


def test_mesh_validation_override_cache_and_physical_pml_bounds():
    nodes=tuple(coordinates([.15]*4+[.1]*12+[.2]*4) for _ in range(3));spans=tuple(v[-1]-v[0] for v in nodes)
    kwargs=dict(mesh_type='explicit',dimension='3d',mesh_coordinates=nodes,size=spans,material_sampling='yee',pml_cells=4)
    r=Region(**kwargs)
    assert r.interior_bounds(0)==pytest.approx((nodes[0][4],nodes[0][-5]))
    for changes,match in [(dict(mesh_coordinates=None),'requires'),(dict(material_sampling='cell'),'Yee'),
                           (dict(size=(1,1,1)),'endpoints'),(dict(time_step_override=r.time_step*1.1),'CFL')]:
        with pytest.raises(ValueError,match=match):Region(**(kwargs|changes))
    shortened=Region(**(kwargs|dict(time_step_override=.5*r.time_step)))
    assert shortened.time_step==.5*r.time_step
    original=r.mesh_nodes[0].copy();updated=list(nodes);axis=list(updated[0]);axis[8]+=.01;updated[0]=tuple(axis)
    r.mesh_coordinates=tuple(updated)
    assert r.mesh_nodes[0][8]==original[8]+.01
    with pytest.raises(ValueError,match='strictly increasing'):
        Region(**(kwargs|dict(mesh_coordinates=(tuple(sorted(nodes[0],reverse=True)),nodes[1],nodes[2]))))


@pytest.mark.parametrize('axis',range(3))
@pytest.mark.parametrize('direction',['+','-'])
def test_rectangular_paired_sources_against_independent_scalar_line(axis,direction):
    from benchmarks.tfsf_sources import box_project,reference
    from benchmarks.oneway_sources import plane_project,long_line_reference
    for kind in ('box','plane'):
        p=box_project(axis,(axis+1)%3,direction) if kind=='box' else plane_project(axis,(axis+1)%3,direction)
        p.region.mesh_steps=(.1,.125,.16);p.region.mesh=.1
        p.region.size=(3.2,4.,5.12);p.region.pml_cells=4;p.region.steps=65
        if kind=='plane':p.region.size=tuple((256 if a==axis else 8)*p.region.mesh_steps[a] for a in range(3))
        p.sources[0].size=tuple((1.6,2.,2.56)[a] if kind=='box' else 0 if a==axis else p.region.size[a] for a in range(3))
        expected,_=reference(p) if kind=='box' else long_line_reference(p)
        actual=Simulation(p).run()
        for got,want in zip((actual.electric,actual.magnetic),expected):
            assert np.linalg.norm(got-want)/np.linalg.norm(want)<1e-7
        if torch.cuda.is_available():
            p.region.backend='cuda';p.region.cuda_kernel='fused'
            gpu=Simulation(p).run()
            np.testing.assert_allclose(gpu.electric,actual.electric,atol=1e-12,rtol=1e-12)


def test_facade_axis_steps_explicit_atomicity_and_http(tmp_path):
    from torchfdtd import FDTD
    from torchfdtd.server import create_app
    from fastapi.testclient import TestClient
    f=FDTD();f.set('dy',.1e-6)
    assert f.project.region.mesh_steps==pytest.approx((.05,.1,.05))
    f.set('mesh step',.1e-6);assert f.project.region.mesh_steps is None
    before=f.project.model_dump()
    with pytest.raises(ValueError):f.setmesh([0,1e-6],[-1e-6,1e-6],[-1e-6,1e-6])
    assert f.project.model_dump()==before
    app=create_app(tmp_path)
    try:
        with TestClient(app) as client:
            nodes=client.post('/api/mesh/coordinates',json=before).json()['nodes_um']
            f.setmesh(*[np.asarray(a)*1e-6 for a in nodes])
            assert f.project.region.mesh_type=='explicit'
            assert client.post('/api/validate',json=f.project.model_dump()).status_code==200
            scope={};exec(f.project.python_script().split('result =')[0],scope)
    finally:app.state.pool.shutdown();app.state.fsp_pool.shutdown()


def test_variable_physical_pml_depth_suppresses_returning_pulse():
    from torchfdtd import Source,Monitor
    nodes=(coordinates(np.r_[.05*(1+.1*np.linspace(1,0,30)**2),np.full(260,.05),.05*(1+.1*np.linspace(0,1,30)**2)]),
           coordinates([.2]*8),coordinates([.2]*8))
    bc=Boundaries(**{a+'_'+s:BoundaryFace(kind='periodic') for a in 'yz' for s in ('min','max')})
    p=Project(region=Region(dimension='3d',mesh_type='explicit',mesh_coordinates=nodes,
            size=tuple(v[-1]-v[0] for v in nodes),material_sampling='yee',precision='float64',
            pml_cells=30,steps=1000,backend='cpu',boundaries=bc,snapshot_interval=1000),
        sources=[Source(kind='plane',injection='oneway',normal='x',size=(0,1.6,1.6),
                        wavelength=.8,time_definition='standard',pulse_length=1.5e-15,pulse_offset=3e-15)],
        monitors=[Monitor(center=(.5,0,0))])
    result=Simulation(p).run();trace=result.signals[:,0]
    # The direct pulse passes within 15 fs. A reflection from the graded PML
    # can return after approximately 40 fs. Observe the full remaining record.
    ratio=np.max(abs(trace[result.times>25e-15]))/np.max(abs(trace))
    assert ratio<1e-3,ratio
