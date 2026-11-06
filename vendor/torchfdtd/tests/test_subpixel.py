import itertools

import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Project,Region,Material,Structure,Simulation,run_tensor_batch
from torchfdtd.boundaries import YeeGrid
from torchfdtd.subpixel import prepare_interfaces,configure_interfaces,interface_tensor
from torchfdtd.subpixel_geometry import DielectricGeometry
from test_solver import small


def periodic_project(dimension='2d',bloch=False,index=3.48):
    bounds={a+'_'+s:dict(kind='bloch' if bloch else 'periodic') for a in ('xy' if dimension=='2d' else 'xyz') for s in ('min','max')}
    r=Region(dimension=dimension,size=(1,1,1),mesh=1/6,material_sampling='yee',interface_method='subpixel',
             boundaries=bounds,bloch_phase=(.43,-.27,.18 if dimension=='3d' else 0) if bloch else (0,0,0),
             precision='float64',backend='cpu',steps=100)
    return Project(region=r,materials=[Material(name='solid',index=index)],
                   structures=[Structure(kind='sphere',material='solid',radius=.31,center=(.02,.015,0))])


def dense_operator(plan):
    size=plan.diagonal.size;matrix=np.diag(plan.diagonal.reshape(-1)).astype(plan.values.dtype)
    for row,cols,values in zip(plan.rows,plan.columns,plan.values):np.add.at(matrix[row],cols,values)
    return matrix


@pytest.mark.parametrize('kind',['rectangle','sphere','circle','ring','polygon'])
def test_line_integrals_resolve_all_primitives_and_overlaps(kind):
    p=periodic_project();p.structures=[Structure(kind=kind,material='solid',radius=.3,inner_radius=.14,
        size=(.6,.4,1),vertices=[(-.3,-.2),(.3,-.2),(.3,.2),(-.3,.2)])]
    g=DielectricGeometry(p);point=np.zeros((1,3));f=.32 if kind=='ring' else .6
    assert g.line_average(point,0,1)[0]==pytest.approx(1+f*(3.48**2-1),rel=1e-13)
    assert g.line_average(point,0,1,inverse=True)[0]==pytest.approx(1-f+f/3.48**2,rel=1e-13)
    p.materials.append(Material(name='hole',index=1))
    p.structures.append(Structure(kind='rectangle',material='hole',size=(.1,2,2)))
    changed=DielectricGeometry(p).line_average(point,0,1)[0]
    assert changed==pytest.approx(1+(f-(0 if kind=='ring' else .1))*(3.48**2-1),rel=1e-13)


@pytest.mark.parametrize('axis',[0,1,2])
def test_off_grid_planar_interface_matches_exact_edge_and_dual_face_fractions(axis):
    p=periodic_project('3d');r=p.region;center=np.zeros(3);center[axis]=.017;size=np.full(3,3.);size[axis]=.237
    p.structures=[Structure(kind='rectangle',material='solid',center=tuple(center),size=tuple(size))]
    plan=prepare_interfaces(p);h=r.mesh;lo=center[axis]-size[axis]/2;hi=center[axis]+size[axis]/2
    for component in range(3):
        x=r.mesh_nodes[axis][:-1]+(h/2 if component==axis else 0)
        fraction=np.maximum(0,np.minimum(x+h/2,hi)-np.maximum(x-h/2,lo))/h
        expected=1-fraction+fraction/3.48**2 if component==axis else 1/(1+fraction*(3.48**2-1))
        broadcast=[1,1,1];broadcast[axis]=len(x)
        np.testing.assert_allclose(plan.diagonal[...,component],np.broadcast_to(expected.reshape(broadcast),r.shape),rtol=2e-13,atol=2e-14)
    assert len(plan.rows)==0


@pytest.mark.parametrize('dimension,bloch,index',list(itertools.product(['2d','3d'],[False,True],[1.5,20])))
def test_assembled_operator_is_hermitian_positive_and_cfl_bounded(dimension,bloch,index):
    p=periodic_project(dimension,bloch,index);plan=prepare_interfaces(p);a=dense_operator(plan)
    np.testing.assert_allclose(a,a.conj().T,atol=2e-15,rtol=0)
    eigen=np.linalg.eigvalsh(a)
    assert eigen.min()>=1/index**2-1e-13 and eigen.max()<=1+1e-13
    rng=np.random.default_rng(8);field=rng.normal(size=(*p.region.shape,3))
    if bloch:field=field+1j*rng.normal(size=field.shape)
    np.testing.assert_allclose(plan.apply(field).reshape(-1),a@field.reshape(-1),rtol=2e-14,atol=3e-15)
    assert len(plan.rows)>0


@pytest.mark.parametrize('bloch',[False,True])
def test_high_contrast_leapfrog_energy_is_conserved_for_3000_steps(bloch):
    p=periodic_project(bloch=bloch,index=20);plan=prepare_interfaces(p);matrix=dense_operator(plan);inverse=np.linalg.inv(matrix)
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64;grid=YeeGrid(p.region);configure_interfaces(grid,plan)
    rng=np.random.default_rng(91);grid.E[:]=rng.normal(size=grid.E.shape);grid.H[:]=rng.normal(size=grid.H.shape)
    if bloch:grid.E[:]+=1j*rng.normal(size=grid.E.shape);grid.H[:]+=1j*rng.normal(size=grid.H.shape)
    def energy():
        e=grid.E.reshape(-1);h=grid.H.reshape(-1)
        return np.vdot(e,inverse@e).real+np.vdot(h,h).real+grid.courant_number*np.vdot(e,grid.curl(grid.H,False).reshape(-1)).real
    initial=energy()
    for i in range(3000):grid.update_E();grid.update_H()
    assert energy()==pytest.approx(initial,rel=3e-12)
    assert np.all(np.isfinite(grid.E))


def test_unsupported_physics_is_explicit_and_old_projects_keep_staircase():
    assert Region().interface_method=='staircase'
    with pytest.raises(ValueError,match='Yee'):Region(interface_method='subpixel')
    p=small();p.region.material_sampling='yee';p.region.interface_method='subpixel'
    next(m for m in p.materials if m.name==p.structures[0].material).model='drude';p.region.pml_dispersion='frozen'
    with pytest.raises(ValueError,match='frozen'):Simulation(p)


@pytest.mark.parametrize('kind',['rectangle','sphere','circle','ring','polygon'])
def test_rotated_elliptical_primitives_match_dense_line_quadrature(kind):
    p=periodic_project('3d');p.structures=[Structure(kind=kind,material='solid',
        radius=.35,radius_2=.23,radius_3=.19,make_ellipsoid=True,
        inner_radius=.13,inner_radius_2=.09,theta_start=25,theta_stop=265,
        size=(.61,.42,.51),vertices=[(-.3,-.2),(.3,-.2),(.13,0),(.3,.2),(-.3,.2)],
        rotation_axes=('x','y','z'),rotation_angles=(19,27,38),center=(.02,-.01,.03))]
    g=DielectricGeometry(p);rng=np.random.default_rng(20)
    centers=rng.uniform(-.22,.22,(4,3));length=.71;n=60000
    for axis in range(3):
        samples=np.broadcast_to(centers[:,None,:],(len(centers),n,3)).copy()
        samples[:,:,axis]+=length*((np.arange(n)+.5)/n-.5)
        eps=g.epsilon(samples)
        np.testing.assert_allclose(g.line_average(centers,axis,length),eps.mean(axis=1),rtol=0,atol=8*(3.48**2-1)/n)
        np.testing.assert_allclose(g.line_average(centers,axis,length,inverse=True),(1/eps).mean(axis=1),rtol=0,atol=8/n)


def test_sphere_face_quadrature_converges_to_independent_disk_area():
    p=periodic_project('3d');p.structures[0].center=(0,0,0);p.structures[0].radius=.47
    g=DielectricGeometry(p);point=np.zeros((1,3));normal=np.array([[1.,0,0]])
    exact=1+np.pi*.47**2*(3.48**2-1)
    errors=[abs(g.face_average(point,2,np.ones(3),order,normal)[0]-exact) for order in (8,16,32)]
    assert errors[-1]<errors[0]
    assert errors[-1]/exact<.003


def test_axis_spacing_operator_and_nonuniform_rejection():
    p=periodic_project('3d',True);p.region.mesh_steps=(.125,.2,1/6)
    p=Project.model_validate(p.model_dump());a=dense_operator(prepare_interfaces(p))
    np.testing.assert_allclose(a,a.conj().T,atol=2e-15,rtol=0)
    assert np.linalg.eigvalsh(a).min()>0
    data=p.model_dump();r=data['region'];r.update(mesh_type='explicit',mesh_steps=None,
        mesh_coordinates=[[-.5,-.4,-.3,-.1,.1,.3,.5],p.region.mesh_nodes[1].tolist(),p.region.mesh_nodes[2].tolist()])
    with pytest.raises(ValueError,match='uniform spacing'):Project.model_validate(data)


def test_empty_homogeneous_case_keeps_legacy_fields_exactly():
    p=small();p.structures=[];p.region.material_sampling='yee'
    expected=Simulation(p).run();p.region.interface_method='subpixel';actual=Simulation(p).run()
    np.testing.assert_array_equal(actual.electric,expected.electric)
    np.testing.assert_array_equal(actual.magnetic,expected.magnetic)
    assert actual.summary['subpixel']['coupled_edges']==0


@pytest.mark.parametrize('cuda',[False,True])
def test_interface_buffers_do_not_keep_finished_grids_alive(cuda):
    import weakref
    from torchfdtd.cuda_kernels import configure_cuda_kernel
    if cuda and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=periodic_project();plan=prepare_interfaces(p);old=torch.get_default_dtype()
    try:
        fdtd.set_backend('torch.cuda.float64' if cuda else 'numpy')
        fdtd.backend.float=torch.float64 if cuda else np.float64
        grid=YeeGrid(p.region);configure_interfaces(grid,plan)
        if cuda:configure_cuda_kernel(grid,'fused')
        reference=weakref.ref(grid)
        del grid
        assert reference() is None
    finally:fdtd.set_backend('numpy');fdtd.backend.float=np.float64;torch.set_default_dtype(old)


def test_python_facade_roundtrip_and_result_metadata(tmp_path):
    from torchfdtd import FDTD,Result
    f=FDTD(project=small());f.set('interface method','subpixel');f.set('subpixel quadrature',16)
    f.save(tmp_path/'scene.json');p=Project.load(tmp_path/'scene.json')
    assert p.region.interface_method=='subpixel' and p.region.material_sampling=='yee'
    assert p.region.subpixel_quadrature==16
    result=f.run();result.save(tmp_path/'result.npz');restored=Result.load(tmp_path/'result.npz')
    assert restored.summary['subpixel']['quadrature']==16
    assert 'reciprocal diagonal' in restored.summary['epsilon_definition'].lower()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('precision',['float32','float64'])
def test_cpu_cuda_graph_and_tensor_cohorts_agree(dimension,precision):
    p=small();p.region.material_sampling='yee';p.region.interface_method='subpixel';p.region.steps=90
    if dimension=='3d':p.region.dimension='3d';p.region.size=(2.4,2.4,2.4);p.sources[0].center=(-.4,0,0);p.monitors[0].center=(.4,0,0)
    p.region.precision=precision
    cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel='torch';reference=Simulation(p).run(cuda_graph=False)
    p.region.cuda_kernel='fused';fused=Simulation(p).run()
    shifted=p.model_copy(deep=True);shifted.structures[0].center=(.07,.03,0)
    empty=p.model_copy(deep=True);empty.structures=[]
    projects=[p,shifted,empty];individual=[fused,Simulation(shifted).run(),Simulation(empty).run()]
    batch=run_tensor_batch(projects);batch.raise_for_errors()
    for actual in (reference,fused):
        tol=2e-6 if precision=='float32' else 2e-12
        np.testing.assert_allclose(actual.electric,cpu.electric,rtol=100*tol,atol=tol)
        np.testing.assert_allclose(actual.magnetic,cpu.magnetic,rtol=100*tol,atol=tol)
    for item,expected in zip(batch.items,individual):
        np.testing.assert_array_equal(item.result.electric,expected.electric)
        np.testing.assert_array_equal(item.result.magnetic,expected.magnetic)
        np.testing.assert_array_equal(item.result.signals,expected.signals)
        assert 'reciprocal diagonal' in item.result.summary['epsilon_definition'].lower()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_complex_bloch_cpu_and_cuda_torch_paths_agree():
    from torchfdtd import Source
    p=periodic_project('3d',True);p.sources=[Source(component='Ey')]
    cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel='torch'
    gpu=Simulation(p).run()
    np.testing.assert_allclose(gpu.electric,cpu.electric,rtol=2e-10,atol=2e-12)
    np.testing.assert_allclose(gpu.magnetic,cpu.magnetic,rtol=2e-10,atol=2e-12)
