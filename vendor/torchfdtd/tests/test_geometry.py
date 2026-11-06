"""Analytic solid, coordinate, preparation and optical representation checks."""
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from torchfdtd import Structure,Material,FDTD,Project,Simulation,run_tensor_batch
from torchfdtd.geometry import contains,object_bounds,rotation_matrix
from torchfdtd.solver import voxelize,field_axes
from test_solver import small


def solids():
    return [Structure(kind='rectangle',size=(.82,.46,.7)),
            Structure(kind='polygon',vertices=((- .5,-.4),(.5,-.4),(.5,0),(0,0),(0,.4),(-.5,.4)),size=(1,1,.6)),
            Structure(kind='sphere',radius=.5,make_ellipsoid=True,radius_2=.28,radius_3=.37),
            Structure(kind='circle',radius=.5,make_ellipsoid=True,radius_2=.3,size=(1,1,.6)),
            Structure(kind='ring',radius=.55,inner_radius=.3,theta_start=315,theta_stop=90,size=(1,1,.6)),
            Structure(kind='ring',radius=.55,radius_2=.4,inner_radius=.3,inner_radius_2=.2,make_ellipsoid=True,size=(1,1,.6))]


@pytest.mark.parametrize('vertices',[
    [(0,0),(1,1),(0,1),(1,0)],[(0,0),(1,0),(1,1),(0,0)],
    [(0,0),(1,0),(.5,0),(.5,1),(0,1)],[(0,0),(1,0),(2,0)],
    [(0,0),(1,0),(1,1),(.5,0),(0,1)],[(0,0),(1,float('nan')),(0,1)]])
def test_invalid_contours_rejected(vertices):
    with pytest.raises(ValueError):Structure(kind='polygon',vertices=vertices)


def test_concave_polygon_winding_and_closed_boundary():
    obj=solids()[1]
    x,y=np.meshgrid(np.linspace(-.7,.7,141),np.linspace(-.6,.6,121),indexing='ij')
    expected=(x>=-.5-1e-14)&(x<=.5+1e-14)&(y>=-.4-1e-14)&(y<=.4+1e-14)&((x<=1e-14)|(y<=1e-14))
    for vertices in (obj.vertices,obj.vertices[::-1]):
        obj.vertices=vertices
        np.testing.assert_array_equal(contains(obj,x,y,np.zeros_like(x)),expected)
    assert not contains(obj,np.array([0]),np.array([0]),np.array([.31]))[0]


def test_ordered_rotations_match_independent_scipy_and_are_noncommuting():
    obj=Structure(rotation=13,rotation_axes=('x','y','z'),rotation_angles=(31,-22,47))
    expected=Rotation.from_euler('xyz',[31,-22,47],degrees=True).as_matrix()@Rotation.from_euler('z',13,degrees=True).as_matrix()
    np.testing.assert_allclose(rotation_matrix(obj),expected,atol=4e-16)
    obj=Structure(rotation_axes=('z','x','none'),rotation_angles=(90,90,17))
    np.testing.assert_array_equal(rotation_matrix(obj)@np.array([1,0,0]),[0,0,1])
    obj.rotation_axes=('x','z','none')
    np.testing.assert_array_equal(rotation_matrix(obj)@np.array([1,0,0]),[0,1,0])


@pytest.mark.parametrize('shape',solids(),ids=lambda s:s.kind)
def test_rigid_membership_and_conservative_support(shape):
    rng=np.random.default_rng(43);q=rng.uniform(-.8,.8,(20000,3))
    original=contains(shape,*q.T)
    rotation=Rotation.from_euler('xyz',[17,31,-29],degrees=True)
    shape.rotation_axes=('x','y','z');shape.rotation_angles=(17,31,-29);shape.center=(.17,-.29,.31)
    world=rotation.apply(q)+shape.center
    np.testing.assert_array_equal(contains(shape,*world.T),original)
    center,size=map(np.array,object_bounds(shape))
    assert np.all(abs(world[original]-center)<=size/2+1e-14)


def test_ellipsoid_and_polar_ring_not_parametric_angle():
    obj=Structure(kind='sphere',radius=2,radius_2=1,radius_3=.5,make_ellipsoid=True)
    np.testing.assert_array_equal(contains(obj,np.array([1.9,0,0,0]),np.array([0,.9,0,0]),np.array([0,0,.49,.51])),[True,True,True,False])
    obj=Structure(kind='ring',radius=2,radius_2=1,inner_radius=0,inner_radius_2=0,make_ellipsoid=True,theta_start=0,theta_stop=45)
    # At (1,.75), physical polar angle is 36.9 degrees, parametric angle 56.3.
    assert contains(obj,np.array([1.]),np.array([.75]),np.array([0.]))[0]
    assert not contains(obj,np.array([.5]),np.array([.75]),np.array([0.]))[0]


@pytest.mark.parametrize('shape,volume',[
    (solids()[0],.82*.46*.7),(solids()[1],.6*.6),
    (solids()[2],4/3*math.pi*.5*.28*.37),(solids()[3],math.pi*.5*.3*.6),
    (solids()[4],135/360*math.pi*(.55**2-.3**2)*.6),
    (solids()[5],math.pi*(.55*.4-.3*.2)*.6)])
def test_rotated_midpoint_volumes_refine_to_analytic_volume(shape,volume):
    shape.rotation_axes=('x','y','z');shape.rotation_angles=(17,31,-29)
    errors=[]
    for step in (.08,.04,.02):
        axis=(np.arange(round(2/step))+.5)*step-1
        count=np.count_nonzero(contains(shape,*np.meshgrid(axis,axis,axis,indexing='ij',sparse=True)))
        errors.append(abs(count*step**3-volume)/volume)
    assert errors[-1]<.012,errors
    assert errors[-1]<max(errors[:-1]),errors


def full_domain_voxelize(p,with_ownership=True):
    """Unpruned reference for evaluating preparation speed and exact parity."""
    outputs=[]
    components=('Ex','Ey','Ez') if p.region.material_sampling=='yee' else (None,)
    for component in components:
        axes=field_axes(p.region,component) if component else [(np.arange(n)+.5)*p.region.mesh-s/2 for n,s in zip(p.region.shape,p.region.actual_size)]
        if p.region.dimension=='2d':axes[2]=np.array([0.])
        xyz=np.meshgrid(*axes,indexing='ij',sparse=True)
        eps=np.full(p.region.shape,p.region.background_index**2,dtype=p.region.precision)
        owners=np.full(p.region.shape,-1,np.int32);counts={}
        materials={m.name:(i,m.instantaneous_epsilon) for i,m in enumerate(p.materials)}
        for obj in sorted(p.structures,key=lambda s:-s.mesh_order):
            if not obj.enabled:continue
            mask=contains(obj,*xyz);idx,value=materials[obj.material]
            eps[mask]=value;owners[mask]=idx;counts[obj.id]=int(np.count_nonzero(mask))
        outputs.append((eps,counts,owners))
    if len(outputs)==1:return outputs[0] if with_ownership else outputs[0][:2]
    return (np.stack([r[0] for r in outputs],axis=-1),{k:max(r[1][k] for r in outputs) for k in outputs[0][1]},np.stack([r[2] for r in outputs],axis=-1))


@pytest.mark.parametrize('dimension,sampling',[('2d','cell'),('2d','yee'),('3d','cell'),('3d','yee')])
def test_support_pruning_is_bitwise_full_domain_including_overlap_and_boundaries(dimension,sampling):
    p=small(dimension=dimension);p.region.material_sampling=sampling;p.structures=solids()
    p.materials.append(Material(name='extra',index=1.4))
    for i,s in enumerate(p.structures):
        s.center=(i*.11-.23,0,.13);s.rotation_axes=('x','y','z');s.rotation_angles=(i*11,i*7,-i*19)
        s.mesh_order=i%3+1;s.material='extra' if i%2 else 'SiN (constant n)'
    p.structures += [Structure(kind='circle',center=(0,0,20)),Structure(kind='polygon',center=(2.,1,0),size=(1,1,1)),Structure(enabled=False)]
    actual=voxelize(p,with_ownership=True);expected=full_domain_voxelize(p)
    for a,b in zip(actual,expected):
        if isinstance(a,dict):assert a==b
        else:np.testing.assert_array_equal(a,b)


def test_facade_coupled_edits_units_and_failure_are_atomic():
    fd=FDTD();poly=fd.addpoly(vertices=np.array([[0,0],[.8,0],[.2,.4]])*1e-6,z_span=.3e-6,first_axis='x',rotation_1=90)
    np.testing.assert_allclose(poly.vertices,((0,0),(.8,0),(.2,.4)),rtol=2e-16)
    assert poly.rotation_angles==(90,0,0)
    before=fd.project.model_dump()
    with pytest.raises(ValueError):fd.setgeometry(vertices=[[0,0],[1,1],[0,1],[1,0]])
    assert fd.project.model_dump()==before
    with pytest.raises(ValueError):fd.addring(inner_radius=.7e-6,outer_radius=.5e-6)
    assert fd.project.model_dump()==before
    obj=fd.addring(make_ellipsoid=True,inner_radius=0,inner_radius_2=0,outer_radius=1e-6,outer_radius_2=.6e-6)
    assert obj.inner_radius==obj.inner_radius_2==0
    fd.setgeometry(theta_start=350,theta_stop=10,second_axis='y',rotation_2=31)
    assert fd._selected().angular_span==20
    fd.set('radius 2',.7e-6);assert fd._selected().radius_2==.7


def test_large_legacy_angle_uses_identical_membership_and_support_arithmetic():
    p=small();p.structures=[Structure(size=(2,.13,1),rotation=1e20)]
    actual=voxelize(p,with_ownership=True);expected=full_domain_voxelize(p)
    np.testing.assert_array_equal(actual[0],expected[0]);assert actual[1]==expected[1]


def geometry_projects(backend='cpu',precision='float64'):
    p=small(backend,'3d',precision);p.region.steps=100;p.region.snapshot_interval=50;p.region.material_sampling='yee'
    box=solids()[0];box.rotation_axes=('x','y','z');box.rotation_angles=(17,23,11);box.center=(.031,.027,.013)
    p.structures=[box];q=p.model_copy(deep=True)
    q.structures[0]=Structure(**{**box.model_dump(),'kind':'polygon','vertices':[[-.41,-.23],[.41,-.23],[.41,.23],[-.41,.23]]})
    return p,q


def test_equivalent_box_polygon_gives_bitwise_complete_optical_solution():
    p,q=geometry_projects();a,b=Simulation(p).run(),Simulation(q).run()
    for name in ('electric','magnetic','epsilon','signals','times','frames'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert np.max(abs(a.signals))>1e-6


@pytest.mark.parametrize('precision',['float32','float64'])
def test_geometry_cuda_batch_independence_and_cpu_parity(precision):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p,q=geometry_projects('cpu',precision);p.structures=solids();q.structures=solids()[::-1]
    for i,s in enumerate(q.structures):s.rotation_axes=('x','y','z');s.rotation_angles=(i*9,22,-17)
    reference=[Simulation(v).run() for v in (p,q)]
    for v in (p,q):v.region.backend='cuda';v.region.cuda_kernel='fused'
    independent=[Simulation(v).run() for v in (p,q)]
    report=run_tensor_batch([p,q]);report.raise_for_errors()
    for item,cpu,single in zip(report.items,reference,independent):
        batched=item.load()
        for name in ('electric','magnetic','epsilon','signals','frames'):
            np.testing.assert_array_equal(getattr(batched,name),getattr(single,name))
            a,b=getattr(single,name),getattr(cpu,name)
            np.testing.assert_allclose(a,b,rtol=2e-4 if precision=='float32' else 1e-10,atol=(2e-6 if precision=='float32' else 2e-13)*max(np.max(abs(b)),1e-20))
    assert not np.array_equal(independent[0].electric,independent[1].electric)
