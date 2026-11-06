"""Private exact-endpoint PMC foundation, not public solver admission."""
import math
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd.pmc_reference import EndpointTopology,EndpointReference,ReferenceState
from torchfdtd import Region,Boundaries,BoundaryFace
from torchfdtd.boundaries import YeeGrid


@pytest.fixture(autouse=True)
def deterministic_cpu_random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1803)
        yield


def topology(faces=None,n=6,nonuniform=False):
    if faces is None:faces=(('pmc','pmc'),)*3
    nodes=[np.linspace(0,n,n+1) for _ in range(3)]
    if nonuniform:nodes=[n*(x/n)**1.2 for x in nodes]
    return EndpointTopology(nodes,faces)


def test_unique_face_edge_ownership_and_payload_bytes():
    t=topology(n=4)
    assert len(t.components['E'])==3*4*5*5
    assert len(t.components['H'])==3*5*4*4
    assert sum(len(b.upper_axes)==2 for b in t.blocks['E'])==3
    assert all(len(b.upper_axes)<=1 for b in t.blocks['H'])
    state=t.zeros()
    for family,values in (('E',state.electric),('H',state.magnetic)):
        views=t.block_views(values,family)
        assert sum(v.numel() for v in views)==values.numel()
        for b,v in zip(t.blocks[family],views):
            assert v.untyped_storage().data_ptr()==values.untyped_storage().data_ptr()
            for axis in b.upper_axes:assert np.all(t.coordinates[family][b.start:b.stop,axis]==4)
    assert t.state_bytes()==(state.electric.numel()+state.magnetic.numel())*4


CASES=[(('pec','pec'),)*3,(('pmc','pmc'),)*3,(('pec','pmc'),)*3,
       (('pmc','pec'),)*3,(('pec','pmc'),('pmc','pmc'),('pmc','pec'))]


def basis(coords,node_axes,faces,wave):
    value=np.ones(len(coords))
    for a in range(3):
        odd=(faces[a][0]=='pec') if a in node_axes else (faces[a][0]=='pmc')
        value*=np.sin(wave[a]*coords[:,a]) if odd else np.cos(wave[a]*coords[:,a])
    return value


@pytest.mark.parametrize('faces',CASES)
@pytest.mark.parametrize('seed_component',range(3))
def test_exact_cavity_and_doubled_domain(faces,seed_component):
    t=topology(faces);op=EndpointReference(t);dt=.3
    wave=np.array([(1 if lo==hi else .5)*math.pi/n for n,(lo,hi) in zip(t.shape,faces)])
    seed=np.zeros(len(t.components['H']),np.float32)
    chosen=t.components['H']==seed_component
    seed[chosen]=basis(t.coordinates['H'][chosen],(seed_component,),faces,wave)
    electric=op.curl(torch.from_numpy(seed),False)
    assert electric.abs().max()>.01
    magnetic=-dt/2*op.curl(electric,True)
    state=ReferenceState(electric,magnetic);eps=torch.ones_like(electric)
    omega=2*math.asin(dt*math.sqrt(sum(np.sin(wave/2)**2)))
    # Mixed parity has an antiperiodic doubled extension, not a shifted wall.
    phase=tuple(0 if lo==hi else math.pi for lo,hi in faces)
    region=Region(dimension='3d',size=(12,12,12),mesh=1.,precision='float32',
        courant_factor=dt*math.sqrt(3),boundaries=Boundaries(**{a+'_'+side:BoundaryFace(kind='bloch')
        for a in 'xyz' for side in ('min','max')}),bloch_phase=phase)
    fdtd.set_backend('numpy');full=YeeGrid(region)
    q=np.indices(region.shape).reshape(3,-1).T.astype(float)
    coords=q+np.array([0 if a==seed_component else .5 for a in range(3)])
    full.H[...,seed_component]=basis(coords,(seed_component,),faces,wave).reshape(region.shape)
    full.E[:]=full.curl(full.H,False)
    full.H[:]=-dt/2*full.curl(full.E,True)
    for _ in range(24):
        state=op.step(state,eps,dt);full.update_E();full.update_H()
    torch.testing.assert_close(state.electric,electric*math.cos(24*omega),atol=2e-6,rtol=3e-5)
    for family,actual,field in (('E',state.electric,full.E),('H',state.magnetic,full.H)):
        q=t.indices[family];expected=field[q[:,0],q[:,1],q[:,2],t.components[family]]
        np.testing.assert_allclose(actual.numpy(),expected,atol=3e-6,rtol=5e-5)


@pytest.mark.parametrize('complex_fields',[False,True])
@torch.enable_grad()
def test_explicit_step_transpose_and_shared_material_gradient(complex_fields):
    t=topology(CASES[-1],n=4,nonuniform=True)
    dtype=torch.complex64 if complex_fields else torch.float32
    op=EndpointReference(t,dtype=dtype)
    state=t.zeros(dtype=dtype)
    e=torch.randn_like(state.electric,requires_grad=True)
    h=torch.randn_like(state.magnetic,requires_grad=True)
    state=ReferenceState(e,h)
    seed=ReferenceState(torch.randn_like(e),torch.randn_like(h))
    volume=(1.2+torch.rand((*t.shape,3),dtype=torch.float32)).requires_grad_()
    eps,index=t.shared_volume_epsilon(volume)
    result=op.step(state,eps,.15)
    loss=(result.electric.conj()*seed.electric).real.sum()+(result.magnetic.conj()*seed.magnetic).real.sum()
    ge,gh,gp=torch.autograd.grad(loss,(e,h,volume))
    bars,g=op.transpose_step(state,seed,eps,.15)
    torch.testing.assert_close(bars.electric,ge,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(bars.magnetic,gh,atol=2e-6,rtol=2e-6)
    shared=torch.zeros(volume.numel(),dtype=volume.dtype).index_add(0,index,g).reshape(volume.shape)
    torch.testing.assert_close(shared,gp,atol=2e-6,rtol=2e-6)
    direction=torch.randn_like(volume);direction/=direction.norm()
    def evaluate(parameter):
        epsilon,_=t.shared_volume_epsilon(parameter)
        value=op.step(state,epsilon,.15)
        return ((value.electric.conj()*seed.electric).real.sum()+(value.magnetic.conj()*seed.magnetic).real.sum()).item()
    delta=.01
    finite=(evaluate(volume+delta*direction)-evaluate(volume-delta*direction))/(2*delta)
    assert finite==pytest.approx((gp*direction).sum().item(),rel=.01,abs=.001)


@torch.enable_grad()
def test_actual_endpoint_geometry_sampler_and_gradient():
    t=topology(n=4);op=EndpointReference(t)
    position=torch.tensor(3.7,dtype=torch.float32,requires_grad=True)
    eps=t.sample_epsilon(lambda xyz,comp:1+torch.sigmoid((xyz[:,0]-position)/.4))
    upper=[b for b in t.blocks['E'] if 0 in b.upper_axes]
    assert upper
    for b in upper:
        torch.testing.assert_close(eps[b.start:b.stop],(1+torch.sigmoid((4-position)/.4)).expand(b.stop-b.start))
    state=t.zeros();state=ReferenceState(torch.randn_like(state.electric),torch.randn_like(state.magnetic))
    seed=ReferenceState(torch.randn_like(state.electric),torch.randn_like(state.magnetic))
    result=op.step(state,eps,.2)
    loss=(result.electric*seed.electric).sum()+(result.magnetic*seed.magnetic).sum()
    expected,=torch.autograd.grad(loss,position,retain_graph=True)
    _,g=op.transpose_step(state,seed,eps,.2)
    actual,=torch.autograd.grad(eps,position,g)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)


def test_magnetic_wall_never_falls_through_to_ordinary_yee():
    region=Region(dimension='3d',size=(1,1,1),mesh=.1,material_sampling='yee',
        boundaries={a+'_'+side:BoundaryFace(kind='pmc') for a in 'xyz' for side in ('min','max')})
    grid=YeeGrid(region)
    assert grid.pmc_blocks['E'] and len(grid.faces['E'])==len(grid.pmc_blocks['E'])
    for update in (grid.update_E,grid.update_H):
        with pytest.raises(ValueError,match='Torch/NumPy grid curl'):
            update()


def test_nonuniform_endpoint_material_sharing_is_explicit():
    t=topology(n=4,nonuniform=True)
    cells=torch.arange(64,dtype=torch.float32).reshape(t.shape)+1
    shared,index=t.shared_volume_epsilon(cells)
    for b in t.blocks['E']:
        if b.upper_axes:
            q=t.indices['E'][b.start:b.stop].copy()
            for a in b.upper_axes:q[:,a]=3
            expected=cells[q[:,0],q[:,1],q[:,2]]
            torch.testing.assert_close(shared[b.start:b.stop],expected)
    # Geometry sampling at the actual upper endpoint need not equal cell sharing.
    sampled=t.sample_epsilon(lambda xyz,comp:xyz[:,0]+1)
    for b in t.blocks['E']:
        if 0 in b.upper_axes:assert bool((sampled[b.start:b.stop]==5).all())
