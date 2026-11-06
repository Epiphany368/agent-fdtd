"""Native modal project workflow: metadata admission and actual CPU S/VJP."""
import json
import numpy as np
import pytest
import torch
from torchfdtd import Project,Region,Source,Material,Structure
from torchfdtd.mode_network_project import ModeNetworkConfig,mode_network_plan,run_mode_network,_section


def config():
    faces={a+'_'+s:dict(kind='periodic' if a!='z' else 'pml',layers=3)
           for a in 'xyz' for s in ('min','max')}
    r=Region(dimension='3d',size=(1.,1.,6.),mesh=.2,steps=160,boundaries=faces,
        material_sampling='yee',precision='float32',backend='cpu')
    p=Project(region=r,materials=[Material(name='design',index=1.2)],
        structures=[Structure(kind='rectangle',center=(0,0,0),size=(.6,.6,.4),material='design')],
        sources=[Source(kind='plane',normal='z',center=(0,0,-1.8),size=(1,1,0),pulse_cycles=2,wavelength=1.55)],monitors=[])
    return dict(project=p.model_dump(mode='json'),normal='z',
        ports=[dict(name='left',coordinate_um=-.8,source_coordinate_um=-1.8,direction=1,mode_indices=[0]),
               dict(name='right',coordinate_um=.8,source_coordinate_um=1.8,direction=-1,mode_indices=[0])],
        execution=dict(device='cpu',checkpoints=2),
        objective=dict(output_channel=['right',0],input_channel=['left',0],quantity='power'),
        differentiate_materials=['design'])


def forbidden(*a,**k): raise AssertionError('unexpected numerical allocation')


def test_strict_schema_and_metadata_only_plan(monkeypatch):
    c=config()
    monkeypatch.setattr('torchfdtd.solver.voxelize',forbidden)
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',forbidden)
    monkeypatch.setattr('torchfdtd.differentiable._System.__init__',forbidden)
    plan=mode_network_plan(c)
    assert plan['gpu_reservation_bytes']==0 and plan['host_reservation_bytes']>0
    assert plan['observation_history_retained'] is False
    assert len(plan['request_digest'])==64
    c['execution']['checkpoints']=True
    with pytest.raises(ValueError): ModeNetworkConfig.model_validate(c)
    c=config();c['unknown']=1
    with pytest.raises(ValueError): ModeNetworkConfig.model_validate(c)
    c=config();c['ports'][0]['coordinate_um']=float('nan')
    with pytest.raises(ValueError): ModeNetworkConfig.model_validate(c)


@pytest.mark.parametrize('budget',['host_budget_bytes','resident_budget_bytes','network_budget_bytes','output_budget_bytes'])
def test_tiny_budget_rejects_before_material_and_modes(monkeypatch,budget):
    c=config();c['execution'][budget]=1
    monkeypatch.setattr('torchfdtd.solver.voxelize',forbidden)
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',forbidden)
    with pytest.raises(ValueError,match='budget'): run_mode_network(c)


def test_native_section_precedence_and_independent_stagger_sampling():
    from torchfdtd.solver import voxelize,field_axes
    c=ModeNetworkConfig.model_validate(config());p=c.project
    # Extend design through the source and overlay a lower-priority object.
    p.structures[0].size=(.6,.6,8.);p.structures[0].rotation=23.
    p.materials.append(Material(name='lower',index=1.8))
    p.structures.append(Structure(kind='circle',radius=.4,size=(1,1,8),material='lower',mesh_order=3))
    epsilon,_=voxelize(p);sample=_section(p,'z',-1.8)
    for component in range(3):
        axes=field_axes(p.region,'E'+'xyz'[component]);x,y=np.meshgrid(axes[0],axes[1],indexing='ij')
        k=int(np.argmin(abs(axes[2]+1.8)))
        np.testing.assert_array_equal(sample(x,y),epsilon[:,:,k,component])


def test_cpu_native_workflow_matches_direct_network_material_vjp():
    from torchfdtd.mode_network import ModeNetwork,FixedModePort
    from torchfdtd.differentiable import AdjointOptions
    from torchfdtd.solver import voxelize
    c=ModeNetworkConfig.model_validate(config());phases=[]
    actual=run_mode_network(c,phases.append)
    p=c.project
    network=ModeNetwork(p,tuple(FixedModePort(**v.model_dump()) for v in c.ports),
        port_permittivities={'left':1.,'right':1.},options=AdjointOptions(checkpoints=2),num_modes=1)
    eps,_,owner=voxelize(p,with_ownership=True)
    parameter=torch.tensor(1.2**2,dtype=torch.float32,requires_grad=True)
    material=torch.where(torch.from_numpy(owner==0),parameter,torch.ones_like(torch.from_numpy(eps)))
    result=network(material);objective=result.s[1,0].abs().square()
    gradient,=torch.autograd.grad(objective,parameter)
    s=np.array(actual['s_real'])+1j*np.array(actual['s_imag'])
    np.testing.assert_allclose(s,result.s.detach().numpy(),rtol=2e-5,atol=2e-6)
    assert actual['objective']==pytest.approx(float(objective.detach()),rel=2e-5,abs=2e-6)
    assert actual['material_gradients']['design']==pytest.approx(float(gradient),rel=2e-4,abs=1e-6)
    assert abs(float(gradient))>1e-5
    assert actual['channels']==[['left',0],['right',0]]
    assert phases[-1]['phase']=='completed'
    assert json.loads(json.dumps(actual,allow_nan=False))['status']=='completed'


def test_wrong_native_exterior_fails_before_fdtd(monkeypatch):
    c=config()
    # A short end object causes variation in the whole guarded exterior,
    # even though the exact source section itself is uniform background.
    c['project']['structures'][0]['center']=[0,0,-2.4]
    monkeypatch.setattr('torchfdtd.mode_network.ModeInjectedPlaneSimulation',forbidden)
    with pytest.raises(ValueError,match='exterior material'): run_mode_network(c)


def test_material_assembly_enables_requested_graph_inside_no_grad():
    from torchfdtd.mode_network_project import _material,mode_network_request_digest
    from torchfdtd.solver import voxelize
    c=ModeNetworkConfig.model_validate(config());epsilon,_,owner=voxelize(c.project,with_ownership=True)
    with torch.no_grad(): value,table=_material(c,epsilon,owner)
    gradient,=torch.autograd.grad(value.sum(),table)
    assert gradient[0]==np.count_nonzero(owner==0)
    snapshot=c.model_dump(mode='json')
    import hashlib
    expected=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert mode_network_request_digest(c)==mode_network_request_digest(snapshot)==expected


def test_open_mode_budget_precedes_section_sampling(monkeypatch):
    c=config();c['project']['region']['size']=[6.4,1.,6.]
    for side in ('min','max'): c['project']['region']['boundaries']['x_'+side]=dict(kind='pml',layers=4)
    c['open_ports']=dict(cladding_epsilon=1.,mode_budget_bytes=1)
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',forbidden)
    monkeypatch.setattr('torchfdtd.solver.voxelize',forbidden)
    with pytest.raises(ValueError,match='mode_budget_bytes'): run_mode_network(c)


def test_live_admission_rechecked_after_modes_before_rasterizer(monkeypatch):
    calls=[]
    def admission(c,report):
        calls.append(report)
        if len(calls)==2: raise ValueError('available host memory changed')
    monkeypatch.setattr('torchfdtd.mode_network_project._admit',admission)
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',lambda *a,**k:object())
    monkeypatch.setattr('torchfdtd.solver.voxelize',forbidden)
    with pytest.raises(ValueError,match='memory changed'): run_mode_network(config())
    assert len(calls)==2


def test_complete_result_budget_rejects_s_only_allowance_before_solvers(monkeypatch):
    c=config();c['execution']['output_budget_bytes']=32
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',forbidden)
    monkeypatch.setattr('torchfdtd.solver.voxelize',forbidden)
    with pytest.raises(ValueError,match='result JSON reservation'):
        run_mode_network(c)


@pytest.mark.parametrize('oversized',[False,True])
def test_direct_result_json_budget_without_numerical_solve(monkeypatch,oversized):
    from types import SimpleNamespace
    c=config();c['differentiate_materials']=[];c['objective']=None
    baseline=mode_network_plan(c)
    assert baseline['s_matrix_bytes']==32
    assert baseline['result_json_reservation_bytes']>len(json.dumps(c).encode())
    c['execution']['output_budget_bytes']=baseline['result_json_reservation_bytes']+1024
    expected_channels=(('left',0),('right',0))
    class Network:
        channels=expected_channels
        _launches=()
        def __init__(self,*a,**k): pass
        def __call__(self,material,**kwargs):
            return SimpleNamespace(s=torch.eye(2,dtype=torch.complex64),
                channels=expected_channels,port_coordinates_um=(-.8,.8),
                report={'diagnostic':'x'*(c['execution']['output_budget_bytes'] if oversized else 0)})
    def raster(project,**kwargs):
        shape=project.region.shape+(3,)
        return np.ones(shape,dtype=np.float32),{},np.full(shape,-1,dtype=np.int32)
    monkeypatch.setattr('torchfdtd.mode_network_project.ModeNetwork',Network)
    monkeypatch.setattr('torchfdtd.solver.voxelize',raster)
    if oversized:
        with pytest.raises(ValueError,match='result JSON exceeds output_budget_bytes'):
            run_mode_network(c)
    else:
        value=run_mode_network(c)
        encoded=json.dumps(value,allow_nan=False).encode('utf8')
        assert len(encoded)<=value['admission']['result_json_reservation_bytes']<=c['execution']['output_budget_bytes']
