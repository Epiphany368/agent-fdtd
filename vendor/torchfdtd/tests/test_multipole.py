import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Material, LorentzPole, Simulation, Project
from torchfdtd.boundaries import YeeGrid
from torchfdtd.materials import MaterialADE, permittivity
from test_solver import small


def multi_material():
    return Material(name='three-pole',model='multipole',epsilon_inf=2,poles=[
        LorentzPole(resonance_rad_s=0,strength_rad_s_squared=1e30,damping_rad_s=6e14),
        LorentzPole(resonance_rad_s=2e15,strength_rad_s_squared=4e30,damping_rad_s=4e14),
        LorentzPole(resonance_rad_s=3e15,strength_rad_s_squared=2e30,damping_rad_s=8e14)])


def test_passive_summed_response_and_json():
    m=multi_material();f=np.linspace(1e14,4e14,100)
    w=2*np.pi*f
    exact=2+1e30/(-w*w-6e14j*w)+4e30/(4e30-w*w-4e14j*w)+2e30/(9e30-w*w-8e14j*w)
    np.testing.assert_allclose(permittivity(m,f),exact,rtol=2e-15)
    assert np.all(exact.imag>0)
    assert Material.model_validate_json(m.model_dump_json())==m
    with pytest.raises(ValueError,match='at least one'):Material(name='empty',model='multipole')
    with pytest.raises(ValueError):LorentzPole(strength_rad_s_squared=-1)


@pytest.mark.parametrize('components',[False,True])
def test_coupled_constitutive_response_all_poles(components):
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64
    g=YeeGrid(small().region);m=multi_material()
    state=MaterialADE(g,m,np.array([0]),components)
    f=1/(64*g.time_step);previous=0.;electric=[];displacement=[]
    for n in range(8192):
        drive=np.sin(2*np.pi*f*(n+1)*g.time_step)
        old,response=state.prepare(g.E)
        g.E[0,0,0,0]+=(drive-previous)/m.epsilon_inf
        state.correct(g.E,old,response)
        previous=drive
        if n>=4096:
            electric.append(g.E[0,0,0,0]);displacement.append(drive)
    phase=np.exp(2j*np.pi*np.arange(4096)/64)
    measured=np.dot(displacement,phase)/np.dot(electric,phase)
    assert measured==pytest.approx(permittivity(m,f,g.time_step),rel=3e-12)


def test_single_pole_representation_preserves_existing_solver():
    p=small();p.region.steps=250;p.region.material_sampling='yee'
    m=Material(name='old',model='lorentz',epsilon_inf=2)
    p.materials.append(m);p.structures[0].material='old'
    old=Simulation(p).run()
    w0,a,gamma=m.oscillator
    p.materials[-1]=Material(name='old',model='multipole',epsilon_inf=2,poles=[
        LorentzPole(resonance_rad_s=w0,strength_rad_s_squared=a,damping_rad_s=gamma)])
    new=Simulation(p).run()
    np.testing.assert_array_equal(new.electric,old.electric)
    np.testing.assert_array_equal(new.signals,old.signals)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('kernel',['torch','fused'])
def test_multipole_gpu_graph_eager_and_cpu(precision,kernel):
    if kernel=='fused':pytest.importorskip('cupy')
    p=small(precision=precision);p.region.steps=220;p.region.material_sampling='yee'
    p.materials.append(multi_material());p.structures[0].material='three-pole'
    cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel=kernel
    graph=Simulation(p).run();eager=Simulation(p).run(cuda_graph=False)
    tol=3e-5 if precision=='float32' else 2e-11
    for result in (graph,eager):
        np.testing.assert_allclose(result.electric,cpu.electric,rtol=tol,atol=tol*1e-2)
        np.testing.assert_allclose(result.signals,cpu.signals,rtol=tol,atol=tol*1e-2)
    assert graph.summary['cuda_graph']


def test_multipole_material_preview_and_python_export(tmp_path):
    from fastapi.testclient import TestClient
    from torchfdtd.server import create_app
    p=small();p.materials.append(multi_material())
    assert Project.model_validate_json(p.model_dump_json()).materials[-1].model=='multipole'
    assert 'strength_rad_s_squared' in p.python_script()
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/materials/preview?dt_fs=.1',json=multi_material().model_dump())
        assert response.status_code==200
        assert min(response.json()['k'])>0


def test_material_state_energy_is_visible_when_electric_field_is_zero():
    from torchfdtd.run_control import StateDiagnostics
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64
    g=YeeGrid(small().region)
    state=MaterialADE(g,multi_material(),np.array([0]))
    g.material_states=[state]
    diagnostics=StateDiagnostics(g)
    assert diagnostics.measure()[0]==0
    state.P[1]=1e-4
    energy,peak=diagnostics.measure()
    assert energy>0 and peak==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('precision',['float32','float64'])
def test_multipole_graded_bloch_cuda_parity(precision):
    p=small(precision=precision);p.region.steps=160;p.region.material_sampling='yee'
    p.region.mesh_type='graded';p.region.mesh_max=.25
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
    p.region.bloch_phase=(0,.2,0)
    p.materials.append(multi_material());p.structures[0].material='three-pole'
    cpu=Simulation(p).run();p.region.backend='cuda';gpu=Simulation(p).run()
    tol=6e-5 if precision=='float32' else 3e-11
    np.testing.assert_allclose(cpu.electric,gpu.electric,rtol=tol,atol=tol*1e-2)
    np.testing.assert_allclose(cpu.signals,gpu.signals,rtol=tol,atol=tol*1e-2)
