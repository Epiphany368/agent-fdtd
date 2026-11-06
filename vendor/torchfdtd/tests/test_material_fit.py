import json

import fdtd
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd import OpticalData,FitOptions,fit_material,material_fit_report,Material,Simulation,Project,run_tensor_batch
from torchfdtd.materials import permittivity,MaterialADE
from torchfdtd.boundaries import YeeGrid
from torchfdtd.server import create_app
from torchfdtd.solver import estimate
from test_solver import small


def authored_data(case='mixed',count=121,dt=0):
    wavelength=np.linspace(.7,2.2,count);frequency=299792458/(wavelength*1e-6)
    omega=2*np.pi*frequency if not dt else 2/dt*np.tan(np.pi*frequency*dt)
    poles={'constant':[], 'lorentz':[(1.7e15,4.913e30,2e14)],
           'drude':[(0,7e30,1.8e14)], 'narrow':[(1.7e15,1e30,1e13)],
           'mixed':[(0,1e30,4e14),(1.7e15,4e30,2e14),(3e15,2e30,5e14)]}[case]
    epsilon=np.full(wavelength.shape,2.3+0j)
    for w0,a,g in poles:epsilon+=a/(w0*w0-omega*omega-1j*g*omega)
    index=np.sqrt(epsilon)
    return OpticalData.from_nk(wavelength,index.real,index.imag,reference='Independently authored analytic samples'),poles


@pytest.mark.parametrize('kind',['constant','lorentz','drude','mixed','narrow'])
def test_passive_fit_matches_independent_unsampled_frequencies(kind):
    data,poles=authored_data(kind);before=data.model_dump_json()
    result=fit_material(data,options=FitOptions(max_poles=5,tolerance=2e-6))
    assert result.converged,result.report['history']
    assert data.model_dump_json()==before
    assert len(result.material.poles)==len(poles)
    wavelength=np.linspace(.711,2.193,197);omega=2*np.pi*299792458/(wavelength*1e-6)
    exact=np.full(wavelength.shape,2.3+0j)
    for w0,a,g in poles:exact+=a/(w0*w0-omega*omega-1j*g*omega)
    fitted=permittivity(result.material,omega/(2*np.pi))
    assert np.max(abs(fitted-exact)/np.maximum(abs(exact),1))<2e-5
    dense=permittivity(result.material,np.geomspace(1e9,1e19,4000))
    assert np.min(dense.imag)>=0 and np.all(np.isfinite(dense))
    assert result.material.fit_band_um==(.7,2.2)
    assert Material.model_validate_json(result.material.model_dump_json())==result.material


@pytest.mark.parametrize('unit,scale',[('um',1),('nm',1000),('m',1e-6)])
@pytest.mark.parametrize('kind',['nk','epsilon'])
def test_data_text_units_order_headers_and_hashes(unit,scale,kind,tmp_path):
    data,_=authored_data('lorentz',9);v=np.sqrt(data.epsilon) if kind=='nk' else data.epsilon
    header=f'wavelength_{unit},'+('n,k' if kind=='nk' else 'epsilon_real,epsilon_imag')
    text=header+'\n# supplied samples\n'+'\n'.join(f'{w*scale:.17g},{n:.17g},{k:.17g}' for w,n,k in
             reversed(list(zip(data.wavelength_um,v.real,v.imag))))
    path=tmp_path/'sample.csv';path.write_text(text,encoding='utf8')
    imported=OpticalData.from_csv(path,unit=unit,kind=kind)
    np.testing.assert_allclose(imported.wavelength_um,data.wavelength_um,rtol=2e-16)
    np.testing.assert_allclose(imported.epsilon,data.epsilon,rtol=1e-15)
    assert imported.reference=='sample.csv'
    copy=imported.model_copy(deep=True);copy.reference='another citation';assert copy.fingerprint==imported.fingerprint


@pytest.mark.parametrize('text,unit',[
    ('1,2,-.1\n2,2,0\n3,2,0','um'),('1,2,0\n1,3,0\n3,2,0','um'),
    ('0,2,0\n2,2,0\n3,2,0','um'),('nan,2,0\n2,2,0\n3,2,0','um'),
    ('wavelength_nm,n,k\n1000,2,0\n2000,2,0\n3000,2,0','um'),
    ('1,2,0\n2,not-a-number,0\n3,2,0','um'),('1,2,0,7\n2,2,0\n3,2,0','um')])
def test_invalid_optical_tables_fail_instead_of_silent_repair(text,unit):
    with pytest.raises(ValueError):OpticalData.from_text(text,unit=unit)


def test_range_fixed_instantaneous_epsilon_and_noisy_samples():
    data,_=authored_data('lorentz');rng=np.random.default_rng(3)
    data.epsilon_real=(np.asarray(data.epsilon_real)+rng.normal(0,1e-4,len(data.wavelength_um))).tolist()
    options=FitOptions(max_poles=3,epsilon_inf=2.3,tolerance=2e-4,wavelength_range_um=(.9,1.9))
    result=fit_material(data,options=options)
    assert result.converged and result.material.epsilon_inf==2.3
    assert result.report['sample_count']<len(data.wavelength_um)
    assert .9<=result.material.fit_band_um[0]<result.material.fit_band_um[1]<=1.9
    again=fit_material(data,options=options)
    assert result.material==again.material
    changed=result.material.model_copy(deep=True);changed.poles[0].strength_rad_s_squared*=1.2
    assert material_fit_report(changed)['analytic']['normalized_rms']>.01


def test_tolerance_failure_and_bad_fit_requests_are_explicit(tmp_path):
    w=np.linspace(1,2,35);data=OpticalData(wavelength_um=w.tolist(),epsilon_real=(3+np.sin(50*w)).tolist(),epsilon_imag=[.1]*len(w))
    result=fit_material(data,options=FitOptions(max_poles=1,tolerance=1e-7))
    assert not result.converged and result.report['analytic']['normalized_rms']>.01
    with pytest.raises(ValueError,match='tolerance'):result.require_tolerance()
    path=tmp_path/'fit.json';result.save(path);assert json.loads(path.read_text())['report']['converged'] is False
    for options in ({'wavelength_range_um':(.5,2)}, {'wavelength_range_um':(1.001,1.002)}, {'target':'ade'}, {'dt_s':1e-12}):
        with pytest.raises(ValueError):fit_material(data,options=options)


def test_fit_discrete_ade_response_and_timestep_warning():
    p=small();dt=p.region.time_step;data,_=authored_data('lorentz',dt=dt)
    result=fit_material(data,options=FitOptions(max_poles=2,target='ade',dt_s=dt,tolerance=1e-7))
    assert result.converged and result.material.fit_dt_s==dt
    assert result.report['ade']['normalized_rms']<1e-7
    assert result.report['analytic']['normalized_rms']>1e-4
    p.materials.append(result.material);p.structures[0].material=result.material.name
    assert not any('different timestep' in w for w in estimate(p)['warnings'])
    p.region.courant_factor*=.8
    assert any('different timestep' in w for w in estimate(p)['warnings'])
    p.sources[0].wavelength=3
    assert any('outside' in w and 'fit band' in w for w in estimate(p)['warnings'])


def test_fitted_material_measured_discrete_D_over_E():
    data,_=authored_data('mixed');m=fit_material(data,options={'max_poles':4,'tolerance':1e-6}).require_tolerance()
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64
    g=YeeGrid(small().region);state=MaterialADE(g,m,np.array([0]));period=16;f=1/(period*g.time_step)
    electric=[];displacement=[];previous=0
    for n in range(8192):
        drive=np.sin(2*np.pi*f*(n+1)*g.time_step);old,response=state.prepare(g.E)
        g.E[0,0,0,0]+=(drive-previous)/m.epsilon_inf;state.correct(g.E,old,response);previous=drive
        if n>=4096:electric.append(g.E[0,0,0,0]);displacement.append(drive)
    phase=np.exp(2j*np.pi*np.arange(4096)/period);measured=np.dot(displacement,phase)/np.dot(electric,phase)
    assert measured==pytest.approx(permittivity(m,f,g.time_step),rel=4e-11)


def test_http_import_fit_preview_and_python_serialization(tmp_path):
    data,_=authored_data('lorentz',35);index=np.sqrt(data.epsilon)
    text='\n'.join(f'{w},{n},{k}' for w,n,k in zip(data.wavelength_um,index.real,index.imag))
    with TestClient(create_app(tmp_path)) as client:
        parsed=client.post('/api/materials/data',json={'text':text});assert parsed.status_code==200
        fitted=client.post('/api/materials/fit',json={'data':parsed.json(),'options':{'max_poles':2},'name':'Measured glass'})
        assert fitted.status_code==200,fitted.text
        m=fitted.json()['material'];assert fitted.json()['report']['converged']
        preview=client.post('/api/materials/preview?dt_fs=.05',json=m)
        assert preview.status_code==200 and preview.json()['samples']['analytic']['normalized_rms']<1e-5
        p=small();p.materials.append(Material.model_validate(m));p.structures[0].material=m['name']
        script=client.post('/api/python',json=p.model_dump()).text;namespace={};exec(script.split('simulation =')[0],namespace)
        assert p.materials[-1].samples.fingerprint in [a.samples.fingerprint for a in namespace['project'].materials if a.samples]
        assert client.post('/api/materials/data',json={'text':'1,2,-1\n2,2,0\n3,2,0'}).status_code==422


def test_facade_fit_is_atomic_and_preserves_named_structure_assignments():
    from torchfdtd import FDTD
    data,_=authored_data('mixed');fd=FDTD(small());name=fd.project.structures[0].material
    before=fd.project.model_dump_json()
    with pytest.raises(ValueError,match='tolerance'):
        fd.fitmaterial(name,data,options={'max_poles':1,'tolerance':1e-8})
    assert fd.project.model_dump_json()==before
    result=fd.fitmaterial(name,data,options={'max_poles':3,'tolerance':1e-6})
    assert result.converged and len(result.material.poles)==3
    assert fd.project.structures[0].material==name
    assert len([m for m in fd.project.materials if m.name==name])==1
    fd.result=object()
    with pytest.raises(RuntimeError,match='switchtolayout'):fd.fitmaterial('New',data)


def test_no_drude_option_and_single_start_retain_distinct_model_topologies():
    data,_=authored_data('drude')
    result=fit_material(data,options={'max_poles':1,'starts':1,'tolerance':1e-6})
    assert result.converged and result.material.poles[0].resonance_rad_s==0
    without=fit_material(data,options={'max_poles':1,'include_drude':False,'tolerance':1e-6})
    assert not without.converged
    assert all(p.resonance_rad_s>0 for p in without.material.poles)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_fitted_material_cpu_cuda_and_independent_tensor_cohorts():
    data,_=authored_data('mixed');m=fit_material(data,options={'max_poles':4,'tolerance':1e-6}).require_tolerance()
    p=small();p.region.steps=150;p.region.material_sampling='yee';p.materials.append(m);p.structures[0].material=m.name
    cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel='fused'
    gpu=Simulation(p).run();batch=run_tensor_batch([p,p.model_copy(deep=True)]);batch.raise_for_errors()
    assert np.max(abs(cpu.electric))>0
    np.testing.assert_allclose(cpu.electric,gpu.electric,rtol=3e-5,atol=3e-6)
    np.testing.assert_allclose(cpu.signals,gpu.signals,rtol=3e-5,atol=3e-6)
    for item in batch.items:
        np.testing.assert_array_equal(item.result.electric,gpu.electric)
        np.testing.assert_array_equal(item.result.signals,gpu.signals)
