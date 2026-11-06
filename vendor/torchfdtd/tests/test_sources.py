import numpy as np
import pytest
from fastapi.testclient import TestClient

from torchfdtd import FDTD, Project, Source, SourceTimeSettings, TimeSignal, Simulation
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import convert_fsp
from torchfdtd.server import create_app
from torchfdtd.source_preview import preview_source
from torchfdtd.solver import source_time_signal
from test_fsp_native import fixture
from test_solver import small


def signal():
    return TimeSignal(time_s=[0,10e-15,40e-15,60e-15], amplitude=[0,1,.5,0], phase_rad=[0,-12,-48,-72])


@pytest.mark.parametrize('patch', [dict(time_s=[0,0,1,2]), dict(time_s=[-1,0,1,2]),
    dict(amplitude=[1,2]), dict(phase_rad=[0,float('nan'),0,0]), dict(time_s=[0,1,float('inf'),3])])
def test_invalid_tables_are_rejected(patch):
    with pytest.raises(ValueError):TimeSignal.model_validate({**signal().model_dump(),**patch})


def test_interpolate_amplitude_and_unwrapped_phase_before_sine_and_zero_outside():
    s=Source(pulse='sampled',signal=TimeSignal(time_s=[10e-15,20e-15],amplitude=[2,4],phase_rad=[0,2*np.pi]),amplitude=.7,phase=90)
    times=np.array([0,10,12.5,15,17.5,20,21])*1e-15
    times[1]=s.signal.time_s[0];times[-2]=s.signal.time_s[-1]
    np.testing.assert_allclose(source_time_signal(s,times),[0,1.4,0,-2.1,0,2.8,0],atol=1e-14)
    # Interpolating the oscillating field would incorrectly give a positive midpoint.
    assert source_time_signal(s,[15e-15])[0]<0


def test_global_inheritance_preserves_local_overrides_and_saved_signal(tmp_path):
    p=small();local=p.sources[0].model_dump()
    p.global_source=SourceTimeSettings(pulse='sampled',signal=signal(),wavelength=1.8)
    p.sources[0].use_global_source=True;p.sources[0].phase=37;p.sources[0].amplitude=.7
    inherited=Simulation(p).run()
    resolved=p.resolved_source(p.sources[0]);assert resolved.phase==37 and resolved.amplitude==.7
    assert resolved.wavelength==1.8 and p.sources[0].wavelength==local['wavelength']
    explicit=p.model_copy(deep=True);explicit.sources[0]=resolved
    np.testing.assert_array_equal(inherited.electric,Simulation(explicit).run().electric)
    p.save(tmp_path/'scene.json');assert Project.load(tmp_path/'scene.json')==p
    inherited.save(tmp_path/'result.npz')
    saved=np.load(tmp_path/'result.npz');assert 'project' in saved
    assert Project.model_validate_json(str(saved['project']))==p
    p.sources[0].use_global_source=False
    assert p.resolved_source(p.sources[0]).wavelength==local['wavelength']
    p.global_source=None;p.sources[0].use_global_source=True
    with pytest.raises(ValueError,match='unavailable'):Project.model_validate(p.model_dump())


def test_familiar_global_and_custom_commands_are_transactional():
    f=FDTD(small());name=f.project.sources[0].name
    f.setglobalsource('frequency',210e12);f.setglobalsource('set time domain',1)
    f.setglobalsource('pulselength',23e-15);f.setglobalsource('offset',67e-15)
    f.setnamed(name,'override global source settings',False)
    assert f.project.resolved_source(f.project.sources[0]).pulse_offset==67e-15
    assert f.getglobalsource('frequency')==pytest.approx(210e12)
    s=signal();f.setsourcesignal(name,np.array(s.time_s)[:,None],s.amplitude,s.phase_rad)
    assert not f.project.sources[0].use_global_source and f.project.sources[0].signal==s
    before=f.project.model_dump()
    with pytest.raises(ValueError):f.setsourcesignal(name,[0,0],[1,1],[0,1])
    with pytest.raises(ValueError):f.setglobalsource('eliminate dc',1)
    assert f.project.model_dump()==before
    f.run()
    with pytest.raises(RuntimeError):f.setglobalsource('frequency',200e12)
    with pytest.raises(RuntimeError):f.setsourcesignal(name,s.time_s,s.amplitude,s.phase_rad)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_custom_global_signal_cpu_gpu_eager_graph_parity(precision):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA GPU not available')
    p=small(precision=precision)
    p.global_source=SourceTimeSettings(pulse='sampled',signal=signal())
    p.sources[0].use_global_source=True;p.sources[0].phase=37;p.sources[0].amplitude=.7
    cpu=Simulation(p).run();p.region.backend='cuda'
    eager=Simulation(p).run(cuda_graph=False);graph=Simulation(p).run(cuda_graph=True)
    assert graph.summary['cuda_graph'] and np.max(abs(graph.signals))>1e-3
    tol=2e-6 if precision=='float32' else 1e-12
    np.testing.assert_allclose(cpu.signals,graph.signals,atol=tol,rtol=tol)
    np.testing.assert_allclose(eager.electric,graph.electric,atol=tol,rtol=tol)


def test_independent_fsp_custom_signal_and_global_settings(monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Independent source import loaded vendor runtime'))
    s=signal()
    conversion=convert_fsp(FspDocument(fixture(dict(frequencyEnvelopeType=2,userTime=np.array(s.time_s),
                    userAmp=np.array(s.amplitude),userPhs=np.array(s.phase_rad)))))
    assert conversion.project is not None,conversion.issues
    source=conversion.project.sources[0]
    assert source.signal==s and source.pulse_length==2e-15
    # Saved effective fields and retained local fields must be treated separately.
    local={'local::pulseTypeActual':0,'local::defineSourceBy':0,'local::frequency':190e12,
           'local::mEliminateDC':0,'local::pulseLength':2e-15,'local::offset':4e-15,'local::optimizeForShortPulse':0,'local::eliminateDiscontinuities':0}
    global_={'frequencyEnvelopeType':0,'sourcePreference':0,'globalFrequency':210e12,'globalEliminateDC':0,
             'pulseLength':23e-15,'offset':67e-15,'optimizeForShortPulse':0,'eliminateDiscontinuities':0}
    raw=fixture(dict(local, useGlobalSource=1,frequency=210e12,pulseLength=23e-15,offset=67e-15),global_)
    p=convert_fsp(FspDocument(raw)).project
    assert p is not None and p.sources[0].use_global_source
    assert p.sources[0].pulse_length==2e-15 and p.resolved_source(p.sources[0]).pulse_length==23e-15
    mismatch=fixture(dict(local,useGlobalSource=1,frequency=220e12,pulseLength=23e-15,offset=67e-15),global_)
    assert convert_fsp(FspDocument(mismatch)).project is None


def test_preview_uses_mesh_times_precision_and_disabled_state(tmp_path):
    p=small();p.sources[0].pulse='sampled';p.sources[0].signal=signal()
    app=create_app(tmp_path)
    try:
        with TestClient(app) as client:
            response=client.post('/api/sources/'+p.sources[0].id+'/preview',json=p.model_dump())
            assert response.status_code==200
            data=response.json();t=np.arange(1,p.region.steps+1)*p.region.time_step
            np.testing.assert_allclose(data['time_fs'],t*1e15,atol=0,rtol=0)
            expected=np.interp(t,signal().time_s,signal().amplitude,right=0)*np.sin(-1.2e15*t)
            np.testing.assert_allclose(data['signal'],expected,atol=2e-14,rtol=0)
            expected_fft=abs(np.fft.rfft(expected)/len(expected))[1:];expected_fft[:-1]*=2
            np.testing.assert_allclose(data['spectrum'],expected_fft,atol=1e-15)
            assert client.post('/api/sources/missing/preview',json=p.model_dump()).status_code==404
            p.sources[0].enabled=False
            assert not np.any(preview_source(p,p.sources[0].id)['signal'])
    finally:
        app.state.pool.shutdown();app.state.fsp_pool.shutdown()
