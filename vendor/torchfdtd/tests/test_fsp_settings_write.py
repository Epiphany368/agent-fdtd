"""Independent authored source/monitor settings, no commercial field fixtures."""
import numpy as np
import pytest

from torchfdtd import Project,SourceTimeSettings,TimeSignal,Simulation,run_tensor_batch
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_geometry import write_fsp_scene
from torchfdtd.fsp_native import convert_fsp
from torchfdtd.spectra import frequency_samples
from torchfdtd.waveforms import source_time_signal
from test_fsp_geometry_write import assert_untouched,shape_fixture
from test_fsp_native import fixture
from test_fsp_monitors import spectral_fixture
from test_fsp_paired_sources import paired_fixture


def settings_fixture(*,inherited=False,axis='x'):
    global_settings=dict(fStart=170e12,fEnd=250e12,nfreqDesired=9,sampleSpacing=0,useWavelengthSpacing=0,useSourceLimits=0,
        sourcePreference=0,frequencyEnvelopeType=0,globalFrequency=200e12,globalEliminateDC=0,
        pulseLength=5e-15,offset=11e-15,optimizeForShortPulse=1,eliminateDiscontinuities=0)
    return spectral_fixture(axis,dict(useGlobalDFT=int(inherited)),global_settings)


def imported(raw):
    document=FspDocument(raw);conversion=convert_fsp(document,backend='cpu')
    assert conversion.project is not None,conversion.issues
    return document,conversion.project


def roundtrip(doc,p):
    output,report=write_fsp_scene(doc,p)
    conversion=convert_fsp(output,backend='cpu')
    assert conversion.project is not None,conversion.issues
    assert doc.fingerprint()==report['source_sha256']
    assert_untouched(doc.data,output.data,report)
    return conversion.project,report


@pytest.mark.parametrize('raw',[fixture(),settings_fixture(),settings_fixture(inherited=True),shape_fixture(),paired_fixture(),paired_fixture('plane')])
def test_scene_noop_preserves_every_byte(raw,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Vendor runtime loaded'))
    doc,p=imported(raw);out,report=write_fsp_scene(doc,p)
    assert out.data==doc.data and report['byte_identical'] and report['edits']==[]


@pytest.mark.parametrize('phase,theta,phi',[(47,None,0),(280,131,47),(19,180,37),(-77,90,270)])
def test_dipole_position_polarization_phase_amplitude_and_time(phase,theta,phi):
    doc,p=imported(fixture());s=p.sources[0]
    s.name='new μ source';s.center=(-.6,.1,.2);s.amplitude=.31;s.phase=phase;s.theta=theta;s.phi=phi
    s.pulse_length=7e-15;s.pulse_offset=15e-15;s.wavelength=1.42;s.eliminate_discontinuities=True
    q,report=roundtrip(doc,p)
    assert q.sources[0].name==s.name and len(report['edits'])>4
    t=np.linspace(0,80e-15,801)
    a,b=p.sources[0],q.sources[0]
    wa,wb=dict(a.polarization_components),dict(b.polarization_components)
    for c in ('Ex','Ey','Ez'):
        np.testing.assert_allclose(wa.get(c,0)*source_time_signal(a,t),wb.get(c,0)*source_time_signal(b,t),atol=3e-14)


@pytest.mark.parametrize('definition',['standard','wavelength','frequency'])
@pytest.mark.parametrize('inherited',[False,True])
def test_broadband_settings_and_global_source_links(definition,inherited):
    doc,p=imported(settings_fixture())
    settings=SourceTimeSettings(time_definition=definition,pulse='broadband',wavelength=1.5,wavelength_start=.8,wavelength_stop=2.1,
        pulse_length=8e-15,pulse_offset=23e-15,chirp_bandwidth_hz=90e12,optimize_for_short_pulse=False,eliminate_discontinuities=True)
    if inherited:p.global_source=settings;p.sources[0].use_global_source=True
    else:
        p.sources[0]=type(p.sources[0]).model_validate({**p.sources[0].model_dump(),**settings.model_dump()})
    q,_=roundtrip(doc,p);assert q.sources[0].use_global_source==inherited
    np.testing.assert_allclose(source_time_signal(p.resolved_source(p.sources[0]),np.linspace(0,80e-15,1201)),
        source_time_signal(q.resolved_source(q.sources[0]),np.linspace(0,80e-15,1201)),atol=1e-12)
    # Turn inheritance off again, retaining the original local pulse settings.
    if inherited:
        out,_=write_fsp_scene(doc,p);q.sources[0].use_global_source=False
        back,_=roundtrip(out,q)
        np.testing.assert_allclose(source_time_signal(back.sources[0],np.linspace(0,30e-15,301)),
                                  source_time_signal(p.sources[0],np.linspace(0,30e-15,301)),atol=1e-13)


def test_sampled_table_resize_and_second_edit():
    doc,p=imported(fixture());s=p.sources[0]
    s.pulse='sampled';s.signal=TimeSignal(time_s=[0,1e-15,5e-15,8e-15],amplitude=[0,.8,.4,0],phase_rad=[0,-1,-7,-9])
    q,_=roundtrip(doc,p);assert q.sources[0].signal==s.signal
    out,_=write_fsp_scene(doc,p);q.sources[0].signal=TimeSignal(time_s=[0,2e-15,7e-15],amplitude=[0,1,0],phase_rad=[0,-4,-12])
    r,_=roundtrip(out,q);assert r.sources[0].signal==q.sources[0].signal


@pytest.mark.parametrize('sampling,apo',[('frequency','none'),('wavelength','full'),('chebyshev','start'),('custom','end')])
@pytest.mark.parametrize('inherited',[False,True])
def test_spectral_controls_selections_stride_and_global_inheritance(sampling,apo,inherited):
    doc,p=imported(settings_fixture(inherited=inherited));m=p.monitors[0]
    target=p.global_monitor if inherited else m.spectrum
    target.sampling=sampling;target.chebyshev_nodes='lobatto';target.frequency_points=13
    target.wavelength_start=1.25;target.wavelength_stop=1.85;target.custom_frequencies_hz=[170e12,185e12,231e12,249e12]
    m.spectrum.apodization=apo;m.spectrum.apodization_center=19e-15;m.spectrum.apodization_time_width=8e-15
    m.center=(.3,.2,.1);m.name='exported DFT μ';m.time_downsample=2;m.downsample_xyz=(1,2,1)
    m.record_fields=('Ey','Ez');m.record_poynting=('x',);m.record_flux=False;m.dft_precision='field';m.spatial_interpolation='specified'
    q,_=roundtrip(doc,p);n=q.monitors[0]
    np.testing.assert_allclose(frequency_samples(p.resolved_monitor(m).spectrum),frequency_samples(q.resolved_monitor(n).spectrum),rtol=1e-14)
    assert n.spectrum.apodization==apo and n.record_fields==('Ey','Ez') and n.time_downsample==2
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.frequency_fields[0]['fields'],b.frequency_fields[0]['fields'],atol=1e-28,rtol=2e-11)


def test_chebyshev_roots_encoded_as_exact_custom_table_and_source_limits():
    doc,p=imported(settings_fixture());m=p.monitors[0]
    m.spectrum.sampling='chebyshev';m.spectrum.chebyshev_nodes='roots';m.spectrum.frequency_points=17
    q,report=roundtrip(doc,p)
    assert q.monitors[0].spectrum.sampling=='custom'
    np.testing.assert_array_equal(frequency_samples(q.monitors[0].spectrum),frequency_samples(m.spectrum))
    assert any('roots' in s for s in report['native_only_settings'])
    p.sources[0].time_definition='wavelength';p.sources[0].pulse='broadband'
    p.sources[0].wavelength_start=1.1;p.sources[0].wavelength_stop=1.9
    m.spectrum.sampling='frequency';m.spectrum.use_source_limits=True
    q,_=roundtrip(doc,p)
    np.testing.assert_allclose(frequency_samples(p.resolved_monitor(m).spectrum),frequency_samples(q.resolved_monitor(q.monitors[0]).spectrum))


@pytest.mark.parametrize('axis',['x','y','z'])
def test_monitor_normal_and_dimensions(axis):
    doc,p=imported(settings_fixture());m=p.monitors[0];m.normal=axis
    spans=[.7,.8,.9];spans['xyz'.index(axis)]=0;m.size=tuple(spans)
    q,_=roundtrip(doc,p);assert q.monitors[0].normal==axis
    np.testing.assert_allclose(q.monitors[0].size,m.size,rtol=0,atol=1e-15)


@pytest.mark.parametrize('kind',['plane','tfsf'])
@pytest.mark.parametrize('axis',range(3))
def test_paired_source_polarization_direction_geometry(kind,axis):
    doc,p=imported(paired_fixture(kind,axis));s=p.sources[0]
    s.direction='-';s.phase=71;s.amplitude=.2;s.wavelength=1.7
    if kind=='tfsf':s.center=(.03,-.02,.01);s.size=(1.5,1.6,1.7)
    else:
        center=list(s.center);center[axis]=.7;s.center=tuple(center)
    from torchfdtd.fsp_native import paired_polarization
    s.theta,s.phi=paired_polarization(axis,211.)
    q,_=roundtrip(doc,p)
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,rtol=2e-5,atol=2e-8)


@pytest.mark.parametrize('mode',['time','pml','periodic','background'])
def test_region_duration_cfl_boundary_layers_and_periodic_nodes(mode):
    doc,p=imported(fixture())
    if mode=='time':p.region.courant_factor=.7;p.region.steps=43
    if mode=='pml':p.region.boundaries.x_min.layers=6;p.region.boundaries.z_max.layers=7
    if mode=='periodic':
        p.region.boundaries.y_min.kind='periodic';p.region.boundaries.y_max.kind='periodic'
    if mode=='background':p.region.background_index=1.23
    q,_=roundtrip(doc,p)
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,atol=2e-7,rtol=2e-6)


def test_point_components_share_one_file_record():
    raw=fixture(monitor_overrides={'outputE':np.array([1,0,1,0,0,0])})
    doc,p=imported(raw)
    for m in p.monitors:m.center=(.7,.1,0.);m.name='shared '+m.component
    q,_=roundtrip(doc,p);assert len(q.monitors)==2
    p.monitors[1].center=(.6,0,0)
    q,report=roundtrip(doc,p)
    np.testing.assert_allclose([m.center for m in q.monitors],[m.center for m in p.monitors],rtol=0,atol=1e-15)
    assert len(report['monitor_list']['groups'])==2


@pytest.mark.parametrize('change,match',[
    (lambda p:setattr(p.sources[0],'component','Hx'),'magnetic'),
    (lambda p:setattr(p.sources[0],'pulse','continuous'),'continuous'),
    (lambda p:setattr(p.region.boundaries.x_min,'kappa',2.),'CPML profile'),
    (lambda p:setattr(p.region.run_control,'auto_shutoff',True),'run_control')])
def test_unmapped_edits_never_silently_drop(change,match):
    doc,p=imported(settings_fixture());before=doc.data;change(p)
    with pytest.raises(ValueError,match=match):write_fsp_scene(doc,p)
    assert doc.data==before


def test_combined_geometry_source_monitor_gpu_cohort():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    doc,p=imported(settings_fixture());p.structures[0].radius=.34
    p.sources[0].phase=41;p.sources[0].wavelength=1.4;p.region.steps=50
    p.monitors[0].spectrum.apodization='full';p.monitors[0].spectrum.apodization_center=5e-15
    q,_=roundtrip(doc,p);cpu=Simulation(p).run()
    q.region.backend='cuda';q.region.cuda_kernel='fused';q.region.cuda_monitor_kernel='fused'
    gpu=Simulation(q).run();batch=run_tensor_batch([q,q.model_copy(deep=True)]);batch.raise_for_errors()
    np.testing.assert_allclose(cpu.electric,gpu.electric,atol=2e-6,rtol=2e-6)
    scale=np.max(abs(cpu.frequency_fields[0]['fields']));assert scale>0
    np.testing.assert_allclose(cpu.frequency_fields[0]['fields']/scale,gpu.frequency_fields[0]['fields']/scale,atol=2e-5,rtol=2e-5)
    for item in batch.items:np.testing.assert_array_equal(item.result.frequency_fields[0]['fields'],gpu.frequency_fields[0]['fields'])


def test_unreferenced_global_settings_and_effective_global_window():
    doc,p=imported(settings_fixture());p.global_source.wavelength=1.32;p.global_source.pulse_offset=25e-15
    p.global_monitor.apodization='start';p.global_monitor.apodization_center=31e-15
    p.global_monitor.apodization_time_width=12e-15;p.monitors[0].use_global_monitor=True;p.monitors[0].inherit_apodization=True
    q,report=roundtrip(doc,p)
    assert q.global_source.wavelength==pytest.approx(1.32)
    assert q.resolved_monitor(q.monitors[0]).spectrum.apodization=='start'
    assert any('effective local windows' in s for s in report['native_only_settings'])


def test_cli_and_http_scene_export_are_independent_and_preserve_original(tmp_path,monkeypatch):
    import json,subprocess,sys,time
    from fastapi.testclient import TestClient
    from torchfdtd import fsp
    from torchfdtd.server import create_app
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Vendor runtime loaded'))
    raw=settings_fixture();doc,p=imported(raw);p.sources[0].phase=17;p.monitors[0].spectrum.apodization='end'
    original=tmp_path/'original.fsp';original.write_bytes(raw);scene=tmp_path/'scene.json';p.save(scene)
    output=tmp_path/'edited.fsp';report=tmp_path/'write.json'
    command=[sys.executable,'-m','torchfdtd.cli','fsp-write-scene',str(original),str(scene),'--output',str(output),'--report',str(report)]
    run=subprocess.run(command,capture_output=True,text=True);assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['scope']=='existing objects and supported settings'
    assert subprocess.run(command,capture_output=True).returncode!=0
    app=create_app(tmp_path/'server')
    try:
        with TestClient(app) as client:
            def wait(key):
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    row=client.get('/api/fsp/'+key).json()
                    if row['status'] in ('ready','failed'):return row
                    time.sleep(.01)
                pytest.fail('FSP export did not finish')
            key=client.post('/api/fsp/native-import',content=raw,headers={'x-filename':'source.fsp'}).json()['id']
            assert wait(key)['status']=='ready'
            response=client.post('/api/fsp/'+key+'/native-scene-export',json=p.model_dump())
            assert response.status_code==202
            newkey=response.json()['id'];assert wait(newkey)['status']=='ready'
            data=client.get('/api/fsp/'+newkey+'/download').content
            assert data==output.read_bytes() and client.get('/api/fsp/'+key+'/download').content==raw
            p.sources[0].component='Hx'
            bad=wait(client.post('/api/fsp/'+key+'/native-scene-export',json=p.model_dump()).json()['id'])
            assert bad['status']=='failed' and 'magnetic' in bad['error']
            assert client.get('/api/fsp/'+bad['id']+'/download').status_code==409
    finally:app.state.pool.shutdown();app.state.fsp_pool.shutdown()


def test_scene_input_controls_are_written_with_resolved_values():
    from torchfdtd.fsp_native import FDTD,DIPOLE,DFT
    doc,p=imported(settings_fixture());s=p.sources[0]
    s.pulse='broadband';s.time_definition='wavelength';s.wavelength_start=1.2;s.wavelength_stop=1.9
    p.monitors[0].time_downsample=2
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='periodic'
    out,_=write_fsp_scene(doc,p);nodes={n.uid:n for n in out.root.children}
    source=nodes[DIPOLE].properties
    assert source['local::defineSourceBy'].value==2
    assert source['local::minGUIFrequency'].value==pytest.approx(299792458/1.9e-6)
    assert source['local::maxGUIFrequency'].value==pytest.approx(299792458/1.2e-6)
    assert source['frequency2'].value==source['local::maxGUIFrequency'].value
    np.testing.assert_array_equal(nodes[FDTD].properties['GUIboundary'].value.reshape(-1),[0,0,2,2,0,0])
    m=nodes[DFT].properties;assert m['useGlobalAdvanced'].value==0
    assert int(1/(p.region.time_step*m['minSamplingPerCycle'].value*max(frequency_samples(p.monitors[0].spectrum))))==2


def test_sampled_source_sampling_override_and_stored_pml_limits():
    from torchfdtd.fsp_settings import PropertyPlan
    from torchfdtd.fsp_native import FDTD
    doc,p=imported(settings_fixture())
    p.sources[0].pulse='sampled';p.sources[0].signal=TimeSignal(time_s=[0,1e-15],amplitude=[0,1],phase_rad=[0,1])
    with pytest.raises(ValueError,match='time_downsample=1'):write_fsp_scene(doc,p)
    p.monitors[0].time_downsample=1;roundtrip(doc,p)
    # Supply independently authored profile limits as input metadata.
    domain=next(n for n in doc.root.children if n.uid==FDTD)
    from torchfdtd.fsp_binary import Value
    domain.properties['minPMLLayers']=Value(np.full((6,1),4),5,0,0,0)
    p.region.boundaries.x_min.layers=3
    with pytest.raises(ValueError,match='profile limits'):write_fsp_scene(doc,p)
    domain.properties['PMLType']=Value(1,2,0,0,0)
    p.region.boundaries.x_min.layers=5
    with pytest.raises(ValueError,match='even PML count'):write_fsp_scene(doc,p)
