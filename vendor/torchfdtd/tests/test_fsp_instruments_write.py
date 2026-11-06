"""Authored source and monitor records with native waveform/field validation."""
import numpy as np
import pytest

from torchfdtd import Source,Monitor,FieldMonitor,SpectrumSettings,TimeSignal,Simulation,run_tensor_batch
from torchfdtd.fsp_geometry import write_fsp_scene
from torchfdtd.fsp_native import convert_fsp,DIPOLE,TIME,DFT
from torchfdtd.waveforms import source_time_signal
from torchfdtd.spectra import frequency_samples,point_spectrum
from test_fsp_native import fixture
from test_fsp_settings_write import settings_fixture
from test_fsp_geometry_write import imported
from test_fsp_objects_write import roundtrip
from test_fsp_mesh_write import authored_2d
from test_fsp_paired_sources import paired_fixture


def source(**changes):
    return Source(name='new source',time_definition='standard',pulse_length=3e-15,pulse_offset=6e-15,
                  center=(-.4,0,0),phase=37,**changes)


def check_instruments(p,q,report):
    assert [s.id for s in q.sources]==[report['id_mapping'][s.id] for s in p.sources]
    assert [m.id for m in q.monitors]==[report['id_mapping'][m.id] for m in p.monitors]
    t=np.linspace(0,60e-15,601)
    for a,b in zip(p.sources,q.sources):
        wa,wb=dict(a.polarization_components),dict(b.polarization_components)
        for c in ('Ex','Ey','Ez'):
            np.testing.assert_allclose(wa.get(c,0)*source_time_signal(p.resolved_source(a),t),
                wb.get(c,0)*source_time_signal(q.resolved_source(b),t),rtol=1e-12,atol=1e-13)
    for a,b in zip(p.monitors,q.monitors):
        left,right=p.resolved_monitor(a).spectrum,q.resolved_monitor(b).spectrum
        if left.sampling!='fft':np.testing.assert_allclose(frequency_samples(left),frequency_samples(right),rtol=2e-14)


@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('definition',['standard','wavelength','sampled','global'])
def test_add_delete_reorder_dipoles_preserves_vector_waveforms(dimension,definition):
    doc,p=imported(authored_2d() if dimension=='2d' else settings_fixture())
    original=p.sources[0];a=source(id='new-a',theta=127,phi=219);b=source(id='new-b',component='Ex')
    if definition=='wavelength':
        a.time_definition='wavelength';a.pulse='broadband';a.wavelength_start=1.1;a.wavelength_stop=1.9
    if definition=='sampled':
        a.pulse='sampled';a.signal=TimeSignal(time_s=[0,2e-15,5e-15,9e-15],amplitude=[0,.7,1,0],phase_rad=[0,-2,-5,-11])
        for m in p.monitors:m.time_downsample=1
    if definition=='global':
        from torchfdtd import SourceTimeSettings
        p.global_source=SourceTimeSettings(time_definition='standard',pulse_length=4e-15,pulse_offset=7e-15)
        a.use_global_source=True
    p.sources=[b,a]
    _,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    assert report['source_list']['removed']==[original.id]
    assert report['source_list']['added']==['new-b','new-a']
    a_result,b_result=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a_result.electric,b_result.electric,rtol=3e-6,atol=3e-7)


@pytest.mark.parametrize('kind',['plane','tfsf'])
@pytest.mark.parametrize('axis',range(3))
def test_new_paired_sources_all_axes_and_directions(kind,axis):
    doc,p=imported(paired_fixture(kind,axis));first=p.sources.pop()
    for i,direction in enumerate(('+','-')):
        s=first.model_copy(deep=True);s.id='paired-'+str(i);s.name='paired '+str(i);s.direction=direction;s.phase+=13*i
        p.sources.append(s)
    _,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,rtol=3e-6,atol=3e-7)


@pytest.mark.parametrize('sampling',['fft','frequency','wavelength','chebyshev','custom','global'])
def test_new_point_monitors_keep_spectra_and_component_order(sampling):
    doc,p=imported(settings_fixture());p.monitors=[]
    spec=SpectrumSettings(sampling='frequency' if sampling=='global' else sampling,apodization='none',frequency_points=7,
                          custom_frequencies_hz=[180e12,190e12,213e12] if sampling=='custom' else [])
    if sampling=='chebyshev':spec.chebyshev_nodes='roots'
    for i,component in enumerate(('Hz','Ex','Ey')):
        p.monitors.append(Monitor(id='point-'+component,name='point '+component,component=component,
            center=(.4,.03*i,0),spectrum=spec.model_copy(deep=True),use_global_monitor=sampling=='global'))
    p.region.steps=65
    _,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_array_equal(a.signals,b.signals)
    for i,m in enumerate(p.monitors):
        aa=point_spectrum(a.times,a.signals[:,i],p.resolved_monitor(m).spectrum)
        bb=point_spectrum(b.times,b.signals[:,i],q.resolved_monitor(q.monitors[i]).spectrum)
        np.testing.assert_allclose(aa['value'],bb['value'],rtol=1e-12,atol=1e-28)


@pytest.mark.parametrize('dimension',['2d','3d'])
def test_new_frequency_planes_axes_outputs_and_local_windows(dimension):
    doc,p=imported(authored_2d() if dimension=='2d' else settings_fixture());p.monitors=[]
    for i,axis in enumerate('xy' if dimension=='2d' else 'xyz'):
        spans=[.7,.9,.6];spans[i]=0
        p.monitors.append(FieldMonitor(id='plane-'+axis,name='plane '+axis,normal=axis,size=tuple(spans),
            spectrum=SpectrumSettings(sampling='frequency',frequency_points=5,apodization=('start','full','end')[i],
                                      apodization_center=5e-15,apodization_time_width=4e-15),
            record_fields=('Ex','Ey','Ez'),record_poynting=(),record_flux=True,downsample_xyz=(2,1,1),
            spatial_interpolation='specified',dft_precision='float64'))
    p.region.steps=60
    _,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    a,b=Simulation(p).run(),Simulation(q).run()
    for aa,bb in zip(a.frequency_fields,b.frequency_fields):
        scale=np.max(abs(aa['fields']));assert scale>0
        np.testing.assert_allclose(aa['fields']/scale,bb['fields']/scale,rtol=2e-6,atol=2e-7)
        np.testing.assert_allclose(aa['flux'],bb['flux'],rtol=2e-6,atol=1e-36)


@pytest.mark.parametrize('edit',['delete_component','delete_all','split_position','interleave','reverse'])
def test_shared_monitor_record_can_be_deleted_or_split_without_dropping_channels(edit):
    doc,p=imported(fixture(monitor_overrides=dict(outputE=np.array([1,1,1,0,0,0]))))
    original=next(n for n in doc.root.children if n.uid==TIME)
    if edit=='delete_component':p.monitors.pop(1)
    if edit=='delete_all':p.monitors=[]
    if edit=='split_position':p.monitors[1].center=(.41,.03,0)
    if edit=='interleave':p.monitors.insert(1,Monitor(id='extra',name='extra',spectrum=SpectrumSettings(apodization='none')))
    if edit=='reverse':p.monitors.reverse()
    out,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    assert len(q.monitors)==len(p.monitors)
    if edit in ('split_position','interleave','reverse'):
        assert report['monitor_list']['splits']
        assert all(r['original_record']==original.start for r in report['monitor_list']['splits'])
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_array_equal(a.signals,b.signals)
    # Reimported IDs/fingerprint are a valid basis for another save.
    assert write_fsp_scene(out,q)[0].data==out.data


def test_delete_all_sources_and_monitors_then_add_to_empty_layout():
    doc,p=imported(fixture());p.sources=[];p.monitors=[]
    out,q,report=roundtrip(doc,p)
    assert not q.sources and not q.monitors
    assert np.max(abs(Simulation(q).run().electric))==0
    q.sources=[source(id='reborn')];q.monitors=[Monitor(spectrum=SpectrumSettings(apodization='none'))]
    _,r,report=roundtrip(out,q);check_instruments(q,r,report)
    assert len(r.sources)==len(r.monitors)==1


@pytest.mark.parametrize('case',['magnetic','continuous','cycles','hann','sheet'])
def test_unmapped_new_instruments_fail_explicitly_and_keep_original(case):
    doc,p=imported(fixture());raw=doc.data
    if case=='hann':p.monitors.append(Monitor())
    else:
        s=source()
        if case=='magnetic':s.component='Hx'
        if case=='continuous':s.pulse='continuous'
        if case=='cycles':s.time_definition='cycles'
        if case=='sheet':s.kind='plane';s.injection='soft'
        p.sources.append(s)
    with pytest.raises(ValueError):write_fsp_scene(doc,p)
    assert doc.data==raw


def test_combined_new_instruments_cuda_and_tensor_outputs():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    doc,p=imported(settings_fixture());p.sources.append(source(id='second',theta=121,phi=41))
    p.monitors.append(Monitor(name='trace',spectrum=SpectrumSettings(apodization='none')))
    p.monitors.append(FieldMonitor(name='new plane',normal='y',size=(.7,0,.9),
        spectrum=SpectrumSettings(sampling='frequency',frequency_points=5,apodization='none')))
    p.region.steps=70
    _,q,report=roundtrip(doc,p);check_instruments(p,q,report)
    cpu=Simulation(p).run();q.region.backend='cuda';q.region.cuda_kernel='fused';q.region.cuda_monitor_kernel='fused'
    gpu=Simulation(q).run();batch=run_tensor_batch([q,q.model_copy(deep=True)]);batch.raise_for_errors()
    np.testing.assert_allclose(cpu.electric,gpu.electric,rtol=4e-5,atol=3e-6)
    np.testing.assert_allclose(cpu.signals,gpu.signals,rtol=4e-5,atol=3e-6)
    for aa,bb in zip(cpu.frequency_fields,gpu.frequency_fields):
        scale=np.max(abs(aa['fields']));assert scale>0
        np.testing.assert_allclose(aa['fields']/scale,bb['fields']/scale,rtol=4e-5,atol=4e-5)
    for item in batch.items:
        np.testing.assert_array_equal(item.result.electric,gpu.electric)
        np.testing.assert_array_equal(item.result.signals,gpu.signals)
        for aa,bb in zip(item.result.frequency_fields,gpu.frequency_fields):np.testing.assert_array_equal(aa['fields'],bb['fields'])
