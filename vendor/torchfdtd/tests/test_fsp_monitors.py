"""Synthetic spectral-layout mappings. No vendor runtime or calculation data."""
import numpy as np
import pytest
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import DFT, convert_fsp
from torchfdtd.models import FieldMonitor
from torchfdtd.spectra import frequency_samples
from test_fsp_native import fixture


def spectral_fixture(axis='x',overrides=None,global_overrides=None):
    a='xyz'.index(axis);spans=np.array([.8,1.,1.2])*1e-6;spans[a]=0
    center=np.array([.12,.23,.34])*1e-6;low=center-spans/2
    settings=dict(monitorShape=4+a,standardDFT=1,partialSpectralAverage=0,totalSpectralAverage=0,
        useGlobalDFT=0,sampleSpacing=0,useWavelengthSpacing=0,useSourceLimits=0,
        userF1=180e12,userF2=230e12,nfreqDesired=7,apodizationType=0,
        apodizationCenter=37e-15,apodizationWidth=11e-15,downsampleT=3,
        downsampleX=2,downsampleY=3,downsampleZ=4,highPrecisionDFT=1,
        left=low[0],bottom=low[1],z1=low[2],width=spans[0],height=spans[1],z2=low[2]+spans[2],
        outputPower=1,unfold=np.zeros(6),customFrequencySamples=np.array([181e12,199e12,227e12]))
    settings.update({'output'+c:1 for c in ('Ex','Ey','Ez','Hx','Hy','Hz')})
    settings.update({'outputP'+c:0 for c in 'xyz'});settings.update(overrides or {})
    return fixture(region_overrides=global_overrides,monitor_class=DFT,monitor_overrides=settings)


@pytest.mark.parametrize('axis','xyz')
@pytest.mark.parametrize('apo,expected',[(0,'none'),(1,'full'),(2,'start'),(3,'end')])
def test_plane_geometry_apodization_strides_outputs_and_immutable_input(axis,apo,expected):
    raw=spectral_fixture(axis,dict(apodizationType=apo));doc=FspDocument(raw);report=convert_fsp(doc)
    assert report.project is not None,report.issues
    m=report.project.monitors[0];assert isinstance(m,FieldMonitor)
    assert m.normal==axis and m.spectrum.apodization==expected
    np.testing.assert_allclose(m.center,[.12,.23,.34],atol=1e-14)
    assert m.downsample_xyz==(2,3,4) and m.time_downsample==3
    assert m.dft_precision=='float64' and m.spatial_interpolation=='nearest'
    assert m.spectrum.apodization_center==37e-15 and m.spectrum.apodization_time_width==11e-15
    assert m.record_poynting==() and m.record_flux and len(m.record_fields)==6
    assert doc.data==raw and report.source_sha256==doc.fingerprint()
    assert any(i['code']=='plane_dft_convention' for i in report.issues)


@pytest.mark.parametrize('spacing,wl',[(0,0),(0,1),(1,0),(1,1),(2,0)])
def test_frequency_tables_match_independent_explicit_formula(spacing,wl):
    report=convert_fsp(FspDocument(spectral_fixture(overrides=dict(sampleSpacing=spacing,useWavelengthSpacing=wl))))
    assert report.project is not None,report.issues
    actual=frequency_samples(report.project.monitors[0].spectrum)
    u=np.linspace(0,1,7) if spacing==0 else (1-np.cos(np.arange(7)*np.pi/6))/2
    expected=np.array([181e12,199e12,227e12]) if spacing==2 else (
        1/((1-u)/230e12+u/180e12) if wl else 180e12+u*50e12)
    np.testing.assert_allclose(actual,expected,rtol=8e-16)


def test_global_frequency_inheritance_keeps_local_apodization_and_power_only():
    switches={'output'+c:0 for c in ('Ex','Ey','Ez','Hx','Hy','Hz')}
    raw=spectral_fixture(overrides=dict(**switches,useGlobalDFT=1,apodizationType=3),global_overrides=dict(
        fStart=170e12,fEnd=250e12,nfreqDesired=9,sampleSpacing=1,useWavelengthSpacing=0,useSourceLimits=0))
    report=convert_fsp(FspDocument(raw));assert report.project is not None,report.issues
    p=report.project;m=p.monitors[0]
    assert m.record_fields==() and m.required_fields==('Ey','Ez','Hy','Hz')
    assert m.use_global_monitor and not m.inherit_apodization
    p.global_monitor.frequency_points=5
    resolved=p.resolved_monitor(m);assert resolved.spectrum.apodization=='end'
    np.testing.assert_allclose(frequency_samples(resolved.spectrum)[[0,-1]],[170e12,250e12],rtol=3e-16)
    assert len(frequency_samples(resolved.spectrum))==5


@pytest.mark.parametrize('changes',[
    dict(spatialAveraging=2),dict(standardDFT=0),dict(partialSpectralAverage=1),
    dict(totalSpectralAverage=1),dict(unfold=np.ones(6)),dict(monitorShape=7),
    dict(downsampleT=2.5),dict(downsampleT=100),dict(downsampleX=0),
    dict(useSourceLimits=1),dict(useGlobalDFT=1),dict(highPrecisionDFT=2),dict(enabled=2)])
def test_unsupported_monitor_settings_never_create_partial_scene(changes):
    report=convert_fsp(FspDocument(spectral_fixture(overrides=changes)))
    assert report.project is None
    assert not report.as_dict()['native_execution_allowed']
    assert any(i['severity']=='error' for i in report.issues)


@pytest.mark.parametrize('shape,normal',[(1,'y'),(2,'x')])
def test_xy_layout_lines_map_to_native_invariant_planes(shape,normal):
    dt=.8*.1e-6/299792458/np.sqrt(2)
    raw=spectral_fixture(normal,dict(monitorShape=shape,z1=0.,z2=0.),
                         dict(dimension=0,dt=dt,MaxSimTime=40*dt))
    report=convert_fsp(FspDocument(raw));assert report.project is not None,report.issues
    m=report.project.monitors[0]
    assert m.normal==normal and m.center[2]==0 and m.size[2]==1


def test_converted_selected_plane_runs_on_cpu_cuda_and_tensor_batch():
    import torch
    from torchfdtd import Simulation,run_tensor_batch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    report=convert_fsp(FspDocument(spectral_fixture(overrides=dict(outputEx=0,outputHx=0))),backend='cpu')
    assert report.project is not None,report.issues
    p=report.project;cpu=Simulation(p).run();p.region.backend='cuda';p.region.cuda_kernel='fused';p.region.cuda_monitor_kernel='fused'
    single=Simulation(p).run();batch=run_tensor_batch([p,p.model_copy(deep=True)]);batch.raise_for_errors()
    scale=np.max(abs(cpu.frequency_fields[0]['fields']))
    assert scale>0
    np.testing.assert_allclose(single.frequency_fields[0]['fields']/scale,cpu.frequency_fields[0]['fields']/scale,rtol=2e-5,atol=2e-5)
    for item in batch.items:
        np.testing.assert_array_equal(item.load().frequency_fields[0]['fields'],single.frequency_fields[0]['fields'])
        np.testing.assert_array_equal(item.load().frequency_fields[0]['flux'],single.frequency_fields[0]['flux'])
