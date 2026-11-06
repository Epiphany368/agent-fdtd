import math

import numpy as np
import pytest

from torchfdtd import FDTD, Project, Source, SourceTimeSettings, Simulation
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import convert_fsp
from torchfdtd.waveforms import TAIL_INNER, TAIL_OUTER, gaussian_envelope, pulse_parameters, source_time_signal
from torchfdtd.solver import estimate
from test_fsp_native import fixture
from test_solver import small


def broad(**kwargs):
    return Source(pulse='broadband',time_definition='wavelength',wavelength_start=1.2,wavelength_stop=1.8,**kwargs)


def test_range_generation_and_short_pulse_switch_are_not_a_fixed_carrier_substitute():
    source=broad(optimize_for_short_pulse=False)
    p=pulse_parameters(source)
    assert p.frequency_hz==pytest.approx(.5*(299792458/1.2e-6+299792458/1.8e-6))
    # Native invariants only. Retired vendor-derived golden timing values are
    # deliberately not used as current validation evidence.
    assert p.as_dict()['pulse_length_s'] > 0
    assert p.offset_s > 3*p.sigma_s
    assert p.chirped
    source.optimize_for_short_pulse=True
    assert pulse_parameters(source).sigma_s==p.sigma_s/4
    source.wavelength_start=1.4;source.wavelength_stop=1.6
    assert not pulse_parameters(source).chirped
    source.wavelength_start=source.wavelength_stop=1.55
    p=pulse_parameters(source)
    assert not p.chirped and p.frequency_span_hz==0


def test_chirp_frequency_sweeps_continuously_between_requested_limits():
    s=broad(optimize_for_short_pulse=False)
    p=pulse_parameters(s);t=p.offset_s+np.linspace(-3,3,6001)*p.sigma_s
    sine=source_time_signal(s,t);s.phase=90;cosine=source_time_signal(s,t)
    phase=np.unwrap(np.arctan2(sine,cosine));frequency=-np.gradient(phase,t)/(2*np.pi)
    assert frequency[500]==pytest.approx(299792458/1.8e-6,rel=1e-9)
    assert frequency[5500]==pytest.approx(299792458/1.2e-6,rel=1e-9)
    assert frequency[3000]==pytest.approx(p.frequency_hz,rel=1e-9)
    assert np.min(np.diff(frequency))>-1e-11*p.frequency_hz  # differentiation roundoff
    for center in (-2,2):
        index=np.argmin(abs((t-p.offset_s)/p.sigma_s-center))
        assert abs(frequency[index+1]-frequency[index-1])<p.frequency_span_hz*1e-6


def test_endpoint_taper_is_continuous_with_continuous_slope_and_compact_support():
    h=1e-5
    for boundary in (TAIL_INNER,TAIL_OUTER):
        x=np.array([boundary-h,boundary,boundary+h]);y=gaussian_envelope(x,True)
        slopes=np.diff(y)/h
        assert abs(slopes[1]-slopes[0])<3e-8
    assert gaussian_envelope(np.array([0,TAIL_OUTER,TAIL_OUTER+1]),True).tolist()==[1,0,0]
    assert gaussian_envelope(np.array([TAIL_INNER]),True)[0]==pytest.approx(1e-4)


def test_familiar_frequency_wavelength_and_manual_switch_preserve_effective_pulse():
    f=FDTD();f.adddipole(name='s')
    f.setglobalsource('set wavelength',1)
    f.setglobalsource('wavelength start',1.2e-6);f.setglobalsource('wavelength stop',1.8e-6)
    f.setglobalsource('optimize for short pulse',0)
    f.setglobalsource('eliminate discontinuities',1)
    f.setnamed('s','use global source settings',1)
    before=pulse_parameters(f.project.resolved_source(f.project.sources[0]))
    assert f.getglobalsource('frequency start')==pytest.approx(299792458/1.8e-6)
    f.setglobalsource('set frequency',1)
    assert pulse_parameters(f.project.global_source)==before
    f.setglobalsource('set time domain',1)
    after=pulse_parameters(f.project.global_source)
    assert after.frequency_hz==pytest.approx(before.frequency_hz) and after.offset_s==before.offset_s
    assert after.sigma_s==pytest.approx(before.sigma_s,rel=1e-14,abs=0)
    assert after.frequency_span_hz==before.frequency_span_hz and after.chirped
    f.setglobalsource('pulse type','standard')
    assert not pulse_parameters(f.project.global_source).chirped
    f.setglobalsource('set wavelength',1)
    with pytest.raises(ValueError,match='Automatic range'):f.setglobalsource('pulse type','standard')


def test_range_validation_and_nyquist_check_use_active_endpoints():
    with pytest.raises(ValueError):Source(pulse='broadband')
    with pytest.raises(ValueError):Source(time_definition='wavelength')
    with pytest.raises(ValueError):Source(wavelength_start=2,wavelength_stop=1)
    with pytest.raises(ValueError):Source(pulse='broadband',time_definition='standard',chirp_bandwidth_hz=1e18)
    p=small();p.sources=[broad()];p.sources[0].wavelength_start=.01
    with pytest.raises(ValueError,match='Nyquist'):Project.model_validate(p.model_dump())


def test_simulation_end_warning_covers_the_complete_taper():
    p=small();p.sources=[broad(optimize_for_short_pulse=False,eliminate_discontinuities=True)]
    parameters=pulse_parameters(p.sources[0])
    p.region.steps=math.ceil((parameters.offset_s+3*parameters.sigma_s)/p.region.time_step)
    assert any('pulse tail' in w for w in estimate(p)['warnings'])
    p.region.steps=math.ceil((parameters.offset_s+TAIL_OUTER*parameters.sigma_s)/p.region.time_step)
    assert not any('pulse tail' in w for w in estimate(p)['warnings'])


@pytest.mark.parametrize('precision',['float32','float64'])
def test_global_broadband_cpu_gpu_graph_and_eager_agree(precision):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA GPU not available')
    p=small(precision=precision);p.global_source=SourceTimeSettings(pulse='broadband',time_definition='frequency',wavelength_start=1.2,wavelength_stop=1.8,eliminate_discontinuities=True)
    p.sources[0].use_global_source=True;p.sources[0].phase=-23;p.sources[0].amplitude=.7
    cpu=Simulation(p).run();p.region.backend='cuda'
    eager=Simulation(p).run(cuda_graph=False);graph=Simulation(p).run(cuda_graph=True)
    tol=2e-6 if precision=='float32' else 1e-12
    assert graph.summary['cuda_graph'] and np.max(abs(graph.signals))>1e-3
    np.testing.assert_allclose(graph.electric,cpu.electric,atol=tol,rtol=tol)
    np.testing.assert_allclose(graph.signals,eager.signals,atol=tol,rtol=tol)


def test_independent_range_mapping_rejects_stale_automatic_pulse_and_dc():
    low,high=299792458/1.8e-6,299792458/1.2e-6
    params=pulse_parameters(broad())
    overrides=dict(sourcePreference=2,frequencyEnvelopeType=1,frequency=(low+high)/2,
                   BBFrequencyStart=low,BBFrequencyStop=high,optimizeForShortPulse=1,
                   pulseLength=params.as_dict()['pulse_length_s'],offset=params.offset_s,eliminateDiscontinuities=1)
    p=convert_fsp(FspDocument(fixture(overrides))).project
    assert p is not None and p.sources[0].time_definition=='wavelength'
    for patch in [dict(pulseLength=1e-15),dict(frequencyEnvelopeType=0),dict(mEliminateDC=1)]:
        assert convert_fsp(FspDocument(fixture({**overrides,**patch}))).project is None
