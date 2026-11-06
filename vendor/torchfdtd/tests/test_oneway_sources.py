import numpy as np
import pytest
import torch
from torchfdtd import Project, Simulation, Source, Structure, RunControl, FDTD, run_tensor_batch
from torchfdtd.injection import oneway_tables
from torchfdtd.source_preview import preview_source
from torchfdtd.run_control import source_end_time
from benchmarks.oneway_sources import plane_project, long_line_reference


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('direction',['+','-'])
def test_oneway_fields_against_independent_light_cone_reference(axis,direction):
    for ec in range(3):
        if ec==axis:continue
        p=plane_project(axis,ec,direction)
        reference,_=long_line_reference(p);result=Simulation(p).run()
        for actual,expected in zip((result.electric,result.magnetic),reference):
            assert np.linalg.norm(expected)>.01
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<5e-6


@pytest.mark.parametrize('direction',['+','-'])
def test_oneway_slab_fresnel_and_energy(direction):
    from examples.oneway_slab import make_project, evaluate
    from torchfdtd.field_monitors import normalize_flux
    p=make_project('cpu',direction)
    sample=Simulation(p).run();p.structures=[];reference=Simulation(p).run()
    reflected=normalize_flux(sample.frequency_fields[0],reference.frequency_fields[0],subtract_incident=True)['ratio']
    transmitted=normalize_flux(sample.frequency_fields[1],reference.frequency_fields[1])
    wavelength=299792458/transmitted['frequency_hz']*1e6
    expected=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    d=1 if direction=='+' else -1
    np.testing.assert_allclose(d*transmitted['ratio'],expected,atol=.004)
    np.testing.assert_allclose(-d*reflected,1-expected,atol=.004)
    np.testing.assert_allclose(d*(transmitted['ratio']-reflected),1,atol=.004)
    metrics=evaluate(sample,reference)
    assert metrics['max_scattered_R_difference']<2e-4
    assert metrics['max_empty_scattered_flux_ratio']<1e-8


@pytest.mark.parametrize('precision',['float32','float64'])
def test_oneway_cuda_vector_graded_tensor_and_preview(precision,tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p=plane_project();p.region.size=(6,.4,.4);p.region.precision=precision
    p.region.mesh_type='graded';p.sources[0].theta=43;p.sources[0].phi=90
    cpu=Simulation(p).run();p.region.backend='cuda'
    tolerance=3e-5 if precision=='float32' else 3e-12
    for kernel in ('torch','fused'):
        p.region.cuda_kernel=kernel
        graph=Simulation(p).run();eager=Simulation(p).run(cuda_graph=False)
        for field in ('electric','magnetic','signals'):
            np.testing.assert_allclose(getattr(graph,field),getattr(cpu,field),rtol=tolerance,atol=tolerance)
            np.testing.assert_array_equal(getattr(graph,field),getattr(eager,field))
    other=p.model_copy(deep=True);other.sources[0].direction='-';other.sources[0].phase=37
    batch=run_tensor_batch([p,other])
    for item,q in zip(batch.items,[p,other]):
        independent=Simulation(q).run()
        for field in ('electric','magnetic','signals'):
            np.testing.assert_array_equal(getattr(item.load(),field),getattr(independent,field))
    preview=preview_source(p,p.sources[0].id)
    assert len(preview['injections'])==4
    np.testing.assert_array_equal(preview['signal'],oneway_tables(p.sources[0],p.region)[0].astype(precision))
    p.save(tmp_path/'plane.json');assert Project.load(tmp_path/'plane.json').sources[0].injection=='oneway'


def test_oneway_validation_facade_and_source_gate():
    p=plane_project(dimension='2d');s=p.sources[0]
    with pytest.raises(ValueError,match='transverse'):Source(kind='plane',injection='oneway',normal='z',component='Ez')
    q=p.model_copy(deep=True);q.region.boundaries.y_min.kind=q.region.boundaries.y_max.kind='pml'
    with pytest.raises(ValueError):Simulation(q).run()
    q=p.model_copy(deep=True);q.sources[0].size=(0,.2,0)
    with pytest.raises(ValueError,match='complete transverse'):Simulation(q).run()
    q=p.model_copy(deep=True);q.structures=[Structure(size=(.2,.4,1))]
    with pytest.raises(ValueError,match='homogeneous'):Simulation(q).run()
    q=p.model_copy(deep=True);q.region.boundaries.y_min.kind=q.region.boundaries.y_max.kind='bloch';q.region.bloch_phase=(0,.4,0)
    with pytest.raises(ValueError,match='zero Bloch'):Simulation(q).run()
    f=FDTD(p.model_copy(deep=True));f.addplane(name='backward',direction='Backward',incident_pml_cells=128)
    assert f.project.sources[-1].direction=='-' and f.project.sources[-1].incident_pml_cells==128
    f.set('injection axis','y-axis')
    assert f.project.sources[-1].size[1]==0 and f.project.sources[-1].size[0]==p.region.actual_size[0]
    f.addplane(name='vector plane',component='Ez',theta=43,phi=90)
    assert len(f.project.sources[-1].polarization_components)==2
    preview=preview_source(p,s.id)
    assert preview['injections'][1]['time_fs'][0]==pytest.approx(1.5*p.region.time_step*1e15)
    assert source_end_time(p)>s.pulse_offset
    p.region.run_control=RunControl(auto_shutoff=True,min_steps=10,check_interval=10,consecutive_checks=2,decay_threshold=.99)
    result=Simulation(p).run()
    assert all(not h['source_finished'] for h in result.summary['diagnostics'] if h['time_s']<source_end_time(p))
    p.sources[0].pulse_offset=1e-9
    assert source_end_time(p)>1e-9
    delayed=Simulation(p).run()
    assert delayed.summary['termination_reason']=='max_steps'
    assert all(not h['source_finished'] for h in delayed.summary['diagnostics'])


def test_incident_line_absorption_convergence_and_background():
    p=plane_project(dimension='2d',index=1.5);p.region.steps=1400
    s=p.sources[0];s.wavelength=1.55;s.pulse_length=6e-15;s.pulse_offset=15e-15
    _,reference=long_line_reference(p);errors=[]
    for layers in (32,96,192):
        s.incident_pml_cells=layers
        errors.append(max(np.linalg.norm(a-b)/np.linalg.norm(b) for a,b in zip(oneway_tables(s,p.region),reference)))
    assert errors[2]<errors[1]<errors[0] and errors[1]<1e-6


@pytest.mark.parametrize('pulse',['continuous','broadband','sampled'])
def test_incident_temporal_modes_and_late_sampled_drive(pulse):
    from torchfdtd import TimeSignal
    p=plane_project(dimension='2d');p.region.steps=400;s=p.sources[0]
    s.pulse=pulse
    if pulse=='broadband':s.time_definition='wavelength';s.wavelength_start=.9;s.wavelength_stop=1.3
    if pulse=='sampled':
        s.signal=TimeSignal(time_s=[0,2e-15,4e-15,20e-15,22e-15,24e-15],amplitude=[0,1,0,0,1,0],phase_rad=[np.pi/2]*6)
    p.global_source=type(p.global_source).model_validate({k:v for k,v in s.model_dump().items() if k in type(p.global_source).model_fields})
    s.use_global_source=True
    result=Simulation(p).run()
    assert np.max(abs(result.electric))>1e-8
    if pulse=='continuous':assert not np.isfinite(source_end_time(p))
    elif pulse=='sampled':assert source_end_time(p)>24e-15
