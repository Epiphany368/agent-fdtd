import numpy as np
import pytest
import torch

from torchfdtd import (Project,Simulation,Source,Structure,Material,RunControl,FDTD,
                        run_tensor_batch,FieldMonitor,SpectrumSettings,TimeSignal)
from torchfdtd.run_control import source_end_time
from torchfdtd.source_preview import preview_source
from benchmarks.tfsf_sources import box_project,reference


@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('direction',['+','-'])
def test_closed_box_against_independent_discrete_fields(dimension,direction):
    for axis in range(2 if dimension=='2d' else 3):
        for component in range(3):
            if component==axis:continue
            p=box_project(axis,component,direction,dimension)
            expected,masks=reference(p);result=Simulation(p).run()
            for a,b,mask in zip((result.electric,result.magnetic),expected,masks):
                assert np.linalg.norm(b)>1e-3
                assert np.linalg.norm(a-b)/np.linalg.norm(b)<2e-5
                assert np.max(abs(a[~mask]))/np.max(abs(a))<2e-7


@pytest.mark.parametrize('precision',['float32','float64'])
def test_tfsf_cuda_graph_eager_tensor_vector_and_scatterer(precision):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p=box_project();p.region.precision=precision;p.region.steps=140
    p.sources[0].theta=43;p.sources[0].phi=90
    p.structures=[Structure(kind='sphere',radius=.3,material='SiN (constant n)')]
    p.monitors.append(FieldMonitor(center=(1.1,0,0),size=(0,1.8,1.8),
        spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')))
    cpu=Simulation(p).run();p.region.backend='cuda'
    tol=3e-5 if precision=='float32' else 2e-12
    outputs=[]
    for kernel in ('torch','fused'):
        p.region.cuda_kernel=kernel
        graph=Simulation(p).run();eager=Simulation(p).run(cuda_graph=False)
        for field in ('electric','magnetic','signals'):
            np.testing.assert_allclose(getattr(graph,field),getattr(cpu,field),atol=tol,rtol=tol)
            np.testing.assert_array_equal(getattr(graph,field),getattr(eager,field))
        outputs.append(graph)
    for field in ('electric','magnetic','signals'):
        np.testing.assert_array_equal(getattr(outputs[0],field),getattr(outputs[1],field))
    other=p.model_copy(deep=True);other.sources[0].direction='-';other.sources[0].phase=37
    batch=run_tensor_batch([p,other])
    for item,q in zip(batch.items,(p,other)):
        a,b=item.load(),Simulation(q).run()
        for field in ('electric','magnetic','signals','frames'):
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
        np.testing.assert_array_equal(a.frequency_fields[0]['fields'],b.frequency_fields[0]['fields'])


def test_tfsf_geometry_material_guards_preview_and_source_end(tmp_path):
    p=box_project(dimension='2d')
    p.save(tmp_path/'box.json');assert Project.load(tmp_path/'box.json').sources[0].kind=='tfsf'
    preview=preview_source(p,p.sources[0].id)
    assert len(preview['injections'])==3 and preview['tfsf']['incident_line_cells']>200
    assert np.max(abs(np.asarray(preview['signal'])))>.1
    assert source_end_time(p)>p.sources[0].pulse_offset
    for size in ((0,1,1),(2.4,2.4,2.4)):
        q=p.model_copy(deep=True);q.sources[0].size=size
        with pytest.raises(ValueError):Simulation(q).run()
    q=p.model_copy(deep=True);q.region.boundaries.y_min.kind=q.region.boundaries.y_max.kind='periodic'
    with pytest.raises(ValueError,match='PML'):Simulation(q).run()
    q=p.model_copy(deep=True);q.structures=[Structure(size=(.2,2,1),center=(.8,0,0))]
    with pytest.raises(ValueError,match='homogeneous'):Simulation(q).run()
    with pytest.raises(ValueError,match='transverse'):Source(kind='tfsf',normal='z',component='Ez')
    with pytest.raises(ValueError,match='paired'):Source(kind='tfsf',injection='soft')
    f=FDTD(p.model_copy(deep=True));f.addtfsf(name='box facade',x_span=1.6e-6,y_span=1.6e-6,
                                           direction='Backward',theta=0,phi=0)
    assert f.project.sources[-1].direction=='-'
    np.testing.assert_allclose(f.project.sources[-1].size[:2],(1.6,1.6),rtol=1e-15)
    Project.model_validate(f.project.model_dump())


def test_tfsf_background_index_and_graded_dispersive_scatterer():
    for index in (1.5,3.):
        p=box_project(direction='-',dimension='2d',index=index);p.region.steps=120
        expected,_=reference(p);actual=Simulation(p).run()
        for a,b in zip((actual.electric,actual.magnetic),expected):
            assert np.linalg.norm(a-b)/np.linalg.norm(b)<2e-5
    from test_multipole import multi_material
    p=box_project(dimension='2d');p.region.size=(6,6,1);p.region.mesh_type='graded'
    p.region.mesh_ppw=6;p.region.mesh_max=.2
    p.materials.append(multi_material());p.structures=[Structure(kind='circle',radius=.25,material='three-pole')]
    a=Simulation(p).run()
    assert a.project.region.shape[0]<60 and np.max(abs(a.electric))>0
    if torch.cuda.is_available():
        p.region.backend='cuda';p.region.cuda_kernel='fused'
        b=Simulation(p).run()
        np.testing.assert_allclose(a.electric,b.electric,rtol=2e-10,atol=2e-12)
        c=run_tensor_batch([p]).items[0].load()
        np.testing.assert_array_equal(c.electric,b.electric)
    p.structures[0].center=(.8,0,0)
    with pytest.raises(ValueError,match='homogeneous|dispersive'):Simulation(p).run()


def test_tfsf_overlapping_boxes_and_soft_source_preserve_superposition():
    p=box_project(dimension='2d');p.region.steps=130
    q=p.model_copy(deep=True);q.sources[0].normal='y';q.sources[0].direction='-';q.sources[0].phase=72
    a,b=Simulation(p).run(),Simulation(q).run()
    both=p.model_copy(deep=True);both.sources.append(q.sources[0].model_copy(update={'id':'second'}))
    c=Simulation(both).run()
    np.testing.assert_allclose(c.electric,a.electric+b.electric,atol=2e-14,rtol=2e-12)
    if torch.cuda.is_available():
        both.region.backend='cuda';both.region.cuda_kernel='fused'
        both.sources.append(Source(id='soft',center=(.2,.2,0),wavelength=.8,pulse_cycles=1))
        independent=Simulation(both).run()
        batch=run_tensor_batch([both,both]).items
        for item in batch:np.testing.assert_array_equal(item.load().electric,independent.electric)


def test_tfsf_incident_energy_is_checked_and_delayed_drive_cannot_stop():
    import fdtd
    from torchfdtd.boundaries import YeeGrid
    from torchfdtd.tfsf import TfsfState
    from torchfdtd.run_control import StateDiagnostics
    p=box_project(dimension='2d')
    g=YeeGrid(p.region);line=TfsfState(p.sources[0],p.region,g)
    line.e[0]=2
    diagnostics=StateDiagnostics(g,fused=False)
    assert diagnostics.measure()[0]>0 and diagnostics.measure()[1]==0
    line.ph[-1]=np.nan
    with pytest.raises(FloatingPointError):diagnostics.measure()
    p.sources[0].pulse_offset=1e-9
    p.region.run_control=RunControl(auto_shutoff=True,min_steps=10,check_interval=10,consecutive_checks=2)
    result=Simulation(p).run()
    assert result.summary['steps']==p.region.steps
    assert all(not s['source_finished'] for s in result.summary['diagnostics'])
    if not torch.cuda.is_available():return
    old=torch.get_default_dtype()
    try:
        fdtd.set_backend('torch.cuda.float64');fdtd.backend.float=torch.float64
        g=YeeGrid(p.region);line=TfsfState(p.sources[0],p.region,g)
        line.e.fill_(1e-25);line.h.fill_(2e-25)
        a,b=StateDiagnostics(g),StateDiagnostics(g,fused=False)
        np.testing.assert_allclose(a.measure(),b.measure(),rtol=1e-13,atol=0)
        assert a.measure()[0]>0
        line.pe[-1]=float('nan')
        for diagnostic in (a,b):
            with pytest.raises(FloatingPointError):diagnostic.measure()
    finally:
        fdtd.set_backend('numpy');fdtd.backend.float=np.float64;torch.set_default_dtype(old)


def test_tfsf_sphere_mie_reference_and_measured_scattering():
    from examples.tfsf_sphere import mie_cross_section,make_project,evaluate
    wavelengths=np.array([1.3,1.55,1.8]);radius=.001;m=1.5
    rayleigh=8*np.pi/3*(2*np.pi/wavelengths)**4*radius**6*((m*m-1)/(m*m+2))**2
    np.testing.assert_allclose(mie_cross_section(wavelengths,radius,m),rayleigh,rtol=1e-5)
    np.testing.assert_array_equal(mie_cross_section(wavelengths,.3,1),0)
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable for sphere integration')
    p=make_project(.05,'cuda');sample=Simulation(p).run();p.structures=[];ref=Simulation(p).run()
    report=evaluate(sample,ref)
    assert report['max_relative_error']<.025
    assert report['max_empty_box_cross_section_um2']<1e-9


def test_tfsf_auto_shutoff_preserves_prefix_after_incident_line_drains():
    p=box_project(dimension='2d');p.region.steps=1800
    times=np.arange(101)*p.region.time_step
    values=np.sin(2*np.pi*np.arange(101)/100)*np.sin(np.pi*np.arange(101)/100)**2
    p.sources[0].pulse='sampled'
    p.sources[0].signal=TimeSignal(time_s=times.tolist(),amplitude=values.tolist(),phase_rad=[0.]*101)
    p.region.run_control=RunControl(auto_shutoff=True,check_interval=20,consecutive_checks=3,decay_threshold=1e-6)
    stopped=Simulation(p).run()
    assert stopped.summary['termination_reason']=='decayed'
    p.region.run_control.auto_shutoff=False
    full=Simulation(p).run()
    np.testing.assert_array_equal(stopped.signals,full.signals[:len(stopped.times)])
    assert np.max(abs(full.signals[len(stopped.times):]))<.001*np.max(abs(full.signals))
