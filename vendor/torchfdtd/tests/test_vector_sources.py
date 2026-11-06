"""Electric/magnetic vector injection against independent Fourier-space updates."""
import math
import numpy as np
import pytest
import torch

from torchfdtd import Project,Region,Source,Monitor,Simulation,RunControl,run_tensor_batch,FDTD,Result
from torchfdtd.models import Boundaries,BoundaryFace
from torchfdtd.source_preview import preview_source
from torchfdtd.waveforms import source_time_signal


from benchmarks.vector_sources import periodic_project, fourier_reference


@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('family',['E','H'])
def test_vector_injection_matches_independent_fourier_solution(dimension,family):
    p=periodic_project(dimension,family);expected=fourier_reference(p);run=Simulation(p).run()
    assert np.linalg.norm(expected[0])>.01 and np.linalg.norm(expected[1])>.01
    np.testing.assert_allclose(run.electric,expected[0],rtol=2e-12,atol=2e-13)
    np.testing.assert_allclose(run.magnetic,expected[1],rtol=2e-12,atol=2e-13)


@pytest.mark.parametrize('family',['E','H'])
def test_vector_is_normalized_linear_superposition_and_preview(family,tmp_path):
    p=periodic_project('3d',family);source=p.sources[0]
    expected=[np.zeros((*p.region.shape,3)) for _ in range(2)]
    for field,weight in source.polarization_components:
        basis=p.model_copy(deep=True);basis.sources[0].component=field;basis.sources[0].theta=None
        result=Simulation(basis).run()
        for accum,values in zip(expected,(result.electric,result.magnetic)):accum+=weight*values
    actual=Simulation(p).run()
    for a,b in zip((actual.electric,actual.magnetic),expected):np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)
    preview=preview_source(p,source.id)
    times=np.arange(1,p.region.steps+1)*p.region.time_step+source.time_offset_steps*p.region.time_step
    np.testing.assert_allclose(preview['time_fs'],times*1e15,rtol=0,atol=0)
    np.testing.assert_array_equal(preview['signal'],source_time_signal(source,times))
    assert sum(v*v for v in preview['polarization_components'].values())==pytest.approx(1)
    actual.save(tmp_path/'vector.npz');loaded=Result.load(tmp_path/'vector.npz')
    assert loaded.project.sources[0].theta==38 and loaded.project.sources[0].component==family+'z'
    np.testing.assert_array_equal(actual.magnetic,loaded.magnetic)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_mixed_vector_sources_cuda_graph_fused_and_tensor_batch(precision):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    p=periodic_project('2d','H');p.region.precision=precision
    p.region.mesh_type='graded';p.region.mesh_auto_refine=False;p.region.mesh_max=.15
    p.sources.append(Source(component='Ez',theta=74,phi=-31,center=(-.22,-.11,0),wavelength=1,pulse_cycles=1))
    # CPML and periodic boundaries coexist, with both source families in a case.
    p.region.boundaries.x_min=p.region.boundaries.x_max=BoundaryFace(layers=3)
    p.region.size=(2.4,1.4,1.2)
    cpu=Simulation(p).run();p.region.backend='cuda'
    tolerance=2e-5 if precision=='float32' else 3e-12
    for kernel in ('torch','fused'):
        p.region.cuda_kernel=kernel
        graph=Simulation(p).run();eager=Simulation(p).run(cuda_graph=False)
        for name in ('electric','magnetic','signals'):
            np.testing.assert_allclose(getattr(graph,name),getattr(cpu,name),rtol=tolerance,atol=tolerance)
            np.testing.assert_array_equal(getattr(graph,name),getattr(eager,name))
    other=p.model_copy(deep=True);other.sources[0].theta=53;other.sources[0].phase=-51
    batch=run_tensor_batch([p,other])
    for item,project in zip(batch.items,[p,other]):
        reference=Simulation(project).run()
        for name in ('electric','magnetic','signals'):
            np.testing.assert_array_equal(getattr(item.load(),name),getattr(reference,name))


def test_fsp_vector_mapping_and_familiar_python_commands():
    from torchfdtd.fsp_native import convert_fsp
    from torchfdtd.fsp_binary import FspDocument
    from test_fsp_native import fixture
    doc=FspDocument(fixture(source_overrides={'theta':43.,'angle':219.,'phase':-27.}))
    original=doc.data
    report=convert_fsp(doc,backend='cpu')
    assert report.project is not None,report.issues
    s=report.project.sources[0]
    assert (s.theta,s.phi,s.phase)==(43,219,-27)
    assert len(s.polarization_components)==3 and doc.data==original
    run=Simulation(report.project).run()
    assert np.linalg.norm(run.electric)>0
    f=FDTD();f.adddipole(name='magnet',dipole_type='Magnetic dipole',theta=43,phi=219)
    assert f.project.sources[0].polarization_components==tuple(('H'+field[1],weight) for field,weight in s.polarization_components)
    f.set('dipole type','Electric dipole');assert f.project.sources[0].component.startswith('E')
    with pytest.raises(ValueError):f.set('theta',181)
    f.set('polarization','Hx');assert f.project.sources[0].theta is None
    with pytest.raises(ValueError):Source(theta=float('nan'))
    assert Source(theta=180).polarization_components==(('Ez',-1.),)
    assert Source(theta=90,phi=90,component='Hx').polarization_components==(('Hy',1.),)
    weak=dict(Source(theta=1e-14,phi=0).polarization_components)
    assert 0<weak['Ex']<1e-15


def test_bloch_magnetic_vector_sheet_and_plane_dft():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_boundaries import pulse_project
    from torchfdtd import FieldMonitor,SpectrumSettings
    p=pulse_project(bloch=True);p.region.steps=140;p.region.material_sampling='yee'
    p.sources[0].component='Hy';p.sources[0].theta=48;p.sources[0].phi=133
    p.monitors.append(FieldMonitor(name='flux',center=(.4,0,0),size=(0,.8,1),
        spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')))
    cpu=Simulation(p).run();p.region.backend='cuda';gpu=Simulation(p).run()
    np.testing.assert_allclose(gpu.electric,cpu.electric,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(gpu.magnetic,cpu.magnetic,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(gpu.field_monitor('flux')['fields'],cpu.field_monitor('flux')['fields'],rtol=3e-12,atol=1e-27)
    with pytest.raises(ValueError,match='real fields'):run_tensor_batch([p])
