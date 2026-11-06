import json
import fdtd
import numpy as np
import pytest
import torch
from torchfdtd import Project,Region,Source,Structure,Monitor,FieldMonitor,Simulation,Boundaries,BoundaryFace,SpectrumSettings
from torchfdtd.boundaries import YeeGrid
from torchfdtd.field_monitors import FrequencyPlane,plane_plan,normalize_flux
from torchfdtd.spectra import frequency_samples
from torchfdtd.solver import C0


def slab_project():
    return Project(region=Region(size=(8,.5,1),mesh=.025,steps=1600,pml_cells=16,backend='cpu',precision='float64',
                   snapshot_interval=1600,material_sampling='yee',boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'),y_max=BoundaryFace(kind='periodic'))),
                   structures=[Structure(center=(.0125,0,0),size=(.2,.5,1),material='SiO2 (constant n)')],
                   sources=[Source(kind='plane',size=(0,.5,0),center=(-1.5,0,0),wavelength=1.55,pulse_cycles=1)],
                   monitors=[FieldMonitor(id='reflection',center=(-.8,0,0),size=(0,.5,1),spectrum=SpectrumSettings(sampling='frequency',frequency_points=31,apodization='none')),
                             FieldMonitor(id='transmission',center=(.8,0,0),size=(0,.5,1),spectrum=SpectrumSettings(sampling='frequency',frequency_points=31,apodization='none'))])


@pytest.mark.parametrize('normal',['x','y','z'])
def test_six_component_dft_half_step_and_surface_quadrature(normal):
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64
    r=Region(dimension='3d',size=(2,2,2),mesh=.1,pml_cells=3,steps=200,precision='float64')
    g=YeeGrid(r);f=7/(r.steps*r.time_step);axis='xyz'.index(normal)
    size=[.63,.47,.51];size[axis]=0
    m=FieldMonitor(normal=normal,size=tuple(size),downsample=3,spectrum=SpectrumSettings(sampling='custom',custom_frequencies_hz=[f],apodization='none'))
    monitor=FrequencyPlane(g,m);ec=(axis+1)%3;hc=(axis+2)%3
    for q in range(r.steps):
        g.E[...,ec]=np.cos(2*np.pi*f*(q+1)*r.time_step)
        g.H[...,hc]=np.cos(2*np.pi*f*(q+1.5)*r.time_step)
        monitor.update(np.array([q]))
    result=monitor.result();amplitude=r.steps*r.time_step/2
    area=np.prod([s*1e-6 for s in size if s])
    assert result['weights'].sum()==pytest.approx(area,rel=1e-14)
    np.testing.assert_allclose(result['fields'][...,ec]/amplitude,1,atol=2e-14)
    np.testing.assert_allclose(result['fields'][...,hc+3]/amplitude,1,atol=2e-14)
    assert result['flux'][0]/(.5*amplitude**2*area)==pytest.approx(1,abs=2e-14)


def test_flux_fresnel_conservation_reference_validation_and_npz(tmp_path):
    p=slab_project();p.materials[1].index=1.5
    sample=Simulation(p).run();p.structures=[];reference=Simulation(p).run()
    reflected=normalize_flux(sample.frequency_fields[0],reference.frequency_fields[0],subtract_incident=True)
    transmitted=normalize_flux(sample.frequency_fields[1],reference.frequency_fields[1])
    assert reflected['valid'].all() and transmitted['valid'].all()
    wavelength=C0/transmitted['frequency_hz']*1e6
    expected=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    np.testing.assert_allclose(transmitted['ratio'],expected,atol=.004)
    np.testing.assert_allclose(-reflected['ratio'],1-expected,atol=.004)
    np.testing.assert_allclose(transmitted['ratio']-reflected['ratio'],1,atol=.004)
    sample.save(tmp_path/'plane.npz');saved=np.load(tmp_path/'plane.npz')
    np.testing.assert_array_equal(saved['field_monitor_0_fields'],sample.frequency_fields[0]['fields'])
    assert json.loads(str(saved['field_monitors']))[0]['normal_axis']=='x'
    changed=dict(reference.frequency_fields[0],run_signature='different')
    with pytest.raises(ValueError,match='run settings'):normalize_flux(sample.frequency_fields[0],changed)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_graded_bloch_plane_cuda_parity_and_global_settings(precision):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p=slab_project();p.region.mesh=.05;p.region.size=(6,1,1);p.region.steps=160;p.region.pml_cells=8
    p.region.mesh_type='graded';p.region.precision=precision;p.region.bloch_phase=(0,.4,0)
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
    p.monitors=p.monitors[:1];p.monitors[0].size=(0,1,1);p.sources[0].size=(0,1,0)
    p.monitors[0].use_global_monitor=True
    p.global_monitor=SpectrumSettings(sampling='custom',custom_frequencies_hz=[180e12,210e12],apodization='none')
    cpu=Simulation(p).run();p.region.backend='cuda';gpu=Simulation(p).run()
    assert gpu.summary['cuda_graph']
    scale=max(abs(cpu.frequency_fields[0]['fields']).ravel())
    tolerance=4e-5 if precision=='float32' else 4e-12
    np.testing.assert_allclose(gpu.frequency_fields[0]['fields']/scale,cpu.frequency_fields[0]['fields']/scale,atol=tolerance,rtol=tolerance)
    assert len(gpu.flux_data()[0]['flux'])==2


def test_custom_chebyshev_and_global_point_validation():
    custom=SpectrumSettings(sampling='custom',custom_frequencies_hz=[100e12,120e12,180e12])
    np.testing.assert_array_equal(frequency_samples(custom),custom.custom_frequencies_hz)
    with pytest.raises(ValueError,match='increasing'):SpectrumSettings(sampling='custom',custom_frequencies_hz=[120e12,100e12])
    with pytest.raises(ValueError,match='Nyquist'):Project(monitors=[Monitor(use_global_monitor=True)],global_monitor=SpectrumSettings(sampling='custom',custom_frequencies_hz=[1e20]))
    c=SpectrumSettings(sampling='chebyshev',frequency_points=5)
    values=frequency_samples(c);assert np.all(np.diff(values)>0)
    assert values[0]>C0/1.8e-6 and values[-1]<C0/1.3e-6
    assert values[2]==pytest.approx(.5*(C0/1.3e-6+C0/1.8e-6))
    p=Project(monitors=[Monitor(use_global_monitor=True)],global_monitor=custom)
    assert p.resolved_monitor(p.monitors[0]).spectrum.custom_frequencies_hz==custom.custom_frequencies_hz
