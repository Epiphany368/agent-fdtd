"""Selective online spectra against an independent constant-space DFT oracle."""
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import FieldMonitor, Monitor, Project, Region, Source, SpectrumSettings, Simulation, Result, FDTD, run_tensor_batch
from torchfdtd.boundaries import YeeGrid
from torchfdtd.field_monitors import FrequencyPlane, FrequencyUpdates, normalize_flux, plane_plan
from torchfdtd.spectra import frequency_samples, apodization_window
from torchfdtd.tuning import _result_digest


OUTPUTS = [dict(record_fields=('Hy',),record_poynting=(),record_flux=False),
           dict(record_fields=('Ez',),record_poynting=(),record_flux=False),
           *[dict(normal=a,record_fields=(),record_poynting=(),record_flux=True) for a in 'xyz'],
           dict(record_fields=('Hz','Ey'),record_poynting=('z','x'),record_flux=False)]


@pytest.mark.parametrize('outputs',OUTPUTS)
@pytest.mark.parametrize('backend,precision,dft_precision',[
    ('numpy','float64','field'),('torch','float32','field'),
    ('torch','float64','field'),('torch.cuda','float32','field'),
    ('torch.cuda','float32','float64'),('torch.cuda','float64','field')])
def test_selective_dft_matches_independent_time_integrals(outputs,backend,precision,dft_precision):
    if 'cuda' in backend and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if 'cuda' in backend:pytest.importorskip('cupy')
    previous=torch.get_default_dtype()
    try:
        fdtd.set_backend(backend if backend=='numpy' else backend+'.'+precision)
        fdtd.backend.float=np.float64 if backend=='numpy' else getattr(torch,precision)
        r=Region(dimension='3d',size=(2,2,2),mesh=.1,pml_cells=3,steps=40,precision=precision,
                 cuda_monitor_kernel='fused' if 'cuda' in backend else 'torch')
        size=[.63,.47,.51];size['xyz'.index(outputs.get('normal','x'))]=0
        m=FieldMonitor(**outputs,size=tuple(size),downsample_xyz=(2,3,4),time_downsample=3,
            dft_precision=dft_precision,spectrum=SpectrumSettings(sampling='custom',
                custom_frequencies_hz=[110e12,180e12],apodization='full',
                apodization_center=3e-15,apodization_time_width=2e-15))
        g=YeeGrid(r);plane=FrequencyPlane(g,m);updates=FrequencyUpdates([plane])
        counter=torch.zeros(1,dtype=torch.long,device=g.E.device) if g.is_torch else np.zeros(1,dtype=int)
        history=[]
        for q in range(r.steps):
            values=np.cos(.19*q+np.arange(6))+.3*np.sin(.07*q*(np.arange(6)+1))
            history.append(values.astype(precision))
            for field,data in ((g.E,values[:3]),(g.H,values[3:])):
                if g.is_torch:field.copy_(torch.as_tensor(data,dtype=field.dtype,device=field.device))
                else:field[:]=data
            counter[0]=q;updates.update(counter)
        got=plane.result();history=np.asarray(history);f=frequency_samples(m.spectrum)
        times=(np.arange(r.steps)+1)*r.time_step;oracle=[]
        for c in range(6):
            t=times+(r.time_step/2 if c>=3 else 0);sel=slice(None,None,m.time_downsample)
            oracle.append(np.exp(2j*np.pi*f[:,None]*t[sel])@
                          (history[sel,c]*apodization_window(t[sel],m.spectrum))*r.time_step*m.time_downsample)
        oracle=np.array(oracle).T;scale=np.max(abs(oracle))
        tol=2e-6 if precision=='float32' else 2e-14
        names=['Ex','Ey','Ez','Hx','Hy','Hz']
        for i,c in enumerate(m.record_fields):
            np.testing.assert_allclose(got['fields'][...,i]/scale,
                np.broadcast_to(oracle[:,names.index(c),None]/scale,got['fields'][...,i].shape),rtol=tol,atol=tol)
        poynting=.5*np.cross(oracle[:,:3],oracle[:,3:].conj()).real
        for i,c in enumerate(m.record_poynting):
            expected=poynting[:, 'xyz'.index(c)]/scale**2
            np.testing.assert_allclose(got['poynting'][...,i]/scale**2,np.broadcast_to(expected[:,None],got['poynting'][...,i].shape),atol=tol,rtol=tol)
        if m.record_flux:
            area=np.prod([v*1e-6 for v in size if v]);assert got['weights'].sum()==pytest.approx(area,rel=1e-14)
            np.testing.assert_allclose(got['flux']/scale**2/area,poynting[:,'xyz'.index(m.normal)]/scale**2,rtol=tol,atol=tol)
        else:assert got['flux'] is None
        assert plane.value.shape[-1]==len(m.required_fields)
        assert plane.value.dtype in (np.complex128,torch.complex128)
        assert got['components']==list(m.record_fields)
    finally:
        fdtd.set_backend('numpy');fdtd.backend.float=np.float64;torch.set_default_dtype(previous)


def test_lobatto_and_local_apodization_with_dynamic_source_limits():
    s=Source(pulse='broadband',time_definition='wavelength',wavelength_start=1.2,wavelength_stop=1.9)
    local=SpectrumSettings(sampling='frequency',apodization='end')
    p=Project(sources=[s],monitors=[FieldMonitor(use_global_monitor=True,inherit_apodization=False,spectrum=local)],
        global_monitor=SpectrumSettings(sampling='chebyshev',chebyshev_nodes='lobatto',frequency_points=7,
                                       use_source_limits=True,apodization='full'))
    m=p.resolved_monitor(p.monitors[0]);f=frequency_samples(m.spectrum)
    assert m.spectrum.apodization=='end' and m.spectrum.wavelength_start==1.2
    np.testing.assert_allclose((f-f[0])/(f[-1]-f[0]),[0,(2-np.sqrt(3))/4,.25,.5,.75,(2+np.sqrt(3))/4,1],atol=1e-15)
    p.sources[0].wavelength_stop=2.1;p.global_monitor.frequency_points=3
    assert p.resolved_monitor(p.monitors[0]).spectrum.wavelength_stop==2.1
    assert len(frequency_samples(p.resolved_monitor(p.monitors[0]).spectrum))==3
    p.monitors[0].inherit_apodization=True
    assert p.resolved_monitor(p.monitors[0]).spectrum.apodization=='full'
    p.sources[0].time_definition='standard'
    with pytest.raises(ValueError,match='explicitly ranged'):p.resolved_monitor(p.monitors[0])
    with pytest.raises(ValueError,match='FFT bins'):SpectrumSettings(use_source_limits=True)
    with pytest.raises(ValueError,match='Nyquist'):Project(monitors=[FieldMonitor(time_downsample=100)])


def test_nearest_normal_and_independent_axis_strides():
    r=Region(dimension='3d',size=(2,2,2),mesh=.1,pml_cells=3)
    m=FieldMonitor(center=(.123,0,0),size=(0,.7,.9),downsample_xyz=(7,2,3),spatial_interpolation='nearest')
    plan=plane_plan(r,m)
    assert plan['shape']==(1,4,4)
    np.testing.assert_allclose(plan['points_um'][:,0],.1,atol=1e-15)
    assert sum(plan['weights'])==pytest.approx(.7*.9e-12)


def test_mixed_precision_batch_compact_results_roundtrip_facade_and_digest(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_tensor_batch import cases
    projects=cases(2,dimension='3d')
    for p in projects:
        p.region.cuda_monitor_kernel='fused'
        p.monitors=[FieldMonitor(id='flux',name='Flux',center=(.7,0,0),size=(0,1,1),
            record_fields=(),record_poynting=(),spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')),
            FieldMonitor(id='field',name='Field',center=(.7,0,0),size=(0,1,1),
                record_fields=('Hy','Ez'),record_poynting=(),record_flux=False,dft_precision='float64',time_downsample=2,
                spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none'))]
    ref=[Simulation(p).run(cuda_graph_steps=8) for p in projects]
    batch=run_tensor_batch(projects,cuda_graph_steps=8);batch.raise_for_errors()
    for item,expected in zip(batch.items,ref):
        actual=item.load();assert _result_digest(actual)==_result_digest(expected)
        assert len(actual.flux_data())==1
        actual.save(tmp_path/'selected.npz');loaded=Result.load(tmp_path/'selected.npz')
        assert _result_digest(loaded)==_result_digest(actual)
        assert loaded.frequency_fields[0]['fields'].shape[-1]==0 and loaded.frequency_fields[1]['flux'] is None
        a=loaded.frequency_fields[0];normalized=normalize_flux(a,a)
        assert normalized['valid'].any()
        np.testing.assert_allclose(normalized['ratio'][normalized['valid']],1)
        with pytest.raises(ValueError,match='stored tangential'):normalize_flux(a,a,subtract_incident=True)
        original=_result_digest(loaded);a['flux']*=2;assert _result_digest(loaded)!=original
    session=FDTD(projects[0]);session.result=ref[0]
    assert session.getresult('Field','E')['components']==['Ez']
    assert session.getresult('Field','H')['components']==['Hy']
    with pytest.raises(ValueError,match='No E components'):session.getresult('Flux','E')
    with pytest.raises(ValueError,match='Flux was not recorded'):session.getresult('Field','flux')
    session.switchtolayout();session.select('Field');session.set('downsample xyz',(1,2,3))
    session.set('record fields',('Hz',));session.set('dft precision','field');session.set('time downsample',3)
    session.setglobalmonitor('chebyshev nodes','lobatto')
    assert session.getglobalmonitor('chebyshev nodes')=='lobatto'
