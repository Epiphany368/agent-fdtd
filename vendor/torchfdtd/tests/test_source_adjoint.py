import pytest
import torch
from torchfdtd import Project,Region,Source,Monitor,FieldMonitor,BoundaryFace,AdjointOptions,DifferentiableSimulation
from torchfdtd.source_adjoint import SourceWaveformSimulation,SourceWaveformPlaneSimulation


def scene(complex_fields=False):
    r=Region(dimension="3d",size=(1.5,1.5,1.5),mesh=.1,pml_cells=3,steps=11,precision='float32',backend='cpu')
    if complex_fields:
        r.boundaries.x_min=r.boundaries.x_max=BoundaryFace(kind='bloch')
        r.bloch_phase=(.2,0,0)
    return Project(region=r,sources=[Source(component='Ex',center=(0,0,0),pulse='continuous'),
        Source(component='Hy',center=(0,0,0),pulse='continuous'),
        Source(component='Ex',center=(0,0,0),pulse='continuous')],
        monitors=[Monitor(component='Ex',center=(0,0,0)),Monitor(component='Hy',center=(0,0,0))])


@pytest.mark.parametrize('complex_fields,complex_wave,diagonal',[(False,False,False),(True,False,True),(True,True,False)])
def test_material_multiple_e_h_waveforms_reference_and_retained_seeds(complex_fields,complex_wave,diagonal):
    torch.set_num_threads(1)
    p=scene(complex_fields)
    model=SourceWaveformSimulation(p,AdjointOptions(checkpoints=2))
    epsilon=torch.full(p.region.shape+((3,) if diagonal else ()),1.6,requires_grad=True)
    dtype=torch.complex64 if complex_wave else torch.float32
    torch.manual_seed(82)
    waves=(torch.randn(p.region.steps,3,dtype=dtype)*.1).requires_grad_()
    expected=model.reference(epsilon,waves)
    actual=model(epsilon,waves)
    torch.testing.assert_close(actual.signals,expected,rtol=3e-5,atol=2e-7)
    for _ in range(2):
        seed=torch.randn(2,p.region.steps,dtype=expected.dtype).T
        assert not seed.is_contiguous()
        want=torch.autograd.grad(expected,(epsilon,waves),seed,retain_graph=True)
        got=torch.autograd.grad(actual.signals,(epsilon,waves),seed,retain_graph=True)
        for a,b in zip(got,want):torch.testing.assert_close(a,b,rtol=8e-5,atol=8e-7)
    assert got[1][:,0].norm()>0 and got[1][:,1].norm()>0
    torch.testing.assert_close(got[1][:,0],got[1][:,2])


def test_defaults_native_equality_snapshot_and_admission(monkeypatch):
    p=scene()
    model=SourceWaveformSimulation(p)
    epsilon=torch.full(p.region.shape,1.4,requires_grad=True)
    waves=model.default_waveforms().requires_grad_()
    torch.testing.assert_close(model(epsilon,waves).signals,DifferentiableSimulation(p)(epsilon).signals)
    actual=model(epsilon,waves).signals
    seed=torch.ones_like(actual)
    expected=torch.autograd.grad(actual,(epsilon,waves),seed,retain_graph=True)
    with torch.no_grad():epsilon.add_(.2);waves.add_(.3)
    got=torch.autograd.grad(actual,(epsilon,waves),seed)
    for a,b in zip(got,expected):torch.testing.assert_close(a,b)
    p.region.memory_mode="budgeted"
    bad=SourceWaveformSimulation(p,AdjointOptions(resident_budget_bytes=1))
    def forbidden(*args,**kwargs):raise AssertionError('packing before admission')
    monkeypatch.setattr(torch,'cat',forbidden)
    with pytest.raises((ValueError,MemoryError)):bad(epsilon,waves)
    model.project.sources[0].amplitude=2
    with pytest.raises(ValueError,match='configuration'):model(epsilon,waves)


def test_online_spectrum_matches_time_history_vjp():
    p=scene(True);model=SourceWaveformSimulation(p)
    epsilon=torch.full(p.region.shape,1.4,requires_grad=True)
    wave=(model.default_waveforms()+.1j*model.default_waveforms()).requires_grad_()
    frequency=torch.tensor([.03,.07])/p.region.time_step
    from torchfdtd.adjoint_spectrum import SpectralObservation
    spectral=SpectralObservation(epsilon,p.region,['Ex','Hy'],frequency,None,4)
    history=model.reference(epsilon,wave)
    want=spectral.zeros()
    for start in range(0,p.region.steps,4):spectral.accumulate(want,history[start:start+4],start)
    result=model.spectrum(epsilon,wave,frequency,block_size=4)
    torch.testing.assert_close(result.fields/p.region.time_step,want/p.region.time_step,rtol=3e-5,atol=2e-6)
    seed=torch.randn_like(want)/p.region.time_step
    a=torch.autograd.grad(result.fields,(epsilon,wave),seed)
    b=torch.autograd.grad(want,(epsilon,wave),seed)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=2e-6)


def test_plane_six_fields_and_waveform_gradient():
    p=scene();p.monitors=[FieldMonitor(normal='z',center=(0,0,.1),size=(.2,.2,0),downsample=1)]
    model=SourceWaveformPlaneSimulation(p,AdjointOptions(checkpoints=1))
    epsilon=torch.full(p.region.shape+(3,),1.4,requires_grad=True)
    wave=model.default_waveforms().requires_grad_();frequency=torch.tensor([.03])/p.region.time_step
    result=model(epsilon,wave,frequency,block_size=4)
    field=next(iter(result.values())).fields
    assert field.shape[-1]==6
    grad=torch.autograd.grad((field/p.region.time_step).abs().square().sum(),(epsilon,wave))
    assert all(torch.isfinite(x).all() and x.norm()>0 for x in grad)



def test_invalid_inputs_and_unsupported_source_before_pack(monkeypatch):
    p=scene();model=SourceWaveformSimulation(p)
    eps=torch.ones(p.region.shape);waves=model.default_waveforms()
    def forbidden(*args,**kwargs):raise AssertionError("unexpected packing")
    monkeypatch.setattr(torch,"cat",forbidden)
    for bad in (waves[:-1],waves.double(),waves.to(torch.complex64),waves*float("nan")):
        with pytest.raises(ValueError):model(eps,bad)
    p.sources[0].injection="oneway"
    with pytest.raises(ValueError):SourceWaveformSimulation(p)


def test_plane_waveform_identity_and_full_autograd_oracle():
    p=scene(True)
    p.sources[0].kind="plane";p.sources[0].normal="z";p.sources[0].size=(.4,.4,0)
    p.monitors=[FieldMonitor(normal="z",center=(0,0,.1),size=(.2,.2,0),downsample=1)]
    model=SourceWaveformPlaneSimulation(p)
    eps=torch.full(p.region.shape,1.4,requires_grad=True)
    wave=(model.default_waveforms()+.02j).requires_grad_()
    freq=torch.tensor([.03])/p.region.time_step
    actual=model(eps,wave,freq,block_size=4)
    spectral=model._spectral(eps,freq,4)
    from torchfdtd.source_adjoint import _SourceSystem
    from torchfdtd.differentiable import DifferentiableResult
    def reference(spec):
        carrier,_=model.model._pack(eps,wave,spec)
        own=carrier[:eps.numel()].view(eps.shape)
        system=_SourceSystem(model.model.project,own,carrier=carrier,wave_shape=tuple(wave.shape),
            wave_complex=True,layout=model.term_layout,observation_monitors=spec.observers)
        state=tuple(torch.zeros_like(x) for x in system.state());history=[]
        for n in range(p.region.steps):
            state=system.reference_step(state,n,own);history.append(system.observe(state))
        history=torch.stack(history);out=spec.zeros()
        for n in range(0,p.region.steps,4):spec.accumulate(out,history[n:n+4],n)
        return spec.result(out,{})
    expected=model._planes(eps,freq,4,reference)
    a=next(iter(actual.values()));b=next(iter(expected.values()))
    torch.testing.assert_close(a.fields/p.region.time_step,b.fields/p.region.time_step,rtol=5e-5,atol=3e-6)
    seed=torch.randn_like(a.fields)/p.region.time_step
    ga=torch.autograd.grad(a.fields,(eps,wave),seed)
    gb=torch.autograd.grad(b.fields,(eps,wave),seed)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-4,atol=3e-6)
    changed=next(iter(model(eps,wave*2,freq,block_size=4).values()))
    with pytest.raises(ValueError):a.normalized_flux(changed)


def test_preparation_helpers_admit_before_numpy_table(monkeypatch):
    p=scene();p.region.memory_mode="budgeted"
    model=SourceWaveformSimulation(p,AdjointOptions(resident_budget_bytes=1))
    import numpy as np
    original=np.arange
    def forbidden(*args,**kwargs):
        if args[:2]==(1,p.region.steps+1):raise AssertionError("table allocation before admission")
        return original(*args,**kwargs)
    monkeypatch.setattr(np,"arange",forbidden)
    for prepare in (model.source_times,model.default_waveforms):
        with pytest.raises(ValueError,match="budget"):prepare()


def test_vector_source_columns_match_native_weights_and_material_gradient():
    p=scene()
    p.sources[0].theta=57.;p.sources[0].phi=31.
    p.sources[1].theta=64.;p.sources[1].phi=117.
    p.sources[2].enabled=False
    model=SourceWaveformSimulation(p,AdjointOptions(checkpoints=1))
    assert [t['component'] for t in model.term_layout]==['Ex','Ey','Ez','Hx','Hy','Hz']
    times=model.source_times()
    assert times.shape==(p.region.steps,6)
    torch.testing.assert_close(times[:,3:]-times[:,:3],
        torch.full_like(times[:,:3],.5*p.region.time_step),rtol=4e-6,atol=0)
    eps=torch.full(p.region.shape,1.5,requires_grad=True)
    waves=model.default_waveforms().requires_grad_()
    actual=model(eps,waves).signals
    native=DifferentiableSimulation(p)(eps).signals
    torch.testing.assert_close(actual,native,rtol=0,atol=0)
    a=torch.autograd.grad(actual.square().sum(),(eps,waves))
    b,=torch.autograd.grad(native.square().sum(),eps)
    torch.testing.assert_close(a[0],b,rtol=8e-5,atol=1e-7)
    assert all(a[1][:,i].norm()>0 for i in range(6))
