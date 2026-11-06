import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions, BoundaryFace, Material, Simulation, Structure
from torchfdtd import DispersiveSimulation, DispersivePlaneSimulation
from test_differentiable import project


@pytest.mark.parametrize('dimension,bloch,diagonal', [('2d', False, False), ('3d', False, True), ('2d', True, True)])
@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_ade_explicit_transpose_matches_full_time_graph(dimension, bloch, diagonal, device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p = project(dimension=dimension, steps=15)
    if bloch:
        p.region.boundaries.x_min = BoundaryFace(kind='bloch')
        p.region.boundaries.x_max = BoundaryFace(kind='bloch')
        p.region.bloch_phase = (.41, 0, 0)
    torch.manual_seed(54)
    shape = p.region.shape+((3,) if diagonal else ())
    epsilon = (1.5+.1*torch.rand(shape, dtype=torch.float64,device=device)).requires_grad_()
    strength = torch.tensor([.8, .4], dtype=torch.float64,device=device,requires_grad=True)
    omega = torch.tensor([0., 1.7], dtype=torch.float64,device=device,requires_grad=True)
    gamma = torch.tensor([.2, .3], dtype=torch.float64,device=device,requires_grad=True)
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
    args = lambda: (epsilon, strength*1e30, omega*1e15, gamma*1e15)
    reference = model.reference(*args())
    weights = torch.randn(reference.shape, dtype=reference.dtype,device=device)
    loss = lambda signals: (signals.conj()*weights).real.sum() + signals.abs().square().sum()
    expected = torch.autograd.grad(loss(reference), (epsilon, strength, omega, gamma))
    actual = model(*args())
    got = torch.autograd.grad(loss(actual.signals), (epsilon, strength, omega, gamma))
    torch.testing.assert_close(actual.signals, reference, rtol=1e-12, atol=1e-13)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=1e-12)
    assert all(torch.linalg.vector_norm(g)>1e-8 for g in got)
    assert actual.report['material_state_bytes'] == 12*np.prod(p.region.shape)*8*(2 if bloch else 1)
    assert actual.report['peak_checkpoints'] <= 2
    assert not actual.report['full_time_autograd']


@pytest.mark.parametrize('kind', ['drude', 'lorentz', 'multipole'])
def test_matches_existing_native_ade_forward(kind):
    p = project(steps=45)
    from torchfdtd import LorentzPole
    material = Material(name='dispersion', model=kind, epsilon_inf=1.8,
        plasma_rad_s=1.2e15, collision_rad_s=2e14,
        resonance_rad_s=1.7e15, linewidth_rad_s=3e14, delta_epsilon=.8,
        poles=[LorentzPole(resonance_rad_s=1.3e15, damping_rad_s=2e14, strength_rad_s_squared=.7e30),
               LorentzPole(resonance_rad_s=1.9e15, damping_rad_s=4e14, strength_rad_s_squared=.5e30)])
    p.materials.append(material)
    p.structures = [Structure(material=material.name, size=(10,10,10))]
    epsilon = torch.full(p.region.shape, material.epsilon_inf, dtype=torch.float64)
    rates = np.array(material.oscillators)
    result = DispersiveSimulation(p)(epsilon, rates[:,1], rates[:,0], rates[:,2])
    expected = Simulation(p).run(cuda_graph=False)
    np.testing.assert_allclose(result.signals.numpy(), expected.signals, rtol=2e-12, atol=1e-13)


@pytest.mark.parametrize('storage,slots', [('device',0), ('host',1), ('disk',3)])
def test_material_states_survive_checkpoint_replay(tmp_path, storage, slots):
    p = project(steps=12)
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    strength = torch.full((1,*p.region.shape), .8, dtype=torch.float64, requires_grad=True)
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=slots, storage=storage,
        checkpoint_directory=tmp_path, disk_budget_bytes=32*1024**2, host_budget_bytes=32*1024**2))
    reference = model.reference(eps, strength*1e30, 1.4e15, 2e14)
    expected = torch.autograd.grad(reference.square().sum(), (eps, strength))
    actual = model(eps, strength*1e30, 1.4e15, 2e14)
    got = torch.autograd.grad(actual.signals.square().sum(), (eps, strength))
    for a,b in zip(got,expected):
        torch.testing.assert_close(a,b,rtol=2e-11,atol=1e-13)
    assert list(tmp_path.iterdir()) == []


def test_geometry_to_ade_taylor_and_online_spectrum():
    p = project(steps=22)
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
    x = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    mask = torch.linspace(-1,1,np.prod(p.region.shape), dtype=torch.float64).reshape(p.region.shape)
    def args(value):
        density = torch.sigmoid(3*(mask+value))
        return (1.2+.4*density, (density*.8+0.1)[None]*1e30, 1.5e15, 2e14)
    def loss(value):
        return model(*args(value)).signals.square().sum()
    baseline = loss(x)
    gradient, = torch.autograd.grad(baseline,x)
    errors = [float((loss(x.detach()+h)-baseline.detach()-h*gradient).abs()) for h in (.002,.001,.0005)]
    assert all(3.8<a/b<4.2 for a,b in zip(errors,errors[1:]))
    f = [1.5e14, 2e14]
    online = model.spectrum(*args(x),f,block_size=5)
    reference = model(*args(x)).spectrum(f)
    torch.testing.assert_close(online.fields, reference, rtol=1e-12, atol=1e-28)
    a, = torch.autograd.grad((online.fields.abs().square()/p.region.time_step**2).sum(), x)
    b, = torch.autograd.grad((reference.abs().square()/p.region.time_step**2).sum(), x)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-13)


def test_invalid_parameters_and_options():
    p = project(steps=10)
    eps = torch.full(p.region.shape, 1.5, dtype=torch.float64)
    with pytest.raises(ValueError,match='requires a CUDA tensor'):
        DispersiveSimulation(p,AdjointOptions(backward_kernel='fused'))(eps,[1e30],1e15,1e14)
    from torchfdtd import StreamedAdjointOptions
    with pytest.raises(ValueError,match='resident AdjointOptions'):
        DispersiveSimulation(p,StreamedAdjointOptions())
    model = DispersiveSimulation(p)
    for strength in ([-1.], [float('nan')], [1+1j]):
        with pytest.raises(ValueError):
            model(eps,strength,1e15,1e14)
    with pytest.raises(ValueError,match='shape'):
        model(eps,[1e30], [1e15,2e15], 1e14)
    with pytest.raises(ValueError,match='Host checkpoint budget'):
        DispersiveSimulation(p,AdjointOptions(storage='host',host_budget_bytes=1))(eps,[1e30],1e15,1e14)
    p.region.memory_mode='streamed'
    with pytest.raises(ValueError,match='StreamedSimulation'):
        DispersiveSimulation(p)(eps,[1e30],1e15,1e14)


def test_reference_size_guard_precedes_parameter_packing(monkeypatch):
    # The oracle is admitted by its graph memory estimate, before packing.
    p=project(steps=10)
    eps=torch.full(p.region.shape,1.5,dtype=torch.float64)
    model=DispersiveSimulation(p)
    model.project.region.steps=10**7
    monkeypatch.setattr(torch,'cat',lambda *a,**k: pytest.fail('Large oracle allocated packed parameters'))
    with pytest.raises(ValueError,match='estimated .* bytes of retained graph'):
        model.reference(eps,[1e30],1e15,2e14)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cpu_cuda_precision_and_parameter_directions(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = project(precision='float32' if dtype == torch.float32 else 'float64', steps=18)
    model = DispersiveSimulation(p, AdjointOptions(checkpoints=2,storage='host'))
    parameters = torch.tensor([1.6,.8,1.7,.2],dtype=dtype,device=device,requires_grad=True)
    def loss(values, reference=False):
        eps = values[0].expand(p.region.shape)
        args = eps,values[1:2]*1e30,values[2]*1e15,values[3]*1e15
        signals = model.reference(*args) if reference else model(*args).signals
        return signals.square().sum()
    got, = torch.autograd.grad(loss(parameters),parameters)
    expected, = torch.autograd.grad(loss(parameters,True),parameters)
    tolerance = 5e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(got,expected,rtol=tolerance,atol=tolerance*1e-3)
    if dtype == torch.float64:
        direction = parameters.new_tensor([.2,-.3,.4,.1]); h = 1e-4
        fd = (loss(parameters.detach()+h*direction)-loss(parameters.detach()-h*direction))/(2*h)
        torch.testing.assert_close(got@direction,fd,rtol=1e-6,atol=1e-9)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_plane_flux_material_gradient_and_zero_pole_limit(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from torchfdtd import DifferentiablePlaneSimulation
    from test_adjoint_planes import scene
    p = scene(); p.region.steps = 24
    eps = torch.full(p.region.shape,1.7,dtype=torch.float64,device=device)
    options = AdjointOptions(checkpoints=2)
    model = DispersivePlaneSimulation(p,options)
    frequency = [.025/p.region.time_step,.06/p.region.time_step]
    zero = model(eps,[0.],1.5e15,2e14,frequency)
    nondispersive = DifferentiablePlaneSimulation(p,options)(eps,frequency)
    for key in zero:
        torch.testing.assert_close(zero[key].fields,nondispersive[key].fields,rtol=2e-12,atol=1e-27)
    strength = torch.tensor(.8,dtype=torch.float64,device=device,requires_grad=True)
    def loss(value):
        planes = model(eps,value[None]*1e30,1.5e15,2e14,frequency,block_size=5)
        return sum(plane.normalized_flux(zero[key]).sum() for key,plane in planes.items())
    gradient, = torch.autograd.grad(loss(strength),strength)
    h=1e-4
    finite=(loss(strength.detach()+h)-loss(strength.detach()-h))/(2*h)
    assert gradient.abs()>1e-5
    torch.testing.assert_close(gradient,finite,rtol=2e-6,atol=1e-8)


def test_ade_replay_releases_system_without_cyclic_gc():
    import gc
    import weakref
    p=project(steps=12)
    model=DispersiveSimulation(p,AdjointOptions(checkpoints=2))
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,requires_grad=True)
    def iteration():
        result=model(eps,[1e30],1.5e15,2e14)
        ref=weakref.ref(result.signals.grad_fn.system)
        torch.autograd.grad(result.signals.square().sum(),eps)
        return ref
    enabled=gc.isenabled();gc.disable()
    try:
        ref=iteration()
        assert ref() is None
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('layout', ['shared', 'spatial_strength', 'spatial_rates', 'diagonal'])
@pytest.mark.parametrize('bloch', [False, True])
def test_compact_parameters_match_expanded_values_and_vjp(layout, bloch):
    p=project(steps=14)
    if bloch:
        p.region.boundaries.x_min=BoundaryFace(kind='bloch')
        p.region.boundaries.x_max=BoundaryFace(kind='bloch')
        p.region.bloch_phase=(.37,0,0)
    shape=p.region.shape
    generator=torch.Generator().manual_seed(917)
    parameter_shapes={
        'shared': [(2,), (), (2,)],
        'spatial_strength': [(2,*shape), (2,), ()],
        'spatial_rates': [(2,), (2,*shape), (2,*shape,3)],
        'diagonal': [(2,*shape,3), (2,*shape,3), ()],
    }[layout]
    epsilon=(1.4+.2*torch.rand(shape+((3,) if layout=='diagonal' else ()),generator=generator,dtype=torch.float64)).requires_grad_()
    variables=[(.2+torch.rand(s,generator=generator,dtype=torch.float64)).requires_grad_() for s in parameter_shapes]
    def inputs(expanded):
        values=[variables[0]*1e30,variables[1]*1e15,variables[2]*1e15]
        if not expanded:return epsilon,*values
        def expand(v):
            if v.ndim==0:v=v.reshape(1,1,1,1,1)
            elif v.ndim==1:v=v.reshape(2,1,1,1,1)
            elif v.ndim==4:v=v[...,None]
            return v.expand(2,*shape,3)
        eps=epsilon[...,None].expand(*shape,3) if epsilon.ndim==3 else epsilon
        return eps,*(expand(v) for v in values)
    model=DispersiveSimulation(p,AdjointOptions(checkpoints=2))
    compact=model(*inputs(False)); expanded=model(*inputs(True))
    loss=lambda result:result.signals.abs().square().sum()
    a=torch.autograd.grad(loss(compact),(epsilon,*variables))
    b=torch.autograd.grad(loss(expanded),(epsilon,*variables))
    torch.testing.assert_close(compact.signals,expanded.signals,rtol=1e-12,atol=1e-14)
    for want,got in zip(b,a):torch.testing.assert_close(got,want,rtol=2e-10,atol=1e-12)
    expected_bytes=(epsilon.numel()+sum(v.numel() for v in variables))*8
    assert compact.report['material_parameter_bytes']==expected_bytes
    assert expected_bytes < expanded.report['material_parameter_bytes']


def test_uniform_poles_do_not_allocate_dense_parameter_copies():
    p=project(steps=10)
    epsilon=torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    result=DispersiveSimulation(p)(epsilon,[.7e30,.4e30],[1.5e15,2e15],2e14)
    system=result.signals.grad_fn.system
    assert system.grid.inverse_permittivity is None
    assert system.parameters.numel()==epsilon.numel()+5
    eps,a,d,k=system.coefficients(system.parameters)
    assert eps.numel()==epsilon.numel()
    assert [value.numel() for value in (a,d,k)]==[2,2,2]
    gradient,=torch.autograd.grad(result.signals.square().sum(),epsilon)
    assert torch.isfinite(gradient).all() and gradient.norm()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_ade_async_hierarchical_cuda_restores_polarization_and_current(tmp_path):
    p=project(steps=14)
    p.region.boundaries.x_min=BoundaryFace(kind='bloch')
    p.region.boundaries.x_max=BoundaryFace(kind='bloch')
    p.region.bloch_phase=(.31,0,0)
    eps=torch.full(p.region.shape,1.5,dtype=torch.float64,device='cuda',requires_grad=True)
    strength=torch.tensor([.5,.7],dtype=torch.float64,device='cuda',requires_grad=True)
    model=DispersiveSimulation(p,AdjointOptions(checkpoints=3,storage='hierarchical',
        device_checkpoints=1,host_checkpoints=1,checkpoint_transfers='async',
        checkpoint_directory=tmp_path,disk_budget_bytes=32*1024**2,host_budget_bytes=32*1024**2))
    expected=model.reference(eps,strength*1e30,[0.,1.8e15],2e14)
    want=torch.autograd.grad(expected.abs().square().sum(),(eps,strength))
    result=model(eps,strength*1e30,[0.,1.8e15],2e14)
    got=torch.autograd.grad(result.signals.abs().square().sum(),(eps,strength))
    for a,b in zip(got,want):torch.testing.assert_close(a,b,rtol=3e-10,atol=1e-12)
    assert result.report['checkpoint_tiers']==['disk','host','device']
    assert result.report['peak_checkpoints']==3
    assert not list(tmp_path.iterdir())
