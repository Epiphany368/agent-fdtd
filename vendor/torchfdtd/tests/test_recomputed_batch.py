import weakref
import pytest
import torch
from torchfdtd import recompute_cases, DifferentiableSimulation, AdjointOptions
from test_bloch_adjoint import scene


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('complex_output', [False, True])
def test_coupled_objective_multiple_parameters(device, complex_output):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    x = torch.tensor([.2, .7], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor(.3, dtype=x.dtype, device=device, requires_grad=True)
    unused = torch.tensor(.9, dtype=x.dtype, device=device, requires_grad=True)
    fixed = torch.tensor(.4, dtype=x.dtype, device=device)
    cases = [lambda a,b,c,d,k=k: torch.sin(a*k+b)+d+(1j*a.square() if complex_output else 0) for k in (1,2,3)]
    def loss(v): return (v.sum(0).abs().square().sum() + (v[0]*v[2].conj()).real.sum())
    expected = torch.stack([f(x,y,unused,fixed) for f in cases]).cpu()
    eg = torch.autograd.grad(loss(expected), (x,y,unused), allow_unused=True)
    actual = recompute_cases(cases,x,y,unused,fixed)
    ag = torch.autograd.grad(loss(actual), (x,y,unused), allow_unused=True)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(ag[:2],eg[:2]): torch.testing.assert_close(a,e)
    assert ag[2] is eg[2] is None


def test_only_one_graph_alive_and_no_forward_graphs():
    live = []
    class Saved(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            assert not any(ref() is not None for ref in live)
            state = torch.ones(100)
            live.append(weakref.ref(state))
            ctx.save_for_backward(state)
            return x.square()
        @staticmethod
        def backward(ctx,g):
            assert ctx.saved_tensors[0].numel() == 100
            return g*2
    x=torch.tensor(1.,requires_grad=True)
    v=recompute_cases([lambda x: Saved.apply(x)]*4,x)
    assert not any(ref() is not None for ref in live)
    v.sum().backward()
    assert not any(ref() is not None for ref in live)
    assert x.grad == 8


@pytest.mark.parametrize('storage', ['device','disk'])
def test_bloch_fdtd_coupled_case_gradients(storage,tmp_path):
    cases=[]
    for phase in (.2,.6,.9):
        p=scene();p.region.bloch_phase=(phase,0,0)
        model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,storage=storage,
            checkpoint_directory=tmp_path,disk_budget_bytes=32*1024**2))
        def case(x,model=model):
            eps=torch.ones(model.project.region.shape,dtype=x.dtype)*(1.4+x)
            return model.spectrum(eps,[.03/model.project.region.time_step],block_size=5).fields.flatten()/model.project.region.time_step
        cases.append(case)
    x=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    def loss(v): return v.sum(0).abs().square().sum()+v.real.square().mean()
    expected=torch.stack([f(x) for f in cases]);eg,=torch.autograd.grad(loss(expected),x)
    actual=recompute_cases(cases,x);ag,=torch.autograd.grad(loss(actual),x)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(ag,eg,rtol=1e-10,atol=1e-12)
    assert ag.abs()>1e-5
    assert list(tmp_path.iterdir())==[]


def test_drift_and_first_order_guards():
    x=torch.tensor(.3,requires_grad=True)
    scale=[1.]
    y=recompute_cases([lambda x: x*scale[0]],x)
    scale[0]=2.
    with pytest.raises(RuntimeError,match='drifted'): y.sum().backward()
    y=recompute_cases([lambda x: x.square()],x)
    with pytest.raises(RuntimeError,match='first-order'): torch.autograd.grad(y.sum(),x,create_graph=True)
    y=recompute_cases([lambda x: x.square()],x)
    with torch.no_grad(): x.add_(1)
    with pytest.raises(RuntimeError,match='modified'): y.sum().backward()


def test_output_contract_budget_and_constant_cases():
    x=torch.tensor(.2,requires_grad=True)
    with pytest.raises(ValueError,match='budget exceeded'): recompute_cases([lambda x: x.expand(10)],x,output_budget_bytes=4)
    with pytest.raises(ValueError,match='identical'): recompute_cases([lambda x:x,lambda x:x[None]],x)
    with pytest.raises(ValueError,match='finite'): recompute_cases([lambda x:x*float('nan')],x)
    y=recompute_cases([lambda x:torch.ones(2)],x)
    assert torch.autograd.grad(y.sum(),x,allow_unused=True)==(None,)


def test_gaussian_information_coupled_cases():
    from torchfdtd import gaussian_target_information,shot_read_covariance
    x=torch.tensor([.2,.4],dtype=torch.float64,requires_grad=True)
    cases=[lambda x,k=k: (x+k).square() for k in (.1,.5,1.)]
    eye=torch.eye(2,dtype=x.dtype)
    def objective(a):
        return gaussian_target_information(a,eye,.7*eye,eye,shot_read_covariance(a.sum(-1),1.5)).information_bits
    expected=torch.stack([f(x) for f in cases]);eg,=torch.autograd.grad(objective(expected),x)
    actual=recompute_cases(cases,x);ag,=torch.autograd.grad(objective(actual),x)
    torch.testing.assert_close(ag,eg,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('storage',['host','disk'])
def test_streamed_cases(storage,tmp_path):
    from torchfdtd import StreamedSimulation,StreamedAdjointOptions
    from test_differentiable import project
    cases=[]
    for wavelength in (1.,1.3):
        p=project(steps=10);p.sources[0].wavelength=wavelength
        model=StreamedSimulation(p,StreamedAdjointOptions(device='cpu',slab_width=6,temporal_depth=2,
            state_storage=storage,state_directory=tmp_path,disk_budget_bytes=64*1024**2))
        cases.append(lambda x,model=model: model.spectrum(torch.ones(model.project.region.shape,dtype=x.dtype)*(1.5+x),
            [.03/model.project.region.time_step]).fields.flatten()/model.project.region.time_step)
    x=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    def loss(v):return v.sum(0).abs().square().sum()
    expected=torch.stack([f(x) for f in cases]);eg,=torch.autograd.grad(loss(expected),x)
    actual=recompute_cases(cases,x);ag,=torch.autograd.grad(loss(actual),x)
    torch.testing.assert_close(ag,eg,rtol=1e-10,atol=1e-12)
    assert list(tmp_path.iterdir())==[]
