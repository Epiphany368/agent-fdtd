"""Direct-gather private PMC CUDA foundation versus the exact CPU reference."""
import math
import numpy as np
import pytest
import torch

from torchfdtd.pmc_reference import EndpointTopology,EndpointReference,ReferenceState
from torchfdtd.pmc_cuda import CompactEndpointTopology,EndpointCUDA


CASES=[(('pec','pec'),)*3,(('pmc','pmc'),)*3,(('pec','pmc'),)*3,
       (('pmc','pec'),)*3,(('pec','pmc'),('pmc','pmc'),('pmc','pec'))]


@pytest.fixture(autouse=True)
def deterministic_random():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1731)
        yield


def topology(faces,n=4,nonuniform=False):
    nodes=[np.linspace(0,n,n+1) for _ in range(3)]
    if nonuniform:nodes=[n*(x/n)**(1+.1*a) for a,x in enumerate(nodes)]
    return EndpointTopology(nodes,faces)


def gpu():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


def copy_state(state,device):return ReferenceState(state.electric.to(device),state.magnetic.to(device))


def test_compact_topology_has_no_volume_metadata():
    t=CompactEndpointTopology([np.arange(257)]*3,CASES[1])
    assert t.counts['E']==3*256*257*257
    assert t.counts['H']==3*257*256*256
    assert t.state_bytes()==4*sum(t.counts.values())
    assert not any(hasattr(t,name) for name in ('indices','coordinates','lookup','active','operators'))
    assert sum(x.nbytes for x in t.nodes)==3*257*8
    assert len(t.blocks['E'])==10 and len(t.blocks['H'])==4


@pytest.mark.parametrize('faces',CASES)
@pytest.mark.parametrize('nonuniform',[False,True])
def test_direct_curls_and_transposes_match_cpu(faces,nonuniform):
    gpu();t=topology(faces,nonuniform=nonuniform)
    cpu=EndpointReference(t);cuda=EndpointCUDA(t)
    assert cuda.topology.blocks==t.blocks
    assert cuda.metadata_bytes==4*sum(2*n+1 for n in t.shape)
    for forward in (False,True):
        source,target=('E','H') if forward else ('H','E')
        value=torch.randn(len(t.components[source]),dtype=torch.float32)
        seed=torch.randn(len(t.components[target]),dtype=torch.float32)
        expected=cpu.curl(value,forward)
        actual=cuda.curl(value.cuda(),forward).cpu()
        transpose=cuda.curl(seed.cuda(),forward,transpose=True).cpu()
        torch.testing.assert_close(actual,expected,rtol=3e-6,atol=2e-6)
        torch.testing.assert_close(transpose,cpu.curl(seed,forward,transpose=True),rtol=3e-6,atol=2e-6)
        torch.testing.assert_close((actual*seed).sum(),(value*transpose).sum(),rtol=2e-5,atol=2e-5)
        # Random seeds include independently owned face/edge states, not volume only.
        for block in t.blocks[target][1:]:
            assert seed[block.start:block.stop].abs().max()>0


@pytest.mark.parametrize('faces',[CASES[1],CASES[-1]])
@torch.enable_grad()
def test_multistep_fields_state_adjoint_and_shared_material_gradient(faces):
    gpu();t=topology(faces,n=5,nonuniform=True)
    cpu=EndpointReference(t);cuda=EndpointCUDA(t)
    initial=ReferenceState(torch.randn(len(t.components['E']),dtype=torch.float32,requires_grad=True),
                           torch.randn(len(t.components['H']),dtype=torch.float32,requires_grad=True))
    parameter=(1.2+torch.rand((*t.shape,3),dtype=torch.float32)).requires_grad_()
    epsilon,index=t.shared_volume_epsilon(parameter)
    material=cuda.prepare_epsilon(epsilon.detach().cuda())
    actual=copy_state(ReferenceState(initial.electric.detach(),initial.magnetic.detach()),'cuda')
    expected=initial;primal=[];dt=.13
    for _ in range(9):
        primal.append(actual)
        expected=cpu.step(expected,epsilon,dt)
        actual=cuda.step(actual,material,dt)
    torch.testing.assert_close(actual.electric.cpu(),expected.electric,rtol=1e-5,atol=3e-6)
    torch.testing.assert_close(actual.magnetic.cpu(),expected.magnetic,rtol=1e-5,atol=3e-6)
    seed=ReferenceState(torch.randn_like(expected.electric),torch.randn_like(expected.magnetic))
    loss=(expected.electric*seed.electric).sum()+(expected.magnetic*seed.magnetic).sum()
    ge,gh,gp=torch.autograd.grad(loss,(initial.electric,initial.magnetic,parameter))
    bars=copy_state(seed,'cuda');gradient=torch.zeros_like(material.epsilon)
    for state in reversed(primal):
        bars,part=cuda.transpose_step(state,bars,material,dt)
        gradient+=part
    torch.testing.assert_close(bars.electric.cpu(),ge,rtol=1e-5,atol=4e-6)
    torch.testing.assert_close(bars.magnetic.cpu(),gh,rtol=1e-5,atol=4e-6)
    shared=torch.zeros(parameter.numel(),dtype=torch.float32).index_add(0,index,gradient.cpu()).reshape(parameter.shape)
    torch.testing.assert_close(shared,gp,rtol=2e-5,atol=5e-6)


def test_rejects_unsupported_precision_device_and_material():
    t=topology(CASES[1])
    with pytest.raises(ValueError,match='FP32'):EndpointCUDA(t,dtype=torch.float64)
    with pytest.raises(ValueError,match='CUDA device'):EndpointCUDA(t,device='cpu')
    gpu();cuda=EndpointCUDA(t);state=cuda.zeros()
    with pytest.raises(ValueError,match='FP32'):cuda.curl(state.electric.to(torch.complex64),True)
    with pytest.raises(ValueError,match='positive'):cuda.prepare_epsilon(torch.zeros_like(state.electric))
    with pytest.raises(ValueError,match='explicit adjoint'):cuda.curl(state.electric.requires_grad_(),True)
    state=cuda.zeros();material=cuda.prepare_epsilon(torch.ones_like(state.electric))
    with pytest.raises(ValueError,match='CFL'):cuda.step(state,material,2.)
    with pytest.raises(ValueError,match='Prepare epsilon'):cuda.step(state,torch.ones_like(state.electric),.1)


def test_nondefault_stream_lifetime_and_larger_direct_grid():
    gpu()
    t=CompactEndpointTopology([np.arange(129)]*3,CASES[1])
    cuda=EndpointCUDA(t)
    assert cuda.metadata_bytes==4*3*257
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        state=cuda.zeros()
        state.electric.normal_();state.magnetic.normal_()
        material=cuda.prepare_epsilon(torch.ones_like(state.electric))
        original=state
        state=cuda.step(state,material,.2)
        seed=ReferenceState(torch.ones_like(state.electric),torch.ones_like(state.magnetic))
        bars,gradient=cuda.transpose_step(original,seed,material,.2)
    stream.synchronize()
    assert bool(torch.isfinite(state.electric).all() & torch.isfinite(bars.magnetic).all() & torch.isfinite(gradient).all())
    assert state.electric.numel()==t.counts['E']
    lhs=(state.electric*seed.electric).sum()+(state.magnetic*seed.magnetic).sum()
    rhs=(original.electric*bars.electric).sum()+(original.magnetic*bars.magnetic).sum()
    torch.testing.assert_close(lhs,rhs,rtol=1e-4,atol=.02)


def test_prepared_material_rejects_inplace_change_and_nodes_are_readonly():
    gpu();t=topology(CASES[1]);cuda=EndpointCUDA(t)
    with pytest.raises(ValueError,match='read-only'):
        cuda.topology.nodes[0][0]=-.1
    state=cuda.zeros();eps=torch.ones_like(state.electric)
    material=cuda.prepare_epsilon(eps)
    eps.add_(.2)
    with pytest.raises(ValueError,match='changed after preparation'):
        cuda.step(state,material,.1)
    with pytest.raises(ValueError,match='changed after preparation'):
        cuda.transpose_step(state,state,material,.1)
    material=cuda.prepare_epsilon(eps)
    result=cuda.step(state,material,.1)
    assert result.electric.count_nonzero()==0
