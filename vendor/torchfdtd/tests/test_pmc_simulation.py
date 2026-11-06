"""Resident endpoint traces against independent full-graph reference/replay."""
import numpy as np
import pytest
import torch

from torchfdtd.pmc_simulation import EndpointSimulation,C_UM_S
from torchfdtd.pmc_reference import EndpointTopology,EndpointReference,ReferenceState


def make(device='cpu',checkpoints=2,faces=None):
    return EndpointSimulation([np.array([0.,.2,.55,1.])]*3,
        faces or [('pmc','pmc'),('pec','pmc'),('pmc','pec')],dt_seconds=.04/C_UM_S,
        sources=[(2,(3,3,1)),(0,(1,1,1))],
        observations=[('E',2,(3,3,1)),('H',0,(3,1,1)),('E',0,(1,1,1))],
        device=device,checkpoints=checkpoints)


def oracle(sim,epsilon,drive):
    topology=EndpointTopology(sim.topology.nodes,sim.topology.faces)
    backend=EndpointReference(topology);state=topology.zeros();trace=[]
    for row in drive:
        state=backend.step(state,epsilon,sim.dt)
        injected=torch.zeros_like(state.electric).index_add(0,sim.source_tensor.cpu(),row)
        state=ReferenceState(state.electric+injected,state.magnetic-sim.dt*backend.curl(injected,True))
        trace.append(torch.stack([(state.electric if f=='E' else state.magnetic)[i] for f,i in sim.observation_ids]))
    return torch.stack(trace)


@pytest.mark.parametrize('checkpoints',[0,1,4,64])
def test_checkpoint_trace_and_exact_endpoint_parameter_gradient(checkpoints):
    torch.manual_seed(42);sim=make(checkpoints=checkpoints)
    parameter=torch.tensor(.3,requires_grad=True)
    epsilon=sim.sample_epsilon(lambda xyz,c:2+parameter*(xyz[:,0]+2*xyz[:,1])+c*.1,chunk_size=17)
    drive=torch.randn(13,2,requires_grad=True);seed=torch.randn(13,3)
    actual=sim(epsilon,drive);expected=oracle(sim,epsilon,drive)
    ag=torch.autograd.grad((actual*seed).sum(),(parameter,drive),retain_graph=True)
    eg=torch.autograd.grad((expected*seed).sum(),(parameter,drive))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,atol=5e-6,rtol=5e-5)
    assert sim.memory_plan(13)['checkpoint_states']==min(checkpoints,12)
    assert actual.grad_fn.__class__.__name__=='_EndpointTraceBackward'


def test_finite_difference_and_saved_tensors_do_not_retain_time_graph():
    sim=make(checkpoints=2);epsilon=torch.full((sim.topology.counts['E'],),2.,requires_grad=True)
    drive=torch.zeros(11,2);drive[0]=torch.tensor([1.,.4]);drive.requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
        trace=sim(epsilon,drive)
    assert len(saved)==2 and saved[0] is epsilon and saved[1] is drive
    grad=torch.autograd.grad(trace.square().sum(),epsilon)[0]
    direction=torch.zeros_like(epsilon);direction[sim.source_ids[0]]=1
    h=.002
    finite=(sim(epsilon.detach()+h*direction,drive.detach()).square().sum()-sim(epsilon.detach()-h*direction,drive.detach()).square().sum())/(2*h)
    torch.testing.assert_close(grad@direction,finite,atol=.003,rtol=.01)


def test_admission_and_exact_endpoint_coordinates():
    sim=make();values=sim.sample_epsilon(lambda xyz,c:1+xyz[:,0]+xyz[:,1]+xyz[:,2])
    index=sim.dof('E',2,(3,3,1));assert values[index].item()==pytest.approx(3.375)
    with pytest.raises(ValueError,match='constrained'):sim.dof('E',0,(1,0,1))
    with pytest.raises(ValueError,match='outside'):sim.dof('H',2,(1,1,3))
    drive=torch.zeros(4,2)
    with pytest.raises(ValueError,match='FP32'):sim(values.double(),drive)
    with pytest.raises(ValueError,match='CFL'):sim(torch.full_like(values,.0001),drive)
    sim.tensor_budget_bytes=sim.memory_plan(4)['tensor_upper_bound_bytes']-1
    with pytest.raises(ValueError,match='budget'):sim(values,drive)
    alias=EndpointSimulation([np.arange(3.)]*3,[('symmetric','symmetric')]*3,
        dt_seconds=.1/C_UM_S,sources=[],observations=[('E',0,(0,0,0))])
    assert alias.topology.faces==(('pmc','pmc'),)*3


def test_input_mutation_rejected_before_backward():
    sim=make();eps=torch.full((sim.topology.counts['E'],),2.,requires_grad=True);drive=torch.zeros(3,2)
    result=sim(eps,drive)
    with torch.no_grad():eps.add_(.1)
    with pytest.raises(RuntimeError,match='modified'):result.sum().backward()


def test_cuda_resident_trace_and_bounded_adjoint_matches_cpu():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    pytest.importorskip('cupy')
    torch.manual_seed(99);cpu=make();cuda=make(device='cuda')
    epsilon=torch.rand(cpu.topology.counts['E'])+.8;drive=torch.randn(15,2)
    e=epsilon.clone().requires_grad_();d=drive.clone().requires_grad_()
    ec=epsilon.cuda().requires_grad_();dc=drive.cuda().requires_grad_()
    reference=cpu(e,d);actual=cuda(ec,dc)
    g=torch.randn_like(reference);expected=torch.autograd.grad((reference*g).sum(),(e,d))
    result=torch.autograd.grad((actual*g.cuda()).sum(),(ec,dc))
    torch.testing.assert_close(actual.cpu(),reference,atol=4e-6,rtol=4e-5)
    for a,b in zip(result,expected):torch.testing.assert_close(a.cpu(),b,atol=2e-5,rtol=1e-4)


def test_binomial_long_duration_logical_schedule_and_actual_replay_reduction():
    from torchfdtd.pmc_simulation import _reverse_schedule
    live=set();reverse=[];replays=0;peak=0
    for action in _reverse_schedule(10000,4):
        if action[0]=='replay':
            assert action[1] is None or action[1] in live
            replays+=action[3]-action[2]
        elif action[0]=='save':live.add(action[1]);peak=max(peak,len(live))
        elif action[0]=='drop':live.remove(action[1])
        else:reverse.append(action[1])
    assert not live and peak<=4 and reverse==list(range(9999,-1,-1))
    assert replays<500000  # zero slots needs 49,995,000 forward replays
    counts=[]
    for slots in (0,3):
        sim=make(checkpoints=slots)
        epsilon=torch.full((sim.topology.counts['E'],),2.,requires_grad=True)
        drive=torch.zeros(31,2);drive[0]=1
        sim(epsilon,drive).square().sum().backward()
        assert sim.last_report['peak_checkpoints']<=slots
        assert sim.last_report['reverse_steps']==31
        counts.append(sim.last_report['replayed_steps'])
    assert counts[0]==465 and counts[1]<counts[0]/2


@pytest.mark.parametrize('mutation',['source','source_replace','operator'])
def test_mutable_replay_metadata_rejected(mutation):
    sim=make();epsilon=torch.full((sim.topology.counts['E'],),2.,requires_grad=True)
    result=sim(epsilon,torch.ones(4,2))
    if mutation=='source':sim.source_tensor.copy_(sim.source_tensor.flip(0))
    elif mutation=='source_replace':sim.source_tensor=sim.source_tensor.flip(0)
    else:sim.backend.operators[True][2].mul_(2)
    with pytest.raises(RuntimeError,match='configuration changed'):result.sum().backward()


def test_budget_admission_precedes_full_input_scans(monkeypatch):
    sim=make();epsilon=torch.full((sim.topology.counts['E'],),2.)
    sim.tensor_budget_bytes=sim.memory_plan(100)['tensor_upper_bound_bytes']-1
    def forbidden(*args,**kwargs):raise AssertionError('Numeric scan before budget admission')
    monkeypatch.setattr(torch,'isfinite',forbidden)
    with pytest.raises(ValueError,match='budget'):sim(epsilon,torch.zeros(100,2))
