"""CPU endpoint/CPML auxiliary adjoints and mirrored-domain pulse evidence."""
import math
import numpy as np
import pytest
import torch

from torchfdtd.pmc_cpml import EndpointCPMLSimulation, CPMLState
from torchfdtd.pmc_simulation import C_UM_S


def make(checkpoints=2,budget=64_000_000):
    return EndpointCPMLSimulation([np.arange(7)*.1,np.arange(5)*.1,np.arange(4)*.1],
        [('pml','pml'),('pmc','pmc'),('pec','pmc')],dt_seconds=.03/C_UM_S,
        pml_cells=1,background_epsilon=2.,sources=[(0,(3,4,3))],
        observations=[('E',0,(3,4,3)),('E',0,(3,4,3)),('H',1,(3,4,2))],
        checkpoints=checkpoints,tensor_budget_bytes=budget)


def independent_terms(sim):
    """Dense tiny scalar-index derivative assembly, no native curl/operator."""
    topology=sim.backend.topology
    terms={}
    for forward in (False,True):
        source,target=('E','H') if forward else ('H','E')
        groups=[]
        # curl component q = d_(q+1) F_(q+2) - d_(q+2) F_(q+1).
        for q in range(3):
            for axis,component,sign in (((q+1)%3,(q+2)%3,1),((q+2)%3,(q+1)%3,-1)):
                matrix=torch.zeros((len(topology.components[target]),len(topology.components[source])))
                rates=torch.zeros(len(matrix))
                for row,(comp,index) in enumerate(zip(topology.components[target],topology.indices[target])):
                    if comp!=q or not topology.active[target][row]:continue
                    i=index[axis];n=topology.shape[axis];h=topology.nodes[axis][1]-topology.nodes[axis][0]
                    if forward:neighbors=[(i+1,1/h),(i,-1/h)]
                    elif i==0:neighbors=[(0,2/h)]
                    elif i==n:neighbors=[(n-1,-2/h)]
                    else:neighbors=[(i,1/h),(i-1,-1/h)]
                    for j,w in neighbors:
                        pos=list(index);pos[axis]=j
                        col=topology.lookup[source].get((component,tuple(pos)))
                        if col is not None and topology.active[source][col]:matrix[row,col]+=sign*w
                    x=topology.coordinates[target][row,axis];length=sim.pml_cells*h
                    depth=0.
                    for side in (0,1):
                        if sim.physical_faces[axis][side]=='pml':
                            distance=x-topology.nodes[axis][0] if side==0 else topology.nodes[axis][-1]-x
                            depth=max(depth,1-distance/length)
                    rates[row]=-2*math.log(sim.reflection)/(length*math.sqrt(sim.background_epsilon))*depth**3
                groups.append((matrix,torch.exp(-rates*sim.dt),torch.expm1(-rates*sim.dt)))
        terms[forward]=groups
    return terms


def independent_trace(sim,epsilon,waveform):
    terms=independent_terms(sim)
    e=torch.zeros_like(epsilon);h=torch.zeros(sim.topology.counts['H'])
    psi={key:[torch.zeros(len(matrix)) for matrix,_,_ in groups] for key,groups in terms.items()}
    samples=[]
    for drive in waveform:
        curls=[]
        for i,(matrix,b,c) in enumerate(terms[False]):
            derivative=matrix@h;psi[False][i]=b*psi[False][i]+c*derivative
            curls.append(derivative+psi[False][i])
        e=e*sim.backend.active['E']+sim.dt/epsilon*sum(curls)
        e=e.index_add(0,sim.source_tensor,drive)
        curls=[]
        for i,(matrix,b,c) in enumerate(terms[True]):
            derivative=matrix@e;psi[True][i]=b*psi[True][i]+c*derivative
            curls.append(derivative+psi[True][i])
        h=h*sim.backend.active['H']-sim.dt*sum(curls)
        samples.append(torch.stack([(e if family=='E' else h)[i] for family,i in sim.observation_ids]))
    return torch.stack(samples)


def test_independent_split_trace_material_source_vjp_and_fd():
    sim=make();torch.manual_seed(14)
    parameter=torch.tensor(.2,requires_grad=True)
    epsilon=2+parameter*(~sim.collar)
    waveform=(.1*torch.randn(13,1)).requires_grad_()
    expected=independent_trace(sim,epsilon,waveform)
    wanted=torch.autograd.grad(expected.square().sum(),(parameter,waveform),retain_graph=True)
    actual=sim(epsilon,waveform)
    gradients=torch.autograd.grad(actual.square().sum(),(parameter,waveform))
    assert sim.last_report['peak_checkpoints']<=2 and sim.last_report['reverse_steps']==13
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-7)
    for a,b in zip(gradients,wanted):torch.testing.assert_close(a,b,rtol=8e-5,atol=2e-7)
    with torch.no_grad():
        high=sim(2+(parameter+.002)*(~sim.collar),waveform).square().sum()
        low=sim(2+(parameter-.002)*(~sim.collar),waveform).square().sum()
    torch.testing.assert_close(gradients[0],(high-low)/.004,rtol=.004,atol=2e-6)


def test_nonzero_auxiliary_fullstate_transpose_and_fixed_collar():
    sim=make();torch.manual_seed(39)
    zero=sim._zero()
    values=[torch.randn_like(v).requires_grad_() for v in (zero.electric,zero.magnetic,*zero.psi)]
    state=CPMLState(values[0],values[1],tuple(values[2:]))
    seeds=[torch.randn_like(v) for v in values]
    seed=CPMLState(seeds[0],seeds[1],tuple(seeds[2:]))
    parameter=torch.tensor(.2,requires_grad=True);epsilon=2+parameter*(~sim.collar)
    drive=torch.tensor([.1],requires_grad=True)
    result=sim._step(state,epsilon,drive)
    loss=sum((a*b).sum() for a,b in zip(seeds,(result.electric,result.magnetic,*result.psi)))
    expected=torch.autograd.grad(loss,(*values,parameter,drive))
    previous,ge,gw=sim._transpose(state,seed,epsilon)
    for a,b in zip((previous.electric,previous.magnetic,*previous.psi),expected[:-2]):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(ge.sum(),expected[-2],rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(gw,expected[-1],rtol=2e-5,atol=2e-6)
    assert ge[sim.collar].count_nonzero()==0
    plan=sim.memory_plan(13)
    assert plan['complete_state_bytes']==sum(v.numel()*4 for v in values)


def test_admission_memory_and_saved_configuration():
    with pytest.raises(ValueError,match='budget'):make(budget=100)
    sim=make();epsilon=torch.full((sim.topology.counts['E'],),2.,requires_grad=True)
    wave=torch.zeros(4,1)
    bad=epsilon.detach().clone();bad[sim.collar]=2.1
    with pytest.raises(ValueError,match='collar'):sim(bad,wave)
    trace=sim(epsilon,wave)
    with torch.no_grad():sim.backend.operators[0][4].add_(.01)
    with pytest.raises(RuntimeError,match='configuration'):trace.sum().backward()


def test_reflected_full_and_reduced_pulse_intersections_and_decay():
    def domain(reduced):
        shift=0 if reduced else 16
        nodes=[np.arange(17)*.1 if reduced else np.arange(-16,17)*.1,
               np.arange(7)*.1,np.arange(7)*.1]
        return EndpointCPMLSimulation(nodes,
            [('pmc' if reduced else 'pml','pml'),('pmc','pmc'),('pmc','pmc')],
            dt_seconds=.04/C_UM_S,pml_cells=4,background_epsilon=2.,
            sources=[(2,(shift,3,2))],observations=[('E',2,(shift+2,6,2)),('H',1,(shift+5,6,3))],
            checkpoints=2)
    half,full=domain(True),domain(False)
    epsh=torch.full((half.topology.counts['E'],),2.)
    epsf=torch.full((full.topology.counts['E'],),2.)
    t=torch.arange(240)
    wave=(torch.exp(-((t-15)/5).square())*torch.sin(.7*t))[:,None]
    hstate,fstate=half._zero(),full._zero()
    lookup=full.backend.topology.lookup
    maps={family:torch.tensor([lookup[family][(int(c),(int(q[0])+16,int(q[1]),int(q[2])))]
        for c,q in zip(half.backend.topology.components[family],half.backend.topology.indices[family])])
        for family in ('E','H')}
    energies=[];maxerror=0.
    with torch.no_grad():
        for drive in wave:
            hstate=half._step(hstate,epsh,drive)
            fstate=full._step(fstate,epsf,drive)
            for family,a,b in (('E',hstate.electric,fstate.electric),('H',hstate.magnetic,fstate.magnetic)):
                maxerror=max(maxerror,float((a-b[maps[family]]).abs().max()))
            energies.append(float(2*hstate.electric.square().sum()+hstate.magnetic.square().sum()))
    assert maxerror<4e-6
    decay=energies[-1]/max(energies)
    assert decay<.5
    hp,fp=half.memory_plan(len(wave)),full.memory_plan(len(wave))
    assert hp['psi_elements']<fp['psi_elements']
    assert hp['complete_state_bytes']<fp['complete_state_bytes']
    derivatives=[];traces=[]
    for sim in (half,full):
        parameter=torch.tensor(.2,requires_grad=True)
        xyz=sim.backend.topology.coordinates['E']
        mask=torch.tensor((abs(xyz[:,0])>.25)&(abs(xyz[:,0])<.65))
        epsilon=2+parameter*mask
        drive=wave[:60].clone().requires_grad_()
        trace=sim(epsilon,drive)
        derivatives.append(torch.autograd.grad(trace.square().sum(),(parameter,drive)))
        traces.append(trace.detach())
    torch.testing.assert_close(traces[0],traces[1],rtol=2e-5,atol=1e-7)
    for a,b in zip(derivatives[0],derivatives[1]):
        torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-7)
    assert derivatives[0][0].abs()>1e-6
    print(dict(max_reflection_field_error=maxerror,late_unweighted_norm_ratio=decay,
        half_field_bytes=hp['field_state_bytes'],full_field_bytes=fp['field_state_bytes'],
        half_psi_bytes=hp['psi_state_bytes'],full_psi_bytes=fp['psi_state_bytes'],
        half_material_vjp=float(derivatives[0][0]),full_material_vjp=float(derivatives[1][0])))
