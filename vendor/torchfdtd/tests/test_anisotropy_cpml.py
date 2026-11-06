"""Finite nodal closure and isotropic-exterior CPML composition diagnostics."""
from itertools import product
import math

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation, Monitor, Project, Region, Source
from torchfdtd.anisotropy import (TensorConstitutive, TensorDielectricSimulation,
    _TensorSystem, _cpml_collar_slices)


def scene(precision='float32', mixed=False):
    faces = {a+'_'+side: dict(kind='bloch' if mixed and a=='y' else 'pml')
        for a in 'xyz' for side in ('min','max')}
    r = Region(dimension='3d', size=(1.2, 1.2, 1.2), mesh=.1, steps=14,
        precision=precision, pml_cells=3, material_sampling='yee', boundaries=faces,
        bloch_phase=(0,.23,0) if mixed else (0,0,0))
    return Project(region=r, sources=[Source(component='Ez', center=(0,0,0), wavelength=.7,
        pulse='gaussian', time_definition='standard', pulse_length=.5e-15, pulse_offset=.7e-15)],
        monitors=[Monitor(component=c, center=(.1,0,0)) for c in ('Ex','Ey','Ez','Hx')])


def design_mask(r):
    mask = torch.ones(r.shape, dtype=torch.bool)
    for index in _cpml_collar_slices(r):
        mask[index] = False
    return mask


def material(r, parameter):
    base = 2*torch.eye(3, dtype=parameter.dtype)
    return torch.where(design_mask(r)[...,None,None], parameter, base)


def dense_constitutive(epsilon, periodic, phases):
    """Independent explicit incidence matrices, only for a tiny diagnostic."""
    shape = epsilon.shape[:3]
    count = math.prod(shape)*3
    dtype = torch.complex128
    matrix = torch.zeros((count,count), dtype=dtype)
    coverage = torch.zeros(count, dtype=torch.float64)
    for node in product(*(range(n) for n in shape)):
        for signs in product((0,1), repeat=3):
            rows = torch.zeros((3,count), dtype=dtype)
            for a in range(3):
                index = list(node)
                coefficient = 1.
                if signs[a]:
                    index[a] -= 1
                    if index[a] < 0:
                        if not periodic[a]:
                            continue
                        index[a] += shape[a]
                        coefficient = 1/phases[a]
                flat = ((index[0]*shape[1]+index[1])*shape[2]+index[2])*3+a
                rows[a,flat] = coefficient
            coverage += rows.abs().square().sum(0)/8
            matrix = matrix+rows.conj().T@torch.linalg.inv(epsilon[node]).to(dtype)@rows/8
    scale = coverage.rsqrt()
    return scale[:,None]*matrix*scale[None,:]


def test_finite_closure_dense_bounds_transpose_and_no_opposite_face_coupling():
    torch.manual_seed(83)
    raw = torch.randn(3,3,3,3,3,dtype=torch.float64)
    epsilon = (torch.eye(3)+raw@raw.transpose(-1,-2)).requires_grad_()
    phases = (1.,complex(math.cos(.4),math.sin(.4)),1.)
    operator = TensorConstitutive(epsilon, phases, (False,True,False))
    matrix = dense_constitutive(epsilon,(False,True,False),phases)
    torch.testing.assert_close(matrix,matrix.conj().T,atol=2e-15,rtol=2e-15)
    eigen = torch.linalg.eigvalsh(matrix)
    assert eigen.min()>0 and eigen.max()<=1+1e-12
    x = torch.randn(3,3,3,3,dtype=torch.complex128)
    y = torch.randn_like(x)
    torch.testing.assert_close(operator.apply(x).flatten(),matrix@x.flatten(),atol=2e-15,rtol=2e-14)
    expected, = torch.autograd.grad((y.conj().flatten()*(matrix@x.flatten())).real.sum(),epsilon)
    expected = (expected+expected.transpose(-1,-2))/2
    torch.testing.assert_close(operator.epsilon_vjp(x,y),expected,atol=2e-14,rtol=2e-13)
    impulse = torch.zeros_like(x);impulse[0,1,1,0]=1
    assert torch.count_nonzero(operator.apply(impulse)[-1])==0


def test_cpml_support_one_node_collar_contains_constitutive_reach():
    p=scene(); mask=design_mask(p.region)
    tensor=torch.tensor([[2.4,.2,.1],[.2,2.1,-.1],[.1,-.1,2.8]])
    epsilon=material(p.region,tensor)
    operator=TensorConstitutive(epsilon,periodic=(False,False,False))
    # All CPML-target field rows, including intersections and terminal edges.
    supported=torch.zeros(p.region.shape,dtype=torch.bool)
    for axis in range(3):
        for side in range(2):
            index=[slice(None)]*3; layers=p.region.pml_layers(axis,side)
            index[axis]=slice(0,layers) if side==0 else slice(-layers,None)
            supported[tuple(index)]=True
    field=torch.randn(p.region.shape+(3,))*supported[...,None]
    torch.testing.assert_close(operator.apply(field),field/2,rtol=2e-6,atol=3e-7)
    assert mask.any()


def test_isotropic_cpml_forward_and_admitted_material_gradient_match_native():
    p=scene()
    density=torch.full(p.region.shape,2.,requires_grad=True)
    mask=design_mask(p.region)
    admitted=torch.where(mask,density,2.)
    native=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))(admitted).signals
    tensor=admitted[...,None,None]*torch.eye(3)
    result=TensorDielectricSimulation(p,AdjointOptions(checkpoints=2),cpml_background_epsilon=2.)(tensor)
    actual,=torch.autograd.grad(result.signals.square().sum(),density)
    torch.testing.assert_close(result.signals,native,rtol=4e-5,atol=2e-7)
    # Nodal scalar interpolation differs from independently sampled Yee scalar
    # perturbations: compare the uniform interior direction away from collars
    # through the matching nodal-to-edge diagonal coefficient mapping below.
    node=admitted.detach().clone().requires_grad_()
    edge=[]
    for axis in range(3):
        inverse=1/node
        shifted=torch.cat((inverse.narrow(axis,1,inverse.shape[axis]-1),inverse.narrow(axis,inverse.shape[axis]-1,1)),axis)
        edge.append(2/(inverse+shifted))
    diagonal=torch.stack(edge,-1)
    scalar=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))(diagonal).signals
    reference,=torch.autograd.grad(scalar.square().sum(),node)
    reference=reference*mask
    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-8)
    assert torch.count_nonzero(actual[~mask])==0
    assert result.report['restart_bytes']>2*math.prod(p.region.shape)*3*4


@pytest.mark.parametrize('mixed',[False,True])
def test_rotated_interior_pulse_checkpoint_vjp_matches_full_autograd(mixed):
    p=scene('float64',mixed)
    coefficient=torch.tensor([[2.4,.2,.1],[.2,2.1,-.1],[.1,-.1,2.8]],dtype=torch.float64,requires_grad=True)
    epsilon=material(p.region,coefficient)
    system=_TensorSystem(p,epsilon)
    state=tuple(torch.zeros_like(v) for v in system.state())
    samples=[]
    for step in range(p.region.steps):
        state=system.reference_step(state,step,epsilon)
        samples.append(system.observe(state))
    oracle=torch.stack(samples)
    expected,=torch.autograd.grad(oracle.abs().square().sum(),coefficient,retain_graph=True)
    expected=(expected+expected.T)/2
    result=TensorDielectricSimulation(p,AdjointOptions(checkpoints=2),cpml_background_epsilon=2.)(epsilon)
    actual,=torch.autograd.grad(result.signals.abs().square().sum(),coefficient)
    torch.testing.assert_close(result.signals,oracle,rtol=2e-12,atol=2e-13)
    torch.testing.assert_close(actual,expected,rtol=2e-11,atol=2e-12)
    assert actual.abs().max()>1e-6


def test_fixed_cpml_admission_zero_collar_vjp_and_saved_input_version(monkeypatch):
    p=scene()
    with pytest.raises(ValueError,match='explicit fixed'):
        TensorDielectricSimulation(p)
    model=TensorDielectricSimulation(p,cpml_background_epsilon=2.)
    eigvalsh=torch.linalg.eigvalsh
    batch_sizes=[]
    def bounded_validation(value):
        batch_sizes.append(value.shape[0])
        assert value.ndim==3 and value.shape[0]<=64
        return eigvalsh(value)
    monkeypatch.setattr(torch.linalg,'eigvalsh',bounded_validation)
    epsilon=(2*torch.eye(3)).expand(p.region.shape+(3,3)).clone().requires_grad_()
    with pytest.raises(ValueError,match='precision'):
        model(epsilon.double())
    bad=epsilon.detach().clone();bad[0,0,0,0,0]=2.1
    with pytest.raises(ValueError,match='fixed isotropic'):
        model(bad)
    result=model(epsilon)
    gradient,=torch.autograd.grad(result.signals.square().sum(),epsilon)
    assert batch_sizes and max(batch_sizes)<=64
    assert torch.count_nonzero(gradient[~design_mask(p.region)])==0
    result=model(epsilon)
    with torch.no_grad():epsilon[5,5,5,0,0]+= .01
    with pytest.raises(RuntimeError,match='modified by an inplace'):
        result.signals.square().sum().backward()


def test_full_cpml_state_transpose_with_nonzero_memories():
    p=scene('float64',True)
    coefficient=torch.tensor([[2.4,.2,.1],[.2,2.1,-.1],[.1,-.1,2.8]],dtype=torch.float64,requires_grad=True)
    epsilon=material(p.region,coefficient)
    system=_TensorSystem(p,epsilon,fixed_collar=True)
    generator=torch.Generator().manual_seed(192)
    state=tuple(torch.randn(v.shape,dtype=v.dtype,generator=generator).requires_grad_() for v in system.state())
    seeds=tuple(torch.randn(v.shape,dtype=v.dtype,generator=generator) for v in state)
    output=system.reference_step(state,0,epsilon)
    loss=sum((a.conj()*b).real.sum() for a,b in zip(seeds,output))
    expected=torch.autograd.grad(loss,(*state,coefficient))
    actual,gradient=system.transpose_step(state,tuple(v.clone() for v in seeds),torch.zeros(4,dtype=state[0].dtype))
    for a,b in zip(actual,expected[:-1]):
        torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)
    reference=(expected[-1]+expected[-1].T)/2
    torch.testing.assert_close(gradient.sum((0,1,2)),reference,rtol=2e-12,atol=2e-12)
    assert len(state)>2 and any(v.abs().max()>0 for v in actual[2:])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('mixed',[False,True])
def test_cpml_cuda_fp32_matches_cpu_autograd_and_reservation(mixed):
    p=scene('float32',mixed)
    coefficient=torch.tensor([[2.4,.2,.1],[.2,2.1,-.1],[.1,-.1,2.8]],requires_grad=True)
    epsilon=material(p.region,coefficient)
    system=_TensorSystem(p,epsilon)
    state=tuple(torch.zeros_like(v) for v in system.state())
    samples=[]
    for step in range(p.region.steps):
        state=system.reference_step(state,step,epsilon)
        samples.append(system.observe(state))
    oracle=torch.stack(samples)
    expected,=torch.autograd.grad(oracle.abs().square().sum(),coefficient)
    expected=(expected+expected.T)/2
    torch.cuda.reset_peak_memory_stats()
    cuda_parameter=coefficient.detach().cuda().requires_grad_()
    cuda_epsilon=torch.where(design_mask(p.region).cuda()[...,None,None],cuda_parameter,2*torch.eye(3,device='cuda'))
    result=TensorDielectricSimulation(p,AdjointOptions(checkpoints=2),cpml_background_epsilon=2.)(cuda_epsilon)
    actual,=torch.autograd.grad(result.signals.abs().square().sum(),cuda_parameter)
    torch.testing.assert_close(result.signals.cpu(),oracle,rtol=7e-5,atol=3e-7)
    torch.testing.assert_close(actual.cpu(),expected,rtol=2e-4,atol=3e-7)
    peak=torch.cuda.max_memory_allocated()
    assert peak<=result.report['gpu_reservation_bytes']
    print(dict(mixed=mixed,cuda_peak_allocated_bytes=peak,reservation_bytes=result.report['gpu_reservation_bytes']))
