"""PEC walls at physical mesh endpoints, with no half-cell displacement."""
import math
import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Monitor, AdjointOptions, DifferentiableSimulation
from torchfdtd.models import Boundaries, BoundaryFace
from torchfdtd.boundaries import YeeGrid
from torchfdtd.differentiable import _System


@pytest.fixture(autouse=True)
def restore_torch_default_dtype():
    """fdtd.set_backend('numpy') sets the global Torch default dtype to float64; leave the session as found."""
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def scene(axis=0, kind="pec", dimension="3d", complex_fields=False):
    faces={a+"_"+s:BoundaryFace(kind=kind if i==axis else "bloch" if complex_fields else "periodic")
           for i,a in enumerate("xyz"[:2 if dimension=="2d" else 3]) for s in ("min","max")}
    phase=tuple(.3 if complex_fields and i!=axis and i<2 else 0. for i in range(3))
    r=Region(dimension=dimension,size=(1.6,1.6,1.6),mesh=.1,steps=18,
             boundaries=Boundaries(**faces),material_sampling="yee",bloch_phase=phase,precision="float32",cuda_kernel="torch")
    return Project(region=r,sources=[],monitors=[])


@pytest.mark.parametrize("axis",range(3))
@pytest.mark.parametrize("kind",["pec","antisymmetric"])
@pytest.mark.parametrize("polarization",[1,2])
def test_cavity_eigenmode_and_full_domain_reduction(axis,kind,polarization):
    p=scene(axis,kind)
    fdtd.set_backend("numpy")
    g=YeeGrid(p.region)
    full=scene(axis)
    setattr(full.region.boundaries,"xyz"[axis]+"_min",BoundaryFace(kind="periodic"))
    setattr(full.region.boundaries,"xyz"[axis]+"_max",BoundaryFace(kind="periodic"))
    size=list(full.region.size);size[axis]*=2;full.region.size=tuple(size)
    f=YeeGrid(full.region)
    n=p.region.shape[axis];q=math.pi/n;omega=2*math.asin(g.courant_number*math.sin(q/2))
    component=(axis+polarization)%3
    for grid in (g,f):
        grid.E=grid.E.astype(np.float32);grid.H=grid.H.astype(np.float32)
        shape=[1,1,1];shape[axis]=grid.E.shape[axis]
        wave=np.sin(q*np.arange(grid.E.shape[axis])).reshape(shape)
        grid.E[...,component]=wave
        grid.H[:]=-g.courant_number/2*grid.curl(grid.E,True)
    initial=g.E.copy()
    for step in range(35):
        g.update_E();g.update_H();f.update_E();f.update_H()
    np.testing.assert_allclose(g.E,initial*np.cos(35*omega),atol=3e-6,rtol=3e-6)
    cut=[slice(None)]*4;cut[axis]=slice(0,n)
    np.testing.assert_allclose(g.E,f.E[tuple(cut)],atol=4e-6,rtol=4e-6)
    np.testing.assert_allclose(g.H,f.H[tuple(cut)],atol=4e-6,rtol=4e-6)
    # On-plane E tangential and H normal remain zero, including corners.
    cut[axis]=0
    assert np.max(np.abs(g.E[tuple(cut)][...,component])) < 2e-6
    assert np.max(np.abs(g.H[tuple(cut)][...,axis])) < 2e-6


@pytest.mark.parametrize("complex_fields",[False,True])
@torch.enable_grad()
def test_exact_curl_transpose(complex_fields):
    p=scene(complex_fields=complex_fields)
    system=_System(p,torch.ones(p.region.shape,dtype=torch.float32))
    for forward in (False,True):
        field=torch.randn_like(system.grid.E,requires_grad=True)
        seed=torch.randn_like(field)
        value,psis=system.curl(field,(),forward)
        expected,=torch.autograd.grad((value.conj()*seed).real.sum(),field)
        actual,_=system.curl_transpose(seed,(),forward)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)


def test_forbidden_wall_forcing_and_subpixel_are_rejected():
    p=scene(dimension="2d")
    p.sources=[Source(center=(-.8,0,0),component="Ez")]
    with pytest.raises(ValueError,match="constrained PEC"):
        Project.model_validate(p.model_dump())
    p.sources=[];p.region.interface_method="subpixel"
    with pytest.raises(ValueError,match="staircase"):
        Project.model_validate(p.model_dump())


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("complex_fields",[False,True])
@torch.enable_grad()
def test_resident_and_streamed_adjoint(device,complex_fields):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    from torchfdtd import StreamedSimulation,StreamedAdjointOptions
    p=scene(dimension="2d",complex_fields=complex_fields)
    p.sources=[Source(center=(.5,0,0),component="Ez",pulse="continuous",wavelength=1.)]
    p.monitors=[Monitor(center=(.6,.1,0),component="Ez"),Monitor(center=(-.7,.1,0),component="Hy")]
    eps=torch.full(p.region.shape,1.4,device=device,dtype=torch.float32,requires_grad=True)
    model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2))
    ref=model.reference(eps)
    expected,=torch.autograd.grad(ref.abs().square().sum(),eps)
    if device=="cuda":
        p.region.cuda_kernel="fused"
        model=DifferentiableSimulation(p,AdjointOptions(checkpoints=2,backward_kernel="fused"))
    result=model(eps)
    actual,=torch.autograd.grad(result.signals.abs().square().sum(),eps)
    torch.testing.assert_close(result.signals,ref,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=4e-5)
    host=eps.detach().cpu().requires_grad_()
    streamed=StreamedSimulation(p,StreamedAdjointOptions(device=device,slab_width=5,temporal_depth=3,checkpoints=2))
    result=streamed(host)
    gradient,=torch.autograd.grad(result.signals.abs().square().sum(),host)
    torch.testing.assert_close(result.signals,ref.cpu(),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(gradient,expected.cpu(),atol=3e-6,rtol=4e-5)


@pytest.mark.parametrize("kind",["pmc","symmetric"])
def test_magnetic_wall_rejects_periodic_endpoint_mixing(kind):
    assert BoundaryFace(kind=kind).kind==kind
    with pytest.raises(ValueError,match="periodic or Bloch mixing"):
        scene(kind=kind)


@pytest.mark.parametrize("axis",range(3))
@torch.enable_grad()
def test_nonuniform_pec_metric_and_transpose(axis):
    p=scene(axis)
    p.region.mesh_type="explicit"
    nodes=tuple(tuple(.8*(.8*u+.2*u**3) for u in np.linspace(-1,1,17)) for _ in range(3))
    p.region.mesh_coordinates=nodes
    p=Project.model_validate(p.model_dump())
    system=_System(p,torch.ones(p.region.shape,dtype=torch.float32))
    field=torch.randn_like(system.grid.E,requires_grad=True)
    seed=torch.randn_like(field)
    result,_=system.curl(field,(),True)
    exact,=torch.autograd.grad((result*seed).sum(),field)
    actual,_=system.curl_transpose(seed,(),True)
    torch.testing.assert_close(actual,exact,atol=2e-6,rtol=2e-6)
    # Constant transverse field has only the physical upper-wall derivative.
    flat=torch.zeros_like(field);flat[...,(axis+1)%3]=1
    curl,_=system.curl(flat,(),True)
    cut=[slice(None)]*4;cut[axis]=-1
    factor=p.region.reference_step/(nodes[axis][-1]-nodes[axis][-2])
    assert curl[tuple(cut)].abs().max().item()==pytest.approx(factor,rel=1e-6)


@torch.enable_grad()
def test_real_cuda_batch_reuses_pec_physics():
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    from torchfdtd.cuda_batch import FusedBatchYeeCUDA
    p=scene()
    systems=[_System(p,torch.ones(p.region.shape,device="cuda",dtype=torch.float32),prepare_kernels=False) for _ in range(2)]
    references=[]
    for system in systems:
        system.grid.region=p.region
        for value in system.state():value.normal_()
        references.append(system.reference_step(system.state(),0,system.epsilon))
    batch=FusedBatchYeeCUDA([s.grid for s in systems])
    batch.update(False);batch.update(True)
    for system,reference in zip(systems,references):
        for actual,expected in zip(system.state(),reference):
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize("family,theta,phi",[("Ex",45,90),("Hy",90,0)])
def test_vector_source_secondary_wall_component_rejected(family,theta,phi):
    p=scene(dimension="2d")
    p.sources=[Source(center=(-.8,0,0),component=family,theta=theta,phi=phi)]
    with pytest.raises(ValueError,match="constrained PEC"):
        Project.model_validate(p.model_dump())


@pytest.mark.parametrize("kind",["plane","tfsf"])
def test_compound_oneway_sources_do_not_bypass_mirror_guard(kind):
    p=scene(dimension="2d")
    p.sources=[Source(kind=kind,injection="oneway",normal="x",center=(0,0,0),size=(0,1.6,0) if kind=="plane" else (.6,.6,0))]
    with pytest.raises(ValueError,match="PML"):
        Project.model_validate(p.model_dump())


def test_three_dimensional_pec_edges_and_corners_preserve_constraints():
    p=scene()
    p.region.boundaries=Boundaries(**{a+"_"+side:BoundaryFace(kind="pec") for a in "xyz" for side in ("min","max")})
    fdtd.set_backend("numpy")
    grid=YeeGrid(p.region)
    rng=np.random.default_rng(741)
    grid.E=rng.normal(size=grid.E.shape).astype(np.float32)
    grid.H=rng.normal(size=grid.H.shape).astype(np.float32)
    for axis in range(3):
        cut=[slice(None)]*4;cut[axis]=0
        for comp in range(3):
            cut[3]=comp
            (grid.H if comp==axis else grid.E)[tuple(cut)]=0
    for _ in range(20):grid.update_E();grid.update_H()
    for axis in range(3):
        cut=[slice(None)]*4;cut[axis]=0
        for comp in range(3):
            cut[3]=comp
            assert np.count_nonzero((grid.H if comp==axis else grid.E)[tuple(cut)])==0
