"""Plane interpolation/flux derivatives against full-state time-domain autograd."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from torchfdtd import (FieldMonitor,Monitor,AdjointOptions,StreamedAdjointOptions,
                        DifferentiablePlaneSimulation,DifferentiablePlaneResult)
from torchfdtd.differentiable import _System
from torchfdtd.field_monitors import plane_plan,interpolation_map
from torchfdtd.adjoint_planes import COMPONENTS
from test_differentiable import project,gpu


def scene(dimension='2d',periodic=False):
    p=project(dimension=dimension,steps=11,periodic=periodic)
    p.monitors=[FieldMonitor(center=(.12,0,0),size=(0,.65,.55),normal='x',downsample=2),
                FieldMonitor(center=(.21,.03,0),size=(0,.55,.45),normal='x',downsample=2)]
    return p


def reference(p,eps,frequencies):
    system=_System(p,eps)
    state=system.state()
    results={m.id:[] for m in p.monitors}
    for step in range(p.region.steps):
        state=system.reference_step(state,step,eps)
        for m in p.monitors:
            plan=plane_plan(p.region,m)
            values=[]
            for c in COMPONENTS:
                indices,weights=interpolation_map(p.region,c,plan['points_um'])
                indices=torch.tensor(indices,device=eps.device)
                weights=torch.tensor(weights,device=eps.device,dtype=eps.dtype)
                spatial=(state[0 if c[0]=='E' else 1].reshape(-1)[indices]*weights).sum(0)
                t=(step+1+(.5 if c[0]=='H' else 0))*p.region.time_step
                values.append(torch.exp(2j*torch.pi*frequencies[:,None]*t)*spatial[None,:]*p.region.time_step)
            results[m.id].append(torch.stack(values,dim=-1))
    return {key:torch.stack(values).sum(0) for key,values in results.items()}


@pytest.mark.parametrize('mode',['resident','host','disk'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_planes_and_flux_vjp_match_full_autograd(tmp_path,mode,device):
    if device=='cuda':gpu()
    p=scene(periodic=True)
    epsilon=torch.full(p.region.shape+(3,),1.7,dtype=torch.float64,device=device if mode=='resident' else 'cpu',requires_grad=True)
    frequency=torch.tensor([.022,.051],dtype=epsilon.dtype,device=epsilon.device)/p.region.time_step
    expected=reference(p,epsilon,frequency)
    def objective(fields):
        z=fields/p.region.time_step
        return (z.real+.4*z.imag).square().sum()+(.5*(z[...,1]*z[...,5].conj()-z[...,2]*z[...,4].conj()).real).sum()
    expected_gradient,=torch.autograd.grad(sum(objective(f) for f in expected.values()),epsilon)
    options=AdjointOptions(checkpoints=2) if mode=='resident' else StreamedAdjointOptions(device=device,slab_width=4,temporal_depth=3,
        state_storage=mode,state_directory=tmp_path if mode=='disk' else None,disk_budget_bytes=32*1024**2 if mode=='disk' else None,
        tile_transfers='async' if device=='cuda' else 'sync')
    model=DifferentiablePlaneSimulation(p,options)
    result=model(epsilon,frequency,block_size=4)
    assert len(model.observers)>32
    for key,plane in result.items():
        torch.testing.assert_close(plane.fields/p.region.time_step,expected[key]/p.region.time_step,rtol=2e-11,atol=1e-12)
        assert plane.report['plane_workspace_reservation_bytes']>0
        torch.testing.assert_close(plane.flux(),plane.poynting()@plane.weights)
    gradient,=torch.autograd.grad(sum(objective(r.fields) for r in result.values()),epsilon)
    torch.testing.assert_close(gradient,expected_gradient,rtol=3e-10,atol=1e-11)
    assert gradient.norm()>1e-5
    if mode=='disk':assert list(tmp_path.iterdir())==[]


def test_three_dimensional_plane_and_flux():
    p=scene('3d')
    epsilon=torch.full(p.region.shape,1.8,dtype=torch.float64,requires_grad=True)
    frequency=torch.tensor([.04],dtype=epsilon.dtype)/p.region.time_step
    expected=reference(p,epsilon,frequency)
    results=DifferentiablePlaneSimulation(p,AdjointOptions(checkpoints=1))(epsilon,frequency)
    for key,result in results.items():
        torch.testing.assert_close(result.fields/p.region.time_step,expected[key]/p.region.time_step,rtol=1e-11,atol=1e-12)
    key=p.monitors[0].id
    a=results[key]
    z=expected[key]/p.region.time_step
    expected_loss=(.5*(z[...,1]*z[...,5].conj()-z[...,2]*z[...,4].conj()).real@a.weights).sum()
    expected_grad,=torch.autograd.grad(expected_loss,epsilon)
    actual_grad,=torch.autograd.grad(a.flux().sum()/p.region.time_step**2,epsilon)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-9,atol=1e-22)


def test_affine_spatial_interpolation_and_quadrature():
    from torchfdtd.solver import field_axes
    p=scene('3d');m=p.monitors[0]
    plan=plane_plan(p.region,m)
    expected=1+plan['points_um']@np.array([2.,-3.,.7])
    for c in COMPONENTS:
        xyz=np.meshgrid(*field_axes(p.region,c),indexing='ij')
        field=np.zeros((*p.region.shape,3))
        field[...,COMPONENTS.index(c)%3]=1+2*xyz[0]-3*xyz[1]+.7*xyz[2]
        indices,weights=interpolation_map(p.region,c,plan['points_um'])
        np.testing.assert_allclose((field.reshape(-1)[indices]*weights).sum(0),expected,rtol=1e-14,atol=1e-14)
    assert plan['weights'].sum()==pytest.approx(.65*.55*1e-12)


def test_reference_normalization_and_field_subtraction_gradients():
    amplitude=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    base=torch.zeros((2,3,6),dtype=torch.complex128)
    base[...,1]=1;base[...,5]=1
    args=dict(frequency_hz=torch.tensor([1e12,2e12],dtype=torch.float64),points_um=torch.zeros((3,3),dtype=torch.float64),
              weights=torch.ones(3,dtype=torch.float64),shape=(1,3,1),normal='x',run_signature='fixed',report={})
    reference_result=DifferentiablePlaneResult(fields=base,**args)
    sample=DifferentiablePlaneResult(fields=base*amplitude,**args)
    ratio=sample.normalized_flux(reference_result)
    torch.testing.assert_close(ratio,torch.full((2,),4.,dtype=torch.float64))
    grad,=torch.autograd.grad(ratio.sum(),amplitude,retain_graph=True)
    assert float(grad)==pytest.approx(8)
    reflected=sample.normalized_flux(reference_result,subtract_incident=True)
    torch.testing.assert_close(reflected,torch.ones(2,dtype=torch.float64))
    with pytest.raises(ValueError,match='mesh'):sample.normalized_flux(replace(reference_result,run_signature='different'))
    with pytest.raises(ValueError,match='weak'):sample.normalized_flux(replace(reference_result,fields=base*0))
    with pytest.raises(ValueError,match='frequency'):sample.normalized_flux(replace(reference_result,frequency_hz=args['frequency_hz']*2))


@pytest.mark.parametrize('subtract',[False,True])
def test_fp32_normalized_flux_preserves_both_graphs_at_photonic_si_scales(subtract):
    def run(dtype):
        amplitude=torch.tensor(2.,dtype=dtype,requires_grad=True)
        incident=torch.tensor(.8,dtype=dtype,requires_grad=True)
        base=torch.zeros((2,3,6),dtype=torch.complex64 if dtype==torch.float32 else torch.complex128)
        base[...,1]=1e-16*(1+.3j);base[...,5]=.9e-16*(1-.2j)
        args=dict(frequency_hz=torch.tensor([1e14,2e14],dtype=dtype),
            points_um=torch.zeros((3,3),dtype=dtype),weights=torch.full((3,),1e-14,dtype=dtype),
            shape=(1,3,1),normal='x',run_signature='fixed',report={})
        reference=DifferentiablePlaneResult(fields=base*incident,**args)
        sample=DifferentiablePlaneResult(fields=base*amplitude,**args)
        ratio=sample.normalized_flux(reference,subtract_incident=subtract)
        gradients=torch.autograd.grad(ratio.sum(),(amplitude,incident))
        return ratio,gradients
    actual,got=run(torch.float32)
    expected,want=run(torch.float64)
    torch.testing.assert_close(actual.double(),expected,rtol=1e-6,atol=1e-6)
    for a,b in zip(got,want):
        assert torch.isfinite(a)
        torch.testing.assert_close(a.double(),b,rtol=1e-6,atol=1e-6)
    # Analytic derivatives of (sample/reference - subtract)^2, two bands.
    residual=2./.8-int(subtract)
    assert float(got[0])==pytest.approx(4*residual/.8,rel=1e-6)
    assert float(got[1])==pytest.approx(-4*residual*2./.8**2,rel=1e-6)


def test_unsupported_monitor_settings_are_rejected():
    p=scene();p.monitors=[Monitor()]
    with pytest.raises(ValueError,match='field monitors'):DifferentiablePlaneSimulation(p)
    p=scene();p.monitors[0].time_downsample=2
    with pytest.raises(ValueError,match='time_downsample'):DifferentiablePlaneSimulation(p)


def test_native_frequency_plane_matches_complex_fields_and_flux():
    from torchfdtd import SpectrumSettings
    from torchfdtd.field_monitors import FrequencyPlane
    p=scene();p.monitors=p.monitors[:1]
    frequencies=[.025/p.region.time_step,.06/p.region.time_step]
    m=p.monitors[0]
    m.spectrum=SpectrumSettings(sampling='custom',custom_frequencies_hz=frequencies,apodization='none')
    eps=torch.full(p.region.shape,1.7,dtype=torch.float64)
    with torch.no_grad():
        system=_System(p,eps)
        system.grid.region=p.region
        system.grid.memory_states=[]
        native=FrequencyPlane(system.grid,m)
        for t in range(p.region.steps):
            system.advance(t,t+1)
            native.update(torch.tensor([t]))
        expected=native.result()
    actual=DifferentiablePlaneSimulation(p)(eps,frequencies)[m.id]
    np.testing.assert_allclose(actual.fields.numpy()/p.region.time_step,expected['fields']/p.region.time_step,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(actual.flux().numpy()/p.region.time_step**2,expected['flux']/p.region.time_step**2,rtol=1e-11,atol=1e-18)


def test_shared_planes_deduplicate_samples_and_budget_before_state(monkeypatch):
    p=scene();p.monitors=p.monitors[:1]
    single=DifferentiablePlaneSimulation(p)
    p.monitors.append(p.monitors[0].model_copy(update={'id':'duplicate-plane'}))
    duplicate=DifferentiablePlaneSimulation(p)
    assert duplicate.observers==single.observers
    def forbidden(*args,**kwargs):raise AssertionError('Physical fields allocated before admission')
    monkeypatch.setattr('torchfdtd.streamed._System',forbidden)
    model=DifferentiablePlaneSimulation(p,StreamedAdjointOptions(device='cpu',host_budget_bytes=1))
    with pytest.raises(ValueError,match='host budget'):
        model(torch.ones(p.region.shape,dtype=torch.float64),[1e12])


def test_normalized_plane_material_derivative_finite_difference():
    p=scene();p.monitors=p.monitors[:1]
    model=DifferentiablePlaneSimulation(p,AdjointOptions(checkpoints=2))
    mask=torch.zeros(p.region.shape,dtype=torch.float64)
    mask[6:9,5:8]=1
    frequency=[.03/p.region.time_step]
    with torch.no_grad():ref=model(torch.ones_like(mask),frequency)[p.monitors[0].id]
    def loss(value):
        return model(1+value*mask,frequency)[p.monitors[0].id].normalized_flux(ref).sum()
    value=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    grad,=torch.autograd.grad(loss(value),value)
    with torch.no_grad():fd=(loss(value+1e-5)-loss(value-1e-5))/(2e-5)
    torch.testing.assert_close(grad,fd,rtol=1e-7,atol=1e-9)
    assert abs(grad)>1e-6


def test_nonuniform_mesh_plane_gradient():
    p=scene(periodic=True);p.monitors=p.monitors[:1]
    p.region.mesh_type='explicit'
    nodes=[]
    for span,count in ((1.6,16),(1.5,15)):
        u=np.linspace(-1,1,count+1)
        nodes.append(tuple(span/2*(.8*u+.2*u**3)))
    p.region.mesh_coordinates=(*nodes,(-.7,.7))
    p.region.material_sampling='yee'
    eps=torch.full(p.region.shape,1.6,dtype=torch.float64,requires_grad=True)
    frequency=torch.tensor([.035/p.region.time_step],dtype=eps.dtype)
    expected=reference(p,eps,frequency)[p.monitors[0].id]/p.region.time_step
    expected_grad,=torch.autograd.grad(expected.abs().square().sum(),eps)
    result=DifferentiablePlaneSimulation(p)(eps,frequency)[p.monitors[0].id]
    actual=result.fields/p.region.time_step
    gradient,=torch.autograd.grad(actual.abs().square().sum(),eps)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(gradient,expected_grad,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('internal',[False,True])
def test_mutated_fixed_configuration_is_rejected(internal):
    p=scene()
    model=DifferentiablePlaneSimulation(p)
    target=model.model.project if internal else model.project
    target.sources[0].amplitude=2
    with pytest.raises(ValueError,match='Rebuild'):
        model(torch.ones(p.region.shape,dtype=torch.float64),[1e12])


def graded_scene(**changes):
    from torchfdtd import Project,Region,Source,SpectrumSettings
    region=dict(dimension='3d',size=(3.2,3.2,3.2),mesh=.1,mesh_type='graded',material_sampling='yee',
                mesh_max=.3,mesh_ppw=6,pml_cells=3,steps=10,backend='cpu',precision='float64')
    return Project(region=Region(**(region|changes)),sources=[Source(id='source',center=(-.4,0,0),wavelength=3)],
        monitors=[FieldMonitor(id='plane',center=(.4,0,0),size=(0,1,1),
            spectrum=SpectrumSettings(sampling='custom',custom_frequencies_hz=[1e14],apodization='none'))])


def test_internal_solver_keeps_the_auto_refined_plane_mesh(monkeypatch):
    from torchfdtd.solver import field_axes
    p=graded_scene()
    assert p.region.shape==(30,30,30)
    model=DifferentiablePlaneSimulation(p)
    inner=model.model.project.region
    assert inner.shape==p.region.shape and inner.time_step==p.region.time_step
    for planned,solved in zip(p.region.mesh_nodes,inner.mesh_nodes):
        np.testing.assert_array_equal(solved,planned)
    for c in COMPONENTS:
        for planned,solved in zip(field_axes(p.region,c),field_axes(inner,c)):
            np.testing.assert_array_equal(solved,planned)
    assert model.project.model_dump()==p.model_dump()
    # Without frozen refinements the point monitor re-meshes the solver.
    monkeypatch.setattr('torchfdtd.adjoint_planes.freeze_refinements',lambda project:project.model_copy(deep=True))
    with pytest.raises(ValueError,match='plane plan on the x axis'):DifferentiablePlaneSimulation(p)
    # Identical nodes are insufficient if the solver time step differs.
    def shortened(project):
        from torchfdtd import freeze_refinements
        frozen=freeze_refinements(project);frozen.region.time_step_override=.5*project.region.time_step
        return frozen
    monkeypatch.setattr('torchfdtd.adjoint_planes.freeze_refinements',shortened)
    with pytest.raises(ValueError,match='time step'):DifferentiablePlaneSimulation(p)
    # Identical nodes and time step are insufficient if Yee positions differ.
    monkeypatch.undo()
    def shifted(region,component):
        axes=field_axes(region,component)
        return axes if region.mesh_auto_refine else [axes[0]+1e-3,axes[1],axes[2]]
    monkeypatch.setattr('torchfdtd.adjoint_planes.field_axes',shifted)
    with pytest.raises(ValueError,match='Ex sampling positions differ from the plane plan on the x axis'):
        DifferentiablePlaneSimulation(p)


def test_plane_signature_tracks_realized_mesh_not_placement_or_structure():
    from torchfdtd import Structure,freeze_refinements
    from torchfdtd.adjoint_planes import _plane_signature,_run_fingerprint
    p=graded_scene();p.structures=[Structure(center=(.9,0,0),size=(.2,.2,.2))]
    q=graded_scene();q.structures=[Structure(center=(.9,0,0),size=(.2,2,.2))]
    p=type(p).model_validate(p.model_dump());q=type(q).model_validate(q.model_dump())
    assert p.region.model_dump()==q.region.model_dump()
    assert (p.region.shape,q.region.shape)==((31,30,30),(31,32,30))
    assert _plane_signature(p)!=_plane_signature(q) and _run_fingerprint(p)!=_run_fingerprint(q)
    placed=graded_scene(backend='cuda',cuda_kernel='fused',cuda_monitor_kernel='fused',memory_mode='streamed')
    placed.structures=list(p.structures);placed=type(p).model_validate(placed.model_dump())
    assert _plane_signature(placed)==_plane_signature(p) and _run_fingerprint(placed)==_run_fingerprint(p)
    # A scatterer and its air reference on the frozen mesh stay compatible,
    # and so does the unfrozen scatterer, while the full fingerprint differs.
    frozen=freeze_refinements(p);air=frozen.model_copy(deep=True);air.structures=[]
    air=type(p).model_validate(air.model_dump())
    for a,b in zip(frozen.region.mesh_nodes,air.region.mesh_nodes):np.testing.assert_array_equal(a,b)
    assert _plane_signature(air)==_plane_signature(frozen)==_plane_signature(p)
    assert _run_fingerprint(air)!=_run_fingerprint(frozen)==_run_fingerprint(p)
    model=DifferentiablePlaneSimulation(p)
    plane=model(torch.ones(p.region.shape,dtype=torch.float64),[1e14])['plane']
    assert (plane.run_signature,plane.run_fingerprint)==(model.signature,model.fingerprint)==(_plane_signature(p),_run_fingerprint(p))


def test_reference_fingerprint_ignores_ids_names_and_display_options():
    from torchfdtd import Structure,BoundaryFace
    from torchfdtd.adjoint_planes import _plane_signature,_run_fingerprint
    base=graded_scene();base.structures=[Structure(center=(.9,0,0),size=(.2,.2,.2))]
    base=type(base).model_validate(base.model_dump())
    signature,fingerprint=_plane_signature(base),_run_fingerprint(base)
    cosmetic=base.model_copy(deep=True)
    for item in cosmetic.structures+cosmetic.sources+cosmetic.monitors:item.id='new-'+item.id;item.name='renamed'
    cosmetic.name='Renamed project';cosmetic.materials[2].color='#123456'
    cosmetic.region.field='Hy';cosmetic.region.slice_axis='y';cosmetic.region.slice_position=.3
    cosmetic.region.snapshot_interval=7;cosmetic.region.complex_display='magnitude'
    cosmetic=type(base).model_validate(cosmetic.model_dump())
    assert cosmetic.model_dump()!=base.model_dump()
    assert _plane_signature(cosmetic)==signature and _run_fingerprint(cosmetic)==fingerprint
    def changed(**region):
        other=base.model_copy(deep=True)
        for key,value in region.items():setattr(other.region,key,value)
        return type(base).model_validate(other.model_dump())
    # A different mesh, time step, duration, source or exterior is rejected.
    variants=dict(mesh=changed(mesh=.08),time_step=changed(courant_factor=.8),duration=changed(steps=12),
                  background=changed(background_index=1.2),bloch=changed(bloch_phase=(0,.3,0),
                      boundaries=base.region.boundaries.model_copy(update=dict(y_min=BoundaryFace(kind='bloch'),y_max=BoundaryFace(kind='bloch')))),
                  pml=changed(boundaries=base.region.boundaries.model_copy(update=dict(x_min=BoundaryFace(kappa=3)))))
    source=base.model_copy(deep=True);source.sources[0].wavelength=2.5;variants['wavelength']=type(base).model_validate(source.model_dump())
    source=base.model_copy(deep=True);source.sources[0].phase=30;variants['phase']=type(base).model_validate(source.model_dump())
    source=base.model_copy(deep=True);source.sources[0].theta=45;variants['polarization']=type(base).model_validate(source.model_dump())
    monitor=base.model_copy(deep=True);monitor.monitors[0].spatial_interpolation='nearest';variants['interpolation']=type(base).model_validate(monitor.model_dump())
    for name,other in variants.items():
        assert _plane_signature(other)!=signature,name
        assert _run_fingerprint(other)!=fingerprint,name
