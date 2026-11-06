"""Full-vector physical limits and fixed-mode plane objective checks, CPU only."""
import math
import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from torchfdtd.mode_ports import solve_waveguide_modes, mode_power_overlap, normalized_mode_power, C0


def slab_neff(polarization, thickness=.5, wavelength=1.55, core=4., clad=2.25):
    a=thickness/2
    k=2*np.pi/wavelength
    v=k*a*np.sqrt(core-clad)
    factor=1 if polarization=='TE' else core/clad
    root=brentq(lambda u:u*np.tan(u)-factor*np.sqrt(v*v-u*u),1e-10,min(v,np.pi/2)-1e-10)
    return np.sqrt(core-(root/(a*k))**2)


def slab_modes(h,length=4.):
    def epsilon(u,v):
        return np.where(abs(u)<.25-1e-9,4.,np.where(abs(abs(u)-.25)<1e-9,3.125,2.25))
    return solve_waveguide_modes(epsilon,shape=(round(length/h),4),spacing_um=(h,.05),
                                wavelength_um=1.55,normal='z')


def test_homogeneous_vector_fourier_dispersion_residual_and_power_orthogonality():
    shape=(20,16)
    modes=solve_waveguide_modes(2.25,shape=shape,spacing_um=(.1,.1),wavelength_um=1.55,num_modes=6)
    k=2*np.pi/1.55
    expected=[1.5]*2+[math.sqrt(2.25-(2/.1*math.sin(np.pi/20)/k)**2)]*4
    np.testing.assert_allclose([m.neff for m in modes],expected,rtol=2e-5,atol=2e-5)
    for mode in modes:
        assert mode.fields.dtype==np.complex64
        assert mode.power()==pytest.approx(1,rel=1e-6)
        assert mode.backward().power()==pytest.approx(-1,rel=1e-6)
        assert mode.eigenpair_residual<1e-4 and mode.maxwell_residual<1e-4
    gram=np.array([[mode_power_overlap(a,b) for b in modes] for a in modes])
    np.testing.assert_allclose(gram,np.eye(len(modes)),rtol=2e-4,atol=2e-4)
    assert max(np.linalg.norm(m.fields[...,0]) for m in modes[2:])>1e4  # longitudinal Ex for normal x


@pytest.mark.parametrize('normal',['x','y','z'])
def test_component_coordinates_match_native_yee_field_axes(normal):
    from torchfdtd import Region,Boundaries,BoundaryFace
    from torchfdtd.solver import field_axes
    mode=solve_waveguide_modes(2.25,shape=(12,14),spacing_um=(.1,.1),wavelength_um=1.55,
                              normal=normal,num_modes=1)[0]
    size=[1.6,1.6,1.6]
    u,v=mode.transverse_axes
    size[u],size[v]=1.2,1.4
    region=Region(dimension='3d',size=tuple(size),mesh=.1,pml_cells=3,steps=10,material_sampling='yee',
        boundaries=Boundaries(**{name:BoundaryFace(kind='periodic') for name in ('x_min','x_max','y_min','y_max','z_min','z_max')}))
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        native=field_axes(region,component)
        got=mode.component_axes(component)
        for a,b in zip(got,(native[u],native[v])):
            np.testing.assert_allclose(a,b,atol=1e-14)


def test_slab_te_tm_mesh_limit_box_convergence_and_evanescent_confinement():
    exact=np.array([slab_neff('TE'),slab_neff('TM')])
    solutions=[slab_modes(h) for h in (.1,.05,.025)]
    errors=np.abs(np.array([[m.neff for m in row] for row in solutions])-exact)
    assert (np.diff(errors,axis=0)<0).all()
    assert errors[-1].max()<5e-4
    boxes=[slab_modes(.05,length) for length in (2.,4.,6.)]
    neff=np.array([[m.neff for m in row] for row in boxes])
    assert np.abs(neff[2]-neff[1]).max()<2e-6
    assert (np.abs(neff[2]-neff[1])<np.abs(neff[1]-neff[0])).all()
    for mode in boxes[-1]:
        assert 1.5<mode.neff<2.
        intensity=(np.abs(mode.fields)**2).sum(-1)
        edge=intensity[:3].sum()+intensity[-3:].sum()
        assert edge/intensity.sum()<1e-7
        assert mode.maxwell_residual<2e-5
    # Both polarizations are obtained from the coupled vector eigenproblem.
    te,tm=solutions[-1]
    assert np.linalg.norm(te.fields[...,1])>100*np.linalg.norm(te.fields[...,0])
    assert np.linalg.norm(tm.fields[...,0])>100*np.linalg.norm(tm.fields[...,1])
    assert np.linalg.norm(tm.fields[...,2])>.1*np.linalg.norm(tm.fields[...,:3])


def test_nonseparable_dielectric_core_has_all_six_components_and_orthogonal_modes():
    modes=solve_waveguide_modes(lambda u,v:np.where((abs(u)<.4)&(abs(v)<.25),4.,2.25),
        shape=(40,32),spacing_um=(.1,.1),wavelength_um=1.55)
    for mode in modes:
        norms=np.array([np.linalg.norm(mode.fields[...,c]) for c in range(6)])
        assert norms.min()>.01*norms.max()
        assert mode.maxwell_residual<2e-5
    assert abs(mode_power_overlap(*modes))<1e-5


def test_fixed_mode_directional_plane_power_and_both_torch_graphs():
    from torchfdtd.adjoint_planes import DifferentiablePlaneResult
    mode=solve_waveguide_modes(2.25,shape=(8,6),spacing_um=(.1,.1),wavelength_um=1.55,num_modes=1)[0]
    u,v=np.meshgrid(-.4+(np.arange(8)+.5)*.1,-.3+(np.arange(6)+.5)*.1,indexing='ij')
    points=np.stack([np.zeros(u.size),u.ravel(),v.ravel()],axis=1)
    basis=torch.tensor(mode.sample_plane(points))
    reverse=torch.tensor(mode.backward().sample_plane(points))
    a=torch.tensor(1.7,requires_grad=True)
    b=torch.tensor(.3,requires_grad=True)
    incident=torch.tensor(.8,requires_grad=True)
    args=dict(frequency_hz=torch.tensor([C0/(1.55e-6)]),points_um=torch.tensor(points,dtype=torch.float32),
        weights=torch.full((len(points),),1e-14),shape=(1,8,6),normal='x',run_signature='fixed',report={})
    reference=DifferentiablePlaneResult(fields=1e-22*incident*basis[None],**args)
    sample=DifferentiablePlaneResult(fields=1e-22*(a*basis[None]+b*reverse[None]),**args)
    forward=normalized_mode_power(sample,reference,mode)
    backward=normalized_mode_power(sample,reference,mode,direction='backward')
    torch.testing.assert_close(forward,(a/incident).square()[None],rtol=2e-5,atol=1e-6)
    torch.testing.assert_close(backward,(b/incident).square()[None],rtol=2e-5,atol=1e-6)
    gradients=torch.autograd.grad((forward+backward).sum(),(a,b,incident))
    expected=(2*a/incident**2,2*b/incident**2,-2*(a*a+b*b)/incident**3)
    for got,want in zip(gradients,expected):
        torch.testing.assert_close(got,want,rtol=2e-5,atol=1e-5)


def test_invalid_material_and_mode_contracts():
    args=dict(shape=(6,6),spacing_um=(.1,.1),wavelength_um=1.55)
    with pytest.raises(ValueError,match='real positive'):
        solve_waveguide_modes(2+1j,**args)
    with pytest.raises(ValueError,match='finite and positive'):
        solve_waveguide_modes(-2.,**args)
    with pytest.raises(ValueError,match='precision'):
        solve_waveguide_modes(2.,precision='complex64',**args)



def test_yee_to_plane_interpolation_matches_native_maps_and_positive_time_dft():
    from torchfdtd import Region,Boundaries,BoundaryFace
    from torchfdtd.field_monitors import interpolation_map
    from torchfdtd.adjoint_planes import _PlaneSpectrum
    modes=solve_waveguide_modes(2.25,shape=(16,12),spacing_um=(.1,.1),wavelength_um=1.55,num_modes=4)
    mode=modes[-1]
    region=Region(dimension='3d',size=(1.2,1.6,1.2),mesh=.1,pml_cells=3,steps=32,
        precision='float32',material_sampling='yee',
        boundaries=Boundaries(**{name:BoundaryFace(kind='periodic') for name in ('x_min','x_max','y_min','y_max','z_min','z_max')}))
    u,v=np.meshgrid(-.8+(np.arange(16)+.37)*.1,-.6+(np.arange(12)+.29)*.1,indexing='ij')
    points=np.stack([np.full(u.size,.123),u.ravel(),v.ravel()],axis=1)
    sampled=mode.sample_plane(points)
    native=np.empty(region.shape+(3,),dtype=np.complex64)
    for c,component in enumerate(('Ex','Ey','Ez','Hx','Hy','Hz')):
        native[...,c%3]=mode.fields[None,...,c]
        index,weight=interpolation_map(region,component,points)
        want=(native.reshape(-1)[index]*weight).sum(0)
        np.testing.assert_allclose(sampled[:,c],want,rtol=2e-5,atol=.5)
    # Native positive-time DFT recovers exp(-i omega t) phasors for both E and
    # H at their distinct time samples. This tests convention, not injection.
    frequency=torch.tensor([C0/(1.55e-6)])
    observation=_PlaneSpectrum(torch.ones(()),region,('Ex','Ey','Ez','Hx','Hy','Hz'),
                              frequency,points=1,maps=0)
    phasor=torch.tensor(sampled[0])
    for family,columns in [('E',slice(0,3)),('H',slice(3,6))]:
        times=(torch.arange(region.steps)+1+(.5 if family=='H' else 0))*region.time_step
        trace=torch.exp(-2j*torch.pi*frequency[0]*times)[:,None]*phasor[None,columns]
        recovered=observation.kernel(0,region.steps,family)@trace/(region.steps*region.time_step)
        torch.testing.assert_close(recovered[0],phasor[columns],rtol=2e-5,atol=.3)

def test_plane_field_vjp_finite_difference_leakage_and_quadrature_rejection():
    from dataclasses import replace
    from torchfdtd.adjoint_planes import DifferentiablePlaneResult
    modes=solve_waveguide_modes(2.25,shape=(8,6),spacing_um=(.1,.1),wavelength_um=1.55,num_modes=2)
    u,v=np.meshgrid(-.4+(np.arange(8)+.5)*.1,-.3+(np.arange(6)+.5)*.1,indexing='ij')
    points=np.stack([np.zeros(u.size),u.ravel(),v.ravel()],axis=1)
    basis=torch.tensor(modes[0].sample_plane(points))[None]
    reference=DifferentiablePlaneResult(fields=basis*1e-22,frequency_hz=torch.tensor([C0/1.55e-6]),
        points_um=torch.tensor(points,dtype=torch.float32),weights=torch.full((len(points),),1e-14),
        shape=(1,8,6),normal='x',run_signature='fixed',report={})
    reverse=replace(reference,fields=torch.tensor(modes[0].backward().sample_plane(points))[None]*1e-22)
    cross=replace(reference,fields=torch.tensor(modes[1].sample_plane(points))[None]*1e-22)
    assert float(normalized_mode_power(reference,reference,modes[0],direction='backward'))<1e-10
    assert float(normalized_mode_power(reverse,reference,modes[0]))<1e-10
    assert float(normalized_mode_power(cross,reference,modes[0]))<1e-10
    field=(reference.fields*1.3+reverse.fields*.2).detach().requires_grad_()
    def objective(value):
        plane=replace(reference,fields=value)
        return (normalized_mode_power(plane,reference,modes[0])+.3*normalized_mode_power(
            plane,reference,modes[0],direction='backward')).sum()
    gradient,=torch.autograd.grad(objective(field),field)
    rng=torch.Generator().manual_seed(293)
    direction=torch.complex(torch.randn(field.shape,generator=rng),torch.randn(field.shape,generator=rng))*field.detach().abs().max()
    h=1e-3
    finite=(objective(field.detach()+h*direction)-objective(field.detach()-h*direction))/(2*h)
    adjoint=(gradient.conj()*direction).sum().real
    torch.testing.assert_close(adjoint,finite,rtol=3e-3,atol=2e-4)
    # Same total area is insufficient: shifted, duplicated or reweighted points fail.
    shifted=reference.points_um.clone()
    shifted[:,1]+=.1
    duplicate=reference.points_um.clone()
    duplicate[0]=duplicate[1]
    for bad in (shifted,duplicate):
        altered=replace(reference,points_um=bad)
        with pytest.raises(ValueError,match='quadrature'):
            normalized_mode_power(altered,altered,modes[0])
    weights=reference.weights.clone()
    weights[0]*=1.5
    weights[1]*=.5
    altered=replace(reference,weights=weights)
    with pytest.raises(ValueError,match='quadrature'):
        normalized_mode_power(altered,altered,modes[0])


def test_degenerate_polarization_basis_is_canonical_and_start_independent(monkeypatch):
    """A uniform section has a doubly degenerate fundamental pair. Its returned
    basis must not depend on the ARPACK Krylov sequence, which differs between
    BLAS builds and CPUs: the two ports of an interface network on another
    workstation received orthogonal polarizations and lost all transmission."""
    import torchfdtd.mode_ports as module
    def polarization(mode):
        power=np.abs(mode.fields[...,:3].astype(np.complex128))**2
        return power.sum(axis=(0,1))/power.sum()
    reference={}
    for eps in (1.,1.44):
        modes=solve_waveguide_modes(eps,shape=(5,5),spacing_um=(.2,.2),wavelength_um=1.55,normal='x',num_modes=2)
        assert math.isclose(modes[0].beta_per_um,modes[1].beta_per_um,rel_tol=1e-6)
        np.testing.assert_allclose(polarization(modes[0]),[0.,1.,0.],atol=1e-6)
        np.testing.assert_allclose(polarization(modes[1]),[0.,0.,1.],atol=1e-6)
        reference[eps]=modes
    assert abs(mode_power_overlap(reference[1.][0],reference[1.44][0]))>.999
    assert abs(mode_power_overlap(reference[1.][0],reference[1.44][1]))<1e-6
    # Rotate the degenerate ARPACK pair by a random unitary before selection,
    # imitating a different Krylov sequence, and require the same modes.
    original=module.eigs
    rng=np.random.default_rng(7)
    def rotated_eigs(*args,**kwargs):
        values,vectors=original(*args,**kwargs)
        vectors=vectors.copy()
        order=np.argsort(np.abs(values-values[0]))
        pair=order[:2]
        if abs(values[pair[0]]-values[pair[1]])<1e-4*abs(values[pair[0]]):
            theta,phi=rng.uniform(0,2*np.pi,2)
            unitary=np.array([[np.cos(theta),-np.sin(theta)*np.exp(1j*phi)],
                              [np.sin(theta)*np.exp(-1j*phi),np.cos(theta)]])
            vectors[:,pair]=vectors[:,pair]@unitary
        return values,vectors
    monkeypatch.setattr(module,'eigs',rotated_eigs)
    for eps,modes in reference.items():
        again=solve_waveguide_modes(eps,shape=(5,5),spacing_um=(.2,.2),wavelength_um=1.55,normal='x',num_modes=2)
        for first,second in zip(modes,again):
            assert math.isclose(first.beta_per_um,second.beta_per_um,rel_tol=1e-6)
            np.testing.assert_allclose(second.fields,first.fields,rtol=2e-4,atol=2e-4*np.abs(first.fields).max())
