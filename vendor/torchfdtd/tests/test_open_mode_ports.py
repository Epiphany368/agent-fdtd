"""Fixed CPML modes against independent continuum slab and Maxwell equations."""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pytest
from torchfdtd import Region
from torchfdtd.open_mode_ports import solve_open_waveguide_modes
from benchmarks.open_mode_oracles import solve_slab


def slab_region(h=.1):
    faces={axis+'_'+side: dict(kind='pml', layers=round(.8/h) if axis=='x' else 3)
           for axis in 'xyz' for side in ('min','max')}
    for side in ('min','max'): faces['y_'+side]=dict(kind='periodic')
    return Region(dimension='3d', size=(6.4,.5,1.6), mesh=h,
                  mesh_steps=(h,.1,h), steps=10, boundaries=faces,
                  material_sampling='yee', precision='float32', backend='cpu')


def material(x,y):
    # Strict interfaces: nodal samples exactly on the boundary are cladding.
    return np.where(np.abs(x)<.4-1e-8, 1.8**2, 1.4**2)


@lru_cache(None)
def modes(h=.1):
    return solve_open_waveguide_modes(material,region=slab_region(h),normal='z',
             wavelength_um=1.55,cladding_epsilon=1.4**2,num_modes=2)


def test_budget_before_material_sampling(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('allocated before admission')
    monkeypatch.setattr('torchfdtd.open_mode_operators.prepare_open_mode_operators',forbidden)
    with pytest.raises(ValueError,match='reservation exceeds'):
        solve_open_waveguide_modes(forbidden,region=slab_region(),normal='z',
            wavelength_um=1.55,cladding_epsilon=1.4**2,mode_budget_bytes=1)


def test_fixed_cladding_rejected_before_operator_assembly(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('assembled invalid material')
    monkeypatch.setattr('torchfdtd.open_mode_operators.prepare_open_mode_operators',forbidden)
    with pytest.raises(ValueError,match='one-cell collar'):
        solve_open_waveguide_modes(3.24,region=slab_region(),normal='z',
            wavelength_um=1.55,cladding_epsilon=1.96)


def test_independent_te_tm_continuum_refinement():
    references=np.array([solve_slab(polarization=p).beta_per_um for p in ('TE','TM')])
    errors=[]; propagation=[]
    for h in (.1,.05,.025):
        found=modes(h)
        actual=np.array([m.beta_per_um.real for m in found])
        errors.append(np.abs(actual/references-1)); propagation.append(actual)
        assert all(m.maxwell_residual<2e-4 for m in found)
        assert all(m.diagnostics['collar_energy_fraction']<1e-4 for m in found)
    errors=np.asarray(errors)
    # Staircase interface and bulk-dispersion errors may cancel, so require
    # shrinking successive corrections, not monotonic absolute error.
    corrections=np.abs(np.diff(propagation,axis=0))
    assert np.all(corrections[1]<corrections[0]), corrections
    assert np.all(errors[-1]<errors[0]), errors
    assert np.max(errors[-1])<.005, errors


def test_six_field_maxwell_and_open_sampling_contract():
    from torchfdtd.open_mode_operators import prepare_open_mode_operators
    ops=prepare_open_mode_operators(slab_region(),normal='z',wavelength_um=1.55)
    for mode in modes():
        ex,ey,ez,hx,hy,hz=mode.fields.reshape(-1,6).T
        b=mode.beta_tilde_per_um;k=ops.temporal_k_per_um
        ce=np.array([ops.vp@ez-1j*b*ey,1j*b*ex-ops.up@ez,ops.up@ey-ops.vp@ex])
        ch=np.array([ops.vm@hz-1j*b*hy,1j*b*hx-ops.um@hz,ops.um@hy-ops.vm@hx])
        eh=np.array([hx,hy,hz]); ee=mode.sampled_epsilon.reshape(-1,3).T*np.array([ex,ey,ez])
        assert np.linalg.norm(ce-1j*k*eh)/np.linalg.norm(k*eh)<2e-4
        assert np.linalg.norm(ch+1j*k*ee)/np.linalg.norm(k*ee)<2e-4
        assert mode.fields.dtype==np.complex64 and not mode.fields.flags.writeable
        assert not mode.sampled_epsilon.flags.writeable
        for value in (mode.fields, mode.sampled_epsilon, mode.backward().fields):
            with pytest.raises(ValueError): value.setflags(write=True)
        assert mode.power()==pytest.approx(1,rel=2e-6)
        assert mode.backward().power()==pytest.approx(-1,rel=2e-6)
        assert mode.backward().beta_per_um==-mode.beta_per_um
        a=mode.sample_plane(np.array([[0,.05,0],[0,.55,0]]))
        np.testing.assert_allclose(a[0],a[1],rtol=1e-6,atol=.01)
        with pytest.raises(ValueError,match='outside'):
            mode.sample_plane(np.array([[20,0,0]]))
    # TE has longitudinal H, TM longitudinal E: this is a six-field solve.
    assert np.linalg.norm(modes()[0].fields[...,5])>0
    assert np.linalg.norm(modes()[1].fields[...,2])>0


def test_physical_rectangle_quadrature():
    mode=modes()[0]
    coords=[lo+(np.arange(n)+.5)*(hi-lo)/n for (lo,hi),n in zip(mode.physical_bounds_um,(96,4))]
    x,y=np.meshgrid(*coords,indexing='ij');points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    area=np.prod([hi-lo for lo,hi in mode.physical_bounds_um])*1e-12
    plane=SimpleNamespace(normal='z',points_um=points,weights=np.full(x.size,area/x.size))
    assert mode.validate_quadrature(plane)>0
    plane.points_um=points.copy();plane.points_um[:,0]+=.01
    with pytest.raises(ValueError,match='physical rectangle'): mode.validate_quadrature(plane)


def _direct_harmonic_derivative(r, fields, axis, forward, wavelength):
    """Native differences plus explicit time-domain auxiliary recurrence."""
    n=fields.shape[axis];h=r.axis_steps[axis]
    derivative=np.zeros_like(fields,dtype=np.complex128)
    target=[slice(None)]*3;left=target.copy();right=target.copy()
    target[axis]=slice(0,n-1) if forward else slice(1,n)
    left[axis]=slice(0,n-1);right[axis]=slice(1,n)
    derivative[tuple(target)]=(fields[tuple(right)]-fields[tuple(left)])/h
    if r.boundaries.pair(axis)[0].kind=='periodic':
        target[axis]=-1 if forward else 0;left[axis]=-1;right[axis]=0
        derivative[tuple(target)]=(fields[tuple(right)]-fields[tuple(left)])/h
        return derivative
    b=np.ones(n,dtype=np.float32);c=np.zeros(n,dtype=np.float32);ki=np.ones(n,dtype=np.float32)
    for side,face in enumerate(r.boundaries.pair(axis)):
        layers=r.pml_layers(axis,side);base=0 if side==0 else n-layers
        for row in range(max(base,0 if forward else 1),min(base+layers,n-1 if forward else n)):
            depth=layers-(row-base)-(1 if forward else .5) if side==0 else row-base+(1 if forward else .5)
            rho=depth/(layers+1)
            sigma=face.sigma_scale*40*rho**face.polynomial/(layers+1)
            kappa=1+(face.kappa-1)*rho**face.polynomial
            alpha=face.alpha*(1-rho)**face.alpha_polynomial
            decay=np.exp(-(sigma/kappa+alpha)*r.rectangular_courant)
            denominator=sigma*kappa+alpha*kappa*kappa
            b[row]=decay;c[row]=0 if denominator==0 else (decay-1)*sigma/denominator;ki[row]=1/kappa
    shape=[1,1,1];shape[axis]=n
    b,c,ki=(v.reshape(shape) for v in (b,c,ki))
    omega_dt=2*np.pi*299792458*r.time_step/(wavelength*1e-6)
    psi=np.zeros_like(derivative)
    for step in range(4000):
        phase=np.exp(-1j*omega_dt*step);drive=derivative*phase
        psi=b*psi+c*drive
    return (ki*drive+psi)/phase


def test_independent_native_time_stagger_and_cpml_recurrence():
    r=slab_region();dt=r.time_step;c_dt=299792458*dt*1e6
    phase=np.exp(-2j*np.pi*299792458*dt/(1.55e-6))
    for forward in modes():
        for mode in (forward,forward.backward()):
            f=mode.fields
            xp=_direct_harmonic_derivative(r,f,0,True,1.55)
            xm=_direct_harmonic_derivative(r,f,0,False,1.55)
            yp=_direct_harmonic_derivative(r,f,1,True,1.55)
            ym=_direct_harmonic_derivative(r,f,1,False,1.55)
            # Actual longitudinal staggered exponential difference.
            z=(np.exp(.5j*mode.beta_per_um*r.axis_steps[2])-np.exp(-.5j*mode.beta_per_um*r.axis_steps[2]))/r.axis_steps[2]
            ce=np.stack((yp[...,2]-z*f[...,1],z*f[...,0]-xp[...,2],xp[...,1]-yp[...,0]),axis=-1)
            ch=np.stack((ym[...,5]-z*f[...,4],z*f[...,3]-xm[...,5],xm[...,4]-ym[...,3]),axis=-1)
            # E(n+1)-E(n), H(n+3/2)-H(n+1/2): native leapfrog.
            e_increment=(phase-1)*f[...,:3]
            h_increment=(phase-1)*np.sqrt(phase)*f[...,3:]
            e_rhs=c_dt*np.sqrt(phase)*ch/mode.sampled_epsilon
            h_rhs=-c_dt*phase*ce
            assert np.linalg.norm(e_increment-e_rhs)/np.linalg.norm(e_increment)<2e-4
            assert np.linalg.norm(h_increment-h_rhs)/np.linalg.norm(h_increment)<2e-4


def test_reactive_flux_rejects_phase_corrupted_magnetic_profile():
    from dataclasses import replace
    mode=modes()[0]
    coords=[lo+(np.arange(n)+.5)*(hi-lo)/n for (lo,hi),n in zip(mode.physical_bounds_um,(96,5))]
    x,y=np.meshgrid(*coords,indexing='ij');points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    area=np.prod([hi-lo for lo,hi in mode.physical_bounds_um])*1e-12
    plane=SimpleNamespace(normal='z',points_um=points,weights=np.full(x.size,area/x.size))
    assert mode.validate_quadrature(plane)>0
    corrupted=mode.fields.copy();corrupted[...,3:]*=np.exp(.01j)
    with pytest.raises(ValueError,match='reactive flux'):
        replace(mode,fields=corrupted).validate_quadrature(plane)


def test_fully_confined_fiber_two_mesh_continuum_comparison():
    from benchmarks.open_mode_oracles import solve_fiber_he11
    oracle=solve_fiber_he11(n_core=2.2,n_clad=1.,radius_um=.4,wavelength_um=1.55)
    for h in (.1,.08):
        r=Region(dimension='3d',size=(4.,4.,1.6),mesh=h,steps=10,
            boundaries={a+'_'+s:dict(kind='pml',layers=round(.8/h) if a!='z' else 3)
                        for a in 'xyz' for s in ('min','max')},
            material_sampling='yee',precision='float32',backend='cpu')
        mode=solve_open_waveguide_modes(lambda x,y:np.where(x*x+y*y<.4**2-1e-10,2.2**2,1.),
            region=r,normal='z',wavelength_um=1.55,cladding_epsilon=1.)[0]
        error=abs(mode.beta_per_um.real/oracle.beta_per_um-1)
        print(dict(h=h,beta=mode.beta_per_um,neff=mode.neff,oracle_beta=oracle.beta_per_um,
            relative_error=error,tail=mode.diagnostics['collar_energy_fraction'],
            residual=mode.maxwell_residual,reservation=mode.diagnostics['engineering_reservation_bytes']))
        assert error<.02
        assert mode.diagnostics['collar_energy_fraction']<1e-4


def test_live_host_admission_precedes_material_and_sparse_allocations(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('allocated before host admission')
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory',lambda:dict(available_bytes=1024))
    monkeypatch.setattr('torchfdtd.open_mode_operators.prepare_open_mode_operators',forbidden)
    with pytest.raises(ValueError,match='available host memory'):
        solve_open_waveguide_modes(forbidden,region=slab_region(),normal='z',
            wavelength_um=1.55,cladding_epsilon=1.96,mode_budget_bytes=100*1024**3)
