"""Independent scalar-root and six-field boundary tests for continuum oracles."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import jv, kve

from benchmarks.open_mode_oracles import solve_slab, solve_fiber_he11


@pytest.mark.parametrize('polarization',['TE','TM'])
def test_slab_continuity_dispersion_scaling_and_limits(polarization):
    mode=solve_slab(polarization=polarization)
    k=2*np.pi/mode.wavelength_um
    assert mode.n_clad<mode.neff<mode.n_core and mode.root_residual<1e-12
    assert mode.beta_per_um**2+mode.transverse_k_per_um**2==pytest.approx((k*mode.n_core)**2)
    assert mode.beta_per_um**2-mode.decay_per_um**2==pytest.approx((k*mode.n_clad)**2)
    a=mode.thickness_um/2
    f=mode.fields(np.array([a-1e-10,a+1e-10]))
    # Tangential fields Ey,Ez,Hy,Hz and normal D/B must be continuous.
    np.testing.assert_allclose(f[0,[1,2,4,5]],f[1,[1,2,4,5]],rtol=2e-8,atol=2e-9)
    assert mode.n_core**2*f[0,0]==pytest.approx(mode.n_clad**2*f[1,0],rel=2e-8,abs=2e-9)
    assert f[0,3]==pytest.approx(f[1,3],rel=2e-8,abs=2e-9)
    scaled=solve_slab(wavelength_um=4.65,thickness_um=2.4,polarization=polarization)
    assert scaled.neff==pytest.approx(mode.neff,rel=1e-13)
    np.testing.assert_allclose(scaled.fields(np.array([-.7,.1,.9])*3),mode.fields([-.7,.1,.9]),rtol=1e-13,atol=1e-13)
    thin=solve_slab(thickness_um=.01,polarization=polarization)
    thick=solve_slab(thickness_um=20,polarization=polarization)
    assert thin.neff-1.4<.001 and 1.8-thick.neff<.001
    assert solve_slab(polarization='TE').neff>solve_slab(polarization='TM').neff


def test_fiber_full_vector_boundary_conditions_and_confinement():
    mode=solve_fiber_he11()
    assert mode.root_residual<1e-12 and mode.boundary_residual<1e-12
    assert mode.n_clad<mode.neff<mode.n_core
    angle=np.linspace(0,2*np.pi,19,endpoint=False)
    x=mode.radius_um*np.cos(angle);y=mode.radius_um*np.sin(angle)
    core=mode.fields(x,y,side='core');clad=mode.fields(x,y,side='cladding')
    def tangent(f):
        return np.stack((-np.sin(angle)*f[:,0]+np.cos(angle)*f[:,1],f[:,2],
                         -np.sin(angle)*f[:,3]+np.cos(angle)*f[:,4],f[:,5]),axis=-1)
    np.testing.assert_allclose(tangent(core),tangent(clad),rtol=2e-12,atol=2e-12)
    for start,scale_core,scale_clad in ((0,mode.n_core**2,mode.n_clad**2),(3,1,1)):
        left=scale_core*(np.cos(angle)*core[:,start]+np.sin(angle)*core[:,start+1])
        right=scale_clad*(np.cos(angle)*clad[:,start]+np.sin(angle)*clad[:,start+1])
        np.testing.assert_allclose(left,right,rtol=2e-12,atol=2e-12)
    assert np.linalg.norm(core[:,2])>0 and np.linalg.norm(core[:,5])>0
    assert np.isfinite(mode.fields(0,0)).all()
    for angle in (0,.7,2.2):
        np.testing.assert_allclose(mode.fields(1e-9*np.cos(angle),1e-9*np.sin(angle)),mode.fields(0,0),atol=1e-8)
    def power(r):
        f=mode.fields(r,0)
        return np.pi*r*np.real(f[0]*f[4].conjugate()-f[1]*f[3].conjugate())
    total=quad(power,0,mode.radius_um)[0]+quad(power,mode.radius_um,mode.radius_um+20/mode.decay_per_um)[0]
    assert total>0
    outer=mode.radius_um+10/mode.decay_per_um
    tail=quad(power,outer,outer+10/mode.decay_per_um)[0]
    assert 0<tail/total<1e-7


def test_fiber_six_component_maxwell_equations_independently_differenced():
    mode=solve_fiber_he11();h=2e-6;k=2*np.pi/mode.wavelength_um
    # Avoid interfaces. This checks all recovered components, not its determinant.
    for x,y in ((.1,.13),(-.2,.04),(.6,.2),(-.5,-.4)):
        f=mode.fields(x,y)
        dx=(mode.fields(x+h,y)-mode.fields(x-h,y))/(2*h)
        dy=(mode.fields(x,y+h)-mode.fields(x,y-h))/(2*h)
        dz=1j*mode.beta_per_um*f
        curls=np.stack((dy[2::3]-dz[1::3],dz[0::3]-dx[2::3],dx[1::3]-dy[0::3]),axis=-1)
        eps=mode.n_core**2 if x*x+y*y<mode.radius_um**2 else mode.n_clad**2
        np.testing.assert_allclose(curls[0],1j*k*f[3:],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(curls[1],-1j*k*eps*f[:3],rtol=1e-8,atol=1e-8)


def test_fiber_scale_weak_guidance_and_large_v_invariants():
    mode=solve_fiber_he11()
    scaled=solve_fiber_he11(wavelength_um=4.65,radius_um=1.2)
    assert scaled.beta_per_um*3==pytest.approx(mode.beta_per_um,rel=1e-13)
    np.testing.assert_allclose(scaled.fields(.2*3,.3*3),mode.fields(.2,.3),rtol=1e-12,atol=1e-12)
    # LP01 used only as an independent weak-contrast LIMIT, not as fiber solver.
    v=2.;nclad=1.4;k=2*np.pi/1.55
    def lp(u):
        w=np.sqrt(v*v-u*u)
        return u*jv(1,u)/jv(0,u)-w*kve(1,w)/kve(0,w)
    scalar_u=brentq(lp,1e-8,v*(1-1e-12),xtol=1e-14)
    errors=[]
    for delta in (.01,.001,.0001):
        ncore=nclad+delta
        radius=v/(k*np.sqrt(ncore*ncore-nclad*nclad))
        value=solve_fiber_he11(n_core=ncore,n_clad=nclad,radius_um=radius)
        errors.append(abs(value.u-scalar_u))
    assert errors[1]<errors[0]/8 and errors[2]<errors[1]/8 and errors[-1]<1e-4
    last=0
    for v in (8,16,32):
        value=solve_fiber_he11(radius_um=v/(k*np.sqrt(1.8**2-1.4**2)))
        assert last<value.u<2.4048255576957728
        last=value.u
    assert 2.4048255576957728-last<.1


@pytest.mark.parametrize('kwargs',[dict(radius_um=0),dict(n_core=1.3),dict(wavelength_um=float('nan')),dict(radius_um=.01)])
def test_invalid_or_unresolved_fiber_parameters_rejected(kwargs):
    with pytest.raises(ValueError): solve_fiber_he11(**kwargs)
