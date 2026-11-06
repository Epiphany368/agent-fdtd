"""Invariant-axis eigensystem and modal preparation; CPU, no time stepping."""
import subprocess
import sys
import types

import numpy as np
import pytest

from torchfdtd import Project, Region, Source
from torchfdtd.invariant_modes import solve_invariant_waveguide_modes
from torchfdtd.mode_ports import solve_waveguide_modes, mode_power_overlap
from torchfdtd.mode_injection import prepare_modal_launch
from torchfdtd.mode_branches import prepare_aperture_modal_launch

BASELINE = '26630ba'


def slab(y, z):
    return np.where(abs(y) < .25-1e-9, 4., np.where(abs(abs(y)-.25)<1e-9, 3.125, 2.25))


def solve(normal='x', spacing_z=.04):
    shape, spacing, origin = ((160, 1), (.025, spacing_z), (-2., -.1)) if normal == 'x' else (
        (1, 160), (spacing_z, .025), (-.1, -2.))
    material = slab if normal == 'x' else lambda z,x:slab(x,z)
    return solve_invariant_waveguide_modes(material, shape=shape, spacing_um=spacing,
        origin_um=origin, wavelength_um=1.55, normal=normal, precision='float64')


@pytest.mark.parametrize('normal', ['x', 'y'])
def test_1d_modes_equal_five_cell_zero_kz_sector(normal):
    modes = solve(normal)
    shape, spacing, origin = ((160,5),(.025,.04),(-2.,-.1)) if normal == 'x' else (
        (5,160),(.04,.025),(-.1,-2.))
    material = slab if normal == 'x' else lambda z,x:slab(x,z)
    reference = solve_waveguide_modes(material, shape=shape, spacing_um=spacing,
        origin_um=origin, wavelength_um=1.55, normal=normal, precision='float64')
    for mode, three in zip(modes, reference):
        assert mode.neff == pytest.approx(three.neff, rel=1e-10)
        expected = np.mean(three.fields, axis=1 if normal=='x' else 0, keepdims=True)*np.sqrt(.2e-6)
        np.testing.assert_allclose(mode.fields, expected, rtol=1e-8, atol=1e-7)
        assert mode.power() == pytest.approx(1., rel=1e-12)
        assert mode.backward().power() == pytest.approx(-1., rel=1e-12)
        assert max(mode.eigenpair_residual, mode.maxwell_residual) < 1e-10
    assert abs(mode_power_overlap(*modes)) < 1e-10


def test_invariant_spacing_does_not_change_operator_profile_or_power():
    for first, second in zip(solve(spacing_z=.04), solve(spacing_z=7.)):
        assert first.neff == second.neff
        np.testing.assert_array_equal(first.fields, second.fields)
        assert first.power() == second.power()
        points = np.array([[0., .12, 0.], [0., .12, 938.]])
        np.testing.assert_array_equal(first.sample_plane(points)[0], first.sample_plane(points)[1])


def test_noninvariant_material_fails_without_thin_box_fallback():
    with pytest.raises(ValueError, match='z-invariant'):
        solve_invariant_waveguide_modes(lambda y,z:2.+z*z, shape=(40,1),
            spacing_um=(.05,.04), origin_um=(-1.,-.1), wavelength_um=1.55)


def test_1d_te_tm_converge_to_independent_analytic_slab_dispersion():
    from scipy.optimize import brentq
    k=2*np.pi/1.55
    v=k*.25*np.sqrt(4.-2.25)
    exact=[]
    for factor in (1.,4./2.25):
        root=brentq(lambda u:u*np.tan(u)-factor*np.sqrt(v*v-u*u),1e-10,min(v,np.pi/2)-1e-10)
        exact.append(np.sqrt(4.-(root/(.25*k))**2))
    errors=[]
    for h in (.1,.05,.025):
        modes=solve_invariant_waveguide_modes(slab,shape=(round(4/h),1),spacing_um=(h,.04),
            origin_um=(-2.,-.1),wavelength_um=1.55,normal='x',precision='float64')
        errors.append(abs(np.array([m.neff for m in modes])-exact))
    assert (np.diff(errors,axis=0)<0).all()
    assert max(errors[-1])<5e-4


def test_float32_1d_keeps_single_precision_profiles():
    modes=solve_invariant_waveguide_modes(2.25,shape=(40,1),spacing_um=(.05,.04),
        origin_um=(-1.,-.1),wavelength_um=1.55,normal='x',precision='float32')
    for m in modes:
        assert m.fields.dtype==np.complex64
        assert m.power()==pytest.approx(1.,rel=1e-6)


def project(dimension, aperture=False, normal='x'):
    faces = {a+'_'+s:dict(kind='periodic') for a in ('y' if normal=='x' else 'x')
             for s in ('min','max')}
    if dimension == '3d':
        faces.update({a+'_'+s:dict(kind='periodic') for a in 'z' for s in ('min','max')})
    r = Region(dimension=dimension, size=(4.,4.,.2), mesh_steps=(.05,.05,.04),
        material_sampling='yee', precision='float64', steps=10, pml_cells=3, boundaries=faces)
    size = (0.,2. if aperture else 4.,.2) if normal=='x' else (2. if aperture else 4.,0.,.2)
    center = (-1.,0.,0.) if normal=='x' else (0.,-1.,0.)
    return Project(region=r, sources=[Source(id='source',kind='plane',normal=normal,
        center=center,size=size,wavelength=1.55,pulse_cycles=2)], monitors=[])


@pytest.mark.parametrize('normal',['x','y'])
@pytest.mark.parametrize('aperture',[False,True])
def test_both_modal_launches_use_1d_with_correct_source_shapes(normal,aperture):
    prepare = prepare_aperture_modal_launch if aperture else prepare_modal_launch
    p = project('2d',aperture,normal)
    material = (lambda y,z:np.where(abs(y)<.2,12.,2.25)) if normal=='x' else (
        lambda z,x:np.where(abs(x)<.2,12.,2.25))
    launch = prepare(p,material,num_modes=2)
    assert launch.mode.invariant_axis == 'z'
    assert launch.mode.power() == pytest.approx(1.,rel=1e-12)
    for _,_,wave,spatial in launch.terms:
        assert len(wave) == 10
        assert spatial.shape[2] == 1
        assert np.isfinite(spatial).all()


def baseline_module(name, path):
    code = subprocess.check_output(['git','show',BASELINE+':'+path],text=True)
    module = types.ModuleType('torchfdtd._t2d_'+name)
    sys.modules[module.__name__] = module
    exec(compile(code,path,'exec'),module.__dict__)
    return module


@pytest.mark.parametrize('aperture',[False,True])
def test_three_dimensional_launch_tables_are_bit_identical_to_pre_t2d(aperture):
    ports = baseline_module('ports','torchfdtd/mode_ports.py')
    module = baseline_module('launch', 'torchfdtd/mode_branches.py' if aperture else 'torchfdtd/mode_injection.py')
    module.solve_waveguide_modes = ports.solve_waveguide_modes
    prepare = prepare_aperture_modal_launch if aperture else prepare_modal_launch
    original = module.prepare_aperture_modal_launch if aperture else module.prepare_modal_launch
    p = project('3d',aperture)
    material = lambda y,z:np.where(abs(y)<.2,12.,2.25)
    before, after = original(p,material,num_modes=2),prepare(p,material,num_modes=2)
    assert before.identity == after.identity
    assert before.mode.neff == after.mode.neff
    assert before.mode.power() == after.mode.power()
    np.testing.assert_array_equal(before.mode.fields,after.mode.fields)
    np.testing.assert_array_equal(before.epsilon,after.epsilon)
    for a,b in zip(before.terms,after.terms):
        assert a[:2] == b[:2]
        np.testing.assert_array_equal(a[2],b[2])
        np.testing.assert_array_equal(a[3],b[3])


def test_invariant_axis_aperture_crop_is_rejected():
    p = project('2d')
    p.sources[0].size=(0.,4.,.1)
    with pytest.raises(ValueError,match='invariant storage'):
        prepare_modal_launch(p,2.25)
