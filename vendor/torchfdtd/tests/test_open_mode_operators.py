"""Independent harmonic CPML recurrence checks for open transverse operators."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from torchfdtd import Region, BoundaryFace
from torchfdtd.open_mode_operators import prepare_open_mode_operators, plan_open_mode_operators


def region(normal='z', both=True, sigma=1.):
    w='xyz'.index(normal);u,v=(w+1)%3,(w+2)%3
    faces={}
    for axis in range(3):
        for side,name in enumerate(('min','max')):
            faces['xyz'[axis]+'_'+name]=(dict(kind='periodic') if axis==v and not both else
                dict(kind='pml',layers=3+side,sigma_scale=sigma*(1+.2*side),alpha=.03,kappa=1 if sigma<1e-20 else 2.1))
    return Region(dimension='3d',size=(1.2,1.3,1.4),mesh=.1,steps=10,
                  boundaries=faces,material_sampling='yee',precision='float32',backend='cpu')


def recurrence(r,axes,field,local_axis,forward,wavelength):
    """Explicit state recurrence, independent of sparse/production derivatives."""
    axis=axes[local_axis];n=field.shape[local_axis];periodic=r.boundaries.pair(axis)[0].kind=='periodic'
    derivative=np.zeros_like(field)
    target=[slice(None),slice(None)];left=target.copy();right=target.copy()
    target[local_axis]=slice(0,n-1) if forward else slice(1,n)
    left[local_axis]=slice(0,n-1);right[local_axis]=slice(1,n)
    derivative[tuple(target)]=(field[tuple(right)]-field[tuple(left)])/r.axis_steps[axis]
    if periodic:
        target[local_axis]=-1 if forward else 0
        left[local_axis]=-1;right[local_axis]=0
        derivative[tuple(target)]=(field[tuple(right)]-field[tuple(left)])/r.axis_steps[axis]
        return derivative
    decay=np.ones(n,dtype=np.float32);coupling=np.zeros(n,dtype=np.float32);invk=np.ones(n,dtype=np.float32)
    for side,face in enumerate(r.boundaries.pair(axis)):
        layers=r.pml_layers(axis,side);base=0 if side==0 else n-layers
        for row in range(max(base,0 if forward else 1),min(base+layers,n-1 if forward else n)):
            depth=layers-(row-base)-(1 if forward else .5) if side==0 else (row-base)+(1 if forward else .5)
            rho=depth/(layers+1);sigma=face.sigma_scale*40*rho**face.polynomial/(layers+1)
            kappa=1+(face.kappa-1)*rho**face.polynomial;alpha=face.alpha*(1-rho)**face.alpha_polynomial
            b=np.exp(-(sigma/kappa+alpha)*r.rectangular_courant)
            denominator=sigma*kappa+alpha*kappa*kappa
            decay[row]=b;coupling[row]=0 if denominator==0 else (b-1)*sigma/denominator;invk[row]=1/kappa
    shape=[1,1];shape[local_axis]=n
    b,c,k=(x.reshape(shape) for x in (decay,coupling,invk))
    omega_dt=2*np.pi*299792458.*r.time_step/(wavelength*1e-6)
    psi=np.zeros_like(field)
    # Recurrence starts at zero and converges under a harmonic forcing.
    for step in range(4000):
        phase=np.complex64(np.exp(-1j*omega_dt*step));drive=derivative*phase
        psi=b*psi+c*drive
    return (k*drive+psi)/phase


@pytest.mark.parametrize('normal,both',[('x',False),('y',True),('z',True)])
def test_explicit_harmonic_psi_recurrence_all_axes_corners_and_asymmetric_faces(normal,both):
    r=region(normal,both);wave=.8;ops=prepare_open_mode_operators(r,normal=normal,wavelength_um=wave)
    rng=np.random.default_rng(45);field=(rng.normal(size=ops.shape)+1j*rng.normal(size=ops.shape)).astype(np.complex64)
    for matrix,axis,forward in ((ops.up,0,True),(ops.um,0,False),(ops.vp,1,True),(ops.vm,1,False)):
        actual=(matrix@field.ravel()).reshape(ops.shape)
        expected=recurrence(r,ops.transverse_axes,field,axis,forward,wave)
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-5
        assert matrix.dtype==np.complex64 and matrix.format=='csr'
        assert not matrix.data.flags.writeable
    with pytest.raises(FrozenInstanceError):ops.shape=(1,1)
    assert ops.origin_um==tuple(-r.actual_size[a]/2 for a in ops.transverse_axes)
    assert ops.physical_bounds_um==tuple(r.interior_bounds(a) for a in ops.transverse_axes)


def test_zero_sigma_limit_retains_distinct_truncated_endpoint_rows():
    r=region(sigma=1e-30);ops=prepare_open_mode_operators(r,normal='z',wavelength_um=.8)
    ramp=np.broadcast_to(np.arange(ops.shape[0],dtype=np.float32)[:,None],ops.shape).copy()
    plus=(ops.up@ramp.ravel()).reshape(ops.shape)
    minus=(ops.um@ramp.ravel()).reshape(ops.shape)
    np.testing.assert_allclose(plus[:-1],10,atol=1e-5)
    np.testing.assert_allclose(minus[1:],10,atol=1e-5)
    assert np.count_nonzero(plus[-1])==0 and np.count_nonzero(minus[0])==0
    assert (ops.um+ops.up.T).nnz>0


def test_metadata_plan_and_rejections_before_sparse_allocation(monkeypatch):
    import torchfdtd.open_mode_operators as implementation
    r=region();plan=plan_open_mode_operators(r,'z',.8)
    assert plan['n']==12*13 and plan['required_workspace_bytes']>4*2*plan['n']*8
    monkeypatch.setattr(implementation,'BoundaryDescription',lambda *a:pytest.fail('Allocated coefficients'))
    monkeypatch.setattr(implementation.sparse,'coo_matrix',lambda *a,**k:pytest.fail('Allocated sparse matrix'))
    assert plan_open_mode_operators(r,'z',.8)['region_signature']==plan['region_signature']
    for normal,wave in [('q',.8),('z',0),('z',float('nan')),('z',True),('z',.001)]:
        with pytest.raises(ValueError):plan_open_mode_operators(r,normal,wave)
    for precision,sampling in [('float64','yee'),('float32','cell')]:
        bad=r.model_copy(update=dict(precision=precision,material_sampling=sampling))
        with pytest.raises(ValueError):plan_open_mode_operators(bad,'z',.8)
    bad=r.model_copy(deep=True)
    for axis in 'xy':
        for side in ('min','max'):setattr(bad.boundaries,axis+'_'+side,BoundaryFace(kind='periodic'))
    with pytest.raises(ValueError):plan_open_mode_operators(bad,'z',.8)
