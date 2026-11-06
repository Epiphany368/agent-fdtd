"""A confined HE11 pair, without FDTD or unrelated solver reruns."""
import numpy as np
import pytest
from torchfdtd import Region
from torchfdtd.open_mode_ports import solve_open_waveguide_modes, _power_pair
from types import SimpleNamespace


def solve():
    r=Region(dimension='3d',size=(4.,4.,1.6),mesh=.1,steps=10,
        boundaries={a+'_'+s:dict(kind='pml',layers=8 if a!='z' else 3)
                    for a in 'xyz' for s in ('min','max')},
        material_sampling='yee',precision='float32',backend='cpu')
    return solve_open_waveguide_modes(lambda x,y:np.where(x*x+y*y<.4**2-1e-10,2.2**2,1.),
        region=r,normal='z',wavelength_um=1.55,cladding_epsilon=1.,num_modes=2)


@pytest.fixture(scope='module')
def baseline(): return solve()


def test_confined_degenerate_pair_power_maxwell_collocation_and_repeatability(baseline):
    modes=baseline;again=solve()
    assert abs(modes[0].beta_per_um-modes[1].beta_per_um)/abs(modes[0].beta_per_um)<1e-5
    assert modes[1].diagnostics['degenerate_predecessors']==1
    weights=modes[0]._power_weights()
    gram=np.array([[_power_pair(a.fields,b.fields,weights,a.transverse_axes) for b in modes] for a in modes])
    np.testing.assert_allclose(gram,np.eye(2),atol=2e-6)
    for a,b in zip(modes,again):
        assert a.maxwell_residual<2e-4 and a.eigenpair_residual<2e-4
        overlap=_power_pair(a.fields,b.fields,weights,a.transverse_axes)
        assert abs(overlap)>1-2e-5
    # Collocated physical-aperture signed forward/backward basis.
    mode=modes[0]
    coords=[lo+(np.arange(48)+.5)*(hi-lo)/48 for lo,hi in mode.physical_bounds_um]
    x,y=np.meshgrid(*coords,indexing='ij');points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    area=np.prod([hi-lo for lo,hi in mode.physical_bounds_um])*1e-12
    plane=SimpleNamespace(normal='z',points_um=points,weights=np.full(x.size,area/x.size))
    profiles=[]
    for m in (*modes,*(v.backward() for v in modes)):
        m.validate_quadrature(plane)
        f=m.sample_plane(points)
        flux=.5*np.sum(plane.weights*(f[:,0]*f[:,4].conj()-f[:,1]*f[:,3].conj())).real
        profiles.append(f/np.sqrt(abs(flux)))
    signed=np.array([[.25*np.sum(plane.weights*(a[:,0]*b[:,4].conj()+b[:,0].conj()*a[:,4]
        -a[:,1]*b[:,3].conj()-b[:,1].conj()*a[:,3])) for b in profiles] for a in profiles])
    error=float(np.max(np.abs(signed-np.diag([1,1,-1,-1]))))
    print({'signed_collocated_gram_max_error':error, 'betas':[m.beta_per_um for m in modes]})
    np.testing.assert_allclose(signed,np.diag([1,1,-1,-1]),atol=1e-4)


def test_rank_deficient_cluster_fails_instead_of_changing_family(baseline,monkeypatch):
    mode=baseline[0]
    vector=np.r_[mode.fields[...,0].ravel(),mode.fields[...,1].ravel()]
    values=np.full(2,mode.beta_tilde_per_um**2,dtype=np.complex64)
    vectors=np.column_stack((vector,vector)).astype(np.complex64)
    monkeypatch.setattr('torchfdtd.open_mode_ports.eigs',lambda *args,**kwargs:(values,vectors))
    with pytest.raises(ValueError,match='rank deficient'):
        solve()


def test_four_channel_network_default_gram_admission_without_fdtd(baseline,monkeypatch):
    from torchfdtd import Project, Source
    from torchfdtd.mode_network import ModeNetwork, FixedModePort
    from torchfdtd.open_mode_injection import OpenPortOptions
    r=Region(dimension='3d',size=(4.,4.,1.6),mesh=.1,steps=10,
        boundaries={a+'_'+s:dict(kind='pml',layers=8 if a!='z' else 3)
                    for a in 'xyz' for s in ('min','max')},
        material_sampling='yee',precision='float32',backend='cpu')
    p=Project(region=r,sources=[Source(kind='plane',normal='z',center=(0,0,-.3),
        size=(2.4,2.4,0),wavelength=1.55,pulse_cycles=2)],monitors=[])
    ports=(FixedModePort('left',-.2,-.3,1,(0,1)),FixedModePort('right',.2,.3,-1,(0,1)))
    monkeypatch.setattr('torchfdtd.open_mode_ports.solve_open_waveguide_modes',lambda *a,**k:baseline)
    def forbidden(*a,**k): raise AssertionError('FDTD allocation during mode basis admission')
    monkeypatch.setattr('torchfdtd.differentiable._System.__init__',forbidden)
    network=ModeNetwork(p,ports,lambda x,y:np.where(x*x+y*y<.4**2-1e-10,2.2**2,1.),
        num_modes=2,open_ports=OpenPortOptions(1.))
    assert network.gram_tolerance==1e-4
    assert len(network.channels)==4
