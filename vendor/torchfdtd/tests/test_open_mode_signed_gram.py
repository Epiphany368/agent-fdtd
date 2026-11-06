"""Signed modal admission independent of eigensolver and native FDTD."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from torchfdtd.mode_ports import WaveguideMode
from torchfdtd.open_mode_ports import OpenWaveguideMode
from torchfdtd.mode_network import _decompose


def fixture(alpha=0., boundary='cpml'):
    # E1=x, E2=y. H1=(2alpha,2), H2=(-2,2alpha).
    # Unit-power C_ij = integral Ei x Hj* = 2I + 2alpha[[0,1],[-1,0]].
    # Thus G=(C+C.H)/4=I but reverse K=(C-C.H)/4 != 0.
    profiles=np.array([[1,0,0,2*alpha,2,0],[0,1,0,-2,2*alpha,0]],dtype=np.complex64)
    extent=1. if boundary=='periodic' else 2.
    coordinates=np.array([-.25,.25])*extent
    x,y=np.meshgrid(coordinates,coordinates,indexing='ij')
    points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    weights=torch.full((4,),extent**2*1e-12/4,dtype=torch.float32)
    modes=[]
    for profile in profiles:
        if boundary=='cpml':
            mode=OpenWaveguideMode(beta_per_um=4.+0j,beta_tilde_per_um=3.9+0j,
                wavelength_um=1.55,normal='z',spacing_um=(1.,1.),origin_um=(-2.,-2.),
                fields=np.broadcast_to(profile,(4,4,6)).copy(),eigenpair_residual=0.,
                maxwell_residual=0.,region_signature='synthetic',physical_bounds_um=((-1.,1.),(-1.,1.)),
                sampled_epsilon=np.ones((4,4,3),dtype=np.float32),periodic_axes=(False,False),diagnostics={})
        else:
            mode=WaveguideMode(4.,1.55,'z',(.5,.5),(-.5,-.5),
                np.broadcast_to(profile,(2,2,6)).copy(),0.,0.,'float32')
        modes.append(mode)
    plane=SimpleNamespace(normal='z',points_um=torch.tensor(points,dtype=torch.float32),
        weights=weights,fields=torch.zeros((1,4,6),dtype=torch.complex64))
    launches=tuple(SimpleNamespace(detector_mode=lambda coordinate,mode=mode:mode) for mode in modes)
    bases=[torch.tensor(mode.sample_plane(points)) for mode in modes]
    bases=[b/(.5*((b[:,0]*b[:,4].conj()-b[:,1]*b[:,3].conj()).real*weights).sum()).sqrt() for b in bases]
    return plane,launches,modes,bases


def test_cross_reactive_counterexample_passes_previous_gram_and_diagonal_checks():
    plane,launches,modes,bases=fixture(.2)
    for mode in modes: assert mode.validate_quadrature(plane)>0
    # Independently construct the electric/magnetic pairing matrix.
    electric=torch.stack([b[:,:2] for b in bases])
    rotated_magnetic=torch.stack([torch.stack((b[:,4],-b[:,3]),-1) for b in bases])
    c=torch.einsum('ipd,jpd,p->ij',electric,rotated_magnetic.conj(),plane.weights.to(torch.complex64))
    torch.testing.assert_close((c+c.mH)/4,torch.eye(2,dtype=torch.complex64),atol=2e-6,rtol=2e-6)
    assert torch.count_nonzero(c.diag().imag)==0
    assert float(((c-c.mH)/4).abs().max())==pytest.approx(.2,rel=2e-6)
    with pytest.raises(ValueError,match='signed forward/backward'):
        _decompose(plane,launches,1e-4)


@pytest.mark.parametrize('boundary',['cpml','periodic'])
def test_valid_signed_basis_recovers_both_directions_and_field_vjp(boundary):
    plane,launches,_,bases=fixture(boundary=boundary)
    coefficients=torch.tensor([.7+.2j,-.1+.3j,.13-.11j,-.06+.08j],dtype=torch.complex64,requires_grad=True)
    backward=[]
    for basis in bases:
        partner=basis.clone();partner[:,[2,3,4]]*=-1
        backward.append(partner)
    plane.fields=sum(a*b for a,b in zip(coefficients,(*bases,*backward)))[None]
    forward,reverse=_decompose(plane,launches,1e-4)
    actual=torch.cat((forward[0],reverse[0]))
    torch.testing.assert_close(actual,coefficients,atol=2e-6,rtol=2e-6)
    gradient,=torch.autograd.grad(actual.abs().square().sum(),coefficients)
    torch.testing.assert_close(gradient,2*coefficients,atol=2e-6,rtol=2e-6)
