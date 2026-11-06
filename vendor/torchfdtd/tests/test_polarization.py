from dataclasses import replace
import pytest
import torch
from torchfdtd import calibrate_plane_polarization,mix_plane_fields,DifferentiablePlaneResult


def bases():
    points=torch.tensor([[-.3,-.2,0],[.3,-.2,0],[-.3,.2,0],[.3,.2,0]],dtype=torch.float64)
    k=torch.tensor([[.7,.3],[.4,.8]],dtype=torch.float64)
    phase=torch.exp(1j*(k@points[:,:2].T))
    matrix=torch.tensor([[1+.3j,.2-.1j],[.1+.2j,.8+.1j]],dtype=torch.complex128)
    results=[]
    for i in range(2):
        fields=torch.zeros(2,4,6,dtype=torch.complex128)
        fields[...,:2]=phase[...,None]*matrix[:,i]
        results.append(DifferentiablePlaneResult(fields,torch.tensor([1.,2.]),points,
            torch.ones(4,dtype=torch.float64),(2,2,1),'z',str(i),{}))
    return results,k,matrix


def test_projection_and_coherent_mixing():
    planes,k,matrix=bases()
    target=torch.tensor([[.6,.8],[.8,-.6]],dtype=torch.complex128)
    coeff=calibrate_plane_polarization(planes,k,target)
    torch.testing.assert_close(coeff,torch.linalg.solve(matrix,target.T).T)
    result=mix_plane_fields(planes,coeff)
    phase=torch.exp(1j*(k@planes[0].points_um[:,:2].T))
    torch.testing.assert_close(result.fields[...,:2],phase[...,None]*target[:,None,:])
    assert not coeff.requires_grad
    changed=mix_plane_fields(planes,coeff*2)
    assert changed.run_signature!=result.run_signature


def test_both_basis_gradients_and_guards():
    planes,k,_=bases()
    coeff=calibrate_plane_polarization(planes,k,torch.ones(2,2,dtype=torch.complex128))
    x=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    def loss(x):
        sample=[replace(planes[0],fields=planes[0].fields*x.square()),replace(planes[1],fields=planes[1].fields*x.sin())]
        return mix_plane_fields(sample,coeff).fields.abs().square().sum()
    assert torch.autograd.gradcheck(loss,(x,))
    with pytest.raises(ValueError,match='singular'):calibrate_plane_polarization([planes[0]]*2,k,torch.ones(2,2))
    with pytest.raises(ValueError,match='fixed'):mix_plane_fields(planes,coeff.requires_grad_())
    wrong=replace(planes[1],frequency_hz=planes[1].frequency_hz+1)
    with pytest.raises(ValueError,match='frequencies'):mix_plane_fields([planes[0],wrong],coeff.detach())
