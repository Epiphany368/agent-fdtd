import numpy as np
import pytest
import torch
from torchfdtd.models import Project,Region,Material,Structure,Source,Monitor
from torchfdtd.tensor_project import tensor_from_project


def scene():
    return Project(region=Region(dimension='3d',size=(.6,.6,.6),mesh=.1,steps=12,backend='cpu',material_sampling='yee',boundaries={a+'_'+s:{'kind':'periodic'} for a in 'xyz' for s in ('min','max')}),
        materials=[Material(name='tensor',model='tensor',epsilon_tensor=(2.3,2.5,2.7,.13,.11,.09)),Material(name='glass',index=1.4)],
        structures=[Structure(material='tensor',size=(.3,.3,.3))],
        sources=[Source(component='Ex',center=(0,0,0),pulse='continuous')],
        monitors=[Monitor(component='Ex',center=(.1,0,0)),Monitor(component='Ey',center=(0,.1,0))])


def test_schema_and_pre_scalar_scope_rejection():
    with pytest.raises(ValueError,match='eigenvalues'):Material(name='bad',model='tensor',epsilon_tensor=(1,1,1,2,0,0))
    m=Material(name='t',model='tensor')
    with pytest.raises(ValueError,match='node-tensor'): _=m.instantaneous_epsilon
    p=scene();assert Project.model_validate_json(p.model_dump_json()).materials[0].epsilon_tensor==p.materials[0].epsilon_tensor
    for key,value in [('precision','float64'),('material_sampling','cell'),('interface_method','subpixel')]:
        payload=p.model_dump();payload['region'][key]=value
        with pytest.raises(ValueError):Project.model_validate(payload)
    payload=p.model_dump();payload['materials'][1]['model']='drude'
    with pytest.raises(ValueError,match='ADE'):Project.model_validate(payload)


def test_node_coordinates_precedence_graph_and_budget():
    p=scene();p.structures.append(Structure(material='glass',size=(.15,.15,.15)))
    adapter=tensor_from_project(p)
    table=torch.tensor([p.materials[0].epsilon_tensor,(1.96,1.96,1.96,0,0,0)],requires_grad=True)
    eps=adapter.rasterize(table)
    # Node origin belongs to the later glass object, .1 x belongs to tensor.
    torch.testing.assert_close(eps[3,3,3],torch.eye(3)*1.96)
    torch.testing.assert_close(eps[4,3,3],torch.tensor([[2.3,.13,.11],[.13,2.5,.09],[.11,.09,2.7]]))
    assert torch.equal(eps[0,0,0],torch.eye(3))
    eps[...,0,1].sum().backward();assert table.grad[0,3]>0
    with pytest.raises(ValueError,match='budget'):tensor_from_project(p,tensor_budget_bytes=1)
    adapter.project.structures[0].size=(.2,.2,.2)
    with pytest.raises(ValueError,match='changed'):adapter.rasterize()


def test_actual_field_six_material_derivatives():
    p=scene();adapter=tensor_from_project(p,checkpoints=2)
    table=torch.tensor([p.materials[0].epsilon_tensor,(1.96,1.96,1.96,0,0,0)],requires_grad=True)
    result=adapter(table);loss=result.signals.square().sum()
    gradient=torch.autograd.grad(loss,table)[0][0]
    assert torch.isfinite(gradient).all() and gradient.abs().min()>1e-8
    values=[];delta=.003
    for component in range(6):
        plus=table.detach().clone();minus=table.detach().clone()
        plus[0,component]+=delta;minus[0,component]-=delta
        values.append((adapter(plus).signals.square().sum()-adapter(minus).signals.square().sum())/(2*delta))
    torch.testing.assert_close(gradient,torch.stack(values),rtol=.025,atol=2e-5)


def test_cpml_fixed_node_collar_rejected_not_overwritten():
    p=scene().model_dump();p['region']['size']=(1.6,.6,.6)
    for side in ('min','max'):p['region']['boundaries']['x_'+side]={'kind':'pml','layers':3}
    adapter=tensor_from_project(Project.model_validate(p));epsilon=adapter.rasterize();adapter.validate_material(epsilon)
    p['structures'][0]['center']=(-.4,0,0)
    adapter=tensor_from_project(Project.model_validate(p))
    with pytest.raises(ValueError,match='collar'):adapter.rasterize()


def test_bloch_adapter_trace_and_material_graph_match_core():
    from torchfdtd.anisotropy import TensorDielectricSimulation
    from torchfdtd.differentiable import AdjointOptions
    payload=scene().model_dump()
    for side in ('min','max'):payload['region']['boundaries']['x_'+side]={'kind':'bloch'}
    payload['region']['bloch_phase']=(.37,0.,0.)
    p=Project.model_validate(payload);adapter=tensor_from_project(p,checkpoints=2)
    table=torch.tensor([p.materials[0].epsilon_tensor,(1.96,1.96,1.96,0,0,0)],requires_grad=True)
    result=adapter(table)
    epsilon=adapter.rasterize(table).detach().requires_grad_()
    direct=TensorDielectricSimulation(p,AdjointOptions(checkpoints=2))(epsilon)
    assert result.signals.dtype==torch.complex64
    torch.testing.assert_close(result.signals,direct.signals)
    material_gradient=torch.autograd.grad(result.signals.abs().square().sum(),table)[0]
    node_gradient=torch.autograd.grad(direct.signals.abs().square().sum(),epsilon)[0]
    reconstructed=torch.autograd.grad(adapter.rasterize(table),table,node_gradient)[0]
    torch.testing.assert_close(material_gradient,reconstructed)
    assert material_gradient[0].abs().max()>0
