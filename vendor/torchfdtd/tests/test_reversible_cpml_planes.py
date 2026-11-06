"""Recorded online plane spectra versus the checkpointed discrete solver."""
import pytest
import torch

from torchfdtd import (Project, Region, Source, FieldMonitor, AdjointOptions,
                       DifferentiablePlaneSimulation, ReversibleCPMLOptions)
from torchfdtd.reversible_cpml_planes import ReversibleCPMLPlaneSimulation


def fixture(complex_fields=False, component='Ex'):
    kind = 'bloch' if complex_fields else 'periodic'
    faces = {f'{a}_{s}': dict(kind=kind) for a in 'xy' for s in ('min', 'max')}
    faces.update(z_min=dict(kind='pml', layers=3, kappa=2, alpha=.03),
                 z_max=dict(kind='pml', layers=4, kappa=2, alpha=.03))
    region = Region(dimension='3d', size=(.6,.7,2), mesh=.1, steps=32,
        precision='float32', courant_factor=.9, backend='cpu', boundaries=faces,
        material_sampling='yee', bloch_phase=(.31,-.47,0) if complex_fields else (0,0,0))
    project = Project(region=region, sources=[Source(kind='plane', normal='z',
        size=(.6,.7,0), center=(0,0,-.4), component=component, wavelength=.6,
        time_definition='standard', pulse_length=.4e-15, pulse_offset=.6e-15)],
        monitors=[FieldMonitor(id=name, normal='z', size=(.6,.7,0), center=(0,0,z),
            spectrum=dict(sampling='frequency', apodization='none'))
            for name,z in [('incident',-.2),('detector',.5)]])
    shape = (*region.shape,3) if complex_fields else region.shape
    material = torch.full(shape,1.4,dtype=torch.float32)
    material[:,:,9:11] = 2.25
    if complex_fields: material[...,1] += .2
    fixed = torch.full_like(material,1.3)
    return project,material,fixed


def relative(actual, expected):
    actual, expected = actual.detach(), expected.detach()
    return float((actual-expected).abs().double().norm()/expected.abs().double().norm().clamp_min(1e-30))


@pytest.mark.parametrize('complex_fields,component,block', [
    (False,'Ex',1), (False,'Ey',7), (True,'Ex',7), (True,'Ey',1)])
def test_online_plane_spectra_material_vjp_and_retained_seeds(complex_fields,component,block):
    torch.set_num_threads(1)
    project,base,fixed = fixture(complex_fields,component)
    counts = {'incident':(3,4),'detector':(3,4)}
    actual = ReversibleCPMLPlaneSimulation(project,quadrature_counts=counts)
    reference = DifferentiablePlaneSimulation(project,AdjointOptions(checkpoints=2),quadrature_counts=counts)
    frequencies = torch.tensor([.033,.057],dtype=torch.float32)/project.region.time_step
    a,b = actual.interior_z
    generator = torch.Generator().manual_seed(142)
    seeds = [torch.randn((6,12,2,2),generator=generator,dtype=torch.complex64).permute(2,3,1,0) for _ in range(2)]
    assert all(not seed.is_contiguous() for seed in seeds)
    rows = []
    for model in (reference,actual):
        parameter = base.clone().requires_grad_()
        if model is reference:
            effective = fixed.clone(); effective[:,:,a:b+1] = parameter[:,:,a:b+1]
            planes = model(effective,frequencies,block_size=block)
        else:
            planes = model(parameter,frequencies,fixed_epsilon=fixed,block_size=block)
        fields = torch.stack([plane.fields for plane in planes.values()])/project.region.time_step
        loss = (fields.real+.37*fields.imag).square().mean()
        gradients = [torch.autograd.grad(loss,parameter,retain_graph=True)[0]]
        gradients += [torch.autograd.grad(fields,parameter,seed,retain_graph=i==0)[0] for i,seed in enumerate(seeds)]
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g[:,:,:a])==0
                   and torch.count_nonzero(g[:,:,b+1:])==0 for g in gradients)
        rows.append((fields.detach(),gradients))
    assert relative(rows[1][0],rows[0][0]) < 1e-4
    assert all(relative(x,y)<1e-4 for x,y in zip(rows[1][1],rows[0][1]))
    report = next(iter(planes.values())).report
    assert report['observation_history_retained'] is False
    assert report['observation_block_shape'] == [block,len(actual.observers)]
    assert report['last_backward']['seed_buffer_shape'] == [block,len(actual.observers)]
    plan = actual.plan(frequencies,material_components=3 if complex_fields else 1,block_size=block)
    assert plan['output_history_bytes'] == 0
    assert plan['material_components'] == (3 if complex_fields else 1)
    assert actual._spectral(base,frequencies,block).layout_reservation_bytes == actual.layout_reservation_bytes


def test_plane_admission_and_snapshot():
    project,base,fixed = fixture()
    frequencies = [.033/project.region.time_step]
    counts = {'incident':(3,4),'detector':(3,4)}
    with pytest.raises(ValueError,match='ReversibleCPMLOptions'):
        ReversibleCPMLPlaneSimulation(project,AdjointOptions(),quadrature_counts=counts)
    model = ReversibleCPMLPlaneSimulation(project,quadrature_counts=counts)
    with pytest.raises(ValueError,match='material_components'):
        model.plan(frequencies,material_components=2)
    with pytest.raises(ValueError,match='block_size'):
        model.plan(frequencies,block_size=0)
    with pytest.raises(ValueError,match='shape|matching|scalar|diagonal'):
        model(base[:,:,:-1],frequencies,fixed_epsilon=fixed)
    model.model.options = ReversibleCPMLOptions(host_budget_bytes=1)
    with pytest.raises(ValueError,match='budget'):
        model.plan(frequencies)
    model.project.monitors[0].center = (0,0,0)
    with pytest.raises(ValueError,match='configuration changed'):
        model.plan(frequencies)
