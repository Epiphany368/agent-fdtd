"""Independent observed work and logical file traffic, not estimator self-tests."""
from collections import defaultdict
import math

import pytest
import torch

from torchfdtd import (AdjointOptions, DifferentiableSimulation, Monitor, Project,
                       Region, Source, StreamedAdjointOptions, StreamedSimulation)
from torchfdtd.differentiable import _System
from torchfdtd.solver import field_axes
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd.state_store import StateStore
from torchfdtd.streamed_work import estimate_streamed_work


def scene(boundary, steps):
    if boundary == 'mixed2d':
        region=Region(dimension='2d',size=(.9,1.,.1),mesh=.1,steps=steps,
            pml_cells=3,precision='float32',backend='cpu',material_sampling='yee',
            boundaries={'x_min':{'kind':'pec'},'x_max':{'kind':'pml','layers':3},
                        'y_min':{'kind':'pml','layers':4},'y_max':{'kind':'pec'}})
        def xy(component,index):
            axes=field_axes(region,component)
            return tuple(float(axes[a][index[a]]) for a in range(3))
        return Project(region=region,
            sources=[Source(component='Ez',center=xy('Ez',(4,5,0)),
                wavelength=.6,pulse='continuous',pulse_cycles=1)],
            monitors=[Monitor(component='Ez',center=xy('Ez',(5,5,0))),
                      Monitor(component='Hx',center=xy('Hx',(4,6,0)))])
    nx = 12 if boundary == 'nonperiodic' else 5
    faces = {f'{a}_{s}': {'kind':'pml', 'layers':layers}
             for a, pair in [('x',(3,4)),('y',(3,4)),('z',(3,5))]
             for s,layers in zip(('min','max'),pair)}
    if boundary != 'nonperiodic':
        faces.update(x_min={'kind':boundary}, x_max={'kind':boundary})
    region = Region(dimension='3d', size=(nx*.1,1.2,1.3), mesh=.1,
        steps=steps, boundaries=faces, precision='float32', backend='cpu',
        material_sampling='yee', bloch_phase=(.27,0,0) if boundary=='bloch' else (0,0,0))
    def position(component, index):
        axes=field_axes(region,component)
        return tuple(float(axes[i][index[i]]) for i in range(3))
    return Project(region=region,
        sources=[Source(component='Ex',center=position('Ex',(nx//2,5,5)),
            wavelength=.6,pulse='continuous',pulse_cycles=1)],
        monitors=[Monitor(component='Ex',center=position('Ex',(nx//2,5,6))),
                  Monitor(component='Hy',center=position('Hy',(nx//2,5,6)))])


class ObservedWork:
    """Count actually executed methods, independent of replay combinatorics."""
    def __init__(self, monkeypatch):
        self.phase='forward';self.role=None;self.counts=defaultdict(int)
        original_forward=SlabBlockOperator.forward
        original_transpose=SlabBlockOperator.transpose
        original_advance=_System.advance
        original_step=_System.transpose_step
        original_tile=SlabBlockOperator._tile
        counter=self
        def forward(operator,*args,**kwargs):
            previous=counter.role
            counter.role='forward' if counter.phase=='forward' else 'global_replay'
            if counter.phase=='backward':
                counter.counts['global_replayed_blocks']+=1
                counter.counts['global_replayed_steps']+=args[3]
            try:return original_forward(operator,*args,**kwargs)
            finally:counter.role=previous
        def transpose(operator,*args,**kwargs):
            previous=counter.role;counter.role='local_replay'
            try:return original_transpose(operator,*args,**kwargs)
            finally:counter.role=previous
        def tile(operator,*args,**kwargs):
            counter.counts[counter.role+'_tile_visits']+=1
            return original_tile(operator,*args,**kwargs)
        def advance(system,start,end):
            if counter.role is not None:
                cells=system.grid.E.numel()//3
                counter.counts['max_tile_x']=max(counter.counts['max_tile_x'],system.grid.E.shape[0])
                counter.counts[counter.role+'_cell_steps']+=cells*(end-start)
                if counter.role=='local_replay':counter.counts['local_replayed_steps']+=end-start
            return original_advance(system,start,end)
        def step(system,*args,**kwargs):
            if counter.role=='local_replay':counter.counts['adjoint_cell_steps']+=system.grid.E.numel()//3
            return original_step(system,*args,**kwargs)
        monkeypatch.setattr(SlabBlockOperator,'forward',forward)
        monkeypatch.setattr(SlabBlockOperator,'transpose',transpose)
        monkeypatch.setattr(SlabBlockOperator,'_tile',tile)
        monkeypatch.setattr(_System,'advance',advance)
        monkeypatch.setattr(_System,'transpose_step',step)


@pytest.mark.parametrize('boundary,steps,depth,width,checkpoints,local,diagonal',[
    ('nonperiodic',10,3,4,0,0,False),
    ('nonperiodic',11,3,4,1,1,True),
    ('periodic',12,6,3,0,1,False),
    ('bloch',13,6,3,1,0,True),
    ('mixed2d',10,13,32,2,2,False),
])
def test_metadata_matches_actual_file_io_and_executed_work(tmp_path,monkeypatch,
        boundary,steps,depth,width,checkpoints,local,diagonal):
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        project=scene(boundary,steps)
        shape=project.region.shape+((3,) if diagonal else ())
        epsilon=torch.full(shape,1.6,dtype=torch.float32,requires_grad=True)
        options=StreamedAdjointOptions(device='cpu',slab_width=width,
            temporal_depth=depth,checkpoints=checkpoints,local_checkpoints=local,
            state_storage='disk',state_directory=tmp_path,disk_budget_bytes=32*1024**2,
            host_budget_bytes=256*1024**2)
        def forbidden(*args,**kwargs):raise AssertionError('Metadata estimate allocated a runtime system or file store.')
        with monkeypatch.context() as guard:
            guard.setattr(_System,'__init__',forbidden)
            guard.setattr(StateStore,'__init__',forbidden)
            predicted=estimate_streamed_work(project,options,diagonal=diagonal)
        observed=ObservedWork(monkeypatch)
        actual=StreamedSimulation(project,options)(epsilon)
        observed.phase='backward'
        generator=torch.Generator().manual_seed(71)
        seed=torch.randn(actual.signals.shape,dtype=actual.signals.dtype,generator=generator)
        gradient,=torch.autograd.grad(actual.signals,epsilon,seed)
        measured=dict(observed.counts)
        forward=actual.report['forward_backing_store'];backward=actual.report['backward_backing_store']
        for predicted_name,value in {
            'state_bytes':actual.report['state_bytes'],
            'forward_state_read_bytes':forward['logical_read_bytes'],
            'forward_state_write_bytes':forward['logical_written_bytes'],
            'backward_state_read_bytes':backward['logical_read_bytes'],
            'backward_state_write_bytes':backward['logical_written_bytes'],
            'total_state_io_bytes':sum(s[k] for s in (forward,backward)
                for k in ('logical_read_bytes','logical_written_bytes')),
            'blocks':math.ceil(steps/depth),
            'global_replayed_blocks':actual.report['replayed_blocks'],
            'local_replayed_steps':actual.report['backward_workspace']['local_replayed_steps'],
        }.items():
            assert predicted[predicted_name]==value,(predicted_name,predicted[predicted_name],value)
        for key in ('global_replayed_blocks','global_replayed_steps','local_replayed_steps',
                    'forward_cell_steps','global_replay_cell_steps','local_replay_cell_steps','adjoint_cell_steps'):
            assert predicted[key]==measured.get(key,0),(key,predicted[key],measured.get(key,0))
        for name,role in [('forward_tile_visits','forward'),('backward_replay_tile_visits','global_replay'),('backward_transpose_tile_visits','local_replay')]:
            assert predicted[name]==measured.get(role+'_tile_visits',0)
        assert predicted['tile_count']==math.ceil(project.region.shape[0]/width)
        if boundary in ('periodic','bloch'):assert measured['max_tile_x']>2*project.region.shape[0]
        if boundary=='mixed2d':
            assert predicted['blocks']==predicted['tile_count']==1
            assert measured.get('global_replayed_blocks',0)==0
            assert measured['max_tile_x']==project.region.shape[0]
        assert predicted['total_cell_steps']==sum(measured.get(k+'_cell_steps',0)
            for k in ('forward','global_replay','local_replay','adjoint'))
        assert all(s['closed'] and s['live_logical_file_bytes']==0 for s in (forward,backward))
        assert not list(tmp_path.iterdir())
        # A compact physical check makes exact traffic insufficient to hide a
        # wrong winding, halo reduction, CPML ownership or checkpoint schedule.
        observed.role=None
        reference_epsilon=epsilon.detach().clone().requires_grad_()
        reference=DifferentiableSimulation(project,AdjointOptions(checkpoints=1))(reference_epsilon)
        expected,=torch.autograd.grad(reference.signals,reference_epsilon,seed)
        torch.testing.assert_close(actual.signals,reference.signals,rtol=5e-5,atol=2e-7)
        assert gradient.norm()>0
        relative=(gradient.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30)
        assert float(relative)<2e-4
    finally:
        torch.set_num_threads(previous_threads)
