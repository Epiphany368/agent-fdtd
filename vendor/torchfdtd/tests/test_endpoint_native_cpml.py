"""Native restricted endpoint CPML dispatch, without CUDA execution."""
import math
import numpy as np
import pytest
import torch
from torchfdtd.models import Project, Region, Source, Monitor
from torchfdtd.solver import Simulation, estimate
from torchfdtd.endpoint_project import endpoint_from_project
from torchfdtd.endpoint_native import endpoint_cpml_options


def scene():
    faces={a+'_'+s: {'kind':'pmc'} for a in 'xyz' for s in ('min','max')}
    faces['x_min']={'kind':'pml','alpha':0,'layers':3,'sigma_scale':.3}
    faces['x_max']={'kind':'pml','alpha':0,'layers':3,'sigma_scale':.3}
    return Project(region=Region(dimension='3d',size=(1.6,.6,.6),mesh=.1,
        steps=20,pml_cells=4,material_sampling='yee',backend='cpu',snapshot_interval=5,boundaries=faces),
        sources=[Source(component='Ez',center=(0,0,0),pulse='continuous')],
        monitors=[Monitor(component='Ez',center=(.1,0,0)),Monitor(component='Hy',center=(.1,0,0))])


def test_native_cpml_matches_adapter_fields_trace_and_profile():
    project=scene();result=Simulation(project).run();adapter=endpoint_from_project(project,checkpoints=0)
    epsilon=adapter.rasterize();waves=adapter.waveforms();expected=adapter(epsilon,waves)
    np.testing.assert_allclose(result.signals,expected.signals.numpy(),rtol=1e-6,atol=1e-8)
    state=adapter.simulation._zero()
    for drive in waves:state=adapter.simulation._step(state,epsilon,drive)
    count=3*math.prod(project.region.shape)
    np.testing.assert_array_equal(result.electric.reshape(-1),state.electric[:count].numpy())
    np.testing.assert_array_equal(result.magnetic.reshape(-1),state.magnetic[:count].numpy())
    report=result.summary['endpoint_plan']['cpml']
    assert report['pml_cells']==3 and report['native_sigma_scale']==.3
    assert report['scalar_yee_profile_equivalence'] is False
    assert result.summary['engine'].endswith('/CPML')
    assert estimate(project)['endpoint_tensor_bytes']>=adapter.simulation.memory_plan(20)['tensor_upper_bound_bytes']
    assert result.summary['endpoint_plan']['memory']['psi_state_bytes']>0
    epsilon[adapter.simulation.collar]=1.1
    with pytest.raises(ValueError,match='collar'):adapter(epsilon,waves)


@pytest.mark.parametrize('field,value', [('alpha',1e-8),('kappa',2),('polynomial',2),('alpha_polynomial',1),('layers',4),('sigma_scale',.4),('kind','periodic')])
def test_native_cpml_rejects_untranslated_profile_parameters(field,value):
    # Per-face profiles are admitted by the Yee paths; the endpoint forward dispatch rejects them at run time.
    payload=scene().model_dump();payload['region']['boundaries']['x_min'][field]=value
    with pytest.raises(ValueError):Simulation(Project.model_validate(payload)).run()


def test_native_cpml_rejects_inside_source_and_checks_auxiliary(monkeypatch):
    payload=scene().model_dump();payload['sources'][0]['center']=(-.7,0,0)
    with pytest.raises(ValueError,match='PML'):Simulation(Project.model_validate(payload)).run()
    from torchfdtd.pmc_cpml import EndpointCPMLSimulation, CPMLState
    step=EndpointCPMLSimulation._step
    def broken(self,*args):
        state=step(self,*args)
        return CPMLState(state.electric,state.magnetic,tuple(torch.full_like(v,float('nan')) for v in state.psi))
    monkeypatch.setattr(EndpointCPMLSimulation,'_step',broken)
    with pytest.raises(FloatingPointError,match='auxiliary'):Simulation(scene()).run()


def test_native_cpml_cli_and_tiny_budget(tmp_path,monkeypatch,capsys):
    import sys
    import json
    from torchfdtd.cli import main
    from torchfdtd.solver import Result
    project=scene();path=tmp_path/'cpml.json';project.save(path);output=tmp_path/'cpml.npz'
    monkeypatch.setattr(sys,'argv',['torchfdtd','run',str(path),'--output',str(output)])
    main();assert json.loads(capsys.readouterr().out)['endpoint_plan']['cpml']['pml_cells']==3
    loaded=Result.load(output)
    assert loaded.endpoint_fields is not None and loaded.summary['engine'].endswith('/CPML')
    with pytest.raises(ValueError,match='budget'):
        endpoint_from_project(project,tensor_budget_bytes=1)
