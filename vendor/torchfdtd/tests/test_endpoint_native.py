"""Native PMC JSON -> solver/NPZ/CLI/API dispatch, with explicit unsupported gates."""
import json
import sys
import threading
import time
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd.models import Project,Region,Source,Monitor,Material,Structure,demo_project
from torchfdtd.solver import Simulation,Result,estimate
from torchfdtd.endpoint_project import endpoint_from_project


def scene(kind='pmc'):
    return Project(region=Region(dimension='3d',size=(.6,.6,.6),mesh=.1,steps=10,
        material_sampling='yee',backend='cpu',snapshot_interval=2,
        boundaries={a+'_'+side:{'kind':kind} for a in 'xyz' for side in ('min','max')}),
        sources=[Source(component='Ez',center=(.3,.3,0),pulse='continuous')],
        monitors=[Monitor(component='Ez',center=(.3,.3,0)),Monitor(component='Hx',center=(.3,0,0))])


def test_native_dispatch_matches_endpoint_adapter_and_npz_retains_upper_fields(tmp_path):
    project=Project.model_validate_json(scene('symmetric').model_dump_json())
    updates=[];result=Simulation(project).run(progress=updates.append)
    expected=endpoint_from_project(project)()
    np.testing.assert_allclose(result.signals,expected.signals.numpy(),rtol=1e-6,atol=1e-8)
    assert result.summary['engine']=='TorchFDTD exact-endpoint PEC/PMC'
    assert result.endpoint_fields['E_upper'].size>0 and np.max(abs(result.endpoint_fields['E_upper']))>0
    assert result.electric.shape==project.region.shape+(3,)
    assert len(result.frames)==len(result.frame_steps)==len(updates)
    assert updates[-1]['step']==10
    plan=result.summary['endpoint_plan'];assert plan['source_terms'][0]['index'][:2]==[6,6]
    path=tmp_path/'endpoint.npz';result.save(path);loaded=Result.load(path)
    for family in ('E_upper','H_upper'):np.testing.assert_array_equal(loaded.endpoint_fields[family],result.endpoint_fields[family])
    np.testing.assert_array_equal(loaded.signals,result.signals)
    assert loaded.project.region.boundaries.x_max.kind=='symmetric'
    assert loaded.monitor_data()[0]['signal']
    assert estimate(project)['endpoint_counts']['E']>3*np.prod(project.region.shape)


def test_cli_and_api_jobs_follow_native_dispatch(tmp_path,monkeypatch,capsys):
    from torchfdtd.cli import main
    from torchfdtd.server import create_app
    project=scene();path=tmp_path/'scene.json';project.save(path);output=tmp_path/'cli.npz'
    monkeypatch.setattr(sys,'argv',['torchfdtd','run',str(path),'--output',str(output)])
    main();assert json.loads(capsys.readouterr().out)['engine']=='TorchFDTD exact-endpoint PEC/PMC'
    assert Result.load(output).endpoint_fields is not None
    client=TestClient(create_app(tmp_path/'server'))
    checked=client.post('/api/validate',json=project.model_dump());assert checked.status_code==200,checked.text
    response=client.post('/api/jobs',json=project.model_dump());assert response.status_code==202,response.text
    key=response.json()['id'];deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        job=client.get('/api/jobs/'+key).json()
        if job['status'] not in ('queued','running'):break
        time.sleep(.02)
    assert job['status']=='completed',job
    assert job['summary']['engine']=='TorchFDTD exact-endpoint PEC/PMC'
    assert client.get('/api/jobs/'+key+'/download').status_code==200


@pytest.mark.parametrize('change',[
    lambda p:p['region']['boundaries']['x_min'].update(kind='pml'),
    lambda p:p['region'].update(precision='float64'),
    lambda p:p['region'].update(material_sampling='cell'),
    lambda p:p['region'].update(memory_mode='streamed'),
    lambda p:p['region'].update(interface_method='subpixel'),
    lambda p:p['sources'][0].update(component='Hx'),
    lambda p:p['sources'][0].update(enabled=False),
    lambda p:p['monitors'][0].update(kind='field'),
    lambda p:p['monitors'][0].update(time_downsample=2),
])
def test_unsupported_native_pmc_contracts_are_rejected_before_dispatch(change):
    # The schema admits what the Yee adjoint, streamed and batch paths implement;
    # the bounded exact-endpoint forward dispatch still rejects at run time.
    payload=scene().model_dump();change(payload)
    with pytest.raises(ValueError):Simulation(Project.model_validate(payload)).run()


def test_endpoint_dispatch_rejects_ade_and_plain_grid_curl_rejects_pmc():
    from torchfdtd.boundaries import BoundaryDescription,YeeGrid
    from torchfdtd.differentiable import DifferentiableSimulation
    project=scene();payload=project.model_dump()
    payload['materials']=[Material(name='metal',model='drude').model_dump()]
    payload['structures']=[Structure(material='metal').model_dump()]
    # ADE next to PMC runs through the Yee adjoint/streamed/batch paths; the endpoint forward has no ADE.
    with pytest.raises(ValueError,match='ADE'):Simulation(Project.model_validate(payload)).run()
    description=BoundaryDescription(project.region)
    assert description.pmc_upper and description.pmc_blocks['E']
    grid=YeeGrid(project.region)
    with pytest.raises(ValueError,match='not implemented by the Torch/NumPy grid curl'):grid.curl(grid.H,False)
    with pytest.raises(ValueError,match='stored row'):
        DifferentiableSimulation(project)(torch.ones(project.region.shape))


def test_pec_stays_ordinary_and_native_pmc_runtime_guards():
    payload=scene().model_dump()
    for face in payload['region']['boundaries'].values():face['kind']='pec'
    for item in payload['sources']+payload['monitors']:item['center']=(0,0,0)
    ordinary=Simulation(Project.model_validate(payload)).run()
    assert ordinary.endpoint_fields is None and 'endpoint' not in ordinary.summary['engine']
    project=scene();project.region.run_control.field_limit=1e-20
    with pytest.raises(FloatingPointError,match='limit'):Simulation(project).run()
    stop=threading.Event();stop.set();cancelled=Simulation(scene()).run(cancel=stop)
    assert cancelled.summary['cancelled'] and cancelled.signals.shape==(0,2)
    assert cancelled.frames.shape[0]==0


def test_native_demo_is_small_supported_project():
    project=demo_project('pmc')
    assert project.region.shape==(16,16,16) and project.region.steps==160
    assert project.region.boundaries.x_max.kind=='pmc'
    assert len(project.monitors)==2


def test_native_cpu_cuda_trace_final_endpoint_and_npz_parity(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    pytest.importorskip('cupy')
    project=scene();cpu=Simulation(project).run()
    project.region.backend='cuda';gpu=Simulation(project).run()
    for name in ('signals','electric','magnetic','frames'):
        np.testing.assert_allclose(getattr(gpu,name),getattr(cpu,name),rtol=3e-5,atol=2e-7)
    for key in ('E_upper','H_upper'):
        np.testing.assert_allclose(gpu.endpoint_fields[key],cpu.endpoint_fields[key],rtol=3e-5,atol=2e-7)
    path=tmp_path/'cuda-endpoints.npz';gpu.save(path);loaded=Result.load(path)
    np.testing.assert_array_equal(loaded.endpoint_fields['E_upper'],gpu.endpoint_fields['E_upper'])
    assert gpu.summary['cuda_kernel']=='endpoint-direct' and gpu.summary['cuda_graph'] is False


def test_native_metadata_admission_scales_beyond_old_fixed_caps():
    from torchfdtd.endpoint_native import estimate_endpoint,admit_endpoint
    payload=scene().model_dump();payload['region'].update(size=(12.8,12.8,12.8),backend='cuda')
    project=Project.model_validate(payload)
    stats=estimate_endpoint(project)
    assert stats['endpoint_tensor_bytes']>256_000_000
    assert stats['endpoint_host_bytes']>64_000_000
    budgets=admit_endpoint(stats,use_cuda=True,gpu_free_bytes=16*1024**3,host_available_bytes=32*1024**3)
    assert budgets['tensor_budget_bytes']==stats['endpoint_tensor_bytes']
    assert budgets['host_preparation_budget_bytes']==stats['endpoint_preparation_bytes']
    with pytest.raises(ValueError,match='CUDA'):
        admit_endpoint(stats,use_cuda=True,gpu_free_bytes=stats['endpoint_tensor_bytes'],host_available_bytes=32*1024**3)
    with pytest.raises(ValueError,match='host'):
        admit_endpoint(stats,use_cuda=True,gpu_free_bytes=16*1024**3,host_available_bytes=stats['endpoint_host_bytes'])
