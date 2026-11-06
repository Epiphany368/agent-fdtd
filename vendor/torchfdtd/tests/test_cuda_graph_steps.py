"""Unrolled graphs must preserve every physical sample and host event."""
import threading

import numpy as np
import pytest
import torch

from torchfdtd import Simulation, run_tensor_batch, FieldMonitor, SpectrumSettings
from torchfdtd.cuda_graph import observation_schedule, validate_graph_steps
from torchfdtd.tuning import _result_digest
from test_tensor_batch import cases
from test_multipole import multi_material


def test_schedule_hits_all_barriers_without_losing_or_repeating_steps():
    for width in (1, 7, 16, 64):
        schedule = list(observation_schedule(173, width, [29, 50, 87, 100, 129, 150]))
        assert sum(n for _, n in schedule) == 173
        assert {29,50,87,100,129,150,173} <= {q for q,_ in schedule}
        assert all(n in (1,width) for _,n in schedule)
        np.testing.assert_array_equal(np.cumsum([n for _,n in schedule]), [q for q,_ in schedule])
    for value in (0, 65, True, 1.5):
        with pytest.raises(ValueError, match='integer'):validate_graph_steps(value)
    with pytest.raises(ValueError, match='requires CUDA'):validate_graph_steps(8, False)
    p=cases(1)[0];p.region.backend='cpu';p.region.cuda_kernel='torch'
    with pytest.raises(ValueError, match='requires CUDA'):Simulation(p).run(cuda_graph_steps=8)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@gpu
@pytest.mark.parametrize('precision', ['float32','float64'])
@pytest.mark.parametrize('kernel', ['torch','fused'])
def test_single_batch_spectral_material_outputs_and_exact_callbacks(precision,kernel):
    pytest.importorskip('cupy')
    projects=cases(2,precision,'3d')
    for p, interval in zip(projects,(37,53)):
        p.region.steps=173;p.region.snapshot_interval=interval
        p.region.cuda_kernel=kernel;p.region.cuda_monitor_kernel=kernel
        p.region.run_control.check_interval=29
        p.region.mesh_type='graded';p.region.mesh_auto_refine=False;p.region.mesh_max=.2
        p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='periodic'
        p.materials.append(multi_material());p.structures[0].material='three-pole'
        p.monitors.append(FieldMonitor(id='plane',center=(.8,0,0),size=(0,1,1),
            spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='end')))
    refs=[Simulation(p).run() for p in projects]
    for p,ref in zip(projects,refs):
        events=[]
        got=Simulation(p).run(cuda_graph_steps=16,progress=lambda s:events.append(s['step']))
        assert _result_digest(got)==_result_digest(ref)
        assert events==ref.frame_steps.tolist()
        assert got.summary['diagnostics']==ref.summary['diagnostics']
        assert got.summary['cuda_graph_replays']<got.summary['steps']
    # Without a callback, long runs between heterogeneous barriers use chunks.
    got=run_tensor_batch(projects,cuda_graph_steps=16)
    for item,ref in zip(got.items,refs):
        assert _result_digest(item.load())==_result_digest(ref)
        assert item.summary['diagnostics']==ref.summary['diagnostics']
    assert got.plan['cuda_graph_replays']<173
    # A callback is itself a barrier, even when that prevents chunking.
    a,b=[],[]
    run_tensor_batch(projects,progress=lambda s:a.append(s['step']))
    run_tensor_batch(projects,cuda_graph_steps=16,progress=lambda s:b.append(s['step']))
    assert a==b


@gpu
def test_decay_cancellation_and_tfsf_counter_reset():
    pytest.importorskip('cupy')
    from test_run_control import decay_project
    from benchmarks.tfsf_sources import box_project
    p=decay_project('cuda','fused')
    a=Simulation(p).run();b=Simulation(p).run(cuda_graph_steps=16)
    assert a.summary['termination_reason']==b.summary['termination_reason']=='decayed'
    assert a.summary['steps']==b.summary['steps']
    assert _result_digest(a)==_result_digest(b)
    event=threading.Event()
    def stop(state):event.set()
    a=Simulation(p).run(progress=stop,cancel=event)
    event.clear()
    b=Simulation(p).run(progress=stop,cancel=event,cuda_graph_steps=16)
    assert _result_digest(a)==_result_digest(b)
    p=box_project(dimension='2d');p.region.backend='cuda';p.region.cuda_kernel='fused'
    p.region.steps=137
    a=run_tensor_batch([p,p]);b=run_tensor_batch([p,p],cuda_graph_steps=16)
    assert [_result_digest(i.load()) for i in a.items]==[_result_digest(i.load()) for i in b.items]
    for width in (1,16):
        event.clear();steps=[]
        def stop_batch(state):steps.append(state['step']);event.set()
        r=run_tensor_batch([p,p,p],cohort_size=2,cuda_graph_steps=width,cancel=event,progress=stop_batch)
        assert all(i.status=='cancelled' for i in r.items)
        assert steps==[1] and r.items[2].pid is None


@gpu
def test_external_graph_unrolling_is_bitwise_and_handles_tail():
    pytest.importorskip('cupy')
    from benchmarks.open_source import upstream_run
    from benchmarks.spectral_ensemble import projects_for
    p=projects_for('sphere',32,173,1)[0]
    _,a=upstream_run(p,with_planes=True,plane_kernel='fused')
    _,b=upstream_run(p,with_planes=True,plane_kernel='fused',cuda_graph_steps=16)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


@gpu
def test_complex_and_oneway_graphs_keep_their_auxiliary_states():
    pytest.importorskip('cupy')
    from torchfdtd import Source
    p=cases(1,'float64','2d')[0]
    p.region.cuda_kernel='torch';p.region.steps=117
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
    p.region.bloch_phase=(0,.37,0)
    a=Simulation(p).run();b=Simulation(p).run(cuda_graph_steps=16)
    assert a.summary['complex_fields'] and _result_digest(a)==_result_digest(b)
    p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='periodic'
    p.region.bloch_phase=(0,0,0)
    p.region.cuda_kernel='fused'
    p.sources=[Source(kind='plane',injection='oneway',normal='x',center=(-.8,0,0),
                      size=(0,p.region.actual_size[1],0),wavelength=1,pulse_cycles=1)]
    a=run_tensor_batch([p,p]);b=run_tensor_batch([p,p],cuda_graph_steps=16)
    assert [_result_digest(i.load()) for i in a.items]==[_result_digest(i.load()) for i in b.items]
