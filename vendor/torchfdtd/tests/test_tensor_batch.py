import threading

import numpy as np
import pytest
import torch

from torchfdtd import run_tensor_batch, Simulation, BatchCase, Result, Source, FieldMonitor, SpectrumSettings
from test_solver import small
from test_multipole import multi_material

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


def cases(count=3,precision='float32',dimension='2d'):
    projects=[]
    for i in range(count):
        p=small('cuda',dimension,precision)
        p.region.steps=160;p.region.snapshot_interval=50;p.region.cuda_kernel='fused'
        p.region.material_sampling='yee';p.structures[0].radius=.3+i*.1
        p.sources[0].amplitude=.7+i*.1
        projects.append(p)
    return projects


@pytest.mark.parametrize('dimension',['2d','3d'])
@pytest.mark.parametrize('precision',['float32','float64'])
def test_tensor_cohort_isolated_and_matches_independent_fields(dimension,precision,tmp_path):
    pytest.importorskip('cupy')
    projects=cases(3,precision,dimension)
    reference=[Simulation(p).run() for p in projects]
    report=run_tensor_batch(projects,output_dir=tmp_path,objective=lambda r:np.max(abs(r.signals)))
    report.raise_for_errors()
    assert report.plan['batch_size']==3
    for item,ref in zip(report.items,reference):
        actual=item.load()
        for key in ('electric','magnetic','signals','times','frames','frame_steps','epsilon'):
            np.testing.assert_array_equal(getattr(actual,key),getattr(ref,key))
        np.testing.assert_array_equal(Result.load(item.output).signals,ref.signals)
        assert item.metrics['objective']==np.max(abs(ref.signals))
        assert actual.summary['timing_scope']=='whole_cohort'
    assert not np.array_equal(report.items[0].load().electric,report.items[1].load().electric)


def test_tensor_material_is_rejected_before_any_grid_is_built():
    from torchfdtd import Material
    p=small('cuda','3d','float32');p.region.material_sampling='yee'
    p.materials.append(Material(name='tensor',model='tensor',epsilon_tensor=(2.4,2.1,2.7,.15,0.,.1)))
    p.structures[0].material='tensor';p.sources[0].component='Ez'
    torch.cuda.synchronize();allocated=torch.cuda.memory_allocated()
    with pytest.raises(ValueError,match='does not implement tensor materials'):run_tensor_batch([p])
    assert torch.cuda.memory_allocated()==allocated


def test_tensor_graded_multipole_frequency_planes_and_eager():
    pytest.importorskip('cupy')
    projects=cases(2,'float64')
    for p in projects:
        p.region.mesh_type='graded';p.region.mesh_auto_refine=False;p.region.mesh_max=.2
        p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='periodic'
        p.materials.append(multi_material());p.structures[0].material='three-pole'
        p.sources=[Source(kind='plane',center=(-.8,0,0),size=(0,1,0),wavelength=1,pulse_cycles=1)]
        p.monitors.append(FieldMonitor(id='plane',center=(.8,0,0),size=(0,1,1),
            spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')))
    reference=[Simulation(p).run() for p in projects]
    for graph in (False,True):
        report=run_tensor_batch(projects,cuda_graph=graph)
        for item,ref in zip(report.items,reference):
            actual=item.load()
            np.testing.assert_array_equal(actual.electric,ref.electric)
            np.testing.assert_array_equal(actual.signals,ref.signals)
            np.testing.assert_array_equal(actual.frames,ref.frames)
            a,b=actual.field_monitor('plane'),ref.field_monitor('plane')
            np.testing.assert_array_equal(a['fields'],b['fields'])
            assert a['run_signature']==b['run_signature']


def test_tensor_rejects_incompatible_semantics_and_preserves_backend_state():
    pytest.importorskip('cupy')
    p,q=cases(2)
    q.region.mesh=.08
    with pytest.raises(ValueError,match='identical'):run_tensor_batch([p,q])
    q=cases(1)[0];q.region.run_control.auto_shutoff=True
    with pytest.raises(ValueError,match='fixed-duration'):run_tensor_batch([q])
    q=cases(1)[0];q.sources.append(q.sources[0].model_copy(update={'id':'another'}))
    before=torch.get_default_dtype()
    with pytest.raises(ValueError,match='non-overlapping'):run_tensor_batch([q])
    assert torch.get_default_dtype()==before
    with pytest.raises(ValueError,match='memory allowance'):run_tensor_batch([p],memory_fraction=1e-12)


def test_tensor_cancellation_empty_sources_and_objective_failure(tmp_path):
    pytest.importorskip('cupy')
    p=cases(1)[0];p.sources=[];p.monitors=[]
    event=threading.Event();event.set()
    report=run_tensor_batch([p],cancel=event,objective=lambda _:1/0)
    assert report.items[0].status=='cancelled'
    assert report.items[0].summary['steps']==0 and not report.items[0].metrics
    event.clear()
    report=run_tensor_batch([BatchCase('zero',p)],objective=lambda _:float('nan'),output_dir=tmp_path,keep_results=False)
    assert report.items[0].status=='failed'
    assert np.count_nonzero(report.items[0].load().electric)==0


def test_split_cohorts_match_and_cancel_pending_cases(tmp_path):
    pytest.importorskip('cupy')
    projects=cases(5)
    reference=run_tensor_batch(projects)
    actual=run_tensor_batch(projects,cohort_size=2,output_dir=tmp_path,keep_results=False)
    assert actual.plan['cohorts_completed']==3
    assert [c['batch_size'] for c in actual.plan['cohorts']]==[2,2,1]
    for a,b in zip(actual.items,reference.items):
        np.testing.assert_array_equal(a.load().electric,b.load().electric)
        np.testing.assert_array_equal(a.load().signals,b.load().signals)
    with pytest.raises(FileExistsError):run_tensor_batch(projects,cohort_size=2,output_dir=tmp_path)
    event=threading.Event()
    def stop(state):event.set()
    cancelled=run_tensor_batch(projects,cohort_size=2,cancel=event,progress=stop)
    assert all(i.status=='cancelled' for i in cancelled.items)
    assert all(i.pid is None for i in cancelled.items[2:])
    assert cancelled.plan['cohorts_completed']==1


def test_sparse_host_schedule_preserves_different_observation_intervals():
    pytest.importorskip('cupy')
    projects=cases(2,'float64')
    for p,snapshot,diagnostic in zip(projects,(37,69),(45,70)):
        p.region.snapshot_interval=snapshot
        p.region.run_control.divergence_check=True
        p.region.run_control.check_interval=diagnostic
    reference=[Simulation(p).run() for p in projects]
    for graph in (False,True):
        report=run_tensor_batch(projects,cuda_graph=graph)
        for item,ref in zip(report.items,reference):
            actual=item.load()
            np.testing.assert_array_equal(actual.frames,ref.frames)
            np.testing.assert_array_equal(actual.frame_steps,ref.frame_steps)
            np.testing.assert_array_equal(actual.electric,ref.electric)
            assert actual.summary['diagnostics']==ref.summary['diagnostics']
