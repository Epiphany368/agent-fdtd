from dataclasses import replace
import sys
import types
import pytest
import torch
from torchfdtd import Project,Region,Source,Monitor
from torchfdtd.streamed import StreamedAdjointOptions
from torchfdtd import streamed_planning as planning


def scene():
    return Project(region=Region(size=(1.6,1.6,1.6),mesh=.1,steps=40,pml_cells=3,precision='float32',backend='cpu'),
        sources=[Source(center=(0,0,0))],monitors=[Monitor(center=(.1,0,0))])


@pytest.fixture
def work(monkeypatch):
    module=types.ModuleType('torchfdtd.streamed_work')
    module.estimate_streamed_work=lambda p,o,diagonal=False:dict(total_state_io_bytes=100//o.temporal_depth if o.state_storage=='disk' else 0,
        total_cell_steps=1000//o.slab_width+o.temporal_depth)
    monkeypatch.setitem(sys.modules,'torchfdtd.streamed_work',module)
    monkeypatch.setattr(planning,'host_memory',lambda:dict(available_bytes=1024**3))


def test_metadata_only_candidates_and_snapshot(work,monkeypatch):
    original=torch.empty
    def guarded(*args,**kwargs):
        assert kwargs.get('device')=='meta','allocated non-meta tensor'
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'empty',guarded)
    p=scene();base=StreamedAdjointOptions(device='cpu',slab_width=2,temporal_depth=3,checkpoints=3)
    result=planning.plan_streamed_work(p,base)
    assert result.options is None and result.report['admitted_indices']
    assert {x['options']['temporal_depth'] for x in result.report['candidates']}=={1,3,6,12}
    assert {x['options']['checkpoints'] for x in result.report['candidates']}=={3}
    assert all(x['options']['state_storage']=='host' for x in result.report['candidates'])
    p.sources[0].amplitude=7
    assert result.project_snapshot['sources'][0]['amplitude']!=7
    exposed=result.report;exposed['candidates'].clear()
    assert result.report['candidates']


def test_metric_selection_pareto_and_rejection(work,monkeypatch):
    monkeypatch.setattr(planning,'estimate_streamed_memory',lambda p,o,diagonal=False:dict(host_reservation_bytes=o.slab_width*10,gpu_reservation_bytes=0))
    base=StreamedAdjointOptions(device='cpu',slab_width=2,temporal_depth=3)
    options=[base,replace(base,slab_width=8)]
    report=planning.plan_streamed_work(scene(),base,candidates=options,selection='min_compute_work')
    assert report.options==options[1]
    host_choice=planning.plan_streamed_work(scene(),base,candidates=options,selection='min_disk_traffic')
    assert host_choice.options==options[1]
    assert report.report['pareto_indices']==[0,1]
    monkeypatch.setattr(planning,'host_memory',lambda:dict(available_bytes=100))
    report=planning.plan_streamed_work(scene(),base,candidates=options,host_free_reserve_bytes=30,selection='min_compute_work')
    assert report.options==base
    assert 'RAM floor' in report.report['candidates'][1]['reason']


def test_unknown_ram_floor_and_bounded_inputs(work,monkeypatch):
    base=StreamedAdjointOptions(device='cpu')
    monkeypatch.setattr(planning,'host_memory',lambda:dict(available_bytes=None))
    with pytest.raises(ValueError,match='RAM'):planning.plan_streamed_work(scene(),base,host_free_reserve_bytes=1)
    with pytest.raises(ValueError,match='64'):planning.plan_streamed_work(scene(),base,candidates=(base for _ in range(65)))
    with pytest.raises(ValueError,match='checkpoint'):planning.plan_streamed_work(scene(),base,candidates=[replace(base,checkpoints=0)])


def test_disk_explicit_floor_without_creating_directory(work,monkeypatch,tmp_path):
    import torchfdtd.state_store as storage
    monkeypatch.setattr(storage,'disk_free',lambda directory:10)
    path=tmp_path/'never-created'
    base=StreamedAdjointOptions(device='cpu',state_storage='disk',state_directory=path,disk_budget_bytes=1024**3,disk_free_reserve_bytes=9)
    result=planning.plan_streamed_work(scene(),base,candidates=[base],selection='min_disk_traffic')
    assert result.options is None and not path.exists()
    assert 'disk' in result.report['candidates'][0]['reason'].lower()


def test_unsupported_field_observer_rejected_before_estimators(work,monkeypatch):
    from torchfdtd import FieldMonitor
    p=scene();p.monitors=[FieldMonitor()]
    monkeypatch.setattr(planning,'estimate_streamed_memory',lambda *a,**k:pytest.fail('estimator called'))
    with pytest.raises(ValueError,match='point'):planning.plan_streamed_work(p,StreamedAdjointOptions(device='cpu'))


def test_real_arithmetic_metadata_integration():
    from torchfdtd.streamed_work import estimate_streamed_work
    base=StreamedAdjointOptions(device="cpu",slab_width=4,temporal_depth=3)
    result=planning.plan_streamed_work(scene(),base,candidates=[base],selection="min_compute_work")
    assert result.options==base
    row=result.report["candidates"][0]
    assert row["work"]["total_cell_steps"]>0
    assert row["work"]["total_state_io_bytes"]==0
    assert row["memory"]["host_reservation_bytes"]>0
