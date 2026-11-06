import threading
import json

import numpy as np
import pytest
import torch

from torchfdtd import Simulation, RunControl, Source, TimeSignal, Result
from torchfdtd.run_control import DecayDecision, source_end_time
from test_solver import small


def decay_project(backend='cpu', kernel='torch'):
    p = small(backend=backend)
    p.structures = []
    p.sources[0].center = (0,0,0)
    p.region.steps = 1400
    p.region.snapshot_interval = 500
    p.region.cuda_kernel = kernel
    p.region.run_control = RunControl(auto_shutoff=True, decay_threshold=1e-5,
                                     check_interval=25, consecutive_checks=3)
    return p


def test_decay_requires_source_end_and_a_sustained_window():
    c = RunControl(auto_shutoff=True, min_steps=10, consecutive_checks=3, decay_threshold=.01)
    d = DecayDecision(c, 20., 1.)
    assert not d.update(10, 1., 1.)
    assert not d.update(15, 0., 0.)
    assert not d.update(20, .001, .001)
    assert not d.update(21, .001, .001)
    assert not d.update(22, .1, .1)  # resets the window
    assert not d.update(23, .001, .001)
    assert not d.update(24, .001, .001)
    assert d.update(25, .001, .001)
    with pytest.raises(FloatingPointError, match='grew'):
        d.update(26, 2e6, 2e3)


def test_auto_shutoff_is_successful_and_preserves_recorded_prefix(tmp_path):
    p = decay_project()
    progress = []
    stopped = Simulation(p).run(progress=progress.append)
    assert stopped.summary['termination_reason']=='decayed'
    assert not stopped.summary['cancelled']
    assert stopped.summary['steps'] < p.region.steps
    assert stopped.frame_steps[-1]==stopped.summary['steps']
    assert progress[-1]['termination_reason']=='decayed'
    p.region.run_control.auto_shutoff=False
    full = Simulation(p).run()
    np.testing.assert_array_equal(stopped.signals, full.signals[:len(stopped.times)])
    relative_tail = np.linalg.norm(full.signals[len(stopped.times):])/np.linalg.norm(full.signals)
    assert relative_tail < .003
    stopped.save(tmp_path/'decayed.npz')
    assert Result.load(tmp_path/'decayed.npz').summary==json.loads(json.dumps(stopped.summary))


def test_delayed_sampled_source_and_continuous_source_cannot_stop_early():
    p = decay_project()
    p.region.steps=120
    p.sources = [Source(pulse='sampled', signal=TimeSignal(
        time_s=[0,1e-12,1.1e-12], amplitude=[0,1,0],phase_rad=[0,1,0]))]
    assert source_end_time(p)==1.1e-12
    assert Simulation(p).run().summary['termination_reason']=='max_steps'
    p.sources=[Source(pulse='continuous',pulse_cycles=1)]
    result=Simulation(p).run()
    assert result.summary['termination_reason']=='max_steps'
    assert result.summary['source_end_s'] is None
    assert any('continuous source' in s for s in result.summary['warnings'])


def test_zero_source_completion_and_explicit_cancellation_are_distinct():
    p=decay_project();p.sources=[]
    assert Simulation(p).run().summary['termination_reason']=='decayed'
    event=threading.Event();event.set()
    cancelled=Simulation(p).run(cancel=event)
    assert cancelled.summary['termination_reason']=='cancelled'
    assert cancelled.summary['cancelled'] and cancelled.summary['steps']==0


def test_nonfinite_fields_outside_snapshot_are_detected(monkeypatch):
    from torchfdtd.boundaries import YeeGrid
    original=YeeGrid.update_H
    def bad_update(grid):
        original(grid)
        grid.H[1,1,1,0]=np.nan  # display is a different z plane and an E component
    monkeypatch.setattr(YeeGrid,'update_H',bad_update)
    p=small(dimension='3d');p.region.run_control.check_interval=10
    with pytest.raises(FloatingPointError,match='step 10'):
        Simulation(p).run()


def test_absolute_field_limit():
    p=small();p.region.run_control=RunControl(field_limit=1e-6,check_interval=10)
    with pytest.raises(FloatingPointError,match='magnitude limit'):
        Simulation(p).run()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('kernel',['torch','fused'])
def test_gpu_shutoff_matches_cpu_and_eager(kernel):
    if kernel=='fused':pytest.importorskip('cupy')
    cpu=Simulation(decay_project()).run()
    gpu=Simulation(decay_project('cuda',kernel)).run()
    eager=Simulation(decay_project('cuda',kernel)).run(cuda_graph=False)
    for result in (gpu,eager):
        assert result.summary['termination_reason']=='decayed'
        assert result.summary['steps']==cpu.summary['steps']
        np.testing.assert_allclose(result.electric,cpu.electric,rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(result.signals,cpu.signals,rtol=1e-8,atol=1e-12)


def test_run_control_invalid_values_rejected():
    for values in ({'decay_threshold':0},{'decay_threshold':1},
                   {'consecutive_checks':1},{'growth_limit':1},{'field_limit':float('nan')}):
        with pytest.raises(ValueError):RunControl(**values)


def completed_steps(result):
    return result.summary['steps']


def test_decayed_batch_evaluates_objective_and_remains_resumable(tmp_path):
    from torchfdtd import BatchCase, BatchRunner
    p=decay_project();p.sources=[]
    with BatchRunner(backend='cpu',max_workers=1) as runner:
        cases=[BatchCase('decay',p)]
        report=runner.run(cases,objective=completed_steps,objective_key='steps-v1',output_dir=tmp_path)
        report.raise_for_errors()
        assert report.items[0].status=='completed'
        assert report.items[0].summary['auto_shutoff']
        assert report.items[0].metrics['objective']<p.region.steps
        resumed=runner.run(cases,objective=completed_steps,objective_key='steps-v1',output_dir=tmp_path,resume=True)
        assert resumed.items[0].resumed
