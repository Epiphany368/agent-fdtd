from dataclasses import replace

import pytest
import torch

from torchfdtd import StreamedAdjointOptions, StreamedSimulation, DifferentiableSimulation, tune_streamed
from test_differentiable import project


def test_tuning_preserves_design_and_selects_full_duration_policy():
    p = project(steps=13, periodic=True)
    epsilon = torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    epsilon.grad = torch.full_like(epsilon,3.)
    original = epsilon.detach().clone()
    base = StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2)
    candidates = [replace(base,host_budget_bytes=1), base, replace(base,slab_width=6,temporal_depth=3)]
    with torch.no_grad():
        tuned = tune_streamed(p,epsilon,candidates=candidates,probe_steps=10,repeats=1)
    assert tuned.report['candidates'][0]['status'] == 'rejected'
    assert tuned.options in candidates[1:]
    assert tuned.report['probe_steps'] == 10 and p.region.steps == 13
    assert tuned.report['calibration_steps'] == [10,12,13]
    assert tuned.report['candidates'][1]['calibration_steps'] == [10,13]
    assert tuned.report['candidates'][2]['calibration_steps'] == [12,13]
    assert tuned.report['reference_policy_index'] == 1
    assert tuned.report['extra_reference_evaluations'] == 1
    assert tuned.report['candidates'][tuned.report['selected_index']]['prediction']['method'] == 'measured full duration'
    torch.testing.assert_close(epsilon,original,rtol=0,atol=0)
    torch.testing.assert_close(epsilon.grad,torch.full_like(epsilon,3.),rtol=0,atol=0)
    result = StreamedSimulation(p,tuned.options)(epsilon)
    reference = DifferentiableSimulation(p).reference(epsilon)
    assert result.signals.shape[0] == 13
    torch.testing.assert_close(result.signals,reference,rtol=1e-11,atol=1e-12)


def test_tuner_rejects_empty_or_inadmissible_search():
    p = project(steps=10)
    epsilon = torch.ones(p.region.shape,dtype=torch.float64)
    with pytest.raises(ValueError,match='one to twelve'):tune_streamed(p,epsilon,candidates=[])
    with pytest.raises(ValueError,match='No streamed'):
        tune_streamed(p,epsilon,candidates=[StreamedAdjointOptions(device='cpu',host_budget_bytes=1)])
    with torch.inference_mode(),pytest.raises(ValueError,match='inference_mode'):
        tune_streamed(p,epsilon)


def test_deep_default_search_compares_local_replay_without_mutating_design():
    p = project(steps=17,periodic=True)
    epsilon = torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    options = StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=8)
    tuned = tune_streamed(p,epsilon,options=options,probe_steps=10,repeats=1)
    measured = [row for row in tuned.report['candidates'] if row['status']=='measured']
    assert {row['policy']['local_checkpoints'] for row in measured} == {0,1}
    assert epsilon.grad is None
    assert tuned.options.local_checkpoints in (0,1)


@pytest.mark.parametrize('steps,probe,depth,expected',[
    (1000,12,32,[32,64]),(1000,10,3,[12,24]),(53,12,32,[32,53]),
    (10,12,32,[10]),(1000,256,7,[259,518]),
])
def test_calibration_uses_whole_blocks_unless_measuring_full_duration(steps,probe,depth,expected):
    from torchfdtd.streamed_tuning import _calibration_lengths
    lengths = _calibration_lengths(steps,probe,depth,'replay_cost',1024)
    assert lengths == expected
    assert all(length == steps or length%depth == 0 for length in lengths)


def test_calibration_limit_rejects_before_any_simulation(monkeypatch):
    p = project(steps=1000)
    epsilon = torch.ones(p.region.shape,dtype=torch.float64)
    def forbidden(*a,**kw):raise AssertionError('Calibration ran beyond its declared limit')
    monkeypatch.setattr('torchfdtd.streamed_tuning.StreamedSimulation',forbidden)
    options = StreamedAdjointOptions(device='cpu',temporal_depth=300)
    with pytest.raises(ValueError,match='max_calibration_steps'):
        tune_streamed(p,epsilon,candidates=[options])


def test_unique_duration_is_checked_against_the_reference_policy(monkeypatch):
    p = project(steps=13,periodic=True)
    epsilon = torch.full(p.region.shape,1.7,dtype=torch.float64)
    base = StreamedAdjointOptions(device='cpu',slab_width=4,temporal_depth=2)
    def corrupt_unique_duration(scene,options):
        model = StreamedSimulation(scene,options)
        def run(design):
            result = model(design)
            if scene.region.steps == 12 and options.temporal_depth == 3:
                result.signals = result.signals*1.1
            return result
        return run
    monkeypatch.setattr('torchfdtd.streamed_tuning.StreamedSimulation',corrupt_unique_duration)
    with pytest.raises(AssertionError):
        tune_streamed(p,epsilon,candidates=[base,replace(base,temporal_depth=3)],probe_steps=10,repeats=1)


@pytest.mark.parametrize('limit,expected',[(30,[12,24]),(50,[12,24,48])])
def test_refinement_uses_longer_blocks_within_the_calibration_limit(limit,expected):
    from torchfdtd.streamed_cost import predict_duration
    p = project(steps=50,periodic=True)
    epsilon = torch.full(p.region.shape,1.7,dtype=torch.float64)
    options = StreamedAdjointOptions(device='cpu',slab_width=8,temporal_depth=3)
    tuned = tune_streamed(p,epsilon,candidates=[options],probe_steps=10,repeats=1,
                          max_calibration_steps=limit,refine_candidates=1)
    row = tuned.report['candidates'][0]
    assert row['calibration_steps'] == expected
    assert tuned.report['refined_indices'] == ([0] if limit==50 else [])
    predicted = predict_duration(*row['probes'][-2:],steps=50,depth=3,checkpoints=options.checkpoints)
    assert row['prediction'] == predicted
