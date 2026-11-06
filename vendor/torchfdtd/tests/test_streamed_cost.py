import pytest
import torch

from torchfdtd import StreamedAdjointOptions, StreamedSimulation
from torchfdtd.streamed_cost import predict_duration, replay_blocks
from test_differentiable import project


@pytest.mark.parametrize('checkpoints',[0,1,2,4])
def test_replay_count_matches_actual_partial_block_schedule(checkpoints):
    p = project(steps=13,periodic=True)
    epsilon = torch.full(p.region.shape,1.7,dtype=torch.float64,requires_grad=True)
    options = StreamedAdjointOptions(device='cpu',slab_width=8,temporal_depth=3,checkpoints=checkpoints)
    result = StreamedSimulation(p,options)(epsilon)
    result.signals.square().sum().backward()
    assert result.report['replayed_blocks'] == replay_blocks(5,checkpoints)


def test_two_duration_prediction_recovers_known_startup_and_replay_costs():
    def measurement(steps):
        blocks=steps//4
        forward=1.+blocks*.1
        backward=2.+replay_blocks(blocks,2)*.1+blocks*.5
        return dict(steps=steps,forward=forward,backward=backward,total=forward+backward+.03)
    predicted=predict_duration(measurement(12),measurement(24),steps=120,depth=4,checkpoints=2)
    assert predicted['seconds'] == pytest.approx(measurement(120)['total'])
    assert predicted['method'] == 'two-duration replay cost'
    assert predicted['startup_forward_seconds'] == pytest.approx(1.)


def test_noisy_negative_slope_falls_back_and_full_measurement_is_preferred():
    short=dict(steps=12,total=3.,forward=2.,backward=1.)
    long=dict(steps=24,total=2.,forward=1.,backward=1.)
    prediction=predict_duration(short,long,steps=120,depth=4,checkpoints=2)
    assert prediction['method'] == 'work-count fallback'
    assert prediction['seconds'] >= long['total']
    prediction=predict_duration(short,long,steps=24,depth=4,checkpoints=2)
    assert prediction['seconds'] == 2.
