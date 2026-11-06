"""Public complex streamed history, online spectra and checkpoint gradients."""
import pytest
import torch
from torchfdtd import DifferentiableSimulation, StreamedSimulation, StreamedAdjointOptions, tune_streamed
from test_bloch_adjoint import scene
from test_differentiable import gpu


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('storage', ['host', 'disk'])
@pytest.mark.parametrize('checkpoints', [0, 2])
def test_complex_streamed_history_and_spectrum(tmp_path, device, storage, checkpoints):
    if device == 'cuda':gpu()
    p = scene(True)
    torch.manual_seed(719)
    parameter = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    pattern = torch.rand(p.region.shape+(3,), dtype=torch.float64)
    eps = 1.4+parameter.sigmoid()*pattern
    oracle = DifferentiableSimulation(p)
    expected = oracle.reference(eps)
    def loss(v):return (v.real+.4*v.imag).square().sum()+.2*v.abs().square().sum()
    wanted, = torch.autograd.grad(loss(expected), parameter, retain_graph=True)
    options = StreamedAdjointOptions(device=device, state_storage=storage,
        state_directory=tmp_path, disk_budget_bytes=128*1024**2, slab_width=5,
        temporal_depth=3, checkpoints=checkpoints, local_checkpoints=1,
        tile_transfers='async' if device == 'cuda' else 'sync', tile_buffers=3)
    model = StreamedSimulation(p, options)
    result = model(eps)
    actual, = torch.autograd.grad(loss(result.signals), parameter, retain_graph=True)
    torch.testing.assert_close(result.signals, expected, rtol=3e-10, atol=3e-12)
    torch.testing.assert_close(actual, wanted, rtol=3e-9, atol=3e-11)
    assert actual.abs() > 1e-8
    assert result.report['complex_fields']
    assert result.report['peak_block_checkpoints'] <= checkpoints
    frequencies = [.025/p.region.time_step, .07/p.region.time_step]
    window = torch.hann_window(p.region.steps, dtype=torch.float64)
    history = oracle(eps).spectrum(frequencies, window=window)/p.region.time_step
    hg, = torch.autograd.grad(loss(history), parameter, retain_graph=True)
    online = model.spectrum(eps, frequencies, window=window)
    fields = online.fields/p.region.time_step
    og, = torch.autograd.grad(loss(fields), parameter, retain_graph=True)
    torch.testing.assert_close(fields, history, rtol=3e-10, atol=3e-12)
    torch.testing.assert_close(og, hg, rtol=3e-9, atol=3e-11)
    again, = torch.autograd.grad(loss(fields), parameter)
    torch.testing.assert_close(again, og, rtol=0, atol=0)
    assert not list(tmp_path.iterdir())


def test_complex_policy_tuning_uses_real_loss():
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    policies = [StreamedAdjointOptions(device='cpu', slab_width=w, temporal_depth=3) for w in (4, 7)]
    result = tune_streamed(p, eps, candidates=policies, probe_steps=10, repeats=1)
    assert all(row['status'] == 'measured' for row in result.report['candidates'])
    lengths=result.report['calibration_steps']
    cached=sum(eps.numel()*eps.element_size()+n*len(p.monitors)*2*eps.element_size() for n in lengths)
    assert result.report['reference_cache_peak_bytes']==cached
    extra=result.report['tuning_reference_reservation_bytes']
    assert extra>=cached+2*eps.numel()*eps.element_size()
    assert all(row['reservation']['host_reservation_bytes']+extra<=policy.host_budget_bytes
               for row,policy in zip(result.report['candidates'],policies))
    assert eps.grad is None
