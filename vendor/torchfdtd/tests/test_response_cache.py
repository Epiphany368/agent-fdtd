from unittest.mock import patch

import pytest
import torch

from torchfdtd import PeriodicLayerResponse, PeriodicResponseCache
from test_periodic_adjoint import model, SPEC, SETTINGS, policy, BUDGET
from torchfdtd import AdjointBatchOptions


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_exact_hard_structure_reuses_seeded_vjp_with_new_latent_chain():
    cache = PeriodicResponseCache(4096)
    calls = []
    def compute(density):
        calls.append(1)
        return density.square().sum().expand(2, 4).clone()
    hard = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    for magnitude, weight, expected_calls in ((.1, 1., 2), (.3, 1., 2), (.3, 2., 3)):
        logits = ((2*hard-1)*magnitude).requires_grad_()
        soft = logits.sigmoid()
        density = hard + (soft-soft.detach())
        response = cache._evaluate('fixed', density, compute, lambda key: None)
        gradient, = torch.autograd.grad(weight*response.sum(), logits)
        expected = 16*weight*hard*soft*(1-soft)
        torch.testing.assert_close(gradient, expected)
        assert len(calls) == expected_calls
    changed = (.9*hard).requires_grad_()
    cache._evaluate('fixed', changed, compute, lambda key: None).sum().backward()
    assert len(calls) == 5
    assert cache.statistics()['vjp_hits'] == 1 and cache.statistics()['vjp_misses'] == 3
    assert all(not value.requires_grad and value.grad_fn is None for value in cache._entries.values())


def test_cache_limits_and_returned_copies_cannot_corrupt_retention():
    cache = PeriodicResponseCache(64, max_entries=1)
    calls = []
    def compute(density):
        calls.append(1)
        return density.square().sum().expand(2, 4).clone()
    density = torch.full((2, 2), .3)
    first = cache._evaluate('one', density, compute, lambda key: None)
    expected = first.clone()
    first.add_(10)
    actual = cache._evaluate('one', density, compute, lambda key: None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cache._evaluate('two', density, compute, lambda key: None)
    assert len(calls) == 2 and cache.statistics()['evictions'] == 1
    assert cache.tensor_bytes <= 64 and cache.statistics()['entries'] == 1
    cache.clear()
    assert cache.tensor_bytes == 0 and cache.statistics()['entries'] == 0
    # Oversize gradients are computed correctly but never exceed retention.
    small = PeriodicResponseCache(8)
    x = density.clone().requires_grad_()
    small._evaluate('one', x, compute, lambda key: None).sum().backward()
    torch.testing.assert_close(x.grad, 16*x.detach())
    assert small.tensor_bytes == 0


def test_cache_rejects_replay_drift_and_higher_derivatives():
    cache = PeriodicResponseCache(4096)
    calls = []
    def drifting(x):
        calls.append(1)
        return (x.square().sum()+len(calls)).expand(2, 4)
    x = torch.full((2, 2), .3, requires_grad=True)
    response = cache._evaluate('drift', x, drifting, lambda key: None)
    with pytest.raises(RuntimeError, match='replay changed'):
        response.sum().backward()
    assert cache.statistics()['vjp_hits'] == 0
    response = cache._evaluate('stable', x, lambda p:p.square().sum().expand(2, 4), lambda key: None)
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(response.sum(), x, create_graph=True)


def test_actual_periodic_response_reuses_same_gradient_without_another_solve(tmp_path):
    cache = PeriodicResponseCache(1024**2, max_entries=16)
    module = model('cpu', 'resident', tmp_path, dtype=torch.float32, response_cache=cache)
    d = torch.tensor([[.2, .4], [.5, .3]], dtype=torch.float32, requires_grad=True)
    first = module(d)
    gradient, = torch.autograd.grad(first.square().sum(), d)
    with patch.object(module, '_compute_response', side_effect=AssertionError('Repeated field solve')):
        again = module(d)
        reused, = torch.autograd.grad(again.square().sum(), d)
    torch.testing.assert_close(again, first, rtol=0, atol=0)
    torch.testing.assert_close(reused, gradient, rtol=0, atol=0)
    assert gradient.norm() > 0 and cache.statistics()['vjp_hits'] == 1
    assert module.plan()['response_cache_allowance_bytes'] > cache.budget_bytes


def test_fixed_model_identity_and_budget_changes_are_not_reused(tmp_path):
    cache = PeriodicResponseCache(1024**2, max_entries=16)
    module = model('cpu', 'resident', tmp_path, dtype=torch.float32, response_cache=cache)
    settings = dict(SETTINGS, density_shape=(2, 2), response_cache=cache,
        policy=policy('cpu', 'resident', tmp_path),
        batch_options=AdjointBatchOptions(host_budget_bytes=BUDGET, gpu_budget_bytes=BUDGET))
    changed = PeriodicLayerResponse(dict(SPEC, design_index=2.2), **settings)
    module.last_report = {}
    changed.last_report = {}
    # Cheap model-specific responses exercise shared-cache identities without
    # repeating optical tests for unchanged solver arithmetic.
    with patch.object(module, '_compute_response', lambda p:p.sum().expand(2, 4)), patch.object(
            changed, '_compute_response', lambda p:(2*p.sum()).expand(2, 4)):
        d = torch.full((2, 2), .3)
        a, b = module(d), changed(d)
        torch.testing.assert_close(b, 2*a)
    assert cache.statistics()['forward_misses'] == 2
    cache.budget_bytes += 1
    with pytest.raises(ValueError, match='Response-cache budget'):
        module(d)


def test_execution_context_change_before_cached_backward_is_rejected(tmp_path):
    cache = PeriodicResponseCache(1024**2, max_entries=16)
    module = model('cpu', 'resident', tmp_path, dtype=torch.float32, response_cache=cache)
    d = torch.full((2, 2), .3, requires_grad=True)
    module.last_report = {}
    with patch.object(module, '_compute_response', lambda p:p.square().sum().expand(2, 4)):
        result = module(d)
        with patch.object(module, '_response_namespace', return_value=('changed',)):
            with pytest.raises(RuntimeError, match='execution context changed'):
                result.sum().backward()
