"""Bounded exact-value CPU response/VJP reuse for fixed periodic models."""
from collections import OrderedDict
import hashlib
import json
import threading

import torch


def _tensor_key(value):
    value = value.detach().contiguous()
    if value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64):
        raise ValueError('Cached periodic tensors must be real CPU FP32 or FP64.')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Cannot cache nonfinite periodic tensors.')
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


class PeriodicResponseCache:
    """Opt-in LRU of detached responses and exact seeded density VJPs.

    Forward keys include fixed-model identity and every density value. A VJP
    additionally requires an identical incoming objective seed. No tolerance,
    interpolation, field history or autograd graph is stored. Tensor retention
    is bounded. Python metadata and returned/transient copies are additional.
    max_entries also bounds small-response metadata growth. The layer reserves
    an additional metadata allowance, not an exact process-RSS limit.
    Locking protects storage and counters, not concurrent model execution.
    Misses compute outside the lock and may duplicate simultaneous work.
    The layer API supplies identities and admission. This is not a general
    cache for mutable Torch modules or hidden trainable closure parameters.
    """
    def __init__(self, budget_bytes=64*1024**2, *, max_entries=4096):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError('Response cache budget must be a nonnegative integer.')
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError('Response cache max_entries must be a positive integer.')
        self.max_entries = max_entries
        self.metadata_allowance_bytes = 1024*max_entries
        self.budget_bytes = budget_bytes
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.tensor_bytes = 0
        self._counts = dict(forward_hits=0, forward_misses=0, vjp_hits=0, vjp_misses=0, evictions=0)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.tensor_bytes = 0

    def statistics(self):
        with self._lock:
            return dict(self._counts, tensor_bytes=self.tensor_bytes,
                budget_bytes=self.budget_bytes, entries=len(self._entries), max_entries=self.max_entries)

    def _get(self, key, kind):
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._counts[kind+'_misses'] += 1
                return None
            self._entries[key] = entry
            self._counts[kind+'_hits'] += 1
            return entry.clone()

    def _put(self, key, value):
        if value.device.type != 'cpu' or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError('Cache entries must be finite detached CPU tensors.')
        size = value.numel()*value.element_size()
        if size > self.budget_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.tensor_bytes -= old.numel()*old.element_size()
            while self.tensor_bytes+size > self.budget_bytes or len(self._entries) >= self.max_entries:
                _, old = self._entries.popitem(last=False)
                self.tensor_bytes -= old.numel()*old.element_size()
                self._counts['evictions'] += 1
            self._entries[key] = value.clone()
            self.tensor_bytes += size

    def _forward(self, key, density, compute):
        result = self._get(('forward', key), 'forward')
        if result is None:
            with torch.no_grad():
                result = compute(density).detach()
            if result.shape != (2, 4) or result.device.type != 'cpu' or result.dtype != density.dtype:
                raise ValueError('Expected a CPU periodic response of shape (2,4) matching density precision.')
            self._put(('forward', key), result)
        return result

    def _evaluate(self, namespace, density, compute, validate):
        if not self.budget_bytes:
            return compute(density)
        key = (namespace, _tensor_key(density))
        if torch.is_grad_enabled() and density.requires_grad:
            return _CachedPeriodic.apply(density, self, key, compute, validate)
        return self._forward(key, density, compute)


class _CachedPeriodic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, cache, key, compute, validate):
        result = cache._forward(key, density, compute)
        ctx.cache, ctx.key, ctx.compute, ctx.validate = cache, key, compute, validate
        ctx.save_for_backward(density, result.detach().clone())
        return result

    @staticmethod
    def backward(ctx, seed):
        if torch.is_grad_enabled():
            raise RuntimeError('Cached periodic responses support first-order derivatives only.')
        density, expected = ctx.saved_tensors
        ctx.validate(ctx.key[0])
        key = ('vjp', ctx.key, _tensor_key(seed))
        gradient = ctx.cache._get(key, 'vjp')
        if gradient is None:
            with torch.enable_grad():
                leaf = density.detach().requires_grad_()
                response = ctx.compute(leaf)
                if not torch.equal(response.detach(), expected):
                    raise RuntimeError('Cached periodic response replay changed. Exact reuse requires deterministic fixed models.')
                gradient, = torch.autograd.grad(response, leaf, seed)
            gradient = gradient.detach()
            ctx.cache._put(key, gradient)
        return gradient, None, None, None, None
