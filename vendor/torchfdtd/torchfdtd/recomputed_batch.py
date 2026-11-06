"""Sequential first-order VJPs for coupled objectives over independent cases."""
import math
import torch


def _output(value):
    if not isinstance(value, torch.Tensor) or not (value.is_floating_point() or value.is_complex()):
        raise ValueError('Each case must return a floating or complex tensor.')
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError('Case outputs must be nonempty and finite.')
    return value


def _case_vjp(case, parameters, expected, seed, rtol, atol):
    # A function scope releases this case graph before constructing the next.
    with torch.enable_grad():
        inputs = tuple(p.detach().requires_grad_(p.requires_grad) for p in parameters)
        value = _output(case(*inputs))
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise RuntimeError('Recomputed case changed output shape or dtype.')
        if not torch.allclose(value.detach().to(expected.device), expected, rtol=rtol, atol=atol):
            raise RuntimeError('Recomputed case output drifted. Cases must be deterministic and unchanged.')
        active = tuple(p for p in inputs if p.requires_grad)
        gradients = torch.autograd.grad(value, active, seed.to(value.device), allow_unused=True) if value.requires_grad else (None,) * len(active)
    iterator = iter(gradients)
    return tuple(next(iterator) if p.requires_grad else None for p in inputs)


class _RecomputedCases(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cases, output_device, budget, rtol, atol, *parameters):
        packed = None
        for index, case in enumerate(cases):
            value = _output(case(*parameters))
            if packed is None:
                required = len(cases) * value.numel() * value.element_size()
                if required > budget:
                    raise ValueError(f'Case output budget exceeded: {required} > {budget} bytes.')
                packed = torch.empty((len(cases), *value.shape), dtype=value.dtype, device=output_device)
            if value.shape != packed.shape[1:] or value.dtype != packed.dtype:
                raise ValueError('All cases must have identical output shape and dtype.')
            packed[index].copy_(value)
            del value
        ctx.cases, ctx.rtol, ctx.atol = cases, rtol, atol
        ctx.save_for_backward(packed, *parameters)
        return packed

    @staticmethod
    def backward(ctx, seed):
        if torch.is_grad_enabled():
            raise RuntimeError('Recomputed cases support first-order derivatives only.')
        packed, *parameters = ctx.saved_tensors
        accumulated = [None] * len(parameters)
        for index, case in enumerate(ctx.cases):
            gradients = _case_vjp(case, parameters, packed[index], seed[index], ctx.rtol, ctx.atol)
            for j, gradient in enumerate(gradients):
                if gradient is not None:
                    if accumulated[j] is None:
                        accumulated[j] = gradient.clone()
                    else:
                        accumulated[j].add_(gradient)
            del gradients
        return (None,) * 5 + tuple(accumulated)


def recompute_cases(cases, *parameters, output_device='cpu', output_budget_bytes=64 * 1024**2,
                    replay_rtol=1e-6, replay_atol=1e-9):
    """Stack compact case results, recomputing one case graph at a time in backward.

    Each callable receives every explicit design tensor and returns one tensor.
    Cases must be deterministic, side-effect free, and unchanged until backward.
    Hidden trainable closure tensors are unsupported. Keep outputs compact, for
    example detector channel powers rather than time histories or field volumes.

    The budget limits the packed result only. Per-case solver/geometry workspaces,
    design inputs, parameter gradients and caller objective are additional memory.
    This is sequential recomputation, not parallel simulation execution. Gradients
    are first order. Replay checks detect output drift, not arbitrary state changes.
    """
    cases = tuple(cases)
    if not cases or not all(callable(case) for case in cases):
        raise ValueError('Provide at least one callable case.')
    if not parameters or not all(isinstance(p, torch.Tensor) and (p.is_floating_point() or p.is_complex()) for p in parameters):
        raise ValueError('Provide explicit floating or complex design tensors.')
    if isinstance(output_budget_bytes, bool) or not isinstance(output_budget_bytes, int) or output_budget_bytes <= 0:
        raise ValueError('Output budget must be a positive integer byte count.')
    if any(not math.isfinite(v) or v < 0 for v in (replay_rtol, replay_atol)):
        raise ValueError('Replay tolerances must be finite and nonnegative.')
    return _RecomputedCases.apply(cases, torch.device(output_device), output_budget_bytes,
                                  replay_rtol, replay_atol, *parameters)
