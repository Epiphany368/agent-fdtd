"""Incoherent spectral/pupil assembly with bounded case graph residency."""
import torch
from .recomputed_batch import recompute_cases


def spectral_pupil_response(cases, density, ray_weights, *, output_device='cpu',
                            output_budget_bytes=64 * 1024**2,
                            replay_rtol=1e-6, replay_atol=1e-9):
    """Return (channel, wavelength) responses from wavelength-major ray cases.

    cases[wavelength][ray](density) returns (polarization, channel) power
    responses. Polarizations are equally weighted and mutually incoherent.
    Ray weights are fixed intensity weights and are NOT normalized here.
    This preserves illumination falloff and exact-weight pupil subsets.
    Cases must include their own fixed material indices, phase and reference.
    Only one case graph is replayed at a time, with first-order derivatives.
    See recompute_cases for determinism and memory-budget restrictions.
    """
    rows=tuple(tuple(row) for row in cases)
    if not rows or not rows[0] or any(len(row)!=len(rows[0]) for row in rows):
        raise ValueError('Cases must be a nonempty rectangular wavelength/ray schedule.')
    if not isinstance(density,torch.Tensor) or density.dtype not in (torch.float32,torch.float64):
        raise ValueError('Density must be a real floating tensor.')
    raw=torch.as_tensor(ray_weights)
    if raw.is_complex() or raw.requires_grad:
        raise ValueError('Ray weights must be fixed real intensity weights.')
    weights=torch.as_tensor(ray_weights,device=output_device,dtype=density.dtype)
    if weights.shape!=(len(rows[0]),) or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()) or not bool(weights.sum()>0):
        raise ValueError('Ray weights must be finite, nonnegative and have positive total.')
    packed=recompute_cases([case for row in rows for case in row],density,
        output_device=output_device,output_budget_bytes=output_budget_bytes,
        replay_rtol=replay_rtol,replay_atol=replay_atol)
    if packed.ndim!=3 or packed.is_complex() or packed.dtype!=density.dtype or not bool((packed>=0).all()):
        raise ValueError('Each case must return real nonnegative (polarization, channel) responses matching density dtype.')
    values=packed.reshape(len(rows),len(rows[0]),*packed.shape[1:])
    return (values.mean(2)*weights[None,:,None]).sum(1).T
