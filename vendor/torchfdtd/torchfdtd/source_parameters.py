"""Differentiable temporal envelopes for fixed spatial source terms.

These functions construct waveform values only. The source-adjoint API consumes
actual additive source increments. A Gaussian defined here is not a promise of
matching every native pulse convention or its source-term scaling.
"""
from __future__ import annotations

import math
import torch


def gaussian_waveform(times_s: torch.Tensor, *, frequency_hz, sigma_s, delay_s,
                      amplitude=1., phase_rad=0., analytic=False) -> torch.Tensor:
    """Return a Gaussian sine or complex analytic waveform on fixed sample times.

    Define ``u = times_s - delay_s`` and ``angle = 2*pi*frequency_hz*u +
    phase_rad``. The envelope is ``amplitude*exp(-0.5*(u/sigma_s)**2)``.
    Real output multiplies the envelope by ``sin(angle)``. ``analytic=True``
    multiplies it by ``exp(1j*angle)`` instead, whose imaginary part is the real
    sine convention. Sigma is the amplitude-envelope standard deviation in
    seconds, not an intensity FWHM. Frequency is in Hz and phase in radians.

    ``times_s`` must be a fixed finite nonempty 1D float32/float64 Tensor. Its
    device and real dtype determine the output (complex64/complex128 for an
    analytic signal). Parameters are finite real scalar numbers or scalar
    Tensors. Tensor casts preserve their autograd graph, including transfers
    from a different device or precision. Frequency and sigma must be positive.
    Positions, profiles and sampling times are fixed. Scalar validation may
    synchronize a CUDA device, but no parameter is detached in the waveform.
    """
    if not isinstance(times_s, torch.Tensor):
        raise TypeError('times_s must be a Torch tensor defining dtype and device.')
    if (times_s.layout != torch.strided or times_s.dtype not in (torch.float32, torch.float64)
            or times_s.ndim != 1 or times_s.numel() == 0 or times_s.requires_grad):
        raise ValueError('times_s must be fixed nonempty 1D strided float32/float64 samples.')
    if not bool(torch.isfinite(times_s).all()):
        raise ValueError('times_s must contain only finite samples.')
    if not isinstance(analytic, bool):
        raise TypeError('analytic must be a bool.')

    def scalar(value, name, positive=False):
        if isinstance(value, bool):
            raise ValueError(f'{name} must be a real scalar, not bool.')
        raw = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if raw.layout != torch.strided or raw.ndim != 0 or raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError(f'{name} must be a real scalar.')
        result = (raw.to(device=times_s.device, dtype=times_s.dtype)
                  if isinstance(value, torch.Tensor) else
                  torch.as_tensor(value, device=times_s.device, dtype=times_s.dtype))
        if not bool(torch.isfinite(result)) or (positive and not bool(result > 0)):
            raise ValueError(f'{name} must be finite' + (' and positive.' if positive else '.'))
        return result

    frequency = scalar(frequency_hz, 'frequency_hz', True)
    sigma = scalar(sigma_s, 'sigma_s', True)
    delay = scalar(delay_s, 'delay_s')
    amplitude_value = scalar(amplitude, 'amplitude')
    phase = scalar(phase_rad, 'phase_rad')
    u = times_s - delay
    envelope = amplitude_value * torch.exp(-.5 * (u / sigma).square())
    angle = (2 * math.pi) * frequency * u + phase
    if analytic:
        return torch.complex(envelope * torch.cos(angle), envelope * torch.sin(angle))
    return envelope * torch.sin(angle)
