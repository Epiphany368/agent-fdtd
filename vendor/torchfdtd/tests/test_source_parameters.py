import math

import pytest
import torch

from torchfdtd.source_parameters import gaussian_waveform


@pytest.mark.parametrize('analytic', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scalar_parameter_vjps_against_independent_finite_differences(dtype, analytic):
    # FP64 here is a small scalar-envelope calculus diagnostic, not FDTD fields.
    times = torch.linspace(-2e-15, 8e-15, 31, dtype=dtype)
    parameters = torch.tensor([.7, .31, 1.3, .42, .18], dtype=dtype, requires_grad=True)
    weights = torch.linspace(-.3, .8, len(times), dtype=dtype)

    def objective(p):
        signal = gaussian_waveform(times, amplitude=p[0], phase_rad=p[1],
            frequency_hz=p[2]*1e14, sigma_s=p[3]*1e-14, delay_s=p[4]*1e-14,
            analytic=analytic)
        assert signal.dtype == ({torch.float32:torch.complex64, torch.float64:torch.complex128}[dtype] if analytic else dtype)
        return (signal.real*weights).sum() + ((signal.imag*weights.flip(0)).sum()*.3 if analytic else 0)

    gradient, = torch.autograd.grad(objective(parameters), parameters)
    step = 2e-3 if dtype == torch.float32 else 1e-5
    estimates = []
    with torch.no_grad():
        for i in range(5):
            delta = torch.zeros_like(parameters); delta[i] = step
            estimates.append((objective(parameters+delta)-objective(parameters-delta))/(2*step))
    torch.testing.assert_close(gradient, torch.stack(estimates),
        rtol=3e-3 if dtype==torch.float32 else 2e-7,
        atol=3e-4 if dtype==torch.float32 else 2e-8)
    assert bool(torch.isfinite(gradient).all()) and bool((gradient.abs() > 1e-4).all())


def test_real_sine_analytic_imaginary_part_and_dtype_independent_of_global_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        times = torch.linspace(0, 2, 16, dtype=torch.float32)[::2]
        assert not times.is_contiguous()
        parameters = dict(frequency_hz=1.3, sigma_s=.6, delay_s=.4, amplitude=-.7, phase_rad=.2)
        real = gaussian_waveform(times, **parameters)
        complex_signal = gaussian_waveform(times, **parameters, analytic=True)
        assert real.dtype == torch.float32 and complex_signal.dtype == torch.complex64
        torch.testing.assert_close(real, complex_signal.imag, rtol=0, atol=0)
        expected = -.7*torch.exp(-.5*((times-.4)/.6).square())*torch.sin(2*math.pi*1.3*(times-.4)+.2)
        torch.testing.assert_close(real, expected, rtol=2e-6, atol=1e-7)
    finally:
        torch.set_default_dtype(previous)


def test_scalar_cast_preserves_trainable_leaf_gradient():
    phase = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0., .1, .3], dtype=torch.float32)
    output = gaussian_waveform(times, frequency_hz=1., sigma_s=1., delay_s=0., phase_rad=phase)
    gradient, = torch.autograd.grad(output.sum(), phase)
    assert gradient.dtype == torch.float64 and gradient.abs() > 0


@pytest.mark.parametrize('patch', [dict(frequency_hz=0), dict(frequency_hz=float('inf')),
    dict(sigma_s=-1), dict(delay_s=float('nan')), dict(amplitude=1j), dict(phase_rad=[0.]),
    dict(amplitude=True), dict(analytic=1)])
def test_invalid_scalar_contract_rejected(patch):
    values = dict(frequency_hz=1., sigma_s=1., delay_s=0.)
    values.update(patch)
    with pytest.raises((TypeError, ValueError)):
        gaussian_waveform(torch.arange(4, dtype=torch.float32), **values)


@pytest.mark.parametrize('times', [torch.ones(3, requires_grad=True), torch.ones(2, 2),
    torch.tensor([0, 1]), torch.tensor([float('nan')]), torch.empty(0)])
def test_invalid_or_trainable_times_rejected(times):
    with pytest.raises(ValueError):
        gaussian_waveform(times, frequency_hz=1., sigma_s=1., delay_s=0.)


def test_python_scalar_precision_is_not_limited_by_default_float32():
    times = torch.tensor([.123456789012345], dtype=torch.float64)
    phase = .123456789012345
    output = gaussian_waveform(times, frequency_hz=1.23456789012345,
        sigma_s=.987654321098765, delay_s=.0123456789012345, phase_rad=phase)
    u = float(times[0])-.0123456789012345
    expected = math.exp(-.5*(u/.987654321098765)**2)*math.sin(2*math.pi*1.23456789012345*u+phase)
    assert abs(float(output[0])-expected) < 2e-15
