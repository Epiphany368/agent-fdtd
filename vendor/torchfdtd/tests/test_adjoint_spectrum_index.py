"""Cached index tensors of the online DFT blocks against the list indexing they replace.

The functions ``legacy_accumulate`` and ``legacy_transpose`` are the 0.16.0
methods. Every comparison is bitwise (``torch.equal``).
"""
from types import SimpleNamespace

import pytest
import torch

from torchfdtd import AdjointOptions, DifferentiableSimulation
from torchfdtd.adjoint_spectrum import SpectralObservation
from test_differentiable import project, gpu


def legacy_accumulate(self, output, samples, start):
    stop = start+samples.shape[0]
    for family, indices in self.groups.items():
        if indices:
            output[:, indices] += self.kernel(start, stop, family)@samples[:, indices].to(self.complex_dtype)


def legacy_transpose(self, seed, start, stop):
    result = torch.empty((stop-start, len(self.components)), dtype=self.complex_dtype if self.complex_samples else self.dtype, device=self.device)
    for family, indices in self.groups.items():
        if indices:
            values=self.kernel(start, stop, family).conj().T@seed[:, indices]
            result[:, indices] = values if self.complex_samples else values.real
    return result


def use_legacy_blocks(monkeypatch):
    monkeypatch.setattr(SpectralObservation, 'accumulate', legacy_accumulate)
    monkeypatch.setattr(SpectralObservation, 'transpose', legacy_transpose)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('complex_samples', [False, True])
@pytest.mark.parametrize('windowed', [False, True])
def test_indexed_blocks_equal_list_indexing(device, dtype, complex_samples, windowed):
    if device == 'cuda':gpu()
    region = SimpleNamespace(steps=37, time_step=project().region.time_step, complex_fields=complex_samples)
    epsilon = torch.ones((), dtype=dtype, device=device)
    # Interleaved families, so neither group is a contiguous column range.
    components = ('Hy', 'Ex', 'Ez', 'Hx', 'Ex', 'Hz', 'Ey', 'Hy')
    frequency = torch.tensor([.013, .041, .087], dtype=dtype)/region.time_step
    window = torch.linspace(.2, 1, region.steps, dtype=dtype) if windowed else None
    legacy = SpectralObservation(epsilon, region, components, frequency, window, block_size=5)
    native = SpectralObservation(epsilon, region, components, frequency, window, block_size=5)
    generator = torch.Generator().manual_seed(4071)
    samples = torch.randn((region.steps, len(components)), dtype=native.complex_dtype if complex_samples else dtype,
                          generator=generator).to(device)
    expected, actual = legacy.zeros(), native.zeros()
    for start in range(0, region.steps, 5):
        block = samples[start:start+5]
        legacy_accumulate(legacy, expected, block, start)
        native.accumulate(actual, block, start)
        assert torch.equal(actual, expected)
    assert set(native._index_tensors) == {'E', 'H'}
    assert all(index.device == actual.device for index in native._index_tensors.values())
    seed = torch.randn((len(components), frequency.numel()), dtype=native.complex_dtype, generator=generator).to(device).T
    assert not seed.is_contiguous()
    for start in range(0, region.steps, 5):
        stop = min(start+5, region.steps)
        reference = legacy_transpose(legacy, seed, start, stop)
        values = native.transpose(seed, start, stop)
        assert values.dtype == reference.dtype and torch.equal(values, reference)
    assert expected.abs().max() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dimension,precision,diagonal', [
    ('2d', 'float64', False), ('2d', 'float64', True), ('3d', 'float32', False), ('3d', 'float32', True)])
def test_differentiable_spectrum_and_gradient_equal_list_indexing(monkeypatch, device, dimension, precision, diagonal):
    if device == 'cuda':gpu()
    p = project(dimension=dimension, precision=precision, steps=19, periodic=diagonal)
    dtype = getattr(torch, precision)
    shape = p.region.shape+((3,) if diagonal else ())
    generator = torch.Generator().manual_seed(3310)
    base = (1.4+.3*torch.rand(shape, dtype=dtype, generator=generator)).to(device)
    frequency = torch.tensor([.017, .052], dtype=dtype)/p.region.time_step
    model = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))

    def run():
        epsilon = base.clone().requires_grad_()
        fields = model.spectrum(epsilon, frequency.to(device), block_size=4).fields/p.region.time_step
        seed = torch.randn(fields.shape, dtype=fields.dtype, generator=torch.Generator().manual_seed(77)).to(device)
        gradient, = torch.autograd.grad(fields, epsilon, seed)
        return fields.detach(), gradient

    native_fields, native_gradient = run()
    with monkeypatch.context() as patch:
        use_legacy_blocks(patch)
        legacy_fields, legacy_gradient = run()
    assert torch.equal(native_fields, legacy_fields)
    assert torch.equal(native_gradient, legacy_gradient)
    assert native_gradient.abs().max() > 0
