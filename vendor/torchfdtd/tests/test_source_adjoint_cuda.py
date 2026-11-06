"""Focused source-transpose gates. GPU cases are collected without CUDA setup."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from torchfdtd.source_adjoint_cuda import SourceFusedAdjointCUDA


def test_source_transpose_order_and_online_observation_index():
    """CPU protocol oracle: observation once, H seed before H transpose, E after."""
    events = []
    e, h = torch.tensor([2.]), torch.tensor([3.])

    def observation(*args):
        events.append(('observation', int(args[-1][-1])))
        h.add_(5)

    def h_transpose(*args):
        events.append(('H transpose',))
        e.add_(h * 2)

    def e_transpose(*args):
        events.append(('E transpose',))

    def source(family, field, gradient, step):
        events.append((family, step, float(field.item())))

    wrapper = SourceFusedAdjointCUDA.__new__(SourceFusedAdjointCUDA)
    wrapper.system = SimpleNamespace(accumulate_waveform_gradient=source)
    wrapper.packed_gradient = torch.zeros(1)
    wrapper.kernel = SimpleNamespace(
        cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext())),
        device=0, stream=nullcontext, e_bar=e, h_bar=h, count=1, phase=0,
        observer=(observation, (), None, 1),
        launches={(True, 0): (h_transpose, (), None),
                  (False, 0): (e_transpose, (), None)})
    wrapper.step(9, observation_index=2)
    assert events == [('observation', 2), ('H', 9, 8.), ('H transpose',),
                      ('E', 9, 18.), ('E transpose',)]
    assert wrapper.phase == 1


def test_source_transpose_rejects_cpu_before_cuda_import():
    with pytest.raises(ValueError, match='CUDA carrier'):
        SourceFusedAdjointCUDA(None, torch.zeros(1), torch.zeros(1))


def _scene(complex_fields):
    from torchfdtd import BoundaryFace, Monitor, Source
    from test_differentiable import project
    p = project('3d', precision='float32', steps=10, periodic=True)
    p.region.cuda_kernel = 'fused'
    if complex_fields:
        p.region.boundaries.x_min = BoundaryFace(kind='bloch')
        p.region.boundaries.x_max = BoundaryFace(kind='bloch')
        p.region.bloch_phase = (.47, 0., 0.)
    p.sources = [Source(kind='plane', normal='z', size=(.6, .5, 0),
                        center=(0, 0, 0), component='Ex'),
                 Source(kind='plane', normal='z', size=(.6, .5, 0),
                        center=(0, 0, 0), component='Hy'),
                 Source(center=(0, 0, 0), component='Ex'),
                 Source(center=(0, 0, 0), component='Hy')]
    p.monitors = [Monitor(center=(0, 0, 0), component=c) for c in ('Ex', 'Hy', 'Ex')]
    return p


def _system(project, epsilon, waveforms):
    """Test-only augmented primitive with explicit real/imag carrier lanes."""
    from torchfdtd.differentiable import _System
    system = _System(project, epsilon)
    terms = []
    column = 0
    for family in ('E', 'H'):
        replaced = []
        for loc, component, _, profile in system.sources[family]:
            replaced.append((loc, component, waveforms[:, column], profile))
            terms.append((family, loc, component, profile, column))
            column += 1
        system.sources[family] = replaced
    n = epsilon.numel()
    count = waveforms.numel()

    def material_view(gradient):
        return gradient[:n].view_as(epsilon)

    def accumulate(family, field, gradient, step):
        real = gradient[n:n + count].view(waveforms.shape)
        imag = gradient[n + count:].view(waveforms.shape) if waveforms.is_complex() else None
        for term_family, loc, component, profile, column in terms:
            if term_family != family:
                continue
            seed = field[loc + (component,)]
            value = seed.sum() if profile is None else (profile.conj() * seed).sum()
            real[step, column].add_(value.real)
            if imag is not None:
                imag[step, column].add_(value.imag)

    system.material_gradient_view = material_view
    system.accumulate_waveform_gradient = accumulate
    return system


@pytest.mark.parametrize('complex_fields,complex_waveforms', [(False, False), (True, False), (True, True)])
def test_fused_augmented_step_matches_cpu_autograd_on_nondefault_stream(complex_fields, complex_waveforms):
    """Independent full state/material/source oracle, including CPML cotangents."""
    from test_differentiable import gpu
    gpu()
    p = _scene(complex_fields)
    generator = torch.Generator().manual_seed(7192)
    epsilon = (1.5 + .1 * torch.rand(p.region.shape + (3,), generator=generator)).requires_grad_()
    wave_dtype = torch.complex64 if complex_waveforms else torch.float32
    waves = (torch.randn((p.region.steps, 4), generator=generator, dtype=wave_dtype) * .03).requires_grad_()
    cpu = _system(p, epsilon, waves)
    state = tuple((torch.randn(v.shape, dtype=v.dtype, generator=generator) * .1).requires_grad_()
                  for v in cpu.state())
    bars = tuple(torch.randn(v.shape, dtype=v.dtype, generator=generator) for v in state)
    seed = torch.randn((1, len(cpu.monitors)), dtype=cpu.field_dtype, generator=generator)
    final = cpu.reference_step(state, 2, epsilon)
    loss = sum((v.conj() * b).real.sum() for v, b in zip(final, bars))
    loss = loss + (cpu.observe(final).conj() * seed[0]).real.sum()
    expected = torch.autograd.grad(loss, (epsilon, waves, *state))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_eps = epsilon.detach().cuda()
        device_waves = waves.detach().cuda()
        system = _system(p, device_eps, device_waves)
        for target, value in zip(system.state(), state):
            target.copy_(value.detach())
        packed = torch.zeros(epsilon.numel() + waves.numel() * (2 if complex_waveforms else 1), device='cuda')
        fused = SourceFusedAdjointCUDA(system, packed, seed.cuda())
        for target, value in zip((fused.e_bar, fused.h_bar, *fused.psi_bars[0]), bars):
            target.copy_(value)
        fused.step(2, observation_index=0)
    stream.synchronize()
    n, count = epsilon.numel(), waves.numel()
    actual_wave = packed[n:n + count].view_as(waves).cpu()
    if complex_waveforms:
        actual_wave = torch.complex(actual_wave, packed[n + count:].view_as(waves).cpu())
    torch.testing.assert_close(packed[:n].view_as(epsilon).cpu(), expected[0], rtol=4e-5, atol=3e-6)
    torch.testing.assert_close(actual_wave, expected[1], rtol=4e-5, atol=3e-6)
    for actual, want in zip((fused.e_bar, fused.h_bar, *fused.psi_bars[fused.phase]), expected[2:]):
        torch.testing.assert_close(actual.cpu(), want, rtol=4e-5, atol=3e-6)
    assert expected[1][2].abs().min() > 0
    assert torch.count_nonzero(actual_wave[[i for i in range(p.region.steps) if i != 2]]) == 0
