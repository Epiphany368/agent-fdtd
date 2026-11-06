"""Exact comparisons for direct recorded E/H observations."""
import pytest
import torch

from tests.test_reversible_cpml_complex_kernels import fixture
from torchfdtd.differentiable import _System
from torchfdtd.recorded_observations import recorder


@pytest.mark.cuda
@pytest.mark.parametrize('complex_fields', [False, True])
def test_recorded_samples_copy_bits_and_duplicate_locations(complex_fields):
    if not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
    prepare_cuda_kernels()
    original, base = fixture(True, complex_fields, steps=10)
    system = _System(original.project, base.cuda(), prepare_kernels=False)
    # All components, both fields, repeated positions, and two different rows.
    system.monitors = [(family+axis, (x, y, z), c)
        for family in ('E', 'H') for c, axis in enumerate('xyz')
        for x, y, z in ((0, 0, 0), (3, 2, 6), (3, 2, 6), (5, 6, 19))]
    system.prepare_observations()
    samples = torch.full((3, len(system.monitors)), -13, device='cuda', dtype=system.field_dtype)
    write = recorder(system, samples)
    for row in (2, 0):
        generator = torch.Generator().manual_seed(130+row)
        for field in (system.grid.E, system.grid.H):
            values = torch.randn(field.shape, generator=generator, dtype=field.dtype)
            # Include zeros and subnormal payloads among sampled locations.
            values[0, 0, 0] = torch.tensor([0., -0., 1e-40], dtype=field.dtype)
            field.copy_(values)
        expected = system.observe(system.state())
        write(row)
        assert torch.equal(samples[row].view(torch.int32), expected.view(torch.int32))
    assert torch.all(samples[1] == -13)
    with pytest.raises(ValueError, match='row'):
        write(3)


def test_cpu_recorder_preserves_observation_order():
    system, _ = fixture(True, True, steps=10)
    system.grid.E.fill_(2+3j)
    system.grid.H.fill_(-4+1j)
    samples = torch.empty((3, 3), dtype=system.field_dtype)
    write = recorder(system, samples)
    write(1)
    assert torch.equal(samples[1], system.observe(system.state()))
