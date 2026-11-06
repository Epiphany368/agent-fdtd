"""Complex/diagonal recorded CPML helpers and explicit online seed rows."""
import pytest
import torch

from torchfdtd.models import Project, Region
from torchfdtd.differentiable import _System
from torchfdtd.reversible_cpml_kernels import InteriorReconstruction


def fixture(diagonal=True, complex_fields=True, phases=(.43, -.37, 0.), steps=32):
    faces = {f'{axis}_{side}': {'kind': 'bloch' if complex_fields else 'periodic'}
             for axis in 'xy' for side in ('min', 'max')}
    faces.update(z_min={'kind': 'pml', 'layers': 3, 'kappa': 2, 'alpha': .03},
                 z_max={'kind': 'pml', 'layers': 4, 'kappa': 2, 'alpha': .03})
    project = Project(region=Region(dimension='3d', size=(.6, .7, 2), mesh=.1,
        steps=steps, precision='float32', backend='cpu', courant_factor=.9,
        boundaries=faces, bloch_phase=phases if complex_fields else (0., 0., 0.)),
        sources=[], monitors=[])
    shape = project.region.shape
    base = (torch.tensor([1.3, 1.6, 2.25]).expand(*shape, 3).clone() if diagonal
            else torch.full(shape, 1.4))
    base[:, :, 9:11] += .3
    system = _System(project, base, prepare_kernels=False)
    t = torch.arange(steps, dtype=torch.float32)
    wave = torch.exp(-((t-6)/2.5)**2)*torch.cos(.8*(t-6))
    system.sources = {'E': [((0, 3, 5), 0, wave, None)], 'H': []}
    system.monitors = [('Ex', (0, 3, 7), 0), ('Ex', (0, 3, 7), 0),
                       ('Hy', (2, 3, 15), 1)]
    system.prepare_observations()
    return system, base


@pytest.mark.parametrize('diagonal,complex_fields,phases,block', [
    (True, False, (0., 0., 0.), False),
    (False, True, (.43, -.37, 0.), False),
    (True, True, (.43, -.37, 0.), False),
    (True, True, (0., 0., 0.), False),
    (True, True, (.43, -.37, 0.), True),
])
def test_material_vjp_and_block_rows_match_functional_autograd(diagonal, complex_fields, phases, block):
    torch.set_num_threads(1)
    system, base = fixture(diagonal, complex_fields, phases)
    a, b = 4, 14
    parameter = base.clone().requires_grad_()
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:, :, a+1:b] = True
    mask[0, 3, 5] = False
    epsilon = torch.where(mask, parameter, base)
    state = system.state()
    archive, histories = [], []
    trajectory = [tuple(t[:, :, a:b+1].clone() for t in state[:2])]
    for n in range(system.region.steps):
        e, h, *psis = state
        curl, psis = system.curl(h, psis, False)
        eps4 = epsilon if diagonal else epsilon[..., None]
        next_e = system.inject(e+system.grid.courant_number/eps4*curl, 'E', n, functional=True)
        archive.append(torch.stack((next_e[:, :, b+1, :2], h[:, :, a-1, :2])).detach())
        # Independent differentiation of the functional step supplies all
        # exterior material and auxiliary sensitivities before our design mask.
        state = system.reference_step(state, n, epsilon)
        trajectory.append(tuple(t[:, :, a:b+1].detach().clone() for t in state[:2]))
        histories.append(system.observe(state))
    generator = torch.Generator().manual_seed(61)
    seed = torch.randn((3, system.region.steps), generator=generator, dtype=system.field_dtype).T
    assert not seed.is_contiguous()
    expected, = torch.autograd.grad(torch.stack(histories), parameter, seed)
    for target, terminal in zip(system.state()[:2], trajectory[-1]):
        target.fill_(complex(float('nan'), float('nan')) if complex_fields else float('nan'))
        target[:, :, a:b+1].copy_(terminal)
    gradient = torch.zeros_like(base)
    buffer = torch.zeros((7, 3), dtype=system.field_dtype) if block else seed
    helper = InteriorReconstruction(system, a, b, gradient, buffer)
    if block:
        assert helper.signal_bar.data_ptr() == buffer.data_ptr()
    psi_peak = 0.
    for n in reversed(range(system.region.steps)):
        if block:
            row = n % 7
            buffer[row].copy_(seed[n])
            helper.step(n, archive[n], observation_index=row)
        else:
            helper.step(n, archive[n])
        for actual, ref in zip(system.state()[:2], trajectory[n]):
            assert float((actual[:, :, a:b+1]-ref).abs().max()) < 1e-4
        psi_peak = max(psi_peak, max(float(t.abs().max()) for t in helper.psi_bars))
    assert psi_peak > 1e-8
    actual = gradient*mask
    assert float((actual.double()-expected.double()).norm()/expected.double().norm()) < 1e-4
    assert torch.count_nonzero(gradient[:, :, :a]) == 0
    assert torch.count_nonzero(gradient[:, :, b+1:]) == 0
    assert torch.isfinite(gradient).all()
    if diagonal:
        assert all(float(actual[..., c].norm()) > 0 for c in range(3))


def test_block_and_boundary_guards_precede_mutation():
    system, base = fixture(steps=10)
    gradient = torch.zeros_like(base)
    helper = InteriorReconstruction(system, 4, 14, gradient, torch.zeros((1, 3), dtype=torch.complex64))
    trace = torch.zeros((2, 6, 7, 2), dtype=torch.complex64)
    with pytest.raises(ValueError, match='explicit observation_index'):
        helper.step(9, trace)
    for row in (-1, 1, True):
        with pytest.raises(ValueError, match='valid seed buffer row'):
            helper.step(9, trace, observation_index=row)
    assert torch.count_nonzero(system.grid.E) == 0
    with pytest.raises(ValueError, match='field dtype'):
        InteriorReconstruction(system, 4, 14, gradient, torch.zeros((10, 3)))
    with pytest.raises(ValueError, match='Seed buffer'):
        InteriorReconstruction(system, 4, 14, gradient, torch.zeros((0, 3), dtype=torch.complex64))
    system.grid.wrap[2] = 1
    with pytest.raises(ValueError, match='z CPML'):
        InteriorReconstruction(system, 4, 14, gradient, torch.zeros((10, 3), dtype=torch.complex64))


def test_singleton_bloch_axis_matches_native_skipped_derivative():
    from torchfdtd.reversible_cpml_kernels import _local_curl
    # Native curl skips a length-one derivative axis. Only that seam has a
    # nontrivial phase, and the other two derivative directions are constant.
    field = torch.tensor([1+2j, 3-1j, -2+.5j], dtype=torch.complex64).expand(1, 3, 12, 3).clone()
    for forward in (False, True):
        actual = _local_curl(field, 3, 8, forward, {0: complex(.8, .6), 1: 1.})
        assert torch.count_nonzero(actual) == 0
