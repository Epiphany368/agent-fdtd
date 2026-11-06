"""Small independent functional-autograd oracle for recorded CPML helpers."""
import os
import weakref

import pytest
import torch

from torchfdtd.models import Project, Region
from torchfdtd.differentiable import _System
from torchfdtd.reversible_cpml_kernels import InteriorReconstruction


def make(epsilon, steps=48):
    faces = {f'{axis}_{side}': {'kind': 'periodic'}
             for axis in 'xy' for side in ('min', 'max')}
    faces.update(z_min={'kind': 'pml', 'layers': 3, 'kappa': 2, 'alpha': .03},
                 z_max={'kind': 'pml', 'layers': 4, 'kappa': 2, 'alpha': .03})
    project = Project(region=Region(dimension='3d', size=(.6, .7, 2), mesh=.1,
        steps=steps, precision='float32', backend='cpu', courant_factor=.9,
        boundaries=faces), sources=[], monitors=[])
    system = _System(project, epsilon.detach(), prepare_kernels=False)
    t = torch.arange(steps, dtype=torch.float32)
    wave = (torch.exp(-((t-6)/2.5)**2)*torch.cos(.8*(t-6))).to(epsilon.device)
    system.sources = {'E': [((2, 3, 5), 0, wave, None)],
                      'H': [((3, 2, 13), 1, .2*wave, None)]}
    # Duplicate electric observations exercise scatter addition independently.
    system.monitors = [('Ex', (2, 3, 7), 0), ('Ex', (2, 3, 7), 0),
                       ('Hy', (2, 3, 12), 1)]
    system.prepare_observations()
    return system


def relative(actual, expected):
    return float((actual.double()-expected.double()).norm()/expected.double().norm())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=[pytest.mark.long, pytest.mark.skipif(
    os.environ.get('TORCHFDTD_RUN_CPML_KERNEL_CUDA_TEST') != '1',
    reason='Explicit small CUDA helper gate only')])])
def test_reconstruction_full_cpml_transpose_and_material_vjp(device):
    torch.set_num_threads(1)
    if device == 'cuda':
        from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
        prepare_cuda_kernels()
    base = torch.ones((6, 7, 20))
    base[:, :, 9:11] = 2.25
    parameter = base.clone().requires_grad_()
    oracle = make(base)
    a, b = 4, 14
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:, :, a+1:b] = True
    mask[2, 3, 5] = mask[3, 2, 13] = False
    effective = torch.where(mask, parameter, base)
    state = oracle.state()
    trajectory = [tuple(t.detach().clone() for t in state[:2])]
    histories = []
    # Direct functional native updates differentiated by Torch, no hand-written
    # transpose or recorded reconstruction in this oracle.
    for n in range(oracle.region.steps):
        state = oracle.reference_step(state, n, effective)
        trajectory.append(tuple(t.detach().clone() for t in state[:2]))
        histories.append(oracle.observe(state))
    generator = torch.Generator().manual_seed(91)
    seed = torch.randn((3, oracle.region.steps), generator=generator).T
    assert not seed.is_contiguous()
    reference_gradient, = torch.autograd.grad(torch.stack(histories), parameter, seed)

    system = make(base.to(device))
    if device == 'cuda':
        from torchfdtd.cuda_kernels import FusedYeeCUDA
        system.kernel = FusedYeeCUDA(system.grid, direct_views=True)
    archive = []
    for n in range(system.region.steps):
        if system.kernel is not None:
            system.kernel.update_E()
            system.inject(system.grid.E, 'E', n)
            archive.append(torch.stack((system.grid.E[:, :, b+1, :2],
                                        system.grid.H[:, :, a-1, :2])).clone())
            system.kernel.update_H()
            system.inject(system.grid.H, 'H', n)
        else:
            e, h, *psi = system.state()
            curl, psi = system.curl(h, psi, False)
            next_e = system.inject(e+system.grid.courant_number/system.eps4*curl,
                                   'E', n, functional=True)
            archive.append(torch.stack((next_e[:, :, b+1, :2], h[:, :, a-1, :2])).clone())
            curl, psi = system.curl(next_e, psi, True)
            next_h = system.inject(h-system.grid.courant_number*curl, 'H', n, functional=True)
            for target, value in zip(system.state(), (next_e, next_h, *psi)):
                target.copy_(value)
    assert max(float(t['psi'].abs().max()) for t in system.segments) > 1e-8
    terminal = tuple(t[:, :, a:b+1].clone() for t in system.state()[:2])
    for target, value in zip(system.state()[:2], terminal):
        target.fill_(float('nan'))
        target[:, :, a:b+1].copy_(value)
    gradient = torch.zeros_like(system.epsilon)
    helper = InteriorReconstruction(system, a, b, gradient, seed.to(device).T.contiguous().T)
    peak = max(float(t.abs().max()) for pair in trajectory for t in pair)
    psi_peak = 0.
    for n in reversed(range(system.region.steps)):
        helper.step(n, archive[n])
        for actual, expected in zip(system.state()[:2], trajectory[n]):
            error = float((actual[:, :, a:b+1].cpu()-expected[:, :, a:b+1]).abs().max())
            assert error < 1e-4 and error/peak < 1e-4
        psi_peak = max(psi_peak, max(float(t.abs().max()) for t in helper.psi_bars))
    assert psi_peak > 1e-8
    masked = gradient.cpu()*mask
    assert relative(masked, reference_gradient) < 1e-4
    assert torch.count_nonzero(gradient[:, :, :a]) == 0
    assert torch.count_nonzero(gradient[:, :, b+1:]) == 0
    assert torch.isfinite(gradient).all()
    assert torch.isnan(system.grid.E[:, :, 0]).all()
    assert torch.isnan(system.grid.H[:, :, -1]).all()
    if device == 'cuda':
        torch.cuda.synchronize()
    ref = weakref.ref(helper)
    del helper
    assert ref() is None


def test_interval_trace_and_order_guards_before_mutation():
    system = make(torch.ones((6, 7, 20)), steps=10)
    gradient = torch.zeros_like(system.epsilon)
    seed = torch.zeros((10, 3))
    with pytest.raises(ValueError, match='collar'):
        InteriorReconstruction(system, 3, 15, gradient, seed)
    helper = InteriorReconstruction(system, 4, 14, gradient, seed)
    trace = torch.zeros((2, 6, 7, 2))
    with pytest.raises(ValueError, match='descending'):
        helper.step(0, trace)
    with pytest.raises(ValueError, match='Trace'):
        helper.step(9, trace.double())
    assert torch.count_nonzero(system.grid.E) == 0
    helper.step(9, trace)
    with pytest.raises(ValueError, match='descending'):
        helper.step(9, trace)

    for n in reversed(range(9)):
        helper.step(n, trace)
    with pytest.raises(ValueError, match="descending"):
        helper.step(-1, trace)
