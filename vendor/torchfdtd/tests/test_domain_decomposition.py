"""Real multi-process single-problem halos, field propagation and material VJPs."""
from datetime import timedelta
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchfdtd import Boundaries, BoundaryFace, Region, Project
from torchfdtd.differentiable import _System
from torchfdtd.domain_decomposition import (DistributedYeeDomain,
    distributed_capabilities, plan_domain_decomposition, _replay_cost)


def region(complex_fields=False, precision='float64'):
    kind = 'bloch' if complex_fields else 'periodic'
    boundaries = Boundaries(**{axis+side: BoundaryFace(kind=kind)
        for axis in 'xyz' for side in ('_min', '_max')})
    return Region(dimension='3d', size=(.5, .6, .5), mesh=.1, steps=20,
        pml_cells=3, precision=precision, material_sampling='yee', boundaries=boundaries,
        bloch_phase=(.43, -.31, .29) if complex_fields else (0., 0., 0.))


def initial(r, begin=0, end=None, diagonal=False, device='cpu'):
    end = r.shape[0] if end is None else end
    dtype = getattr(torch, r.precision)
    x = torch.arange(begin, end, dtype=dtype, device=device)[:, None, None]
    y = torch.arange(r.shape[1], dtype=dtype, device=device)[None, :, None]
    z = torch.arange(r.shape[2], dtype=dtype, device=device)[None, None, :]
    index = ((x*r.shape[1]+y)*r.shape[2]+z)[..., None]*3+torch.arange(3, dtype=dtype, device=device)
    e, h = .03*torch.sin(.17*index), .02*torch.cos(.11*index)
    if r.complex_fields:
        e, h = e+1j*.021*torch.cos(.13*index), h-1j*.013*torch.sin(.19*index)
    eps = 1.3+.1*torch.sin(.3*x+.2*y+.4*z)
    if diagonal:
        eps = eps[..., None]+.05*torch.arange(3, dtype=dtype, device=device)
    return e, h, eps


def objective(e, h):
    return .7*e.abs().square().sum()+.4*h.abs().square().sum()+.03*e.real.sum()


def worker(rank, world, port, output, complex_fields, precision, diagonal, checkpoints, backend):
    torch.set_num_threads(1)
    os.environ['USE_LIBUV'] = '0'
    device = 'cpu' if backend == 'gloo' else f'cuda:{rank}'
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        r = region(complex_fields, precision)
        domain = DistributedYeeDomain(r, device=device, checkpoints=checkpoints,
                                      rank_budget_bytes=16*1024**2)
        own = domain.ownership
        e, h, eps = [t.detach().requires_grad_(True) for t in initial(r, own.begin, own.end, diagonal, device)]
        halo_errors = []
        for side in ('left', 'right'):
            sample = domain.exchange_halo(e.detach(), side=side)
            seed = h.detach()[:1]+.13
            lhs = (sample.conj()*seed).sum().real
            transpose = domain.transpose_halo(seed, side=side)
            owned_row = e.detach()[-1:] if side == 'left' else e.detach()[:1]
            rhs = (owned_row.conj()*transpose).sum().real
            difference = lhs-rhs
            dist.all_reduce(difference)
            halo_errors.append(float(difference))
        result = domain.propagate(e, h, eps, steps=4)
        loss = domain.global_sum(objective(*result))
        loss.backward()
        gradients = tuple(t.grad.detach().cpu() for t in (e, h, eps))
        report = dict(domain.last_report)
        # Independently perturb only rank zero's material at a slab corner.
        direction = torch.zeros_like(eps)
        if rank == 0:
            direction.reshape(-1)[0] = 1.
        step = 2e-3 if precision == 'float32' else 1e-5
        analytic = (eps.grad*direction).sum()
        dist.all_reduce(analytic)
        with torch.no_grad():
            high = domain.global_sum(objective(*domain.propagate(e, h, eps+step*direction, steps=4)))
            low = domain.global_sum(objective(*domain.propagate(e, h, eps-step*direction, steps=4)))
        torch.save(dict(e=result[0].detach().cpu(), h=result[1].detach().cpu(), gradients=gradients,
            loss=loss.detach().cpu(), halo_errors=halo_errors, finite_difference=float((high-low)/(2*step)),
            directional=float(analytic), report=report, own=(own.begin, own.end)), f'{output}/{rank}.pt')
    finally:
        dist.destroy_process_group()


def launch(tmp_path, world, complex_fields, precision, diagonal, checkpoints, backend='gloo'):
    if backend == 'gloo':
        try:
            dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')
        except RuntimeError as error:
            pytest.skip(f'Gloo runtime transport unavailable: {error}')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(world, port, str(tmp_path), complex_fields, precision,
        diagonal, checkpoints, backend), nprocs=world, join=True)
    return [torch.load(tmp_path/f'{rank}.pt', weights_only=True) for rank in range(world)]


def launch_emulated(monkeypatch, tmp_path, world, complex_fields, precision, diagonal, checkpoints):
    """In-process protocol oracle, explicitly not distributed backend evidence."""
    local = threading.local()
    barrier = threading.Barrier(world, timeout=30)
    values = [None]*world

    def gather(value):
        values[local.rank] = value
        barrier.wait()
        result = list(values)
        barrier.wait()
        return result

    def initialize(*args, rank, **kwargs):
        local.rank = rank

    def all_gather(output, value, **kwargs):
        output[:] = gather(value)

    def all_reduce(value, **kwargs):
        result = sum(gather(value.detach().clone()))
        value.copy_(result)

    def transfer(domain, send, destination, source, tag):
        messages = gather((destination, send.detach().clone(), tag))
        assert messages[source][0] == local.rank
        assert messages[source][2] == tag
        domain.last_report['communicated_bytes'] += 2*send.numel()*send.element_size()
        return messages[source][1].clone()

    for name, function in dict(init_process_group=initialize, destroy_process_group=lambda: None,
            is_initialized=lambda: True, get_rank=lambda group=None: local.rank,
            get_world_size=lambda group=None: world, get_backend=lambda group=None: 'gloo',
            all_gather_object=all_gather, all_reduce=all_reduce).items():
        monkeypatch.setattr(dist, name, function)
    monkeypatch.setattr(DistributedYeeDomain, '_transfer', transfer)
    with ThreadPoolExecutor(max_workers=world) as executor:
        futures = [executor.submit(worker, rank, world, 0, str(tmp_path), complex_fields,
            precision, diagonal, checkpoints, 'gloo') for rank in range(world)]
        for future in futures:
            future.result()
    return [torch.load(tmp_path/f'{rank}.pt', weights_only=True) for rank in range(world)]


@pytest.mark.optional
@pytest.mark.parametrize('world,complex_fields,precision,diagonal,checkpoints',
    [(2, False, 'float64', False, 0), (3, True, 'float64', True, 2),
         (2, True, 'float32', False, 1)])
@pytest.mark.parametrize('transport', ['gloo', 'emulated'])
def test_actual_gloo_domain_matches_resident_and_independent_material_difference(
        monkeypatch, tmp_path, world, complex_fields, precision, diagonal, checkpoints, transport):
    if not distributed_capabilities()['gloo']:
        pytest.skip('Gloo unavailable')
    rows = (launch(tmp_path, world, complex_fields, precision, diagonal, checkpoints)
        if transport == 'gloo' else launch_emulated(monkeypatch, tmp_path, world,
            complex_fields, precision, diagonal, checkpoints))
    r = region(complex_fields, precision)
    e, h, eps = [t.requires_grad_(True) for t in initial(r, diagonal=diagonal)]
    system = _System(Project(region=r, sources=[], monitors=[]), eps)
    state = (e, h)
    for step in range(4):
        state = system.reference_step(state, step, eps)
    loss = objective(*state)
    expected = torch.autograd.grad(loss, (e, h, eps))
    tolerance = 2e-5 if precision == 'float32' else 2e-12
    torch.testing.assert_close(torch.cat([row['e'] for row in rows]), state[0], rtol=tolerance, atol=tolerance*.01)
    torch.testing.assert_close(torch.cat([row['h'] for row in rows]), state[1], rtol=tolerance, atol=tolerance*.01)
    for index, gradient in enumerate(expected):
        torch.testing.assert_close(torch.cat([row['gradients'][index] for row in rows]), gradient,
                                   rtol=tolerance*4, atol=tolerance*.02)
    for row in rows:
        torch.testing.assert_close(row['loss'], loss, rtol=tolerance, atol=tolerance*.01)
        assert max(abs(v) for v in row['halo_errors']) < tolerance
        assert row['report']['peak_checkpoints'] <= checkpoints
        assert row['report']['replayed_steps'] == _replay_cost(4, checkpoints)
        assert row['report']['replicated_global_state'] is False
        assert row['e'].shape[0] < r.shape[0]
        assert row['report']['communicated_bytes'] > 0
    torch.testing.assert_close(torch.tensor(rows[0]['directional']), torch.tensor(rows[0]['finite_difference']),
        rtol=3e-2 if precision == 'float32' else 3e-5, atol=2e-5 if precision == 'float32' else 1e-8)


def test_admission_capabilities_ownership_and_fixed_memory_contract():
    r = region()
    plan = plan_domain_decomposition(r, 3, checkpoints=2)
    assert [(part.begin, part.end) for part in plan['ownership']] == [(0, 1), (1, 3), (3, 5)]
    assert sum(part.shape[0] for part in plan['ownership']) == r.shape[0]
    assert _replay_cost(1000, 2) < _replay_cost(1000, 0)//10
    with pytest.raises(ValueError, match='budget'):
        plan_domain_decomposition(r, 3, rank_budget_bytes=1)
    with pytest.raises(ValueError, match='one x cell'):
        plan_domain_decomposition(r, 6)
    with pytest.raises(ValueError, match='unavailable'):
        plan_domain_decomposition(r, 2, backend='invalid')
    r.boundaries.z_min = r.boundaries.z_max = BoundaryFace(kind='pml')
    with pytest.raises(ValueError, match='CPML'):
        plan_domain_decomposition(r, 2)


@pytest.mark.optional
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason='Requires at least two visible CUDA GPUs and NCCL; not emulated by CPU ranks.')
def test_two_cuda_devices_single_problem(tmp_path):
    rows = launch(tmp_path, 2, True, 'float32', True, 2, backend='nccl')
    r = region(True, 'float32')
    e, h, eps = initial(r, diagonal=True)
    system = _System(Project(region=r, sources=[], monitors=[]), eps)
    state = (e, h)
    for step in range(4):
        state = system.reference_step(state, step, eps)
    torch.testing.assert_close(torch.cat([row['e'] for row in rows]), state[0], rtol=5e-5, atol=1e-7)
    assert all(row['report']['backend'] == 'nccl' for row in rows)


def test_single_rank_replay_admission_and_immutable_configuration(monkeypatch):
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_rank', lambda group=None: 0)
    monkeypatch.setattr(dist, 'get_world_size', lambda group=None: 1)
    monkeypatch.setattr(dist, 'get_backend', lambda group=None: 'gloo')
    monkeypatch.setattr(dist, 'all_gather_object', lambda output, value, **kwargs: output.__setitem__(0, value))
    with pytest.raises(ValueError, match='Distributed admission failed'):
        DistributedYeeDomain(region(), device='bad-device')
    domain = DistributedYeeDomain(region(), checkpoints=0, max_replayed_steps=1)
    with pytest.raises(AttributeError, match='immutable'):
        domain.courant = .1
    e, h, eps = initial(region())
    eps.requires_grad_()
    with torch.inference_mode():
        inference_e = e.clone()
    with pytest.raises(ValueError, match='inference tensors'):
        domain.propagate(inference_e, h, eps, steps=1)
    with pytest.raises(ValueError, match='dense strided'):
        domain.propagate(e.to_sparse(), h, eps, steps=1)
    with pytest.raises(ValueError, match='Replay schedule'):
        domain.propagate(e, h, eps, steps=4)
    result = domain.propagate(e, h, eps, steps=2)
    with torch.no_grad():
        eps.add_(.01)
    with pytest.raises(ValueError, match='modified by an inplace operation'):
        objective(*result).backward()
