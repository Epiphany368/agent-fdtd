"""G5-03: the planner's byte admission bounds the measured peaks, and concurrent reservations are atomic.

A ladder of three streamed fixtures (65,536 to 1,048,576 cells) with host and
disk banks runs on CUDA; the reservation of every rung must bound the measured
peak Torch allocated bytes (GPU tier) and the measured peak bank bytes (host
banks or scratch files), with the margins recorded in
docs/validation/g5/G5-03.json. The in-process reservation registry
(torchfdtd/reservation_registry.py) then refuses the second of two runs whose
combined reservation exceeds the budget of a shared tier, under one lock, so
two threads that each see enough free memory cannot both be admitted.
"""
from dataclasses import replace
import threading

import pytest
import torch

from torchfdtd import Monitor, Project, Region, Source, StreamedAdjointOptions, StreamedSimulation, smooth_sphere_epsilon
from torchfdtd.reservation_registry import REGISTRY, ReservationRegistry, streamed_requests
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd.streamed import _reservation
from g5_support import CUDA, Record, SOURCE, limits, scene, sphere

CASE = 'G5-03'
RECORD = Record('G5-03', CASE, 'tests/test_admission_peak_g5.py')
cuda = pytest.mark.skipif(not CUDA, reason='CUDA unavailable')
LADDER = ((64, 32, 32), (128, 48, 48), (256, 64, 64))
MESH = .05
REFUSAL = 'Concurrent streamed reservation refused'


def rung(shape, steps=12):
    nx, ny, nz = shape
    region = Region(dimension='3d', size=(nx*MESH, ny*MESH, nz*MESH), mesh=MESH, pml_cells=4, steps=steps,
                    precision='float32', backend='cpu', material_sampling='yee', memory_mode='streamed')
    source = dict(SOURCE, center=(-.3, 0, 0), pulse_length=.5e-15, pulse_offset=.5e-15)
    project = Project(region=region, sources=[Source(**source)],
                      monitors=[Monitor(component='Ez', center=(.3, 0, 0)), Monitor(component='Hy', center=(.2, .1, 0))])
    assert tuple(project.region.shape) == tuple(shape), project.region.shape
    return project


def ladder_options(storage, scratch):
    return StreamedAdjointOptions(device='cuda', slab_width=8, temporal_depth=4, checkpoints=1, state_storage=storage,
                                  state_directory=scratch if storage == 'disk' else None,
                                  disk_budget_bytes=4*2**30 if storage == 'disk' else None,
                                  gpu_budget_bytes=2*2**30, host_budget_bytes=16*2**30)


def measured(report):
    forward, backward = report['forward_memory'], report['backward_memory']
    def peak(name, key='bytes'):
        return max(forward[name][key], backward[name][key])
    return dict(peak_torch_allocated_delta=peak('peak_torch_allocated_bytes', 'delta_bytes'),
                exact=forward['peak_torch_allocated_bytes']['exact'] and backward['peak_torch_allocated_bytes']['exact'],
                peak_torch_reserved=peak('peak_torch_reserved_bytes'),
                peak_host_bank_bytes=peak('host_bank_bytes'), peak_file_bytes=peak('scratch_disk_peak_file_bytes'),
                rss_delta=peak('peak_process_rss_bytes', 'delta_bytes'))


@cuda
def test_reservation_bounds_measured_peaks_on_every_rung(tmp_path):
    minimum = limits(CASE, 'minimum_ratio')
    # One small warm-up run loads the CUDA kernels so that the rungs' working-set deltas are their own.
    warm = scene('float32', 'point')
    StreamedSimulation(warm, ladder_options('host', None))(sphere(warm, inside=4., outside=1.)).signals.sum()
    for shape in LADDER:
        project = rung(shape)
        epsilon = smooth_sphere_epsilon(project.region, torch.tensor(.3), inside=4., outside=1., width=.08).detach().contiguous().requires_grad_()
        for storage in ('host', 'disk'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            result = StreamedSimulation(project, ladder_options(storage, tmp_path/'banks'))(epsilon)
            gradient, = torch.autograd.grad(result.signals.square().sum(), epsilon)
            assert bool(torch.isfinite(gradient).all())
            report = result.report
            values = measured(report)
            assert values['exact'], 'the allocator peak of a rung must be exact; the test resets the peak counters first'
            gpu_ratio = report['gpu_reservation_bytes']/values['peak_torch_allocated_delta']
            if storage == 'host':
                bank_ratio = report['host_reservation_bytes']/values['peak_host_bank_bytes']
                assert values['peak_file_bytes'] == 0
            else:
                bank_ratio = report['disk_reservation_bytes']/values['peak_file_bytes']
                assert values['peak_host_bank_bytes'] == 0
            assert gpu_ratio >= minimum and bank_ratio >= minimum
            assert values['peak_torch_allocated_delta'] <= values['peak_torch_reserved']
            rss_ratio = report['host_reservation_bytes']/values['rss_delta'] if values['rss_delta'] > 0 else None
            RECORD.add(f'{"x".join(map(str, shape))}/{storage}', shape=list(shape), cells=shape[0]*shape[1]*shape[2],
                       storage=storage, state_bytes=report['state_bytes'], state_bank_capacity=report['state_bank_capacity'],
                       gpu_reservation_bytes=report['gpu_reservation_bytes'], host_reservation_bytes=report['host_reservation_bytes'],
                       disk_reservation_bytes=report['disk_reservation_bytes'], **values,
                       gpu_ratio=gpu_ratio, gpu_margin_bytes=report['gpu_reservation_bytes']-values['peak_torch_allocated_delta'],
                       bank_ratio=bank_ratio,
                       bank_margin_bytes=(report['host_reservation_bytes'] if storage == 'host' else report['disk_reservation_bytes'])
                       -(values['peak_host_bank_bytes'] if storage == 'host' else values['peak_file_bytes']),
                       rss_ratio_informational=rss_ratio, minimum_ratio=minimum,
                       forward_seconds=report['forward_seconds'], backward_seconds=report['backward_seconds'])


# ----------------------------------------------------------------------------------------------- registry
def test_registry_admits_only_one_of_two_simultaneous_requests():
    registry = ReservationRegistry()
    barrier = threading.Barrier(2)
    outcomes = {}

    def attempt(name):
        barrier.wait()
        try:
            outcomes[name] = registry.acquire([('host', 'process', 600, 1000)])
        except ValueError as exc:
            outcomes[name] = exc

    threads = [threading.Thread(target=attempt, args=(name,)) for name in ('a', 'b')]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    leases = [v for v in outcomes.values() if not isinstance(v, Exception)]
    refused = [v for v in outcomes.values() if isinstance(v, Exception)]
    assert len(leases) == 1 and len(refused) == 1
    assert str(refused[0]).startswith(REFUSAL) and 'per process' in str(refused[0])
    assert registry.held_bytes('host', 'process') == 600
    leases[0].release()
    assert registry.held_bytes('host', 'process') == 0
    with registry.acquire([('host', 'process', 600, 1000)]):
        pass
    assert registry.held_bytes('host', 'process') == 0


def test_registry_requests_are_all_or_nothing_across_tiers():
    registry = ReservationRegistry()
    with registry.acquire([('host', 'process', 500, 1000), ('disk', 'D:\\', 100, 1000)]):
        with pytest.raises(ValueError, match=REFUSAL):
            registry.acquire([('host', 'process', 100, 1000), ('disk', 'D:\\', 950, 1000)])
        # The refused request must not have registered its first tier.
        assert registry.held_bytes('host', 'process') == 500
    assert registry.held_bytes('disk', 'D:\\') == 0


def test_streamed_requests_name_every_tier_of_the_options(tmp_path):
    project = scene('float32', 'point')
    epsilon = sphere(project, inside=4., outside=1.)
    options = StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4, state_storage='disk',
                                     state_directory=tmp_path/'banks', disk_budget_bytes=64*2**20)
    reservation = _reservation(project, epsilon, options)
    requests = streamed_requests(options, reservation)
    assert [r[0] for r in requests] == ['host', 'disk']
    assert requests[0][2:] == (reservation['host_reservation_bytes'], options.host_budget_bytes)
    assert requests[1][2:] == (reservation['disk_reservation_bytes'], options.disk_budget_bytes)
    if CUDA:
        with_gpu = streamed_requests(StreamedAdjointOptions(device='cuda', slab_width=6, temporal_depth=4), reservation)
        assert [r[0] for r in with_gpu] == ['host', 'gpu']


@pytest.mark.parametrize('storage', ['host', 'disk'])
def test_second_concurrent_run_is_refused_and_the_lease_is_released(storage, tmp_path, monkeypatch):
    project = scene('float32', 'point')
    epsilon = sphere(project, inside=4., outside=1.).requires_grad_()
    base = StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4, checkpoints=1, state_storage=storage,
                                  state_directory=tmp_path/'banks' if storage == 'disk' else None,
                                  disk_budget_bytes=64*2**20 if storage == 'disk' else None)
    single = _reservation(project, epsilon, base)
    tier = 'host' if storage == 'host' else 'disk'
    budget = int(1.5*single[f'{tier}_reservation_bytes'])
    options = replace(base, **{f'{tier}_budget_bytes': budget})
    entered, release = threading.Event(), threading.Event()
    original = SlabBlockOperator.forward

    def blocking_forward(self, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(60), 'the first run was never released'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SlabBlockOperator, 'forward', blocking_forward)
    outcome = {}

    def first():
        try:
            result = StreamedSimulation(project, options)(epsilon)
            outcome['result'] = result
        except Exception as exc:  # noqa: BLE001 - reported through the assertion below
            outcome['error'] = exc
    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(60)
    key = 'process' if tier == 'host' else streamed_requests(options, single)[-1][1]
    assert REGISTRY.held_bytes(tier, key) == single[f'{tier}_reservation_bytes']
    with pytest.raises(ValueError, match=REFUSAL):
        StreamedSimulation(project, options)(epsilon)
    release.set()
    thread.join(120)
    assert 'error' not in outcome, outcome.get('error')
    assert REGISTRY.held_bytes(tier, key) == 0
    # With the first run finished, the same budget admits the second one.
    second = StreamedSimulation(project, options)(epsilon)
    torch.testing.assert_close(second.signals, outcome['result'].signals, rtol=0, atol=0)
    assert REGISTRY.held_bytes(tier, key) == 0


def test_failed_phase_releases_its_lease(tmp_path, monkeypatch):
    project = scene('float32', 'point')
    epsilon = sphere(project, inside=4., outside=1.)
    options = StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4)

    def failing_forward(self, *args, **kwargs):
        raise RuntimeError('injected tile failure')
    monkeypatch.setattr(SlabBlockOperator, 'forward', failing_forward)
    with pytest.raises(RuntimeError, match='injected tile failure'):
        StreamedSimulation(project, options)(epsilon)
    assert REGISTRY.held_bytes('host', 'process') == 0
