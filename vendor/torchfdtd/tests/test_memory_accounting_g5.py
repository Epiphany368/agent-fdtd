"""G5-02: one streamed run reports every memory quantity as its own named, instrumented field.

torchfdtd/memory_accounting.py measures each phase of a StreamedSimulation run:
peak Torch allocated and reserved bytes, the device memory the process holds,
process working set, private and committed bytes, the OS file cache change, the
scratch bank bytes written and read with their effective rate, and the host bank
ledger. The record never sums them. The checks here follow the case file
docs/validation/cases/G5-02.json: the fields are present with instrument and
caveat, allocated never exceeds reserved, the working set is a host quantity,
and the disk bytes of a disk-bank run equal the bank file sizes.
"""
import json

import pytest
import torch

from torchfdtd import StreamedAdjointOptions, StreamedSimulation
from torchfdtd.memory_accounting import (MemoryMeter, SCOPE, process_io, process_memory, system_file_cache)
from g5_support import CUDA, Record, scene, sphere, streamed_options

CASE = 'G5-02'
RECORD = Record('G5-02', CASE, 'tests/test_memory_accounting_g5.py')
cuda = pytest.mark.skipif(not CUDA, reason='CUDA unavailable')
DEVICE_FIELDS = ('peak_torch_allocated_bytes', 'peak_torch_reserved_bytes', 'cuda_process_memory_bytes')
HOST_FIELDS = ('peak_process_rss_bytes', 'peak_process_private_bytes', 'peak_process_committed_bytes', 'host_bank_bytes')
SYSTEM_FIELDS = ('os_file_cache_bytes',)
DISK_FIELDS = ('scratch_disk_written_bytes', 'scratch_disk_read_bytes', 'scratch_disk_peak_file_bytes')
FIELDS = DEVICE_FIELDS+HOST_FIELDS+SYSTEM_FIELDS+DISK_FIELDS+('scratch_disk_bandwidth_bytes_per_second',)


def run(path, tmp_path, *, device='cuda'):
    project = scene('float32', 'point')
    epsilon = sphere(project, inside=4., outside=1.).requires_grad_()
    if device == 'cpu':
        options = StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4, checkpoints=2)
    else:
        options = streamed_options(path, tmp_path/'banks')
    result = StreamedSimulation(project, options)(epsilon)
    torch.autograd.grad(result.signals.square().sum(), epsilon)
    return result.report


def check_shape(record, *, device):
    for name in FIELDS:
        field = record[name]
        assert {'domain', 'instrument', 'caveat'} <= set(field), name
        assert field['domain'] in ('device', 'host', 'system', 'disk')
    assert record['total_memory_bytes'] is None and 'never summed' in record['total_memory_note']
    assert record['accounting_scope'] == SCOPE
    for name in DEVICE_FIELDS:
        assert record[name]['domain'] == 'device'
        if device:
            assert isinstance(record[name]['bytes'], int) and record[name]['bytes'] >= 0
            assert 'torch.cuda' in record[name]['instrument']
        else:
            assert record[name]['bytes'] is None and 'not applicable' in record[name]['instrument']
    for name in HOST_FIELDS:
        assert record[name]['domain'] == 'host' and isinstance(record[name]['bytes'], int)
    rss = record['peak_process_rss_bytes']
    assert 'not device memory' in rss['caveat'] and 'torch.cuda' not in rss['instrument']
    assert rss['bytes'] >= rss['at_start_bytes'] > 0 and rss['delta_bytes'] == rss['bytes']-rss['at_start_bytes']
    assert rss['os_lifetime_peak_bytes'] >= rss['bytes']
    assert record['peak_process_committed_bytes']['bytes'] > 0 and record['peak_process_private_bytes']['bytes'] > 0
    assert isinstance(record['os_file_cache_bytes']['bytes'], int) and 'system-wide' in record['os_file_cache_bytes']['caveat'].lower()
    assert record['sampler']['samples'] >= 1 and record['seconds'] > 0
    json.dumps(record)


@cuda
@pytest.mark.parametrize('path', ['host_sync', 'disk_sync', 'host_async'])
def test_streamed_report_exposes_instrumented_memory_fields(path, tmp_path):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report = run(path, tmp_path)
    for phase in ('forward', 'backward'):
        record = report[phase+'_memory']
        check_shape(record, device=True)
        allocated, reserved = record['peak_torch_allocated_bytes'], record['peak_torch_reserved_bytes']
        assert allocated['exact'] and reserved['exact']
        assert 0 < allocated['bytes'] <= reserved['bytes']
        assert allocated['at_start_bytes'] <= reserved['at_start_bytes']
        assert allocated['delta_bytes'] > 0
        assert allocated['bytes'] not in (record['peak_process_rss_bytes']['bytes'], record['peak_process_committed_bytes']['bytes'])
        assert record['cuda_process_memory_bytes']['device_free_at_start_bytes'] >= record['cuda_process_memory_bytes']['device_min_free_bytes']
        if path.startswith('host'):
            assert record['host_bank_bytes']['bytes'] > 0 and record['host_bank_bytes']['banks_created'] > 0
            assert record['host_bank_bytes']['bytes'] % report['state_bytes'] == 0
            assert record['scratch_disk_written_bytes']['bytes'] == record['scratch_disk_read_bytes']['bytes'] == 0
        RECORD.add(f'{path}/{phase}', path=path, phase=phase, state_bytes=report['state_bytes'],
                   **{name: {k: v for k, v in record[name].items() if k not in ('instrument', 'caveat')} for name in FIELDS},
                   sampler=record['sampler'], seconds=record['seconds'])


@cuda
def test_disk_bytes_match_the_bank_file_sizes(tmp_path):
    report = run('disk_sync', tmp_path)
    state = report['state_bytes']
    for phase in ('forward', 'backward'):
        record, store = report[phase+'_memory'], report[phase+'_backing_store']
        assert store['closed'] and store['created_file_bytes'] == store['created_banks']*state
        written, read = record['scratch_disk_written_bytes'], record['scratch_disk_read_bytes']
        assert written['bytes'] == store['logical_written_bytes'] and read['bytes'] == store['logical_read_bytes']
        peak = record['scratch_disk_peak_file_bytes']
        assert peak['bytes'] == store['peak_logical_file_bytes'] and peak['bytes'] % state == 0
        assert 1 <= peak['bytes']//state <= report['state_bank_capacity']
        assert record['host_bank_bytes']['bytes'] == 0 and record['host_bank_bytes']['banks_created'] == 0
        rate = record['scratch_disk_bandwidth_bytes_per_second']
        assert rate['written'] == pytest.approx(written['bytes']/record['seconds'])
        assert rate['read'] == pytest.approx(read['bytes']/record['seconds'])
        assert 'not physical' in rate['caveat']
        if written['process_io_delta_bytes'] is not None:
            assert written['process_io_delta_bytes'] >= written['bytes']
    forward = report['forward_memory']['scratch_disk_written_bytes']['bytes']
    # Every owned row of every forward bank is committed exactly once.
    assert forward == report['forward_backing_store']['created_file_bytes']
    assert report['backward_memory']['scratch_disk_written_bytes']['bytes'] >= report['backward_backing_store']['created_file_bytes']
    assert report['backward_memory']['scratch_disk_read_bytes']['bytes'] > 0
    RECORD.add('disk_sync/bank_files', state_bytes=state,
               forward=dict(created_banks=report['forward_backing_store']['created_banks'],
                            created_file_bytes=report['forward_backing_store']['created_file_bytes'],
                            written_bytes=forward, read_bytes=report['forward_memory']['scratch_disk_read_bytes']['bytes'],
                            peak_file_bytes=report['forward_memory']['scratch_disk_peak_file_bytes']['bytes']),
               backward=dict(created_banks=report['backward_backing_store']['created_banks'],
                             created_file_bytes=report['backward_backing_store']['created_file_bytes'],
                             written_bytes=report['backward_memory']['scratch_disk_written_bytes']['bytes'],
                             read_bytes=report['backward_memory']['scratch_disk_read_bytes']['bytes'],
                             peak_file_bytes=report['backward_memory']['scratch_disk_peak_file_bytes']['bytes']))


def test_cpu_tiles_report_host_fields_and_no_device_fields(tmp_path):
    report = run('host_sync', tmp_path, device='cpu')
    for phase in ('forward', 'backward'):
        check_shape(report[phase+'_memory'], device=False)
        assert report[phase+'_memory']['host_bank_bytes']['bytes'] > 0


def test_operating_system_instruments_return_integers():
    memory = process_memory()
    assert memory is not None and memory['working_set_bytes'] > 0 and memory['lifetime_peak_working_set_bytes'] >= memory['working_set_bytes']
    cache = system_file_cache()
    assert cache is not None and cache['bytes'] >= 0
    io = process_io()
    assert io is not None and io['read_bytes'] >= 0 and io['write_bytes'] >= 0


@cuda
def test_allocator_peak_is_exact_only_when_the_phase_sets_a_new_process_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with MemoryMeter('cuda') as meter:
        block = torch.empty(2**22, dtype=torch.uint8, device='cuda')
        del block
    record = meter.record(1.)
    assert record['peak_torch_allocated_bytes']['exact'] and record['peak_torch_allocated_bytes']['delta_bytes'] >= 2**22
    with MemoryMeter('cuda') as meter:
        small = torch.empty(2**10, dtype=torch.uint8, device='cuda')
        del small
    record = meter.record(1.)
    assert not record['peak_torch_allocated_bytes']['exact']
    assert 'lower bound' in record['peak_torch_allocated_bytes']['instrument']
    assert record['peak_torch_allocated_bytes']['prior_process_peak_bytes'] >= 2**22
