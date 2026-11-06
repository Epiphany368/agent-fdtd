"""Protocol/metadata mocks only. No CUDA runtime calls or tensors allocated."""
from types import SimpleNamespace

import pytest

import torch
import torchfdtd.reversible_trace as module


def owner(monkeypatch):
    result = module.AsyncBoundaryTrace.__new__(module.AsyncBoundaryTrace)
    counters = dict(compute=0, copy=0)
    def sync(name):
        counters[name] += 1
    result.compute = SimpleNamespace(synchronize=lambda: sync('compute'))
    result.copy = SimpleNamespace(synchronize=lambda: sync('copy'))
    result.device = 'cuda:0'
    result.state, result.reading = 'complete', False
    result.shape, result.k = (96, 2, 6, 7, 2), 7
    result.next_step, result.borrowed = 0, None
    result.slots, result.archive = [], object()
    monkeypatch.setattr(module.torch.cuda, 'current_stream', lambda device: result.compute)
    return result, counters


@pytest.mark.parametrize('dtype,item', [(torch.float32, 4), (torch.complex64, 8)])
def test_metadata_fixed_bytes_and_rejections(dtype, item):
    result = module.metadata((96, 2, 6, 7, 2), 7, dtype)
    assert result['archive_bytes'] == 64512*item//4
    assert result['pinned_staging_bytes'] == result['device_staging_bytes'] == 9408*item//4
    assert result['host_reservation_bytes'] == (64512+9408)*item//4+65536
    assert result['reusable_event_count'] == 8
    for shape, k in [((0, 2, 6, 7, 2), 7), ((96, 4, 6, 7, 2), 7),
                     ((96, 2, 6, 7, 2), True), ((96, 2, 6, 7, 2), 97),
                     ((2**63, 2, 6, 7, 2), 7)]:
        with pytest.raises(ValueError):
            module.metadata(shape, k)


def test_forward_order_and_completion_guards(monkeypatch):
    value, _ = owner(monkeypatch)
    value.state = 'writing'
    for step in (1, -1, True):
        with pytest.raises(RuntimeError, match='ascending'):
            value.frame(step)
    with pytest.raises(RuntimeError, match='ascending'):
        value.commit(0)
    value.borrowed = 0
    with pytest.raises(RuntimeError, match='ascending'):
        value.frame(0)
    with pytest.raises(RuntimeError, match='incomplete'):
        value.finish()
    value.borrowed, value.next_step = None, 96
    value.finish()
    assert value.state == 'complete'
    with pytest.raises(RuntimeError, match='finalized'):
        value.finish()


def test_reader_initialization_failure_drains_and_resets(monkeypatch):
    value, counts = owner(monkeypatch)
    loads = []
    def load(reader, index, chunk):
        loads.append((index, chunk))
        if index == 1:
            raise RuntimeError('injected second prefetch failure')
    monkeypatch.setattr(module._ReverseReader, '_load', load)
    with pytest.raises(RuntimeError, match='second prefetch'):
        value.reverse()
    assert loads == [(0, 13), (1, 12)]
    assert not value.reading and value.state == 'failed'
    assert counts == dict(compute=1, copy=1)
    with pytest.raises(RuntimeError, match='complete trace'):
        value.reverse()


def test_context_cancellation_releases_reader_and_keeps_completed_archive(monkeypatch):
    value, counts = owner(monkeypatch)
    monkeypatch.setattr(module._ReverseReader, '_load', lambda *args: None)
    with pytest.raises(RuntimeError, match='caller cancelled'):
        with value.reverse():
            with pytest.raises(RuntimeError, match='other active reader'):
                value.reverse()
            raise RuntimeError('caller cancelled')
    assert not value.reading and value.state == 'complete'
    assert counts == dict(compute=1, copy=1)
    with value.reverse():
        pass
    assert not value.reading


def test_stream_contract_and_cleanup_original_stream(monkeypatch):
    value, counts = owner(monkeypatch)
    monkeypatch.setattr(module.torch.cuda, 'current_stream', lambda device: object())
    with pytest.raises(RuntimeError, match='captured compute stream'):
        value.reverse()
    value.close()  # Cleanup works even when the caller has changed streams.
    assert counts == dict(compute=1, copy=1)
    assert value.state == 'closed' and value.archive is None
    value.close()
    assert counts == dict(compute=1, copy=1)


def test_reader_close_failure_resets_owner_without_exception_storage(monkeypatch):
    value, counts = owner(monkeypatch)
    monkeypatch.setattr(module._ReverseReader, '_load', lambda *args: None)
    reader = value.reverse()
    def failed_record(stream):
        raise RuntimeError('event record failed')
    value.slots = [SimpleNamespace(consumed=SimpleNamespace(record=failed_record))]
    reader.previous, reader.map = 13, {13: 0}
    with pytest.raises(RuntimeError, match='event record failed'):
        reader.close()
    assert reader.closed and not value.reading and value.state == 'failed'
    assert counts == dict(compute=1, copy=1)
    assert not any(isinstance(item, BaseException) for item in vars(value).values())


def test_drain_attempts_compute_even_when_copy_stream_fails(monkeypatch):
    value, counts = owner(monkeypatch)
    def failed_sync():
        raise RuntimeError('copy stream failed')
    value.copy.synchronize = failed_sync
    with pytest.raises(RuntimeError, match='copy stream failed'):
        value.close()
    assert counts['compute'] == 1 and value.state == 'failed'


def test_missing_add_note_preserves_original_error_and_cleanup(monkeypatch):
    value, counts = owner(monkeypatch)
    class LegacyError(RuntimeError):
        @property
        def add_note(self):
            raise AttributeError('Python 3.10 style exception')
    original = LegacyError('original prefetch failure')
    def fail_load(*args):
        raise original
    def fail_copy_sync():
        raise RuntimeError('secondary cleanup failure')
    monkeypatch.setattr(module._ReverseReader, '_load', fail_load)
    value.copy.synchronize = fail_copy_sync
    with pytest.raises(LegacyError, match='original prefetch failure') as caught:
        value.reverse()
    assert caught.value is original
    assert not value.reading and value.state == 'failed'
    assert counts['compute'] == 1
    assert not any(isinstance(item, BaseException) for item in vars(value).values())


def test_invalid_transport_metadata_precedes_device_or_tensor_allocation(monkeypatch):
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: pytest.fail('allocation before admission'))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: pytest.fail('CUDA before validation'))
    with pytest.raises(ValueError, match='dtype'):
        module.AsyncBoundaryTrace((10, 2, 3, 4, 2), 'cuda', chunk_steps=7, dtype=torch.float64)
    with pytest.raises(ValueError, match='CUDA device'):
        module.AsyncBoundaryTrace((10, 2, 3, 4, 2), 'cpu', chunk_steps=7)
    with pytest.raises(ValueError, match='chunk_steps'):
        module.AsyncBoundaryTrace((10, 2, 3, 4, 2), 'cuda', chunk_steps=11)


def test_final_partial_chunk_commit_and_retire_without_cuda(monkeypatch):
    from contextlib import nullcontext
    value, _ = owner(monkeypatch)
    operations = []
    class Buffer:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, index):
            operations.append((self.name, 'slice', index))
            return self
        def copy_(self, source, **kwargs):
            operations.append((self.name, 'copy', source.name, kwargs))
    def event(name):
        return SimpleNamespace(record=lambda stream: operations.append((name, 'record')),
                               synchronize=lambda: operations.append((name, 'synchronize')))
    value.state, value.shape, value.k = 'writing', (9, 2, 6, 7, 2), 7
    value.next_step, value.borrowed = 8, 8
    value.copy.wait_event = lambda event: operations.append(('copy', 'wait'))
    slot = SimpleNamespace(cpu=Buffer('pinned'), gpu=Buffer('device'),
                           ready=event('ready'), d2h=event('d2h'), pending=None)
    value.slots = [SimpleNamespace(pending=None), slot]
    value.archive = Buffer('archive')
    monkeypatch.setattr(module.torch.cuda, 'stream', lambda stream: nullcontext())
    value.commit(8)
    assert slot.pending == (7, 2) and value.next_step == 9 and value.borrowed is None
    assert ('pinned', 'slice', slice(None, 2)) in operations
    value.finish()
    assert value.state == 'complete' and slot.pending is None
    assert operations.index(('d2h', 'synchronize')) < operations.index(('archive', 'slice', slice(7, 9)))
