"""Bounded two-slot CUDA/CPU boundary-history transport.

Caller admission precedes construction. All frame consumers must use the
captured compute stream. Returned frames are borrowed until the next call.
Close explicitly, including error paths; no destructor, worker or executor.
"""
from types import SimpleNamespace
import math
import sys

import torch


def metadata(shape, chunk_steps=32, dtype=torch.float32):
    if dtype not in (torch.float32, torch.complex64):
        raise ValueError('Trace dtype must be float32 or complex64.')
    item = 4 if dtype == torch.float32 else 8
    shape = tuple(shape)
    if (len(shape) != 5 or shape[1] != 2 or shape[4] != 2
            or any(type(n) is not int or n <= 0 for n in shape)
            or type(chunk_steps) is not int or not 1 <= chunk_steps <= shape[0]):
        raise ValueError('Expected positive integer (T,2,Nx,Ny,2) and 1 <= chunk_steps <= T.')
    archive = item*math.prod(shape)
    chunk = item*chunk_steps*math.prod(shape[1:])
    host = archive+2*chunk+65536
    device = 2*chunk
    if max(host, device) > min(sys.maxsize, 2**63-1):
        raise ValueError('Trace storage exceeds signed 64-bit addressing.')
    return dict(shape=shape, chunk_steps=chunk_steps, dtype=str(dtype), item_bytes=item, archive_bytes=archive,
                pinned_staging_bytes=2*chunk, device_staging_bytes=device,
                metadata_allowance_bytes=65536, host_reservation_bytes=host,
                device_payload_bytes=device, reusable_event_count=8,
                owned_copy_stream_count=1)


def _optional_note(error, message):
    """Python 3.10 exceptions have no add_note; never replace the original."""
    try:
        add_note = getattr(error, 'add_note', None)
        if callable(add_note):
            add_note(message)
    except BaseException:
        # A custom exception attribute/method must not break error cleanup.
        pass


class AsyncBoundaryTrace:
    def __init__(self, shape, device, *, chunk_steps=32, dtype=torch.float32):
        self._metadata = metadata(shape, chunk_steps, dtype)
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('This transport requires a CUDA device.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        self.shape, self.k, self.device = tuple(shape), chunk_steps, device
        self.compute = torch.cuda.current_stream(device)
        self.copy = torch.cuda.Stream(device=device)
        self.state, self.reading = 'writing', False
        self.next_step, self.borrowed = 0, None
        self.slots = []
        self.archive = torch.empty(self.shape, dtype=dtype, device='cpu')
        # No copy or kernel work is enqueued during construction. Allocation
        # errors release partial slots through ordinary local ownership.
        for _ in range(2):
            self.slots.append(SimpleNamespace(
                cpu=torch.empty((self.k, *self.shape[1:]), dtype=dtype, pin_memory=True),
                gpu=torch.empty((self.k, *self.shape[1:]), dtype=dtype, device=device),
                ready=torch.cuda.Event(), d2h=torch.cuda.Event(),
                h2d=torch.cuda.Event(), consumed=torch.cuda.Event(),
                pending=None, uploaded=False, consumed_valid=False))

    def allocation_metadata(self):
        return dict(self._metadata)

    def _stream_guard(self):
        if torch.cuda.current_stream(self.device) != self.compute:
            raise RuntimeError('Trace packing and consumption must use the captured compute stream.')

    def _writing_guard(self, n, *, commit=False):
        self._stream_guard()
        if (self.state != 'writing' or type(n) is not int or n != self.next_step
                or not 0 <= n < self.shape[0]
                or (self.borrowed != n if commit else self.borrowed is not None)):
            raise RuntimeError('Expected one frame(n), then commit(n), in ascending order.')

    def _retire(self, slot):
        if slot.pending is not None:
            start, count = slot.pending
            slot.d2h.synchronize()
            self.archive[start:start+count].copy_(slot.cpu[:count])
            slot.pending = None

    def _drain(self):
        # Always drain the original streams, never the caller's current stream.
        # Attempt both even if one synchronize raises. No exception is stored.
        try:
            self.copy.synchronize()
        finally:
            self.compute.synchronize()

    def _cleanup_after_error(self, error):
        self.state = 'failed'
        try:
            self._drain()
        except BaseException as cleanup_error:
            _optional_note(error, 'Trace stream drain also failed: '+type(cleanup_error).__name__)

    def frame(self, n):
        self._writing_guard(n)
        try:
            slot = self.slots[(n//self.k)%2]
            if n % self.k == 0:
                self._retire(slot)
            self.borrowed = n
            return slot.gpu[n % self.k]
        except BaseException as error:
            self._cleanup_after_error(error)
            raise

    def commit(self, n):
        self._writing_guard(n, commit=True)
        try:
            if n % self.k == self.k-1 or n == self.shape[0]-1:
                start = (n//self.k)*self.k
                count = n-start+1
                slot = self.slots[(n//self.k)%2]
                slot.ready.record(self.compute)
                with torch.cuda.stream(self.copy):
                    self.copy.wait_event(slot.ready)
                    slot.cpu[:count].copy_(slot.gpu[:count], non_blocking=True)
                    slot.d2h.record(self.copy)
                slot.pending = (start, count)
            self.borrowed = None
            self.next_step += 1
        except BaseException as error:
            self._cleanup_after_error(error)
            raise

    def finish(self):
        self._stream_guard()
        if self.state != 'writing' or self.borrowed is not None or self.next_step != self.shape[0]:
            raise RuntimeError('Cannot complete an incomplete, borrowed, failed or finalized trace.')
        try:
            for slot in self.slots:
                self._retire(slot)
            self.state = 'complete'
        except BaseException as error:
            self._cleanup_after_error(error)
            raise

    def reverse(self):
        self._stream_guard()
        if self.state != 'complete' or self.reading:
            raise RuntimeError('Reverse requires a complete trace and no other active reader.')
        self.reading = True
        try:
            return _ReverseReader(self)
        except BaseException as error:
            self.reading = False
            self._cleanup_after_error(error)
            raise

    def close(self):
        if self.state == 'closed':
            return
        if self.reading:
            raise RuntimeError('Close the active reverse context before closing its transport.')
        try:
            self._drain()
        except BaseException:
            self.state = 'failed'
            raise
        self.state = 'closed'
        self.slots.clear()
        self.archive = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.state = 'failed'
            try:
                self.close()
            except BaseException as cleanup_error:
                _optional_note(error, 'Trace close also failed: '+type(cleanup_error).__name__)
        else:
            self.close()


class _ReverseReader:
    def __init__(self, owner):
        self.owner = owner
        self.device, self.shape = owner.device, owner.shape
        self.expected, self.previous = self.shape[0]-1, None
        self.map, self.closed = {}, False
        top = self.expected//owner.k
        self._load(0, top)
        if top:
            self._load(1, top-1)

    def _load(self, index, chunk):
        owner, slot = self.owner, self.owner.slots[index]
        if slot.uploaded:
            slot.h2d.synchronize()
        start = chunk*owner.k
        count = min(owner.k, self.shape[0]-start)
        slot.cpu[:count].copy_(owner.archive[start:start+count])
        with torch.cuda.stream(owner.copy):
            if slot.consumed_valid:
                owner.copy.wait_event(slot.consumed)
            slot.gpu[:count].copy_(slot.cpu[:count], non_blocking=True)
            slot.h2d.record(owner.copy)
        slot.uploaded = True
        self.map[chunk] = index

    def __getitem__(self, n):
        owner = self.owner
        owner._stream_guard()
        if self.closed or owner.state != 'complete' or type(n) is not int or n != self.expected or n < 0:
            raise RuntimeError('Reverse frames must be consumed once in descending order.')
        try:
            chunk = n//owner.k
            if self.previous is not None and chunk != self.previous:
                # Preserve map entry until recording succeeds, so close can
                # still identify the most recent consumer after an exception.
                old = self.map[self.previous]
                slot = owner.slots[old]
                slot.consumed.record(owner.compute)
                slot.consumed_valid = True
                del self.map[self.previous]
                self.previous = None
                if chunk > 0:
                    self._load(old, chunk-1)
            slot = owner.slots[self.map[chunk]]
            owner.compute.wait_event(slot.h2d)
            self.previous = chunk
            self.expected -= 1
            return slot.gpu[n % owner.k]
        except BaseException as error:
            owner._cleanup_after_error(error)
            raise

    def close(self):
        if self.closed:
            return
        owner = self.owner
        try:
            if self.previous is not None:
                slot = owner.slots[self.map[self.previous]]
                slot.consumed.record(owner.compute)
                slot.consumed_valid = True
            owner._drain()
        except BaseException as error:
            owner._cleanup_after_error(error)
            raise
        finally:
            owner.reading = False
            self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Reverse reader is closed.')
        return self

    def __exit__(self, kind, error, traceback):
        if error is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException as cleanup_error:
                _optional_note(error, 'Reverse close also failed: '+type(cleanup_error).__name__)
