"""Bounded slab allocations and cached CUDA argument views.

Every geometry specialization aliases the same high-water buffers. Cache growth
cannot retain a separate field bank per slab. Reallocation invalidates bindings
before replacing an allocation. The tile driver waits for the previous slot's
device work and host consumer before handing it to the next tile.
"""
from collections import OrderedDict
from .tensor_packet import TensorLayout, pack_tensors

import torch


class HostTransfer:
    def __init__(self, values, event=None, owner=None):
        self.values, self.event, self.owner = values, event, owner

    def wait(self):
        if self.event is not None:self.event.synchronize()
        self.owner = None
        return self.values


class TileWorkspace:
    def __init__(self, device, *, cache_entries=32, asynchronous=False):
        self.device = torch.device(device)
        self.buffers = {}
        self.bindings = OrderedDict()
        self.cache_entries = cache_entries
        self.allocations = 0
        self.binding_hits = 0
        self.binding_misses = 0
        self.asynchronous = asynchronous
        self.pinned = {}
        self.host_staging = {}
        self.host_allocations = self.pinned_allocations = 0
        self.h2d_bytes = self.d2h_bytes = 0
        self.events = []
        if asynchronous:
            if self.device.type != 'cuda':raise ValueError('Asynchronous tiles require CUDA.')
            self.h2d = torch.cuda.Stream(device=self.device)
            self.d2h = torch.cuda.Stream(device=self.device)

    def pinned_array(self, name, value):
        return self.host_array(name, value.shape, value.dtype, pinned=True)

    def host_array(self, name, shape, dtype, *, pinned=False):
        count = 1
        for length in shape:count *= length
        pool = self.pinned if pinned else self.host_staging
        current = pool.get(name)
        if current is None or current.numel() < count or current.dtype != dtype:
            current = torch.empty(count, dtype=dtype, device='cpu', pin_memory=pinned)
            pool[name] = current
            if pinned:self.pinned_allocations += 1
            else:self.host_allocations += 1
        return current[:count].view(shape)

    def copy_packet(self, name, tensors):
        """Pack CPU values directly into this slot's transfer storage."""
        values = tuple(tensors)
        layout = TensorLayout.from_tensors(values)
        if values[0].device.type != 'cpu':
            raise ValueError('Tile input packets must originate on CPU.')
        target = self.array(name, (layout.count,), layout.dtype)
        if self.device.type == 'cpu':
            return pack_tensors(values, out=target)
        staging = self.host_array('input:'+name, (layout.count,), layout.dtype,
                                  pinned=self.asynchronous)
        pack_tensors(values, out=staging)
        self._copy_into(name, target, staging, staged=True)
        return target, layout

    def _copy_into(self, name, target, value, *, staged=False):
        if self.asynchronous and value.device.type == 'cpu':
            pinned = value if staged else self.pinned_array('input:'+name, value)
            if not staged:pinned.copy_(value)
            with torch.cuda.stream(self.h2d):
                target.copy_(pinned, non_blocking=True)
                target.record_stream(self.h2d)
                ready = torch.cuda.Event()
                ready.record(self.h2d)
            torch.cuda.current_stream(self.device).wait_event(ready)
            self.events.append(ready)
            self.h2d_bytes += value.numel()*value.element_size()
            return target
        target.copy_(value)
        if target.is_cuda and value.device.type == 'cpu':
            self.h2d_bytes += value.numel()*value.element_size()
        return target

    def array(self, name, shape, dtype, *, invalidate_bindings=True):
        count = 1
        for length in shape:count *= length
        value = self.buffers.get(name)
        if value is None or value.numel() < count or value.dtype != dtype:
            # Cached views own allocations. Invalidate them before replacement.
            if invalidate_bindings:self.bindings.clear()
            # The slot is drained before reuse. In particular, output buffers
            # are not kernel arguments. Release the expired pool allocation
            # before asking CUDA for its larger replacement, avoiding old+new
            # peak memory at the forward-to-transpose size transition.
            self.buffers.pop(name,None)
            del value
            value = torch.empty(count, dtype=dtype, device=self.device)
            self.buffers[name] = value
            self.allocations += 1
        if value.is_cuda:value.record_stream(torch.cuda.current_stream(self.device))
        return value[:count].view(shape)

    def zeros(self, name, like):
        return self.array(name, like.shape, like.dtype).zero_()

    def copy(self, name, value):
        target = self.array(name, value.shape, value.dtype)
        return self._copy_into(name, target, value)

    def to_host(self, tensors):
        tensors = tuple(tensors)
        layout = TensorLayout.from_tensors(tensors)
        # Forward packets can be homogeneous complex, whereas backward adds a
        # real gradient. A byte buffer reuses one allocation across both layouts.
        # Output packets never appear in a cached Yee/adjoint argument list.
        count = layout.count*(1 if layout.dtype == torch.uint8 else tensors[0].element_size())
        packed = self.array('output_packet', (count,), torch.uint8, invalidate_bindings=False)
        pack_tensors(tensors, out=packed.view(layout.dtype))
        event = None
        if self.asynchronous:
            host = self.pinned_array('output', packed)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(ready)
                host.copy_(packed, non_blocking=True)
                packed.record_stream(self.d2h)
                event = torch.cuda.Event()
                event.record(self.d2h)
            self.events.append(event)
        elif packed.is_cuda:
            host = self.host_array('output', packed.shape, packed.dtype)
            host.copy_(packed)
        else:host = packed
        if packed.is_cuda:self.d2h_bytes += packed.numel()*packed.element_size()
        values = layout.unpack(host.view(layout.dtype))
        return HostTransfer(values, event, packed if self.asynchronous else None)

    def drain(self):
        for event in self.events:event.synchronize()
        self.events.clear()

    def cuda_arguments(self, source, tensors, view):
        key = (source, tuple((t.data_ptr(), tuple(t.shape), t.dtype, t.device.index) for t in tensors))
        if key in self.bindings:
            self.binding_hits += 1
            self.bindings.move_to_end(key)
            return self.bindings[key]
        self.binding_misses += 1
        arrays = tuple(view(t) for t in tensors)
        self.bindings[key] = arrays
        if len(self.bindings) > self.cache_entries:self.bindings.popitem(last=False)
        return arrays

    @property
    def allocated_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers.values())

    @property
    def pinned_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.pinned.values())

    @property
    def host_staging_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.host_staging.values())
