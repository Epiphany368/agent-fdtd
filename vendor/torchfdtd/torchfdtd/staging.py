"""Bounded event-owned, lossless CUDA/host checkpoint staging.

Pageable persistent host checkpoints are separate from the small pinned pool.
A slot cannot be reused until both its CPU task and its CUDA consumer finish.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _Slot:
    host: tuple
    device: tuple
    leased: bool = False
    kind: str = ''
    future: object = None
    last_use: object = None


@dataclass
class LoadTicket:
    slot: _Slot
    future: object
    consumed: bool = False


class AsyncStateStaging:
    def __init__(self,state,slots,report):
        if not state or not all(t.is_cuda for t in state):
            raise ValueError('Asynchronous state staging requires CUDA state.')
        self.device=state[0].device
        self.report=report
        self.closed=False
        self.d2h=torch.cuda.Stream(device=self.device)
        self.h2d=torch.cuda.Stream(device=self.device)
        self.slots=[_Slot(tuple(torch.empty(t.shape,dtype=t.dtype,pin_memory=True) for t in state),
                          tuple(torch.empty_like(t) for t in state)) for _ in range(slots)]
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='torchfdtd-state-io')
        self.tasks=set()
        self.error=None
        size=sum(t.numel()*t.element_size() for t in state)
        report.update(staging_slots=slots,pinned_staging_bytes=size*slots,device_staging_bytes=size*slots,
                      async_saves=0,async_loads=0,peak_staging_leases=0)

    def acquire(self,kind):
        if self.closed:raise RuntimeError('State staging is closed.')
        available=[s for s in self.slots if not s.leased and (s.future is None or s.future.done())]
        if not available:
            pending=next((s for s in self.slots if s.kind=='save' and s.future is not None),None)
            if pending is None:raise RuntimeError('All staging slots are owned by unconsumed loads.')
            pending.future.result()
            available=[pending]
        slot=next((s for s in available if s.last_use is None or s.last_use.query()),available[0])
        if slot.future is not None:slot.future.result()
        if slot.last_use is not None:slot.last_use.synchronize()
        slot.leased=True;slot.kind=kind;slot.future=None;slot.last_use=None
        self.report['peak_staging_leases']=max(self.report['peak_staging_leases'],sum(s.leased for s in self.slots))
        return slot

    def track(self,future):
        self.tasks.add(future)
        def finished(done):
            try:
                error=done.exception()
                if error is not None and self.error is None:self.error=error
            finally:self.tasks.discard(done)
        future.add_done_callback(finished)

    def forget(self,future):
        # Completed Future results own persistent CPU arrays. Dropping a
        # checkpoint must not retain those arrays in a reusable staging slot.
        for slot in self.slots:
            if slot.future is future and slot.kind=='save':slot.future=None

    def save(self,state,*,path=None,max_file_bytes=None):
        slot=self.acquire('save')
        compute=torch.cuda.current_stream(self.device)
        for dst,src in zip(slot.device,state):dst.copy_(src,non_blocking=True)
        ready=torch.cuda.Event();ready.record(compute)
        with torch.cuda.stream(self.d2h):
            self.d2h.wait_event(ready)
            for dst,src in zip(slot.host,slot.device):dst.copy_(src,non_blocking=True)
            copied=torch.cuda.Event();copied.record(self.d2h)
        def finish():
            try:
                copied.synchronize()
                if path is None:
                    stored=tuple(torch.empty(t.shape,dtype=t.dtype,device='cpu',pin_memory=False) for t in slot.host)
                    for dst,src in zip(stored,slot.host):dst.copy_(src)
                    return stored
                try:
                    np.savez(path,**{f's{i}':t.numpy() for i,t in enumerate(slot.host)})
                    if max_file_bytes is not None and path.stat().st_size>max_file_bytes:
                        raise ValueError('Checkpoint archive exceeded its reserved file bytes.')
                    return path
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
            finally:slot.leased=False
        slot.future=self.executor.submit(finish)
        self.track(slot.future)
        self.report['async_saves']+=1
        return slot.future

    def load(self,stored,*,disk=False):
        slot=self.acquire('load')
        def prepare():
            value=stored.result() if hasattr(stored,'result') else stored
            if disk:
                with np.load(value,allow_pickle=False) as archive:
                    for i,dst in enumerate(slot.host):dst.copy_(torch.from_numpy(archive[f's{i}']))
            else:
                for dst,src in zip(slot.host,value):dst.copy_(src)
            with torch.cuda.device(self.device),torch.cuda.stream(self.h2d):
                for dst,src in zip(slot.device,slot.host):dst.copy_(src,non_blocking=True)
                ready=torch.cuda.Event();ready.record(self.h2d)
            return ready
        slot.future=self.executor.submit(prepare)
        self.track(slot.future)
        self.report['async_loads']+=1
        return LoadTicket(slot,slot.future)

    def consume(self,ticket,state=None):
        if ticket.consumed:raise RuntimeError('A staging load can only be consumed once.')
        ready=ticket.future.result()
        compute=torch.cuda.current_stream(self.device)
        compute.wait_event(ready)
        if state is not None:
            for dst,src in zip(state,ticket.slot.device):dst.copy_(src,non_blocking=True)
        used=torch.cuda.Event();used.record(compute)
        ticket.slot.last_use=used
        ticket.slot.leased=False
        ticket.consumed=True

    def close(self):
        if self.closed:return
        self.closed=True
        error=None
        for future in list(self.tasks):
            try:future.result()
            except BaseException as exc:
                if error is None:error=exc
        self.executor.shutdown(wait=True)
        self.d2h.synchronize();self.h2d.synchronize()
        for slot in self.slots:
            if slot.last_use is not None:slot.last_use.synchronize()
            slot.future=None
        if error is None:error=self.error
        if error is not None:raise error
