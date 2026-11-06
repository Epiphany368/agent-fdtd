"""Bounded slab I/O for transient field banks, without whole-file mappings.

The OS may cache buffered file I/O. Logical traffic is not physical SSD traffic.
Only files created by this store are removed, and the private directory must be
empty before it is removed. This is scratch storage, not a durable restart.
"""
import math
import os
from pathlib import Path
import shutil
import tempfile
import weakref

import torch


def disk_free(directory):
    path = Path(directory).expanduser().resolve()
    while not path.exists():
        if path.parent == path:raise ValueError('State backing volume is unavailable.')
        path = path.parent
    if not path.is_dir():raise ValueError('State backing directory must be a directory.')
    return shutil.disk_usage(path).free


class StateStore:
    def __init__(self,directory,budget,free_reserve_bytes=0):
        if isinstance(free_reserve_bytes,bool) or not isinstance(free_reserve_bytes,int) or free_reserve_bytes<0:
            raise ValueError('free_reserve_bytes must be a nonnegative integer.')
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True,exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='torchfdtd-state-',dir=parent))
        self.budget = budget
        self.free_reserve_bytes = free_reserve_bytes
        self.banks = weakref.WeakValueDictionary()
        self.closed = False
        self.live_bytes = self.peak_bytes = self.created_banks = self.created_file_bytes = 0
        self.read_bytes = self.written_bytes = self.max_read_bytes = 0

    def new_state(self,templates):
        if self.closed:raise RuntimeError('State store is closed.')
        sizes = [math.prod(s.shape)*s.element_size() for s in templates]
        size = sum(sizes)
        if self.live_bytes+size > self.budget:raise MemoryError('Live field banks exceed the disk budget.')
        if size+self.free_reserve_bytes > disk_free(self.root):
            raise OSError('Insufficient free space for a field bank and reserved disk headroom.')
        key = self.created_banks
        bank = _Bank(self,key,size)
        self.banks[key] = bank
        self.created_banks += 1
        self.created_file_bytes += bank.file_bytes
        self.live_bytes += size
        self.peak_bytes = max(self.peak_bytes,self.live_bytes)
        offset, arrays = 0, []
        for template,size in zip(templates,sizes):
            arrays.append(DiskArray(bank,offset,template.shape,template.dtype))
            offset += size
        return tuple(arrays)

    def report(self):
        return dict(peak_logical_file_bytes=self.peak_bytes,live_logical_file_bytes=self.live_bytes,
                    free_reserve_bytes=self.free_reserve_bytes,
                    created_banks=self.created_banks,created_file_bytes=self.created_file_bytes,logical_read_bytes=self.read_bytes,
                    logical_written_bytes=self.written_bytes,max_single_read_bytes=self.max_read_bytes,
                    closed=self.closed,io_scope='Buffered file I/O. OS page cache and physical storage traffic are not measured. No whole-state file mapping.')

    def close(self):
        if self.closed:return
        # No recursive removal and no traversal of user-owned directory contents.
        for bank in list(self.banks.values()):bank.close()
        self.root.rmdir()
        self.closed = True

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()


class _Bank:
    def __init__(self,store,key,size):
        self.store,self.key,self.size = store,key,size
        self.closed = True
        self.path = store.root/f'bank-{key}.bin'
        self.file = self.path.open('x+b',buffering=0)
        try:
            self.file.truncate(size)
            # The file size the OS reports is the bank's on-disk footprint.
            self.file_bytes = os.fstat(self.file.fileno()).st_size
            if self.file_bytes != size:raise OSError('Field bank file size differs from the bank size.')
        except BaseException:
            self.file.close()
            self.path.unlink()
            raise
        self.closed = False

    def close(self):
        if self.closed:return
        self.file.close()
        self.path.unlink()
        self.store.live_bytes -= self.size
        self.store.banks.pop(self.key,None)
        self.closed = True

    def __del__(self):
        if not getattr(self,'closed',True):self.close()


class DiskArray:
    device = torch.device('cpu')

    def __init__(self,bank,offset,shape,dtype):
        self.bank,self.offset,self.shape,self.dtype = bank,offset,torch.Size(shape),dtype
        self.item = torch.empty((),dtype=dtype,device='cpu').element_size()
        self.row_bytes = math.prod(self.shape[1:])*self.item
        # Newly truncated banks are zero. Track complete row writes so the
        # first adjoint contribution need not read those zeros from disk.
        self.written_rows = set()

    def element_size(self):return self.item

    def _groups(self,indices):
        if self.bank.closed:raise RuntimeError('Field bank is closed.')
        if indices.device.type != 'cpu' or indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError('Disk slab indices must be a one-dimensional CPU int64 tensor.')
        values = indices.tolist()
        if any(i<0 or i>=self.shape[0] for i in values):raise IndexError('Field bank index out of bounds.')
        begin = 0
        while begin < len(values):
            end = begin+1
            while end < len(values) and values[end] == values[end-1]+1:end += 1
            yield begin,end,values[begin]
            begin = end

    def index_select(self,axis,indices):
        if axis != 0:raise ValueError('Field banks only support x-slab I/O.')
        groups = list(self._groups(indices))
        output = torch.empty((len(indices),*self.shape[1:]),dtype=self.dtype,device='cpu')
        if not groups:return output
        raw = memoryview(output.numpy()).cast('B')
        for begin,end,row in groups:
            piece = raw[begin*self.row_bytes:end*self.row_bytes]
            self.bank.file.seek(self.offset+row*self.row_bytes)
            done = 0
            while done < len(piece):
                count = self.bank.file.readinto(piece[done:])
                if not count:raise OSError('Truncated field bank read.')
                done += count
            self.bank.store.read_bytes += len(piece)
            self.bank.store.max_read_bytes = max(self.bank.store.max_read_bytes,len(piece))
        return output

    def __getitem__(self,selection):
        if not isinstance(selection,slice):raise TypeError('Field banks require an x slice.')
        start,stop,step = selection.indices(self.shape[0])
        return self.index_select(0,torch.arange(start,stop,step,dtype=torch.int64,device='cpu'))

    def index_copy_(self,axis,indices,value):
        if axis != 0:raise ValueError('Field banks only support x-slab I/O.')
        if value.device.type != 'cpu' or value.dtype != self.dtype or value.shape != (len(indices),*self.shape[1:]):
            raise ValueError('Field bank write shape, dtype or device mismatch.')
        groups = list(self._groups(indices))
        if not groups:return self
        # Conjugate/negative views carry logical values not reflected in their
        # underlying storage. Resolve those flags before exposing bytes to I/O.
        payload = value.detach().resolve_conj().resolve_neg().contiguous()
        raw = memoryview(payload.numpy()).cast('B')
        for begin,end,row in groups:
            piece = raw[begin*self.row_bytes:end*self.row_bytes]
            self.bank.file.seek(self.offset+row*self.row_bytes)
            done = 0
            while done < len(piece):
                count = self.bank.file.write(piece[done:])
                if not count:raise OSError('Incomplete field bank write.')
                done += count
            self.bank.store.written_bytes += len(piece)
            self.written_rows.update(range(row,row+end-begin))
        return self

    def index_add_(self,axis,indices,value):
        if axis != 0:raise ValueError('Field banks only support x-slab I/O.')
        if value.device.type != 'cpu' or value.dtype != self.dtype or value.shape != (len(indices),*self.shape[1:]):
            raise ValueError('Field bank reduction shape, dtype or device mismatch.')
        # Consecutive runs preserve duplicate-index accumulation order across
        # periodic halo copies. Never materialize the whole global field bank.
        for begin,end,row in self._groups(indices):
            offset=0
            while offset<end-begin:
                written=row+offset in self.written_rows
                stop=offset+1
                while stop<end-begin and ((row+stop in self.written_rows)==written):stop+=1
                selected=torch.arange(row+offset,row+stop,dtype=torch.int64,device='cpu')
                contribution=value[begin+offset:begin+stop]
                if written:
                    current=self.index_select(0,selected)
                    current.add_(contribution)
                else:current=contribution
                self.index_copy_(0,selected,current)
                offset=stop
        return self
