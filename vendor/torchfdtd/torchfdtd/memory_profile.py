"""Bounded transfer measurements for future checkpoint/tile policy selection.

These are measured copy and file-I/O costs, not a prediction that out-of-core
FDTD will preserve resident throughput. No OS caches or system settings change.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import numpy as np
import torch


def host_memory():
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                      *[(name,ctypes.c_ulonglong) for name in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]]
        status=Status();status.length=ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Could not query host physical memory.')
        return {'total_bytes':status.total,'available_bytes':status.available}
    try:
        page=os.sysconf('SC_PAGE_SIZE')
        total=page*os.sysconf('SC_PHYS_PAGES')
        # On Linux SC_AVPHYS_PAGES resolves to MemFree, which excludes reclaimable page
        # cache. That under-reports usable memory by ~28x whenever the page cache is warm
        # (the steady state), and the memory admission refuses runs at ~1/28 of the real
        # budget. Prefer MemAvailable; keep the sysconf value as the fallback.
        available=None
        try:
            with open('/proc/meminfo') as handle:
                for line in handle:
                    if line.startswith('MemAvailable:'):
                        available=int(line.split()[1])*1024
                        break
        except (OSError,ValueError,IndexError):
            available=None
        if available is None:
            available=page*os.sysconf('SC_AVPHYS_PAGES')
        return {'total_bytes':total,'available_bytes':available}
    except (ValueError,OSError,AttributeError):
        return {'total_bytes':None,'available_bytes':None}


def profile_memory_transfers(*,device='cuda',copy_bytes=64*1024**2,repeats=5,
                             storage_directory=None,file_bytes=256*1024**2):
    """Measure pinned host copies and optional ordinary sequential file I/O.

    File reads are explicitly warm-cache measurements. Writes include fsync,
    but a small probe does not establish sustained physical NVMe bandwidth.
    The returned report is suitable input evidence, not an optimal tile plan.
    """
    if not isinstance(repeats,int) or isinstance(repeats,bool) or not 1<=repeats<=100:
        raise ValueError('repeats must be an integer from 1 to 100.')
    if not isinstance(copy_bytes,int) or isinstance(copy_bytes,bool) or not 4096<=copy_bytes<=1024**3:
        raise ValueError('copy_bytes must be between 4 KiB and 1 GiB.')
    if storage_directory is not None and (not isinstance(file_bytes,int) or isinstance(file_bytes,bool) or not 4096<=file_bytes<=4*1024**3):
        raise ValueError('file_bytes must be between 4 KiB and 4 GiB.')
    report={'host':host_memory(),'copy_bytes':copy_bytes,'repeats':repeats,
            'gpu':None,'storage':None,'scope':'Transfer probes only. No FDTD speedup, optimal policy or GDS support is inferred.'}
    if device is not None:
        chosen=torch.device(device)
        if chosen.type!='cuda' or not torch.cuda.is_available():raise ValueError('A CUDA device is required for GPU copy profiling.')
        with torch.cuda.device(chosen):
            free,total=torch.cuda.mem_get_info()
            if copy_bytes>free//4:raise ValueError('Copy probe exceeds one quarter of available GPU memory.')
            available=report['host']['available_bytes']
            if available is not None and copy_bytes>available//4:raise ValueError('Pinned probe exceeds one quarter of available host memory.')
            host=torch.empty(copy_bytes,dtype=torch.uint8,pin_memory=True)
            host.fill_(37)
            gpu=torch.empty(copy_bytes,dtype=torch.uint8,device=chosen)
            def measure(target,source):
                target.copy_(source,non_blocking=True);torch.cuda.synchronize(chosen)
                times=[]
                for _ in range(repeats):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();target.copy_(source,non_blocking=True);end.record();end.synchronize()
                    times.append(start.elapsed_time(end)/1000)
                return {'seconds':times,'median_seconds':statistics.median(times),
                        'effective_bytes_per_second':copy_bytes/statistics.median(times)}
            h2d=measure(gpu,host);d2h=measure(host,gpu)
            if not bool((host==37).all()):raise RuntimeError('Transfer probe data mismatch.')
            report['gpu']={'name':torch.cuda.get_device_name(chosen),'total_bytes':total,'available_bytes':free,
                           'h2d':h2d,'d2h':d2h,'mode':'pinned, sequential directions, CUDA-event timing',
                           'concurrent_copy_compute_measured':False}
    if storage_directory is not None:
        root=Path(storage_directory).resolve();root.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(root).free
        if file_bytes>free//4:raise ValueError('File probe exceeds one quarter of available storage.')
        descriptor,path=tempfile.mkstemp(prefix='torchfdtd-io-',suffix='.bin',dir=root)
        path=Path(path).resolve()
        if path.parent!=root:
            os.close(descriptor);raise RuntimeError('File probe escaped its selected directory.')
        chunk=np.random.default_rng(729).integers(0,256,min(file_bytes,4*1024**2),dtype=np.uint8).tobytes()
        try:
            started=time.perf_counter()
            with os.fdopen(descriptor,'wb',buffering=0) as output:
                remaining=file_bytes
                while remaining:
                    data=chunk[:min(remaining,len(chunk))]
                    offset=0
                    while offset<len(data):
                        written=output.write(data[offset:])
                        if not written:raise OSError('File probe write made no progress.')
                        offset+=written
                    remaining-=len(data)
                os.fsync(output.fileno())
            write_seconds=time.perf_counter()-started
            started=time.perf_counter();received=0
            with path.open('rb',buffering=0) as source:
                while data:=source.read(len(chunk)):
                    if data!=chunk[:len(data)]:raise RuntimeError('File probe data mismatch.')
                    received+=len(data)
            read_seconds=time.perf_counter()-started
            if received!=file_bytes:raise RuntimeError('File probe was truncated.')
            report['storage']={'file_bytes':file_bytes,'available_bytes':free,
                               'fsynced_write_seconds':write_seconds,'warm_read_seconds':read_seconds,
                               'fsynced_write_bytes_per_second':file_bytes/write_seconds,
                               'warm_read_bytes_per_second':file_bytes/read_seconds,
                               'physical_medium':'not identified','read_cache':'not flushed, warm',
                               'gds':False,'sustained_nvme_bandwidth_measured':False}
        finally:path.unlink(missing_ok=True)
    return report
