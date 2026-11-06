"""Per-phase memory measurements of one streamed run, one instrument per field.

Every field of a record names the instrument that produced it and the way it
overlaps the others. Nothing is summed into a total: Torch allocated bytes are a
subset of Torch reserved bytes, both are a subset of the device memory the
process holds, pinned tile staging is counted in the working set and in the
commit charge, host field banks are counted in the working set and in the bank
ledger, and the OS file cache is system-wide. Sampled quantities can miss a
transient between two samples; the allocator peak counters are exact only when
the phase pushed them above the earlier process peak, which a caller obtains
by calling torch.cuda.reset_peak_memory_stats before the run.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path

import torch

INTERVAL_SECONDS = .02
SCOPE = ('Each field carries its own instrument and overlaps the others; the fields are never added into one total. '
         'Torch allocated is a subset of Torch reserved, both are a subset of the device memory of the process; pinned '
         'staging appears in the working set and the commit charge; host field banks appear in the working set and in '
         'the bank ledger; the OS file cache is system-wide. Caller graphs, optimizer state and the CUDA context are '
         'measured only where the process-level instruments include them.')


# ---------------------------------------------------------------------------
# Operating-system instruments (no third-party dependency)
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    import ctypes.wintypes as _w

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [('cb', _w.DWORD), ('PageFaultCount', _w.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]

    class _PerformanceInformation(ctypes.Structure):
        _fields_ = [('cb', _w.DWORD)]+[(name, ctypes.c_size_t) for name in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable', 'SystemCache',
            'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')]+[
            (name, _w.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi
    _kernel32.GetCurrentProcess.restype = _w.HANDLE
    _psapi.GetProcessMemoryInfo.argtypes = [_w.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), _w.DWORD]
    _psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(_PerformanceInformation), _w.DWORD]
    _kernel32.GetProcessIoCounters.argtypes = [_w.HANDLE, ctypes.POINTER(_IoCounters)]


def process_memory():
    """Working set, private and committed bytes of this process, from the OS.

    Windows: GetProcessMemoryInfo (WorkingSetSize, PeakWorkingSetSize, PrivateUsage,
    PagefileUsage, PeakPagefileUsage). Linux: /proc/self/status (VmRSS, VmHWM,
    RssAnon, VmSize; the commit charge of one process is not exposed, so the
    committed field is the virtual size). The peaks are process-lifetime peaks.
    """
    if sys.platform == 'win32':
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return dict(working_set_bytes=counters.WorkingSetSize, lifetime_peak_working_set_bytes=counters.PeakWorkingSetSize,
                    private_bytes=counters.PrivateUsage, committed_bytes=counters.PagefileUsage,
                    lifetime_peak_committed_bytes=counters.PeakPagefileUsage,
                    instrument='psapi.GetProcessMemoryInfo: WorkingSetSize, PeakWorkingSetSize, PrivateUsage, PagefileUsage, PeakPagefileUsage')
    status = Path('/proc/self/status')
    if not status.is_file():
        return None
    values = {}
    for line in status.read_text(encoding='utf-8', errors='replace').splitlines():
        name, _, rest = line.partition(':')
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0])*1024
    if 'VmRSS' not in values:
        return None
    return dict(working_set_bytes=values['VmRSS'], lifetime_peak_working_set_bytes=values.get('VmHWM'),
                private_bytes=values.get('RssAnon'), committed_bytes=values.get('VmSize'),
                lifetime_peak_committed_bytes=values.get('VmPeak'),
                instrument='/proc/self/status: VmRSS, VmHWM, RssAnon, VmSize (virtual size, not a commit charge), VmPeak')


def system_file_cache():
    """System-wide file cache bytes: psapi.GetPerformanceInfo SystemCache on Windows, /proc/meminfo Cached on Linux."""
    if sys.platform == 'win32':
        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not _psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        return dict(bytes=info.SystemCache*info.PageSize, instrument='psapi.GetPerformanceInfo: SystemCache pages times PageSize')
    meminfo = Path('/proc/meminfo')
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('Cached:'):
            return dict(bytes=int(line.split()[1])*1024, instrument='/proc/meminfo Cached')
    return None


def process_io():
    """Bytes this process has read and written through the OS, lifetime counters.

    Both platforms count the bytes passed through read and write calls, page-cache hits
    included: Windows ReadTransferCount and WriteTransferCount, Linux rchar and wchar. The
    Linux read_bytes and write_bytes count storage traffic instead (a page rewritten while
    dirty once, a cached read not at all), so they can fall below the bytes a file received.
    """
    if sys.platform == 'win32':
        counters = _IoCounters()
        if not _kernel32.GetProcessIoCounters(_kernel32.GetCurrentProcess(), ctypes.byref(counters)):
            return None
        return dict(read_bytes=counters.ReadTransferCount, write_bytes=counters.WriteTransferCount,
                    instrument='kernel32.GetProcessIoCounters: ReadTransferCount, WriteTransferCount (all handles, not only scratch files)')
    io = Path('/proc/self/io')
    if not io.is_file():
        return None
    values = {}
    try:
        for line in io.read_text(encoding='utf-8', errors='replace').splitlines():
            name, _, value = line.partition(':')
            values[name.strip()] = int(value)
    except OSError:
        return None
    if 'rchar' not in values or 'wchar' not in values:
        return None
    return dict(read_bytes=values['rchar'], write_bytes=values['wchar'],
                instrument='/proc/self/io rchar and wchar (bytes through read and write calls on every descriptor, not only scratch files)')


_smi_usable = None


def nvidia_smi_process_memory(device):
    """Per-process device memory from nvidia-smi, or None when the driver does not report it.

    Under WDDM on Windows nvidia-smi prints N/A for every process; after one
    such answer the tool is not queried again in this process.
    """
    global _smi_usable
    if _smi_usable is False:
        return None
    try:
        completed = subprocess.run(['nvidia-smi', f'--id={device.index or 0}', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        _smi_usable = False
        return None
    if completed.returncode != 0:
        _smi_usable = False
        return None
    rows = [line.partition(',') for line in completed.stdout.splitlines() if line.strip()]
    if rows and not any(used.strip().isdigit() for _, _, used in rows):
        _smi_usable = False
        return None
    _smi_usable = True
    for pid, _, used in rows:
        if pid.strip() == str(os.getpid()) and used.strip().isdigit():
            return int(used.strip())*1024**2
    return None


# ---------------------------------------------------------------------------
# The meter
# ---------------------------------------------------------------------------
class _Sampler(threading.Thread):
    def __init__(self, device, interval):
        super().__init__(daemon=True)
        self.device, self.interval = device, interval
        self.stop = threading.Event()
        self.samples = 0
        self.max_allocated = self.max_reserved = 0
        self.min_free = None
        self.max_working_set = self.max_private = self.max_committed = 0
        self.max_cache = None

    def sample(self):
        if self.device is not None:
            self.max_allocated = max(self.max_allocated, torch.cuda.memory_allocated(self.device))
            self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(self.device))
            free = torch.cuda.mem_get_info(self.device)[0]
            self.min_free = free if self.min_free is None else min(self.min_free, free)
        memory = process_memory()
        if memory is not None:
            self.max_working_set = max(self.max_working_set, memory['working_set_bytes'])
            self.max_private = max(self.max_private, memory['private_bytes'] or 0)
            self.max_committed = max(self.max_committed, memory['committed_bytes'] or 0)
        cache = system_file_cache()
        if cache is not None:
            self.max_cache = cache['bytes'] if self.max_cache is None else max(self.max_cache, cache['bytes'])
        self.samples += 1

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)


def _field(bytes_value, instrument, caveat, domain, **extra):
    return dict(bytes=bytes_value, domain=domain, instrument=instrument, caveat=caveat, **extra)


class MemoryMeter:
    """Measure one phase (forward or backward) of a streamed run.

    Use as a context manager around the phase and call ``record`` afterwards
    with the phase wall time, the closed scratch store's report and the host
    bank ledger of the slab operator. The record is JSON-compatible.
    """
    def __init__(self, device, *, interval_seconds=INTERVAL_SECONDS):
        chosen = torch.device(device)
        self.device = None
        if chosen.type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda', chosen.index if chosen.index is not None else torch.cuda.current_device())
        self.interval = interval_seconds
        self.sampler = None
        self.start = self.end = None

    def _snapshot(self):
        snapshot = dict(process=process_memory(), cache=system_file_cache(), io=process_io())
        if self.device is not None:
            snapshot.update(allocated=torch.cuda.memory_allocated(self.device),
                            allocated_peak=torch.cuda.max_memory_allocated(self.device),
                            reserved=torch.cuda.memory_reserved(self.device),
                            reserved_peak=torch.cuda.max_memory_reserved(self.device),
                            free=torch.cuda.mem_get_info(self.device)[0])
        return snapshot

    def __enter__(self):
        self.start = self._snapshot()
        self.sampler = _Sampler(self.device, self.interval)
        self.sampler.start()
        return self

    def __exit__(self, *exc):
        self.sampler.stop.set()
        self.sampler.join()
        if self.device is not None:
            torch.cuda.synchronize(self.device)
        self.sampler.sample()
        self.end = self._snapshot()
        return False

    def _peak(self, name, sampled_max):
        """Exact allocator peak when the phase set a new process peak, else the sampled lower bound."""
        start_peak, end_peak = self.start[name+'_peak'], self.end[name+'_peak']
        at_start = self.start[name]
        if end_peak > start_peak:
            value, exact = end_peak, True
            instrument = f'torch.cuda.max_memory_{name} after the phase (allocator peak counter; the phase set a new process peak, so it is exact)'
        elif at_start == start_peak:
            # The phase started at the process peak and never exceeded it.
            value, exact = start_peak, True
            instrument = f'torch.cuda.max_memory_{name} after the phase (allocator peak counter; the phase began at the process peak and stayed at or below it, so it is exact)'
        else:
            value, exact = max(sampled_max, at_start, self.end[name]), False
            instrument = (f'sampled torch.cuda.memory_{name} every {self.interval} s (lower bound: the allocator peak counter '
                          f'still holds an earlier process peak of {start_peak} bytes; call torch.cuda.reset_peak_memory_stats before the run for an exact value)')
        return value, at_start, exact, instrument, start_peak

    def record(self, seconds, *, store=None, banks=None):
        if self.end is None:
            raise RuntimeError('The meter has not finished a phase.')
        sampler = self.sampler
        record = dict(accounting_scope=SCOPE, total_memory_bytes=None,
                      total_memory_note='Not defined: the fields overlap and are never summed.',
                      sampler=dict(interval_seconds=self.interval, samples=sampler.samples), seconds=seconds)
        if self.device is None:
            unavailable = 'not applicable: CPU tiles, no CUDA device in this phase'
            for name in ('peak_torch_allocated_bytes', 'peak_torch_reserved_bytes', 'cuda_process_memory_bytes'):
                record[name] = _field(None, unavailable, 'no device memory is used by CPU tiles', 'device')
        else:
            allocated, at_start, exact, instrument, prior = self._peak('allocated', sampler.max_allocated)
            record['peak_torch_allocated_bytes'] = _field(allocated, instrument,
                'Torch caching-allocator bytes handed to tensors on this device; a subset of the reserved bytes and of the '
                'device memory of the process; excludes the CUDA context, CuPy kernel modules and other processes',
                'device', at_start_bytes=at_start, delta_bytes=allocated-at_start, exact=exact, prior_process_peak_bytes=prior)
            reserved, at_start, exact, instrument, prior = self._peak('reserved', sampler.max_reserved)
            record['peak_torch_reserved_bytes'] = _field(reserved, instrument,
                'Torch caching-allocator bytes held from the driver, including cached blocks not handed to any tensor; '
                'contains the allocated bytes, so the two are not added',
                'device', at_start_bytes=at_start, delta_bytes=reserved-at_start, exact=exact, prior_process_peak_bytes=prior)
            min_free = sampler.min_free if sampler.min_free is not None else self.end['free']
            min_free = min(min_free, self.end['free'])
            smi = nvidia_smi_process_memory(self.device)
            record['cuda_process_memory_bytes'] = _field(max(0, self.start['free']-min_free),
                f'torch.cuda.mem_get_info free-memory decrease during the phase, sampled every {self.interval} s and at the end '
                '(device-wide, from the driver)',
                'Device-wide: includes the Torch reserved bytes (so it is not added to them), the CUDA context, CuPy '
                'allocations and any other process on the same device; nvidia-smi per-process used_memory is reported '
                'separately when the driver exposes it (it does not under WDDM on Windows)',
                'device', device_free_at_start_bytes=self.start['free'], device_min_free_bytes=min_free,
                nvidia_smi_process_used_bytes=smi,
                nvidia_smi_instrument='nvidia-smi --query-compute-apps=pid,used_memory for this pid' if smi is not None
                else 'nvidia-smi --query-compute-apps reports N/A or is unavailable for this process')
        process_start, process_end = self.start['process'], self.end['process']
        if process_start is None or process_end is None:
            for name in ('peak_process_rss_bytes', 'peak_process_private_bytes', 'peak_process_committed_bytes'):
                record[name] = _field(None, 'unavailable on this platform', '', 'host')
        else:
            rss = max(sampler.max_working_set, process_start['working_set_bytes'], process_end['working_set_bytes'])
            record['peak_process_rss_bytes'] = _field(rss,
                f"{process_start['instrument']}, working set sampled every {self.interval} s and at the phase boundaries",
                'Host working set (RSS) of the whole process: interpreter, libraries, CUDA runtime host mappings, pinned '
                'staging, host field banks and caller tensors; host memory, not device memory; the OS lifetime peak is '
                'a process-wide high-water mark that may predate this phase',
                'host', at_start_bytes=process_start['working_set_bytes'], delta_bytes=rss-process_start['working_set_bytes'],
                os_lifetime_peak_bytes=process_end['lifetime_peak_working_set_bytes'])
            private = max(sampler.max_private, process_start['private_bytes'] or 0, process_end['private_bytes'] or 0)
            record['peak_process_private_bytes'] = _field(private,
                f"{process_start['instrument']}, private bytes sampled every {self.interval} s and at the phase boundaries",
                'Private (non-shared) host bytes of the process; overlaps the working set and the commit charge',
                'host', at_start_bytes=process_start['private_bytes'], delta_bytes=private-(process_start['private_bytes'] or 0))
            committed = max(sampler.max_committed, process_start['committed_bytes'] or 0, process_end['committed_bytes'] or 0)
            record['peak_process_committed_bytes'] = _field(committed,
                f"{process_start['instrument']}, committed bytes sampled every {self.interval} s and at the phase boundaries",
                'Commit charge of the process (Windows PagefileUsage; virtual size on Linux): includes pinned staging '
                'and pageable allocations whether resident or not; overlaps the working set',
                'host', at_start_bytes=process_start['committed_bytes'],
                delta_bytes=committed-(process_start['committed_bytes'] or 0),
                os_lifetime_peak_bytes=process_end['lifetime_peak_committed_bytes'])
        cache_start, cache_end = self.start['cache'], self.end['cache']
        if cache_start is None or cache_end is None:
            record['os_file_cache_bytes'] = _field(None, 'unavailable on this platform', '', 'system')
        else:
            peak = max(sampler.max_cache if sampler.max_cache is not None else cache_end['bytes'], cache_end['bytes'])
            record['os_file_cache_bytes'] = _field(cache_end['bytes']-cache_start['bytes'],
                f"{cache_start['instrument']}, at the phase boundaries and sampled every {self.interval} s",
                'System-wide file cache change during the phase: every process and file on the machine contributes, '
                'so it is not attributable to the scratch files alone and can be negative',
                'system', at_start_bytes=cache_start['bytes'], at_end_bytes=cache_end['bytes'],
                peak_delta_bytes=peak-cache_start['bytes'])
        io_start, io_end = self.start['io'], self.end['io']
        io_delta = None if io_start is None or io_end is None else dict(
            read_bytes=io_end['read_bytes']-io_start['read_bytes'], write_bytes=io_end['write_bytes']-io_start['write_bytes'],
            instrument=io_start['instrument'])
        written = store['logical_written_bytes'] if store is not None else 0
        read = store['logical_read_bytes'] if store is not None else 0
        store_instrument = ('StateStore logical byte counters of the scratch field banks (buffered file I/O)' if store is not None
                            else 'no scratch store in this phase (host banks)')
        record['scratch_disk_written_bytes'] = _field(written, store_instrument,
            'Logical bytes written to the scratch bank files; the OS may absorb them in its page cache, so they are not '
            'physical storage traffic; the process I/O counters cover every file handle of the process',
            'disk', process_io_delta_bytes=None if io_delta is None else io_delta['write_bytes'],
            process_io_instrument=None if io_delta is None else io_delta['instrument'])
        record['scratch_disk_read_bytes'] = _field(read, store_instrument,
            'Logical bytes read from the scratch bank files, warm-cache reads included; not physical storage traffic',
            'disk', process_io_delta_bytes=None if io_delta is None else io_delta['read_bytes'],
            process_io_instrument=None if io_delta is None else io_delta['instrument'])
        record['scratch_disk_bandwidth_bytes_per_second'] = dict(
            written=written/seconds if seconds else None, read=read/seconds if seconds else None,
            seconds=seconds, domain='disk',
            instrument='logical scratch bytes divided by the phase wall time, which includes compute and transfers',
            caveat='an effective rate over the whole phase, not a storage benchmark and not physical bandwidth')
        record['scratch_disk_peak_file_bytes'] = _field(store['peak_logical_file_bytes'] if store is not None else 0,
            store_instrument, 'Peak sum of live scratch bank file sizes (files are truncated to the bank size at creation)',
            'disk', created_banks=store['created_banks'] if store is not None else 0)
        ledger = banks or {}
        record['host_bank_bytes'] = _field(ledger.get('peak_bytes', 0),
            'SlabBlockOperator bank ledger: bytes of live host field banks, released through weak references',
            'Host tensors that also appear in the working set and the commit charge; zero for disk banks, whose '
            'sizes are the scratch file bytes', 'host', created_bytes=ledger.get('created_bytes', 0),
            banks_created=ledger.get('banks_created', 0), live_bytes_at_end=ledger.get('live_bytes', 0))
        return record
