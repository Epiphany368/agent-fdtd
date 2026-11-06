"""CUDA admission that can release unused allocator cache before denying work."""
import ctypes
import functools
import sys

import torch


class _NVMLMemory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


@functools.lru_cache(maxsize=None)
def _nvml():
    for name in (('nvml.dll',) if sys.platform == 'win32' else ('libnvidia-ml.so.1', 'libnvidia-ml.so')):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        if library.nvmlInit_v2() == 0:
            return library
    return None


def nvml_free_bytes(device):
    """Device-wide free memory from NVML (every process's allocations counted), or None without NVML."""
    library = _nvml()
    if library is None:
        return None
    handle = ctypes.c_void_p()
    try:
        uuid = ('GPU-' + str(torch.cuda.get_device_properties(device).uuid)).encode()
    except (RuntimeError, AssertionError, AttributeError):
        return None
    memory = _NVMLMemory()
    if library.nvmlDeviceGetHandleByUUID(uuid, ctypes.byref(handle)) or library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)):
        return None
    return int(memory.free)


def cuda_mem_info(device=None):
    """(free, total) bytes of a CUDA device for admission: the smaller of the CUDA runtime's and NVML's free memory.

    Under the Windows WDDM driver cudaMemGetInfo reports what this process could still allocate while the
    allocations of other processes can be paged out, so it can exceed the physically free memory (measured:
    10.98 GiB free reported to a second process while the first held 6 GiB of a 12 GiB RTX 3060); NVML counts
    every process. Without NVML the runtime value is returned.
    """
    device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
    free, total = torch.cuda.mem_get_info(device)
    nvml = nvml_free_bytes(device)
    return (free if nvml is None else min(free, nvml)), total


def cuda_budget_limit(device,required,budget=None):
    """Return min(explicit budget, 80% of freshly observed free CUDA memory).

    Repeated solves leave unused blocks in Torch's allocator. If those blocks
    could make an otherwise denied reservation fit, release unused cache once
    and query the device again. Never count reserved-minus-allocated bytes as
    guaranteed free memory. Live tensors and other processes are not evicted.
    Planning callers may therefore release unused cache without creating fields.
    This is not an exclusive device reservation or an allocation/OOM retry.
    """
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('CUDA capacity requires a CUDA device.')
    free,total=cuda_mem_info(device)
    cap=total if budget is None else budget
    limit=min(cap,int(free*.8))
    if required<=limit or required>min(cap,int(total*.8)):
        return limit
    unused=max(0,torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    if unused and required<=min(cap,int(min(total,free+unused)*.8)):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        free,_=cuda_mem_info(device)
        limit=min(cap,int(free*.8))
    return limit
