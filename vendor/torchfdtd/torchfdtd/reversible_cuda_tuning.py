"""Instance-local launch selection for the recorded CPML CUDA path.

Timing runs on copies of writable arguments, never on simulation state. The
cache contains only launch metadata, not tensors or CUDA allocation owners.
"""
from __future__ import annotations

import hashlib
import re
import statistics
import threading
from collections import OrderedDict

import torch

from .cuda_kernels import _compile, _direct_cuda_view


BLOCKS = (128, 256, 512, 1024)
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def validate_cells_per_thread(value):
    if value != 'auto' and (type(value) is not int or value not in (1, 2, 4)):
        raise ValueError('cells_per_thread must be 1, 2, 4 or auto.')


def multicell_source(source, name, count, cells_per_thread):
    """Unroll an unchanged per-cell body with coalesced, block-strided loads.

    The flattened Yee layout has z as its fastest spatial axis. For each
    unrolled iteration, neighboring threads still read neighboring cells:
    base + iteration * blockDim.x + threadIdx.x. A thread therefore handles
    several z-major entries, which may cross a row boundary. There is no
    requirement that Nz divide the number of entries per thread.

    Only loop-free, one-entry-per-thread kernels are eligible. Void returns
    skip the current entry rather than the remaining entries. Value returns
    inside PMC read lambdas are preserved. Launch-index arithmetic is 64-bit
    and checked before narrowing; the resident field-offset guard still
    bounds every original 32-bit component offset.
    """
    validate_cells_per_thread(cells_per_thread)
    if cells_per_thread == 'auto':
        raise ValueError('Resolve cells_per_thread before generating CUDA source.')
    if type(count) is not int or not 0 < count < 2**31:
        raise ValueError('CUDA launch indexing exceeds the signed 32-bit range.')
    if cells_per_thread == 1:
        return source
    declaration = re.search(r'\bvoid\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{', source)
    if declaration is None:
        raise ValueError('Expected one named CUDA kernel body.')
    start, end = declaration.end(), source.rfind('}')
    body = source[start:end]
    index = re.match(r'\s*const\s+(int|long long)\s+(\w+)\s*=\s*'
                     r'(?:\(long long\))?blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;', body)
    if index is None or re.search(r'\b(for|while|do|goto|break|continue)\b', body):
        raise ValueError('Multi-cell CUDA requires a loop-free per-entry kernel.')
    integer, variable = index.groups()
    body = re.sub(r'\breturn\s*;', 'continue;', body[index.end():])
    loop = (f'\nconst long long launch_base=(long long)blockIdx.x*blockDim.x*{cells_per_thread};\n'
            '#pragma unroll\n'
            f'for(int launch_entry=0;launch_entry<{cells_per_thread};++launch_entry){{\n'
            'const long long launch_index=launch_base+(long long)launch_entry*blockDim.x+threadIdx.x;\n'
            f'if(launch_index>={count}LL)continue;\n'
            f'const {integer} {variable}=({integer})launch_index;\n' + body + '\n}\n')
    return source[:start] + loop + source[end:]


class _MultiCellLaunch:
    """Keep callers' logical-entry launch sizes while reducing physical blocks."""
    def __init__(self, function, cells_per_thread):
        self.function = function
        self.cells_per_thread = cells_per_thread

    def __call__(self, grid, block, arguments, **kwargs):
        if len(grid) != 1 or len(block) != 1:
            raise ValueError('Multi-cell CUDA launches require a one-dimensional grid and block.')
        groups = (grid[0]+self.cells_per_thread-1)//self.cells_per_thread
        return self.function((groups,), block, arguments, **kwargs)


def select_launch(source, name, tensors, arrays, count, device, requested,
                  *, default=256, label, report, cells_per_thread=1):
    """Compile a fixed launch, or measure interleaved candidates on scratch.

    A failed optional specialization falls back to the original source and
    block size. Runtime launch errors propagate, as the CUDA context may no
    longer be usable. Allocation failures also propagate.
    """
    import cupy

    validate_cells_per_thread(cells_per_thread)
    source = re.sub(r'__launch_bounds__\(\d+\)\s*', '', source)
    capability = cupy.cuda.Device(device).compute_capability
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = (device, capability, digest, name, count, requested, cells_per_thread, default)
    record = dict(source_sha256=digest, requested=requested, cached=False,
                  cells_per_thread_requested=cells_per_thread, layout='strided')

    def compiled(block, cells):
        head = f'void {name}('
        if source.count(head) != 1:
            raise ValueError('Expected exactly one CUDA kernel declaration.')
        bounded = multicell_source(source, name, count, cells)
        bounded = bounded.replace(head, f'void __launch_bounds__({block}) {name}(')
        fn, module = _compile(bounded, device, capability, name)
        return (_MultiCellLaunch(fn, cells) if cells > 1 else fn), module

    stream = torch.cuda.current_stream(device)
    with _LOCK, cupy.cuda.Device(device), cupy.cuda.ExternalStream(stream.cuda_stream, device_id=device):
        if requested == 'auto' or cells_per_thread == 'auto':
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('CUDA launch tuning must precede graph capture.')
            if key in _CACHE:
                record.update(_CACHE[key], cached=True)
                _CACHE.move_to_end(key)
                block = record['block_size']
                cells = record['cells_per_thread']
                try:
                    fn, module = compiled(block, cells)
                except RuntimeError as exc:
                    if not str(exc).startswith('CUDA kernel compilation failed:'):
                        raise
                    # The compiled-module cache may evict an entry before the
                    # metadata cache. Its next compilation can then fail.
                    del _CACHE[key]
                    record.pop('median_ms', None)
                    record.update(cache_invalidated=True, fallback_reason=str(exc),
                                  compile_errors={str(block): str(exc)})
                    block = default
                    cells = 1
                    fn, module = _compile(source, device, capability, name)
                    record['block_size'] = block
                    record['cells_per_thread'] = cells
            else:
                head = source.split(f'{name}(', 1)[1].split(')', 1)[0]
                parameters = head.split(',')
                if len(parameters) != len(arrays) or len(tensors) != len(arrays):
                    raise ValueError('CUDA tuning argument metadata does not match the signature.')
                scratch = {}
                arguments = []
                for parameter, tensor, array in zip(parameters, tensors, arrays):
                    if '*' in parameter and not parameter.strip().startswith('const '):
                        # Preserve aliases if a kernel declares the same writable
                        # storage more than once. Coefficients stay read-only.
                        identity = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
                        if identity not in scratch:
                            scratch[identity] = tensor.clone()
                        arguments.append(_direct_cuda_view(cupy, scratch[identity]))
                    else:
                        arguments.append(array)
                timings, candidates, errors = {}, {}, {}
                blocks = BLOCKS if requested == 'auto' else (default if requested is None else requested,)
                cell_counts = (1, 2, 4) if cells_per_thread == 'auto' else (cells_per_thread,)
                for block, cells in ((b, c) for b in blocks for c in cell_counts):
                    try:
                        candidates[block, cells] = compiled(block, cells)
                    except RuntimeError as exc:
                        if not str(exc).startswith('CUDA kernel compilation failed:'):
                            raise
                        errors[f'{block}x{cells}'] = str(exc)
                for (block, cells), (fn, _) in candidates.items():
                    fn(((count + block - 1) // block,), (block,), tuple(arguments))
                    timings[block, cells] = []
                for sweep in range(5):
                    order = list(candidates) if sweep % 2 == 0 else list(reversed(candidates))
                    for block, cells in order:
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        candidates[block, cells][0](((count + block - 1) // block,), (block,), tuple(arguments))
                        stop.record(stream)
                        stop.synchronize()
                        timings[block, cells].append(start.elapsed_time(stop))
                del arguments, scratch
                if timings:
                    medians = {pair: statistics.median(values) for pair, values in timings.items()}
                    block, cells = min(medians, key=medians.get)
                    fn, module = candidates[block, cells]
                    # Keep the original block-only report keys for callers
                    # that did not request the new multi-cell setting.
                    reported_medians = ({b: value for (b, _), value in medians.items()}
                        if cells_per_thread == 1 else
                        {f'{b}x{c}': value for (b, c), value in medians.items()})
                    record.update(block_size=block, cells_per_thread=cells,
                        median_ms=reported_medians, compile_errors=errors)
                    _CACHE[key] = dict(record)
                    while len(_CACHE) > 128:
                        _CACHE.popitem(last=False)
                else:
                    block = default
                    cells = 1
                    fn, module = _compile(source, device, capability, name)
                    record.update(block_size=block, cells_per_thread=cells,
                                  fallback_reason='No launch specialization compiled.', compile_errors=errors)
        else:
            block = default if requested is None else requested
            cells = cells_per_thread
            try:
                fn, module = compiled(block, cells)
            except RuntimeError as exc:
                if not str(exc).startswith('CUDA kernel compilation failed:'):
                    raise
                block = default
                cells = 1
                fn, module = _compile(source, device, capability, name)
                record['fallback_reason'] = str(exc)
            record['block_size'] = block
            record['cells_per_thread'] = cells
    if report is not None:
        report.setdefault('cuda_launches', {})[label] = record
        if record.get('fallback_reason'):
            report.setdefault('fallback_reason', {})['block_size:'+label] = record['fallback_reason']
    return fn, module, block


def tune_yee(kernel, requested, report, *, cells_per_thread=1):
    """Install instance-local launches without replacing classes or functions."""
    kernel.tuned_blocks = {}
    for forward in (False, True):
        _, arrays, _ = kernel.launches[forward]
        source, tensors = kernel._source(forward)
        count = kernel.launch_count(kernel.grid, forward)
        fn, module, block = select_launch(source, 'yee_update', tensors, arrays,
            count, kernel.device, requested, label='yee_H' if forward else 'yee_E', report=report,
            cells_per_thread=cells_per_thread)
        kernel.launches[forward] = fn, arrays, module
        kernel.tuned_blocks[forward] = block


def tune_adjoint(kernel, requested, report, *, cells_per_thread=1):
    kernel.tuned_blocks = {}
    for (forward, phase), (_, arrays, _) in kernel.launches.items():
        source, tensors = kernel.source(forward, phase)
        fn, module, block = select_launch(source, 'adjoint_update', tensors, arrays,
            kernel.count, kernel.device, requested,
            label=f'adjoint_{"H" if forward else "E"}_{phase}', report=report,
            cells_per_thread=cells_per_thread)
        kernel.launches[forward, phase] = fn, arrays, module
        kernel.tuned_blocks[forward, phase] = block
