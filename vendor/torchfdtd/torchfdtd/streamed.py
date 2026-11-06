"""Experimental host/file-backed differentiable FDTD with bounded CUDA slabs."""
from dataclasses import dataclass, replace
from contextlib import contextmanager
import contextvars
from pathlib import Path
import math
import os
import time

import torch

from .boundaries import BoundaryDescription, material_shape
from .differentiable import DifferentiableResult, DifferentiableSimulation, _System, _split
from .spacetime import SlabBlockOperator
from .memory_profile import host_memory
from .memory_accounting import MemoryMeter
from .cuda_memory import cuda_budget_limit
from .plan import check_snapshot
from .reservation_registry import streamed_lease


@dataclass(frozen=True)
class StreamedAdjointOptions:
    slab_width: int = 32
    temporal_depth: int = 4
    checkpoints: int = 2
    gpu_budget_bytes: int = 1024**3
    host_budget_bytes: int = 8*1024**3
    device: str = 'cuda'
    cuda_binding: str = 'direct'
    reuse_tile_buffers: bool = True
    tile_transfers: str = 'sync'
    tile_buffers: int = 2
    local_checkpoints: int = 0
    state_storage: str = 'host'
    state_directory: str | Path | None = None
    disk_budget_bytes: int | None = None
    disk_free_reserve_bytes: int = 0
    restart_directory: str | Path | None = None
    restart_every_blocks: int = 1
    optimizer_moments: int = 0

    def __post_init__(self):
        if isinstance(self.disk_free_reserve_bytes,bool) or not isinstance(self.disk_free_reserve_bytes,int) or self.disk_free_reserve_bytes<0:
            raise ValueError('disk_free_reserve_bytes must be a nonnegative integer.')
        if self.state_storage not in ('host','disk'):raise ValueError('state_storage must be host or disk.')
        if self.disk_budget_bytes is not None and (isinstance(self.disk_budget_bytes,bool) or not isinstance(self.disk_budget_bytes,int) or self.disk_budget_bytes<1):
            raise ValueError('disk_budget_bytes must be a positive integer.')
        if self.state_directory is not None:
            if not isinstance(self.state_directory,(str,Path)):raise ValueError('state_directory must be a path or string.')
            object.__setattr__(self,'state_directory',str(self.state_directory))
        if self.state_storage == 'disk' and (not self.state_directory or self.disk_budget_bytes is None):
            raise ValueError('Disk field banks require a state_directory and explicit disk_budget_bytes.')
        for name in ('slab_width', 'temporal_depth', 'gpu_budget_bytes', 'host_budget_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if isinstance(self.checkpoints, bool) or not isinstance(self.checkpoints, int) or not 0 <= self.checkpoints <= 32:
            raise ValueError('checkpoints must be an integer between zero and 32.')
        if isinstance(self.local_checkpoints,bool) or not isinstance(self.local_checkpoints,int) or not 0 <= self.local_checkpoints <= 32:
            raise ValueError('local_checkpoints must be an integer between zero and 32.')
        if torch.device(self.device).type not in ('cpu', 'cuda'):
            raise ValueError('Streamed execution supports CPU or CUDA tiles.')
        if self.cuda_binding not in ('direct', 'dlpack'):raise ValueError('cuda_binding must be direct or dlpack.')
        if not isinstance(self.reuse_tile_buffers,bool):raise ValueError('reuse_tile_buffers must be boolean.')
        if self.tile_transfers not in ('sync','async'):raise ValueError('tile_transfers must be sync or async.')
        if isinstance(self.tile_buffers,bool) or not isinstance(self.tile_buffers,int) or not 1 <= self.tile_buffers <= 3:
            raise ValueError('tile_buffers must be between one and three.')
        if self.tile_transfers == 'async' and (not self.reuse_tile_buffers or torch.device(self.device).type != 'cuda'):
            raise ValueError('Asynchronous tiles require reusable CUDA buffers.')
        if self.restart_directory is not None:
            if not isinstance(self.restart_directory,(str,Path)):raise ValueError('restart_directory must be a path or string.')
            object.__setattr__(self,'restart_directory',str(self.restart_directory))
        if isinstance(self.restart_every_blocks,bool) or not isinstance(self.restart_every_blocks,int) or self.restart_every_blocks < 1:
            raise ValueError('restart_every_blocks must be a positive integer.')
        if isinstance(self.optimizer_moments,bool) or not isinstance(self.optimizer_moments,int) or not 0 <= self.optimizer_moments <= 8:
            raise ValueError('optimizer_moments must be an integer between zero and eight.')


def _journal_bytes(directory):
    """Bytes already held by restart records under the journal directory, if any.

    `*.tmp` entries are interrupted writes that the journal removes when it
    opens, so they are not part of the space the records will occupy.
    """
    root = Path(directory).expanduser()
    if not root.exists():return 0
    total = 0
    for child in root.iterdir():
        if child.name.endswith('.tmp'):continue
        total += sum(p.stat().st_size for p in ([child] if child.is_file() else child.rglob('*')) if p.is_file())
    return total


def _volume(directory):
    """Identity of the volume that files under directory will occupy.

    The drive of the resolved path on Windows, the device number of the
    nearest existing ancestor elsewhere; never a path-prefix comparison.
    """
    path = Path(os.path.realpath(Path(directory).expanduser()))
    if os.name == 'nt':return os.path.splitdrive(str(path))[0].lower()
    while not path.exists() and path.parent != path:path = path.parent
    return os.stat(path).st_dev


# Host bytes this process already holds for the run being admitted (the forward's
# retained field banks when the backward is admitted). The operating system
# reports them as unavailable, so they are added back before the budget check.
_HELD_HOST_BYTES = contextvars.ContextVar('torchfdtd_streamed_held_host_bytes', default=0)


@contextmanager
def held_host_bytes(count):
    """Admit the next reservation with `count` bytes of this run's own host banks counted as available."""
    token = _HELD_HOST_BYTES.set(int(count))
    try:
        yield
    finally:
        _HELD_HOST_BYTES.reset(token)


def _reservation(project, epsilon, options, spectral=None, *, pole_count=0, parameter_shapes=None):
    region = project.region
    n = math.prod(region.shape)
    boundary = BoundaryDescription(region)
    cpml = sum(math.prod(s['shape']) for segments in boundary.cpml.values() for s in segments)
    # Stored upper PMC faces/edges count their exact elements, not a padded volume.
    faces = sum(math.prod(shape) for blocks in boundary.pmc_blocks.values() for _, _, shape in blocks)
    electric_faces = sum(math.prod(shape) for _, _, shape in boundary.pmc_blocks['E'])
    pmc = bool(boundary.pmc_lower or boundary.pmc_upper)
    material_item = epsilon.element_size()
    # Fields, CPML memories and source/observation histories are complex for
    # Bloch propagation even though epsilon and its gradient remain real.
    item = material_item*(2 if region.complex_fields else 1)
    state = (6*n+cpml+faces+6*pole_count*n+2*pole_count*electric_faces)*item
    depth = min(options.temporal_depth, region.steps)
    width = min(options.slab_width, region.shape[0])+2*depth
    if 0 not in boundary.wrap:width = min(width, region.shape[0])
    stored = material_shape(region)
    tile_cells = (width+stored[0]-region.shape[0])*stored[1]*stored[2]
    if (6 if region.complex_fields else 3)*tile_cells >= 2**31:
        raise ValueError('A CUDA tile exceeds the supported integer index range.')
    monitors = sum(m.enabled for m in project.monitors) if spectral is None else len(spectral.components)
    terms = sum(len(s.polarization_components)*(2 if s.injection == 'oneway' else 1)
                for s in project.sources if s.enabled)
    history = region.steps*(2*monitors+terms)*item if spectral is None else region.steps*terms*item+spectral.reservation(depth)['spectral_reservation_bytes']
    # Conservative bounds cover restart banks, two adjoint banks, tile gather/
    # scatter copies, validation temporaries and recursive saved block states.
    # Geometry graphs and optimizer/objective allocations remain caller-owned.
    source_copies = math.ceil(width/region.shape[0])+1
    tile_history = depth*(monitors+terms*source_copies)*item
    local_slots = min(options.local_checkpoints,depth-1)
    # A complete local state contains six fields and at most twelve CPML arrays.
    # ADE includes P/Q and their adjoints, compact material packets and
    # explicit Torch transpose temporaries. Keep a conservative pole-dependent
    # bound for both CPU and native CUDA execution until large runs calibrate it.
    tile_workspace = (128+144*pole_count+(18+6*pole_count)*local_slots)*tile_cells*item
    buffers = options.tile_buffers if options.tile_transfers == 'async' else 1
    state_arrays = (2+sum(len(segments) for segments in boundary.cpml.values())
                    +sum(len(blocks) for blocks in boundary.pmc_blocks.values())
                    +((2+2*len(boundary.pmc_blocks['E'])) if pole_count else 0))
    initial_storage = state_arrays*item
    # The immutable all-zero host initial bank is represented by scalar views.
    # At most C saved block states, one current adjoint and two evolving
    # primal banks coexist during replay. Transpose instead holds a primal,
    # old adjoint and new adjoint, and forward holds two banks. Distinct live
    # bank identities are counted in benchmarks/streamed_bank_lifetime.py and
    # tests/test_streamed_bank_lifetime.py; they reach exactly this bound.
    # The initial all-zero bank is scalar-backed and charged separately.
    state_bank_capacity = options.checkpoints+3
    state_banks = state_bank_capacity*state
    disk = state_banks if options.state_storage == 'disk' else 0
    disk_io_workspace = (36+12*pole_count)*tile_cells*item if disk else 0
    parameter_count = sum(math.prod(s) for s in parameter_shapes) if parameter_shapes is not None else epsilon.numel()
    # Host ledger (benchmarks/streamed_host_ledger.py, docs/validation/
    # streamed_host_ledger_3060*.json): with a contiguous real CPU scalar epsilon,
    # real fields, synchronous reusable CUDA tiles, file-backed banks and point
    # observations, forward allocates no full-size host tensor and backward adds
    # exactly two, the accumulated gradient and one block contribution, to the
    # caller's input. Reserve four there: input, gradient, contribution and one
    # margin for accumulation temporaries. Every other path keeps eight. The
    # metadata estimate passes a meta tensor and must select the same figure.
    measured_scope = (pole_count == 0 and parameter_shapes is None and spectral is None and not pmc
                      and epsilon.device.type in ('cpu', 'meta') and epsilon.is_contiguous()
                      and epsilon.ndim == 3 and not epsilon.is_complex() and not region.complex_fields
                      and torch.device(options.device).type == 'cuda' and options.state_storage == 'disk'
                      and options.tile_transfers == 'sync' and options.reuse_tile_buffers)
    parameter_multiplier = 4 if measured_scope else 8
    dense_parameters = parameter_multiplier*parameter_count*material_item
    # The caller's CPU design tensor is the first of those copies. Optimizer
    # state (Adam keeps two moments) is parameter-sized host memory the caller
    # holds through the whole optimization; charge the declared count here.
    design = parameter_count*material_item
    optimizer_state = options.optimizer_moments*design
    host = (0 if disk else state_banks)+initial_storage+dense_parameters+optimizer_state+history+buffers*(tile_workspace+2*tile_history)+disk_io_workspace+16*monitors
    gpu = buffers*(tile_workspace+tile_history)
    cuda=torch.device(options.device).type=='cuda'
    observer_layout=32*monitors+8 if cuda and monitors and not region.complex_fields else 0
    observer_preparation=512*monitors+8 if observer_layout else 0
    # Each active slot owns its device index maps, plus an optional host/pinned
    # copy of the real fused observer packet. Charge all slots conservatively.
    gpu+=buffers*(16*monitors+observer_layout) if cuda else 0
    host+=buffers*(observer_preparation+observer_layout)
    available = host_memory()['available_bytes']
    if available is not None:
        available += _HELD_HOST_BYTES.get()
    host_limit = min(options.host_budget_bytes, int(available*.8)) if available is not None else options.host_budget_bytes
    if host > host_limit:
        raise ValueError(f'Streamed state and checkpoint reservation exceed the host budget: required={host} bytes, admissible={host_limit} bytes.')
    if disk:
        from .state_store import disk_free
        free_disk = disk_free(options.state_directory)
        disk_limit=min(options.disk_budget_bytes,int(free_disk*.8),free_disk-options.disk_free_reserve_bytes)
        if disk > disk_limit:
            raise ValueError(f'Field bank reservation exceeds the disk budget or available disk space: required={disk} bytes, admissible={max(0,disk_limit)} bytes, free={free_disk} bytes.')
    # A restart journal holds one published record while the next is written.
    # Forward records carry a full state and the whole signal history (steps x
    # monitors, field dtype); the completed forward record keeps that history
    # while backward records, each an adjoint state plus the partial material
    # gradient, rotate beside it. Arrays are raw bytes; the JSON metadata is
    # bounded per record array, per block start (the contract lists them) and
    # a fixed allowance for the contract's hashes, options and pointers.
    # Charge the largest coexisting set on the journal volume, together with
    # the bank reservation when they share that volume.
    signal_history = region.steps*monitors*item if spectral is None else 0
    material_gradient = parameter_count*material_item
    journal_metadata = 64*1024+16*(region.steps//options.temporal_depth+2)+2*(512+256*(state_arrays+1))
    restart = (max(2*(state+signal_history), signal_history+2*(state+material_gradient))
               +journal_metadata) if options.restart_directory else 0
    existing_journal = 0
    if restart:
        from .state_store import disk_free
        journal_free = disk_free(options.restart_directory)
        shared = bool(disk) and _volume(options.restart_directory) == _volume(options.state_directory)
        # Records already on disk from the interrupted run are part of the
        # journal's reservation, not additional space that must still be free.
        existing_journal = _journal_bytes(options.restart_directory)
        required = max(0, restart-existing_journal)+(disk if shared else 0)
        if required+options.disk_free_reserve_bytes > journal_free:
            raise ValueError(f'Restart journal reservation exceeds available disk space: required={required} bytes, free={journal_free} bytes, existing journal={existing_journal} bytes.')
    if torch.device(options.device).type == 'cuda':
        gpu_limit=cuda_budget_limit(options.device,gpu,options.gpu_budget_bytes)
        if gpu > gpu_limit:
            raise ValueError(f'Streamed tile workspace reservation exceeds the GPU budget: required={gpu} bytes, admissible={gpu_limit} bytes.')
    return dict(host_reservation_bytes=host, gpu_reservation_bytes=gpu,observation_index_bytes=16*monitors,
                observer_layout_bytes=buffers*observer_layout,observer_preparation_bytes=buffers*observer_preparation,
                state_bank_capacity=state_bank_capacity,
                dense_parameter_multiplier=parameter_multiplier,dense_parameter_reservation_bytes=dense_parameters,
                design_tensor_bytes=design,optimizer_moments=options.optimizer_moments,optimizer_state_reservation_bytes=optimizer_state,
                disk_reservation_bytes=disk,disk_io_workspace_bytes=disk_io_workspace,restart_reservation_bytes=restart,
                restart_journal_existing_bytes=existing_journal,
                restart_signal_history_bytes=signal_history if options.restart_directory else 0,
                restart_journal_metadata_bytes=journal_metadata if options.restart_directory else 0,
                host_initial_state_reservation_bytes=initial_storage,
                state_bytes=state, halo_cells_per_side=depth, max_extended_tile_cells=tile_cells, local_checkpoint_reservation_bytes=buffers*(18+6*pole_count)*local_slots*tile_cells*item,
                source_and_output_history_bytes=history, host_tile_reservation_bytes=buffers*(tile_workspace+2*tile_history))


@contextmanager
def _backing(options,report,phase):
    if options.state_storage == 'host':
        yield None
        return
    from .state_store import StateStore
    store = StateStore(options.state_directory,options.disk_budget_bytes,options.disk_free_reserve_bytes)
    try:yield store
    finally:
        try:store.close()
        finally:report[phase+'_backing_store'] = store.report()


def estimate_streamed_memory(project, options=None, *, diagonal=False, frequency_hz=None, window=None):
    """Check time-history or spectral admission without domain allocation.

    Uses current free resources and the same reservation as StreamedSimulation.
    Does not validate all physics or include caller-owned geometry/optimizer
    graphs or OS file cache. Supplied spectral settings are validated on CPU
    and copied, including an optional O(steps) window, but no fields are built.
    Admission is checked again at execution because resources may change.
    """
    if not isinstance(diagonal,bool):raise ValueError('diagonal must be boolean.')
    options=options or StreamedAdjointOptions()
    epsilon=torch.empty(material_shape(project.region,diagonal),dtype=getattr(torch,project.region.precision),device='meta')
    if window is not None and frequency_hz is None:
        raise ValueError('A spectral window requires frequency_hz.')
    spectral=None
    if frequency_hz is not None:
        from .adjoint_spectrum import SpectralObservation
        scalar=torch.empty((),dtype=epsilon.dtype,device='cpu')
        spectral=SpectralObservation(scalar,project.region,
            [m.component for m in project.monitors if m.enabled],frequency_hz,window)
    reservation=_reservation(project,epsilon,options,spectral)
    if spectral is not None:reservation.update(spectral.reservation(min(options.temporal_depth,project.region.steps)))
    return reservation


@dataclass(frozen=True)
class StreamedStoragePlan:
    options: StreamedAdjointOptions
    reservation: dict
    rejected: dict


def select_streamed_storage(project, options=None, *, diagonal=False, frequency_hz=None, window=None):
    """Prefer admitted DRAM banks, otherwise use explicitly configured disk.

    Capacity selection only, without domain allocation or performance probes.
    Tile/checkpoint policies and budgets are preserved. This does not select
    resident execution, change physical precision or bypass execution admission.
    """
    base=options or StreamedAdjointOptions()
    rejected={}
    for storage in ('host','disk'):
        if storage=='disk' and (base.state_directory is None or base.disk_budget_bytes is None):
            rejected[storage]='No explicit disk directory and budget configured.'
            continue
        candidate=replace(base,state_storage=storage)
        try:reservation=estimate_streamed_memory(project,candidate,diagonal=diagonal,frequency_hz=frequency_hz,window=window)
        except ValueError as exc:
            rejected[storage]=str(exc)
            continue
        return StreamedStoragePlan(candidate,reservation,rejected)
    raise ValueError('No streamed storage policy fits: '+str(rejected))


class _StreamedExecution:
    """Physics factories for the common bounded block replay schedule."""
    @property
    def operator_type(self):
        return SlabBlockOperator

    def host(self, project, value, spectral):
        return _System(project, value, prepare_updates=False,
                       observation_monitors=None if spectral is None else spectral.observers)

    def reservation(self, project, value, options, spectral):
        return _reservation(project, value, options, spectral)

    def operator(self, host, options, store):
        return self.operator_type(host, options.slab_width, options.device, cuda_binding=options.cuda_binding,
            reuse_buffers=options.reuse_tile_buffers, tile_transfers=options.tile_transfers,
            tile_buffers=options.tile_buffers, local_checkpoints=options.local_checkpoints,
            state_factory=store.new_state if store is not None else None)


def _journal(project, epsilon, options, spectral, starts):
    """Open the durable restart journal when configured, with a strict input contract."""
    if options.restart_directory is None:return None
    if spectral is not None:raise ValueError('Restart journals support time-history observations only.')
    from .streamed_restart import RestartJournal, journal_contract
    return RestartJournal(options.restart_directory, journal_contract(project, epsilon, options, starts),
                          options.restart_every_blocks)


class StreamCancelled(InterruptedError):
    """A streamed pass stopped at a block boundary because its cancel event was set."""


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _forward(ctx, epsilon, project, options, report, spectral, execution, cancel=None):
    """The forward phase of _Streamed, run under its reservation lease and memory meter."""
    host = execution.host(project, epsilon, spectral)
    report['host_initial_state_storage_bytes'] = sum(s.untyped_storage().nbytes() for s in host.state())
    report['host_inverse_permittivity_bytes'] = 0
    starts = list(range(0, project.region.steps, options.temporal_depth))+[project.region.steps]
    journal = _journal(project, epsilon, options, spectral, starts)
    ctx.journal = journal
    total = len(starts)-1
    index = 0
    try:
        with _backing(options,report,'forward') as store:
            operator = execution.operator(host, options, store)
            state = host.state()
            signals = torch.empty((project.region.steps, len(host.monitors)), dtype=host.field_dtype,
                                  device='cpu') if spectral is None else spectral.zeros()
            first = 0
            resumed = journal.load_forward(operator.new_state, signals) if journal is not None else None
            if resumed is not None:
                completed, saved_state, signals = resumed
                if completed is None:
                    first = total
                    report['forward_resumed_from_block'] = 'complete'
                else:
                    first, state = completed, saved_state
                    report['forward_resumed_from_block'] = completed
            recorded = first
            for index in range(first, total):
                if _cancelled(cancel):
                    # Stop at the block boundary. The journal keeps the state
                    # reached here unless it already holds this block.
                    if journal is not None:
                        if index > recorded and index > 0:
                            journal.record_forward(index, state, signals)
                        journal.cancel('forward', index)
                        report.update(journal.report())
                    raise StreamCancelled(f'Streamed forward cancelled at block {index} of {total}.')
                start = starts[index]
                depth = starts[index+1]-start
                state, values = operator.forward(epsilon, state, start, depth)
                if spectral is None:signals[start:start+depth].copy_(values)
                else:spectral.accumulate(signals, values, start)
                if journal is not None and journal.forward_due(index+1, total):
                    journal.record_forward(index+1, state, signals)
                    recorded = index+1
            if journal is not None and first < total:
                journal.record_signals(signals)
            if operator.workspace is not None:
                report['forward_workspace'] = operator.workspace_report()
            if journal is not None:report.update(journal.report())
            report['forward_bank_ledger'] = operator.bank_ledger()
    except StreamCancelled:
        raise
    except BaseException as exc:
        if journal is not None:
            journal.fail('forward', exc, block=index)
            report.update(journal.report())
        raise
    return signals


class _Streamed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, epsilon, project, options, report, spectral, execution, cancel=None):
        ctx.save_for_backward(epsilon)
        ctx.project, ctx.options, ctx.report = project.model_copy(deep=True), options, report
        ctx.spectral = spectral
        ctx.execution = execution
        ctx.cancel = cancel
        started = time.perf_counter()
        # The phase holds its reservation in the process registry while it runs.
        with streamed_lease(options, report), MemoryMeter(options.device) as meter:
            signals = _forward(ctx, epsilon, project, options, report, spectral, execution, cancel)
        report['forward_seconds'] = time.perf_counter()-started
        report['forward_memory'] = meter.record(report['forward_seconds'], store=report.get('forward_backing_store'),
                                                banks=report['forward_bank_ledger'])
        # No physical time history is retained by the autograd context.
        return signals

    @staticmethod
    def backward(ctx, signal_bar):
        if torch.is_grad_enabled():raise RuntimeError('Higher-order streamed derivatives are not implemented.')
        epsilon, = ctx.saved_tensors
        options, project, report = ctx.options, ctx.project, ctx.report
        # The forward's retained host banks belong to this run; the OS counts them as used.
        retained = (report.get('forward_bank_ledger') or {}).get('live_bytes', 0) + report.get('host_initial_state_storage_bytes', 0)
        with held_host_bytes(retained):
            reservation = ctx.execution.reservation(project, epsilon, options, ctx.spectral)
        started = time.perf_counter()
        with streamed_lease(options, reservation), MemoryMeter(options.device) as meter:
            gradient = _backward(ctx, signal_bar, epsilon, project, options, report)
        report['backward_seconds'] = time.perf_counter()-started
        report['backward_memory'] = meter.record(report['backward_seconds'], store=report.get('backward_backing_store'),
                                                 banks=report['backward_bank_ledger'])
        return gradient, None, None, None, None, None, None


def _backward(ctx, signal_bar, epsilon, project, options, report):
    """The backward phase of _Streamed, run under its reservation lease and memory meter."""
    cancel = getattr(ctx, 'cancel', None)
    host = ctx.execution.host(project, epsilon, ctx.spectral)
    journal = getattr(ctx, 'journal', None)
    with _backing(options,report,'backward') as store:
        operator = ctx.execution.operator(host, options, store)
        steps = project.region.steps
        starts = list(range(0, steps, options.temporal_depth))+[steps]
        total = len(starts)-1
        signal_bar_sha256 = None
        resumed = None
        if journal is not None:
            from .streamed_restart import sha256_tensor
            signal_bar_sha256 = sha256_tensor(signal_bar.detach().contiguous())
            # A journal written for another objective is refused here, before
            # any work; that refusal is not a failure of this run.
            resumed = journal.load_backward(signal_bar_sha256, operator.new_state, epsilon)
        if resumed is not None:
            limit, adjoint, gradient = resumed
            report['backward_resumed_from_block'] = limit
        else:
            limit = total
            gradient = torch.zeros_like(epsilon)
            adjoint = host.state()
        live = 0
        recorded = limit
        report.update(peak_block_checkpoints=0, replayed_blocks=0)

        def stop(position):
            # Cancellation at a transpose boundary: the adjoint and partial
            # gradient reached here are recorded unless the journal already
            # holds this position, then the journal is marked cancelled.
            if journal is not None:
                if position < recorded and position < total:
                    journal.record_backward(position, adjoint, gradient, signal_bar_sha256)
                journal.cancel('backward', position)
                report.update(journal.report())
            raise StreamCancelled(f'Streamed backward cancelled at block {position} of {total}.')

        def replay(state, begin, end):
            for block in range(begin, end):
                if _cancelled(cancel):stop(recorded)
                state, _ = operator.forward(epsilon, state, starts[block], starts[block+1]-starts[block])
                report['replayed_blocks'] += 1
            return state

        def reverse(begin, end, restart, slots):
            nonlocal adjoint, live, recorded
            while end > begin:
                if end-begin == 1 or slots == 0:
                    for block in reversed(range(begin, end)):
                        if _cancelled(cancel):stop(block+1)
                        state = replay(restart, begin, block)
                        start, stop_step = starts[block], starts[block+1]
                        adjoint, contribution = operator.transpose(epsilon, state, start, stop_step-start,
                                                                   adjoint, signal_bar[start:stop_step] if ctx.spectral is None else ctx.spectral.transpose(signal_bar,start,stop_step))
                        gradient.add_(contribution)
                        del state, contribution
                        if journal is not None and journal.backward_due(block, total):
                            journal.record_backward(block, adjoint, gradient, signal_bar_sha256)
                            recorded = block
                    return
                middle = begin+_split(end-begin, slots)
                saved = replay(restart, begin, middle)
                live += 1
                report['peak_block_checkpoints'] = max(live, report['peak_block_checkpoints'])
                try:reverse(middle, end, saved, slots-1)
                finally:
                    del saved
                    live -= 1
                end = middle

        try:reverse(0, limit, host.state(), options.checkpoints)
        except StreamCancelled:
            raise
        except BaseException as exc:
            if journal is not None:
                journal.fail('backward', exc, block=recorded)
                report.update(journal.report())
            raise
        finally:reverse=None  # Release recursive replay captures without cyclic GC.
        if journal is not None:
            journal.complete()
            report.update(journal.report())
        if operator.workspace is not None:
            report['backward_workspace'] = operator.workspace_report()
        report['backward_bank_ledger'] = operator.bank_ledger()
    return gradient


class StreamedSimulation(DifferentiableSimulation):
    """First-order epsilon-to-signal operation with host or file-backed state.

    Geometry tensors and outputs reside on CPU. CUDA receives one extended slab
    per reusable slot. Optional asynchronous transfers use bounded pinned pools.
    A manual policy or a policy from tune_streamed is explicit, and conservative
    admission does not itself establish a throughput advantage.
    """
    def __init__(self, project, options=None, *, cancel=None):
        super().__init__(project)
        self.streaming_options = options or StreamedAdjointOptions()
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancel must be an event with an is_set() method, such as threading.Event.')
        # Checked at block boundaries of the forward pass and at transpose and
        # replay boundaries of the backward pass; a set event raises
        # StreamCancelled after the journal, if any, has recorded the boundary.
        self.cancel = cancel

    def spectrum(self, epsilon, frequency_hz, *, window=None, block_size=32):
        """Accumulate a fixed-frequency DFT without retaining time signals."""
        from .adjoint_spectrum import SpectralObservation
        spectral = SpectralObservation(epsilon, self.project.region,
                                       [m.component for m in self.project.monitors if m.enabled], frequency_hz, window, block_size)
        return self._run(epsilon, spectral)

    def forward(self, epsilon):
        return self._run(epsilon, None)

    def _run(self, epsilon, spectral, *, execution=None):
        check_snapshot(self.project, self._plan_snapshot, type(self).__name__)
        region = self.project.region
        if not isinstance(epsilon, torch.Tensor) or epsilon.device.type != 'cpu':
            raise ValueError('Streamed epsilon must be a CPU tensor to avoid full-volume VRAM allocation.')
        if epsilon.dtype not in (torch.float32, torch.float64) or (epsilon.dtype == torch.float64) != (region.precision == 'float64'):
            raise ValueError('Epsilon dtype must match the real project precision.')
        if tuple(epsilon.shape) == region.shape+(3, 3):
            raise ValueError('Node permittivity tensors stream through StreamedTensorSimulation in torchfdtd.streamed_tensor.')
        if tuple(epsilon.shape) not in (material_shape(region), material_shape(region, True)):
            raise ValueError('Epsilon shape must match the project grid plus one stored row on every upper PMC/symmetric axis.')
        options = self.streaming_options
        reservation = _reservation(self.project, epsilon, options, spectral)
        if not bool(torch.isfinite(epsilon).all()) or bool((epsilon < 1).any()):
            raise ValueError('The CFL contract requires finite epsilon >= 1.')
        report = dict(experimental=True, spatial_streaming=True, full_time_autograd=False,
                      complex_fields=region.complex_fields,
                      higher_order=False, slab_width=options.slab_width,
                      temporal_depth=options.temporal_depth, checkpoint_capacity=options.checkpoints,
                      local_checkpoint_capacity=options.local_checkpoints,
                      state_storage=options.state_storage,
                      execution_device=options.device, tile_transfers=options.tile_transfers,
                      tile_buffers=options.tile_buffers if options.tile_transfers == 'async' else 1,
                      cuda_binding=options.cuda_binding,
                      reuse_tile_buffers=options.reuse_tile_buffers,
                      policy='manual', precision=str(epsilon.dtype), **reservation)
        report['observation_storage'] = 'time_history' if spectral is None else 'online_spectrum'
        if spectral is not None:report.update(spectral.reservation(min(options.temporal_depth,region.steps)))
        signals = _Streamed.apply(epsilon, self.project, options, report, spectral, execution or _StreamedExecution(),
                                  getattr(self, 'cancel', None))
        if spectral is not None:return spectral.result(signals,report)
        return DifferentiableResult(signals, region.time_step,
                                    tuple(m.component for m in self.project.monitors if m.enabled), report)
