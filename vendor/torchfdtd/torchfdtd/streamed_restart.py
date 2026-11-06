"""Durable restart journal for streamed adjoints.

The journal records, at block boundaries, everything a later process needs to
continue: the forward state and partial signals, or the adjoint state, the
partial material gradient and the block that was completed. Resuming a backward
pass replays from the all-zero initial state up to the recorded block, so no
saved restart bank has to be preserved. A strict contract (project, epsilon,
options, runtime sources, signal adjoint) rejects a journal written for other
inputs. Records are written to a temporary directory, every array carries its
SHA-256, byte count, dtype and shape, the directory is renamed into place and
only then published through `latest-<kind>.json`. One earlier record of each
kind is kept as a fallback: a newest record that fails an integrity check is
rolled back to it, and the rollback is named in the report. One process owns a
journal at a time through `owner.json`; a lock left by a dead process is taken
over and the previous run is reported as `partial`. This is process-kill
consistency for our own files, not a filesystem quota, a mid-block checkpoint
or a guarantee against power loss (docs/STREAMED_RESTART.md).
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

import numpy as np
import torch

CHUNK_BYTES = 256 * 1024 ** 2
MARKER = 'torchfdtd-streamed-restart'
STATES = ('running', 'completed', 'cancelled', 'failed', 'partial')
RECORD_KINDS = ('forward', 'backward')


class RestartRecordDamaged(ValueError):
    """A record failed an integrity check; the reason names the file and the check."""
    def __init__(self, record, reason):
        super().__init__(f'Restart record {record} is damaged: {reason}')
        self.record, self.reason = record, reason


def sha256_tensor(value):
    """Hash a CPU tensor in bounded row chunks without copying the whole array."""
    digest = hashlib.sha256()
    if value.numel() == 0:
        return digest.hexdigest()
    rows = max(1, CHUNK_BYTES // max(1, value[0].numel() * value.element_size())) if value.ndim else 1
    if value.ndim == 0:
        digest.update(value.detach().contiguous().numpy().tobytes())
        return digest.hexdigest()
    for lo in range(0, value.shape[0], rows):
        chunk = value[lo:lo + rows].detach().contiguous()
        digest.update(memoryview(chunk.numpy()).cast('B'))
    return digest.hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_sources(package=None):
    """SHA-256 of every Python file under the package, keyed by relative POSIX path.

    Boundaries, waveforms, differentiable systems and the inline CUDA kernel
    sources all take part in the numerics, so the whole package is the runtime
    contract rather than the streamed modules alone.
    """
    root = Path(package if package is not None else Path(__file__).resolve().parent)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob('*.py'))}


# ---- durability primitives ---------------------------------------------------------
def _fsync_directory(path):
    """Make a rename in this directory durable on POSIX; Windows has no directory fsync."""
    if os.name == 'nt':
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH = 0x1, 0x8


def _move_file_ex(source, target, flags):
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel32.MoveFileExW(str(source), str(target), flags):
        raise ctypes.WinError(ctypes.get_last_error())


def _durable_replace(source, target, *, directory=False):
    """Rename into place so that a process kill leaves either the old or the new entry.

    POSIX: `os.replace` followed by an fsync of the parent directory, which
    commits the directory entry. Windows: `MoveFileExW` with
    MOVEFILE_WRITE_THROUGH, which returns only after the move has been
    written; a directory target cannot carry MOVEFILE_REPLACE_EXISTING and is
    removed by the caller first. Neither is a power-loss guarantee for the
    drive's own write cache.
    """
    source, target = Path(source), Path(target)
    if os.name == 'nt':
        flags = MOVEFILE_WRITE_THROUGH | (0 if directory else MOVEFILE_REPLACE_EXISTING)
        _move_file_ex(source, target, flags)
        return
    os.replace(source, target)
    _fsync_directory(target.parent)


def _pid_alive(pid):
    """Whether a process id is running on this host; None when it cannot be decided.

    Limitation: a recycled pid of an unrelated process looks alive, and a pid
    on another host cannot be checked at all.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            return True if error == 5 else False  # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- array records -------------------------------------------------------------------
def _size_reason(file, size, expected):
    if size < expected:
        return f'{file} is truncated: {size} of {expected} bytes'
    return f'{file} holds {size} bytes, {expected} declared'


def _write_array(path, array):
    """Write a CPU tensor or file bank to a raw file in row chunks, hash it and sync it."""
    shape = tuple(int(n) for n in array.shape)
    dtype = array.dtype
    item = torch.empty((), dtype=dtype).element_size()
    row_bytes = int(np.prod(shape[1:])) * item if len(shape) > 1 else item
    rows = max(1, CHUNK_BYTES // max(1, row_bytes))
    digest = hashlib.sha256()
    with open(path, 'wb', buffering=0) as handle:
        for lo in range(0, shape[0], rows):
            chunk = array[lo:lo + rows]
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().contiguous()
            view = memoryview(chunk.numpy()).cast('B')
            digest.update(view)
            done = 0
            while done < len(view):
                count = handle.write(view[done:])
                if not count:
                    raise OSError('Incomplete restart record write.')
                done += count
        handle.flush()
        os.fsync(handle.fileno())
    return dict(file=Path(path).name, shape=shape, dtype=str(dtype).replace('torch.', ''),
                bytes=int(np.prod(shape)) * item, sha256=digest.hexdigest())


def _read_array(path, description, target):
    """Fill a CPU tensor or file bank from a raw record in row chunks, verifying every byte.

    The byte count is checked against the file before reading, the dtype and
    shape against the target, and the SHA-256 while reading; each failure is
    a RestartRecordDamaged naming the file and the check.
    """
    path = Path(path)
    record = path.parent.name
    shape = tuple(description['shape'])
    dtype = getattr(torch, description['dtype'], None)
    if not isinstance(dtype, torch.dtype):
        raise RestartRecordDamaged(record, f'{path.name} declares an unknown dtype {description["dtype"]!r}')
    item = torch.empty((), dtype=dtype).element_size()
    expected = int(np.prod(shape)) * item
    if description.get('bytes') != expected:
        raise RestartRecordDamaged(record, f'{path.name} declares {description.get("bytes")} bytes for shape {list(shape)} {description["dtype"]} ({expected} expected)')
    if not path.exists():
        raise RestartRecordDamaged(record, f'{path.name} is missing')
    size = path.stat().st_size
    if size != expected:
        raise RestartRecordDamaged(record, _size_reason(path.name, size, expected))
    if tuple(target.shape) != shape or target.dtype != dtype:
        raise RestartRecordDamaged(record, f'{path.name} has shape {list(shape)} {description["dtype"]}, the current state needs {list(target.shape)} {str(target.dtype).replace("torch.", "")}')
    if not isinstance(description.get('sha256'), str) or len(description['sha256']) != 64:
        raise RestartRecordDamaged(record, f'{path.name} has no recorded checksum')
    row_bytes = int(np.prod(shape[1:])) * item if len(shape) > 1 else item
    rows = max(1, CHUNK_BYTES // max(1, row_bytes))
    digest = hashlib.sha256()
    with open(path, 'rb', buffering=0) as handle:
        for lo in range(0, shape[0], rows):
            hi = min(shape[0], lo + rows)
            buffer = torch.empty((hi - lo, *shape[1:]), dtype=dtype)
            view = memoryview(buffer.numpy()).cast('B')
            done = 0
            while done < len(view):
                count = handle.readinto(view[done:])
                if not count:
                    raise RestartRecordDamaged(record, f'{path.name} is truncated: read ended after {lo * row_bytes + done} of {expected} bytes')
                done += count
            digest.update(view)
            if isinstance(target, torch.Tensor):
                target[lo:hi].copy_(buffer)
            else:
                target.index_copy_(0, torch.arange(lo, hi, dtype=torch.int64), buffer)
    if digest.hexdigest() != description['sha256']:
        raise RestartRecordDamaged(record, f'checksum mismatch in {path.name}')
    return target


def _atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    data = (json.dumps(payload, indent=1, sort_keys=True) + '\n').encode('utf-8')
    with open(temp, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temp, path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _record_bytes(directory):
    return sum(f.stat().st_size for f in Path(directory).rglob('*') if f.is_file())


def _validate_record(root, name, kind):
    """Cheap integrity check of a published record: marker, kind, array descriptions and byte counts."""
    record = root / name
    if not record.is_dir():
        raise RestartRecordDamaged(name, 'record directory is missing')
    meta = record / 'meta.json'
    if not meta.exists():
        raise RestartRecordDamaged(name, 'meta.json is missing')
    try:
        payload = _read_json(meta)
    except (OSError, ValueError) as exc:
        raise RestartRecordDamaged(name, f'meta.json is unreadable ({exc})') from exc
    if payload.get('marker') != MARKER:
        raise ValueError('Restart record was not written by this journal.')
    if payload.get('kind') != kind:
        raise RestartRecordDamaged(name, f'meta.json declares kind {payload.get("kind")!r}, pointer expects {kind!r}')
    arrays = payload.get('arrays')
    if not isinstance(arrays, list) or not arrays:
        raise RestartRecordDamaged(name, 'meta.json lists no arrays')
    for description in arrays:
        file = description.get('file') if isinstance(description, dict) else None
        if not file or not isinstance(description.get('shape'), list) or not isinstance(description.get('dtype'), str):
            raise RestartRecordDamaged(name, 'an array description lacks file, shape or dtype')
        dtype = getattr(torch, description['dtype'], None)
        if not isinstance(dtype, torch.dtype):
            raise RestartRecordDamaged(name, f'{file} declares an unknown dtype {description["dtype"]!r}')
        expected = int(np.prod(description['shape'])) * torch.empty((), dtype=dtype).element_size()
        if description.get('bytes') != expected:
            raise RestartRecordDamaged(name, f'{file} declares {description.get("bytes")} bytes, its shape and dtype need {expected}')
        path = record / file
        if not path.exists():
            raise RestartRecordDamaged(name, f'{file} is missing')
        size = path.stat().st_size
        if size != expected:
            raise RestartRecordDamaged(name, _size_reason(file, size, expected))
        if not isinstance(description.get('sha256'), str) or len(description['sha256']) != 64:
            raise RestartRecordDamaged(name, f'{file} has no recorded checksum')
    payload['directory'] = record
    return payload

class RestartJournal:
    def __init__(self, directory, contract, every_blocks=1):
        self.root = Path(directory).expanduser().resolve()
        self.contract = contract
        self.every = every_blocks
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract_path = self.root / 'contract.json'
        self.written = dict(forward=0, backward=0)
        self.seconds = 0.
        self.rollbacks = []
        self.run_id = uuid.uuid4().hex
        self.pid, self.host = os.getpid(), socket.gethostname()
        self.state = None
        self.previous_state = None
        self._owned = False
        if self.contract_path.exists():
            existing = _read_json(self.contract_path)
            if existing != contract:
                changed = sorted(k for k in set(existing) | set(contract) if existing.get(k) != contract.get(k))
                for group in ('options', 'runtime_sha256'):
                    if group in changed:
                        old, new = existing.get(group) or {}, contract.get(group) or {}
                        changed += [f'{group}.{k}' for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
                raise ValueError('Restart journal contract mismatch. Use a new directory for changed inputs, options or runtime. Differences: '+', '.join(changed))
        else:
            _atomic_json(self.contract_path, contract)
        self.previous_state = self._acquire()
        # A crash leaves the record it was writing as `<name>.tmp` beside the
        # published records, never pointed at by a latest-*.json. Only such
        # entries directly under this journal's own directory are removed.
        for stale in self.root.glob('*.tmp'):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        self._set_state('running')

    # ---- ownership and terminal states --------------------------------------------
    @property
    def owner_path(self):
        return self.root / 'owner.json'

    @property
    def status_path(self):
        return self.root / 'status.json'

    def _owner_payload(self):
        return dict(marker=MARKER, run_id=self.run_id, pid=self.pid, host=self.host, started=time.time())

    def _acquire(self):
        """Take exclusive ownership; refuse a live owner, take over a dead one as `partial`."""
        previous = self._read_status()
        try:
            fd = os.open(self.owner_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                owner = _read_json(self.owner_path)
            except (OSError, ValueError) as exc:
                raise ValueError(f'Restart journal {self.root} has an unreadable owner.json ({exc}); remove it only if no run owns the journal.') from exc
            if owner.get('marker') != MARKER:
                raise ValueError(f'Restart journal {self.root} owner.json was not written by this journal.')
            if owner.get('host') != self.host:
                raise ValueError(f'Restart journal {self.root} is owned by run {owner.get("run_id")} on host {owner.get("host")!r}; liveness cannot be checked across hosts. Remove owner.json only if that run is dead.')
            alive = _pid_alive(owner.get('pid'))
            if alive is None or alive:
                raise ValueError(f'Restart journal {self.root} is owned by run {owner.get("run_id")} (pid {owner.get("pid")}), which is still running. One writer at a time owns a journal.')
            # The owner is dead. Its lock is replaced, and a run it left as
            # `running` ended without a terminal state: partial.
            _atomic_json(self.owner_path, self._owner_payload())
            if _read_json(self.owner_path).get('run_id') != self.run_id:
                raise ValueError(f'Restart journal {self.root} was taken over by another run while replacing a stale lock.')
            if previous in (None, 'running'):
                previous = 'partial'
        else:
            with os.fdopen(fd, 'wb') as handle:
                handle.write((json.dumps(self._owner_payload(), indent=1, sort_keys=True) + '\n').encode('utf-8'))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.root)
            if previous == 'running':
                previous = 'partial'
        self._owned = True
        return previous

    def _read_status(self):
        if not self.status_path.exists():
            return 'completed' if (self.root / 'complete.json').exists() else None
        try:
            payload = _read_json(self.status_path)
        except (OSError, ValueError):
            return None
        state = payload.get('state')
        return state if state in STATES else None

    def _set_state(self, state, **details):
        self.state = state
        _atomic_json(self.status_path, dict(marker=MARKER, state=state, run_id=self.run_id, pid=self.pid,
                                            host=self.host, updated=time.time(), **details))

    def _owns(self):
        try:
            return _read_json(self.owner_path).get('run_id') == self.run_id
        except (OSError, ValueError):
            return False

    def _ensure_owner(self):
        if self._owned and self._owns():
            return
        # Ownership was released by a terminal state of this run (a second
        # backward on the same result, say) or removed by hand: take it again.
        self._acquire()
        self._set_state('running')

    def _release(self):
        if self._owned and self._owns():
            try:
                self.owner_path.unlink()
            except OSError:
                pass
        self._owned = False

    def close(self):
        self._release()

    def __del__(self):
        try:
            self._release()
        except Exception:
            pass

    def cancel(self, phase, block):
        """Mark the run cancelled at a block boundary; the records stay readable."""
        self._set_state('cancelled', phase=phase, block=block)
        self._release()

    def fail(self, phase, exc, block=None):
        """Mark the run failed with the exception named; best effort, never raises."""
        try:
            self._set_state('failed', phase=phase, block=block, reason=f'{type(exc).__name__}: {exc}')
        except BaseException:
            pass
        self._release()

    # ---- records --------------------------------------------------------------------
    def _pointer(self, kind):
        # Forward and backward records are published independently: the completed
        # forward signals stay available while backward records rotate.
        return self.root / f'latest-{kind}.json'

    def _read_pointer(self, kind):
        pointer = self._pointer(kind)
        if not pointer.exists():
            return None
        try:
            latest = _read_json(pointer)
        except (OSError, ValueError) as exc:
            raise ValueError(f'Restart pointer {pointer.name} is unreadable ({exc}); the journal cannot decide its latest {kind} record.') from exc
        record = self.root / latest['directory']
        if record.parent != self.root or record.name.endswith('.tmp'):
            raise ValueError('Restart pointer names a record outside this journal directory.')
        return latest

    def _validate(self, name, kind):
        return _validate_record(self.root, name, kind)

    def _rollback(self, kind, name, previous, damaged):
        """Point the kind at its fallback record, remove the damaged one and name the reason."""
        fallback = None
        if previous and (self.root / previous).is_dir() and previous != name:
            try:
                block = self._validate(previous, kind)['block']
                fallback = previous
            except RestartRecordDamaged as second:
                self.rollbacks.append(dict(kind=kind, record=previous, reason=second.reason, resumed_from=None))
                shutil.rmtree(self.root / previous, ignore_errors=True)
        self.rollbacks.append(dict(kind=kind, record=name, reason=damaged.reason, resumed_from=fallback))
        if fallback is not None:
            _atomic_json(self._pointer(kind), dict(directory=fallback, kind=kind, block=block, previous=None))
        else:
            self._pointer(kind).unlink(missing_ok=True)
        shutil.rmtree(self.root / name, ignore_errors=True)
        return fallback

    def _latest(self, kind):
        """The latest record of a kind that passes the cheap checks, rolling a damaged newest one back."""
        latest = self._read_pointer(kind)
        while latest is not None:
            name, previous = latest['directory'], latest.get('previous')
            try:
                payload = self._validate(name, kind)
            except RestartRecordDamaged as damaged:
                if self._rollback(kind, name, previous, damaged) is None:
                    return None
                latest = self._read_pointer(kind)
                continue
            payload['previous'] = previous
            return payload
        return None

    def _load(self, kind, loader):
        """Run a loader on the latest record; a read-time integrity failure rolls back and retries."""
        latest = self._latest(kind)
        while latest is not None:
            try:
                return loader(latest)
            except RestartRecordDamaged as damaged:
                if self._rollback(kind, latest['directory'].name, latest.get('previous'), damaged) is None:
                    return None
                latest = self._latest(kind)
        return None

    def _drop_fallback(self, kind):
        latest = self._read_pointer(kind)
        if latest is None or not latest.get('previous'):
            return
        fallback = self.root / latest['previous']
        if fallback.is_dir() and fallback.name != latest['directory']:
            shutil.rmtree(fallback)

    def _publish(self, name, meta, writer):
        started = time.perf_counter()
        kind = meta['kind']
        self._ensure_owner()
        # One earlier record of each kind stays as the rollback target, so the
        # record two generations back goes before the new one is written and
        # at most two records of a kind coexist. The forward fallback goes
        # before the first backward record: the completed forward record has
        # been used by then, and the reservation charges either two forward
        # records or the completed one plus two backward records.
        self._drop_fallback(kind)
        if kind == 'backward':
            self._drop_fallback('forward')
        temp = self.root / (name + '.tmp')
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        try:
            arrays = writer(temp)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        meta = dict(meta, marker=MARKER, run_id=self.run_id, arrays=arrays, written=time.time())
        _atomic_json(temp / 'meta.json', meta)
        final = self.root / name
        if final.exists():
            shutil.rmtree(final)
        for attempt in range(5):
            try:
                _durable_replace(temp, final, directory=True)
                break
            except PermissionError:
                # Windows can briefly deny renaming a directory whose fresh
                # files an indexer still holds open. Retry before giving up.
                if attempt == 4:
                    raise
                time.sleep(.05*(attempt+1))
        current = self._read_pointer(kind)
        previous = current['directory'] if current is not None and current['directory'] != name and (self.root / current['directory']).is_dir() else None
        _atomic_json(self._pointer(kind), dict(directory=name, kind=kind, block=meta['block'], previous=previous))
        self.seconds += time.perf_counter() - started
        self.written[kind] += 1

    # ---- forward ------------------------------------------------------------------
    def forward_due(self, completed_blocks, total_blocks):
        return 0 < completed_blocks < total_blocks and completed_blocks % self.every == 0

    def record_forward(self, completed_blocks, state, signals):
        def writer(directory):
            arrays = [_write_array(directory / f'state-{i}.bin', value) for i, value in enumerate(state)]
            arrays.append(_write_array(directory / 'signals.bin', signals))
            return arrays
        self._publish(f'forward-{completed_blocks}', dict(kind='forward', block=completed_blocks, complete=False), writer)

    def record_signals(self, signals):
        def writer(directory):
            return [_write_array(directory / 'signals.bin', signals)]
        self._publish('forward-complete', dict(kind='forward', block=None, complete=True), writer)

    def load_forward(self, state_factory, signals_template):
        """Return (completed_blocks, state, signals) or (None, None, signals) when complete, else None."""
        def loader(latest):
            directory, arrays = latest['directory'], latest['arrays']
            name = directory.name
            if latest['complete']:
                if len(arrays) != 1:
                    raise RestartRecordDamaged(name, f'array count {len(arrays)} for a completed forward record (1 expected)')
                signals = _read_array(directory / arrays[-1]['file'], arrays[-1], torch.empty_like(signals_template))
                return None, None, signals
            state = state_factory()
            if len(arrays) != len(state) + 1:
                raise RestartRecordDamaged(name, f'array count {len(arrays)} ({len(state) + 1} expected: {len(state)} state arrays and the signals)')
            for value, description in zip(state, arrays[:-1]):
                _read_array(directory / description['file'], description, value)
            signals = _read_array(directory / arrays[-1]['file'], arrays[-1], torch.empty_like(signals_template))
            return latest['block'], state, signals
        return self._load('forward', loader)

    # ---- backward -----------------------------------------------------------------
    def backward_due(self, block, total_blocks):
        return block > 0 and (total_blocks - block) % self.every == 0

    def record_backward(self, block, adjoint, gradient, signal_bar_sha256):
        def writer(directory):
            arrays = [_write_array(directory / f'adjoint-{i}.bin', value) for i, value in enumerate(adjoint)]
            arrays.append(_write_array(directory / 'gradient.bin', gradient))
            return arrays
        self._publish(f'backward-{block}', dict(kind='backward', block=block, signal_bar_sha256=signal_bar_sha256), writer)

    def load_backward(self, signal_bar_sha256, adjoint_factory, gradient_template):
        """Return (block, adjoint, gradient) for a recorded backward, else None."""
        latest = self._latest('backward')
        if latest is None:
            return None
        if latest['signal_bar_sha256'] != signal_bar_sha256:
            raise ValueError('Restart journal was written for a different signal adjoint. Use a new directory. Differences: signal_bar_sha256')

        def loader(latest):
            directory, arrays = latest['directory'], latest['arrays']
            adjoint = adjoint_factory()
            if len(arrays) != len(adjoint) + 1:
                raise RestartRecordDamaged(directory.name, f'array count {len(arrays)} ({len(adjoint) + 1} expected: {len(adjoint)} adjoint arrays and the gradient)')
            for value, description in zip(adjoint, arrays[:-1]):
                _read_array(directory / description['file'], description, value)
            gradient = _read_array(directory / arrays[-1]['file'], arrays[-1], torch.empty_like(gradient_template))
            return latest['block'], adjoint, gradient
        return self._load('backward', loader)

    def complete(self):
        """Mark the run complete and remove every state record; the contract and marker stay."""
        _atomic_json(self.root / 'complete.json', dict(marker=MARKER, written=time.time(), records=self.written))
        for kind in RECORD_KINDS:
            self._pointer(kind).unlink(missing_ok=True)
        for child in self.root.iterdir():
            if child.is_dir() and child.name.split('-')[0] in RECORD_KINDS and not child.name.endswith('.tmp'):
                shutil.rmtree(child)
        self._set_state('completed')
        self._release()

    def report(self):
        return dict(restart_journal=str(self.root), restart_every_blocks=self.every,
                    journal_records_written=dict(self.written), journal_seconds=self.seconds,
                    restart_run_id=self.run_id, restart_state=self.state,
                    restart_previous_state=self.previous_state, restart_rollbacks=list(self.rollbacks))


def inspect_journal(directory):
    """Read a journal's state and records without owning it, so partial results stay readable.

    Returns the terminal or running state, the owner (with its pid liveness on
    this host), the pointers of both record kinds with each record's block,
    byte count and cheap integrity verdict, and the recorded rollbacks are
    left to the run report. Nothing is locked, removed or rolled back.
    """
    root = Path(directory).expanduser().resolve()
    result = dict(journal=str(root), contract=(root / 'contract.json').exists(), complete=(root / 'complete.json').exists(),
                  state=None, owner=None, records={})
    if (root / 'status.json').exists():
        try:
            status = _read_json(root / 'status.json')
            result['state'] = status.get('state')
            result['status'] = {k: status.get(k) for k in ('run_id', 'pid', 'host', 'phase', 'block', 'reason', 'updated')}
        except (OSError, ValueError) as exc:
            result['state'] = f'unreadable status.json ({exc})'
    elif result['complete']:
        result['state'] = 'completed'
    if (root / 'owner.json').exists():
        try:
            owner = _read_json(root / 'owner.json')
            same_host = owner.get('host') == socket.gethostname()
            result['owner'] = dict(run_id=owner.get('run_id'), pid=owner.get('pid'), host=owner.get('host'),
                                   alive=_pid_alive(owner.get('pid')) if same_host else None)
        except (OSError, ValueError) as exc:
            result['owner'] = dict(error=str(exc))
    if result['state'] == 'running' and result['owner'] is not None and result['owner'].get('alive') is False:
        result['state'] = 'partial'
    for kind in RECORD_KINDS:
        pointer = root / f'latest-{kind}.json'
        if not pointer.exists():
            continue
        try:
            latest = _read_json(pointer)
        except (OSError, ValueError) as exc:
            result['records'][kind] = dict(error=f'unreadable pointer ({exc})')
            continue
        entry = dict(record=latest.get('directory'), block=latest.get('block'), previous=latest.get('previous'))
        record = root / str(latest.get('directory'))
        entry['bytes'] = _record_bytes(record) if record.is_dir() else 0
        try:
            payload = _validate_record(root, str(latest.get('directory')), kind)
            entry.update(complete=payload.get('complete', False), arrays=len(payload['arrays']), valid=True)
        except (RestartRecordDamaged, ValueError, KeyError) as exc:
            entry.update(valid=False, reason=str(exc))
        result['records'][kind] = entry
    return result


def journal_contract(project, epsilon, options, starts):
    settings = asdict(options)
    settings.pop('restart_directory', None)
    settings.pop('restart_every_blocks', None)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()}
    # The scene enters through the restart-contract field selection of torchfdtd.identity
    # (docs/IDENTITY_CONDITIONS.md): the resolved plan's cache sections plus the kernel
    # scheme. Labels, placement, display settings, the revision and the content hash a
    # workbench save adds between the interruption and the resume are not hashed.
    from .identity import cache_payload, KERNEL_SCHEME_FIELDS
    from .plan import resolve_plan
    plan = resolve_plan(project)
    scene = json.dumps(dict(cache_payload(plan), kernel_scheme={k: plan.placement[k] for k in KERNEL_SCHEME_FIELDS}),
                       sort_keys=True, allow_nan=False)
    return dict(project_sha256=hashlib.sha256(scene.encode('utf-8')).hexdigest(),
                epsilon_sha256=sha256_tensor(epsilon), epsilon_shape=list(epsilon.shape), epsilon_dtype=str(epsilon.dtype),
                options=settings, starts=list(starts), runtime_sha256=runtime_sources(), torch=torch.__version__)
