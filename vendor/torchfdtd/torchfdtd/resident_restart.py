"""Crash-proof persistent restart journal for resident in-memory FDTD simulations (LEG 45).

Zero-Trust Rule: Simulations can be checkpointed and resumed at block boundaries
with 100% bit-identity (0.0 relative error). The journal maintains dual ping-pong
slots ('slot_a' and 'slot_b') and updates pointer 'latest-forward.json' atomically,
with rollback capability if a slot write is damaged or incomplete.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Dict, Optional, Sequence, Tuple
import uuid

import numpy as np
import torch

from .streamed_restart import (
    RestartRecordDamaged,
    _atomic_json,
    _durable_replace,
    _fsync_directory,
    _pid_alive,
    _read_array,
    _read_json,
    _size_reason,
    _write_array,
    runtime_sources,
    sha256_file,
    sha256_tensor,
)
from .telemetry import hash_array

MARKER = 'torchfdtd-resident-restart'
SLOT_A = 'slot_a'
SLOT_B = 'slot_b'
STATES = ('running', 'completed', 'cancelled', 'failed', 'partial')


def build_resident_contract(
    system,
    epsilon: torch.Tensor,
    spectral=None,
    options=None,
    project_canonical_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an immutable contract dictionary for resident restart validation."""
    r = system.region
    eps_cpu = epsilon.detach().contiguous().cpu()
    contract = {
        'marker': MARKER,
        'project_canonical_sha256': str(project_canonical_sha256) if project_canonical_sha256 else None,
        'epsilon_sha256': hash_array(eps_cpu),
        'epsilon_shape': [int(x) for x in epsilon.shape],
        'epsilon_dtype': str(epsilon.dtype).replace('torch.', ''),
        'dt': float(r.time_step),
        'steps': int(r.steps),
        'grid_shape': [int(x) for x in r.shape],
        'block_size': int(spectral.block_size) if spectral is not None else 32,
        'runtime_sha256': runtime_sources(),
    }
    return contract


class ResidentRestartJournal:
    """Manages atomic ping-pong checkpointing and bit-identical resume for resident _FDTD."""

    SLOT_A = SLOT_A
    SLOT_B = SLOT_B

    def __init__(
        self,
        directory: Path | str,
        contract: Optional[Dict[str, Any]] = None,
        *,
        every_blocks: int = 64,
        every_seconds: Optional[float] = None,
    ):
        self.root = Path(directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.every_blocks = every_blocks
        self.every_seconds = every_seconds

        self.contract_path = self.root / 'contract.json'
        self.owner_path = self.root / 'owner.json'
        self.status_path = self.root / 'status.json'
        self.pointer_path = self.root / 'latest-forward.json'
        self.complete_path = self.root / 'complete.json'

        self.run_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.rollbacks = []
        self._last_save_time = time.time()
        self._owned = False

        self._validate_or_init_contract()
        self._acquire_lock()
        self._clean_stale_tmp()
        self._set_state('running')

    def _validate_or_init_contract(self) -> None:
        if self.contract_path.exists():
            existing = _read_json(self.contract_path)
            if self.contract is not None:
                # Compare fields that define physical/computational invariants
                diffs = []
                check_keys = {
                    'project_canonical_sha256',
                    'epsilon_sha256',
                    'epsilon_shape',
                    'epsilon_dtype',
                    'dt',
                    'steps',
                    'grid_shape',
                    'block_size',
                }
                for k in sorted(check_keys):
                    e_val = existing.get(k)
                    c_val = self.contract.get(k)
                    if e_val is not None and c_val is not None:
                        if isinstance(e_val, float) and isinstance(c_val, (int, float)):
                            scale = max(abs(e_val), abs(c_val), 1e-30)
                            if abs(e_val - c_val) / scale > 1e-9:
                                diffs.append(f"{k}: saved={e_val!r} != current={c_val!r}")
                        elif e_val != c_val:
                            diffs.append(f"{k}: saved={e_val!r} != current={c_val!r}")
                if diffs:
                    raise ValueError(
                        f"Restart journal contract mismatch. Use a new directory for changed inputs. Differences: {', '.join(diffs)}"
                    )
            else:
                self.contract = existing
        elif self.contract is not None:
            _atomic_json(self.contract_path, self.contract)

    def _acquire_lock(self) -> None:
        """Acquire single-writer exclusive ownership via owner.json."""
        if self.owner_path.exists():
            try:
                owner = _read_json(self.owner_path)
            except Exception:
                owner = None
            if owner and isinstance(owner, dict):
                pid = owner.get('pid')
                if pid and _pid_alive(pid) and pid != self.pid:
                    raise ValueError(
                        f"Restart journal {self.root} is owned by run {owner.get('run_id')} (pid {pid}), which is still running."
                    )
        payload = {
            'marker': MARKER,
            'run_id': self.run_id,
            'pid': self.pid,
            'host': self.host,
            'started': time.time(),
        }
        _atomic_json(self.owner_path, payload)
        self._owned = True

    def _release_lock(self) -> None:
        if self._owned:
            self.owner_path.unlink(missing_ok=True)
            self._owned = False

    def _clean_stale_tmp(self) -> None:
        for stale in self.root.glob('*.tmp'):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)

    def _set_state(self, state: str, **details) -> None:
        payload = {
            'marker': MARKER,
            'state': state,
            'run_id': self.run_id,
            'pid': self.pid,
            'host': self.host,
            'updated': time.time(),
            **details,
        }
        _atomic_json(self.status_path, payload)

    def _read_pointer(self) -> Optional[Dict[str, Any]]:
        if not self.pointer_path.exists():
            return None
        try:
            return _read_json(self.pointer_path)
        except Exception:
            return None

    def has_checkpoint(self) -> bool:
        if self.complete_path.exists():
            return False
        ptr = self._read_pointer()
        if not ptr or 'directory' not in ptr:
            return False
        return (self.root / ptr['directory']).is_dir()

    def should_commit(self, completed_blocks: int, current_time: float) -> bool:
        """Check if checkpoint is due at the current completed block boundary."""
        if completed_blocks <= 0:
            return False
        if self.every_blocks and (completed_blocks % self.every_blocks == 0):
            return True
        if self.every_seconds is not None and (current_time - self._last_save_time >= self.every_seconds):
            return True
        return False

    def save_checkpoint(
        self,
        step: int,
        block: int,
        system_state: Sequence[torch.Tensor],
        signals: torch.Tensor,
        peak_energy: float = 0.0,
    ) -> Dict[str, Any]:
        """Atomically persist state and signals into alternating ping-pong slot."""
        ptr = self._read_pointer()
        current_slot = ptr.get('directory') if ptr else None
        next_slot = self.SLOT_B if current_slot == self.SLOT_A else self.SLOT_A
        previous_slot = current_slot

        if signals.is_cuda:
            torch.cuda.synchronize(signals.device)

        tmp_dir = self.root / f"{next_slot}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        arrays = []
        for i, tensor in enumerate(system_state):
            cpu_t = tensor.detach().cpu().contiguous()
            arrays.append(_write_array(tmp_dir / f"state-{i}.bin", cpu_t))

        cpu_sig = signals.detach().cpu().contiguous()
        arrays.append(_write_array(tmp_dir / "signals.bin", cpu_sig))

        meta = {
            'marker': MARKER,
            'kind': 'forward',
            'step': int(step),
            'block': int(block),
            'peak_energy': float(peak_energy),
            'arrays': arrays,
            'written': time.time(),
        }
        _atomic_json(tmp_dir / 'meta.json', meta)

        target_dir = self.root / next_slot
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        _durable_replace(tmp_dir, target_dir, directory=True)

        pointer_payload = {
            'directory': next_slot,
            'step': int(step),
            'block': int(block),
            'peak_energy': float(peak_energy),
            'previous': previous_slot,
            'written': time.time(),
        }
        _atomic_json(self.pointer_path, pointer_payload)
        self._last_save_time = time.time()
        return pointer_payload

    def _validate_slot(self, slot_dir: Path) -> Dict[str, Any]:
        if not slot_dir.is_dir():
            raise RestartRecordDamaged(slot_dir.name, 'slot directory is missing')
        meta_file = slot_dir / 'meta.json'
        if not meta_file.is_file():
            raise RestartRecordDamaged(slot_dir.name, 'meta.json is missing')
        try:
            meta = _read_json(meta_file)
        except Exception as e:
            raise RestartRecordDamaged(slot_dir.name, f'unreadable meta.json: {e}')
        if meta.get('marker') != MARKER:
            raise RestartRecordDamaged(slot_dir.name, f'invalid marker: {meta.get("marker")}')
        arrays = meta.get('arrays')
        if not isinstance(arrays, list) or not arrays:
            raise RestartRecordDamaged(slot_dir.name, 'no array descriptions')

        for desc in arrays:
            file_name = desc.get('file')
            expected_bytes = desc.get('bytes')
            expected_sha = desc.get('sha256')
            if not file_name or expected_bytes is None or not expected_sha:
                raise RestartRecordDamaged(slot_dir.name, f'corrupted descriptor {desc}')
            arr_path = slot_dir / file_name
            if not arr_path.is_file():
                raise RestartRecordDamaged(slot_dir.name, f'missing file {file_name}')
            actual_bytes = arr_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RestartRecordDamaged(slot_dir.name, _size_reason(file_name, actual_bytes, expected_bytes))
        return meta

    def _load_arrays(
        self,
        slot_dir: Path,
        array_descriptions: list,
        system_state: Sequence[torch.Tensor],
        signals: torch.Tensor,
    ) -> None:
        state_map = {desc['file']: desc for desc in array_descriptions if desc['file'].startswith('state-')}
        sig_desc = next((desc for desc in array_descriptions if desc['file'] == 'signals.bin'), None)
        if sig_desc is None:
            raise RestartRecordDamaged(slot_dir.name, 'signals.bin missing from descriptions')

        if len(state_map) != len(system_state):
            raise RestartRecordDamaged(
                slot_dir.name,
                f"state tensor count mismatch: journal has {len(state_map)} vs system {len(system_state)}"
            )

        # In-place restore of state tensors
        for i, target in enumerate(system_state):
            fname = f"state-{i}.bin"
            desc = state_map.get(fname)
            if not desc:
                raise RestartRecordDamaged(slot_dir.name, f"{fname} not found in journal")
            cpu_buf = torch.empty(tuple(desc['shape']), dtype=getattr(torch, desc['dtype']))
            _read_array(slot_dir / fname, desc, cpu_buf)
            target.copy_(cpu_buf)

        # In-place restore of signals
        cpu_sig = torch.empty(tuple(sig_desc['shape']), dtype=getattr(torch, sig_desc['dtype']))
        _read_array(slot_dir / 'signals.bin', sig_desc, cpu_sig)
        signals.copy_(cpu_sig)

        if signals.is_cuda:
            torch.cuda.synchronize(signals.device)

    def load_latest(
        self,
        system_state: Optional[Sequence[torch.Tensor]] = None,
        signals: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int, float, bool]:
        """
        Load latest valid checkpoint with rollback fallback if damaged.
        Returns: (step, block, peak_energy, is_complete)
        """
        if self.complete_path.exists():
            return 0, 0, 0.0, True

        latest = self._read_pointer()
        while latest is not None:
            slot_name = latest.get('directory')
            fallback = latest.get('previous')
            if not slot_name:
                break
            slot_dir = self.root / slot_name
            try:
                meta = self._validate_slot(slot_dir)
                if system_state is not None and signals is not None:
                    self._load_arrays(slot_dir, meta['arrays'], system_state, signals)
                return int(meta['step']), int(meta['block']), float(meta.get('peak_energy', 0.0)), False
            except RestartRecordDamaged as exc:
                self.rollbacks.append({'slot': slot_name, 'reason': exc.reason, 'fallback': fallback})
                shutil.rmtree(slot_dir, ignore_errors=True)
                if fallback and (self.root / fallback).is_dir():
                    try:
                        fb_meta = self._validate_slot(self.root / fallback)
                        pointer_payload = {
                            'directory': fallback,
                            'step': fb_meta['step'],
                            'block': fb_meta['block'],
                            'peak_energy': fb_meta.get('peak_energy', 0.0),
                            'previous': None,
                            'written': time.time(),
                        }
                        _atomic_json(self.pointer_path, pointer_payload)
                        latest = self._read_pointer()
                        continue
                    except Exception:
                        pass
                self.pointer_path.unlink(missing_ok=True)
                latest = None
                raise
        return 0, 0, 0.0, False

    def complete(self) -> None:
        """Mark simulation completed and release lock."""
        _atomic_json(self.complete_path, {'marker': MARKER, 'completed': time.time()})
        self._set_state('completed')
        self._release_lock()

    def close(self) -> None:
        self._release_lock()

    def __del__(self):
        try:
            self._release_lock()
        except Exception:
            pass
