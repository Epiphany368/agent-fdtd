"""Durable checkpoint of an optimization loop that drives a streamed adjoint.

The streamed restart journal carries the solver state of one forward or
backward pass. An optimization needs, beside it, the design parameters, the
optimizer moments and step count, the projection and continuation state, the
random number generator states, the effective source waveform the solver
injects and the configuration fingerprint (`torchfdtd.identity.restart_key`)
so that a resumed loop reproduces the uninterrupted history. `DesignCheckpoint`
writes those as one file with a checksum beside a JSON description, renamed
into place, and refuses a checkpoint that lacks any of them by name, that was
written for another configuration, or that fails its checksum
(docs/STREAMED_RESTART.md). The file holds tensors and plain containers only
and is read with `weights_only=True`, so a pickled object in it is refused
rather than executed.
"""
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from .identity import restart_key
from .streamed_restart import MARKER, _atomic_json, _durable_replace, _read_json, sha256_file

REQUIRED_STATES = ('iteration', 'parameters', 'optimizer', 'projection', 'rng', 'waveform', 'fingerprint', 'history')


def rng_state():
    """Every generator a loop may draw from: torch CPU and CUDA, NumPy and Python.

    The NumPy key array is stored as a tensor; the Python state is plain ints."""
    name, key, *rest = np.random.get_state()
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=(name, torch.from_numpy(key.astype(np.int64)), *rest), python=random.getstate())


def set_rng_state(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])
    name, key, *rest = state['numpy']
    np.random.set_state((name, key.numpy().astype(np.uint32), *rest))
    random.setstate(state['python'])


def effective_waveforms(plan):
    """The sampled source terms the solver injects, from the resolved plan."""
    return [dict(source=source.name, kind=source.kind, injection=source.injection, component=term.component,
                 location=[int(v) if isinstance(v, (int, np.integer)) else [int(v.start), int(v.stop)] for v in term.location],
                 sample_times=torch.from_numpy(np.array(term.sample_times, dtype=np.float64)),
                 samples=torch.from_numpy(np.array(term.samples)))
            for source in plan.sources for term in source.terms]


def projection_state(parameterization):
    """Continuation value, threshold, update count and the filter configuration."""
    return dict(beta=float(parameterization.beta), eta=float(parameterization.eta),
                continuation_updates=int(parameterization.continuation_updates),
                **parameterization.get_extra_state())


def _same_waveforms(saved, current):
    if len(saved) != len(current):
        return False
    for a, b in zip(saved, current):
        if any(a.get(k) != b.get(k) for k in ('source', 'kind', 'injection', 'component', 'location')):
            return False
        if not torch.equal(a['sample_times'], b['sample_times']) or not torch.equal(a['samples'], b['samples']):
            return False
    return True


class DesignCheckpoint:
    """One checkpoint file per optimization, replaced atomically at every save."""
    def __init__(self, directory, plan, *, options=None):
        self.root = Path(directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = restart_key(plan, options=options)
        self.waveform = effective_waveforms(plan)
        self.path = self.root / 'checkpoint.pt'
        self.meta_path = self.root / 'checkpoint.json'

    def save(self, *, iteration, parameterization, optimizer, history, extra=None):
        payload = dict(marker=MARKER, iteration=int(iteration), parameters=parameterization.state_dict(),
                       optimizer=optimizer.state_dict(), projection=projection_state(parameterization),
                       rng=rng_state(), waveform=self.waveform, fingerprint=self.fingerprint,
                       history=list(history), extra=extra, saved=time.time())
        temp = self.root / 'checkpoint.pt.tmp'
        with open(temp, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest, size = sha256_file(temp), temp.stat().st_size
        _durable_replace(temp, self.path)
        _atomic_json(self.meta_path, dict(marker=MARKER, iteration=int(iteration), sha256=digest, bytes=size,
                                          fingerprint=self.fingerprint, saved=payload['saved']))
        return dict(iteration=int(iteration), sha256=digest, bytes=size)

    def load(self, parameterization, optimizer):
        """Restore every state into the given objects; None when no checkpoint exists."""
        if not self.meta_path.exists():
            return None
        meta = _read_json(self.meta_path)
        if meta.get('marker') != MARKER:
            raise ValueError('Design checkpoint description was not written by this module.')
        if not self.path.exists():
            raise ValueError('Design checkpoint is damaged: checkpoint.pt is missing.')
        size = self.path.stat().st_size
        if size != meta.get('bytes'):
            raise ValueError(f'Design checkpoint is damaged: checkpoint.pt holds {size} of {meta.get("bytes")} bytes.')
        digest = sha256_file(self.path)
        if digest != meta.get('sha256'):
            raise ValueError('Design checkpoint is damaged: checksum mismatch in checkpoint.pt.')
        try:
            payload = torch.load(self.path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError('Design checkpoint holds a pickled object that is not a tensor or a plain container; '
                             'it was refused and nothing in it was executed.') from exc
        if not isinstance(payload, dict) or payload.get('marker') != MARKER:
            raise ValueError('Design checkpoint was not written by this module.')
        missing = [name for name in REQUIRED_STATES if name not in payload]
        if missing:
            raise ValueError('Design checkpoint is missing the state '+', '.join(missing)+'. Every state of REQUIRED_STATES must be present.')
        if payload['fingerprint'] != self.fingerprint:
            raise ValueError('Design checkpoint was written for another configuration: the fingerprint (restart_key) differs.')
        if not _same_waveforms(payload['waveform'], self.waveform):
            raise ValueError('Design checkpoint was written for another effective source waveform.')
        parameterization.load_state_dict(payload['parameters'])
        expected = projection_state(parameterization)
        if payload['projection'] != expected:
            raise ValueError('Design checkpoint projection state disagrees with its parameters: '
                             +', '.join(sorted(k for k in set(expected) | set(payload['projection']) if expected.get(k) != payload['projection'].get(k))))
        optimizer.load_state_dict(payload['optimizer'])
        set_rng_state(payload['rng'])
        return dict(iteration=int(payload['iteration']), history=list(payload['history']), extra=payload.get('extra'))
