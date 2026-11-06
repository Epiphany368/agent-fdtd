"""The three identity conditions, expressed on the resolved plan (docs/IDENTITY_CONDITIONS.md).

Each condition hashes a declared subset of the plan's canonical sections:

* reference compatibility: mesh, time, exterior, sources and the observation
  geometry; a scatterer and its matched reference share this key;
* cache validity: reference compatibility plus the material, the monitors'
  frequency samples and the precision;
* restart contract: cache validity plus the kernel scheme, the exact epsilon
  tensor, the execution options, the runtime sources and the torch version.

Placement (backend, memory and execution mode, tiling) and display settings
enter none of them. The keys are pure functions of their inputs; nothing here
stores or invalidates anything.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import torch

from .plan import SimulationPlan

REFERENCE_SECTIONS = ('mesh', 'time', 'exterior', 'sources')
# Observation geometry a matched reference must share: the field monitors' quadrature,
# interpolation, time downsampling, DFT precision and apodization, not their frequency
# samples and not the point monitors, which do not affect the fields.
OBSERVATION_KEYS = ('normal', 'points_um', 'weights', 'shape', 'maps', 'time_downsample', 'dft_precision', 'apodization')
CACHE_SECTIONS = REFERENCE_SECTIONS + ('material', 'monitors')
KERNEL_SCHEME_FIELDS = ('cuda_kernel', 'cuda_monitor_kernel')


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def _plan(plan):
    if not isinstance(plan, SimulationPlan):
        raise TypeError('Identity conditions take a SimulationPlan from resolve_plan.')
    return plan.sections


def reference_payload(plan):
    """What a matched reference must share: mesh, time base, sources, exterior and observation geometry."""
    sections = _plan(plan)
    observation = [{k: m[k] for k in OBSERVATION_KEYS} for m in sections['monitors'] if m['kind'] == 'field']
    return {**{k: sections[k] for k in REFERENCE_SECTIONS}, 'observation': observation}


def reference_key(plan):
    return _digest(reference_payload(plan))


def cache_payload(plan):
    """Reference compatibility plus the sampled material, every monitor's samples and the precision."""
    sections = _plan(plan)
    return {**{k: sections[k] for k in CACHE_SECTIONS}, 'precision': plan.placement['precision']}


def cache_key(plan):
    return _digest(cache_payload(plan))


def _options(options):
    if options is None:
        return None
    settings = asdict(options) if is_dataclass(options) else dict(options)
    for key in ('restart_directory', 'restart_every_blocks'):
        settings.pop(key, None)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()}


def restart_payload(plan, *, epsilon=None, options=None):
    """Cache validity plus the numerical path: kernel scheme, exact epsilon, options, runtime sources, torch."""
    from .streamed_restart import runtime_sources, sha256_tensor
    payload = dict(cache_payload(plan), kernel_scheme={k: plan.placement[k] for k in KERNEL_SCHEME_FIELDS},
                   options=_options(options), runtime_sha256=runtime_sources(), torch=torch.__version__)
    if epsilon is not None:
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError('The restart contract hashes the exact epsilon tensor.')
        payload['epsilon'] = dict(sha256=sha256_tensor(epsilon.detach().cpu()), shape=list(epsilon.shape), dtype=str(epsilon.dtype))
    return payload


def restart_key(plan, *, epsilon=None, options=None):
    return _digest(restart_payload(plan, epsilon=epsilon, options=options))


def identity(plan, *, epsilon=None, options=None):
    """All three keys of one plan."""
    return dict(reference=reference_key(plan), cache=cache_key(plan),
                restart=restart_key(plan, epsilon=epsilon, options=options))


def invalidated(before, after, *, epsilon=(None, None), options=(None, None)):
    """Names of the conditions that differ between two plans (with optional per-plan epsilon and options)."""
    a = identity(before, epsilon=epsilon[0], options=options[0])
    b = identity(after, epsilon=epsilon[1], options=options[1])
    return tuple(name for name in ('reference', 'cache', 'restart') if a[name] != b[name])
