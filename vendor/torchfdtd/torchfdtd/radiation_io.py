"""Native stored frequency planes to continuum radiation postprocessing.

No field reconstruction from snapshots or flux-only monitors is attempted.
The exterior-medium and closed-surface assumptions remain caller obligations.
"""
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .adjoint_planes import COMPONENTS, DifferentiablePlaneResult
from .models import Project


def native_radiation_plane(record, *, frequency_index=None, max_samples=2_000_000):
    """Adapt a native Result.frequency_fields entry on CPU, without changing DFT sign."""
    names=tuple(record.get('components',()))
    if len(names)!=6 or set(names)!=set(COMPONENTS):
        raise ValueError('Radiation needs stored Ex, Ey, Ez, Hx, Hy and Hz. Enable all six field outputs and rerun; flux alone is insufficient.')
    if record.get('flux_units')!='reduced E*H * s^2 * m^2':
        raise ValueError('Radiation requires a 3D plane with area weights, not a 2D invariant-length result.')
    settings=record.get('settings',{})
    if settings.get('spectrum',{}).get('apodization')!='none' or settings.get('time_downsample',1)!=1:
        raise ValueError('Use unapodized frequency monitors with time downsample 1 and rerun.')
    signature=record.get('run_signature')
    if not isinstance(signature,str) or not signature:
        raise ValueError('Stored run signature is missing. Rerun with native frequency monitors to retain reference metadata.')
    fields=np.asarray(record['fields']);frequency=np.asarray(record['frequency_hz'])
    if fields.ndim!=3 or fields.shape[0]!=len(frequency) or fields.shape[-1]!=6 or fields.dtype not in (np.complex64,np.complex128):
        raise ValueError('Stored frequency field shape or complex dtype is invalid.')
    if frequency_index is not None:
        if isinstance(frequency_index,bool) or not isinstance(frequency_index,int) or not 0<=frequency_index<len(frequency):
            raise ValueError('Frequency index is outside the stored frequency range.')
        fields=fields[frequency_index:frequency_index+1];frequency=frequency[frequency_index:frequency_index+1]
    if fields.size>max_samples:
        raise ValueError('Radiation adapter sample limit exceeded. Select one frequency or reduce monitor quadrature.')
    value=torch.as_tensor(fields)
    if names!=COMPONENTS:
        value=value[...,list(names.index(c) for c in COMPONENTS)]
    real=value.real.dtype
    return DifferentiablePlaneResult(value,torch.as_tensor(frequency,dtype=real),
        torch.as_tensor(record['points_um'],dtype=real),torch.as_tensor(record['weights'],dtype=real),
        tuple(record['shape']),record['normal_axis'],signature,
        dict(source='native stored frequency plane',autograd=False),components=COMPONENTS)


def load_native_radiation_plane(path, monitor_id, *, frequency_index=None, max_bytes=256*1024**2):
    """Load only one monitor from a native NPZ, never its full E/H or frame arrays.

    Returns (plane, project). The archive is data only; pickle is disabled.
    """
    with np.load(Path(path),allow_pickle=False) as archive:
        metadata=json.loads(str(archive['field_monitors']))
        matches=[(i,m) for i,m in enumerate(metadata) if m.get('id')==monitor_id]
        if len(matches)!=1:
            raise ValueError('Select one stored frequency monitor ID from field_monitors metadata.')
        i,record=matches[0];record=dict(record)
        names=['fields','frequency_hz','points_um','weights']
        prefix=f'field_monitor_{i}_'
        with zipfile.ZipFile(path) as zipped:
            required=sum(zipped.getinfo(prefix+name+'.npy').file_size for name in names)
        if required>max_bytes:
            raise ValueError('Selected stored monitor exceeds the NPZ loading byte budget.')
        for name in names:record[name]=archive[prefix+name]
        project=Project.model_validate_json(str(archive['project']))
    return native_radiation_plane(record,frequency_index=frequency_index),project
