"""Angular-spectrum post-processing of a finished job's stored DFT plane, without FDTD.

The plane of any completed forward job (resident, streamed or tiled) propagates
through the homogeneous exterior with the same implementation the tiled focal
plane uses: ``propagate_section`` for an xz/yz section and ``propagate_volume``
at one distance for a parallel plane. The computation runs on the job's device.
"""
import math
from typing import Literal

import numpy as np
import torch
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from .models import Project

MAX_PLANES = 2000
MAX_IMAGE = 512
C0 = 299792458.0


class PropagationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    monitor: str = Field(min_length=1, max_length=128)
    kind: Literal['section', 'plane'] = 'section'
    axis: Literal['x', 'y', 'z'] | None = None      # transverse axis of the section
    offset_um: StrictFloat = 0.                      # fixed coordinate of the other transverse axis
    index: StrictFloat = Field(default=1., gt=0, le=20)
    direction: Literal['auto', '+', '-'] = 'auto'
    frequency_index: StrictInt = Field(default=0, ge=0)
    z_start_um: StrictFloat = Field(default=0., ge=0)
    z_stop_um: StrictFloat = Field(default=10., gt=0)
    planes: StrictInt = Field(default=200, ge=1, le=MAX_PLANES)
    distance_um: StrictFloat = Field(default=10., ge=0)
    pad: StrictInt = Field(default=2, ge=1, le=8)

    @model_validator(mode='after')
    def valid(self):
        if self.kind == 'section' and self.z_stop_um < self.z_start_um:
            raise ValueError('The distance range must end at or beyond its start.')
        if self.kind == 'section' and self.planes > 1 and self.z_stop_um == self.z_start_um:
            raise ValueError('A range of one distance needs a single plane.')
        return self


def _budget_bytes(device):
    """Half the free device memory on CUDA, a quarter of the available host memory on the CPU."""
    if device.type == 'cuda':
        from .cuda_memory import cuda_mem_info
        free, _ = cuda_mem_info(device)
        return int(free*.5)
    from .memory_profile import host_memory
    available = host_memory()['available_bytes']
    return int(available*.25) if available else 4*2**30


def _job_device(job):
    return torch.device('cuda' if job.get('summary', {}).get('backend') == 'cuda' and torch.cuda.is_available() else 'cpu')


def _direction(project, record):
    """+1 when the enabled sources lie behind the plane along its normal, -1 when in front."""
    axis = 'xyz'.index(record['normal_axis'])
    offset = float(np.asarray(record['points_um'])[0, axis])
    sides = {int(math.copysign(1, offset-s.center[axis])) for s in project.sources if s.enabled and s.center[axis] != offset}
    if len(sides) == 1:
        return sides.pop(), 'inferred from the sources'
    return 1, 'sources on both sides or on the plane; assuming the positive normal'


def _spacing(record):
    points = np.asarray(record['points_um'])
    steps = {}
    for a in range(3):
        if 'xyz'[a] == record['normal_axis']:
            continue
        unique = np.unique(np.round(points[:, a], 9))
        if len(unique) > 1:
            steps['xyz'[a]] = float(np.diff(unique).mean())
    return steps


def _monitor_summary(project, record):
    axis = record['normal_axis']
    a = 'xyz'.index(axis)
    points = np.asarray(record['points_um'])
    frequency = np.asarray(record['frequency_hz'], dtype=np.float64)
    direction, note = _direction(project, record)
    transverse = [c for c in 'xyz' if c != axis and (project.region.dimension == '3d' or c != 'z')]
    extent = {c: [float(points[:, 'xyz'.index(c)].min()), float(points[:, 'xyz'.index(c)].max())] for c in transverse}
    spacing = _spacing(record)
    return dict(id=record['id'], name=record['name'], normal=axis, position_um=float(points[0, a]), direction=direction,
                direction_note=note, transverse=transverse, extent_um=extent, spacing_um=spacing,
                frequency_thz=(frequency*1e-12).tolist(), wavelength_um=(C0/frequency*1e6).tolist(),
                components=list(record.get('components', [])), shape=list(record['shape']), tiled=record.get('tiled'))


def _image(values):
    """Decimate a 2D intensity map to at most MAX_IMAGE samples per axis for display."""
    array = np.asarray(values, dtype=np.float64)
    strides = tuple(max(1, math.ceil(n/MAX_IMAGE)) for n in array.shape)
    return array[::strides[0], ::strides[1]], strides


def _fwhm_along(profile, peak, coordinates):
    from .angular_spectrum import _fwhm
    return _fwhm(torch.as_tensor(np.asarray(profile, dtype=np.float64)), int(peak), torch.as_tensor(np.asarray(coordinates, dtype=np.float64)))


def register_propagation_routes(app, get_job):
    @app.get('/api/jobs/{key}/propagation-monitors')
    def propagation_monitors(key: str):
        job = get_job(key)
        if job['status'] not in ('completed', 'cancelled') or 'frequency_fields' not in job:
            raise HTTPException(409, 'Angular-spectrum propagation needs a finished run with stored frequency planes.')
        project = Project.model_validate(job['project'])
        return dict(job=key, name=job['project']['name'], device=_job_device(job).type, mode=job.get('execution', {}).get('mode'),
                    background_index=project.region.background_index, dimension=project.region.dimension,
                    monitors=[_monitor_summary(project, m) for m in job['frequency_fields']],
                    note='Homogeneous, lossless, source-free exterior beyond the plane with outgoing waves only; the recorded window is zero padded.')

    @app.post('/api/jobs/{key}/propagate')
    def propagate(key: str, request: PropagationRequest):
        from .angular_spectrum import plane_grid, propagate_section, propagate_volume, volume_bytes
        job = get_job(key)
        if 'frequency_fields' not in job:
            raise HTTPException(409, 'Angular-spectrum propagation needs a finished run with stored frequency planes.')
        records = [m for m in job['frequency_fields'] if m['id'] == request.monitor]
        if len(records) != 1:
            raise HTTPException(422, f'{request.monitor} is not a stored DFT plane of this run.')
        record = records[0]
        project = Project.model_validate(job['project'])
        frequency = np.asarray(record['frequency_hz'], dtype=np.float64)
        if request.frequency_index >= len(frequency):
            raise HTTPException(422, 'Frequency index out of range.')
        device = _job_device(job)
        inferred, note = _direction(project, record)
        direction = inferred if request.direction == 'auto' else (1 if request.direction == '+' else -1)
        f = request.frequency_index
        single = {**record, 'fields': torch.as_tensor(np.asarray(record['fields'])[f:f+1], device=device), 'frequency_hz': frequency[f:f+1]}
        try:
            grid = plane_grid(single, direction=direction)
            electric = [c for c in grid.components if c.startswith('E')]
            if not electric:
                raise ValueError('The stored plane holds no electric component; record Ex, Ey or Ez.')
            budget = _budget_bytes(device)
            item = grid.fields.element_size()
            spacing = float(grid.spacing_um[0])
            wavelength = C0/float(frequency[f])*1e6/request.index
            aliasing = spacing > wavelength/2
            if request.kind == 'section':
                transverse = [c for c in grid.axes if project.region.dimension == '3d' or c != 'z']
                axis = request.axis or transverse[0]
                if axis not in transverse:
                    raise ValueError(f'The section axis must be one of {transverse} for this {grid.normal}-normal plane.')
                z = np.linspace(request.z_start_um, request.z_stop_um, request.planes)
                pu = request.pad*grid.fields.shape[1]
                pv = request.pad*grid.fields.shape[2] if grid.fields.shape[2] > 1 else 1
                along = grid.fields.shape[1] if axis == grid.axes[0] else grid.fields.shape[2]
                required = len(z)*along*len(electric)*item+6*pu*pv*item
                if required > budget:
                    raise ValueError(f'Section needs about {required:,} bytes, above the budget of {budget:,} on {device.type}. Use fewer planes or components or a smaller pad.')
                with torch.no_grad():
                    section = propagate_section(grid, z, index=request.index, section=axis+grid.normal, offset_um=request.offset_um,
                                                components=tuple(electric), pad=request.pad)
                intensity = section.intensity()[0].cpu().numpy()
                focus = section.focus(0)
                image, strides = _image(intensity)
                geometry = dict(axes=[grid.normal, axis], z_um=section.z_um.cpu().numpy()[::strides[0]].tolist(),
                                normal_um=section.normal_um.cpu().numpy()[::strides[0]].tolist(),
                                a_um=section.a_um.cpu().numpy()[::strides[1]].tolist(), offset_um=request.offset_um,
                                extent_um=dict(z=[float(z[0]), float(z[-1])], a=[float(section.a_um[0]), float(section.a_um[-1])]))
                report = section.report
            else:
                required = volume_bytes(grid, [request.distance_um], components=tuple(electric), pad=request.pad, chunk=1, index=request.index)
                with torch.no_grad():
                    volume = propagate_volume(grid, [request.distance_um], index=request.index, components=tuple(electric), pad=request.pad,
                                              chunk=1, budget_bytes=budget)
                intensity = volume.intensity()[0, 0].cpu().numpy()
                iu, iv = np.unravel_index(int(intensity.argmax()), intensity.shape)
                focus = dict(z_um=request.distance_um, normal_um=float(volume.normal_um[0]), a_um=float(volume.u_um[iu]),
                             b_um=float(volume.v_um[iv]), peak_intensity=float(intensity[iu, iv]),
                             fwhm_um=_fwhm_along(intensity[:, iv], iu, volume.u_um))
                image, strides = _image(intensity)
                geometry = dict(axes=list(volume.axes), u_um=volume.u_um.cpu().numpy()[::strides[0]].tolist(),
                                v_um=volume.v_um.cpu().numpy()[::strides[1]].tolist(), normal_um=float(volume.normal_um[0]),
                                extent_um=dict(u=[float(volume.u_um[0]), float(volume.u_um[-1])], v=[float(volume.v_um[0]), float(volume.v_um[-1])]))
                report = dict(volume.report, bytes=required)
                required = required['working_bytes']+required['output_bytes']
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not np.isfinite(image).all():
            raise HTTPException(422, 'The propagated intensity is not finite.')
        max_angle = float(report['max_angle_deg'][0])
        spectrum = dict(max_angle_deg=max_angle, evanescent_fraction=float(report['evanescent_fraction'][0]), spacing_um=spacing,
                        wavelength_um=wavelength, index=request.index, pad=request.pad, padded_shape=list(report['padded_shape']),
                        aliasing=aliasing,
                        warning=(f'The monitor spacing {spacing:.4g} um exceeds half the wavelength in the exterior ({wavelength/2:.4g} um): '
                                 f'angles above {max_angle:.1f} degrees alias instead of failing.') if aliasing else None)
        focus = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in focus.items()}
        return dict(job=key, monitor=request.monitor, kind=request.kind, device=device.type, direction=direction,
                    direction_note=note if request.direction == 'auto' else 'chosen', frequency_thz=float(frequency[f]*1e-12),
                    image=image.tolist(), image_strides=list(strides), focus=focus, spectrum=spectrum, bytes=required, budget_bytes=budget,
                    geometry=geometry, components=electric, method=report['method'],
                    scope='Angular spectrum through a homogeneous lossless exterior with outgoing waves only; the recorded window is zero padded and the field beyond it is assumed zero.')
