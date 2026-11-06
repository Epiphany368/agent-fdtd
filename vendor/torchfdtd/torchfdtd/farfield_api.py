"""Bounded postprocessing of six stored native radiation faces, without FDTD."""
import hashlib
import json
import threading
from typing import Literal

import numpy as np
import torch
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

from .models import Project

FACES = tuple(a + '_' + side for a in 'xyz' for side in ('min', 'max'))
MAX_DIRECTIONS = 8192
MAX_DIRECTION_POINTS = 50_000_000
MAX_FIELD_SCALARS = 2_000_000
HOST_BUDGET_BYTES = 512 * 1024**2
OUTPUT_BUDGET_BYTES = 16 * 1024**2


class _Config(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class AngularRange(_Config):
    start: StrictFloat
    stop: StrictFloat
    count: StrictInt = Field(ge=1, le=8192)


class FarFieldRequest(_Config):
    version: StrictInt = Field(default=1, ge=1, le=1)
    faces: dict[Literal['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'], str]
    frequency_index: StrictInt = Field(default=0, ge=0)
    bounds_um: tuple[tuple[StrictFloat, StrictFloat], tuple[StrictFloat, StrictFloat], tuple[StrictFloat, StrictFloat]]
    refractive_index: StrictFloat = Field(default=1., gt=0, le=20)
    phase_origin_um: tuple[StrictFloat, StrictFloat, StrictFloat] = (0., 0., 0.)
    theta_deg: AngularRange = Field(default_factory=lambda: AngularRange(start=0., stop=180., count=19))
    phi_deg: AngularRange = Field(default_factory=lambda: AngularRange(start=0., stop=360., count=36))
    reference: str | None = Field(default=None, min_length=1, max_length=128)
    subtract_incident: StrictBool = False
    confirm_homogeneous_closed_surface: StrictBool = False

    @model_validator(mode='after')
    def valid(self):
        if set(self.faces) != set(FACES) or len(set(self.faces.values())) != 6:
            raise ValueError('Select six distinct monitor IDs, one for every closed-box face.')
        if any(not v or len(v) > 128 for v in self.faces.values()):
            raise ValueError('Monitor IDs must contain 1 to 128 characters.')
        if any(hi <= lo for lo, hi in self.bounds_um):
            raise ValueError('Closed-box bounds require three positive spans.')
        t, p = self.theta_deg, self.phi_deg
        if not 0 <= t.start <= t.stop <= 180 or not 0 <= p.start < p.stop <= 360:
            raise ValueError('Use 0 to 180 degrees for theta and an increasing range inside 0 to 360 for phi.')
        if t.count * p.count > MAX_DIRECTIONS:
            raise ValueError('The angular grid exceeds the 8192-direction limit.')
        if bool(self.reference) != self.subtract_incident:
            raise ValueError('A reference and explicit incident-field subtraction must be selected together.')
        return self


def _input(job):
    return dict(project=job['project'], summary=job.get('summary', {}),
                frequency_fields=job.get('frequency_fields', []))


def _selected_points(job, request):
    records = job.get('frequency_fields', [])
    total = 0
    for name in FACES:
        matches = [r for r in records if r.get('id') == request.faces[name]]
        if len(matches) != 1:
            raise ValueError('Each face must select one stored monitor ID. Reference runs retain these same IDs.')
        record = matches[0]
        if request.frequency_index >= len(record.get('frequency_hz', [])):
            raise ValueError('Frequency index is outside a selected face frequency range.')
        total += len(record.get('weights', []))
    if total == 0 or total * 6 > MAX_FIELD_SCALARS:
        raise ValueError('The six-face field sample limit is exceeded or no samples were stored.')
    return total


def register_farfield_routes(app, get_job):
    # Bounded CPU postprocessing has its own admission, independent of FDTD jobs.
    gate = threading.BoundedSemaphore(1)

    def completed(key):
        job = get_job(key)
        if job.get('status') != 'completed':
            raise HTTPException(409, 'Finish a native six-plane run before far-field postprocessing.')
        return job

    @app.get('/api/jobs/{key}/farfield-monitors')
    def monitors(key: str):
        job = completed(key)
        project = Project.model_validate(job['project'])
        region = project.region
        rows = []
        for record in job.get('frequency_fields', []):
            normal = record.get('normal_axis')
            points = np.asarray(record.get('points_um', []))
            position = (float(points[0, 'xyz'.index(normal)]) if normal in ('x', 'y', 'z')
                        and points.ndim == 2 and points.shape[1] == 3 and len(points) else None)
            rows.append(dict(id=record['id'], name=record.get('name', record['id']),
                normal=normal, position_um=position, shape=list(record.get('shape', [])),
                points=len(record.get('weights', [])), components=list(record.get('components', [])),
                frequency_thz=(np.asarray(record.get('frequency_hz', [])) * 1e-12).tolist(),
                dtype=str(getattr(record.get('fields'), 'dtype', 'not stored'))))
        return dict(job_id=key, name=project.name,
            isolated_pml=region.dimension == '3d' and all(f.kind == 'pml' for a in range(3) for f in region.boundaries.pair(a)),
            background_index=region.background_index,
            interior_bounds_um=[list(region.interior_bounds(a)) for a in range(3)], monitors=rows,
            note='Select six complete midpoint planes surrounding an isolated object. Stored data are postprocessed without another FDTD run.')

    @app.post('/api/jobs/{key}/farfield')
    def calculate(key: str, request: FarFieldRequest):
        if not request.confirm_homogeneous_closed_surface:
            raise HTTPException(422, 'Confirm a closed surface in homogeneous lossless isotropic material, outside PML, enclosing the radiating sources or incident-subtracted scatterers.')
        if not gate.acquire(blocking=False):
            raise HTTPException(409, 'Another far-field calculation is using the bounded postprocessing workspace. Try again when it finishes.')
        try:
            job = completed(key)
            reference = completed(request.reference) if request.reference else None
            points = _selected_points(job, request)
            if reference is not None:
                _selected_points(reference, request)
            count = request.theta_deg.count * request.phi_deg.count
            if count * points > MAX_DIRECTION_POINTS:
                raise ValueError('The angular-grid/face-point work limit is exceeded. Request fewer directions.')
            snapshot = request.model_dump(mode='json')
            # JSON float text, containers, amplitude/power and conversion copies.
            output_bound = count * 1024 + 65536 + 4 * len(json.dumps(snapshot).encode())
            if output_bound > OUTPUT_BUDGET_BYTES:
                raise ValueError('Far-field result serialization exceeds its byte budget.')
            from .radiation_box import native_radiation_box
            box = native_radiation_box(_input(job), request.faces,
                bounds_um=request.bounds_um, refractive_index=request.refractive_index,
                frequency_index=request.frequency_index,
                reference=None if reference is None else _input(reference),
                host_budget_bytes=HOST_BUDGET_BYTES - 4 * output_bound)
            theta = np.linspace(request.theta_deg.start, request.theta_deg.stop, request.theta_deg.count)
            phi = np.linspace(request.phi_deg.start, request.phi_deg.stop, request.phi_deg.count, endpoint=False)
            t, p = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing='ij')
            directions = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)), axis=-1).reshape(-1, 3)
            with torch.no_grad():
                far = box.project(directions, phase_origin_um=request.phase_origin_um,
                    direction_chunk=16, point_chunk=2048,
                    host_budget_bytes=HOST_BUDGET_BYTES - 4 * output_bound)
                amplitude = far.electric_amplitude[0].cpu()
                # Only the compact output reduction uses double precision. The
                # stored field and projection dtype stay unchanged. This avoids
                # squaring optical FP32 spectral amplitudes into underflow.
                intensity = .5 * request.refractive_index * (amplitude.real.double().square() + amplitude.imag.double().square()).sum(-1)
                if not bool(torch.isfinite(amplitude).all()) or not bool(torch.isfinite(intensity).all()):
                    raise ValueError('Nonfinite far-field output is not publishable.')
                maximum = float(intensity.max())
                relative = intensity / maximum if maximum > 0 else torch.zeros_like(intensity)
            result = dict(request=snapshot,
                request_digest=hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                job_id=key, reference=request.reference, field_kind=box.report['field_kind'],
                frequency_hz=float(far.frequency_hz[0]), frequency_thz=float(far.frequency_hz[0]) * 1e-12,
                refractive_index=request.refractive_index, phase_origin_um=far.phase_origin_um.tolist(),
                bounds_um=snapshot['bounds_um'], theta_deg=theta.tolist(), phi_deg=phi.tolist(),
                directions=far.directions.tolist(), electric_real=amplitude.real.tolist(), electric_imag=amplitude.imag.tolist(),
                intensity=intensity.reshape(len(theta), len(phi)).tolist(),
                relative_intensity=relative.reshape(len(theta), len(phi)).tolist(), zero_pattern=maximum == 0,
                intensity_units='reduced E*H * s^2 * m^2 / sr', amplitude_units='reduced E * s * m',
                report={**box.report, 'json_reservation_bytes': output_bound, 'intensity_reduction_dtype': 'float64'},
                note='Raw spectral intensity and relative I/max(I) are not efficiency, directivity or calibrated watts. Theta is measured from +z and phi from +x toward +y. Phi excludes its stop angle. No FDTD simulation was launched.')
            if len(json.dumps(result, allow_nan=False).encode()) > min(output_bound, OUTPUT_BUDGET_BYTES):
                raise ValueError('Serialized far-field output exceeds its admitted byte bound.')
            return result
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            gate.release()
