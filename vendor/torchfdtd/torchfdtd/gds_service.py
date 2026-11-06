"""Bounded optional GDS upload/conversion routes. No arbitrary server paths."""
from pathlib import Path
from uuid import uuid4
from urllib.parse import unquote
import hashlib

from fastapi import HTTPException, Request
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .models import Model, Project
from . import gds

MAX_UPLOAD_BYTES = 32_000_000
# Uploads keep the vertex limit that the Python API leaves open (docs/SECURITY.md).
SERVER_GDS_LIMITS = gds.GDSLimits(max_total_vertices=1_000_000)


class GDSConversion(Model):
    project: Project
    cell: str = Field(min_length=1, max_length=1024)
    layers: list[dict] = Field(min_length=1, max_length=1000)
    port_layers: list[dict] = Field(default_factory=list, max_length=1000)
    unmapped: str = 'error'
    replace_geometry: bool = False


class GDSExport(Model):
    project: Project
    cell: str = Field(default='TOP', min_length=1, max_length=32)
    # structure id -> [layer, datatype] for every enabled structure
    layers: dict[str, tuple[int, int]] = Field(min_length=1, max_length=1000)
    unit_m: float = Field(default=1e-6, gt=0)
    precision_m: float = Field(default=1e-9, gt=0)


def attach_gds_routes(app, root):
    folder = Path(root) / 'gds'
    folder.mkdir(parents=True, exist_ok=True)
    uploads = {}

    def inspect(data):
        gds._audit_records(data)
        dependency = gds._gdstk()
        with gds._gds_io() as path:
            path.write_bytes(data)
            unit, precision = dependency.gds_units(path)
            library = dependency.read_gds(path)
        cells = []
        for cell in sorted(library.cells, key=lambda value: value.name):
            pairs = {(p.layer, p.datatype) for p in cell.polygons}
            for path in cell.paths:
                pairs.update(zip(path.layers, path.datatypes))
            cells.append({'name': cell.name, 'geometry_pairs': sorted(pairs),
                          'text_pairs': sorted({(p.layer, p.texttype) for p in cell.labels})})
        return {'cells': cells, 'unit_m': unit, 'precision_m': precision,
                'sha256': hashlib.sha256(data).hexdigest()}

    @app.post('/api/gds/inspect')
    async def upload(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, 'GDS upload exceeds 32 MB.')
            data.extend(chunk)
        try:
            result = await run_in_threadpool(inspect, bytes(data))
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        key = uuid4().hex
        path = folder / (key + '.gds')
        path.write_bytes(data)
        uploads[key] = path
        return {**result, 'id': key, 'filename': unquote(request.headers.get('x-filename', 'layout.gds'))[:512]}

    @app.post('/api/gds/export')
    def export(payload: GDSExport):
        # XY geometry only; the returned sidecar carries the Z/material stack the file cannot hold.
        import base64
        path = folder / ('export-' + uuid4().hex + '.gds')
        try:
            sidecar = gds.export_gds(path, payload.project.structures, layers=payload.layers, cell=payload.cell,
                                     unit_m=payload.unit_m, precision_m=payload.precision_m)
            data = path.read_bytes()
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        return {'gds_base64': base64.b64encode(data).decode('ascii'), 'sidecar': sidecar, 'bytes': len(data)}

    @app.post('/api/gds/{key}/convert')
    def convert(key: str, payload: GDSConversion):
        if key not in uploads:
            raise HTTPException(404, 'GDS upload not found in this server session.')
        try:
            imported = gds.import_gds(uploads[key], cell=payload.cell,
                layers=[gds.GDSLayer(**row) for row in payload.layers],
                port_layers=[gds.GDSPortLayer(**row) for row in payload.port_layers],
                unmapped=payload.unmapped, limits=SERVER_GDS_LIMITS)
            project = payload.project
            if payload.replace_geometry:
                project = project.model_copy(update={'structures': []})
            candidate = imported.add_to(project)
            return {'project': candidate.model_dump(), 'report': imported.report}
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
