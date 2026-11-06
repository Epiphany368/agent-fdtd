"""Serialized, asynchronous vendor bridge jobs for the loopback workbench."""
import json
import threading
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import Field

from . import fsp
from .models import Model, Project


class ExportRequest(Model):
    patches: list[fsp.PropertyPatch] = Field(default_factory=list, max_length=500)


def attach_fsp_routes(app, root):
    root = Path(root) / 'fsp'
    root.mkdir(parents=True, exist_ok=True)
    from .server import ServerExecutor
    # Imported scenes are validated in this pool, under the server limits of the upload request.
    pool = ServerExecutor(max_workers=1, thread_name_prefix='fsp-bridge')
    jobs, lock = {}, threading.Lock()
    app.state.fsp_pool = pool

    def get(key):
        if key not in jobs:
            raise HTTPException(404, 'FSP job not found in this server session.')
        return jobs[key]

    def allocate(name):
        with lock:
            if sum(j['status'] in ('uploading', 'queued', 'running') for j in jobs.values()) >= 3:
                raise HTTPException(409, 'FSP bridge queue is full.')
            key = uuid4().hex
            folder = root / key
            folder.mkdir()
            job = {'id': key, 'filename': name, 'status': 'uploading'}
            jobs[key] = job
            return job, folder

    def inspect_work(job, folder, original=None, patches=()):
        job['status'] = 'running'
        try:
            source = folder / 'project.fsp'
            if original:
                job['export_verification'] = fsp.export_fsp(original, source, patches)
            manifest = fsp.inspect_fsp(source)
            manifest['source']['filename'] = job['filename']
            fsp.write_inspection(manifest, folder / 'inspection.json')
            fsp.archive_fsp(source, folder / 'project.pwfsp', manifest)
            job['inspection'] = manifest
            job['status'] = 'ready'
        except Exception as exc:
            job['status'] = 'failed'
            job['error'] = str(exc)

    def native_work(job, folder):
        from .fsp_binary import FspDocument
        from .fsp_native import convert_fsp
        job['status'] = 'running'
        try:
            report = convert_fsp(FspDocument.load(folder / 'project.fsp'), Path(job['filename']).stem)
            job['conversion'] = report.as_dict()
            (folder / 'conversion.json').write_text(json.dumps(job['conversion'], indent=2, allow_nan=False), encoding='utf-8')
            job['status'] = 'ready'
        except Exception as exc:
            job['status'], job['error'] = 'failed', str(exc)

    def native_export_work(job, folder, original, project, settings=False):
        from .fsp_binary import FspDocument
        from .fsp_geometry import write_fsp_geometry, write_fsp_scene
        job['status'] = 'running'
        try:
            writer = write_fsp_scene if settings else write_fsp_geometry
            document, report = writer(FspDocument.load(original), project)
            document.save(folder / 'project.fsp')
            job['export_verification'] = report
            (folder / 'write-report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            native_work(job, folder)
        except Exception as exc:
            job['status'], job['error'] = 'failed', str(exc)

    @app.get('/api/fsp/status')
    def bridge_status():
        return fsp.availability()

    @app.post('/api/fsp/import', status_code=202)
    async def upload(request: Request):
        if not fsp.availability()['installed']:
            raise HTTPException(503, fsp.availability()['reason'])
        return await receive_upload(request, False)

    @app.post('/api/fsp/native-import', status_code=202)
    async def native_upload(request: Request):
        return await receive_upload(request, True)

    async def receive_upload(request, native):
        name = unquote(request.headers.get('x-filename', 'project.fsp')).replace('\\', '/').split('/')[-1]
        if not name.lower().endswith('.fsp') or len(name) > 200:
            raise HTTPException(400, 'Choose a Lumerical .fsp project.')
        job, folder = allocate(name)
        job['mode'] = 'native' if native else 'bridge'
        try:
            size = 0
            with (folder / 'project.fsp').open('xb') as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > fsp.MAX_FSP_BYTES:
                        raise HTTPException(413, 'FSP upload exceeds 128 MiB.')
                    out.write(chunk)
            fsp.validate_file(folder / 'project.fsp')
        except Exception as exc:
            (folder / 'project.fsp').unlink(missing_ok=True)
            job['status'], job['error'] = 'failed', 'Upload did not complete.'
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(400, str(exc)) from exc
        job['status'] = 'queued'
        pool.submit(native_work if native else inspect_work, job, folder)
        return {'id': job['id'], 'status': 'queued'}

    @app.get('/api/fsp/{key}')
    def status(key: str):
        return dict(get(key))

    @app.post('/api/fsp/{key}/export', status_code=202)
    def export(key: str, payload: ExportRequest):
        parent = get(key)
        if parent['status'] != 'ready':
            raise HTTPException(409, 'Wait for FSP inspection to complete.')
        job, folder = allocate('edited-' + parent['filename'][:190])
        job['status'] = 'queued'
        pool.submit(inspect_work, job, folder, root / key / 'project.fsp', payload.patches)
        return {'id': job['id'], 'status': 'queued'}

    @app.get('/api/fsp/{key}/download')
    def download(key: str):
        job = get(key)
        if job['status'] != 'ready':
            raise HTTPException(409, 'FSP file is not ready.')
        return FileResponse(root / key / 'project.fsp', filename=job['filename'], media_type='application/octet-stream')

    @app.post('/api/fsp/{key}/native-export', status_code=202)
    def native_export(key: str, payload: Project):
        return submit_native_export(key, payload, False)

    @app.post('/api/fsp/{key}/native-scene-export', status_code=202)
    def native_scene_export(key: str, payload: Project):
        return submit_native_export(key, payload, True)

    def submit_native_export(key, payload, settings):
        parent = get(key)
        if parent['status'] != 'ready' or not parent.get('conversion', {}).get('project'):
            raise HTTPException(409, 'A supported native import must be ready before geometry export.')
        job, folder = allocate('edited-' + parent['filename'][:190])
        job.update(status='queued', mode='native')
        pool.submit(native_export_work, job, folder, root / key / 'project.fsp', payload, settings)
        return {'id': job['id'], 'status': 'queued'}

    @app.get('/api/fsp/{key}/write-report')
    def write_report(key: str):
        job = get(key)
        if job['status'] != 'ready' or job.get('mode') != 'native' or 'export_verification' not in job:
            raise HTTPException(409, 'Independent geometry export report is not ready.')
        return FileResponse(root / key / 'write-report.json', filename=Path(job['filename']).stem + '-write-report.json')

    @app.get('/api/fsp/{key}/archive')
    def archive(key: str):
        job = get(key)
        if job['status'] != 'ready' or job.get('mode') == 'native':
            raise HTTPException(409, 'FSP archive is not ready.')
        return FileResponse(root / key / 'project.pwfsp', filename=Path(job['filename']).stem + '.pwfsp', media_type='application/zip')

    @app.get('/api/fsp/{key}/conversion')
    def conversion_download(key: str):
        job = get(key)
        if job['status'] != 'ready' or 'conversion' not in job:
            raise HTTPException(409, 'Native conversion report is not ready.')
        return FileResponse(root / key / 'conversion.json', filename=Path(job['filename']).stem + '-conversion.json')
