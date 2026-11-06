"""Modal network jobs share the native single-worker queue and own their child."""
import csv
import io
import json
import multiprocessing
import pprint
import threading
import time
from concurrent.futures import wait
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from .models import server_admission
from .solver import admit_planning
from .mode_network_project import ModeNetworkConfig, mode_network_plan, mode_network_request_digest
from .mode_network_worker import MAX_MESSAGE_BYTES, close_owned_process, mode_network_worker


def request_digest(snapshot):
    return mode_network_request_digest(snapshot)


def _validated_s(result, snapshot):
    if result.get('status') != 'completed':
        raise ValueError('Modal result did not complete.')
    if result.get('request_digest') != request_digest(snapshot):
        raise ValueError('Modal result request digest does not match the submitted snapshot.')
    channels = [[p['name'], index] for p in snapshot['ports'] for index in p['mode_indices']]
    if result.get('channels') != channels:
        raise ValueError('Modal result channels do not match the submitted basis.')
    real = np.asarray(result['s_real'], dtype=np.float32)
    imag = np.asarray(result['s_imag'], dtype=np.float32)
    shape = (len(channels), len(channels))
    if real.shape != shape or imag.shape != shape or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError('Modal result requires finite square FP32 S components for the configured channels.')
    return real, imag


def _execute_owned(snapshot, scratch, cancel, progress):
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    # A spawned child starts without the job thread's context, so the admission is passed.
    child = context.Process(target=mode_network_worker,
        args=(snapshot, str(scratch), send, server_admission() == 'memory'), daemon=True)
    try:
        if cancel.is_set():
            return None
        child.start()
        send.close()
        while True:
            if cancel.is_set():
                return None
            if receive.poll(.05):
                try:
                    message = json.loads(receive.recv_bytes(MAX_MESSAGE_BYTES))
                except EOFError as exc:
                    raise RuntimeError('Modal worker exited without a result.') from exc
                if message['type'] == 'progress':
                    progress(message['data'])
                elif message['type'] == 'error':
                    raise RuntimeError(message['error'])
                elif message['type'] == 'completed':
                    child.join(timeout=5)
                    if child.is_alive() or child.exitcode != 0:
                        raise RuntimeError('Modal worker did not exit successfully.')
                    budget = snapshot['execution']['output_budget_bytes']
                    if scratch.stat().st_size > budget:
                        raise ValueError('Modal result exceeds the output budget.')
                    return json.loads(scratch.read_bytes())
                else:
                    raise RuntimeError('Unexpected modal worker message.')
            elif not child.is_alive():
                raise RuntimeError(f'Modal worker exited without a result (exit {child.exitcode}).')
    finally:
        try:
            close_owned_process(child)
        finally:
            receive.close()
            send.close()
            scratch.unlink(missing_ok=True)


def attach_mode_network_routes(app, root, pool, jobs, lock):
    futures = {}
    stopping = threading.Event()

    @app.on_event('shutdown')
    def shutdown():
        with lock:
            stopping.set()
            for job in jobs.values():
                if job.get('kind') == 'mode_network' and job['status'] in ('queued', 'running'):
                    job['cancel'].set()
            pending = list(futures)
        if pending:
            _, unfinished = wait(pending, timeout=12)
            # Queued callbacks may sit behind unrelated ordinary jobs. Cancel their
            # futures, while active modal callbacks reap their sole owned child.
            for future in unfinished:
                if future.cancel():
                    with lock:
                        job = jobs.get(futures[future])
                        if job is not None:
                            job['status'] = 'cancelled'
    def validate_plan(config):
        try:
            admit_planning(config.project)
            return mode_network_plan(config)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/mode-networks/validate')
    def validate(config: ModeNetworkConfig):
        return validate_plan(config)

    @app.post('/api/mode-networks/python')
    def python(config: ModeNetworkConfig):
        validate_plan(config)
        source = ('import json\nfrom pathlib import Path\n'
            'from torchfdtd.mode_network_project import ModeNetworkConfig, run_mode_network\n\n'
            'config = ModeNetworkConfig.model_validate(' +
            pprint.pformat(config.model_dump(mode='json'), sort_dicts=True) + ')\n'
            'if __name__ == "__main__":\n'
            '    result = run_mode_network(config)\n'
            '    Path("mode-network-result.json").write_text(json.dumps(result, indent=2), encoding="utf8")\n')
        return Response(source, media_type='text/plain')

    def work(key, snapshot):
        job = jobs[key]
        temporary = root / f'{key}.modal.tmp'
        scratch = root / f'{key}.modal.worker.json'
        try:
            with lock:
                if job['cancel'].is_set():
                    job['status'] = 'cancelled'
                    return
                job['status'] = 'running'
            def progress(data):
                with lock:
                    job['progress'] = data
            result = _execute_owned(snapshot, scratch, job['cancel'], progress)
            if result is None:
                with lock:
                    job['status'] = 'cancelled'
                return
            real, imag = _validated_s(result, snapshot)
            encoded = json.dumps(result, allow_nan=False)
            if len(encoded.encode()) > snapshot['execution']['output_budget_bytes']:
                raise ValueError('Modal result exceeds the output budget.')
            with temporary.open('wb') as output:
                np.savez_compressed(output, result_json=np.asarray(encoded),
                    s=(real + np.complex64(1j) * imag), s_real=real, s_imag=imag)
            with lock:
                if job['cancel'].is_set():
                    job['status'] = 'cancelled'
                    return
                temporary.replace(root / f'{key}.modal.npz')
                job['result'] = result
                job['summary'] = {k: result.get(k) for k in ('channels', 'objective', 'request_digest')}
                job['status'] = 'completed'
        except Exception as exc:
            with lock:
                job['status'] = 'cancelled' if job['cancel'].is_set() else 'failed'
                if job['status'] == 'failed':
                    job['error'] = str(exc)
        finally:
            temporary.unlink(missing_ok=True)
            scratch.unlink(missing_ok=True)

    @app.post('/api/mode-network-jobs', status_code=202)
    def submit(config: ModeNetworkConfig):
        validate_plan(config)
        snapshot = config.model_dump(mode='json')
        with lock:
            if stopping.is_set():
                raise HTTPException(503, 'The server is shutting down.')
            if sum(j['status'] in ('queued', 'running') for j in jobs.values()) >= 3:
                raise HTTPException(409, 'The shared run queue is full (one running and two waiting).')
            if len(jobs) >= 12:
                old = next((k for k, j in jobs.items()
                    if j['status'] in ('completed', 'cancelled', 'failed')), None)
                if old is None:
                    raise HTTPException(409, 'The shared job history is full.')
                del jobs[old]
            key = uuid4().hex
            jobs[key] = dict(id=key, kind='mode_network', status='queued', progress={},
                cancel=threading.Event(), project={'name': config.project.name},
                created=time.time(), request_digest=request_digest(snapshot))
            try:
                future = pool.submit(work, key, snapshot)
                for done in [f for f in futures if f.done()]:
                    del futures[done]
                futures[future] = key
            except Exception as exc:
                jobs[key]['status'] = 'failed'
                jobs[key]['error'] = str(exc)
        return {'id': key, 'status': jobs[key]['status']}

    def get_job(key):
        job = jobs.get(key)
        if job is None or job.get('kind') != 'mode_network':
            raise HTTPException(404, 'Modal job not found in this server session.')
        return job

    @app.get('/api/mode-network-jobs/{key}')
    def status(key: str):
        with lock:
            job = get_job(key)
            return {**{k: v for k, v in job.items() if k not in ('cancel',)},
                'cancel_requested': job['cancel'].is_set()}

    @app.post('/api/mode-network-jobs/{key}/cancel')
    def cancel(key: str):
        with lock:
            job = get_job(key)
            if job['status'] in ('queued', 'running'):
                job['cancel'].set()
            return {'id': key, 'status': job['status'], 'cancel_requested': job['cancel'].is_set()}

    @app.get('/api/mode-network-jobs/{key}/download')
    def download(key: str):
        with lock:
            job = get_job(key)
            if job['status'] != 'completed':
                raise HTTPException(409, 'Modal results are not ready.')
        return FileResponse(root / f'{key}.modal.npz', filename=f'mode-network-{key[:8]}.npz')

    @app.get('/api/mode-network-jobs/{key}/s.csv')
    def csv_result(key: str):
        with lock:
            job = get_job(key)
            if job['status'] != 'completed':
                raise HTTPException(409, 'Modal results are not ready.')
            result = job['result']
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['output_channel', 'input_channel', 'real', 'imag', 'power'])
        for i, row in enumerate(result['s_real']):
            for j, real in enumerate(row):
                imag = result['s_imag'][i][j]
                writer.writerow([json.dumps(result['channels'][i]), json.dumps(result['channels'][j]),
                    real, imag, real * real + imag * imag])
        return Response(stream.getvalue(), media_type='text/csv')
