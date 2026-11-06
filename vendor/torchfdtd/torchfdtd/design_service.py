"""Periodic design routes sharing the ordinary forward-run queue."""
import json
import pprint
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from .periodic_design import PeriodicDesignConfig, periodic_design_plan, run_periodic_design


def attach_design_routes(app, root, pool, jobs, lock):
    state_root = (Path(root)/'design-state').resolve()

    def contained(config, create=False):
        # A client names a subdirectory of the server's design state root, never a filesystem path.
        name = config.state_directory
        if name is None:
            return config
        # The same names are refused on every platform, before any path is resolved: a control character, a
        # backslash, a colon (drive or stream syntax), a leading slash (a root or a UNC share), a component over
        # 255 characters, or a name resolving outside; a name the file system refuses answers 422 as well.
        refusal = HTTPException(422, "The state directory names a subdirectory of the server's design state directory "
                                     '(<results>/design-state), written with forward slashes, not a path of its own.')
        if (any(ord(c) < 32 for c in name) or '\\' in name or ':' in name or name.startswith('/')
                or any(len(part) > 255 for part in name.split('/'))):
            raise refusal
        try:
            path = (state_root/name).resolve()
            inside = path != state_root and path.is_relative_to(state_root)
            if inside and create:
                path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, 'The state directory name cannot be used as a directory on this server.') from exc
        if not inside:
            raise refusal
        return config.model_copy(update={'state_directory': str(path)})

    @app.get('/api/design/defaults')
    def defaults():
        return PeriodicDesignConfig().model_dump(mode='json')

    @app.post('/api/design/config')
    def validate(config: PeriodicDesignConfig):
        contained(config)
        return config.model_dump(mode='json')

    @app.post('/api/design/plan')
    def plan(config: PeriodicDesignConfig):
        config = contained(config, create=True)
        try:return periodic_design_plan(config)
        except (ValueError, RuntimeError) as exc:raise HTTPException(422,str(exc)) from exc

    @app.post('/api/design/python')
    def python(config: PeriodicDesignConfig):
        source = ('import json\nfrom pathlib import Path\n'
            'from torchfdtd import PeriodicDesignConfig, run_periodic_design\n\n'
            'config = PeriodicDesignConfig.model_validate('+pprint.pformat(config.model_dump(mode='json'),sort_dicts=False)+')\n'
            'result = run_periodic_design(config, on_progress=lambda p: print(p["stage"], p["updates_completed"]))\n'
            'Path("periodic-design-result.json").write_text(json.dumps(result, indent=2), encoding="utf8")\n')
        return Response(source,media_type='text/plain')

    def work(key, config):
        job = jobs[key]
        if job['cancel'].is_set():job['status']='cancelled';return
        job['status']='running'
        try:
            def progress(data):job['progress']=data
            result = run_periodic_design(config,on_progress=progress,cancel=job['cancel'])
            path = root/f'{key}.design.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result,allow_nan=False),encoding='utf8')
            temporary.replace(path)
            job['summary'] = dict(updates_completed=result['updates_completed'],
                best_objective=None if result['best'] is None else result['best']['objective'],
                last_objective=None if result['last_evaluated'] is None else result['last_evaluated']['objective'])
            job['status']=result['status']
        except Exception as exc:
            job['error']=str(exc)
            job['status']='failed'

    @app.post('/api/design/jobs',status_code=202)
    def run(config: PeriodicDesignConfig):
        config = contained(config, create=True)
        with lock:
            if sum(j['status'] in ('queued','running') for j in jobs.values()) >= 3:
                raise HTTPException(409,'The shared run queue is full (one running and two waiting).')
            if len(jobs)>=12:
                old=next((k for k,j in jobs.items() if j['status'] in ('completed','cancelled','failed')),None)
                if old:del jobs[old]
            key=uuid4().hex
            jobs[key]=dict(id=key,kind='periodic_design',status='queued',progress={},
                cancel=threading.Event(),project={'name':config.name},config=config.model_dump(mode='json'),created=time.time())
            pool.submit(work,key,config.model_copy(deep=True))
        return dict(id=key,status='queued')

    @app.get('/api/design/jobs/{key}/download')
    def download(key: str):
        job=jobs.get(key)
        if job is None or job.get('kind')!='periodic_design':raise HTTPException(404,'Design job not found.')
        path=root/f'{key}.design.json'
        if not path.exists():raise HTTPException(409,'Evaluated design results are not ready.')
        return FileResponse(path,filename=f'periodic-design-{key[:8]}.json',media_type='application/json')
