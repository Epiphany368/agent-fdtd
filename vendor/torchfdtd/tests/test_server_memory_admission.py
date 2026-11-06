"""Memory admission of the workbench server: torchfdtd serve --memory-admission.

By default the server applies SERVER_LIMITS to every request, job thread and
modal worker (tests/test_server_security.py, tests/test_resident_guards.py).
With create_app(memory_admission=True) the size limits are lifted and resident
execution is admitted by the memory estimate as on the Python API, in the
requests and in the job threads and modal worker they start, while the input
limits of the server stay (docs/SECURITY.md). The host arrays of a resident
run, its frames included, bound it on the host in this mode. Memory is
simulated: the resolver reads CPU_RECORD and the CPU Simulation a patched
host_memory.
"""
import contextlib
import inspect
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient

import test_server_security as security
from test_mode_network_integration import payload as modal_payload
from test_resident_guards import CPU_RECORD, LARGE, SERVER_CASES, oversized, single_oversized, structures, wide_plane
from torchfdtd import FieldMonitor, Monitor, Project, Region, Simulation, Source, cli, server
from torchfdtd.models import MEMORY_ADMISSION_LIMITS, SERVER_LIMITS, demo_project, server_admission, server_limit, server_limits
from torchfdtd.execution_modes import _resident_fit
from torchfdtd.solver import display_host_bytes, estimate


@pytest.fixture(autouse=True)
def simulated_resources(monkeypatch):
    monkeypatch.setattr(server, 'execution_resources', lambda: dict(CPU_RECORD))


@contextlib.contextmanager
def workbench(root, memory_admission):
    app = server.create_app(root, memory_admission=memory_admission)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.state.pool.shutdown()
        app.state.fsp_pool.shutdown()


@pytest.fixture
def memory_app(tmp_path):
    application = server.create_app(tmp_path / 'results', memory_admission=True)
    yield application
    application.state.pool.shutdown()
    application.state.fsp_pool.shutdown()


@pytest.fixture
def memory_client(memory_app):
    with TestClient(memory_app, raise_server_exceptions=False) as c:
        yield c


def resident(**region):
    """An explicit resident grid of 8,388,608 cells, above the server's resident cell limit."""
    return Project(region=Region(**LARGE, execution_mode='resident', **region), sources=[Source(center=(0, 0, 0))])


def finished(client, path, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = client.get(path).json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.02)
    raise AssertionError(f'{path} did not finish: {job}')


# ---------------------------------------------------------------------------------------------- switch
def test_serve_passes_the_memory_admission_flag(monkeypatch):
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kw: calls.append((app, kw)))
    monkeypatch.setattr(server, 'create_app', lambda **kwargs: kwargs)
    cli.main(['serve'])
    cli.main(['serve', '--memory-admission', '--port', '8123'])
    assert calls == [({'memory_admission': False}, {'host': '127.0.0.1', 'port': 8765}),
                     ({'memory_admission': True}, {'host': '127.0.0.1', 'port': 8123})]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['serve', '--memory-admission', '--host', '0.0.0.0'])
    assert exit_info.value.code == 2 and len(calls) == 2


def test_health_reports_the_admission(tmp_path):
    with workbench(tmp_path / 'fixed', False) as client:
        health = client.get('/api/health').json()
        assert health['admission'] == 'fixed' and health['server_limits'] == SERVER_LIMITS
    with workbench(tmp_path / 'memory', True) as client:
        health = client.get('/api/health').json()
        assert health['admission'] == 'memory' and health['server_limits'] is None


def test_server_executor_tasks_keep_the_admission_of_the_submitting_request():
    with server.ServerExecutor(max_workers=1) as pool:
        assert pool.submit(server_admission).result() == 'fixed'
        with server_limits(memory_admission=True):
            assert pool.submit(server_admission).result() == 'memory'
            assert pool.submit(server_limit, 'resident_cells').result() is None
        with server_limits():
            assert pool.submit(server_limit, 'resident_cells').result() == SERVER_LIMITS['resident_cells']
    assert server_admission() is None


# -------------------------------------------------------------------------------------------- admission
def test_a_resident_grid_above_the_server_cell_limit_is_admitted_by_the_estimate(tmp_path):
    body = resident().model_dump(mode='json')
    with workbench(tmp_path / 'fixed', False) as client:
        response = client.post('/api/validate', json=body)
        assert response.status_code == 422 and 'server limits resident execution to 8,000,000 cells' in response.text
    with workbench(tmp_path / 'memory', True) as client:
        response = client.post('/api/validate', json=body)
        assert response.status_code == 200, response.text
        execution = response.json()['execution']
        assert execution['mode'] == 'resident' and execution['warnings'] == [] and execution['cells'] == 8_388_608
        assert execution['resident']['fits'] and execution['resident']['cell_limit'] is None
        assert 'within 80% of available host memory' in execution['resident']['reason']
        auto = dict(body, region=dict(body['region'], execution_mode='auto'))
        assert client.post('/api/validate', json=auto).json()['execution']['mode'] == 'resident'
        for route in ('/api/mesh/preview', '/api/python'):
            assert client.post(route, json=body).status_code == 200, route
        # A cap the project carries still applies, as on the Python API.
        capped = dict(body, region=dict(body['region'], resident_cell_limit=8_000_000))
        response = client.post('/api/validate', json=capped)
        assert response.status_code == 422 and 'resident_cell_limit=8,000,000 cells' in response.text


def test_memory_admission_refuses_a_resident_grid_whose_estimate_does_not_fit(tmp_path, monkeypatch):
    project = resident()
    required = int(estimate(project)['estimated_memory_mb']*2**20)
    # With the estimate itself available, 80% of it cannot hold the grid: in the resolver and in Simulation.
    monkeypatch.setattr(server, 'execution_resources', lambda: dict(CPU_RECORD, host_available_bytes=required))
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=4*required, available_bytes=required))
    body = project.model_dump(mode='json')
    with workbench(tmp_path, True) as client:
        execution = client.post('/api/validate', json=body).json()['execution']
        assert not execution['resident']['fits'] and 'exceeds 80% of available host memory' in execution['resident']['reason']
        assert execution['warnings'] == ['Resident execution was requested but '+execution['resident']['reason']+'.']
        response = client.post('/api/jobs', json=body)
        assert response.status_code == 202, response.text
        job = finished(client, '/api/jobs/'+response.json()['id'])
        assert job['status'] == 'failed' and 'Insufficient available host memory' in job['error'], job
        # Auto passes over the resident rung.
        auto = dict(body, region=dict(body['region'], execution_mode='auto'))
        execution = client.post('/api/validate', json=auto).json()['execution']
        assert execution['mode'] != 'resident' and not execution['resident']['fits']


def test_memory_admission_lifts_every_server_size_limit(memory_client):
    payload = oversized()
    for field, message in SERVER_CASES.items():
        project = single_oversized(field, payload)
        response = memory_client.post('/api/validate', json=project)
        assert response.status_code == 200, (field, response.text[:300])
        # The nested project of a GDS export and a mode-network request validates too.
        for route, body in (('/api/gds/export', {'project': project, 'layers': {'s0': [1, 0]}}),
                            ('/api/mode-networks/validate', {'project': project})):
            assert message not in memory_client.post(route, json=body).text, (field, route)
    assert memory_client.post('/api/validate', json=wide_plane().model_dump(mode='json')).status_code == 200
    response = memory_client.post('/api/validate', json=wide_plane(max_monitor_samples=20_000_000).model_dump(mode='json'))
    assert response.status_code == 422 and 'exceeds the limit of 20,000,000' in response.text


# ------------------------------------------------------------------------------------------ host arrays
def test_memory_admission_refuses_a_run_whose_frames_exceed_the_host_memory(tmp_path, monkeypatch):
    """The stored frames are part of the host estimate; without them the scene would fit 80% of the host memory."""
    p = demo_project()
    p.region = Region(**{**p.region.model_dump(), 'backend': 'cpu', 'execution_mode': 'resident'})
    # 160 x 120 cells and 1000 steps: 101 stored frames outweigh the grid.
    required, frames = int(estimate(p)['estimated_memory_mb']*2**20), display_host_bytes(p)
    assert frames > required/2
    available = int((required-frames//2)/.8)
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=4*available, available_bytes=available))
    monkeypatch.setattr(server, 'execution_resources', lambda: dict(CPU_RECORD, host_available_bytes=available))
    with server_limits(memory_admission=True), pytest.raises(ValueError, match='Insufficient available host memory'):
        Simulation(p).run()
    body = p.model_dump(mode='json')
    with workbench(tmp_path, True) as client:
        execution = client.post('/api/validate', json=body).json()['execution']
        reason = execution['resident']['reason']
        assert not execution['resident']['fits'] and reason.startswith('resident estimate')
        assert 'exceeds 80% of available host memory' in reason
        assert execution['warnings'] == ['Resident execution was requested but '+reason+'.']
        job = finished(client, '/api/jobs/'+client.post('/api/jobs', json=body).json()['id'])
        assert job['status'] == 'failed' and job['error'] == 'Insufficient available host memory. Increase mesh spacing or reduce the domain.', job
        auto = dict(body, region=dict(body['region'], execution_mode='auto'))
        assert client.post('/api/validate', json=auto).json()['execution']['mode'] != 'resident'
        # Three stored frames (one every 500 steps) fit.
        fewer = dict(body, region=dict(body['region'], snapshot_interval=500))
        assert client.post('/api/validate', json=fewer).json()['execution']['resident']['fits']


def test_on_the_gpu_the_host_arrays_bound_a_grid_above_the_server_cell_limit():
    p = Project(region=Region(**{**LARGE, 'backend': 'cuda', 'execution_mode': 'resident'}), sources=[Source(center=(0, 0, 0))])
    summary, cells = estimate(p), math.prod(p.region.shape)
    host = int(summary['host_estimated_mb']*2**20)
    device = dict(CPU_RECORD, cuda=True, gpu='simulated', gpu_free_bytes=64*2**30, gpu_total_bytes=64*2**30)
    with server_limits():
        fit = _resident_fit(p, summary, 'cuda', dict(device, host_available_bytes=2**40), cells)
        assert not fit['fits'] and fit['reason'] == '8,388,608 cells exceed the resident limit of 8,000,000'
    with server_limits(memory_admission=True):
        assert _resident_fit(p, summary, 'cuda', dict(device, host_available_bytes=int(host/.8)+2**20), cells)['fits']
        fit = _resident_fit(p, summary, 'cuda', dict(device, host_available_bytes=int(host/.8)-2**20), cells)
        assert not fit['fits'] and fit['reason'].startswith('resident host estimate')


def simulate_host_memory(monkeypatch, available):
    """Every torchfdtd module reading the host memory sees available bytes, the ones that imported host_memory by name too."""
    import sys
    from torchfdtd import memory_profile
    original = memory_profile.host_memory
    for name, module in list(sys.modules.items()):
        if name.startswith('torchfdtd') and getattr(module, 'host_memory', None) is original:
            monkeypatch.setattr(module, 'host_memory', lambda: dict(total_bytes=2*available, available_bytes=available))


def test_budgeted_mode_network_and_design_routes_are_admitted_by_their_budgets(tmp_path, monkeypatch):
    """The counterpart of test_server_security::test_budgeted_mode_network_and_design_routes_keep_the_resident_cell_limit.

    With 1 TiB of host memory simulated, the byte budgets of the requests decide, on any host."""
    from test_mode_network_project import config
    from torchfdtd.periodic_design import PeriodicDesignConfig
    simulate_host_memory(monkeypatch, 2**40)
    network = config()
    network['project']['region'] = dict(network['project']['region'], size=[14., 14., 6.], mesh=.05)  # 9,408,000 cells
    network['project']['sources'][0]['size'] = [14., 14., 0.]
    network['execution'] = dict(network['execution'], resident_budget_bytes=64*2**30, network_budget_bytes=16*2**30,
                                host_budget_bytes=40*2**30)
    design = dict(PeriodicDesignConfig().model_dump(mode='json'), period_um=[6.4, 6.4], mesh_um=.025)  # 9,568,256 cells
    with workbench(tmp_path, True) as client:
        for route, body in (('/api/mode-networks/validate', network), ('/api/design/plan', dict(design, execution='resident'))):
            response = client.post(route, json=body)
            assert 'server limits resident execution' not in response.text and 'budget' in response.text, (route, response.text[:300])
        response = client.post('/api/design/plan', json=design)
        assert response.status_code == 200, response.text
        assert not any('server limits' in a['reason'] for a in response.json()['selection']['attempts'] if a.get('reason'))


# ------------------------------------------------------------------------------------ threads and workers
def test_job_threads_run_under_the_admission_of_their_request(tmp_path, monkeypatch):
    seen, release = [], threading.Event()

    class Recording:
        def __init__(self, project):
            self.cells = math.prod(project.region.shape)

        def run(self, progress, cancel):
            seen.append((server_admission(), server_limit('resident_cells'), self.cells))
            release.wait(10)
            raise ValueError('recorded')
    monkeypatch.setattr(server, 'Simulation', Recording)
    small = demo_project()
    small.region.backend = 'cpu'
    for memory_admission, project in ((False, small), (True, resident())):
        release.clear()
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            body = project.model_dump(mode='json')
            keys = [client.post('/api/jobs', json=body).json()['id'] for _ in range(3)]
            # The queue cap holds under either admission.
            response = client.post('/api/jobs', json=body)
            assert response.status_code == 409 and 'queue is full' in response.text
            release.set()
            for key in keys:
                job = finished(client, '/api/jobs/'+key)
                assert job['status'] == 'failed' and job['error'] == 'recorded', job
    cells = math.prod(small.region.shape)
    assert seen == [('fixed', SERVER_LIMITS['resident_cells'], cells)]*3+[('memory', None, 8_388_608)]*3


def test_design_jobs_run_under_the_admission_of_their_request(tmp_path, monkeypatch):
    from torchfdtd import design_service
    seen = []

    def run(config, on_progress, cancel):
        seen.append(server_admission())
        raise ValueError('recorded')
    monkeypatch.setattr(design_service, 'run_periodic_design', run)
    for memory_admission in (False, True):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            config = client.get('/api/design/defaults').json()
            key = client.post('/api/design/jobs', json=config).json()['id']
            job = finished(client, '/api/jobs/'+key)
            assert job['status'] == 'failed' and job['error'] == 'recorded', job
    assert seen == ['fixed', 'memory']


def test_the_modal_worker_receives_the_admission_of_its_request(tmp_path, monkeypatch):
    """The job thread passes the admission to the spawned worker, which applies it to the snapshot it validates and runs."""
    from torchfdtd import mode_network_project, mode_network_service as service
    from torchfdtd.mode_network_worker import mode_network_worker
    spawned, seen, sent = [], [], []

    class Unstarted:
        pid = None

        def start(self):
            raise OSError('recorded')

        def close(self):
            pass

    class Context:
        def Pipe(self, duplex):
            return Unstarted(), Unstarted()

        def Process(self, target, args, daemon):
            spawned.append(args)
            return Unstarted()

    class Connection:
        def send_bytes(self, data):
            sent.append(data)

        def close(self):
            pass

    def run(config, on_progress=None):
        seen.append((server_admission(), server_limit('structures'), len(config.project.structures)))
        raise ValueError('recorded')
    monkeypatch.setattr(service, 'mode_network_plan', lambda config: {})
    monkeypatch.setattr(service.multiprocessing, 'get_context', lambda kind: Context())
    monkeypatch.setattr(mode_network_project, 'run_mode_network', run)
    small = modal_payload()
    large = dict(small, project=dict(small['project'], structures=structures(1001)))
    for memory_admission, body in ((False, small), (True, large)):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            key = client.post('/api/mode-network-jobs', json=body).json()['id']
            job = finished(client, '/api/mode-network-jobs/'+key)
            assert job['status'] == 'failed' and job['error'] == 'recorded', job
    assert [args[3] for args in spawned] == [False, True]
    # A spawned child starts without the job thread's context; a new thread does too.
    for snapshot, output, _, memory_admission in spawned:
        worker = threading.Thread(target=mode_network_worker, args=(snapshot, output, Connection(), memory_admission))
        worker.start()
        worker.join(30)
    assert seen == [('fixed', SERVER_LIMITS['structures'], len(small['project']['structures'])), ('memory', MEMORY_ADMISSION_LIMITS['structures'], 1001)]
    assert all(b'recorded' in data for data in sent)


# ---------------------------------------------------------------------------------------- input limits
@pytest.mark.parametrize('name', ['test_foreign_host_headers_are_rejected', 'test_the_test_host_is_allowed_only_through_the_environment',
                                  'test_foreign_origins_are_rejected_on_state_changing_routes',
                                  'test_path_parameters_and_static_paths_stay_inside_their_directories',
                                  'test_upload_filename_headers_cannot_choose_the_stored_path',
                                  'test_gds_upload_names_are_display_only_and_hostile_layouts_are_rejected',
                                  'test_request_size_limits_apply_to_declared_and_chunked_bodies',
                                  'test_malformed_json_bodies_get_a_client_error_not_a_server_error',
                                  'test_gds_uploads_keep_the_vertex_limit_the_python_api_lifts'])
def test_security_checks_hold_under_memory_admission(name, tmp_path, monkeypatch, memory_app, memory_client):
    """The server security checks, run against memory-admission apps, including the apps a check builds itself."""
    create = server.create_app
    monkeypatch.setattr(server, 'create_app', lambda result_dir=None: create(result_dir, memory_admission=True))
    check = getattr(security, name)
    available = dict(app=memory_app, client=memory_client, tmp_path=tmp_path, monkeypatch=monkeypatch)
    check(**{parameter: available[parameter] for parameter in inspect.signature(check).parameters})


def test_input_limits_of_the_models_hold_under_memory_admission(memory_client):
    base = demo_project().model_dump(mode='json')
    lorentz = dict(base['materials'][1], model='lorentz', poles=[{}]*17)
    index_bound = dict(dimension='3d', size=(100., 100., 71.6), mesh=.1, pml_cells=3, execution_mode='resident')
    cases = {'at most 16 items': dict(base, materials=[base['materials'][0], lorentz, *base['materials'][2:]]),
             'at most 120 characters': dict(base, name='x'*121),
             'at most one million cells': dict(base, region=dict(base['region'], mesh=1e-9)),
             'signed 32-bit field index of the fused CUDA kernels), which 716,000,000 cells reach': dict(base, region=dict(base['region'], **index_bound))}
    for message, project in cases.items():
        response = memory_client.post('/api/validate', json=project)
        assert response.status_code == 422 and message in response.text, (message, response.text[:300])
    response = memory_client.post('/api/materials/data', json={'text': '1 1.5 0', 'reference': 'x'*2001})
    assert response.status_code == 422 and 'at most 2000 characters' in response.text


# ---------------------------------------------------------------------------------------------- scalars
def huge(field):
    """A demo project asking for 10**12 steps, frequency points or plane points; its source is src0."""
    base = demo_project().model_dump(mode='json')
    base['sources'] = [dict(s, id='src0') for s in base['sources']]
    spectrum = dict(sampling='frequency', frequency_points=10**12, wavelength_start=1.5, wavelength_stop=1.6, apodization='none')
    if field == 'steps':
        return dict(base, region=dict(base['region'], steps=10**12))
    if field == 'frequency_points':
        return dict(base, monitors=[dict(base['monitors'][0], spectrum=spectrum)])
    # 10**12 plane points: a 1e6 x 1e6-cell layer with a z-normal plane across it (the old 500 of G9 review)
    region = Region(dimension='3d', size=(1e5, 1e5, 1.2), mesh=.1, pml_cells=3, steps=10, backend='cpu')
    plane = FieldMonitor(id='plane', normal='z', center=(0, 0, 0), size=(9.9e4, 9.9e4, 0),
                         spectrum=dict(sampling='frequency', frequency_points=2, wavelength_start=1.5, wavelength_stop=1.6, apodization='none'))
    return Project(region=region, sources=[Source(id='src0', center=(0, 0, 0))], monitors=[plane]).model_dump(mode='json')


@pytest.mark.parametrize('field', ['steps', 'frequency_points', 'plane'])
def test_10_to_the_12_steps_frequencies_or_plane_points_answer_422_in_both_modes(field, tmp_path):
    body = huge(field)
    network = dict(modal_payload(), project=body)
    planning = [*((r, body) for r in ('/api/validate', '/api/jobs', '/api/mesh/preview', '/api/sources/src0/preview')),
                ('/api/mode-networks/validate', network)]
    # These plan nothing: they may accept the project, but never answer 500.
    exports = [('/api/python', body), ('/api/gds/export', {'project': body, 'layers': {'x': [1, 0]}})]
    for memory_admission in (False, True):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            for route, payload in planning+exports:
                started = time.monotonic()
                response = client.post(route, json=payload)
                # The source preview reads no monitor: under the fixed limits it previews the source of the huge plane.
                preview_only = field == 'plane' and route == '/api/sources/src0/preview' and not memory_admission
                expected = (422,) if (route, payload) in planning and not preview_only else (200, 422)
                assert response.status_code in expected, (memory_admission, field, route, response.status_code, response.text[:200])
                assert time.monotonic()-started < 20, (memory_admission, field, route)


def test_the_nyquist_check_and_the_counts_need_no_frequency_array():
    import random
    from torchfdtd.models import SpectrumSettings
    from torchfdtd.spectra import frequency_count, frequency_samples, highest_frequency
    rng = random.Random(1)
    for _ in range(600):
        a = rng.uniform(.2, 3)
        b = a+rng.choice([0, rng.uniform(1e-6, 3)])
        spectrum = SpectrumSettings(sampling=rng.choice(['frequency', 'wavelength', 'chebyshev']), wavelength_start=a, wavelength_stop=b,
                                    frequency_points=1 if a == b else rng.choice([1, 2, 3, rng.randint(2, 5000)]),
                                    chebyshev_nodes=rng.choice(['roots', 'lobatto']), chebyshev_wavelength=rng.random() < .5)
        samples = frequency_samples(spectrum)
        assert highest_frequency(spectrum) == samples.max() and frequency_count(spectrum) == len(samples), spectrum
    custom = SpectrumSettings(sampling='custom', custom_frequencies_hz=[1e14, 2e14, 3e14])
    assert highest_frequency(custom) == 3e14 and frequency_count(custom) == 3
    # The Python API validates 10**12 frequency points without building them.
    started = time.monotonic()
    Project.model_validate(huge('frequency_points'))
    assert time.monotonic()-started < 5


def _planning_peak(body):
    """Peak host bytes of validation, estimate() and resolve_plan, from tracemalloc."""
    import gc
    import tracemalloc
    from torchfdtd.plan import resolve_plan
    gc.collect()
    tracemalloc.start()
    try:
        with server_limits(memory_admission=True):
            project = Project.model_validate(body)
            estimate(project)
            resolve_plan(project)
        return tracemalloc.get_traced_memory()[1], project
    finally:
        tracemalloc.stop()


def test_preadmission_bytes_bound_what_planning_holds():
    from torchfdtd import BoundaryFace
    from torchfdtd.solver import preadmission_bytes
    base = demo_project().model_dump(mode='json')
    spectrum = dict(sampling='chebyshev', frequency_points=300_000, wavelength_start=1.5, wavelength_stop=1.6, apodization='none')
    bloch = {f'{a}_{s}': BoundaryFace(kind='bloch') for a in 'xy' for s in ('min', 'max')}
    cases = {
        'steps': dict(base, region=dict(base['region'], steps=300_000)),
        'terms': dict(base, region=dict(base['region'], steps=200_000), sources=[dict(base['sources'][0], id=f's{i}') for i in range(5)]),
        'frequencies': dict(base, monitors=[dict(base['monitors'][0], spectrum=spectrum)]),
        'items': dict(base, structures=[{'id': f'q{i}'} for i in range(3000)], monitors=[{'id': f'm{i}'} for i in range(2000)]),
        'plane': Project(region=Region(dimension='3d', size=(3.4, 3.4, 1.), mesh=.01, pml_cells=4, steps=100, backend='cpu'),
                         sources=[Source(center=(0, 0, 0))],
                         monitors=[FieldMonitor(id='plane', normal='z', center=(0, 0, 0), size=(3., 3., 0),
                                                spectrum=dict(sampling='frequency', frequency_points=2, wavelength_start=1.5,
                                                              wavelength_stop=1.6, apodization='none'))]).model_dump(mode='json'),
        'oneway': Project(region=Region(dimension='2d', size=(4., 2., 1.), mesh=.1, pml_cells=5, steps=100_000, backend='cpu',
                                        boundaries={'y_min': BoundaryFace(kind='periodic'), 'y_max': BoundaryFace(kind='periodic')}),
                          sources=[Source(id='ow', kind='plane', injection='oneway', normal='x', center=(-1., 0., 0.), size=(0., 2., 0.),
                                          component='Ez')]).model_dump(mode='json'),
        'tfsf': Project(region=Region(dimension='2d', size=(4., 4., 1.), mesh=.1, pml_cells=5, steps=200_000, backend='cpu'),
                        sources=[Source(id='box', kind='tfsf', normal='x', center=(0., 0., 0.), size=(2., 2., 1.), component='Ez')]).model_dump(mode='json'),
        'bloch sheet': Project(region=Region(dimension='3d', size=(3.2, 3.2, 1.), mesh=.01, pml_cells=5, steps=100, backend='cpu',
                                             material_sampling='yee', boundaries=bloch, bloch_phase=(.5, .3, 0.)),
                               sources=[Source(id='sheet', kind='plane', normal='z', center=(0., 0., -.2), size=(3.2, 3.2, 0.),
                                               component='Ex')]).model_dump(mode='json')}
    for name, body in cases.items():
        peak, project = _planning_peak(body)
        assert peak <= preadmission_bytes(project), (name, peak, preadmission_bytes(project))


def test_the_preadmission_check_refuses_before_anything_is_planned(tmp_path, monkeypatch):
    from torchfdtd.solver import preadmission_bytes
    base = demo_project().model_dump(mode='json')
    # 10,000 monitors of 999,999 frequency points each: every list and value within its ceiling, 450 GiB to plan.
    spectrum = dict(sampling='frequency', frequency_points=999_999, wavelength_start=1.5, wavelength_stop=1.6, apodization='none')
    many = dict(base, monitors=[dict(base['monitors'][0], id=f'm{i}', spectrum=spectrum) for i in range(10_000)])
    with workbench(tmp_path / 'memory', True) as client:
        response = client.post('/api/validate', json=many)
        assert response.status_code == 422 and 'Planning this project would take' in response.text, response.text[:300]
    # Memory admission compares the planning bytes with the available host memory where planning starts.
    # The fixed limits bound planning on their own: the check does not apply there, nor in model validation.
    with server_limits():
        needed = preadmission_bytes(Project.model_validate(base))
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2*needed, available_bytes=needed))
    for memory_admission, status in ((False, 200), (True, 422)):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            for route in ('/api/validate', '/api/jobs', '/api/mesh/preview', '/api/sources/src0/preview'):
                response = client.post(route, json=dict(base, sources=[dict(base['sources'][0], id='src0')]))
                assert response.status_code == status or (status == 200 and response.status_code == 202), (route, response.text[:300])
                assert status == 200 or 'above 80% of the' in response.text, response.text[:300]
            response = client.post('/api/python', json=base)
            assert response.status_code == 200
    with server_limits(memory_admission=True):
        Project.model_validate(base)   # validation alone never reads the free memory


# ---------------------------------------------------------------------------------------- list ceilings
CEILING_ITEMS = {'structures': lambda n: {'structures': [{} for _ in range(n)]},
                 'sources': lambda n: {'sources': [{} for _ in range(n)]},
                 'monitors': lambda n: {'monitors': [{} for _ in range(n)]},
                 'materials': lambda n: {'materials': [*demo_project().model_dump(mode='json')['materials'], *({'name': str(i)} for i in range(n))]},
                 'mesh_refinements': lambda n: {'region': {'mesh_refinements': [{} for _ in range(n)]}}}
# Numbers in lists: host bytes per request byte, measured below; polygon vertices are the densest.
NUMBER_ITEMS = {'vertices': lambda n: {'structures': [{'kind': 'polygon', 'vertices': [[i % 7, i % 5] for i in range(n)]}]},
                'custom_frequencies': lambda n: {'monitors': [{'spectrum': {'sampling': 'custom', 'custom_frequencies_hz': list(range(1, n+1))}}]},
                'signal_samples': lambda n: {'sources': [{'pulse': 'sampled', 'use_global_source': False,
                                                          'signal': {'time_s': list(range(n)), 'amplitude': [1]*n, 'phase_rad': [0]*n}}]}}
ITEM_BYTES = 3*1024
NUMBER_BYTES_PER_REQUEST_BYTE = 80
VALIDATION_BOUND = 3.3e9


def _request(parts):
    import json
    base = demo_project().model_dump(mode='json')
    for key, value in parts.items():
        base[key] = dict(base[key], **value) if isinstance(value, dict) else value
    return json.dumps(base, separators=(',', ':'))


def _request_peak(client, text):
    import gc
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    try:
        response = client.post('/api/validate', content=text, headers={'content-type': 'application/json'})
        return response, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize('kind', list(CEILING_ITEMS))
def test_memory_admission_keeps_each_list_ceiling_before_the_items_are_validated(kind, memory_client, monkeypatch):
    ceiling = MEMORY_ADMISSION_LIMITS[kind]
    extra = len(demo_project().materials) if kind == 'materials' else 0
    response, peak = _request_peak(memory_client, _request(CEILING_ITEMS[kind](ceiling+1-extra)))
    assert response.status_code == 422 and f'exceed the limit of {ceiling:,}' in response.text, response.text[:200]
    # Refused before the items are validated (the count check runs first): 200,001 structures would take 230 MB.
    assert kind != 'structures' or peak < 64*2**20, peak
    # Below the ceiling the items are validated; the pre-check then refuses, so only validation is measured.
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2**20, available_bytes=1))
    n = min(2000, ceiling//4)
    (_, low), (_, high) = (_request_peak(memory_client, _request(CEILING_ITEMS[kind](count))) for count in (n, 3*n))
    assert (high-low)/(2*n) <= ITEM_BYTES, (kind, (high-low)/(2*n))


@pytest.mark.parametrize('kind', list(NUMBER_ITEMS))
def test_numbers_validate_within_the_stated_bytes_per_request_byte(kind, memory_client, monkeypatch):
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2**20, available_bytes=1))
    texts = [_request(NUMBER_ITEMS[kind](n)) for n in (100_000, 300_000)]
    (_, low), (_, high) = (_request_peak(memory_client, text) for text in texts)
    assert (high-low)/(len(texts[1])-len(texts[0])) <= NUMBER_BYTES_PER_REQUEST_BYTE, (kind, (high-low)/(len(texts[1])-len(texts[0])))


def test_the_stated_validation_bound_covers_a_full_request():
    items = sum(MEMORY_ADMISSION_LIMITS[kind] for kind in CEILING_ITEMS)
    assert ITEM_BYTES*items+NUMBER_BYTES_PER_REQUEST_BYTE*server.MAX_REQUEST_BYTES <= VALIDATION_BOUND


# ----------------------------------------------------------------------------------------- result JSON
def _plane_and_trace_project():
    region = Region(dimension='3d', size=(2.4, 2.4, 1.2), mesh=.05, pml_cells=4, steps=400, backend='cpu')
    plane = FieldMonitor(id='plane', normal='z', center=(0, 0, 0), size=(1.4, 1.4, 0),
                         spectrum=dict(sampling='frequency', frequency_points=3, wavelength_start=1.5, wavelength_stop=1.6, apodization='none'))
    return Project(region=region, sources=[Source(center=(0, 0, .2))], monitors=[Monitor(id='probe', center=(.1, 0, 0)), plane])


MONITOR_SERIES = ('time_fs', 'signal', 'signal_imag', 'window', 'frequency_thz', 'wavelength_um', 'spectrum', 'spectrum_real', 'spectrum_imag')


@pytest.mark.parametrize('memory_admission', [False, True])
def test_result_routes_send_bounded_json(memory_admission, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'FIELD_JSON_POINTS', 100)
    monkeypatch.setattr(server, 'MONITOR_JSON_VALUES', 300)
    body = _plane_and_trace_project().model_dump(mode='json')
    with workbench(tmp_path, memory_admission) as client:
        key = client.post('/api/jobs', json=body).json()['id']
        assert finished(client, '/api/jobs/'+key, 120)['status'] == 'completed'
        plane = client.get(f'/api/jobs/{key}/field-monitors/plane', params=dict(component='Ez')).json()
        assert plane['full_shape'] == [28, 28] and plane['stride'] == [3, 3] and plane['shape'] == [10, 10]
        assert len(plane['real']) == 10 and len(plane['points_um']) == 100
        status = client.get('/api/jobs/'+key).json()
        probe, = status['monitors']
        assert sum(len(probe[k]) for k in MONITOR_SERIES) <= 300 and probe['spectrum_stride'] > 1 and probe['trace_stride'] > 1
        # The CSV exports come from the saved result: every one of the 200 FFT bins above DC of 400 steps,
        # and the traces decimated as always (400 samples, under the 2000 of Result.monitor_data).
        assert len(client.get(f'/api/jobs/{key}/spectra.csv').text.strip().splitlines()) == 1+200
        assert len(client.get(f'/api/jobs/{key}/monitors.csv').text.strip().splitlines()) == 1+400


def test_the_stored_series_budget_is_a_total_over_every_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'MONITOR_JSON_VALUES', 9000)
    monitors = [Monitor(id=f'p{i}', center=(.02*i-.4, .1, 0)) for i in range(40)]
    body = Project(region=Region(dimension='2d', size=(2., 2., 1.), mesh=.05, pml_cells=4, steps=3000, backend='cpu'),
                   sources=[Source(center=(0, 0, 0))], monitors=monitors).model_dump(mode='json')
    with workbench(tmp_path, True) as client:
        key = client.post('/api/jobs', json=body).json()['id']
        assert finished(client, '/api/jobs/'+key, 300)['status'] == 'completed'
        stored = client.get('/api/jobs/'+key).json()['monitors']
        assert len(stored) == 40 and sum(len(m[k]) for m in stored for k in MONITOR_SERIES) <= 9000
        # spectra.csv streams every sample of every monitor from the saved result, with the stride of none.
        rows = client.get(f'/api/jobs/{key}/spectra.csv').text.strip().splitlines()
        assert len(rows)-1 == 40*1500


def test_flux_json_is_strided_and_the_job_list_names_planes_only(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'MONITOR_JSON_VALUES', 600)
    plane = FieldMonitor(id='flux', normal='x', center=(.4, 0, 0), size=(0, 1., 1.),
                         spectrum=dict(sampling='frequency', frequency_points=500, wavelength_start=1.3, wavelength_stop=1.8, apodization='none'))
    body = Project(region=Region(dimension='2d', size=(2., 2., 1.), mesh=.05, pml_cells=4, steps=400, backend='cpu'),
                   sources=[Source(center=(0, 0, 0))], monitors=[plane]).model_dump(mode='json')
    for memory_admission in (False, True):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            key = client.post('/api/jobs', json=body).json()['id']
            assert finished(client, '/api/jobs/'+key, 120)['status'] == 'completed'
            flux, = client.get('/api/jobs/'+key).json()['flux_monitors']
            assert flux['flux_stride'] == 3 and len(flux['flux']) == len(flux['frequency_thz']) == 167
            listed, = client.get('/api/jobs').json()
            assert listed['flux_monitors'] == [{'id': 'flux', 'name': flux['name'], 'normal': flux['normal']}]
            assert len(client.get(f'/api/jobs/{key}/flux.csv').text.strip().splitlines()) == 1+500


def test_the_plane_stride_is_chosen_per_axis():
    assert server._plane_strides((28, 28), 100) == (3, 3)
    assert server._plane_strides((3, 10**6), 512*512) == (1, 12)
    assert server._plane_strides((10**6, 3), 512*512) == (12, 1)
    assert server._plane_strides((200, 300), 512*512) == (1, 1)
    for shape in ((3, 10**6), (700, 900), (5, 7000), (1, 10**6), (2000, 2000, 1)):
        strides = server._plane_strides(shape, 512*512)
        counts = [-(-n//s) for n, s in zip(shape, strides)]
        assert math.prod(counts) <= 512*512, (shape, counts)
    counts = [-(-n//s) for n, s in zip((3, 10**6), server._plane_strides((3, 10**6), 512*512))]
    assert math.prod(counts) > 250_000


def test_stored_projects_answer_without_reading_the_free_memory(tmp_path, monkeypatch):
    """A finished job's project is validated again by the post-processing routes; the planning check is not part of it."""
    body = _plane_and_trace_project().model_dump(mode='json')
    for memory_admission in (False, True):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            key = client.post('/api/jobs', json=body).json()['id']
            assert finished(client, '/api/jobs/'+key, 120)['status'] == 'completed'
            with monkeypatch.context() as scarce:
                scarce.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2**20, available_bytes=1))
                for route in ('farfield-monitors', 'propagation-monitors', 'diffraction-monitors'):
                    response = client.get(f'/api/jobs/{key}/{route}')
                    assert response.status_code == 200, (memory_admission, route, response.text[:200])


def test_a_finished_job_can_be_deleted_to_release_its_results(tmp_path, monkeypatch):
    release = threading.Event()

    class Blocking:
        def __init__(self, project):
            pass

        def run(self, progress, cancel):
            release.wait(10)
            raise ValueError('released')
    small = demo_project()
    small.region.backend = 'cpu'
    small.region.steps = 40
    with workbench(tmp_path, True) as client:
        key = client.post('/api/jobs', json=small.model_dump(mode='json')).json()['id']
        assert finished(client, '/api/jobs/'+key)['status'] == 'completed'
        assert (tmp_path/f'{key}.npz').exists()
        assert client.delete('/api/jobs/'+key, headers={'Origin': 'https://evil.example'}).status_code == 403
        assert client.delete('/api/jobs/'+key).json() == {'id': key, 'deleted': True}
        assert client.get('/api/jobs/'+key).status_code == 404 and not (tmp_path/f'{key}.npz').exists()
        assert client.delete('/api/jobs/'+key).status_code == 404
        monkeypatch.setattr(server, 'Simulation', Blocking)
        running = client.post('/api/jobs', json=small.model_dump(mode='json')).json()['id']
        assert client.delete('/api/jobs/'+running).status_code == 409
        release.set()
        assert finished(client, '/api/jobs/'+running)['status'] == 'failed'
        assert client.delete('/api/jobs/'+running).status_code == 200


# ------------------------------------------------------------------------------------------ previews
def test_the_source_preview_covers_at_most_its_window(monkeypatch):
    from torchfdtd import BoundaryFace, source_preview
    monkeypatch.setattr(source_preview, 'PREVIEW_STEPS', 500)
    point = demo_project()
    point.region.steps = 2000
    tfsf = Project(region=Region(dimension='2d', size=(4., 4., 1.), mesh=.1, pml_cells=5, steps=2000, backend='cpu'),
                   sources=[Source(id='box', kind='tfsf', normal='x', center=(0., 0., 0.), size=(2., 2., 1.), component='Ez')])
    oneway = Project(region=Region(dimension='2d', size=(4., 2., 1.), mesh=.1, pml_cells=5, steps=2000, backend='cpu',
                                   boundaries={'y_min': BoundaryFace(kind='periodic'), 'y_max': BoundaryFace(kind='periodic')}),
                     sources=[Source(id='ow', kind='plane', injection='oneway', normal='x', center=(-1., 0., 0.), size=(0., 2., 0.), component='Ez')])
    for project, source in ((point, point.sources[0].id), (tfsf, 'box'), (oneway, 'ow')):
        out = source_preview.preview_source(project, source)
        assert out['steps'] == 2000 and out['preview_steps'] == 500 and len(out['signal']) == len(out['time_fs']) == 500
        assert all(len(i['signal']) == 500 for i in out['injections']) and 'first 500 of the 2,000 steps' in out['note']
        assert project.region.steps == 2000
    point.region.steps = 400
    out = source_preview.preview_source(point, point.sources[0].id)
    assert out['preview_steps'] == 400 and len(out['signal']) == 400 and 'first' not in out['note']


# ------------------------------------------------------------------------------------------ host estimate
def test_the_host_estimate_counts_the_structures_beyond_the_covered_ones():
    from torchfdtd.solver import HOST_COVERED_STRUCTURES, HOST_STRUCTURE_BYTES
    base = demo_project().model_dump(mode='json')
    def project(n, backend):
        return Project.model_validate(dict(base, region=dict(base['region'], backend=backend),
                                           structures=[dict(base['structures'][0], id=f's{i}') for i in range(n)]))
    for backend, key in (('cuda', 'host_estimated_mb'), ('cpu', 'estimated_memory_mb')):
        low, high = (estimate(project(n, backend))[key] for n in (HOST_COVERED_STRUCTURES, HOST_COVERED_STRUCTURES+2000))
        assert high-low == pytest.approx(2000*HOST_STRUCTURE_BYTES/2**20, abs=.2), (backend, high-low)
    # The plan keeps no more per structure than the estimate counts.
    import gc
    import tracemalloc
    from torchfdtd.plan import resolve_plan
    kept = []
    for n in (1000, 3000):
        p = project(n, 'cpu')
        gc.collect()
        tracemalloc.start()
        plan = resolve_plan(p)
        kept.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        del plan
    assert (kept[1]-kept[0])/2000 <= HOST_STRUCTURE_BYTES


# ------------------------------------------------------------------------------------- design state paths
def test_the_design_state_directory_stays_inside_the_server_root(tmp_path):
    from torchfdtd.periodic_design import PeriodicDesignConfig
    config = PeriodicDesignConfig().model_dump(mode='json')
    for memory_admission in (False, True):
        root = tmp_path / str(memory_admission)
        with workbench(root, memory_admission) as client:
            for path in [str(tmp_path / 'outside'), 'C:/Windows/Temp', '\\\\server\\share', '/tmp/x', '../escape', 'runs/../../escape',
                         'C:relative', 'a:b', '.', 'with\0nul', 'tab\there', 'x'*300, 'runs\\first', '//server/share/x']:
                for route in ('/api/design/config', '/api/design/plan', '/api/design/jobs'):
                    response = client.post(route, json=dict(config, state_directory=path, disk_budget_gib=1))
                    assert response.status_code == 422 and 'design state directory' in response.text, (path, route, response.text[:200])
            assert not (tmp_path / 'outside').exists() and not (tmp_path / 'escape').exists()
            response = client.post('/api/design/plan', json=dict(config, state_directory='runs/first', disk_budget_gib=1))
            assert response.status_code == 200, response.text[:300]
            assert (root / 'design-state' / 'runs' / 'first').is_dir()


# --------------------------------------------------------------------------------------------- documents
def test_documents_state_the_fixed_limits_as_the_default_only():
    from pathlib import Path
    from torchfdtd.models import ProjectLimits
    docs = Path(__file__).resolve().parents[1] / 'docs'
    for name in ('ADJOINT_MEMORY_PLAN.md', 'EXECUTION_MODES.md', 'STREAMED_FDTD.md', 'SECURITY.md'):
        for paragraph in (docs / name).read_text(encoding='utf-8').split('\n\n'):
            if '8,000,000' in paragraph and 'server' in paragraph and not paragraph.startswith('|'):
                assert 'memory-admission' in paragraph or 'memory admission' in paragraph, (name, paragraph[:200])
    assert 'memory admission' in ProjectLimits.__doc__


# ------------------------------------------------------------------------------- fixed planning limits
def _bloch_sheet_project(side_um, mesh=.01):
    """A 3D Bloch cell whose z-normal sheet source spans (side_um/mesh)**2 cells."""
    from torchfdtd import BoundaryFace
    bloch = {f'{a}_{s}': BoundaryFace(kind='bloch') for a in 'xy' for s in ('min', 'max')}
    region = Region(dimension='3d', size=(side_um, side_um, 1.2), mesh=mesh, pml_cells=5, steps=100, backend='cpu',
                    material_sampling='yee', boundaries=bloch, bloch_phase=(.5, .3, 0.))
    return Project(region=region, sources=[Source(id='sheet', kind='plane', normal='z', center=(0., 0., -.2),
                                                  size=(side_um, side_um, 0.), component='Ex')]).model_dump(mode='json')


def _many_planes_project(planes, side=9.9):
    """planes z-normal planes of side/mesh points squared (990 x 990 by default), 2 frequencies and 6 components
    each: every default plane (11,761,200 samples) under the per-plane limit, 5,880,600 points x components each."""
    region = Region(dimension='3d', size=(10.2, 10.2, 2.2), mesh=.01, pml_cells=4, steps=100, backend='cpu')
    spectrum = dict(sampling='frequency', frequency_points=2, wavelength_start=1.5, wavelength_stop=1.6, apodization='none')
    monitors = [FieldMonitor(id=f'plane{i}', normal='z', center=(0, 0, round(-.99+.02*i, 2)), size=(side, side, 0), spectrum=spectrum)
                for i in range(planes)]
    return Project(region=region, sources=[Source(id='src0', center=(0, 0, 0))], monitors=monitors).model_dump(mode='json')


@pytest.mark.parametrize('case', ['bloch sheet of 10**8 cells', '100 planes of 5.9e8 points x components'])
def test_fixed_limits_count_bloch_sheets_and_planes_before_building_them(case, tmp_path, monkeypatch):
    from torchfdtd.solver import planning_counts
    body, source = ((_bloch_sheet_project(100.), 'sheet') if case.startswith('bloch') else (_many_planes_project(100), 'src0'))
    name, what = ('sheet_cells', 'Bloch source sheet cells') if case.startswith('bloch') else ('plane_points', 'plane points x recorded components')
    with server_limits():
        count = planning_counts(Project.model_validate(body))[name]
    assert 10**8 <= count <= 10**9
    message = f'{count:,} {what} exceed the limit of {SERVER_LIMITS[name]:,}'
    # With 4 GiB free, memory admission refuses the same projects by their planning memory.
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2**33, available_bytes=2**32))
    for memory_admission in (False, True):
        with workbench(tmp_path / str(memory_admission), memory_admission) as client:
            for route in ('/api/validate', '/api/jobs', '/api/mesh/preview', f'/api/sources/{source}/preview'):
                started = time.monotonic()
                response, peak = _request_peak(client, __import__('json').dumps(body)) if route == '/api/validate' else (client.post(route, json=body), 0)
                assert response.status_code == 422, (memory_admission, route, response.text[:300])
                assert (message if not memory_admission else 'Planning this project would take') in response.text, response.text[:300]
                assert time.monotonic()-started < 20 and peak < 64*2**20, (route, time.monotonic()-started, peak)


def test_fixed_limits_admit_sheets_and_planes_within_them_whatever_the_free_memory(tmp_path, monkeypatch):
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: dict(total_bytes=2**20, available_bytes=1))
    with workbench(tmp_path, False) as client:
        assert client.post('/api/validate', json=_bloch_sheet_project(1.6)).status_code == 200
        assert client.post('/api/sources/sheet/preview', json=_bloch_sheet_project(1.6)).status_code == 200
        response = client.post('/api/validate', json=_many_planes_project(20, side=1.))   # 20 planes of 100 x 100 points
        assert response.status_code == 200, response.text[:300]


# --------------------------------------------------------------------------------------- result files
def test_monitors_csv_reads_the_traces_once(tmp_path, monkeypatch):
    import numpy
    loads = []
    real_load = numpy.load

    class Counting:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            self.archive.__enter__()
            return self

        def __exit__(self, *exc):
            return self.archive.__exit__(*exc)

        def __getitem__(self, name):
            loads.append(name)
            return self.archive[name]

    monkeypatch.setattr(numpy, 'load', lambda *a, **k: Counting(real_load(*a, **k)))
    body = Project(region=Region(dimension='2d', size=(2., 2., 1.), mesh=.05, pml_cells=4, steps=300, backend='cpu'),
                   sources=[Source(center=(0, 0, 0))],
                   monitors=[Monitor(id=f'p{i}', component=c, center=(.1*i, .1, 0)) for i, c in enumerate(('Ez', 'Hx', 'Hy'))]).model_dump(mode='json')
    with workbench(tmp_path, False) as client:
        key = client.post('/api/jobs', json=body).json()['id']
        assert finished(client, '/api/jobs/'+key, 120)['status'] == 'completed'
        stored = client.get('/api/jobs/'+key).json()['monitors']
        loads.clear()
        rows = client.get(f'/api/jobs/{key}/monitors.csv').text.strip().splitlines()
    assert loads.count('signals') == 1 and len(rows) == 1+3*300
    # The rows are the stored traces (below the JSON budget, so not strided), monitor after monitor.
    expected = [f"{m['name']},{m['component']},{t},{s},{i}" for m in stored for t, s, i in zip(m['time_fs'], m['signal'], m['signal_imag'])]
    assert rows[1:] == expected


def test_a_job_whose_result_file_is_in_use_is_kept_on_delete(tmp_path, monkeypatch):
    import pathlib
    small = demo_project()
    small.region.backend = 'cpu'
    small.region.steps = 40
    with workbench(tmp_path, False) as client:
        key = client.post('/api/jobs', json=small.model_dump(mode='json')).json()['id']
        assert finished(client, '/api/jobs/'+key)['status'] == 'completed'
        real_unlink = pathlib.Path.unlink

        def unlink(path, missing_ok=False):
            if path.name == f'{key}.npz':
                raise PermissionError(13, 'The process cannot access the file because it is being used by another process')
            return real_unlink(path, missing_ok=missing_ok)
        with monkeypatch.context() as busy:
            busy.setattr(pathlib.Path, 'unlink', unlink)
            response = client.delete('/api/jobs/'+key)
            assert response.status_code == 409 and 'in use' in response.text
        assert client.get('/api/jobs/'+key).status_code == 200 and (tmp_path/f'{key}.npz').exists()
        assert client.delete('/api/jobs/'+key).status_code == 200 and not (tmp_path/f'{key}.npz').exists()


def test_the_job_status_is_rendered_in_its_route_within_a_400k_value_budget(tmp_path):
    from fastapi.responses import JSONResponse
    assert server.MONITOR_JSON_VALUES <= 400_000
    app = server.create_app(tmp_path)
    try:
        route = next(r for r in app.routes if getattr(r, 'path', None) == '/api/jobs/{key}' and 'GET' in r.methods)
        app.state.jobs['k'] = dict(id='k', status='completed', cancel=threading.Event(), project={'name': 'x'},
                                   monitors=[dict(id='p', time_fs=[0., 1.], signal=[0., 1.])], frames=None)
        response = route.endpoint('k')
        # A rendered response: the synchronous route encodes it in the threadpool, off the event loop.
        assert isinstance(response, JSONResponse)
        assert __import__('json').loads(response.body) == dict(id='k', status='completed', project={'name': 'x'},
                                                             monitors=[dict(id='p', time_fs=[0., 1.], signal=[0., 1.])],
                                                             cancel_requested=False)
    finally:
        app.state.pool.shutdown()
        app.state.fsp_pool.shutdown()
