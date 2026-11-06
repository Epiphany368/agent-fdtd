"""G9-01: the loopback workbench refuses remote binds, foreign origins and hosts,
path escapes, oversized or undeclared bodies and hostile JSON, NPZ and GDS inputs.

No simulation is run; every check is a request against the FastAPI app or a
Python-side loader on a synthetic file. The threat model is in docs/SECURITY.md.
"""
import ast
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from torchfdtd import cli, fsp, server
from torchfdtd.models import Project, demo_project
from torchfdtd.solver import Result

PACKAGE = Path(server.__file__).parent
FOREIGN_HOSTS = ['0.0.0.0', '192.168.0.5', '::', 'fdtd.example', '10.0.0.1']
ESCAPES = ['../pyproject.toml', '..%2F..%2Fpyproject.toml', '..%5C..%5Cpyproject.toml',
           'C:%5CWindows%5Cwin.ini', '%5C%5Cserver%5Cshare%5Cwin.ini', '%2Fetc%2Fpasswd']


@pytest.fixture
def app(tmp_path):
    application = server.create_app(tmp_path / 'results')
    yield application
    application.state.pool.shutdown()
    application.state.fsp_pool.shutdown()


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def post_json(client, path, payload, **kwargs):
    return client.post(path, content=payload.encode() if isinstance(payload, str) else payload,
                       headers={'content-type': 'application/json', **kwargs.pop('headers', {})}, **kwargs)


def test_serve_binds_loopback_by_default_and_refuses_other_hosts(monkeypatch):
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kw: calls.append(kw))
    monkeypatch.setattr(server, 'create_app', lambda **kwargs: object())
    monkeypatch.setattr(sys, 'argv', ['torchfdtd', 'serve', '--port', '8123'])
    cli.main()
    assert calls == [{'host': '127.0.0.1', 'port': 8123}]
    monkeypatch.setattr(sys, 'argv', ['torchfdtd', 'serve', '--host', 'localhost'])
    cli.main()
    assert calls[-1] == {'host': 'localhost', 'port': 8765}
    for host in FOREIGN_HOSTS + ['::1']:
        monkeypatch.setattr(sys, 'argv', ['torchfdtd', 'serve', '--host', host])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
    assert len(calls) == 2


def test_foreign_host_headers_are_rejected(client):
    # /api/capabilities is a file read, so the check never touches a GPU.
    for host in ['localhost:8765', '127.0.0.1:8765', 'localhost', '127.0.0.1']:
        assert client.get('/api/capabilities', headers={'Host': host}).status_code == 200, host
    # '[::1]' is loopback and on the allowlist; whether '[::1]:8765' matches depends on the Starlette version's IPv6
    # host parsing (older releases split on the first colon), so it is not asserted either way here.
    for host in FOREIGN_HOSTS + ['evil.example:8765', '127.0.0.1.evil.example']:
        response = client.get('/api/capabilities', headers={'Host': host})
        assert response.status_code == 400, (host, response.text)
        assert 'features' not in response.text


def test_the_test_host_is_allowed_only_through_the_environment(tmp_path, monkeypatch):
    """Without TORCHFDTD_ALLOWED_HOSTS the allowlist is loopback only: the TestClient default host, testserver, answers 400."""
    monkeypatch.delenv('TORCHFDTD_ALLOWED_HOSTS', raising=False)
    application = server.create_app(tmp_path / 'default')
    try:
        with TestClient(application, raise_server_exceptions=False) as c:
            assert c.get('/api/capabilities').status_code == 400
            assert c.get('/api/capabilities', headers={'Host': 'testserver'}).status_code == 400
            for host in ['localhost', '127.0.0.1', 'localhost:8765']:
                assert c.get('/api/capabilities', headers={'Host': host}).status_code == 200, host
    finally:
        application.state.pool.shutdown()
        application.state.fsp_pool.shutdown()
    monkeypatch.setenv('TORCHFDTD_ALLOWED_HOSTS', 'testserver')
    application = server.create_app(tmp_path / 'tests')
    try:
        with TestClient(application, raise_server_exceptions=False) as c:
            assert c.get('/api/capabilities').status_code == 200
            for host in FOREIGN_HOSTS:
                assert c.get('/api/capabilities', headers={'Host': host}).status_code == 400, host
    finally:
        application.state.pool.shutdown()
        application.state.fsp_pool.shutdown()


def test_foreign_origins_are_rejected_on_state_changing_routes(app, client):
    project = demo_project().model_dump()
    routes = ['/api/validate', '/api/jobs', '/api/mesh/preview', '/api/mesh/freeze', '/api/python',
              '/api/materials/data', '/api/design/config', '/api/design/jobs', '/api/mode-networks/validate',
              '/api/mode-network-jobs', '/api/gds/inspect', '/api/fsp/native-import', '/api/jobs/none/cancel']
    for origin in ['https://evil.example', 'http://testserver.evil.example', 'http://localhost:8765', 'null']:
        for route in routes:
            response = client.post(route, json=project, headers={'Origin': origin})
            assert response.status_code == 403, (origin, route, response.status_code)
    assert app.state.jobs == {}
    assert client.post('/api/validate', json=project, headers={'Origin': 'http://testserver'}).status_code == 200
    assert client.post('/api/validate', json=project).status_code == 200


def test_path_parameters_and_static_paths_stay_inside_their_directories(client):
    project = demo_project().model_dump()
    conversion = {'project': project, 'cell': 'TOP',
                  'layers': [{'layer': 1, 'datatype': 0, 'z_min': 0, 'z_max': .1, 'material': project['materials'][0]['name']}]}
    for key in ESCAPES:
        for route in [f'/api/jobs/{key}', f'/api/jobs/{key}/download', f'/api/jobs/{key}/fields',
                      f'/api/jobs/{key}/monitors.csv', f'/api/fsp/{key}', f'/api/fsp/{key}/download',
                      f'/api/fsp/{key}/archive', f'/api/fsp/{key}/conversion', f'/api/design/jobs/{key}/download',
                      f'/api/mode-network-jobs/{key}/download', f'/api/examples/{key}']:
            response = client.get(route)
            # An unencoded ../ is normalized away by the client, which lands on
            # a route without the method or without the resource.
            assert response.status_code in (404, 405), (route, response.status_code)
            assert b'[project]' not in response.content and b'[fonts]' not in response.content
        response = client.post(f'/api/gds/{key}/convert', json=conversion)
        assert response.status_code in (404, 405), (key, response.status_code)
    assert client.get('/').status_code == 200
    asset = next(p for p in (Path(server.__file__).parent / 'web' / 'assets').iterdir() if p.suffix == '.css')
    assert client.get('/assets/' + asset.name).status_code == 200
    # A drive letter or UNC prefix must never reach os.path.join, where it would
    # replace the web directory and let realpath probe a drive or a network share.
    for path in ['/../pyproject.toml', '/%2e%2e/pyproject.toml', '/assets/../../pyproject.toml',
                 '/..%5Cpyproject.toml', '/C:/Windows/win.ini', '/C:%5CWindows%5Cwin.ini', '/assets/C:/Windows/win.ini',
                 '/%5C%5Cserver%5Cshare%5Cwin.ini', '//server/share/win.ini', '/index.html::$DATA',
                 '/../../.git/config', '/assets/..%2F..%2FLICENSE']:
        response = client.get(path)
        assert response.status_code == 404, (path, response.status_code)
        assert b'[project]' not in response.content and b'MIT License' not in response.content and b'[fonts]' not in response.content


def test_upload_filename_headers_cannot_choose_the_stored_path(tmp_path, client):
    root = tmp_path / 'results'
    names = ['../../escape.fsp', '..\\..\\escape.fsp', 'C:\\Windows\\escape.fsp',
             '\\\\server\\share\\escape.fsp', '/etc/escape.fsp', '%2e%2e%2Fescape.fsp', '..%5Cescape.fsp']
    keys = []
    for name in names:
        response = client.post('/api/fsp/native-import', content=fsp.HEADER + b'1.1\0fixture', headers={'x-filename': name})
        assert response.status_code == 202, (name, response.text)
        keys.append(response.json()['id'])
    for key in keys:
        assert client.get('/api/fsp/' + key).json()['filename'] == 'escape.fsp'
    assert client.post('/api/fsp/native-import', content=fsp.HEADER, headers={'x-filename': '../escape.txt'}).status_code == 400
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(client.get('/api/fsp/' + k).json()['status'] in ('queued', 'running') for k in keys):
        time.sleep(.02)
    stored = [p for p in tmp_path.rglob('*') if p.is_file()]
    assert stored, 'the accepted uploads were stored'
    for path in stored:
        relative = path.relative_to(root / 'fsp')
        assert re.fullmatch(r'[0-9a-f]{32}', relative.parts[0]), path
    assert not list(tmp_path.rglob('escape*')) and not Path('escape.fsp').exists()


def test_gds_upload_names_are_display_only_and_hostile_layouts_are_rejected(tmp_path, client):
    pytest.importorskip('gdstk')
    from test_gds import rectangle_file, write
    import gdstk
    root = tmp_path / 'results'
    (tmp_path / 'fixture').mkdir()
    (tmp_path / 'array').mkdir()
    layout = rectangle_file(tmp_path / 'fixture')
    data = layout.read_bytes()
    response = client.post('/api/gds/inspect', content=data, headers={'x-filename': '../../escape.gds'})
    assert response.status_code == 200, response.text
    assert response.json()['filename'] == '../../escape.gds'
    stored = list((root / 'gds').iterdir())
    assert [re.fullmatch(r'[0-9a-f]{32}\.gds', p.name) is not None for p in stored] == [True]
    assert not list(tmp_path.rglob('escape*'))
    for hostile in [data[:len(data) // 2], data[:-2], data + b'\0\4\4\0', b'\0' * 64, data[:4] + b'\xff' * 60]:
        response = client.post('/api/gds/inspect', content=hostile)
        assert response.status_code == 422, (response.status_code, response.text)
    library = gdstk.Library()
    child = library.new_cell('C')
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    top = library.new_cell('TOP')
    top.add(gdstk.Reference(child, columns=4000, rows=4000, spacing=(2, 2)))
    upload = client.post('/api/gds/inspect', content=write(tmp_path / 'array', library).read_bytes())
    assert upload.status_code == 200, upload.text
    project = demo_project().model_dump()
    body = {'project': project, 'cell': 'TOP',
            'layers': [{'layer': 1, 'datatype': 0, 'z_min': 0, 'z_max': .1, 'material': project['materials'][0]['name']}]}
    started = time.monotonic()
    response = client.post('/api/gds/' + upload.json()['id'] + '/convert', json=body)
    assert response.status_code == 422 and 'admission limits' in response.text, response.text
    assert time.monotonic() - started < 30


def test_request_size_limits_apply_to_declared_and_chunked_bodies(tmp_path, monkeypatch):
    from torchfdtd import gds_service
    monkeypatch.setattr(server, 'MAX_REQUEST_BYTES', 64)
    monkeypatch.setattr(gds_service, 'MAX_UPLOAD_BYTES', 20)
    monkeypatch.setattr(fsp, 'MAX_FSP_BYTES', 16)
    application = server.create_app(tmp_path / 'results')
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            headers = {'content-type': 'application/json'}
            assert client.post('/api/validate', content=b'{"name": "' + b'x' * 80 + b'"}', headers=headers).status_code == 413
            assert client.post('/api/validate', content=b'{"name": "x"}', headers=headers).status_code == 200
            chunked = client.post('/api/validate', content=iter([b'{"name":', b' "x"}']), headers=headers)
            assert chunked.status_code == 411, chunked.text
            for declared in ['abc', '1e9', '-1', '0x10', '']:
                response = client.post('/api/validate', content=b'{}', headers={**headers, 'content-length': declared})
                assert response.status_code == 400, (declared, response.status_code)
            assert client.post('/api/gds/inspect', content=b'\0' * 32).status_code == 413
            assert client.post('/api/fsp/native-import', content=fsp.HEADER + b'1.1' * 8).status_code == 413
            assert client.get('/api/capabilities').status_code == 200
    finally:
        application.state.pool.shutdown()
        application.state.fsp_pool.shutdown()


def test_malformed_json_bodies_get_a_client_error_not_a_server_error(client):
    project = demo_project().model_dump()
    text = json.dumps(project)
    cases = {
        'binary': b'\x00\xff\xfe',
        'array': '[1, 2]',
        'trailing': '{"name": "x"}}}}',
        'empty': b'',
        'string': '"project"',
        'deep': '[' * 100000 + ']' * 100000,
        'nan': text.replace('"mesh": 0.05', '"mesh": NaN'),
        'infinity': text.replace('"mesh": 0.05', '"mesh": Infinity'),
        'overflow': text.replace('"mesh": 0.05', '"mesh": 1e999'),
        'wrong_type': text.replace('"steps": 1000', '"steps": "many"'),
        'extra_key': text.replace('"name":', '"__class__": "os.system", "name":'),
        'huge_grid': text.replace('"mesh": 0.05', '"mesh": 1e-9'),
        'negative_steps': text.replace('"steps": 1000', '"steps": -5'),
        'bad_kind': text.replace('"kind": "pml"', '"kind": "shell"', 1),
    }
    assert 'NaN' in cases['nan'] and '"many"' in cases['wrong_type'] and '"shell"' in cases['bad_kind']
    for name, body in cases.items():
        response = post_json(client, '/api/validate', body)
        assert 400 <= response.status_code < 500, (name, response.status_code, response.text[:200])
        assert response.json()['detail'], name
    for name in ['nan', 'infinity', 'overflow']:
        response = post_json(client, '/api/validate', cases[name])
        assert response.status_code == 422 and 'finite' in response.text, (name, response.text[:200])
    assert client.get('/api/capabilities').status_code == 200
    assert post_json(client, '/api/validate', text).status_code == 200


def test_project_files_are_data_not_code(tmp_path, client):
    hostile = "x'); import os; os.system('echo pwned'); print('"
    project = demo_project()
    project.name = hostile
    project.materials[0].name = "__import__('os').system('id')"
    project.structures[0].material = project.materials[0].name
    project.structures[0].name = '${IFS}`whoami`'
    saved = tmp_path / 'hostile.json'
    project.save(saved)
    loaded = Project.load(saved)
    assert loaded.name == hostile and loaded.structures[0].name == '${IFS}`whoami`'
    with pytest.raises(ValueError):
        Project.model_validate_json(saved.read_text(encoding='utf-8').replace('"name":', '"__reduce__": "os.system", "name":', 1))
    response = client.post('/api/python', json=loaded.model_dump())
    assert response.status_code == 200
    module = ast.parse(response.text)
    assert [type(node).__name__ for node in module.body] == ['ImportFrom', 'Assign', 'Assign', 'Expr', 'Expr']
    called = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.attr if isinstance(func, ast.Attribute) else func.id)
    assert called == {'model_validate', 'Simulation', 'run', 'save', 'print'}
    constants = {node.value for node in ast.walk(module) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert hostile in constants and project.materials[0].name in constants
    for name in ['server.py', 'cli.py', 'models.py', 'gds_service.py', 'fsp_service.py', 'design_service.py',
                 'mode_network_service.py', 'radiation_api.py', 'farfield_api.py']:
        source = (PACKAGE / name).read_text(encoding='utf-8')
        assert not re.search(r'(?<![\w.])(eval|exec|compile)\(|os\.system|subprocess|__import__|importlib', source), name


def test_npz_readers_never_unpickle_and_fail_closed_on_hostile_archives(tmp_path):
    from torchfdtd.radiation_io import load_native_radiation_plane
    from torchfdtd.radiation_box_io import load_native_radiation_box
    sources = {path.name: path.read_text(encoding='utf-8') for path in PACKAGE.glob('*.py')}
    loads = [(name, match.group(0)) for name, text in sources.items() for match in re.finditer(r'\bnp\.load\([^\n]*', text)]
    assert loads, 'the package loads NPZ files somewhere'
    assert all('allow_pickle=False' in call for _, call in loads), [name for name, call in loads if 'allow_pickle=False' not in call]
    # Checkpoints are the only torch.load callers; each restores tensors and plain containers only.
    checkpoints = [(name, match.group(0)) for name, text in sources.items() for match in re.finditer(r'\btorch\.load\([^\n]*', text)]
    assert {name for name, _ in checkpoints} == {'design_checkpoint.py', 'design_problem.py'}
    assert all('weights_only=True' in call for _, call in checkpoints), [name for name, call in checkpoints if 'weights_only=True' not in call]
    for pattern in ['allow_pickle=True', 'weights_only=False', 'pickle.load', 'joblib.load', 'numpy.load(']:
        assert not [name for name, text in sources.items() if pattern in text], pattern
    project = demo_project()
    strings = dict(project=np.asarray(project.model_dump_json()), summary=np.asarray('{"cancelled": false}'),
                   field_monitors=np.asarray(json.dumps([{'id': 'plane'}])))

    def entry(value):
        stream = io.BytesIO()
        np.save(stream, value)
        return stream.getvalue()

    def header(shape, descr='<f8'):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {'descr': descr, 'fortran_order': False, 'shape': shape})
        return stream.getvalue()

    pickled = tmp_path / 'object.npz'
    np.savez(pickled, **strings, frames=np.asarray([object()], dtype=object),
             field_monitor_0_fields=np.asarray([object()], dtype=object), field_monitor_0_frequency_hz=np.ones(1),
             field_monitor_0_points_um=np.zeros((4, 3)), field_monitor_0_weights=np.ones(4))
    with pytest.raises(ValueError, match='allow_pickle'):
        Result.load(pickled)
    with pytest.raises(ValueError, match='allow_pickle'):
        load_native_radiation_plane(pickled, 'plane')
    with pytest.raises(ValueError):
        load_native_radiation_box(pickled, {'x_min': 'plane'}, bounds_um=[[-1, 1]] * 3, refractive_index=1., open_surface=True)

    bomb = tmp_path / 'bomb.npz'
    with zipfile.ZipFile(bomb, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in strings.items():
            archive.writestr(name + '.npy', entry(value))
        archive.writestr('frames.npy', header((1 << 40,)) + b'\0' * 64)
        archive.writestr('field_monitor_0_fields.npy', header((1 << 30, 4, 6), '<c8') + b'\0' * 64)
        archive.writestr('field_monitor_0_frequency_hz.npy', entry(np.ones(1)))
        archive.writestr('field_monitor_0_points_um.npy', entry(np.zeros((4, 3))))
        archive.writestr('field_monitor_0_weights.npy', entry(np.ones(4)))
    assert bomb.stat().st_size < 8192
    started = time.monotonic()
    with pytest.raises((MemoryError, ValueError)):
        Result.load(bomb)
    with pytest.raises(ValueError):
        load_native_radiation_box(bomb, {'x_min': 'plane'}, bounds_um=[[-1, 1]] * 3, refractive_index=1., open_surface=True)
    assert time.monotonic() - started < 5

    truncated = tmp_path / 'truncated.npz'
    truncated.write_bytes(pickled.read_bytes()[:200])
    for loader in [Result.load, lambda path: load_native_radiation_plane(path, 'plane')]:
        with pytest.raises((zipfile.BadZipFile, ValueError, OSError, EOFError)):
            loader(truncated)


PROJECT_ROUTES = ('/api/validate', '/api/jobs', '/api/mesh/preview', '/api/python', '/api/sources/src0/preview')


def test_every_server_limit_refuses_an_oversized_request(client):
    """Each size constraint the Python API lifted stays a limit of every route that takes a project or its parts.

    The projects below are valid on the Python API. Through the server each one answers 422, on the
    project routes and nested in a GDS export and a mode-network request, whatever caps it carries.
    """
    from torchfdtd import Region, Source
    from torchfdtd.solver import estimate
    from test_resident_guards import LARGE, SERVER_CASES, oversized, single_oversized, wide_plane
    payload = oversized()
    raised = {name: None for name in ('max_structures', 'max_sources', 'max_monitors', 'max_materials',
                                      'max_mesh_refinements', 'max_monitor_samples')}
    for field, message in SERVER_CASES.items():
        project = dict(single_oversized(field, payload), limits=raised)
        project['sources'] = [dict(s, id='src0' if i == 0 else s['id']) for i, s in enumerate(project['sources'])]
        Project.model_validate(project)
        for route in PROJECT_ROUTES:
            response = client.post(route, json=project)
            assert response.status_code == 422 and message in response.text, (field, route, response.text[:300])
        for route, body in (('/api/gds/export', {'project': project, 'layers': {'s0': [1, 0]}}),
                            ('/api/mode-networks/validate', {'project': project})):
            response = client.post(route, json=body)
            assert response.status_code == 422 and message in response.text, (field, route, response.text[:300])
    # Monitor samples are admitted by the estimate: 23.5 million samples pass the Python API and not the server.
    wide = wide_plane().model_dump(mode='json')
    estimate(Project.model_validate(wide))
    for route in ('/api/validate', '/api/jobs', '/api/mesh/preview'):
        response = client.post(route, json=wide)
        assert response.status_code == 422 and 'exceeds the limit of 12,000,000' in response.text, route
    # Resident cells: an explicit resident grid above 8 million cells, admitted by memory on the Python API.
    resident = Project(region=Region(**LARGE, execution_mode='resident'), sources=[Source(center=(0, 0, 0))]).model_dump(mode='json')
    for route in ('/api/validate', '/api/jobs'):
        response = client.post(route, json=resident)
        assert response.status_code == 422 and 'server limits resident execution to 8,000,000 cells' in response.text, route
    # A small project carrying caps validates, and the server echoes it unchanged.
    capped = dict(demo_project().model_dump(), limits={'max_structures': 10})
    response = client.post('/api/validate', json=capped)
    assert response.status_code == 200 and response.json()['project']['limits']['max_structures'] == 10


def test_budgeted_mode_network_and_design_routes_keep_the_resident_cell_limit(client):
    """A byte budget (mode networks) or the design planner's generated budgets do not lift the 8,000,000-cell limit."""
    from test_mode_network_project import config
    from torchfdtd.periodic_design import PeriodicDesignConfig
    network = config()
    network['project']['region'] = dict(network['project']['region'], size=[14., 14., 6.], mesh=.05)  # 9,408,000 cells
    network['project']['sources'][0]['size'] = [14., 14., 0.]
    network['execution'] = dict(network['execution'], resident_budget_bytes=64*2**30, network_budget_bytes=16*2**30,
                                host_budget_bytes=40*2**30)
    for route in ('/api/mode-networks/validate', '/api/mode-networks/python'):
        response = client.post(route, json=network)
        assert response.status_code == 422 and 'server limits resident execution to 8,000,000 cells' in response.text, route
    design = dict(PeriodicDesignConfig().model_dump(mode='json'), period_um=[6.4, 6.4], mesh_um=.025)  # 9,568,256 cells
    response = client.post('/api/design/plan', json=dict(design, execution='resident'))
    assert response.status_code == 422 and 'server limits resident execution to 8,000,000 cells' in response.text
    response = client.post('/api/design/plan', json=design)
    assert response.status_code == 200, response.text
    selection = response.json()['selection']
    assert selection['mode'] != 'resident'
    assert any(a['mode'] == 'resident' and not a['admitted'] and 'server limits resident execution' in a['reason']
               for a in selection['attempts'])


def test_gds_uploads_keep_the_vertex_limit_the_python_api_lifts(tmp_path, client):
    pytest.importorskip('gdstk')
    import gdstk
    from test_gds import write
    from torchfdtd import gds, gds_service
    assert gds.GDSLimits().max_total_vertices is None and gds_service.SERVER_GDS_LIMITS.max_total_vertices == 1_000_000
    # 130 polygons of 8000 vertices: 1.04 million vertices in 130 instances.
    library = gdstk.Library()
    top = library.new_cell('TOP')
    angles = np.linspace(0, 2*np.pi, 8000, endpoint=False)
    for k in range(130):
        top.add(gdstk.Polygon(np.c_[np.cos(angles)+3*k, np.sin(angles)], layer=1))
    (tmp_path / 'dense').mkdir()
    upload = client.post('/api/gds/inspect', content=write(tmp_path / 'dense', library).read_bytes())
    assert upload.status_code == 200, upload.text
    project = demo_project().model_dump()
    body = {'project': project, 'cell': 'TOP',
            'layers': [{'layer': 1, 'datatype': 0, 'z_min': 0, 'z_max': .1, 'material': project['materials'][0]['name']}]}
    response = client.post('/api/gds/' + upload.json()['id'] + '/convert', json=body)
    assert response.status_code == 422 and 'instance/vertex admission limits' in response.text, response.text
