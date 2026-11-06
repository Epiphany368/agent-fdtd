"""G8-04: the versioned project (edit counter and content hash), the plan hash a
job carries for stale-result marking, field validation at the API and the GDS export route."""
import base64
import time

import pytest
from fastapi.testclient import TestClient

from torchfdtd import Project
from torchfdtd.models import demo_project
from torchfdtd.server import create_app


def test_content_hash_ignores_the_version_fields_and_tracks_the_content():
    project = demo_project()
    assert project.revision == 0 and project.content_sha256 is None and project.content_matches() is None
    baseline = project.content_hash()
    assert len(baseline) == 64
    assert project.model_copy(update={'revision': 5}).content_hash() == baseline
    stamped = project.stamped(revision=3)
    assert stamped.revision == 3 and stamped.content_sha256 == baseline and stamped.content_matches() is True
    assert stamped.content_hash() == baseline, 'the stored hash does not enter the hash'
    edited = stamped.model_copy(update={'region': stamped.region.model_copy(update={'steps': stamped.region.steps + 1})})
    assert edited.content_hash() != baseline and edited.content_matches() is False
    assert stamped.model_copy(update={'content_sha256': '0' * 64}).content_matches() is False
    with pytest.raises(ValueError):
        Project.model_validate({**project.model_dump(), 'revision': -1})
    with pytest.raises(ValueError):
        Project.model_validate({**project.model_dump(), 'content_sha256': 'xyz'})


def test_project_file_round_trips_the_version_fields(tmp_path):
    stamped = demo_project('scatterer').stamped(revision=7)
    stamped.save(tmp_path / 'scene.json')
    loaded = Project.load(tmp_path / 'scene.json')
    assert loaded.revision == 7 and loaded.content_sha256 == stamped.content_sha256
    assert loaded.content_matches() is True and loaded.model_dump() == stamped.model_dump()
    text = (tmp_path / 'scene.json').read_text(encoding='utf-8').replace('"steps": 1000', '"steps": 1200')
    assert text != (tmp_path / 'scene.json').read_text(encoding='utf-8')
    (tmp_path / 'edited.json').write_text(text, encoding='utf-8')
    assert Project.load(tmp_path / 'edited.json').content_matches() is False


def test_validate_reports_the_version_fields_a_tampered_hash_and_rejects_bad_values(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = demo_project()
        p.region.backend = 'cpu'
        first = client.post('/api/validate', json=p.model_dump()).json()
        assert first['revision'] == 0 and first['content_sha256'] == p.content_hash()
        assert first['stored_content_sha256_matches'] is None
        assert first['project']['content_sha256'] == first['content_sha256']
        echoed = first['project']
        echoed['revision'] = 4
        second = client.post('/api/validate', json=echoed).json()
        assert second['revision'] == 4 and second['stored_content_sha256_matches'] is True
        assert second['content_sha256'] == first['content_sha256'], 'the revision counter does not change the content hash'
        echoed['content_sha256'] = '0' * 64
        assert client.post('/api/validate', json=echoed).json()['stored_content_sha256_matches'] is False
        for patch in ({'size': [-1, 6, 2]}, {'steps': 5}, {'mesh': 0}, {'pml_cells': 1}):
            response = client.post('/api/validate', json={**p.model_dump(), 'region': {**p.region.model_dump(), **patch}})
            assert response.status_code == 422, patch
        assert client.post('/api/validate', content=b'{"name": "x", "region": {"mesh": NaN}}',
                           headers={'Content-Type': 'application/json'}).status_code == 422
        assert client.post('/api/validate', json={**p.model_dump(), 'revision': -2}).status_code == 422
    app.state.pool.shutdown()


def test_jobs_carry_the_plan_hash_of_the_submitted_project(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = demo_project()
        p.region.backend = 'cpu'
        p.region.steps = 20
        current = client.post('/api/validate', json=p.model_dump()).json()['plan_hash']
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        job = client.get('/api/jobs/' + key).json()
        assert job['plan_hash'] == current and job['revision'] == 0
        deadline = time.monotonic() + 30
        while client.get('/api/jobs/' + key).json()['status'] not in ('completed', 'failed') and time.monotonic() < deadline:
            time.sleep(.05)
        assert client.get('/api/jobs/' + key).json()['status'] == 'completed'
        assert client.get('/api/jobs/' + key).json()['plan_hash'] == current, 'the hash of the run does not move after completion'
        renamed = {**p.model_dump(), 'name': 'other name', 'revision': 9}
        assert client.post('/api/validate', json=renamed).json()['plan_hash'] == current, 'names and the edit counter are not physics'
        moved = p.model_dump()
        moved['structures'][0]['center'] = [0.3, 0, 0]
        assert client.post('/api/validate', json=moved).json()['plan_hash'] != current, 'a geometry edit changes the plan hash'
        longer = {**p.model_dump(), 'region': {**p.region.model_dump(), 'steps': 30}}
        assert client.post('/api/validate', json=longer).json()['plan_hash'] != current, 'the step count changes the plan hash'
        placement = {**p.model_dump(), 'region': {**p.region.model_dump(), 'precision': 'float64'}}
        assert client.post('/api/validate', json=placement).json()['plan_hash'] == current, 'placement stays outside the plan hash (torchfdtd.plan)'
    app.state.pool.shutdown()


def test_gds_export_route_returns_the_file_and_its_stack_sidecar(tmp_path):
    gdstk = pytest.importorskip('gdstk')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = demo_project()
        response = client.post('/api/gds/export', json={'project': p.model_dump(), 'layers': {'waveguide': [1, 0]}, 'cell': 'TOP'})
        assert response.status_code == 200, response.text
        body = response.json()
        data = base64.b64decode(body['gds_base64'])
        assert len(data) == body['bytes'] > 0
        sidecar = body['sidecar']
        assert sidecar['cell'] == 'TOP' and sidecar['structures'] == 1
        assert sidecar['layer_stack'] == [dict(structure_id='waveguide', layer=1, datatype=0, material='SiN (constant n)', z_min=-0.2, z_max=0.2, mesh_order=2)]
        # gdstk 0.9 opens ASCII relative paths only on Windows; the package helper provides one.
        from torchfdtd.gds import _gds_io
        with _gds_io() as path:
            path.write_bytes(data)
            library = gdstk.read_gds(path)
        cell = next(c for c in library.cells if c.name == 'TOP')
        assert len(cell.polygons) == 1 and cell.polygons[0].layer == 1
        assert client.post('/api/gds/export', json={'project': p.model_dump(), 'layers': {'missing': [1, 0]}}).status_code == 422
        assert client.post('/api/gds/export', json={'project': p.model_dump(), 'layers': {'waveguide': [1, 0]}, 'cell': 'x' * 40}).status_code == 422
        assert not list(tmp_path.glob('gds/export-*.gds')), 'the temporary export file is removed'
    app.state.pool.shutdown()
