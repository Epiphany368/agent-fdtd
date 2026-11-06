import time

from fastapi.testclient import TestClient

from torchfdtd import fsp
from torchfdtd.server import create_app


def test_upload_inspection_and_original_download(tmp_path, monkeypatch):
    monkeypatch.setattr(fsp, 'availability', lambda: {'installed': True})
    def inspect(path):
        return {'source': {'filename': path.name, 'sha256': fsp.fingerprint(path)},
                'objects': [], 'native_execution': {'allowed': False}}
    monkeypatch.setattr(fsp, 'inspect_fsp', inspect)
    app = create_app(tmp_path)
    content = fsp.HEADER + b'1.1\0fixture'
    try:
        with TestClient(app) as c:
            assert c.post('/api/fsp/import', content=b'{}', headers={'x-filename': 'bad.fsp'}).status_code == 400
            assert c.post('/api/fsp/import', content=content, headers={'Origin': 'https://unrelated.example'}).status_code == 403
            response = c.post('/api/fsp/import', content=content, headers={'x-filename': '../input.fsp'})
            assert response.status_code == 202
            key = response.json()['id']
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                data = c.get('/api/fsp/' + key).json()
                if data['status'] in ('ready', 'failed'):
                    break
                time.sleep(.02)
            assert data['status'] == 'ready', data
            assert data['filename'] == 'input.fsp'
            assert not data['inspection']['native_execution']['allowed']
            assert c.get(f'/api/fsp/{key}/download').content == content
            assert c.get(f'/api/fsp/{key}/archive').content.startswith(b'PK')
            assert c.get('/api/fsp/missing').status_code == 404
            assert c.post(f'/api/fsp/{key}/export', json={'patches': [{'foo': 'bar'}]}).status_code == 422
    finally:
        app.state.pool.shutdown()
        app.state.fsp_pool.shutdown()
