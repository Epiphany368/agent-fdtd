"""Workbench angular-spectrum post-processing of a stored DFT plane through the FastAPI client."""
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd import FieldMonitor, Project, Simulation, propagate_section, propagate_volume
import torchfdtd.propagation_api as propagation_api
from torchfdtd.server import create_app
from test_tiled import pillar_row, pillar_array, spectrum


def _finished(client, key, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/'+key).json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.05)
    raise AssertionError('job did not finish')


def _row():
    p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
    # A second, coarse plane whose spacing exceeds half the wavelength in a dense exterior.
    p.monitors.append(FieldMonitor(id='coarse', normal='y', center=(0., .6, 0.), size=(4.8, 0., 1.), downsample=16,
                                   spectrum=spectrum(1.55)))
    return Project.model_validate(p.model_dump())


def test_section_and_plane_match_the_library_and_report_focus_and_spectrum(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = _row()
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        job = _finished(client, key)
        assert job['status'] == 'completed'
        listing = client.get(f'/api/jobs/{key}/propagation-monitors').json()
        assert listing['device'] == 'cpu' and listing['dimension'] == '2d' and [m['id'] for m in listing['monitors']] == ['out', 'coarse']
        out = listing['monitors'][0]
        assert out['normal'] == 'y' and out['direction'] == 1 and out['transverse'] == ['x'] and out['spacing_um']['x'] == pytest.approx(.05)
        z = np.linspace(0., 12., 241)
        body = dict(monitor='out', kind='section', axis='x', z_start_um=0., z_stop_um=12., planes=241, index=1.)
        response = client.post(f'/api/jobs/{key}/propagate', json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['kind'] == 'section' and result['device'] == 'cpu' and result['direction'] == 1
        assert result['geometry']['axes'] == ['y', 'x'] and result['components'] == ['Ex', 'Ey', 'Ez']
        plane = Simulation(p).run().field_monitor('out')
        section = propagate_section(plane, z, index=1., section='xy', components=('Ex', 'Ey', 'Ez'), pad=2)
        expected = section.intensity()[0].numpy()
        np.testing.assert_allclose(np.array(result['image']), expected, rtol=1e-4, atol=1e-6*expected.max())
        focus = section.focus(0)
        for name in ('z_um', 'normal_um', 'a_um', 'fwhm_um'):
            assert result['focus'][name] == pytest.approx(focus[name], rel=1e-6)
        assert result['focus']['peak_intensity'] == pytest.approx(focus['peak_intensity'], rel=1e-4)
        assert result['spectrum']['max_angle_deg'] == pytest.approx(90.) and not result['spectrum']['aliasing']
        assert result['spectrum']['evanescent_fraction'] == pytest.approx(section.report['evanescent_fraction'][0])
        assert result['spectrum']['warning'] is None and result['bytes'] <= result['budget_bytes']
        # A parallel plane at one distance uses the volume kernel of the tiled focal plane.
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='plane', distance_um=7., index=1.))
        assert response.status_code == 200, response.text
        result = response.json()
        volume = propagate_volume(plane, [7.], index=1., components=('Ex', 'Ey', 'Ez'), pad=2, chunk=1)
        expected = volume.intensity()[0, 0].numpy()
        np.testing.assert_allclose(np.array(result['image']), expected, rtol=1e-4, atol=1e-6*expected.max())
        assert result['focus']['z_um'] == 7. and result['focus']['normal_um'] == pytest.approx(7.5)
        assert result['method'] == 'angular spectrum volume'
        # The coarse plane aliases in a dense exterior and the route says so.
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='coarse', kind='section', axis='x', z_stop_um=2., planes=11, index=1.5))
        assert response.status_code == 200, response.text
        aliased = response.json()['spectrum']
        assert aliased['aliasing'] and aliased['max_angle_deg'] < 90 and 'alias' in aliased['warning']
        assert aliased['spacing_um'] == pytest.approx(.8) and aliased['wavelength_um'] == pytest.approx(1.55/1.5)
    app.state.pool.shutdown()


def test_route_refuses_unknown_planes_axes_and_budgets(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        assert _finished(client, key)['status'] == 'completed'
        assert client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='missing', kind='plane', distance_um=1.)).status_code == 422
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='section', axis='z', z_stop_um=2.))
        assert response.status_code == 422 and 'section axis' in response.json()['detail']
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='section', axis='x', z_start_um=3., z_stop_um=1.))
        assert response.status_code == 422
        monkeypatch.setattr(propagation_api, '_budget_bytes', lambda device: 1000)
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='section', axis='x', z_stop_um=2.))
        assert response.status_code == 422 and 'above the budget' in response.json()['detail']
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='plane', distance_um=2.))
        assert response.status_code == 422 and 'above the budget' in response.json()['detail']
        # A point-monitor-only run stores no plane.
        q = Project.model_validate(p.model_dump())
        q.monitors = []
        q = Project.model_validate(q.model_dump())
        key = client.post('/api/jobs', json=q.model_dump()).json()['id']
        assert _finished(client, key)['status'] == 'completed'
        assert client.get(f'/api/jobs/{key}/propagation-monitors').json()['monitors'] == []
        assert client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='plane', distance_um=1.)).status_code == 422
    app.state.pool.shutdown()


def test_tiled_job_planes_propagate_too(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
        p.region.execution_mode = 'tiled'
        p.region.tiling = p.region.tiling.model_copy(update=dict(size_um=3., overlap_um=2.))
        p = Project.model_validate(p.model_dump())
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        job = _finished(client, key)
        assert job['status'] == 'completed' and job['execution']['mode'] == 'tiled'
        listing = client.get(f'/api/jobs/{key}/propagation-monitors').json()
        assert listing['mode'] == 'tiled' and listing['monitors'][0]['tiled']['plane'] == 'stitched output plane'
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='section', axis='x', z_stop_um=12., planes=121))
        assert response.status_code == 200, response.text
        stitched = response.json()
        whole = Simulation(Project.model_validate({**p.model_dump(), 'region': {**p.region.model_dump(), 'execution_mode': 'resident'}})).run().field_monitor('out')
        expected = propagate_section(whole, np.linspace(0., 12., 121), section='xy', pad=2).intensity()[0].numpy()
        np.testing.assert_allclose(np.array(stitched['image']), expected, rtol=1e-3, atol=1e-3*expected.max())
    app.state.pool.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_job_propagates_on_the_device(tmp_path):
    pytest.importorskip('cupy')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = pillar_array(backend='cuda')
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        assert _finished(client, key)['status'] == 'completed'
        assert client.get(f'/api/jobs/{key}/propagation-monitors').json()['device'] == 'cuda'
        response = client.post(f'/api/jobs/{key}/propagate', json=dict(monitor='out', kind='section', axis='y', offset_um=.3, z_stop_um=4., planes=41))
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['device'] == 'cuda' and result['geometry']['axes'] == ['z', 'y'] and len(result['image']) == 41
        assert result['focus']['fwhm_um'] is None or result['focus']['fwhm_um'] > 0
    app.state.pool.shutdown()
