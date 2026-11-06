"""Workbench execution modes: Auto resolution from a resource record and streamed browser jobs."""
from dataclasses import replace
import re
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd import FieldMonitor, Monitor, Project, Region, Simulation, Source
from torchfdtd.execution_modes import (_base_options, _browser_scene, resolve_execution)
from torchfdtd.models import SERVER_LIMITS, server_limits
from torchfdtd.server import create_app
from torchfdtd.solver import estimate

CPU_RECORD = dict(cuda=False, cupy=False, gpu=None, gpu_free_bytes=0, gpu_total_bytes=0,
                  host_total_bytes=64*2**30, host_available_bytes=32*2**30)


def scene(length_um=1.6, steps=60, **region):
    settings = {**dict(dimension='3d', size=(length_um, 1.2, 1.2), mesh=.1, pml_cells=3, steps=steps, backend='cpu'), **region}
    return Project(name='modes', region=Region(**settings),
                   sources=[Source(center=(-.3, 0, 0), component='Ez', wavelength=1., pulse_cycles=2)],
                   monitors=[Monitor(id='pt', center=(.3, .1, 0), component='Ez'),
                             Monitor(id='ph', center=(.2, 0, .1), component='Hy'),
                             FieldMonitor(id='plane', center=(.4, 0, 0), normal='x', size=(0, .6, .6),
                                          spectrum=dict(sampling='frequency', wavelength_start=.9, wavelength_stop=1.2,
                                                        frequency_points=5, apodization='none'))])


def test_auto_picks_resident_for_a_small_scene(tmp_path):
    p = scene()
    record = resolve_execution(p, health=CPU_RECORD, scratch=tmp_path/'scratch', summary=estimate(p))
    assert (record['requested'], record['mode'], record['backend']) == ('auto', 'resident', 'cpu')
    assert record['resident']['fits'] and record['resident']['limit_bytes'] == int(32*2**30*.8)
    assert record['error'] is None and 'streamed_host' not in record
    gpu = dict(CPU_RECORD, cuda=True, cupy=True, gpu='fake', gpu_free_bytes=8*2**30, gpu_total_bytes=12*2**30)
    p.region.backend = 'auto'
    record = resolve_execution(p, health=gpu, scratch=tmp_path/'scratch', summary=estimate(p))
    assert (record['mode'], record['backend'], record['gpu']) == ('resident', 'cuda', 'fake')
    assert record['resident']['limit_bytes'] == int(8*2**30*.75)
    p.region.execution_mode = 'streamed_host'
    record = resolve_execution(p, health=CPU_RECORD, scratch=tmp_path/'scratch', summary=estimate(p))
    assert record['mode'] == 'streamed_host' and record['policy']['checkpoints'] == 0
    assert record['policy']['tile_device'] == 'cpu' and record['options']['state_storage'] == 'host'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA tile admission needs a CUDA device')
def test_auto_streams_through_host_above_the_injected_device_memory(tmp_path):
    pytest.importorskip('cupy')
    p = scene(25.6, backend='cuda')
    summary = estimate(p)
    resident_bytes = int(summary['estimated_memory_mb']*2**20)
    # The record's free device memory admits the tile workspace but not the resident estimate.
    free = int(resident_bytes/.75*.6)
    fake = dict(cuda=True, cupy=True, gpu='fake', gpu_free_bytes=free, gpu_total_bytes=12*2**30,
                host_total_bytes=64*2**30, host_available_bytes=32*2**30)
    record = resolve_execution(p, health=fake, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] == 'streamed_host' and record['backend'] == 'cuda'
    assert not record['resident']['fits'] and 'exceeds 75%' in record['resident']['reason']
    assert record['policy']['tile_device'] == 'cuda' and record['policy']['state_storage'] == 'host'
    assert record['options']['gpu_budget_bytes'] == int(free*.8)
    assert record['reservation']['gpu_reservation_bytes'] <= int(free*.8)
    assert record['reservation']['host_reservation_bytes'] <= int(32*2**30*.8)
    assert 'streamed_disk' not in record


def test_auto_refuses_instead_of_disk_and_explicit_disk_still_works(tmp_path):
    from torchfdtd.streamed import _reservation
    from torchfdtd.streamed_planning import plan_streamed_work
    from torchfdtd.boundaries import material_shape
    p = scene(51.2)
    summary = estimate(p)
    internal, observations = _browser_scene(p)
    epsilon = torch.empty(material_shape(internal.region), device='meta')
    def reservation(storage, **policy):
        base = _base_options(storage, 'cpu', CPU_RECORD, tmp_path/'scratch')
        return _reservation(internal, epsilon, replace(base, **policy), observations)['host_reservation_bytes']
    disk_default = reservation('disk')
    host_base = _base_options('host', 'cpu', CPU_RECORD, tmp_path/'scratch')
    smallest_host = min(reservation('host', slab_width=c['options']['slab_width'], temporal_depth=c['options']['temporal_depth'])
                        for c in plan_streamed_work(internal, host_base).report['candidates'])
    assert disk_default < smallest_host
    available = int((disk_default+smallest_host)/2/.8)
    short = dict(CPU_RECORD, host_available_bytes=available)
    # Disk banks would fit, but Auto never selects them: it refuses and names the options.
    record = resolve_execution(p, health=short, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] is None and 'streamed_disk' not in record
    assert not record['resident']['fits'] and 'exceeds 80%' in record['resident']['reason']
    assert not record['streamed_host']['admitted'] and 'host budget' in record['streamed_host']['reason']
    assert record['error'].startswith('No automatic tier fits')
    for option in ('allow approximate tiling', 'Streamed through disk', '1.9 to 2.4 times the DRAM time', 'coarsen the mesh'):
        assert option in record['error']
    assert [rung['tier'] for rung in record['auto']['rungs']] == ['resident', 'streamed_host', 'tiled'] and record['auto']['chosen'] is None
    assert 'not allowed' in record['auto']['rungs'][2]['reason']
    p.region.execution_mode = 'streamed_disk'
    explicit = resolve_execution(p, health=short, scratch=tmp_path/'scratch', summary=summary)
    assert explicit['mode'] == 'streamed_disk' and explicit['policy']['state_storage'] == 'disk'
    assert explicit['options']['state_directory'] == str(tmp_path/'scratch') and explicit['options']['host_budget_bytes'] == int(available*.8)
    assert explicit['reservation']['disk_reservation_bytes'] > 0 and explicit['scratch_directory'] == str(tmp_path/'scratch')
    p.region.execution_mode = 'auto'
    nothing = resolve_execution(p, health=dict(CPU_RECORD, host_available_bytes=1), scratch=tmp_path/'scratch', summary=summary)
    assert nothing['mode'] is None and nothing['error'].startswith('No automatic tier fits')
    p.region.execution_mode = 'resident'
    forced = resolve_execution(p, health=dict(CPU_RECORD, host_available_bytes=1), scratch=tmp_path/'scratch', summary=summary)
    assert forced['mode'] == 'resident' and forced['warnings']


def _planar_row(**tiling):
    """The 2D sparse pillar row of the tiling record under the automatic policy."""
    from test_tiled import pillar_row
    p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
    # Two stored frames: 85 frames of this 8400-cell grid would outweigh the arrays the rungs are ordered by.
    p.region.snapshot_interval = p.region.steps
    p.region.tiling = p.region.tiling.model_copy(update={**dict(size_um=1.5, overlap_um=.5), **tiling})
    return Project.model_validate(p.model_dump())


def _rung_thresholds(p, tmp_path):
    """Resident estimate, the smallest admitted DRAM reservation and the largest tile estimate of a planar scene."""
    from torchfdtd.streamed import _reservation
    from torchfdtd.streamed_planning import plan_streamed_work
    from torchfdtd.boundaries import material_shape
    from torchfdtd.execution_modes import _tiled_candidate
    resident = int(estimate(p)['estimated_memory_mb']*2**20)
    internal, observations = _browser_scene(p)
    epsilon = torch.empty(material_shape(internal.region), device='meta')
    base = _base_options('host', 'cpu', CPU_RECORD, tmp_path/'scratch')
    host = min(_reservation(internal, epsilon, replace(base, slab_width=c['options']['slab_width'], temporal_depth=c['options']['temporal_depth']), observations)['host_reservation_bytes']
               for c in plan_streamed_work(internal, base).report['candidates'])
    tile = _tiled_candidate(p, 'cpu', CPU_RECORD)['plan']['largest_tile_estimate_bytes']
    return resident, host, tile


def test_auto_policy_rungs_resident_dram_tiled_with_consent_then_refuse(tmp_path):
    p = _planar_row()
    summary = estimate(p)
    resident, host, tile = _rung_thresholds(p, tmp_path)
    assert tile < host < resident
    # Rung 1: the estimate fits the device (GPU record) or the RAM (CPU record).
    gpu = dict(CPU_RECORD, cuda=True, cupy=True, gpu='fake', gpu_free_bytes=8*2**30, gpu_total_bytes=12*2**30)
    q = Project.model_validate(p.model_dump())
    q.region.backend = 'auto'
    record = resolve_execution(q, health=gpu, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] == 'resident' and record['backend'] == 'cuda'
    record = resolve_execution(p, health=CPU_RECORD, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] == 'resident'
    # Rung 2: only the DRAM banks fit.
    ram_only = dict(CPU_RECORD, host_available_bytes=int((host+resident)/2/.8))
    record = resolve_execution(p, health=ram_only, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] == 'streamed_host' and record['auto']['chosen'] == 'streamed_host'
    assert record['reason'].startswith('resident: ') and '-> DRAM banks:' in record['reason']
    assert 'tiled' not in record
    # Rung 3: neither fits; the planar device tiles only with consent.
    tiles_only = dict(CPU_RECORD, host_available_bytes=int((tile+host)/2/.8))
    record = resolve_execution(p, health=tiles_only, scratch=tmp_path/'scratch', summary=summary)
    assert record['mode'] is None and 'No automatic tier fits' in record['error'] and 'allow approximate tiling' in record['error']
    assert record['auto']['rungs'][2]['reason'].startswith('approximate tiling not allowed')
    consented = _planar_row(allow_approximate=True)
    record = resolve_execution(consented, health=tiles_only, scratch=tmp_path/'scratch', summary=estimate(consented))
    assert record['mode'] == 'tiled' and record['auto']['chosen'] == 'tiled' and record['tiled']['plan']['tiles'] == 4
    assert '-> approximate tiles (allowed):' in record['reason']
    assert any('read the mismatch indicator' in w for w in record['warnings'])
    # Rung 4: nothing fits, with consent but without the memory for one tile, and for a device the planner rejects.
    record = resolve_execution(consented, health=dict(CPU_RECORD, host_available_bytes=int(tile/2/.8)), scratch=tmp_path/'scratch', summary=estimate(consented))
    assert record['mode'] is None and 'largest tile' in record['auto']['rungs'][2]['reason'] and 'coarsen the mesh' in record['error']
    point = Project.model_validate(consented.model_dump())
    point.sources[0] = Source(kind='point', center=(0, -.85, 0), component='Ex', wavelength=1.55)
    point = Project.model_validate(point.model_dump())
    record = resolve_execution(point, health=tiles_only, scratch=tmp_path/'scratch', summary=estimate(point))
    assert record['mode'] is None and 'not partitioned' in record['auto']['rungs'][2]['reason']
    assert record['error'].startswith('No automatic tier fits')


def test_auto_policy_through_the_validate_route(tmp_path, monkeypatch):
    p = _planar_row()
    resident, host, tile = _rung_thresholds(p, tmp_path)
    tiles_only = dict(CPU_RECORD, host_available_bytes=int((tile+host)/2/.8))
    from torchfdtd import server
    monkeypatch.setattr(server, 'execution_resources', lambda: tiles_only)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/validate', json=p.model_dump())
        assert response.status_code == 200
        execution = response.json()['execution']
        assert execution['mode'] is None and execution['requested'] == 'auto' and 'allow approximate tiling' in execution['error']
        assert [rung['tier'] for rung in execution['auto']['rungs']] == ['resident', 'streamed_host', 'tiled']
        refused = client.post('/api/jobs', json=p.model_dump())
        assert refused.status_code == 422 and 'No automatic tier fits' in refused.json()['detail']
        consented = _planar_row(allow_approximate=True)
        assert consented.region.tiling.allow_approximate and not p.region.tiling.allow_approximate
        execution = client.post('/api/validate', json=consented.model_dump()).json()['execution']
        assert execution['mode'] == 'tiled' and execution['auto']['chosen'] == 'tiled'
        job = client.post('/api/jobs', json=consented.model_dump()).json()['id']
        finished = _finished(client, job)
        assert finished['status'] == 'completed' and finished['execution']['mode'] == 'tiled' and finished['execution']['requested'] == 'auto'
        assert finished['execution']['indicator']['max_mismatch_center'] is not None
        disk = Project.model_validate(p.model_dump())
        disk.region.execution_mode = 'streamed_disk'
        execution = client.post('/api/validate', json=disk.model_dump()).json()['execution']
        assert execution['mode'] == 'streamed_disk'
    app.state.pool.shutdown()


def test_resident_cell_caps_apply_to_explicit_resident_regions():
    # The Python API admits resident grids by memory; an explicit cap or the server limit refuses at validation.
    settings = dict(dimension='3d', size=(25.6, 25.6, 12.8), mesh=.1, pml_cells=3)
    region = Region(**settings, execution_mode='resident')
    assert np.prod(region.shape) > SERVER_LIMITS['resident_cells']
    Simulation(Project(region=region, sources=[Source(center=(0, 0, 0))]))
    with pytest.raises(ValueError, match='resident_cell_limit=8,000,000'):
        Region(**settings, execution_mode='resident', resident_cell_limit=8_000_000)
    with pytest.raises(ValueError, match='resident_cell_limit=8,000,000'):
        Simulation(Project(region=Region(**settings, resident_cell_limit=8_000_000), sources=[Source(center=(0, 0, 0))]))
    with server_limits():
        with pytest.raises(ValueError, match='server limits resident execution to 8,000,000 cells'):
            Region(**settings, execution_mode='resident')
        with pytest.raises(ValueError, match='server limits resident execution to 8,000,000 cells'):
            Simulation(Project(region=Region(**settings), sources=[Source(center=(0, 0, 0))]))


def _finished(client, key, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/'+key).json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.05)
    raise AssertionError('job did not finish')


def test_streamed_browser_job_matches_the_resident_job(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCHFDTD_SCRATCH', str(tmp_path/'scratch'))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = scene()
        def run(mode):
            q = Project.model_validate(p.model_dump())
            q.region.execution_mode = mode
            validated = client.post('/api/validate', json=q.model_dump())
            assert validated.status_code == 200 and validated.json()['execution']['mode'] == mode
            key = client.post('/api/jobs', json=q.model_dump()).json()['id']
            job = _finished(client, key)
            assert job['status'] == 'completed', job.get('error')
            job['fields'] = client.get(f'/api/jobs/{key}/fields').json()
            job['plane'] = client.get(f'/api/jobs/{key}/field-monitors/plane', params=dict(component='Ey')).json()
            job['csv'] = client.get(f'/api/jobs/{key}/monitors.csv').text
            job['flux_csv'] = client.get(f'/api/jobs/{key}/flux.csv').text
            assert client.get(f'/api/jobs/{key}/download').content[:2] == b'PK'
            return job
        resident = run('resident')
        assert resident['execution']['mode'] == 'resident' and resident['summary']['steps'] == 60
        for mode in ('streamed_host', 'streamed_disk'):
            streamed = run(mode)
            execution = streamed['execution']
            assert execution['mode'] == mode and execution['policy']['checkpoints'] == 0
            assert execution['policy']['state_storage'] == ('disk' if mode == 'streamed_disk' else 'host')
            assert execution['reservation']['host_reservation_bytes'] > 0 and execution['scratch_directory'] == str(tmp_path/'scratch')
            assert execution['report']['completed_blocks'] == execution['report']['blocks'] == 15
            assert (execution['report']['disk_reservation_bytes'] > 0) == (mode == 'streamed_disk')
            summary = streamed['summary']
            assert summary['steps'] == 60 and not summary['cancelled'] and summary['execution']['snapshot'] == 'final step only'
            assert summary['backend'] == 'cpu' and summary['engine'].startswith('TorchFDTD streamed')
            assert [m['id'] for m in streamed['monitors']] == [m['id'] for m in resident['monitors']] == ['pt', 'ph']
            for a, b in zip(streamed['monitors'], resident['monitors']):
                scale = np.abs(b['signal']).max()
                np.testing.assert_allclose(a['signal'], b['signal'], rtol=1e-5, atol=1e-6*scale)
                np.testing.assert_allclose(a['time_fs'], b['time_fs'], rtol=1e-12)
                np.testing.assert_allclose(a['spectrum'], b['spectrum'], rtol=1e-5, atol=1e-6*np.abs(b['spectrum']).max())
            flux_a, flux_b = streamed['flux_monitors'][0], resident['flux_monitors'][0]
            assert flux_a['id'] == flux_b['id'] == 'plane' and flux_a['points'] == flux_b['points']
            np.testing.assert_allclose(flux_a['flux'], flux_b['flux'], rtol=1e-5, atol=1e-6*np.abs(flux_b['flux']).max())
            np.testing.assert_allclose(flux_a['frequency_thz'], flux_b['frequency_thz'], rtol=1e-12)
            plane_a, plane_b = streamed['plane'], resident['plane']
            assert plane_a['shape'] == plane_b['shape'] and plane_a['component'] == 'Ey'
            np.testing.assert_allclose(plane_a['magnitude'], plane_b['magnitude'], rtol=1e-5, atol=1e-6*np.abs(plane_b['magnitude']).max())
            np.testing.assert_allclose(plane_a['real'], plane_b['real'], rtol=1e-5, atol=1e-6*np.abs(plane_b['magnitude']).max())
            assert streamed['fields']['frame_steps'] == [60] and resident['fields']['frame_steps'][-1] == 60
            frame_a, frame_b = np.array(streamed['fields']['frames'][0]), np.array(resident['fields']['frames'][-1])
            assert frame_a.shape == frame_b.shape
            np.testing.assert_allclose(frame_a, frame_b, rtol=1e-5, atol=1e-6*np.abs(frame_b).max())
            np.testing.assert_allclose(streamed['fields']['epsilon'], resident['fields']['epsilon'], rtol=0, atol=0)
            assert streamed['csv'].count('\n') == resident['csv'].count('\n')
            assert streamed['flux_csv'].count('\n') == resident['flux_csv'].count('\n')
            ratio = client.get(f'/api/jobs/{streamed["id"]}/normalize-flux', params=dict(reference=resident['id'], monitor='plane'))
            assert ratio.status_code == 200
            np.testing.assert_allclose([v for v in ratio.json()['ratio'] if v is not None], 1., rtol=1e-5)
        assert not list((tmp_path/'scratch').glob('torchfdtd-state-*'))
    app.state.pool.shutdown()


def test_streamed_browser_job_cancels_between_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCHFDTD_SCRATCH', str(tmp_path/'scratch'))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = scene(steps=4000, execution_mode='streamed_host')
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/'+key).json()
            if job['status'] == 'running' and job['progress'].get('block', 0) >= 2:
                break
            time.sleep(.02)
        assert job['progress']['blocks'] == 1000 and job['progress']['tiles'] >= 1
        assert client.post(f'/api/jobs/{key}/cancel').json()['cancel_requested']
        job = _finished(client, key)
        assert job['status'] == 'cancelled' and job['summary']['cancelled']
        assert 0 < job['summary']['steps'] < 4000 and job['summary']['steps'] % 4 == 0
        fields = client.get(f'/api/jobs/{key}/fields').json()
        assert fields['frame_steps'] == [job['summary']['steps']] and len(fields['frames']) == 1
        assert len(job['monitors'][0]['signal']) == job['summary']['steps']
    app.state.pool.shutdown()


def test_large_auto_scene_validates_and_resolves_streamed(tmp_path):
    from torchfdtd.memory_profile import host_memory
    available = host_memory()['available_bytes']
    if available is not None and available < 6*2**30:
        pytest.skip('needs a few GiB of free RAM for the streamed reservation')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = Project(region=Region(dimension='3d', size=(25.6, 25.6, 12.8), mesh=.1, pml_cells=3, steps=100, backend='cpu'),
                    sources=[Source(center=(0, 0, 0))], monitors=[Monitor(center=(1, 0, 0))])
        assert np.prod(p.region.shape) > SERVER_LIMITS['resident_cells']
        response = client.post('/api/validate', json=p.model_dump())
        assert response.status_code == 200
        execution = response.json()['execution']
        assert not execution['resident']['fits'] and 'exceed the resident limit' in execution['resident']['reason']
        assert execution['mode'] == 'streamed_host', execution
        p.region.execution_mode = 'resident'
        assert client.post('/api/validate', json=p.model_dump()).status_code == 422
    app.state.pool.shutdown()


def test_budgeted_scene_is_refused_by_the_preflight_before_a_job_is_accepted(tmp_path):
    """memory_mode='budgeted' belongs to the adjoint API; the preflight reports the dispatch refusal and the job route returns 422."""
    p = scene(memory_mode='budgeted')
    message = 'Budgeted scenes require the adjoint API and an explicit resident byte budget.'
    with pytest.raises(ValueError, match=re.escape(message)):
        p.region.require_resident()
    record = resolve_execution(p, health=CPU_RECORD, scratch=tmp_path, summary=estimate(p))
    assert record['mode'] is None and record['error'] == message
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/validate', json=p.model_dump())
        assert response.status_code == 200
        assert response.json()['execution']['mode'] is None and response.json()['execution']['error'] == message
        response = client.post('/api/jobs', json=p.model_dump())
        assert response.status_code == 422 and response.json()['detail'] == message
        assert client.get('/api/jobs').json() == []
    app.state.pool.shutdown()


def _sparse_row(**overrides):
    """The 2D sparse pillar row of the tiling record: two 3 um tiles at 2 um overlap stitch it exactly."""
    from test_tiled import pillar_row
    p = pillar_row(period=1.25, count=4, half=(.1, .15), seed=5)
    p.region.execution_mode = 'tiled'
    p.region.tiling = p.region.tiling.model_copy(update={**dict(size_um=3., overlap_um=2., propagation_um=10.), **overrides})
    return Project.model_validate(p.model_dump())


def test_tiled_browser_job_matches_the_whole_device_plane(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        p = _sparse_row()
        validated = client.post('/api/validate', json=p.model_dump())
        assert validated.status_code == 200
        execution = validated.json()['execution']
        assert execution['mode'] == 'tiled' and execution['requested'] == 'tiled'
        plan = execution['tiled']['plan']
        assert plan['tiles'] == 2 and plan['overlap_um'] == pytest.approx(2.) and plan['propagation']['distance_um'] == 10.
        assert plan['suggestion']['overlap_um'] > 0 and not plan['below_suggestion']
        whole = Project.model_validate(p.model_dump())
        whole.region.execution_mode = 'resident'
        reference = client.post('/api/jobs', json=whole.model_dump()).json()['id']
        key = client.post('/api/jobs', json=p.model_dump()).json()['id']
        assert _finished(client, reference)['status'] == 'completed'
        job = _finished(client, key)
        assert job['status'] == 'completed', job.get('error')
        assert job['execution']['mode'] == 'tiled' and job['summary']['engine'].startswith('TorchFDTD resident tiles')
        report = job['execution']['report']
        assert len(report['pairs']) == 1 and set(report['pairs'][0]) >= {'tiles', 'axis', 'mismatch', 'mismatch_center', 'band_um'}
        indicator = job['execution']['indicator']
        # The record: this row stitches to machine precision once both tiles hold all four pillars.
        assert 0 <= indicator['max_mismatch_center'] < 1e-3 and 0 <= indicator['max_mismatch'] < 1e-3
        assert 'approximate' in indicator['explanation']
        ids = [m['id'] for m in job['flux_monitors']]
        assert ids == ['out', 'out-focal'] and job['monitors'] == []
        stitched = client.get(f'/api/jobs/{key}/field-monitors/out', params=dict(component='Ex')).json()
        direct = client.get(f'/api/jobs/{reference}/field-monitors/out', params=dict(component='Ex')).json()
        assert stitched['shape'] == direct['shape']
        a, b = np.array(stitched['real'])+1j*np.array(stitched['imag']), np.array(direct['real'])+1j*np.array(direct['imag'])
        assert np.linalg.norm(a-b)/np.linalg.norm(b) < 1e-3
        flux_tiled = [m for m in job['flux_monitors'] if m['id'] == 'out'][0]['flux']
        flux_whole = _finished(client, reference)['flux_monitors'][0]['flux']
        np.testing.assert_allclose(flux_tiled, flux_whole, rtol=1e-3)
        focal = client.get(f'/api/jobs/{key}/field-monitors/out-focal', params=dict(component='Ex')).json()
        assert focal['shape'] == stitched['shape'] and np.max(np.abs(focal['magnitude'])) > 0
        fields = client.get(f'/api/jobs/{key}/fields').json()
        assert len(fields['frames']) == 2 and job['summary']['execution']['frames'][1]['plane'] == 'focal'
        assert 'wavelength_um,signed_flux' in client.get(f'/api/jobs/{key}/flux.csv').text
        assert client.get(f'/api/jobs/{key}/download').content[:2] == b'PK'
    app.state.pool.shutdown()


def test_tiled_mode_reports_the_planner_rejection(tmp_path):
    p = _sparse_row()
    q = Project.model_validate(p.model_dump())
    q.sources[0] = Source(kind='point', center=(0, -.85, 0), component='Ex', wavelength=1.55)
    q = Project.model_validate(q.model_dump())
    record = resolve_execution(q, health=CPU_RECORD, scratch=tmp_path, summary=estimate(q))
    assert record['mode'] is None and 'Tiling rejected the scene' in record['error']
    assert 'point, one-way and TFSF sources are not partitioned' in record['error']
    below = _sparse_row(overlap_um=.25)
    record = resolve_execution(below, health=CPU_RECORD, scratch=tmp_path, summary=estimate(below))
    assert record['mode'] == 'tiled' and record['tiled']['plan']['below_suggestion']
    assert any('below the suggested' in w for w in record['warnings'])
    app = create_app(tmp_path)
    with TestClient(app) as client:
        # The preflight rejection refuses the submission; nothing is queued to fail at dispatch.
        response = client.post('/api/jobs', json=q.model_dump())
        assert response.status_code == 422 and 'Tiling rejected the scene' in response.json()['detail']
        assert client.get('/api/jobs').json() == []
    app.state.pool.shutdown()


def test_point_traces_and_spectra_enter_the_resident_estimate_and_can_be_refused(tmp_path):
    from pydantic import ValidationError
    steps = 4000
    one = scene(steps=steps).model_copy(update=dict(monitors=[Monitor(id='m0', center=(.3, .1, 0), component='Ez')]))
    many = one.model_copy(update=dict(monitors=[Monitor(id=f'm{i}', center=(.3, .1, 0), component='Ez') for i in range(400)]))
    per_trace = steps*4+(steps//2+1)*16  # float32 samples plus the complex128 FFT spectrum
    assert estimate(one)['point_trace_estimated_bytes'] == per_trace
    assert estimate(many)['point_trace_estimated_bytes'] == 400*per_trace
    # Each trace also has its host copy in the result, one float32 sample per step.
    assert estimate(many)['estimated_memory_mb']-estimate(one)['estimated_memory_mb'] == pytest.approx(399*(per_trace+steps*4)/2**20, abs=.1)
    from torchfdtd import Boundaries, BoundaryFace
    bloch = one.model_copy(update=dict(region=one.region.model_copy(update=dict(
        boundaries=Boundaries(y_min=BoundaryFace(kind='bloch'), y_max=BoundaryFace(kind='bloch')), bloch_phase=(0, .3, 0)))))
    assert estimate(bloch)['point_trace_estimated_bytes'] == steps*8+(steps//2+1)*16
    dft = one.model_copy(update=dict(monitors=[Monitor(id='m0', center=(.3, .1, 0), component='Ez', time_downsample=2,
                                                       spectrum=dict(sampling='frequency', wavelength_start=.9, wavelength_stop=1.2, frequency_points=7, apodization='none'))]))
    assert estimate(dft)['point_trace_estimated_bytes'] == steps*4+7*16
    available = int((int(estimate(one)['estimated_memory_mb']*2**20)+200*per_trace)/.8)
    fits = resolve_execution(one, health=dict(CPU_RECORD, host_available_bytes=available), scratch=tmp_path/'scratch', summary=estimate(one))
    refused = resolve_execution(many, health=dict(CPU_RECORD, host_available_bytes=available), scratch=tmp_path/'scratch', summary=estimate(many))
    assert fits['resident']['fits'] and fits['mode'] == 'resident'
    assert not refused['resident']['fits'] and 'exceeds 80%' in refused['resident']['reason'] and refused['mode'] != 'resident'
    assert len(one.model_copy(update=dict(monitors=[Monitor(id=f'm{i}', center=(.3, .1, 0), component='Ez') for i in range(512)])).monitors) == 512
    # The Python API admits the trace memory instead of counting monitors; the server keeps 512, a user cap refuses.
    beyond = dict(one.model_dump(), monitors=[Monitor(id=f'm{i}', center=(.3, .1, 0), component='Ez').model_dump() for i in range(513)])
    assert len(Project.model_validate(beyond).monitors) == 513
    with pytest.raises(ValidationError, match='513 monitors exceed the limit of 512'):
        Project.model_validate(dict(beyond, limits=dict(max_monitors=512)))
    with server_limits(), pytest.raises(ValidationError, match='513 monitors exceed the limit of 512'):
        Project.model_validate(beyond)
    sources = dict(one.model_dump(), sources=[Source(id=f's{i}', center=(-.3, 0, 0), component='Ez', wavelength=1., pulse_cycles=2).model_dump() for i in range(513)])
    assert len(Project.model_validate(sources).sources) == 513
    with server_limits(), pytest.raises(ValidationError, match='513 sources exceed the limit of 512'):
        Project.model_validate(sources)
