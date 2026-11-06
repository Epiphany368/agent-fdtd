"""Synthetic native conversion fixtures and physics checks, no vendor files."""
import math
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import DIPOLE, FDTD, ROOT, TIME, ZERO_UUID, convert_fsp
from torchfdtd.models import Project, Region, Source
from torchfdtd.server import create_app
from torchfdtd.solver import Simulation, source_time_signal
from test_fsp_binary import legacy_circle, mapping, node, scripted_group, string, u


def items(data):
    return {k: (0 if isinstance(v, str) else 2 if isinstance(v, int) else 3 if isinstance(v, np.ndarray) else 1, v)
            for k, v in data.items()}


def dipole_settings(name='::model::source', **overrides):
    source = dict(name=name, enabled=1, sourceType=0, useGlobalSource=0,
                  sourcePreference=0, frequencyEnvelopeType=0, optimizeForShortPulse=0, mEliminateDC=0, eliminateDiscontinuities=0,
                  theta=0., angle=0., frequency=299792458/1.55e-6, amplitude0=1.,
                  pulseLength=2e-15, offset=4e-15, phase=23., xcoord=-.5e-6, ycoord=0., zcoord=0.)
    return {**source, **overrides}


def time_monitor_settings(name='::model::monitor', **overrides):
    monitor = dict(name=name, enabled=1, monitorShape=0, spatialAveraging=1,
                   recordInPML=0, simulationType=0, outputPower=0, startTime=0., stopMethod=0, downsampleT=1,
                   outputP=np.zeros(3), outputE=np.array([0, 0, 1, 0, 0, 0]), left=.5e-6, bottom=0., z1=0.)
    return {**monitor, **overrides}


def analysis_group(name, members, enabled=1, setupscript='adddipole;'):
    return node(ROOT, items(dict(name=name, enabled=enabled, x=.2e-6, y=0., z=0., use_relative_coordinates=1,
                                 setupscript=setupscript, analysisscript='', constructionflag=1)), members)


def fixture(source_overrides=None, region_overrides=None, *, source_class=DIPOLE,
            monitor_class=TIME, monitor_overrides=None, root_overrides=None, extra=()):
    dx = .1e-6; dt = .8*dx/299792458/math.sqrt(3)
    region = dict(name='::model::FDTD', enabled=1, customGrid=2, meshRefineDesired=5,
                  fullSymmetry=0, forceComplex=0, splitFieldFDTD=0, checkpointDuringSimulation=0,
                  checkpointAtShutoff=0, mMaterialId=ZERO_UUID, mRefIndex='1.0', dimension=1,
                  dx=dx, dy=dx, dz=dx, PMLLayersV7p0=np.full((6, 1), 4), GUIx=0., GUIy=0.,
                  GUIz1=-2e-6, GUIz2=2e-6, GUIwidth=4e-6, GUIheight=4e-6,
                  courantFactor=.8, dt=dt, MaxSimTime=20*dt, useAutoShutoffMin=0, useAutoShutoffMax=1)
    region.update({f'BCType{i}':0 for i in range(6)})
    region.update({a+'Grid':np.linspace(-2.4e-6, 2.4e-6, 49) for a in 'xyz'})
    source = dipole_settings(**(source_overrides or {}))
    region.update(region_overrides or {})
    monitor = time_monitor_settings(**(monitor_overrides or {}))
    # Controlled legacy sphere with real scalar transforms and material order.
    tail = bytearray(167)
    tail[:23] = b'\0'+u(0)+b'\1'+struct.pack('<d', .3e-6)+b'\1'+struct.pack('<d', .3e-6)
    tail[23:38] = (b'\0'+struct.pack('<i', -1))*3
    tail[38:91] = u(3)+b'\0'+u(3)+u(2)+u(3)+u(1)+u(1)+u(1)+bytes(24)
    tail[-56:-52] = u(1); tail[-51:-47] = u(2); tail[-9:-5] = u(1)
    body = u(8)+u(25)+struct.pack('<i5d', -1, 2., .3e-6, 0., 0., 0.)+string('2')+bytes(8)+string('sphere')
    body += tail+mapping(items(dict(materialuuid=ZERO_UUID, use_relative_coordinates=1, gridAttributeName='')))+u(0)
    sphere = u(1000)+u(38)+b'{23046316-141b-4111-aa2f-9e18de790b6c}'+body
    root = node(ROOT, items({**dict(name='::model', setupscript='', analysisscript='', enabled=1,
                                    constructionflag=0, x=0., y=0., z=0.), **(root_overrides or {})}),
                [node(FDTD, items(region)), sphere, node(source_class, items(source)), node(monitor_class, items(monitor)), *extra])
    return (b'LUMERICAL file version 1.1\0'+struct.pack('<5I', 3, 10, 4, 8, 1)+bytes(24)
            +b'table of contents\0end table of contents\0Lumerical material data file version 2.0\0'
            +u(0)+root+u(0)+bytes(12))


def test_conversion_retains_geometry_timestep_pulse_and_fingerprint(monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent conversion loaded Lumerical'))
    raw = fixture(); doc = FspDocument(raw); report = convert_fsp(doc, 'Synthetic sphere', 'cpu')
    assert report.project is not None, report.issues
    p = report.project
    assert p.region.shape == (48, 48, 48)
    assert p.region.courant_factor == .8
    assert p.structures[0].radius == pytest.approx(.3)
    assert p.materials[0].index == 2
    assert p.sources[0].center == pytest.approx((-.5, 0, 0))
    assert p.sources[0].pulse_length == 2e-15 and p.sources[0].phase == 23
    assert p.import_provenance.source_sha256 == doc.fingerprint()
    result = Simulation(p).run()
    np.testing.assert_allclose(np.diff(result.times), p.region.time_step, rtol=1e-13)
    assert np.max(abs(result.electric)) > 0
    assert doc.data == raw


@pytest.mark.parametrize('class_id,key,bad', [(DIPOLE,'frequencyEnvelopeType',3),
    (DIPOLE,'sourceType',1), (DIPOLE,'useGlobalSource',1), (FDTD,'meshRefineDesired',0),
    (FDTD,'mMaterialId','a dispersive material'), (FDTD,'dt',1e-12),
    (TIME,'outputPower',1), (TIME,'monitorShape',6), (FDTD,'BCType0',5)])
def test_unsupported_physics_never_produces_a_runnable_partial_scene(class_id, key, bad):
    doc = FspDocument(fixture())
    target = next(n for n in doc.nodes() if n.uid == class_id)
    target.properties[key].value = bad
    report = convert_fsp(doc)
    assert report.project is None
    assert any(i['severity'] == 'error' for i in report.issues)
    assert not report.as_dict()['native_execution_allowed']


def test_nonzero_rotations_and_ellipsoids_are_not_silently_dropped():
    for change in ({'rotation_axes':['x','none','none'], 'rotation_angles':[17,0,0]},
                   {'make ellipsoid':True,'rotation_axes':['z','none','none'],'rotation_angles':[17,0,0]}):
        doc = FspDocument(fixture()); shape = next(n for n in doc.nodes() if n.legacy)
        shape.legacy.update(change)
        p=convert_fsp(doc).project
        assert p is not None
        assert p.structures[0].rotation_axes==tuple(change['rotation_axes'])
        assert p.structures[0].rotation_angles==tuple(change['rotation_angles'])
        assert p.structures[0].make_ellipsoid==change.get('make ellipsoid',False)


def test_standard_pulse_fwhm_phase_and_legacy_compatibility():
    source = Source(time_definition='standard', pulse_length=16e-15, pulse_offset=40e-15, phase=90)
    assert source_time_signal(source, [40e-15])[0] == pytest.approx(1)
    t = np.linspace(0, 100e-15, 201)
    w = source.pulse_length/(2*math.sqrt(math.log(2)))
    expected = np.exp(-.5*((t-source.pulse_offset)/w)**2)*np.cos(-2*np.pi*299792458/(source.wavelength*1e-6)*(t-source.pulse_offset))
    np.testing.assert_allclose(source_time_signal(source, t), expected, atol=1e-14)
    assert np.exp(-.5*((source.pulse_length/2)/w)**2)**2 == pytest.approx(.5)
    legacy = Source(); w = legacy.pulse_cycles*legacy.wavelength*1e-6/299792458
    np.testing.assert_allclose(source_time_signal(legacy,t), np.exp(-.5*((t-4*w)/w)**2)*np.sin(2*np.pi*299792458/(legacy.wavelength*1e-6)*t))
    assert Region(size=(6.000000000000001, 6, 6), mesh=6/128, dimension='3d').shape == (128,128,128)
    assert Region(size=(6.000001, 6, 6), mesh=6/128, dimension='3d').shape == (129,128,128)


@pytest.mark.parametrize('precision', ['float32','float64'])
def test_imported_standard_pulse_with_nondefault_cfl_cpu_cuda_parity(precision):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA GPU not available')
    p = convert_fsp(FspDocument(fixture()), backend='cpu').project
    p.region.precision = precision
    cpu = Simulation(p).run()
    p.region.backend = 'cuda'
    gpu = Simulation(p).run()
    tolerance = 2e-6 if precision=='float32' else 1e-13
    assert gpu.summary['cuda_graph']
    np.testing.assert_allclose(gpu.electric, cpu.electric, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(gpu.signals, cpu.signals, rtol=tolerance, atol=tolerance)


def test_native_upload_download_and_run_without_vendor_runtime(tmp_path, monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp, 'availability', lambda: {'installed':False, 'reason':'Not installed'})
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent import loaded Lumerical'))
    app = create_app(tmp_path)
    try:
        with TestClient(app) as client:
            raw = fixture()
            response = client.post('/api/fsp/native-import', content=raw, headers={'x-filename':'test.fsp'})
            assert response.status_code == 202
            key = response.json()['id']
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                job = client.get('/api/fsp/'+key).json()
                if job['status'] in ('ready', 'failed'):break
                time.sleep(.01)
            assert job['status'] == 'ready', job
            assert job['conversion']['native_execution_allowed']
            assert client.get('/api/fsp/'+key+'/download').content == raw
            assert client.get('/api/fsp/'+key+'/conversion').json()['requires_lumerical'] is False
            assert client.get('/api/fsp/'+key+'/archive').status_code == 409
            assert client.post('/api/validate', json=job['conversion']['project']).status_code == 200
    finally:
        app.state.pool.shutdown(); app.state.fsp_pool.shutdown()


def sampled_material(uid, name='Synthetic glass', wavelengths_um=np.geomspace(.4, 20, 80)):
    """A type-7 record: embedded frequency/permittivity samples of a lossy dielectric."""
    frequency = 299792458/(wavelengths_um*1e-6);omega = 2*np.pi*frequency
    w0 = 2*np.pi*299792458/.25e-6  # one ultraviolet Lorentz resonance: Sellmeier-like dispersion with small loss
    epsilon = 2.25+.6*w0**2/(w0**2-omega**2-1j*.05*w0*omega)
    return mapping(items(dict(name=name, materialuuid=uid, type=7, anisotropy=0, priority=3, maxpoles=4, red=.1, green=.6, blue=.9,
                              frequency=frequency.reshape(-1, 1), permittivity=epsilon.reshape(-1, 1))))


def bloch_fixture(units=1, kx=0., ky=.5, based_on_source=0, extra=(), sampled=None, **overrides):
    """3D fixture with paired Bloch faces on x and y, ghost planes included, PML on z."""
    grid = np.linspace(-2.1e-6, 2.1e-6, 43)  # 41 interior nodes = the 4 um cell plus one ghost plane per side
    region = dict(BCType0=5, BCType1=5, BCType2=5, BCType3=5, PMLLayersV7p0=np.array([0, 0, 0, 0, 4, 4]).reshape(6, 1),
                  xGrid=grid, yGrid=grid, blochUnits=units, blochBasedOnSource=based_on_source, kx=kx, ky=ky, kz=0.,
                  frequencyEnvelopeType=0, sourcePreference=0, globalFrequency=299792458/1.55e-6, globalEliminateDC=0,
                  pulseLength=2e-15, offset=4e-15, optimizeForShortPulse=0, eliminateDiscontinuities=0)
    region.update(overrides.pop('region_overrides', None) or {})
    raw = fixture(region_overrides=region, extra=extra, **overrides)
    if sampled is not None:
        marker = b'Lumerical material data file version 2.0\0'+u(0)
        raw = raw.replace(marker, marker[:-4]+u(1)+sampled, 1)
    return raw


def test_bloch_boundaries_map_the_saved_wavevector_in_both_unit_systems(monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent conversion loaded Lumerical'))
    report = convert_fsp(FspDocument(bloch_fixture(units=0, kx=.25, ky=.5)), 'Bloch', 'cpu')
    assert report.project is not None, report.issues
    p = first = report.project
    assert [f.kind for a in range(3) for f in p.region.boundaries.pair(a)] == ['bloch']*4+['pml']*2
    assert p.region.bloch_phase == pytest.approx((2*math.pi*.25, math.pi, 0))
    assert p.region.shape == (40, 40, 48) and p.region.size[:2] == pytest.approx((4, 4))
    assert sum(i['code'] == 'bloch_phase' and i['severity'] == 'info' and 'exp(+i*phase)' in i['message'] for i in report.issues) == 2
    p = convert_fsp(FspDocument(bloch_fixture(units=1, kx=0., ky=2.5e5)), 'Bloch', 'cpu').project
    assert p is not None and p.region.bloch_phase == pytest.approx((0, 2.5e5*4e-6, 0))
    report = convert_fsp(FspDocument(bloch_fixture(units=0, kx=.25, ky=.5, based_on_source=1)), 'Bloch', 'cpu')
    assert report.project is not None and report.project.region.bloch_phase == (0, 0, 0)
    assert sum(i['code'] == 'bloch_source_angle' and i['severity'] == 'warning' for i in report.issues) == 2
    report = convert_fsp(FspDocument(bloch_fixture(units=0, region_overrides={'BCType1': 1})), 'Bloch', 'cpu')
    assert report.project is None and any('[5, 1]' in i['message'] for i in report.issues)
    result = Simulation(first).run()
    assert np.iscomplexobj(result.electric) and np.isfinite(result.electric).all() and np.max(abs(result.electric)) > 0


def test_scripted_group_geometry_sampled_material_and_skipped_objects_convert_to_a_runnable_project(tmp_path, monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent conversion loaded Lumerical'))
    uid = '{11111111-2222-3333-4444-555555555555}'
    inside = legacy_circle(x=.5e-6, y=-.4e-6, index=1.71)
    outside = legacy_circle(x=3e-6, y=0., index=1.71)  # entirely beyond the 4 um cell after the group shift
    glass = legacy_circle(x=-.6e-6, y=.6e-6, material=uid, radius=.2e-6)
    group = scripted_group('array', 'addcircle;', [('nx', 0, '2')], [inside, outside, glass], x=.1e-6, y=0., z=.3e-6)
    raw = bloch_fixture(units=0, kx=0., ky=.5, extra=[group], sampled=sampled_material(uid),
                        root_overrides={'setupscript': 'set("x",0);'}, monitor_overrides={'enabled': 0, 'monitorShape': 6})
    doc = FspDocument(raw)
    report = convert_fsp(doc, 'Scripted', 'cpu')
    assert report.project is not None, report.issues
    p = report.project
    by_name = {s.name: s for s in p.structures}
    assert sorted(by_name) == ['array circle 1', 'array circle 3', 'sphere']
    assert by_name['array circle 1'].center == pytest.approx((.6, -.4, .3)) and by_name['array circle 1'].material == 'FSP n=1.71'
    assert by_name['array circle 3'].radius == pytest.approx(.2) and by_name['array circle 3'].mesh_order == 3
    glass_material = next(m for m in p.materials if m.name == by_name['array circle 3'].material)
    assert glass_material.model == 'multipole' and glass_material.poles and glass_material.provenance.source.startswith('FSP embedded sampled data')
    assert glass_material.color == '#1a99e6'
    codes = {(i['severity'], i['code']) for i in report.issues}
    assert ('warning', 'model_script') in codes and ('info', 'group_script') in codes and ('error', 'object_mapping') not in codes
    assert any(i['code'] == 'group_script' and '3 saved generated objects' in i['message'] and '1 lying entirely outside' in i['message'] for i in report.issues)
    fit = next(i for i in report.issues if i['code'] == 'material_fit')
    assert fit['severity'] == 'warning' and 'normalized RMS' in fit['message'] and 'Synthetic glass' in fit['message']
    assert '(union of the enabled source ranges)' in fit['message']  # a standard global pulse stores no limits
    limited = convert_fsp(FspDocument(bloch_fixture(units=0, extra=[group], sampled=sampled_material(uid),
                                                    region_overrides=dict(sourcePreference=2, BBFrequencyStart=299792458/1.8e-6, BBFrequencyStop=299792458/1.2e-6))), 'Limits', 'cpu')
    fit = next(i for i in limited.issues if i['code'] == 'material_fit')
    assert 'fitted over 1.2-1.8 um (FSP global source limits)' in fit['message'], fit['message']
    assert next(m for m in limited.project.materials if 'Synthetic glass' in m.name).fit_band_um[0] <= 1.2
    assert any(i['code'] == 'object_mapping' and i['severity'] == 'warning' and i['message'].startswith('Disabled object is not imported') for i in report.issues)
    assert p.monitors == [] and len(p.sources) == 1
    assert {m['object_id'] for m in report.mappings} == {'array', 'sphere', '::model::source'}
    assert next(m for m in report.mappings if m['object_id'] == 'array')['native_ids'] == [by_name['array circle 1'].id, by_name['array circle 3'].id]
    rotated = scripted_group('twist', '', [], [inside], axes=(2, -1, -1), angles=(90., 0., 0.))
    report = convert_fsp(FspDocument(bloch_fixture(extra=[rotated])), 'Rotated', 'cpu')
    assert report.project is None and any('Rotated structure groups' in i['message'] for i in report.issues)
    monkeypatch.setattr(fsp, 'availability', lambda: {'installed': False, 'reason': 'Not installed'})
    app = create_app(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post('/api/fsp/native-import', content=raw, headers={'x-filename': 'moir%C3%A9%20cavity%20-%2090%20twist.fsp'})
            assert response.status_code == 202
            key = response.json()['id']
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                job = client.get('/api/fsp/'+key).json()
                if job['status'] in ('ready', 'failed'):break
                time.sleep(.01)
            assert job['status'] == 'ready', job
            assert job['filename'] == 'moiré cavity - 90 twist.fsp' and job['conversion']['native_execution_allowed']
            validated = client.post('/api/validate', json=job['conversion']['project'])
            assert validated.status_code == 200, validated.text
            assert validated.json()['cells'] == 40*40*48
            # Generated objects execute natively but are never written back.
            export = client.post('/api/fsp/'+key+'/native-export', json=job['conversion']['project'])
            assert export.status_code == 202
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                edited = client.get('/api/fsp/'+export.json()['id']).json()
                if edited['status'] in ('ready', 'failed'):break
                time.sleep(.01)
            assert edited['status'] == 'ready', edited
            moved = dict(job['conversion']['project']);moved['structures'] = [dict(s, center=(s['center'][0]+.1, s['center'][1], s['center'][2])) if s['name'] == 'array circle 1' else s for s in moved['structures']]
            export = client.post('/api/fsp/'+key+'/native-export', json=moved)
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                edited = client.get('/api/fsp/'+export.json()['id']).json()
                if edited['status'] in ('ready', 'failed'):break
                time.sleep(.01)
            assert edited['status'] == 'failed' and 'Script-generated group objects are not written back' in edited['error']
    finally:
        app.state.pool.shutdown(); app.state.fsp_pool.shutdown()


def test_enabled_analysis_group_members_import_with_global_coordinates_and_caps(tmp_path, monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent conversion loaded Lumerical'))
    cloud = analysis_group('::model::cloud', [
        node(DIPOLE, items(dipole_settings('::model::cloud::s1', xcoord=.3e-6, ycoord=-.2e-6, theta=40., angle=30., phase=5.))),
        node(TIME, items(time_monitor_settings('::model::cloud::m1', left=.4e-6, bottom=.1e-6, spatialAveraging=0))),
        node(TIME, items(time_monitor_settings('::model::cloud::m2', left=-.4e-6, enabled=0, monitorShape=6)))])
    off = analysis_group('::model::off', [node(TIME, items(time_monitor_settings('::model::off::t1')))], enabled=0)
    raw = bloch_fixture(units=0, kx=0., ky=.5, extra=[cloud, off])
    report = convert_fsp(FspDocument(raw), 'Cloud', 'cpu')
    assert report.project is not None, report.issues
    p = report.project
    assert [s.name for s in p.sources] == ['source', 's1'] and [m.name for m in p.monitors] == ['monitor', 'm1']
    # Members keep their stored global coordinates: the group offset (0.2 um) is not added.
    assert p.sources[1].center == pytest.approx((.3, -.2, 0)) and p.sources[1].theta == 40 and p.sources[1].phi == 30 and p.sources[1].phase == 5
    assert p.monitors[1].center == pytest.approx((.4, .1, 0)) and p.monitors[1].component == 'Ez'
    issues = {(i['severity'], i['code'], i['object_id']) for i in report.issues}
    assert ('info', 'group_script', '::model::cloud') in issues
    assert ('warning', 'point_interpolation', '::model::cloud::m1') in issues
    assert ('warning', 'object_mapping', '::model::cloud::m2') in issues and ('warning', 'object_mapping', '::model::off') in issues
    assert not any(i['severity'] == 'error' for i in report.issues)
    row = next(m for m in report.mappings if m['object_id'] == '::model::cloud')
    assert row['script_generated'] and row['native_ids'] == [p.sources[1].id, p.monitors[1].id]
    result = Simulation(p).run()
    assert np.isfinite(result.electric).all() and np.max(abs(result.electric)) > 0
    big = analysis_group('::model::big', [node(TIME, items(time_monitor_settings(f'::model::big::t{i}', left=(i % 30)*.1e-6, bottom=(i // 30)*.1e-6, outputE=np.ones(6))))
                                          for i in range(86)])  # 516 traces, above the 512-trace product limit
    report = convert_fsp(FspDocument(bloch_fixture(extra=[big])), 'Big', 'cpu')
    assert report.project is None
    cap = next(i for i in report.issues if i['object_id'] == '::model::big' and i['severity'] == 'error')
    assert '516 native monitors' in cap['message'] and '1 already imported' in cap['message'] and 'limit of 512' in cap['message']
    monkeypatch.setattr(fsp, 'availability', lambda: {'installed': False, 'reason': 'Not installed'})
    app = create_app(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post('/api/fsp/native-import', content=raw, headers={'x-filename': 'cloud.fsp'})
            key = response.json()['id']
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                job = client.get('/api/fsp/'+key).json()
                if job['status'] in ('ready', 'failed'):break
                time.sleep(.01)
            assert job['status'] == 'ready' and job['conversion']['native_execution_allowed'], job
            assert client.post('/api/validate', json=job['conversion']['project']).status_code == 200
            outcomes = {}
            for route in ('native-export', 'native-scene-export'):
                export = client.post(f'/api/fsp/{key}/{route}', json=job['conversion']['project'])
                assert export.status_code == 202
                deadline = time.monotonic()+20
                while time.monotonic() < deadline:
                    edited = client.get('/api/fsp/'+export.json()['id']).json()
                    if edited['status'] in ('ready', 'failed'):break
                    time.sleep(.01)
                outcomes[route] = edited
            assert outcomes['native-export']['status'] == 'ready', outcomes['native-export']
            assert outcomes['native-scene-export']['status'] == 'failed' and 'Analysis-group members' in outcomes['native-scene-export']['error']
    finally:
        app.state.pool.shutdown(); app.state.fsp_pool.shutdown()
