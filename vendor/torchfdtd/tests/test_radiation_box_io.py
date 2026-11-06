"""Synthetic data-only archive gates, without a native field solve."""
import io
import json
import sys
import struct
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from torchfdtd.radiation_box_io import load_native_radiation_box


FACES = [a+s for a in 'xyz' for s in ('_min', '_max')]
IDS = {name: name for name in FACES}
KW = dict(bounds_um=[[-.5, .5]]*3, refractive_index=1.)


def archive(path, *, frequencies=2, points=4):
    from torchfdtd.models import Project, Region
    project = Project(name='synthetic', region=Region(dimension='3d', size=(2.,)*3,
        mesh=.1, pml_cells=3, steps=10), sources=[], monitors=[])
    arrays = dict(project=np.asarray(project.model_dump_json()),
        summary=np.asarray(json.dumps({'cancelled': False})),
        field_monitors=np.asarray(json.dumps([{'id': name} for name in FACES])),
        E=np.asarray('not a numerical field'), frames=np.asarray('never load'))
    for index in range(6):
        prefix = f'field_monitor_{index}_'
        arrays.update({prefix+'fields': np.full((frequencies, points, 6), index+1j, dtype=np.complex64),
            prefix+'frequency_hz': np.arange(1, frequencies+1, dtype=np.float64),
            prefix+'points_um': np.zeros((points, 3), dtype=np.float64),
            prefix+'weights': np.ones(points, dtype=np.float64)})
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture
def adapter(monkeypatch):
    calls = []
    def adapt(value, mapping, **kwargs):
        calls.append((value, mapping, kwargs))
        return SimpleNamespace(report={'adapter': True})
    monkeypatch.setitem(sys.modules, 'torchfdtd.radiation_box', SimpleNamespace(native_radiation_box=adapt))
    return calls


def rewrite(source, target, name, payload):
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, 'w') as new:
        for item in old.infolist():
            new.writestr(item.filename, payload if item.filename == name else old.read(item))
    return target


def npy(value):
    output = io.BytesIO()
    np.save(output, value)
    return output.getvalue()


def test_selected_six_faces_reference_and_full_frequency_forwarding(tmp_path, adapter, monkeypatch):
    path = archive(tmp_path/'sample.npz')
    opened = []
    original = zipfile.ZipFile.open
    def track(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'open', track)
    result = load_native_radiation_box(path, IDS, reference_path=path, frequency_index=1, **KW)
    assert result.report == {'adapter': True}
    sample, mapping, kwargs = adapter[0]
    assert mapping == IDS and kwargs['reference_monitor_ids'] == IDS
    assert kwargs['frequency_index'] == 1
    assert all(face['fields'].shape == (2, 4, 6) for face in sample['frequency_fields'])
    assert kwargs['reference']['frequency_fields'][0]['fields'] is not sample['frequency_fields'][0]['fields']
    assert not {'E.npy', 'frames.npy'} & set(opened)


def test_open_surface_subset_is_forwarded_and_flag_required(tmp_path, adapter):
    path = archive(tmp_path/'sample.npz')
    subset = {'z_max': 'z_max', 'x_min': 'x_min'}
    load_native_radiation_box(path, subset, open_surface=True, **KW)
    sample, mapping, kwargs = adapter[0]
    assert mapping == subset and kwargs['open_surface'] is True
    assert [face['id'] for face in sample['frequency_fields']] == ['z_max', 'x_min']
    with pytest.raises(ValueError, match='Exactly six'):
        load_native_radiation_box(path, subset, **KW)
    with pytest.raises(ValueError, match='one to five'):
        load_native_radiation_box(path, IDS, open_surface=True, **KW)
    with pytest.raises(ValueError, match='boolean'):
        load_native_radiation_box(path, subset, open_surface=1, **KW)
    assert len(adapter) == 1


def test_aggregate_reference_and_full_frequency_bytes_before_numeric_load(tmp_path, adapter, monkeypatch):
    path = archive(tmp_path/'many-frequencies.npz', frequencies=64, points=64)
    budget = 10*1024**2
    load_native_radiation_box(path, IDS, host_budget_bytes=budget, **KW)
    original = np.load
    def only_metadata(source, **kwargs):
        assert source.name in ('project.npy', 'summary.npy', 'field_monitors.npy')
        return original(source, **kwargs)
    monkeypatch.setattr(np, 'load', only_metadata)
    with pytest.raises(ValueError, match='Aggregate full-frequency'):
        load_native_radiation_box(path, IDS, reference_path=path, host_budget_bytes=budget, **KW)
    # Selecting a single frequency cannot hide the full decompression footprint.
    with pytest.raises(ValueError, match='Aggregate full-frequency'):
        load_native_radiation_box(path, IDS, frequency_index=0, host_budget_bytes=1024**2, **KW)
    assert len(adapter) == 1


@pytest.mark.parametrize('defect', ['shape', 'dtype', 'object', 'huge_shape'])
def test_invalid_numeric_headers_rejected_before_numeric_allocation(tmp_path, adapter, monkeypatch, defect):
    source = archive(tmp_path/'source.npz')
    key = 'field_monitor_0_fields.npy'
    if defect == 'shape': payload = npy(np.zeros((2, 4, 5), np.complex64))
    elif defect == 'dtype': payload = npy(np.zeros((2, 4, 6), np.float64))
    elif defect == 'object': payload = npy(np.asarray([object()], dtype=object))
    else:
        data = io.BytesIO()
        np.lib.format.write_array_header_1_0(data, {'descr': '<c8', 'fortran_order': False,
            'shape': (999999999999, 4, 6)})
        payload = data.getvalue()
    path = rewrite(source, tmp_path/'bad.npz', key, payload)
    original = np.load
    def metadata_only(handle, **kwargs):
        assert 'field_monitor_' not in handle.name
        return original(handle, **kwargs)
    monkeypatch.setattr(np, 'load', metadata_only)
    with pytest.raises(ValueError): load_native_radiation_box(path, IDS, **KW)
    assert adapter == []


def test_metadata_limit_before_any_numpy_load(tmp_path, adapter, monkeypatch):
    path = archive(tmp_path/'sample.npz')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Allocated before metadata admission.'))
    with pytest.raises(ValueError, match='metadata'):
        load_native_radiation_box(path, IDS, archive_metadata_budget_bytes=100, **KW)


def test_duplicate_entry_and_invalid_frequency_are_explicit(tmp_path, adapter):
    path = archive(tmp_path/'sample.npz')
    with pytest.raises(ValueError, match='Frequency index'):
        load_native_radiation_box(path, IDS, frequency_index=2, **KW)
    with zipfile.ZipFile(path, 'a') as output:
        with pytest.warns(UserWarning):
            output.writestr('field_monitor_0_weights.npy', npy(np.ones(4)))
    with pytest.raises(ValueError, match='exactly one'):
        load_native_radiation_box(path, IDS, **KW)


def test_reject_missing_face_mapping_or_reference_path(tmp_path, adapter):
    path = archive(tmp_path/'sample.npz')
    with pytest.raises(ValueError, match='six'):
        load_native_radiation_box(path, {'x_min': 'x_min'}, **KW)
    with pytest.raises(ValueError, match='reference archive'):
        load_native_radiation_box(path, IDS, reference_monitor_ids=IDS, **KW)


def test_live_memory_rejection_precedes_numpy_allocation(tmp_path, monkeypatch):
    path = archive(tmp_path/'sample.npz')
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory', lambda: {'available_bytes': 1})
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Loaded before live RAM admission.'))
    with pytest.raises(ValueError, match='available RAM'):
        load_native_radiation_box(path, IDS, **KW)


def test_analytic_six_face_archive_matches_real_box_adapter(tmp_path):
    # Independent closed-dipole fixture, no FDTD execution. This checks the actual
    # loader-to-box boundary as well as the synthetic allocation probes above.
    from test_radiation import dipole_faces
    from torchfdtd.models import Project, Region, Source
    from torchfdtd.radiation_box import native_radiation_box
    faces, bounds, _ = dipole_faces(4)
    project = Project(region=Region(dimension='3d', size=(3.,)*3, mesh=.1,
        pml_cells=3, background_index=1.3, precision='float64', steps=10),
        sources=[Source(component='Ez', center=(0., 0., 0.))], structures=[], monitors=[])
    summary = dict(steps=10, termination_reason='steps', cancelled=False)
    records, metadata, arrays = [], [], {}
    for index, name in enumerate(FACES):
        face = faces[name]
        info = dict(id=name, components=list(face.components), flux_units=face.flux_units,
            run_signature=face.run_signature, shape=list(face.shape), normal_axis=face.normal,
            settings=dict(spectrum=dict(apodization='none'), time_downsample=1))
        values = {key: getattr(face, key).numpy() for key in ('fields','frequency_hz','points_um','weights')}
        records.append({**info, **values}); metadata.append(info)
        arrays.update({f'field_monitor_{index}_{key}': value for key, value in values.items()})
    path = tmp_path/'analytic.npz'
    np.savez_compressed(path, project=np.asarray(project.model_dump_json()),
        summary=np.asarray(json.dumps(summary)), field_monitors=np.asarray(json.dumps(metadata)), **arrays)
    direct = native_radiation_box(dict(project=project, summary=summary, frequency_fields=records),
        IDS, bounds_um=bounds, refractive_index=1.3)
    loaded = load_native_radiation_box(path, IDS, bounds_um=bounds, refractive_index=1.3)
    directions = [[1.,0.,0.], [0.,1.,0.]]
    np.testing.assert_array_equal(loaded.project(directions).electric_amplitude.numpy(),
        direct.project(directions).electric_amplitude.numpy())
    scattered = load_native_radiation_box(path, IDS, reference_path=path, bounds_um=bounds,
        refractive_index=1.3)
    assert np.count_nonzero(scattered.project(directions).electric_amplitude.numpy()) == 0


def zip64_central_directory(path, *, oversized=False, multiple_disks=False):
    # Produce standard ZIP64 end records around a tiny ordinary archive. No huge
    # field allocation is necessary to exercise 64-bit central-directory parsing.
    raw = path.read_bytes()
    end = raw.rfind(b'PK\x05\x06')
    _, _, _, count_disk, count, central_size, central_offset, _ = struct.unpack('<4s4H2LH', raw[end:end+22])
    record = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0,
        count_disk, count, (1 << 40) if oversized else central_size, central_offset)
    locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, end, 2 if multiple_disks else 1)
    sentinel = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, 65535, 65535, 0xffffffff, 0xffffffff, 0)
    path.write_bytes(raw[:end]+record+locator+sentinel)


def test_standard_single_disk_zip64_directory_loads_selected_planes(tmp_path, adapter):
    path = archive(tmp_path/'zip64.npz')
    zip64_central_directory(path)
    load_native_radiation_box(path, IDS, **KW)
    assert len(adapter) == 1
    assert adapter[0][0]['frequency_fields'][0]['fields'].shape == (2, 4, 6)


@pytest.mark.parametrize('defect', ['oversized', 'multiple_disks'])
def test_zip64_directory_limits_precede_zipfile_and_array_allocation(tmp_path, monkeypatch, defect):
    path = archive(tmp_path/'zip64.npz')
    zip64_central_directory(path, **{defect: True})
    monkeypatch.setattr(zipfile, 'ZipFile', lambda *a, **k: pytest.fail('Opened unadmitted central directory.'))
    with pytest.raises(ValueError):
        load_native_radiation_box(path, IDS, **KW)


def test_long_native_axes_admitted_before_selected_numeric_decode(tmp_path, monkeypatch):
    path = archive(tmp_path/'short.npz')
    with np.load(path, allow_pickle=False) as source:
        project = json.loads(str(source['project']))
    project['region']['size'] = [100000., 2., 2.]
    project['region']['memory_mode'] = 'budgeted'
    path = rewrite(path, tmp_path/'long.npz', 'project.npy', npy(np.asarray(json.dumps(project))))
    original = np.load
    def metadata_only(source, **kwargs):
        assert source.name in ('project.npy', 'summary.npy', 'field_monitors.npy'), 'Numeric array decoded before axis admission.'
        return original(source, **kwargs)
    monkeypatch.setattr(np, 'load', metadata_only)
    with pytest.raises(ValueError, match='host_budget_bytes'):
        load_native_radiation_box(path, IDS, host_budget_bytes=3*1024**2, **KW)
