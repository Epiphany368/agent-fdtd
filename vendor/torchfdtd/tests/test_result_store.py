"""G8-02: the chunked HDF5 result format round-trips a run, reads by slice and
keeps a plane read of a multi-gigabyte-shaped volume out of process memory."""
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from torchfdtd import FieldMonitor, Monitor, Project, Region, Result, Simulation, Source, Structure
from torchfdtd.result_store import FORMAT, LAYOUT, ResultFile, storage_format

h5py = pytest.importorskip('h5py')

# Nominal shape of the synthetic volume and the declared memory budget of one
# plane read, as docs/validation/cases/G8-02_chunked_result_storage.json fixes them.
LARGE_SHAPE = (512, 512, 512, 3)
LARGE_BYTES = int(np.prod(LARGE_SHAPE)) * 4
DECLARED_FRACTION = 0.05
DECLARED_STORED_FRACTION = 0.01


def plane_project():
    """2D, 24 x 20 cells, 60 steps, one point monitor and one x-normal frequency plane with flux."""
    return Project(name='plane result', region=Region(dimension='2d', size=(2.4, 2, .1), mesh=.1, pml_cells=3, steps=60,
                                                      snapshot_interval=20, backend='cpu'),
                   structures=[Structure(id='slab', name='slab', center=(.5, 0, 0), size=(.4, 2, .1))],
                   sources=[Source(id='src', center=(-.6, 0, 0), wavelength=1, pulse_cycles=1)],
                   monitors=[Monitor(id='probe', center=(.2, 0, 0)),
                             FieldMonitor(id='plane', name='plane', normal='x', center=(.8, 0, 0), size=(0, 1, 1),
                                          record_fields=('Ez', 'Hy'), record_poynting=('x',))])


@pytest.fixture(scope='module')
def result():
    return Simulation(plane_project()).run()


@pytest.fixture(scope='module')
def stored(result, tmp_path_factory):
    path = tmp_path_factory.mktemp('store') / 'plane.h5'
    result.save(path)
    return path


def as_json(value):
    return json.loads(json.dumps(value))


def assert_results_equal(a, b):
    assert a.project.model_dump() == b.project.model_dump() and as_json(a.summary) == as_json(b.summary)
    for name in ('frames', 'frame_steps', 'epsilon', 'signals', 'times', 'electric', 'magnetic'):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        np.testing.assert_array_equal(x, y, err_msg=name)
    assert len(a.frequency_fields) == len(b.frequency_fields)
    for pa, pb in zip(a.frequency_fields, b.frequency_fields):
        assert set(pa) == set(pb)
        for key in pa:
            if isinstance(pa[key], np.ndarray):
                assert pa[key].dtype == pb[key].dtype, key
                np.testing.assert_array_equal(pa[key], pb[key], err_msg=key)
            else:
                assert as_json(pa[key]) == as_json(pb[key]), key
    assert (a.endpoint_fields is None) == (b.endpoint_fields is None)


def test_format_is_selected_by_argument_then_suffix():
    assert storage_format('r.npz') == 'npz' and storage_format('r.h5') == 'hdf5' and storage_format('r.HDF5') == 'hdf5'
    assert storage_format('r.h5', 'npz') == 'npz' and storage_format('r.bin', 'hdf5') == 'hdf5'
    with pytest.raises(ValueError, match='zarr'):
        storage_format('r.h5', 'zarr')


def test_hdf5_round_trips_complex_planes_coordinates_and_units(result, stored):
    assert result.frequency_fields[0]['fields'].dtype == np.complex128
    with h5py.File(stored, 'r') as file:
        assert file.attrs['format'] == FORMAT and int(file.attrs['layout']) == LAYOUT
        assert file.attrs['units'] == result.summary['units']
        assert file['mesh/x_um'].attrs['units'] == 'um' and file['times'].attrs['units'] == 's'
        assert file['field_monitors/0/fields'].dtype.kind == 'c'
        for axis, nodes in zip('xyz', result.project.region.mesh_nodes):
            np.testing.assert_array_equal(file[f'mesh/{axis}_um'][()], nodes)
    with Result.open(stored) as opened:
        assert opened.units == result.summary['units']
        for axis_nodes, nodes in zip(opened.mesh_nodes, result.project.region.mesh_nodes):
            np.testing.assert_array_equal(axis_nodes, nodes)
        loaded = opened.load()
    assert_results_equal(loaded, result)
    assert_results_equal(Result.load(stored), result)
    plane = loaded.field_monitor('plane')
    assert plane['field_units'] == result.frequency_fields[0]['field_units']
    assert plane['flux_units'] == result.frequency_fields[0]['flux_units']
    for k, spectrum in enumerate(result.spectra):
        np.testing.assert_allclose(loaded.spectra[k]['value'], spectrum['value'], rtol=1e-12, atol=0)


def test_explicit_hdf5_format_and_npz_path_keep_working(result, tmp_path):
    result.save(tmp_path / 'explicit.dat', format='hdf5')
    with Result.open(tmp_path / 'explicit.dat') as opened:
        assert_results_equal(opened.load(), result)
    result.save(tmp_path / 'plain.npz')
    with np.load(tmp_path / 'plain.npz', allow_pickle=False) as archive:
        assert {'project', 'summary', 'frames', 'E', 'H', 'field_monitor_0_fields'} <= set(archive.files)
    assert_results_equal(Result.load(tmp_path / 'plain.npz'), result)
    # numpy appends .npz to a savez path without that suffix; the explicit format keeps that rule.
    result.save(tmp_path / 'forced.dat', format='npz')
    with np.load(tmp_path / 'forced.dat.npz', allow_pickle=False) as archive:
        assert 'project' in archive.files


def test_lazy_reads_match_the_full_arrays_and_use_the_documented_chunks(result, stored):
    with Result.open(stored) as opened:
        assert opened.shape == result.electric.shape[:3]
        assert opened.project.model_dump() == result.project.model_dump() and opened.summary == as_json(result.summary)
        for i in range(len(result.frame_steps)):
            np.testing.assert_array_equal(opened.frame(i), result.frames[i])
        np.testing.assert_array_equal(opened.frame_steps, result.frame_steps)
        np.testing.assert_array_equal(opened.times, result.times)
        np.testing.assert_array_equal(opened.field_slice('E', 'Ez', 'z', 0), result.electric[:, :, 0, 2])
        np.testing.assert_array_equal(opened.field_slice('H', 1, 0, 5), result.magnetic[5, :, :, 1])
        np.testing.assert_array_equal(opened.field_slice('E', 'Ex', 'y', 3), result.electric[:, 3, :, 0])
        np.testing.assert_array_equal(opened.signal(0), result.signals[:, 0])
        spectrum = opened.monitor_spectrum(0)
        np.testing.assert_allclose(spectrum['value'], result.spectra[0]['value'], rtol=1e-12, atol=0)
        assert spectrum['units'] == result.spectra[0]['units'] and spectrum['id'] == 'probe'
        plane, full = opened.field_monitor('plane'), result.frequency_fields[0]
        assert plane.components == ['Ez', 'Hy'] and plane.shape == tuple(full['shape'])
        for f in range(len(full['frequency_hz'])):
            np.testing.assert_array_equal(plane.fields(f, 'Hy'), full['fields'][f, :, 1])
            np.testing.assert_array_equal(plane.plane(f, 'Ez'), full['fields'][f, :, 0].reshape(full['shape']).squeeze(axis=0))
        np.testing.assert_array_equal(plane.flux, full['flux'])
        np.testing.assert_array_equal(plane.points_um, full['points_um'])
        np.testing.assert_array_equal(plane.weights, full['weights'])
        with pytest.raises(ValueError, match='not recorded'):
            plane.fields(0, 'Ex')
        with pytest.raises(ValueError, match='one enabled frequency plane'):
            opened.field_monitor('missing')
    with h5py.File(stored, 'r') as file:
        nx, ny, nz, _ = result.electric.shape
        assert file['E'].chunks == (1, ny, nz, 1) and file['H'].chunks == (1, ny, nz, 1)
        assert file['frames'].chunks == (1, *result.frames.shape[1:])
        nf, points, _ = full['fields'].shape
        assert file['field_monitors/0/fields'].chunks == (1, points, 1)
        assert file['signals'].chunks == (len(result.times), 1)


def test_foreign_and_newer_files_are_refused(stored, tmp_path):
    other = tmp_path / 'other.h5'
    with h5py.File(other, 'w') as file:
        file.create_dataset('x', data=np.zeros(3))
    with pytest.raises(ValueError, match='not a torchfdtd result file'):
        ResultFile(other)
    newer = tmp_path / 'newer.h5'
    with h5py.File(stored, 'r') as source, h5py.File(newer, 'w') as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs['layout'] = LAYOUT + 1
    with pytest.raises(ValueError, match=f'layout {LAYOUT + 1}, newer than layout {LAYOUT}'):
        Result.open(newer)


def test_a_corrupted_metadata_attribute_does_not_leak_the_handle(stored, tmp_path, monkeypatch):
    """A project or summary attribute that fails to parse after the format check closes the file before raising."""
    import shutil
    opened = []

    class Recording(h5py.File):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)
    monkeypatch.setattr(h5py, 'File', Recording)
    for attribute, value in [('project', '{"region": {"mesh": -1}}'), ('summary', 'not json'), ('field_monitors', '[unterminated')]:
        corrupt = tmp_path / f'corrupt_{attribute}.h5'
        shutil.copy(stored, corrupt)
        with Recording(corrupt, 'a') as file:
            file.attrs[attribute] = value
        with pytest.raises(ValueError):
            ResultFile(corrupt)
        assert len(opened) == 2 and not any(opened), attribute   # a closed h5py.File is falsy
        opened.clear()
        corrupt.unlink()   # an open read handle keeps the file locked on Windows


def test_plane_slice_of_a_multi_gigabyte_shaped_volume_stays_within_the_declared_memory(result, tmp_path):
    """The E volume is replaced by a (512, 512, 512, 3) float32 dataset, 1.6 GB nominal,
    of which one x plane of Ez is written; the file stores only that chunk. Opening the
    file and reading that plane, and a y plane crossing every x chunk, must grow the
    process working set by less than DECLARED_FRACTION of the nominal volume."""
    psutil = pytest.importorskip('psutil')
    path = tmp_path / 'large.h5'
    result.save(path)
    x_index, y_index, component = 100, 200, 2
    rng = np.random.default_rng(7)
    plane = rng.standard_normal(LARGE_SHAPE[1:3]).astype(np.float32)
    with h5py.File(path, 'a') as file:
        del file['E']
        dataset = file.create_dataset('E', shape=LARGE_SHAPE, dtype=np.float32, chunks=(1, LARGE_SHAPE[1], LARGE_SHAPE[2], 1))
        dataset[x_index, :, :, component] = plane
    stored_bytes = path.stat().st_size
    assert stored_bytes < DECLARED_STORED_FRACTION * LARGE_BYTES, (stored_bytes, LARGE_BYTES)
    process = psutil.Process()
    gc.collect()
    before = process.memory_info()
    with Result.open(path) as opened:
        assert opened.shape == LARGE_SHAPE[:3]
        x_plane = opened.field_slice('E', 'Ez', 'x', x_index)
        y_plane = opened.field_slice('E', 'Ez', 'y', y_index)
        after = process.memory_info()
    np.testing.assert_array_equal(x_plane, plane)
    expected_y = np.zeros((LARGE_SHAPE[0], LARGE_SHAPE[2]), dtype=np.float32)
    expected_y[x_index] = plane[y_index]
    np.testing.assert_array_equal(y_plane, expected_y)
    rss_growth = after.rss - before.rss
    peak_growth = (after.peak_wset - before.peak_wset) if hasattr(after, 'peak_wset') else None
    observed = dict(nominal_volume_bytes=LARGE_BYTES, stored_file_bytes=stored_bytes, plane_bytes=int(plane.nbytes),
                    rss_growth_bytes=int(rss_growth), peak_working_set_growth_bytes=None if peak_growth is None else int(peak_growth),
                    rss_growth_fraction=rss_growth / LARGE_BYTES, declared_fraction=DECLARED_FRACTION,
                    declared_stored_fraction=DECLARED_STORED_FRACTION)
    record = os.environ.get('TORCHFDTD_G8_OBSERVED')
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        Path(record).write_text(json.dumps({'G8-02': observed}, indent=2), encoding='utf-8')
    assert rss_growth < DECLARED_FRACTION * LARGE_BYTES, observed
    if peak_growth is not None:
        assert peak_growth < DECLARED_FRACTION * LARGE_BYTES, observed


def test_endpoint_fields_of_a_pmc_run_round_trip(tmp_path):
    """The exact-endpoint PMC cavity (3D, 16 x 16 x 16 cells, 160 steps) stores its endpoint E/H planes under endpoint/."""
    from torchfdtd.models import demo_project
    pmc = Simulation(demo_project('pmc')).run()
    assert pmc.endpoint_fields is not None and set(pmc.endpoint_fields) == {'E_upper', 'H_upper'}
    pmc.save(tmp_path / 'pmc.h5')
    with h5py.File(tmp_path / 'pmc.h5', 'r') as file:
        assert set(file['endpoint']) == {'E_upper', 'H_upper'}
    loaded = Result.load(tmp_path / 'pmc.h5')
    assert_results_equal(loaded, pmc)
    for key, value in pmc.endpoint_fields.items():
        np.testing.assert_array_equal(loaded.endpoint_fields[key], value)
