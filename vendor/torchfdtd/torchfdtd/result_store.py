"""Chunked HDF5 result files with lazy, per-slice reads.

``Result.save(path, format='hdf5')`` writes the same members as the NPZ layout
into one HDF5 file; ``Result.open(path)`` returns a :class:`ResultFile` that
reads a frame, a field plane slice, one point spectrum or one frequency and
component of a plane monitor without loading the full volume. HDF5 was chosen
over Zarr because h5py is a single binary wheel with no further dependency,
stores complex arrays natively as a compound type that round-trips through
numpy, and keeps a result in one file; Zarr would add zarr and numcodecs and
spread a result over a directory tree.

Layout 1 (``format`` and ``layout`` root attributes):

- root attributes ``project`` (the project JSON with its schema version),
  ``summary``, ``field_monitors`` and ``monitor_spectra`` (JSON strings) and
  ``units`` (the summary's unit statement);
- ``mesh/x_um``, ``mesh/y_um``, ``mesh/z_um``: Yee node coordinates;
- ``frames`` (nframes, a, b) chunked one frame per chunk, ``frame_steps``,
  ``epsilon``, ``signals`` (steps, monitors) chunked one monitor per chunk,
  ``times``;
- ``E`` and ``H`` (nx, ny, nz, 3) chunked one component and one x plane per
  chunk, so a plane slice touches only the chunks it needs;
- ``monitors/<k>/frequency_hz`` and ``monitors/<k>/spectrum`` per point monitor;
- ``field_monitors/<k>/<array>`` per frequency plane: ``fields`` and
  ``poynting`` (frequencies, points, components) chunked one frequency and one
  component per chunk, ``points_um``, ``weights``, ``frequency_hz``, ``flux``;
- ``endpoint/<name>`` when PMC endpoint fields exist.

Chunks that were never written are not stored, so a file can carry a large
nominal shape with few stored bytes; reading returns the fill value there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = 'torchfdtd-result'
LAYOUT = 1
SUFFIXES = ('.h5', '.hdf5')
FIELD_COMPONENTS = ('x', 'y', 'z')


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError('Chunked result files need h5py: pip install "torchfdtd[hdf5]".') from exc
    return h5py


def storage_format(path, format=None):
    """The format a path selects: an explicit name, else the suffix, else npz."""
    if format is not None:
        if format not in ('npz', 'hdf5'):
            raise ValueError(f"Unknown result format {format!r}; use 'npz' or 'hdf5'.")
        return format
    return 'hdf5' if Path(path).suffix.lower() in SUFFIXES else 'npz'


def _chunks(array, leading=1, components=False):
    """One chunk per index of the first ``leading`` axes and, when the last
    axis holds field components, per component; the rest of the array whole."""
    if array.ndim < 2 or array.size == 0:
        return None
    chunks = [1 if i < leading else n for i, n in enumerate(array.shape)]
    if components:
        chunks[-1] = 1
    return tuple(chunks)


def _write(group, name, array, leading=1, components=False, **attrs):
    array = np.asarray(array)
    dataset = group.create_dataset(name, data=array, chunks=_chunks(array, leading, components))
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def save_hdf5(result, path):
    h5py = _h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    project = result.project
    metadata, plane_metadata = [], []
    with h5py.File(path, 'w') as file:
        file.attrs['format'] = FORMAT
        file.attrs['layout'] = LAYOUT
        file.attrs['schema_version'] = project.schema_version
        file.attrs['project'] = project.model_dump_json()
        file.attrs['summary'] = json.dumps(result.summary)
        file.attrs['units'] = str(result.summary.get('units', ''))
        mesh = file.create_group('mesh')
        for axis, nodes in zip('xyz', project.region.mesh_nodes):
            _write(mesh, f'{axis}_um', nodes, units='um')
        _write(file, 'frames', result.frames, units='reduced field')
        _write(file, 'frame_steps', result.frame_steps)
        _write(file, 'epsilon', result.epsilon, units='relative permittivity')
        signals = np.asarray(result.signals)
        file.create_dataset('signals', data=signals, chunks=(signals.shape[0], 1) if signals.ndim == 2 and signals.size else None)
        _write(file, 'times', result.times, units='s')
        _write(file, 'E', result.electric, components=True, units='reduced field', component_names=json.dumps(['Ex', 'Ey', 'Ez']))
        _write(file, 'H', result.magnetic, components=True, units='reduced field', component_names=json.dumps(['Hx', 'Hy', 'Hz']))
        monitors = file.create_group('monitors')
        for k, (m, spec) in enumerate(zip(result.point_monitors, result.spectra)):
            group = monitors.create_group(str(k))
            _write(group, 'frequency_hz', spec['frequency_hz'], units='Hz')
            _write(group, 'spectrum', spec['value'], units=spec['units'])
            metadata.append({'id': m.id, 'name': m.name, 'component': m.component,
                             'settings': m.spectrum.model_dump(), 'units': spec['units'], 'transform': spec['transform']})
        planes = file.create_group('field_monitors')
        for k, plane in enumerate(result.frequency_fields):
            group = planes.create_group(str(k))
            plane_metadata.append({key: v for key, v in plane.items() if not isinstance(v, np.ndarray)})
            for key, v in plane.items():
                if isinstance(v, np.ndarray):
                    _write(group, key, v, leading=1 if v.ndim == 3 else 0, components=v.ndim == 3)
        if result.endpoint_fields is not None:
            endpoint = file.create_group('endpoint')
            for key, value in result.endpoint_fields.items():
                _write(endpoint, key, value)
        file.attrs['monitor_spectra'] = json.dumps(metadata)
        file.attrs['field_monitors'] = json.dumps(plane_metadata)


class PlaneFile:
    """One frequency plane of an open result file, read per frequency and component."""

    def __init__(self, group, metadata):
        self._group = group
        self.metadata = metadata
        self.id, self.name = metadata['id'], metadata['name']
        self.components = list(metadata.get('components', ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']))
        self.shape = tuple(metadata['shape'])
        self.normal_axis = metadata['normal_axis']
        self.field_units = metadata['field_units']
        self.flux_units = metadata['flux_units']

    @property
    def frequency_hz(self):
        return self._group['frequency_hz'][()]

    @property
    def points_um(self):
        return self._group['points_um'][()]

    @property
    def weights(self):
        return self._group['weights'][()]

    @property
    def flux(self):
        return self._group['flux'][()] if 'flux' in self._group else None

    def fields(self, frequency_index, component):
        """The complex sampled field of one frequency and one component, shape (points,)."""
        if component not in self.components:
            raise ValueError(f'{self.name}: component {component} was not recorded; recorded {self.components}.')
        return self._group['fields'][frequency_index, :, self.components.index(component)]

    def plane(self, frequency_index, component):
        """``fields`` reshaped to the monitor's plane, the normal axis squeezed."""
        return self.fields(frequency_index, component).reshape(self.shape).squeeze(axis='xyz'.index(self.normal_axis))

    def load(self):
        """The full plane dictionary as ``Result.frequency_fields`` holds it."""
        plane = dict(self.metadata)
        plane.update({key: self._group[key][()] for key in self._group})
        return plane


class ResultFile:
    """A result HDF5 file open for lazy reads; ``load()`` materializes a ``Result``."""

    def __init__(self, path):
        from .models import Project
        h5py = _h5py()
        self.path = Path(path)
        self._file = h5py.File(self.path, 'r')
        try:
            attrs = self._file.attrs
            if attrs.get('format') != FORMAT:
                raise ValueError(f'{self.path} is not a torchfdtd result file.')
            self.layout = int(attrs.get('layout', 0))
            if self.layout > LAYOUT:
                raise ValueError(f'{self.path} uses result layout {self.layout}, newer than layout {LAYOUT} this torchfdtd reads.')
            self.project = Project.model_validate_json(str(attrs['project']))
            self.summary = json.loads(str(attrs['summary']))
            self.units = str(attrs['units'])
            self.monitor_spectra = json.loads(str(attrs['monitor_spectra']))
            self._plane_metadata = json.loads(str(attrs['field_monitors']))
        except BaseException:
            # A corrupted or foreign metadata attribute must not leave the handle open.
            self._file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()

    @property
    def mesh_nodes(self):
        return tuple(self._file['mesh'][f'{axis}_um'][()] for axis in 'xyz')

    @property
    def frame_steps(self):
        return self._file['frame_steps'][()]

    @property
    def times(self):
        return self._file['times'][()]

    @property
    def shape(self):
        """(nx, ny, nz) of the stored final fields, without reading them."""
        return tuple(self._file['E'].shape[:3])

    def frame(self, index):
        return self._file['frames'][index]

    def field_slice(self, field, component, axis, index):
        """One plane of the final ``E`` or ``H`` volume: ``field`` 'E' or 'H',
        ``component`` 'Ex'..'Hz' or 0..2, ``axis`` 'x', 'y', 'z' or 0..2."""
        dataset = self._file[field]
        c = component if isinstance(component, int) else FIELD_COMPONENTS.index(component[-1].lower())
        a = axis if isinstance(axis, int) else 'xyz'.index(axis)
        selection = [slice(None)] * 3
        selection[a] = index
        return dataset[(*selection, c)]

    def signal(self, monitor_index):
        return self._file['signals'][:, monitor_index]

    def monitor_spectrum(self, monitor_index):
        group = self._file['monitors'][str(monitor_index)]
        meta = self.monitor_spectra[monitor_index]
        return {'frequency_hz': group['frequency_hz'][()], 'value': group['spectrum'][()],
                'units': meta['units'], 'transform': meta['transform'], **{k: meta[k] for k in ('id', 'name', 'component')}}

    @property
    def field_monitors(self):
        return [PlaneFile(self._file['field_monitors'][str(k)], meta) for k, meta in enumerate(self._plane_metadata)]

    def field_monitor(self, name_or_id):
        matches = [m for m in self.field_monitors if name_or_id in (m.id, m.name)]
        if len(matches) != 1:
            raise ValueError('Expected one enabled frequency plane with this name or id.')
        return matches[0]

    def load(self):
        """Read every member and return the in-memory ``Result``."""
        from .solver import Result
        f = self._file
        endpoint = {key: f['endpoint'][key][()] for key in f['endpoint']} if 'endpoint' in f else None
        return Result(self.project, self.summary, f['frames'][()], f['frame_steps'][()], f['epsilon'][()],
                      f['signals'][()], f['times'][()], f['E'][()], f['H'][()],
                      [plane.load() for plane in self.field_monitors], endpoint)
