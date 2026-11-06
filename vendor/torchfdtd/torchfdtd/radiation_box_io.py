"""Bounded, data-only loading of six native spectral faces from NPZ archives."""
import ast
from contextlib import ExitStack
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np

from .radiation_box import _axis_workspace_bytes


_FACES = tuple(axis + side for axis in 'xyz' for side in ('_min', '_max'))
_ARRAYS = ('fields', 'frequency_hz', 'points_um', 'weights')
_METADATA = ('project', 'summary', 'field_monitors')
_HEADER_LIMIT = 16384


def _directory_bytes(path, budget):
    # ZipFile builds its entry list eagerly. Bound that metadata before opening
    # it, rather than trusting only the later selected-array admission.
    with Path(path).open('rb') as source:
        source.seek(0, 2)
        size = source.tell()
        source.seek(max(0, size-65557))
        tail = source.read(65557)
    position = tail.rfind(b'PK\x05\x06')
    if position < 0 or len(tail)-position < 22:
        raise ValueError('Missing ZIP end-of-directory record.')
    _, disk, central_disk, entries_disk, entries, central_bytes, offset, comment = struct.unpack(
        '<4s4H2LH', tail[position:position+22])
    if position+22+comment != len(tail) or disk or central_disk or entries_disk != entries:
        raise ValueError('Invalid or multi-disk NPZ archive.')
    directory_end = size-len(tail)+position
    if entries == 65535 or central_bytes == 0xffffffff or offset == 0xffffffff:
        locator_position = directory_end-20
        if locator_position < 0:
            raise ValueError('Missing ZIP64 central-directory locator.')
        with Path(path).open('rb') as source:
            source.seek(locator_position)
            locator = source.read(20)
            magic, disk64, end64, disks = struct.unpack('<4sLQL', locator)
            if magic != b'PK\x06\x07' or disk64 != 0 or disks != 1 or end64+56 > locator_position:
                raise ValueError('Invalid or multi-disk ZIP64 locator.')
            source.seek(end64)
            fixed = source.read(56)
        if len(fixed) != 56:
            raise ValueError('Truncated ZIP64 central-directory record.')
        magic, record_size, _, _, disk, central_disk, entries_disk, entries, central_bytes, offset = struct.unpack(
            '<4sQ2H2L4Q', fixed)
        if (magic != b'PK\x06\x06' or not 44 <= record_size <= 4096 or
                end64+12+record_size != locator_position or disk or central_disk or entries_disk != entries):
            raise ValueError('Invalid or unbounded ZIP64 central-directory record.')
        directory_end = end64
    if central_bytes > budget or entries*512 > budget or offset+central_bytes != directory_end:
        raise ValueError('NPZ central-directory metadata exceeds the archive metadata budget.')
    return max(central_bytes, entries*512)


def _positive_bytes(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer byte budget.')


def _admit(required, budget, message):
    from .memory_profile import host_memory
    available = host_memory()['available_bytes']
    limit = budget if available is None else min(budget, int(.8*available))
    if required > limit:
        raise ValueError(message + ' (host budget or available RAM).')


def _ids(mapping, open_surface=False):
    if not isinstance(mapping, dict):
        raise ValueError('Face keys must be a dict of named box faces.')
    if open_surface:
        if not mapping or set(mapping) - set(_FACES) or len(mapping) == 6:
            raise ValueError('Open-surface projection takes one to five named box faces; use the closed box for all six.')
    elif set(mapping) != set(_FACES):
        raise ValueError('Exactly six closed-box face keys are required.')
    if any(not isinstance(value, str) or not value for value in mapping.values()) or len(set(mapping.values())) != len(mapping):
        raise ValueError('Select distinct monitor IDs for the named faces.')
    return dict(mapping)


def _header(archive, name, *, metadata=False):
    """Inspect a bounded literal NPY header without requesting an array allocation."""
    matches = [entry for entry in archive.infolist() if entry.filename == name]
    if len(matches) != 1:
        raise ValueError(f'Archive must contain exactly one {name}.')
    entry = matches[0]
    if entry.flag_bits & 1:
        raise ValueError('Encrypted NPZ entries are unsupported.')
    with archive.open(entry) as source:
        prefix = source.read(8)
        if len(prefix) != 8 or prefix[:6] != b'\x93NUMPY':
            raise ValueError(f'Invalid NPY header for {name}.')
        version = tuple(prefix[6:])
        if version not in ((1, 0), (2, 0)):
            raise ValueError('Only native NPY v1/v2 data entries are supported.')
        length_bytes = source.read(2 if version == (1, 0) else 4)
        if len(length_bytes) != (2 if version == (1, 0) else 4):
            raise ValueError('Truncated NPY header length.')
        length = struct.unpack('<H' if version == (1, 0) else '<I', length_bytes)[0]
        if not 0 < length <= _HEADER_LIMIT:
            raise ValueError('NPY header exceeds the bounded header limit.')
        encoded = source.read(length)
        if len(encoded) != length:
            raise ValueError('Truncated NPY header.')
        try:
            header = ast.literal_eval(encoded.decode('latin1').strip())
        except (ValueError, SyntaxError, RecursionError) as exc:
            raise ValueError('Invalid literal NPY header.') from exc
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Unexpected NPY header fields.')
    shape = header['shape']
    if (not isinstance(shape, tuple) or len(shape) > 3 or
            any(type(n) is not int or n < 0 for n in shape) or
            type(header['fortran_order']) is not bool or not isinstance(header['descr'], str)):
        raise ValueError('Invalid NPY shape, order or dtype descriptor.')
    try:
        dtype = np.dtype(header['descr'])
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid NPY dtype descriptor.') from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError('Object/structured NPY arrays are unsupported.')
    if metadata:
        if shape != () or dtype.kind not in ('U', 'S') or dtype.itemsize <= 0:
            raise ValueError('Native archive metadata must be scalar JSON strings.')
    elif dtype.kind not in ('f', 'c') or dtype.itemsize not in (4, 8, 16):
        raise ValueError('Native face arrays require floating or complex numeric dtype.')
    size = math.prod(shape) * dtype.itemsize
    if size + 8 + len(length_bytes) + length != entry.file_size:
        raise ValueError('NPY decoded shape size does not match the uncompressed ZIP entry size.')
    return dict(name=name, shape=shape, dtype=dtype, bytes=size, archive_bytes=entry.file_size)


def _load(archive, description):
    with archive.open(description['name']) as source:
        value = np.load(source, allow_pickle=False, max_header_size=_HEADER_LIMIT)
    if value.shape != description['shape'] or value.dtype != description['dtype']:
        raise ValueError('NPY payload changed after preflight.')
    return value


def _json(archive, description):
    scalar = _load(archive, description).item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode('utf8')
    try:
        return json.loads(scalar, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON metadata.')))
    except (ValueError, RecursionError) as exc:
        raise ValueError('Invalid native archive JSON metadata.') from exc


def _face_headers(archive, metadata, mapping, frequency_index):
    if not isinstance(metadata, list):
        raise ValueError('field_monitors metadata must be a list.')
    selected = []
    for identifier in mapping.values():
        matches = [(index, record) for index, record in enumerate(metadata)
            if isinstance(record, dict) and record.get('id') == identifier]
        if len(matches) != 1:
            raise ValueError('Each selected monitor ID must occur exactly once in native metadata.')
        index, record = matches[0]
        descriptions = {key: _header(archive, f'field_monitor_{index}_{key}.npy') for key in _ARRAYS}
        f, nu, p, w = (descriptions[key] for key in _ARRAYS)
        if (len(f['shape']) != 3 or f['shape'][2] != 6 or f['dtype'].kind != 'c' or
                f['dtype'].itemsize not in (8, 16) or min(f['shape']) <= 0):
            raise ValueError('Stored face fields must be nonempty (frequency, point, 6) complex64/128 arrays.')
        nf, points, _ = f['shape']
        if (nu['shape'] != (nf,) or p['shape'] != (points, 3) or w['shape'] != (points,) or
                any(descriptions[key]['dtype'].kind != 'f' or descriptions[key]['dtype'].itemsize not in (4, 8)
                    for key in ('frequency_hz', 'points_um', 'weights'))):
            raise ValueError('Stored face frequency/point/weight headers do not match the field shape.')
        if frequency_index >= nf:
            raise ValueError('Frequency index is outside the stored frequency range.')
        selected.append((record, descriptions))
    return selected


def load_native_radiation_box(path, monitor_ids, *, bounds_um, refractive_index,
        frequency_index=0, reference_path=None, reference_monitor_ids=None, open_surface=False,
        host_budget_bytes=512*1024**2, archive_metadata_budget_bytes=4*1024**2):
    """Read only selected monitor arrays after aggregate sample/reference admission.

    Full stored frequency arrays count toward decompression, even when only one
    frequency is requested. Main E/H/epsilon/frame members are never opened.
    ``open_surface`` selects one to five named faces for the approximate mode.
    """
    _positive_bytes(host_budget_bytes, 'host_budget_bytes')
    _positive_bytes(archive_metadata_budget_bytes, 'archive_metadata_budget_bytes')
    if type(frequency_index) is not int or frequency_index < 0:
        raise ValueError('frequency_index must be a nonnegative integer.')
    if type(open_surface) is not bool:
        raise ValueError('open_surface must be a boolean.')
    mappings = [_ids(monitor_ids, open_surface)]
    paths = [path]
    if reference_path is not None:
        paths.append(reference_path)
        mappings.append(_ids(reference_monitor_ids if reference_monitor_ids is not None else monitor_ids, open_surface))
    elif reference_monitor_ids is not None:
        raise ValueError('Reference monitor IDs require a reference archive.')
    with ExitStack() as stack:
        directory_bytes = sum(_directory_bytes(value, archive_metadata_budget_bytes) for value in paths)
        if directory_bytes > archive_metadata_budget_bytes:
            raise ValueError('Aggregate central-directory metadata exceeds archive_metadata_budget_bytes.')
        archives = [stack.enter_context(zipfile.ZipFile(Path(value))) for value in paths]
        metadata_headers = [{name: _header(archive, name+'.npy', metadata=True) for name in _METADATA}
            for archive in archives]
        metadata_bytes = directory_bytes + sum(d['archive_bytes'] for headers in metadata_headers for d in headers.values())
        if metadata_bytes > archive_metadata_budget_bytes:
            raise ValueError('Aggregate archive metadata exceeds archive_metadata_budget_bytes.')
        metadata_reservation = 16*metadata_bytes + 65536
        _admit(metadata_reservation, host_budget_bytes, 'Decoded archive metadata exceeds host_budget_bytes')
        decoded = [{name: _json(archive, description) for name, description in headers.items()}
            for archive, headers in zip(archives, metadata_headers)]
        if any(not isinstance(item['project'], dict) or not isinstance(item['summary'], dict) for item in decoded):
            raise ValueError('Native project and summary metadata must be JSON objects.')
        selected = [_face_headers(archive, item['field_monitors'], mapping, frequency_index)
            for archive, item, mapping in zip(archives, decoded, mappings)]
        payload_bytes = sum(description['archive_bytes'] for faces in selected
            for _, descriptions in faces for description in descriptions.values())
        points = sum(descriptions['fields']['shape'][1] for faces in selected for _, descriptions in faces)
        axis_workspace = sum(_axis_workspace_bytes(item['project']) for item in decoded)
        reservation = metadata_reservation + 6*payload_bytes + 512*points + axis_workspace
        _admit(reservation, host_budget_bytes, 'Aggregate full-frequency sample/reference loading exceeds host_budget_bytes')
        inputs = []
        for archive, item, faces in zip(archives, decoded, selected):
            records = []
            for record, descriptions in faces:
                records.append({**record, **{key: _load(archive, description) for key, description in descriptions.items()}})
            inputs.append(dict(project=item['project'], summary=item['summary'], frequency_fields=records))
        from .radiation_box import native_radiation_box
        result = native_radiation_box(inputs[0], mappings[0], bounds_um=bounds_um,
            refractive_index=refractive_index, frequency_index=frequency_index,
            reference=inputs[1] if len(inputs) == 2 else None,
            reference_monitor_ids=mappings[1] if len(inputs) == 2 else None,
            open_surface=open_surface, host_budget_bytes=host_budget_bytes)
        return result
