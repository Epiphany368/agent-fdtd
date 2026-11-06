"""Loss-aware FSP inspection and editing through an installed Lumerical API.

The original FSP is always retained. This is a licensed interoperability bridge,
not an independent FSP parser or a claim that the native solver supports its
contents. No vendor modules or material database are shipped in this package.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
import threading
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .models import Model

BRIDGE_LOCK = threading.Lock()
MAX_FSP_BYTES = 128 * 1024 * 1024
HEADER = b'LUMERICAL file version '


class BridgeUnavailable(RuntimeError):
    pass


class PropertyPatch(Model):
    object_id: str = Field(min_length=1, max_length=2048)
    property: str = Field(min_length=1, max_length=512)
    value: Any


def encode_value(value):
    """JSON encoding preserving array shapes, complex values and nonfinite data."""
    if isinstance(value, np.ndarray):
        return {'__type__': 'ndarray', 'shape': list(value.shape),
                'dtype': str(value.dtype), 'data': encode_value(value.tolist())}
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, complex):
        return {'__type__': 'complex', 'real': encode_value(value.real), 'imag': encode_value(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return {'__type__': 'float', 'value': repr(value)}
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    raise TypeError(f'Unsupported API value type: {type(value).__name__}')


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('__type__')
    if kind == 'complex':
        return complex(decode_value(value['real']), decode_value(value['imag']))
    if kind == 'float':
        if value['value'] not in ('inf', '-inf', 'nan'):
            raise ValueError('Invalid special float.')
        return float(value['value'])
    if kind == 'ndarray':
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are accepted.')
        array = np.asarray(decode_value(value['data']), dtype=dtype)
        shape = value['shape']
        if array.size == 0 and isinstance(shape, list) and shape and all(type(n) is int and 0 <= n <= MAX_FSP_BYTES for n in shape) and math.prod(shape) == 0:
            return array.reshape(shape)
        if list(array.shape) != value['shape']:
            raise ValueError('Array shape differs from its encoded data.')
        return array
    return {k: decode_value(v) for k, v in value.items()}


def api_path(explicit=None):
    supplied = explicit or os.environ.get('TORCHFDTD_LUMAPI') or os.environ.get('PHOTONWEAVE_LUMAPI')
    if supplied:
        path = Path(supplied).expanduser()
        if not path.is_file():
            raise BridgeUnavailable('Configured lumapi.py does not exist.')
        return path.resolve()
    candidates = []
    for base in (Path('C:/Program Files/Lumerical'), Path('C:/Program Files/ANSYS Inc'), Path('/opt/lumerical')):
        if base.exists():
            candidates.extend(base.glob('v*/api/python/lumapi.py'))
            candidates.extend(base.glob('v*/Lumerical/api/python/lumapi.py'))
    if candidates:
        return sorted(candidates, reverse=True)[0].resolve()
    raise BridgeUnavailable('FSP bridge requires an installed Lumerical FDTD and CAD license. Set TORCHFDTD_LUMAPI to lumapi.py.')


def availability():
    try:
        path = api_path()
        return {'installed': True, 'api_path': str(path), 'license_checked': False,
                'independent_parser': False, 'native_execution': False}
    except BridgeUnavailable as exc:
        return {'installed': False, 'reason': str(exc), 'license_checked': False,
                'independent_parser': False, 'native_execution': False}


def load_api(explicit=None):
    path = api_path(explicit)
    spec = importlib.util.spec_from_file_location('torchfdtd_vendor_lumapi', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_file(path):
    path = Path(path).resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FSP_BYTES:
        raise ValueError('FSP file must exist and be no larger than 128 MiB.')
    with path.open('rb') as stream:
        if not stream.read(len(HEADER)).startswith(HEADER):
            raise ValueError('File does not have a Lumerical project header.')
    return path


def fingerprint(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _properties(names, getter):
    values, errors = {}, {}
    for name in str(names).splitlines():
        if not name:
            continue
        try:
            values[name] = encode_value(getter(name))
        except Exception as exc:
            errors[name] = str(exc)
    return {'properties': values, 'read_errors': errors}


def inspect_session(f, source):
    objects, diagnostics, seen = [], [], set()

    def visit(oid, parent=None, vendor_name=None, index=1):
        if oid in seen:
            raise RuntimeError(f'Duplicate traversal ID: {oid}')
        if len(seen) >= 10000:
            raise ValueError('FSP inspector limits a project to 10,000 objects.')
        seen.add(oid)
        name = vendor_name or oid
        record = {'id': oid, 'parent': parent,
                  'vendor_reference': {'name': name, 'index': index},
                  **_properties(f.getnamed(name), lambda prop: f.getnamed(name, prop, index))}
        objects.append(record)
        props = record['properties']
        kind = props.get('type', '')
        if index > 1 and kind != f.getnamed(name, 'type', 1):
            diagnostics.append({'object_id': oid, 'kind': 'ambiguous_property_list',
                                'message': 'Installed API cannot enumerate separate property lists for duplicate names with different object types. Original FSP remains preserved.'})
        if kind == 'ObjectModelRoot' or 'Group' in kind or kind == 'FDTD':
            try:
                f.groupscope(oid)
                f.selectall()
                ids = f.getid().splitlines() if f.getnumber() else []
            except Exception as exc:
                # FDTD can expose ports in newer versions; never assume a
                # failed traversal proves the absence of children.
                diagnostics.append({'object_id': oid, 'kind': 'children_not_read', 'message': str(exc)})
                ids = []
            counts, indices = Counter(ids), Counter()
            for child in ids:
                indices[child] += 1
                visit(child + (f'#{indices[child]}' if counts[child] > 1 else ''), oid, child, indices[child])
    visit('::model')
    f.groupscope('::model')
    globals_ = {}
    for key, getter in [('source', f.getglobalsource), ('monitor', f.getglobalmonitor)]:
        try:
            globals_[key] = _properties(getter(), getter)
        except Exception as exc:
            globals_[key] = {'properties': {}, 'read_errors': {'enumeration': str(exc)}}
    material_names = sorted({str(o['properties'][k]) for o in objects
                             for k in ('material', 'background material')
                             if k in o['properties'] and not str(o['properties'][k]).startswith('<')})
    materials = {}
    for name in material_names:
        try:
            materials[name] = _properties(f.getmaterial(name), lambda key: f.getmaterial(name, key))
        except Exception as exc:
            materials[name] = {'properties': {}, 'read_errors': {'enumeration': str(exc)}}
    issues = native_diagnostics(objects)
    return {'schema': 'photonweave.fsp-inspection', 'schema_version': 1,
            'source': {'filename': Path(source).name, 'bytes': Path(source).stat().st_size,
                       'sha256': fingerprint(source)},
            'bridge': {'vendor_version': f.version(), 'requires_lumerical': True},
            'layout_mode': bool(f.layoutmode()), 'objects': objects, 'globals': globals_,
            'referenced_materials': materials, 'read_diagnostics': diagnostics,
            'preservation': {'original_fsp': 'byte-exact when no edits are applied',
                             'results': 'retained in original; cleared in edited exports',
                             'sweeps_and_unread_data': 'retained in original FSP; not decoded'},
            'native_execution': {'allowed': False, 'issues': issues}}


def native_diagnostics(objects):
    issues = [{'code': 'fsp_native_mapping_unvalidated', 'object_id': None,
               'message': 'FSP settings are inspectable and editable through Lumerical. Native GPU execution is blocked until their mapping and physics are validated.'}]
    basic = {'ObjectModelRoot', 'FDTD', 'Rectangle', 'Circle', 'Ring', 'Sphere', 'DipoleSource', 'TimeMonitor'}
    for obj in objects:
        p, oid = obj['properties'], obj['id']
        def issue(code, message):
            issues.append({'code': code, 'object_id': oid, 'message': message})
        if obj.get('read_errors'):
            issue('unread_properties', 'Some properties could not be decoded. They remain in the original FSP.')
        kind = p.get('type')
        if kind not in basic:
            issue('unsupported_object', f'Native solver does not implement {kind}.')
        if kind == 'FDTD':
            for key in ('x min bc', 'x max bc', 'y min bc', 'y max bc', 'z min bc', 'z max bc'):
                if p.get(key) not in (None, 'PML', 'Periodic', 'Bloch'):
                    issue('unsupported_boundary', f'{key}: {p[key]} is not implemented by the native solver.')
                elif p.get(key) in ('Periodic', 'Bloch'):
                    issue('boundary_mapping_required', f'{key}: native {p[key]} is implemented, but imported grid period and Bloch wavevector mapping still require validation.')
            if p.get('mesh type') != 'uniform':
                issue('unsupported_mesh', 'Native solver currently requires a uniform mesh.')
            issue('different_pml', 'Lumerical PML profiles and native PML coefficients are not equivalent.')
        if 'apodization' in p and p['apodization'] != 'None':
            issue('apodization_mapping_required', f"Native point-trace apodization is implemented, but this FSP monitor's spatial sampling and normalization have not been mapped or validated.")
        if p.get('material', '<Object defined dielectric>') != '<Object defined dielectric>':
            issue('material_mapping_required', 'Database material dispersion must be mapped and validated before native execution.')
        if p.get('script') or p.get('setup script') or p.get('analysis script'):
            issue('script_not_translated', 'Lumerical scripts are preserved as text; the native solver does not execute them.')
    return issues


def inspect_fsp(source, api=None):
    source = validate_file(source)
    with BRIDGE_LOCK:
        vendor = api or load_api()
        with vendor.FDTD(hide=True) as f:
            f.load(str(source))
            return inspect_session(f, source)


def write_inspection(manifest, target):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(target).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def archive_fsp(source, target, manifest):
    """Bundle the original and decoded metadata, without renaming JSON to FSP."""
    source = validate_file(source)
    target = Path(target).resolve()
    if source == target:
        raise ValueError('Archive cannot overwrite the source FSP.')
    if fingerprint(source) != manifest['source']['sha256']:
        raise ValueError('Original FSP changed after inspection.')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source, 'original.fsp')
        zf.writestr('inspection.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


def _equivalent(left, right):
    try:
        a, b = np.asarray(left), np.asarray(right)
        if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
            return a.size == b.size and bool(np.allclose(a.ravel(), b.ravel(), rtol=1e-10, atol=0, equal_nan=True))
    except (ValueError, TypeError):
        pass
    return left == right


def _publish(source, target):
    """Copy fully before publishing, and never replace an existing destination."""
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.fsp-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        # A hard link makes publication atomic without os.replace's overwrite
        # race. If this filesystem cannot link, fail with the destination intact.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def export_fsp(source, target, patches=(), api=None):
    """Apply edits to a private copy, save, reopen and verify requested values.

    Original file and destination are untouched on failure. An unedited export
    is byte-identical and needs no vendor installation or license.
    """
    source = validate_file(source)
    target = Path(target).resolve()
    if target == source:
        raise ValueError('Export to a new path; the original FSP is immutable.')
    if target.exists():
        raise FileExistsError('Export destination already exists.')
    patches = [p if isinstance(p, PropertyPatch) else PropertyPatch.model_validate(p) for p in patches]
    pairs = [(p.object_id, p.property) for p in patches]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Each object property may be patched only once per export.')
    if any(p.property in ('name', 'type') for p in patches):
        raise ValueError('Renaming or changing object types is not supported by this bridge yet.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not patches:
        _publish(source, target)
        return {'edited': False, 'sha256': fingerprint(target), 'verified': True, 'patches': []}
    with BRIDGE_LOCK, tempfile.TemporaryDirectory(prefix='torchfdtd-fsp-') as scratch:
        private = Path(scratch) / 'working.fsp'
        output = Path(scratch) / 'exported.fsp'
        shutil.copyfile(source, private)
        vendor = api or load_api()
        with vendor.FDTD(hide=True) as f:
            f.load(str(private))
            f.switchtolayout()
            # Collect every stable ID first to reject misspelled or ambiguous IDs.
            before = inspect_session(f, private)
            known = {o['id']: o for o in before['objects']}
            for patch in patches:
                if patch.object_id not in known or patch.property not in known[patch.object_id]['properties']:
                    raise ValueError(f'Unknown object/property: {patch.object_id} / {patch.property}')
                ref = known[patch.object_id]['vendor_reference']
                f.setnamed(ref['name'], patch.property, decode_value(patch.value), ref['index'])
            f.save(str(output))
            f.load(str(output))
            verified = []
            for patch in patches:
                ref = known[patch.object_id]['vendor_reference']
                actual = f.getnamed(ref['name'], patch.property, ref['index'])
                if not _equivalent(actual, decode_value(patch.value)):
                    raise ValueError(f'Saved value differs from requested value: {patch.object_id} / {patch.property}')
                verified.append({**patch.model_dump(), 'readback': encode_value(actual)})
        validate_file(output)
        _publish(output, target)
        return {'edited': True, 'sha256': fingerprint(target), 'verified': True,
                'results_cleared': True, 'patches': verified}
