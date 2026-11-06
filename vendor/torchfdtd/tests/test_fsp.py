import json
import os
import zipfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from torchfdtd import fsp


def test_array_and_complex_encoding():
    data = {'a': np.array([[1+2j, np.nan+1j], [0, np.inf]]), 'b': np.zeros((0, 3)), 'text': '파장'}
    payload = json.loads(json.dumps(fsp.encode_value(data), allow_nan=False))
    restored = fsp.decode_value(payload)
    np.testing.assert_equal(restored['a'], data['a'])
    # Empty arrays still carry their full shape.
    assert restored['b'].shape == (0, 3)
    assert restored['text'] == '파장'


def test_original_preservation_and_hash_validation(tmp_path):
    source = tmp_path / 'original.fsp'
    source.write_bytes(fsp.HEADER + b'1.1\0' + bytes(range(256)))
    manifest = {'source': {'sha256': fsp.fingerprint(source)}}
    fsp.archive_fsp(source, tmp_path / 'project.pwfsp', manifest)
    with zipfile.ZipFile(tmp_path / 'project.pwfsp') as zf:
        assert zf.read('original.fsp') == source.read_bytes()
    result = fsp.export_fsp(source, tmp_path / 'copy.fsp')
    assert result['verified'] and not result['edited']
    assert (tmp_path / 'copy.fsp').read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='immutable'):
        fsp.export_fsp(source, source)
    with pytest.raises(FileExistsError):
        fsp.export_fsp(source, tmp_path / 'copy.fsp')
    with pytest.raises(FileExistsError):
        fsp.write_inspection(manifest, source)
    source.write_bytes(source.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        fsp.archive_fsp(source, tmp_path / 'bad.pwfsp', manifest)


def test_unsupported_physics_cannot_be_silently_accepted():
    issues = fsp.native_diagnostics([
        {'id': 'fdtd', 'properties': {'type': 'FDTD', 'x min bc': 'Periodic', 'mesh type': 'auto non-uniform'}},
        {'id': 'monitor', 'properties': {'type': 'DFTMonitor', 'apodization': 'Full'}},
        {'id': 'metal', 'properties': {'type': 'Rectangle', 'material': 'Au (Gold) - Palik'}},
    ])
    assert {'fsp_native_mapping_unvalidated', 'boundary_mapping_required', 'unsupported_mesh',
            'unsupported_object', 'apodization_mapping_required', 'material_mapping_required'} <= {i['code'] for i in issues}


@pytest.mark.optional
@pytest.mark.skipif(os.environ.get('TORCHFDTD_LUMERICAL_TESTS') != '1', reason='Requires installed licensed Lumerical')
def test_installed_lumerical_fsp_roundtrip(tmp_path):
    api = fsp.load_api()
    source = tmp_path / 'source.fsp'
    with api.FDTD(hide=True) as f:
        f.addfdtd()
        f.set('dimension', '3D')
        f.set('x min bc', 'Periodic')
        f.set('x max bc', 'Periodic')
        f.set('pml layers', 12)
        f.addsphere()
        f.set('name', 'sphere')
        f.set('radius', 0.4e-6)
        f.addplane()
        f.set('name', 'source')
        f.set('override global source settings', True)
        f.set('wavelength start', 1.3e-6)
        f.set('wavelength stop', 1.8e-6)
        f.addpower()
        f.set('name', 'dft')
        f.set('apodization', 'Full')
        f.set('apodization center', 100e-15)
        f.set('apodization time width', 50e-15)
        f.addpoly()
        f.set('name', 'polygon')
        f.set('vertices', np.array([[0, 0], [1e-6, 0], [0, 1e-6]]))
        f.set('material', 'Si (Silicon) - Palik')
        f.addgroup()
        f.set('name', 'nested')
        f.groupscope('::model::nested')
        for radius in (0.2e-6, 0.3e-6):
            f.addsphere()
            f.set('name', 'duplicate')
            f.set('radius', radius)
        f.groupscope('::model')
        f.addsphere()
        f.set('name', 'literal#2')
        f.set('radius', 0.12e-6)
        f.save(str(source))
    sha = fsp.fingerprint(source)
    first = fsp.inspect_fsp(source, api)
    assert not first['native_execution']['allowed']
    assert not first['read_diagnostics'], first['read_diagnostics']
    assert all(not obj['read_errors'] for obj in first['objects'])
    assert not first['referenced_materials']['Si (Silicon) - Palik']['read_errors']
    objects = {o['id']: o['properties'] for o in first['objects']}
    duplicates = sorted(k for k in objects if 'duplicate' in k)
    assert len(duplicates) == 2, list(objects)
    assert objects['::model::literal#2']['radius'] == pytest.approx(0.12e-6)
    patches = [
        {'object_id': '::model::sphere', 'property': 'radius', 'value': 0.55e-6},
        {'object_id': '::model::source', 'property': 'wavelength start', 'value': 1.4e-6},
        {'object_id': '::model::dft', 'property': 'apodization center', 'value': 120e-15},
        {'object_id': '::model::FDTD', 'property': 'pml layers', 'value': 16},
        {'object_id': duplicates[1], 'property': 'radius', 'value': 0.35e-6},
    ]
    out = tmp_path / 'edited.fsp'
    verified = fsp.export_fsp(source, out, patches, api)
    assert verified['verified'] and len(verified['patches']) == 5
    second = fsp.inspect_fsp(out, api)
    after = {o['id']: o['properties'] for o in second['objects']}
    assert objects.keys() == after.keys()
    assert after['::model::FDTD']['x min bc'] == 'Periodic'
    assert after['::model::dft']['apodization'] == 'Full'
    assert after['::model::source']['wavelength stop'] == pytest.approx(1.8e-6)
    assert after[duplicates[0]]['radius'] == objects[duplicates[0]]['radius']
    assert after['::model::polygon']['vertices'] == objects['::model::polygon']['vertices']
    assert first['referenced_materials'] == second['referenced_materials']
    assert fsp.fingerprint(source) == sha
    artifacts = os.environ.get('TORCHFDTD_FSP_TEST_ARTIFACTS')
    if artifacts:
        folder = Path(artifacts)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / 'source.fsp')
        shutil.copyfile(out, folder / 'edited.fsp')
        summary = {'vendor_version': first['bridge']['vendor_version'],
                   'object_count': len(first['objects']),
                   'decoded_property_count': sum(len(o['properties']) for o in first['objects']),
                   'read_diagnostics': first['read_diagnostics'],
                   'original_sha256': sha, 'original_unchanged': fsp.fingerprint(source) == sha,
                   'verification': verified,
                   'preserved_checks': ['Periodic x boundaries', 'Full DFT apodization',
                                        'source wavelength stop', 'first duplicate radius',
                                        'polygon vertices', 'referenced material properties'],
                   'native_fsp_execution': False, 'independent_parser': False}
        (folder / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with pytest.raises(Exception):
        fsp.export_fsp(source, tmp_path / 'failed.fsp', [{'object_id': 'missing', 'property': 'radius', 'value': 1}], api)
    assert not (tmp_path / 'failed.fsp').exists()
    assert fsp.fingerprint(source) == sha
