"""The examples archive preserves inputs without shadowing an installed solver."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_examples_archive_has_only_declared_inputs_and_no_solver(tmp_path):
    spec = importlib.util.spec_from_file_location('package_g7', ROOT/'scripts/package_g7_examples.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path/'examples.zip'
    manifest = module.build(target)
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == set(manifest['files']) | {'manifest.json'}
        assert json.loads(archive.read('manifest.json')) == manifest
        for name, digest in manifest['files'].items():
            assert not name.startswith('torchfdtd/')
            assert not name.endswith('.pt') and '/__pycache__/' not in name
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert 'examples/g7/metagrating/wheel_entry_r4.py' in archive.namelist()
        assert 'examples/meep_comparison/metalens/design_2d.json' in archive.namelist()
        assert any('/G7-03/development/' in name for name in archive.namelist())
