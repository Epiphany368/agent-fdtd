"""Record-only tests of the metalens Meep comparison: no simulation is run here."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'meep_comparison' / 'metalens'
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'
PARTS = ('2d', '3d')
SOLVERS = ('torchfdtd', 'meep')


def load_module(name):
    spec = importlib.util.spec_from_file_location(f'metalens_{name}', EXAMPLE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLE))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def compare():
    return load_module('compare')


@pytest.fixture(scope='module')
def result(compare):
    return compare.compare_all()


@pytest.fixture(scope='module')
def committed():
    return json.loads((RECORDS / 'metalens_comparison.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def records():
    return {part: {s: json.loads((RECORDS / f'metalens_{part}_{s}.json').read_text(encoding='utf-8')) for s in SOLVERS} for part in PARTS}


def close(a, b, path=''):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            close(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f'{path}[{i}]')
    elif isinstance(a, float) or isinstance(b, float):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_compare_reproduces_committed_json(result, committed):
    computed, _ = result
    close(computed, committed)


def test_readme_carries_the_rendered_tables(result):
    computed, _ = result
    readme = (EXAMPLE / 'README.md').read_text(encoding='utf-8')
    assert computed['tables_markdown'] in readme


def test_criteria_file_is_the_declared_one(committed):
    digest = hashlib.sha256((EXAMPLE / 'criteria.json').read_bytes()).hexdigest()
    assert committed['criteria_sha256'] == digest


@pytest.mark.parametrize('part', PARTS)
def test_records_share_the_grid_and_geometry(records, part):
    t, m = records[part]['torchfdtd'], records[part]['meep']
    assert t['grid']['shape'] == m['grid']['shape']
    assert t['grid']['mesh_um'] == m['grid']['mesh_um']
    assert abs(t['grid']['dt_s'] - m['grid']['dt_s']) <= 1e-9 * t['grid']['dt_s']
    assert t['grid']['steps'] == m['grid']['steps']
    assert t['grid']['pml_cells'] == m['grid']['pml_cells']
    assert t['geometry_sha256'] == m['geometry_sha256']
    geometry = EXAMPLE / ('geometry.json' if part == '2d' else 'geometry_3d.json')
    assert t['geometry_sha256'] == hashlib.sha256(geometry.read_bytes()).hexdigest()
    for component, count in m['silicon_cells']['per_component'].items():
        assert t['silicon_cells']['per_component'][component] == count
    assert t['source']['support'] == m['source']['support']


@pytest.mark.parametrize('part', PARTS)
def test_declared_criteria_pass(result, part):
    computed, _ = result
    for row in computed[part]['criteria_rows']:
        assert row['difference'] <= row['limit'], row
        assert row['pass']
    assert computed[part]['grid_agreement']['all']
    assert computed['all_pass']


@pytest.mark.parametrize('part', PARTS)
def test_records_are_ratios_to_bare_runs(records, part):
    for s in SOLVERS:
        obs = records[part][s]['observables']
        assert all(0 < v < 1 for v in obs['incident']['transmission'])
        assert all(0 < row['efficiency'] < 1 for row in obs['summary'])
