"""Record-only tests of the microring Meep comparison: no simulation is run here."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'meep_comparison' / 'microring'
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'
SOLVERS = ('torchfdtd', 'meep')


@pytest.fixture(scope='module')
def compare():
    spec = importlib.util.spec_from_file_location('microring_compare', EXAMPLE / 'compare.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['microring_compare'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def records(compare):
    return compare.load_records(RECORDS)


@pytest.fixture(scope='module')
def criteria():
    return json.loads((EXAMPLE / 'criteria.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def result(compare, records, criteria):
    return compare.compute(records, criteria)


@pytest.fixture(scope='module')
def committed():
    return json.loads((RECORDS / 'microring_comparison.json').read_text(encoding='utf-8'))


def assert_same(a, b, path=''):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{path}[{i}]')
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_compare_reproduces_the_committed_comparison(result, committed):
    assert_same(result, committed)


def test_readme_contains_the_rendered_tables(compare, result):
    readme = (EXAMPLE / 'README.md').read_text(encoding='utf-8')
    assert compare.render_table(result) in readme


def test_records_agree_on_grid_and_geometry(records):
    tf, me = records['torchfdtd'], records['meep']
    assert tf['grid']['cells'] == me['grid']['cells']
    assert tf['grid']['steps'] == me['grid']['steps']
    assert tf['grid']['pml_cells'] == me['grid']['pml_cells']
    assert math.isclose(tf['grid']['mesh_um'], me['grid']['mesh_um'], rel_tol=1e-9)
    assert math.isclose(tf['grid']['dt_s'], me['grid']['dt_s'], rel_tol=1e-9)
    assert math.isclose(tf['grid']['courant_number'], me['grid']['courant_number'], rel_tol=1e-9)
    geometry_sha = hashlib.sha256((EXAMPLE / 'geometry.json').read_bytes()).hexdigest()
    assert tf['geometry_sha256'] == me['geometry_sha256'] == geometry_sha
    assert tf['staircase']['interior_ez_epsilon_sha256'] == me['staircase']['interior_ez_epsilon_sha256']
    assert tf['staircase']['interior_core_nodes'] == me['staircase']['interior_core_nodes']
    assert tf['source']['shared_waveform_sha256'] == me['source']['shared_waveform_sha256']
    assert np.allclose(tf['wavelength_um'], me['wavelength_um'], rtol=0, atol=1e-12)
    for s in SOLVERS:
        assert records[s]['grid']['steps'] == records[s]['timing']['samples'][0]['steps']


def test_geometry_matches_the_declared_lattice():
    g = json.loads((EXAMPLE / 'geometry.json').read_text(encoding='utf-8'))
    dx = g['mesh_um']
    assert g['cells'][0] * dx == pytest.approx(g['size_um'][0]) and g['cells'][1] * dx == pytest.approx(g['size_um'][1])
    assert g['cells'][0] % 2 == 0 and g['cells'][1] % 2 == 0
    assert 1.5 / g['core_index'] / dx >= 22
    assert g['steps'] * g['dt_s'] * 1e12 == pytest.approx(g['run_time_ps'])
    assert g['dt_s'] == pytest.approx(g['courant_number'] * dx * 1e-6 / 299792458.0, rel=1e-12)
    for value in (g['bus']['center_y_um'] + g['size_um'][1] / 2, g['source']['x_um'] + g['size_um'][0] / 2,
                  g['monitors']['in_x_um'] + g['size_um'][0] / 2, g['monitors']['out_x_um'] + g['size_um'][0] / 2):
        assert abs(value / dx - round(value / dx)) < 1e-9
    assert g['bus']['gap_um'] == pytest.approx(-(g['bus']['center_y_um'] + g['bus']['width_um'] / 2) - g['ring']['outer_radius_um'])


def test_declared_criteria_pass(result, criteria):
    limits = criteria['criteria']
    c = result['criteria']
    assert c['resonance_wavelength_nm']['value'] <= limits['resonance_wavelength_nm']['limit']
    assert c['q_relative']['value'] <= limits['q_relative']['limit']
    assert c['extinction_db']['value'] <= limits['extinction_db']['limit']
    assert c['rms_T']['value'] <= limits['rms_T']['limit']
    assert c['grid_match']['passed']
    assert result['all_passed']
