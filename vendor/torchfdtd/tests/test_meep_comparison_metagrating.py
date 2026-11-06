"""Metagrating Meep comparison: the committed records reproduce the comparison, agree on the grid and pass the criteria.

No simulation runs here; everything is read from docs/validation/meep_comparison.
"""
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'meep_comparison' / 'metagrating'
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'


def load_compare():
    spec = importlib.util.spec_from_file_location('metagrating_compare', EXAMPLE / 'compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def records():
    names = ('metagrating_torchfdtd.json', 'metagrating_meep.json', 'metagrating_rcwa.json', 'metagrating_comparison.json')
    out = {}
    for name in names:
        path = RECORDS / name
        if not path.exists():
            pytest.skip(f'{path} is not committed')
        out[name.split('_', 1)[1][:-5]] = json.loads(path.read_text(encoding='utf-8'))
    return out


@pytest.fixture(scope='module')
def recomputed(records):
    compare = load_compare()
    import hashlib
    raw = (EXAMPLE / 'geometry.json').read_bytes()
    geometry = json.loads(raw.decode('utf-8'))
    geometry['_sha256'] = hashlib.sha256(raw).hexdigest()
    criteria = json.loads((EXAMPLE / 'criteria.json').read_text(encoding='utf-8'))
    comparison, results, wavelength, i_design = compare.compare(records['torchfdtd'], records['meep'], records['rcwa'], geometry, criteria)
    return comparison


def test_compare_reproduces_committed_comparison(records, recomputed):
    committed = records['comparison']
    assert recomputed['geometry_sha256'] == committed['geometry_sha256']
    assert recomputed['orders'] == committed['orders']
    assert np.allclose(recomputed['wavelength_um'], committed['wavelength_um'], rtol=0, atol=1e-12)
    for solver in ('torchfdtd', 'meep', 'rcwa'):
        for kind in ('T', 'R'):
            for order, values in committed['efficiencies'][solver][kind].items():
                assert np.allclose(recomputed['efficiencies'][solver][kind][order], values, rtol=0, atol=1e-12), (solver, kind, order)
    for name, metric in committed['metrics'].items():
        assert math.isclose(recomputed['metrics'][name]['value'], metric['value'], rel_tol=0, abs_tol=1e-12), name
        assert recomputed['metrics'][name]['passed'] == metric['passed']
    assert recomputed['table_markdown'] == committed['table_markdown']


def test_readme_table_matches_records(records, recomputed):
    readme = (EXAMPLE / 'README.md').read_text(encoding='utf-8')
    for line in recomputed['table_markdown']:
        assert line in readme, line
    for name, metric in recomputed['metrics'].items():
        assert re.search(rf'{name}[^\n]*{metric["value"]:.5f}', readme), name


def test_records_agree_on_grid_and_geometry(records, recomputed):
    t, m = records['torchfdtd'], records['meep']
    assert t['grid']['cells'] == m['grid']['cells'] == [100, 182]
    assert math.isclose(t['grid']['dt_s'], m['grid']['dt_s'], rel_tol=1e-9)
    assert t['grid']['steps'] == m['grid']['steps'] == m['grid']['steps_run'] == 12000
    assert t['grid']['pml_cells'] == m['grid']['pml_cells'] == 20
    assert t['geometry_sha256'] == m['geometry_sha256'] == records['rcwa']['geometry_sha256']
    assert t['staircase']['silicon_columns'] == m['staircase']['silicon_columns']
    assert t['staircase']['silicon_rows'] == m['staircase']['silicon_rows']
    assert t['staircase']['substrate_rows'] == m['staircase']['substrate_rows']
    assert all(recomputed['grid_match'].values()), recomputed['grid_match']


def test_declared_criteria_pass(records, recomputed):
    criteria = json.loads((EXAMPLE / 'criteria.json').read_text(encoding='utf-8'))
    assert criteria['criteria']['torchfdtd_vs_meep_order_efficiency']['limit_abs'] == 0.01
    assert criteria['criteria']['torchfdtd_vs_rcwa_order_efficiency']['limit_abs'] == 0.02
    assert criteria['criteria']['torchfdtd_energy_balance']['limit_abs'] == 0.01
    for name, spec in criteria['criteria'].items():
        metric = recomputed['metrics'][name]
        assert metric['limit_abs'] == spec['limit_abs']
        assert metric['value'] <= spec['limit_abs'], (name, metric)
        assert metric['passed']
    assert recomputed['all_criteria_passed']


def test_rcwa_convergence_recorded(records):
    rcwa = records['rcwa']
    assert rcwa['harmonics'] == rcwa['convergence'][-1]['harmonics'] or any(
        row['harmonics'] == rcwa['harmonics'] and row['max_abs_change_from_previous'] < rcwa['convergence_tolerance'] for row in rcwa['convergence'][1:])
    assert rcwa['max_abs_total_minus_one'] < 1e-9
    assert 'Comput. Phys. Commun. 282, 108552 (2023)' in rcwa['citation']


def test_reference_runs_reproduce_fresnel(recomputed):
    for solver in ('torchfdtd', 'meep'):
        diagnostics = recomputed['diagnostics'][solver]
        assert diagnostics['max_abs_reference_T0_minus_fresnel'] < 1e-3
        assert diagnostics['max_abs_reference_R0_minus_fresnel'] < 1e-3
        assert diagnostics['max_reference_nonzero_order_power_over_incident'] < 1e-12
        assert diagnostics['max_reference_transmission_backward_over_incident'] < 1e-5
        assert diagnostics['all_orders_propagating']
