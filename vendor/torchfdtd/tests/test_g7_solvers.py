"""G7-04: derived geometries, matched-error interpolation and the committed accuracy-versus-cost records.

The default tests run no solver. They check the derived geometry of every mesh (edge-on-node detection
and the half-cell shift), the log-log interpolation, that the recorded Fourier means reproduce the
committed metagrating comparison, and that the committed G7-04 records reproduce summary.json, the
figure data and the tables of docs/G7_RESULTS.md. The acceptance test from the committed records runs
with TORCHFDTD_G7_FULL=1; the sweeps themselves are launched by the commands in docs/G7_RESULTS.md.
"""
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ROOT / 'examples' / 'g7' / 'solvers'
RECORDS = ROOT / 'docs' / 'validation' / 'g7' / 'G7-04'


def load(name):
    spec = importlib.util.spec_from_file_location(f'g7_solvers_{name}', SOLVERS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('common')


@pytest.fixture(scope='module')
def base():
    return common.load_geometry()


def test_base_edges_and_node_conventions(base):
    edges = common.material_edges(base)
    assert edges == dict(x=[-0.97, -0.89, -0.43, -0.21], y=[0.01, 0.51])
    # Even counts: both solvers share the Ez nodes; odd counts: Meep's nodes sit half a cell away.
    assert np.allclose(common.ez_nodes('torchfdtd', 2.0, 0.02, 100), common.ez_nodes('meep', 2.0, 0.02, 100), rtol=0, atol=1e-12)
    t = common.ez_nodes('torchfdtd', 3.64, 0.04, 91)
    m = common.ez_nodes('meep', 3.64, 0.04, 91)
    assert np.allclose(t - m, -0.02, rtol=0, atol=1e-12)
    assert math.isclose(m[45], 0.0, abs_tol=1e-15) and math.isclose(t[0], -1.82, abs_tol=1e-12)


@pytest.mark.parametrize('mesh,cells,pml,steps,tied', [(0.04, [50, 91], 10, 6000, False), (0.02, [100, 182], 20, 12000, False),
                                                        (0.01, [200, 364], 40, 24000, True), (0.005, [400, 728], 80, 48000, True)])
def test_derived_geometry(base, mesh, cells, pml, steps, tied):
    stair = common.derive_geometry(base, mesh, 'staircase')
    smooth = common.derive_geometry(base, mesh, 'smoothed')
    for g in (stair, smooth):
        assert g['cells'] == cells and g['pml_cells'] == pml and g['steps'] == steps
        assert math.isclose(g['pml_cells'] * g['mesh_um'], 0.4, rel_tol=1e-12)
        assert g['courant_number'] == base['courant_number']
        assert math.isclose(g['g7_04']['physical_time_s'], 12000 * 0.700035713374682 * 0.02e-6 / 299792458.0, rel_tol=1e-12)
        assert g['g7_04']['resolution_per_um'] == round(1 / mesh)
        before = g['g7_04']['edges_on_nodes_before_shift']
        for solver in ('torchfdtd', 'meep'):
            assert before[solver] == (dict(x=[-0.97, -0.89, -0.43, -0.21], y=[0.01, 0.51]) if tied else dict(x=[], y=[]))
    shift = mesh / 2 if tied else 0.0
    assert stair['g7_04']['shift_um'] == [shift, shift]
    assert smooth['g7_04']['shift_um'] == [0.0, 0.0]
    assert not common.any_tie(stair['g7_04']['edges_on_nodes']) and not common.any_tie(common.edge_ties(stair))
    assert common.any_tie(smooth['g7_04']['edges_on_nodes']) == tied
    assert math.isclose(stair['substrate_top_y_um'], 0.01 + shift, abs_tol=1e-12)
    for ridge, original in zip(stair['ridges'], base['ridges']):
        assert math.isclose(ridge['center_x_um'], original['center_x_um'] + shift, abs_tol=1e-12)
        assert ridge['width_um'] == original['width_um']
    # Source and DFT lines never move.
    assert stair['source'] == base['source'] and stair['monitors'] == base['monitors']


def test_shift_puts_every_edge_midway_between_nodes(base):
    for mesh in (0.01, 0.005):
        g = common.derive_geometry(base, mesh, 'staircase')
        edges = common.material_edges(g)
        for axis, name in enumerate('xy'):
            for solver in ('torchfdtd', 'meep'):
                nodes = common.ez_nodes(solver, g['cell_size_um'][axis], mesh, g['cells'][axis])
                for e in edges[name]:
                    assert math.isclose(np.min(abs(nodes - e)), mesh / 2, rel_tol=0, abs_tol=1e-9), (mesh, solver, name, e)


def test_tie_detection_on_a_synthetic_edge(base):
    g = common.derive_geometry(base, 0.02, 'smoothed')
    g['ridges'][0]['center_x_um'] = -0.92  # sides at -0.96 and -0.88: both Ez nodes of the 0.02 um grid
    ties = common.edge_ties(g)
    assert ties['torchfdtd']['x'] == ties['meep']['x'] == [-0.96, -0.88]
    assert ties['torchfdtd']['y'] == []


def test_cost_to_reach_log_log_interpolation():
    costs = [1.0, 8.0, 64.0, 512.0]
    errors = [0.04 * c ** -(2 / 3) for c in costs]  # a power law is a straight line in log-log
    for target in (0.01, 0.005, 0.002):
        entry = common.cost_to_reach(costs, errors, target)
        assert entry['status'] == 'interpolated'
        assert math.isclose(entry['cost'], (target / 0.04) ** -1.5, rel_tol=1e-12)
    assert common.cost_to_reach(costs, errors, 1e-4)['status'] == 'not reached'
    first = common.cost_to_reach(costs, errors, 0.05)
    assert first == dict(status='coarsest point', cost=1.0, between=[0, 0])
    # Non-monotone: the coarse point meets the target by cancellation, the next one does not; the last crossing counts.
    entry = common.cost_to_reach([1, 2, 4, 8], [0.001, 0.02, 0.004, 0.001], 0.01)
    assert entry['status'] == 'interpolated' and entry['between'] == [1, 2]
    assert math.isclose(entry['cost'], math.exp(math.log(2) + math.log(0.5) / math.log(0.2) * math.log(2)), rel_tol=1e-12)


def test_amplitudes_reproduce_committed_metagrating_comparison(base):
    records = ROOT / 'docs' / 'validation' / 'meep_comparison'
    comparison = json.loads((records / 'metagrating_comparison.json').read_text(encoding='utf-8'))
    for solver in ('torchfdtd', 'meep'):
        record = json.loads((records / f'metagrating_{solver}.json').read_text(encoding='utf-8'))
        T, R = common.efficiencies_from_amplitudes(common.point_amplitudes(record), base['substrate_index'])
        for j, m in enumerate(common.ORDERS):
            assert np.allclose(T[:, j], comparison['efficiencies'][solver]['T'][str(m)], rtol=0, atol=1e-12)
            assert np.allclose(R[:, j], comparison['efficiencies'][solver]['R'][str(m)], rtol=0, atol=1e-12)


@pytest.fixture(scope='module')
def committed():
    path = RECORDS / 'summary.json'
    if not path.exists():
        pytest.skip(f'{path} is not committed')
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def recomputed(committed):
    return load('analyze').analyze(RECORDS)


def assert_same(a, b, where):
    """Same structure and values; floats within 1e-12 relative, since log and exp may differ in the last bit between C libraries."""
    if isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0), (where, a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            assert_same(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{where}[{i}]')
    else:
        assert a == b, (where, a, b)


def test_records_reproduce_summary(committed, recomputed):
    assert recomputed['tables'] == committed['tables']
    for part in ('matched_costs', 'reference', 'acceptance', 'curves'):
        assert_same(recomputed[part], committed[part], part)


def test_results_page_carries_the_rendered_tables(recomputed):
    page = (ROOT / 'docs' / 'G7_RESULTS.md').read_text(encoding='utf-8')
    assert '## G7-04' in page
    for table in recomputed['tables'].values():
        for line in table:
            assert line in page, line
    assert recomputed['hardware_statement'] in page


def test_figure_renders_from_the_summary(recomputed, tmp_path):
    pytest.importorskip('matplotlib')
    path = tmp_path / 'G7-04.png'
    load('analyze').render_figure(recomputed, path)
    assert path.stat().st_size > 10_000
    assert (ROOT / 'docs' / 'figures' / 'g7' / 'G7-04.png').exists()


@pytest.mark.skipif(os.environ.get('TORCHFDTD_G7_FULL') != '1', reason='set TORCHFDTD_G7_FULL=1 for the G7-04 acceptance check')
def test_g7_04_acceptance_from_committed_records(base, committed, recomputed):
    case = json.loads((ROOT / 'docs' / 'validation' / 'cases' / 'G7-04r2.json').read_text(encoding='utf-8'))
    assert case['case_id'] == 'G7-04r2' and case['task'] == 'G7-04'
    fixture = case['fixture']
    analyze = load('analyze')
    # Every declared point exists, with three timed runs after one warm-up and the declared fixture.
    for solver, series, precision in analyze.CURVES:
        for mesh, resolution in zip(fixture['torchfdtd']['meshes_um'], fixture['meep']['resolutions_per_um']):
            record = json.loads(analyze.record_path(RECORDS, solver, series, precision, mesh).read_text(encoding='utf-8'))
            assert record['case'] == 'G7-04' and record['series'] == series and record['precision'] == precision
            assert record['geometry']['g7_04']['base_geometry_sha256'] == base['_sha256']
            assert record['geometry'] == common.derive_geometry(base, mesh, series)
            assert record['geometry_sha256'] == record['geometry']['_sha256']
            grid = record['grid']
            assert math.isclose(grid['physical_time_s'] * 1e15, fixture['physical_time_fs'], rel_tol=1e-3)
            assert math.isclose(grid['pml_um'], fixture['absorber_um'], rel_tol=1e-12)
            assert math.isclose(grid['courant_number'], base['courant_number'], rel_tol=1e-9)
            assert grid['cells'] == [round(2.0 / mesh), round(3.64 / mesh)]
            timing = record['timing']
            assert timing['repeats'] == fixture['repeats'] == len(timing['samples']) and 'warmup' in timing
            full = [s['full_seconds'] for s in timing['samples']]
            assert timing['median_full_seconds'] == float(np.median(full))
            assert timing['min_full_seconds'] == min(full) and timing['max_full_seconds'] == max(full)
            if solver == 'meep':
                assert record['resolution_per_um'] == resolution and record['environment']['meep_version'].startswith('1.34')
                assert record['environment']['mpi_processes'] == 4 and grid['eps_averaging'] == (series == 'smoothed')
            else:
                assert grid['backend'] == 'cuda' and grid['cuda_kernel'] == 'fused'
                assert grid['interface_method'] == ('subpixel' if series == 'smoothed' else 'staircase')
                assert 'RTX 3060' in timing['device']
            if series == 'staircase':
                assert not common.any_tie(record['geometry']['g7_04']['edges_on_nodes'])
    acceptance = committed['acceptance']
    assert all(item['passed'] for item in acceptance.values()), acceptance
    for key in committed['matched_costs']:
        for obs in analyze.OBSERVABLES:
            assert sorted(committed['matched_costs'][key][obs]) == sorted(f'{t:g}' for t in (0.01, 0.005, 0.002))
    assert committed['tables']['equal_cell_count'] and committed['tables']['equal_error']
    assert 'GPU' in committed['hardware_statement'] and 'CPU' in committed['hardware_statement'] and 'no conclusion' in committed['hardware_statement']
    # Reference: the order counts and the limit of the case, and every efficiency within the limit.
    orders, against, limit = re.fullmatch(r'TORCWA (\d+) Fourier orders, checked against (\d+) orders \(every efficiency at most ([0-9.e+-]+)\)',
                                          fixture['reference']).groups()
    check = committed['reference']['check']
    assert committed['reference']['order_count'] == int(orders) and check['against_order_count'] == int(against)
    assert check['limit_abs'] == float(limit) and check['case'] == 'docs/validation/cases/G7-04r2.json'
    rcwa = json.loads((RECORDS / 'rcwa_reference.json').read_text(encoding='utf-8'))
    ref, other = rcwa['bands'][orders], rcwa['bands'][against]
    largest = max(abs(a - b) for kind in ('T', 'R') for m in ref['T'] for a, b in zip(ref[kind][m], other[kind][m]))
    assert math.isclose(largest, check['max_abs_difference'], rel_tol=0, abs_tol=1e-15)
    assert largest <= float(limit) and check['passed'] and committed['reference_check_passed']
