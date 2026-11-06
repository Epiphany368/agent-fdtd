"""G7-03: fabrication-aware passive PIC coupler workflow (examples/g7/coupler/workflow.py).

The declaration is docs/G7_WORKFLOWS.md (section G7-03) and docs/validation/cases/G7-03.json;
the limits asserted by the judged tests are the case's acceptance criteria, copied into
workflow.LIMITS, and nothing is tuned after a run. The default tests run on the CPU at G6's
coarse mesh and check the mechanics: S-matrix assembly, passivity and reciprocity of a straight
guide, the adjoint derivative against central differences, the fabrication check with the
0.1 um sub-pixel erosion and dilation, the GDS round trip, and the record schema. They also
re-judge the committed records under docs/validation/g7/G7-03 when present.

TORCHFDTD_G7_FULL=1 runs the declared workflow (three seeds, 0.05 um, CUDA float32) after the
non-judged same-mesh baseline diagnostic, one test each so every one can take the GPU lock
separately, then judges the records; TORCHFDTD_G7_RECORD=<dir> keeps the records there
(docs/validation/g7/G7-03 for the evidence).
"""
import sys
from pathlib import Path

# examples/ ships with the repository, not with the wheel, so a run against an installed package still reads
# it from the checkout that holds this test; appended, never prepended, so the installed torchfdtd wins.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from examples import design_mode_coupler as base
from examples.g7.coupler import workflow
from torchfdtd import AdjointOptions, FixedModePort, ModeNetwork
from torchfdtd.design_problem import DesignProblem
from torchfdtd.fabrication import measure_feature_sizes

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT/'docs/validation/g7/G7-03'
FULL = os.environ.get('TORCHFDTD_G7_FULL') == '1'
CUDA = torch.cuda.is_available()
RECORD = os.environ.get('TORCHFDTD_G7_RECORD')


def test_case_declaration_matches_the_workflow_constants():
    case = json.loads((ROOT/workflow.CASE).read_text(encoding='utf-8'))
    fixture = case['fixture']
    settings = workflow.Settings()
    assert (fixture['core_eps'], fixture['cladding_eps']) == (base.CORE_EPSILON, base.CLADDING_EPSILON)
    assert fixture['guide_width_um'] == base.GUIDE_WIDTH_UM and fixture['offset_um'] == 2*base.GUIDE_OFFSET_UM
    assert case['case_id'] == 'G7-03r2'
    assert tuple(fixture['design_box_pixels']) == workflow.PIXELS and fixture['pixel_um'] == workflow.PIXEL_UM
    # The revised pixels tile the example's unchanged design box.
    assert (workflow.PIXELS[0]*workflow.PIXEL_UM, workflow.PIXELS[1]*workflow.PIXEL_UM) == pytest.approx(
        (base.BOX_UM[1]-base.BOX_UM[0], base.BOX_UM[3]-base.BOX_UM[2]))
    assert (fixture['mesh_um'], fixture['check_mesh_um']) == (settings.mesh_um, settings.check_mesh_um)
    assert fixture['holdout_wavelength_um'] == workflow.HOLDOUT_UM
    assert tuple(fixture['evaluation']['wavelengths_um']) == workflow.WAVELENGTHS_UM
    assert fixture['fabrication'] == dict(min_linewidth_um=workflow.MIN_FEATURE_UM, min_gap_um=workflow.MIN_FEATURE_UM,
                                          erosion_dilation_um=workflow.EROSION_UM)
    assert case['baseline']['value'] == workflow.LIMITS['performance']
    # The same physical time as G6's 500 steps at 0.2 um at both meshes.
    assert settings.steps*settings.mesh_um == pytest.approx(500*.2) and settings.check_steps*settings.check_mesh_um == pytest.approx(500*.2)
    assert settings.device == 'cuda' and case['precision'] == 'float32 on CUDA'
    # The filter radius is chosen on development seeds only, ascending from the declared minimum feature.
    assert not set(workflow.DEVELOPMENT_SEEDS) & set(workflow.SEEDS)
    candidates = workflow.RADIUS_CANDIDATES_UM
    assert candidates[0] == workflow.MIN_FEATURE_UM and list(candidates) == sorted(candidates)
    assert all(b-a == pytest.approx(workflow.PIXEL_UM) for a, b in zip(candidates, candidates[1:]))


def test_radius_selection_takes_the_smallest_complete_compliant_candidate_and_refuses_declared_seeds():
    def run(radius, seed, violations=()):
        return dict(radius_um=radius, seed=seed, violations=list(violations))
    seeds = workflow.DEVELOPMENT_SEEDS
    failing = [run(.4, s, ['min_gap'] if s == seeds[0] else []) for s in seeds]
    passing = [run(.5, s) for s in seeds]
    assert workflow.select_radius(failing+passing) == .5
    assert workflow.select_radius(failing) is None                       # no compliant candidate yet
    assert workflow.select_radius(failing+passing[:2]) is None           # 0.5 um not run for every development seed
    assert workflow.select_radius(passing) is None                       # 0.4 um never tried, so 0.5 um is not known smallest
    with pytest.raises(ValueError, match='declared seeds'):
        workflow.develop(.4, workflow.SEEDS[0], workflow.REDUCED)
    with pytest.raises(ValueError, match='development seeds first'):
        workflow.build_problem(1, replace(workflow.Settings(), filter_radius_um=None), lambda density: density.sum())


def test_committed_development_records_select_the_radius_in_use():
    if not (RECORDED/'development').is_dir():
        pytest.skip('no recorded development selection under docs/validation/g7/G7-03/development')
    selection = workflow.development_selection(RECORDED)
    assert selection['judged_seeds_used'] is False
    assert {run['seed'] for run in selection['runs']} == set(workflow.DEVELOPMENT_SEEDS)
    assert selection['selected_um'] == workflow.Settings().filter_radius_um
    for run in selection['runs']:
        record = json.loads((RECORDED/'development'/f"radius-{run['radius_um']:g}-seed{run['seed']}.json").read_text(encoding='utf-8'))
        sizes = measure_feature_sizes(np.array(record['binary']), workflow.PIXEL_UM, boundary='extend')
        assert (sizes.linewidth_pixels, sizes.gap_pixels) == (run['linewidth_px'], run['gap_px'])


def test_development_run_records_feature_sizes_and_transmissions(tmp_path):
    settings = replace(workflow.REDUCED, steps=100, iterations=1)
    record = workflow.develop(.5, workflow.DEVELOPMENT_SEEDS[0], settings)
    assert record['settings']['filter_radius_um'] == .5 and len(record['trace']) == 1
    assert np.array(record['binary']).shape == workflow.PIXELS
    assert record['linewidth_px'] == round(record['feature_sizes']['min_linewidth_um']/workflow.PIXEL_UM)
    assert set(record['transmission']) == {'last_iterate', 'binary_density', 'structures'}
    workflow.write_json(tmp_path/'development'/'radius-0.5-seed11.json', record)
    selection = workflow.development_selection(tmp_path)
    assert selection['runs'][0]['seed'] == workflow.DEVELOPMENT_SEEDS[0] and selection['judged_seeds_used'] is False
    assert selection['selected_um'] is None                              # one run is never a complete selection


def test_s_record_maps_the_ports_and_measures_passivity_and_reciprocity():
    s = torch.tensor([[.1+.2j, .5-.3j], [.5-.301j, -.2+.1j]], dtype=torch.complex64)
    record = workflow.s_record(s)
    assert record['S21']['real'] == pytest.approx(.5) and record['S21']['imag'] == pytest.approx(-.301)
    assert record['S12']['imag'] == pytest.approx(-.3)
    assert record['S11']['power'] == pytest.approx(.05, rel=1e-6)
    assert record['S22']['phase_rad'] == pytest.approx(np.angle(-.2+.1j), rel=1e-6)
    assert record['transmission'] == record['S21']['power']
    assert record['column_power'] == pytest.approx([.05+.5**2+.301**2, .34+.05], rel=1e-6)
    assert record['reciprocity'] == pytest.approx(1e-3, rel=1e-3)
    with pytest.raises(ValueError, match='finite 2 x 2'):
        workflow.s_record(torch.zeros(3, 3, dtype=torch.complex64))


def test_straight_guide_through_the_coupler_region_is_passive_reciprocal_and_phased():
    # Both ports on one guide centred at y = 0, no design: |S21| = 1 and S21 = exp(i beta L).
    project = base.build_project(.2, 500, workflow.DESIGN_UM)
    ports = (FixedModePort('left', -base.PORT_UM, -base.SOURCE_UM, 1), FixedModePort('right', base.PORT_UM, base.SOURCE_UM, -1))
    straight = base.section(0.)
    network = ModeNetwork(project, ports, options=AdjointOptions(checkpoints=4), num_modes=1,
                          port_permittivities=dict(left=straight, right=straight))
    with torch.no_grad():
        s = network(network.reference_epsilon(port='left')).s
    record = workflow.s_record(s)
    assert max(abs(p-1) for p in record['column_power']) <= .01
    assert max(record['column_power']) <= workflow.LIMITS['passivity']
    assert record['reciprocity'] <= workflow.LIMITS['reciprocity']
    assert record['S11']['power'] <= 1e-6 and record['S22']['power'] <= 1e-6
    expected = np.exp(1j*network._launches[0].mode.beta_per_um*2*base.PORT_UM)
    assert abs(complex(s[1, 0])-expected) <= .02


def test_adjoint_derivative_matches_central_differences_on_the_reduced_coupler():
    settings = workflow.REDUCED
    model = workflow.Coupler(settings.mesh_um, settings.steps, workflow.DESIGN_UM, 'cpu')
    problem = workflow.build_problem(1, settings, model.objective)
    start = problem.parameterization.design.detach().clone()
    check = workflow.gradient_check(problem, model, settings)
    assert torch.equal(problem.parameterization.design.detach(), start)
    assert problem.parameterization.design.grad is None
    assert len(check['pixels']) == settings.fd_pixels
    assert len({workflow.orbit(tuple(row['pixel'])) for row in check['pixels']}) == settings.fd_pixels
    assert check['max_relative_error'] <= workflow.LIMITS['gradient'], check['pixels']
    # The checked pixels are the largest adjoint magnitudes of distinct orbits, in decreasing order.
    magnitudes = [abs(row['adjoint']) for row in check['pixels']]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_fabrication_check_and_the_sub_pixel_erosion_and_dilation():
    design = np.zeros(workflow.PIXELS, dtype=int)
    design[4:8, 8:12] = 1                                  # a 0.4 um square of 4 by 4 pixels: meets the declared 0.4 um
    sizes = measure_feature_sizes(design, workflow.PIXEL_UM, boundary='extend')
    assert sizes.violations(min_linewidth_um=workflow.MIN_FEATURE_UM, min_gap_um=workflow.MIN_FEATURE_UM) == ()
    thin = design.copy()
    thin[4:8, 11] = 0                                      # a 0.3 um line
    assert 'min_linewidth' in measure_feature_sizes(thin, workflow.PIXEL_UM, boundary='extend').violations(
        min_linewidth_um=workflow.MIN_FEATURE_UM)
    fine, eroded, dilated, realized = workflow.perturbed_designs(design)
    assert realized == pytest.approx(workflow.EROSION_UM) and fine.shape == (24, 40)
    rows, columns = np.nonzero(fine)
    assert (rows.min(), rows.max(), columns.min(), columns.max()) == (8, 15, 16, 23)
    for array, bounds in ((eroded, (10, 13, 18, 21)), (dilated, (6, 17, 14, 25))):  # 0.1 um = two 0.05 um cells per side
        rows, columns = np.nonzero(array)
        assert (rows.min(), rows.max(), columns.min(), columns.max()) == bounds
        assert array.sum() == (bounds[1]-bounds[0]+1)*(bounds[3]-bounds[2]+1)
    # The box edge is extended, not eroded: a solid pixel on the edge keeps its edge side.
    edge = np.zeros(workflow.PIXELS, dtype=int)
    edge[0:4, 0:4] = 1
    _, eroded, _, _ = workflow.perturbed_designs(edge)
    rows, columns = np.nonzero(eroded)
    assert (rows.min(), rows.max(), columns.min(), columns.max()) == (0, 5, 0, 5)
    structures = workflow.structures_of(eroded, 'edge')
    assert sum(s.size[0]*s.size[1] for s in structures) == pytest.approx(36*.05**2, rel=1e-6)


def test_gds_round_trip_of_a_small_design_voxelizes_like_the_rectangles(tmp_path):
    settings = workflow.REDUCED
    model = workflow.Coupler(settings.mesh_um, 20, workflow.DESIGN_UM, 'cpu')
    fine = workflow.Coupler(.05, 20, workflow.DESIGN_UM, 'cpu')
    problem = workflow.build_problem(2, settings, model.objective)
    binary = problem.density(hard=True)
    assert 0 < binary.sum() < binary.numel()
    exported = problem.export(tmp_path, origin_um=(base.BOX_UM[0], base.BOX_UM[2]), spacing_um=workflow.PIXEL_UM,
                              z_min_um=-base.SLAB_UM/2, z_max_um=base.SLAB_UM/2, material=base.MATERIAL)
    back = DesignProblem.reimport(exported)
    assert len(back['structures']) == len(exported['structures']) and back['gds_structures']
    assert json.loads(Path(exported['binary_path']).read_text(encoding='utf-8'))['pixels'] == binary.int().tolist()
    for forward in (model, fine):
        before = forward.structure_epsilon(exported['structures'])
        assert not torch.equal(before, forward.base)
        assert torch.equal(forward.structure_epsilon(back['structures']), before)
        assert torch.equal(forward.structure_epsilon(back['gds_structures']), before)


def test_reduced_workflow_writes_complete_records_and_judges_them(tmp_path):
    settings = replace(workflow.REDUCED, steps=100, check_steps=200, iterations=1, fd_pixels=1)
    records, summary = workflow.run([3], settings, tmp_path)
    record = json.loads((tmp_path/'seed3.json').read_text(encoding='utf-8'))
    assert record == json.loads(json.dumps(records[0]))
    assert {'history', 'gradient_check', 'fabrication', 'evaluations', 'transmission', 'differences', 'holdout',
            'export', 'wall_time_s', 'environment', 'binary', 'final_density', 'settings'} <= set(record)
    assert set(record['evaluations']) == {'smooth_density', 'binary_density', 'structures', 'gds', 'eroded', 'dilated',
                                          'gds_check_mesh'}
    for stage in record['evaluations'].values():
        assert set(stage) == {'1.50', '1.55', '1.60'}
        for entry in stage.values():
            assert {'S11', 'S21', 'S12', 'S22', 'column_power', 'reciprocity', 'transmission'} <= set(entry)
            assert {'real', 'imag', 'power', 'phase_rad'} == set(entry['S21'])
    assert len(record['history']) == 1 and record['iterations'] == 1
    assert (tmp_path/record['export']['gds']).is_file()
    assert all(b'\r' not in path.read_bytes() for path in (tmp_path/'export').rglob('*.json'))
    assert record['differences']['1.55']['gds_round_trip'] == 0.
    saved = json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    assert saved == json.loads(json.dumps(summary)) and saved['judged'] is False
    names = [c['criterion'] for c in saved['criteria']]
    assert names == ['a_all_seeds', 'b_performance', 'c_passivity', 'd_reciprocity', 'e_gradient', 'f_gds_round_trip', 'g_mesh']
    assert saved['criteria'][0]['passed'] is False     # one seed of three is never a complete run
    assert saved['criteria'][2]['cases'] == 7*3
    selection = saved['development_selection']
    assert selection['runs'] == [] and selection['selected_um'] is None and selection['judged_seeds_used'] is False
    assert selection['radius_used_um'] == settings.filter_radius_um
    assert 'diagnostics_same_mesh_baseline' not in saved
    environment = record['environment']
    assert environment['commit'] and Path(environment['torchfdtd_file']).name == '__init__.py'
    assert 'torchfdtd_version' in environment and environment['torchfdtd_import'] in ('checkout', 'installed')


def test_same_mesh_baseline_reevaluates_the_g6_designs_without_judging_them():
    settings = replace(workflow.REDUCED, steps=100, check_steps=200)
    record = workflow.same_mesh_baseline(workflow.Models(settings), settings)
    assert record['judged'] is False and 'not optimized in G7' in record['note']
    assert sorted(record['designs']) == list(workflow.SEEDS)
    for seed, design in record['designs'].items():
        g6 = json.loads((ROOT/f'docs/validation/g6/coupler-seed{seed}.json').read_text(encoding='utf-8'))
        assert design['binary'] == g6['binary']
        # G6 declared the same 0.4 um rule, so the measured violations agree with its record.
        assert design['violations'] == g6['fabrication']['violations']
        assert design['committed_gds_voxelizes_identically'] is True
        assert design['g6_recorded_fine_gds_transmission'] == g6['final_evaluation']['stages']['fine_gds']['transmission']
        for stage in ('structures', 'gds', 'gds_check_mesh'):
            assert set(design['transmission'][stage]) == {'1.50', '1.55', '1.60'}
        assert design['transmission']['gds'] == design['transmission']['structures']


def judge_directory(directory):
    summary = workflow.summarize(directory)
    table = json.dumps({c['criterion']: dict(value=c['value'], limit=c['limit'], passed=c['passed'])
                        for c in summary['criteria']}, indent=1)
    print(table)
    assert summary['judged'], 'the records are not a complete declared run'
    failing = [c['criterion'] for c in summary['criteria'] if not c['passed']]
    assert not failing, f'G7-03 criteria not met: {failing}\n{table}'


CRITERIA = ('a_all_seeds', 'b_performance', 'c_passivity', 'd_reciprocity', 'e_gradient', 'f_gds_round_trip', 'g_mesh')
# Criteria the committed judged run did not meet. They stay visible as strict expected failures, so any change of
# the records (either way) fails the suite until this mapping is revised; revising the case is the owner's decision.
RECORDED_FAILURES = {}


@pytest.fixture(scope='module')
def rejudged(tmp_path_factory):
    paths = sorted(RECORDED.glob('seed*.json'))
    if not paths:
        pytest.skip('no recorded G7-03 run under docs/validation/g7/G7-03')
    directory = tmp_path_factory.mktemp('g7-coupler-recorded')
    for path in paths:
        (directory/path.name).write_bytes(path.read_bytes())
    return workflow.summarize(directory)


def test_recorded_run_is_complete_and_reproduces_its_summary(rejudged):
    recorded = json.loads((RECORDED/'summary.json').read_text(encoding='utf-8'))
    assert rejudged['judged'] and recorded['judged'] and rejudged['seeds'] == list(workflow.SEEDS)
    assert json.loads(json.dumps(rejudged['criteria'])) == recorded['criteria']
    assert [c['criterion'] for c in recorded['criteria']] == list(CRITERIA)
    selection = recorded['development_selection']
    assert selection['judged_seeds_used'] is False
    assert selection['selected_um'] == selection['radius_used_um'] == workflow.Settings().filter_radius_um
    assert recorded['diagnostics_same_mesh_baseline']['judged'] is False
    assert all(path.is_file() for path in (RECORDED/'export').glob('seed*/*.gds'))
    assert len(list((RECORDED/'export').glob('seed*/*.gds'))) == len(workflow.SEEDS)
    assert all(b'\r' not in path.read_bytes() for path in RECORDED.rglob('*.json'))


@pytest.mark.parametrize('criterion', [pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=RECORDED_FAILURES[name]))
                                       if name in RECORDED_FAILURES else name for name in CRITERIA])
def test_recorded_run_meets_the_criterion(rejudged, criterion):
    entry = next(c for c in rejudged['criteria'] if c['criterion'] == criterion)
    print(json.dumps({key: entry[key] for key in ('criterion', 'value', 'limit', 'passed')}))
    assert entry['passed'], entry


def _full_directory(tmp_path_factory):
    return Path(RECORD) if RECORD else tmp_path_factory.getbasetemp()/'g7-coupler'


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G7_FULL=1 for the judged G7-03 run (about half an hour per seed)')
def test_full_same_mesh_baseline(tmp_path_factory):
    assert CUDA, 'the judged G7-03 run needs CUDA'
    record = workflow.run_baseline(workflow.Settings(), _full_directory(tmp_path_factory))
    assert record['judged'] is False and sorted(record['designs']) == list(workflow.SEEDS)


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G7_FULL=1 for the judged G7-03 run (about half an hour per seed)')
@pytest.mark.parametrize('seed', workflow.SEEDS)
def test_full_declared_seed(seed, tmp_path_factory):
    assert CUDA, 'the judged G7-03 run needs CUDA'
    directory = _full_directory(tmp_path_factory)
    records, _ = workflow.run([seed], workflow.Settings(), directory)
    assert records[0]['seed'] == seed and len(records[0]['history']) == workflow.Settings().iterations


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G7_FULL=1 for the judged G7-03 run (about half an hour per seed)')
def test_full_declared_run_meets_every_criterion(tmp_path_factory):
    judge_directory(_full_directory(tmp_path_factory))
