"""G7-01: the metagrating application workflow of docs/G7_WORKFLOWS.md, judged on the criteria of
docs/validation/cases/G7-01r3.json.

The fast tests check the mechanics at a CPU size: the permittivity transfer, the diffraction-order
decomposition against the fixture's compare.py, the Rayleigh anomalies, the robust projection and
its objective, the threshold-selection rule, the objective's adjoint gradient against finite
differences, the bare-substrate Fresnel balance for TE and TM at both incidences, the TORCWA script
against a uniform film, and the records of a short run. TORCHFDTD_G7_FULL=1 runs the declared
workflow (the RTX 3060 and the TORCWA interpreter) from the threshold selection recorded in the
records directory: one seed stage per declared seed and the baseline stage as parallel subprocesses,
each prefixed by TORCHFDTD_G7_LAUNCHER (the GPU lock command on the shared workstation), then one
TORCWA process for every seed with TORCHFDTD_G7_RCWA_THREADS threads (started once
TORCHFDTD_G7_PAUSE_FILE, if named, does not exist), then the judgement. TORCHFDTD_G7_RECORD=<dir>
names the records directory (docs/validation/g7/G7-01 for the committed evidence) and
TORCHFDTD_G7_CHECKPOINT_DIR=<dir> keeps the design states so that an interrupted run resumes. The
subprocesses import the same torchfdtd as this test (checkout or installed wheel). Without the flag,
the committed records are re-judged when present. Every seed is judged; none is selected.
"""
import importlib.util
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.g7.metagrating import workflow

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/'docs'/'validation'/'g7'/'G7-01'
FULL = os.environ.get('TORCHFDTD_G7_FULL') == '1'
RECORD = os.environ.get('TORCHFDTD_G7_RECORD')
CHECKPOINTS = os.environ.get('TORCHFDTD_G7_CHECKPOINT_DIR')
RCWA_PYTHON = os.environ.get('TORCHFDTD_RCWA_PYTHON', 'C:/anaconda3/python.exe')
LAUNCHER = os.environ.get('TORCHFDTD_G7_LAUNCHER', '').split()
RCWA_THREADS = os.environ.get('TORCHFDTD_G7_RCWA_THREADS')
PAUSE = os.environ.get('TORCHFDTD_G7_PAUSE_FILE')
CRITERIA = ('all_seeds', 'performance', 'rcwa_agreement', 'mesh', 'energy_balance', 'fabrication')


@pytest.fixture(scope='module')
def fixture():
    case, g, _ = workflow.declared()
    return case, g


def small(**changes):
    return replace(workflow.Settings.small_run(), **changes)


def fake_plane(x_um, fields):
    x = torch.as_tensor(x_um, dtype=torch.float64)
    return SimpleNamespace(fields=torch.as_tensor(fields), points_um=torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1),
                           weights=torch.full_like(x, x[1]-x[0]))


def test_binary_density_is_an_exact_staircase_on_the_design_mesh(fixture):
    _, g = fixture
    region = workflow.build_project(g, .02, 'TE', 'normal', small(extension_um=0.), 56.).region
    ez = workflow.layer_epsilon(torch.as_tensor(workflow.two_ridge_density(g)), region, g)[:, :, 0, 2].numpy()
    si, sub = g['ridge_index']**2, g['substrate_index']**2
    # the fixture's staircase: Ez columns 2-5 and 29-39, rows 92-116, substrate rows 0-91
    assert np.flatnonzero(np.isclose(ez[:, 100], si)).tolist() == list(range(2, 6))+list(range(29, 40))
    assert np.flatnonzero(np.isclose(ez[3], si)).tolist() == list(range(92, 117))
    assert np.flatnonzero(np.isclose(ez[0], sub)).tolist() == list(range(92))
    assert set(np.unique(np.round(ez, 6))) <= {1., round(sub, 6), round(si, 6)}
    fine = workflow.build_project(g, .01, 'TM', 'normal', small(extension_um=0.), 56.).region
    eps = workflow.layer_epsilon(torch.as_tensor(workflow.two_ridge_density(g)), fine, g)[:, :, 0].numpy()
    # on the 0.01 um grid Ez column 3 (x = -0.97 um) lies on the left edge of the first ridge and row 183 on the substrate top:
    # both receive the mean of the two sides; the normal components Ex (walls) and Ey (layer faces) never straddle
    assert math.isclose(eps[3, 200, 2], (si+1)/2, rel_tol=1e-6) and math.isclose(eps[0, 183, 2], (sub+1)/2, rel_tol=1e-6)
    assert set(np.unique(np.round(eps[:, 184:233, 0], 5))) == {1., round(si, 5)}
    assert set(np.unique(np.round(eps[4:11, :, 1], 5))) == {1., round(sub, 5), round(si, 5)}


def test_revised_case_sets_the_declared_run(fixture):
    case, g = fixture
    p = workflow.case_parameters(case)
    assert p['shift_candidates'] == [.1, .2, .25] and p['shift_fallback'] == .25 and p['development_seeds'] == [11, 12, 13]
    assert (p['selection_mesh_um'], p['selection_time_fs'], p['design_time_fs'], p['evaluation_time_fs']) == (.02, 560., 1120., 2240.)
    assert (p['exclusion_um'], p['rcwa_error_target'], p['absorber_um'], p['extension_um']) == (.02, .003, .4, 3.)
    settings = workflow.Settings.declared(case, dict(chosen_threshold_shift=.2))
    assert (settings.design_mesh_um, settings.fine_mesh_um, settings.physical_time_fs, settings.evaluation_time_fs) == (.01, .005, 1120., 2240.)
    assert (settings.filter_radius_um, settings.threshold_shift, settings.open_close, settings.extension_um) == (.06, .2, True, 3.)
    assert (settings.seeds, settings.betas, settings.iterations, settings.band_points) == ((1, 2, 3), (8., 16., 32., 64.), 80, 41)
    selection = workflow.Settings.for_selection(case, .1, 12)
    assert (selection.design_mesh_um, selection.physical_time_fs, selection.evaluation_time_fs, selection.extension_um) == (.02, 560., 560., 3.)
    assert (selection.threshold_shift, selection.open_close, selection.seeds) == (.1, False, (12,))
    # evaluations at 0.005 um: 1928 rows for the 9.64 um cell, the 0.4 um absorber as 80 cells per face, 2240 fs
    region = workflow.build_project(g, .005, 'TE', 'normal', settings, settings.evaluation_time_fs).region
    assert region.shape[:2] == (400, 1928) and region.pml_layers(1, 0) == region.pml_layers(1, 1) == 80
    assert math.isclose(region.steps*region.time_step, 2240e-15, rel_tol=1e-4)
    with pytest.raises(ValueError, match='chose no shift'):
        workflow.Settings.declared(case, dict(chosen_threshold_shift=None))


def robust_density(shift=.2, beta=16., seed=4):
    initial = .8*torch.randn((workflow.pixel_count(workflow.declared()[1]), 1), generator=torch.Generator().manual_seed(seed))
    options = dict(spacing_um=(.02, .5), initial=initial, mode='logits', filter_radius_um=.06, boundary='periodic', beta=beta, eta=.5)
    return workflow.RobustDensity((100, 1), threshold_shift=shift, **options), workflow.DensityParameterization((100, 1), **options)


def test_robust_projection_brackets_the_nominal_design():
    robust, plain = robust_density()
    nominal = robust()
    torch.testing.assert_close(nominal, plain(), rtol=0, atol=1e-6)
    torch.testing.assert_close(robust(hard=True), plain(hard=True), rtol=0, atol=0)
    eroded, dilated = nominal.realizations['eroded'], nominal.realizations['dilated']
    assert bool((eroded <= nominal+1e-7).all()) and bool((nominal <= dilated+1e-7).all()) and bool((eroded < dilated).any())
    # the thresholded eroded design keeps less silicon than the nominal one, the dilated one more
    assert (eroded >= .5).sum() <= (nominal >= .5).sum() <= (dilated >= .5).sum()
    eroded.sum().backward()
    assert robust.design.grad is not None and float(robust.design.grad.abs().sum()) > 0
    assert float(robust.beta) == 16. and robust.get_extra_state()['threshold_shift'] == .2
    with pytest.raises(ValueError, match='inside'):
        robust_density(shift=.5)


def test_robust_objective_takes_the_smallest_realization_and_its_gradient(fixture):
    _, g = fixture
    objective = workflow.TransmissionObjective(g, small())
    robust, _ = robust_density(beta=8.)
    density = robust()
    loss, metrics = objective(density)
    values = [metrics[f'T+1 mean {name}'] for name in workflow.REALIZATIONS]
    active = workflow.REALIZATIONS[int(metrics['active realization'])]
    assert math.isclose(-loss.item(), min(values), rel_tol=1e-6) and values[workflow.REALIZATIONS.index(active)] == min(values)
    loss.backward()
    reference, _ = robust_density(beta=8.)
    again = reference()
    chosen = (again if active == 'nominal' else again.realizations[active])*1     # the same graph, without the realizations
    objective(chosen)[0].backward()
    torch.testing.assert_close(robust.design.grad, reference.design.grad, rtol=1e-5, atol=1e-9)
    # a fixed density (DesignProblem.evaluate) is evaluated alone
    with torch.no_grad():
        plain_loss, plain_metrics = objective(density.detach().clone())
    assert 'active realization' not in plain_metrics and math.isclose(-plain_loss.item(), metrics['T+1 mean nominal'], rel_tol=1e-6)


def test_rayleigh_anomalies_and_the_excluded_wavelengths(fixture):
    _, g = fixture
    kx = workflow.bloch_kx(g)
    inside = [a for a in workflow.rayleigh_anomalies(g, kx) if 1.5 <= a['wavelength_um'] <= 1.6]
    assert [(a['order'], a['medium']) for a in inside] == [(1, 'air')] and math.isclose(inside[0]['wavelength_um'], 1.5111, abs_tol=1e-4)
    wavelength = np.linspace(1.5, 1.6, 41)
    kept = workflow.anomaly_distance(g, kx, wavelength) >= .02-1e-12
    assert wavelength[~kept].round(4).tolist() == [round(1.5+.0025*k, 4) for k in range(13)]
    assert workflow.anomaly_distance(g, 0., wavelength).min() > .05     # normal incidence: 1.444 um in the substrate, 2 um in air


def test_open_close_gives_three_pixel_lines_and_gaps_across_the_period():
    pattern = np.zeros(100, int)
    pattern[[10, 11]] = 1                  # a two-pixel line, removed by the opening
    pattern[20:30] = 1
    pattern[31:40] = 1                     # a one-pixel gap, filled by the closing
    pattern[:2] = pattern[-2:] = 1         # a four-pixel line across the periodic seam
    processed = workflow.open_close(pattern, 3)[:, 0].astype(int)
    expected = np.zeros(100, int)
    expected[20:40] = 1
    expected[:2] = expected[-2:] = 1
    assert processed.tolist() == expected.tolist()
    sizes = workflow.measure_feature_sizes(processed[:, None], .02, boundary=('periodic', 'extend'))
    assert not sizes.violations(min_linewidth_um=.06, min_gap_um=.06)


def test_threshold_selection_takes_the_smallest_shift_that_meets_the_rule(tmp_path):
    def write(directory, shift, seed, violations, with_open_close=True):
        run = dict(threshold_shift=shift, seed=seed, violations=violations, record={}, environment=dict(commit='c'))
        if with_open_close:
            run['open_close'] = dict(violations=[], changed_pixels=0, environment=dict(commit='d'))
        workflow.write_json(directory/f'selection-d{shift:g}-seed{seed}.json', run)
    write(tmp_path, .1, 11, [], with_open_close=False)
    selection = workflow.select_decide(tmp_path)
    assert selection['chosen_threshold_shift'] is None and selection['pending'] == [.1]
    write(tmp_path, .1, 12, ['min_gap'])   # the first violating design rejects 0.1; seed 13 need not run
    write(tmp_path, .2, 11, [])
    write(tmp_path, .2, 12, [])
    selection = workflow.select_decide(tmp_path)
    assert selection['chosen_threshold_shift'] is None and selection['pending'] == [.2] and selection['rejected'] == [.1]
    write(tmp_path, .2, 13, [])
    selection = workflow.select_decide(tmp_path)
    assert selection['chosen_threshold_shift'] == .2 and not selection['chosen_by_fallback'] and len(selection['runs']) == 5
    assert selection['development_seeds'] == [11, 12, 13] and not selection['judged_seeds_used'] and selection['commits'] == ['c', 'd']
    case, _, _ = workflow.declared()
    settings = workflow.Settings.declared(case, selection)
    assert settings.threshold_shift == .2 and settings.open_close
    rejected = tmp_path/'rejected'
    for shift in (.1, .2, .25):
        write(rejected, shift, 11, ['min_linewidth'])
    selection = workflow.select_decide(rejected)
    assert selection['chosen_threshold_shift'] == .25 and selection['chosen_by_fallback'] and selection['rejected'] == [.1, .2, .25]
    with pytest.raises(ValueError, match='not a development seed'):
        workflow.select_run(.1, 2, tmp_path)


def test_te_decomposition_matches_the_fixture_routine(fixture):
    _, g = fixture
    spec = importlib.util.spec_from_file_location('metagrating_compare', ROOT/'examples'/'meep_comparison'/'metagrating'/'compare.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    rng = np.random.default_rng(7)
    x = -.99+.02*np.arange(100)
    wavelength = np.array([1.5, 1.55, 1.6])
    ez, hx = (rng.normal(size=(3, 100))+1j*rng.normal(size=(3, 100)) for _ in range(2))
    fields = np.zeros((3, 100, 6), complex)
    fields[..., 2], fields[..., 3] = ez, hx
    for index, name in ((g['substrate_index'], 'reflection'), (1., 'transmission')):
        forward, backward, _, _ = compare.order_powers(x, ez, hx, wavelength, g['period_um'], index, list(workflow.ORDERS))
        plus, minus, factor, _ = workflow.order_branches(fake_plane(x, fields), wavelength, index, 0., g['period_um'], 'TE', workflow.ORDERS)
        np.testing.assert_allclose(.5*(factor*plus.abs()**2).numpy(), forward, rtol=1e-12, err_msg=name)
        np.testing.assert_allclose(.5*(factor*minus.abs()**2).numpy(), backward, rtol=1e-12, err_msg=name)


@pytest.mark.parametrize('polarization', ['TE', 'TM'])
def test_decomposition_separates_known_waves_at_the_bloch_wavevector(fixture, polarization):
    _, g = fixture
    n, wavelength, kx0 = g['substrate_index'], np.array([1.52]), workflow.bloch_kx(g)
    x = -.99+.02*np.arange(100)
    orders = (-2, -1, 0, 1)
    k0, kx, ky, propagating = workflow.order_wavevectors(wavelength, n, kx0, g['period_um'], orders)
    assert propagating.all()
    up, down = np.array([.3+.1j, -.2j, 1., .05]), np.array([.1, .4-.3j, .2j, -.6])
    fields = np.zeros((1, 100, 6), complex)
    for j in range(len(orders)):
        wave = np.exp(1j*kx[j].item()*x)
        ratio = ky[0, j].item()/k0[0, 0].item()
        if polarization == 'TE':        # Hx = +-(k_y/k_0) Ez
            fields[0, :, 2] += (up[j]+down[j])*wave
            fields[0, :, 3] += ratio*(up[j]-down[j])*wave
        else:                           # Ex = -+(k_y/(k_0 n^2)) Hz
            fields[0, :, 5] += (up[j]+down[j])*wave
            fields[0, :, 0] += -ratio/n**2*(up[j]-down[j])*wave
    plus, minus, _, _ = workflow.order_branches(fake_plane(x, fields), wavelength, n, kx0, g['period_um'], polarization, orders)
    np.testing.assert_allclose(plus[0].numpy(), up, atol=1e-12)
    np.testing.assert_allclose(minus[0].numpy(), down, atol=1e-12)


@pytest.mark.parametrize('polarization,incidence', [('TE', 'normal'), ('TM', 'normal'), ('TE', 'bloch'), ('TM', 'bloch')])
def test_bare_substrate_matches_fresnel(fixture, polarization, incidence):
    _, g = fixture
    evaluator = workflow.CaseEvaluator(g, small(evaluation_time_fs=168., band_points=3), .04, polarization, incidence)
    d = evaluator.diagnostics
    assert d['max_abs_R0_minus_fresnel'] < 3e-3 and d['max_abs_T0_minus_fresnel'] < 3e-3, d
    assert d['max_other_order_power_over_incident'] < 1e-8, d
    e = evaluator.evaluate(np.zeros(workflow.pixel_count(g)))
    assert e['max_abs_energy_residual'] < 3e-3 and e['propagating_orders']['R'] == ([-2, -1, 0, 1] if incidence == 'bloch' else [-1, 0, 1])


def test_objective_gradient_matches_central_differences(fixture):
    _, g = fixture
    objective = workflow.TransmissionObjective(g, small())
    rho = (.2+.6*torch.rand((workflow.pixel_count(g), 1), generator=torch.Generator().manual_seed(3))).requires_grad_(True)
    loss, metrics = objective(rho)
    assert set(metrics) == {'T+1 1.50 um', 'T+1 1.55 um', 'T+1 1.60 um', 'T-1 mean', 'T+0 mean', 'T+1 mean'}
    loss.backward()
    h = 1e-2
    for k in (3, 30, 71):
        with torch.no_grad():
            plus, minus = rho.detach().clone(), rho.detach().clone()
            plus[k] += h
            minus[k] -= h
            fd = (objective(plus)[0].item()-objective(minus)[0].item())/(2*h)
        assert abs(rho.grad[k, 0].item()-fd) <= 1e-2*abs(fd)+1e-7, (k, rho.grad[k, 0].item(), fd)


def uniform_film(g, wavelength, kx, polarization):
    """Power reflectance of a uniform silicon layer between the substrate and air (Airy sum)."""
    k0 = 2*np.pi/wavelength
    eps = [g['substrate_index']**2, g['ridge_index']**2, 1.]
    kz = [np.sqrt(e*k0**2-kx**2+0j) for e in eps]
    q = kz if polarization == 'TE' else [k/e for k, e in zip(kz, eps)]
    r12, r23 = (q[0]-q[1])/(q[0]+q[1]), (q[1]-q[2])/(q[1]+q[2])
    phase = np.exp(2j*kz[1]*g['ridge_height_um'])
    return abs((r12+r23*phase)/(1+r12*r23*phase))**2


def rcwa_available():
    try:
        return subprocess.run([RCWA_PYTHON, '-c', 'import torcwa'], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def test_rcwa_script_reproduces_a_uniform_film(fixture, tmp_path):
    if not rcwa_available():
        pytest.skip(f'TORCWA interpreter {RCWA_PYTHON} not found (set TORCHFDTD_RCWA_PYTHON)')
    _, g = fixture
    settings = small(band_points=2, rcwa_harmonics=(3, 5), rcwa_device='auto')
    job = workflow.rcwa_job(g, settings, dict(film=[1]*workflow.pixel_count(g)))
    result = workflow.run_rcwa(job, RCWA_PYTHON, tmp_path)
    assert result['package']['torcwa'] == '0.1.4.2'
    assert len(result['cases']) == 4
    for c in result['cases']:
        kx = c['kx_per_um']
        expected = uniform_film(g, np.asarray(c['wavelengths_um']), kx, c['polarization'])
        np.testing.assert_allclose(c['R']['0'], expected, atol=1e-8, err_msg=f"{c['polarization']} {c['incidence']}")
        assert c['max_abs_total_minus_one'] < 1e-8
        assert all(v < 1e-12 for m in c['T'] if m != '0' for v in c['T'][m])


def load_records(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(directory.glob('seed*.json'))]
    baseline = json.loads((directory/workflow.BASELINE).read_text(encoding='utf-8')) if (directory/workflow.BASELINE).exists() else None
    return summary, records, baseline


def rejudge(directory):
    """Recompute every criterion from the seed records and check it against the summary."""
    summary, records, baseline = load_records(directory)
    previous = workflow.CASE_PATH
    try:
        workflow.CASE_PATH = ROOT/summary['provenance']['case']
        case, g, provenance = workflow.declared()
    finally:
        workflow.CASE_PATH = previous
    assert summary['provenance'] == provenance, 'the records were judged against another case file or geometry'
    settings = workflow.Settings(**{k: tuple(v) if isinstance(v, list) else v for k, v in summary['settings'].items()})
    criteria = json.loads(json.dumps(workflow.judge(records, case, settings, g, baseline)))
    assert criteria == summary['criteria']
    assert set(criteria) == set(CRITERIA)
    return summary, records, criteria


def test_short_run_writes_complete_records(tmp_path):
    settings = small(band_points=3, open_close=True, threshold_shift=.2)
    workflow.run(settings, tmp_path, rcwa_python=RCWA_PYTHON, skip_rcwa=True)
    summary, records, criteria = rejudge(tmp_path)
    assert [r['seed'] for r in records] == [1]
    record = records[0]
    assert len(record['history']) == settings.iterations and len(record['binary']) == workflow.pixel_count(workflow.declared()[1])
    assert [h['beta'] for h in record['history']] == [8., 16., 32., 64.]
    assert all({'active realization', 'T+1 mean eroded', 'T+1 mean nominal', 'T+1 mean dilated'} <= set(h['metrics']) for h in record['history'])
    assert record['robust']['threshold_shift'] == .2 and set(record['robust']['feature_sizes']) == set(workflow.REALIZATIONS)
    assert record['robust']['thresholded']['nominal'] == record['open_close']['thresholded']
    baseline = json.loads((tmp_path/workflow.BASELINE).read_text(encoding='utf-8'))
    assert baseline['open_close']['changed_pixels'] == 0 and baseline['evaluation']['physical_time_fs'] == record['evaluations'][0]['physical_time_fs']
    performance = criteria['performance']
    assert performance['judged_path_baseline'] == baseline['band_mean_T_plus1'] == summary['baseline']['band_mean_T_plus1']
    assert performance['limit_min'] == max(.7706, baseline['band_mean_T_plus1'])
    assert len(record['evaluations']) == 8
    for e in record['evaluations']:
        assert len(e['wavelength_um']) == 3 and set(e['T']) == set(e['R']) == {str(m) for m in workflow.ALL_ORDERS}
        assert set(e['amplitudes']['t']) == set(e['amplitudes']['r']) == {'-1', '0', '1'}
    assert record['fabrication']['declared'] == dict(min_linewidth_um=.06, min_gap_um=.06, boundary=['periodic', 'extend'])
    assert record['fabrication']['thresholded']['declared']['perturbation_um'] == .02 and not record['fabrication']['violations']
    closed = record['open_close']
    assert closed['processed'] == record['binary'] and closed['changed_pixels'] == sum(a != b for a, b in zip(closed['thresholded'], closed['processed']))
    assert closed['band_mean_T_plus1']['processed'] == float(np.mean(workflow.find(record, settings.design_mesh_um, 'TE', 'normal')['T']['1']))
    # a development run is not the declared workflow: one seed and no TORCWA check fail (a) and (c)
    assert not criteria['all_seeds']['passed'] and not criteria['rcwa_agreement']['passed'] and not summary['all_passed']
    assert summary['rcwa'] == {'1': None} and len(summary['references']['1']) == 8
    assert all(math.isclose(e['physical_time_fs'], 56., rel_tol=1e-3) for e in record['evaluations'])
    # every record names the torchfdtd it imported (checkout or installed wheel) and the checkout commit
    for environment in (record['environment'], summary['environment']):
        assert Path(environment['torchfdtd']['file']).name == '__init__.py' and environment['torchfdtd']['version']
        assert 'commit' in environment and 'tracked_changes' in environment


def assert_every_criterion(directory):
    summary, records, criteria = rejudge(directory)
    report = {name: dict(value=c['value'], passed=c['passed']) for name, c in criteria.items()}
    assert summary['all_passed'] == all(c['passed'] for c in criteria.values())
    assert all(c['passed'] for c in criteria.values()), json.dumps(report, indent=1)


def test_two_ridge_fixture_reproduces_the_committed_native_record(fixture):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    _, g = fixture
    result = workflow.fixture_reproduction(g)
    assert result['steps'] == g['steps']
    assert result['max_abs_order_efficiency_difference'] < 1e-4, result


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G7_FULL=1 for the declared three-seed workflow and its TORCWA check (hours)')
def test_declared_workflow_meets_every_acceptance_criterion():
    directory = Path(RECORD) if RECORD else RECORDS
    # Run with the installed-wheel interpreter from outside the source checkout.
    # r4 fixes six steps and does not support replaying r3 checkpoints.
    script = str(ROOT/'examples/g7/metagrating/wheel_entry_r4.py')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    subprocess.run(LAUNCHER+[sys.executable, script, '--stage', 'all',
                            '--out', str(directory.resolve()), '--rcwa-python', RCWA_PYTHON],
                   cwd=Path(sys.executable).parent, env=environment, check=True)
    assert_every_criterion(directory)


def test_committed_records_meet_every_acceptance_criterion():
    if not (RECORDS/'summary.json').exists():
        pytest.skip('no committed G7-01 records under docs/validation/g7/G7-01')
    assert_every_criterion(RECORDS)
