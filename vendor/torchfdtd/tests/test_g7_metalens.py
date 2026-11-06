"""G7-02r2: the small finite metalens workflow of examples/g7/metalens/metalens_workflow.py.

The fixed declaration is docs/G7_WORKFLOWS.md (G7-02 and its revision) and docs/validation/cases/G7-02r2.json. The
fast tests run on the CPU and check the mechanics on the declared lens description and on a reduced five-ridge lens:
the three starts, the time rule's energy against the native StateDiagnostics history, the shape gradient of a ridge
width against central differences, the plane-forward lines against the native monitors, the side-lobe and width
extraction, the angular-spectrum propagation against a direct line and the record schema. The judged run of the
declared lens (float32 on CUDA, one to two hours on the RTX 3060, the installed wheel imported from a working
directory outside the checkout; judged() refuses records that name another torchfdtd) runs only with TORCHFDTD_G7_FULL=1;
TORCHFDTD_G7_RECORD=<dir> writes its records there (docs/validation/g7/G7-02 for the recorded evidence), and
TORCHFDTD_G7_LAUNCHER=<command prefix> runs its stages as parallel processes behind that prefix (a GPU lock command,
for example). Without the flag, test_recorded_run_meets_every_criterion re-judges the committed records and skips
when there are none. Every start is judged; none is selected or dropped.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.g7.metalens import metalens_workflow as wf

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT/'docs/validation/cases/G7-02r2.json').read_text(encoding='utf-8'))
FULL = os.environ.get('TORCHFDTD_G7_FULL') == '1'
RECORD = os.environ.get('TORCHFDTD_G7_RECORD')
LAUNCHER = os.environ.get('TORCHFDTD_G7_LAUNCHER')
RECORDED = ROOT/'docs'/'validation'/'g7'/'G7-02'


@pytest.fixture(autouse=True)
def few_threads():
    """The reduced grids are a few thousand cells; many Torch threads only contend on a shared host."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_declared_lens_and_limits_are_those_of_the_case():
    fixture, spec = CASE['fixture'], wf.declared_spec()
    assert len(spec['ridges']) == fixture['ridges'] and spec['pitch_um'] == fixture['pitch_um'] and spec['aperture_um'] == fixture['aperture_um']
    assert spec['ridge_height_um'] == fixture['height_um'] and spec['focal_length_um'] == fixture['focal_length_um']
    assert spec['mesh_um'] == fixture['mesh_um'] and spec['steps'] == spec['window_steps'] == 5200 and spec['pml_um'] == 1.
    assert spec['polarization'] == 'Ez' and spec['source']['waveform']['wavelength_um'] == fixture['wavelength_um']
    assert spec['monitors']['incident']['y_um'] == -7.9 and spec['monitors']['focal']['y_um'] == 5.6 and spec['monitors']['axis']['x_um'] == 0.
    text = json.dumps(CASE['acceptance'])
    for key, phrase in dict(efficiency_min='at least {:g}', mesh_efficiency='by at most {:g}', mesh_axis_peak_um='by at most {:g} um',
                            time_efficiency='by at most {:g}', fwhm_um='= {:g} um').items():
        assert phrase.format(wf.LIMITS[key]) in text, key
    assert f"{wf.LIMITS['propagation_relative_l2'] * 100:g} percent relative L2" in text
    assert wf.LIMITS['efficiency_min'] == CASE['baseline']['focusing_efficiency']
    rule = fixture['time_rule']
    assert spec['stop'] == wf.STOP_RULE == dict(threshold=1e-3, check_steps=20, cap_factor=10)
    assert 'below 1e-3 of its peak after the source has ended, checked every 20 steps, capped at 10 times the declared 303.6 fs window' in rule
    assert 'of the lens and of the bare cell' in rule and 'StateDiagnostics' in rule and abs(spec['run_time_fs'] - 303.56) < .01
    assert 'twice the stop time' in fixture['validation']['longer_time'] and 'doubling the stop time' in CASE['acceptance']['time']
    assert 'finds the 1e-3 stop' in fixture['refinement']['step_count']
    fine = wf.fine_spec(spec, CASE)
    assert fine['pml_cells'] == 80 and fine['steps'] == fine['window_steps'] == 10400 and abs(fine['run_time_fs'] - spec['run_time_fs']) < 1e-9
    # Ten fine-mesh windows exceed the Region limit, so the fine-mesh cap is 100000 steps (9.6 windows).
    assert wf.cap_steps(spec) == 52000 and wf.cap_steps(fine) == wf.MAX_STEPS == 100000
    project = wf.build(fine, backend='cpu', monitors='none')
    assert tuple(project.region.shape[:2]) == (1920, 1720) and project.region.steps == 10400
    assert all(project.region.pml_layers(axis, side) == 80 for axis in (0, 1) for side in (0, 1))


def test_three_starts_follow_the_library():
    library = wf.library_widths()
    design = [r['width_um'] for r in wf.declared_spec()['ridges']]
    starts = {name: wf.start_widths(design, library, name) for name in wf.STARTS}
    assert starts['library'] == design
    for name, step in (('wider', 1), ('narrower', -1)):
        for original, width in zip(design, starts[name]):
            i = library.index(original)
            assert width == (library[i + step] if 0 <= i + step < len(library) else original)
        assert starts[name] == starts[name][::-1], 'the declared lens is mirror symmetric and so are its starts'
    # Four ridges already have the widest library width; no ridge has the narrowest.
    assert sum(a == b for a, b in zip(design, starts['wider'])) == 4 and sum(a == b for a, b in zip(design, starts['narrower'])) == 0
    assert starts == {name: wf.start_widths(design, library, name) for name in wf.STARTS}


def test_stop_rule_uses_the_native_energy_measure():
    """find_stop's energy after every check equals the native StateDiagnostics norm, and the rule on the native history gives its stop."""
    from torchfdtd import RunControl, Simulation
    from torchfdtd.solver import voxelize
    spec = wf.reduced_spec()
    project = wf.native_project(spec, wf.cap_steps(spec), with_lens=True, backend='cpu', run_control=RunControl(check_interval=spec['stop']['check_steps']))
    native = [(row['step'], row['state_norm']) for row in Simulation(project).run().summary['diagnostics']]
    stop = wf.find_stop(spec, torch.as_tensor(voxelize(project)[0], dtype=torch.float32), backend='cpu', record=True)
    assert stop['reached'] and stop['steps'] < wf.cap_steps(spec)
    mine = dict(stop['energy'])
    peak = max(energy for _, energy in native)
    compared = [(step, energy) for step, energy in native if step in mine]
    assert len(compared) == len(mine) == stop['steps'] // spec['stop']['check_steps']
    assert max(abs(mine[step] - energy) for step, energy in compared) <= 1e-4 * peak
    running, native_stop = 0., None
    for step, energy in native:
        running = max(running, energy)
        if step >= stop['source_end_step'] and energy <= spec['stop']['threshold'] * running:
            native_stop = step
            break
    assert native_stop == stop['steps']


def test_ridge_width_gradient_matches_central_differences():
    """float64 on the CPU: the shape VJP of the width through the discrete adjoint against central differences of the forward."""
    spec = wf.reduced_spec()
    objective = wf.FocalObjective(spec, backend='cpu', precision='float64')
    design = [r['width_um'] for r in spec['ridges']]
    widths = torch.tensor(design, dtype=torch.float64) + torch.tensor([.013, -.021, .007, .017, -.011], dtype=torch.float64)
    _, gradient = objective.value_and_gradient(widths)
    assert bool((gradient != 0).all())
    step = 1e-4
    for ridge in {int(gradient.abs().argmax()), 1}:
        values = []
        for sign in (1., -1.):
            shifted = widths.clone()
            shifted[ridge] += sign * step
            with torch.no_grad():
                values.append(float(objective(shifted)))
        central = (values[0] - values[1]) / (2 * step)
        assert abs(float(gradient[ridge]) - central) <= 1e-4 * abs(central), (ridge, float(gradient[ridge]), central)


def test_plane_forward_lines_reproduce_the_native_monitors():
    """The validation path (DifferentiablePlaneSimulation) against the native Simulation on the staircase reduced lens."""
    from torchfdtd import Simulation
    from torchfdtd.solver import voxelize
    spec = wf.reduced_spec()
    lens_project = wf.tm.build_2d(spec, with_lens=True, backend='cpu')
    native = wf.line_observables(Simulation(lens_project).run(), Simulation(wf.tm.build_2d(spec, with_lens=False, backend='cpu')).run(), spec)
    lines = wf.LineModel(spec, backend='cpu')
    bare, _ = lines.bare()
    planes, _ = lines.run(torch.as_tensor(voxelize(lens_project)[0], dtype=torch.float32))
    plane = wf.line_observables(planes, bare, spec)
    k = wf.centre_index(spec)
    for name in ('focal', 'axis'):
        a, b = np.asarray(native[name]['intensity'])[k], np.asarray(plane[name]['intensity'])[k]
        assert np.max(np.abs(a - b)) <= 1e-4 * np.max(np.abs(a)), name
    a, b = wf.centre_row(native, spec), wf.centre_row(plane, spec)
    assert abs(a['efficiency'] - b['efficiency']) <= 1e-5 and abs(a['fwhm_um'] - b['fwhm_um']) <= 1e-5 * a['fwhm_um']


def test_width_and_side_lobe_extraction_on_a_closed_form_profile():
    x = np.arange(-6000, 6001) * 1e-3
    profile = np.sinc(x / .8) ** 2
    width, _, _ = wf.mc.fwhm(x, profile)
    assert abs(width - .8 * .885893) < 2e-3
    lobe = wf.side_lobe_ratio(x, profile)
    assert abs(lobe['ratio'] - .0471904) < 1e-4 and np.allclose(lobe['main_lobe_um'], [-.8, .8], atol=2e-3)
    # A peak shared by two samples (a symmetric focus between two cell centres) stays in the main lobe.
    shifted = np.sinc((x + 5e-4) / .8) ** 2 + np.sinc((x - 5e-4) / .8) ** 2
    assert abs(wf.side_lobe_ratio(x, shifted)['ratio'] - .0471904) < 1e-3


def test_non_finite_values_are_recorded_as_null():
    value = wf.finite_or_none(dict(a=[1., float('nan')], b=(float('inf'), 2), c=True, d=np.float64('nan')))
    assert value == dict(a=[1., None], b=[None, 2], c=True, d=None)


def test_angular_spectrum_reproduces_a_direct_focal_line_on_a_small_lens():
    """Aperture line widened to the interior, 0.05 um grid: the propagation against the direct focal line, within the declared 5 percent."""
    spec = wf.wide_line_spec(wf.reduced_spec(mesh=.05))
    lines = wf.LineModel(spec, backend='cpu')
    bare, _ = lines.bare()
    planes, _ = lines.lens([r['width_um'] for r in spec['ridges']])
    obs = wf.line_observables(planes, bare, spec)
    result = wf.angular_spectrum_check(planes, bare, obs, spec)
    assert result['focal_relative_l2'] <= wf.LIMITS['propagation_relative_l2'], result['focal_relative_l2']
    assert result['axis_relative_l2'] <= wf.LIMITS['propagation_relative_l2'], result['axis_relative_l2']
    assert len(result['propagated_focal_intensity']) == len(obs['focal']['x_um'])


def test_reduced_workflow_reports_every_start_and_criterion(tmp_path):
    """The whole workflow on the reduced lens with one iteration: every start and every criterion is recorded (the values have no meaning)."""
    spec = wf.reduced_spec(ridges=3, margin=1., run_time_fs=100.)
    summary, records = wf.run(spec=spec, backend='cpu', iterations=1, output_dir=tmp_path, log=lambda *_: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['common.json', 'start-library.json', 'start-narrower.json', 'start-wider.json', 'summary.json']
    assert json.loads((tmp_path/'summary.json').read_text(encoding='utf-8')) == json.loads(json.dumps(summary))
    assert summary['reduced'] and [r['id'] for r in summary['criteria']] == ['a', 'b', 'c', 'd', 'e', 'f', 'scope']
    assert all(isinstance(r['passed'], bool) for r in summary['criteria']) and summary['criteria'][0]['passed']
    assert summary['grids']['fine_mesh']['window_steps'] == 2 * summary['grids']['design_grid']['window_steps']
    assert summary['consistency']['efficiency_difference'] < 1e-5 and len(summary['gradient_check']['central_differences']) == 3
    common = json.loads((tmp_path/'common.json').read_text(encoding='utf-8'))
    assert common['native_shutoff']['termination_reason'] == 'decayed' and common['staircase_stop']['reached']
    for record in records:
        assert json.loads((tmp_path/f"start-{record['start']}.json").read_text(encoding='utf-8'))['final_widths_um'] == record['final_widths_um']
        assert len(record['history']) == 1 and np.allclose(record['initial_widths_um'], record['history'][0]['widths_um'], rtol=0, atol=1e-7)
        assert 0 < record['history'][0]['steps'] <= wf.cap_steps(spec)
        assert set(record['final_design']) == {'design_grid', 'fine_mesh', 'longer_time'} and set(record['lines_1550']) == set(record['final_design'])
        stops = record['final_stops']
        assert stops['longer_time']['lens_steps'] == min(2 * stops['design_grid']['lens_steps'], wf.cap_steps(spec))
        assert stops['longer_time']['bare_steps'] == min(2 * stops['design_grid']['bare_steps'], wf.cap_steps(spec))
        assert stops['design_grid']['bare_steps'] == record['bare_stops']['design_grid']['steps']
        assert set(record['checks']) == {'mesh_efficiency_change', 'mesh_axis_peak_change_um', 'time_efficiency_change', 'propagation_focal_relative_l2'}
        assert 'full_width_line' in record['propagation']
        assert record['provenance'] == {k: summary['environment'][k] for k in wf.PROVENANCE_KEYS}
        bounds = record['width_bounds_um']
        assert all(bounds[0] - 1e-7 <= w <= bounds[1] + 1e-7 for w in record['final_widths_um'])
    import torchfdtd
    assert common['environment']['torchfdtd_file'] == str(Path(torchfdtd.__file__).resolve()) and common['environment']['torchfdtd_version']
    assert (common['environment']['torchfdtd_import'] == 'checkout') == common['environment']['checkout_import']


def judged(summary, records):
    """Re-judge the records against the case and check the summary's verdicts; returns the criterion rows."""
    assert not summary['reduced'] and not any(r['reduced'] for r in records)
    assert sorted(r['start'] for r in records) == sorted(wf.STARTS), 'every declared start must be present'
    for record in (summary['environment'], *summary['stage_provenance'].values(), *(r['provenance'] for r in records)):
        assert record['torchfdtd_file'] and record['torchfdtd_version'] and record['commit'], 'every record names the torchfdtd and the commit that ran'
        assert record['torchfdtd_import'] == 'installed' and record['torchfdtd_wheel']['sha256'], 'the declared run imports the installed wheel'
    refinement = CASE['fixture']['refinement']
    for record in records:
        assert record['iterations'] == refinement['iterations'] == len(record['history'])
        assert record['stop_rule'] == wf.STOP_RULE and all(0 < h['steps'] <= wf.cap_steps(wf.declared_spec()) for h in record['history'])
        assert record['learning_rate_um'] == refinement['learning_rate_um']
        assert len(record['final_widths_um']) == CASE['fixture']['ridges']
    rejudged = wf.judge(records, CASE, wf.declared_spec())
    assert rejudged['criteria'] == json.loads(json.dumps(summary['criteria']))
    report = json.dumps([{k: r.get(k) for k in ('id', 'value', 'limit', 'passed')} for r in rejudged['criteria']], indent=1)
    print(report)
    failing = [r['id'] for r in rejudged['criteria'] if not r['passed']]
    assert not failing, f'criteria not met: {failing}\n{report}'
    return rejudged['criteria']


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G7_FULL=1 for the judged run of the declared lens (one to two hours on the RTX 3060 workstation)')
def test_declared_workflow_meets_every_acceptance_criterion(tmp_path):
    assert torch.cuda.is_available(), 'the declared G7-02r2 run is float32 on CUDA'
    directory = Path(RECORD) if RECORD else tmp_path
    summary, records = wf.run(spec=wf.declared_spec(), backend='cuda', output_dir=directory, launcher=LAUNCHER)
    judged(summary, records)


def test_recorded_run_meets_every_criterion():
    if not (RECORDED/'summary.json').exists():
        pytest.skip('no recorded G7-02r2 run under docs/validation/g7/G7-02')
    summary = json.loads((RECORDED/'summary.json').read_text(encoding='utf-8'))
    records = [json.loads((RECORDED/name).read_text(encoding='utf-8')) for name in summary['records']]
    judged(summary, records)
