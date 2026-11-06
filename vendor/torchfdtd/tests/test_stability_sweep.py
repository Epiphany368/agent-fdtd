"""Stability sweep: the recorded 20,000-step matrix, its rendered document, the validation warning
for dispersive media inside the PML, and short regressions of the sweep fixtures."""
import json
import math
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from benchmarks import stability_sweep as sweep
from torchfdtd import Project, Region, Structure, Source, Monitor, Material
from torchfdtd.server import create_app
from torchfdtd.stability_checks import dispersive_structures_in_pml, stability_warnings

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT/'docs'/'validation'/'cases'/'STABILITY_SWEEP.json').read_text(encoding='utf-8'))
RECORD_PATH = ROOT/'docs'/'validation'/'stability_sweep_3060.json'


@pytest.fixture(scope='module')
def record():
    return json.loads(RECORD_PATH.read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------------------------- the record
def test_record_covers_the_declared_matrix(record):
    ids = {row['id'] for row in record['rows']}
    assert ids == {spec['id'] for spec in sweep.ROWS}, sorted(ids ^ {spec['id'] for spec in sweep.ROWS})
    assert record['acceptance'] == CASE['acceptance']
    assert record['case']['declared_at_commit'] == CASE['declared_at_commit']
    assert not record['quick'] and record['spec']['steps'] == 20000 and record['spec']['sample_interval'] == 250
    for row in record['rows']:
        if row['expect'] == 'rejected':
            continue
        assert row['steps_completed'] == 20000, row['id']
        assert len(row['samples']) == 80, row['id']


def test_every_row_verdict_follows_from_its_samples_and_the_case_limits(record):
    limits = CASE['acceptance']
    for row in record['rows']:
        copy = {k: v for k, v in row.items() if k not in ('judgement', 'passed', 'rejected_as_declared')}
        again = sweep.verdict(copy, limits)
        assert again['passed'] == row['passed'], row['id']
        if row['expect'] == 'rejected':
            assert row['rejected_as_declared'] and 'geometric PML stability criterion' in row['error'], row['id']
            continue
        j = row['judgement']
        assert j['judged'] and j['state_norm'] is not None, row['id']
        for name in ('state_norm', 'interior_energy'):
            if j.get(name) is None:
                continue
            assert j[name]['growth_ratio'] <= limits['growth_ratio_max'], (row['id'], name, j[name])
            limit = limits['last_over_peak_max_closed'] if row['closed'] else limits['last_over_peak_max_open']
            assert j[name]['last_over_peak'] <= limit, (row['id'], name, j[name])
        assert not row['divergence_check_fired'] and row['error'] is None, (row['id'], row['error'])


def test_no_failing_row_is_undocumented(record):
    failing = [row['id'] for row in record['rows'] if not row['passed']]
    assert failing == record['failing_ids']
    for id in failing:
        assert any(id in finding for finding in record['findings']), f'{id} fails without a recorded finding'


def test_document_is_rendered_from_the_record(record):
    document = (ROOT/'docs'/'STABILITY_SWEEP.md').read_text(encoding='utf-8')
    assert document == sweep.render(record)
    for row in record['rows']:
        assert f"| {row['id']} |" in document


# ----------------------------------------------------------------------------------------------- the warning
def dispersive_project(mode='ade', crossing=True, dimension='2d'):
    r = Region(dimension=dimension, size=(2., 2., 2.) if dimension == '3d' else (2., 2., 1.), mesh=.1, pml_cells=4, steps=10,
               precision='float64', backend='cpu', material_sampling='yee', pml_dispersion=mode)
    slab = Structure(name='drude slab', center=(0, 0, 0), size=(.4, 100. if crossing else .8, 100. if crossing else .8), material='metal')
    return Project(region=r, materials=[Material(name='void', index=1), Material(name='metal', model='drude', plasma_rad_s=2e15, collision_rad_s=1e14)],
                   structures=[slab], sources=[Source(component='Ez', center=(-.5, 0, 0))], monitors=[Monitor(component='Ez', center=(.5, 0, 0))])


def pml_warnings(summary):
    return [w for w in summary['warnings'] if 'Dispersive material inside PML layers' in w]


def test_stability_checks_warn_for_dispersive_structures_inside_the_pml_and_name_them():
    p = dispersive_project()
    assert dispersive_structures_in_pml(p) == [('drude slab', ['y_min', 'y_max'])]
    warnings = stability_warnings(p)
    assert len(warnings) == 1
    assert 'drude slab (y_min, y_max)' in warnings[0] and 'pml_dispersion="absorber"' in warnings[0] and 'end the structure before the PML' in warnings[0]
    three = dispersive_project(dimension='3d')
    assert dispersive_structures_in_pml(three) == [('drude slab', ['y_min', 'y_max', 'z_min', 'z_max'])]
    assert len(stability_warnings(three)) == 1


def test_stability_checks_stay_silent_when_frozen_interior_or_nondispersive():
    assert stability_warnings(dispersive_project(mode='frozen')) == []
    assert stability_warnings(dispersive_project(crossing=False)) == []
    nondispersive = dispersive_project()
    nondispersive.materials[1] = Material(name='metal', index=3.5)
    assert dispersive_structures_in_pml(nondispersive) == [] and stability_warnings(nondispersive) == []
    disabled = dispersive_project()
    disabled.structures[0].enabled = False
    assert stability_warnings(disabled) == []


def test_api_validate_returns_the_warning_with_the_estimate_warnings(tmp_path):
    p = dispersive_project()
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/validate', json=p.model_dump())
        assert response.status_code == 200
        body = response.json()
        assert pml_warnings(body) == stability_warnings(p)
        assert any('trapezoidal ADE' in w for w in body['warnings'])   # the estimate's own warnings are kept
        assert not pml_warnings(client.post('/api/validate', json=dispersive_project(mode='frozen').model_dump()).json())
    app.state.pool.shutdown()


def test_cli_run_prints_the_warning(tmp_path, capsys):
    from torchfdtd.cli import main
    path = tmp_path/'project.json'
    path.write_text(dispersive_project().model_dump_json(), encoding='utf-8')
    main(['run', str(path), '--output', str(tmp_path/'out.npz')])
    printed = json.loads(capsys.readouterr().out)
    assert pml_warnings(printed) == stability_warnings(dispersive_project())
    assert printed['termination_reason'] == 'max_steps'


# ----------------------------------------------------------------------------------------------- short regressions
def spec(id):
    return next(s for s in sweep.ROWS if s['id'] == id)


@pytest.mark.parametrize('id,decay', [('drude-slab-ade-2d', 1e-3), ('lorentz-slab-frozen-2d', 1e-3), ('metal-slab-ade-2d', 1e-3),
                                      ('src-tfsf-2d', 1e-3), ('pec-2d', .5)])
def test_short_run_of_a_sweep_fixture_decays_after_the_source(id, decay):
    # 3000 steps (350 fs): the open rows have lost their energy through the CPML; the PEC row is still
    # draining through its two remaining CPML faces, so only its non-growth is checked tightly.
    out = sweep.verdict(sweep.execute(spec(id), 3000), CASE['acceptance'])
    assert out['passed'], (id, out['error'], out['judgement'])
    assert out['judgement']['state_norm']['last_over_peak'] < decay


def test_rejected_tensor_row_is_refused_by_the_admission_not_run():
    out = sweep.verdict(sweep.execute(spec('tensor-rejected-3d'), 100), CASE['acceptance'])
    assert out['passed'] and out['rejected_as_declared'] and out['samples'] == []


def test_capture_records_the_sample_of_a_fired_divergence_check():
    """A run that trips run_control keeps the sample that tripped it, so a divergence is recorded with its numbers."""
    p = sweep.project('2d', name='vacuum')
    p = p.model_copy(update=dict(region=p.region.model_copy(update=dict(steps=2000))))
    p.region.run_control.field_limit = 1e-9
    samples, info, error = sweep.run_simulation(p)
    assert error is not None and 'Field magnitude limit exceeded' in error and info['termination'] == 'raised'
    assert samples and samples[-1]['field_peak'] > 1e-9 and samples[-1]['interior_energy'] > 0


def test_interior_energy_excludes_the_pml_and_matches_the_state_norm_in_vacuum():
    from torchfdtd.run_control import StateDiagnostics
    from torchfdtd.boundaries import YeeGrid
    import fdtd
    import numpy as np
    r = sweep.project('2d', name='vacuum').region
    fdtd.set_backend('numpy')
    fdtd.backend.float = np.float64
    g = YeeGrid(r)
    g.inverse_permittivity[:] = 1.
    g.material_states = []
    g.incident_states = []
    g.E[...] = 1.
    diag = StateDiagnostics(g, fused=False)
    whole, _ = diag.measure()
    interior = sweep.interior_from_diagnostics(diag)
    cells = math.prod(r.shape)
    pml = 2*10*r.shape[1]+2*10*(r.shape[0]-20)
    assert whole == pytest.approx(3*cells)
    assert interior == pytest.approx(3*(cells-pml))
