import threading

import numpy as np
import pytest
import torch

from torchfdtd import (BatchCase, FieldMonitor, Simulation, SpectrumSettings,
                        plan_grouped_batch, run_grouped_batch)
from test_solver import small


def mixed():
    cases = []
    for i in range(6):
        p = small('cuda', '2d', 'float32' if i % 2 == 0 else 'float64')
        p.region.cuda_kernel = p.region.cuda_monitor_kernel = 'fused'
        p.region.material_sampling = 'yee'
        p.region.steps = 91 if i % 2 == 0 else 123
        p.region.snapshot_interval = 40 + i
        p.sources[0].amplitude = .7 + .1 * i
        p.monitors.append(FieldMonitor(id='spectrum', center=(.8, 0, 0), size=(0, 1, 1),
            spectrum=SpectrumSettings(sampling='frequency', frequency_points=3, apodization='none')))
        cases.append(BatchCase(f'mixed-{i}', p, {'original': i}))
    return cases


def test_plan_stable_groups_memory_splits_and_no_cuda_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('Planner touched CUDA'))
    cases = mixed()
    plan = plan_grouped_batch(cases, cohort_size=2)
    assert plan['group_count'] == 2
    assert [c['indices'] for c in plan['cohorts']] == [[0, 2], [4], [1, 3], [5]]
    limit = max(max(g['estimated_bytes']) for g in plan['groups'])
    tight = plan_grouped_batch(cases, cohort_size=6, memory_limit_bytes=limit)
    assert all(c['estimated_bytes'] <= limit for c in tight['cohorts'])
    assert tight['padding_cells'] == 0
    with pytest.raises(ValueError, match='one case exceeds'):
        plan_grouped_batch(cases, memory_limit_bytes=1)


@pytest.mark.parametrize('mutation', ['nodes', 'duration', 'boundary', 'precision', 'default_pml'])
def test_changed_topology_never_padded_or_shortened(mutation):
    a = mixed()[0].project
    b = a.model_copy(deep=True)
    if mutation == 'nodes': b.region.mesh = .08
    if mutation == 'duration': b.region.steps += 1
    if mutation == 'boundary': b.region.boundaries.y_min.kind = b.region.boundaries.y_max.kind = 'periodic'
    if mutation == 'precision': b.region.precision = 'float64'
    if mutation == 'default_pml': b.region.pml_cells = 6
    assert plan_grouped_batch([a, b])['group_count'] == 2


@pytest.mark.parametrize('mutation,match', [('cpu', 'CPU'), ('auto', 'fixed-duration'), ('complex', 'Bloch')])
def test_unsupported_cases_fail_in_preflight(mutation, match):
    cases = mixed()
    r = cases[-1].project.region
    if mutation == 'cpu': r.backend = 'cpu'
    if mutation == 'auto': r.run_control.auto_shutoff = True
    if mutation == 'complex':
        r.boundaries.y_min.kind = r.boundaries.y_max.kind = 'bloch'
        r.bloch_phase = (0, .2, 0)
    with pytest.raises(ValueError, match=match): plan_grouped_batch(cases)


def test_invalid_planning_arguments():
    with pytest.raises(ValueError, match='at least one'): plan_grouped_batch([])
    a = mixed()[0]
    with pytest.raises(ValueError, match='Duplicate'): plan_grouped_batch([a, a])
    with pytest.raises(ValueError, match='integer'): plan_grouped_batch([a], cohort_size=True)
    with pytest.raises(ValueError, match='finite'): plan_grouped_batch([a], memory_limit_bytes=float('nan'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_mixed_cuda_full_outputs_objectives_order_and_saved_results(tmp_path):
    pytest.importorskip('cupy')
    cases = mixed()
    before = [c.project.model_dump() for c in cases]
    reference = [Simulation(c.project).run() for c in cases]
    events = []
    report = run_grouped_batch(cases, cohort_size=2, output_dir=tmp_path/'run', keep_results=False,
        objective=lambda r: np.max(abs(r.signals)), progress=events.append)
    report.raise_for_errors()
    assert [i.id for i in report.items] == [c.id for c in cases]
    assert report.plan['group_count'] == 2
    assert report.plan['cohorts_started'] == 4
    assert events[0]['input_indices'] == [0, 2]
    for item, ref, case in zip(report.items, reference, cases):
        actual = item.load()
        for key in ('electric', 'magnetic', 'epsilon', 'signals', 'times', 'frames', 'frame_steps'):
            np.testing.assert_array_equal(getattr(actual, key), getattr(ref, key))
        np.testing.assert_array_equal(actual.frequency_fields[0]['fields'], ref.frequency_fields[0]['fields'])
        np.testing.assert_array_equal(actual.frequency_fields[0]['flux'], ref.frequency_fields[0]['flux'])
        assert item.parameters == case.parameters
        assert item.metrics['objective'] == float(np.max(abs(ref.signals)))
    assert before == [c.project.model_dump() for c in cases]
    with pytest.raises(FileExistsError): run_grouped_batch(cases, output_dir=tmp_path/'run')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cancellation_before_and_during_grouped_execution():
    pytest.importorskip('cupy')
    event = threading.Event()
    event.set()
    untouched = run_grouped_batch(mixed(), cancel=event)
    assert untouched.plan['cohorts_started'] == 0
    assert all(i.status == 'cancelled' and i.pid is None for i in untouched.items)
    event.clear()
    report = run_grouped_batch(mixed(), cohort_size=2, cancel=event, progress=lambda _: event.set())
    assert report.plan['cohorts_started'] == 1
    assert all(i.status == 'cancelled' for i in report.items)
    assert all(report.items[i].pid is None for i in [1, 3, 4, 5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_objective_failure_does_not_drop_other_groups():
    pytest.importorskip('cupy')
    def objective(result):
        return float('nan') if result.project.region.precision == 'float64' else 1.
    report = run_grouped_batch(mixed(), objective=objective)
    assert [i.status for i in report.items] == ['completed', 'failed'] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_effective_default_pml_layers_are_separate_cuda_groups():
    pytest.importorskip('cupy')
    cases = mixed()[::2]
    cases[1].project.region.pml_cells = 6
    references = [Simulation(c.project).run() for c in cases]
    report = run_grouped_batch(cases)
    assert report.plan['group_count'] == 2
    report.raise_for_errors()
    for item, ref in zip(report.items, references):
        np.testing.assert_array_equal(item.load().electric, ref.electric)
        np.testing.assert_array_equal(item.load().magnetic, ref.magnetic)
