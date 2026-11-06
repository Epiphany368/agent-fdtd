"""G7-05 record mechanics before the timed GPU acceptance run."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from examples.g7.cost.timing import (STAGES, StageTimes, autotuner_break_even,
                                     stage_regressions, summarize_fresh_processes)


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/'docs/validation/g7/G7-05'
SCENARIOS = ('g701-resident-broadband', 'g701-host-broadband',
             'g701-resident-sequential', 'g703-resident')


def records():
    return [dict(iterations=[{stage: float(1+10*process+iteration)
                              for stage in STAGES} for iteration in range(6)])
            for process in range(5)]


def test_declared_stages_and_cold_warm_medians():
    case = json.loads((ROOT/'docs/validation/cases/G7-05r2.json').read_text(encoding='utf-8'))
    assert tuple(case['fixture']['stages']) == STAGES
    summary = summarize_fresh_processes(records())
    assert summary['process_count'] == 5 and summary['iterations_per_process'] == 6
    cold, warm = summary['cold']['T_forward'], summary['warm']['T_forward']
    assert (cold['min'], cold['median'], cold['max']) == (1., 21., 41.)
    assert (warm['min'], warm['median'], warm['max']) == (4., 24., 44.)
    assert warm['samples'] == [4., 14., 24., 34., 44.]
    assert summary['full_iteration']['cold']['median'] == 7*21.
    assert summary['full_iteration']['warm']['median'] == 7*24.


def test_regression_threshold_and_invalid_records():
    baseline = summarize_fresh_processes(records())
    later = summarize_fresh_processes(records())
    later['warm']['T_forward']['median'] = 1.26*baseline['warm']['T_forward']['median']
    regressions = stage_regressions(baseline, later)
    assert [(r['temperature'], r['stage']) for r in regressions] == [('warm', 'T_forward')]
    later['warm']['T_forward']['median'] = 1.25*baseline['warm']['T_forward']['median']
    assert stage_regressions(baseline, later) == []
    with pytest.raises(ValueError, match='five fresh'):
        summarize_fresh_processes(records()[:4])
    broken = records()
    broken[0]['iterations'][0]['T_forward'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        summarize_fresh_processes(broken)


def test_cpu_stage_timer_reports_only_declared_stages():
    timer = StageTimes()
    assert timer.call('T_geometry', lambda: 42) == 42
    assert set(timer.report()) == set(STAGES)
    assert timer.report()['T_geometry'] >= 0
    with timer.stage('T_forward'):
        with pytest.raises(RuntimeError, match='double-count'):
            timer.call('T_monitor', lambda: None)
    with pytest.raises(KeyError):
        timer.call('T_unknown', lambda: None)


def test_autotuner_break_even_counts_full_iterations():
    assert autotuner_break_even(5., 1.25, 1.) == 20
    assert autotuner_break_even(5.1, 1.25, 1.) == 21
    assert autotuner_break_even(0., 1.25, 1.) == 1
    assert autotuner_break_even(4., 1., 1.) is None
    assert autotuner_break_even(4., 1., 1.1) is None
    assert autotuner_break_even(12., 80., 60.,
                                untuned_cold_seconds=100., tuned_cold_seconds=90.) == 2
    assert autotuner_break_even(5., 80., 60.,
                                untuned_cold_seconds=100., tuned_cold_seconds=120.) == 3
    with pytest.raises(ValueError, match='nonnegative'):
        autotuner_break_even(float('nan'), 1., .9)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_recorded_five_process_statistics_and_installation(scenario):
    folder = RECORDS/scenario
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    records = [json.loads((folder/f'process{k}.json').read_text(encoding='utf-8')) for k in range(1, 6)]
    assert summary['statistics'] == summarize_fresh_processes(records)
    assert summary['scenario'] == scenario and summary['backend'] == 'cuda'
    assert summary['reduced'] is False and summary['torchfdtd_import'] == 'installed'
    assert len(summary['wheel_sha256']) == 64
    assert summary['cpu_threads'] == {'omp': '2', 'mkl': '2'}
    assert 'site-packages' in summary['package_path']
    for record in records:
        assert record['execution_policy'] == summary['execution_policy']
        assert record['package_version'] == summary['package_version']
        assert record['package_path'] == summary['package_path']
        assert len(record['observations']) == 6
        assert all(math.isfinite(row[key]) for row in record['observations']
                   for key in ('objective', 'gradient_norm'))
    assert stage_regressions(summary['statistics'], summary['statistics']) == []
    later = json.loads(json.dumps(summary['statistics']))
    later['warm']['T_forward']['median'] *= 1.2501
    assert [r['stage'] for r in stage_regressions(summary['statistics'], later)] == ['T_forward']


def test_recorded_autotuner_and_break_even():
    record = json.loads((RECORDS/'autotuner.json').read_text(encoding='utf-8'))
    case = ROOT/'docs/validation/cases/G7-05r2.json'
    assert record['case_sha256'] == hashlib.sha256(case.read_bytes()).hexdigest()
    assert record['status'] == 'acceptance' and record['backend'] == 'cuda'
    assert record['reduced'] is False and record['tuner']['seconds'] > 0
    measured = record['measured_full_iterations']
    selected = measured['selected_storage']
    resident = json.loads((RECORDS/'g701-resident-broadband/summary.json').read_text(encoding='utf-8'))
    tuned = json.loads((RECORDS/f'g701-{selected}-broadband/summary.json').read_text(encoding='utf-8'))
    before, after = resident['statistics']['full_iteration'], tuned['statistics']['full_iteration']
    assert measured['untuned_resident_warm_seconds'] == before['warm']['median']
    assert measured['tuned_policy_warm_seconds'] == after['warm']['median']
    assert measured['break_even_iterations'] == autotuner_break_even(
        record['tuner']['seconds'], before['warm']['median'], after['warm']['median'],
        untuned_cold_seconds=before['cold']['median'], tuned_cold_seconds=after['cold']['median'])


def test_recorded_cost_payload_is_the_release_wheel():
    execution = json.loads((RECORDS/'execution.json').read_text(encoding='utf-8'))
    assert execution['gpu_lock'] == 'exclusive' and execution['cpu_threads'] == 2
    assert execution['installation']['payload_matches']
    assert execution['installation']['payload_files'] > 0
    for scenario in SCENARIOS:
        summary = json.loads((RECORDS/scenario/'summary.json').read_text(encoding='utf-8'))
        assert summary['wheel_sha256'] == execution['wheel_sha256']
        assert summary['package_version'] == execution['installation']['versions']['torchfdtd']
