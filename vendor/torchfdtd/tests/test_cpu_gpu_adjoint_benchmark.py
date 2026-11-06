"""Full-path comparisons must validate every derivative and expose failures."""
import json

import pytest
import torch

from benchmarks.cpu_gpu_adjoint import check_agreement, main, summarize, trial_order


def test_comparison_rejects_a_missing_group_tail_corruption_and_nonfinite():
    reference = (torch.ones(300000), torch.tensor([1e-20]))
    assert len(check_agreement(reference, reference)) == 2
    with pytest.raises(AssertionError, match='group count'):
        check_agreement(reference[:1], reference)
    corrupt = reference[0].clone()
    corrupt[-1] = 2
    with pytest.raises(AssertionError, match='relative L2'):
        check_agreement((corrupt, reference[1]), reference)
    with pytest.raises(AssertionError, match='relative L2'):
        check_agreement((reference[0], reference[1]*2), reference)
    with pytest.raises(AssertionError, match='Nonfinite'):
        check_agreement((reference[0], torch.tensor([float('nan')])), reference)


def test_ratios_use_cpu_medians_and_keep_gpu_regressions():
    records = {n:[dict(seconds=t) for t in ts] for n, ts in
        dict(cpu_t6=[9, 1, 10], cpu_t12=[7, 8, 7], cuda_resident=[2, 3, 2], cuda_dram=[9, 10, 9]).items()}
    result = summarize(records, ['cpu_t6', 'cpu_t12'])
    assert result['fastest_tested_cpu'] == 'cpu_t12'
    assert result['speedup_over_fastest_tested_cpu']['cuda_resident'] == 3.5
    assert result['speedup_over_fastest_tested_cpu']['cuda_dram'] < 1
    names = list(records)
    assert all(sorted(trial_order(names, i)) == sorted(names) for i in range(4))
    assert trial_order(names, 0) != trial_order(names, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dispersive', [False, True])
def test_actual_cpu_resident_cuda_and_async_dram_driver(tmp_path, dispersive):
    args = ['--output', str(tmp_path/'run.json'), '--execute', '--smoke', '--size', '16',
            '--steps', '19', '--repeats', '1', '--cpu-threads', '1', '2', '--gpu-host-threads', '1',
            '--slab-width', '8', '--temporal-depth', '3', '--gpu-budget-gib', '.5', '--host-budget-gib', '1']
    previous = torch.get_num_threads()
    report = main(args + (['--dispersive'] if dispersive else []))
    assert report['stage'] == 'complete' and report['driver_smoke']
    assert torch.get_num_threads() == previous
    assert len(report['records']) == 4
    for name, rows in report['records'].items():
        assert len(rows) == 1 and rows[0]['seconds'] > 0
        assert len(rows[0]['agreement']) == (5 if dispersive else 2)
        assert rows[0]['sampled_peak_process_rss_bytes'] >= rows[0]['rss_before_bytes']
        assert rows[0]['solver']['execution_reservation']['plane_output_bytes'] > 0
        if name.startswith('cuda'):
            assert rows[0]['peak_torch_cuda_allocated_bytes'] > 0
    assert json.loads((tmp_path/'run.json').read_text())['stage'] == 'complete'


def test_non_smoke_and_host_denial_precede_execution(tmp_path, monkeypatch):
    import benchmarks.cpu_gpu_adjoint as driver
    with pytest.raises(ValueError, match='Non-smoke'):
        main(['--output', str(tmp_path/'invalid.json'), '--size', '16', '--steps', '19'])
    assert not (tmp_path/'invalid.json').exists()
    monkeypatch.setattr(driver, 'host_memory', lambda: dict(available_bytes=1))
    with pytest.raises(ValueError, match='headroom'):
        driver.require_headroom(2, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_failure_cannot_publish_a_speed_ratio(tmp_path, monkeypatch):
    import benchmarks.cpu_gpu_adjoint as driver
    def fail(*a, **kw):
        raise AssertionError('intentional full-gradient mismatch')
    monkeypatch.setattr(driver, 'check_agreement', fail)
    path = tmp_path/'failed.json'
    with pytest.raises(AssertionError, match='full-gradient'):
        main(['--output', str(path), '--execute', '--smoke', '--size', '16', '--steps', '16',
              '--repeats', '1', '--cpu-threads', '1', '--slab-width', '8', '--temporal-depth', '3'])
    record = json.loads(path.read_text())
    assert record['stage'] == 'failed' and record['failed_during'] == 'warmup'
    assert 'speedup_over_fastest_tested_cpu' not in record


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_budget_denial_and_dry_plan_do_not_execute_fields(tmp_path, monkeypatch):
    import benchmarks.cpu_gpu_adjoint as driver
    monkeypatch.setattr(driver, 'measure', lambda *a, **kw: pytest.fail('Ran denied/dry fields'))
    base = ['--smoke', '--size', '16', '--steps', '16', '--cpu-threads', '1',
            '--slab-width', '8', '--temporal-depth', '3']
    denied = tmp_path/'denied.json'
    with pytest.raises(ValueError, match='budget'):
        main(['--output', str(denied), '--execute', '--host-budget-gib', '.000001', *base])
    assert json.loads(denied.read_text())['stage'] == 'failed'
    plan = main(['--output', str(tmp_path/'plan.json'), *base])
    assert plan['stage'] == 'admitted_not_executed'
    assert len(plan['reservations']) == 3 and not plan['warmups']
    assert 'speedup_over_fastest_tested_cpu' not in plan
