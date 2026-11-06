"""Launcher/compact reporting checks with transport explicitly emulated."""
import json

import pytest
import torch.distributed as dist

from examples.distributed_pulse import aggregate_reports, main


@pytest.mark.parametrize('data_group', [None, 'emulated-data-group'])
def test_pulse_launcher_local_protocol_and_fourier_oracle(monkeypatch, tmp_path, data_group):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    for name, function in dict(init_process_group=lambda *args, **kwargs: None,
            destroy_process_group=lambda *args: None, is_initialized=lambda: True,
            get_rank=lambda group=None: 0, get_world_size=lambda group=None: 1,
            get_backend=lambda group=None: 'gloo', barrier=lambda **kwargs: None,
            all_reduce=lambda value, **kwargs: None,
            all_gather_object=lambda output, value, **kwargs: output.__setitem__(0, value)).items():
        monkeypatch.setattr(dist, name, function)
    if data_group is not None:
        import examples.distributed_pulse as launcher
        original = launcher.DistributedYeeDomain
        def domain(*args, group, **kwargs):
            assert group == data_group
            return original(*args, group=group, **kwargs)
        monkeypatch.setattr(launcher, 'DistributedYeeDomain', domain)
        monkeypatch.setattr(launcher, 'prepare_data_group', lambda *args: (
            __import__('torch').device('cpu'), dict(rank=0, device_uuid=None), data_group))
        def reduction(value, **kwargs):
            assert kwargs['group'] == data_group
        monkeypatch.setattr(dist, 'all_reduce', reduction)
    output = tmp_path/'result.json'
    main(['--shape','6','5','5','--steps','10','--warmup','1','--repetitions','2','--output',str(output)])
    result = json.loads(output.read_text())
    assert result['accuracy_passed'] is True
    assert len(result['ranks'][0]['measured_repetitions']) == 2
    assert result['torch_version']
    assert len(result['source_sha256']) == 2
    assert result['global_cells'] == 150
    assert result['hardware_scaling_verified'] is False
    assert result['physical_gpu_count'] == 0
    row = result['ranks'][0]
    assert row['objective'] == pytest.approx(row['oracle_objective'], rel=2e-5)
    assert row['uniform_epsilon_gradient'] == pytest.approx(row['oracle_uniform_epsilon_gradient'], rel=5e-5)
    assert result['forward_gcups'] == pytest.approx(1500/result['forward_seconds_max_rank']/1e9)
    assert row['peak_checkpoints'] <= 2
    assert not list(tmp_path.glob('*.tmp'))


def test_compact_aggregation_rejects_duplicate_gpu_and_uses_slowest_rank():
    rows = [dict(rank=rank, owned_cells=12, device_uuid=f'GPU-{rank}',
        forward_seconds=rank+1., backward_seconds=rank+2.,
        cuda_peak_allocated_bytes=10+rank, cuda_peak_reserved_bytes=20+rank,
        workspace_reservation_bytes=30+rank, objective=1., oracle_objective=1.,
        uniform_epsilon_gradient=.2, oracle_uniform_epsilon_gradient=.2) for rank in range(2)]
    result = aggregate_reports(rows, shape=(4,3,2), steps=10, backend='nccl', precision='float32')
    assert result['forward_gcups'] == 24*10/2/1e9
    assert result['max_rank_cuda_peak_allocated_bytes'] == 11
    assert result['physical_gpu_count'] == 2
    rows[1]['device_uuid'] = rows[0]['device_uuid']
    with pytest.raises(ValueError, match='UUID'):
        aggregate_reports(rows, shape=(4,3,2), steps=10, backend='nccl', precision='float32')


def test_failed_accuracy_aggregation_withholds_throughput():
    row = dict(rank=0, owned_cells=24, device_uuid=None, forward_seconds=1., backward_seconds=2.,
        cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0, workspace_reservation_bytes=100,
        objective=99., oracle_objective=1., uniform_epsilon_gradient=.2, oracle_uniform_epsilon_gradient=.2)
    result = aggregate_reports([row], shape=(4,3,2), steps=10, backend='gloo', precision='float32')
    assert result['accuracy_passed'] is False
    assert result['forward_gcups'] is None
    assert result['accuracy_checks'][0]['absolute_error'] == 98.


def test_cuda_bootstrap_collective_failure_and_valid_group(monkeypatch):
    from types import SimpleNamespace
    import torch
    from examples.distributed_pulse import prepare_data_group
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(dist, 'all_gather_object', lambda output, value: output.__setitem__(0, value))
    monkeypatch.setattr(dist, 'new_group', lambda **kwargs: calls.append(kwargs) or 'data-group')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: (_ for _ in ()).throw(RuntimeError('bad binding')))
    with pytest.raises(ValueError, match='Collective CUDA device admission'):
        prepare_data_group('nccl', 0)
    assert not calls
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(uuid='GPU-abc', name='fake'))
    monkeypatch.setattr('examples.distributed_pulse.subprocess.run', lambda *args, **kwargs: SimpleNamespace(stdout='GPU-abc\n'))
    device, identity, group = prepare_data_group('nccl', 0)
    assert str(device) == 'cuda:0' and group == 'data-group'
    assert calls[0]['backend'] == 'nccl'
    assert identity['device_uuid'] == 'abc'


@pytest.mark.parametrize('failure', [AssertionError, AttributeError])
def test_cuda_bootstrap_cpu_build_exceptions_are_collective(monkeypatch, failure):
    import torch
    from examples.distributed_pulse import prepare_data_group
    seen = []
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: (_ for _ in ()).throw(failure('CPU build')))
    def gather(output, value):
        seen.append(value)
        output[0] = value
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    monkeypatch.setattr(dist, 'new_group', lambda **kwargs: pytest.fail('NCCL must not be initialized'))
    with pytest.raises(ValueError, match='CPU build'):
        prepare_data_group('nccl', 0)
    assert seen[0]['error'] == 'CPU build'


@pytest.mark.parametrize('rank', [0, 1])
def test_result_write_failure_reaches_every_rank(monkeypatch, rank):
    import examples.distributed_pulse as launcher
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)
    def write(*args):
        assert rank == 0
        raise OSError('disk full')
    monkeypatch.setattr(launcher, 'atomic_json', write)
    def gather(output, value):
        assert value == ('OSError: disk full' if rank == 0 else None)
        output[:] = ['OSError: disk full', None]
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    with pytest.raises(RuntimeError, match='Collective result write failed: OSError: disk full'):
        launcher.collective_write_json('unused.json', {})
