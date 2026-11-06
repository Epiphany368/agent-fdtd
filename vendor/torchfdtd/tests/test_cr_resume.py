import hashlib
import json

import numpy as np
import pytest
import torch

from benchmarks.cr_resume import CaseJournal, load_gradient_record, write_json
from benchmarks.cr_spectral_objective import check_density_direction
from torchfdtd import spectral_pupil_response


def test_interrupted_cases_resume_without_detaching_gradient(tmp_path):
    x = torch.tensor([[.2, .7], [.6, .4]], dtype=torch.float64, requires_grad=True)
    path = tmp_path / 'cases.json'
    journal = CaseJournal(path, {'steps': 123})
    calls = []
    fail = [True]

    def case(index, d):
        def compute():
            calls.append((index, torch.is_grad_enabled()))
            if index == (1, 0) and fail[0]:
                raise InterruptedError('simulated interruption between completed cases')
            return (d.exp().sum() * (1 + index[0] + index[1])).expand(2, 4)
        return journal.evaluate(d, index, compute)

    cases = [[lambda d, w=w, r=r: case((w, r), d) for r in range(2)] for w in range(2)]
    with pytest.raises(InterruptedError):
        spectral_pupil_response(cases, x, [.3, .7])
    assert len(journal.record['cases']) == 2
    journal.close()
    # A leftover temporary file never replaces the last complete journal.
    path.with_suffix('.json.tmp').write_text('incomplete')
    journal = CaseJournal(path, {'steps': 123}, resume=True)
    fail[0] = False
    calls.clear()
    response = spectral_pupil_response(cases, x, [.3, .7])
    gradient, = torch.autograd.grad(response.sum(), x)
    assert journal.hits == 2 and journal.computed == 2
    assert sum(not grad for _, grad in calls) == 2
    assert sum(grad for _, grad in calls) == 4  # Every case replays a real graph.
    expected = 4 * (1.7 + 2.7) * x.exp().sum()
    expected_gradient, = torch.autograd.grad(expected, x)
    torch.testing.assert_close(response.sum(), expected)
    torch.testing.assert_close(gradient, expected_gradient)
    journal.close()


def test_directional_resume_keys_include_perturbed_density(tmp_path):
    x = torch.tensor([[.2, .4], [.6, .8]], dtype=torch.float64)
    gradient = 8 * x.exp()
    path = tmp_path / 'probes.json'
    journal = CaseJournal(path, {'contract': 'unchanged'})
    computed = [0]
    fail = [True]

    def objective(d):
        def compute():
            computed[0] += 1
            if computed[0] == 4 and fail[0]:
                raise InterruptedError()
            return d.exp().sum().expand(2, 4)
        return journal.evaluate(d, (0, 0), compute).sum()

    with pytest.raises(InterruptedError):
        check_density_direction(objective, x, gradient, [.002, .001, .0005])
    journal.close()
    journal = CaseJournal(path, {'contract': 'unchanged'}, resume=True)
    fail[0] = False
    result = check_density_direction(objective, x, gradient, [.002, .001, .0005])
    fresh = check_density_direction(lambda d: 8*d.exp().sum(), x, gradient, [.002, .001, .0005])
    assert result == fresh and result['passed']
    assert journal.hits == 3 and journal.computed == 4
    assert len(journal.record['cases']) == 7
    journal.close()


def test_journal_lock_mismatch_and_checksum_fail_closed(tmp_path):
    path = tmp_path / 'cases.json'
    contract = {'inputs': 'hash', 'source': 'hash', 'steps': 100, 'runtime': 'runtime'}
    journal = CaseJournal(path, contract)
    with pytest.raises(RuntimeError, match='Another process'):
        CaseJournal(path, contract, resume=True)
    with torch.no_grad():
        journal.evaluate(torch.ones(1), (0, 0), lambda: torch.ones(2, 4))
    journal.close()
    with pytest.raises(FileExistsError):
        CaseJournal(path, contract)
    for key in contract:
        with pytest.raises(ValueError, match='contract mismatch'):
            CaseJournal(path, dict(contract, **{key: 'changed'}), resume=True)
    data = json.loads(path.read_text())
    entry = next(iter(data['cases'].values()))
    entry['value'][0][0] = 2.
    write_json(path, data)
    journal = CaseJournal(path, contract, resume=True)
    with torch.no_grad(), pytest.raises(ValueError, match='checksum'):
        journal.evaluate(torch.ones(1), (0, 0), lambda: pytest.fail('Must not compute'))
    journal.close()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_invalid_response_not_persisted(tmp_path, bad):
    journal = CaseJournal(tmp_path/'cases.json', {})
    with torch.no_grad(), pytest.raises(ValueError, match='finite and nonnegative'):
        journal.evaluate(torch.ones(1), (0, 0), lambda: torch.full((2, 4), bad))
    assert not json.loads(journal.path.read_text())['cases']
    journal.close()


def test_saved_gradient_requires_matching_contract_and_artifact(tmp_path):
    output = tmp_path/'run.json'
    gradient_path = output.with_suffix('.gradient.npy')
    x = torch.tensor([[.3, .7]], dtype=torch.float64)
    gradient = x.exp()
    np.save(gradient_path, gradient.numpy(), allow_pickle=False)
    contract = {'input': 'hash', 'source': 'hash'}
    record = dict(restart_contract=contract, gradient_artifact=dict(
        file=gradient_path.name, sha256=hashlib.sha256(gradient_path.read_bytes()).hexdigest(),
        shape=list(x.shape), dtype=str(x.dtype), variable='relaxed density',
        objective='weighted_bits_per_pixel'))
    write_json(output, record)
    _, restored = load_gradient_record(output, contract, x)
    torch.testing.assert_close(restored, gradient, rtol=0, atol=0)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_gradient_record(output, {'input': 'other'}, x)
    with pytest.raises(ValueError, match='shape or dtype'):
        load_gradient_record(output, contract, x.float())
    np.save(gradient_path, -gradient.numpy(), allow_pickle=False)
    with pytest.raises(ValueError, match='checksum'):
        load_gradient_record(output, contract, x)
