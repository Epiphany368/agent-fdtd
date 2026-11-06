"""A sharing violation must not discard the last complete benchmark result."""
import json
from pathlib import Path

import pytest

from benchmarks import atomic_output
from benchmarks.cr_resume import write_json


def windows_error(code):
    error = PermissionError('injected file replacement error')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
def test_json_replace_retries_without_deleting_complete_record(tmp_path, monkeypatch, code):
    path = tmp_path/'record.json'
    write_json(path, {'stage': 'previous'})
    original = Path.replace
    calls, sleeps = [], []

    def replace(source, destination):
        calls.append(source)
        assert json.loads(path.read_text()) == {'stage': 'previous'}
        assert json.loads(source.read_text()) == {'stage': 'complete'}
        if len(calls) < 3:
            raise windows_error(code)
        return original(source, destination)

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(atomic_output.time, 'sleep', sleeps.append)
    write_json(path, {'stage': 'complete'})
    assert json.loads(path.read_text()) == {'stage': 'complete'}
    assert len(calls) == 3 and sleeps == [.01, .02]
    assert not path.with_suffix('.json.tmp').exists()


@pytest.mark.parametrize('code, attempts', [(5, 8), (112, 1), (None, 1)])
def test_replace_keeps_outputs_and_propagates_persistent_or_unrelated_errors(
        tmp_path, monkeypatch, code, attempts):
    path = tmp_path/'record.json'
    write_json(path, {'stage': 'previous'})
    error = windows_error(code)
    calls, sleeps = [], []

    def replace(source, destination):
        calls.append(source)
        raise error

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(atomic_output.time, 'sleep', sleeps.append)
    with pytest.raises(PermissionError) as raised:
        write_json(path, {'stage': 'complete'})
    assert raised.value is error and len(calls) == attempts
    assert len(sleeps) == attempts-1 and sum(sleeps) <= 1.27
    assert json.loads(path.read_text()) == {'stage': 'previous'}
    assert json.loads(path.with_suffix('.json.tmp').read_text()) == {'stage': 'complete'}
