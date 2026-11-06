"""G4-01: the platform report runs, the platform matrix lists only recorded platforms, and a platform's G4 run record passes."""
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'docs' / 'PLATFORM_MATRIX.md'
PLATFORMS = ROOT / 'docs' / 'validation' / 'platforms'
G4_RUNS = PLATFORMS / 'g4'
GATES = ROOT / 'docs' / 'validation' / 'completion_gates.json'
HOME_PATH = re.compile(r'(?i)/root/|/home/|/Users/|[A-Z]:\\Users\\')
COLUMNS = ['Platform id', 'GPU', 'Compute capability', 'Driver', 'CUDA runtime', 'torch', 'CuPy', 'Python', 'OS', 'Record', 'Verified by']
RECORD_FIELDS = ['record_version', 'recorded_at', 'os', 'python', 'torch', 'cupy', 'cuda_runtime', 'torch_cuda_available', 'driver', 'gpus']


def matrix_rows():
    lines = [line for line in MATRIX.read_text(encoding='utf-8').splitlines() if line.startswith('|')]
    header = [cell.strip() for cell in lines[0].strip('|').split('|')]
    assert header == COLUMNS, header
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        assert len(cells) == len(COLUMNS), line
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recorder = load_script('record_gate_evidence')
platform_g4 = load_script('record_platform_g4')


def test_platform_report_runs_and_records_every_field(tmp_path):
    output = tmp_path / 'probe.json'
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'platform_report.py'), '--id', 'probe', '--output', str(output)],
                               capture_output=True, text=True, encoding='utf-8', cwd=ROOT, timeout=300,
                               env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    assert completed.returncode == 0, completed.stderr
    assert str(output) in completed.stdout
    record = json.loads(output.read_text(encoding='utf-8'))
    assert record['platform_id'] == 'probe'
    for field in RECORD_FIELDS:
        assert field in record, field
    assert record['python'] == sys.version.split()[0]
    if record['torch_cuda_available']:
        assert record['gpus'] and all(re.fullmatch(r'\d+\.\d+', g['compute_capability']) for g in record['gpus'])
        assert all(g['total_memory_bytes'] > 0 and 0 <= g['free_memory_bytes_at_record'] <= g['total_memory_bytes'] for g in record['gpus'])
    else:
        assert record['gpus'] == []


def test_platform_report_replaces_the_home_directory(tmp_path):
    # The interpreter lies under the home directory the child sees, as a venv under ~ does on a Linux host.
    home = Path(sys.executable).parent.parent
    output = tmp_path / 'probe.json'
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'platform_report.py'), '--id', 'probe', '--output', str(output)],
                               capture_output=True, text=True, encoding='utf-8', cwd=ROOT, timeout=300,
                               env=dict(os.environ, HOME=str(home), USERPROFILE=str(home), PYTHONIOENCODING='utf-8'))
    assert completed.returncode == 0, completed.stderr
    assert str(output) in completed.stdout
    record = json.loads(output.read_text(encoding='utf-8'))
    assert record['python_executable'].startswith('<user home>'), record['python_executable']
    assert not [key for key, value in record.items() if str(home).lower() in str(value).lower()]


def test_matrix_rows_come_from_record_files_only():
    rows = matrix_rows()
    assert rows, 'the matrix lists no platform'
    ids = [row['Platform id'] for row in rows]
    assert len(ids) == len(set(ids))
    hardware = ['GPU', 'Compute capability', 'Driver', 'CUDA runtime', 'torch', 'CuPy', 'Python', 'OS']
    recorded = 0
    for row in rows:
        link = re.fullmatch(r'\[(?P<id>[^\]]+)\]\((?P<path>validation/platforms/[^)]+\.json)\)', row['Record'])
        if link is None:
            assert row['Record'] == 'not recorded', row['Record']
            assert all(row[column] == 'not recorded' for column in hardware), f"{row['Platform id']} lists values without a record"
            continue
        assert link['id'] == row['Platform id']
        path = ROOT / 'docs' / link['path']
        assert path.is_file(), f"{row['Platform id']} names a record that does not exist: {link['path']}"
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['platform_id'] == row['Platform id']
        assert len(record['gpus']) == 1, 'one GPU per row; several GPUs need several rows'
        gpu = record['gpus'][0]
        expected = {'GPU': gpu['name'], 'Compute capability': gpu['compute_capability'], 'Driver': record['driver'],
                    'CUDA runtime': record['cuda_runtime'], 'torch': record['torch'], 'CuPy': record['cupy'],
                    'Python': record['python'], 'OS': record['os']}
        for column, value in expected.items():
            assert row[column] == str(value), f"{row['Platform id']} {column}: matrix says {row[column]!r}, record says {value!r}"
        recorded += 1
    assert recorded >= 1, 'no platform row is backed by a record file'


def test_every_record_file_has_a_matrix_row():
    listed = {row['Platform id'] for row in matrix_rows()}
    for path in (ROOT / 'docs' / 'validation' / 'platforms').glob('*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['platform_id'] == path.stem
        assert path.stem in listed, f'{path.name} has no matrix row'


@pytest.mark.cuda
def test_local_record_matches_this_gpu_when_present():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    names = {json.loads(p.read_text(encoding='utf-8'))['gpus'][0]['name']
             for p in (ROOT / 'docs' / 'validation' / 'platforms').glob('*.json')}
    assert torch.cuda.get_device_name(0) in names, 'this GPU has no platform record; run scripts/platform_report.py'


def test_platform_g4_run_records_pass_every_g4_required_test():
    rows = {row['Platform id']: row for row in matrix_rows()}
    records = sorted(G4_RUNS.glob('*.json'))
    linked = {link for row in rows.values() for link in re.findall(r'\((validation/platforms/g4/[^)]+\.json)\)', row['Verified by'])}
    assert linked == {f'validation/platforms/g4/{path.name}' for path in records}, 'a G4 run record and the matrix links to it disagree'
    g4 = next(stage for stage in json.loads(GATES.read_text(encoding='utf-8'))['stages'] if stage['id'] == 'G4')['tasks']
    for path in records:
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['platform_id'] == path.stem and (PLATFORMS / f'{path.stem}.json').is_file()
        copies = sorted((G4_RUNS / path.stem).glob('*.xml'))
        assert {f'docs/validation/platforms/g4/{path.stem}/{copy.name}' for copy in copies} == {run['junit'] for run in record['runs']}
        for item in [path, *copies]:
            assert not HOME_PATH.search(item.read_text(encoding='utf-8')), f'{item.name} names a home directory'
        runs = {}
        for run in record['runs']:
            copy = ROOT / run['junit']
            assert recorder.file_sha256(copy) == run['junit_sha256'], f"{run['label']}: the JUnit copy is not the one recorded"
            parsed, _, unresolved = recorder.parse_junit(ROOT, copy)
            assert not unresolved, unresolved
            assert {key: parsed[key] if key == 'total' else len(parsed[key]) for key in run['counts']} == run['counts'], run['label']
            assert (parsed['failed'], parsed['errors'], parsed['skipped_reasons']) == (run['failed'], run['errors'], run['skipped']), run['label']
            assert run['exit_code'] == 0 and not parsed['failed'] and not parsed['errors'], f"{run['label']}: {parsed['failed'] + parsed['errors']}"
            classes = {test: platform_g4.skip_class(reason, run['command']) for test, reason in parsed['skipped_reasons'].items()}
            assert classes == run['skip_classes'], run['label']
            unexpected = [test for test, kind in classes.items() if kind is None]
            assert not unexpected, f"{run['label']} skipped {unexpected}"
            runs[run['label']] = (run['task'], parsed)
        assert [run for run in record['runs'] if run['task'] == 'suite' and 'run_suite.py gpu-nightly' in run['command']], 'no gpu-nightly run'
        for task in g4:
            recorded = record['required_tests'].get(task['id'], {})
            assert sorted(recorded) == sorted(task['required_tests']), f"{task['id']}: the record does not name every required test"
            judged = [label for label, (owner, _) in runs.items() if owner in (task['id'], 'suite')]
            assert any(runs[label][0] == task['id'] for label in judged), f"{path.stem} has no run of the {task['id']} command"
            for item, tests in recorded.items():
                assert tests, f'{item} names no test'
                for test, states in tests.items():
                    assert recorder.matches(item, test), test
                    assert states == {label: 'passed' for label in judged}, f'{test}: {states}'
                    assert all(test in runs[label][1]['passed'] for label in judged), test


def test_platform_g4_skip_classes():
    command = 'python scripts/run_suite.py gpu-nightly --junitxml=x.xml'
    assert platform_g4.skip_class('optional platform check: Requires installed licensed Lumerical', command) == 'optional platform check'
    assert platform_g4.skip_class('finest mesh: set TORCHFDTD_G3_FINE=1 (the recorded run does)', command) == 'opt-in, not set by the run: TORCHFDTD_G3_FINE'
    assert platform_g4.skip_class('finest mesh: set TORCHFDTD_G3_FINE=1', 'TORCHFDTD_G3_FINE=1 ' + command) is None
    assert platform_g4.skip_class('the recorded wheel is not on this host: D:/x.whl', command) == 'file of another host'
    for reason in ('CUDA unavailable', 'needs a few GiB of free RAM', 'fixture file missing', ''):
        assert platform_g4.skip_class(reason, command) is None, reason


def _write_junit(path, cases):
    def outcome(state):
        if state == 'failed':
            return '<failure message="boom">boom</failure>'
        return f'<skipped type="pytest.skip" message="{state[8:]}">{state[8:]}</skipped>' if state.startswith('skipped:') else ''
    body = ''.join(f'<testcase classname="{classname}" name="{name}" time="0.001">{outcome(state)}</testcase>' for classname, name, state in cases)
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" failures="0" errors="0" skipped="0" '
                    f'tests="{len(cases)}" time="0.01" timestamp="{datetime.datetime.now().astimezone().isoformat()}">{body}</testsuite></testsuites>',
                    encoding='utf-8')


def test_platform_g4_recorder_names_every_required_test_and_fails_a_failure(tmp_path):
    root = tmp_path / 'repo'
    (root / 'tests').mkdir(parents=True)
    (root / 'tests' / 'test_alpha.py').write_text('def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n', encoding='utf-8')
    (root / 'tests' / 'test_beta.py').write_text('def test_b():\n    assert True\n', encoding='utf-8')
    (root / 'docs' / 'validation' / 'platforms').mkdir(parents=True)
    (root / 'docs' / 'validation' / 'platforms' / 'lab.json').write_text('{"platform_id": "lab", "gpus": []}\n', encoding='utf-8')
    tasks = [dict(id='G4-01', required_tests=['tests/test_alpha.py']), dict(id='G4-02', required_tests=['tests/test_beta.py::test_b'])]
    (root / 'docs' / 'validation' / 'completion_gates.json').write_text(json.dumps(dict(stages=[dict(id='G4', tasks=tasks)])) + '\n', encoding='utf-8')
    (root / '.gitignore').write_text('__pycache__/\n', encoding='utf-8')
    (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding='utf-8')
    git = ['git', '-c', 'user.name=gate', '-c', 'user.email=gate@example.invalid', '-c', 'core.autocrlf=false', '-c', 'commit.gpgsign=false']
    for args in (['init', '-q'], ['add', '.'], ['commit', '-q', '-m', 'baseline']):
        subprocess.run([*git, *args], cwd=root, capture_output=True, check=True)
    alpha = [('tests.test_alpha', 'test_one', 'passed'), ('tests.test_alpha', 'test_two', 'passed')]
    beta = [('tests.test_beta', 'test_b', 'passed')]
    for name, cases in (('a.xml', alpha), ('b.xml', beta), ('suite.xml', alpha + beta)):
        _write_junit(tmp_path / name, cases)
    python = Path(sys.executable).as_posix()
    runs = [['--run', 'g4-01', 'G4-01', '0', str(tmp_path / 'a.xml'), f'{python} -m pytest -q tests/test_alpha.py'],
            ['--run', 'g4-02', 'G4-02', '0', str(tmp_path / 'b.xml'), f'{python} -m pytest -q tests/test_beta.py'],
            ['--run', 'gpu-nightly', 'suite', '0', str(tmp_path / 'suite.xml'), 'python scripts/run_suite.py gpu-nightly']]
    argv = ['--root', str(root), '--platform', 'lab', *[arg for run in runs for arg in run]]
    output = root / 'docs' / 'validation' / 'platforms' / 'g4' / 'lab.json'
    assert platform_g4.main(argv) == 0
    record = json.loads(output.read_text(encoding='utf-8'))
    assert record['all_passed'] and record['totals']['failed'] == 0 and record['totals']['passed'] == 6
    assert record['required_tests']['G4-01']['tests/test_alpha.py'] == {
        f'tests/test_alpha.py::{name}': {'g4-01': 'passed', 'gpu-nightly': 'passed'} for name in ('test_one', 'test_two')}
    assert record['required_tests']['G4-02'] == {'tests/test_beta.py::test_b': {'tests/test_beta.py::test_b': {'g4-02': 'passed', 'gpu-nightly': 'passed'}}}
    assert (root / record['runs'][2]['junit']).read_bytes() == (tmp_path / 'suite.xml').read_bytes()
    _write_junit(tmp_path / 'suite.xml', [alpha[0], ('tests.test_alpha', 'test_two', 'failed'), *beta])
    assert platform_g4.main(argv) == 1
    record = json.loads(output.read_text(encoding='utf-8'))
    assert not record['all_passed'] and record['runs'][2]['failed'] == ['tests/test_alpha.py::test_two']
    assert record['required_tests_not_passed'] == ['G4-01 tests/test_alpha.py::test_two (gpu-nightly: failed)']
    extra = [('tests.test_alpha', 'test_opt_in', 'skipped:set TORCHFDTD_X_FULL=1 for it'), ('tests.test_alpha', 'test_ram', 'skipped:needs RAM')]
    _write_junit(tmp_path / 'suite.xml', [*alpha, *beta, *extra])
    assert platform_g4.main(argv) == 1
    record = json.loads(output.read_text(encoding='utf-8'))
    assert record['runs'][2]['skip_classes'] == {'tests/test_alpha.py::test_opt_in': 'opt-in, not set by the run: TORCHFDTD_X_FULL',
                                                 'tests/test_alpha.py::test_ram': None}
    assert record['unexpected_skips'] == ['tests/test_alpha.py::test_ram'] and not record['required_tests_not_passed']
    with pytest.raises(SystemExit, match='missing: G4-02'):
        platform_g4.main(['--root', str(root), '--platform', 'lab', *runs[0], *runs[2]])
    _write_junit(tmp_path / 'suite.xml', [*alpha, *beta, ('tests.test_alpha', 'test_opt_in', f'skipped:set TORCHFDTD_X_FULL=1 in {Path.home()}')])
    with pytest.raises(SystemExit, match='names the home directory'):
        platform_g4.main(argv)
    (root / 'tests' / 'test_beta.py').write_text('def test_b():\n    assert False\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='dirty tree'):
        platform_g4.main(argv)
