"""The release-candidate re-recorder replays recorded commands on a temporary repository with a tiny test module."""
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
REAL_GATES = ROOT / 'docs' / 'validation' / 'completion_gates.json'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rerecord = load_script('rerecord_gates')
recorder = rerecord.recorder

# The module passes only when the environment variable of the recorded command prefix is set.
TEST_SOURCE = ('import os\n\n\ndef test_flag_from_the_recorded_prefix():\n'
               "    assert os.environ.get('RERECORD_FLAG') == '1'\n\n\ndef test_two():\n    assert True\n")


def junit_text():
    """A passing two-test report timestamped now, so it postdates the fixture repository's commits as a real run would."""
    return ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="2" failures="0" errors="0" skipped="0" '
            f'time="0.01" timestamp="{datetime.datetime.now().astimezone().isoformat()}">'
            '<testcase classname="tests.test_alpha" name="test_flag_from_the_recorded_prefix" time="0.001"></testcase>'
            '<testcase classname="tests.test_alpha" name="test_two" time="0.001"></testcase></testsuite></testsuites>')


def git(root, *args):
    return subprocess.run(['git', '-c', 'user.name=gate', '-c', 'user.email=gate@example.invalid', '-c', 'core.autocrlf=false',
                           '-c', 'commit.gpgsign=false', *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def task(task_id, title, required_tests=()):
    return dict(id=task_id, title=title, specification=title, required_by_current_plan=True, implementation_state='IMPLEMENTED',
                verification_state='NOT_RUN', owner=None, code_paths=[], planned_test_commands=[], actual_test_commands=[],
                required_tests=list(required_tests), evidence=[], blocker=None, scope_change_approval=None)


@pytest.fixture
def repo(tmp_path):
    """A committed miniature repository with one recorded task whose command carries a PowerShell environment prefix."""
    root = tmp_path / 'repo'
    (root / 'tests').mkdir(parents=True)
    (root / 'docs' / 'validation' / 'runs').mkdir(parents=True)
    (root / 'docs' / 'validation' / 'cases').mkdir()
    (root / 'docs' / 'validation' / 'platforms').mkdir()
    (root / 'docs' / 'validation' / 'platforms' / 'lab.json').write_text(json.dumps(dict(platform_id='lab', gpus=[])), encoding='utf-8')
    (root / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE, encoding='utf-8')
    (root / '.gitignore').write_text('.local/\n__pycache__/\n', encoding='utf-8')
    # The same pytest root as the real repository, so JUnit class names read tests.test_alpha.
    (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding='utf-8')
    case = root / 'docs' / 'validation' / 'cases' / 'G1-03.json'
    case.write_text(json.dumps(dict(kind='acceptance_case', seed=1, acceptance=dict(limit=1))), encoding='utf-8')
    real = json.loads(REAL_GATES.read_text(encoding='utf-8'))
    gates = {key: real[key] for key in ('schema_version', 'kind', 'implementation_states', 'verification_states',
                                        'required_evidence_fields', 'release_rule')}
    gates['adopted_commit'] = None
    gates['profiles'] = {'WORKSTATION': {'required_stages': ['G0', 'G1'], 'scope_status': 'TEST'}}
    gates['stages'] = [
        dict(id='G0', title='baseline', depends_on_stages=[], profile='WORKSTATION', priority='P0',
             tasks=[task('G0-01', 'no evidence yet')]),
        dict(id='G1', title='regressions', depends_on_stages=['G0'], profile='WORKSTATION', priority='P0',
             tasks=[task('G1-03', 'flag check', ['tests/test_alpha.py::test_flag_from_the_recorded_prefix']),
                    task('G1-04', 'recorded on a GPU host', ['tests/test_alpha.py::test_two'])]),
    ]
    (root / 'docs' / 'validation' / 'completion_gates.json').write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    git(root, 'init', '-q')
    git(root, 'add', '.')
    git(root, 'commit', '-q', '-m', 'baseline')
    junit = tmp_path / 'first.xml'
    junit.write_text(junit_text(), encoding='utf-8')
    command = "$env:RERECORD_FLAG='1'; python -m pytest -q -p no:cacheprovider tests/test_alpha.py --junitxml=D:/elsewhere/first.xml"
    assert recorder.main(['--root', str(root), '--task', 'G1-03', '--command', command, '--junit', str(junit), '--exit-code', '0',
                          '--fixture', str(case), '--scope', 'first run of the flag check']) == 0
    assert recorder.main(['--root', str(root), '--task', 'G1-04', '--command', 'python -m pytest -q tests/test_alpha.py::test_two',
                          '--junit', str(junit), '--exit-code', '0']) == 0
    # G1-04 pretends to have been recorded on a CUDA host.
    gates_path = root / 'docs' / 'validation' / 'completion_gates.json'
    run_id = task_of(root, 'G1-04')['evidence'][-1]
    evidence_path = root / 'docs' / 'validation' / 'runs' / run_id / 'evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    evidence['environment']['torch_cuda_available'] = True
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    assert gates_path.is_file()
    git(root, 'add', '.')
    git(root, 'commit', '-q', '-m', 'first evidence')
    return root


def gates_of(root):
    return json.loads((root / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))


def task_of(root, task_id):
    return next(t for s in gates_of(root)['stages'] for t in s['tasks'] if t['id'] == task_id)


def evidence_of(root, task_id, index=-1):
    run_id = task_of(root, task_id)['evidence'][index]
    return run_id, json.loads((root / 'docs' / 'validation' / 'runs' / run_id / 'evidence.json').read_text(encoding='utf-8'))


def test_parse_command_reads_powershell_and_posix_environment_prefixes():
    env, argv = rerecord.parse_command("$env:TORCHFDTD_G3_FINE='1'; D:/x/.venv/Scripts/python.exe -m pytest -q tests/test_a.py::TestG301 --junitxml=D:/x/j.xml")
    assert env == {'TORCHFDTD_G3_FINE': '1'}
    assert argv == ['D:/x/.venv/Scripts/python.exe', '-m', 'pytest', '-q', 'tests/test_a.py::TestG301', '--junitxml=D:/x/j.xml']
    env, argv = rerecord.parse_command('TORCHFDTD_G3_FULL=1 TORCHFDTD_G3_RECORD=docs/validation/g3 python -m pytest -q tests/test_b.py -k g3_04')
    assert env == {'TORCHFDTD_G3_FULL': '1', 'TORCHFDTD_G3_RECORD': 'docs/validation/g3'}
    assert argv[:3] == ['python', '-m', 'pytest'] and argv[-2:] == ['-k', 'g3_04']
    env, argv = rerecord.parse_command('$env:A="x y"; $env:B=2; python -m pytest')
    assert env == {'A': 'x y', 'B': '2'} and argv == ['python', '-m', 'pytest']
    env, argv = rerecord.parse_command(r'D:\x\.venv\Scripts\python.exe -m pytest "tests/test a.py"')
    assert argv == [r'D:\x\.venv\Scripts\python.exe', '-m', 'pytest', 'tests/test a.py']
    with pytest.raises(ValueError):
        rerecord.parse_command("$env:A='1'; ")


def test_prepare_replaces_the_interpreter_and_the_junit_path_and_keeps_the_prefix(tmp_path):
    env, argv = rerecord.prepare("$env:F='1'; python -m pytest -q tests/test_a.py --junitxml old.xml -k x", tmp_path, Path('/venv/python.exe'),
                                 tmp_path / 'new.xml', absolute_paths=False)
    assert env == {'F': '1'}
    assert argv[0] == '/venv/python.exe' and 'old.xml' not in argv and '--junitxml' not in argv
    assert argv[-1] == f'--junitxml={(tmp_path / "new.xml").as_posix()}' and argv[-3:-1] == ['-k', 'x']
    shown = rerecord.shown_command(env, argv)
    assert shown.startswith("$env:F='1'; ") and shown.endswith('new.xml')
    with pytest.raises(ValueError):
        rerecord.prepare('make check', tmp_path, Path('/venv/python.exe'), tmp_path / 'new.xml', absolute_paths=True)


@pytest.mark.parametrize('absolute_paths', [True, False])
def test_prepare_moves_paths_of_the_recording_checkout_into_the_replaying_one(tmp_path, absolute_paths):
    """A command recorded with absolute paths in one checkout, replayed from another (a worktree of the candidate),
    must run that checkout's tests and write its records there, not into the recording checkout."""
    old, new = tmp_path / 'old', tmp_path / 'new'
    for checkout in (old, new):
        (checkout / 'docs' / 'validation').mkdir(parents=True)
        (checkout / 'docs' / 'validation' / 'completion_gates.json').write_text('{}', encoding='utf-8')
        (checkout / 'tests').mkdir()
        (checkout / 'tests' / 'test_a.py').write_text('', encoding='utf-8')
    elsewhere = Path(tmp_path.anchor) / 'no-checkout-here' / 'x.txt'   # tmp_path itself may lie inside a checkout
    command = (f"$env:TORCHFDTD_G3_RECORD='{(old / 'docs/validation/g3').as_posix()}'; {(old / '.venv/python.exe').as_posix()} "
               f"-m pytest -q {(old / 'tests/test_a.py').as_posix()}::test_x {elsewhere.as_posix()} "
               f"--junitxml={(old / 'old.xml').as_posix()}")
    env, argv = rerecord.prepare(command, new, Path('/venv/python.exe'), new / 'new.xml', absolute_paths=absolute_paths)
    assert env == {'TORCHFDTD_G3_RECORD': (new / 'docs/validation/g3').as_posix()}
    assert f"{(new / 'tests/test_a.py').as_posix()}::test_x" in argv
    assert elsewhere.as_posix() in argv   # outside any checkout: unchanged
    assert not any(old.as_posix() in token for token in [*argv, *env.values()])
    env, argv = rerecord.prepare(command, old, Path('/venv/python.exe'), old / 'new.xml', absolute_paths=absolute_paths)
    assert f"{(old / 'tests/test_a.py').as_posix()}::test_x" in argv and env['TORCHFDTD_G3_RECORD'] == (old / 'docs/validation/g3').as_posix()


@pytest.mark.parametrize('absolute_paths', [True, False])
def test_prepare_relocates_a_removed_nested_checkout_using_recorded_test_sources(tmp_path, absolute_paths):
    outer, new = tmp_path / 'outer', tmp_path / 'new'
    for checkout in (outer, new):
        (checkout / rerecord.GATE_FILE).parent.mkdir(parents=True)
        (checkout / rerecord.GATE_FILE).write_text('{}', encoding='utf-8')
    old = outer / '.local/worktrees/removed'
    assert not old.exists()
    (new / 'tests').mkdir()
    (new / 'tests/test_a.py').write_text('', encoding='utf-8')
    unrelated = tmp_path / 'unrelated.json'
    command = (f"$env:TORCHFDTD_G3_RECORD='{(old / 'docs/validation/g3').as_posix()}'; python -m pytest "
               f"{(old / 'tests/test_a.py').as_posix()}::test_one {unrelated.as_posix()}")
    env, argv = rerecord.prepare(command, new, Path('/venv/python.exe'), new / 'new.xml', absolute_paths,
                                 test_sources={'tests/test_a.py': 'recorded-hash'})
    assert env['TORCHFDTD_G3_RECORD'] == (new / 'docs/validation/g3').as_posix()
    assert (new / 'tests/test_a.py').as_posix() + '::test_one' in argv
    assert unrelated.as_posix() in argv
    assert not any('/worktrees/removed' in token for token in [*argv, *env.values()])


@pytest.mark.parametrize('options', ['', '--exclusive', '--short --vram-mib 5000', '--vram-mib 5000'])
def test_prepare_replays_locked_pytest_with_the_wheel_interpreter_and_outer_lock(tmp_path, options):
    old, new = tmp_path / 'old', tmp_path / 'new'
    (new / 'tests').mkdir(parents=True)
    (new / 'tests/test_a.py').write_text('', encoding='utf-8')
    command = (f"$env:TORCHFDTD_G3_FULL='1'; python {(old / '.local/gpu_lock.py').as_posix()} {options} "
               f"{(old / '.venv/Scripts/python.exe').as_posix()} -m pytest -q "
               f"{(old / 'tests/test_a.py').as_posix()}::test_one --junitxml=old.xml")
    with pytest.raises(ValueError, match='requires --external-gpu-lock'):
        rerecord.prepare(command, new, Path('/wheel/python.exe'), new / 'new.xml', True)
    env, argv = rerecord.prepare(command, new, Path('/wheel/python.exe'), new / 'new.xml', True,
                                 test_sources={'tests/test_a.py': 'recorded-hash'}, external_gpu_lock=True)
    assert env == {'TORCHFDTD_G3_FULL': '1'}
    assert argv == ['/wheel/python.exe', '-m', 'pytest', '-q',
                    (new / 'tests/test_a.py').as_posix() + '::test_one', f'--junitxml={(new / "new.xml").as_posix()}']


@pytest.mark.parametrize('child', [
    '--unknown python -m pytest', '--vram-mib nope python -m pytest', '--vram-mib',
    '--exclusive --short python -m pytest', 'python another_script.py', 'not-python -m pytest',
])
def test_prepare_refuses_unknown_or_malformed_lock_launchers(tmp_path, child):
    with pytest.raises(ValueError):
        rerecord.prepare(f'python gpu_lock.py {child}', tmp_path, Path('/wheel/python.exe'),
                         tmp_path / 'new.xml', True, external_gpu_lock=True)


def test_command_preflight_fails_before_installation_or_any_replay(repo, tmp_path, monkeypatch, capsys):
    run_id, evidence = evidence_of(repo, 'G1-03')
    evidence['command'] = 'python -m pytest tests/test_missing.py'
    (repo / rerecord.RUNS_DIR / run_id / 'evidence.json').write_text(json.dumps(evidence), encoding='utf-8')
    wheel = tmp_path / 'fake.whl'
    wheel.write_bytes(b'not installed')
    monkeypatch.setattr(rerecord, 'create_rc_venv', lambda *args: pytest.fail('installed before preflight'))
    monkeypatch.setattr(rerecord, 'run_command', lambda *args: pytest.fail('ran before preflight'))
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03', '--wheel', str(wheel)]) == 2
    assert 'G1-03 command preflight failed: replayed test file is missing:' in capsys.readouterr().out
    assert task_of(repo, 'G1-03')['evidence'][-1] == run_id


def test_selection_skips_tasks_without_evidence_and_cuda_runs_on_a_host_without_a_device(repo):
    runs = repo / 'docs' / 'validation' / 'runs'
    # The fixture's G1-03 run stands for a CPU-only recording whatever host recorded it.
    for run_id in task_of(repo, 'G1-03')['evidence']:
        evidence_path = runs / run_id / 'evidence.json'
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        evidence['environment']['torch_cuda_available'] = False
        evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    rows = {row[1]['id']: row[4] for row in rerecord.select(gates_of(repo), repo, runs, cuda=False)}
    assert rows['G0-01'] == 'no evidence run to replay'
    assert rows['G1-03'] is None
    assert 'CUDA' in rows['G1-04']
    rows = {row[1]['id']: row[4] for row in rerecord.select(gates_of(repo), repo, runs, cuda=False, everything=True)}
    assert rows['G1-04'] is None and rows['G0-01'] == 'no evidence run to replay'
    rows = {row[1]['id']: row[4] for row in rerecord.select(gates_of(repo), repo, runs, cuda=True, stage='G1')}
    assert set(rows) == {'G1-03', 'G1-04'} and rows['G1-04'] is None
    with pytest.raises(SystemExit):
        rerecord.select(gates_of(repo), repo, runs, tasks=['G7-99'])
    (repo / 'tests' / 'test_alpha.py').unlink()
    rows = {row[1]['id']: row[4] for row in rerecord.select(gates_of(repo), repo, runs, cuda=True)}
    assert rows['G1-03'].startswith('recorded test sources missing')


def test_rerecord_replays_the_command_with_its_environment_prefix_and_appends_a_verified_run(repo, capsys):
    previous, first = evidence_of(repo, 'G1-03')
    code = rerecord.main(['--root', str(repo), '--tasks', 'G1-03'])
    out = capsys.readouterr().out
    assert code == 0, out
    new_id, evidence = evidence_of(repo, 'G1-03')
    assert new_id != previous and task_of(repo, 'G1-03')['evidence'] == [previous, new_id]
    assert evidence['verification_state_assigned'] == 'VERIFIED' and evidence['exit_code'] == 0
    assert evidence['test_results']['passed'] == ['tests/test_alpha.py::test_flag_from_the_recorded_prefix', 'tests/test_alpha.py::test_two']
    assert evidence['command'].startswith("$env:RERECORD_FLAG='1'; ") and Path(sys.executable).as_posix() in evidence['command']
    assert '--junitxml=' in evidence['command'] and 'D:/elsewhere/first.xml' not in evidence['command']
    assert evidence['fixture_path'] == first['fixture_path'] and evidence['fixture_sha256'] == first['fixture_sha256']
    commit = git(repo, 'rev-parse', 'HEAD')
    assert evidence['applicable_scope'] == f'first run of the flag check re-recorded on {commit[:12]} for the release candidate.'
    assert evidence['note'] == f'rerecord_gates.py replay of {previous}'
    assert evidence['dirty_source_manifest'] == []
    assert f'G1-03   {previous}' in out and new_id in out and 'VERIFIED' in out
    # A second re-record replaces the release-candidate note instead of chaining it.
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'second evidence')
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03']) == 0
    _, again = evidence_of(repo, 'G1-03')
    assert again['applicable_scope'].count('re-recorded on') == 1


def test_a_failing_replay_is_recorded_as_failed_and_the_exit_status_says_so(repo, capsys):
    (repo / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE.replace("== '1'", "== '2'"), encoding='utf-8')
    git(repo, 'commit', '-q', '-am', 'break the flag test')
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03']) == 1
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['verification_state_assigned'] == 'FAILED' and evidence['exit_code'] != 0
    assert task_of(repo, 'G1-03')['verification_state'] == 'FAILED'
    assert 'FAILED' in capsys.readouterr().out


def test_dirty_tree_is_refused_before_anything_runs(repo, capsys):
    (repo / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE + '\n', encoding='utf-8')
    before = task_of(repo, 'G1-03')['evidence']
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03']) == 2
    assert 'dirty' in capsys.readouterr().out and task_of(repo, 'G1-03')['evidence'] == before
    # The gate file and the runs directory are the recorder's own outputs and do not count.
    git(repo, 'checkout', '--', 'tests/test_alpha.py')
    (repo / 'docs' / 'validation' / 'runs' / 'scratch').mkdir()
    (repo / 'docs' / 'validation' / 'runs' / 'scratch' / 'x.txt').write_text('x', encoding='utf-8')
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03', '--dry-run']) == 0
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03']) == 0


def test_dry_run_prints_the_plan_and_records_nothing(repo, capsys):
    before = task_of(repo, 'G1-03')['evidence']
    assert rerecord.main(['--root', str(repo), '--dry-run', '--all']) == 0
    out = capsys.readouterr().out
    assert 'G1-03' in out and 'G1-04' in out and "$env:RERECORD_FLAG='1'; " in out and 'skip G0-01' in out
    assert task_of(repo, 'G1-03')['evidence'] == before and task_of(repo, 'G1-04')['evidence']
    assert rerecord.main(['--root', str(repo), '--dry-run', '--all', '--exclude', 'G1-04']) == 0
    out = capsys.readouterr().out
    assert '== G1-03' in out and '== G1-04' not in out


@pytest.mark.parametrize('wrapped', [False, True])
def test_wheel_option_runs_with_the_installed_interpreter_outside_the_tree_and_records_the_wheel_hash(repo, tmp_path, monkeypatch, wrapped):
    if wrapped:
        run_id, prior = evidence_of(repo, 'G1-03')
        prior['command'] = prior['command'].replace('python -m pytest', 'python gpu_lock.py --vram-mib 5000 python -m pytest')
        (repo / rerecord.RUNS_DIR / run_id / 'evidence.json').write_text(json.dumps(prior), encoding='utf-8')
    wheel = tmp_path / 'torchfdtd-9.9.9-py3-none-any.whl'
    wheel.write_bytes(b'not a real wheel')
    fake_python = tmp_path / 'rc-venv' / 'Scripts' / 'python.exe'
    calls = {}

    def fake_venv(root, wheel_path, base_python, torch_spec, torch_index, find_links, extras):
        calls['venv'] = dict(root=root, wheel=wheel_path, torch=torch_spec, extras=extras)
        return fake_python

    def fake_probe(python, root, cwd, env):
        calls['probe'] = dict(python=python, cwd=cwd)
        return Path('site-packages/torchfdtd/__init__.py')

    def fake_run(argv, env, cwd):
        calls['run'] = dict(argv=argv, env=env, cwd=Path(cwd))
        junit = next(token for token in argv if token.startswith('--junitxml='))[len('--junitxml='):]
        Path(junit).write_text(junit_text(), encoding='utf-8')
        return 0

    monkeypatch.setattr(rerecord, 'create_rc_venv', fake_venv)
    monkeypatch.setattr(rerecord, 'probe_installed_package', fake_probe)
    monkeypatch.setattr(rerecord, 'run_command', fake_run)
    monkeypatch.setattr(recorder, 'environment_of', lambda interpreter, root: dict(recorder.environment(), python_executable=str(interpreter)))
    monkeypatch.setenv('PYTHONPATH', str(repo))
    code = rerecord.main(['--root', str(repo), '--tasks', 'G1-03', '--wheel', str(wheel), '--torch', 'torch==2.10.0+cu126',
                         '--platform', 'lab', *(['--external-gpu-lock'] if wrapped else [])])
    assert code == 0
    assert calls['venv']['wheel'] == wheel.resolve() and calls['venv']['torch'] == 'torch==2.10.0+cu126'
    assert calls['probe']['python'] == fake_python
    argv, env, cwd = calls['run']['argv'], calls['run']['env'], calls['run']['cwd']
    assert argv[0] == fake_python.as_posix()
    assert argv[1:3] == ['-m', 'pytest'] and 'gpu_lock.py' not in argv
    assert (repo / '.local' / 'tmp' / 'rc-work') in cwd.parents and 'PYTHONPATH' not in env
    assert env['RERECORD_FLAG'] == '1' and env['TMP'] == str(repo / '.local' / 'tmp')
    test_arg = next(token for token in argv if token.endswith('test_alpha.py'))
    assert Path(test_arg).is_absolute() and Path(test_arg) == (repo / 'tests' / 'test_alpha.py').resolve()
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['package_or_wheel_sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert evidence['command'].startswith(f"$env:RERECORD_FLAG='1'; {fake_python.as_posix()}")
    assert evidence['environment']['python_executable'] == str(fake_python)
    assert evidence['verification_state_assigned'] == 'VERIFIED'
    assert evidence['platform_id'] == 'lab' and 'platform_id' not in evidence['null_reasons']


def test_generated_record_warning_does_not_replace_a_passing_verdict(repo, capsys, monkeypatch):
    original_run = rerecord.run_command

    def run_and_generate(argv, env, cwd):
        code = original_run(argv, env, cwd)
        (repo / 'docs/validation/generated.json').write_text('{"measured": 1}', encoding='utf-8')
        return code

    monkeypatch.setattr(rerecord, 'run_command', run_and_generate)
    assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03']) == 0
    assert 'VERIFIED (dirty: 1 paths)' in capsys.readouterr().out
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['verification_state_assigned'] == 'VERIFIED'
    assert evidence['dirty_guarded_paths'] == []
    assert [entry['path'] for entry in evidence['dirty_source_manifest']] == ['docs/validation/generated.json']


def test_wheel_dry_run_preserves_existing_work_files_and_junit(repo, tmp_path):
    wheel = tmp_path / 'fake.whl'
    wheel.write_bytes(b'not installed')
    work = repo / '.local/tmp/rc-work'
    work.mkdir(parents=True)
    sentinel = work / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')
    report = repo / '.local/tmp/junit/rerecord/G1-03.xml'
    report.parent.mkdir(parents=True)
    report.write_text('previous report', encoding='utf-8')
    for _ in range(2):
        assert rerecord.main(['--root', str(repo), '--tasks', 'G1-03', '--wheel', str(wheel), '--dry-run']) == 0
        assert sentinel.read_text(encoding='utf-8') == 'keep'
        assert report.read_text(encoding='utf-8') == 'previous report'


def test_recorder_refuses_a_platform_without_a_record_and_records_none_otherwise(repo, tmp_path):
    junit = tmp_path / 'again.xml'
    junit.write_text(junit_text(), encoding='utf-8')
    with pytest.raises(SystemExit):
        recorder.main(['--root', str(repo), '--task', 'G1-03', '--command', 'pytest', '--junit', str(junit), '--platform', 'nowhere'])
    assert recorder.main(['--root', str(repo), '--task', 'G1-03', '--command', 'pytest', '--junit', str(junit)]) == 0
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['platform_id'] is None and 'platform_id' in evidence['null_reasons']
