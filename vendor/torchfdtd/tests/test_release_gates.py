"""Failure-injection tests of the completion-gate recorder and judge on a temporary repository."""
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


recorder = load_script('record_gate_evidence')
judge = load_script('check_release_gates')

TEST_SOURCE = 'def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n'


def git(root, *args):
    return subprocess.run(['git', '-c', 'user.name=gate', '-c', 'user.email=gate@example.invalid',
                           '-c', 'core.autocrlf=false', '-c', 'commit.gpgsign=false', *args],
                          cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def now_iso():
    return datetime.datetime.now().astimezone().isoformat()


def junit(path, cases, timestamp=None):
    """Write a pytest-shaped JUnit file; cases are (classname, name, outcome) with outcome pass/fail/skip/error.

    ``outcome`` may also be ``('skip', reason)`` to inject a specific skip reason. The suite timestamp is
    now unless given, so the run postdates the fixture repository's commits as a real run would.
    """
    rows = []
    for classname, name, outcome in cases:
        reason = 'injected skip'
        if isinstance(outcome, tuple):
            outcome, reason = outcome
        inner = {'pass': '', 'fail': '<failure message="injected failure">assert False</failure>',
                 'error': '<error message="injected error">boom</error>',
                 'skip': f'<skipped type="pytest.skip" message="{reason}">{reason}</skipped>'}[outcome]
        rows.append(f'<testcase classname="{classname}" name="{name}" time="0.001">{inner}</testcase>')
    counts = dict(tests=len(cases), failures=sum(o == 'fail' for _, _, o in cases),
                  errors=sum(o == 'error' for _, _, o in cases), skipped=sum(o == 'skip' for _, _, o in cases))
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
                    + ' '.join(f'{k}="{v}"' for k, v in counts.items())
                    + f' time="0.01" timestamp="{timestamp or now_iso()}">' + ''.join(rows)
                    + '</testsuite></testsuites>', encoding='utf-8')
    return path


@pytest.fixture
def repo(tmp_path):
    """A committed miniature repository with the real gate-file contract and two tasks."""
    root = tmp_path / 'repo'
    (root / 'tests').mkdir(parents=True)
    (root / 'docs' / 'validation' / 'runs').mkdir(parents=True)
    (root / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE, encoding='utf-8')
    real = json.loads(REAL_GATES.read_text(encoding='utf-8'))

    def task(task_id, title, required_tests=()):
        return dict(id=task_id, title=title, specification=title, required_by_current_plan=True,
                    implementation_state='IMPLEMENTED', verification_state='NOT_RUN', owner=None, code_paths=[],
                    planned_test_commands=[], actual_test_commands=[], required_tests=list(required_tests), evidence=[],
                    blocker=None, scope_change_approval=None)

    gates = {key: real[key] for key in ('schema_version', 'kind', 'profiles', 'implementation_states', 'verification_states',
                                        'required_evidence_fields', 'release_rule', 'proposed_thresholds')}
    gates['adopted_commit'] = None
    gates['profiles'] = {'WORKSTATION': {'required_stages': ['G0', 'G1'], 'scope_status': 'TEST'},
                         'HPC': {'required_stages': ['G0', 'G1', 'H1'], 'scope_status': 'TEST'}}
    gates['stages'] = [
        dict(id='G0', title='baseline', depends_on_stages=[], profile='WORKSTATION', priority='P0',
             tasks=[task('G0-01', 'baseline check')]),
        dict(id='G1', title='regressions', depends_on_stages=['G0'], profile='WORKSTATION', priority='P0',
             tasks=[task('G1-03', 'allocation scaling', ['tests/test_alpha.py::test_two']),
                    task('G1-05', 'whole file required', ['tests/test_alpha.py']),
                    dict(task('G1-06', 'reads data files', ['tests/test_alpha.py::test_one']), watch_paths=['data/*.json', 'data/notes.md'])]),
        dict(id='H1', title='hpc', depends_on_stages=['G1'], profile='HPC', priority='P1',
             tasks=[dict(task('H1-02', 'two gpu'), blocker='BLOCKED_EXTERNAL: no second GPU')]),
    ]
    gate_path = root / 'docs' / 'validation' / 'completion_gates.json'
    gate_path.write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    (root / 'data').mkdir()
    (root / 'data' / 'a.json').write_text('{"a": 1}\n', encoding='utf-8')
    (root / 'data' / 'notes.md').write_text('notes\n', encoding='utf-8')
    git(root, 'init', '-q')
    git(root, 'add', '.')
    git(root, 'commit', '-q', '-m', 'baseline')
    return root


def record(root, task_id, junit_path, command=None, **extra):
    argv = ['--root', str(root), '--task', task_id, '--command', command or f'pytest tests/test_alpha.py ({task_id})', '--junit', str(junit_path)]
    for key, value in extra.items():
        argv += [f'--{key.replace("_", "-")}', str(value)]
    return recorder.main(argv)


PYTEST_ALPHA = f'{Path(sys.executable).as_posix()} -m pytest -q -p no:cacheprovider tests/test_alpha.py'


def evidence_of(root, task_id, index=-1):
    run_id = task_of(root, task_id)['evidence'][index]
    run_dir = root / 'docs' / 'validation' / 'runs' / run_id
    return run_dir, json.loads((run_dir / 'evidence.json').read_text(encoding='utf-8'))


def rewrite_evidence(run_dir, evidence):
    (run_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')


def gates_of(root):
    return json.loads((root / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))


def task_of(root, task_id):
    return next(t for s in gates_of(root)['stages'] for t in s['tasks'] if t['id'] == task_id)


def judge_run(root, capsys, *args):
    code = judge.main(['--root', str(root), *args])
    return code, capsys.readouterr().out


def passing_junit(tmp_path):
    return junit(tmp_path / 'pass.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'pass')])


def test_passing_junit_records_verified_evidence_with_every_required_field(repo, tmp_path, capsys):
    assert record(repo, 'G1-03', passing_junit(tmp_path)) == 0
    task = task_of(repo, 'G1-03')
    assert task['verification_state'] == 'VERIFIED'
    assert len(task['evidence']) == 1 and task['actual_test_commands'] == ['pytest tests/test_alpha.py (G1-03)']
    run_dir = repo / 'docs' / 'validation' / 'runs' / task['evidence'][0]
    evidence = json.loads((run_dir / 'evidence.json').read_text(encoding='utf-8'))
    assert (run_dir / 'junit.xml').is_file()
    for field in gates_of(repo)['required_evidence_fields']:
        assert field in evidence, field
    for field in ('fixture_sha256', 'acceptance_criteria_sha256', 'package_or_wheel_sha256', 'seed', 'observed_metrics'):
        assert evidence[field] is None and field in evidence['null_reasons']
    assert evidence['source_commit'] == git(repo, 'rev-parse', 'HEAD')
    assert evidence['dirty_source_manifest'] == []
    assert evidence['test_source_sha256'] == {'tests/test_alpha.py': recorder.file_sha256(repo / 'tests' / 'test_alpha.py')}
    assert evidence['test_results']['passed'] == ['tests/test_alpha.py::test_one', 'tests/test_alpha.py::test_two']
    assert evidence['skipped_required_tests'] == [] and evidence['exit_code'] == 0
    assert evidence['environment']['python'] and 'cpu_model' in evidence['hardware']
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'G1-03 passes' in out


def test_injected_failure_marks_failed_and_judge_rejects(repo, tmp_path, capsys):
    failing = junit(tmp_path / 'fail.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'fail')])
    record(repo, 'G1-03', failing)
    assert task_of(repo, 'G1-03')['verification_state'] == 'FAILED'
    assert '-> FAILED' in capsys.readouterr().out
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'verification_state is FAILED' in out and 'NOT RELEASABLE' in out


def test_injected_error_marks_failed(repo, tmp_path):
    erroring = junit(tmp_path / 'error.xml', [('tests.test_alpha', 'test_one', 'error'), ('tests.test_alpha', 'test_two', 'pass')])
    record(repo, 'G1-03', erroring)
    assert task_of(repo, 'G1-03')['verification_state'] == 'FAILED'


def test_skipped_required_test_stays_not_run_and_judge_exits_nonzero(repo, tmp_path, capsys):
    skipping = junit(tmp_path / 'skip.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'skip')])
    record(repo, 'G1-03', skipping)
    out = capsys.readouterr().out
    task = task_of(repo, 'G1-03')
    assert task['verification_state'] == 'NOT_RUN' and len(task['evidence']) == 1
    assert 'required tests were skipped: tests/test_alpha.py::test_two' in out
    evidence = json.loads((repo / 'docs' / 'validation' / 'runs' / task['evidence'][0] / 'evidence.json').read_text(encoding='utf-8'))
    assert evidence['skipped_required_tests'] == ['tests/test_alpha.py::test_two']
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'verification_state is NOT_RUN' in out


def test_skipped_unrequired_test_still_verifies_but_required_absence_does_not(repo, tmp_path):
    skipping = junit(tmp_path / 'skip.xml', [('tests.test_alpha', 'test_one', 'skip'), ('tests.test_alpha', 'test_two', 'pass')])
    record(repo, 'G0-01', skipping)
    assert task_of(repo, 'G0-01')['verification_state'] == 'VERIFIED'
    absent = junit(tmp_path / 'absent.xml', [('tests.test_alpha', 'test_one', 'pass')])
    record(repo, 'G1-03', absent)
    assert task_of(repo, 'G1-03')['verification_state'] == 'NOT_RUN'


def test_gpu_required_skip_of_an_unlisted_test_is_a_failure(repo, tmp_path, capsys):
    """G4-05: a CUDA test that skipped in a recorded run fails the task even when required_tests does not name it."""
    skipping = junit(tmp_path / 'gpu.xml', [('tests.test_alpha', 'test_one', ('skip', 'CUDA unavailable')), ('tests.test_alpha', 'test_two', 'pass')])
    record(repo, 'G1-03', skipping)
    out = capsys.readouterr().out
    task = task_of(repo, 'G1-03')
    assert task['verification_state'] == 'FAILED'
    assert 'GPU-required tests skipped in a required run: tests/test_alpha.py::test_one' in out
    evidence = json.loads((repo / 'docs' / 'validation' / 'runs' / task['evidence'][0] / 'evidence.json').read_text(encoding='utf-8'))
    assert evidence['gpu_required_skips'] == ['tests/test_alpha.py::test_one'] and evidence['skipped_required_tests'] == []
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'verification_state is FAILED' in out


@pytest.mark.parametrize('reason', ["could not import 'cupy': No module named cupy", 'Requires installed GPU', 'CUDA GPU not available'])
def test_gpu_skip_reasons_are_classified_and_optional_checks_are_not(repo, tmp_path, reason):
    record(repo, 'G0-01', junit(tmp_path / 'gpu.xml', [('tests.test_alpha', 'test_one', ('skip', reason)), ('tests.test_alpha', 'test_two', 'pass')]))
    assert task_of(repo, 'G0-01')['verification_state'] == 'FAILED'
    optional = junit(tmp_path / 'optional.xml', [('tests.test_alpha', 'test_one', ('skip', 'optional platform check: ' + reason)),
                                                 ('tests.test_alpha', 'test_two', 'pass')])
    record(repo, 'G1-03', optional)
    task = task_of(repo, 'G1-03')
    assert task['verification_state'] == 'VERIFIED'
    evidence = json.loads((repo / 'docs' / 'validation' / 'runs' / task['evidence'][0] / 'evidence.json').read_text(encoding='utf-8'))
    assert evidence['gpu_required_skips'] == []


def test_judge_classifies_gpu_skips_in_evidence_recorded_without_the_field(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    run_dir, evidence = evidence_of(repo, 'G1-03')
    del evidence['gpu_required_skips']
    evidence['test_results']['skipped'] = ['tests/test_alpha.py::test_three']
    evidence['test_results']['skipped_reasons'] = {'tests/test_alpha.py::test_three': 'CUDA unavailable'}
    # An old-format record is consistent with its junit copy; only the classification field is absent.
    copy = run_dir / 'junit.xml'
    text = copy.read_text(encoding='utf-8').replace('</testsuite>', '<testcase classname="tests.test_alpha" name="test_three" time="0.001">'
                                                    '<skipped type="pytest.skip" message="CUDA unavailable">CUDA unavailable</skipped></testcase></testsuite>')
    copy.write_text(text, encoding='utf-8')
    for artifact in evidence['artifact_paths_and_sha256']:
        if artifact['path'].endswith('/junit.xml'):
            artifact['sha256'] = hashlib.sha256(copy.read_bytes()).hexdigest()
    rewrite_evidence(run_dir, evidence)
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'GPU-required tests skipped in a required run: tests/test_alpha.py::test_three' in out


def test_nonzero_exit_code_without_failures_does_not_verify(repo, tmp_path):
    record(repo, 'G1-03', passing_junit(tmp_path), exit_code=3)
    assert task_of(repo, 'G1-03')['verification_state'] == 'NOT_RUN'


def test_parametrized_required_test_matches_instances(repo, tmp_path):
    cases = [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two[cpu]', 'pass'), ('tests.test_alpha', 'test_two[cuda]', 'skip')]
    record(repo, 'G1-03', junit(tmp_path / 'param.xml', cases))
    assert task_of(repo, 'G1-03')['verification_state'] == 'NOT_RUN'


def test_modified_test_file_after_recording_is_a_hash_mismatch(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    (repo / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE + '\n# edited after the run\n', encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'STALE: test source changed since the run: tests/test_alpha.py' in out
    code, out = judge_run(repo, capsys, '--task', 'G1-03', '--allow-stale')
    assert code == 0 and 'STALE-ACCEPTED' in out and 'NOT a release judgement' in out and 'stale evidence accepted for: G1-03' in out


def test_evidence_from_a_commit_that_is_not_an_ancestor_is_stale(repo, tmp_path, capsys):
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'note.txt').write_text('second commit\n', encoding='utf-8')
    git(repo, 'add', 'note.txt')
    git(repo, 'commit', '-q', '-m', 'second')
    record(repo, 'G1-03', passing_junit(tmp_path))
    assert judge_run(repo, capsys, '--task', 'G1-03')[0] == 0
    git(repo, 'checkout', '-q', base)  # the uncommitted gate file and evidence directory carry over to the older commit
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'is not an ancestor of HEAD' in out
    code, out = judge_run(repo, capsys, '--task', 'G1-03', '--allow-stale')
    assert code == 0 and 'STALE-ACCEPTED' in out


def test_missing_evidence_file_is_reported(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    run_id = task_of(repo, 'G1-03')['evidence'][0]
    (repo / 'docs' / 'validation' / 'runs' / run_id / 'evidence.json').unlink()
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'evidence file missing' in out


def test_hand_marked_verified_without_evidence_is_rejected(repo, capsys):
    gate_path = repo / 'docs' / 'validation' / 'completion_gates.json'
    gates = gates_of(repo)
    for stage in gates['stages']:
        for task in stage['tasks']:
            if task['id'] == 'G1-03':
                task['verification_state'] = 'VERIFIED'
    gate_path.write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'VERIFIED without any evidence run' in out


def test_hand_marked_verified_over_failed_evidence_is_rejected(repo, tmp_path, capsys):
    failing = junit(tmp_path / 'fail.xml', [('tests.test_alpha', 'test_one', 'fail'), ('tests.test_alpha', 'test_two', 'pass')])
    record(repo, 'G1-03', failing)
    gate_path = repo / 'docs' / 'validation' / 'completion_gates.json'
    gate_path.write_text(gate_path.read_text(encoding='utf-8').replace('"verification_state": "FAILED"', '"verification_state": "VERIFIED"'), encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and "assigned 'FAILED'" in out


def test_profile_judgement_needs_every_required_task_and_reports_blockers(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    code, out = judge_run(repo, capsys)
    assert code == 1 and 'G0-01' in out and 'verification_state is NOT_RUN' in out
    record(repo, 'G0-01', junit(tmp_path / 'g0.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    record(repo, 'G1-05', passing_junit(tmp_path), command=PYTEST_ALPHA)
    record(repo, 'G1-06', junit(tmp_path / 'g1-06.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    code, out = judge_run(repo, capsys)
    assert code == 0 and 'all judged tasks pass' in out
    code, out = judge_run(repo, capsys, '--profile', 'HPC')
    assert code == 1 and 'external blocker unresolved: BLOCKED_EXTERNAL: no second GPU' in out


def test_failed_task_outside_the_profile_blocks_release(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    record(repo, 'G0-01', junit(tmp_path / 'g0.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    record(repo, 'G1-05', passing_junit(tmp_path), command=PYTEST_ALPHA)
    record(repo, 'G1-06', junit(tmp_path / 'g1-06.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    record(repo, 'H1-02', junit(tmp_path / 'h1.xml', [('tests.test_alpha', 'test_one', 'fail')]))
    code, out = judge_run(repo, capsys)
    assert code == 1 and 'FAILED tasks outside the selection: H1-02' in out


def test_dirty_required_test_is_refused_and_allow_dirty_evidence_fails_the_judge(repo, tmp_path, capsys):
    (repo / 'untracked.txt').write_text('x\n', encoding='utf-8')
    record(repo, 'G1-03', passing_junit(tmp_path))  # an unrelated untracked file is a warning, not a refusal
    run_dir, evidence = evidence_of(repo, 'G1-03')
    manifest = {row['path']: row for row in evidence['dirty_source_manifest']}
    assert manifest['untracked.txt']['status'] == '??' and manifest['untracked.txt']['head_sha256'] is None
    assert evidence['dirty_allowed'] is False and evidence['dirty_guarded_paths'] == []
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'recorded on a dirty tree' in out
    (repo / 'tests' / 'test_alpha.py').write_text(TEST_SOURCE + '\n# uncommitted weakening\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='refusing to record'):
        record(repo, 'G1-03', passing_junit(tmp_path))
    assert len(task_of(repo, 'G1-03')['evidence']) == 1
    assert record(repo, 'G1-03', passing_junit(tmp_path), allow_dirty='reviewing an unmerged fix') == 0
    run_dir, evidence = evidence_of(repo, 'G1-03')
    manifest = {row['path']: row for row in evidence['dirty_source_manifest']}
    assert manifest['tests/test_alpha.py']['status'] == 'M' and manifest['tests/test_alpha.py']['head_sha256']
    assert evidence['dirty_allowed'] is True and evidence['dirty_allowed_reason'] == 'reviewing an unmerged fix'
    assert evidence['dirty_guarded_paths'] == ['tests/test_alpha.py']
    assert task_of(repo, 'G1-03')['verification_state'] == 'VERIFIED'
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'recorded with --allow-dirty (reviewing an unmerged fix)' in out


def test_fixture_and_criteria_hashes_are_recorded_and_checked(repo, tmp_path, capsys):
    case = repo / 'docs' / 'validation' / 'cases' / 'demo.json'
    case.parent.mkdir(parents=True)
    case.write_text(json.dumps(dict(fixture=dict(mesh=0.05), seed=7, precision='float32', backend='cpu',
                                    reference_method='analytic', observables=['ratio'], acceptance=dict(rtol=1e-4))) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='refusing to record'):  # the case file must be committed before the run counts
        record(repo, 'G1-03', passing_junit(tmp_path), fixture=case)
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'declare the case')
    record(repo, 'G1-03', passing_junit(tmp_path), fixture=case)
    run_dir, evidence = evidence_of(repo, 'G1-03')
    assert evidence['fixture_sha256'] == evidence['acceptance_criteria_sha256'] == recorder.file_sha256(case)
    assert evidence['seed'] == 7 and evidence['acceptance_limits'] == dict(rtol=1e-4) and evidence['physics_configuration'] == dict(mesh=0.05)
    assert evidence['declared_before_run_verified'] is True and evidence['case_first_commit'] == git(repo, 'rev-parse', 'HEAD')
    assert judge_run(repo, capsys, '--task', 'G1-03')[0] == 0
    edited = case.read_text(encoding='utf-8').replace('0.0001', '0.01')
    assert edited != case.read_text(encoding='utf-8')
    case.write_text(edited, encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'fixture changed since the run' in out


def test_unknown_task_is_refused(repo, tmp_path):
    with pytest.raises(SystemExit):
        record(repo, 'G9-99', passing_junit(tmp_path))


def test_repository_gate_file_is_adopted():
    gates = json.loads(REAL_GATES.read_text(encoding='utf-8'))
    assert gates['kind'] == 'completion_gates' and gates['adopted_commit'] and gates['planning_snapshot_commit']
    ids = [t['id'] for s in gates['stages'] for t in s['tasks']]
    assert len(ids) == len(set(ids)) == gates['task_count']
    # Readiness changes as evidence is recorded. Controlled repositories above
    # test the passing, missing, failed and stale states independently of this tree.


def test_junit_older_than_the_commit_is_refused_unless_allowed_and_then_warned(repo, tmp_path, capsys):
    stale_start = (datetime.datetime.now().astimezone() - datetime.timedelta(hours=1)).isoformat()
    early = junit(tmp_path / 'early.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'pass')], timestamp=stale_start)
    with pytest.raises(SystemExit, match='before the source commit'):
        record(repo, 'G1-03', early)
    assert task_of(repo, 'G1-03')['evidence'] == []
    assert record(repo, 'G1-03', early, allow_precommit_junit='historic run kept for the record') == 0
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['junit_started_before_commit'] is True and evidence['precommit_junit_reason'] == 'historic run kept for the record'
    assert evidence['source_commit_time']
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'run predates its commit' in out and 'historic run kept for the record' in out
    later = passing_junit(tmp_path)
    assert record(repo, 'G1-03', later) == 0
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['junit_started_before_commit'] is False and evidence['precommit_junit_reason'] is None
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'run predates its commit' not in out


def test_judge_derives_the_precommit_warning_for_evidence_without_the_field(repo, tmp_path, capsys):
    record(repo, 'G1-03', passing_junit(tmp_path))
    run_dir, evidence = evidence_of(repo, 'G1-03')
    del evidence['junit_started_before_commit']
    evidence['execution_timestamp'] = (datetime.datetime.now().astimezone() - datetime.timedelta(days=1)).isoformat()
    rewrite_evidence(run_dir, evidence)
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'run predates its commit' in out


def test_watched_data_files_are_hashed_and_a_change_or_a_new_match_is_stale(repo, tmp_path, capsys):
    record(repo, 'G1-06', junit(tmp_path / 'w.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    _, evidence = evidence_of(repo, 'G1-06')
    assert evidence['watch_paths'] == ['data/*.json', 'data/notes.md']
    assert set(evidence['watch_sha256']) == {'data/a.json', 'data/notes.md'}
    assert evidence['watch_sha256']['data/notes.md'] == recorder.file_sha256(repo / 'data' / 'notes.md')
    assert judge_run(repo, capsys, '--task', 'G1-06')[0] == 0
    (repo / 'data' / 'notes.md').write_text('notes changed\n', encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-06')
    assert code == 1 and 'STALE: watched file changed since the run: data/notes.md' in out
    git(repo, 'checkout', '--', 'data/notes.md')
    (repo / 'data' / 'b.json').write_text('{"b": 2}\n', encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-06')
    assert code == 1 and 'a file matching a watched pattern did not exist at recording: data/b.json' in out
    (repo / 'data' / 'b.json').unlink()
    (repo / 'data' / 'a.json').unlink()
    code, out = judge_run(repo, capsys, '--task', 'G1-06')
    assert code == 1 and 'watched file missing: data/a.json' in out


def test_evidence_that_predates_the_watch_list_is_stale(repo, tmp_path, capsys):
    record(repo, 'G1-06', junit(tmp_path / 'w.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    run_dir, evidence = evidence_of(repo, 'G1-06')
    del evidence['watch_sha256']
    rewrite_evidence(run_dir, evidence)
    code, out = judge_run(repo, capsys, '--task', 'G1-06')
    assert code == 1 and 'evidence predates the watch list' in out


def test_file_level_required_tests_are_enumerated_and_a_partial_run_is_not_verified(repo, tmp_path, capsys):
    partial = junit(tmp_path / 'partial.xml', [('tests.test_alpha', 'test_one', 'pass')])
    record(repo, 'G1-05', partial, command=PYTEST_ALPHA + ' -k test_one')
    out = capsys.readouterr().out
    assert task_of(repo, 'G1-05')['verification_state'] == 'NOT_RUN'
    assert 'partial run' in out and 'tests/test_alpha.py::test_two' in out
    record(repo, 'G1-05', passing_junit(tmp_path), command=PYTEST_ALPHA)
    assert task_of(repo, 'G1-05')['verification_state'] == 'VERIFIED'
    run_dir, evidence = evidence_of(repo, 'G1-05')
    assert evidence['enumerated_required_tests'] == {'tests/test_alpha.py': ['tests/test_alpha.py::test_one', 'tests/test_alpha.py::test_two']}
    assert judge_run(repo, capsys, '--task', 'G1-05')[0] == 0
    # The judge checks the stored enumeration against the junit again: a test dropped from the copy is a partial run.
    copy = run_dir / 'junit.xml'
    text = copy.read_text(encoding='utf-8')
    kept = text[:text.index('<testcase classname="tests.test_alpha" name="test_two"')] + '</testsuite></testsuites>'
    copy.write_text(kept, encoding='utf-8')
    evidence['test_results']['passed'] = ['tests/test_alpha.py::test_one']
    for artifact in evidence['artifact_paths_and_sha256']:
        if artifact['path'].endswith('/junit.xml'):
            artifact['sha256'] = hashlib.sha256(copy.read_bytes()).hexdigest()
    rewrite_evidence(run_dir, evidence)
    code, out = judge_run(repo, capsys, '--task', 'G1-05')
    assert code == 1 and 'collected tests of tests/test_alpha.py absent from the junit (partial run): tests/test_alpha.py::test_two' in out


def test_optional_platform_skip_inside_a_file_level_entry_is_allowed_but_another_skip_is_not(repo, tmp_path, capsys):
    optional = junit(tmp_path / 'opt.xml', [('tests.test_alpha', 'test_one', 'pass'),
                                            ('tests.test_alpha', 'test_two', ('skip', 'optional platform check: no second GPU'))])
    record(repo, 'G1-05', optional, command=PYTEST_ALPHA)
    assert task_of(repo, 'G1-05')['verification_state'] == 'VERIFIED'
    assert judge_run(repo, capsys, '--task', 'G1-05')[0] == 0
    plain = junit(tmp_path / 'plain.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'skip')])
    record(repo, 'G1-05', plain, command=PYTEST_ALPHA)
    out = capsys.readouterr().out
    assert task_of(repo, 'G1-05')['verification_state'] == 'NOT_RUN' and 'required tests were skipped: tests/test_alpha.py' in out


def test_file_level_entry_needs_an_enumerable_command(repo, tmp_path):
    with pytest.raises(SystemExit, match='cannot enumerate'):
        record(repo, 'G1-05', passing_junit(tmp_path), command='make check')
    assert task_of(repo, 'G1-05')['evidence'] == []


def test_hand_edited_evidence_or_junit_copy_is_rejected_by_the_judge(repo, tmp_path, capsys):
    failing = junit(tmp_path / 'fail.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'fail')])
    record(repo, 'G1-03', failing)
    run_dir, evidence = evidence_of(repo, 'G1-03')
    evidence['test_results']['failed'] = []
    evidence['test_results']['passed'].append('tests/test_alpha.py::test_two')
    evidence['verification_state_assigned'] = 'VERIFIED'
    evidence['exit_code'] = 0
    rewrite_evidence(run_dir, evidence)
    gate_path = repo / 'docs' / 'validation' / 'completion_gates.json'
    gate_path.write_text(gate_path.read_text(encoding='utf-8').replace('"verification_state": "FAILED"', '"verification_state": "VERIFIED"'), encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 1 and 'evidence does not match its junit: passed, failed differ' in out
    record(repo, 'G0-01', passing_junit(tmp_path))
    run_dir, _ = evidence_of(repo, 'G0-01')
    copy = run_dir / 'junit.xml'
    copy.write_text(copy.read_text(encoding='utf-8').replace('time="0.001"', 'time="0.002"', 1), encoding='utf-8')
    code, out = judge_run(repo, capsys, '--task', 'G0-01')
    assert code == 1 and 'junit copy does not match the SHA-256 stored in the evidence' in out
    copy.unlink()
    code, out = judge_run(repo, capsys, '--task', 'G0-01')
    assert code == 1 and 'junit copy missing' in out


def test_environment_records_package_locations_and_the_original_junit_path(repo, tmp_path):
    report = passing_junit(tmp_path)
    record(repo, 'G1-03', report)
    _, evidence = evidence_of(repo, 'G1-03')
    for key in ('fdtd', 'fdtd_location', 'torchfdtd_location'):
        assert key in evidence['environment'], key
    assert evidence['junit_original_path'] == recorder.redact_home(report.resolve().as_posix())
    assert evidence['recorder_version'] == 2
    home = str(Path.home())
    for value in (evidence['junit_original_path'], evidence['environment']['python_executable'],
                  evidence['environment']['fdtd_location'], evidence['environment']['torchfdtd_location']):
        assert value is None or home.lower() not in str(value).lower(), value  # the account name never enters the evidence
    assert recorder.redact_home(home + '/x/y.py') == '<user home>/x/y.py'


def test_case_first_committed_after_the_run_is_a_declaration_warning_not_a_failure(repo, tmp_path, capsys):
    case = repo / 'docs' / 'validation' / 'cases' / 'late.json'
    case.parent.mkdir(parents=True)
    case.write_text(json.dumps(dict(seed=1, acceptance=dict(limit=1))) + '\n', encoding='utf-8')
    started = (datetime.datetime.now().astimezone() - datetime.timedelta(seconds=5)).isoformat()  # the run started, then the case is committed
    report = junit(tmp_path / 'pass.xml', [('tests.test_alpha', 'test_one', 'pass'), ('tests.test_alpha', 'test_two', 'pass')], timestamp=started)
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'declare the case after the run')
    record(repo, 'G1-03', report, fixture=case, allow_precommit_junit='the run started before this commit')
    _, evidence = evidence_of(repo, 'G1-03')
    assert evidence['declared_before_run_verified'] is False and 'after the run started' in evidence['declaration_note']
    code, out = judge_run(repo, capsys, '--task', 'G1-03')
    assert code == 0 and 'declaration order not verified' in out


def test_scope_changes_without_approval_are_warned_and_listed(repo, tmp_path, capsys):
    cases = repo / 'docs' / 'validation' / 'cases'
    cases.mkdir(parents=True)
    (cases / 'G1-03_r2.json').write_text(json.dumps(dict(task='G1-03', supersedes=dict(case='G1-03_first.json'),
                                                         acceptance=dict(rtol=1e-4))) + '\n', encoding='utf-8')
    (cases / 'G0-01_loose.json').write_text(json.dumps(dict(task='G0-01', acceptance=dict(gradient=dict(rtol=1e-3)))) + '\n', encoding='utf-8')
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'cases')
    record(repo, 'G1-03', passing_junit(tmp_path))
    record(repo, 'G0-01', junit(tmp_path / 'g0.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    record(repo, 'G1-05', passing_junit(tmp_path), command=PYTEST_ALPHA)
    record(repo, 'G1-06', junit(tmp_path / 'g1-06.xml', [('tests.test_alpha', 'test_one', 'pass')]))
    code, out = judge_run(repo, capsys)
    assert code == 0
    assert 'G1-03: scope change pending approval (1 declaration(s), scope_change_approval is null): docs/validation/cases/G1-03_r2.json declares supersedes' in out
    assert 'G0-01: scope change pending approval (1 declaration(s), scope_change_approval is null): docs/validation/cases/G0-01_loose.json acceptance/gradient/rtol = 0.001 is looser' in out
    assert 'scope changes pending owner approval (scope_change_approval is null): G0-01, G1-03' in out
    pending = judge.pending_scope_changes(repo, gates_of(repo))
    assert set(pending) == {'G0-01', 'G1-03'}
    gate_path = repo / 'docs' / 'validation' / 'completion_gates.json'
    gates = gates_of(repo)
    for stage in gates['stages']:
        for task in stage['tasks']:
            if task['id'] == 'G1-03':
                task['scope_change_approval'] = 'owner, 2026-09-22: revision accepted'
    gate_path.write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    code, out = judge_run(repo, capsys)
    assert code == 0 and 'G1-03: scope change pending approval' not in out and 'G0-01: scope change pending approval' in out
