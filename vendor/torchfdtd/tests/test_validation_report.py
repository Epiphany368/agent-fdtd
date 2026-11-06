"""docs/VALIDATION_REPORT.md is the renderer's output for the committed records, and it says what it must and must not say (G9-07)."""
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs' / 'VALIDATION_REPORT.md'
SCOPE = ROOT / 'docs' / 'RELEASE_SCOPE.md'
FORBIDDEN = ('certificate', 'certified', 'certification', 'certify')


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script('build_validation_report')


@pytest.fixture(scope='module')
def rendered():
    return builder.render(ROOT, check_scope=True)


def test_report_is_the_renderer_output_for_the_current_tree(rendered, tmp_path):
    fresh = tmp_path / 'VALIDATION_REPORT.md'
    fresh.write_bytes(rendered['report'].encode('utf-8'))
    assert fresh.read_bytes() == REPORT.read_bytes(), \
        'docs/VALIDATION_REPORT.md differs from a fresh render of the records; run scripts/build_validation_report.py on a clean tree'


def test_release_scope_cells_and_stage_block_match_the_gate_file(rendered):
    assert rendered['scope_cells'] >= 20
    assert not rendered['scope_changed'], 'docs/RELEASE_SCOPE.md verification cells or stage block drifted from the gate file; rerun the builder'
    text = SCOPE.read_text(encoding='utf-8')
    assert text.count(builder.STAGE_BEGIN) == 1 and text.count(builder.STAGE_END) == 1
    gates = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    block = text[text.index(builder.STAGE_BEGIN):text.index(builder.STAGE_END)]
    for stage in gates['stages']:
        assert f"| {stage['id']} |" in block


def test_report_title_and_forbidden_words():
    text = REPORT.read_text(encoding='utf-8')
    assert text.startswith('# TorchFDTD internal validation report\n')
    assert 'internal validation report' in text.lower()
    lower = text.lower()
    for word in FORBIDDEN:
        assert word not in lower, word
    assert '**PROVISIONAL**' not in text, 'the committed report must be rendered on a clean tree'


def test_consistency_section_reports_no_mismatch(rendered):
    failing = [(name, detail) for name, ok, detail in rendered['checks'] if not ok]
    assert not failing, failing
    consistency = rendered['report'].split('## Consistency', 1)[1]
    assert '**MISMATCH**' not in consistency and ', 0 mismatch(es).' in consistency
    names = [name for name, _, _ in rendered['checks']]
    assert 'package version' in names and 'RELEASE_SCOPE.md support claims' in names and 'attestation wording' in names
    assert sum(name.startswith('README row check') for name in names) >= 5
    assert 'third-party notices and SBOM' in names


def test_platform_section_lists_every_record_with_its_g4_evidence(rendered):
    section = rendered['report'].split('## Platform records', 1)[1].split('\n## ', 1)[0]
    records = sorted((ROOT / 'docs' / 'validation' / 'platforms').glob('*.json'))
    assert len(records) >= 2
    for path in records:
        platform_id = json.loads(path.read_text(encoding='utf-8'))['platform_id']
        assert section.count(f'| {platform_id} |') == 2, platform_id
    gates = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    g4 = next(stage for stage in gates['stages'] if stage['id'] == 'G4')
    for task in g4['tasks']:
        if task['evidence']:
            assert f"{task['id']} `{task['evidence'][-1]}`" in section


def test_report_lists_every_gate_task_and_every_g3_task_once(rendered):
    gates = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    report = rendered['report']
    physics = report.split('## Physics validation (stage G3)', 1)[1].split('\n## ', 1)[0]
    for stage in gates['stages']:
        for task in stage['tasks']:
            assert f"| {task['id']} | " in report
            if stage['id'] == 'G3':
                assert physics.count(f"| {task['id']} | ") == 1
    assert '**FAIL**' in physics or 'FAILED' in physics  # the recorded G3-05 and G3-08 findings stay visible


def test_report_lists_evidence_warnings_and_pending_approvals(rendered):
    report = rendered['report']
    assert '## Evidence warnings' in report and '## Pending owner approvals' in report
    pending = report.split('## Pending owner approvals', 1)[1].split('## Consistency', 1)[0]
    gates = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    expected = builder.judge.pending_scope_changes(ROOT, gates)
    for task_id in expected:
        assert f'| {task_id} |' in pending
    declared = builder.declared_scope_changes(ROOT, gates)
    assert 'G3-02' in declared and 'G3-08' in declared  # the revised cases of the record, pending or approved
    approved = pending.split('### Approved scope changes', 1)[1] if '### Approved scope changes' in pending else ''
    for task_id, (_, approval) in declared.items():
        assert (task_id in expected) != bool(approval), f'{task_id} must be either pending or approved'
        if approval:
            assert f'| {task_id} |' in approved, f'the approved scope change of {task_id} is not listed with its approval'
    warnings = report.split('## Evidence warnings', 1)[1].split('## Pending owner approvals', 1)[0]
    for _, task in builder.all_tasks(gates):
        if task['id'] == builder.SELF_TASK:
            continue  # the report's own gate is shown without a judgement; the judge reports its warnings
        for warning in rendered_warnings(task['id']):
            assert warning[:60] in warnings
    assert f'| {builder.SELF_TASK} |' not in warnings


def rendered_warnings(task_id):
    gates = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    task = next(t for _, t in builder.all_tasks(gates) if t['id'] == task_id)
    return builder.judge.judge_task(ROOT, gates, task, ROOT / 'docs' / 'validation' / 'runs')[2]


def test_check_mode_passes_on_the_committed_tree(capsys):
    assert builder.main(['--check']) == 0
    out = capsys.readouterr().out
    assert 'report matches docs/VALIDATION_REPORT.md' in out and 'MISMATCH' not in out


def test_scope_cells_are_regenerated_only_where_they_state_a_gate_result():
    gates = dict(stages=[dict(id='G4', tasks=[
        dict(id='G4-01', verification_state='VERIFIED', blocker=None), dict(id='G4-02', verification_state='VERIFIED', blocker=None),
        dict(id='G4-03', verification_state='NOT_RUN', blocker=None), dict(id='G4-04', verification_state='NOT_RUN', blocker='BLOCKED_EXTERNAL: x')])])
    assert builder.task_ids_in('NOT_RUN (G4-01 to G4-03)', gates) == ['G4-01', 'G4-02', 'G4-03']
    assert builder.render_cell(['G4-01', 'G4-02'], gates) == 'VERIFIED (G4-01, G4-02)'
    assert builder.render_cell(['G4-01', 'G4-03', 'G4-04'], gates) == 'MIXED: VERIFIED G4-01; NOT_RUN G4-03; BLOCKED_EXTERNAL G4-04'
    document = '\n'.join([
        '| Row | Implemented scope and record | Verified for release |', '| --- | --- | --- |',
        '| A | N-port \\|S_ij\\|² objectives | NOT_RUN (G4-01 to G4-02) |',
        '| B | prose | Not a gate row; decision pending (G4-03) |',
        '| C | prose | VERIFIED (G4-01, G4-04) |', '',
        '| Other | table | without the column |', '| --- | --- | --- |', '| x | NOT_RUN (G4-01) | y |', '',
        '## Stage status', '', builder.STAGE_BEGIN, 'old', builder.STAGE_END, ''])

    gates.update(profiles={}, stages=[dict(id='G4', title='cuda', profile='WORKSTATION', tasks=gates['stages'][0]['tasks'])])
    verdicts = {task['id']: (('PASS', 'ok') if task['verification_state'] == 'VERIFIED' else ('FAIL', 'not verified'))
                for task in gates['stages'][0]['tasks']}
    text, count = builder.render_scope(document, gates, verdicts)
    again, _ = builder.render_scope(text, gates, verdicts)
    assert count == 2
    lines = text.split('\n')
    assert lines[2] == '| A | N-port \\|S_ij\\|² objectives | VERIFIED (G4-01, G4-02) |'
    assert lines[3] == '| B | prose | Not a gate row; decision pending (G4-03) |'
    assert lines[4] == '| C | prose | MIXED: VERIFIED G4-01; BLOCKED_EXTERNAL G4-04 |'
    assert lines[8] == '| x | NOT_RUN (G4-01) | y |'
    assert 'old' not in text and '| G4 | cuda | WORKSTATION | 4 | 2 | 0 | 1 | 1 | 2 pass, 2 fail |' in text
    assert again == text


def _git(root, *args):
    return subprocess.run(['git', '-c', 'user.name=gate', '-c', 'user.email=gate@example.invalid', '-c', 'core.autocrlf=false',
                           '-c', 'commit.gpgsign=false', *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _fake_repository(root):
    """A committed miniature repository: one whole-file task with a watched data glob, a case file and a platform record."""
    real = json.loads((ROOT / 'docs' / 'validation' / 'completion_gates.json').read_text(encoding='utf-8'))
    (root / 'tests').mkdir(parents=True)
    (root / 'tests' / 'test_alpha.py').write_text('def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n', encoding='utf-8')
    (root / 'data').mkdir()
    (root / 'data' / 'a.json').write_text('{"a": 1}\n', encoding='utf-8')
    cases = root / 'docs' / 'validation' / 'cases'
    cases.mkdir(parents=True)
    (cases / 'G1-03_demo.json').write_text(json.dumps(dict(task='G1-03', title='demo case', seed=1, acceptance=dict(rtol=1e-4))) + '\n', encoding='utf-8')
    platforms = root / 'docs' / 'validation' / 'platforms'
    platforms.mkdir()
    (platforms / 'lab.json').write_text(json.dumps(dict(platform_id='lab', recorded_at='2026-09-22T00:00:00+00:00', os='Linux-6.8', python='3.11.9',
                                                        torch='2.10.0', cupy=None, cuda_runtime=None, driver=None, gpus=[])) + '\n', encoding='utf-8')
    (root / 'docs' / 'validation' / 'runs').mkdir()
    (root / '.gitignore').write_text('.local/\n__pycache__/\n', encoding='utf-8')
    (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding='utf-8')
    task = dict(id='G1-03', title='demo task', specification='demo', required_by_current_plan=True, implementation_state='IMPLEMENTED',
                verification_state='NOT_RUN', owner=None, code_paths=['tests/test_alpha.py'], planned_test_commands=[], actual_test_commands=[],
                required_tests=['tests/test_alpha.py::test_one'], evidence=[], blocker=None, scope_change_approval=None, watch_paths=['data/*.json'])
    gates = {key: real[key] for key in ('schema_version', 'kind', 'implementation_states', 'verification_states', 'required_evidence_fields',
                                        'release_rule', 'proposed_thresholds')}
    gates.update(adopted_commit=None, task_count=1, profiles={'WORKSTATION': {'required_stages': ['G1'], 'scope_status': 'TEST'}},
                 stages=[dict(id='G1', title='demo stage', depends_on_stages=[], profile='WORKSTATION', priority='P0', tasks=[task])])
    (root / 'docs' / 'validation' / 'completion_gates.json').write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    _git(root, 'init', '-q')
    _git(root, 'add', '.')
    _git(root, 'commit', '-q', '-m', 'baseline')
    junit = root.parent / 'run.xml'
    junit.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="2" failures="0" errors="0" skipped="0" '
                     f'time="0.01" timestamp="{datetime.datetime.now().astimezone().isoformat()}">'
                     '<testcase classname="tests.test_alpha" name="test_one" time="0.001"></testcase>'
                     '<testcase classname="tests.test_alpha" name="test_two" time="0.001"></testcase></testsuite></testsuites>', encoding='utf-8')
    assert builder.recorder.main(['--root', str(root), '--task', 'G1-03', '--command', 'pytest tests/test_alpha.py', '--junit', str(junit),
                                  '--exit-code', '0', '--fixture', str(cases / 'G1-03_demo.json'), '--platform', 'lab']) == 0


def _gate_sections(root):
    gates = builder.load_json(root / builder.GATE_FILE)
    runs = root / builder.RUNS_DIR
    verdicts = builder.judge_all(root, gates, runs)
    lines = builder.section_gates(root, gates, runs, verdicts) + builder.stage_status_block(gates, verdicts)
    lines += builder.section_platforms(root, gates, runs) + builder.section_warnings(gates, verdicts) + builder.section_pending_approvals(root, gates)
    lines += builder.render_scope('## Stage status\n\n' + builder.STAGE_BEGIN + '\n' + builder.STAGE_END + '\n', gates, verdicts)[0].split('\n')
    return '\n'.join(lines)


def test_gate_sections_render_identically_for_windows_and_posix_root_spellings(tmp_path):
    """The rendered sections carry POSIX paths only and do not depend on how the root is spelled or on the host's path flavour."""
    root = tmp_path / 'repo'
    _fake_repository(root)
    native = Path(str(root))
    posix = Path(root.as_posix())  # forward slashes on every platform; on POSIX hosts the same object as native
    first, second = _gate_sections(native), _gate_sections(posix)
    assert first == second
    assert '\\' not in first, first
    assert '| G1-03 | demo task | IMPLEMENTED | VERIFIED |' in first and '| PASS |' in first
    assert '| lab |' in first and '| G1 | demo stage | WORKSTATION | 1 | 1 | 0 | 0 | 0 | 1 pass, 0 fail |' in first
    run_dir = next((root / 'docs' / 'validation' / 'runs').iterdir())
    evidence = json.loads((run_dir / 'evidence.json').read_text(encoding='utf-8'))
    for value in [evidence['fixture_path'], *evidence['watch_sha256'], *evidence['test_source_sha256'], evidence['gate_file']]:
        assert value == PurePosixPath(value).as_posix() and not value.startswith('/') and ':' not in value, value
    assert list(evidence['watch_sha256']) == ['data/a.json']
    # A watched file changed under either spelling is the same STALE judgement.
    (root / 'data' / 'a.json').write_text('{"a": 2}\n', encoding='utf-8')
    for spelling in (native, posix):
        gates = builder.load_json(spelling / builder.GATE_FILE)
        label, reason, _ = builder.judge_all(spelling, gates, spelling / builder.RUNS_DIR)['G1-03']
        assert label == 'FAIL' and reason == 'STALE: watched file changed since the run: data/a.json'
