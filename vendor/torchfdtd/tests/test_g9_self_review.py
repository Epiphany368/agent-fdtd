"""G9-05 documents an installation self-review, not another person's usability."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'docs/validation/g9-05-self-review.json'


def test_three_documented_workflows_completed_from_independent_installation():
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    assert record['task'] == 'G9-05' and record['workflows_completed'] == 3
    assert record['independent_user_test'] is False
    assert record['case_sha256'] == hashlib.sha256(
        (ROOT/'docs/validation/cases/G9-05.json').read_bytes()).hexdigest()
    assert set(record['preconditions']) == {'G7-01', 'G7-02', 'G7-03'}
    assert all(row['judge_exit_code'] == 0 and row['evidence'] for row in record['preconditions'].values())
    environment = record['installation']
    assert environment['fresh_venv'] and not environment['system_site_packages']
    assert 'site-packages' in environment['package_file']
    assert environment['solver_source_on_sys_path'] is False
    assert environment['payload_matches'] and environment['payload_files'] > 0
    assert len(environment['wheel_sha256']) == 64
    assert record['examples']['contains_solver_source'] is False
    assert set(record['workflows']) == {'metagrating', 'metalens', 'coupler'}
    for name, row in record['workflows'].items():
        assert row['exit_code'] == 0 and row['wall_seconds'] > 0
        assert row['commands'] and row['completed']
        summary = json.loads((ROOT/row['summary_path']).read_text(encoding='utf-8'))
        assert hashlib.sha256((ROOT/row['summary_path']).read_bytes()).hexdigest() == row['summary_sha256']
        if name == 'metagrating':
            assert summary['all_passed'] and set(summary['seeds']) == {'7', '8', '9'}
        elif name == 'metalens':
            assert summary['all_pass']
            assert len(summary['records']) == 3
        else:
            assert summary['judged'] and summary['passed'] and summary['seeds'] == [1, 2, 3]


def test_self_review_tracks_documented_commands_and_issue_resolutions():
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    assert record['instructions'] == ['README.md', 'docs/G7_RUN.md', 'docs/G7_WORKFLOWS.md']
    assert all(issue.get('fix_commit') or issue.get('open_issue') for issue in record['issues_found'])
    assert record['cpu_development']['metalens']['exit_code'] == 0
    assert record['cpu_development']['coupler']['exit_code'] == 0
    assert 'docs/G7_RUN.md' in (ROOT/'README.md').read_text(encoding='utf-8')
