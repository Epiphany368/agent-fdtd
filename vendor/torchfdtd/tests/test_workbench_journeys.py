"""The newest workbench journey record (scripts/run_workbench_journeys.py) is complete,
every journey passed on a CPU-only server, and the record describes the committed
specs and browser bundle (G8-03, G8-04).

The record is written by the runner on a clean tree; these tests fail when a
spec file or a committed browser asset changed since, so an edit to the
workbench needs a new record before its evidence counts.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'docs' / 'validation' / 'workbench'
JOURNEY = 'tests/ui/g8-journey.spec.js'
EDITING = 'tests/ui/g8-editing.spec.js'
JOURNEY_STEPS = ['gds_import', 'materials', 'source_monitor', 'boundaries', 'mesh_preview', 'preflight', 'cancel', 'rerun',
                 'results_overlay', 'exports']
EDITING_TITLES = ['undo and redo restore edits and the history is bounded',
                  'multi-select, duplicate all, copy, paste and delete all',
                  'autosave to browser storage and recovery after reload',
                  'the saved project carries a monotonic revision and a content hash the server confirms',
                  'non-finite and out-of-range values are refused with a visible message',
                  'results are marked stale when the plan changes after a run and current again when it is restored']


def _record():
    records = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in sorted(RECORDS.glob('*.json'))
               if not path.name.endswith('.playwright.json')]
    records = [(path, record) for path, record in records if record.get('kind') == 'workbench_journey_record']
    assert records, f'no workbench journey record under {RECORDS}; run scripts/run_workbench_journeys.py'
    return records[-1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tests(record, file):
    return [t for t in record['tests'] if t['file'].replace('\\', '/') == file]


def test_record_is_complete_and_every_journey_passed():
    path, record = _record()
    assert record['all_passed'] is True and record['playwright_exit_code'] == 0, path.name
    assert record['dirty_paths'] == [], 'the record must be taken on a clean tree'
    assert record['wall_seconds'] > 0 and record['server_started_here'] is True
    assert record['environment']['cuda_visible_to_server'] is False, 'the journeys must run against a CPU-only server'
    assert record['server_health']['cuda'] is False
    for entry in record['tests']:
        assert entry['status'] == 'expected' and entry['outcome'] == 'passed' and entry['attempts'] == 1, entry
        assert entry['duration_ms'] > 0 and entry['error'] is None, entry


def test_journey_covers_every_step_and_records_its_wall_time():
    path, record = _record()
    journeys = _tests(record, JOURNEY)
    assert len(journeys) == 1, [t['title'] for t in record['tests']]
    journey = journeys[0]
    timing = journey['journey_timing_ms']
    assert list(timing) == JOURNEY_STEPS + ['total_ms'], list(timing)
    laps = [timing[step] for step in JOURNEY_STEPS]
    assert laps == sorted(laps) and laps[0] > 0 and timing['total_ms'] >= laps[-1]
    assert journey['duration_ms'] >= timing['total_ms']


def test_editing_behaviours_are_each_verified():
    path, record = _record()
    titles = [t['title'] for t in _tests(record, EDITING)]
    assert titles == EDITING_TITLES, titles


def test_record_matches_the_committed_specs_and_bundle():
    path, record = _record()
    assert set(record['specs']) == {JOURNEY, EDITING}
    for spec, digest in record['specs'].items():
        assert _sha256(ROOT / spec) == digest, f'{spec} changed after {path.name}; run scripts/run_workbench_journeys.py again'
    web = ROOT / 'torchfdtd' / 'web'
    current = {str(p.relative_to(ROOT)).replace('\\', '/'): _sha256(p) for p in sorted(web.rglob('*')) if p.is_file()}
    assert record['bundle'] == current, f'the browser bundle changed after {path.name}; rebuild and record again'
    assert 'torchfdtd/web/index.html' in current
    report = ROOT / record['report']['path']
    assert report.is_file() and _sha256(report) == record['report']['sha256']


def test_record_commit_is_in_the_history_of_head():
    path, record = _record()
    completed = subprocess.run(['git', 'merge-base', '--is-ancestor', record['commit'], 'HEAD'], cwd=ROOT, capture_output=True)
    if completed.returncode not in (0, 1):
        pytest.skip('git is unavailable here: ' + completed.stderr.decode('utf-8', 'replace'))
    assert completed.returncode == 0, f"{record['commit']} of {path.name} is not an ancestor of HEAD"
