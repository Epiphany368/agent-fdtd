"""The completion program's documents of record stay consistent with the gate file."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gates():
    return json.loads((ROOT / 'docs/validation/completion_gates.json').read_text(encoding='utf-8'))


def test_program_text_matches_the_recorded_specification_hash():
    lines = (ROOT / 'docs/COMPLETION_PROGRAM_KO.md').read_bytes().split(b'\n')
    body = b'\n'.join(lines[3:])
    assert hashlib.sha256(body).hexdigest() == _gates()['master_prompt_sha256']


def test_gate_file_lists_every_stage_with_dependencies_and_profiles():
    gates = _gates()
    ids = [task['id'] for stage in gates['stages'] for task in stage['tasks']]
    assert len(ids) == gates['task_count'] == len(set(ids))
    stage_ids = {stage['id'] for stage in gates['stages']}
    for stage in gates['stages']:
        assert set(stage['depends_on_stages']) <= stage_ids
        assert stage['profile'] in gates['profiles']
    for profile in gates['profiles'].values():
        assert set(profile['required_stages']) <= stage_ids


def test_release_scope_separates_profiles_and_blocks_hpc_without_two_devices():
    text = (ROOT / 'docs/RELEASE_SCOPE.md').read_text(encoding='utf-8')
    assert 'WORKSTATION' in text and 'HPC' in text
    assert 'BLOCKED_EXTERNAL' in text
    for heading in ('## Physics', '## Platforms', '## Inputs and outputs', '## Differentiation', '## Capacity'):
        assert heading in text


def test_plan_points_to_the_program_and_handoff_has_the_required_sections():
    plan = (ROOT / 'docs/COMPLETION_PLAN_KO.md').read_text(encoding='utf-8')
    assert 'COMPLETION_PROGRAM_KO.md' in plan and 'completion_gates.json' in plan
    handoff = (ROOT / 'docs/DEVELOPMENT_HANDOFF.md').read_text(encoding='utf-8')
    for word in ('commit', 'command', 'evidence', 'blocker', 'next'):
        assert word in handoff.lower()
