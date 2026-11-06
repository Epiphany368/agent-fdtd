"""The newest clean-install record (scripts/clean_install_check.py) is complete and describes the current packaging inputs.

The record is written on a clean tree by the check script; it is stale, and
these tests fail, when a packaging input changed since: pyproject.toml, the
committed browser assets, the two scripts or a runnable README block. Solver
source changes do not invalidate it; the gate evidence of the physics tasks
covers those.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'docs' / 'validation' / 'clean_install'
sys.path.insert(0, str(ROOT / 'scripts'))
from clean_install_check import packaging_inputs_sha256  # noqa: E402
from run_readme_examples import parse_blocks, runnable_blocks_sha256  # noqa: E402


def _record():
    """The newest clean-install record; the directory also holds version-trial records of another kind."""
    records = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in sorted(RECORDS.glob('*.json'))]
    records = [(path, record) for path, record in records if record.get('kind') == 'clean_install_record']
    assert records, f'no clean-install record under {RECORDS}; run scripts/clean_install_check.py'
    path, record = records[-1]
    return record, path


def _committed_sha256(path):
    completed = subprocess.run(['git', 'show', f'HEAD:{path}'], cwd=ROOT, capture_output=True, check=True)
    return hashlib.sha256(completed.stdout).hexdigest()


def test_record_is_complete_and_every_step_passed():
    record, path = _record()
    assert record['kind'] == 'clean_install_record' and record['all_passed'] is True, path.name
    names = [step['name'] for step in record['steps']]
    for required in ('build_wheel', 'cpu_venv_create', 'cpu_pip_install_torch', 'cpu_pip_install_wheel', 'cpu_package_list',
                     'cpu_import_run_save_load', 'cpu_server_index_assets_api', 'cpu_doctor', 'cuda_venv_create',
                     'cuda_pip_install_torch', 'cuda_pip_install_wheel_extras', 'cuda_package_list',
                     'cuda_fused_forward_run', 'cuda_doctor', 'readme_examples'):
        assert required in names, f'{required} missing from {path.name}'
    assert all(step['status'] == 'passed' for step in record['steps']), [s for s in record['steps'] if s['status'] != 'passed']
    assert record['dirty_paths'] == [], 'the record must be taken on a clean tree'
    for name in ('g8-cpu', 'g8-cuda'):
        packages = record['environments'][name]['packages']
        assert any(p.startswith('torchfdtd==') for p in packages) and any(p.startswith('torch==') for p in packages)
    assert any(p.startswith('cupy-cuda12x==') for p in record['environments']['g8-cuda']['packages'])
    assert not any(p.startswith('cupy') for p in record['environments']['g8-cpu']['packages'])


def test_wheel_in_the_record_carries_the_committed_browser_assets():
    record, _ = _record()
    wheel = record['wheel']
    assets = wheel['web_assets']
    assert 'torchfdtd/web/index.html' in assets and any(name.startswith('torchfdtd/web/assets/') for name in assets)
    assert wheel['web_assets_match_committed'] is True and wheel['frontend_assets_current'] is True
    committed = {f'torchfdtd/web/{p.relative_to(ROOT / "torchfdtd" / "web").as_posix()}' for p in (ROOT / 'torchfdtd' / 'web').rglob('*') if p.is_file()}
    assert set(assets) == committed, 'the wheel must carry exactly the committed assets'
    for name, entry in assets.items():
        assert entry['sha256'] == _committed_sha256(name), f'{name} in the wheel differs from HEAD'


def test_record_matches_the_current_packaging_inputs_and_wheel():
    record, path = _record()
    assert record['packaging_inputs_sha256'] == packaging_inputs_sha256(), \
        f'{path.name} predates a change to pyproject.toml, torchfdtd/web or the clean-install scripts; rerun scripts/clean_install_check.py'
    known = subprocess.run(['git', 'cat-file', '-e', record['source_commit'] + '^{commit}'], cwd=ROOT, capture_output=True)
    if known.returncode == 0:  # a shallow clone cannot judge ancestry
        ancestry = subprocess.run(['git', 'merge-base', '--is-ancestor', record['source_commit'], 'HEAD'], cwd=ROOT, capture_output=True)
        assert ancestry.returncode == 0, f'record commit {record["source_commit"][:12]} is not an ancestor of HEAD'
    wheel = Path(record['wheel']['path'])
    if not wheel.is_file():
        pytest.skip(f'the recorded wheel is not on this host: {wheel}')
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == record['wheel']['sha256']


def test_every_runnable_readme_block_passed_on_the_installed_wheel():
    record, path = _record()
    blocks = parse_blocks((ROOT / 'README.md').read_text(encoding='utf-8'))
    runnable = [(b['mode'], b['sha256']) for b in blocks if b['mode'] in ('cpu', 'cuda')]
    assert runnable, 'README.md has no runnable python block'
    assert record['readme']['runnable_blocks_sha256'] == runnable_blocks_sha256(blocks), \
        f'{path.name} predates a change to a runnable README block; rerun scripts/clean_install_check.py'
    outcome = record['readme_examples']
    assert outcome['all_runnable_passed'] is True and outcome['runnable_blocks'] == len(runnable)
    recorded = {(b['mode'], b['sha256']): b for b in outcome['blocks'] if b['mode'] in ('cpu', 'cuda')}
    assert set(recorded) == set(runnable)
    for key, block in recorded.items():
        assert block['status'] == 'passed' and block['exit_code'] == 0, (key, block.get('stderr_tail'))
    # The record's paths belong to the recording host; parse them with that host's path class, not this host's.
    record_path = PureWindowsPath if record['host']['os'].startswith('Windows') else PurePosixPath
    for name, interpreter in outcome['interpreters'].items():
        assert record_path(interpreter['file']).is_relative_to(record_path(interpreter['prefix'])), name
        assert not record_path(interpreter['file']).is_relative_to(record_path(record['checkout'])), f'{name} imported torchfdtd from the checkout'


def test_doctor_reports_are_consistent_with_each_environment():
    record, _ = _record()
    cpu, cuda = record['doctor']['cpu'], record['doctor']['cuda']
    assert cpu['exit_code'] == 0 and cpu['report']['ok'] and cpu['report']['backend']['backend'] == 'cpu'
    assert cpu['report']['torch']['cuda_available'] is False and cpu['report']['fused_kernels']['status'] == 'not_applicable'
    assert cuda['exit_code'] == 0 and cuda['report']['ok'] and cuda['report']['backend'] == dict(
        backend='cuda', cuda_kernel='fused', reason='CUDA device and CuPy available')
    assert cuda['report']['fused_kernels']['status'] == 'ok' and cuda['report']['cupy']['importable'] is True


def test_readme_block_parser_reads_markers_and_languages():
    text = ('text\n```powershell\npip install x\n```\n\n<!-- readme-example: cuda -->\n```python\nimport torch\n```\n'
            '<!-- readme-example: skip: needs a licence -->\n```python\nimport vendor\n```\n```python\nprint(1)\n```\n')
    blocks = parse_blocks(text)
    assert [(b['language'], b['mode'], b['reason']) for b in blocks] == [
        ('powershell', 'not_python', None), ('python', 'cuda', None), ('python', 'skip', 'needs a licence'), ('python', 'cpu', None)]
    assert blocks[3]['code'] == 'print(1)\n' and blocks[1]['line'] == 7
    with pytest.raises(ValueError, match='never closed'):
        parse_blocks('```python\nprint(1)\n')


def test_block_timeout_is_recorded_as_failed_with_its_output(tmp_path, monkeypatch):
    """A block that exceeds the timeout yields a failed record with the captured output; the runner does not die."""
    import run_readme_examples as runner
    monkeypatch.setattr(runner, 'TIMEOUT_SECONDS', 2)
    monkeypatch.setattr(runner, 'check_interpreter', lambda python, workdir, checkout: dict(prefix=sys.prefix, file='stub', python='3'))
    blocks = parse_blocks("```python\nimport time\nprint('started', flush=True)\ntime.sleep(30)\n```\n```python\nprint('next')\n```\n")
    result = runner.run_blocks(blocks, sys.executable, workdir=tmp_path / 'work')
    timed_out, following = result['blocks']
    assert timed_out['status'] == 'failed' and timed_out['exit_code'] is None
    assert timed_out['stderr_tail'] == 'timeout after 2 s'
    # Windows may drop the partial pipe contents when the child is killed at the timeout.
    assert timed_out['stdout_tail'] in ('started\n', '')
    assert following['status'] == 'passed' and following['stdout_tail'] == 'next\n'
    assert result['runnable_blocks'] == 2 and result['all_runnable_passed'] is False


def test_local_root_defaults_outside_the_checkout_and_refuses_one_inside():
    """The probes assert that the installed package does not resolve under the checkout, so its environments live elsewhere."""
    import argparse
    import clean_install_check as check
    assert not check.DEFAULT_LOCAL_ROOT.resolve().is_relative_to(ROOT.resolve())
    with pytest.raises(SystemExit, match='inside the checkout'):
        check.Check(argparse.Namespace(local_root=str(ROOT / '.local')))
