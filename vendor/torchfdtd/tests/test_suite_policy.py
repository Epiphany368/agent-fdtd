"""G4-05: the three declared suites, the cuda marker rules and the --gpu-required policy."""
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFTEST = (ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8').replace("pytest_plugins = ['pytester']\n", '')

POLICY_TESTS = '''
import pytest
import torch


def test_plain():
    assert True


def test_inline_cuda_skip():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


@pytest.mark.optional
def test_optional_two_gpu():
    pytest.skip('Requires at least two visible CUDA GPUs and NCCL')


def test_ram_skip():
    pytest.skip('needs a few GiB of free RAM')


@pytest.mark.cuda
def test_marked_cuda_skips_for_another_reason():
    pytest.skip('fixture file missing')


@pytest.mark.skipif(True, reason='Local CUDA unavailable')
def test_skipif_cuda():
    assert True


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_device_with_gate(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_device_metadata_only(device):
    assert device in ('cpu', 'cuda')


def gpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


def test_unconditional_gate():
    gpu()


def test_conditional_gate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        pytest.importorskip('cupy')
'''


@pytest.fixture
def policy_dir(pytester, monkeypatch):
    # A rootdir of its own keeps the repository pyproject out; UTF-8 child output survives non-ASCII temp paths.
    pytester.makepyprojecttoml('[tool.pytest.ini_options]\n')
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_policy=POLICY_TESTS)
    monkeypatch.setenv('PYTHONIOENCODING', 'utf-8')
    return pytester


def collected(pytester, *args):
    result = pytester.runpytest_subprocess('-p', 'no:cacheprovider', '--collect-only', '-q', *args)
    return sorted(line.split('::')[-1] for line in result.outlines if line.startswith('test_policy.py::'))


def test_cuda_marker_rules(policy_dir):
    marked = collected(policy_dir, '-m', 'cuda')
    assert marked == ['test_device_with_gate[cuda]', 'test_inline_cuda_skip', 'test_marked_cuda_skips_for_another_reason',
                      'test_skipif_cuda', 'test_unconditional_gate']
    unmarked = collected(policy_dir, '-m', 'not cuda')
    assert unmarked == ['test_conditional_gate', 'test_device_metadata_only[cpu]', 'test_device_metadata_only[cuda]',
                        'test_device_with_gate[cpu]', 'test_optional_two_gpu', 'test_plain', 'test_ram_skip']


def test_gpu_required_turns_cuda_skips_into_failures(policy_dir, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    junit = policy_dir.path / 'junit.xml'
    result = policy_dir.runpytest_subprocess('-p', 'no:cacheprovider', '-q', '--gpu-required', '--junitxml', str(junit), '-k', 'not device_with_gate')
    result.assert_outcomes(passed=4, skipped=2, failed=3, errors=1)
    outcomes = {}
    for case in ET.parse(junit).getroot().iter('testcase'):
        child = next(iter(case), None)
        outcomes[case.get('name')] = (child.tag if child is not None else 'passed', (child.get('message') if child is not None else ''))
    assert outcomes['test_plain'] == ('passed', '')
    assert outcomes['test_ram_skip'][0] == 'skipped'
    assert outcomes['test_optional_two_gpu'] == ('skipped', 'optional platform check: Requires at least two visible CUDA GPUs and NCCL')
    assert outcomes['test_inline_cuda_skip'][0] == 'failure' and 'GPU-required test skipped under --gpu-required: CUDA unavailable' in outcomes['test_inline_cuda_skip'][1]
    assert outcomes['test_marked_cuda_skips_for_another_reason'][0] == 'failure' and 'fixture file missing' in outcomes['test_marked_cuda_skips_for_another_reason'][1]
    assert outcomes['test_skipif_cuda'][0] == 'error' and 'Local CUDA unavailable' in outcomes['test_skipif_cuda'][1]
    assert outcomes['test_unconditional_gate'][0] == 'failure'


def test_without_gpu_required_cuda_skips_stay_skips(policy_dir, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    result = policy_dir.runpytest_subprocess('-p', 'no:cacheprovider', '-q', '-k', 'not device_with_gate')
    result.assert_outcomes(passed=4, skipped=6)


def dry_run(suite, *extra):
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run_suite.py'), suite, '--dry-run', *extra],
                               capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert completed.returncode == 0, completed.stderr
    lines = dict(line.split(': ', 1) for line in completed.stdout.strip().splitlines())
    return lines['command'], lines['environment']


def test_cpu_pr_suite_hides_cuda_and_deselects_cuda_and_long_tests():
    command, environment = dry_run('cpu-pr', '--junitxml=out.xml')
    assert '-m "not cuda and not long"' in command and '--gpu-required' not in command and command.endswith('--junitxml=out.xml')
    assert "CUDA_VISIBLE_DEVICES='-1'" in environment


def test_gpu_nightly_suite_requires_the_gpu_and_excludes_long_tests():
    command, environment = dry_run('gpu-nightly')
    assert '-m "not long"' in command and '--gpu-required' in command
    assert environment == 'inherited'


def test_release_full_suite_enables_opt_in_tests():
    command, environment = dry_run('release-full')
    assert '--gpu-required' in command and ' -m not ' not in command
    assert "TORCHFDTD_RUN_CUDA_BOOTSTRAP_TEST='1'" in environment and "TORCHFDTD_RUN_CPML_KERNEL_CUDA_TEST='1'" in environment


def test_gpu_suites_refuse_to_run_without_a_cuda_device(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run_suite.py'), 'gpu-nightly', '--collect-only'],
                               capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert completed.returncode == 2 and 'requires a CUDA device' in completed.stdout


def test_markers_are_declared_in_pyproject():
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    for marker in ('cuda', 'long', 'optional'):
        assert re.search(rf'^\s*"{marker}: ', text, re.MULTILINE), marker
