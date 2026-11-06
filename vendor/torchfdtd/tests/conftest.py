"""Suite markers and the GPU-required policy shared by every test module.

Three suites are declared in docs/GPU_RUNNER_POLICY.md and launched by
scripts/run_suite.py:

  cpu-pr        -m "not cuda and not long"       CPU only, GitHub Actions
  gpu-nightly   -m "not long" --gpu-required     trusted commits on a lab GPU host
  release-full  --gpu-required, opt-in tests on  release candidates on a lab GPU host

Existing modules gate CUDA with inline skips rather than markers, so the
``cuda`` marker is applied at collection from these signals: an explicit
``@pytest.mark.cuda``; a ``skipif`` whose reason names CUDA, CuPy or a GPU; a
parameter value naming CUDA when the test source also contains a gate idiom
(``torch.cuda.is_available()``, ``importorskip('cupy')``, ``gpu()``,
``require_cuda()``), since a CUDA-parametrized test without any gate runs on
the CPU-only job today; or, without such a parameter, an unconditional gate
statement in the function body (``gpu()``, ``require_cuda()``,
``pytest.importorskip('cupy')`` or ``if not torch.cuda.is_available():
pytest.skip(...)`` at the top level of the body). A parameter value of ``cpu``
or ``numpy`` keeps that instance out of the marker.

``--gpu-required`` turns a skip whose reason names CUDA, CuPy or a GPU, or any
skip of a ``cuda``-marked test, into a failure unless the test carries the
``optional`` marker (mixed-platform checks such as the two-GPU NCCL case).
Skips of ``optional`` tests are reported with the prefix
``optional platform check:`` so scripts/record_gate_evidence.py can tell them
apart from a missing GPU.
"""
import ast
import inspect
import os
import re
import textwrap

import pytest
# The upstream grid package sets the Torch default dtype to float64 and disables
# autograd when it is imported; torchfdtd restores both only when it is the first
# importer. Sixteen test modules import fdtd before torchfdtd, so a subset that
# starts with one of them would run under the leaked defaults. Importing torchfdtd
# here, before any test module is collected, fixes the session defaults.
import torchfdtd  # noqa: E402,F401  (must precede any test module import of fdtd)
import sys as _sys
from pathlib import Path as _Path
# benchmarks/, examples/ and the tests package ship with the repository, not with the wheel. Appending the
# repository root after torchfdtd is imported keeps the installed package in charge while a run against the
# wheel still finds those repository-only modules.
_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
try:
    import record_output
except ImportError:  # this conftest is also copied into throwaway test directories by pytester-based tests
    record_output = None

pytest_plugins = ['pytester']

GPU_REASON = re.compile(r'cuda|cupy|gpu', re.IGNORECASE)
GATE_IDIOMS = re.compile(r"torch\.cuda\.is_available\(\)|importorskip\(['\"]cupy['\"]\)|(?<![\w.])gpu\(\)|require_cuda\(\)")
OPTIONAL_PREFIX = 'optional platform check: '


def pytest_addoption(parser):
    parser.addoption('--gpu-required', action='store_true', default=False,
                     help='fail instead of skipping a CUDA test whose GPU prerequisite is missing')


@pytest.fixture(scope='session', autouse=True)
def record_scratch(tmp_path_factory):
    """Records written by the test modules land here unless TORCHFDTD_WRITE_RECORDS=1 (tests/record_output.py)."""
    if record_output is not None:
        record_output.SCRATCH = tmp_path_factory.mktemp('records')


@pytest.fixture(autouse=True)
def simulated_cuda_memory(request, monkeypatch):
    """Tests that set torch.cuda.mem_get_info describe the device alone; the NVML reading of the host joins only under the nvml marker."""
    if request.node.get_closest_marker('nvml') is None:
        monkeypatch.setattr('torchfdtd.cuda_memory.nvml_free_bytes', lambda device: None)


def pytest_configure(config):
    # The server allows loopback Host headers only; the TestClient default host is admitted here, for the tests alone.
    os.environ.setdefault('TORCHFDTD_ALLOWED_HOSTS', 'testserver')
    config.addinivalue_line('markers', 'cuda: needs a CUDA device (and CuPy for fused kernels); deselected in cpu-pr, a skip fails under --gpu-required')
    config.addinivalue_line('markers', 'long: opt-in long or isolated test enabled only by the release-full suite')
    config.addinivalue_line('markers', 'nvml: reads the device-wide NVML free memory in CUDA admission (other tests see the runtime value only)')
    config.addinivalue_line('markers', 'optional: mixed-platform check whose skip is permitted in every suite (two-GPU NCCL, Gloo, licensed tools)')


def _parameter_verdict(item):
    """True for a CUDA instance, False for a CPU/numpy instance, None when parameters do not say."""
    spec = getattr(item, 'callspec', None)
    if spec is None:
        return None
    for name, value in spec.params.items():
        if isinstance(value, str) and 'cuda' in value.lower():
            return True
        if name.lower() in ('cuda', 'gpu') and isinstance(value, bool):
            return value
    if any(isinstance(value, str) and value.lower() in ('cpu', 'numpy') for value in spec.params.values()):
        return False
    return None


def _source(item):
    function = getattr(item, 'obj', None)
    if function is None:
        return ''
    try:
        return inspect.getsource(function)
    except (OSError, TypeError):
        return ''


def _unconditional_gate(source):
    """A gate call or a CUDA skip at the top level of the function body, not inside a branch."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if function is None:
        return False
    for node in function.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            arguments = [getattr(a, 'value', None) for a in node.value.args]
            if name in ('gpu', 'require_cuda') or (name.endswith('importorskip') and arguments[:1] == ['cupy']):
                return True
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'not torch.cuda.is_available()':
            if any('skip' in ast.unparse(statement) for statement in node.body):
                return True
    return False


def needs_cuda(item):
    if item.get_closest_marker('cuda') is not None:
        return True
    for mark in item.iter_markers('skipif'):
        reason = mark.kwargs.get('reason') or (mark.args[1] if len(mark.args) > 1 else '')
        if GPU_REASON.search(str(reason)):
            return True
    verdict = _parameter_verdict(item)
    if verdict is not None:
        return verdict and bool(GATE_IDIOMS.search(_source(item)))
    return _unconditional_gate(_source(item))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker('cuda') is None and needs_cuda(item):
            item.add_marker(pytest.mark.cuda)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or not isinstance(report.longrepr, tuple):
        return
    path, line, message = report.longrepr
    reason = message[len('Skipped: '):] if message.startswith('Skipped: ') else message
    if item.get_closest_marker('optional') is not None:
        report.longrepr = (path, line, 'Skipped: ' + OPTIONAL_PREFIX + reason)
        return
    if not item.config.getoption('--gpu-required'):
        return
    if item.get_closest_marker('cuda') is not None or GPU_REASON.search(reason):
        report.outcome = 'failed'
        report.longrepr = f'GPU-required test skipped under --gpu-required: {reason}'
