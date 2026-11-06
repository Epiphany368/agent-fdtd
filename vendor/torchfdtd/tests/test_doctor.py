"""`torchfdtd doctor` reports the environment and names each unsupported situation."""
import io
import json

import pytest
import torch

from torchfdtd import doctor
from torchfdtd.cli import main


def _check(report, name):
    matches = [item for item in report['checks'] if item['name'] == name]
    assert len(matches) == 1, name
    return matches[0]


def _no_probe():
    raise AssertionError('the fused kernel probe must not run without a CUDA device')


def test_no_cuda_device_reports_cpu_execution_and_skips_the_kernel_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    monkeypatch.setattr(torch.version, 'cuda', None)
    report = doctor.diagnose(probe=_no_probe)
    assert report['torch']['cuda_available'] is False and report['device'] is None
    cuda = _check(report, 'cuda')
    assert cuda['status'] == 'notice' and 'no CUDA support' in cuda['message'] and 'CPU' in cuda['message']
    assert report['fused_kernels']['status'] == 'not_applicable'
    assert report['backend'] == dict(backend='cpu', cuda_kernel='torch', reason='no CUDA device')
    assert report['ok'] is True
    assert '[NOTICE] cuda:' in doctor.format_report(report)


def test_cuda_build_without_driver_names_the_driver(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    report = doctor.diagnose(probe=_no_probe)
    cuda = _check(report, 'cuda')
    assert cuda['status'] == 'notice' and 'no NVIDIA driver' in cuda['message'] and 'CUDA 12.6' in cuda['message']
    assert report['backend']['backend'] == 'cpu'


def test_cuda_build_with_driver_but_no_device_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: ('580.00', '13.0'))
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    report = doctor.diagnose(probe=_no_probe)
    cuda = _check(report, 'cuda')
    assert cuda['status'] == 'error' and 'no usable CUDA device' in cuda['message'] and '580.00' in cuda['message']
    assert report['ok'] is False


def test_old_driver_is_reported_against_the_torch_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: ('470.00', '11.4'))
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    report = doctor.diagnose(probe=_no_probe)
    driver = _check(report, 'driver')
    assert driver['status'] == 'error'
    assert 'supports CUDA 11.4 at most' in driver['message'] and 'CUDA 12.x' in driver['message']


def test_missing_cupy_is_a_notice_with_the_extra_to_install(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    monkeypatch.setattr(doctor, 'probe_cupy', lambda: dict(version=None, importable=False, error=None))
    report = doctor.diagnose(probe=_no_probe)
    cupy = _check(report, 'cupy')
    assert cupy['status'] == 'notice' and 'torchfdtd[cuda-kernels]' in cupy['message']


def test_broken_cupy_is_an_error_with_the_import_message(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    monkeypatch.setattr(doctor, 'probe_cupy',
                        lambda: dict(version='13.6.0', importable=False, error='ImportError: DLL load failed: nvrtc64_120_0.dll'))
    report = doctor.diagnose(probe=_no_probe)
    cupy = _check(report, 'cupy')
    assert cupy['status'] == 'error' and 'nvrtc64_120_0.dll' in cupy['message']
    assert report['ok'] is False


class _Distribution:
    def __init__(self, name, version):
        self.metadata, self.version = {'Name': name}, version


def _installed(monkeypatch, *distributions):
    """Replace the installed-distribution listing that probe_cupy reads."""
    monkeypatch.setattr(doctor.metadata, 'distributions',
                        lambda: [_Distribution(name, version) for name, version in distributions] + [_Distribution('numpy', '1.26.4')])


@pytest.mark.parametrize('name', ['cupy-cuda11x', 'cupy-cuda13x', 'cupy-rocm-5-0', 'cupy'])
def test_any_cupy_distribution_that_fails_to_import_is_an_error(monkeypatch, name):
    """A CuPy wheel other than cupy-cuda12x is found by the probe, and its import failure is not reported as absence."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    _installed(monkeypatch, (name, '12.3.0'))
    import torchfdtd.cuda_bootstrap as bootstrap
    def failing_import():
        raise RuntimeError('CUDA kernel dependency initialization failed. Import reported ImportError: DLL load failed: nvrtc64_111_0.dll')
    monkeypatch.setattr(bootstrap, 'prepare_cuda_kernels', failing_import)
    record = doctor.probe_cupy()
    assert record['version'] == '12.3.0' and record['distributions'] == {name: '12.3.0'} and record['importable'] is False
    assert 'nvrtc64_111_0.dll' in record['error']
    report = doctor.diagnose(probe=_no_probe)
    cupy = _check(report, 'cupy')
    assert cupy['status'] == 'error' and name in cupy['message'] and 'nvrtc64_111_0.dll' in cupy['message']
    assert report['ok'] is False


def test_no_cupy_distribution_at_all_stays_a_notice(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    _installed(monkeypatch)
    import torchfdtd.cuda_bootstrap as bootstrap
    def failing_import():
        raise RuntimeError("CUDA kernel dependency initialization failed. Import reported ModuleNotFoundError: No module named 'cupy'")
    monkeypatch.setattr(bootstrap, 'prepare_cuda_kernels', failing_import)
    record = doctor.probe_cupy()
    assert record['version'] is None and record['distributions'] == {} and record['importable'] is False
    cupy = _check(doctor.diagnose(probe=_no_probe), 'cupy')
    assert cupy['status'] == 'notice' and 'torchfdtd[cuda-kernels]' in cupy['message']


def test_kernel_compile_failure_is_reported_and_falls_back_to_the_torch_kernel(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (7 * 2**30, 8 * 2**30))
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda index: type('P', (), dict(name='Fake GPU', major=8, minor=6))())
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: ('591.86', '13.1'))
    monkeypatch.setattr(doctor, 'probe_cupy', lambda: dict(version='13.6.0', importable=True, error=None))

    def failing_probe():
        raise RuntimeError('CUDA kernel compilation failed: nvrtc: error: invalid value for --gpu-architecture')

    report = doctor.diagnose(probe=failing_probe)
    fused = _check(report, 'fused_kernels')
    assert fused['status'] == 'error' and '--gpu-architecture' in fused['message']
    assert report['fused_kernels']['status'] == 'error'
    assert report['backend'] == dict(backend='cuda', cuda_kernel='torch', reason='CUDA device available; fused kernels failed')
    assert report['device']['name'] == 'Fake GPU' and report['ok'] is False


def test_cli_doctor_prints_json_and_exits_with_the_report_status(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(doctor, '_nvidia_smi', lambda: (None, None))
    monkeypatch.setattr(torch.version, 'cuda', None)
    with pytest.raises(SystemExit) as exit_info:
        main(['doctor', '--json'])
    assert exit_info.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report['backend']['backend'] == 'cpu' and report['python'] and report['torch']['version']
    stream = io.StringIO()
    assert doctor.main(stream=stream) == 0
    assert stream.getvalue().startswith('torchfdtd ')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_real_fused_kernel_probe_launches_on_this_device():
    pytest.importorskip('cupy')
    report = doctor.diagnose()
    assert report['fused_kernels']['status'] == 'ok'
    assert report['fused_kernels']['detail'] == dict(cells=400, steps=20)
    assert report['backend'] == dict(backend='cuda', cuda_kernel='fused', reason='CUDA device and CuPy available')
    assert report['ok'] is True
