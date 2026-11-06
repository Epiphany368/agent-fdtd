"""The test session keeps Torch's defaults: float32 and autograd enabled.

The upstream grid package changes both at import; tests/conftest.py imports
torchfdtd first so that every module, in any order, sees the same defaults.
"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_session_defaults_are_float32_and_grad_enabled():
    assert torch.get_default_dtype() == torch.float32
    assert torch.is_grad_enabled()


def test_an_fdtd_first_module_does_not_leak_into_later_modules(tmp_path):
    """A subset that starts with a module importing fdtd before torchfdtd keeps the defaults."""
    (tmp_path / 'conftest.py').write_text((ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8'), encoding='utf-8')
    (tmp_path / 'test_a_fdtd_first.py').write_text(
        'import fdtd\nimport torchfdtd\n\n\ndef test_first():\n    assert fdtd.backend is not None\n', encoding='utf-8')
    (tmp_path / 'test_b_defaults.py').write_text(
        'import torch\n\n\ndef test_defaults():\n    assert torch.get_default_dtype() == torch.float32\n    assert torch.is_grad_enabled()\n',
        encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(tmp_path)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
