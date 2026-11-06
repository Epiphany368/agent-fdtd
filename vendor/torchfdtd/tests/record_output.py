"""Where the test modules write their JSON records.

The records under docs/validation/ are committed evidence, so refreshing them is
an explicit act: record_path() returns the committed path only when
TORCHFDTD_WRITE_RECORDS=1 is set and otherwise the same relative path under the
pytest session's temporary directory (tests/conftest.py fills SCRATCH from
tmp_path_factory), so a test still writes and re-reads its record without
touching the tracked file. The modules that read committed records for judging
keep reading the committed path directly.
"""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = None  # set by the session fixture in tests/conftest.py


def record_path(relative):
    """The committed record path under the repository root, or its stand-in under the temporary directory."""
    if os.environ.get('TORCHFDTD_WRITE_RECORDS') == '1':
        return ROOT / relative
    global SCRATCH
    if SCRATCH is None:
        SCRATCH = Path(tempfile.mkdtemp(prefix='torchfdtd-records-'))
    return SCRATCH / relative
