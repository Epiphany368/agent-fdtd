"""Run the approved G7-01r4 workflow with an installed release wheel.

Invoke by absolute path from a working directory outside the checkout. Import
the installed package before exposing the repository-only examples package.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torchfdtd

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(torchfdtd.__file__).resolve()
if ROOT == PACKAGE or ROOT in PACKAGE.parents or 'site-packages' not in PACKAGE.parts:
    raise RuntimeError(f'G7-01r4 requires an installed wheel, got {PACKAGE}.')
sys.path.append(str(ROOT))

from examples.g7.metagrating.workflow_r4 import main  # noqa: E402


if __name__ == '__main__':
    main()
