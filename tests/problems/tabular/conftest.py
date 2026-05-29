from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

_COMMON = Path(__file__).resolve().parents[3] / "problems" / "tabular" / "_common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))


@pytest.fixture(scope="session")
def data_root() -> Path:
    root = os.environ.get("GIGAEVO_TABULAR_DATA")
    if not root or not Path(root).is_dir():
        pytest.skip("GIGAEVO_TABULAR_DATA unset or not a directory")
    return Path(root)
