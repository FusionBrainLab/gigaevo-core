"""Worker subprocess inherits intra-worker thread caps from the wrapper.

These caps stop concurrent mutants from oversubscribing the box when user
code calls `n_jobs=-1`. The OMP/MKL/OpenBLAS/NUMEXPR caps were already in
place; joblib (sklearn / XGBoost-via-joblib / LightGBM-via-joblib) resolves
`n_jobs=-1` through `loky.cpu_count()`, which respects `LOKY_MAX_CPU_COUNT`
before falling back to `os.cpu_count()`. Without that cap, n_jobs=-1 spawns
one thread per physical core regardless of OMP_NUM_THREADS.
"""

from __future__ import annotations

import pytest

from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner

_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


@pytest.fixture(autouse=True)
def _clear_ambient_caps(monkeypatch):
    """The wrapper uses ``setdefault`` so a user-set cap wins; clear ambient
    values so these tests exercise the default-injection path deterministically."""
    for _var in _CAP_VARS:
        monkeypatch.delenv(_var, raising=False)


_READ_CAPS = """
import os
def read_caps():
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT"),
    }
"""


class TestWorkerThreadCaps:
    async def test_loky_max_cpu_count_is_capped_in_worker(self, monkeypatch) -> None:
        monkeypatch.setenv("EVO_EXEC_THREADS", "11")

        result, _, _ = await run_exec_runner(
            code=_READ_CAPS, function_name="read_caps", timeout=30
        )

        assert result["LOKY_MAX_CPU_COUNT"] == "11"

    async def test_all_thread_caps_share_one_value(self, monkeypatch) -> None:
        monkeypatch.setenv("EVO_EXEC_THREADS", "7")

        result, _, _ = await run_exec_runner(
            code=_READ_CAPS, function_name="read_caps", timeout=30
        )

        assert result["OMP_NUM_THREADS"] == "7"
        assert result["MKL_NUM_THREADS"] == "7"
        assert result["OPENBLAS_NUM_THREADS"] == "7"
        assert result["NUMEXPR_NUM_THREADS"] == "7"
        assert result["LOKY_MAX_CPU_COUNT"] == "7"
