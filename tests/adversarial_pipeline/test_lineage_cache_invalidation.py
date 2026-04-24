"""Cache-key invalidation via engine snapshot refresh_pass.

Verifies that ``SharedBenchmarkFilteredLineageStage.compute_hash`` reads
``get_current_snapshot().refresh_pass`` and that the legacy classvar
(``_refresh_pass_token``) + its ``bump_refresh_pass`` classmethod are gone.
"""

from __future__ import annotations

import pytest

from gigaevo.adversarial.shared_benchmark_lineage import (
    SharedBenchmarkFilteredLineageStage,
)
from gigaevo.evolution.engine.snapshot import (
    EngineSnapshot,
    _reset_current_snapshot_for_tests,
    set_current_snapshot,
)
from gigaevo.programs.stages.common import CacheOnlyInput


@pytest.fixture(autouse=True)
def _reset():
    _reset_current_snapshot_for_tests()
    yield
    _reset_current_snapshot_for_tests()


@pytest.fixture
def stage_io_fixture() -> CacheOnlyInput:
    # LineageStage.InputsModel = CacheOnlyInput.
    # Any stable payload is fine — we only care about the rp-suffix.
    return CacheOnlyInput(cache_on="opponents-abc")


def test_compute_hash_reflects_current_refresh_pass(stage_io_fixture):
    set_current_snapshot(EngineSnapshot(refresh_pass=0))
    h0 = SharedBenchmarkFilteredLineageStage.compute_hash(stage_io_fixture)

    set_current_snapshot(EngineSnapshot(refresh_pass=1))
    h1 = SharedBenchmarkFilteredLineageStage.compute_hash(stage_io_fixture)

    assert h0 != h1
    assert h0.endswith(":rp0")
    assert h1.endswith(":rp1")


def test_compute_hash_stable_within_same_refresh_pass(stage_io_fixture):
    set_current_snapshot(EngineSnapshot(refresh_pass=2))
    h0 = SharedBenchmarkFilteredLineageStage.compute_hash(stage_io_fixture)
    h1 = SharedBenchmarkFilteredLineageStage.compute_hash(stage_io_fixture)
    assert h0 == h1
    assert h0.endswith(":rp2")


def test_classvar_is_gone():
    assert not hasattr(SharedBenchmarkFilteredLineageStage, "_refresh_pass_token")
    assert not hasattr(SharedBenchmarkFilteredLineageStage, "bump_refresh_pass")
