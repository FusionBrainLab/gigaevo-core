"""Tests for SharedBenchmarkFilteredLineageStage.

Behavior:
  - Filters parent→child transitions by shared-opponent intersection
    (|tracker.faced_by_d(child) ∩ faced_by_d(parent)| >= min_shared).
  - Builds TransitionEvidence per surviving parent with sentinel + is_valid
    gating via MetricsContext.is_valid / is_sentinel.
  - Returns ProgramStageResult.skipped (no LLM call) when no parents or
    no parents survive the filter.
  - Honors inject_shared_evidence flag to disable evidence emission while
    keeping filtering active (ablation).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.metrics.context import (
    MIN_VALUE_DEFAULT,
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import CacheOnlyInput


@pytest.fixture
async def tracker():
    t = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="test")
    t._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield t
    await t.close()


@pytest.fixture
def metrics_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            VALIDITY_KEY: MetricSpec(
                description="validity",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _mk_program(pid: str, parent_ids: list[str] | None = None) -> Program:
    p = MagicMock(spec=Program)
    p.id = pid
    p.lineage = MagicMock()
    p.lineage.parents = parent_ids or []
    return p


def _mk_storage(prog_map: dict[str, Program]):
    storage = MagicMock()

    async def _mget(ids):
        return [prog_map[pid] for pid in ids if pid in prog_map]

    storage.mget = AsyncMock(side_effect=_mget)
    return storage


async def _record(tracker, d_id: str, g_id: str, metrics: dict[str, float]) -> None:
    await tracker.record_metrics(d_id, g_id, metrics)


def _mk_stage(
    tracker,
    metrics_ctx,
    storage,
    *,
    min_shared: int = 1,
    inject_shared_evidence: bool = True,
):
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )

    s = SharedBenchmarkFilteredLineageStage.__new__(SharedBenchmarkFilteredLineageStage)
    s._tracker = tracker
    s._min_shared = min_shared
    s._inject_shared_evidence = inject_shared_evidence
    s._metrics_context = metrics_ctx
    s.storage = storage
    return s


# ---------------------------------------------------------------------------
# Subclass contract
# ---------------------------------------------------------------------------


def test_is_subclass_of_lineage_stage():
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )
    from gigaevo.programs.stages.insights_lineage import LineageStage

    assert issubclass(SharedBenchmarkFilteredLineageStage, LineageStage)


def test_min_shared_zero_rejected_at_init(tracker, metrics_ctx):
    """min_shared=0 keeps parents with empty shared sets and degrades the
    HoF-invariant evidence block to all-sentinel — the filter must refuse
    it at construction time, not silently emit garbage to the LLM."""
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )

    with pytest.raises(ValueError, match="min_shared must be >= 1"):
        SharedBenchmarkFilteredLineageStage(
            tracker=tracker,
            metrics_context=metrics_ctx,
            agent=MagicMock(),
            min_shared=0,
            storage=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Filter behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_keeps_parent_with_shared_benchmark(tracker, metrics_ctx):
    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    g1 = str(uuid.uuid4())
    await _record(tracker, child_id, g1, {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, parent_id, g1, {"fitness": 0.1, VALIDITY_KEY: 1.0})

    parent_prog = _mk_program(parent_id)
    storage = _mk_storage({parent_id: parent_prog})
    stage = _mk_stage(tracker, metrics_ctx, storage)

    program = _mk_program(child_id, [parent_id])
    result = await stage.preprocess(program, CacheOnlyInput())

    assert isinstance(result, dict)
    assert result["parents"] == [parent_prog]
    assert len(result["evidence"]) == 1
    ev = result["evidence"][0]
    assert ev.parent_id == parent_id
    assert ev.shared_opponent_ids == [g1]


@pytest.mark.asyncio
async def test_filter_drops_parent_with_no_shared_benchmark(tracker, metrics_ctx):
    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, parent_id, "g2", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    storage = _mk_storage({parent_id: _mk_program(parent_id)})
    stage = _mk_stage(tracker, metrics_ctx, storage)

    program = _mk_program(child_id, [parent_id])
    result = await stage.preprocess(program, CacheOnlyInput())

    assert isinstance(result, ProgramStageResult)
    storage.mget.assert_not_called()


@pytest.mark.asyncio
async def test_min_shared_respected(tracker, metrics_ctx):
    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, child_id, "g2", {"fitness": 0.2, VALIDITY_KEY: 1.0})
    await _record(tracker, parent_id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    storage = _mk_storage({parent_id: _mk_program(parent_id)})
    stage = _mk_stage(tracker, metrics_ctx, storage, min_shared=2)

    program = _mk_program(child_id, [parent_id])
    result = await stage.preprocess(program, CacheOnlyInput())

    assert isinstance(result, ProgramStageResult)


@pytest.mark.asyncio
async def test_no_parents_returns_skipped(tracker, metrics_ctx):
    storage = _mk_storage({})
    stage = _mk_stage(tracker, metrics_ctx, storage)
    stage._tracker = MagicMock()
    stage._tracker.metrics_by_d = AsyncMock(
        side_effect=AssertionError("no tracker call")
    )

    result = await stage.preprocess(
        _mk_program(str(uuid.uuid4()), []), CacheOnlyInput()
    )
    assert isinstance(result, ProgramStageResult)


# ---------------------------------------------------------------------------
# Evidence aggregation (sentinel + is_valid semantics)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evidence_means_exclude_invalid_g(tracker, metrics_ctx):
    """G where parent or child has is_valid=0 is dropped from per-metric means."""
    c, p = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, c, "g1", {"fitness": 0.40, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g1", {"fitness": 0.10, VALIDITY_KEY: 1.0})
    await _record(tracker, c, "g2", {"fitness": MIN_VALUE_DEFAULT, VALIDITY_KEY: 0.0})
    await _record(tracker, p, "g2", {"fitness": 0.20, VALIDITY_KEY: 1.0})

    storage = _mk_storage({p: _mk_program(p)})
    stage = _mk_stage(tracker, metrics_ctx, storage)
    result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())

    ev = result["evidence"][0]
    assert ev.shared_parent_metrics["fitness"] == pytest.approx(0.10)
    assert ev.shared_child_metrics["fitness"] == pytest.approx(0.40)
    assert ev.per_metric_shared_count["fitness"] == 1
    assert ev.shared_child_metrics[VALIDITY_KEY] == pytest.approx(0.5)
    assert ev.per_metric_shared_count[VALIDITY_KEY] == 2


@pytest.mark.asyncio
async def test_evidence_means_exclude_sentinels(tracker, metrics_ctx):
    """A valid row whose metric value is a sentinel is still dropped for that metric."""
    c, p = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, c, "g1", {"fitness": 0.40, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g1", {"fitness": 0.10, VALIDITY_KEY: 1.0})
    await _record(tracker, c, "g2", {"fitness": MIN_VALUE_DEFAULT, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g2", {"fitness": 0.20, VALIDITY_KEY: 1.0})

    storage = _mk_storage({p: _mk_program(p)})
    stage = _mk_stage(tracker, metrics_ctx, storage)
    result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())

    ev = result["evidence"][0]
    assert ev.per_metric_shared_count["fitness"] == 1
    assert ev.shared_child_metrics[VALIDITY_KEY] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_all_pairs_sentinel_emits_sentinel_value(tracker, metrics_ctx):
    """If no G pairs survive for a metric, both sides render as the sentinel."""
    c, p = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, c, "g1", {"fitness": MIN_VALUE_DEFAULT, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g1", {"fitness": MIN_VALUE_DEFAULT, VALIDITY_KEY: 1.0})

    storage = _mk_storage({p: _mk_program(p)})
    stage = _mk_stage(tracker, metrics_ctx, storage)
    result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())

    ev = result["evidence"][0]
    assert ev.per_metric_shared_count["fitness"] == 0
    assert metrics_ctx.is_sentinel("fitness", ev.shared_parent_metrics["fitness"])
    assert metrics_ctx.is_sentinel("fitness", ev.shared_child_metrics["fitness"])


@pytest.mark.asyncio
async def test_multiple_parents_mixed(tracker, metrics_ctx):
    c = str(uuid.uuid4())
    p_keep_a, p_keep_b, p_drop = (str(uuid.uuid4()) for _ in range(3))
    await _record(tracker, c, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, c, "g2", {"fitness": 0.4, VALIDITY_KEY: 1.0})
    await _record(tracker, p_keep_a, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})
    await _record(tracker, p_keep_b, "g2", {"fitness": 0.2, VALIDITY_KEY: 1.0})
    await _record(tracker, p_drop, "g_other", {"fitness": 0.05, VALIDITY_KEY: 1.0})

    prog_map = {
        p_keep_a: _mk_program(p_keep_a),
        p_keep_b: _mk_program(p_keep_b),
        p_drop: _mk_program(p_drop),
    }
    storage = _mk_storage(prog_map)
    stage = _mk_stage(tracker, metrics_ctx, storage)
    result = await stage.preprocess(
        _mk_program(c, [p_keep_a, p_keep_b, p_drop]), CacheOnlyInput()
    )

    assert isinstance(result, dict)
    kept_parents = {p.id for p in result["parents"]}
    assert kept_parents == {p_keep_a, p_keep_b}
    assert {e.parent_id for e in result["evidence"]} == {p_keep_a, p_keep_b}


# ---------------------------------------------------------------------------
# Ablation: inject_shared_evidence=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_shared_evidence_false_filters_but_no_evidence(
    tracker, metrics_ctx
):
    c, p = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, c, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    storage = _mk_storage({p: _mk_program(p)})
    stage = _mk_stage(tracker, metrics_ctx, storage, inject_shared_evidence=False)

    result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
    assert isinstance(result, dict)
    assert result["evidence"] is None
    assert len(result["parents"]) == 1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_logs_kept_ratio(tracker, metrics_ctx):
    """The filter stage emits '[LineageStage:SharedBenchmark] ... kept X/Y ...'."""
    from loguru import logger

    c, p = str(uuid.uuid4()), str(uuid.uuid4())
    await _record(tracker, c, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await _record(tracker, p, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    storage = _mk_storage({p: _mk_program(p)})
    stage = _mk_stage(tracker, metrics_ctx, storage)

    captured: list[str] = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    try:
        await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
    finally:
        logger.remove(handler_id)

    assert any(
        "[LineageStage:SharedBenchmark]" in m and "kept 1/1" in m for m in captured
    ), f"Expected kept-ratio log line in captured logs: {captured}"


# ---------------------------------------------------------------------------
# Refresh-pass cache-invariant
#
# The two-sided cross-program tracker race (pass 1 re-runs DGTrackerStage;
# pass 2 re-runs this stage against the globally-fresh tracker) only closes
# if pass 2 actually cache-misses on the stage.  Counting token bumps isn't
# enough — the token must change the cache KEY.  These tests prove the
# mechanism end-to-end:
#   - compute_hash(params) differs across bump_refresh_pass()
#   - compute_hash (execution path) == compute_hash_from_inputs (cache-check path)
#   - None base hash propagates rather than being stringified
# ---------------------------------------------------------------------------


class TestRefreshPassTokenCacheInvariant:
    """Proves the token mechanism actually invalidates the cache key.

    `_refresh_pass_token` is class-level state shared across tests in the
    process — `_reset_token` snapshots and restores so tests are order-
    independent.
    """

    @pytest.fixture(autouse=True)
    def _reset_token(self):
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        saved = SharedBenchmarkFilteredLineageStage._refresh_pass_token
        yield
        SharedBenchmarkFilteredLineageStage._refresh_pass_token = saved

    def test_compute_hash_differs_after_bump(self):
        """Same params, different token ⇒ different cache key.

        Without this, pass 2 of the engine's two-pass refresh would
        cache-HIT on pass 1's stage output and never re-read the tracker
        data that pass 1 wrote.  This is the load-bearing invariant of
        the whole refresh_passes=2 design.
        """
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        params = CacheOnlyInput(cache_on="same")
        h_pass1 = SharedBenchmarkFilteredLineageStage.compute_hash(params)
        SharedBenchmarkFilteredLineageStage.bump_refresh_pass()
        h_pass2 = SharedBenchmarkFilteredLineageStage.compute_hash(params)

        assert h_pass1 != h_pass2, (
            "compute_hash returned the same key before and after "
            "bump_refresh_pass() — cache would HIT on pass 2 and the "
            "two-pass refresh would be a no-op semantically."
        )

    def test_compute_hash_suffix_encodes_current_token(self):
        """Hash format is '<base>:rp<N>' — stable suffix so a grep for
        `:rp1` vs `:rp2` in Redis keys can confirm pass identity post-hoc."""
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        params = CacheOnlyInput(cache_on="x")
        t_before = SharedBenchmarkFilteredLineageStage._refresh_pass_token
        h_before = SharedBenchmarkFilteredLineageStage.compute_hash(params)
        assert h_before is not None
        assert h_before.endswith(f":rp{t_before}")

        SharedBenchmarkFilteredLineageStage.bump_refresh_pass()
        h_after = SharedBenchmarkFilteredLineageStage.compute_hash(params)
        assert h_after.endswith(f":rp{t_before + 1}")

    def test_execution_and_cache_check_paths_stay_in_lockstep(self):
        """compute_inputs_hash (taken at execution time) and
        compute_hash_from_inputs (taken at cache-check time, without
        instantiating the stage) must return the SAME value.  Drift
        between the two is the classic silent-cache-bug: the stage
        runs under key A, the cache is probed with key B ⇒ permanent
        cache miss OR worse, stale-read from a different program's cache.
        """
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        params = CacheOnlyInput(cache_on="probe")
        raw = {"cache_on": "probe"}

        h_exec = SharedBenchmarkFilteredLineageStage.compute_hash(params)
        h_check = SharedBenchmarkFilteredLineageStage.compute_hash_from_inputs(raw)
        assert h_exec == h_check

        SharedBenchmarkFilteredLineageStage.bump_refresh_pass()
        h_exec_b = SharedBenchmarkFilteredLineageStage.compute_hash(params)
        h_check_b = SharedBenchmarkFilteredLineageStage.compute_hash_from_inputs(raw)
        assert h_exec_b == h_check_b
        assert h_exec != h_exec_b, "token bump must change both paths together"

    def test_compute_hash_returns_none_when_base_returns_none(self):
        """Subclass must propagate a None base hash, not stringify it.

        A stringified None (`"None:rp3"`) would be a valid Redis key and
        silently collide across programs whose inputs fail to validate.
        """
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        h = SharedBenchmarkFilteredLineageStage.compute_hash_from_inputs(
            {"nonexistent_field": object()}
        )
        assert h is None

    def test_bump_returns_new_token_value(self):
        """bump_refresh_pass returns the new token — callers (the engine
        logs this for post-hoc verification) rely on the return value."""
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        before = SharedBenchmarkFilteredLineageStage._refresh_pass_token
        returned = SharedBenchmarkFilteredLineageStage.bump_refresh_pass()
        assert returned == before + 1
        assert SharedBenchmarkFilteredLineageStage._refresh_pass_token == before + 1
