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
