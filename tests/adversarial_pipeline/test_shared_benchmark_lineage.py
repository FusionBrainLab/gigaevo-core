"""Tests for SharedBenchmarkFilteredLineageStage (aggregator-DI rewrite).

Contract under test:
  - Stage takes a required ``aggregator: MetricsAggregator`` kwarg; ``None`` ⇒
    ``ValueError`` at construction; omitting the kwarg ⇒ ``TypeError``.
  - ``preprocess`` builds per-opponent *record lists* (dicts from the tracker)
    restricted to the shared-G set, and hands them to
    ``aggregator.aggregate(records, intrinsic)`` twice: once with the parent's
    own ``program.metrics`` as ``intrinsic``, once with the child's.
  - ``TransitionEvidence.shared_parent_metrics`` / ``.shared_child_metrics`` are
    exactly the aggregator output dicts.
  - ``per_metric_shared_count`` maps every aggregator output key to
    ``len(shared_opponent_ids)`` (uniform denominator — the per-metric
    filtering is the aggregator's concern, not this stage's).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.metrics.aggregators import (
    ConfigurableAggregator,
    ConstantSpec,
    IntrinsicSpec,
    ReduceSpec,
)
from gigaevo.programs.metrics.context import (
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import CacheOnlyInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _mk_program(
    pid: str,
    parent_ids: list[str] | None = None,
    metrics: dict[str, float] | None = None,
) -> Program:
    p = MagicMock(spec=Program)
    p.id = pid
    p.lineage = MagicMock()
    p.lineage.parents = parent_ids or []
    p.metrics = metrics if metrics is not None else {"fitness": 0.0, VALIDITY_KEY: 1.0}
    return p


def _mk_storage(prog_map: dict[str, Program]):
    storage = MagicMock()

    async def _mget(ids):
        return [prog_map[pid] for pid in ids if pid in prog_map]

    storage.mget = AsyncMock(side_effect=_mget)
    return storage


async def _record(tracker, d_id: str, g_id: str, record: dict[str, float]) -> None:
    """Record a per-opponent dict via the v4 tracker path.

    Uses ``record_batch`` so we flow through the same code path production
    uses; ``delta`` is required (drives d_wins / g_resisted routing) but the
    rest of the dict is passed through verbatim.
    """
    rec = dict(record)
    rec.setdefault("delta", 0.0)
    await tracker.record_batch([(d_id, g_id, rec)])


def _mk_aggregator(metrics_ctx: MetricsContext) -> ConfigurableAggregator:
    """A small real aggregator: ``fitness`` = mean(delta), ``is_valid`` = const 1.0.

    Both outputs are valid names for metrics_ctx, so MetricsFormatter
    downstream wouldn't choke.
    """
    return ConfigurableAggregator(
        outputs={
            "fitness": ReduceSpec(op="mean", field="delta"),
            VALIDITY_KEY: ConstantSpec(1.0),
        },
        invalid_defaults={"fitness": 0.0, VALIDITY_KEY: 0.0},
        metrics_context=metrics_ctx,
    )


def _mk_stage(
    tracker,
    metrics_ctx,
    storage,
    aggregator,
    *,
    min_shared: int = 1,
    inject_shared_evidence: bool = True,
):
    """Instantiate stage via ``__new__`` to bypass the LangGraphStage chain.

    The LineageStage base constructor expects an LLM + task_description +
    prompts_dir; wiring all that up here is pointless (the tests don't
    exercise the LLM call path) and brittle. Only the fields ``preprocess``
    reads are set manually.
    """
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )

    s = SharedBenchmarkFilteredLineageStage.__new__(SharedBenchmarkFilteredLineageStage)
    s._tracker = tracker
    s._aggregator = aggregator
    s._min_shared = min_shared
    s._inject_shared_evidence = inject_shared_evidence
    s._metrics_context = metrics_ctx
    s.storage = storage
    return s


# ---------------------------------------------------------------------------
# Subclass + construction contract
# ---------------------------------------------------------------------------


def test_is_subclass_of_lineage_stage():
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )
    from gigaevo.programs.stages.insights_lineage import LineageStage

    assert issubclass(SharedBenchmarkFilteredLineageStage, LineageStage)


class TestAggregatorDI:
    """The aggregator is required — no silent fallback."""

    def test_none_aggregator_raises_at_construction(self, tracker, metrics_ctx):
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        with pytest.raises(ValueError, match="aggregator.*required"):
            SharedBenchmarkFilteredLineageStage(
                tracker=tracker,
                aggregator=None,
                metrics_context=metrics_ctx,
                llm=MagicMock(),
                task_description="T",
                storage=MagicMock(),
            )

    def test_missing_aggregator_kwarg_raises(self, tracker, metrics_ctx):
        """aggregator is keyword-only — omitting it is a TypeError."""
        from gigaevo.adversarial.shared_benchmark_lineage import (
            SharedBenchmarkFilteredLineageStage,
        )

        with pytest.raises(TypeError):
            SharedBenchmarkFilteredLineageStage(  # type: ignore[call-arg]
                tracker=tracker,
                metrics_context=metrics_ctx,
                llm=MagicMock(),
                task_description="T",
                storage=MagicMock(),
            )


def test_min_shared_zero_rejected_at_init(tracker, metrics_ctx):
    """min_shared=0 still rejected at construction (carry-forward)."""
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )

    with pytest.raises(ValueError, match="min_shared must be >= 1"):
        SharedBenchmarkFilteredLineageStage(
            tracker=tracker,
            aggregator=_mk_aggregator(metrics_ctx),
            metrics_context=metrics_ctx,
            llm=MagicMock(),
            task_description="T",
            storage=MagicMock(),
            min_shared=0,
        )


# ---------------------------------------------------------------------------
# Aggregator invocation
# ---------------------------------------------------------------------------


class TestAggregatorInvocation:
    @pytest.mark.asyncio
    async def test_aggregator_called_on_shared_subset(self, tracker, metrics_ctx):
        """Aggregator is called exactly twice (parent, child) with records whose
        length equals |shared|."""
        child_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())
        g_shared_a, g_shared_b = str(uuid.uuid4()), str(uuid.uuid4())
        g_parent_only = str(uuid.uuid4())
        g_child_only = str(uuid.uuid4())

        for g in (g_shared_a, g_shared_b, g_child_only):
            await _record(tracker, child_id, g, {"delta": 0.1, VALIDITY_KEY: 1.0})
        for g in (g_shared_a, g_shared_b, g_parent_only):
            await _record(tracker, parent_id, g, {"delta": 0.05, VALIDITY_KEY: 1.0})

        parent_prog = _mk_program(
            parent_id, metrics={"fitness": 0.05, VALIDITY_KEY: 1.0}
        )
        storage = _mk_storage({parent_id: parent_prog})

        real_agg = _mk_aggregator(metrics_ctx)
        spy = MagicMock(wraps=real_agg)
        # Proxy the `output_keys` property read (MagicMock wrap doesn't carry it).
        spy.output_keys = real_agg.output_keys
        stage = _mk_stage(tracker, metrics_ctx, storage, spy)

        child_prog = _mk_program(child_id, [parent_id], metrics={"fitness": 0.1})
        result = await stage.preprocess(child_prog, CacheOnlyInput())

        assert isinstance(result, dict)
        # Aggregator called exactly twice: once for parent, once for child.
        assert spy.aggregate.call_count == 2
        for call in spy.aggregate.call_args_list:
            records = call.args[0]
            assert len(records) == 2  # |shared| == 2

    @pytest.mark.asyncio
    async def test_aggregator_receives_only_records_from_shared_g(
        self, tracker, metrics_ctx
    ):
        """Seed tracker so parent faced {g1,g2,g3}, child faced {g2,g3,g4};
        shared={g2,g3}. Every record passed to aggregator must be g2 or g3."""
        c = str(uuid.uuid4())
        p = str(uuid.uuid4())
        g1, g2, g3, g4 = [str(uuid.uuid4()) for _ in range(4)]

        # Mark records with a "tag" field that echoes which g they came from so
        # we can assert identity without dict-equality on floats.
        await _record(tracker, p, g1, {"delta": 0.01, "tag": g1, VALIDITY_KEY: 1.0})
        await _record(tracker, p, g2, {"delta": 0.02, "tag": g2, VALIDITY_KEY: 1.0})
        await _record(tracker, p, g3, {"delta": 0.03, "tag": g3, VALIDITY_KEY: 1.0})
        await _record(tracker, c, g2, {"delta": 0.20, "tag": g2, VALIDITY_KEY: 1.0})
        await _record(tracker, c, g3, {"delta": 0.30, "tag": g3, VALIDITY_KEY: 1.0})
        await _record(tracker, c, g4, {"delta": 0.40, "tag": g4, VALIDITY_KEY: 1.0})

        parent_prog = _mk_program(p, metrics={"fitness": 0.02, VALIDITY_KEY: 1.0})
        storage = _mk_storage({p: parent_prog})

        real_agg = _mk_aggregator(metrics_ctx)
        spy = MagicMock(wraps=real_agg)
        spy.output_keys = real_agg.output_keys
        stage = _mk_stage(tracker, metrics_ctx, storage, spy)

        await stage.preprocess(
            _mk_program(c, [p], metrics={"fitness": 0.25}), CacheOnlyInput()
        )

        shared_ids = {g2, g3}
        for call in spy.aggregate.call_args_list:
            records = call.args[0]
            tags = {r["tag"] for r in records}
            assert tags == shared_ids, (
                f"aggregator saw records tagged {tags}; expected only {shared_ids}"
            )

    @pytest.mark.asyncio
    async def test_aggregator_intrinsic_arg_uses_program_metrics(
        self, tracker, metrics_ctx
    ):
        """Parent call's intrinsic is parent.metrics; child call's is program.metrics."""
        c = str(uuid.uuid4())
        p = str(uuid.uuid4())
        g1 = str(uuid.uuid4())
        await _record(tracker, c, g1, {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, p, g1, {"delta": 0.1, VALIDITY_KEY: 1.0})

        parent_metrics = {"fitness": 0.111, VALIDITY_KEY: 1.0, "_side": "parent"}
        child_metrics = {"fitness": 0.999, VALIDITY_KEY: 1.0, "_side": "child"}

        parent_prog = _mk_program(p, metrics=parent_metrics)
        storage = _mk_storage({p: parent_prog})

        real_agg = _mk_aggregator(metrics_ctx)
        spy = MagicMock(wraps=real_agg)
        spy.output_keys = real_agg.output_keys
        stage = _mk_stage(tracker, metrics_ctx, storage, spy)

        child_prog = _mk_program(c, [p], metrics=child_metrics)
        await stage.preprocess(child_prog, CacheOnlyInput())

        # call.args[1] is the `intrinsic` positional arg.
        seen_sides = [
            call.args[1].get("_side") for call in spy.aggregate.call_args_list
        ]
        assert set(seen_sides) == {"parent", "child"}


# ---------------------------------------------------------------------------
# Evidence shape
# ---------------------------------------------------------------------------


class TestEvidenceShape:
    @pytest.mark.asyncio
    async def test_evidence_uses_aggregator_output_dicts(self, tracker, metrics_ctx):
        """Evidence's shared_*_metrics are *exactly* the aggregator output dicts."""
        c = str(uuid.uuid4())
        p = str(uuid.uuid4())
        g1 = str(uuid.uuid4())
        await _record(tracker, c, g1, {"delta": 0.4, VALIDITY_KEY: 1.0})
        await _record(tracker, p, g1, {"delta": 0.1, VALIDITY_KEY: 1.0})

        parent_prog = _mk_program(p, metrics={"fitness": 0.1, VALIDITY_KEY: 1.0})
        storage = _mk_storage({p: parent_prog})

        aggregator = ConfigurableAggregator(
            outputs={
                "fitness": ReduceSpec(op="mean", field="delta"),
                "marker": ConstantSpec(42.0),
            },
            invalid_defaults={"fitness": 0.0, "marker": 0.0},
            metrics_context=metrics_ctx,
        )
        stage = _mk_stage(tracker, metrics_ctx, storage, aggregator)

        child_prog = _mk_program(c, [p], metrics={"fitness": 0.4})
        result = await stage.preprocess(child_prog, CacheOnlyInput())

        ev = result["evidence"][0]
        assert ev.parent_id == p
        assert ev.shared_opponent_ids == [g1]
        assert ev.shared_parent_metrics == {
            "fitness": pytest.approx(0.1),
            "marker": 42.0,
        }
        assert ev.shared_child_metrics == {
            "fitness": pytest.approx(0.4),
            "marker": 42.0,
        }

    @pytest.mark.asyncio
    async def test_per_metric_shared_count_uniform(self, tracker, metrics_ctx):
        """per_metric_shared_count maps every output key to |shared| (uniform)."""
        c = str(uuid.uuid4())
        p = str(uuid.uuid4())
        g1, g2, g3 = [str(uuid.uuid4()) for _ in range(3)]
        for g in (g1, g2, g3):
            await _record(tracker, c, g, {"delta": 0.1, VALIDITY_KEY: 1.0})
            await _record(tracker, p, g, {"delta": 0.05, VALIDITY_KEY: 1.0})

        parent_prog = _mk_program(p, metrics={"fitness": 0.05, VALIDITY_KEY: 1.0})
        storage = _mk_storage({p: parent_prog})

        aggregator = ConfigurableAggregator(
            outputs={
                "fitness": ReduceSpec(op="mean", field="delta"),
                "actual_fitness": IntrinsicSpec("fitness", default=0.0),
                VALIDITY_KEY: ConstantSpec(1.0),
            },
            invalid_defaults={"fitness": 0.0, "actual_fitness": 0.0, VALIDITY_KEY: 0.0},
            metrics_context=metrics_ctx,
        )
        stage = _mk_stage(tracker, metrics_ctx, storage, aggregator)

        result = await stage.preprocess(
            _mk_program(c, [p], metrics={"fitness": 0.3}), CacheOnlyInput()
        )
        ev = result["evidence"][0]

        assert set(ev.per_metric_shared_count.keys()) == {
            "fitness",
            "actual_fitness",
            VALIDITY_KEY,
        }
        assert all(v == 3 for v in ev.per_metric_shared_count.values())


# ---------------------------------------------------------------------------
# Filter semantics (carry-forward)
# ---------------------------------------------------------------------------


class TestFilterSemantics:
    @pytest.mark.asyncio
    async def test_empty_shared_returns_skipped(self, tracker, metrics_ctx):
        """When no parent shares an opponent, preprocess returns skipped."""
        c, p = str(uuid.uuid4()), str(uuid.uuid4())
        await _record(tracker, c, "g1", {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, p, "g2", {"delta": 0.1, VALIDITY_KEY: 1.0})

        storage = _mk_storage({p: _mk_program(p)})
        stage = _mk_stage(tracker, metrics_ctx, storage, _mk_aggregator(metrics_ctx))

        result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
        assert isinstance(result, ProgramStageResult)
        storage.mget.assert_not_called()

    @pytest.mark.asyncio
    async def test_min_shared_enforced(self, tracker, metrics_ctx):
        """min_shared=2 with only 1 shared opponent drops the parent."""
        c, p = str(uuid.uuid4()), str(uuid.uuid4())
        await _record(tracker, c, "g1", {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, c, "g2", {"delta": 0.2, VALIDITY_KEY: 1.0})
        await _record(tracker, p, "g1", {"delta": 0.1, VALIDITY_KEY: 1.0})

        storage = _mk_storage({p: _mk_program(p)})
        stage = _mk_stage(
            tracker, metrics_ctx, storage, _mk_aggregator(metrics_ctx), min_shared=2
        )

        result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
        assert isinstance(result, ProgramStageResult)

    @pytest.mark.asyncio
    async def test_no_parents_returns_skipped(self, tracker, metrics_ctx):
        storage = _mk_storage({})
        stage = _mk_stage(tracker, metrics_ctx, storage, _mk_aggregator(metrics_ctx))
        stage._tracker = MagicMock()
        stage._tracker.metrics_by_d = AsyncMock(
            side_effect=AssertionError("no tracker call")
        )

        result = await stage.preprocess(
            _mk_program(str(uuid.uuid4()), []), CacheOnlyInput()
        )
        assert isinstance(result, ProgramStageResult)

    @pytest.mark.asyncio
    async def test_multiple_parents_mixed(self, tracker, metrics_ctx):
        c = str(uuid.uuid4())
        p_keep_a, p_keep_b, p_drop = (str(uuid.uuid4()) for _ in range(3))
        await _record(tracker, c, "g1", {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, c, "g2", {"delta": 0.4, VALIDITY_KEY: 1.0})
        await _record(tracker, p_keep_a, "g1", {"delta": 0.1, VALIDITY_KEY: 1.0})
        await _record(tracker, p_keep_b, "g2", {"delta": 0.2, VALIDITY_KEY: 1.0})
        await _record(tracker, p_drop, "g_other", {"delta": 0.05, VALIDITY_KEY: 1.0})

        prog_map = {
            p_keep_a: _mk_program(p_keep_a),
            p_keep_b: _mk_program(p_keep_b),
            p_drop: _mk_program(p_drop),
        }
        storage = _mk_storage(prog_map)
        stage = _mk_stage(tracker, metrics_ctx, storage, _mk_aggregator(metrics_ctx))
        result = await stage.preprocess(
            _mk_program(c, [p_keep_a, p_keep_b, p_drop]), CacheOnlyInput()
        )

        assert isinstance(result, dict)
        kept_parents = {p.id for p in result["parents"]}
        assert kept_parents == {p_keep_a, p_keep_b}
        assert {e.parent_id for e in result["evidence"]} == {p_keep_a, p_keep_b}


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


class TestInjectSharedEvidenceFalse:
    @pytest.mark.asyncio
    async def test_inject_shared_evidence_false_filters_but_no_evidence(
        self, tracker, metrics_ctx
    ):
        c, p = str(uuid.uuid4()), str(uuid.uuid4())
        await _record(tracker, c, "g1", {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, p, "g1", {"delta": 0.1, VALIDITY_KEY: 1.0})

        storage = _mk_storage({p: _mk_program(p)})
        stage = _mk_stage(
            tracker,
            metrics_ctx,
            storage,
            _mk_aggregator(metrics_ctx),
            inject_shared_evidence=False,
        )

        result = await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
        assert isinstance(result, dict)
        assert result["evidence"] is None
        assert len(result["parents"]) == 1


# ---------------------------------------------------------------------------
# Log emission
# ---------------------------------------------------------------------------


class TestLogEmission:
    @pytest.mark.asyncio
    async def test_kept_ratio_log(self, tracker, metrics_ctx):
        """The filter stage emits '[LineageStage:SharedBenchmark] ... kept X/Y ...'."""
        from loguru import logger

        c, p = str(uuid.uuid4()), str(uuid.uuid4())
        await _record(tracker, c, "g1", {"delta": 0.3, VALIDITY_KEY: 1.0})
        await _record(tracker, p, "g1", {"delta": 0.1, VALIDITY_KEY: 1.0})

        storage = _mk_storage({p: _mk_program(p)})
        stage = _mk_stage(tracker, metrics_ctx, storage, _mk_aggregator(metrics_ctx))

        captured: list[str] = []
        handler_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
        try:
            await stage.preprocess(_mk_program(c, [p]), CacheOnlyInput())
        finally:
            logger.remove(handler_id)

        assert any(
            "[LineageStage:SharedBenchmark]" in m and "kept 1/1" in m for m in captured
        ), f"Expected kept-ratio log line in captured logs: {captured}"
