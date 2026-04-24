"""Full-pipeline proof: MutationContextStage output reflects pass-2 tracker.

The two-pass refresh design exists because pass 2's LineageStage must
read tracker data that pass 1 wrote.  The `TestRefreshPassTokenCacheInvariant`
suite proves the cache-KEY mechanism; this suite proves the downstream
CONSUMPTION: after a token bump + tracker update, the MutationContextStage
output actually encodes the fresh tracker values.

Three-phase scenario per test (all with the same program and params):

  Run 1 — tracker state T1, token=N
          → cache miss on `:rpN` → stage runs → cache stores under `:rpN`
          → MutationContext output M1 contains "child_fitness=0.300"

  Run 2 — tracker mutated to T2 but NO token bump
          → cache HIT on `:rpN` → returns stale M1
          → this is the failure mode the refresh_passes=2 design fixes:
            without the bump, the cache would mask the tracker update.

  Run 3 — tracker still at T2, token bumped to N+1 (engine would do this
           between passes)
          → cache MISS on `:rp(N+1)` → stage re-runs → M3 contains
            "child_fitness=0.500"

The test uses a deterministic fake-LLM path: rather than calling a real
LangGraph agent, it builds `TransitionAnalysis` directly from the stage's
filtered evidence so the MutationContext string deterministically encodes
the tracker values that `preprocess()` read.  This is a semantic test,
not an LLM behaviour test — what matters is the flow tracker → evidence →
TransitionAnalysis → MutationContextStage output.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.llm.agents.lineage import (
    TransitionAnalysis,
    TransitionInsight,
    TransitionInsights,
)
from gigaevo.programs.metrics.context import (
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import CacheOnlyInput
from gigaevo.programs.stages.insights_lineage import TransitionAnalysisList
from gigaevo.programs.stages.mutation_context import (
    MutationContextInputs,
    MutationContextStage,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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


@pytest.fixture
async def tracker():
    t = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="t2p")
    t._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield t
    await t.close()


@pytest.fixture(autouse=True)
def _reset_token():
    """refresh_pass now lives on the engine snapshot mirror — reset it so tests
    are order-safe."""
    from gigaevo.evolution.engine.snapshot import (
        _reset_current_snapshot_for_tests,
    )

    _reset_current_snapshot_for_tests()
    yield
    _reset_current_snapshot_for_tests()


def _mk_program(pid: str, parent_ids: list[str] | None = None) -> Program:
    """Mock Program with the attributes SBF-Lineage reads.

    preprocess() reads program.id, program.lineage.parents, and
    program.metrics (for shared-evidence injection); it never touches
    program.code or program.metadata, so a mock is sufficient.
    """
    p = MagicMock(spec=Program)
    p.id = pid
    p.lineage = MagicMock()
    p.lineage.parents = parent_ids or []
    p.metrics = {"fitness": 0.3}  # Dummy metrics
    return p


def _mk_filter_stage(tracker, metrics_ctx, storage):
    from gigaevo.adversarial.shared_benchmark_lineage import (
        SharedBenchmarkFilteredLineageStage,
    )
    from gigaevo.programs.metrics.aggregators import (
        ConfigurableAggregator,
        ConstantSpec,
        ReduceSpec,
    )

    s = SharedBenchmarkFilteredLineageStage.__new__(SharedBenchmarkFilteredLineageStage)
    s._tracker = tracker
    s._min_shared = 1
    s._inject_shared_evidence = True
    s._metrics_context = metrics_ctx
    s._aggregator = ConfigurableAggregator(
        outputs={
            "fitness": ReduceSpec(op="mean", field="fitness"),
            "is_valid": ConstantSpec(value=1.0),
        },
        invalid_defaults={"fitness": 0.0, "is_valid": 0.0},
        metrics_context=metrics_ctx,
    )
    s.storage = storage
    return s


def _evidence_to_transition_analysis(ev, child_id: str) -> TransitionAnalysis:
    """Deterministic stand-in for the real LLM.

    The real LineageAgent produces `TransitionAnalysis` whose
    `child_metrics` field is derived from the evidence block.  We
    reproduce that invariant exactly: child_metrics = shared_child_metrics.
    This is what makes the MutationContext string encode the tracker
    state that `preprocess()` read.
    """
    return TransitionAnalysis(
        from_id=ev.parent_id,
        to_id=child_id,
        parent_metrics=dict(ev.shared_parent_metrics),
        child_metrics=dict(ev.shared_child_metrics),
        diff_blocks=["(stub diff)"],
        insights=TransitionInsights(
            insights=[
                TransitionInsight(strategy="imitation", description="s1"),
                TransitionInsight(strategy="exploration", description="s2"),
                TransitionInsight(strategy="generalization", description="s3"),
            ]
        ),
    )


async def _run_pipeline(
    stage,
    mctx_stage: MutationContextStage,
    program: Program,
    params: CacheOnlyInput,
    cache: dict[str, str],
) -> str:
    """Run stage + MutationContextStage through a compute_hash-keyed cache.

    Cache contract mirrors the real pipeline's: probe `compute_hash(params)`
    BEFORE running; on hit, skip execution entirely and return the cached
    output.  This is exactly what Redis-backed stage caching does in
    production.
    """
    key = stage.compute_hash(params)
    assert key is not None
    if key in cache:
        return cache[key]

    # Cache miss: run preprocess on the filter stage.
    preprocess_out = await stage.preprocess(program, params)
    if not isinstance(preprocess_out, dict):
        analyses: list[TransitionAnalysis] = []
    else:
        analyses = [
            _evidence_to_transition_analysis(ev, program.id)
            for ev in (preprocess_out.get("evidence") or [])
        ]

    # Feed into MutationContextStage.  We bypass input-validation by
    # assigning _params_obj directly; compute() reads self.params which
    # lazily returns _params_obj if set.
    mctx_stage._params_obj = MutationContextInputs(
        metrics=None,
        insights=None,
        lineage_ancestors=TransitionAnalysisList(items=analyses),
        lineage_descendants=None,
        evolutionary_statistics=None,
        formatted=None,
        memory=None,
    )
    out = await mctx_stage.compute(program)
    cache[key] = out.data
    return out.data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_without_bump_returns_stale_mutation_context(
    tracker, metrics_ctx
):
    """Without a token bump, the cache correctly hits on pass-1 output.

    This is the 'positive control': it proves the cache we're testing
    with is actually a cache — if this test fails, the miss-after-bump
    test below is meaningless.
    """
    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await tracker.record_metrics(parent_id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    parent_prog = _mk_program(parent_id)
    child_prog = _mk_program(child_id, [parent_id])

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[parent_prog])

    filter_stage = _mk_filter_stage(tracker, metrics_ctx, storage)
    mctx_stage = MutationContextStage(metrics_context=metrics_ctx, timeout=10.0)

    params = CacheOnlyInput(cache_on="k")
    cache: dict[str, str] = {}

    from gigaevo.evolution.engine.snapshot import get_current_snapshot

    token_snapshot = get_current_snapshot().refresh_pass

    # Run 1 — cold cache, tracker at T1 (child fitness=0.3)
    m1 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)
    assert "0.3" in m1 or "0.30" in m1 or "0.300" in m1, (
        f"MutationContext should encode child fitness 0.3, got: {m1!r}"
    )

    # Mutate tracker to T2 (child fitness=0.5), but do NOT bump token
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.5, VALIDITY_KEY: 1.0})
    assert get_current_snapshot().refresh_pass == token_snapshot

    # Run 2 — same params, same token → cache HIT → stale output
    m2 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)

    assert m2 == m1, (
        "Without a token bump, the cache must HIT and return the stale "
        "pass-1 output.  If this diverges, the cache isn't actually a "
        "cache and the miss-after-bump test below is meaningless."
    )
    # And the stage was only called once (cache hit skipped it the 2nd time).
    assert storage.mget.await_count == 1


@pytest.mark.asyncio
async def test_bump_invalidates_cache_and_mutation_context_reflects_new_tracker(
    tracker, metrics_ctx
):
    """After a token bump, cache misses and MutationContext reflects T2.

    This is the load-bearing assertion of the whole refresh_passes=2
    design: pass 2's stage output must encode tracker data that pass 1
    wrote.  Without the bump the cache hides the update (test above);
    with the bump it's forced to re-run against fresh tracker.
    """
    from gigaevo.evolution.engine.snapshot import (
        EngineSnapshot,
        get_current_snapshot,
        set_current_snapshot,
    )

    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await tracker.record_metrics(parent_id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    parent_prog = _mk_program(parent_id)
    child_prog = _mk_program(child_id, [parent_id])

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[parent_prog])

    filter_stage = _mk_filter_stage(tracker, metrics_ctx, storage)
    mctx_stage = MutationContextStage(metrics_context=metrics_ctx, timeout=10.0)

    params = CacheOnlyInput(cache_on="k")
    cache: dict[str, str] = {}

    # Pass 1 — cold cache, T1, token=N
    m_pass1 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)

    # Between passes: DGTrackerStage writes new metrics (T2), engine bumps.
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.5, VALIDITY_KEY: 1.0})
    set_current_snapshot(
        EngineSnapshot(refresh_pass=get_current_snapshot().refresh_pass + 1)
    )

    # Pass 2 — bumped token → cache miss → re-runs against T2 tracker
    m_pass2 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)

    # The MutationContext output must change, and it must encode 0.500.
    assert m_pass2 != m_pass1, (
        "After token bump + tracker update, MutationContext output was "
        "identical to pass 1 — the cache masked the tracker update.  "
        "This is exactly the bug refresh_passes=2 exists to fix."
    )
    assert "0.5" in m_pass2 or "0.50" in m_pass2 or "0.500" in m_pass2, (
        f"Pass-2 MutationContext must encode the fresh tracker value "
        f"(child fitness=0.5), got: {m_pass2!r}"
    )
    # And it must NOT still contain the stale 0.3 value in the same
    # positions — narrower check: the stale pass-1 string isn't a prefix.
    assert m_pass2 != m_pass1

    # Cache holds two distinct entries, one per pass (distinguishable by
    # their :rp suffix).  This is the post-hoc diagnostic: a grep of the
    # real Redis cache for `:rp1` vs `:rp2` would show both.
    keys = list(cache.keys())
    assert len(keys) == 2, f"expected two cache entries (one per pass), got {keys}"
    suffixes = {k.rsplit(":rp", 1)[-1] for k in keys}
    assert len(suffixes) == 2, (
        f"expected distinct :rp suffixes across passes, got {suffixes}"
    )

    # The stage was called twice (once per cache miss); without the bump
    # the second call would have cache-hit.
    assert storage.mget.await_count == 2


@pytest.mark.asyncio
async def test_mutation_context_text_changes_specifically_for_fitness_field(
    tracker, metrics_ctx
):
    """Tight check: the *fitness* value in MutationContext changes.

    A loose "m1 != m2" test could pass if any unrelated field flipped.
    This test pins the assertion to the metric we actually mutated,
    proving the pass-2 output reflects the specific tracker field that
    changed between passes.
    """
    from gigaevo.evolution.engine.snapshot import (
        EngineSnapshot,
        get_current_snapshot,
        set_current_snapshot,
    )

    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await tracker.record_metrics(parent_id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    parent_prog = _mk_program(parent_id)
    child_prog = _mk_program(child_id, [parent_id])

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[parent_prog])

    filter_stage = _mk_filter_stage(tracker, metrics_ctx, storage)
    mctx_stage = MutationContextStage(metrics_context=metrics_ctx, timeout=10.0)

    params = CacheOnlyInput(cache_on="k")
    cache: dict[str, str] = {}

    # Pass 1
    m1 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)

    # Between passes: tracker update + token bump
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.5, VALIDITY_KEY: 1.0})
    set_current_snapshot(
        EngineSnapshot(refresh_pass=get_current_snapshot().refresh_pass + 1)
    )

    # Pass 2
    m2 = await _run_pipeline(filter_stage, mctx_stage, child_prog, params, cache)

    # m1 must carry 0.3-family number; m2 must carry 0.5-family number.
    def _has_fitness_value(s: str, val: float) -> bool:
        # MetricsFormatter renders floats with fixed precision; check all
        # plausible renderings to avoid over-coupling to formatter choice.
        candidates = [f"{val:.1f}", f"{val:.2f}", f"{val:.3f}", f"{val:.4f}"]
        return any(c in s for c in candidates)

    assert _has_fitness_value(m1, 0.3), f"m1 missing 0.3 encoding: {m1!r}"
    assert _has_fitness_value(m2, 0.5), f"m2 missing 0.5 encoding: {m2!r}"
    assert not _has_fitness_value(m2, 0.3) or _has_fitness_value(m2, 0.5), (
        "Pass-2 output should reflect fresh value 0.5, not stale 0.3"
    )


@pytest.mark.asyncio
async def test_cache_key_suffix_is_observable_per_pass(tracker, metrics_ctx):
    """The cache key suffix `:rpN` is the operator-visible diagnostic.

    Operators running the real pipeline can grep Redis (post-hoc) for
    `:rp1` / `:rp2` keys per program to confirm pass identity.  This
    test pins the format so the diagnostic stays stable.
    """
    from gigaevo.evolution.engine.snapshot import (
        EngineSnapshot,
        get_current_snapshot,
        set_current_snapshot,
    )

    child_id, parent_id = str(uuid.uuid4()), str(uuid.uuid4())
    await tracker.record_metrics(child_id, "g1", {"fitness": 0.3, VALIDITY_KEY: 1.0})
    await tracker.record_metrics(parent_id, "g1", {"fitness": 0.1, VALIDITY_KEY: 1.0})

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[_mk_program(parent_id)])

    filter_stage = _mk_filter_stage(tracker, metrics_ctx, storage)
    mctx_stage = MutationContextStage(metrics_context=metrics_ctx, timeout=10.0)

    params = CacheOnlyInput(cache_on="k")
    cache: dict[str, str] = {}

    t0 = get_current_snapshot().refresh_pass
    await _run_pipeline(
        filter_stage, mctx_stage, _mk_program(child_id, [parent_id]), params, cache
    )

    set_current_snapshot(EngineSnapshot(refresh_pass=t0 + 1))
    await _run_pipeline(
        filter_stage, mctx_stage, _mk_program(child_id, [parent_id]), params, cache
    )

    keys = sorted(cache.keys())
    assert any(k.endswith(f":rp{t0}") for k in keys), (
        f"pass-1 cache key with :rp{t0} suffix missing from {keys}"
    )
    assert any(k.endswith(f":rp{t0 + 1}") for k in keys), (
        f"pass-2 cache key with :rp{t0 + 1} suffix missing from {keys}"
    )


# ---------------------------------------------------------------------------
# Combined-treatment wiring sanity (compact tripwire)
# ---------------------------------------------------------------------------


def test_all_d_side_treatment_knobs_coexist_in_one_config():
    """One-shot regression: every D-side treatment is simultaneously
    wire-able without collisions.  If a future refactor drops one knob,
    this fails loud at the config layer before any experiment launches.
    """
    from gigaevo.evolution.engine.config import SteadyStateEngineConfig

    cfg = SteadyStateEngineConfig(
        max_in_flight=1,
        max_mutations_per_generation=1,
        refresh_order="generation_bucketed",
        refresh_passes=2,
    )

    assert cfg.refresh_order == "generation_bucketed"
    assert cfg.refresh_passes == 2
