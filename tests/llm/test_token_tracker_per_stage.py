"""Per-stage attribution for LLM token usage.

Why this matters
----------------
A single ``MultiModelRouter`` is shared across InsightsAgent, LineageAgent,
and MutationAgent (see ``gigaevo/entrypoint/default_pipelines.py``), so the
existing per-model TensorBoard scalars merge tokens from every stage into
one series. Operators cannot tell whether a sudden cost spike came from
mutation-time generation, lineage analysis, or insight extraction.

The fix is a ``ContextVar`` set by each ``LangGraphAgent.acall_llm`` around
its ``await llm.ainvoke(...)`` call. The router invocation reads the var
and threads the stage name through ``TokenTracker.track`` so:

- ``cumulative`` accounting splits by ``(stage, model)`` instead of ``model``.
- TensorBoard scalar paths gain a ``<stage>`` segment (``llm/tokens/default/
  MutationAgent/<model>/total_tokens``) so each stage gets its own panel.

The legacy "no stage set" path must keep working (e.g. direct router use
outside an agent) and land under an ``unattributed`` bucket so we never
silently lose tokens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gigaevo.llm.token_tracking import (
    TokenTracker,
    TokenUsage,
    llm_stage_context,
)
from tests.conftest import NullWriter


def _mock_response(ctx: int = 100, gen: int = 50, total: int = 150) -> MagicMock:
    resp = MagicMock()
    resp.response_metadata = {
        "token_usage": {
            "prompt_tokens": ctx,
            "completion_tokens": gen,
            "total_tokens": total,
        }
    }
    return resp


class _RecordingWriter(NullWriter):
    """Captures every scalar call so we can inspect path + metric + value."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, tuple[str, ...]]] = []

    def scalar(self, metric: str, value: float, **kw: Any) -> None:
        path = tuple(kw.get("path", ())) if kw.get("path") else ()
        self.calls.append((metric, float(value), path))


class TestPerStageAttribution:
    def test_stage_context_var_default_is_none(self) -> None:
        """Outside any context, the stage var is None — track() falls back
        to the legacy single-key behavior under an 'unattributed' bucket."""
        from gigaevo.llm.token_tracking import get_current_llm_stage

        assert get_current_llm_stage() is None

    def test_track_uses_stage_from_context(self) -> None:
        """Calling track() inside `llm_stage_context("X")` keys the
        cumulative dict by (stage, model)."""
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)

        with llm_stage_context("MutationAgent"):
            tracker.track(_mock_response(ctx=100, gen=50, total=150), "model_a")

        assert ("MutationAgent", "model_a") in tracker.cumulative_by_stage
        bucket = tracker.cumulative_by_stage[("MutationAgent", "model_a")]
        assert bucket.context == 100
        assert bucket.generated == 50
        assert bucket.total == 150

    def test_aggregate_cumulative_still_tracked(self) -> None:
        """The legacy aggregate dict (per-model only) must keep working
        so the existing TB panels and downstream readers don't break."""
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)

        with llm_stage_context("InsightsAgent"):
            tracker.track(_mock_response(ctx=100, gen=50, total=150), "model_a")
        with llm_stage_context("MutationAgent"):
            tracker.track(_mock_response(ctx=200, gen=80, total=280), "model_a")

        # Legacy aggregate sums across stages.
        assert tracker.cumulative["model_a"].total == 430
        # Per-stage breakdown is also kept.
        assert tracker.cumulative_by_stage[("InsightsAgent", "model_a")].total == 150
        assert tracker.cumulative_by_stage[("MutationAgent", "model_a")].total == 280

    def test_writer_path_includes_stage(self) -> None:
        """TB paths under per-stage attribution must contain the stage name
        so each stage gets its own panel."""
        writer = _RecordingWriter()
        tracker = TokenTracker(name="default", writer=writer)

        with llm_stage_context("MutationAgent"):
            tracker.track(_mock_response(total=150), "openai/gpt-4o-mini")

        # Each scalar should be written at a path containing "MutationAgent"
        # AND the safe model name (slashes replaced).
        per_stage_calls = [
            (m, v, p) for m, v, p in writer.calls if "MutationAgent" in p
        ]
        assert per_stage_calls, f"no per-stage scalar emitted: {writer.calls}"
        # Path shape is [name, stage, safe_model].
        assert ("default", "MutationAgent", "openai_gpt-4o-mini") in {
            p for _, _, p in per_stage_calls
        }

    def test_no_stage_falls_back_to_unattributed_bucket(self) -> None:
        """A bare track() outside any stage context must still book tokens —
        under a sentinel stage so they don't silently vanish."""
        writer = _RecordingWriter()
        tracker = TokenTracker(name="default", writer=writer)

        tracker.track(_mock_response(total=150), "model_a")

        # Either keyed under 'unattributed' or some equivalent sentinel — the
        # test just requires the tokens are bookable somewhere per-stage.
        per_stage_keys = list(tracker.cumulative_by_stage.keys())
        assert per_stage_keys, "tokens dropped when no stage context active"
        stage_used, model_used = per_stage_keys[0]
        assert stage_used  # non-empty marker
        assert model_used == "model_a"

    def test_stage_context_is_async_safe(self) -> None:
        """ContextVar must isolate concurrent agents. Two stages set in
        nested with-blocks must restore correctly."""
        from gigaevo.llm.token_tracking import get_current_llm_stage

        with llm_stage_context("Outer"):
            assert get_current_llm_stage() == "Outer"
            with llm_stage_context("Inner"):
                assert get_current_llm_stage() == "Inner"
            assert get_current_llm_stage() == "Outer"
        assert get_current_llm_stage() is None

    @pytest.mark.asyncio
    async def test_stage_context_isolated_across_tasks(self) -> None:
        """Two concurrent asyncio tasks must not see each other's stage."""
        import asyncio

        from gigaevo.llm.token_tracking import get_current_llm_stage

        observed: dict[str, str | None] = {}

        async def stage_task(name: str) -> None:
            with llm_stage_context(name):
                # Yield to interleave with the sibling task.
                await asyncio.sleep(0)
                observed[name] = get_current_llm_stage()

        await asyncio.gather(stage_task("A"), stage_task("B"))
        assert observed == {"A": "A", "B": "B"}


def _harness_response(cost: float = 0.0125, turns: int = 4) -> MagicMock:
    """A harness message: token counts plus the cost and turn count only the
    CLI itself knows."""
    resp = _mock_response()
    resp.response_metadata["total_cost_usd"] = cost
    resp.response_metadata["num_turns"] = turns
    return resp


class TestCostAndTurns:
    """Cost and turn count must reach the durable metrics, not die on the
    message: turn count is the dominant cost variable of a harness backend,
    and total_cost_usd cannot be reconstructed from token counts."""

    def test_from_response_reads_cost_and_turns(self) -> None:
        usage = TokenUsage.from_response(_harness_response(cost=0.0125, turns=4))

        assert usage is not None
        assert usage.cost_usd == 0.0125
        assert usage.turns == 4

    def test_api_responses_leave_them_none(self) -> None:
        usage = TokenUsage.from_response(_mock_response())

        assert usage is not None
        assert usage.cost_usd is None
        assert usage.turns is None

    def test_nonsense_values_are_discarded(self) -> None:
        resp = _mock_response()
        resp.response_metadata["total_cost_usd"] = "free"
        resp.response_metadata["num_turns"] = True

        usage = TokenUsage.from_response(resp)

        assert usage is not None
        assert usage.cost_usd is None
        assert usage.turns is None

    def test_degenerate_costs_are_discarded(self) -> None:
        """A negative or non-finite cost skews the cumulative forever — NaN
        poisons every sum after it — and json.loads happily produces all
        three."""
        for cost in (-0.01, float("nan"), float("inf")):
            resp = _mock_response()
            resp.response_metadata["total_cost_usd"] = cost

            usage = TokenUsage.from_response(resp)

            assert usage is not None
            assert usage.cost_usd is None

    def test_a_turn_count_beyond_float_range_is_discarded(self) -> None:
        """json accepts integers float() cannot represent; the writer emits
        floats, so keeping one turns a served call into a crashed metric
        write."""
        resp = _mock_response()
        resp.response_metadata["num_turns"] = 10**400

        usage = TokenUsage.from_response(resp)

        assert usage is not None
        assert usage.turns is None
        tracker = TokenTracker(name="default", writer=_RecordingWriter())
        tracker.track(resp, "model_a")

    def test_tracker_accumulates_and_writes_scalars(self) -> None:
        writer = _RecordingWriter()
        tracker = TokenTracker(name="default", writer=writer)

        with llm_stage_context("MutationAgent"):
            tracker.track(_harness_response(cost=0.01, turns=4), "claude-code/sonnet")
            tracker.track(_harness_response(cost=0.02, turns=6), "claude-code/sonnet")

        cum = tracker.cumulative["claude-code/sonnet"]
        assert cum.cost_usd == pytest.approx(0.03)
        assert cum.turns == 10
        stage_cum = tracker.cumulative_by_stage[("MutationAgent", "claude-code/sonnet")]
        assert stage_cum.cost_usd == pytest.approx(0.03)
        assert stage_cum.turns == 10

        metrics = {m for m, _, _ in writer.calls}
        assert {
            "cost_usd",
            "cumulative_cost_usd",
            "num_turns",
            "cumulative_num_turns",
        } <= metrics
        last_cumulative_cost = [
            v for m, v, _ in writer.calls if m == "cumulative_cost_usd"
        ][-1]
        assert last_cumulative_cost == pytest.approx(0.03)
        # The per-stage panel carries them too.
        stage_paths = {
            p for m, _, p in writer.calls if m == "num_turns" and "MutationAgent" in p
        }
        assert ("default", "MutationAgent", "claude-code_sonnet") in stage_paths

    def test_api_responses_emit_no_cost_scalars(self) -> None:
        """An API model reports no cost or turns; zero-filled series for every
        OpenAI call would bury the harness series they exist to monitor."""
        writer = _RecordingWriter()
        tracker = TokenTracker(name="default", writer=writer)

        tracker.track(_mock_response(), "model_a")

        metrics = {m for m, _, _ in writer.calls}
        assert "cost_usd" not in metrics
        assert "num_turns" not in metrics


class TestTokenUsageStaticAdd:
    """Sanity check that the per-stage bucket aggregates the same way
    as the legacy bucket — no surprise rounding or off-by-one."""

    def test_per_stage_accumulates_across_calls(self) -> None:
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)

        with llm_stage_context("MutationAgent"):
            tracker.track(_mock_response(ctx=100, gen=50, total=150), "model_a")
            tracker.track(_mock_response(ctx=200, gen=80, total=280), "model_a")

        bucket: TokenUsage = tracker.cumulative_by_stage[("MutationAgent", "model_a")]
        assert bucket.context == 300
        assert bucket.generated == 130
        assert bucket.total == 430
