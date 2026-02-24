"""Tests for MultiModelRouter, TokenTracker, and generate_mutations integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.llm.models import MultiModelRouter, _StructuredOutputRouter
from gigaevo.llm.token_tracking import TokenTracker, TokenUsage
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


def _mock_response(model_name: str, ctx=100, gen=50, total=150) -> MagicMock:
    """Create a mock LLM response with token usage metadata."""
    resp = MagicMock()
    resp.response_metadata = {
        "token_usage": {
            "prompt_tokens": ctx,
            "completion_tokens": gen,
            "total_tokens": total,
        }
    }
    return resp


# ---------------------------------------------------------------------------
# TokenUsage.from_response
# ---------------------------------------------------------------------------


class TestTokenUsageExtraction:
    def test_from_response_standard(self):
        resp = _mock_response("test", ctx=200, gen=100, total=300)
        usage = TokenUsage.from_response(resp)
        assert usage.context == 200
        assert usage.generated == 100
        assert usage.total == 300

    def test_from_response_no_metadata(self):
        resp = MagicMock(spec=[])  # no response_metadata attribute
        assert TokenUsage.from_response(resp) is None

    def test_from_response_empty_metadata(self):
        resp = MagicMock()
        resp.response_metadata = {}
        assert TokenUsage.from_response(resp) is None

    def test_from_response_reasoning_tokens_openai_style(self):
        resp = MagicMock()
        resp.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 30},
            }
        }
        usage = TokenUsage.from_response(resp)
        assert usage.reasoning == 30

    def test_from_response_reasoning_tokens_direct(self):
        resp = MagicMock()
        resp.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "reasoning_tokens": 20,
            }
        }
        usage = TokenUsage.from_response(resp)
        assert usage.reasoning == 20

    def test_from_response_thinking_tokens_qwen(self):
        resp = MagicMock()
        resp.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "thinking_tokens": 15,
            }
        }
        usage = TokenUsage.from_response(resp)
        assert usage.reasoning == 15

    def test_from_response_usage_key_fallback(self):
        """Falls back to 'usage' key if 'token_usage' not present."""
        resp = MagicMock()
        resp.response_metadata = {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            }
        }
        usage = TokenUsage.from_response(resp)
        assert usage.total == 75


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


class TestTokenTracker:
    def test_cumulative_tracking(self):
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)

        resp1 = _mock_response("m", ctx=100, gen=50, total=150)
        resp2 = _mock_response("m", ctx=200, gen=80, total=280)

        tracker.track(resp1, "model_a")
        tracker.track(resp2, "model_a")

        assert tracker.cumulative["model_a"].context == 300
        assert tracker.cumulative["model_a"].generated == 130
        assert tracker.cumulative["model_a"].total == 430

    def test_per_model_separation(self):
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)

        tracker.track(_mock_response("a", ctx=100, gen=50, total=150), "model_a")
        tracker.track(_mock_response("b", ctx=200, gen=80, total=280), "model_b")

        assert tracker.cumulative["model_a"].total == 150
        assert tracker.cumulative["model_b"].total == 280

    def test_no_writer_is_noop(self):
        tracker = TokenTracker(name="test", writer=None)
        tracker.track(_mock_response("m"), "model_a")
        assert len(tracker.cumulative) == 0

    def test_no_usage_in_response_skipped(self):
        writer = NullWriter()
        tracker = TokenTracker(name="test", writer=writer)
        resp = MagicMock()
        resp.response_metadata = {}
        tracker.track(resp, "model_a")
        assert len(tracker.cumulative) == 0


# ---------------------------------------------------------------------------
# MultiModelRouter
# ---------------------------------------------------------------------------


class TestMultiModelRouter:
    def test_initialization_normalizes_probabilities(self):
        models = [_mock_model("a"), _mock_model("b")]
        router = MultiModelRouter(models, [3.0, 1.0], name="test")
        assert router.probabilities == pytest.approx([0.75, 0.25])

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            MultiModelRouter([_mock_model("a")], [0.5, 0.5], name="test")

    def test_validation_zero_probability(self):
        with pytest.raises(ValueError, match="positive"):
            MultiModelRouter(
                [_mock_model("a"), _mock_model("b")], [1.0, 0.0], name="test"
            )

    def test_select_returns_valid_model(self):
        models = [_mock_model("a"), _mock_model("b")]
        router = MultiModelRouter(models, [0.5, 0.5], name="test")
        model, name = router._select()
        assert name in ["a", "b"]
        assert model in models

    def test_invoke_calls_selected_model(self):
        model = _mock_model("a")
        model.invoke.return_value = _mock_response("a")
        router = MultiModelRouter([model], [1.0], name="test")
        router._langfuse = None

        router.invoke("hello")
        model.invoke.assert_called_once()

    async def test_ainvoke_async(self):
        model = _mock_model("a")
        model.ainvoke = AsyncMock(return_value=_mock_response("a"))
        router = MultiModelRouter([model], [1.0], name="test")
        router._langfuse = None

        await router.ainvoke("hello")
        model.ainvoke.assert_called_once()

    async def test_get_last_model_tracks_task(self):
        models = [_mock_model("model_x")]
        router = MultiModelRouter(models, [1.0], name="test")

        async def task_fn():
            router._select()
            return router.get_last_model()

        result = await asyncio.create_task(task_fn())
        assert result == "model_x"

    async def test_get_last_model_returns_none_without_select(self):
        models = [_mock_model("model_x")]
        router = MultiModelRouter(models, [1.0], name="test")

        result = router.get_last_model()
        assert result is None

    def test_with_structured_output(self):
        models = [_mock_model("a"), _mock_model("b")]
        router = MultiModelRouter(models, [0.5, 0.5], name="test")

        structured = router.with_structured_output(dict)
        assert isinstance(structured, _StructuredOutputRouter)

    def test_on_mutation_outcome_default_noop(self):
        """Default on_mutation_outcome is a no-op (doesn't raise)."""
        models = [_mock_model("a")]
        router = MultiModelRouter(models, [1.0], name="test")

        prog = MagicMock()
        # Should not raise
        router.on_mutation_outcome(prog, [], None)


# ---------------------------------------------------------------------------
# _StructuredOutputRouter
# ---------------------------------------------------------------------------


class TestStructuredOutputRouter:
    def test_invoke_with_select_override(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = {
            "raw": _mock_response("a"),
            "parsed": {"key": "val"},
        }

        tracker = TokenTracker(name="test", writer=NullWriter())

        router = _StructuredOutputRouter(
            models=[mock_model],
            model_names=["a"],
            probabilities=[1.0],
            langfuse=None,
            tracker=tracker,
        )

        result = router.invoke("hello")
        assert result == {"key": "val"}

    async def test_ainvoke(self):
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            return_value={"raw": _mock_response("a"), "parsed": {"key": "val"}}
        )

        tracker = TokenTracker(name="test", writer=NullWriter())

        router = _StructuredOutputRouter(
            models=[mock_model],
            model_names=["a"],
            probabilities=[1.0],
            langfuse=None,
            tracker=tracker,
        )

        result = await router.ainvoke("hello")
        assert result == {"key": "val"}

    def test_select_override_used(self):
        mock_model_a = MagicMock()
        mock_model_b = MagicMock()
        mock_model_b.invoke.return_value = {
            "raw": _mock_response("b"),
            "parsed": "b_result",
        }

        tracker = TokenTracker(name="test", writer=NullWriter())

        router = _StructuredOutputRouter(
            models=[mock_model_a, mock_model_b],
            model_names=["a", "b"],
            probabilities=[0.5, 0.5],
            langfuse=None,
            tracker=tracker,
            select_override=lambda: (mock_model_b, "b"),
        )

        result = router.invoke("hello")
        mock_model_a.invoke.assert_not_called()
        mock_model_b.invoke.assert_called_once()
        assert result == "b_result"
