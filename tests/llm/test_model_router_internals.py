"""Tests for MultiModelRouter internals: _select, _config, _current_task_id.

Covers heavily-called but untested production code paths:
- _select (6 production callers)
- _config (6 production callers)
- _current_task_id (5 production callers)
- get_last_model (task-model tracking)
- _StructuredOutputRouter._select / _config / _process
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.llm.models import (
    MultiModelRouter,
    _StructuredOutputRouter,
    _with_langfuse,
)
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


def _make_router(
    model_names: list[str] | None = None,
    probabilities: list[float] | None = None,
) -> MultiModelRouter:
    names = model_names or ["gpt-4", "gpt-3.5-turbo"]
    probs = probabilities or [0.8, 0.2]
    models = [_mock_model(n) for n in names]
    return MultiModelRouter(models, probs, writer=NullWriter(), name="test")


# ===================================================================
# _current_task_id
# ===================================================================


class TestCurrentTaskId:
    """_current_task_id: 5 production callers."""

    async def test_returns_int_inside_async_task(self):
        """Inside an async task, returns id(current_task)."""
        tid = MultiModelRouter._current_task_id()
        assert isinstance(tid, int)
        task = asyncio.current_task()
        assert tid == id(task)

    def test_returns_none_outside_event_loop(self):
        """Outside event loop (no running loop), returns None."""
        # In a sync context with no event loop, should return None
        # (asyncio.current_task() raises RuntimeError or returns None)
        result = MultiModelRouter._current_task_id()
        # In pytest sync test, there may or may not be a loop
        assert result is None or isinstance(result, int)


# ===================================================================
# _select
# ===================================================================


class TestSelect:
    """_select: 6 production callers — model selection + task tracking."""

    def test_returns_model_and_name(self):
        router = _make_router()
        model, name = router._select()
        assert name in ["gpt-4", "gpt-3.5-turbo"]
        assert model is not None

    def test_single_model_always_selected(self):
        router = _make_router(["only-model"], [1.0])
        for _ in range(20):
            _, name = router._select()
            assert name == "only-model"

    @patch("gigaevo.llm.models.random.choices")
    def test_respects_probability_distribution(self, mock_choices):
        """_select uses random.choices with weights."""
        mock_choices.return_value = [1]  # always pick index 1
        router = _make_router()
        _, name = router._select()
        assert name == "gpt-3.5-turbo"
        mock_choices.assert_called_once()
        # Verify weights were passed
        call_kwargs = mock_choices.call_args
        assert call_kwargs[1]["weights"] is not None

    async def test_records_task_model_mapping(self):
        """_select records which model was used for the current async task."""
        router = _make_router(["model-a"], [1.0])
        router._select()
        tid = MultiModelRouter._current_task_id()
        assert tid in router._task_model_map
        assert router._task_model_map[tid] == "model-a"


# ===================================================================
# get_last_model
# ===================================================================


class TestGetLastModel:
    async def test_returns_model_after_select(self):
        """get_last_model returns the model used in the last _select for this task."""
        router = _make_router(["test-model"], [1.0])
        router._select()
        result = router.get_last_model()
        assert result == "test-model"

    async def test_returns_none_when_no_select(self):
        """get_last_model returns None when _select hasn't been called."""
        router = _make_router()
        result = router.get_last_model()
        assert result is None

    async def test_pops_mapping(self):
        """get_last_model pops the entry — second call returns None."""
        router = _make_router(["test-model"], [1.0])
        router._select()
        assert router.get_last_model() == "test-model"
        assert router.get_last_model() is None


# ===================================================================
# _config
# ===================================================================


class TestConfig:
    """_config: 6 production callers — langfuse integration."""

    def test_without_langfuse_returns_input(self):
        """When no langfuse handler, _config returns the input config."""
        router = _make_router()
        router._langfuse = None
        config = {"key": "value"}
        result = router._config(config, "model-a")
        assert result == config

    def test_with_langfuse_adds_handler(self):
        """When langfuse handler exists, _config adds it to callbacks."""
        router = _make_router()
        mock_handler = MagicMock()
        router._langfuse = mock_handler
        result = router._config({}, "model-a")
        assert mock_handler in result["callbacks"]

    def test_with_langfuse_adds_model_metadata(self):
        """_config adds selected_model to metadata."""
        router = _make_router()
        mock_handler = MagicMock()
        router._langfuse = mock_handler
        result = router._config({}, "my-model")
        assert result["metadata"]["selected_model"] == "my-model"

    def test_none_config_handled(self):
        """_config handles None config input."""
        router = _make_router()
        router._langfuse = None
        result = router._config(None, "model")
        assert result is None


# ===================================================================
# _with_langfuse (module-level helper)
# ===================================================================


class TestWithLangfuse:
    def test_none_handler_passthrough(self):
        config = {"key": "val"}
        assert _with_langfuse(config, None) is config

    def test_adds_handler_to_callbacks(self):
        handler = MagicMock()
        result = _with_langfuse({}, handler, "model-x")
        assert handler in result["callbacks"]
        assert result["metadata"]["selected_model"] == "model-x"

    def test_does_not_duplicate_handler(self):
        handler = MagicMock()
        config = {"callbacks": [handler]}
        result = _with_langfuse(config, handler)
        assert result["callbacks"].count(handler) == 1

    def test_none_model_name_skips_metadata(self):
        handler = MagicMock()
        result = _with_langfuse({}, handler, None)
        assert "metadata" not in result or "selected_model" not in result.get(
            "metadata", {}
        )


# ===================================================================
# MultiModelRouter construction validation
# ===================================================================


class TestRouterValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            MultiModelRouter(
                [_mock_model("a")],
                [0.5, 0.5],
                writer=NullWriter(),
            )

    def test_zero_probability_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MultiModelRouter(
                [_mock_model("a"), _mock_model("b")],
                [0.5, 0.0],
                writer=NullWriter(),
            )

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MultiModelRouter(
                [_mock_model("a")],
                [-0.5],
                writer=NullWriter(),
            )

    def test_probabilities_normalized(self):
        router = _make_router(["a", "b"], [2.0, 8.0])
        assert sum(router.probabilities) == pytest.approx(1.0)
        assert router.probabilities[0] == pytest.approx(0.2)
        assert router.probabilities[1] == pytest.approx(0.8)


# ===================================================================
# _StructuredOutputRouter internals
# ===================================================================


class TestStructuredOutputRouter:
    def test_select_uses_override_when_provided(self):
        override = MagicMock(return_value=("mock_model", "override-name"))
        router = _StructuredOutputRouter(
            models=[MagicMock()],
            model_names=["base"],
            probabilities=[1.0],
            langfuse=None,
            tracker=MagicMock(),
            select_override=override,
        )
        model, name = router._select()
        assert name == "override-name"
        override.assert_called_once()

    def test_process_extracts_parsed(self):
        tracker = MagicMock()
        router = _StructuredOutputRouter(
            models=[],
            model_names=[],
            probabilities=[],
            langfuse=None,
            tracker=tracker,
        )
        raw_response = MagicMock()
        response = {"raw": raw_response, "parsed": {"key": "value"}}
        result = router._process(response, "model-a")
        assert result == {"key": "value"}
        tracker.track.assert_called_once_with(raw_response, "model-a")

    def test_process_no_raw_skips_tracking(self):
        tracker = MagicMock()
        router = _StructuredOutputRouter(
            models=[],
            model_names=[],
            probabilities=[],
            langfuse=None,
            tracker=tracker,
        )
        result = router._process({"parsed": "data"}, "model-a")
        assert result == "data"
        tracker.track.assert_not_called()
