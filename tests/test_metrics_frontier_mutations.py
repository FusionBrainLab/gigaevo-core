"""Mutation-killing tests for MetricsTracker frontier comparison logic.

These tests are designed to catch specific operator mutations that would escape
the existing test suite. Each test verifies a boundary condition where flipping
an operator (> to >=, < to <=, etc.) would produce a different result.

Gap identified by mutation testing analysis: +12-18% kill rate improvement.
"""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.utils.metrics_tracker import MetricsTracker
from gigaevo.utils.trackers.base import LogWriter
from typing import Any
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingWriter(LogWriter):
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, dict[str, Any]]] = []

    def bind(self, path: list[str]) -> "RecordingWriter":
        return self

    def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
        self.scalars.append((metric, value, kwargs))

    def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
        pass

    def text(self, tag: str, text: str, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _ctx(higher_is_better: bool = True) -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="Primary score",
                is_primary=True,
                higher_is_better=higher_is_better,
            ),
        }
    )


def _tracker(higher_is_better: bool = True) -> MetricsTracker:
    storage = AsyncMock()
    storage.get_all_program_ids = AsyncMock(return_value=[])
    storage.mget = AsyncMock(return_value=[])
    return MetricsTracker(
        storage=storage,
        metrics_context=_ctx(higher_is_better),
        writer=RecordingWriter(),
    )


# ===========================================================================
# Operator mutation: > vs >= (higher_is_better=True)
# ===========================================================================


class TestFrontierOperatorMutations:
    """These tests would FAIL if the comparison operator is mutated."""

    def test_equal_value_does_not_improve_higher(self) -> None:
        """MUTATION TARGET: `value > best_val` must NOT be `value >= best_val`.

        If mutated to >=, equal values would be treated as improvements,
        causing unnecessary frontier rewrites.
        """
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", 10.0, iteration=1)
        result = t._maybe_update_frontier("score", 10.0, iteration=2)
        assert result is False, "Equal value must NOT be an improvement (> not >=)"
        assert t._best_valid["score"] == (10.0, 1), "Original iteration preserved"

    def test_equal_value_does_not_improve_lower(self) -> None:
        """MUTATION TARGET: `value < best_val` must NOT be `value <= best_val`."""
        t = _tracker(higher_is_better=False)
        t._maybe_update_frontier("score", 10.0, iteration=1)
        result = t._maybe_update_frontier("score", 10.0, iteration=2)
        assert result is False, "Equal value must NOT be an improvement (< not <=)"

    def test_strictly_greater_improves_higher(self) -> None:
        """Sanity: strictly greater value IS an improvement for higher_is_better."""
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", 10.0, iteration=1)
        result = t._maybe_update_frontier("score", 10.001, iteration=2)
        assert result is True

    def test_strictly_less_improves_lower(self) -> None:
        """Sanity: strictly less value IS an improvement for lower_is_better."""
        t = _tracker(higher_is_better=False)
        t._maybe_update_frontier("score", 10.0, iteration=1)
        result = t._maybe_update_frontier("score", 9.999, iteration=2)
        assert result is True


# ===========================================================================
# Operator mutation: direction inversion (> swapped with <)
# ===========================================================================


class TestFrontierDirectionInversion:
    """These tests would FAIL if higher_is_better logic is inverted."""

    def test_higher_value_rejected_when_lower_is_better(self) -> None:
        """MUTATION TARGET: if > and < are swapped, this fails."""
        t = _tracker(higher_is_better=False)
        t._maybe_update_frontier("score", 5.0, iteration=1)
        result = t._maybe_update_frontier("score", 10.0, iteration=2)
        assert result is False, "Higher value must be rejected when lower_is_better"
        assert t._best_valid["score"] == (5.0, 1)

    def test_lower_value_rejected_when_higher_is_better(self) -> None:
        """MUTATION TARGET: if > and < are swapped, this fails."""
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", 10.0, iteration=1)
        result = t._maybe_update_frontier("score", 5.0, iteration=2)
        assert result is False, "Lower value must be rejected when higher_is_better"
        assert t._best_valid["score"] == (10.0, 1)


# ===========================================================================
# Boolean mutation: `if best is None` → `if best is not None`
# ===========================================================================


class TestFrontierFirstValue:
    """These tests catch mutations in the 'first value always wins' logic."""

    def test_first_value_returns_true(self) -> None:
        """MUTATION TARGET: `if best is None: return True` → `return False`."""
        t = _tracker()
        result = t._maybe_update_frontier("score", 42.0, iteration=1)
        assert result is True, "First value must always be recorded"

    def test_first_value_stored_correctly(self) -> None:
        """MUTATION TARGET: tuple stored as (value, iteration) not (iteration, value)."""
        t = _tracker()
        t._maybe_update_frontier("score", 42.0, iteration=7)
        assert t._best_valid["score"] == (42.0, 7)
        # Explicitly check order — (value, iteration) not (iteration, value)
        stored_val, stored_iter = t._best_valid["score"]
        assert stored_val == 42.0, "First element must be value"
        assert stored_iter == 7, "Second element must be iteration"

    def test_first_negative_value_accepted(self) -> None:
        """Even negative values should be recorded as first."""
        t = _tracker()
        result = t._maybe_update_frontier("score", -100.0, iteration=1)
        assert result is True
        assert t._best_valid["score"] == (-100.0, 1)


# ===========================================================================
# Boundary precision tests
# ===========================================================================


class TestFrontierBoundaryPrecision:
    """Tests with tiny differences that catch floating-point comparison bugs."""

    def test_epsilon_improvement_higher(self) -> None:
        """A tiny improvement (1e-15) should still count."""
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", 1.0, iteration=1)
        result = t._maybe_update_frontier("score", 1.0 + 1e-15, iteration=2)
        assert result is True

    def test_epsilon_improvement_lower(self) -> None:
        """A tiny improvement (1e-15) should still count for lower_is_better."""
        t = _tracker(higher_is_better=False)
        t._maybe_update_frontier("score", 1.0, iteration=1)
        result = t._maybe_update_frontier("score", 1.0 - 1e-15, iteration=2)
        assert result is True

    def test_zero_to_positive_is_improvement(self) -> None:
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", 0.0, iteration=1)
        result = t._maybe_update_frontier("score", 0.001, iteration=2)
        assert result is True

    def test_zero_to_negative_is_improvement_lower(self) -> None:
        t = _tracker(higher_is_better=False)
        t._maybe_update_frontier("score", 0.0, iteration=1)
        result = t._maybe_update_frontier("score", -0.001, iteration=2)
        assert result is True

    def test_negative_to_zero_is_improvement_higher(self) -> None:
        t = _tracker(higher_is_better=True)
        t._maybe_update_frontier("score", -1.0, iteration=1)
        result = t._maybe_update_frontier("score", 0.0, iteration=2)
        assert result is True

    def test_multiple_updates_only_best_survives(self) -> None:
        """After many updates, only the absolute best value should remain."""
        t = _tracker(higher_is_better=True)
        values = [1.0, 5.0, 3.0, 7.0, 2.0, 6.0]
        for i, v in enumerate(values):
            t._maybe_update_frontier("score", v, iteration=i + 1)
        assert t._best_valid["score"] == (7.0, 4), "7.0 at iteration 4 is the best"

    def test_multiple_updates_lower_is_better(self) -> None:
        t = _tracker(higher_is_better=False)
        values = [5.0, 3.0, 7.0, 1.0, 4.0, 2.0]
        for i, v in enumerate(values):
            t._maybe_update_frontier("score", v, iteration=i + 1)
        assert t._best_valid["score"] == (1.0, 4), "1.0 at iteration 4 is the best"
