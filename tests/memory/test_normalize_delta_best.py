"""Tests for normalize_delta_best helper (CARD_STRUCTURE_v4 §1.5)."""

from __future__ import annotations

import pytest

from gigaevo.memory.shared_memory.card_conversion import normalize_delta_best


class TestNormalizeDeltaBest:
    def test_higher_is_better_positive_delta_stays_positive(self) -> None:
        # accuracy: child=0.95, parent=0.90 → +0.05 → +0.05
        assert normalize_delta_best(0.05, lower_is_better=False) == pytest.approx(0.05)

    def test_higher_is_better_negative_delta_stays_negative(self) -> None:
        assert normalize_delta_best(-0.05, lower_is_better=False) == pytest.approx(
            -0.05
        )

    def test_lower_is_better_negative_delta_becomes_positive(self) -> None:
        # loss: child=0.10, parent=0.15 → -0.05 → +0.05 (improvement)
        assert normalize_delta_best(-0.05, lower_is_better=True) == pytest.approx(0.05)

    def test_lower_is_better_positive_delta_becomes_negative(self) -> None:
        # loss: child=0.20, parent=0.15 → +0.05 → -0.05 (regression)
        assert normalize_delta_best(0.05, lower_is_better=True) == pytest.approx(-0.05)

    def test_zero_unchanged(self) -> None:
        assert normalize_delta_best(0.0, lower_is_better=True) == 0.0
        assert normalize_delta_best(0.0, lower_is_better=False) == 0.0

    def test_handles_string_input(self) -> None:
        assert normalize_delta_best("0.02", lower_is_better=True) == pytest.approx(
            -0.02
        )

    def test_invalid_input_returns_zero(self) -> None:
        assert normalize_delta_best("not_a_number", lower_is_better=True) == 0.0
        assert normalize_delta_best(None, lower_is_better=False) == 0.0
