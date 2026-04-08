"""Unit tests for gigaevo.memory.utils — shared utility functions."""

from __future__ import annotations

import pytest

from gigaevo.memory.utils import median, parse_cell, to_float


class TestToFloat:
    def test_valid_int(self) -> None:
        assert to_float(42) == 42.0

    def test_valid_float(self) -> None:
        assert to_float(3.14) == pytest.approx(3.14)

    def test_valid_string(self) -> None:
        assert to_float("3.14") == pytest.approx(3.14)

    def test_negative_is_valid(self) -> None:
        assert to_float(-1e5) == pytest.approx(-1e5)

    def test_zero_is_valid(self) -> None:
        assert to_float(0) == 0.0
        assert to_float("0") == 0.0

    def test_invalid_string_returns_none(self) -> None:
        assert to_float("not a number") is None

    def test_none_returns_none(self) -> None:
        assert to_float(None) is None

    def test_nan_returns_none(self) -> None:
        assert to_float(float("nan")) is None

    def test_inf_returns_none(self) -> None:
        assert to_float(float("inf")) is None
        assert to_float(float("-inf")) is None

    def test_default_returned_on_invalid(self) -> None:
        assert to_float("bad", default=0.0) == 0.0

    def test_default_returned_on_nan(self) -> None:
        assert to_float(float("nan"), default=0.0) == 0.0


class TestParseCell:
    def test_json_dict_string(self) -> None:
        result = parse_cell('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_list_string(self) -> None:
        result = parse_cell("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_plain_string_unchanged(self) -> None:
        assert parse_cell("hello") == "hello"

    def test_invalid_json_returns_original_string(self) -> None:
        s = "[not valid json{]"
        assert parse_cell(s) == s

    def test_empty_string_unchanged(self) -> None:
        assert parse_cell("") == ""

    def test_non_string_int(self) -> None:
        assert parse_cell(42) == 42

    def test_non_string_list(self) -> None:
        assert parse_cell([1, 2]) == [1, 2]

    def test_whitespace_prefix_stripped_for_detection(self) -> None:
        result = parse_cell('  {"key": 1}')
        assert result == {"key": 1}

    def test_empty_json_array(self) -> None:
        assert parse_cell("[]") == []

    def test_empty_json_object(self) -> None:
        assert parse_cell("{}") == {}


class TestMedian:
    def test_odd_length_list(self) -> None:
        assert median([1.0, 2.0, 3.0]) == 2.0

    def test_even_length_list(self) -> None:
        assert median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single_element(self) -> None:
        assert median([5.0]) == 5.0

    def test_empty_returns_none(self) -> None:
        assert median([]) is None

    def test_unsorted_list(self) -> None:
        assert median([3.0, 1.0, 2.0]) == 2.0

    def test_negative_values(self) -> None:
        assert median([-3.0, -1.0, -2.0]) == -2.0
