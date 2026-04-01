"""Extended tests for memory_write_example — edge cases for load_memory_cards.

Complements test_memory_write_program_cards.py with adversarial inputs.
"""

import json

import pytest

from gigaevo.memory.memory_write_example import (
    _latest_snapshot,
    _top_percent_count,
    _card_type,
    load_memory_cards,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _make_banks(tmp_path, active_bank=None):
    path = tmp_path / "banks.json"
    _write_json(path, [{"active_bank": active_bank or []}])
    return path


def _make_best_ideas(tmp_path, best_ideas=None):
    path = tmp_path / "best_ideas.json"
    _write_json(path, [{"best_ideas": best_ideas or []}])
    return path


def _make_programs(tmp_path, programs=None):
    path = tmp_path / "programs.json"
    _write_json(path, [{"programs": programs or []}])
    return path


# ===========================================================================
# _latest_snapshot
# ===========================================================================


class TestLatestSnapshot:
    def test_dict_with_key(self):
        result = _latest_snapshot({"active_bank": [1]}, "active_bank")
        assert result == {"active_bank": [1]}

    def test_list_takes_last(self):
        payload = [
            {"active_bank": [1]},
            {"active_bank": [2]},
        ]
        result = _latest_snapshot(payload, "active_bank")
        assert result["active_bank"] == [2]

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing key"):
            _latest_snapshot({"other": 1}, "active_bank")

    def test_list_no_matching_key_raises(self):
        with pytest.raises(ValueError, match="No snapshot"):
            _latest_snapshot([{"other": 1}], "active_bank")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid snapshot"):
            _latest_snapshot("string", "active_bank")


# ===========================================================================
# _top_percent_count
# ===========================================================================


class TestTopPercentCount:
    def test_basic(self):
        assert _top_percent_count(100, 5.0) == 5

    def test_rounds_up(self):
        assert _top_percent_count(10, 5.0) == 1  # ceil(0.5) = 1

    def test_minimum_one(self):
        assert _top_percent_count(1000, 0.01) >= 1

    def test_zero_total(self):
        assert _top_percent_count(0, 5.0) == 0

    def test_zero_percent(self):
        assert _top_percent_count(100, 0.0) == 0


# ===========================================================================
# _card_type
# ===========================================================================


class TestCardType:
    def test_program_by_category(self):
        assert _card_type({"category": "program"}) == "programs"

    def test_program_by_program_id(self):
        assert _card_type({"program_id": "p1"}) == "programs"

    def test_idea(self):
        assert _card_type({"category": "general"}) == "ideas"

    def test_empty(self):
        assert _card_type({}) == "ideas"


# ===========================================================================
# load_memory_cards edge cases
# ===========================================================================


class TestLoadMemoryCardsEdgeCases:
    def test_empty_active_bank(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        cards = load_memory_cards(banks, best)
        assert cards == []

    def test_no_programs_path(self, tmp_path):
        banks = _make_banks(
            tmp_path,
            active_bank=[{"id": "i1", "description": "idea"}],
        )
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "i1"}],
        )
        cards = load_memory_cards(banks, best, programs_path=None)
        # Should return ideas only, no program cards
        assert len(cards) == 1
        assert cards[0].get("id") == "i1"

    def test_zero_best_programs_percent(self, tmp_path):
        banks = _make_banks(
            tmp_path,
            active_bank=[{"id": "i1", "description": "idea"}],
        )
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "i1"}],
        )
        programs = _make_programs(
            tmp_path,
            programs=[{"id": "p1", "fitness": 90.0, "code": "pass"}],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=0.0
        )
        # Zero percent = no program cards
        program_cards = [c for c in cards if c.get("category") == "program"]
        assert program_cards == []

    def test_best_idea_missing_from_bank_creates_minimal_card(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "missing-1", "description": "desc"}],
        )
        cards = load_memory_cards(banks, best)
        assert len(cards) == 1
        assert cards[0]["id"] == "missing-1"

    def test_programs_sorted_by_fitness(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {"id": "p1", "fitness": 50.0, "code": "a", "task_description_summary": "t"},
                {"id": "p2", "fitness": 90.0, "code": "b", "task_description_summary": "t"},
                {"id": "p3", "fitness": 70.0, "code": "c", "task_description_summary": "t"},
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.get("category") == "program"]
        fitnesses = [c["fitness"] for c in program_cards]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_program_without_fitness_skipped(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {"id": "p1", "code": "a"},  # no fitness
                {"id": "p2", "fitness": 80.0, "code": "b", "task_description_summary": "t"},
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.get("category") == "program"]
        assert len(program_cards) == 1
        assert program_cards[0]["program_id"] == "p2"

    def test_missing_banks_file_raises(self, tmp_path):
        best = _make_best_ideas(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_memory_cards(tmp_path / "nonexistent.json", best)

    def test_invalid_json_format_raises(self, tmp_path):
        path = tmp_path / "banks.json"
        _write_json(path, {"no_active_bank": True})
        best = _make_best_ideas(tmp_path)
        with pytest.raises(ValueError):
            load_memory_cards(path, best)
