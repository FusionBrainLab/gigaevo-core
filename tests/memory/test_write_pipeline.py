"""Extended tests for write_pipeline — edge cases for load_memory_cards.

Complements test_memory_write_program_cards.py with adversarial inputs.
"""

import json

import pytest

from gigaevo.memory.write_pipeline import (
    _card_type,
    _latest_snapshot,
    _top_percent_count,
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
        assert cards[0].id == "i1"

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
        program_cards = [c for c in cards if c.category == "program"]
        assert program_cards == []

    def test_best_idea_missing_from_bank_is_skipped(self, tmp_path):
        """best_ideas ID not present in banks.json must be skipped — no ghost cards."""
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "missing-1", "description": "desc"}],
        )
        cards = load_memory_cards(banks, best)
        assert cards == []

    def test_best_idea_present_in_bank_is_included(self, tmp_path):
        """An idea that exists in both best_ideas and banks must be returned."""
        banks = _make_banks(
            tmp_path,
            active_bank=[{"id": "real-1", "description": "real idea"}],
        )
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "real-1", "fitness": 0.9}],
        )
        cards = load_memory_cards(banks, best)
        assert len(cards) == 1
        assert cards[0].id == "real-1"

    def test_programs_sorted_by_fitness(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": "p1",
                    "fitness": 50.0,
                    "code": "a",
                    "task_description_summary": "t",
                },
                {
                    "id": "p2",
                    "fitness": 90.0,
                    "code": "b",
                    "task_description_summary": "t",
                },
                {
                    "id": "p3",
                    "fitness": 70.0,
                    "code": "c",
                    "task_description_summary": "t",
                },
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.category == "program"]
        fitnesses = [c.fitness for c in program_cards]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_program_without_fitness_skipped(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {"id": "p1", "code": "a"},  # no fitness
                {
                    "id": "p2",
                    "fitness": 80.0,
                    "code": "b",
                    "task_description_summary": "t",
                },
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.category == "program"]
        assert len(program_cards) == 1
        assert program_cards[0].program_id == "p2"

    def test_invalid_program_skipped(self, tmp_path):
        """Programs with is_valid=0 must not be written to memory."""
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": "p-invalid",
                    "fitness": 90.0,
                    "is_valid": 0.0,
                    "code": "a",
                    "task_description_summary": "t",
                },
                {
                    "id": "p-valid",
                    "fitness": 80.0,
                    "is_valid": 1.0,
                    "code": "b",
                    "task_description_summary": "t",
                },
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.category == "program"]
        assert len(program_cards) == 1
        assert program_cards[0].program_id == "p-valid"

    def test_program_missing_is_valid_accepted(self, tmp_path):
        """Programs without is_valid field are accepted — ideas_tracker pre-filters invalids
        before writing programs.json, so absence of is_valid means already-valid."""
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": "p-no-validity",
                    "fitness": 85.0,
                    "code": "a",
                    "task_description_summary": "t",
                    # no is_valid: treated as valid (ideas_tracker format)
                },
            ],
        )
        cards = load_memory_cards(
            banks, best, programs_path=programs, best_programs_percent=100.0
        )
        program_cards = [c for c in cards if c.category == "program"]
        assert len(program_cards) == 1
        assert program_cards[0].program_id == "p-no-validity"

    def test_ideas_tracker_dict_aliases_preserved(self, tmp_path):
        """Integration: ideas_tracker writes aliases as list[dict] version history.

        This is the boundary where ideas_tracker output enters Pydantic land.
        Bug #2 (PR #161): MemoryCard.aliases was list[str], crashed on list[dict].
        Fixed by changing to list[Any].
        """
        aliases = [
            {
                "exp1-prog1": {
                    "description": "old description",
                    "programs": ["p1"],
                    "explanations": ["initial"],
                }
            },
        ]
        banks = _make_banks(
            tmp_path,
            active_bank=[
                {
                    "id": "idea-1",
                    "description": "current description",
                    "aliases": aliases,
                    "keywords": ["retrieval"],
                }
            ],
        )
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[{"idea_id": "idea-1"}],
        )
        cards = load_memory_cards(banks, best)
        assert len(cards) == 1
        assert cards[0].id == "idea-1"
        assert cards[0].aliases == aliases
        assert isinstance(cards[0].aliases[0], dict)

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


# ===========================================================================
# usage_updates_path regression test (top-level function)
# ===========================================================================


def test_usage_updates_path_uses_config_when_caller_omits_it(tmp_path):
    """Regression: caller passing banks_path but not usage_updates_path must get config path."""
    from unittest.mock import MagicMock, patch

    from gigaevo.memory.write_pipeline import main

    # Create banks file so the existence check inside main() passes
    banks_file = tmp_path / "banks.json"
    banks_file.write_text("{}")

    # MagicMock without spec so any attribute access auto-returns a Mock
    cfg = MagicMock()
    cfg.banks_path = banks_file
    cfg.best_ideas_path = tmp_path / "best_ideas.json"
    cfg.programs_path = tmp_path / "programs.json"
    cfg.usage_updates_path = tmp_path / "usage_updates.json"
    cfg.use_api = False
    cfg.memory_dir = tmp_path / "memory"
    cfg.search_limit = 5
    cfg.rebuild_interval = 10
    cfg.enable_llm_synthesis = False
    cfg.should_evolve = False
    cfg.fill_missing_fields_with_llm = False
    cfg.enable_bm25 = False
    cfg.allowed_gam_tools = []
    cfg.gam_top_k_by_tool = {}
    cfg.gam_pipeline_mode = "default"
    cfg.card_update_dedup_config = {}
    cfg.best_programs_percent = 5.0
    cfg.sync_batch_size = 100
    cfg.sync_on_init = True
    cfg.channel = "latest"
    cfg.author = None
    cfg.namespace = "default"
    cfg.enable_usage_tracking = True
    cfg.settings_path = tmp_path / "settings.yaml"

    captured: dict = {}

    def fake_load_cards(*_args, usage_updates_path=None, **_kwargs):
        captured["usage_updates_path"] = usage_updates_path
        return []  # empty list so main() iterates nothing and returns cleanly

    with (
        patch("gigaevo.memory.write_pipeline.load_config", return_value=cfg),
        patch(
            "gigaevo.memory.write_pipeline.load_memory_cards",
            side_effect=fake_load_cards,
        ),
        patch("gigaevo.memory.write_pipeline.AmemGamMemory") as MockAmem,
    ):
        MockAmem.return_value.get_card_write_stats.return_value = {}
        main(banks_path=banks_file, usage_updates_path=None)

    assert captured["usage_updates_path"] == tmp_path / "usage_updates.json", (
        "Config's usage_updates_path should be used when caller passes None"
    )
