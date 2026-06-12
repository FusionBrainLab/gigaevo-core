"""Extended tests for write_pipeline — edge cases for load_memory_cards.

Complements test_memory_write_program_cards.py with adversarial inputs.
"""

import json

import pytest

from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import (
    CardAlias,
    CardStatsBlock,
    ConnectedIdea,
    MemoryCard,
    ProgramCard,
    Quartile,
)
from gigaevo.memory.write_pipeline import (
    ProgramRow,
    WriteStats,
    build_program_cards_from_top_programs,
    classify_card_type,
    extract_latest_snapshot,
    load_best_idea_bank_cards,
    load_memory_cards,
    parse_best_ideas,
    parse_programs,
    top_percent_count,
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
# extract_latest_snapshot
# ===========================================================================


class TestLatestSnapshot:
    def test_dict_with_key(self):
        result = extract_latest_snapshot({"active_bank": [1]}, "active_bank")
        assert result == {"active_bank": [1]}

    def test_list_takes_last(self):
        payload = [
            {"active_bank": [1]},
            {"active_bank": [2]},
        ]
        result = extract_latest_snapshot(payload, "active_bank")
        assert result["active_bank"] == [2]

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing key"):
            extract_latest_snapshot({"other": 1}, "active_bank")

    def test_list_no_matching_key_raises(self):
        with pytest.raises(ValueError, match="No snapshot"):
            extract_latest_snapshot([{"other": 1}], "active_bank")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid snapshot"):
            extract_latest_snapshot("string", "active_bank")


# ===========================================================================
# top_percent_count
# ===========================================================================


class TestTopPercentCount:
    def test_basic(self):
        assert top_percent_count(100, 5.0) == 5

    def test_rounds_up(self):
        assert top_percent_count(10, 5.0) == 1  # ceil(0.5) = 1

    def test_minimum_one(self):
        assert top_percent_count(1000, 0.01) >= 1

    def test_zero_total(self):
        assert top_percent_count(0, 5.0) == 0

    def test_zero_percent(self):
        assert top_percent_count(100, 0.0) == 0


# ===========================================================================
# classify_card_type
# ===========================================================================


class TestCardType:
    def test_program_by_category(self):
        card = normalize_memory_card({"id": "program-p1", "category": "program"})
        assert classify_card_type(card) == "programs"

    def test_program_by_program_id(self):
        card = normalize_memory_card({"id": "program-p1", "program_id": "p1"})
        assert classify_card_type(card) == "programs"

    def test_idea(self):
        card = normalize_memory_card({"id": "idea-1", "category": "general"})
        assert classify_card_type(card) == "ideas"

    def test_empty(self):
        assert classify_card_type(normalize_memory_card({})) == "ideas"


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
            best_ideas=[{"idea_id": "i1", "quartile": "ALL"}],
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
            best_ideas=[{"idea_id": "i1", "quartile": "ALL"}],
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
            best_ideas=[
                {"idea_id": "missing-1", "quartile": "ALL", "description": "desc"}
            ],
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
            best_ideas=[{"idea_id": "real-1", "quartile": "ALL"}],
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


# ===========================================================================
# Direction-aware program selection (higher_is_better)
# ===========================================================================


class TestProgramSelectionDirection:
    """For lower-is-better problems (e.g. vartodd_ham_high, validation loss),
    the "best" programs are the LOWEST fitness, not the highest. The memory
    pipeline must honour metrics.yaml direction so the program cards fed to
    the mutator are actually the better programs.
    """

    def test_lower_is_better_picks_lowest_fitness(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": "p_worst",
                    "fitness": 500.0,
                    "code": "a",
                    "task_description_summary": "t",
                },
                {
                    "id": "p_best",
                    "fitness": 400.0,
                    "code": "b",
                    "task_description_summary": "t",
                },
                {
                    "id": "p_mid",
                    "fitness": 450.0,
                    "code": "c",
                    "task_description_summary": "t",
                },
            ],
        )
        cards = load_memory_cards(
            banks,
            best,
            programs_path=programs,
            best_programs_percent=1.0,
            higher_is_better=False,
        )
        program_cards = [c for c in cards if c.category == "program"]
        assert len(program_cards) == 1
        assert program_cards[0].program_id == "p_best"
        assert program_cards[0].fitness == 400.0

    def test_lower_is_better_orders_by_ascending_fitness(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": f"p{i}",
                    "fitness": float(fit),
                    "code": "x",
                    "task_description_summary": "t",
                }
                for i, fit in enumerate([500.0, 410.0, 450.0, 405.0, 460.0])
            ],
        )
        cards = load_memory_cards(
            banks,
            best,
            programs_path=programs,
            best_programs_percent=100.0,
            higher_is_better=False,
        )
        program_cards = [c for c in cards if c.category == "program"]
        fitnesses = [c.fitness for c in program_cards]
        assert fitnesses == sorted(fitnesses)

    def test_higher_is_better_default_picks_highest_fitness(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[])
        best = _make_best_ideas(tmp_path, best_ideas=[])
        programs = _make_programs(
            tmp_path,
            programs=[
                {
                    "id": "p_low",
                    "fitness": 0.1,
                    "code": "a",
                    "task_description_summary": "t",
                },
                {
                    "id": "p_high",
                    "fitness": 0.9,
                    "code": "b",
                    "task_description_summary": "t",
                },
            ],
        )
        cards = load_memory_cards(
            banks,
            best,
            programs_path=programs,
            best_programs_percent=50.0,
        )
        program_cards = [c for c in cards if c.category == "program"]
        assert len(program_cards) == 1
        assert program_cards[0].program_id == "p_high"

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

    def test_ideas_tracker_aliases_preserved(self, tmp_path):
        """Integration: alias version history written by ideas_tracker loads
        back as typed CardAlias entries."""
        aliases = [
            {
                "key": "idea-1-update",
                "description": "old description",
                "programs": ["p1"],
                "explanations": ["initial"],
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
            best_ideas=[{"idea_id": "idea-1", "quartile": "ALL"}],
        )
        cards = load_memory_cards(banks, best)
        assert len(cards) == 1
        assert cards[0].id == "idea-1"
        assert isinstance(cards[0].aliases[0], CardAlias)
        assert cards[0].aliases[0].key == "idea-1-update"

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
# Typed snapshot rows
# ===========================================================================


class TestBestIdeasRows:
    def test_parse_returns_typed_rows(self, tmp_path):
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[
                {
                    "idea_id": "i1",
                    "quartile": Quartile.ALL,
                    "description": "d",
                    "IntroGain_best_median": 0.5,
                }
            ],
        )
        idea_ids, by_id = parse_best_ideas(best)
        assert idea_ids == ["i1"]
        row = by_id["i1"]
        assert isinstance(row, IdeaStats)
        assert row.description == "d"
        assert row.IntroGain_best_median == 0.5

    def test_duplicate_and_blank_ids_skipped(self, tmp_path):
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[
                {"idea_id": "i1", "quartile": Quartile.ALL},
                {"idea_id": "i1", "quartile": Quartile.ALL},
                {"idea_id": " ", "quartile": Quartile.ALL},
            ],
        )
        idea_ids, _ = parse_best_ideas(best)
        assert idea_ids == ["i1"]


class TestProgramRow:
    def test_parse_returns_typed_rows(self, tmp_path):
        programs = _make_programs(
            tmp_path,
            programs=[{"id": "p1", "fitness": "0.5", "code": "x", "extra_metric": 1}],
        )
        rows = parse_programs(programs)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, ProgramRow)
        assert row.resolved_program_id == "p1"
        assert row.fitness == 0.5
        assert row.code == "x"

    def test_program_id_key_fallback(self, tmp_path):
        programs = _make_programs(
            tmp_path, programs=[{"program_id": "p9", "fitness": 1.0}]
        )
        rows = parse_programs(programs)
        assert rows[0].resolved_program_id == "p9"

    def test_unparseable_fitness_is_none(self, tmp_path):
        programs = _make_programs(
            tmp_path, programs=[{"id": "p1", "fitness": "not-a-number"}]
        )
        rows = parse_programs(programs)
        assert rows[0].fitness is None


class TestTypedCardBuilders:
    def test_build_program_cards_returns_program_cards(self, tmp_path):
        banks = _make_banks(
            tmp_path,
            active_bank=[{"id": "i1", "description": "idea desc", "programs": ["p1"]}],
        )
        programs = _make_programs(
            tmp_path,
            programs=[
                {"id": "p1", "fitness": 1.0, "code": "x", "task_description": "t"}
            ],
        )
        cards = build_program_cards_from_top_programs(
            programs_path=programs,
            banks_path=banks,
            best_programs_percent=100.0,
        )
        assert len(cards) == 1
        card = cards[0]
        assert isinstance(card, ProgramCard)
        assert card.id == "program-p1"
        assert card.connected_ideas == [
            ConnectedIdea(idea_id="i1", description="idea desc")
        ]
        assert card.description == "idea desc"

    def test_load_best_idea_bank_cards_returns_typed(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[{"id": "i1", "description": "d"}])
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[
                {
                    "idea_id": "i1",
                    "quartile": Quartile.ALL,
                    "IntroGain_best_median": 0.25,
                }
            ],
        )
        cards = load_best_idea_bank_cards(banks, best)
        assert len(cards) == 1
        assert isinstance(cards[0], MemoryCard)
        snapshot = cards[0].evolution_statistics.best_ideas_snapshot
        assert isinstance(snapshot, CardStatsBlock)
        assert snapshot.IntroGain_best_median == 0.25

    def test_best_idea_description_fills_missing_card_description(self, tmp_path):
        banks = _make_banks(tmp_path, active_bank=[{"id": "i1"}])
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[
                {"idea_id": "i1", "quartile": Quartile.ALL, "description": "from best"}
            ],
        )
        cards = load_best_idea_bank_cards(banks, best)
        assert cards[0].description == "from best"

    def test_best_idea_description_does_not_overwrite(self, tmp_path):
        banks = _make_banks(
            tmp_path, active_bank=[{"id": "i1", "description": "original"}]
        )
        best = _make_best_ideas(
            tmp_path,
            best_ideas=[
                {"idea_id": "i1", "quartile": Quartile.ALL, "description": "from best"}
            ],
        )
        cards = load_best_idea_bank_cards(banks, best)
        assert cards[0].description == "original"


class TestWriteStats:
    def test_delta_to(self):
        before = WriteStats(processed=2, added=1)
        after = WriteStats(processed=5, added=1, rejected=1)
        assert before.delta_to(after) == WriteStats(processed=3, rejected=1)

    def test_delta_clamps_at_zero(self):
        assert WriteStats(processed=5).delta_to(WriteStats(processed=2)).processed == 0

    def test_accumulate(self):
        total = (
            WriteStats()
            .accumulate(WriteStats(added=2))
            .accumulate(WriteStats(added=3, rejected=1))
        )
        assert total.added == 5
        assert total.rejected == 1

    def test_validates_backend_dict_ignoring_unknown_counters(self):
        stats = WriteStats.model_validate(
            {"processed": 3, "added": 1, "unknown_counter": 9}
        )
        assert stats.processed == 3
        assert stats.added == 1
