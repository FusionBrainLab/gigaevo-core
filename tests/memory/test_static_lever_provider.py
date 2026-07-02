"""Tests for gigaevo.memory.provider.StaticLeverMemoryProvider."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.core import MemorySelection
from gigaevo.memory.provider import StaticLeverMemoryProvider
from gigaevo.programs.program import Program

LEVERS = """\
Target the bottleneck triangle: move only the points of the current minimum.

---

Escape via reheating: raise the SA temperature after a stall.
---
Project every proposal back into the feasible simplex.
"""


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


def _write_levers(tmp_path, text: str = LEVERS, name: str = "levers_core6.md"):
    path = tmp_path / name
    path.write_text(text)
    return path


class TestStaticLeverMemoryProvider:
    @pytest.mark.asyncio
    async def test_returns_one_block_per_separator_segment(self, tmp_path) -> None:
        provider = StaticLeverMemoryProvider(levers_file=_write_levers(tmp_path))
        result = await provider.select_cards(
            program=_make_program(),
            task_description="some task",
            metrics_description="fitness: higher is better",
        )
        assert isinstance(result, MemorySelection)
        assert len(result.cards) == 3
        assert result.cards[0].startswith("Target the bottleneck")
        assert result.cards[1].startswith("Escape via reheating")
        assert result.cards[2].startswith("Project every proposal")

    @pytest.mark.asyncio
    async def test_card_ids_are_stable_and_stem_scoped(self, tmp_path) -> None:
        provider = StaticLeverMemoryProvider(levers_file=_write_levers(tmp_path))
        result = await provider.select_cards(
            program=_make_program(),
            task_description="",
            metrics_description="",
        )
        assert result.card_ids == [
            "static:levers_core6:1",
            "static:levers_core6:2",
            "static:levers_core6:3",
        ]

    @pytest.mark.asyncio
    async def test_selection_is_identical_for_every_program(self, tmp_path) -> None:
        provider = StaticLeverMemoryProvider(levers_file=_write_levers(tmp_path))
        first = await provider.select_cards(
            program=_make_program("def a(): pass"),
            task_description="task one",
            metrics_description="m1",
        )
        second = await provider.select_cards(
            program=_make_program("def b(): pass"),
            task_description="task two",
            metrics_description="m2",
            parent_context="parent notes",
        )
        assert first.cards == second.cards
        assert first.card_ids == second.card_ids

    def test_accepts_and_ignores_assembler_component_kwargs(self, tmp_path) -> None:
        partial = functools.partial(
            StaticLeverMemoryProvider, levers_file=_write_levers(tmp_path)
        )
        provider = partial(
            backend=None,
            retriever=None,
            selector=None,
            auctioneer=None,
            budgeter=None,
            reputation=None,
            excluder=None,
        )
        assert isinstance(provider, StaticLeverMemoryProvider)

    def test_missing_file_raises_at_construction(self, tmp_path) -> None:
        with pytest.raises(MemoryStorageError, match="levers file"):
            StaticLeverMemoryProvider(levers_file=tmp_path / "absent.md")

    def test_empty_file_raises_at_construction(self, tmp_path) -> None:
        with pytest.raises(MemoryStorageError, match="levers file"):
            StaticLeverMemoryProvider(
                levers_file=_write_levers(tmp_path, text="\n\n---\n\n")
            )

    def test_block_count_mismatch_raises_at_construction(self, tmp_path) -> None:
        with pytest.raises(MemoryStorageError, match="expected 6"):
            StaticLeverMemoryProvider(
                levers_file=_write_levers(tmp_path), expected_blocks=6
            )

    def test_block_count_match_accepted(self, tmp_path) -> None:
        provider = StaticLeverMemoryProvider(
            levers_file=_write_levers(tmp_path), expected_blocks=3
        )
        assert isinstance(provider, StaticLeverMemoryProvider)


EXPERIMENT_LEVER_FILES = [
    "experiments/static_lever_prompt_baseline/levers_core6.md",
    "experiments/static_lever_prompt_baseline/levers_tail.md",
]


@pytest.mark.parametrize("levers_file", EXPERIMENT_LEVER_FILES)
def test_experiment_lever_files_parse_into_six_blocks(levers_file) -> None:
    StaticLeverMemoryProvider(
        levers_file=Path(__file__).parents[2] / levers_file, expected_blocks=6
    )
