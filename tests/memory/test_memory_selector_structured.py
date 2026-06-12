"""Unit tests for the structured-output path in MemoryReadPipeline.

These pin the contract that:
- ``select()`` extracts card IDs from ``ExperimentalDecision.top_ideas[].card_id``
  via Pydantic validation (no regex on prose).
- ``select()`` resolves card text via ``memory.get_card(card_id).description``
  (no regex on prose).
- ``select()`` runs a Thompson auction over the candidate slate: each card's
  downside posterior competes against a no-card arm, so the emergent winner count
  is 0..N (not a fixed ``max_cards`` slice), and the per-candidate decisions land
  in ``selection.slate``.
- Invalid ``raw_memory`` shapes degrade to an empty selection, not a crash.
"""

from __future__ import annotations

from typing import Any

import pytest

from gigaevo.memory.core import (
    EfficacyCardRenderer,
    LLMCardSelector,
    MemorySelection,
)
from gigaevo.memory.shared_memory.models import MemoryCard
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.read_pipeline import make_read_pipeline

_PROVEN = (200.0, 1.0)
_SUSPECT = (1.0, 200.0)
_SEED = 20260604


class _StubResearchOutput:
    def __init__(self, *, integrated_memory: str = "", raw_memory: Any = None) -> None:
        self.integrated_memory = integrated_memory
        self.raw_memory = raw_memory


class _StubMemory:
    """Minimal memory backend exposing ``research`` + ``get_card`` like AmemGamMemory."""

    def __init__(self, *, raw_memory: Any, cards: dict[str, Any] | None = None) -> None:
        self._raw_memory = raw_memory
        self._cards = cards or {}
        self.research_calls: list[dict[str, Any]] = []

    def research(
        self,
        request: str,
        memory_state: str | None = None,
        planning_request: str | None = None,
    ):
        self.research_calls.append(
            {"request": request, "planning_request": planning_request}
        )
        return _StubResearchOutput(integrated_memory="", raw_memory=self._raw_memory)

    def get_card(self, card_id: str) -> Any:
        return self._cards.get(card_id)


def _card(description: str, posterior: tuple[float, float] | None = None) -> MemoryCard:
    stats: dict = {}
    if posterior is not None:
        stats = {"ALL": {"posterior_a": posterior[0], "posterior_b": posterior[1]}}
    return MemoryCard(id="idea-x", description=description, evolution_statistics=stats)


def _make_program(code: str = "def solve(): return 1") -> Program:
    return Program(code=code, state=ProgramState.DONE)


@pytest.mark.asyncio
async def test_select_pulls_ids_from_top_ideas_card_id():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "idea-A"}, {"card_id": "idea-B"}],
                "additional_queries": [],
            }
        },
        cards={
            "idea-A": _card("Try simulated annealing", _PROVEN),
            "idea-B": _card("Filter low-confidence hops", _PROVEN),
        },
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=2,
    )

    assert selection.card_ids == ["idea-A", "idea-B"]
    assert any("simulated annealing" in c for c in selection.cards)
    assert any("low-confidence" in c for c in selection.cards)


@pytest.mark.asyncio
async def test_select_resolves_description_from_typed_card():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "idea-1"}],
                "additional_queries": [],
            }
        },
        cards={"idea-1": _card("Use a heap for sorted retrieval", _PROVEN)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=1,
    )

    assert selection.cards == ["Use a heap for sorted retrieval"]
    assert selection.card_ids == ["idea-1"]


@pytest.mark.asyncio
async def test_max_cards_hard_caps_auction_winners():
    # The auction prunes emergently (0..N winners), but max_cards is a hard
    # ceiling on what reaches the mutator: with five PROVEN winners and
    # max_cards=2 exactly the two highest-theta winners survive. The slate keeps
    # all five per-candidate draws for offline audit.
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": f"id-{i}"} for i in range(5)],
                "additional_queries": [],
            }
        },
        cards={f"id-{i}": _card(f"card {i}", _PROVEN) for i in range(5)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=2,
    )

    assert len(selection.card_ids) == 2
    assert len(selection.cards) == 2
    theta = {bid.card_id: bid.theta for bid in selection.slate}
    assert len(selection.slate) == 5
    assert selection.card_ids == sorted(theta, key=theta.get, reverse=True)[:2]


@pytest.mark.asyncio
async def test_max_cards_one_returns_single_card_when_auction_keeps_three():
    # Live config (max_cards=1): even when the LLM proposes three cards and the
    # auction keeps all three, the mutator sees exactly one.
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": f"id-{i}"} for i in range(3)],
                "additional_queries": [],
            }
        },
        cards={f"id-{i}": _card(f"card {i}", _PROVEN) for i in range(3)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=1,
    )

    assert len(selection.card_ids) == 1
    assert len(selection.cards) == 1
    assert selection.card_ids[0] in {f"id-{i}" for i in range(3)}


@pytest.mark.asyncio
async def test_select_invalid_raw_memory_returns_empty():
    memory = _StubMemory(
        raw_memory={"final_decision": {"mode": "nope", "top_ideas": "not-a-list"}}
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


@pytest.mark.asyncio
async def test_select_missing_final_decision_returns_empty():
    memory = _StubMemory(raw_memory={"other_key": "irrelevant"})
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


@pytest.mark.asyncio
async def test_select_skips_missing_cards_silently():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [
                    {"card_id": "exists"},
                    {"card_id": "missing"},
                ],
                "additional_queries": [],
            }
        },
        cards={"exists": _card("real card", _PROVEN)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=5,
    )

    # 'missing' is not fetchable -> excluded from the auction (not a phantom id).
    assert selection.card_ids == ["exists"]
    assert selection.cards == ["real card"]
    assert [bid.card_id for bid in selection.slate] == ["exists"]


@pytest.mark.asyncio
async def test_select_research_exception_returns_empty():
    class _ThrowingMemory:
        def research(self, request, memory_state=None, planning_request=None):
            raise RuntimeError("backend exploded")

        def get_card(self, card_id):
            return None

    pipeline = make_read_pipeline(_ThrowingMemory(), seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


@pytest.mark.asyncio
async def test_auction_keeps_proven_drops_suspect():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "good"}, {"card_id": "bad"}],
                "additional_queries": [],
            }
        },
        cards={
            "good": _card("proven move", _PROVEN),
            "bad": _card("suspect move", _SUSPECT),
        },
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=2,
    )

    assert selection.card_ids == ["good"]
    assert selection.cards == ["proven move"]
    # The slate records BOTH candidates and the suspect's rejection (auditable).
    assert [bid.card_id for bid in selection.slate] == ["good", "bad"]
    assert [bid.selected for bid in selection.slate] == [True, False]


@pytest.mark.asyncio
async def test_auction_can_select_zero_cards():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": f"s{i}"} for i in range(3)],
                "additional_queries": [],
            }
        },
        cards={f"s{i}": _card(f"suspect {i}", _SUSPECT) for i in range(3)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=3,
    )

    assert selection.card_ids == []
    assert selection.cards == []
    # The "no-card" outcome is still recorded: 3 rejected candidates.
    assert len(selection.slate) == 3
    assert all(bid.selected is False for bid in selection.slate)


@pytest.mark.asyncio
async def test_slate_records_posteriors_and_baseline_arm():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "good"}],
                "additional_queries": [],
            }
        },
        cards={"good": _card("proven move", _PROVEN)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=1,
    )

    bid = selection.slate[0]
    assert bid.card_id == "good"
    assert bid.posterior_a == 200.0
    assert bid.posterior_b == 1.0
    assert bid.baseline_a == 3.0
    assert bid.baseline_b == 3.0


@pytest.mark.asyncio
async def test_card_without_track_record_is_cold_one_one():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "cold"}],
                "additional_queries": [],
            }
        },
        cards={"cold": _card("untried move")},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    selection = await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=1,
    )

    bid = selection.slate[0]
    assert bid.posterior_a == 1.0
    assert bid.posterior_b == 1.0


@pytest.mark.asyncio
async def test_select_passes_role_free_planning_request():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "idea-A"}],
                "additional_queries": [],
            }
        },
        cards={"idea-A": _card("proven move", _PROVEN)},
    )
    pipeline = make_read_pipeline(memory, seed=_SEED)

    await pipeline.select(
        parents=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        max_cards=1,
    )

    call = memory.research_calls[0]
    planning = call["planning_request"]
    # planning request is the role-free core: retrieval planning doesn't need
    # the selector ROLE; the full request (role + core) drives reflection.
    assert planning is not None
    assert planning.startswith("MUTATION INPUTS")
    assert call["request"].endswith(planning)
    assert call["request"] != planning


def test_missing_final_decision_warns_once_per_selector_instance():
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="WARNING")
    try:
        selector = LLMCardSelector()
        selector.shortlist({"pipeline_mode": "default"})
        selector.shortlist({"pipeline_mode": "default"})
        LLMCardSelector().shortlist({"pipeline_mode": "default"})
    finally:
        logger.remove(sink_id)

    hits = [r for r in records if "final_decision" in r]
    assert len(hits) == 2
    assert "default" in hits[0]


def test_parse_final_decision_handles_non_dict_raw_memory():
    assert LLMCardSelector().shortlist(None) == []
    assert LLMCardSelector().shortlist("not a dict") == []


def test_render_card_handles_typed_card_and_none():
    render = EfficacyCardRenderer().render
    assert render(None) == ""
    assert render(MemoryCard(id="idea-1", description=" trim me  ")) == "trim me"
    assert render(MemoryCard(id="idea-2")) == ""
