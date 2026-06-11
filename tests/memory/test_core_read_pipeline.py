"""Golden + unit tests for the modular read pipeline.

``MemoryReadPipeline`` composed of GamRetriever / LLMCardSelector /
ThompsonAuctioneer / TopThetaBudgeter / EfficacyCardRenderer.
End-to-end goldens were frozen from the legacy ``MemorySelectorAgent.select()``
(seed-exact) before its deletion.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gigaevo.memory.core import (
    BetaBinomialReputation,
    EfficacyCardRenderer,
    GamRetriever,
    LLMCardSelector,
    MemorySelection,
    TopThetaBudgeter,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.read_pipeline import make_read_pipeline

_PROVEN = (200.0, 1.0)
_SUSPECT = (1.0, 200.0)
_SEED = 20260610


class _StubResearchOutput:
    def __init__(self, *, integrated_memory: str = "", raw_memory: Any = None) -> None:
        self.integrated_memory = integrated_memory
        self.raw_memory = raw_memory


class _StubBackend:
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


class _ExplodingResearchBackend(_StubBackend):
    def research(self, request, memory_state=None, planning_request=None):
        raise RuntimeError("research boom")


class _ExplodingGetCardBackend(_StubBackend):
    def get_card(self, card_id):
        raise RuntimeError("get_card boom")


def _card(description: str, posterior: tuple[float, float] | None = None) -> dict:
    card: dict = {"description": description}
    if posterior is not None:
        card["evolution_statistics"] = {
            "ALL": {"posterior_a": posterior[0], "posterior_b": posterior[1]}
        }
    return card


def _make_program(code: str = "def solve(): return 1") -> Program:
    return Program(code=code, state=ProgramState.DONE)


def _decision(ids: list[str]) -> dict:
    return {
        "final_decision": {
            "mode": "final",
            "top_ideas": [{"card_id": cid} for cid in ids],
            "additional_queries": [],
        }
    }


class TestLLMCardSelector:
    def test_query_embeds_core_request_and_inputs(self):
        parents = [_make_program("def solve(): return 42")]
        sel = LLMCardSelector()
        kwargs = dict(
            parents=parents,
            mutation_mode="rewrite",
            task_description="predict prices",
            metrics_description="r2",
            max_cards=2,
        )
        core = sel.build_core_request(**kwargs)
        query = sel.build_query(**kwargs)
        assert "MUTATION INPUTS" in core
        assert "pick up to 2 card(s)" in core
        assert "predict prices" in core
        assert "r2" in core
        assert "rewrite" in core
        assert "def solve(): return 42" in core
        assert core in query
        assert len(query) > len(core)

    def test_shortlist_orders_ids_from_top_ideas(self):
        raw = _decision(["x", "y"])
        assert LLMCardSelector().shortlist(raw) == ["x", "y"]

    def test_shortlist_dedups_repeated_ids_keeping_first_position(self):
        raw = _decision(["x", "y", "x", "z", "y"])
        assert LLMCardSelector().shortlist(raw) == ["x", "y", "z"]

    def test_shortlist_empty_on_bad_shapes(self):
        for raw in [
            None,
            "prose answer",
            {},
            {"final_decision": "nope"},
            {"final_decision": {"mode": "final", "top_ideas": "bad"}},
        ]:
            assert LLMCardSelector().shortlist(raw) == []


class TestGamRetriever:
    def test_get_card_fail_to_none(self):
        retriever = GamRetriever(_ExplodingGetCardBackend(raw_memory=None))
        assert retriever.get_card("x") is None

    def test_get_card_passthrough(self):
        retriever = GamRetriever(
            _StubBackend(raw_memory=None, cards={"a": {"description": "d"}})
        )
        assert retriever.get_card("a") == {"description": "d"}
        assert retriever.get_card("missing") is None


class TestTopThetaBudgeter:
    def test_within_cap_preserves_auction_order(self):
        slate = [{"card_id": "a", "theta": 0.1}, {"card_id": "b", "theta": 0.9}]
        assert TopThetaBudgeter().cap(["a", "b"], slate, 3) == ["a", "b"]

    def test_over_cap_keeps_top_theta(self):
        slate = [
            {"card_id": "a", "theta": 0.2},
            {"card_id": "b", "theta": 0.9},
            {"card_id": "c", "theta": 0.5},
        ]
        assert TopThetaBudgeter().cap(["a", "b", "c"], slate, 2) == ["b", "c"]

    def test_missing_theta_defaults_to_zero(self):
        slate = [{"card_id": "b", "theta": 0.4}]
        assert TopThetaBudgeter().cap(["a", "b"], slate, 1) == ["b"]


_RENDER_GOLDENS = [
    (None, ""),
    (
        {
            "description": "Use pairwise ratios",
            "explanation": {"summary": "exposes scale-free structure"},
            "evolution_statistics": {
                "ALL": {
                    "intro_events": 4,
                    "IntroGain_best_median": 0.05,
                    "DownsideRate_best": 0.1,
                    "efficacy_confident": True,
                }
            },
        },
        "Use pairwise ratios\n"
        "mechanism: exposes scale-free structure\n"
        "efficacy: introduced in 4 children; median improvement +0.0500; "
        "downside 10% (confident)",
    ),
    (
        {"description": "Same text", "explanation": {"summary": "Same text"}},
        "Same text",
    ),
    (
        {
            "description": "Adjusted median wins",
            "evolution_statistics": {
                "ALL": {
                    "intro_events": 2,
                    "IntroGain_best_median": 0.5,
                    "IntroGain_best_adj_median": -0.01,
                    "efficacy_confident": True,
                }
            },
        },
        "Adjusted median wins\n"
        "efficacy: introduced in 2 children; median improvement vs cohort "
        "-0.0100 (caution: non-positive median)",
    ),
    (
        {
            "description": "Not confident stays silent",
            "evolution_statistics": {
                "ALL": {
                    "intro_events": 5,
                    "IntroGain_best_median": 0.2,
                    "efficacy_confident": False,
                }
            },
        },
        "Not confident stays silent",
    ),
    (
        {"id": "program-1", "description": "Exemplar", "fitness": 0.8123},
        "Exemplar\nefficacy: exemplar fitness 0.8123",
    ),
    (
        SimpleNamespace(
            description="Attr card",
            explanation=SimpleNamespace(summary="attr mechanism"),
            evolution_statistics=None,
        ),
        "Attr card\nmechanism: attr mechanism",
    ),
]


class TestEfficacyCardRenderer:
    @pytest.mark.parametrize("card, expected", _RENDER_GOLDENS)
    def test_render_golden(self, card, expected):
        assert EfficacyCardRenderer().render(card) == expected


class TestCardPosterior:
    @pytest.mark.parametrize(
        "card, expected",
        [
            (None, (1.0, 1.0)),
            ({}, (1.0, 1.0)),
            (
                {
                    "evolution_statistics": {
                        "ALL": {"posterior_a": 4.0, "posterior_b": 2.0}
                    }
                },
                (4.0, 2.0),
            ),
            ({"evolution_statistics": {"ALL": {"posterior_a": 4.0}}}, (1.0, 1.0)),
            ({"evolution_statistics": "malformed"}, (1.0, 1.0)),
            (
                SimpleNamespace(
                    evolution_statistics={
                        "ALL": {"posterior_a": 7.0, "posterior_b": 3.0}
                    }
                ),
                (7.0, 3.0),
            ),
            (SimpleNamespace(evolution_statistics=None), (1.0, 1.0)),
        ],
    )
    def test_card_posterior_golden(self, card, expected):
        assert BetaBinomialReputation().card_posterior(card) == expected

    def test_cold_prior_configurable(self):
        assert BetaBinomialReputation(cold_prior=(2.0, 5.0)).card_posterior({}) == (
            2.0,
            5.0,
        )


def _mk_backend() -> _StubBackend:
    return _StubBackend(
        raw_memory=_decision([f"idea-{i}" for i in range(5)]),
        cards={
            f"idea-{i}": _card(f"Card {i}", _PROVEN if i % 2 == 0 else _SUSPECT)
            for i in range(5)
        },
    )


_SELECT_KWARGS = dict(
    mutation_mode="rewrite",
    task_description="t",
    metrics_description="m",
    max_cards=2,
)


class TestMemoryReadPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_golden(self):
        backend = _mk_backend()
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-2", "idea-4"]
        assert got.cards == ["Card 2", "Card 4"]
        assert len(got.slate) == 5
        by_id = {e["card_id"]: e for e in got.slate}
        assert [by_id[f"idea-{i}"]["selected"] for i in range(5)] == [
            True,
            False,
            True,
            False,
            True,
        ]
        assert len(backend.research_calls) == 1

    @pytest.mark.asyncio
    async def test_empty_when_max_cards_nonpositive(self):
        got = await make_read_pipeline(_mk_backend(), seed=_SEED).select(
            parents=[_make_program()],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=0,
        )
        assert got == MemorySelection(cards=[], card_ids=[])

    @pytest.mark.asyncio
    async def test_empty_when_retriever_missing(self):
        got = await make_read_pipeline(None, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got == MemorySelection(cards=[], card_ids=[])

    @pytest.mark.asyncio
    async def test_empty_when_research_raises(self):
        backend = _ExplodingResearchBackend(raw_memory=None)
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got == MemorySelection(cards=[], card_ids=[])

    @pytest.mark.asyncio
    async def test_empty_render_drops_card_and_id_in_lockstep(self):
        # An empty render must not desync cards from card_ids — every
        # downstream consumer zips them (prompt block, citation analytics).
        backend = _StubBackend(
            raw_memory=_decision(["idea-blank", "idea-good"]),
            cards={
                "idea-blank": _card("", _PROVEN),
                "idea-good": _card("Good card", _PROVEN),
            },
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-good"]
        assert got.cards == ["Good card"]

    @pytest.mark.asyncio
    async def test_empty_when_card_posterior_malformed(self):
        # A corrupt persisted posterior must degrade to an empty selection,
        # never an exception that sinks the mutation.
        bad = _card("Bad card")
        bad["evolution_statistics"] = {
            "ALL": {"posterior_a": "corrupt", "posterior_b": 1.0}
        }
        backend = _StubBackend(
            raw_memory=_decision(["idea-bad"]),
            cards={"idea-bad": bad},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got == MemorySelection(cards=[], card_ids=[])

    @pytest.mark.asyncio
    async def test_nonfinite_posterior_never_selected(self):
        bad = _card("NaN card", (float("nan"), 1.0))
        backend = _StubBackend(
            raw_memory=_decision(["idea-nan"]),
            cards={"idea-nan": bad},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == []
        assert got.cards == []

    @pytest.mark.asyncio
    async def test_unfetchable_cards_skipped(self):
        backend = _StubBackend(
            raw_memory=_decision(["idea-A", "idea-missing"]),
            cards={"idea-A": _card("Only card", _PROVEN)},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-A"]
        assert len(got.slate) == 1
