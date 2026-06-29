"""Golden + unit tests for the modular read pipeline.

``MemoryReadPipeline`` composed of GamRetriever / LLMCardSelector /
ThompsonAuctioneer / TopThetaBudgeter / EfficacyCardRenderer.
End-to-end goldens were frozen from the legacy ``MemorySelectorAgent.select()``
(seed-exact) before its deletion.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core import (
    AuctionBid,
    BetaBinomialReputation,
    EfficacyCardRenderer,
    EVThompsonAuctioneer,
    GamRetriever,
    LLMCardSelector,
    MemoryReadPipeline,
    MemorySelection,
    TopBidBudgeter,
    TopThetaBudgeter,
)
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.read_pipeline import make_read_pipeline

_SEED = 20260610


def _events(gains: list[float]) -> list[ContextualGain]:
    return [
        ContextualGain(
            context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=g
        )
        for g in gains
    ]


# 199 equal wins / losses resolve to the Beta(200, 1) / Beta(1, 200) downside
# posteriors the seed-exact auction goldens were frozen against.
_PROVEN_EVENTS = _events([0.01] * 199)
_SUSPECT_EVENTS = _events([-0.01] * 199)
# The efficacy line a PROVEN card (199 confident wins, +0.01 median) renders.
_PROVEN_EFFICACY = (
    "efficacy: introduced in 199 children; median improvement +0.0100 (confident)"
)


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
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ):
        self.research_calls.append(
            {
                "request": request,
                "planning_request": planning_request,
                "exclude_ids": exclude_ids,
                "random_drop_dose": random_drop_dose,
            }
        )
        return _StubResearchOutput(integrated_memory="", raw_memory=self._raw_memory)

    def get_card(self, card_id: str) -> Any:
        return self._cards.get(card_id)


class _ExplodingResearchBackend(_StubBackend):
    def research(
        self,
        request,
        memory_state=None,
        planning_request=None,
        *,
        exclude_ids=frozenset(),
        random_drop_dose=0,
    ):
        raise RuntimeError("research boom")


class _ExplodingGetCardBackend(_StubBackend):
    def get_card(self, card_id):
        raise RuntimeError("get_card boom")


def _card(description: str, events: list[ContextualGain] | None = None) -> MemoryCard:
    return MemoryCard(
        id=f"card-{description or 'blank'}",
        description=description,
        gain_events=events,
    )


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


def _event_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


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
        card = MemoryCard(id="a", description="d")
        retriever = GamRetriever(_StubBackend(raw_memory=None, cards={"a": card}))
        assert retriever.get_card("a") == card
        assert retriever.get_card("missing") is None

    def test_get_card_normalizes_legacy_dict_backend(self):
        # A dict-returning backend can hand back raw dicts; the retriever is
        # the typed boundary for them.
        raw = {
            "description": "legacy idea",
            "gain_events": [
                {"context": {"parent_metrics": {"min_area": 0.5}}, "gain": 0.01}
            ],
        }
        retriever = GamRetriever(_StubBackend(raw_memory=None, cards={"a": raw}))
        card = retriever.get_card("a")
        assert isinstance(card, MemoryCard)
        assert card.id == "a"
        assert card.description == "legacy idea"
        assert card.gain_events[0].gain == 0.01

    def test_get_card_corrupt_legacy_dict_fails_to_none(self):
        raw = {
            "description": "corrupt legacy",
            "gain_events": [
                {"context": {"parent_metrics": {"min_area": 0.5}}, "gain": "corrupt"}
            ],
        }
        retriever = GamRetriever(_StubBackend(raw_memory=None, cards={"a": raw}))
        assert retriever.get_card("a") is None


def _bid(card_id: str, theta: float) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=1.0,
        posterior_b=1.0,
        theta=theta,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=True,
    )


class TestTopThetaBudgeter:
    def test_within_cap_preserves_auction_order(self):
        slate = [_bid("a", 0.1), _bid("b", 0.9)]
        assert TopThetaBudgeter().cap(["a", "b"], slate, 3) == ["a", "b"]

    def test_over_cap_keeps_top_theta(self):
        slate = [_bid("a", 0.2), _bid("b", 0.9), _bid("c", 0.5)]
        assert TopThetaBudgeter().cap(["a", "b", "c"], slate, 2) == ["b", "c"]

    def test_missing_theta_defaults_to_zero(self):
        slate = [_bid("b", 0.4)]
        assert TopThetaBudgeter().cap(["a", "b"], slate, 1) == ["b"]


# Four wins -> Beta(5, 1), intro 4, median +0.05, confident.
_RATIOS_EVENTS = _events([0.05] * 4)
# Three wins + two losses -> intro 5, median +0.2, not confident (stays silent).
_QUIET_EVENTS = _events([0.2] * 3 + [-0.5] * 2)

_RENDER_GOLDENS = [
    (None, ""),
    (
        MemoryCard(
            id="m-ratios",
            description="Use pairwise ratios",
            gain_events=_RATIOS_EVENTS,
        ),
        "Use pairwise ratios\n"
        "efficacy: introduced in 4 children; median improvement +0.0500 (confident)",
    ),
    (
        MemoryCard(
            id="m-same",
            description="Same text",
        ),
        "Same text",
    ),
    (
        MemoryCard(
            id="m-quiet",
            description="Not confident stays silent",
            gain_events=_QUIET_EVENTS,
        ),
        "Not confident stays silent",
    ),
    (
        ProgramCard(
            id="program-1",
            program_id="1",
            description="Exemplar",
            fitness=0.8123,
        ),
        "Exemplar\nefficacy: exemplar fitness 0.8123",
    ),
]


class TestEfficacyCardRenderer:
    @pytest.mark.parametrize("card, expected", _RENDER_GOLDENS)
    def test_render_golden(self, card, expected):
        block = BetaBinomialReputation().card_stats(card) if card is not None else None
        assert EfficacyCardRenderer().render(card, block) == expected


class TestCardPosterior:
    @pytest.mark.parametrize(
        "card, expected",
        [
            (MemoryCard(id="m-cold"), (1.0, 1.0)),
            (
                MemoryCard(
                    id="m-events",
                    gain_events=_events([0.01] * 3 + [-0.5]),
                ),
                (4.0, 2.0),
            ),
            (
                ProgramCard(
                    id="program-7",
                    program_id="7",
                    gain_events=_events([0.01] * 6 + [-0.5] * 2),
                ),
                (7.0, 3.0),
            ),
            (ProgramCard(id="program-cold", program_id="c"), (1.0, 1.0)),
        ],
    )
    def test_card_posterior_golden(self, card, expected):
        assert BetaBinomialReputation().card_posterior(card) == expected

    def test_cold_prior_configurable(self):
        assert BetaBinomialReputation(cold_prior=(2.0, 5.0)).card_posterior(
            MemoryCard(id="m-cold")
        ) == (2.0, 5.0)


def _mk_backend() -> _StubBackend:
    return _StubBackend(
        raw_memory=_decision([f"idea-{i}" for i in range(5)]),
        cards={
            f"idea-{i}": _card(
                f"Card {i}", _PROVEN_EVENTS if i % 2 == 0 else _SUSPECT_EVENTS
            )
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
        assert got.cards == [
            f"Card 2\n{_PROVEN_EFFICACY}",
            f"Card 4\n{_PROVEN_EFFICACY}",
        ]
        assert len(got.slate) == 5
        by_id = {bid.card_id: bid for bid in got.slate}
        assert [by_id[f"idea-{i}"].selected for i in range(5)] == [
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
        # idea-blank wins the auction (one positive gain -> Beta(2,1)) yet renders
        # empty: blank description + non-confident block emits no efficacy line.
        backend = _StubBackend(
            raw_memory=_decision(["idea-blank", "idea-good"]),
            cards={
                "idea-blank": _card("", _events([0.01])),
                "idea-good": _card("Good card", _PROVEN_EVENTS),
            },
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-good"]
        assert got.cards == [f"Good card\n{_PROVEN_EFFICACY}"]

    @pytest.mark.asyncio
    async def test_unfetchable_cards_skipped(self):
        backend = _StubBackend(
            raw_memory=_decision(["idea-A", "idea-missing"]),
            cards={"idea-A": _card("Only card", _PROVEN_EVENTS)},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-A"]
        assert len(got.slate) == 1

    @pytest.mark.asyncio
    async def test_canonical_events_record_read_decision_and_child_events(
        self, tmp_path
    ):
        path = tmp_path / "memory_events.jsonl"
        backend = _mk_backend()

        got = await make_read_pipeline(backend, seed=_SEED, event_path=path).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )

        rows = _event_rows(path)
        request = [row for row in rows if row["event_type"] == "read.request"][-1]
        retrieval = [row for row in rows if row["event_type"] == "read.retrieval"][-1]
        read = [row for row in rows if row["event_type"] == "read.selection"][-1]
        auction = [row for row in rows if row["event_type"] == "auction.run"][-1]
        budget = [row for row in rows if row["event_type"] == "budget.cap"][-1]
        assert read["schema_version"] == "memory_event.v1"
        assert read["decision_id"]
        assert request["decision_id"] == read["decision_id"]
        assert retrieval["decision_id"] == read["decision_id"]
        assert auction["decision_id"] == read["decision_id"]
        assert budget["decision_id"] == read["decision_id"]
        assert retrieval["payload"]["duration_ms"] >= 0
        assert read["payload"]["candidate_count"] == 5
        assert read["payload"]["selected_ids"] == got.card_ids
        assert read["payload"]["empty_reason"] == ""
        assert len(read["payload"]["slate"]) == 5
        assert read["payload"]["timing_ms"]["total"] >= 0
        assert auction["payload"]["winner_count"] == 3
        assert budget["payload"]["dropped_ids"] == ["idea-0"]

    @pytest.mark.asyncio
    async def test_canonical_event_records_auction_empty_reason(self, tmp_path):
        path = tmp_path / "memory_events.jsonl"
        backend = _StubBackend(
            raw_memory=_decision(["bad"]),
            cards={"bad": _card("Bad card", _SUSPECT_EVENTS)},
        )

        got = await make_read_pipeline(backend, seed=_SEED, event_path=path).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )

        assert got.cards == []
        assert got.card_ids == []
        assert len(got.slate) == 1
        read = [
            row for row in _event_rows(path) if row["event_type"] == "read.selection"
        ][-1]
        assert read["payload"]["candidate_ids"] == ["bad"]
        assert read["payload"]["auction_winner_ids"] == []
        assert read["payload"]["empty_reason"] == "auction_rejected"


def _ev_card(description: str, magnitude: float) -> MemoryCard:
    # Twelve equal wins -> Beta(13, 1) (gate passes near-deterministically) with
    # IntroGain_best_median == magnitude, the one field the EV auction bids on.
    return MemoryCard(
        id=f"card-{description}",
        description=description,
        gain_events=_events([magnitude] * 12),
    )


def _ev_pipeline(backend, *, seed, event_path=None) -> MemoryReadPipeline:
    return MemoryReadPipeline(
        retriever=GamRetriever(backend),
        selector=LLMCardSelector(),
        auctioneer=EVThompsonAuctioneer(prior_magnitude=0.1),
        budgeter=TopBidBudgeter(),
        renderer=EfficacyCardRenderer(),
        reputation=BetaBinomialReputation(),
        rng=np.random.default_rng(seed),
        event_path=event_path,
    )


class TestShortlistKThreading:
    @pytest.mark.asyncio
    async def test_shortlist_k_widens_selector_ask_but_budget_caps(self):
        # The selector LLM is asked for up to shortlist_k, while the injection
        # budget (max_cards) still caps what reaches the mutator.
        backend = _StubBackend(
            raw_memory=_decision([f"idea-{i}" for i in range(5)]),
            cards={f"idea-{i}": _card(f"Card {i}", _PROVEN_EVENTS) for i in range(5)},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=1,
            shortlist_k=10,
        )
        ask = backend.research_calls[0]["planning_request"]
        assert "pick up to 10 card(s)" in ask
        assert len(got.card_ids) <= 1

    @pytest.mark.asyncio
    async def test_shortlist_k_default_one_keeps_legacy_ask(self):
        backend = _mk_backend()
        await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert "pick up to 1 card(s)" in backend.research_calls[0]["planning_request"]


class TestEVReadPath:
    @pytest.mark.asyncio
    async def test_magnitude_populated_and_winner_ranked_by_bid(self, tmp_path):
        # Two near-certain-safe cards (gate passes) with distinct magnitudes; the
        # budget=1 cap must keep the higher-EV one, and the auction event must
        # carry the per-card magnitude (not None).
        path = tmp_path / "memory_events.jsonl"
        backend = _StubBackend(
            raw_memory=_decision(["idea-hi", "idea-lo"]),
            cards={
                "idea-hi": _ev_card("Hi EV", 0.5),
                "idea-lo": _ev_card("Lo EV", 0.02),
            },
        )
        got = await _ev_pipeline(backend, seed=_SEED, event_path=path).select(
            parents=[_make_program()],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=1,
            shortlist_k=10,
        )
        assert got.card_ids == ["idea-hi"]
        auction = [
            row for row in _event_rows(path) if row["event_type"] == "auction.run"
        ][-1]
        mags = {b["card_id"]: b["magnitude"] for b in auction["payload"]["bids"]}
        assert mags == {"idea-hi": 0.5, "idea-lo": 0.02}
