"""Golden + unit tests for the modular read pipeline.

``MemoryReadPipeline`` composed of GamRetriever / LLMCardSelector /
ThompsonAuctioneer / TopThetaBudgeter / EfficacyCardRenderer.
End-to-end goldens were frozen from the legacy ``MemorySelectorAgent.select()``
(seed-exact) before its deletion.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gigaevo.memory.core import (
    AuctionBid,
    BetaBinomialReputation,
    EfficacyCardRenderer,
    GamRetriever,
    LLMCardSelector,
    MemorySelection,
    TopThetaBudgeter,
)
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard
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


def _card(description: str, posterior: tuple[float, float] | None = None) -> MemoryCard:
    stats = (
        {"ALL": {"posterior_a": posterior[0], "posterior_b": posterior[1]}}
        if posterior is not None
        else {}
    )
    return MemoryCard(
        id=f"card-{description or 'blank'}",
        description=description,
        evolution_statistics=stats,
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
            "evolution_statistics": {"ALL": {"posterior_a": 3.0, "posterior_b": 1.0}},
        }
        retriever = GamRetriever(_StubBackend(raw_memory=None, cards={"a": raw}))
        card = retriever.get_card("a")
        assert isinstance(card, MemoryCard)
        assert card.id == "a"
        assert card.description == "legacy idea"
        assert card.evolution_statistics.ALL.posterior_a == 3.0

    def test_get_card_corrupt_legacy_dict_fails_to_none(self):
        raw = {
            "description": "corrupt legacy",
            "evolution_statistics": {"ALL": {"posterior_a": "corrupt"}},
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


_RENDER_GOLDENS = [
    (None, ""),
    (
        MemoryCard(
            id="m-ratios",
            description="Use pairwise ratios",
            explanation={"summary": "exposes scale-free structure"},
            evolution_statistics={
                "ALL": {
                    "intro_events": 4,
                    "IntroGain_best_median": 0.05,
                    "DownsideRate_best": 0.1,
                    "efficacy_confident": True,
                }
            },
        ),
        "Use pairwise ratios\n"
        "mechanism: exposes scale-free structure\n"
        "efficacy: introduced in 4 children; median improvement +0.0500; "
        "downside 10% (confident)",
    ),
    (
        MemoryCard(
            id="m-same",
            description="Same text",
            explanation={"summary": "Same text"},
        ),
        "Same text",
    ),
    (
        MemoryCard(
            id="m-adj",
            description="Adjusted median wins",
            evolution_statistics={
                "ALL": {
                    "intro_events": 2,
                    "IntroGain_best_median": 0.5,
                    "IntroGain_best_adj_median": -0.01,
                    "efficacy_confident": True,
                }
            },
        ),
        "Adjusted median wins\n"
        "efficacy: introduced in 2 children; median improvement vs cohort "
        "-0.0100 (caution: non-positive median)",
    ),
    (
        MemoryCard(
            id="m-quiet",
            description="Not confident stays silent",
            evolution_statistics={
                "ALL": {
                    "intro_events": 5,
                    "IntroGain_best_median": 0.2,
                    "efficacy_confident": False,
                }
            },
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
    (
        MemoryCard(
            id="m-mech",
            description="Mechanism only",
            explanation={"summary": "attr mechanism"},
        ),
        "Mechanism only\nmechanism: attr mechanism",
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
            (MemoryCard(id="m-cold"), (1.0, 1.0)),
            (
                MemoryCard(
                    id="m-stamped",
                    evolution_statistics={
                        "ALL": {"posterior_a": 4.0, "posterior_b": 2.0}
                    },
                ),
                (4.0, 2.0),
            ),
            (
                MemoryCard(
                    id="m-half",
                    evolution_statistics={"ALL": {"posterior_a": 4.0}},
                ),
                (1.0, 1.0),
            ),
            (
                ProgramCard(
                    id="program-7",
                    program_id="7",
                    evolution_statistics={
                        "ALL": {"posterior_a": 7.0, "posterior_b": 3.0}
                    },
                ),
                (7.0, 3.0),
            ),
            (ProgramCard(id="program-cold", program_id="c"), (1.0, 1.0)),
        ],
    )
    def test_card_posterior_golden(self, card, expected):
        assert BetaBinomialReputation().card_posterior(card) == expected

    @pytest.mark.parametrize(
        "posterior",
        [(0.0, 1.0), (-1.0, 2.0), (1.0, 0.0), (float("inf"), 1.0), (float("nan"), 1.0)],
    )
    def test_malformed_stamped_posterior_degrades_to_cold(self, posterior):
        # Beta(a, b) requires a > 0 and b > 0; a corrupt stamped block must
        # degrade to the cold prior, not poison the auction's rng.beta draw.
        card = MemoryCard(
            id="m-bad",
            evolution_statistics={
                "ALL": {"posterior_a": posterior[0], "posterior_b": posterior[1]}
            },
        )
        assert BetaBinomialReputation().card_posterior(card) == (1.0, 1.0)

    def test_cold_prior_configurable(self):
        assert BetaBinomialReputation(cold_prior=(2.0, 5.0)).card_posterior(
            MemoryCard(id="m-cold")
        ) == (2.0, 5.0)


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
    async def test_empty_when_persisted_posterior_corrupt(self):
        # A corrupt persisted posterior fails typed validation at the backend
        # boundary; the read path must degrade to an empty selection, never an
        # exception that sinks the mutation.
        class _CorruptCardBackend(_StubBackend):
            def get_card(self, card_id):
                return normalize_memory_card(
                    {
                        "id": card_id,
                        "description": "Bad card",
                        "evolution_statistics": {
                            "ALL": {"posterior_a": "corrupt", "posterior_b": 1.0}
                        },
                    }
                )

        backend = _CorruptCardBackend(raw_memory=_decision(["idea-bad"]))
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got == MemorySelection(cards=[], card_ids=[])

    @pytest.mark.asyncio
    async def test_nonfinite_posterior_competes_as_cold(self):
        # A corrupt stamped posterior degrades to the cold prior — the card
        # competes as COLD (same contract as a half-stamped block) instead of
        # NaN silently propagating through the auction draw.
        bad = _card("NaN card", (float("nan"), 1.0))
        backend = _StubBackend(
            raw_memory=_decision(["idea-nan"]),
            cards={"idea-nan": bad},
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert got.card_ids == ["idea-nan"]

    @pytest.mark.asyncio
    async def test_nonpositive_posterior_does_not_blank_healthy_cards(self):
        # One poisoned stamped posterior must not raise inside the auction and
        # fail-to-empty the WHOLE selection — healthy candidates still compete.
        backend = _StubBackend(
            raw_memory=_decision(["idea-bad", "idea-good"]),
            cards={
                "idea-bad": _card("Poisoned card", (0.0, 1.0)),
                "idea-good": _card("Good card", _PROVEN),
            },
        )
        got = await make_read_pipeline(backend, seed=_SEED).select(
            parents=[_make_program()], **_SELECT_KWARGS
        )
        assert "idea-good" in got.card_ids

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
            cards={"bad": _card("Bad card", (1.0, 1_000_000.0))},
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
