"""MemoryReader end-to-end: stage wiring, fail-to-empty, the empty_reason
taxonomy, and seed-exact reproducibility of the full select pass."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.events import MemoryReadSelection
from gigaevo.memory.read.auction import (
    AuctionBid,
    AuctionCandidate,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)
from gigaevo.memory.read.reader import MemoryReader, MemorySelection
from gigaevo.memory.read.render import EfficacyCardRenderer
from gigaevo.memory.read.reputation import BetaBinomialReputation
from gigaevo.memory.storage.base import ResearchResult


class _Parent:
    def __init__(self, pid: str = "prog-1", metrics: dict | None = None) -> None:
        self.id = pid
        self.metrics = metrics or {"fitness": 0.5}


class _Shortlister:
    def __init__(self, result: ResearchResult | None = None) -> None:
        self.result = result or ResearchResult()
        self.calls: list[dict] = []

    async def shortlist(self, **kwargs) -> ResearchResult:
        self.calls.append(kwargs)
        return self.result


class _ExplodingShortlister:
    async def shortlist(self, **kwargs) -> ResearchResult:
        raise RuntimeError("shortlist blew up")


class _WinAllAuctioneer:
    def run(self, candidates, rng):
        slate = [
            AuctionBid(
                card_id=c.card_id,
                posterior_a=c.posterior_a,
                posterior_b=c.posterior_b,
                theta=1.0 - 0.01 * i,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.0,
                selected=True,
            )
            for i, c in enumerate(candidates)
        ]
        return [c.card_id for c in candidates], slate


class _RejectAllAuctioneer:
    def run(self, candidates, rng):
        slate = [
            AuctionBid(
                card_id=c.card_id,
                posterior_a=c.posterior_a,
                posterior_b=c.posterior_b,
                theta=0.0,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=1.0,
                selected=False,
            )
            for c in candidates
        ]
        return [], slate


class _EmptyBudgeter:
    def cap(self, card_ids, slate, max_cards):
        return []


class _BlankRenderer:
    def render(self, card, block=None):
        return ""


def _reader(
    *,
    shortlister=None,
    auctioneer=None,
    budgeter=None,
    renderer=None,
    max_cards: int = 1,
    rng=None,
) -> MemoryReader:
    return MemoryReader(
        shortlister=shortlister if shortlister is not None else _Shortlister(),
        reputation=BetaBinomialReputation(),
        auctioneer=auctioneer if auctioneer is not None else _WinAllAuctioneer(),
        budgeter=budgeter if budgeter is not None else TopThetaBudgeter(),
        renderer=renderer if renderer is not None else EfficacyCardRenderer(),
        max_cards=max_cards,
        rng=rng if rng is not None else np.random.default_rng(0),
    )


async def _select(reader: MemoryReader, **overrides) -> MemorySelection:
    params = {
        "parents": [_Parent()],
        "mutation_mode": "rewrite",
        "task_description": "task",
        "metrics_description": "metrics",
    }
    params.update(overrides)
    return await reader.select(**params)


def _selection_events(events) -> list[MemoryReadSelection]:
    return [e for e in events if isinstance(e, MemoryReadSelection)]


class TestSelect:
    async def test_happy_path_renders_budgeted_winners(
        self, make_card, captured_events
    ):
        cards = (make_card(description="alpha"), make_card(description="beta"))
        shortlister = _Shortlister(ResearchResult(cards=cards, iterations=2))
        reader = _reader(shortlister=shortlister, max_cards=1)
        selection = await _select(reader)
        assert selection.card_ids == (cards[0].id,)
        assert selection.cards == ("alpha",)
        assert len(selection.slate) == 2
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == ""
        assert event.candidate_ids == (cards[0].id, cards[1].id)
        assert event.auction_winner_ids == (cards[0].id, cards[1].id)
        assert event.selected_ids == (cards[0].id,)
        assert event.research_iterations == 2
        assert set(event.timing_ms) == {
            "research",
            "reputation",
            "auction",
            "budget",
            "render",
            "total",
        }

    async def test_reputation_feeds_auction_candidates(self, make_card, make_event):
        card = make_card(gain_events=(make_event(0.2), make_event(0.4)))
        seen: list[AuctionCandidate] = []

        class _Spy:
            def run(self, candidates, rng):
                seen.extend(candidates)
                return [], []

        reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(card,))),
            auctioneer=_Spy(),
        )
        await _select(reader)
        (candidate,) = seen
        assert (candidate.posterior_a, candidate.posterior_b) == (3.0, 1.0)
        assert candidate.magnitude == pytest.approx(0.3)

    async def test_exclude_ids_thread_to_shortlister(self, make_card):
        shortlister = _Shortlister()
        reader = _reader(shortlister=shortlister)
        await _select(reader, exclude_ids=frozenset({"m-used"}))
        (call,) = shortlister.calls
        assert call["exclude_ids"] == frozenset({"m-used"})
        assert call["mutation_mode"] == "rewrite"

    async def test_nonpositive_budget_short_circuits(self, captured_events):
        shortlister = _Shortlister()
        reader = _reader(shortlister=shortlister, max_cards=0)
        selection = await _select(reader)
        assert selection == MemorySelection()
        assert shortlister.calls == []
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "max_cards_nonpositive"

    async def test_empty_research(self, captured_events):
        selection = await _select(_reader())
        assert selection == MemorySelection()
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "research_empty"

    async def test_auction_rejects_all(self, make_card, captured_events):
        reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(make_card(),))),
            auctioneer=_RejectAllAuctioneer(),
        )
        selection = await _select(reader)
        assert selection.card_ids == ()
        assert len(selection.slate) == 1
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "auction_rejected"

    async def test_budget_empties(self, make_card, captured_events):
        reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(make_card(),))),
            budgeter=_EmptyBudgeter(),
        )
        selection = await _select(reader)
        assert selection == MemorySelection(slate=selection.slate)
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "budget_empty"

    async def test_render_empties(self, make_card, captured_events):
        card = make_card()
        reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(card,))),
            renderer=_BlankRenderer(),
        )
        selection = await _select(reader)
        assert selection.card_ids == ()
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "render_empty"
        assert event.render_dropped_ids == (card.id,)
        assert event.budgeted_ids == (card.id,)

    async def test_component_failure_degrades_to_empty(self, captured_events):
        reader = _reader(shortlister=_ExplodingShortlister())
        selection = await _select(reader)
        assert selection == MemorySelection()
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "exception"
        assert "shortlist blew up" in event.error

    async def test_no_parents_is_safe(self, make_card, captured_events):
        reader = _reader(shortlister=_Shortlister(ResearchResult(cards=(make_card(),))))
        selection = await _select(reader, parents=[])
        assert len(selection.card_ids) == 1
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == ""


class TestSeedExactReproducibility:
    async def test_same_seed_same_selection(self, make_card, make_event):
        def _cards() -> tuple[Card, ...]:
            counter = iter(range(100))

            def _mk(**kw):
                n = next(counter)
                return Card(
                    id=f"mem-seed{n:03d}",
                    description=f"idea {n}",
                    **kw,
                )

            return tuple(
                _mk(gain_events=(make_event(0.1 * (i + 1)), make_event(-0.05)))
                for i in range(4)
            )

        def _build(seed: int) -> MemoryReader:
            return _reader(
                shortlister=_Shortlister(ResearchResult(cards=_cards())),
                auctioneer=ThompsonAuctioneer(),
                max_cards=2,
                rng=np.random.default_rng(seed),
            )

        first = await _select(_build(11))
        second = await _select(_build(11))
        third = await _select(_build(12))
        assert first == second
        assert [b.theta for b in first.slate] != [b.theta for b in third.slate]
