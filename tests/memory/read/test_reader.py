"""MemoryReader end-to-end: stage wiring, fail-to-empty, the empty_reason
taxonomy, and seed-exact reproducibility of the full select pass."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.context.beta import BetaPrior
from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.events import MemoryReadSelection
from gigaevo.memory.read.auction import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)
from gigaevo.memory.read.projection import AuctionCandidateProjector
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
    def run(self, candidates, rng, *, baseline=None):
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
    def run(self, candidates, rng, *, baseline=None):
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
            def run(self, candidates, rng, *, baseline=None):
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

    async def test_default_pending_counts_is_byte_identical_zero(self, make_card):
        card = make_card(description="pending seam")
        omitted_seen: list[AuctionCandidate] = []
        explicit_seen: list[AuctionCandidate] = []

        class _Spy:
            def __init__(self, seen):
                self._seen = seen

            def run(self, candidates, rng, *, baseline=None):
                self._seen.extend(candidates)
                return _WinAllAuctioneer().run(candidates, rng, baseline=baseline)

        omitted_reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(card,))),
            auctioneer=_Spy(omitted_seen),
        )
        explicit_reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(card,))),
            auctioneer=_Spy(explicit_seen),
        )

        omitted = await _select(omitted_reader)
        explicit_none = await _select(explicit_reader, pending_counts=None)

        assert [candidate.pending_count for candidate in omitted_seen] == [0]
        assert [candidate.pending_count for candidate in explicit_seen] == [0]
        assert explicit_none.model_dump_json() == omitted.model_dump_json()

    async def test_card_stats_resolved_once_per_candidate(self, make_card, make_event):
        # One block_from_events pass per candidate per select: the auction's
        # posterior/magnitude and the winner's render all read the same
        # resolved block (BD proximity deep-copies the behavior space per
        # card_stats call, so re-resolving is real per-read cost, and a
        # render-time re-resolve could even disagree with what was bid on).
        calls: list[str] = []

        class _Counting(BetaBinomialReputation):
            def card_stats(self, card, context=None):
                calls.append(card.id)
                return super().card_stats(card, context)

        cards = tuple(make_card(gain_events=(make_event(0.2),)) for _ in range(3))
        reader = MemoryReader(
            shortlister=_Shortlister(ResearchResult(cards=cards)),
            reputation=_Counting(),
            auctioneer=_WinAllAuctioneer(),
            budgeter=TopThetaBudgeter(),
            renderer=EfficacyCardRenderer(),
            max_cards=3,
            rng=np.random.default_rng(0),
        )
        await _select(reader)
        assert sorted(calls) == sorted(c.id for c in cards)

    async def test_no_card_baseline_resolved_once_per_decision(
        self, make_card, make_event
    ):
        # The decision-level no-card summary reaches the auction as one keyword
        # argument, resolved once per select — never once per candidate (the
        # JSON-backed store takes a file lock per summary_for call).
        class _CountingNoCard:
            def __init__(self) -> None:
                self.calls = 0

            def summary_for(self, context):
                self.calls += 1
                return NoCardGateSummary(
                    prior=BetaPrior(alpha=4.0, beta=2.0), source="dynamic"
                )

        received: list = []

        class _Spy:
            def run(self, candidates, rng, *, baseline=None):
                received.append(baseline)
                return [], []

        evidence = _CountingNoCard()
        cards = tuple(make_card(gain_events=(make_event(0.2),)) for _ in range(3))
        reader = MemoryReader(
            shortlister=_Shortlister(ResearchResult(cards=cards)),
            reputation=BetaBinomialReputation(),
            auctioneer=_Spy(),
            budgeter=TopThetaBudgeter(),
            renderer=EfficacyCardRenderer(),
            candidate_projector=AuctionCandidateProjector(no_card_evidence=evidence),
            max_cards=3,
            rng=np.random.default_rng(0),
        )
        await _select(reader)
        assert evidence.calls == 1
        (baseline,) = received
        assert (baseline.prior.alpha, baseline.prior.beta) == (4.0, 2.0)
        assert baseline.source == "dynamic"

    async def test_founding_only_card_borrows_warm_scale_end_to_end(
        self, make_card, make_event
    ):
        # A founding-only card resolves to magnitude None through the
        # reputation, and the EV auction prices it on the warm pool's positive
        # gain scale — birth evidence never prices the bid directly, in either
        # direction.
        warm = make_card(gain_events=(make_event(0.4), make_event(0.4)))
        newborn = make_card(gain_events=(make_event(0.9, founding=True),))
        reader = _reader(
            shortlister=_Shortlister(ResearchResult(cards=(warm, newborn))),
            auctioneer=EVThompsonAuctioneer(),
            max_cards=2,
        )
        selection = await _select(reader)
        newborn_bid = next(b for b in selection.slate if b.card_id == newborn.id)
        assert newborn_bid.magnitude == pytest.approx(0.4)

    async def test_exclude_ids_thread_to_shortlister(self, make_card):
        shortlister = _Shortlister()
        reader = _reader(shortlister=shortlister)
        await _select(reader, exclude_ids=frozenset({"m-used"}))
        (call,) = shortlister.calls
        assert call["exclude_ids"] == frozenset({"m-used"})
        assert call["mutation_mode"] == "rewrite"

    async def test_exclude_ids_filtered_after_shortlisting(
        self, make_card, captured_events
    ):
        excluded = make_card(description="excluded")
        allowed = make_card(description="allowed")
        shortlister = _Shortlister(ResearchResult(cards=(excluded, allowed)))
        reader = _reader(shortlister=shortlister, max_cards=2)

        selection = await _select(reader, exclude_ids=frozenset({excluded.id}))

        assert selection.card_ids == (allowed.id,)
        assert selection.cards == ("allowed",)
        (event,) = _selection_events(captured_events)
        assert event.candidate_ids == (allowed.id,)

    async def test_exclude_ids_filter_absorbed_alias_after_shortlisting(
        self, make_card
    ):
        survivor = make_card(
            id="mem-new",
            absorbed_ids=("mem-old",),
            description="survivor",
        )
        allowed = make_card(description="allowed")
        shortlister = _Shortlister(ResearchResult(cards=(survivor, allowed)))
        reader = _reader(shortlister=shortlister, max_cards=2)

        selection = await _select(reader, exclude_ids=frozenset({"mem-old"}))

        assert selection.card_ids == (allowed.id,)
        assert selection.cards == ("allowed",)

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
