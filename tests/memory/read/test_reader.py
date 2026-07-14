"""MemoryReader end-to-end: stage wiring, fail-to-empty, the empty_reason
taxonomy, and seed-exact reproducibility of the full select pass."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.memory.cards import Card, DecisionContext
from gigaevo.memory.context import ContextKey
from gigaevo.memory.context.beta import BetaPrior
from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.events import MemoryAssignment, MemoryReadSelection
from gigaevo.memory.read.auction import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)
from gigaevo.memory.read.probe import ColdProbePolicy
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.reader import MemoryReader, MemorySelection
from gigaevo.memory.read.render import EfficacyCardRenderer
from gigaevo.memory.read.reputation import BetaBinomialReputation
from gigaevo.memory.storage.base import ResearchResult


class _Parent:
    def __init__(self, pid: str = "prog-1", metrics: dict | None = None) -> None:
        self.id = pid
        self.metrics = metrics or {"fitness": 0.5}
        self.iteration = 7


class _CountingRng:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


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


class _ColdRejectAllAuctioneer:
    def run(self, candidates, rng, *, baseline=None):
        del rng, baseline
        slate = [
            AuctionBid(
                card_id=c.card_id,
                posterior_a=c.posterior_a,
                posterior_b=c.posterior_b,
                theta=0.2,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.8,
                selected=False,
                bid=0.1,
                support_kind="cold_prior",
                support_n=0.0,
            )
            for c in candidates
        ]
        return [], slate


class _DRBaselineAuctioneer:
    def run(self, candidates, rng, *, baseline=None):
        del rng, baseline
        warm, bootstrap, cold, irrelevant = candidates
        return [warm.card_id, bootstrap.card_id], [
            AuctionBid(
                card_id=warm.card_id,
                posterior_a=3.0,
                posterior_b=1.0,
                theta=0.9,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.2,
                selected=True,
                magnitude=2.0,
                bid=0.01,
                pending_discount=0.1,
            ),
            AuctionBid(
                card_id=bootstrap.card_id,
                posterior_a=4.0,
                posterior_b=1.0,
                theta=0.85,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.2,
                selected=True,
                magnitude=0.4,
                bid=0.004,
                support_kind="ev_rewards",
                support_n=4.0,
                pending_discount=0.1,
            ),
            AuctionBid(
                card_id=cold.card_id,
                posterior_a=1.0,
                posterior_b=3.0,
                theta=0.8,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.9,
                selected=False,
                magnitude=None,
                bid=0.8,
                support_kind="cold_prior",
                support_n=0.0,
                no_card_baseline=-0.2,
            ),
            AuctionBid(
                card_id=irrelevant.card_id,
                posterior_a=9.0,
                posterior_b=1.0,
                theta=0.9,
                baseline_a=3.0,
                baseline_b=3.0,
                baseline_theta=0.8,
                selected=False,
                magnitude=10.0,
                bid=9.0,
            ),
        ]


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
    probe_policy=None,
    max_cards: int = 1,
    rng=None,
) -> MemoryReader:
    return MemoryReader(
        shortlister=shortlister if shortlister is not None else _Shortlister(),
        reputation=BetaBinomialReputation(),
        auctioneer=auctioneer if auctioneer is not None else _WinAllAuctioneer(),
        budgeter=budgeter if budgeter is not None else TopThetaBudgeter(),
        renderer=renderer if renderer is not None else EfficacyCardRenderer(),
        probe_policy=probe_policy,
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


def _assignment_events(events) -> list[MemoryAssignment]:
    return [e for e in events if isinstance(e, MemoryAssignment)]


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
        (assignment_event,) = _assignment_events(captured_events)
        assert assignment_event.decision_id == event.decision_id
        assert assignment_event.assignment.decision_id == selection.decision_id
        assert assignment_event.assignment.assigned_ids == (cards[0].id,)
        assert assignment_event.assignment.context.search_phase == "iteration:7"
        assert set(event.timing_ms) == {
            "research",
            "reputation",
            "auction",
            "budget",
            "render",
            "total",
        }

    async def test_assignment_records_assigned_card_pending_state(
        self, make_card, captured_events
    ):
        card = make_card(description="pending assignment")

        class _AuditedWin:
            def run(self, candidates, rng, *, baseline=None):
                del rng, baseline
                candidate = candidates[0]
                return [candidate.card_id], [
                    AuctionBid(
                        card_id=candidate.card_id,
                        posterior_a=candidate.posterior_a,
                        posterior_b=candidate.posterior_b,
                        theta=1.0,
                        baseline_a=3.0,
                        baseline_b=3.0,
                        baseline_theta=0.0,
                        selected=True,
                        pending_count=candidate.pending_count,
                        pending_effective_support=2.5,
                        pending_discount=0.8,
                    )
                ]

        selection = await _select(
            _reader(
                shortlister=_Shortlister(ResearchResult(cards=(card,))),
                auctioneer=_AuditedWin(),
            ),
            pending_counts={card.id: 3},
        )

        assert selection.assignment is not None
        assert selection.assignment.propensity_kind == "observational"
        assert selection.assignment.pending_by_card == {card.id: 3}
        assert selection.assignment.pending_discount_by_card == {card.id: 0.8}
        (assignment_event,) = _assignment_events(captured_events)
        assert assignment_event.assignment.pending_by_card == {card.id: 3}
        (selection_event,) = _selection_events(captured_events)
        assert selection_event.slate[0]["pending_count"] == 3
        assert selection_event.slate[0]["pending_discount"] == 0.8

    async def test_assignment_records_dr_baselines_from_assigned_and_offered_slate(
        self, make_card
    ):
        warm = make_card(description="warm")
        bootstrap = make_card(description="bootstrap")
        cold = make_card(description="cold")
        irrelevant = make_card(description="irrelevant")
        selection = await _select(
            _reader(
                shortlister=_Shortlister(
                    ResearchResult(cards=(warm, bootstrap, cold, irrelevant))
                ),
                auctioneer=_DRBaselineAuctioneer(),
                probe_policy=ColdProbePolicy(warm_override_probe_rate=0.0),
                max_cards=2,
                rng=np.random.default_rng(0),
            )
        )

        assert selection.card_ids == (warm.id, bootstrap.id)
        assert selection.assignment is not None
        assert selection.assignment.propensities == {cold.id: 0.0}
        assert selection.assignment.predicted_help == {
            warm.id: 0.75,
            bootstrap.id: 0.8,
            cold.id: 0.25,
        }
        assert selection.assignment.predicted_gain == {
            warm.id: 1.5,
            bootstrap.id: 0.4,
        }
        assert selection.assignment.predicted_no_card_gain == {cold.id: -0.2}
        assert irrelevant.id not in selection.assignment.predicted_help

    async def test_assignment_records_withheld_empty_probe_as_control(
        self, make_card, captured_events
    ):
        card = make_card(description="withheld cold probe")
        rate = 0.5
        selection = await _select(
            _reader(
                shortlister=_Shortlister(ResearchResult(cards=(card,))),
                auctioneer=_ColdRejectAllAuctioneer(),
                probe_policy=ColdProbePolicy(empty_selection_probe_rate=rate),
                rng=np.random.default_rng(0),
            )
        )

        assert selection.card_ids == ()
        assert selection.assignment is not None
        assert selection.assignment.probe_arm == "control"
        assert selection.assignment.arm == "none"
        assert selection.assignment.randomized is True
        assert selection.assignment.propensity_kind == "probe_bernoulli"
        assert selection.assignment.propensities == {card.id: rate}
        (offered,) = selection.slate
        assert offered.probe_offered is True
        assert offered.probe_propensity == rate
        assert offered.probe_selected is False
        assert card.id not in selection.assignment.assigned_ids
        (assignment_event,) = _assignment_events(captured_events)
        assert assignment_event.assignment == selection.assignment

    async def test_assignment_records_fired_empty_probe_as_treated(
        self, make_card, captured_events
    ):
        card = make_card(description="fired cold probe")
        rate = 0.5
        selection = await _select(
            _reader(
                shortlister=_Shortlister(ResearchResult(cards=(card,))),
                auctioneer=_ColdRejectAllAuctioneer(),
                probe_policy=ColdProbePolicy(empty_selection_probe_rate=rate),
                rng=np.random.default_rng(2),
            )
        )

        assert selection.assignment is not None
        assert selection.assignment.probe_arm == "treated"
        assert selection.assignment.randomized is True
        assert selection.assignment.propensity_kind == "probe_bernoulli"
        assert selection.assignment.propensities == {card.id: rate}
        assert selection.assignment.assigned_ids == (card.id,)
        (offered,) = selection.slate
        assert offered.probe_offered is True
        assert offered.probe_propensity == rate
        assert offered.probe_selected is True
        (assignment_event,) = _assignment_events(captured_events)
        assert assignment_event.assignment == selection.assignment

    async def test_assignment_records_no_probe_offer_as_observational(self, make_card):
        card = make_card(description="ineligible for probing")
        selection = await _select(
            _reader(
                shortlister=_Shortlister(ResearchResult(cards=(card,))),
                auctioneer=_RejectAllAuctioneer(),
                probe_policy=ColdProbePolicy(empty_selection_probe_rate=1.0),
            )
        )

        assert selection.assignment is not None
        assert selection.assignment.probe_arm == "none"
        assert selection.assignment.randomized is False
        assert selection.assignment.propensity_kind == "observational"
        assert selection.assignment.propensities == {}
        (ineligible,) = selection.slate
        assert ineligible.probe_offered is False
        assert ineligible.probe_propensity is None

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
        assert explicit_none.model_dump_json(
            exclude={"decision_id", "assignment"}
        ) == omitted.model_dump_json(exclude={"decision_id", "assignment"})

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
        assert selection.cards == selection.card_ids == selection.slate == ()
        assert selection.assignment is not None
        assert selection.assignment.arm == "none"
        assert shortlister.calls == []
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "max_cards_nonpositive"

    async def test_empty_research(self, captured_events):
        selection = await _select(_reader())
        assert selection.cards == selection.card_ids == selection.slate == ()
        assert selection.assignment is not None
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
        assert selection.cards == selection.card_ids == ()
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
        assert selection.cards == selection.card_ids == selection.slate == ()
        assert selection.assignment is not None
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == "exception"
        assert "shortlist blew up" in event.error

    async def test_no_parents_is_safe(self, make_card, captured_events):
        reader = _reader(shortlister=_Shortlister(ResearchResult(cards=(make_card(),))))
        selection = await _select(reader, parents=[])
        assert len(selection.card_ids) == 1
        (event,) = _selection_events(captured_events)
        assert event.empty_reason == ""

    async def test_assignment_freezes_decision_time_bd_cell(self, make_card):
        class _BDContext:
            task_key = "hover"

            def read_context(self, parents):
                return DecisionContext(
                    task_key=self.task_key,
                    parent_metrics=dict(parents[0].metrics),
                    parent_id=parents[0].id,
                )

            def key_for(self, context=None):
                return ContextKey(kind="bd_cell", parts=("3", "5"))

        card = make_card(description="cell card")
        reader = MemoryReader(
            shortlister=_Shortlister(ResearchResult(cards=(card,))),
            reputation=BetaBinomialReputation(),
            auctioneer=_WinAllAuctioneer(),
            budgeter=TopThetaBudgeter(),
            renderer=EfficacyCardRenderer(),
            context_model=_BDContext(),
            max_cards=1,
            rng=np.random.default_rng(0),
        )

        selection = await _select(reader)

        assert selection.assignment is not None
        assert selection.assignment.bd_cell == (3, 5)


class TestSeedExactReproducibility:
    async def test_dr_recording_preserves_selection_and_rng_position_for_64_seeds(
        self, make_card
    ):
        card = make_card(description="probe selection")
        rate = 0.37
        observed_outcomes: set[bool] = set()

        for seed in range(64):
            expected_fired = float(np.random.default_rng(seed).random()) < rate
            observed_outcomes.add(expected_fired)
            rng = _CountingRng(seed)
            selection = await _select(
                _reader(
                    shortlister=_Shortlister(ResearchResult(cards=(card,))),
                    auctioneer=_ColdRejectAllAuctioneer(),
                    probe_policy=ColdProbePolicy(empty_selection_probe_rate=rate),
                    rng=rng,
                )
            )
            expected = MemorySelection(
                cards=((card.description,) if expected_fired else ()),
                card_ids=((card.id,) if expected_fired else ()),
            )

            assert rng.calls == 1
            assert selection.model_dump_json(include={"cards", "card_ids"}) == (
                expected.model_dump_json(include={"cards", "card_ids"})
            )
            assert selection.assignment is not None
            assert selection.assignment.predicted_help == {card.id: 0.5}

        assert observed_outcomes == {False, True}

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
        assert first.model_dump(exclude={"decision_id", "assignment"}) == (
            second.model_dump(exclude={"decision_id", "assignment"})
        )
        assert [b.theta for b in first.slate] != [b.theta for b in third.slate]
