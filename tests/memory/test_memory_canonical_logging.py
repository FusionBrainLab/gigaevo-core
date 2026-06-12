"""Every decision-making memory component must emit a canonical
``[Memory][Component]`` event at its decision point, so live runs are
debuggable from the log alone (auction draws, budget drops, evictions,
admissions, posterior bridge summary).
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger
import numpy as np

from gigaevo.memory.core.admitter import SignBasedAdmitter
from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    ThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_injection_posterior,
)
from gigaevo.memory.shared_memory.models import MemoryCard


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


@contextmanager
def capture_logs():
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink_id)


class TestAuctioneerLogging:
    def test_run_emits_canonical_event_with_draws(self):
        with capture_logs() as captured:
            winners, records = ThompsonAuctioneer().run(
                [
                    AuctionCandidate(
                        card_id="card-hot", posterior_a=9.0, posterior_b=1.0
                    ),
                    AuctionCandidate(
                        card_id="card-cold", posterior_a=1.0, posterior_b=9.0
                    ),
                ],
                np.random.default_rng(0),
            )
        text = "".join(captured)
        assert "[Memory][Auction]" in text
        assert "card-hot" in text
        assert str(len(winners)) in text

    def test_no_event_for_empty_candidates(self):
        with capture_logs() as captured:
            ThompsonAuctioneer().run([], np.random.default_rng(0))
        assert "[Memory][Auction]" not in "".join(captured)


class TestBudgeterLogging:
    def test_cap_drop_is_logged_with_dropped_ids(self):
        slate = [_bid("a", 0.9), _bid("b", 0.2)]
        with capture_logs() as captured:
            kept = TopThetaBudgeter().cap(["a", "b"], slate, max_cards=1)
        text = "".join(captured)
        assert kept == ["a"]
        assert "[Memory][Budgeter]" in text
        assert "b" in text

    def test_within_budget_is_silent(self):
        with capture_logs() as captured:
            TopThetaBudgeter().cap(["a"], [_bid("a", 0.9)], 3)
        assert "[Memory][Budgeter]" not in "".join(captured)


def _harmful_card(card_id: str) -> MemoryCard:
    return MemoryCard(
        id=card_id,
        evolution_statistics={
            "ALL": {"intro_events": 6, "posterior_a": 1.0, "posterior_b": 9.0}
        },
    )


class TestEvictorLogging:
    def test_sweep_logs_evicted_ids(self):
        bank = {
            "bad-card": _harmful_card("bad-card"),
            "good-card": MemoryCard(id="good-card"),
        }
        with capture_logs() as captured:
            evicted = HarmEvictor().sweep(bank)
        text = "".join(captured)
        assert evicted == ["bad-card"]
        assert "[Memory][Evictor]" in text
        assert "bad-card" in text

    def test_clean_sweep_is_silent(self):
        with capture_logs() as captured:
            HarmEvictor().sweep({"good-card": MemoryCard(id="good-card")})
        assert "[Memory][Evictor]" not in "".join(captured)


def _admittable_row() -> IdeaStats:
    return IdeaStats(
        idea_id="idea-1",
        quartile="ALL",
        intro_events=4,
        IntroGain_best_median=0.05,
        IntroGain_best_rel_median=0.05,
        DownsideRate_best=0.1,
        SiblingWinRate_allgens=0.8,
    )


class TestAdmitterLogging:
    def test_select_logs_admitted_count(self):
        with capture_logs() as captured:
            kept = SignBasedAdmitter().select([_admittable_row()])
        text = "".join(captured)
        assert len(kept) == 1
        assert "[Memory][Admitter]" in text
        assert "SignBasedAdmitter" in text
        assert "idea-1" in text

    def test_empty_input_is_silent(self):
        with capture_logs() as captured:
            SignBasedAdmitter().select([])
        assert "[Memory][Admitter]" not in "".join(captured)


class TestInjectionPosteriorLogging:
    def test_bridge_logs_summary(self):
        programs = [
            InjectionOutcome(id="p1", fitness=0.5, selected_ids=["card-x"]),
            InjectionOutcome(id="c1", fitness=0.6, parents=["p1"]),
        ]
        with capture_logs() as captured:
            result = compute_injection_posterior(programs)
        text = "".join(captured)
        assert "card-x" in result
        assert "[Memory][InjectionPosterior]" in text

    def test_no_events_is_silent(self):
        with capture_logs() as captured:
            compute_injection_posterior([])
        assert "[Memory][InjectionPosterior]" not in "".join(captured)
