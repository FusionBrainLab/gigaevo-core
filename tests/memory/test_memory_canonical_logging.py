"""Every decision-making memory component must emit a canonical
``[Memory][Component]`` event at its decision point, so live runs are
debuggable from the log alone (auction draws, budget drops, evictions,
admissions, posterior bridge summary).
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger
import numpy as np

from gigaevo.memory.core.admitter import TieredAdmitter
from gigaevo.memory.core.auctioneer import ThompsonAuctioneer
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.shared_memory.injection_posterior import (
    compute_injection_posterior,
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
                [("card-hot", 9.0, 1.0), ("card-cold", 1.0, 9.0)],
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
        slate = [
            {"card_id": "a", "theta": 0.9},
            {"card_id": "b", "theta": 0.2},
        ]
        with capture_logs() as captured:
            kept = TopThetaBudgeter().cap(["a", "b"], slate, max_cards=1)
        text = "".join(captured)
        assert kept == ["a"]
        assert "[Memory][Budgeter]" in text
        assert "b" in text

    def test_within_budget_is_silent(self):
        with capture_logs() as captured:
            TopThetaBudgeter().cap(["a"], [{"card_id": "a", "theta": 0.9}], 3)
        assert "[Memory][Budgeter]" not in "".join(captured)


def _harmful_stats() -> dict:
    return {"ALL": {"intro_events": 6, "posterior_a": 1.0, "posterior_b": 9.0}}


class TestEvictorLogging:
    def test_sweep_logs_evicted_ids(self):
        bank = {
            "bad-card": {"evolution_statistics": _harmful_stats()},
            "good-card": {"evolution_statistics": None},
        }
        with capture_logs() as captured:
            evicted = HarmEvictor().sweep(bank)
        text = "".join(captured)
        assert evicted == ["bad-card"]
        assert "[Memory][Evictor]" in text
        assert "bad-card" in text

    def test_clean_sweep_is_silent(self):
        with capture_logs() as captured:
            HarmEvictor().sweep({"good-card": {"evolution_statistics": None}})
        assert "[Memory][Evictor]" not in "".join(captured)


def _admittable_row() -> IdeaStats:
    return IdeaStats(
        idea_id="idea-1",
        quartile="ALL",
        intro_events=4,
        IntroGain_best_rel_median=0.05,
        DownsideRate_best=0.1,
        SiblingWinRate_allgens=0.8,
    )


class TestAdmitterLogging:
    def test_select_logs_admitted_count(self):
        with capture_logs() as captured:
            kept = TieredAdmitter().select([_admittable_row()])
        text = "".join(captured)
        assert len(kept) == 1
        assert "[Memory][Admitter]" in text
        assert "TieredAdmitter" in text
        assert "idea-1" in text

    def test_empty_input_is_silent(self):
        with capture_logs() as captured:
            TieredAdmitter().select([])
        assert "[Memory][Admitter]" not in "".join(captured)


class TestInjectionPosteriorLogging:
    def test_bridge_logs_summary(self):
        programs = [
            {"id": "p1", "fitness": 0.5, "selected_ids": ["card-x"], "parents": []},
            {"id": "c1", "fitness": 0.6, "selected_ids": [], "parents": ["p1"]},
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
