"""Behavioral pins for TieredAdmitter admission conditions.

Single-intro-event ideas are admitted only on BornInElite evidence;
descendant-count popularity alone no longer qualifies.
"""

from gigaevo.memory.core.admitter import TieredAdmitter
from gigaevo.memory.core.idea_stats import IdeaStats


def make_idea_row(
    idea_id: str,
    *,
    quartile: str = "ALL",
    intro_events: int = 3,
    intro_gain_rel_median: float = 0.05,
    downside_rate: float = 0.1,
    sibling_win_rate: float = 0.8,
    intro_gain_p10: float = 0.01,
    born_in_elite_rate: float = 0.0,
    reaches_elite_rate: float = 0.0,
    descendant_count_median: float = 1.0,
    intro_gain_median: float = 0.02,
) -> dict:
    return {
        "idea_id": idea_id,
        "quartile": quartile,
        "intro_events": intro_events,
        "IntroGain_best_rel_median": intro_gain_rel_median,
        "DownsideRate_best": downside_rate,
        "SiblingWinRate_allgens": sibling_win_rate,
        "IntroGain_best_p10": intro_gain_p10,
        "BornInElite_rate": born_in_elite_rate,
        "ReachesElite_k_rate": reaches_elite_rate,
        "DescendantCount_k_median": descendant_count_median,
        "IntroGain_best_median": intro_gain_median,
    }


def select(rows: list[dict]) -> list[IdeaStats]:
    return TieredAdmitter().select([IdeaStats.model_validate(r) for r in rows])


def kept_ids(rows: list[dict]) -> set[str]:
    return {s.idea_id for s in select(rows)}


class TestMultiEventBranchesUnchanged:
    def test_three_events_with_majority_sibling_wins_kept(self):
        rows = [make_idea_row("idea-a", intro_events=3, sibling_win_rate=0.5)]
        assert kept_ids(rows) == {"idea-a"}

    def test_three_events_below_majority_sibling_wins_dropped(self):
        rows = [make_idea_row("idea-a", intro_events=3, sibling_win_rate=0.4)]
        assert kept_ids(rows) == set()

    def test_two_events_positive_p10_and_perfect_sibling_wins_kept(self):
        rows = [
            make_idea_row(
                "idea-a",
                intro_events=2,
                intro_gain_p10=0.01,
                sibling_win_rate=1.0,
            )
        ]
        assert kept_ids(rows) == {"idea-a"}

    def test_two_events_nonpositive_p10_dropped(self):
        rows = [
            make_idea_row(
                "idea-a",
                intro_events=2,
                intro_gain_p10=0.0,
                sibling_win_rate=1.0,
            )
        ]
        assert kept_ids(rows) == set()


class TestSingleEventRequiresBornInElite:
    def test_born_in_elite_kept(self):
        rows = [make_idea_row("idea-a", intro_events=1, born_in_elite_rate=1.0)]
        assert kept_ids(rows) == {"idea-a"}

    def test_popularity_and_reaches_elite_alone_dropped(self):
        rows = [
            make_idea_row(
                "idea-a",
                intro_events=1,
                born_in_elite_rate=0.0,
                reaches_elite_rate=1.0,
                descendant_count_median=100.0,
            )
        ]
        assert kept_ids(rows) == set()

    def test_no_elite_evidence_dropped(self):
        rows = [make_idea_row("idea-a", intro_events=1, born_in_elite_rate=0.0)]
        assert kept_ids(rows) == set()


class TestBaseGate:
    def test_nonpositive_relative_gain_dropped(self):
        rows = [make_idea_row("idea-a", intro_gain_rel_median=0.005)]
        assert kept_ids(rows) == set()

    def test_high_downside_rate_dropped(self):
        rows = [make_idea_row("idea-a", downside_rate=0.5)]
        assert kept_ids(rows) == set()


class TestQuartileDedup:
    def test_one_row_per_idea_preferring_all_quartile(self):
        rows = [
            make_idea_row("idea-a", quartile="Q4", intro_gain_median=0.9),
            make_idea_row("idea-a", quartile="ALL", intro_gain_median=0.1),
            make_idea_row("idea-b", quartile="Q2"),
        ]
        result = select(rows)
        assert sorted(s.idea_id for s in result) == ["idea-a", "idea-b"]
        row_a = next(s for s in result if s.idea_id == "idea-a")
        assert row_a.quartile == "ALL"
