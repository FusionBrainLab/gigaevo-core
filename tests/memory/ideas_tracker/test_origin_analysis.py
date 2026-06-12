"""Tests for the origin_analysis subpackage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.efficacy import EfficacyEvent, EfficacyScorer, GenerationBucketer
from gigaevo.memory.ideas_tracker.utils.origin_analysis import AnalysisResult, analyse
from gigaevo.memory.ideas_tracker.utils.origin_analysis.aggregation import (
    aggregate_idea_rows,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.events import (
    compute_descendant_metrics,
    compute_intro_events,
    pick_best_parent,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.loader import (
    build_children,
    build_parents,
    compute_roots_memoized,
    invert_idea_to_programs,
    load_ideas,
    load_programs,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.siblings import (
    build_sibling_groups,
    build_sibling_groups_allgens,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    elite_threshold_by_top_k,
    mad,
    nancount,
    nanmedian,
    nanquantile,
    nanrate_bool,
    percentile_rank,
    robust_median,
    robust_quantile,
)
from gigaevo.memory.shared_memory.models import Quartile


class TestRobustMedian:
    def test_odd_list(self):
        assert robust_median([1.0, 3.0, 5.0]) == 3.0

    def test_even_list(self):
        assert robust_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_empty_returns_nan(self):
        assert math.isnan(robust_median([]))

    def test_single_element(self):
        assert robust_median([7.0]) == 7.0


class TestRobustQuantile:
    def test_q0_returns_min(self):
        assert robust_quantile([1.0, 2.0, 3.0], 0.0) == 1.0

    def test_q1_returns_max(self):
        assert robust_quantile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_q0_5_returns_median(self):
        assert robust_quantile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_empty_returns_nan(self):
        assert math.isnan(robust_quantile([], 0.5))


class TestMad:
    def test_known_values(self):
        # median=3, deviations=[2,1,0,1,2], mad=1
        result = mad([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == 1.0

    def test_empty_returns_nan(self):
        assert math.isnan(mad([]))


class TestPercentileRank:
    def test_value_at_max(self):
        assert percentile_rank([1.0, 2.0, 3.0], 3.0) == 1.0

    def test_value_at_min(self):
        assert percentile_rank([1.0, 2.0, 3.0], 0.5) == 0.0

    def test_empty_returns_nan(self):
        assert math.isnan(percentile_rank([], 1.0))

    def test_middle_value(self):
        assert percentile_rank([1.0, 2.0, 3.0], 2.0) == pytest.approx(2 / 3)


class TestEliteThreshold:
    def test_top_50_pct(self):
        threshold, count = elite_threshold_by_top_k([1.0, 2.0, 3.0, 4.0], 0.5)
        assert count == 2
        assert threshold == 3.0

    def test_empty_returns_nan(self):
        threshold, count = elite_threshold_by_top_k([], 0.1)
        assert math.isnan(threshold)
        assert count == 0


class TestNanHelpers:
    def test_nanmedian_skips_nan(self):
        assert nanmedian([1.0, float("nan"), 3.0]) == 2.0

    def test_nanmedian_all_nan(self):
        assert math.isnan(nanmedian([float("nan"), float("nan")]))

    def test_nanquantile_skips_nan(self):
        assert nanquantile([1.0, float("nan"), 3.0], 0.0) == 1.0

    def test_nanrate_bool_counts_gt_half(self):
        assert nanrate_bool([0.0, 1.0, 1.0]) == pytest.approx(2 / 3)

    def test_nanrate_bool_all_nan(self):
        assert math.isnan(nanrate_bool([float("nan")]))

    def test_nancount(self):
        assert nancount([1.0, float("nan"), 3.0]) == 2


class TestGenerationBucketer:
    def test_from_generations_equal_span_bounds(self):
        b = GenerationBucketer.from_generations([0, 1, 2, 3])
        # gmin=0, gmax=3, span=4; b1=1.0, b2=2.0, b3=3.0
        assert b.b1 == pytest.approx(1.0)
        assert b.b2 == pytest.approx(2.0)
        assert b.b3 == pytest.approx(3.0)

    def test_from_generations_empty_raises(self):
        with pytest.raises(ValueError):
            GenerationBucketer.from_generations([])

    def test_bucket_bounds_are_exclusive_upper(self):
        b = GenerationBucketer(b1=1.0, b2=2.0, b3=3.0)
        assert b.bucket(0) is Quartile.Q1
        assert b.bucket(1) is Quartile.Q2
        assert b.bucket(2) is Quartile.Q3
        assert b.bucket(3) is Quartile.Q4

    def test_generations_by_bucket_partitions_quarters(self):
        b = GenerationBucketer(b1=1.0, b2=2.0, b3=3.0)
        assert b.generations_by_bucket([0, 1, 2, 3, 3]) == {
            Quartile.Q1: {0},
            Quartile.Q2: {1},
            Quartile.Q3: {2},
            Quartile.Q4: {3},
        }


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


BANKS_FIXTURE = [
    {
        "active_bank": [
            {"id": "idea_a", "programs": ["p1", "p2"], "description": "Idea A"},
            {"id": "idea_b", "programs": ["p3"], "description": "Idea B"},
        ]
    }
]

PROGRAMS_FIXTURE = [
    {
        "programs": [
            {"id": "p1", "generation": 0, "fitness": 0.5, "parents": []},
            {"id": "p2", "generation": 1, "fitness": 0.6, "parents": ["p1"]},
            {"id": "p3", "generation": 2, "fitness": 0.7, "parents": ["p2"]},
            {"id": "p4", "generation": 3, "fitness": 0.8, "parents": ["p2", "p3"]},
        ]
    }
]


class TestLoadIdeas:
    def test_loads_idea_to_programs(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        _write_json(banks_file, BANKS_FIXTURE)
        idea_to_progs, idea_desc = load_ideas(str(banks_file))
        assert idea_to_progs["idea_a"] == {"p1", "p2"}
        assert idea_to_progs["idea_b"] == {"p3"}

    def test_loads_descriptions(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _, idea_desc = load_ideas(str(banks_file))
        assert idea_desc["idea_a"] == "Idea A"

    def test_invalid_format_raises(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        _write_json(banks_file, {"no_active_bank": []})
        with pytest.raises(ValueError):
            load_ideas(str(banks_file))


class TestLoadPrograms:
    def test_loads_programs_by_id(self, tmp_path):
        progs_file = tmp_path / "programs.json"
        _write_json(progs_file, PROGRAMS_FIXTURE)
        programs = load_programs(str(progs_file))
        assert set(programs.keys()) == {"p1", "p2", "p3", "p4"}
        assert programs["p1"]["generation"] == 0

    def test_deduplicates_keeps_best_fitness(self, tmp_path):
        progs_file = tmp_path / "programs.json"
        data = [
            {
                "programs": [
                    {"id": "p1", "generation": 0, "fitness": 0.3, "parents": []}
                ]
            },
            {
                "programs": [
                    {"id": "p1", "generation": 0, "fitness": 0.9, "parents": []}
                ]
            },
        ]
        _write_json(progs_file, data)
        programs = load_programs(str(progs_file))
        assert programs["p1"]["fitness"] == 0.9

    def test_deduplicates_keeps_best_fitness_lower_is_better(self, tmp_path):
        progs_file = tmp_path / "programs.json"
        data = [
            {
                "programs": [
                    {"id": "p1", "generation": 0, "fitness": 0.3, "parents": []}
                ]
            },
            {
                "programs": [
                    {"id": "p1", "generation": 0, "fitness": 0.9, "parents": []}
                ]
            },
        ]
        _write_json(progs_file, data)
        programs = load_programs(str(progs_file), higher_is_better=False)
        assert programs["p1"]["fitness"] == 0.3


class TestBuildParentsAndChildren:
    def test_build_parents(self, tmp_path):
        progs_file = tmp_path / "programs.json"
        _write_json(progs_file, PROGRAMS_FIXTURE)
        programs = load_programs(str(progs_file))
        parents_of = build_parents(programs)
        assert parents_of["p1"] == []
        assert parents_of["p2"] == ["p1"]
        assert set(parents_of["p4"]) == {"p2", "p3"}

    def test_build_children(self, tmp_path):
        progs_file = tmp_path / "programs.json"
        _write_json(progs_file, PROGRAMS_FIXTURE)
        programs = load_programs(str(progs_file))
        parents_of = build_parents(programs)
        children_of = build_children(parents_of)
        assert "p2" in children_of["p1"]
        assert "p3" in children_of["p2"]


class TestInvertIdeaToPrograms:
    def test_invert(self):
        mapping = {"idea_a": {"p1", "p2"}, "idea_b": {"p2"}}
        prog_to_ideas = invert_idea_to_programs(mapping)
        assert "idea_a" in prog_to_ideas["p1"]
        assert "idea_a" in prog_to_ideas["p2"]
        assert "idea_b" in prog_to_ideas["p2"]


class TestComputeRootsMemoized:
    def test_roots_of_root_is_itself(self):
        parents_of = {"p1": [], "p2": ["p1"], "p3": ["p2"]}
        roots = compute_roots_memoized(parents_of)
        assert roots["p1"] == {"p1"}

    def test_roots_trace_back(self):
        parents_of = {"p1": [], "p2": ["p1"], "p3": ["p2"]}
        roots = compute_roots_memoized(parents_of)
        assert roots["p3"] == {"p1"}


SIBLING_PROGRAMS = {
    "p1": {"generation": 0, "fitness": 0.5, "parents": []},
    "p2": {"generation": 1, "fitness": 0.6, "parents": ["p1"]},
    "p3": {"generation": 1, "fitness": 0.4, "parents": ["p1"]},
    "p4": {"generation": 2, "fitness": 0.7, "parents": ["p2"]},
    "p5": {"generation": 2, "fitness": 0.3, "parents": ["p2"]},
}
SIBLING_PARENTS_OF = {
    "p1": [],
    "p2": ["p1"],
    "p3": ["p1"],
    "p4": ["p2"],
    "p5": ["p2"],
}


class TestBuildSiblingGroups:
    def test_groups_children_of_same_parent(self):
        groups = build_sibling_groups(
            SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent", 0
        )
        # p2 and p3 share best_parent p1 at generation 1
        key = ("best_parent", "p1", 1)
        assert set(groups[key]) == {"p2", "p3"}

    def test_gen_window_buckets_generations(self):
        groups = build_sibling_groups(
            SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent", 1
        )
        # gen_window=1: bucket = gen // 2; gen=1 -> bucket=0
        key_gen1 = ("best_parent", "p1", 0)
        assert set(groups[key_gen1]) == {"p2", "p3"}


class TestBuildSiblingGroupsAllgens:
    def test_groups_ignoring_generation(self):
        groups = build_sibling_groups_allgens(
            SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent"
        )
        key = ("best_parent_allgens", "p1")
        assert set(groups[key]) == {"p2", "p3"}


EVENTS_PROGRAMS = {
    "p1": {"generation": 0, "fitness": 0.5, "parents": []},
    "p2": {"generation": 1, "fitness": 0.7, "parents": ["p1"]},
    "p3": {"generation": 2, "fitness": 0.8, "parents": ["p2"]},
}
EVENTS_PARENTS_OF = {"p1": [], "p2": ["p1"], "p3": ["p2"]}
PROG_TO_ORIGIN_IDEAS = {
    "p1": {"idea_a"},
    "p2": {"idea_a"},
    "p3": {"idea_b"},  # idea_b not in p2 → intro event for p3
}


class TestPickBestParent:
    def test_picks_highest_fitness(self):
        programs = {
            "a": {"fitness": 0.3},
            "b": {"fitness": 0.8},
        }
        best_pid, best_fit = pick_best_parent(["a", "b"], programs)
        assert best_pid == "b"
        assert best_fit == pytest.approx(0.8)

    def test_returns_none_for_empty(self):
        assert pick_best_parent([], {}) is None


class TestComputeIntroEvents:
    def test_detects_intro_event(self):
        events = compute_intro_events(
            programs=EVENTS_PROGRAMS,
            prog_to_origin_ideas=PROG_TO_ORIGIN_IDEAS,
            parents_of=EVENTS_PARENTS_OF,
            bucketer=GenerationBucketer(b1=0.5, b2=1.5, b3=2.5),
        )
        # p3 introduces idea_b (not in parent p2's idea set)
        assert len(events) == 1
        ev = events[0]
        assert ev.idea_id == "idea_b"
        assert ev.child_id == "p3"
        assert (
            ev.quartile is Quartile.Q3
        )  # gen=2, b1=0.5, b2=1.5, b3=2.5 → 2 >= 1.5 and 2 < 2.5 → Q3

    def test_no_event_when_idea_in_parent(self):
        prog_to_ideas = {"p1": {"idea_a"}, "p2": {"idea_a"}, "p3": {"idea_a"}}
        events = compute_intro_events(
            programs=EVENTS_PROGRAMS,
            prog_to_origin_ideas=prog_to_ideas,
            parents_of=EVENTS_PARENTS_OF,
            bucketer=GenerationBucketer(b1=0.5, b2=1.5, b3=2.5),
        )
        assert len(events) == 0


class TestComputeDescendantMetrics:
    def test_no_descendants(self):
        children_of: dict[str, list[str]] = {"p1": [], "p2": [], "p3": []}
        dm = compute_descendant_metrics(
            child_id="p3",
            child_gen=2,
            programs=EVENTS_PROGRAMS,
            children_of=children_of,
            elite_pids=set(),
            gmax=2,
            k=5,
        )
        assert dm.desc_count_k == 0
        assert dm.branching_factor == 0
        assert dm.reaches_elite_k == 0.0


class TestPipelineEndToEnd:
    def test_returns_analysis_result(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        result = analyse(str(banks_file), str(progs_file))

        assert isinstance(result, AnalysisResult)
        assert all(isinstance(s, IdeaStats) for s in result.summary)
        assert all(isinstance(s, IdeaStats) for s in result.best_ideas)

    def test_summary_has_five_rows_per_idea(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        result = analyse(str(banks_file), str(progs_file))

        # 2 ideas × 5 quartile rows each = 10 rows
        assert len(result.summary) == 10

    def test_summary_rows_have_expected_keys(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        result = analyse(str(banks_file), str(progs_file))

        row = result.summary[0].as_row()
        for key in [
            "idea_id",
            "quartile",
            "intro_events",
            "IntroGain_best_median",
            "IntroGain_best_adj_median",
            "description",
        ]:
            assert key in row


class TestHigherIsBetterDirection:
    def test_lower_is_better_negates_gains_on_ingestion(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        up = analyse(str(banks_file), str(progs_file))
        down = analyse(str(banks_file), str(progs_file), higher_is_better=False)

        def all_row(result, idea_id):
            return next(
                s
                for s in result.summary
                if s.idea_id == idea_id and s.quartile == "ALL"
            )

        # idea_b is introduced by p3 (fit 0.7) from parent p2 (fit 0.6):
        # gain +0.1 when maximizing, -0.1 when minimizing (fitness negated
        # on ingestion so "positive gain" always means improvement).
        assert all_row(up, "idea_b").IntroGain_best_median == pytest.approx(0.1)
        assert all_row(down, "idea_b").IntroGain_best_median == pytest.approx(-0.1)

    def test_lower_is_better_dedup_keeps_better_duplicate(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        # A second snapshot re-exports p3 with a worse (higher, since
        # minimizing) fitness; dedup must keep the 0.7 copy, so idea_b's gain
        # stays -0.1 instead of -(5.0) - -(0.6) = -4.4.
        worse_p3 = {"id": "p3", "generation": 2, "fitness": 5.0, "parents": ["p2"]}
        _write_json(progs_file, PROGRAMS_FIXTURE + [{"programs": [worse_p3]}])

        down = analyse(str(banks_file), str(progs_file), higher_is_better=False)

        row = next(
            s for s in down.summary if s.idea_id == "idea_b" and s.quartile == "ALL"
        )
        assert row.IntroGain_best_median == pytest.approx(-0.1)


def _make_events(
    rows: list[tuple[str, str, float, float]],
) -> list[EfficacyEvent]:
    """Build minimal events from (idea_id, quartile, IntroGain_best,
    best_parent_fit) tuples; every metric not under test stays at its NaN
    default."""
    return [
        EfficacyEvent(
            idea_id=idea_id,
            quartile=Quartile(quartile),
            child_id=f"child_{i}",
            IntroGain_best=gain,
            best_parent_fit=best_parent_fit,
        )
        for i, (idea_id, quartile, gain, best_parent_fit) in enumerate(rows, start=1)
    ]


# Neutral counterfactual population: a spread of equally-fit (best_parent_fit=0)
# mutations whose gains are centred on zero. Harm is judged relative to this
# base-rate, so a card is only penalised when its children fall below it by more
# than the population's robust noise scale.
_BG = [
    ("bg", "Q4", g, 0.0)
    for g in (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04)
]


class TestPosteriorFields:
    """Beta-Binomial downside posterior per (idea, quartile) row.

    Harm is parent-fitness-local and noise-aware: a child counts as harmful only
    when its gain falls below the typical gain of equally-fit parents' mutations
    (``_BG``) by more than the robust noise band. A bare ``< 0`` test would
    mislabel sub-noise jitter and plateau regression as harm.
    """

    def _aggregate(self, rows, idea_ids):
        return aggregate_idea_rows(
            events=_make_events(rows),
            idea_to_origin_programs={i: set() for i in idea_ids},
            idea_desc={i: i for i in idea_ids},
            programs={},
            elite_pids=set(),
            roots_memo={},
            bucketer=GenerationBucketer(b1=0.5, b2=1.5, b3=2.5),
            gens_by_quartile={q: set() for q in Quartile.quarters()},
            total_distinct_gens=1,
            scorer=EfficacyScorer(),
        )

    @staticmethod
    def _all_row(stats: list[IdeaStats], idea_id: str) -> dict:
        return next(
            s for s in stats if s.idea_id == idea_id and s.quartile == "ALL"
        ).as_row()

    def test_within_noise_regressions_are_not_harm(self):
        # Small negative gains, all within the population's noise band -> no harm,
        # so the card stays confident. A bare ``< 0`` test would penalise four of
        # the five and break confidence.
        card = [
            ("noisy", "Q4", g, 0.0) for g in (-0.005, -0.003, -0.004, 0.002, -0.001)
        ]
        row = self._all_row(self._aggregate(_BG + card, ["bg", "noisy"]), "noisy")
        assert row["intro_events"] == 5
        assert row["posterior_a"] == 6.0
        assert row["posterior_b"] == 1.0
        assert row["DownsideRate_best"] == pytest.approx(0.0)
        assert bool(row["efficacy_confident"]) is True

    def test_consistent_regression_below_baseline_is_not_confident(self):
        # Children consistently far below the local base-rate (beyond the noise
        # band) -> genuine harm survives: no false negative.
        card = [("bad", "Q4", -0.2, 0.0) for _ in range(4)]
        row = self._all_row(self._aggregate(_BG + card, ["bg", "bad"]), "bad")
        assert row["intro_events"] == 4
        assert row["posterior_a"] == 1.0
        assert row["posterior_b"] == 5.0
        assert row["p_help_mean"] == pytest.approx(1 / 6)
        assert row["DownsideRate_best"] == pytest.approx(1.0)
        assert bool(row["efficacy_confident"]) is False

    def test_cold_card_with_no_events_is_not_confident(self):
        out = self._aggregate(_BG + [("good", "Q4", 0.05, 0.0)], ["bg", "good", "cold"])
        row = self._all_row(out, "cold")
        assert row["intro_events"] == 0
        assert row["posterior_a"] == 1.0
        assert row["posterior_b"] == 1.0
        assert math.isnan(row["p_help_lo20"])
        assert bool(row["efficacy_confident"]) is False


# Frontier-regression cohort: every event jumps weak parents (+0.07-ish), so the
# typical child gains ~+0.07 regardless of which idea it carries.
_FRONTIER_BG = [
    ("fbg", "Q4", g, 0.59)
    for g in (0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.076, 0.078)
]


class TestAdjustedMedian:
    """``IntroGain_best_adj_median``: the displayed median must be measured against
    the parent-fitness-local counterfactual the posterior already uses, not raw
    child-minus-parent. A card whose children merely regress weak parents to the
    population frontier shows a large raw median but contributes nothing beyond
    the cohort baseline."""

    _aggregate = TestPosteriorFields._aggregate
    _all_row = staticmethod(TestPosteriorFields._all_row)

    def test_frontier_regression_card_has_zero_adjusted_median(self):
        card = [("rtf", "Q4", 0.070, 0.59) for _ in range(5)]
        row = self._all_row(self._aggregate(_FRONTIER_BG + card, ["fbg", "rtf"]), "rtf")
        assert row["IntroGain_best_median"] == pytest.approx(0.070)
        assert row["IntroGain_best_adj_median"] == pytest.approx(0.0, abs=1e-9)

    def test_genuinely_better_card_keeps_positive_adjusted_median(self):
        card = [("gen", "Q4", 0.100, 0.59) for _ in range(3)]
        row = self._all_row(self._aggregate(_FRONTIER_BG + card, ["fbg", "gen"]), "gen")
        assert row["IntroGain_best_adj_median"] == pytest.approx(0.030, abs=0.005)

    def test_cold_card_adjusted_median_is_nan(self):
        out = self._aggregate(_FRONTIER_BG, ["fbg", "cold"])
        row = self._all_row(out, "cold")
        assert math.isnan(row["IntroGain_best_adj_median"])


class _MarkerAdmitter:
    def __init__(self) -> None:
        self.seen: list[IdeaStats] | None = None

    def select(self, stats: list[IdeaStats]) -> list[IdeaStats]:
        self.seen = stats
        return stats[:1]


class TestAdmitterInjection:
    def test_analyse_uses_injected_admitter(self, tmp_path):
        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        marker = _MarkerAdmitter()
        result = analyse(str(banks_file), str(progs_file), admitter=marker)

        assert marker.seen is not None
        assert result.best_ideas == marker.seen[:1]

    def test_default_admitter_is_sign_based(self, tmp_path):
        from gigaevo.memory.core.admitter import SignBasedAdmitter

        banks_file = tmp_path / "banks.json"
        progs_file = tmp_path / "programs.json"
        _write_json(banks_file, BANKS_FIXTURE)
        _write_json(progs_file, PROGRAMS_FIXTURE)

        default = analyse(str(banks_file), str(progs_file)).best_ideas
        sign_based = analyse(
            str(banks_file), str(progs_file), admitter=SignBasedAdmitter()
        ).best_ideas
        # NaN-bearing models from separate runs never compare equal; pin identity
        # by (idea_id, quartile) instead.
        assert [(s.idea_id, s.quartile) for s in default] == [
            (s.idea_id, s.quartile) for s in sign_based
        ]


class TestMultiIdeaChildWeighsOnce:
    """A child that introduces several ideas is ONE mutation outcome: it weighs
    once in the counterfactual baseline and noise band, matching the card-side
    injection posterior's per-child cohort."""

    @staticmethod
    def _aggregate(events: list[EfficacyEvent]) -> list[IdeaStats]:
        idea_ids = sorted({e.idea_id for e in events})
        return aggregate_idea_rows(
            events=events,
            idea_to_origin_programs={i: set() for i in idea_ids},
            idea_desc={i: i for i in idea_ids},
            programs={},
            elite_pids=set(),
            roots_memo={},
            bucketer=GenerationBucketer(b1=0.5, b2=1.5, b3=2.5),
            gens_by_quartile={q: set() for q in Quartile.quarters()},
            total_distinct_gens=1,
            scorer=EfficacyScorer(),
        )

    def test_duplicated_outlier_child_does_not_shift_probe_posterior(self):
        background = _make_events(_BG)
        outlier = EfficacyEvent(
            idea_id="x",
            quartile=Quartile.Q4,
            child_id="outlier_child",
            IntroGain_best=-0.30,
            best_parent_fit=0.0,
        )
        probe = EfficacyEvent(
            idea_id="probe",
            quartile=Quartile.Q4,
            child_id="probe_child",
            IntroGain_best=-0.02,
            best_parent_fit=0.0,
        )
        single_idea = self._aggregate([*background, outlier, probe])
        multi_idea = self._aggregate(
            [*background, outlier, outlier.model_copy(update={"idea_id": "y"}), probe]
        )

        def probe_all(rows: list[IdeaStats]) -> IdeaStats:
            return next(
                r for r in rows if r.idea_id == "probe" and r.quartile is Quartile.ALL
            )

        before, after = probe_all(single_idea), probe_all(multi_idea)
        for field in (
            "posterior_a",
            "posterior_b",
            "DownsideRate_best",
            "IntroGain_best_adj_median",
            "efficacy_confident",
        ):
            assert getattr(after, field) == getattr(before, field), field


class TestWriteCsv:
    def test_empty_rows_still_write_canonical_header(self, tmp_path):
        from gigaevo.memory.ideas_tracker.utils.origin_analysis.pipeline import (
            _write_csv,
        )

        path = tmp_path / "out.csv"
        _write_csv(path, [])

        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(IdeaStats.model_fields)
