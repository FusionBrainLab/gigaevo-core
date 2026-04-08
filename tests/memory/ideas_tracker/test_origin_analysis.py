"""Tests for the origin_analysis subpackage."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gigaevo.memory.ideas_tracker.utils.origin_analysis.events import (
    compute_descendant_metrics,
    compute_intro_events,
    mean_parent_fitness,
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
from gigaevo.memory.ideas_tracker.utils.origin_analysis.quartiles import (
    generation_quantile_bounds,
    generation_range_bounds,
    generation_to_quartile,
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


class TestGenerationQuantileBounds:
    def test_symmetric_list(self):
        b1, b2, b3 = generation_quantile_bounds([0, 1, 2, 3, 4, 5, 6, 7])
        assert b1 == 2.0
        assert b2 == 4.0  # q=0.5, idx=round(0.5*7)=round(3.5)=4 (banker's rounding)
        assert b3 == 5.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            generation_quantile_bounds([])


class TestGenerationRangeBounds:
    def test_gens_0_to_3(self):
        b1, b2, b3 = generation_range_bounds([0, 1, 2, 3])
        # gmin=0, gmax=3, span=4; b1=1.0, b2=2.0, b3=3.0
        assert b1 == pytest.approx(1.0)
        assert b2 == pytest.approx(2.0)
        assert b3 == pytest.approx(3.0)


class TestGenerationToQuartile:
    def test_q1(self):
        assert generation_to_quartile(0, 1.0, 2.0, 3.0) == "Q1"

    def test_q2(self):
        assert generation_to_quartile(1, 1.0, 2.0, 3.0) == "Q2"

    def test_q3(self):
        assert generation_to_quartile(2, 1.0, 2.0, 3.0) == "Q3"

    def test_q4(self):
        assert generation_to_quartile(3, 1.0, 2.0, 3.0) == "Q4"


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
            {"programs": [{"id": "p1", "generation": 0, "fitness": 0.3, "parents": []}]},
            {"programs": [{"id": "p1", "generation": 0, "fitness": 0.9, "parents": []}]},
        ]
        _write_json(progs_file, data)
        programs = load_programs(str(progs_file))
        assert programs["p1"]["fitness"] == 0.9


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
        groups = build_sibling_groups(SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent", 0)
        # p2 and p3 share best_parent p1 at generation 1
        key = ("best_parent", "p1", 1)
        assert set(groups[key]) == {"p2", "p3"}

    def test_gen_window_buckets_generations(self):
        groups = build_sibling_groups(SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent", 1)
        # gen_window=1: bucket = gen // 2; gen=1 -> bucket=0
        key_gen1 = ("best_parent", "p1", 0)
        assert set(groups[key_gen1]) == {"p2", "p3"}


class TestBuildSiblingGroupsAllgens:
    def test_groups_ignoring_generation(self):
        groups = build_sibling_groups_allgens(SIBLING_PROGRAMS, SIBLING_PARENTS_OF, "best_parent")
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


class TestMeanParentFitness:
    def test_mean_of_two(self):
        programs = {"a": {"fitness": 0.4}, "b": {"fitness": 0.6}}
        result = mean_parent_fitness(["a", "b"], programs)
        assert result == pytest.approx(0.5)

    def test_returns_none_for_empty(self):
        assert mean_parent_fitness([], {}) is None


class TestComputeIntroEvents:
    def test_detects_intro_event(self):
        events = compute_intro_events(
            programs=EVENTS_PROGRAMS,
            prog_to_origin_ideas=PROG_TO_ORIGIN_IDEAS,
            parents_of=EVENTS_PARENTS_OF,
            b1=0.5,
            b2=1.5,
            b3=2.5,
        )
        # p3 introduces idea_b (not in parent p2's idea set)
        assert len(events) == 1
        ev = events[0]
        assert ev.idea_id == "idea_b"
        assert ev.child_id == "p3"
        assert ev.quartile == "Q3"  # gen=2, b1=0.5, b2=1.5, b3=2.5 → 2 >= 1.5 and 2 < 2.5 → Q3

    def test_no_event_when_idea_in_parent(self):
        prog_to_ideas = {"p1": {"idea_a"}, "p2": {"idea_a"}, "p3": {"idea_a"}}
        events = compute_intro_events(
            programs=EVENTS_PROGRAMS,
            prog_to_origin_ideas=prog_to_ideas,
            parents_of=EVENTS_PARENTS_OF,
            b1=0.5,
            b2=1.5,
            b3=2.5,
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
