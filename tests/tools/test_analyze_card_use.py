"""Tests for tools/analyze_card_use.py — offline card injection→use funnel.

Pre-registration: docs/audits/card_use_offline_prereg_2026-06-25.md.
Fixtures are synthetic normalized-child dicts built by factories; no live run is read.
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_card_use",
    Path(__file__).resolve().parents[2] / "tools" / "analyze_card_use.py",
)
acu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acu)


def make_child(
    cid="c",
    injected=(),
    base_selected=(),
    card_ids_used=(),
    fitness=None,
    base_fitness=None,
    iteration=0,
):
    return {
        "id": cid,
        "injected": list(injected),
        "base_selected": list(base_selected),
        "card_ids_used": list(card_ids_used),
        "fitness": fitness,
        "base_fitness": base_fitness,
        "iteration": iteration,
    }


def test_cited_cards_is_base_selected_intersect_used():
    child = make_child(
        injected=["a", "b", "c"], base_selected=["a", "b"], card_ids_used=["b", "z"]
    )
    # only b is both selected-for-base-parent AND declared used (z is a donor, not credited)
    assert acu.cited_cards(child) == {"b"}


def test_cited_cards_empty_when_no_overlap():
    child = make_child(injected=["a"], base_selected=["a"], card_ids_used=["x"])
    assert acu.cited_cards(child) == set()


def test_funnel_counts_injected_used_unread():
    children = [
        # injected 3, used 1 (a) -> unread 2
        make_child(injected=["a", "b", "c"], base_selected=["a"], card_ids_used=["a"]),
        # injected 2, used 0 -> unread 2
        make_child(injected=["d", "e"], base_selected=["d"], card_ids_used=[]),
        # no injection -> not in injected denominators
        make_child(injected=[], base_selected=[], card_ids_used=[]),
    ]
    f = acu.funnel(children)
    assert f["n_children"] == 3
    assert f["children_injected"] == 2
    assert f["children_used_any"] == 1
    assert f["injected_instances"] == 5
    assert f["used_instances"] == 1
    assert f["unread_instances"] == 4
    assert f["unread_frac"] == 4 / 5


def test_funnel_handles_no_injections():
    f = acu.funnel([make_child(injected=[], base_selected=[], card_ids_used=[])])
    assert f["injected_instances"] == 0
    assert f["unread_frac"] == 0.0  # defined as 0 when nothing injected


def test_use_conditional_gain_splits_used_vs_unread():
    children = [
        # used a card; gain = 0.10 - 0.02 = 0.08
        make_child(
            injected=["a"],
            base_selected=["a"],
            card_ids_used=["a"],
            fitness=0.10,
            base_fitness=0.02,
        ),
        # used a card; gain = 0.06 - 0.02 = 0.04
        make_child(
            injected=["b"],
            base_selected=["b"],
            card_ids_used=["b"],
            fitness=0.06,
            base_fitness=0.02,
        ),
        # injected, not used; gain = 0.03 - 0.02 = 0.01
        make_child(
            injected=["c"],
            base_selected=["c"],
            card_ids_used=[],
            fitness=0.03,
            base_fitness=0.02,
        ),
    ]
    g = acu.use_conditional_gain(children, higher_is_better=True)
    assert g["n_used"] == 2
    assert g["n_unread"] == 1
    assert abs(g["median_gain_used"] - 0.06) < 1e-9  # median(0.08, 0.04)
    assert abs(g["median_gain_unread"] - 0.01) < 1e-9


def test_use_conditional_gain_flips_sign_for_minimization():
    # lower fitness is better: a child that drops below its base parent IMPROVED
    child = make_child(
        injected=["a"],
        base_selected=["a"],
        card_ids_used=["a"],
        fitness=0.01,
        base_fitness=0.05,
    )
    g = acu.use_conditional_gain([child], higher_is_better=False)
    assert abs(g["median_gain_used"] - 0.04) < 1e-9  # (0.05 - 0.01) is a gain


def test_use_conditional_gain_excludes_invalid_fitness():
    children = [
        make_child(
            injected=["a"],
            base_selected=["a"],
            card_ids_used=["a"],
            fitness=None,  # invalid child
            base_fitness=0.02,
        ),
        make_child(
            injected=["b"],
            base_selected=["b"],
            card_ids_used=["b"],
            fitness=0.05,
            base_fitness=0.02,
        ),
    ]
    g = acu.use_conditional_gain(children, higher_is_better=True)
    assert g["n_used"] == 1  # the invalid one is dropped


def test_basin_share_fraction_below_threshold():
    children = [
        make_child(
            injected=["a"],
            base_selected=["a"],
            card_ids_used=["a"],
            fitness=0.022,
            base_fitness=0.020,
        ),  # gain 0.002 < 0.007 basin
        make_child(
            injected=["b"],
            base_selected=["b"],
            card_ids_used=["b"],
            fitness=0.040,
            base_fitness=0.020,
        ),  # gain 0.020 >= 0.007 jump
    ]
    s = acu.basin_share(children, threshold=0.007, higher_is_better=True)
    assert s["n_positive"] == 2
    assert abs(s["basin_share"] - 0.5) < 1e-9


def test_basin_share_counts_below_not_above_threshold():
    # asymmetric split so the < vs >= direction is observable (kills the boundary mutant)
    children = [
        make_child(
            injected=["a"],
            base_selected=["a"],
            card_ids_used=["a"],
            fitness=0.022,
            base_fitness=0.020,
        ),  # gain 0.002 < 0.007
        make_child(
            injected=["b"],
            base_selected=["b"],
            card_ids_used=["b"],
            fitness=0.025,
            base_fitness=0.020,
        ),  # gain 0.005 < 0.007
        make_child(
            injected=["c"],
            base_selected=["c"],
            card_ids_used=["c"],
            fitness=0.040,
            base_fitness=0.020,
        ),  # gain 0.020 >= 0.007
    ]
    s = acu.basin_share(children, threshold=0.007, higher_is_better=True)
    assert s["n_positive"] == 3
    assert abs(s["basin_share"] - 2 / 3) < 1e-9  # 2 below, 1 above


def test_normalize_program_extracts_fields():
    # the loader's field extraction from a raw disk-storage program JSON
    raw = {
        "id": "child1",
        "metrics": {"fitness": 0.05, "is_valid": 1.0},
        "iteration": 12,
        "metadata": {
            "memory_injected_idea_ids": ["a", "b"],
            "memory_base_selected_idea_ids": ["a"],
            "memory_base_metrics": {"fitness": 0.02, "is_valid": 1.0},
            "mutation_output": {"card_ids_used": ["a"], "base_parent": 1},
        },
    }
    c = acu.normalize_program(raw)
    assert c["id"] == "child1"
    assert c["injected"] == ["a", "b"]
    assert c["base_selected"] == ["a"]
    assert c["card_ids_used"] == ["a"]
    assert c["fitness"] == 0.05
    assert c["base_fitness"] == 0.02


def test_normalize_program_invalid_fitness_is_none():
    raw = {
        "id": "c",
        "metrics": {"fitness": 0.0, "is_valid": 0.0},
        "iteration": 1,
        "metadata": {"memory_base_metrics": {"fitness": 0.02, "is_valid": 1.0}},
    }
    c = acu.normalize_program(raw)
    assert c["fitness"] is None  # invalid child carries no scorable fitness
