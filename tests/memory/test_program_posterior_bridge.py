"""Fix B bridge: extracting the injection posterior from live ``Program`` objects.

``run_increment`` receives full ``Program`` objects (metadata intact, since
``EXCLUDE_STAGE_RESULTS`` does not strip metadata). The adapter pulls each
program's id, parents, fitness (under the configured key), and the
``memory_selected_idea_ids`` stamped when that program was mutated (the cards
its CHILDREN's prompts contained), then hands them to the pure
``compute_injection_posterior``. These tests pin the extraction seam.
"""

from __future__ import annotations

import uuid

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.ideas_tracker.ideas_tracker import (
    _card_posterior_from_programs,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Lineage, Program


def _uuid(name: str) -> str:
    # Program.id is UUID-validated; map friendly labels to stable UUID5s so
    # parent references resolve to the right program.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"bridge-test-{name}"))


def _prog(
    name: str,
    *,
    fitness: float | None,
    parents: list[str] | None = None,
    selected: list[str] | None = None,
    fitness_key: str = "fitness",
    is_valid: float = 1.0,
) -> Program:
    metrics = {VALIDITY_KEY: float(is_valid)}
    if fitness is not None:
        metrics[fitness_key] = float(fitness)
    metadata = {}
    if selected is not None:
        metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] = list(selected)
    return Program(
        id=_uuid(name),
        code=f"# {name}",
        metrics=metrics,
        metadata=metadata,
        lineage=Lineage(parents=[_uuid(p) for p in (parents or [])]),
    )


def test_credits_card_injected_into_child() -> None:
    # The card rides on the parent's selected_ids; the child is the outcome.
    programs = [
        _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
        _prog("c1", fitness=0.85, parents=["root"], selected=[]),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert set(post) == {"program-A"}
    assert (post["program-A"]["posterior_a"], post["program-A"]["posterior_b"]) == (
        2.0,
        1.0,
    )


def test_harm_event_lowers_posterior() -> None:
    # Against a counterfactual of near-neutral sibling mutations, the card's
    # children fall far below the local base-rate -> a genuine harm event.
    siblings = [
        _prog(f"s{i}", fitness=0.85 + d, parents=["root"], selected=[])
        for i, d in enumerate((0.0, 0.005, -0.005, 0.003, -0.003, 0.002))
    ]
    programs = [
        _prog("root", fitness=0.85, parents=[], selected=[]),
        _prog("rootA", fitness=0.85, parents=[], selected=["program-A"]),
        *siblings,
        _prog("c1", fitness=0.70, parents=["rootA"], selected=[]),
        _prog("c2", fitness=0.71, parents=["rootA"], selected=[]),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert post["program-A"]["k_harm"] == 2
    assert (post["program-A"]["posterior_a"], post["program-A"]["posterior_b"]) == (
        1.0,
        3.0,
    )


def test_uses_configured_fitness_key() -> None:
    programs = [
        _prog(
            "root",
            fitness=0.80,
            parents=[],
            selected=["program-A"],
            fitness_key="r2_mean",
        ),
        _prog("c1", fitness=0.85, parents=["root"], selected=[], fitness_key="r2_mean"),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="r2_mean", higher_is_better=True
    )
    assert post["program-A"]["posterior_a"] == 2.0


def test_missing_fitness_key_treated_as_none() -> None:
    # Child with no metric under the configured key -> no gain / no event.
    programs = [
        _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
        _prog("c1", fitness=None, parents=["root"], selected=[]),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert post == {}


def test_direction_flip_lower_is_better() -> None:
    programs = [
        _prog("root", fitness=0.20, parents=[], selected=["program-A"]),
        _prog("c1", fitness=0.10, parents=["root"], selected=[]),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=False
    )
    assert post["program-A"]["k_harm"] == 0
    assert post["program-A"]["posterior_a"] == 2.0


def test_program_without_selected_metadata_contributes_nothing() -> None:
    # No memory_selected_idea_ids key at all -> empty selected -> no event.
    programs = [
        _prog("root", fitness=0.80, parents=[], selected=None),
        _prog("c1", fitness=0.85, parents=["root"], selected=None),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert post == {}


def test_invalid_child_with_sentinel_fitness_creates_no_harm_event() -> None:
    # Invalid programs carry a sentinel floor fitness (-100000) in this suite.
    # Reading it raw would register a catastrophic harm event; the validated
    # signal treats invalid fitness as absent, so the card is not credited.
    programs = [
        _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
        _prog(
            "c1",
            fitness=-100000.0,
            parents=["root"],
            selected=[],
            is_valid=-100000.0,
        ),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert post == {}


def test_child_with_only_invalid_parent_has_no_baseline() -> None:
    # The sole parent is invalid (sentinel fitness). Treating that sentinel as a
    # baseline would make the child look like a +100000 improvement; the
    # validated signal has no valid baseline here, so no event is recorded.
    programs = [
        _prog(
            "bad_parent",
            fitness=-100000.0,
            parents=[],
            selected=["program-A"],
            is_valid=-100000.0,
        ),
        _prog("c1", fitness=0.85, parents=["bad_parent"], selected=[]),
    ]
    post = _card_posterior_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert post == {}
