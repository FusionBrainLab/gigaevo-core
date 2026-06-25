"""Fix B bridge: extracting per-card gain events from live ``Program`` objects.

``run_increment`` receives full ``Program`` objects (metadata intact, since
``EXCLUDE_STAGE_RESULTS`` does not strip metadata). The adapter pulls each
program's use-attribution stamps — the base parent's selected card ids
(``memory_base_selected_idea_ids``), the base parent's frozen metrics
(``memory_base_metrics``), and the cards the mutator declared applied
(``mutation_output.card_ids_used``) — into ``InjectionOutcome`` rows and hands
them to the pure ``compute_contextual_gains``. Reputation then computes every
per-card statistic from those events at read time. These tests pin the
extraction seam, resolving the resulting events through
``BetaBinomialReputation().card_stats`` to read the posterior block.
"""

from __future__ import annotations

import uuid

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.context import ContextualGain
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.ideas_tracker.ideas_tracker import (
    _card_gain_events_from_programs,
)
from gigaevo.memory.shared_memory.models import CardStatsBlock, MemoryCard
from gigaevo.programs.metrics.context import (
    MAX_VALUE_DEFAULT,
    MIN_VALUE_DEFAULT,
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)
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
    base_selected: list[str] | None = None,
    base_fitness: float | None = None,
    used: list[str] | None = None,
    fitness_key: str = "fitness",
    is_valid: float = 1.0,
    base_is_valid: float = 1.0,
) -> Program:
    """A child program stamped with the use-attribution metadata the bridge
    reads: the base parent's selected cards, its frozen metrics (carrying the
    base fitness under ``fitness_key``), and the cards declared used."""
    metrics = {VALIDITY_KEY: float(is_valid)}
    if fitness is not None:
        metrics[fitness_key] = float(fitness)
    metadata: dict = {}
    if base_selected is not None:
        metadata[MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY] = list(base_selected)
    if base_fitness is not None:
        metadata[MUTATION_MEMORY_BASE_METRICS_METADATA_KEY] = {
            VALIDITY_KEY: float(base_is_valid),
            fitness_key: float(base_fitness),
        }
    if used is not None:
        metadata[MUTATION_OUTPUT_METADATA_KEY] = {"card_ids_used": list(used)}
    return Program(
        id=_uuid(name),
        code=f"# {name}",
        metrics=metrics,
        metadata=metadata,
        lineage=Lineage(parents=[_uuid(p) for p in (parents or [])]),
    )


def _block(
    events: dict[str, list[ContextualGain]],
    card_id: str,
    *,
    reputation: BetaBinomialReputation | None = None,
) -> CardStatsBlock | None:
    """Resolve a card's gain events to its posterior block at read time."""
    rep = reputation or BetaBinomialReputation()
    return rep.card_stats(MemoryCard(id=card_id, gain_events=events[card_id]))


def test_credits_card_injected_into_child() -> None:
    # The card was selected for the base parent and declared used; the child is
    # the outcome, base-relative to the base parent's frozen fitness.
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert set(events) == {"program-A"}
    block = _block(events, "program-A")
    assert (block.posterior_a, block.posterior_b) == (2.0, 1.0)


def test_harm_event_lowers_posterior() -> None:
    # Two credited children fall far below the base parent's frozen fitness ->
    # two genuine harm events.
    programs = [
        _prog(
            "c1",
            fitness=0.70,
            parents=["rootA"],
            base_selected=["program-A"],
            base_fitness=0.85,
            used=["program-A"],
        ),
        _prog(
            "c2",
            fitness=0.71,
            parents=["rootA"],
            base_selected=["program-A"],
            base_fitness=0.85,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    block = _block(events, "program-A")
    assert block.k_harm == 2
    assert (block.posterior_a, block.posterior_b) == (1.0, 3.0)


def test_uses_configured_fitness_key() -> None:
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
            fitness_key="r2_mean",
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="r2_mean", higher_is_better=True
    )
    block = _block(events, "program-A")
    assert block.posterior_a == 2.0


def test_missing_fitness_key_treated_as_none() -> None:
    # Child with no metric under the configured key -> no gain / no event.
    programs = [
        _prog(
            "c1",
            fitness=None,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert events == {}


def test_direction_flip_lower_is_better() -> None:
    programs = [
        _prog(
            "c1",
            fitness=0.10,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.20,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=False
    )
    block = _block(events, "program-A")
    assert block.k_harm == 0
    assert block.posterior_a == 2.0


def test_program_without_base_selected_metadata_contributes_nothing() -> None:
    # No base-selected stamp at all -> empty base_selected -> no credit.
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["root"],
            base_selected=None,
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert events == {}


def test_invalid_child_with_sentinel_fitness_is_one_forced_harm_event() -> None:
    # Invalid programs carry a sentinel floor fitness (-100000) in this suite.
    # Reading it raw would register a catastrophic gain magnitude; the child is
    # instead a single forced harm event with no gain.
    programs = [
        _prog(
            "c1",
            fitness=-100000.0,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
            is_valid=-100000.0,
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    block = _block(events, "program-A")
    assert block.intro_events == 1
    assert block.k_harm == 1
    assert (block.posterior_a, block.posterior_b) == (1.0, 2.0)


def test_child_missing_is_valid_is_excluded() -> None:
    # Validity is a contract: a program without the is_valid metric is treated
    # as invalid, matching record eligibility.
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    del programs[0].metrics[VALIDITY_KEY]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert events == {}


def test_sentinel_fitness_via_metrics_context_is_forced_harm() -> None:
    # Belt over the is_valid braces: a child claiming validity but carrying the
    # metric's sentinel floor is caught through MetricsContext.is_sentinel —
    # judged invalid, so one forced harm event, never a -100000 gain.
    ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Primary fitness.",
                is_primary=True,
                higher_is_better=True,
            )
        }
    )
    programs = [
        _prog(
            "c1",
            fitness=MIN_VALUE_DEFAULT,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs,
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=ctx,
    )
    block = _block(events, "program-A")
    assert (block.intro_events, block.k_harm) == (1, 1)


def test_sentinel_fitness_lower_is_better_is_forced_harm() -> None:
    # Lower-is-better metrics default to the +1e5 sentinel ceiling; the
    # MetricsContext gate must catch that side too.
    ctx = MetricsContext(
        specs={
            "cost": MetricSpec(
                description="Primary cost.",
                is_primary=True,
                higher_is_better=False,
            )
        }
    )
    programs = [
        _prog(
            "c1",
            fitness=MAX_VALUE_DEFAULT,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.30,
            used=["program-A"],
            fitness_key="cost",
        ),
    ]
    events = _card_gain_events_from_programs(
        programs,
        fitness_key="cost",
        higher_is_better=False,
        metrics_context=ctx,
    )
    block = _block(events, "program-A")
    assert (block.intro_events, block.k_harm) == (1, 1)


def test_child_with_only_invalid_base_has_no_baseline() -> None:
    # The frozen base metrics carry a sentinel/invalid is_valid. Treating that
    # sentinel fitness as a baseline would make the child look like a +100000
    # improvement; with no valid base baseline, no event is recorded.
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["bad_parent"],
            base_selected=["program-A"],
            base_fitness=-100000.0,
            used=["program-A"],
            base_is_valid=-100000.0,
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    assert events == {}


def test_reputation_knobs_reach_computation() -> None:
    # One success event -> Beta(2,1). Default quantile 0.20 reads sqrt(0.2)~0.447
    # (not confident); an optimistic 0.90 quantile reads sqrt(0.9)~0.949
    # (confident) — proving the read-time reputation's knobs reach the math.
    programs = [
        _prog(
            "c1",
            fitness=0.85,
            parents=["root"],
            base_selected=["program-A"],
            base_fitness=0.80,
            used=["program-A"],
        ),
    ]
    events = _card_gain_events_from_programs(
        programs, fitness_key="fitness", higher_is_better=True
    )
    default = _block(events, "program-A")
    assert default.efficacy_confident is False

    optimist = BetaBinomialReputation(confident_quantile=0.9)
    block = _block(events, "program-A", reputation=optimist)
    assert block.efficacy_confident is True
    assert block.p_help_lo20 == pytest.approx(0.9**0.5)
