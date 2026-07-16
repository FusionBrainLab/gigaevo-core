from __future__ import annotations

import uuid

import numpy as np
import pytest

from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
from gigaevo.memory.cards import Card
from gigaevo.memory_v2.features import FeatureConfig, HierarchicalFeatureMap
from gigaevo.memory_v2.models import (
    BehaviorCoordinate,
    CardSnapshot,
    CausalObservation,
    EnvironmentFingerprint,
    EvolutionContext,
    LLMFingerprint,
    MapElitesContext,
    OutcomeMeasurement,
    RewardDefinition,
)
from gigaevo.memory_v2.posterior import (
    HierarchicalTerminalUtilityPosterior,
    TerminalUtilityPosteriorConfig,
)


@pytest.fixture
def environment() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        task_key="task",
        problem_name="task",
        llm=LLMFingerprint(
            model_name="model",
            base_url="http://localhost/v1",
            temperature=0.6,
        ),
        mutation_operator=LLMMutationOperator,
        program_format="json_document",
        pipeline="memory_guided_noise",
        algorithm="chains_bd3d",
    )


@pytest.fixture
def evolution_context(environment: EnvironmentFingerprint) -> EvolutionContext:
    coordinates = tuple(
        BehaviorCoordinate(
            key=key,
            raw_value=raw,
            semantic_normalized=normalized,
            dynamic_normalized=normalized,
            cell_index=cell,
            num_bins=bins,
            dynamic_lower_bound=0.0,
            dynamic_upper_bound=upper,
        )
        for key, raw, normalized, cell, bins, upper in (
            ("hop_depth", 2.0, 0.4, 2, 5, 5.0),
            ("passages_fetched", 10.0, 0.5, 1, 5, 45.0),
            ("instr_chars", 500.0, 0.7, 1, 6, 2500.0),
        )
    )
    return EvolutionContext(
        run_id="run-1",
        environment=environment,
        parent_id=str(uuid.uuid4()),
        parent_iteration=20,
        parent_generation=5,
        parent_metrics={
            "fitness": 0.5,
            "is_valid": 1.0,
            "hop_depth": 2.0,
            "passages_fetched": 10.0,
            "instr_chars": 500.0,
        },
        reward=RewardDefinition(
            primary_metric="fitness",
            higher_is_better=True,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        ),
        map_elites=MapElitesContext(
            island_id="main",
            strategy_generation=10,
            archive_size=25,
            total_cells=150,
            coverage=1.0 / 6.0,
            parent_quality_quantile=0.5,
            parent_cell=(2, 1, 1),
            parent_cell_occupied=True,
            neighbor_occupancy=0.2,
            coordinates=coordinates,
            semantic_schema_hash="c" * 64,
            behavior_schema_hash="a" * 64,
            archive_fingerprint="b" * 64,
        ),
    )


@pytest.fixture
def revisions() -> tuple[CardSnapshot, CardSnapshot]:
    return (
        CardSnapshot.from_card(
            Card(id="good", task_key="task", description="use the reliable lever")
        ),
        CardSnapshot.from_card(
            Card(id="bad", task_key="task", description="use the unsafe lever")
        ),
    )


@pytest.fixture
def posterior_model() -> HierarchicalTerminalUtilityPosterior:
    return HierarchicalTerminalUtilityPosterior(
        feature_map=HierarchicalFeatureMap(
            config=FeatureConfig(
                behavior_keys=("hop_depth", "passages_fetched", "instr_chars"),
            )
        ),
        config=TerminalUtilityPosteriorConfig(),
    )


def synthetic_observations(
    context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    *,
    per_arm: int = 60,
    seed: int = 7,
) -> tuple[CausalObservation, ...]:
    rng = np.random.default_rng(seed)
    rows: list[CausalObservation] = []
    ordinal = 0
    good, bad = revisions
    for card in revisions:
        for treatment in (False, True):
            for _ in range(per_arm):
                invalid_probability = (
                    (0.82 if treatment else 0.04) if card is bad else 0.04
                )
                invalid = bool(rng.random() < invalid_probability)
                effect = 0.20 if card is good else -0.10
                value = float(rng.normal(effect if treatment else 0.0, 0.08))
                rows.append(
                    CausalObservation(
                        decision_id=f"decision-{ordinal}",
                        event_ordinal=ordinal,
                        card=card,
                        context=context,
                        treatment=treatment,
                        offer_propensity=0.5,
                        proposal_propensity=0.5,
                        joint_action_propensity=0.25,
                        status="invalid" if invalid else "outcome",
                        measurement=(
                            None
                            if invalid
                            else OutcomeMeasurement(value=value, se=None, kind="scalar")
                        ),
                        reward_q_hat_control=0.0,
                        reward_q_hat_treated=0.0,
                        risk_q_hat_control=0.05,
                        risk_q_hat_treated=0.05,
                    )
                )
                ordinal += 1
    return tuple(rows)
