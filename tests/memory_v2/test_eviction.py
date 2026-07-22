from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.embedding import CardEmbedder
from gigaevo.memory_v2.eviction import CausalRetirementEvictor
from gigaevo.memory_v2.features import EmbeddingPriorConfig
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvidenceSnapshot,
    EvolutionContext,
    OutcomeMeasurement,
    RagApplicability,
)
from gigaevo.memory_v2.policy import SafetyConstraint
from gigaevo.memory_v2.posterior import PosteriorFitError


@dataclass
class _Ledger:
    evidence: EvidenceSnapshot

    def snapshot(self) -> EvidenceSnapshot:
        return self.evidence


class _FittedPosterior:
    reward = SimpleNamespace(
        optimizer_success=True,
        hyperparameters_at_boundary=False,
        residual_boundary_probability=0.0,
    )
    lineage_reward = SimpleNamespace(
        optimizer_success=True,
        hyperparameters_at_boundary=False,
        residual_boundary_probability=0.0,
    )
    safety_integration_tolerance = 1e-8

    def __init__(self, viability) -> None:
        self.viability = viability
        self.applicability_states: list[RagApplicability] = []
        self.context_fitnesses: list[float] = []
        self.minimum_helpful_effects: list[float] = []
        self.safety_integration_error = 0.0
        self.prediction_error: Exception | None = None

    def prediction(self, *_args, **kwargs):
        if self.prediction_error is not None:
            raise self.prediction_error
        applicability = kwargs["rag_applicability"]
        context = _args[1]
        self.applicability_states.append(applicability)
        self.context_fitnesses.append(
            context.parent_metrics[context.reward.primary_metric]
        )
        self.minimum_helpful_effects.append(kwargs["minimum_helpful_effect"])
        probability = (
            self.viability(context, applicability)
            if callable(self.viability)
            else (
                self.viability[applicability]
                if isinstance(self.viability, dict)
                else self.viability
            )
        )
        return SimpleNamespace(
            probability_safe_and_helpful=probability,
            safety_integration_error=self.safety_integration_error,
        )


class _Posterior:
    def __init__(self, viability) -> None:
        self.fitted = _FittedPosterior(viability)
        self.error: Exception | None = None
        self.fit_kwargs: dict = {}

    def fit(self, *_args, **_kwargs) -> _FittedPosterior:
        self.fit_kwargs = _kwargs
        if self.error is not None:
            raise self.error
        return self.fitted


class _ConstantEmbedder(CardEmbedder):
    """Deterministic fake embedder for retirement-seam tests."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts) -> np.ndarray:
        return np.ones((len(list(texts)), self._dimension), dtype=float)


def _observation(
    card: CardSnapshot,
    context: EvolutionContext,
    *,
    ordinal: int,
    treated: bool,
) -> CausalObservation:
    return CausalObservation(
        decision_id=f"observation-{ordinal}",
        event_ordinal=ordinal,
        card=card,
        context=context.model_copy(
            update={
                "parent_iteration": context.parent_iteration + ordinal,
                "map_elites": context.map_elites.model_copy(
                    update={
                        "island_id": (f"{context.map_elites.island_id}-{ordinal // 2}")
                    }
                ),
            }
        ),
        treatment=treated,
        card_used=treated,
        offer_propensity=0.5,
        proposal_propensity=1.0,
        joint_action_propensity=0.5,
        status="outcome",
        measurement=OutcomeMeasurement(value=-0.1, se=None, kind="scalar"),
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.0,
        risk_q_hat_control=0.05,
        risk_q_hat_treated=0.05,
    )


def _evidence(
    revision: CardSnapshot,
    context: EvolutionContext,
    *,
    version: str = "evidence",
    pending: bool = False,
) -> EvidenceSnapshot:
    observations = tuple(
        _observation(
            revision,
            context,
            ordinal=index,
            treated=index % 2 == 0,
        )
        for index in range(4)
    )
    return EvidenceSnapshot(
        version=version,
        model_version=f"{version}-model",
        observations=observations,
        lineage_pending_by_bank_card=({revision.bank_card_id: 1} if pending else {}),
    )


def _with_first_map_cell_fitness(
    evidence: EvidenceSnapshot,
    fitness: float,
) -> EvidenceSnapshot:
    observations = tuple(
        (
            row.model_copy(
                update={
                    "context": row.context.model_copy(
                        update={
                            "parent_metrics": {
                                **row.context.parent_metrics,
                                row.context.reward.primary_metric: fitness,
                            }
                        }
                    )
                }
            )
            if row.event_ordinal < 2
            else row
        )
        for row in evidence.observations
    )
    return evidence.model_copy(update={"observations": observations})


def _evictor(
    ledger: _Ledger,
    viability,
    **overrides,
) -> tuple[CausalRetirementEvictor, _Posterior]:
    posterior = _Posterior(viability)
    return (
        CausalRetirementEvictor(
            ledger=ledger,  # type: ignore[arg-type]
            posterior=posterior,  # type: ignore[arg-type]
            safety=SafetyConstraint(),
            posterior_samples=256,
            **overrides,
        ),
        posterior,
    )


def test_supported_nonviable_revision_gets_one_shot_verdict(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="bad", task_key="task", description="bad treatment")
    ledger = _Ledger(_evidence(CardSnapshot.from_card(card), evolution_context))
    evictor, _ = _evictor(ledger, viability=0.0)

    assert evictor.sweep((card,)) == [card.id]
    assert evictor.should_evict(card)
    assert not evictor.should_evict(card)


def test_embedding_prior_without_an_embedder_fails_construction_loudly(
    evolution_context: EvolutionContext,
) -> None:
    # The read (provider) and write (retirement) seams share one feature config.
    # If the prior is on but the evictor is handed no embedder, its fit would
    # raise and be swallowed as "fit failed closed", silently disabling the whole
    # retirement subsystem. Fail loud at construction instead.
    ledger = _Ledger(
        _evidence(
            CardSnapshot.from_card(Card(id="bad", task_key="task", description="x")),
            evolution_context,
        )
    )
    with pytest.raises(ValueError):
        _evictor(
            ledger,
            viability=0.0,
            embedding_prior=EmbeddingPriorConfig(raw_dimension=8, dimension=4),
        )


def test_embedding_prior_supplies_projected_features_to_the_retirement_fit(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="bad", task_key="task", description="bad treatment")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))
    evictor, posterior = _evictor(
        ledger,
        viability=0.0,
        embedding_prior=EmbeddingPriorConfig(raw_dimension=8, dimension=4),
        card_embedder=_ConstantEmbedder(dimension=8),
    )

    evictor.sweep((card,))

    embeddings = posterior.fit_kwargs["card_embeddings"]
    assert embeddings is not None
    assert revision.bank_card_id in embeddings
    assert embeddings[revision.bank_card_id].shape == (4,)


def test_disabled_prior_passes_no_embeddings_to_the_retirement_fit(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="bad", task_key="task", description="bad treatment")
    ledger = _Ledger(_evidence(CardSnapshot.from_card(card), evolution_context))
    evictor, posterior = _evictor(ledger, viability=0.0)

    evictor.sweep((card,))

    assert posterior.fit_kwargs["card_embeddings"] is None


def test_verdict_ignores_audit_only_change_but_revalidates_model_and_card(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="bad", task_key="task", description="bad treatment")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))
    evictor, _ = _evictor(ledger, viability=0.0)

    assert evictor.sweep((card,)) == [card.id]
    ledger.evidence = ledger.evidence.model_copy(update={"version": "changed"})
    assert evictor.should_evict(card)

    ledger.evidence = _evidence(revision, evolution_context, version="fresh")
    assert evictor.sweep((card,)) == [card.id]
    ledger.evidence = ledger.evidence.model_copy(
        update={"model_version": "changed-model"}
    )
    assert not evictor.should_evict(card)

    ledger.evidence = _evidence(revision, evolution_context, version="pending-sweep")
    assert evictor.sweep((card,)) == [card.id]
    ledger.evidence = ledger.evidence.model_copy(
        update={"pending_by_bank_card": {card.id: 1}}
    )
    assert not evictor.should_evict(card)

    ledger.evidence = _evidence(revision, evolution_context, version="new-sweep")
    assert evictor.sweep((card,)) == [card.id]
    changed = card.model_copy(update={"description": "changed treatment"})
    assert not evictor.should_evict(changed)


def test_pending_or_optimistically_viable_card_is_retained(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="conditional treatment")
    revision = CardSnapshot.from_card(card)

    pending_ledger = _Ledger(_evidence(revision, evolution_context, pending=True))
    pending_evictor, _ = _evictor(pending_ledger, viability=0.0)
    assert pending_evictor.sweep((card,)) == []

    viable_ledger = _Ledger(_evidence(revision, evolution_context))
    viable_evictor, posterior = _evictor(
        viable_ledger,
        viability={
            RagApplicability.UNASSESSED: 0.0,
            RagApplicability.APPLICABLE: 0.2,
        },
    )
    assert viable_evictor.sweep((card,)) == []
    assert posterior.fitted.applicability_states == [
        RagApplicability.UNASSESSED,
        RagApplicability.APPLICABLE,
    ]
    assert posterior.fitted.minimum_helpful_effects == [0.1, 0.1]


def test_unhealthy_lineage_posterior_vetoes_retirement(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="conditional treatment")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))
    evictor, posterior = _evictor(ledger, viability=0.0)
    posterior.fitted.lineage_reward = SimpleNamespace(
        optimizer_success=False,
        hyperparameters_at_boundary=False,
        residual_boundary_probability=0.0,
    )

    assert evictor.sweep((card,)) == []


def test_absorbed_lineage_evidence_supports_survivor_retirement(
    evolution_context: EvolutionContext,
) -> None:
    old = Card(id="old", task_key="task", description="same treatment")
    survivor = Card(
        id="survivor",
        task_key="task",
        description="same treatment",
        absorbed_ids=(old.id,),
    )
    ledger = _Ledger(_evidence(CardSnapshot.from_card(old), evolution_context))
    evictor, _ = _evictor(ledger, viability=0.0)

    assert evictor.sweep((survivor,)) == [survivor.id]


def test_support_gates_block_below_treated_or_control_minimum(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    complete = _evidence(revision, evolution_context)

    one_treated = complete.model_copy(
        update={
            "observations": tuple(
                row for row in complete.observations if row.event_ordinal != 2
            )
        }
    )
    one_control = complete.model_copy(
        update={
            "observations": tuple(
                row for row in complete.observations if row.event_ordinal != 3
            )
        }
    )

    assert (
        _evictor(
            _Ledger(one_treated),
            viability=0.0,
            min_distinct_contexts=1,
        )[0].sweep((card,))
        == []
    )
    assert _evictor(_Ledger(one_control), viability=0.0)[0].sweep((card,)) == []


def test_uncited_treatments_supply_retirement_support(
    evolution_context: EvolutionContext,
) -> None:
    """Delivered-but-uncited randomized evidence retires a card that never earns
    citations, instead of leaving it immortal (intention-to-treat support)."""

    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    observations = tuple(
        row.model_copy(update={"card_used": False}) if row.treatment else row
        for row in evidence.observations
    )
    evictor, _ = _evictor(
        _Ledger(evidence.model_copy(update={"observations": observations})),
        viability=0.0,
        min_distinct_contexts=1,
    )

    assert evictor.sweep((card,)) == [card.id]


def test_iteration_changes_inside_one_map_cell_are_not_distinct_contexts(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    same_cell = tuple(
        row.model_copy(
            update={
                "context": row.context.model_copy(
                    update={"map_elites": evolution_context.map_elites}
                )
            }
        )
        for row in evidence.observations
    )
    ledger = _Ledger(evidence.model_copy(update={"observations": same_cell}))

    assert _evictor(ledger, viability=0.0)[0].sweep((card,)) == []


def test_fit_and_prediction_failures_keep_cards(
    evolution_context: EvolutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "gigaevo.memory_v2.eviction.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))

    fit_evictor, fit_posterior = _evictor(ledger, viability=0.0)
    fit_posterior.error = PosteriorFitError("bad fit")
    assert fit_evictor.sweep((card,)) == []

    integration_evictor, integration_posterior = _evictor(ledger, viability=0.0)
    integration_posterior.fitted.safety_integration_error = 1.0
    assert integration_evictor.sweep((card,)) == []
    assert any("safety integration failed closed" in warning for warning in warnings)

    prediction_evictor, prediction_posterior = _evictor(ledger, viability=0.0)
    prediction_posterior.fitted.prediction_error = PosteriorFitError("bad prediction")
    assert prediction_evictor.sweep((card,)) == []


def test_practical_threshold_tracks_randomized_control_gain_scale(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    values = (0.001, -0.002, 0.004, -0.008)
    observations = tuple(
        row.model_copy(
            update={"measurement": row.measurement.model_copy(update={"value": value})}
        )
        for row, value in zip(evidence.observations, values, strict=True)
        if row.measurement is not None
    )
    evictor, _ = _evictor(
        _Ledger(evidence.model_copy(update={"observations": observations})),
        viability=0.0,
    )

    assert evictor._practical_effect_threshold(observations) == pytest.approx(0.0026)


@pytest.mark.parametrize(
    ("control_values", "min_global_control", "expected"),
    (
        ((0.0, 0.0), 1, 0.0),
        ((0.0, 0.008), 1, 0.008),
        ((0.0, 0.008), 2, 0.0),
    ),
)
def test_practical_threshold_requires_enough_nonzero_measured_controls(
    evolution_context: EvolutionContext,
    control_values: tuple[float, float],
    min_global_control: int,
    expected: float,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    controls = iter(control_values)
    observations = [
        (
            row.model_copy(
                update={
                    "measurement": row.measurement.model_copy(
                        update={"value": next(controls)}
                    )
                }
            )
            if not row.treatment and row.measurement is not None
            else row
        )
        for row in evidence.observations
    ]
    invalid_control = next(row for row in evidence.observations if not row.treatment)
    observations.append(
        invalid_control.model_copy(
            update={
                "decision_id": "invalid-control",
                "event_ordinal": 99,
                "status": "invalid",
                "measurement": None,
            }
        )
    )
    evictor, _ = _evictor(
        _Ledger(evidence.model_copy(update={"observations": tuple(observations)})),
        viability=0.0,
        min_global_control=min_global_control,
    )

    assert evictor._practical_effect_threshold(observations) == pytest.approx(expected)


def test_headroom_clipped_context_can_still_vote_to_keep(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _with_first_map_cell_fitness(
        _evidence(revision, evolution_context),
        0.95,
    )
    ledger = _Ledger(evidence)
    evictor, posterior = _evictor(
        ledger,
        viability=lambda context, _applicability: (
            0.2 if context.parent_metrics["fitness"] > 0.9 else 0.0
        ),
        min_distinct_contexts=1,
    )

    assert evictor.sweep((card,)) == []
    assert posterior.fitted.context_fitnesses == [0.95]


def test_distinct_context_support_counts_only_assessable_contexts(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _with_first_map_cell_fitness(
        _evidence(revision, evolution_context),
        0.95,
    )
    ledger = _Ledger(evidence)

    assert _evictor(ledger, viability=0.0)[0].sweep((card,)) == []
    assert _evictor(
        ledger,
        viability=0.0,
        min_distinct_contexts=1,
    )[0].sweep((card,)) == [card.id]


def test_context_without_practical_headroom_cannot_certify_but_can_keep(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    near_optimum = evolution_context.model_copy(
        update={
            "parent_metrics": {
                **evolution_context.parent_metrics,
                "fitness": 0.95,
            }
        }
    )
    ledger = _Ledger(_evidence(revision, near_optimum))
    evictor, posterior = _evictor(ledger, viability=0.0)

    assert evictor.sweep((card,)) == []
    assert posterior.fitted.applicability_states == []


def test_invalid_controls_do_not_supply_reward_retirement_support(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    observations = tuple(
        (
            row.model_copy(update={"status": "invalid", "measurement": None})
            if not row.treatment
            else row
        )
        for row in evidence.observations
    )
    ledger = _Ledger(evidence.model_copy(update={"observations": observations}))
    evictor, posterior = _evictor(ledger, viability=0.0)

    assert evictor.sweep((card,)) == []
    assert posterior.fitted.applicability_states == []


def test_both_reward_heads_and_residual_boundaries_veto_retirement(
    evolution_context: EvolutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "gigaevo.memory_v2.eviction.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))

    unhealthy_evictor, unhealthy = _evictor(ledger, viability=0.0)
    unhealthy.fitted.reward = SimpleNamespace(
        optimizer_success=False,
        hyperparameters_at_boundary=False,
        residual_boundary_probability=0.0,
    )
    assert unhealthy_evictor.sweep((card,)) == []

    boundary_evictor, boundary = _evictor(ledger, viability=0.0)
    boundary.fitted.reward = SimpleNamespace(
        optimizer_success=True,
        hyperparameters_at_boundary=False,
        residual_boundary_probability=0.2,
    )
    assert boundary_evictor.sweep((card,)) == []
    assert any(
        "memory.posterior_config.reward_residual_sd_bounds" in warning
        for warning in warnings
    )


def test_wilson_upper_bound_prevents_borderline_mc_retirement(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))

    evictor, _ = _evictor(ledger, viability=0.07)

    assert evictor.sweep((card,)) == []


def test_real_posterior_retirement_smoke(
    evolution_context: EvolutionContext,
    posterior_model,
) -> None:
    card = Card(id="card", task_key="task", description="candidate")
    revision = CardSnapshot.from_card(card)
    evidence = _evidence(revision, evolution_context)
    fitted = posterior_model.fit(
        evidence.observations,
        (revision,),
        lineage_observations=evidence.lineage_observations,
    )

    class FixedPosterior:
        def fit(self, *_args, **_kwargs):
            return fitted

    evictor = CausalRetirementEvictor(
        ledger=_Ledger(evidence),
        posterior=FixedPosterior(),
        safety=SafetyConstraint(),
        posterior_samples=256,
    )

    assert set(evictor.sweep((card,))) <= {card.id}


def test_real_posterior_retires_confidently_harmful_card(
    evolution_context: EvolutionContext,
    posterior_model,
) -> None:
    card = Card(id="harmful", task_key="task", description="harmful candidate")
    revision = CardSnapshot.from_card(card)
    noise = (-0.04, -0.02, 0.02, 0.04)
    observations = tuple(
        _observation(
            revision,
            evolution_context,
            ordinal=index,
            treated=index % 2 == 0,
        ).model_copy(
            update={
                "card_used": index % 4 == 0,
                "measurement": OutcomeMeasurement(
                    value=(-0.20 if index % 2 == 0 else 0.0) + noise[index % 4],
                    se=0.0,
                    kind="deterministic",
                ),
            }
        )
        for index in range(80)
    )
    evidence = EvidenceSnapshot(
        version="harmful-evidence",
        model_version="harmful-model",
        observations=observations,
    )

    evictor = CausalRetirementEvictor(
        ledger=_Ledger(evidence),
        posterior=posterior_model,
        safety=SafetyConstraint(),
        posterior_samples=4096,
    )

    assert evictor.sweep((card,)) == [card.id]
