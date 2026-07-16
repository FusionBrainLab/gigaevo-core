from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.eviction import CausalPosteriorEvictor
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvidenceSnapshot,
    EvolutionContext,
    OutcomeMeasurement,
)
from gigaevo.memory_v2.policy import SafetyConstraint


@dataclass
class _Ledger:
    evidence: EvidenceSnapshot

    def snapshot(self) -> EvidenceSnapshot:
        return self.evidence


class _FittedPosterior:
    reward = SimpleNamespace(
        optimizer_success=True,
        hyperparameters_at_boundary=False,
    )

    def __init__(self, viability: float) -> None:
        self.viability = viability
        self.safety_integration_tolerance = 1e-8
        self.space = SimpleNamespace(
            context_features=lambda context: (float(context.parent_iteration),)
        )

    def prediction(self, *_args, **_kwargs):
        return SimpleNamespace(
            probability_safe_and_helpful=self.viability,
            safety_integration_error=0.0,
        )


class _Posterior:
    def __init__(self, viability: float) -> None:
        self.viability = viability

    def fit(self, *_args, **_kwargs) -> _FittedPosterior:
        return _FittedPosterior(self.viability)


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
            update={"parent_iteration": context.parent_iteration + ordinal}
        ),
        treatment=treated,
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


def _evictor(ledger: _Ledger, viability: float) -> CausalPosteriorEvictor:
    return CausalPosteriorEvictor(
        ledger=ledger,  # type: ignore[arg-type]
        posterior=_Posterior(viability),  # type: ignore[arg-type]
        safety=SafetyConstraint(),
        posterior_samples=4096,
    )


def test_causal_posterior_evictor_uses_current_supported_posterior(
    evolution_context: EvolutionContext,
) -> None:
    bank_card = Card(id="bad", task_key="task", description="bad treatment")
    revision = CardSnapshot.from_card(bank_card)
    observations = tuple(
        _observation(
            revision,
            evolution_context,
            ordinal=index,
            treated=index % 2 == 0,
        )
        for index in range(4)
    )
    ledger = _Ledger(
        EvidenceSnapshot(
            version="evidence-1",
            model_version="model-1",
            observations=observations,
        )
    )
    evictor = _evictor(ledger, viability=0.0)

    assert evictor.sweep((bank_card,)) == [bank_card.id]
    assert evictor.should_evict(bank_card)


def test_causal_posterior_evictor_retains_pending_undertested_or_viable_cards(
    evolution_context: EvolutionContext,
) -> None:
    bank_card = Card(id="card", task_key="task", description="treatment")
    revision = CardSnapshot.from_card(bank_card)
    observations = tuple(
        _observation(
            revision,
            evolution_context,
            ordinal=index,
            treated=index % 2 == 0,
        )
        for index in range(4)
    )

    pending = _Ledger(
        EvidenceSnapshot(
            version="pending",
            model_version="pending-model",
            observations=observations,
            pending_by_bank_card={revision.bank_card_id: 1},
        )
    )
    assert _evictor(pending, viability=0.0).sweep((bank_card,)) == []

    under_tested = _Ledger(
        EvidenceSnapshot(
            version="under-tested",
            model_version="under-tested-model",
            observations=observations[:3],
        )
    )
    assert _evictor(under_tested, viability=0.0).sweep((bank_card,)) == []

    supported = _Ledger(
        EvidenceSnapshot(
            version="viable",
            model_version="viable-model",
            observations=observations,
        )
    )
    assert _evictor(supported, viability=0.2).sweep((bank_card,)) == []


def test_eviction_requires_distinct_modeled_contexts_not_distinct_parent_ids(
    evolution_context: EvolutionContext,
) -> None:
    bank_card = Card(id="card", task_key="task", description="treatment")
    revision = CardSnapshot.from_card(bank_card)
    observations = tuple(
        _observation(
            revision,
            evolution_context,
            ordinal=index,
            treated=index % 2 == 0,
        ).model_copy(
            update={
                "context": evolution_context.model_copy(
                    update={"parent_id": f"parent-{index}", "parent_iteration": 0}
                )
            }
        )
        for index in range(4)
    )
    ledger = _Ledger(
        EvidenceSnapshot(
            version="same-model-context",
            model_version="same-model-context-model",
            observations=observations,
        )
    )

    assert _evictor(ledger, viability=0.0).sweep((bank_card,)) == []


def test_eviction_vetoes_uncertified_safety_integration(
    evolution_context: EvolutionContext,
) -> None:
    bank_card = Card(id="card", task_key="task", description="treatment")
    revision = CardSnapshot.from_card(bank_card)
    observations = tuple(
        _observation(
            revision,
            evolution_context,
            ordinal=index,
            treated=index % 2 == 0,
        )
        for index in range(4)
    )
    ledger = _Ledger(
        EvidenceSnapshot(
            version="integration-failed",
            model_version="integration-failed-model",
            observations=observations,
        )
    )
    evictor = _evictor(ledger, viability=0.0)
    fitted = evictor.posterior.fit(observations, (revision,))
    fitted.prediction = lambda *_args, **_kwargs: SimpleNamespace(
        probability_safe_and_helpful=0.0,
        safety_integration_error=1.0,
    )
    evictor.posterior.fit = lambda *_args, **_kwargs: fitted

    assert evictor.sweep((bank_card,)) == []
