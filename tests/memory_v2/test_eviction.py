from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.eviction import CausalRetirementEvictor
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvidenceSnapshot,
    EvolutionContext,
    OutcomeMeasurement,
    RagApplicability,
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
    lineage_reward = SimpleNamespace(
        optimizer_success=True,
        hyperparameters_at_boundary=False,
    )
    safety_integration_tolerance = 1e-8

    def __init__(self, viability) -> None:
        self.viability = viability
        self.applicability_states: list[RagApplicability] = []
        self.space = SimpleNamespace(
            context_features=lambda context: (float(context.parent_iteration),)
        )

    def prediction(self, *_args, **kwargs):
        applicability = kwargs["rag_applicability"]
        self.applicability_states.append(applicability)
        probability = (
            self.viability[applicability]
            if isinstance(self.viability, dict)
            else self.viability
        )
        return SimpleNamespace(
            probability_safe_and_helpful=probability,
            safety_integration_error=0.0,
        )


class _Posterior:
    def __init__(self, viability) -> None:
        self.fitted = _FittedPosterior(viability)

    def fit(self, *_args, **_kwargs) -> _FittedPosterior:
        return self.fitted


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


def _evictor(
    ledger: _Ledger,
    viability,
) -> tuple[CausalRetirementEvictor, _Posterior]:
    posterior = _Posterior(viability)
    return (
        CausalRetirementEvictor(
            ledger=ledger,  # type: ignore[arg-type]
            posterior=posterior,  # type: ignore[arg-type]
            safety=SafetyConstraint(),
            posterior_samples=256,
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


def test_verdict_fails_closed_when_evidence_or_card_revision_changes(
    evolution_context: EvolutionContext,
) -> None:
    card = Card(id="bad", task_key="task", description="bad treatment")
    revision = CardSnapshot.from_card(card)
    ledger = _Ledger(_evidence(revision, evolution_context))
    evictor, _ = _evictor(ledger, viability=0.0)

    assert evictor.sweep((card,)) == [card.id]
    ledger.evidence = ledger.evidence.model_copy(update={"version": "changed"})
    assert not evictor.should_evict(card)

    ledger.evidence = _evidence(revision, evolution_context, version="fresh")
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
