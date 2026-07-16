from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory_v2.candidates import WholeBankCandidateSource
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import (
    CandidateActionProbability,
    CardSnapshot,
    EvolutionContext,
    PolicyDecision,
    PolicySpecification,
)
from gigaevo.memory_v2.policy import ProbabilityMatchingConfig
from gigaevo.memory_v2.posterior import HierarchicalTerminalUtilityPosterior
from gigaevo.memory_v2.provider import CausalBanditMemoryProvider
from gigaevo.memory_v2.render import ImmutableCardRenderer
from gigaevo.memory_v2.rng import EventRNG
from gigaevo.memory_v2.writer import CausalV2ContentOnlyUpdater
from gigaevo.programs.program import Program

from .factories import prediction


class _Store:
    def __init__(self, card: Card) -> None:
        self.card = card

    def get(self, card_id: str) -> Card | None:
        return self.card if card_id == self.card.id else None

    def snapshot(self) -> tuple[Card, ...]:
        return (self.card,)


class _ContextSource:
    def __init__(self, context: EvolutionContext) -> None:
        self.context = context

    async def snapshot(self, program: Program) -> EvolutionContext:
        assert program.id == self.context.parent_id
        return self.context


class _AlwaysTreatPolicy:
    def __init__(self) -> None:
        self.config = ProbabilityMatchingConfig(
            offer_probability=0.50,
            proposal_exploration_probability=0.0,
            posterior_summary_samples=128,
            proposal_worlds=64,
        )
        self.specification = PolicySpecification(
            safety_gate_mode="credible_joint_safe",
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.10,
            safety_alpha=0.10,
            offer_probability=0.50,
            proposal_exploration_probability=0.0,
            posterior_summary_samples=128,
            proposal_worlds=64,
            abstain_effect=0.0,
            max_pending_per_card=2,
        )

    def eligible_candidates(
        self,
        candidates: Sequence[CardSnapshot],
        *,
        pending_by_bank_card: Mapping[str, int],
    ) -> tuple[CardSnapshot, ...]:
        del pending_by_bank_card
        return tuple(candidates)

    def choose(
        self,
        *,
        posterior,
        candidates: Sequence[CardSnapshot],
        context: EvolutionContext,
        rng: EventRNG,
    ) -> PolicyDecision:
        del posterior, context, rng
        card = candidates[0]
        row = CandidateActionProbability(
            treatment_id=card.treatment_id,
            bank_card_id=card.bank_card_id,
            proposal_probability=1.0,
            proposal_mc_se=0.0,
            offer_probability=0.5,
            joint_treated_probability=0.5,
            joint_control_probability=0.5,
            safe=True,
            prediction=prediction(card),
        )
        return PolicyDecision(
            proposed_card=card,
            delivered=True,
            offer_probability=0.5,
            proposal_probability=1.0,
            joint_action_probability=0.5,
            action_probabilities=(row,),
            abstain_probability=0.0,
        )


class _AlwaysControlPolicy(_AlwaysTreatPolicy):
    def choose(self, **kwargs) -> PolicyDecision:
        treated = super().choose(**kwargs)
        return PolicyDecision(
            proposed_card=treated.proposed_card,
            delivered=False,
            offer_probability=treated.offer_probability,
            proposal_probability=treated.proposal_probability,
            joint_action_probability=treated.joint_action_probability,
            action_probabilities=treated.action_probabilities,
            abstain_probability=treated.abstain_probability,
        )


@pytest.mark.asyncio
async def test_provider_creates_one_decision_per_active_mutation_attempt(
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    card = Card(id="card", task_key="task", description="exact treatment")
    store = _Store(card)
    registry = InFlightSelectionRegistry()
    ledger = SqliteCausalLedger(
        path=tmp_path / "provider.sqlite3", environment=environment
    )
    ledger.activate()
    provider = CausalBanditMemoryProvider(
        candidate_source=WholeBankCandidateSource(store=store),  # type: ignore[arg-type]
        context_source=_ContextSource(evolution_context),  # type: ignore[arg-type]
        ledger=ledger,
        posterior=posterior_model,
        policy=_AlwaysTreatPolicy(),  # type: ignore[arg-type]
        renderer=ImmutableCardRenderer(),
        store=store,  # type: ignore[arg-type]
        selection_leases=registry,
        task_key="task",
        run_seed=9,
    )
    parent = Program(
        id=evolution_context.parent_id,
        code="def parent(): return 1",
        iteration=evolution_context.parent_iteration,
    )

    child_dag_selection = await provider.select_cards(
        parent, task_description="task", metrics_description="fitness"
    )
    assert child_dag_selection.decision_id == ""
    assert ledger.decisions() == ()

    decision_ids: list[str] = []
    for attempt_id in ("attempt-a", "attempt-b"):
        lease = registry.open_attempt(attempt_id, parent.id)
        with registry.activate_attempt(attempt_id, (parent.id,)):
            selection = await provider.select_cards(
                parent, task_description="task", metrics_description="fitness"
            )
            assert selection.preformatted
            assert selection.cards == (f"[card 1] id=card\n{card.description}",)
            assert registry.is_leased(card.id)
            decision_ids.append(selection.decision_id)
        lease.release()

    assert len(set(decision_ids)) == 2
    assert [row.attempt_id for row in ledger.decisions()] == [
        "attempt-a",
        "attempt-b",
    ]


def test_v2_writer_releases_leases_without_restamping_card_efficacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment,
) -> None:
    card = Card(id="card", task_key="task", description="unchanged treatment")
    store = _Store(card)
    ledger = SqliteCausalLedger(
        path=tmp_path / "writer.sqlite3", environment=environment
    )
    ledger.activate()
    leases = Mock()
    events: list[object] = []
    monkeypatch.setattr("gigaevo.memory_v2.writer.emit_memory_event", events.append)
    updater = CausalV2ContentOnlyUpdater(
        ledger=ledger,
        selection_leases=leases,
    )
    child_ids = (str(uuid4()), str(uuid4()))
    children = [
        Program(id=child_id, code=code) for child_id, code in zip(child_ids, ("a", "b"))
    ]
    monkeypatch.setattr(
        ledger,
        "terminals",
        lambda: (SimpleNamespace(child_id=child_ids[0]),),
    )

    gate = Mock()
    updater.update(children, store=store, gate=gate)

    assert store.snapshot() == (card,)
    assert [call.args[0] for call in leases.release_child.call_args_list] == [
        child_ids[0]
    ]
    gate.sweep.assert_called_once_with()
    assert len(events) == 1
    assert events[0].evidence_count == 0
    assert events[0].bank_size == 1
    assert events[0].released_child_count == 1


@pytest.mark.asyncio
async def test_withheld_control_reserves_proposed_card_until_child_handoff(
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    card = Card(id="card", task_key="task", description="frozen control treatment")
    store = _Store(card)
    registry = InFlightSelectionRegistry()
    ledger = SqliteCausalLedger(
        path=tmp_path / "control.sqlite3", environment=environment
    )
    ledger.activate()
    provider = CausalBanditMemoryProvider(
        candidate_source=WholeBankCandidateSource(store=store),  # type: ignore[arg-type]
        context_source=_ContextSource(evolution_context),  # type: ignore[arg-type]
        ledger=ledger,
        posterior=posterior_model,
        policy=_AlwaysControlPolicy(),  # type: ignore[arg-type]
        renderer=ImmutableCardRenderer(),
        store=store,  # type: ignore[arg-type]
        selection_leases=registry,
        task_key="task",
        run_seed=9,
    )
    parent = Program(
        id=evolution_context.parent_id,
        code="def parent(): return 1",
        iteration=evolution_context.parent_iteration,
    )
    lease = registry.open_attempt("control-attempt", parent.id)

    with registry.activate_attempt("control-attempt", (parent.id,)):
        selection = await provider.select_cards(
            parent, task_description="task", metrics_description="fitness"
        )
        assert selection.cards == ()
        assert registry.is_leased(card.id)
        with registry.eviction_guard():
            assert registry.is_leased(card.id)

    record = ledger.decisions()[0]
    assert record.proposed_treatment_id is not None
    assert record.delivered is False
    child_id = str(uuid4())
    lease.transfer_to_child(child_id, (card.id,))
    assert registry.is_leased(card.id)
    registry.release_child(child_id)
    assert not registry.is_leased(card.id)
