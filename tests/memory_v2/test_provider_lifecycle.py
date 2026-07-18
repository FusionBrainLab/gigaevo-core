from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import ResearchResult
from gigaevo.memory_v2.candidates import (
    AgenticApplicabilityProvider,
    WholeBankCandidateSource,
)
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


class _MultiStore:
    def __init__(self, cards: Sequence[Card]) -> None:
        self.cards = {card.id: card for card in cards}

    def get(self, card_id: str) -> Card | None:
        return self.cards.get(card_id)

    def snapshot(self) -> tuple[Card, ...]:
        return tuple(self.cards.values())


class _ContextSource:
    def __init__(self, context: EvolutionContext) -> None:
        self.context = context

    async def snapshot(self, program: Program) -> EvolutionContext:
        assert program.id == self.context.parent_id
        return self.context


class _PerParentContextSource:
    def __init__(self, context: EvolutionContext) -> None:
        self.context = context

    async def snapshot(self, program: Program) -> EvolutionContext:
        return self.context.model_copy(
            update={
                "parent_id": program.id,
                "parent_iteration": program.iteration,
            }
        )


class _Shortlister:
    def __init__(self, card: Card) -> None:
        self.card = card

    async def shortlist(self, **kwargs) -> ResearchResult:
        del kwargs
        return ResearchResult(cards=(self.card,), iterations=1)


class _LeaseBumpingCandidateSource:
    def __init__(self, source: WholeBankCandidateSource, contender) -> None:
        self.source = source
        self.contender = contender
        self.calls = 0

    @property
    def specification(self):
        return self.source.specification

    @property
    def applicability_specification(self):
        return self.source.applicability_specification

    async def prepare(self, *args, **kwargs):
        return await self.source.prepare(*args, **kwargs)

    async def candidate_snapshot(self, *args, **kwargs):
        slate = await self.source.candidate_snapshot(*args, **kwargs)
        self.calls += 1
        if self.calls == 1:
            self.contender.attach_cards(("foreign-card",))
        return slate


class _BarrierCandidateSource:
    def __init__(self, source: WholeBankCandidateSource) -> None:
        self.source = source
        self.started = 0
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def specification(self):
        return self.source.specification

    @property
    def applicability_specification(self):
        return self.source.applicability_specification

    async def prepare(self, *args, **kwargs):
        self.started += 1
        if self.started == 2:
            self.both_started.set()
        await self.release.wait()
        return await self.source.prepare(*args, **kwargs)

    async def candidate_snapshot(self, *args, **kwargs):
        return await self.source.candidate_snapshot(*args, **kwargs)


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
        assessed_bank_card_ids: frozenset[str] = frozenset(),
        applicable_bank_card_ids: frozenset[str] = frozenset(),
        pending_by_bank_card: Mapping[str, int] | None = None,
        lineage_pending_by_bank_card: Mapping[str, int] | None = None,
    ) -> PolicyDecision:
        del (
            posterior,
            context,
            rng,
            assessed_bank_card_ids,
            applicable_bank_card_ids,
            pending_by_bank_card,
            lineage_pending_by_bank_card,
        )
        card = candidates[0]
        rows = tuple(
            CandidateActionProbability(
                treatment_id=candidate.treatment_id,
                bank_card_id=candidate.bank_card_id,
                proposal_probability=1.0 if candidate == card else 0.0,
                proposal_mc_se=0.0,
                offer_probability=0.5,
                joint_treated_probability=0.5 if candidate == card else 0.0,
                joint_control_probability=0.5 if candidate == card else 0.0,
                safe=True,
                prediction=prediction(candidate),
            )
            for candidate in candidates
        )
        return PolicyDecision(
            proposed_card=card,
            delivered=True,
            offer_probability=0.5,
            proposal_probability=1.0,
            joint_action_probability=0.5,
            action_probabilities=rows,
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


class _AlwaysAbstainPolicy(_AlwaysTreatPolicy):
    def choose(self, **kwargs) -> PolicyDecision:
        candidates = tuple(kwargs["candidates"])
        return PolicyDecision(
            action_probabilities=tuple(
                CandidateActionProbability(
                    treatment_id=card.treatment_id,
                    bank_card_id=card.bank_card_id,
                    proposal_probability=0.0,
                    proposal_mc_se=0.0,
                    offer_probability=0.5,
                    joint_treated_probability=0.0,
                    joint_control_probability=0.0,
                    safe=True,
                    prediction=prediction(card),
                )
                for card in candidates
            ),
            abstain_probability=1.0,
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
        candidate_source=WholeBankCandidateSource(
            store=store,  # type: ignore[arg-type]
            applicability=AgenticApplicabilityProvider(shortlister=_Shortlister(card)),
        ),
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
    assert all(
        row.candidate_universe.status == "eligible_bank" for row in ledger.decisions()
    )
    assert all(
        row.applicability.applicable_bank_card_ids == (card.id,)
        for row in ledger.decisions()
    )


@pytest.mark.asyncio
async def test_provider_runs_slow_candidate_preparation_concurrently(
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    card = Card(id="card", task_key="task", description="shared treatment")
    store = _Store(card)
    registry = InFlightSelectionRegistry()
    source = _BarrierCandidateSource(
        WholeBankCandidateSource(store=store)  # type: ignore[arg-type]
    )
    ledger = SqliteCausalLedger(
        path=tmp_path / "concurrent-retrieval.sqlite3", environment=environment
    )
    ledger.activate()
    provider = CausalBanditMemoryProvider(
        candidate_source=source,  # type: ignore[arg-type]
        context_source=_PerParentContextSource(evolution_context),  # type: ignore[arg-type]
        ledger=ledger,
        posterior=posterior_model,
        policy=_AlwaysTreatPolicy(),  # type: ignore[arg-type]
        renderer=ImmutableCardRenderer(),
        store=store,  # type: ignore[arg-type]
        selection_leases=registry,
        task_key="task",
        run_seed=9,
    )
    parents = (
        Program(id=str(uuid4()), code="def first(): return 1", iteration=1),
        Program(id=str(uuid4()), code="def second(): return 2", iteration=2),
    )
    leases = tuple(
        registry.open_attempt(f"attempt-{index}", parent.id)
        for index, parent in enumerate(parents)
    )

    with registry.activate_attempt("attempt-0", (parents[0].id,)):
        with registry.activate_attempt("attempt-1", (parents[1].id,)):
            selections = tuple(
                asyncio.create_task(
                    provider.select_cards(
                        parent,
                        task_description="task",
                        metrics_description="fitness",
                    )
                )
                for parent in parents
            )
            await asyncio.wait_for(source.both_started.wait(), timeout=1.0)
            source.release.set()
            results = await asyncio.gather(*selections)

    assert source.started == 2
    assert all(result.card_ids == (card.id,) for result in results)
    assert [row.event_ordinal for row in ledger.decisions()] == [0, 1]
    for lease in leases:
        lease.release()


@pytest.mark.asyncio
async def test_provider_leases_full_slate_through_commit_then_retains_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    cards = (
        Card(id="card-a", task_key="task", description="first treatment"),
        Card(id="card-b", task_key="task", description="second treatment"),
    )
    store = _MultiStore(cards)
    registry = InFlightSelectionRegistry()
    ledger = SqliteCausalLedger(
        path=tmp_path / "full-slate.sqlite3", environment=environment
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
    lease = registry.open_attempt("full-slate-attempt", parent.id)
    original_commit = provider._commit

    def checked_commit(**kwargs):
        assert registry.leased_ids() == frozenset(card.id for card in cards)
        return original_commit(**kwargs)

    monkeypatch.setattr(provider, "_commit", checked_commit)
    with registry.activate_attempt("full-slate-attempt", (parent.id,)):
        selection = await provider.select_cards(
            parent, task_description="task", metrics_description="fitness"
        )

    record = ledger.decisions()[0]
    proposed = next(
        row
        for row in record.action_probabilities
        if row.treatment_id == record.proposed_treatment_id
    )
    assert selection.card_ids == (proposed.bank_card_id,)
    assert registry.leased_ids() == frozenset(selection.card_ids)
    lease.release()


def test_provider_rejects_changed_card_revision_during_atomic_reservation(
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    card = Card(id="card", task_key="task", description="frozen treatment")
    store = _Store(card)
    registry = InFlightSelectionRegistry()
    ledger = SqliteCausalLedger(
        path=tmp_path / "changed-slate.sqlite3", environment=environment
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
    parent = Program(id=evolution_context.parent_id, code="parent")
    lease = registry.open_attempt("changed-slate-attempt", parent.id)
    frozen = CardSnapshot.from_card(card)

    with registry.activate_attempt("changed-slate-attempt", (parent.id,)):
        version = registry.selection_snapshot().version
        store.card = card.model_copy(update={"description": "changed treatment"})
        assert not provider._reserve_candidate_slate(
            parent,
            attempt_id="changed-slate-attempt",
            expected_lease_version=version,
            candidates=(frozen,),
        )

    assert registry.leased_ids() == frozenset()
    lease.release()


@pytest.mark.asyncio
async def test_abstention_retries_after_stale_lease_version(
    tmp_path,
    environment,
    evolution_context: EvolutionContext,
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    card = Card(id="card", task_key="task", description="candidate treatment")
    store = _Store(card)
    registry = InFlightSelectionRegistry()
    contender = registry.open_attempt("contender", "other-parent")
    source = _LeaseBumpingCandidateSource(
        WholeBankCandidateSource(store=store),  # type: ignore[arg-type]
        contender,
    )
    ledger = SqliteCausalLedger(
        path=tmp_path / "stale-abstain.sqlite3", environment=environment
    )
    ledger.activate()
    provider = CausalBanditMemoryProvider(
        candidate_source=source,  # type: ignore[arg-type]
        context_source=_ContextSource(evolution_context),  # type: ignore[arg-type]
        ledger=ledger,
        posterior=posterior_model,
        policy=_AlwaysAbstainPolicy(),  # type: ignore[arg-type]
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
    lease = registry.open_attempt("abstain-attempt", parent.id)

    with registry.activate_attempt("abstain-attempt", (parent.id,)):
        selection = await provider.select_cards(
            parent, task_description="task", metrics_description="fitness"
        )

    assert source.calls == 2
    assert selection.cards == ()
    assert len(ledger.decisions()) == 1
    assert ledger.decisions()[0].proposed_treatment_id is None
    assert registry.leased_ids() == frozenset({"foreign-card"})
    lease.release()
    contender.release()


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
    gate.sweep.return_value = ["retired-card"]
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
    assert events[0].retired_card_ids == ("retired-card",)


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
