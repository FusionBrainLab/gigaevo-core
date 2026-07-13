"""Cross-task CONSENSUS lifecycle over one real shared memory bank."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
import uuid

import numpy as np
import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import (
    LibrarianCard,
    ReconcileItem,
    ReconcileResponse,
)
from gigaevo.llm.agents.task_summary import TaskSummaryResponse
from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.events import new_decision_id
from gigaevo.memory.read.auction import BootstrapThompsonAuctioneer, TopBidBudgeter
from gigaevo.memory.read.prior import EmpiricalBayesMemoryPrior
from gigaevo.memory.read.probe import ColdProbePolicy
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.render import EfficacyCardRenderer
from gigaevo.memory.read.reputation import (
    BetaBinomialReputation,
    BootstrapReputation,
)
from gigaevo.memory.storage.bank import new_card_id
from gigaevo.memory.storage.base import ResearchRequest
from gigaevo.memory.storage.config import ResearchConfig, StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.storage.research import (
    ScopedQuery,
    SearchPlan,
    ShortlistDecision,
    candidate_brief,
)
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.eviction import HarmEvictor, NullEvictor
from gigaevo.memory.write.merge import ProgramExemplarPolicy
from gigaevo.memory.write.stats import CardStatsUpdater
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import Lineage, Program
from tests.fakes.embedding import FakeEmbeddingFunction
from tests.fakes.llm_router import FakeMemoryRouter


def test_shared_bank_generated_ids_use_full_uuid_entropy():
    card_ids = (new_card_id(), new_card_id())
    decision_ids = (new_decision_id(), new_decision_id())

    assert card_ids[0] != card_ids[1]
    assert decision_ids[0] != decision_ids[1]
    assert all(re.fullmatch(r"mem-[0-9a-f]{32}", card_id) for card_id in card_ids)
    assert all(
        re.fullmatch(r"memsel-[0-9a-f]{32}", decision_id)
        for decision_id in decision_ids
    )


@dataclass(frozen=True)
class SharedBank:
    store: LocalMemoryStore
    router: FakeMemoryRouter
    metrics: MetricsContext
    parent: Program
    authored: Program
    insight_id: str
    exemplar_id: str


def _program(
    number: int,
    *,
    code: str,
    loss: float,
    parents: tuple[str, ...] = (),
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> Program:
    return Program(
        id=str(uuid.UUID(int=number)),
        code=code,
        metrics={VALIDITY_KEY: 1.0, "loss": loss},
        metadata=metadata or {},
        lineage=Lineage(parents=list(parents), generation=2, mutation=None),
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


def _outcome_program(
    number: int,
    *,
    parent: Program,
    loss: float,
    selected_id: str = "",
    task_offset: int = 0,
) -> Program:
    selected = [selected_id] if selected_id else []
    output = {"card_ids_used": selected} if selected else {}
    return _program(
        number,
        code=f"def solve():\n    return {loss}",
        loss=loss,
        parents=(parent.id,),
        metadata={
            MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: selected,
            MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: dict(parent.metrics),
            MUTATION_MEMORY_BASE_ID_METADATA_KEY: parent.id,
            MUTATION_OUTPUT_METADATA_KEY: output,
        },
        created_at=datetime(2026, 1, 2, tzinfo=UTC)
        + timedelta(minutes=number + task_offset),
    )


def _context(task_key: str) -> DecisionContext:
    return DecisionContext(task_key=task_key, parent_metrics={"loss": 10.0})


def _event(task_key: str, gain: float, minute: int) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(
            task_key=task_key,
            parent_metrics={"loss": 10.0},
            timestamp=datetime(2026, 1, 3, tzinfo=UTC) + timedelta(minutes=minute),
        ),
        gain=gain,
    )


def _stats_updater(metrics: MetricsContext, task_key: str) -> CardStatsUpdater:
    return CardStatsUpdater(
        fitness_key="loss",
        higher_is_better=False,
        metrics_context=metrics,
        task_key=task_key,
    )


def _reputation(store: LocalMemoryStore) -> BootstrapReputation:
    return BootstrapReputation(
        BetaBinomialReputation(),
        store,
        n_bootstrap=64,
        half_life_cycles=1.0,
    )


def _writer_router(selected: dict[str, str]) -> FakeMemoryRouter:
    def respond(schema, messages):
        if schema is TaskSummaryResponse:
            return TaskSummaryResponse(summary="minimize deterministic loss")
        if schema is ReconcileResponse:
            return ReconcileResponse(
                items=[
                    ReconcileItem(
                        decision="NEW",
                        card=LibrarianCard(
                            description="reuse the task-a descent schedule",
                            explanation_summary="the schedule crosses the loss plateau",
                            keywords=["descent", "schedule"],
                        ),
                    )
                ]
            )
        if schema is ProgramAuthorResponse:
            return ProgramAuthorResponse(
                description="task-a staged-descent exemplar",
                explanation_summary="staged descent lowers the objective",
                keywords=["staged descent"],
            )
        if schema is SearchPlan:
            return SearchPlan(
                queries=[
                    ScopedQuery(
                        scope="desc_expl",
                        query="task-a descent schedule crosses the loss plateau",
                    )
                ]
            )
        if schema is ShortlistDecision:
            return ShortlistDecision(
                mode="final",
                reasoning="the shared schedule matches task-b's plateau",
                selected_ids=[selected["insight_id"]],
            )
        raise AssertionError(f"unexpected memory schema {schema.__name__}")

    return FakeMemoryRouter(respond=respond)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()


@pytest.fixture
async def shared_bank(tmp_path) -> SharedBank:
    selected: dict[str, str] = {}
    router = _writer_router(selected)
    metrics = MetricsContext(
        specs={
            "loss": MetricSpec(
                description="loss",
                higher_is_better=False,
                is_primary=True,
                significant_change=0.1,
            )
        }
    )
    store = LocalMemoryStore(
        StoreConfig(
            path=tmp_path,
            research=ResearchConfig(default_top_k=10, max_iters=1, max_cards=10),
        ),
        llm=router,
    )
    parent = _program(1, code="def solve():\n    return 10", loss=10.0)
    authored = _program(
        2,
        code="def solve():\n    return staged_descent()",
        loss=8.0,
        parents=(parent.id,),
        metadata={
            MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: dict(parent.metrics),
            MUTATION_MEMORY_BASE_ID_METADATA_KEY: parent.id,
            MUTATION_OUTPUT_METADATA_KEY: {
                "changes": [
                    {
                        "description": "introduced a staged descent schedule",
                        "explanation": "cross the loss plateau",
                    }
                ]
            },
        },
    )
    writer = MemoryWriter(
        llm=router,
        evictor=NullEvictor(),
        store=store,
        checkpoint_dir=tmp_path,
        metrics_context=metrics,
        task_key="task-a",
        task_description="Minimize loss for task A.",
        best_programs_percent=100.0,
        program_exemplars=ProgramExemplarPolicy(
            top_k_per_refresh=1,
            max_cards=1,
            store_code=True,
        ),
    )
    await writer.run_increment([parent, authored])

    insight = next(card for card in store.snapshot() if card.kind is CardKind.INSIGHT)
    exemplar = next(card for card in store.snapshot() if card.kind is CardKind.PROGRAM)
    selected["insight_id"] = insight.id

    control = _outcome_program(100, parent=parent, loss=9.0)
    gains = (7.0, 11.0, 12.0, 13.0, 14.0)
    outcomes = [control]
    for card_offset, card_id in enumerate((insight.id, exemplar.id)):
        outcomes.extend(
            _outcome_program(
                110 + card_offset * 10 + index,
                parent=parent,
                loss=loss,
                selected_id=card_id,
            )
            for index, loss in enumerate(gains)
        )
    _stats_updater(metrics, "task-a").update(
        outcomes,
        store=store,
        gate=CardAdmissionGate(store=store, evictor=NullEvictor()),
    )
    return SharedBank(
        store=store,
        router=router,
        metrics=metrics,
        parent=parent,
        authored=authored,
        insight_id=insight.id,
        exemplar_id=exemplar.id,
    )


async def test_1_task_a_writer_admits_and_restamps_direction_normalized(shared_bank):
    insight = shared_bank.store.get(shared_bank.insight_id)
    exemplar = shared_bank.store.get(shared_bank.exemplar_id)

    assert insight is not None and exemplar is not None
    assert (insight.task_key, exemplar.task_key) == ("task-a", "task-a")
    assert (insight.kind, exemplar.kind) == (CardKind.INSIGHT, CardKind.PROGRAM)
    assert [event.gain for event in insight.gain_events if event.founding] == [2.0]
    assert [event.gain for event in insight.gain_events if not event.founding] == [
        2.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    ]
    assert {
        event.context.task_key
        for event in (*insight.gain_events, *exemplar.gain_events)
    } == {"task-a"}


async def test_2a_task_b_research_reaches_task_a_card_and_sees_origin(shared_bank):
    result = await shared_bank.store.research(
        ResearchRequest(query="find a descent schedule for task-b")
    )

    assert [card.id for card in result.cards] == [shared_bank.insight_id]
    assert candidate_brief(result.cards[0])["origin_task"] == "task-a"
    reflection = next(
        messages
        for _, schema, messages in shared_bank.router.calls
        if schema is ShortlistDecision
    )
    assert '"origin_task": "task-a"' in reflection[1].content


def test_2b_foreign_stats_fold_signs_without_leaking_magnitudes(shared_bank):
    card = shared_bank.store.get(shared_bank.insight_id)
    reputation = _reputation(shared_bank.store)
    block = reputation.card_stats(card, _context("task-b"))

    assert block is not None
    assert block.intro_events == 0
    assert block.IntroGain_best_median is None
    assert block.IntroGain_bootstrap_ev_mean is None
    assert block.IntroGain_bootstrap_ev_lo20 is None
    assert block.foreign_help_events == 1
    assert block.foreign_total_events == 5
    assert (block.posterior_a, block.posterior_b) == (2.0, 5.0)


def test_2c_foreign_only_card_uses_probe_lane_below_support_floor(shared_bank):
    card = shared_bank.store.get(shared_bank.insight_id)
    context = _context("task-b")
    reputation = _reputation(shared_bank.store)
    block = reputation.card_stats(card, context)
    prior = EmpiricalBayesMemoryPrior(store=shared_bank.store)
    candidate = AuctionCandidateProjector(prior=prior).project(
        card=card,
        block=block,
        reputation=reputation,
        context=context,
    )
    winners, slate = BootstrapThompsonAuctioneer().run(
        [candidate], np.random.default_rng(7)
    )
    budgeted = TopBidBudgeter().cap(winners, slate, max_cards=1)
    probed, marked = ColdProbePolicy(
        probe_until_effective_events=3.0,
        empty_selection_probe_rate=1.0,
        warm_override_probe_rate=0.0,
    ).apply(
        budgeted_ids=budgeted,
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(8),
    )

    assert candidate.prior_source == "reputation"
    assert candidate.deltas == ()
    assert reputation.staleness_weights(card, context) == ()
    assert winners == []
    assert marked[0].support_kind == "cold_prior"
    assert marked[0].support_n == 0.0
    assert marked[0].probe_eligible is True
    assert marked[0].probe_selected is True
    assert marked[0].selection_reason == "cold_probe_empty"
    assert probed == [card.id]


def test_2d_foreign_program_render_has_sign_rate_without_fitness(shared_bank):
    exemplar = shared_bank.store.get(shared_bank.exemplar_id)
    block = _reputation(shared_bank.store).card_stats(exemplar, _context("task-b"))

    assert block is not None and block.foreign_total_events == 5
    text = EfficacyCardRenderer(task_key="task-b").render(exemplar, block)
    assert text.splitlines() == [
        "task-a staged-descent exemplar",
        "helped in 1 of 5 uses on other tasks",
        "evidence from a different task (task-a)",
    ]
    assert "exemplar fitness" not in text


def test_2e_harm_eviction_is_scoped_to_the_evidence_task(shared_bank):
    card = shared_bank.store.get(shared_bank.insight_id)
    reputation = _reputation(shared_bank.store)

    assert HarmEvictor(reputation, task_key="task-b").should_evict(card) is False
    assert HarmEvictor(reputation, task_key="task-a").should_evict(card) is True


def test_3_task_b_restamp_keeps_native_magnitudes_isolated(shared_bank):
    card = shared_bank.store.get(shared_bank.insight_id)
    reputation = _reputation(shared_bank.store)
    context_a = _context("task-a")
    before_a = reputation.card_stats(card, context_a)
    before_weights = reputation.staleness_weights(card, context_a)
    control = _outcome_program(
        200, parent=shared_bank.parent, loss=9.0, task_offset=500
    )
    task_b = [
        control,
        *(
            _outcome_program(
                210 + index,
                parent=shared_bank.parent,
                loss=loss,
                selected_id=card.id,
                task_offset=500,
            )
            for index, loss in enumerate((8.0, 7.0, 12.0))
        ),
    ]
    _stats_updater(shared_bank.metrics, "task-b").update(
        task_b,
        store=shared_bank.store,
        gate=CardAdmissionGate(store=shared_bank.store, evictor=NullEvictor()),
    )

    restamped = shared_bank.store.get(card.id)
    after_a = reputation.card_stats(restamped, context_a)
    block_b = reputation.card_stats(restamped, _context("task-b"))
    assert before_a is not None and after_a is not None and block_b is not None
    assert (block_b.intro_events, block_b.IntroGain_best_median) == (3, 1.0)
    assert block_b.IntroGain_bootstrap_ev_mean is not None
    assert block_b.foreign_total_events == 5
    assert (
        after_a.intro_events,
        after_a.IntroGain_best_median,
        after_a.IntroGain_bootstrap_ev_mean,
        after_a.IntroGain_bootstrap_ev_lo20,
    ) == (
        before_a.intro_events,
        before_a.IntroGain_best_median,
        before_a.IntroGain_bootstrap_ev_mean,
        before_a.IntroGain_bootstrap_ev_lo20,
    )
    assert (after_a.foreign_help_events, after_a.foreign_total_events) == (2, 3)
    assert reputation.staleness_weights(restamped, context_a) == before_weights
    assert {
        event.context.task_key for event in restamped.gain_events if not event.founding
    } == {"task-a", "task-b"}


async def test_4_task_b_twin_and_cap_do_not_interfere_with_task_a(
    shared_bank, tmp_path
):
    original = shared_bank.store.get(shared_bank.exemplar_id)
    task_b_program = _program(
        300,
        code=shared_bank.authored.code,
        loss=7.0,
    )
    writer = MemoryWriter(
        llm=shared_bank.router,
        evictor=NullEvictor(),
        store=shared_bank.store,
        checkpoint_dir=tmp_path,
        metrics_context=shared_bank.metrics,
        task_key="task-b",
        task_description="Minimize loss for task B.",
        best_programs_percent=100.0,
        program_exemplars=ProgramExemplarPolicy(
            top_k_per_refresh=1,
            max_cards=1,
            store_code=True,
        ),
    )
    await writer.run_increment([task_b_program])

    incoming_id = f"program-{task_b_program.id}"
    assert shared_bank.store.get(shared_bank.exemplar_id) == original
    assert shared_bank.store.get(incoming_id) is not None
    programs = [
        card for card in shared_bank.store.snapshot() if card.kind is CardKind.PROGRAM
    ]
    assert [(card.task_key, card.id) for card in programs] == [
        ("task-a", shared_bank.exemplar_id),
        ("task-b", incoming_id),
    ]


def test_5_legacy_events_are_native_only_to_legacy_context(shared_bank):
    legacy = Card(
        id="legacy-shared-card",
        task_key="",
        description="legacy shared descent advice",
        gain_events=(
            _event("", 0.4, 1),
            _event("", -0.2, 2),
            _event("", 0.1, 3),
        ),
    )
    result = CardAdmissionGate(store=shared_bank.store, evictor=NullEvictor()).admit(
        legacy
    )
    reputation = _reputation(shared_bank.store)

    assert result.card_id == legacy.id
    for task_key in ("task-a", "task-b"):
        block = reputation.card_stats(legacy, _context(task_key))
        assert block is not None
        assert block.intro_events == 0
        assert block.IntroGain_best_median is None
        assert block.IntroGain_bootstrap_ev_mean is None
        assert (block.foreign_help_events, block.foreign_total_events) == (2, 3)
        assert (block.posterior_a, block.posterior_b) == (3.0, 2.0)

    native = reputation.card_stats(legacy, _context(""))
    assert native is not None
    assert native.intro_events == 3
    assert native.IntroGain_best_median == pytest.approx(0.1)
    assert native.IntroGain_bootstrap_ev_mean is not None
    assert (native.foreign_help_events, native.foreign_total_events) == (0, 0)
