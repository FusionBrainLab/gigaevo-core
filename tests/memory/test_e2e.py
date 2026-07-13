"""Two-run e2e over one checkpoint dir: a writer run authors the bank on disk,
then a fresh reader run (new store, new components — nothing shared in-process)
retrieves those cards into a mutation-facing selection.

Both runs use the production component stack from ``config/memory/full.yaml``
(real ``LocalMemoryStore`` + in-memory Chroma); only the embedder and the
LLM router are scripted doubles.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gigaevo.evolution.mutation.constants import MUTATION_OUTPUT_METADATA_KEY
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import (
    LibrarianCard,
    ReconcileItem,
    ReconcileResponse,
)
from gigaevo.llm.agents.task_summary import TaskSummaryResponse
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.provider import ReaderMemoryProvider
from gigaevo.memory.read.auction import BootstrapThompsonAuctioneer, TopBidBudgeter
from gigaevo.memory.read.probe import ColdProbePolicy
from gigaevo.memory.read.reader import MemoryReader
from gigaevo.memory.read.render import EfficacyCardRenderer
from gigaevo.memory.read.reputation import BetaBinomialReputation
from gigaevo.memory.read.shortlist import ResearchShortlister
from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.storage.research import (
    ScopedQuery,
    SearchPlan,
    ShortlistDecision,
)
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import Lineage, Program
from tests.fakes.embedding import FakeEmbeddingFunction
from tests.fakes.llm_router import FakeMemoryRouter

TASK = "Place points on the sphere maximizing the minimal pairwise distance."
METRICS = "fitness: minimal pairwise distance (higher is better)"
IDEA = LibrarianCard(
    description=(
        "Swap the greedy placement for simulated annealing so the search "
        "accepts uphill moves and escapes the greedy local optimum"
    ),
    explanation_summary=(
        "annealing's acceptance temperature lets the search cross fitness "
        "barriers the greedy step is stuck behind"
    ),
    keywords=["simulated annealing", "local optimum"],
)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()


def make_program(code: str, fitness: float, parents: list[str]) -> Program:
    return Program(
        code=code,
        metrics={VALIDITY_KEY: 1.0, "fitness": fitness},
        metadata=(
            {
                MUTATION_OUTPUT_METADATA_KEY: {
                    "changes": [{"description": "swapped solver", "explanation": ""}]
                }
            }
            if parents
            else {}
        ),
        lineage=Lineage(parents=parents, generation=1, mutation=None),
    )


def writer_router() -> FakeMemoryRouter:
    def respond(schema, messages):
        if schema is TaskSummaryResponse:
            return TaskSummaryResponse(summary="maximize sphere point spread")
        if schema is ReconcileResponse:
            return ReconcileResponse(items=[ReconcileItem(decision="NEW", card=IDEA)])
        if schema is ProgramAuthorResponse:
            return ProgramAuthorResponse(
                description="exemplar anneals point positions from a greedy seed",
                explanation_summary="cooling schedule balances exploration",
                keywords=["annealing"],
            )
        raise AssertionError(f"unexpected writer schema {schema.__name__}")

    return FakeMemoryRouter(respond=respond)


def reader_router(insight_id: str) -> FakeMemoryRouter:
    def respond(schema, messages):
        if schema is SearchPlan:
            return SearchPlan(
                queries=[
                    ScopedQuery(
                        scope="desc_expl",
                        query="simulated annealing escapes greedy local optimum",
                    )
                ]
            )
        if schema is ShortlistDecision:
            return ShortlistDecision(
                mode="final",
                reasoning="the annealing lever targets the greedy parent's plateau",
                selected_ids=[insight_id],
            )
        raise AssertionError(f"unexpected reader schema {schema.__name__}")

    return FakeMemoryRouter(respond=respond)


async def test_writer_bank_feeds_fresh_reader_run(tmp_path):
    metrics_context = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            )
        }
    )
    parent = make_program("x = greedy()", fitness=0.5, parents=[])
    child = make_program("x = anneal()", fitness=0.7, parents=[parent.id])

    writer = MemoryWriter(
        llm=writer_router(),
        evictor=NullEvictor(),
        store=LocalMemoryStore(StoreConfig(path=tmp_path)),
        checkpoint_dir=tmp_path,
        metrics_context=metrics_context,
        task_description=TASK,
        best_programs_percent=100.0,
    )
    await writer.run_increment([parent, child])

    bank_file = tmp_path / "cards.json"
    assert bank_file.exists()
    assert (tmp_path / "write_ledger.jsonl").exists()
    dumped = json.loads(bank_file.read_text())["cards"]
    cards = [Card.model_validate(raw) for raw in dumped.values()]
    by_kind = {kind: [c for c in cards if c.kind is kind] for kind in CardKind}
    assert len(by_kind[CardKind.INSIGHT]) == 1
    assert {c.id for c in by_kind[CardKind.PROGRAM]} == {
        f"program-{parent.id}",
        f"program-{child.id}",
    }
    insight = by_kind[CardKind.INSIGHT][0]
    assert insight.description == IDEA.description
    assert insight.task_description_summary == "maximize sphere point spread"

    store = LocalMemoryStore(StoreConfig(path=tmp_path), llm=reader_router(insight.id))
    assert {c.id for c in store.snapshot()} == {c.id for c in cards}
    reader = MemoryReader(
        shortlister=ResearchShortlister(store=store),
        reputation=BetaBinomialReputation(),
        auctioneer=BootstrapThompsonAuctioneer(),
        budgeter=TopBidBudgeter(),
        renderer=EfficacyCardRenderer(),
        probe_policy=ColdProbePolicy(empty_selection_probe_rate=1.0),
        max_cards=1,
        rng=np.random.default_rng(1),
    )
    provider = ReaderMemoryProvider(reader=reader)

    selection = await provider.select_cards(
        parent, task_description=TASK, metrics_description=METRICS
    )

    assert selection.card_ids == (insight.id,)
    assert IDEA.description in selection.cards[0]
    assert [bid.card_id for bid in selection.slate] == [insight.id]
