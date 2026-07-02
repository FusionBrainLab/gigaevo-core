"""ResearchAgent loop semantics with a scripted structured-output LLM.

The router replays scripted SearchPlan/ShortlistDecision responses (or raises
scripted exceptions), so every loop property — termination, id validation,
caps, scope filtering, candidate aggregation, fail-to-empty — is asserted
deterministically against a real bank + Chroma index.
"""

from __future__ import annotations

import json

import pytest

from gigaevo.memory.events import memory_event_context
from gigaevo.memory.storage.bank import CardBank
from gigaevo.memory.storage.base import ResearchRequest
from gigaevo.memory.storage.config import EmbedConfig, ResearchConfig
from gigaevo.memory.storage.index import VectorIndex
from gigaevo.memory.storage.research import (
    ResearchAgent,
    ScopedQuery,
    SearchPlan,
    ShortlistDecision,
)
from tests.fakes.llm_router import FakeMemoryRouter


def scripted_router(plans=(), decisions=()) -> FakeMemoryRouter:
    remaining = {SearchPlan: list(plans), ShortlistDecision: list(decisions)}
    defaults = {
        SearchPlan: lambda: SearchPlan(),
        ShortlistDecision: lambda: ShortlistDecision(mode="continue"),
    }

    def respond(schema, messages):
        script = remaining[schema]
        item = script.pop(0) if script else defaults[schema]()
        if isinstance(item, Exception):
            raise item
        return item

    return FakeMemoryRouter(respond=respond)


def plan(*queries: tuple[str, str]) -> SearchPlan:
    return SearchPlan(
        queries=[ScopedQuery(scope=scope, query=query) for scope, query in queries]
    )


def calls_for(router: FakeMemoryRouter, schema: type) -> list:
    return [messages for _, called, messages in router.calls if called is schema]


@pytest.fixture
def make_agent(tmp_path):
    def _make_agent(cards, router, research=None):
        embed = EmbedConfig(
            embed_scopes={"description": ("description",)},
            nearest_scope="description",
        )
        bank = CardBank(tmp_path / "cards.json")
        for card in cards:
            bank.put(card)
        index = VectorIndex(tmp_path / "chroma", embed)
        index.rebuild(cards)
        return ResearchAgent(
            router,
            bank,
            index,
            embed,
            research or ResearchConfig(default_top_k=1),
            query_scopes=("description",),
        )

    return _make_agent


async def test_final_decision_returns_shortlist(make_agent, make_card):
    target = make_card(description="zebra lattice")
    other = make_card(description="ordinary gradient descent")
    router = scripted_router(
        plans=[plan(("description", "zebra lattice"))],
        decisions=[
            ShortlistDecision(mode="final", reasoning="fits", selected_ids=[target.id])
        ],
    )
    agent = make_agent([target, other], router)

    result = await agent.research(ResearchRequest(query="need lattice ideas"))

    assert [card.id for card in result.cards] == [target.id]
    assert result.summary == "fits"
    assert result.iterations == 1


async def test_selected_ids_validated_against_candidates(make_agent, make_card):
    retrieved = make_card(description="zebra lattice")
    never_retrieved = make_card(description="ordinary gradient descent")
    router = scripted_router(
        plans=[plan(("description", "zebra lattice"))],
        decisions=[
            ShortlistDecision(
                mode="final",
                selected_ids=[never_retrieved.id, "mem-invented9999", retrieved.id],
            )
        ],
    )
    agent = make_agent([retrieved, never_retrieved], router)

    result = await agent.research(ResearchRequest(query="anything"))

    assert [card.id for card in result.cards] == [retrieved.id]


async def test_max_cards_caps_shortlist(make_agent, make_card):
    cards = [make_card(description=f"shared topic variant-{i}") for i in range(3)]
    router = scripted_router(
        plans=[plan(("description", "shared topic"))],
        decisions=[ShortlistDecision(mode="final", selected_ids=[c.id for c in cards])],
    )
    agent = make_agent(
        cards, router, research=ResearchConfig(default_top_k=3, max_cards=2)
    )

    result = await agent.research(ResearchRequest(query="anything"))

    assert len(result.cards) == 2
    assert [card.id for card in result.cards] == [cards[0].id, cards[1].id]


async def test_never_final_exhausts_iterations_empty(make_agent, make_card):
    card = make_card(description="zebra lattice")
    router = scripted_router()
    agent = make_agent(
        [card], router, research=ResearchConfig(default_top_k=1, max_iters=2)
    )

    result = await agent.research(ResearchRequest(query="anything"))

    assert result.cards == ()
    assert result.iterations == 2
    assert len(calls_for(router, SearchPlan)) == 2
    assert len(calls_for(router, ShortlistDecision)) == 2


async def test_planner_failure_degrades_to_empty_plan(make_agent, make_card):
    router = scripted_router(
        plans=[RuntimeError("planner exploded")],
        decisions=[ShortlistDecision(mode="final", selected_ids=[])],
    )
    agent = make_agent([make_card()], router)

    result = await agent.research(ResearchRequest(query="anything"))

    assert result.cards == ()
    assert result.iterations == 1
    reflect_messages = calls_for(router, ShortlistDecision)[0]
    assert "[]" in reflect_messages[1].content


async def test_reflector_failure_treated_as_continue(make_agent, make_card):
    card = make_card(description="zebra lattice")
    router = scripted_router(
        plans=[plan(("description", "zebra lattice"))],
        decisions=[RuntimeError("reflector exploded")],
    )
    agent = make_agent(
        [card], router, research=ResearchConfig(default_top_k=1, max_iters=1)
    )

    result = await agent.research(ResearchRequest(query="anything"))

    assert result.cards == ()
    assert result.iterations == 1


async def test_disallowed_scope_and_blank_queries_dropped(make_agent, make_card):
    card = make_card(description="zebra lattice")
    router = scripted_router(
        plans=[plan(("desc_expl", "zebra lattice"), ("description", "   "))],
        decisions=[ShortlistDecision(mode="final", selected_ids=[card.id])],
    )
    agent = make_agent([card], router)

    result = await agent.research(ResearchRequest(query="anything"))

    assert result.cards == ()
    reflect_messages = calls_for(router, ShortlistDecision)[0]
    assert "[]" in reflect_messages[1].content


async def test_candidates_aggregate_across_iterations(make_agent, make_card):
    first = make_card(description="zebra lattice")
    second = make_card(description="quantum annealing")
    router = scripted_router(
        plans=[
            plan(("description", "zebra lattice")),
            plan(("description", "quantum annealing")),
        ],
        decisions=[
            ShortlistDecision(mode="continue"),
            ShortlistDecision(mode="final", selected_ids=[first.id, second.id]),
        ],
    )
    agent = make_agent([first, second], router)

    result = await agent.research(ResearchRequest(query="anything"))

    assert [card.id for card in result.cards] == [first.id, second.id]
    assert result.iterations == 2


async def test_followup_queries_fold_into_next_planner_request(make_agent, make_card):
    router = scripted_router(
        decisions=[
            ShortlistDecision(
                mode="continue", additional_queries=["focus on symmetry breaking"]
            ),
            ShortlistDecision(mode="final", selected_ids=[]),
        ],
    )
    agent = make_agent(
        [make_card()], router, research=ResearchConfig(default_top_k=1, max_iters=2)
    )

    await agent.research(ResearchRequest(query="base request"))

    first_prompt, second_prompt = (
        messages[1].content for messages in calls_for(router, SearchPlan)
    )
    assert "Follow-up retrieval focus:" not in first_prompt
    assert "Follow-up retrieval focus:" in second_prompt
    assert "1. focus on symmetry breaking" in second_prompt
    assert "base request" in second_prompt


async def test_exclude_ids_never_become_candidates(make_agent, make_card):
    excluded = make_card(description="shared topic alpha")
    allowed = make_card(description="shared topic beta")
    router = scripted_router(
        plans=[plan(("description", "shared topic"))],
        decisions=[
            ShortlistDecision(mode="final", selected_ids=[excluded.id, allowed.id])
        ],
    )
    agent = make_agent(
        [excluded, allowed], router, research=ResearchConfig(default_top_k=2)
    )

    result = await agent.research(
        ResearchRequest(query="anything", exclude_ids=frozenset({excluded.id}))
    )

    assert [card.id for card in result.cards] == [allowed.id]


async def test_research_step_events_recorded(make_agent, make_card, tmp_path):
    card = make_card(description="zebra lattice")
    router = scripted_router(
        plans=[
            plan(("description", "zebra lattice")),
            plan(("description", "zebra lattice")),
        ],
        decisions=[
            ShortlistDecision(mode="continue"),
            ShortlistDecision(mode="final", selected_ids=[card.id]),
        ],
    )
    agent = make_agent([card], router)
    events_file = tmp_path / "events.jsonl"

    with memory_event_context(event_path=events_file):
        await agent.research(ResearchRequest(query="anything"))

    rows = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["MEMORY_RESEARCH_STEP"] * 2
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["decision"] for row in rows] == ["continue", "final"]
    assert rows[0]["hit_ids"] == [card.id]
