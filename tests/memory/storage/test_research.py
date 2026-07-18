"""ResearchAgent loop semantics with a scripted structured-output LLM.

The router replays scripted SearchPlan/ShortlistDecision responses (or raises
scripted exceptions), so every loop property — termination, id validation,
caps, scope filtering, candidate aggregation, fail-to-empty — is asserted
deterministically against a real bank + in-memory Chroma index.
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
    candidate_brief,
    render_candidate_briefs,
    render_candidate_briefs_with_visible_ids,
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
        index = VectorIndex(embed)
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


async def test_final_step_continue_falls_back_to_empty(make_agent, make_card):
    card = make_card(description="zebra lattice")
    router = scripted_router(
        plans=[
            plan(("description", "zebra lattice")),
            plan(("description", "zebra lattice")),
        ],
        decisions=[
            ShortlistDecision(mode="continue", additional_queries=["more lattice"]),
            ShortlistDecision(mode="continue", reasoning="still searching"),
        ],
    )
    agent = make_agent(
        [card], router, research=ResearchConfig(default_top_k=1, max_iters=2)
    )

    result = await agent.research(ResearchRequest(query="anything"))

    assert result.cards == ()
    assert result.summary == "still searching"
    assert result.iterations == 2


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


async def test_absorbed_alias_exclude_suppresses_survivor(make_agent, make_card):
    survivor = make_card(
        id="mem-new",
        absorbed_ids=("mem-old",),
        description="shared topic alpha",
    )
    allowed = make_card(description="shared topic beta")
    router = scripted_router(
        plans=[plan(("description", "shared topic"))],
        decisions=[
            ShortlistDecision(mode="final", selected_ids=[survivor.id, allowed.id])
        ],
    )
    agent = make_agent(
        [survivor, allowed], router, research=ResearchConfig(default_top_k=2)
    )

    result = await agent.research(
        ResearchRequest(query="anything", exclude_ids=frozenset({"mem-old"}))
    )

    assert [card.id for card in result.cards] == [allowed.id]


def test_candidate_brief_contains_only_semantic_applicability_evidence(make_card):
    card = make_card(
        description="line one\n  line two " + "x " * 300,
        explanation_summary="e " * 300,
        task_description_summary="t " * 300,
        task_key="origin-task",
    )

    brief = candidate_brief(card)

    assert "\n" not in brief["description"]
    assert len(brief["description"]) <= 300
    assert len(brief["evidence_summary"]) <= 160
    assert len(brief["task_description_summary"]) <= 100
    assert set(brief) == {
        "card_id",
        "kind",
        "description",
        "evidence_summary",
        "task_description_summary",
    }
    assert not {"category", "fitness", "origin_task"}.intersection(brief)


def test_render_briefs_under_budget_keeps_all_cards(make_card):
    cards = [make_card() for _ in range(3)]

    payload = json.loads(render_candidate_briefs(cards, 24000))

    assert [brief["card_id"] for brief in payload] == [card.id for card in cards]
    assert all("omitted" not in brief for brief in payload)


def test_render_briefs_over_budget_drops_whole_tail_with_marker(make_card):
    cards = [make_card(description=f"variant {i} " + "pad " * 120) for i in range(6)]
    budget = int(len(render_candidate_briefs(cards, 10**6)) * 0.6)

    rendered = render_candidate_briefs(cards, budget)

    assert len(rendered) <= budget
    payload = json.loads(rendered)
    kept, marker = payload[:-1], payload[-1]
    assert kept
    assert [brief["card_id"] for brief in kept] == [c.id for c in cards[: len(kept)]]
    assert marker == {"omitted": len(cards) - len(kept)}


def test_render_briefs_keeps_head_brief_under_tiny_budget(make_card):
    cards = [make_card(), make_card()]

    payload = json.loads(render_candidate_briefs(cards, 10))

    assert payload[0]["card_id"] == cards[0].id
    assert payload[-1] == {"omitted": 1}


def test_render_briefs_reports_only_visible_ids(make_card):
    cards = [make_card(), make_card()]

    _, visible_ids = render_candidate_briefs_with_visible_ids(cards, 10)

    assert visible_ids == {cards[0].id}


async def test_reflect_filters_ids_not_visible_in_payload(make_agent, make_card):
    shown = make_card(description="visible candidate")
    omitted = make_card(description="omitted candidate")
    router = scripted_router(
        decisions=[
            ShortlistDecision(
                mode="final", selected_ids=[omitted.id, shown.id], reasoning="fits"
            )
        ]
    )
    agent = make_agent(
        [shown, omitted],
        router,
        research=ResearchConfig(default_top_k=2, reflect_payload_chars=10),
    )

    decision = await agent._reflect(
        "anything",
        {shown.id: (shown, 1.0), omitted.id: (omitted, 0.0)},
        step=1,
        observations="",
    )

    assert decision.selected_ids == [shown.id]


async def test_reflect_briefs_order_by_reciprocal_rank_fusion(make_agent, make_card):
    far = make_card(description="quantum annealing")
    near = make_card(description="zebra lattice")
    agent = make_agent([far, near], scripted_router())

    ordered = agent._ordered_candidates(
        "anything",
        {
            far.id: (far, 1.0 / 61.0),
            near.id: (near, 2.0 / 61.0),
        },
    )

    assert [card.id for card in ordered] == [near.id, far.id]


async def test_final_step_notice_forces_final_mode_instruction(make_agent, make_card):
    router = scripted_router(
        decisions=[
            ShortlistDecision(mode="continue"),
            ShortlistDecision(mode="continue"),
        ],
    )
    agent = make_agent(
        [make_card()], router, research=ResearchConfig(default_top_k=1, max_iters=2)
    )

    await agent.research(ResearchRequest(query="anything"))

    first, second = (m[1].content for m in calls_for(router, ShortlistDecision))
    assert "Retrieval step 1 of 2." in first
    assert "FINAL STEP" not in first
    assert "Retrieval step 2 of 2." in second
    assert "FINAL STEP" in second
    assert 'mode MUST be "final"' in second


async def test_held_id_requery_and_no_new_cards_reported_next_step(
    make_agent, make_card
):
    card = make_card(description="zebra lattice")
    router = scripted_router(
        plans=[
            plan(("description", "zebra lattice")),
            plan(("description", "zebra lattice")),
        ],
        decisions=[
            ShortlistDecision(
                mode="continue", additional_queries=[f"more about {card.id}"]
            ),
            ShortlistDecision(mode="final", selected_ids=[card.id]),
        ],
    )
    agent = make_agent(
        [card], router, research=ResearchConfig(default_top_k=1, max_iters=2)
    )

    result = await agent.research(ResearchRequest(query="anything"))

    first, second = (m[1].content for m in calls_for(router, ShortlistDecision))
    assert "ALREADY HELD" not in first
    assert "NO NEW CARDS" not in first
    assert f"ALREADY HELD: [{card.id}]" in second
    assert "NO NEW CARDS matched the follow-up queries." in second
    assert [c.id for c in result.cards] == [card.id]


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
