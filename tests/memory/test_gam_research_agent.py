"""GAM ResearchAgent + research-prompt contract.

Pins the post-audit behavior:
- final decisions are never padded with unvetted retrieved card ids;
- the top-ideas cap is parametrized (``max_cards``), not hardcoded to 3;
- the planning prompt advertises only active tools (allowed AND top_k > 0),
  with per-tool guidance rendered dynamically;
- prompts carry no dead card fields, no project-specific examples, and no
  ``<think>`` scaffolding (structured-output schemas already enforce JSON);
- ``top_k == 0`` disables a tool instead of silently restoring the default;
- a role-free ``planning_request`` drives retrieval planning while the full
  request (selector role) still drives the reflection/selection decision.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from gigaevo.memory._vendor.GAM_root.gam.agents.research_agent import ResearchAgent
from gigaevo.memory._vendor.GAM_root.gam.prompts import research_prompts
from gigaevo.memory._vendor.GAM_root.gam.schemas import (
    DECISION_SCHEMA,
    PLANNING_SCHEMA,
    Hit,
    MemoryState,
)
from gigaevo.memory.core.events import memory_event_context

_PLAN = {
    "tools": ["vector"],
    "keyword_collection": [],
    "vector_queries": ["alpha"],
    "vector_description_queries": [],
    "vector_task_description_queries": [],
    "vector_explanation_summary_queries": [],
    "page_index": [],
}

_CONTINUE = {"mode": "continue", "top_ideas": [], "additional_queries": ["more"]}


def _final(*ids: str) -> dict:
    return {
        "mode": "final",
        "top_ideas": [{"card_id": i} for i in ids],
        "additional_queries": [],
    }


def _event_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _FakePage:
    def __init__(self, card_id: str) -> None:
        self.header = card_id
        self.content = f"content {card_id}"
        self.meta = {
            "amem_id": card_id,
            "amem": {
                "id": card_id,
                "category": "general",
                "description": f"desc {card_id}",
                "task_description_summary": f"task summary {card_id}",
                "task_description": f"task context {card_id}",
                "strategy": "exploration",
                "keywords": [f"kw-{card_id}", "mechanism"],
                "explanation": {"summary": f"why {card_id}"},
                "works_with": ["ally-card"],
                "links": ["related-card"],
                "gain_events": [
                    {
                        "context": {"parent_metrics": {"min_area": 0.5}},
                        "gain": 0.25,
                    }
                ],
            },
        }


class _FakePageStore:
    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = list(pages)

    def load(self) -> list[_FakePage]:
        return self._pages

    def get(self, idx: int):
        return self._pages[idx] if 0 <= idx < len(self._pages) else None


class _FakeMemoryStore:
    def load(self) -> MemoryState:
        return MemoryState()


class _StubRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def build(self, page_store) -> None:
        pass

    def update(self, page_store) -> None:
        pass

    def search(self, query_list, top_k=3):
        return [list(self._hits[:top_k])]


class _ScriptedGenerator:
    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.planning_prompts: list[str] = []
        self.decision_prompts: list[str] = []

    def generate_single(self, prompt: str, schema=None) -> dict:
        if schema is PLANNING_SCHEMA:
            self.planning_prompts.append(prompt)
            return {"json": dict(_PLAN)}
        if schema is DECISION_SCHEMA:
            self.decision_prompts.append(prompt)
            return {"json": self._decisions.pop(0)}
        raise AssertionError(f"unexpected schema: {schema}")


def _make_agent(
    card_ids: tuple[str, ...] = ("c1", "c2", "c3"),
    decisions: list[dict] | None = None,
    **kwargs,
) -> tuple[ResearchAgent, _ScriptedGenerator]:
    hits = [
        Hit(page_id=c, snippet=f"snippet {c}", source="vector", meta={"score": 1.0})
        for c in card_ids
    ]
    gen = _ScriptedGenerator(decisions or [])
    agent = ResearchAgent(
        page_store=_FakePageStore([_FakePage(c) for c in card_ids]),
        memory_store=_FakeMemoryStore(),
        retrievers={"vector": _StubRetriever(hits)},
        generator=gen,
        **kwargs,
    )
    return agent, gen


# --- no padding of final decisions ------------------------------------------


def test_final_top_ideas_not_padded_with_retrieved_ids():
    agent, _ = _make_agent(decisions=[_final("c1")])
    out = agent.research("pick cards")
    ideas = out.raw_memory["final_decision"]["top_ideas"]
    assert [i["card_id"] for i in ideas] == ["c1"]


def test_exhausted_iterations_yield_empty_top_ideas():
    agent, _ = _make_agent(decisions=[_CONTINUE], max_iters=1)
    out = agent.research("pick cards")
    assert out.raw_memory["final_decision"]["top_ideas"] == []
    assert "No final top ideas" in out.integrated_memory


# --- max_cards parametrization ----------------------------------------------


def test_reflection_caps_top_ideas_at_max_cards():
    agent, _ = _make_agent(
        card_ids=("c1", "c2", "c3", "c4"),
        decisions=[_final("c1", "c2", "c3", "c4")],
        max_cards=2,
    )
    out = agent.research("pick cards")
    ideas = out.raw_memory["final_decision"]["top_ideas"]
    assert [i["card_id"] for i in ideas] == ["c1", "c2"]


def test_decision_prompt_parametrized_by_max_cards():
    agent, gen = _make_agent(decisions=[_final("c1")], max_cards=2)
    agent.research("pick cards")
    prompt = gen.decision_prompts[0]
    assert "top 2" in prompt
    assert "top 3" not in prompt
    assert "exactly 3" not in prompt


# --- dynamic tool section -----------------------------------------------------


def test_tool_guidance_covers_all_supported_tools():
    assert set(research_prompts.TOOL_GUIDANCE) == {
        "vector",
        "vector_description",
        "vector_task_description",
        "vector_explanation_summary",
        "page_index",
    }


def test_render_tool_section_only_active_tools():
    section = research_prompts.render_tool_section(
        ["vector_description", "vector_task_description"]
    )
    assert research_prompts.TOOL_GUIDANCE["vector_description"] in section
    assert research_prompts.TOOL_GUIDANCE["vector_task_description"] in section
    assert research_prompts.TOOL_GUIDANCE["page_index"] not in section
    assert research_prompts.TOOL_GUIDANCE["vector_explanation_summary"] not in section


def test_planning_prompt_lists_only_active_tools():
    agent, gen = _make_agent(
        decisions=[_final("c1")],
        allowed_tools=["vector_description", "vector_task_description"],
    )
    agent.research("pick cards")
    prompt = gen.planning_prompts[0]
    assert research_prompts.TOOL_GUIDANCE["vector_description"] in prompt
    assert research_prompts.TOOL_GUIDANCE["vector_task_description"] in prompt
    assert research_prompts.TOOL_GUIDANCE["page_index"] not in prompt
    assert research_prompts.TOOL_GUIDANCE["vector_explanation_summary"] not in prompt


def test_top_k_zero_removes_tool_from_planning_prompt():
    agent, gen = _make_agent(decisions=[_final("c1")], top_k_by_tool={"vector": 0})
    agent.research("pick cards")
    assert research_prompts.TOOL_GUIDANCE["vector"] not in gen.planning_prompts[0]


# --- top_k == 0 disables a tool ----------------------------------------------


def test_normalize_top_k_accepts_zero_to_disable():
    normalized = ResearchAgent._normalize_top_k_by_tool({"vector": 0})
    assert normalized["vector"] == 0


def test_normalize_top_k_negative_still_ignored():
    normalized = ResearchAgent._normalize_top_k_by_tool({"vector": -2})
    assert normalized["vector"] == 5


def test_filter_tools_drops_zero_top_k_tools():
    agent, _ = _make_agent(decisions=[_final("c1")], top_k_by_tool={"vector": 0})
    assert agent._filter_tools(["vector", "page_index"]) == ["page_index"]


def test_extract_explanation_summary_prefers_top_level_field():
    # Cards now model_dump to a top-level ``explanation_summary``; the final
    # top-ideas renderer must read it, not only the legacy nested
    # ``explanation.summary`` shape, or WHEN_TO_USE renders "(not provided)".
    assert (
        ResearchAgent._extract_explanation_summary({"explanation_summary": "why X"})
        == "why X"
    )


def test_extract_explanation_summary_falls_back_to_legacy_nested():
    assert (
        ResearchAgent._extract_explanation_summary(
            {"explanation": {"summary": "legacy why"}}
        )
        == "legacy why"
    )


# --- prompt hygiene -----------------------------------------------------------


def test_prompts_free_of_think_scaffolding():
    for template in (
        research_prompts.Planning_PROMPT,
        research_prompts.Decision_PROMPT,
    ):
        assert "<think>" not in template
        assert "</think>" not in template
        assert "THINKING STEP" not in template


def test_prompts_free_of_dead_card_fields():
    template = research_prompts.Planning_PROMPT
    assert "last_generation" not in template
    assert "usage" not in template
    assert '"programs"' not in template


def test_no_project_specific_retrieval_examples():
    blob = research_prompts.Planning_PROMPT + "".join(
        research_prompts.TOOL_GUIDANCE.values()
    )
    assert "DenseRetriever" not in blob
    assert "GPU" not in blob


def test_gam_card_text_drops_dead_placeholder_fields():
    from gigaevo.memory._vendor.A_mem.agentic_memory.memory_system import (
        AgenticMemorySystem,
    )

    note = SimpleNamespace(
        content="d", context="t", category="c", strategy="s", keywords=["k"], links=[]
    )
    text = AgenticMemorySystem._build_gam_card_text(None, note)
    assert "description: d" in text
    assert "last_generation" not in text
    assert "programs:" not in text
    assert "usage:" not in text


# --- planning_request seam ----------------------------------------------------


def test_planning_request_separates_role_from_planning():
    agent, gen = _make_agent(decisions=[_final("c1")])
    agent.research(
        "ROLE_BLOCK_SENTINEL\n\nCORE_REQUEST", planning_request="CORE_REQUEST"
    )
    assert "CORE_REQUEST" in gen.planning_prompts[0]
    assert "ROLE_BLOCK_SENTINEL" not in gen.planning_prompts[0]
    assert "ROLE_BLOCK_SENTINEL" in gen.decision_prompts[0]


def test_planning_request_defaults_to_full_request():
    agent, gen = _make_agent(decisions=[_final("c1")])
    agent.research("FULL_REQUEST_ONLY")
    assert "FULL_REQUEST_ONLY" in gen.planning_prompts[0]


# --- prompt quality -----------------------------------------------------------


def test_decision_prompt_includes_compact_candidate_context():
    agent, gen = _make_agent(decisions=[_final("c1")])
    agent.research("pick cards")

    prompt = gen.decision_prompts[0]
    assert '"task_description_summary": "task summary c1"' in prompt
    assert '"task_description": "task context c1"' in prompt
    assert '"strategy": "exploration"' in prompt
    assert '"keywords": [' in prompt
    assert "kw-c1" in prompt
    assert '"works_with": [' in prompt
    assert '"links": [' in prompt
    # Memory cards no longer carry an efficacy line in the GAM candidate context:
    # the per-card endorsement is resolved at read time from gain_events, and the
    # cohort adjustment is gone. The rendered candidate JSON exposes no efficacy.
    ideas_json = prompt.split("RETRIEVED_IDEAS:\n", 1)[1].split(
        "\n\nDecision rules:", 1
    )[0]
    assert '"efficacy"' not in ideas_json
    assert "vs cohort" not in prompt


# --- canonical telemetry ------------------------------------------------------


def test_pipeline_emits_canonical_gam_events(tmp_path):
    path = tmp_path / "memory_events.jsonl"
    agent, _ = _make_agent(decisions=[_final("c1")])

    with memory_event_context(decision_id="d-gam", event_path=path):
        agent.research("pick cards")

    rows = _event_rows(path)
    types = [row["event_type"] for row in rows]

    assert "gam.research.start" in types
    assert "gam.retriever_update" in types
    assert "gam.plan" in types
    assert "gam.search.tool" in types
    assert "gam.search" in types
    assert "gam.reflection" in types
    assert "gam.iteration" in types
    assert "gam.research.complete" in types
    assert all(row["decision_id"] == "d-gam" for row in rows)

    plan = [row for row in rows if row["event_type"] == "gam.plan"][-1]
    assert plan["payload"]["outcome"] == "ok"
    assert plan["payload"]["filtered_tools"] == ["vector"]

    search = [row for row in rows if row["event_type"] == "gam.search"][-1]
    assert search["payload"]["mode"] == "no_integrate"
    assert search["payload"]["idea_count"] == 3

    reflection = [row for row in rows if row["event_type"] == "gam.reflection"][-1]
    assert reflection["payload"]["mode"] == "final"
    assert reflection["payload"]["top_idea_ids"] == ["c1"]


# --- page_index hits must carry the card id, not the numeric page index ------


def test_index_retriever_emits_amem_id_not_numeric_page_index(tmp_path):
    """The live page_index retriever must surface the card's ``amem_id`` as the
    Hit page_id; the numeric store index belongs in meta, not as the card id.
    Emitting ``str(pid)`` makes the hit unfetchable from the card store.
    """
    from gigaevo.memory._vendor.GAM_root.gam.retriever.index_retriever import (
        IndexRetriever,
    )
    from gigaevo.memory._vendor.GAM_root.gam.schemas import InMemoryPageStore, Page

    source = InMemoryPageStore()
    source._pages = [
        Page(header="[A-MEM] mem-a", content="card a body", meta={"amem_id": "mem-a"})
    ]
    retriever = IndexRetriever({"index_dir": str(tmp_path)})
    retriever.build(source)

    [hits] = retriever.search(["0"], top_k=1)

    assert [h.page_id for h in hits] == ["mem-a"]
    assert hits[0].meta.get("page_index") == 0


def test_agent_page_index_fallback_emits_amem_id_not_numeric_index():
    """The agent's in-method page_index fallback (used when no page_index
    retriever is wired) is the twin of the IndexRetriever path and must also
    resolve the numeric index to the page's ``amem_id``.
    """
    agent, _ = _make_agent(card_ids=("mem-a",))
    assert agent.retrievers.get("page_index") is None  # fallback loop runs

    [hits] = agent._search_by_page_index([0])

    assert [h.page_id for h in hits] == ["mem-a"]
    assert hits[0].meta.get("page_index") == 0
