"""GAM ResearchAgent + research-prompt contract (experimental pipeline).

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

from types import SimpleNamespace

from gigaevo.memory._vendor.GAM_root.gam.agents.research_agent import ResearchAgent
from gigaevo.memory._vendor.GAM_root.gam.prompts import research_prompts
from gigaevo.memory._vendor.GAM_root.gam.schemas import (
    EXPERIMENTAL_DECISION_SCHEMA,
    PLANNING_SCHEMA,
    Hit,
    MemoryState,
)

_PLAN = {
    "tools": ["keyword"],
    "keyword_collection": ["alpha"],
    "vector_queries": [],
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


class _FakePage:
    def __init__(self, card_id: str) -> None:
        self.header = card_id
        self.content = f"content {card_id}"
        self.meta = {
            "amem_id": card_id,
            "amem": {
                "id": card_id,
                "description": f"desc {card_id}",
                "explanation": {"summary": f"why {card_id}"},
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
        if schema is EXPERIMENTAL_DECISION_SCHEMA:
            self.decision_prompts.append(prompt)
            return {"json": self._decisions.pop(0)}
        raise AssertionError(f"unexpected schema: {schema}")


def _make_agent(
    card_ids: tuple[str, ...] = ("c1", "c2", "c3"),
    decisions: list[dict] | None = None,
    **kwargs,
) -> tuple[ResearchAgent, _ScriptedGenerator]:
    hits = [
        Hit(page_id=c, snippet=f"snippet {c}", source="keyword", meta={"score": 1.0})
        for c in card_ids
    ]
    gen = _ScriptedGenerator(decisions or [])
    agent = ResearchAgent(
        page_store=_FakePageStore([_FakePage(c) for c in card_ids]),
        memory_store=_FakeMemoryStore(),
        retrievers={"keyword": _StubRetriever(hits)},
        generator=gen,
        pipeline_mode="experimental",
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
        "keyword",
        "vector",
        "vector_description",
        "vector_task_description",
        "vector_explanation_summary",
        "page_index",
    }


def test_render_tool_section_only_active_tools():
    section = research_prompts.render_tool_section(
        ["keyword", "vector_task_description"]
    )
    assert research_prompts.TOOL_GUIDANCE["keyword"] in section
    assert research_prompts.TOOL_GUIDANCE["vector_task_description"] in section
    assert research_prompts.TOOL_GUIDANCE["page_index"] not in section
    assert research_prompts.TOOL_GUIDANCE["vector"] not in section


def test_planning_prompt_lists_only_active_tools():
    agent, gen = _make_agent(
        decisions=[_final("c1")],
        allowed_tools=["keyword", "vector_task_description"],
    )
    agent.research("pick cards")
    prompt = gen.planning_prompts[0]
    assert research_prompts.TOOL_GUIDANCE["keyword"] in prompt
    assert research_prompts.TOOL_GUIDANCE["vector_task_description"] in prompt
    assert research_prompts.TOOL_GUIDANCE["page_index"] not in prompt
    assert research_prompts.TOOL_GUIDANCE["vector"] not in prompt


def test_top_k_zero_removes_tool_from_planning_prompt():
    agent, gen = _make_agent(decisions=[_final("c1")], top_k_by_tool={"keyword": 0})
    agent.research("pick cards")
    assert research_prompts.TOOL_GUIDANCE["keyword"] not in gen.planning_prompts[0]


# --- top_k == 0 disables a tool ----------------------------------------------


def test_normalize_top_k_accepts_zero_to_disable():
    normalized = ResearchAgent._normalize_top_k_by_tool({"keyword": 0})
    assert normalized["keyword"] == 0


def test_normalize_top_k_negative_still_ignored():
    normalized = ResearchAgent._normalize_top_k_by_tool({"keyword": -2})
    assert normalized["keyword"] == 5


def test_filter_tools_drops_zero_top_k_tools():
    agent, _ = _make_agent(decisions=[_final("c1")], top_k_by_tool={"keyword": 0})
    assert agent._filter_tools(["keyword", "page_index"]) == ["page_index"]


# --- prompt hygiene -----------------------------------------------------------


def test_prompts_free_of_think_scaffolding():
    for template in (
        research_prompts.Planning_PROMPT,
        research_prompts.Integrate_PROMPT,
        research_prompts.InfoCheck_PROMPT,
        research_prompts.GenerateRequests_PROMPT,
        research_prompts.ExperimentalDecision_PROMPT,
    ):
        assert "<think>" not in template
        assert "</think>" not in template
        assert "THINKING STEP" not in template


def test_prompts_free_of_dead_card_fields():
    for template in (
        research_prompts.Planning_PROMPT,
        research_prompts.Integrate_PROMPT,
    ):
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
