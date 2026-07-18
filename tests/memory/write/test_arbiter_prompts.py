from __future__ import annotations

from gigaevo.llm.agents.card_author import CardAuthorAgent
from gigaevo.llm.agents.equivalence import EquivalenceAgent
from gigaevo.memory.cards import Card
from gigaevo.memory.write.decisions import ArchiveStatus, ValidityStatus


def uninitialized(agent_type, template):
    agent = object.__new__(agent_type)
    agent.system_prompt = "system"
    agent.user_prompt_template = template
    return agent


def test_card_author_renders_all_outcome_signal_and_mutator_explanation():
    agent = uninitialized(
        CardAuthorAgent,
        "{parent_fitness}|{child_fitness}|{signed_gain}|{validity_status}|"
        "{archive_status}|{mutation_report}|{base_parent_code}|{child_code}|"
        "{unified_diff}",
    )
    state = agent.build_prompt(
        {
            "base_parent_code": "parent",
            "child_code": "child",
            "unified_diff": "diff",
            "mutation_report": "EXPLANATION_MARKER",
            "parent_fitness": 1.0,
            "child_fitness": 2.0,
            "signed_gain": 1.0,
            "validity_status": ValidityStatus.VALID,
            "archive_status": ArchiveStatus.ARCHIVED,
        }
    )
    prompt = state["messages"][1].content
    assert prompt == "1.0|2.0|1.0|valid|archived|EXPLANATION_MARKER|parent|child|diff"


def test_equivalence_renders_candidate_and_only_offered_neighbors():
    agent = uninitialized(EquivalenceAgent, "{kind}\n{candidate}\n{neighbors}")
    candidate = Card(
        id="", description="candidate action", explanation_summary="candidate why"
    )
    neighbor = Card(
        id="mem-neighbor",
        description="neighbor action",
        explanation_summary="neighbor why",
    )
    state = agent.build_prompt({"candidate": candidate, "neighbors": [neighbor]})
    prompt = state["messages"][1].content
    assert "insight" in prompt
    assert "candidate action | why: candidate why" in prompt
    assert "mem-neighbor: neighbor action | why: neighbor why" in prompt


def test_retrieval_prompts_do_not_rank_on_dead_keyword_metadata():
    from gigaevo.prompts import load_prompt

    assert "keywords" not in load_prompt("retrieval_reflection", "system")
    assert "keywords" not in load_prompt("equivalence", "system")
