from __future__ import annotations

from gigaevo.llm.agents.card_author import CardAuthorAgent
from gigaevo.llm.agents.equivalence import EquivalenceAgent
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.decisions import ArchiveStatus
from gigaevo.prompts import load_prompt


def uninitialized(agent_type, template):
    agent = object.__new__(agent_type)
    agent.system_prompt = "system"
    agent.user_prompt_template = template
    if agent_type is CardAuthorAgent:
        agent.fitness_key = "score"
    return agent


def test_card_author_renders_all_outcome_signal_and_mutator_explanation():
    agent = uninitialized(
        CardAuthorAgent,
        "{fitness_key}|{parent_fitness}|{child_fitness}|{signed_gain}|"
        "{fitness_direction}|{archive_status}|{mutation_report}|"
        "{base_parent_code}|{child_code}|{unified_diff}",
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
            "higher_is_better": False,
            "archive_status": ArchiveStatus.ARCHIVED,
        }
    )
    prompt = state["messages"][1].content
    assert prompt == (
        "score|1.0|2.0|1.0|lower is better|archived|EXPLANATION_MARKER|"
        "parent|child|diff"
    )


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


def test_equivalence_renders_program_kind():
    agent = uninitialized(EquivalenceAgent, "{kind}\n{candidate}\n{neighbors}")
    candidate = Card(
        id="",
        kind=CardKind.PROGRAM,
        program_id="candidate-program",
        description="candidate strategy",
    )
    neighbor = Card(
        id="program-neighbor",
        kind=CardKind.PROGRAM,
        program_id="neighbor-program",
        description="neighbor strategy",
    )

    state = agent.build_prompt({"candidate": candidate, "neighbors": [neighbor]})

    assert state["messages"][1].content.startswith("program\n")


def test_equivalence_contract_is_kind_specific():
    system = load_prompt("equivalence", "system")
    user = load_prompt("equivalence", "user")

    assert "INSIGHT CARDS: STRICT INTERVENTIONAL IDENTITY" in system
    assert "PROGRAM CARDS: STRATEGY-FAMILY IDENTITY" in system
    assert (
        "core transformation, algorithm, procedure, pipeline, or control flow" in system
    )
    assert "update, selection, or output policy" in system
    assert "Do not fill in a missing load-bearing axis" in system
    assert "load-bearing axis changes" in system
    assert "comparison_summary" in system
    assert "`applicability`:" in system
    assert "`representation_or_state`:" in system
    assert "neither card's strategy involves the role at all" in system
    assert "one card names the role and the other's text is too thin" in system
    assert "final decision and target are computed from `program_axes`" in system
    assert "training regime" not in system
    assert "model families" not in system
    assert "simulated annealing" not in system
    assert "global optimization" not in system
    assert "card kind" in user


def test_program_author_targets_stable_strategy_families():
    system = load_prompt("program_author", "system")

    assert "strategy-family rather than implementation-instance resolution" in system
    assert "core algorithm, model, pipeline, or control flow" in system
    assert "update, selection, or output policy" in system
    assert "Abstract benchmark-specific identifiers and instance sizes" in system
    assert "keep an instance size in the condition" in system


def test_retrieval_prompts_do_not_rank_on_dead_keyword_metadata():
    assert "keywords" not in load_prompt("retrieval_reflection", "system")
    assert "keywords" not in load_prompt("equivalence", "system")
