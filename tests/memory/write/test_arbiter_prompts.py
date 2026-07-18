"""Task provenance rendered for cross-task write-side LLM arbiters."""

from __future__ import annotations

from pathlib import Path

from gigaevo.llm.agents.consolidate_cards import ConsolidateAgent
from gigaevo.llm.agents.reconcile import ReconcileAgent

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "gigaevo" / "prompts"


def _uninitialized(agent_type, user_prompt_template):
    agent = object.__new__(agent_type)
    agent.system_prompt = "system"
    agent.user_prompt_template = user_prompt_template
    return agent


def test_reconcile_prompt_renders_nonempty_origin_tasks_only(make_card):
    agent = _uninitialized(
        ReconcileAgent,
        "note={note}\nneighbors:\n{neighbors}\n{base_parent_code}{child_code}{unified_diff}",
    )
    foreign = make_card(id="foreign", task_key="foreign-task")
    legacy = make_card(id="legacy", task_key="")

    state = agent.build_prompt(
        {
            "base_parent_code": "a",
            "child_code": "b",
            "unified_diff": "diff",
            "note": "mutation note",
            "task_key": "current-task",
            "neighbors": [foreign, legacy],
        }
    )

    prompt = state["messages"][1].content
    assert "origin task: current-task" in prompt
    assert "origin task: foreign-task" in prompt
    assert prompt.count("origin task:") == 2


def test_reconcile_prompt_is_unchanged_for_empty_task_keys(make_card):
    agent = _uninitialized(ReconcileAgent, "{note}\n{neighbors}")
    legacy = make_card(id="legacy", task_key="")

    state = agent.build_prompt(
        {
            "base_parent_code": "a",
            "child_code": "b",
            "unified_diff": "diff",
            "note": "mutation note",
            "task_key": "",
            "neighbors": [legacy],
        }
    )

    assert (
        state["messages"][1].content
        == f"mutation note\n- legacy: {legacy.description} | why: {legacy.explanation_summary}"
    )
    assert "origin task:" not in state["messages"][1].content


def test_consolidate_prompt_renders_nonempty_origin_tasks_only(make_card):
    agent = _uninitialized(ConsolidateAgent, "A={card_a}\nB={card_b}")
    foreign = make_card(task_key="foreign-task")
    legacy = make_card(task_key="")

    state = agent.build_prompt({"card_a": foreign, "card_b": legacy})

    prompt = state["messages"][1].content
    assert "origin task: foreign-task" in prompt
    assert prompt.count("origin task:") == 1


def test_consolidate_prompt_is_unchanged_for_empty_task_keys(make_card):
    agent = _uninitialized(ConsolidateAgent, "A={card_a}\nB={card_b}")
    card_a = make_card(task_key="")
    card_b = make_card(task_key="")

    state = agent.build_prompt({"card_a": card_a, "card_b": card_b})

    assert state["messages"][1].content == (
        f"A={card_a.description} | why: {card_a.explanation_summary}\n"
        f"B={card_b.description} | why: {card_b.explanation_summary}"
    )
    assert "origin task:" not in state["messages"][1].content


def test_retrieval_prompts_use_semantic_context_without_metadata_ranking():
    reflection = (_PROMPTS_DIR / "retrieval_reflection" / "system.txt").read_text()
    planner = (_PROMPTS_DIR / "retrieval_planner" / "system.txt").read_text()

    assert "task_description_summary" in reflection
    assert "origin_task" not in reflection
    assert "keywords" not in reflection
    assert "positive incremental utility" in planner
    assert "Task similarity alone is not utility" in planner


def test_arbiter_system_prompts_explain_cross_task_origin_lines():
    reconcile = (_PROMPTS_DIR / "reconcile" / "system.txt").read_text()
    consolidate = (_PROMPTS_DIR / "consolidate" / "system.txt").read_text()

    assert (
        'Neighbors may originate from different tasks (shown as "origin task"); '
        "MERGE/DUPLICATE across tasks only when the mechanism genuinely transfers beyond "
        "task-specific artifacts."
    ) in reconcile
    assert (
        'The two cards may originate from different tasks (shown as "origin task"); '
        "merge across tasks only when the mechanism genuinely transfers, never merely "
        "because prose overlaps."
    ) in consolidate
