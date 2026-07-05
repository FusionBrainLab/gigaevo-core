"""Unit tests for the full7 adaptive soft-coverage evaluator (no network)."""

from types import SimpleNamespace

import pytest

from problems.chains.hover.full7.validate import evaluate_soft_coverage_adaptive


def make_chain(step_types):
    return SimpleNamespace(steps=[SimpleNamespace(step_type=t) for t in step_types])


def make_result(step_outputs):
    return SimpleNamespace(step_outputs=step_outputs)


def passages(*titles):
    return "\n".join(f"[{i + 1}] {t} | body text" for i, t in enumerate(titles))


def test_scans_all_tool_steps():
    chain = make_chain(["llm", "tool", "llm", "tool"])
    dataset = [{"supporting_facts": ["Alpha Doc", "Beta Doc"]}]
    results = [
        make_result(
            [
                "draft text",
                passages("Alpha Doc"),
                "reasoning text",
                passages("Beta Doc"),
            ]
        )
    ]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [1.0]


def test_llm_step_mentioning_missing_gold_does_not_count():
    # The LLM step emits retrieval-shaped text naming the gold title the tools
    # never found; only tool outputs may earn coverage, so the score stays 0.5.
    chain = make_chain(["llm", "tool"])
    dataset = [{"supporting_facts": ["Alpha Doc", "Beta Doc"]}]
    results = [make_result([passages("Beta Doc"), passages("Alpha Doc")])]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [0.5]


def test_partial_coverage_is_fractional():
    chain = make_chain(["tool"])
    dataset = [{"supporting_facts": ["Alpha Doc", "Beta Doc"]}]
    results = [make_result([passages("Alpha Doc", "Unrelated")])]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [0.5]


def test_title_matching_is_normalized():
    chain = make_chain(["tool"])
    dataset = [{"supporting_facts": ["Alpha Doc!"]}]
    results = [make_result([passages("ALPHA doc")])]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [1.0]


def test_empty_gold_scores_one():
    chain = make_chain(["tool"])
    dataset = [{"supporting_facts": []}]
    results = [make_result([passages("Anything")])]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [1.0]


def test_truncated_step_outputs_do_not_crash():
    chain = make_chain(["llm", "tool", "tool"])
    dataset = [{"supporting_facts": ["Alpha Doc", "Beta Doc"]}]
    results = [make_result(["reasoning", passages("Alpha Doc")])]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == [0.5]


def test_per_sample_scores_align_with_dataset_order():
    chain = make_chain(["tool"])
    dataset = [
        {"supporting_facts": ["Alpha Doc"]},
        {"supporting_facts": ["Beta Doc"]},
    ]
    results = [
        make_result([passages("Alpha Doc")]),
        make_result([passages("Unrelated")]),
    ]
    assert evaluate_soft_coverage_adaptive(dataset, results, chain) == pytest.approx(
        [1.0, 0.0]
    )
