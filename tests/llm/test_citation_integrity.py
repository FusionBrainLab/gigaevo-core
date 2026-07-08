"""Exact grounding of cited insight and card references against the prompt.

Each parent block ('=== Parent N ===') carries its own '## Program Insights'
list numbered from 1, so the mutator cites insights as (parent, insight)
pairs via ``insight_ids_used`` (structured output) and exact card ids via
``card_ids_used``; grounding checks each citation was actually offered in the
rendered prompt — the insight inside that parent's block. Counts land in
child metadata for run-level statistics; nothing is gated on them.
"""

from langchain_core.messages import HumanMessage

from gigaevo.evolution.mutation.context import InsightsMutationContext
from gigaevo.llm.agents.insights import ProgramInsight, ProgramInsights
from gigaevo.llm.agents.mutation import InsightCitation, MutationAgent


def _integrity(citations, cards, prompt):
    return MutationAgent._citation_integrity(
        citations, cards, [HumanMessage(content=prompt)]
    )


def _rendered_insights(n: int) -> str:
    insights = ProgramInsights(
        insights=[
            ProgramInsight(
                type="threshold_tuning",
                tag="rigid",
                severity="medium",
                insight=f"tune threshold {i}",
            )
            for i in range(n)
        ]
    )
    return InsightsMutationContext(insights=insights).format()


def _prompt(*insights_per_parent: int) -> str:
    blocks = [
        f"=== Parent {i} ===\n```python\ncode\n```\n\n{_rendered_insights(n)}\n"
        for i, n in enumerate(insights_per_parent, start=1)
    ]
    return "\n\n".join(blocks)


def test_cited_pairs_grounded_against_rendered_insight_list():
    citations = [
        InsightCitation(parent=1, insight=1),
        InsightCitation(parent=1, insight=2),
    ]
    res = _integrity(citations, [], _prompt(3))
    assert res["cited"] == 2
    assert res["grounded"] == 2


def test_insight_beyond_offered_list_is_ungrounded():
    citations = [
        InsightCitation(parent=1, insight=2),
        InsightCitation(parent=1, insight=7),
    ]
    res = _integrity(citations, [], _prompt(2))
    assert res["cited"] == 2
    assert res["grounded"] == 1


def test_two_parents_ground_independently():
    citations = [
        InsightCitation(parent=1, insight=3),
        InsightCitation(parent=2, insight=3),
    ]
    res = _integrity(citations, [], _prompt(3, 1))
    assert res["cited"] == 2
    assert res["grounded"] == 1


def test_parent_beyond_offered_blocks_is_ungrounded():
    res = _integrity([InsightCitation(parent=3, insight=1)], [], _prompt(2, 2))
    assert res["cited"] == 1
    assert res["grounded"] == 0


def test_no_insights_section_grounds_nothing():
    res = _integrity(
        [InsightCitation(parent=1, insight=1)],
        [],
        "=== Parent 1 ===\nmutate the program; no insights offered",
    )
    assert res["cited"] == 1
    assert res["grounded"] == 0


def test_non_positive_numbers_ignored():
    citations = [
        InsightCitation(parent=0, insight=1),
        InsightCitation(parent=1, insight=-3),
        InsightCitation(parent=1, insight=1),
    ]
    res = _integrity(citations, [], _prompt(1))
    assert res["cited"] == 1
    assert res["grounded"] == 1


def test_card_ids_used_grounded_only_against_explicit_card_renderers():
    prompt = (
        "free text mentioning program-abc is not enough\n"
        "[card 1] id=mem-a\n"
        "description\n"
        "1. **[x][rigid][medium]** — mechanism: x | card: program-abc"
    )
    res = _integrity([], ["program-abc", "mem-a", "ghost-1"], prompt)
    assert res["cards_cited"] == 3
    assert res["cards_grounded"] == 2


def test_empty_citations():
    res = _integrity([], [], "any prompt")
    assert res == {"cited": 0, "grounded": 0, "cards_cited": 0, "cards_grounded": 0}
