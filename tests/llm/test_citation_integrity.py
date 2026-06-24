from langchain_core.messages import HumanMessage

from gigaevo.llm.agents.mutation import MutationAgent


def _integrity(insights, cards, prompt):
    return MutationAgent._citation_integrity(
        insights, cards, [HumanMessage(content=prompt)]
    )


def test_card_ids_used_counted_against_prompt():
    res = _integrity([], ["program-abc", "ghost-1"], "consider card program-abc here")
    assert res["cards_cited"] == 2
    assert res["cards_grounded"] == 1


def test_insights_still_counted():
    res = _integrity(["raise temperature"], [], "you could raise temperature now")
    assert res["cited"] == 1
    assert res["grounded"] == 1
