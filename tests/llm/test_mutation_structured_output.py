from gigaevo.llm.agents.mutation import MutationStructuredOutput


def _payload(**over):
    base = dict(
        archetype="Precision Optimization",
        justification="x",
        insights_used=[],
        changes=[],
        code="def f():\n    return 1\n",
    )
    base.update(over)
    return base


def test_base_parent_and_card_ids_used_round_trip():
    out = MutationStructuredOutput(
        **_payload(base_parent=2, card_ids_used=["program-a", "idea-b"])
    )
    assert out.base_parent == 2
    assert out.card_ids_used == ["program-a", "idea-b"]


def test_card_ids_used_defaults_empty_and_base_parent_defaults_to_one():
    out = MutationStructuredOutput(**_payload())
    assert out.card_ids_used == []
    assert out.base_parent == 1
