from __future__ import annotations

from gigaevo.memory._vendor.GAM_root.gam.agents.research_agent import (
    _drop_random_ideas,
)


def _ideas(n):
    return [{"card_id": f"c{i}", "score": float(n - i)} for i in range(n)]


def test_dose_zero_is_identity():
    ideas = _ideas(5)
    assert _drop_random_ideas(ideas, 0, seed_basis="x") == ideas


def test_empty_pool_is_identity():
    assert _drop_random_ideas([], 3, seed_basis="x") == []


def test_drops_exactly_dose_cards():
    kept = _drop_random_ideas(_ideas(5), 2, seed_basis="x")
    assert len(kept) == 3


def test_top_ranked_card_survives_partial_drop():
    kept = _drop_random_ideas(_ideas(5), 3, seed_basis="x")
    assert kept[0]["card_id"] == "c0"


def test_dose_at_or_above_pool_empties_slate():
    assert _drop_random_ideas(_ideas(4), 4, seed_basis="x") == []
    assert _drop_random_ideas(_ideas(4), 9, seed_basis="x") == []


def test_drop_is_deterministic_under_fixed_seed_basis():
    a = _drop_random_ideas(_ideas(8), 3, seed_basis="same")
    b = _drop_random_ideas(_ideas(8), 3, seed_basis="same")
    assert [i["card_id"] for i in a] == [i["card_id"] for i in b]


def test_drop_preserves_relative_order_of_survivors():
    kept = _drop_random_ideas(_ideas(8), 3, seed_basis="x")
    ids = [i["card_id"] for i in kept]
    assert ids == sorted(ids, key=lambda c: int(c[1:]))
