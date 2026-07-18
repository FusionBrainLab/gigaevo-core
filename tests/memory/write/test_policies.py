from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.memory.write.policies import DedupPolicy, ProgramExemplarPolicy


def test_dedup_policy_is_bounded_and_frozen() -> None:
    policy = DedupPolicy(online_top_k=3)
    assert policy.online_top_k == 3
    with pytest.raises(ValidationError):
        policy.online_top_k = 4
    with pytest.raises(ValidationError):
        DedupPolicy(online_top_k=-1)


def test_program_exemplar_policy_is_bounded_and_frozen() -> None:
    policy = ProgramExemplarPolicy(
        enabled=True,
        top_k_per_refresh=2,
        max_cards=8,
        min_fitness_gap=0.1,
        store_code=False,
    )
    assert policy.top_k_per_refresh == 2
    with pytest.raises(ValidationError):
        policy.max_cards = 7
    with pytest.raises(ValidationError):
        ProgramExemplarPolicy(top_k_per_refresh=-1)
    with pytest.raises(ValidationError):
        ProgramExemplarPolicy(max_cards=-1)
