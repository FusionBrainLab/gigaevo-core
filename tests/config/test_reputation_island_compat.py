"""BD-proximity reputation is single-island only.

``BDProximityReputation`` reads one ``${ref:behavior_space}`` tessellation, but a
multi-island algorithm defines a per-island behavior space (and no top-level
``behavior_space`` for the ref to resolve). Composing the two used to crash deep
inside Hydra interpolation; the guard fails fast with a clear NotImplementedError.

The guard keys on the *structural* dependency — any ``${ref:behavior_space}``
leaf in the reputation subtree — so a subclass, alias, or nested ``fallback``
cannot slip past an exact class-name match.
"""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_reputation_island_compat

_REF = "${ref:behavior_space}"
_BD = "gigaevo.memory.core.bd_proximity.BDProximityReputation"
_RW = "gigaevo.memory.core.reward.RewardWeightedReputation"


def _cfg(*, n_islands: int, reputation: dict | None) -> object:
    islands = [{"island_id": f"i{i}"} for i in range(n_islands)]
    memory: dict = {}
    if reputation is not None:
        memory["reputation"] = reputation
    return OmegaConf.create({"islands": islands, "memory": memory})


def test_multi_island_with_bd_reputation_raises() -> None:
    rep = {"_target_": _BD, "behavior_space": _REF}
    with pytest.raises(NotImplementedError, match="single-island|behavior_space"):
        validate_reputation_island_compat(_cfg(n_islands=2, reputation=rep))


def test_multi_island_with_nested_behavior_space_ref_raises() -> None:
    rep = {"_target_": _RW, "fallback": {"_target_": _BD, "behavior_space": _REF}}
    with pytest.raises(NotImplementedError):
        validate_reputation_island_compat(_cfg(n_islands=2, reputation=rep))


def test_single_island_with_bd_reputation_is_allowed() -> None:
    rep = {"_target_": _BD, "behavior_space": _REF}
    validate_reputation_island_compat(_cfg(n_islands=1, reputation=rep))


def test_multi_island_with_non_bd_reputation_is_allowed() -> None:
    validate_reputation_island_compat(_cfg(n_islands=2, reputation={"_target_": _RW}))


def test_multi_island_without_reputation_is_allowed() -> None:
    validate_reputation_island_compat(_cfg(n_islands=2, reputation=None))


def test_missing_islands_key_is_allowed() -> None:
    validate_reputation_island_compat(
        OmegaConf.create(
            {"memory": {"reputation": {"_target_": _BD, "behavior_space": _REF}}}
        )
    )
