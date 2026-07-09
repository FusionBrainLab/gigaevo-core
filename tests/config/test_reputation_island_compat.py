"""BD-local memory components are single-island only.

BD-local memory components read one ``${ref:behavior_space}`` tessellation, but
a multi-island algorithm defines a per-island behavior space (and no top-level
``behavior_space`` for the ref to resolve). Composing the two used to crash deep
inside Hydra interpolation; the guard fails fast with a clear NotImplementedError.

The guard keys on the *structural* dependency — any ``${ref:behavior_space}``
leaf in the memory subtree — so a subclass, alias, or nested ``fallback`` cannot
slip past an exact class-name match.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_reputation_island_compat

_REF = "${ref:behavior_space}"
_BD = "gigaevo.memory.read.reputation.BDProximityReputation"
_BB = "gigaevo.memory.read.reputation.BetaBinomialReputation"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(
    *, n_islands: int, reputation: dict | None, context_model: dict | None = None
) -> object:
    islands = [{"island_id": f"i{i}"} for i in range(n_islands)]
    memory: dict = {}
    if reputation is not None:
        memory["reputation"] = reputation
    if context_model is not None:
        memory["context_model"] = context_model
    return OmegaConf.create({"islands": islands, "memory": memory})


def _reputation_from_policy(policy: str):
    read_policy = OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "read_policy" / f"{policy}.yaml"
    )
    defaults = OmegaConf.to_container(read_policy.defaults, resolve=False)
    reputation_name = next(
        item["/memory/reputation"]
        for item in defaults
        if isinstance(item, dict) and "/memory/reputation" in item
    )
    return OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "reputation" / f"{reputation_name}.yaml"
    )


def test_multi_island_with_bd_reputation_raises() -> None:
    rep = {"_target_": _BD, "behavior_space": _REF}
    with pytest.raises(NotImplementedError, match="single-island|behavior_space"):
        validate_reputation_island_compat(_cfg(n_islands=2, reputation=rep))


def test_multi_island_with_nested_behavior_space_ref_raises() -> None:
    rep = {"_target_": _BB, "fallback": {"_target_": _BD, "behavior_space": _REF}}
    with pytest.raises(NotImplementedError):
        validate_reputation_island_compat(_cfg(n_islands=2, reputation=rep))


def test_multi_island_with_bd_context_model_raises() -> None:
    context = {
        "_target_": "gigaevo.memory.context.models.BDCellMemoryContext",
        "behavior_space": _REF,
    }
    with pytest.raises(NotImplementedError):
        validate_reputation_island_compat(
            _cfg(n_islands=2, reputation={"_target_": _BB}, context_model=context)
        )


def test_single_island_with_bd_reputation_is_allowed() -> None:
    rep = {"_target_": _BD, "behavior_space": _REF}
    validate_reputation_island_compat(_cfg(n_islands=1, reputation=rep))


def test_multi_island_with_non_bd_reputation_is_allowed() -> None:
    validate_reputation_island_compat(_cfg(n_islands=2, reputation={"_target_": _BB}))


def test_multi_island_without_reputation_is_allowed() -> None:
    validate_reputation_island_compat(_cfg(n_islands=2, reputation=None))


def test_missing_islands_key_is_allowed() -> None:
    validate_reputation_island_compat(
        OmegaConf.create(
            {"memory": {"reputation": {"_target_": _BD, "behavior_space": _REF}}}
        )
    )


def test_multi_island_with_recommended_read_policy_raises() -> None:
    cfg = _cfg(n_islands=2, reputation=None)
    cfg.memory.reputation = _reputation_from_policy("recommended")
    with pytest.raises(NotImplementedError, match="behavior_space"):
        validate_reputation_island_compat(cfg)


def test_multi_island_with_portable_read_policy_is_allowed() -> None:
    cfg = _cfg(n_islands=2, reputation=None)
    cfg.memory.reputation = _reputation_from_policy("portable")
    validate_reputation_island_compat(cfg)
