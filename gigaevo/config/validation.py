"""Pre-instantiate config compatibility guards.

Run against the raw Hydra ``cfg`` before ``instantiate`` so an incompatible
composition fails with a clear message instead of crashing deep inside an
interpolation or silently mis-wiring a shared object.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

_SHARED_BEHAVIOR_SPACE_REF = "${ref:behavior_space}"


def _references_shared_behavior_space(node: Any) -> bool:
    """True if any leaf in ``node`` interpolates the shared ``behavior_space``.

    Matches the structural dependency, not a class name — so a subclass, an
    import alias, or a BD reputation nested as another reputation's ``fallback``
    is caught the same as a top-level ``bd_proximity``.
    """
    if isinstance(node, str):
        return _SHARED_BEHAVIOR_SPACE_REF in node
    if isinstance(node, dict):
        return any(_references_shared_behavior_space(v) for v in node.values())
    if isinstance(node, list):
        return any(_references_shared_behavior_space(v) for v in node)
    return False


def validate_reputation_island_compat(cfg: DictConfig) -> None:
    """Reject a multi-island algorithm paired with BD-proximity reputation.

    ``BDProximityReputation`` resolves a single ``${ref:behavior_space}`` and
    partitions every card's gain events by that one tessellation. A multi-island
    algorithm has a per-island behavior space and no top-level ``behavior_space``
    for the ref to bind to, so the pairing has no coherent meaning — fail fast.
    """
    islands = OmegaConf.select(cfg, "islands", default=None)
    if not islands or len(islands) <= 1:
        return
    reputation = OmegaConf.select(cfg, "memory.reputation", default=None)
    if reputation is None:
        return
    raw = OmegaConf.to_container(reputation, resolve=False)
    if _references_shared_behavior_space(raw):
        raise NotImplementedError(
            "memory/reputation=bd_proximity reads one ${ref:behavior_space}, but "
            f"this algorithm configures {len(islands)} islands with per-island "
            "behavior spaces and no top-level behavior_space for the ref to bind "
            "to. Use a single-island algorithm, or a non-BD reputation for "
            "multi-island runs."
        )
