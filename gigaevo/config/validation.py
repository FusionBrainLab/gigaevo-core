"""Pre-instantiate config compatibility guards.

Run against the raw Hydra ``cfg`` before ``instantiate`` so an incompatible
composition fails with a clear message instead of crashing deep inside an
interpolation or silently mis-wiring a shared object.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

_SHARED_BEHAVIOR_SPACE_REF = "${ref:behavior_space}"
_DEFAULT_CHECKPOINT_DIR = "${hydra:runtime.output_dir}/memory"
_READER_PROVIDER_TARGET = "gigaevo.memory.provider.ReaderMemoryProvider"


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


def _raw_select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Select from ``cfg`` without resolving interpolations."""

    node: Any = OmegaConf.to_container(cfg, resolve=False)
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


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
            "BD-proximity memory reputation reads one ${ref:behavior_space}, but "
            f"this algorithm configures {len(islands)} islands with per-island "
            "behavior spaces and no top-level behavior_space for the ref to bind "
            "to. Use a single-island algorithm, or switch to "
            "memory/read_policy=portable for multi-island runs."
        )


def validate_memory_pipeline_compat(cfg: DictConfig) -> None:
    """Reject incompatible memory read/write and DAG pipeline pairings.

    External-memory read is a DAG capability: the pipeline must include
    ``MemoryContextStage``. External-memory write is an engine-hook capability:
    ``memory/write`` controls whether the engine installs only the finalizer or
    also the live refresh hook. This guard keeps those axes explicit instead of
    allowing silent no-op arms or late ``LiveMemoryRefreshHook`` type errors.
    """

    pipeline_id = str(_raw_select(cfg, "pipeline.id", "<unknown>"))
    pipeline_reads = bool(_raw_select(cfg, "pipeline.reads_external_memory", False))
    memory_reads = bool(_raw_select(cfg, "memory.capabilities.read", False))
    memory_writes = bool(_raw_select(cfg, "memory.capabilities.write", False))
    write_mode = str(_raw_select(cfg, "memory.write.mode", "off"))
    write_enabled = bool(_raw_select(cfg, "memory.write.enabled", False))

    if memory_reads and not pipeline_reads:
        raise ValueError(
            f"memory read is enabled but pipeline={pipeline_id} does not read "
            "external memory cards. Use pipeline=memory_guided, or switch to "
            "memory=none / memory=writer for a non-reading run."
        )

    if pipeline_reads and not memory_reads:
        raise ValueError(
            f"pipeline={pipeline_id} reads external memory cards, but the selected "
            "memory preset has read=false. Use memory=reader, memory=full, or "
            "memory=static; use pipeline=guided for no external-memory reads."
        )

    if (write_enabled or write_mode in {"end_of_run", "live"}) and not memory_writes:
        raise ValueError(
            f"memory/write={write_mode} requires a writer-enabled memory preset. "
            "Use memory=writer or memory=full, or switch to memory/write=none."
        )

    if write_mode == "live":
        hook_target = _raw_select(cfg, "post_step_hook._target_", None)
        if hook_target != "gigaevo.memory.live_memory_hook.LiveMemoryRefreshHook":
            raise ValueError(
                "memory/write=live must install LiveMemoryRefreshHook as "
                "post_step_hook; check config/memory/write/live.yaml."
            )

    provider_target = _raw_select(cfg, "memory.provider._target_", None)
    checkpoint_dir = _raw_select(cfg, "checkpoint_dir", None)
    if (
        memory_reads
        and not memory_writes
        and provider_target == _READER_PROVIDER_TARGET
        and checkpoint_dir == _DEFAULT_CHECKPOINT_DIR
    ):
        raise ValueError(
            "memory=reader is read-only and should point at an existing bank. "
            "Set checkpoint_dir=/path/to/shared/bank, or use memory=full when "
            "the same run should create the bank."
        )
