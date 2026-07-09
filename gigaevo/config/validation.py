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
_ARCHIVE_GATE_PROVIDER_TARGET = "gigaevo.config.helpers.build_archive_gate_provider"
_MISSING = object()


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
    """Reject multi-island memory configs that require one shared behavior space.

    BD-local memory components resolve a single ``${ref:behavior_space}`` and
    partition card/no-card evidence by that one tessellation. A multi-island
    algorithm has a per-island behavior space and no top-level ``behavior_space``
    for the ref to bind to, so the pairing has no coherent meaning — fail fast.
    """
    islands = OmegaConf.select(cfg, "islands", default=None)
    if not islands or len(islands) <= 1:
        return
    memory = OmegaConf.select(cfg, "memory", default=None)
    if memory is None:
        return
    raw = OmegaConf.to_container(memory, resolve=False)
    if _references_shared_behavior_space(raw):
        raise NotImplementedError(
            "BD-local memory components read one ${ref:behavior_space}, but "
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
        and provider_target == _READER_PROVIDER_TARGET
        and checkpoint_dir == _DEFAULT_CHECKPOINT_DIR
        and write_mode != "live"
    ):
        raise ValueError(
            "memory reads from the default per-run bank, but no live writer will "
            "populate that bank during the run. Set checkpoint_dir=/path/to/an/"
            "existing/bank, use memory/write=live for same-run read+write, or "
            "use pipeline=guided memory=writer to build a bank for a later run."
        )


def validate_archive_gate_pipeline_compat(cfg: DictConfig) -> None:
    """Reject archive-gate settings that are not wired into the selected pipeline."""

    enabled = bool(OmegaConf.select(cfg, "archive_gate_enabled", default=False))
    if not enabled:
        return

    pipeline_id = str(_raw_select(cfg, "pipeline.id", "<unknown>"))
    mode = str(_raw_select(cfg, "pipeline.archive_gate_mode", "none"))

    if mode == "builder":
        provider_target = _raw_select(cfg, "archive_gate_provider._target_", None)
        if provider_target != _ARCHIVE_GATE_PROVIDER_TARGET:
            raise ValueError(
                f"pipeline={pipeline_id} declares archive_gate_mode=builder, but "
                "archive_gate_provider is not wired. Include "
                "/pipeline_feature/archive_gate in the pipeline config."
            )
        context_provider = _raw_select(
            cfg, "evolution_context.archive_gate_provider", None
        )
        if context_provider != "${ref:archive_gate_provider}":
            raise ValueError(
                f"pipeline={pipeline_id} declares archive_gate_mode=builder, but "
                "evolution_context.archive_gate_provider is not "
                "${ref:archive_gate_provider}."
            )
        builder_flag = _raw_select(
            cfg, "pipeline_builder.archive_gate_enabled", _MISSING
        )
        if builder_flag is _MISSING:
            raise ValueError(
                f"pipeline={pipeline_id} declares archive_gate_mode=builder, but "
                "pipeline_builder.archive_gate_enabled is missing."
            )
        return

    if mode == "declarative":
        gate_node = _raw_select(
            cfg, "dag_blueprint.nodes.ArchivePotentialGateStage", _MISSING
        )
        if gate_node is _MISSING:
            raise ValueError(
                f"pipeline={pipeline_id} declares archive_gate_mode=declarative, "
                "but dag_blueprint.nodes.ArchivePotentialGateStage is missing."
            )
        return

    if mode == "none":
        raise ValueError(
            f"archive_gate_enabled=true, but pipeline={pipeline_id} declares "
            "archive_gate_mode=none. Use pipeline=guided/memory_guided/optuna_opt, "
            "set archive_gate_enabled=false, or wire ArchivePotentialGateStage "
            "explicitly in the custom pipeline."
        )

    raise ValueError(
        f"pipeline={pipeline_id} has unsupported archive_gate_mode={mode!r}. "
        "Expected builder, declarative, or none."
    )


def validate_program_format_pipeline_compat(cfg: DictConfig) -> None:
    """Reject program-format choices that the selected pipeline cannot consume."""

    program_format = str(_raw_select(cfg, "program_format.id", "python_source"))
    if program_format == "python_source":
        return

    pipeline_id = str(_raw_select(cfg, "pipeline.id", "<unknown>"))
    if bool(OmegaConf.select(cfg, "enable_optuna_stage", default=False)):
        raise ValueError(
            f"program_format={program_format} cannot be used with "
            "enable_optuna_stage=true; the current Optuna stage optimizes "
            "Python source programs only."
        )
    if pipeline_id == "optuna_opt":
        raise ValueError(
            f"program_format={program_format} cannot be used with "
            "pipeline=optuna_opt; the current Optuna pipeline optimizes Python "
            "source programs only."
        )

    feature_ref = _raw_select(cfg, "pipeline_builder.program_format_feature", None)
    if feature_ref is None:
        raise ValueError(
            f"program_format={program_format} is selected, but pipeline={pipeline_id} "
            "does not consume program_format.evaluation_feature. Use "
            "pipeline=guided or pipeline=memory_guided, or add explicit support "
            "to the custom pipeline."
        )
