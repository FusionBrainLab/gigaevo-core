"""Pre-instantiate config compatibility guards.

Run against the raw Hydra ``cfg`` before ``instantiate`` so an incompatible
composition fails with a clear message instead of crashing deep inside an
interpolation or silently mis-wiring a shared object.
"""

from __future__ import annotations

import inspect
from typing import Any

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from gigaevo.config.helpers import build_archive_gate_provider
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.strategies.paired_selectors import (
    PairedBootstrapArchiveSelector,
)
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory_v2.candidates import (
    AgenticApplicabilityProvider,
    NullApplicabilityProvider,
    WholeBankCandidateSource,
)
from gigaevo.memory_v2.eviction import CausalRetirementEvictor
from gigaevo.memory_v2.writer import CausalV2ContentOnlyUpdater


def _target_path(symbol: Any) -> str:
    """Dotted ``_target_`` path derived from the symbol itself, so moving or
    renaming it breaks these guards loudly instead of silently skipping them."""
    return f"{symbol.__module__}.{symbol.__qualname__}"


_ARCHIVE_GATE_PROVIDER_TARGET = _target_path(build_archive_gate_provider)
_PAIRED_SELECTOR_TARGET = _target_path(PairedBootstrapArchiveSelector)
_MEMORY_V2_WRITER_TARGET = _target_path(CausalV2ContentOnlyUpdater)
_MEMORY_V2_WHOLE_BANK_SOURCE_TARGET = _target_path(WholeBankCandidateSource)
_MEMORY_V2_AGENTIC_APPLICABILITY_TARGET = _target_path(AgenticApplicabilityProvider)
_MEMORY_V2_NULL_APPLICABILITY_TARGET = _target_path(NullApplicabilityProvider)
_MEMORY_V2_EVICTOR_TARGET = _target_path(CausalRetirementEvictor)
_MEMORY_V2_NULL_EVICTOR_TARGET = _target_path(NullEvictor)
_MISSING = object()


def _raw_select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Select from ``cfg`` without resolving interpolations."""

    node: Any = OmegaConf.to_container(cfg, resolve=False)
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


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
            "memory=none for a non-reading run."
        )

    if pipeline_reads and not memory_reads:
        raise ValueError(
            f"pipeline={pipeline_id} reads external memory cards, but the selected "
            "memory preset has read=false. Use memory=v2; use pipeline=guided for "
            "no external-memory reads."
        )

    if (write_enabled or write_mode in {"end_of_run", "live"}) and not memory_writes:
        raise ValueError(
            f"memory/write={write_mode} requires a writer-enabled memory preset. "
            "Use memory=v2, or switch to memory/write=none."
        )

    if write_mode == "live":
        hook_target = _raw_select(cfg, "post_step_hook._target_", None)
        if hook_target != "gigaevo.memory.live_memory_hook.LiveMemoryRefreshHook":
            raise ValueError(
                "memory/write=live must install LiveMemoryRefreshHook as "
                "post_step_hook; check config/memory/write/live.yaml."
            )


def validate_memory_v2_scope(cfg: DictConfig) -> None:
    """Fail fast outside the first causally identified memory-v2 scope."""

    if not bool(_raw_select(cfg, "memory.capabilities.causal_v2", False)):
        return
    num_parents = int(_raw_select(cfg, "num_parents", 1))
    if num_parents != 1:
        raise ValueError(
            "memory=v2 currently requires num_parents=1. Per-parent decisions in "
            "crossover create post-treatment anchor selection and unidentified "
            "interference; v2 will support crossover only after the decision moves "
            "to the complete mutation-level parent slate."
        )
    if bool(_raw_select(cfg, "memory.coalesce_refresh", True)):
        raise ValueError(
            "memory=v2 requires memory.coalesce_refresh=false so every mutation "
            "attempt receives a distinct persisted behavior-policy decision."
        )
    if not bool(_raw_select(cfg, "pipeline_builder.fresh_context_reorder", False)):
        raise ValueError(
            "memory=v2 requires pipeline_builder.fresh_context_reorder=true so "
            "the frozen decision context and mutation baseline use the same "
            "fresh parent evaluation."
        )
    mutation_target = _raw_select(cfg, "mutation_operator._target_", None)
    try:
        mutation_operator = get_class(str(mutation_target))
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"memory=v2 cannot resolve mutation operator {mutation_target!r}"
        ) from exc
    if not issubclass(mutation_operator, MutationOperator) or inspect.isabstract(
        mutation_operator
    ):
        raise ValueError(
            "memory=v2 requires a concrete MutationOperator class, got "
            f"{mutation_operator!r}"
        )
    environment_operator = OmegaConf.select(
        cfg, "memory.environment.mutation_operator", default=None
    )
    if environment_operator is not mutation_operator:
        raise ValueError(
            "memory=v2 environment fingerprint must reference the configured "
            "concrete mutation operator class"
        )
    exploration = float(
        _raw_select(
            cfg,
            "memory.policy_config.proposal_exploration_probability",
            0.0,
        )
    )
    if exploration < 0.01:
        raise ValueError(
            "memory=v2 requires proposal_exploration_probability>=0.01 to "
            "preserve practical support for every eligible card."
        )
    offer_probability = float(
        _raw_select(cfg, "memory.policy_config.offer_probability", 0.0)
    )
    if not 0.10 <= offer_probability <= 0.90:
        raise ValueError(
            "memory=v2 requires offer_probability in [0.10, 0.90] for "
            "treatment/control overlap."
        )
    candidate_source_target = _raw_select(cfg, "memory.candidate_source._target_", None)
    if candidate_source_target != _MEMORY_V2_WHOLE_BANK_SOURCE_TARGET:
        raise ValueError(
            "memory=v2 candidate_source must be WholeBankCandidateSource: RAG "
            "may annotate the full bank but must not gate its candidate set."
        )
    applicability_target = _raw_select(cfg, "memory.applicability._target_", None)
    if applicability_target not in {
        _MEMORY_V2_AGENTIC_APPLICABILITY_TARGET,
        _MEMORY_V2_NULL_APPLICABILITY_TARGET,
    }:
        raise ValueError(
            "memory=v2 applicability must be AgenticApplicabilityProvider or "
            "NullApplicabilityProvider."
        )
    feature_enabled = bool(
        OmegaConf.select(
            cfg,
            "memory.feature_config.retrieval_applicability_contrast",
            default=False,
        )
    )
    applicability_feature_enabled = bool(
        _raw_select(
            cfg,
            "memory.applicability.retrieval_applicability_contrast",
            False,
        )
    )
    if feature_enabled != applicability_feature_enabled:
        raise ValueError(
            "memory=v2 feature_config.retrieval_applicability_contrast must "
            "match the applicability provider."
        )
    updater_target = _raw_select(cfg, "memory.causal_writer_updater._target_", None)
    if updater_target != _MEMORY_V2_WRITER_TARGET:
        raise ValueError(
            "memory=v2 requires the content-only causal writer updater; legacy "
            "gain restamping would create a second observational efficacy policy."
        )
    evictor_target = _raw_select(cfg, "memory.evictor._target_", None)
    if evictor_target not in {
        _MEMORY_V2_EVICTOR_TARGET,
        _MEMORY_V2_NULL_EVICTOR_TARGET,
    }:
        raise ValueError(
            "memory=v2 causal-retirement must use CausalRetirementEvictor or the "
            "explicit NullEvictor ablation; observational card-stat evictors are "
            "incompatible."
        )
    if evictor_target == _MEMORY_V2_EVICTOR_TARGET and bool(
        _raw_select(cfg, "memory.candidate_source.allow_cross_task", False)
    ):
        raise ValueError(
            "memory=v2 causal retirement requires allow_cross_task=false until "
            "retirement evidence is identified per source task."
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


def validate_paired_selector_pipeline_compat(cfg: DictConfig) -> None:
    """Reject the paired archive gate when no pipeline routes per-sample scores.

    ``PairedBootstrapArchiveSelector`` reads ``metadata["per_sample_scores"]``,
    which an artifact-aware pipeline populates from the reserved validator
    namespace. Under any other pipeline every comparison silently falls back to
    the point rule — an inert treatment the run would never surface.
    """

    group_target = _raw_select(cfg, "archive_selector._target_", None)

    def island_selector_target(island: Any) -> Any:
        selector = island.get("archive_selector") if isinstance(island, dict) else None
        if selector == "${archive_selector}":
            return group_target
        if isinstance(selector, dict):
            return selector.get("_target_")
        return None

    islands = _raw_select(cfg, "islands", default=None) or []
    if not any(island_selector_target(i) == _PAIRED_SELECTOR_TARGET for i in islands):
        return
    if bool(_raw_select(cfg, "pipeline.routes_program_metadata", False)):
        return
    pipeline_id = str(_raw_select(cfg, "pipeline.id", "<unknown>"))
    raise ValueError(
        "archive_selector=paired_bootstrap needs per_sample_scores on "
        f"program.metadata, but pipeline={pipeline_id} does not route "
        "_program_metadata. Pick a pipeline that declares "
        "routes_program_metadata: true and a problem whose validate() emits "
        "the vector, or archive_selector=point for a control arm."
    )


def validate_algorithm_requirements(cfg: DictConfig) -> None:
    """Enforce the config prerequisites an algorithm preset declares.

    Presets whose behavior-space keys are produced by opt-in wiring declare
    ``algorithm_requires: {<dotted.cfg.path>: <required value>}``; a mismatch
    would otherwise surface as a KeyError at first archive insert or, worse,
    as a silently degenerate behavior space. Domain knowledge stays in the
    preset yaml — this check is generic.
    """

    requirements = OmegaConf.select(cfg, "algorithm_requires", default=None) or {}
    for path, required in requirements.items():
        actual = OmegaConf.select(cfg, path, default=None)
        if actual != required:
            raise ValueError(
                f"this algorithm preset requires {path}={required}, got "
                f"{actual}. See algorithm_requires in the preset config."
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
