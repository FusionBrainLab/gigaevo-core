"""Unit tests for the pipeline preset builders."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gigaevo.config.defaults import DEFAULT_DAG_TIMEOUT_S, DEFAULT_STAGE_TIMEOUT_S
from gigaevo.config.pipeline_presets import (
    ProblemSpecificPipelineBuilderConfig,
    build_algotune_speed,
    build_auto,
    build_cma_opt,
    build_hotpotqa_asi,
    build_hotpotqa_colbert,
    build_hotpotqa_reflective,
    build_hover_feedback,
    build_optuna_opt,
    build_standard,
    build_structural_metrics,
    build_with_context,
)
from gigaevo.config.schemas import (
    AlgoTuneSpeedPipelineBuilderConfig,
    AutoPipelineBuilderConfig,
    CMAOptPipelineBuilderConfig,
    ContextPipelineBuilderConfig,
    DefaultPipelineBuilderConfig,
    OptunaOptPipelineBuilderConfig,
    PipelineConfig,
    StructuralMetricsPipelineBuilderConfig,
)


class TestStandardPresets:
    @pytest.mark.parametrize(
        "builder_func,expected_type",
        [
            (build_standard, DefaultPipelineBuilderConfig),
            (build_with_context, ContextPipelineBuilderConfig),
            (build_auto, AutoPipelineBuilderConfig),
            (build_algotune_speed, AlgoTuneSpeedPipelineBuilderConfig),
            (build_cma_opt, CMAOptPipelineBuilderConfig),
            (build_optuna_opt, OptunaOptPipelineBuilderConfig),
            (build_structural_metrics, StructuralMetricsPipelineBuilderConfig),
        ],
    )
    def test_each_preset_uses_correct_builder_variant(
        self, builder_func, expected_type
    ) -> None:
        cfg = builder_func()
        assert isinstance(cfg, PipelineConfig)
        assert isinstance(cfg.builder, expected_type)

    def test_default_timeouts_match_constants(self) -> None:
        cfg = build_standard()
        assert cfg.builder.dag_timeout == DEFAULT_DAG_TIMEOUT_S
        assert cfg.builder.stage_timeout == DEFAULT_STAGE_TIMEOUT_S

    def test_custom_timeouts_propagate(self) -> None:
        cfg = build_standard(dag_timeout=1200.0, stage_timeout=300.0)
        assert cfg.builder.dag_timeout == 1200.0
        assert cfg.builder.stage_timeout == 300.0

    def test_prompts_dir_propagates(self) -> None:
        cfg = build_with_context(prompts_dir=Path("/srv/prompts"))
        assert cfg.prompts_dir == Path("/srv/prompts")


class TestProblemSpecificPipelineBuilder:
    def test_construct(self) -> None:
        cfg = ProblemSpecificPipelineBuilderConfig(
            builder_path="problems.chains.hotpotqa.static.pipeline.ReflectivePipelineBuilder"
        )
        assert cfg.kind == "problem_specific"
        assert cfg.dag_timeout == 3600.0
        assert cfg.stage_timeout == 300.0

    def test_empty_builder_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProblemSpecificPipelineBuilderConfig(builder_path="")

    def test_build_rejects_non_dotted_path(self) -> None:
        cfg = ProblemSpecificPipelineBuilderConfig(builder_path="NoDots")
        with pytest.raises(ValueError, match="fully-qualified"):
            cfg.build(ctx=None)  # type: ignore[arg-type]

    def test_build_raises_clean_error_on_missing_module(self) -> None:
        cfg = ProblemSpecificPipelineBuilderConfig(
            builder_path="nonexistent.module.Builder"
        )
        with pytest.raises(ModuleNotFoundError):
            cfg.build(ctx=None)  # type: ignore[arg-type]


class TestProblemSpecificPresets:
    @pytest.mark.parametrize(
        "builder_func,expected_path",
        [
            (
                build_hotpotqa_reflective,
                "problems.chains.hotpotqa.static.pipeline.ReflectivePipelineBuilder",
            ),
            (
                build_hotpotqa_asi,
                "problems.chains.hotpotqa.static_a.pipeline.ASIPipelineBuilder",
            ),
            (
                build_hotpotqa_colbert,
                "problems.chains.hotpotqa.static_colbert_f1_600.pipeline.ColBERTPipelineBuilder",
            ),
            (
                build_hover_feedback,
                "problems.chains.hover.static.pipeline.HoVerFeedbackPipelineBuilder",
            ),
        ],
    )
    def test_each_problem_specific_preset_targets_correct_builder(
        self, builder_func, expected_path: str
    ) -> None:
        cfg = builder_func()
        assert isinstance(cfg, PipelineConfig)
        assert isinstance(cfg.builder, ProblemSpecificPipelineBuilderConfig)
        assert cfg.builder.builder_path == expected_path


class TestPipelineConfigRoundTrip:
    """The standard presets dispatch through the discriminated union;
    the problem-specific preset is a custom schema outside that union
    (it lives on PipelineConfig.builder as a parallel descriptor that
    PipelineConfig accepts via Union[...] expansion). Verify both
    paths round-trip cleanly."""

    def test_standard_preset_json_round_trips(self) -> None:
        cfg = build_with_context(dag_timeout=1500.0)
        parsed = PipelineConfig.model_validate_json(cfg.model_dump_json())
        assert isinstance(parsed.builder, ContextPipelineBuilderConfig)
        assert parsed.builder.dag_timeout == 1500.0


class TestProblemSpecificPathsImportCleanly:
    """Defense-in-depth: every problem-specific preset must point at
    a path that resolves to a real class on disk. Catches renames or
    moves of the problem-directory pipeline modules that would only
    otherwise fail at experiment startup."""

    @pytest.mark.parametrize(
        "builder_func",
        [
            build_hotpotqa_reflective,
            build_hotpotqa_asi,
            build_hotpotqa_colbert,
            build_hover_feedback,
        ],
    )
    def test_builder_path_resolves_in_subprocess(self, builder_func) -> None:
        """The ``problems`` namespace package and its descendants are
        implicit PEP-420 packages without ``__init__.py``. pytest's
        sys.path manipulation interferes with namespace package
        discovery — a fresh subprocess with the repo root on sys.path
        is the clean way to verify each preset's ``builder_path``
        resolves to a real class on disk. Catches preset rot from
        renames or moves of the problem-directory pipeline modules."""
        import subprocess
        import sys

        cfg = builder_func()
        builder = cfg.builder
        assert isinstance(builder, ProblemSpecificPipelineBuilderConfig)
        module_name, _, class_name = builder.builder_path.rpartition(".")
        repo_root = Path(__file__).resolve().parents[2]
        script = (
            "import sys, importlib;"
            f"sys.path.insert(0, {str(repo_root)!r});"
            f"m = importlib.import_module({module_name!r});"
            f"cls = getattr(m, {class_name!r}, None);"
            "sys.exit(0 if cls is not None else 1)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"{builder.builder_path} did not resolve. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


class TestUniqueDiscriminators:
    def test_problem_specific_kind_distinct_from_built_in_variants(self) -> None:
        """``problem_specific`` must not collide with the seven
        kind literals on PipelineBuilderConfig — a duplicate would
        let one variant silently shadow the other during round trip."""
        existing = {
            "default",
            "context",
            "algotune_speed",
            "cma_opt",
            "optuna_opt",
            "structural_metrics",
            "auto",
        }
        assert (
            ProblemSpecificPipelineBuilderConfig.model_fields["kind"].default
            not in existing
        )
