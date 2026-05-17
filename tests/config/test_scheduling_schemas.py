"""Unit tests for the scheduling-side typed schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    ChainFeatureExtractorConfig,
    CodeFeatureExtractorConfig,
    FIFOConfig,
    FeatureExtractorConfig,
    LPTConfig,
    PredictorConfig,
    RidgePredictorConfig,
    SchedulingConfig,
    SimpleHeuristicPredictorConfig,
)


class TestFeatureExtractorConfigs:
    def test_code_extractor_kind(self) -> None:
        assert CodeFeatureExtractorConfig().kind == "code"

    def test_chain_extractor_kind(self) -> None:
        assert ChainFeatureExtractorConfig().kind == "chain"

    def test_union_round_trip(self) -> None:
        ta: TypeAdapter[FeatureExtractorConfig] = TypeAdapter(FeatureExtractorConfig)
        for cfg in (CodeFeatureExtractorConfig(), ChainFeatureExtractorConfig()):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_code_extractor_builds_runtime(self) -> None:
        from gigaevo.evolution.scheduling.feature_extractor import (
            CodeFeatureExtractor,
        )

        assert isinstance(CodeFeatureExtractorConfig().build(), CodeFeatureExtractor)

    def test_chain_extractor_builds_runtime(self) -> None:
        from gigaevo.evolution.scheduling.feature_extractor import (
            ChainFeatureExtractor,
        )

        assert isinstance(
            ChainFeatureExtractorConfig().build(), ChainFeatureExtractor
        )


class TestSimpleHeuristicPredictor:
    def test_default_construct(self) -> None:
        cfg = SimpleHeuristicPredictorConfig()
        assert cfg.kind == "simple_heuristic"
        assert cfg.default_rate == 0.1
        assert cfg.window_size == 50

    def test_default_rate_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimpleHeuristicPredictorConfig(default_rate=0.0)

    def test_window_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimpleHeuristicPredictorConfig(window_size=0)

    def test_builds_runtime_predictor(self) -> None:
        from gigaevo.evolution.scheduling.predictor import (
            SimpleHeuristicPredictor,
        )

        predictor = SimpleHeuristicPredictorConfig(
            default_rate=0.2, window_size=30
        ).build()
        assert isinstance(predictor, SimpleHeuristicPredictor)


class TestRidgePredictor:
    def test_default_construct(self) -> None:
        cfg = RidgePredictorConfig()
        assert cfg.kind == "ridge"
        assert isinstance(cfg.feature_extractor, CodeFeatureExtractorConfig)
        assert cfg.buffer_size == 500
        assert cfg.alpha == 1.0

    def test_chain_feature_extractor_override(self) -> None:
        cfg = RidgePredictorConfig(feature_extractor=ChainFeatureExtractorConfig())
        assert isinstance(cfg.feature_extractor, ChainFeatureExtractorConfig)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RidgePredictorConfig(alpha=0.0)

    def test_default_prediction_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RidgePredictorConfig(default_prediction=0.0)

    def test_builds_runtime_predictor(self) -> None:
        from gigaevo.evolution.scheduling.predictor import RidgePredictor

        predictor = RidgePredictorConfig().build()
        assert isinstance(predictor, RidgePredictor)


class TestPredictorUnion:
    def test_round_trip_each_variant(self) -> None:
        ta: TypeAdapter[PredictorConfig] = TypeAdapter(PredictorConfig)
        for cfg in (
            SimpleHeuristicPredictorConfig(),
            RidgePredictorConfig(feature_extractor=ChainFeatureExtractorConfig()),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)


class TestFIFOConfig:
    def test_kind(self) -> None:
        assert FIFOConfig().kind == "fifo"

    def test_builds_runtime_prioritizer(self) -> None:
        from gigaevo.evolution.scheduling.prioritizer import FIFOPrioritizer

        assert isinstance(FIFOConfig().build(), FIFOPrioritizer)


class TestLPTConfig:
    def test_default_uses_simple_heuristic(self) -> None:
        cfg = LPTConfig()
        assert cfg.kind == "lpt"
        assert isinstance(cfg.eval_predictor, SimpleHeuristicPredictorConfig)

    def test_ridge_chain_override(self) -> None:
        cfg = LPTConfig(
            eval_predictor=RidgePredictorConfig(
                feature_extractor=ChainFeatureExtractorConfig(),
                buffer_size=500,
                min_samples=10,
                default_prediction=300.0,
                alpha=1.0,
            )
        )
        assert isinstance(cfg.eval_predictor, RidgePredictorConfig)
        assert isinstance(
            cfg.eval_predictor.feature_extractor, ChainFeatureExtractorConfig
        )

    def test_builds_runtime_prioritizer(self) -> None:
        from gigaevo.evolution.scheduling.prioritizer import LPTPrioritizer

        cfg = LPTConfig(
            eval_predictor=SimpleHeuristicPredictorConfig(
                default_rate=0.1, window_size=50
            )
        )
        prioritizer = cfg.build()
        assert isinstance(prioritizer, LPTPrioritizer)


class TestSchedulingUnion:
    def test_round_trip_fifo(self) -> None:
        ta: TypeAdapter[SchedulingConfig] = TypeAdapter(SchedulingConfig)
        cfg = FIFOConfig()
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, FIFOConfig)

    def test_round_trip_lpt_chain(self) -> None:
        """Matches the lpt_chain.yaml shape: LPT with Ridge predictor +
        Chain feature extractor."""
        ta: TypeAdapter[SchedulingConfig] = TypeAdapter(SchedulingConfig)
        cfg = LPTConfig(
            eval_predictor=RidgePredictorConfig(
                feature_extractor=ChainFeatureExtractorConfig()
            )
        )
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, LPTConfig)
        assert isinstance(parsed.eval_predictor, RidgePredictorConfig)
        assert isinstance(
            parsed.eval_predictor.feature_extractor, ChainFeatureExtractorConfig
        )

    def test_json_round_trip(self) -> None:
        ta: TypeAdapter[SchedulingConfig] = TypeAdapter(SchedulingConfig)
        cfg = LPTConfig(
            eval_predictor=RidgePredictorConfig(alpha=2.5, buffer_size=750)
        )
        parsed = ta.validate_json(ta.dump_json(cfg))
        assert isinstance(parsed, LPTConfig)
        assert isinstance(parsed.eval_predictor, RidgePredictorConfig)
        assert parsed.eval_predictor.alpha == 2.5
        assert parsed.eval_predictor.buffer_size == 750

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[SchedulingConfig] = TypeAdapter(SchedulingConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_scheduler"})

    def test_each_variant_has_unique_kind(self) -> None:
        """Discriminator-uniqueness regression — same defense-in-depth
        check applied to PipelineBuilderConfig in hydra-1.6."""
        kinds = [
            FIFOConfig.model_fields["kind"].default,
            LPTConfig.model_fields["kind"].default,
        ]
        assert len(set(kinds)) == len(kinds)
        assert set(kinds) == {"fifo", "lpt"}
