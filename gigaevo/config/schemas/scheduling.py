from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.evolution.scheduling.feature_extractor import FeatureExtractor
    from gigaevo.evolution.scheduling.predictor import EvalTimePredictor
    from gigaevo.evolution.scheduling.prioritizer import ProgramPrioritizer


class CodeFeatureExtractorConfig(FrozenStrictModel):
    """Extracts features from program code length and structure. The
    default for the Ridge predictor when the experiment does not pin a
    chain-specific extractor."""

    kind: Literal["code"] = "code"

    def build(self) -> "FeatureExtractor":
        from gigaevo.evolution.scheduling.feature_extractor import (
            CodeFeatureExtractor,
        )

        return CodeFeatureExtractor()


class ChainFeatureExtractorConfig(FrozenStrictModel):
    """Chain-specific feature extractor exposing n_steps, n_tool_steps,
    n_llm_steps, content length, retrieval count, and dag_depth.
    Used by problems with multi-step reasoning chains (HotpotQA, HoVer)
    where the structural shape predicts evaluation time better than
    code length alone."""

    kind: Literal["chain"] = "chain"

    def build(self) -> "FeatureExtractor":
        from gigaevo.evolution.scheduling.feature_extractor import (
            ChainFeatureExtractor,
        )

        return ChainFeatureExtractor()


FeatureExtractorConfig = Annotated[
    CodeFeatureExtractorConfig | ChainFeatureExtractorConfig,
    Field(discriminator="kind"),
]


class SimpleHeuristicPredictorConfig(FrozenStrictModel):
    """Online running-average predictor mapping code length to eval
    time. No sklearn dependency; cold-starts cleanly via
    ``default_rate``."""

    kind: Literal["simple_heuristic"] = "simple_heuristic"
    default_rate: float = Field(default=0.1, gt=0.0)
    window_size: int = Field(default=50, ge=1)

    def build(self) -> "EvalTimePredictor":
        from gigaevo.evolution.scheduling.predictor import (
            SimpleHeuristicPredictor,
        )

        return SimpleHeuristicPredictor(
            default_rate=self.default_rate,
            window_size=self.window_size,
        )


class RidgePredictorConfig(FrozenStrictModel):
    """Ridge regression over user-defined features. Retrains on the
    bounded replay buffer after every update. sklearn is a soft
    dependency, imported lazily by the runtime predictor."""

    kind: Literal["ridge"] = "ridge"
    feature_extractor: FeatureExtractorConfig = Field(
        default_factory=lambda: CodeFeatureExtractorConfig()
    )
    buffer_size: int = Field(default=500, ge=1)
    min_samples: int = Field(default=10, ge=1)
    default_prediction: float = Field(default=300.0, gt=0.0)
    alpha: float = Field(default=1.0, gt=0.0)

    def build(self) -> "EvalTimePredictor":
        from gigaevo.evolution.scheduling.predictor import RidgePredictor

        return RidgePredictor(
            feature_extractor=self.feature_extractor.build(),
            buffer_size=self.buffer_size,
            min_samples=self.min_samples,
            default_prediction=self.default_prediction,
            alpha=self.alpha,
        )


PredictorConfig = Annotated[
    SimpleHeuristicPredictorConfig | RidgePredictorConfig,
    Field(discriminator="kind"),
]


class FIFOConfig(FrozenStrictModel):
    """First-in-first-out scheduling — programs launch in insertion
    order. The runtime FIFOPrioritizer carries no parameters; the
    schema is a marker for the discriminated union."""

    kind: Literal["fifo"] = "fifo"

    def build(self) -> "ProgramPrioritizer":
        from gigaevo.evolution.scheduling.prioritizer import FIFOPrioritizer

        return FIFOPrioritizer()


class LPTConfig(FrozenStrictModel):
    """Longest-Processing-Time-first scheduling. Starts the
    predicted-longest program first to reduce tail idle time. Uses
    :class:`SimpleHeuristicPredictorConfig` by default; the chain-LPT
    variant overrides with :class:`RidgePredictorConfig` carrying a
    :class:`ChainFeatureExtractorConfig`."""

    kind: Literal["lpt"] = "lpt"
    eval_predictor: PredictorConfig = Field(
        default_factory=lambda: SimpleHeuristicPredictorConfig()
    )

    def build(self) -> "ProgramPrioritizer":
        from gigaevo.evolution.scheduling.prioritizer import LPTPrioritizer

        return LPTPrioritizer(eval_predictor=self.eval_predictor.build())


SchedulingConfig = Annotated[
    FIFOConfig | LPTConfig,
    Field(discriminator="kind"),
]
