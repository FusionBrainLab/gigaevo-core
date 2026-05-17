from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel
from gigaevo.config.schemas.algorithm import AlgorithmConfig, MultiIslandConfig
from gigaevo.config.schemas.engine import EngineConfig, SteadyStateEngineConfig
from gigaevo.config.schemas.llm import BanditRouterConfig, LLMConfig
from gigaevo.config.schemas.pipeline import PipelineConfig
from gigaevo.config.schemas.problem import ProblemConfig
from gigaevo.config.schemas.prompt import PromptFetcherConfig
from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig
from gigaevo.config.schemas.runner import DAGRunnerConfig


_KEY_PREFIX_TEMPLATE = "gigaevo:{name}"


class ExperimentConfig(FrozenStrictModel):
    """Root of the typed experiment configuration tree.

    Every shipped experiment exports ``build() -> ExperimentConfig``
    returning a fully-frozen instance of this class. The CLI loads the
    instance, applies tyro-driven overrides, and hands it to
    ``build_object_graph`` which materialises the runtime tree.

    Cross-field invariants live on this model rather than on the leaf
    schemas so that violations are surfaced where the experiment
    composes its parts.
    """

    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    seed: int = Field(default=42)
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))

    @field_validator("output_dir")
    @classmethod
    def _output_dir_not_empty_or_cwd(cls, value: Path) -> Path:
        if str(value) in ("", "."):
            raise ValueError(
                "output_dir must be a real path; "
                f"got {value!r} which resolves to the current working directory"
            )
        return value

    redis: RedisConfig
    dataplane: DataPlaneSettings
    problem: ProblemConfig
    algorithm: AlgorithmConfig
    engine: EngineConfig
    pipeline: PipelineConfig
    llm: LLMConfig
    runner: DAGRunnerConfig = Field(default_factory=DAGRunnerConfig)
    prompt_fetcher: PromptFetcherConfig | None = None

    @model_validator(mode="after")
    def _key_prefix_follows_convention(self) -> "ExperimentConfig":
        expected = _KEY_PREFIX_TEMPLATE.format(name=self.name)
        if self.dataplane.key_prefix != expected:
            raise ValueError(
                f"dataplane.key_prefix must equal {expected!r} for experiment "
                f"{self.name!r}; got {self.dataplane.key_prefix!r}. The "
                "convention isolates each experiment's Redis namespace from "
                "concurrent runs of other experiments."
            )
        return self

    @model_validator(mode="after")
    def _multi_island_min_size_matches_remover(self) -> "ExperimentConfig":
        """If the algorithm is multi-island and any island carries a
        ``max_size`` cap, that island must also declare an
        ``archive_remover`` — the IslandConfig validator already enforces
        this, but the cross-field check surfaces the failure with the
        experiment name attached for easier debugging."""
        if isinstance(self.algorithm, MultiIslandConfig):
            for island in self.algorithm.islands:
                if island.max_size is not None and island.archive_remover is None:
                    raise ValueError(
                        f"experiment {self.name!r}: island "
                        f"{island.island_id!r} has max_size={island.max_size} "
                        "but no archive_remover"
                    )
        return self

    @model_validator(mode="after")
    def _bus_engine_requires_bandit_router(self) -> "ExperimentConfig":
        """The bus-flavor engine is only useful with a bandit router
        because the bus migrates programs across engines and rewards
        the LLM that produced the winner. With a static ensemble there
        is no learning signal to propagate.

        Bus engine schemas land in hydra-2.5; this validator runs as a
        no-op until that variant joins the EngineConfig union."""
        engine_kind = getattr(self.engine, "kind", None)
        if engine_kind == "bus" and not isinstance(self.llm, BanditRouterConfig):
            raise ValueError(
                f"experiment {self.name!r}: bus engine requires a "
                "BanditRouterConfig llm to receive migration rewards; "
                f"got {type(self.llm).__name__}"
            )
        return self

    @model_validator(mode="after")
    def _steady_state_in_flight_below_engine_count(self) -> "ExperimentConfig":
        """SteadyStateEngineConfig.max_in_flight bounds the in-flight
        DAG queue. Setting it above max_mutations_per_generation is a
        cargo-cult configuration that wastes memory without raising
        throughput because the parent selector cannot produce more
        children per generation than that ceiling."""
        if isinstance(self.engine, SteadyStateEngineConfig):
            if self.engine.max_in_flight > self.engine.max_mutations_per_generation:
                raise ValueError(
                    f"experiment {self.name!r}: engine.max_in_flight "
                    f"({self.engine.max_in_flight}) exceeds "
                    f"engine.max_mutations_per_generation "
                    f"({self.engine.max_mutations_per_generation})"
                )
        return self

    @property
    def experiment_id(self) -> str:
        """Stable 12-character hash of the resolved config tree. Used
        as the leaf directory under ``output_dir`` so two runs with
        identical configs share an output namespace and two runs with
        any field differing land in different directories."""
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def expected_key_prefix(self) -> str:
        """The dataplane key prefix the cross-field validator enforces.
        Exposed so experiment files can construct the right
        DataPlaneSettings without re-implementing the template."""
        return _KEY_PREFIX_TEMPLATE.format(name=self.name)
