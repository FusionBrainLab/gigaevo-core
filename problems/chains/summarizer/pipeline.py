"""Pipeline builder for JSON chain genomes: parse instead of execute."""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT, MAX_CODE_LENGTH
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    DEFAULT_INTRA_MAX_CHILDREN,
    IntraMemoryPipelineBuilder,
)
from gigaevo.programs.stages.json_genome import ParseJsonProgram


class JsonChainPipelineBuilder(IntraMemoryPipelineBuilder):
    """Standard (intra-memory) pipeline where ValidateCodeStage and
    CallProgramFunction both become ParseJsonProgram: parsing is the syntax
    gate, and the parsed document is the validator payload."""

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = 5,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        intra_max_children: int = DEFAULT_INTRA_MAX_CHILDREN,
        mutation_mode: str | None = None,
        enable_optuna_stage: bool = False,
        optimization_time_budget: float | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            intra_max_children=intra_max_children,
            mutation_mode=mutation_mode,
            enable_optuna_stage=enable_optuna_stage,
            optimization_time_budget=optimization_time_budget,
        )
        self.replace_stage(
            "ValidateCodeStage", lambda: ParseJsonProgram(timeout=self._stage_timeout)
        )
        self.replace_stage(
            "CallProgramFunction", lambda: ParseJsonProgram(timeout=self._stage_timeout)
        )
