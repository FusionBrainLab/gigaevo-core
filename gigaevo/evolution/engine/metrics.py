from __future__ import annotations

from pydantic import BaseModel, Field


class EngineMetrics(BaseModel):
    """Simplified metrics tracking (extracted)."""

    total_mutants: int = Field(
        default=0,
        description=(
            "Total number of mutants produced (incremented once per "
            "successful generate_mutations call, before DAG evaluation). "
            "Monotone, single source of truth for engine progress."
        ),
    )
    programs_processed: int = Field(
        default=0, description="Total number of programs processed"
    )
    mutations_created: int = Field(
        default=0, description="Total number of mutations created"
    )
    added: int = Field(default=0, description="Total programs added to evolution")
    rejected_validation: int = Field(
        default=0, description="Total programs rejected by validation"
    )
    rejected_strategy: int = Field(
        default=0, description="Total programs rejected by strategy"
    )
    elites_selected: int = Field(
        default=0, description="Total elites cumulatively selected for mutation"
    )
    submitted_for_refresh: int = Field(
        default=0, description="Total programs submitted for refresh"
    )

    def record_ingestion_metrics(
        self,
        added: int,
        rejected_validation: int,
        rejected_strategy: int,
    ) -> None:
        """Record metrics from program ingestion."""
        self.added += added
        self.rejected_validation += rejected_validation
        self.rejected_strategy += rejected_strategy

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
