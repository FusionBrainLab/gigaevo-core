from __future__ import annotations

from pydantic import BaseModel, Field


class EngineMetrics(BaseModel):
    """Simplified metrics tracking (extracted)."""

    total_generations: int = Field(
        default=0, description="Total number of generations run"
    )
    programs_processed: int = Field(
        default=0, description="Total number of programs processed"
    )
    mutations_created: int = Field(
        default=0, description="Total number of mutations created"
    )
    errors_encountered: int = Field(
        default=0, description="Total number of errors encountered"
    )
    added: int = Field(default=0, description="Total programs added to evolution")
    restored: int = Field(default=0, description="Total programs restored to evolution")
    rejected_validation: int = Field(
        default=0, description="Total programs rejected by validation"
    )
    rejected_strategy: int = Field(
        default=0, description="Total programs rejected by strategy"
    )
    elites_selected: int = Field(
        default=0, description="Total elites selected across all generations"
    )
    elites_selection_errors: int = Field(
        default=0, description="Total elite selection errors"
    )
    submitted_for_refresh: int = Field(
        default=0, description="Total programs submitted for refresh"
    )
    mutations_creation_errors: int = Field(
        default=0, description="Total mutation creation errors"
    )

    def record_ingestion_metrics(
        self,
        added: int,
        restored: int,
        rejected_validation: int,
        rejected_strategy: int,
    ) -> None:
        """Record metrics from program ingestion."""
        self.added += added
        self.restored += restored
        self.rejected_validation += rejected_validation
        self.rejected_strategy += rejected_strategy

    def record_elite_selection_metrics(
        self, elites_selected: int, elites_selection_errors: int
    ) -> None:
        """Record metrics from elite selection."""
        self.elites_selected += elites_selected
        self.elites_selection_errors += elites_selection_errors

    def record_reprocess_metrics(self, submitted_for_refresh: int) -> None:
        """Record metrics from reprocessing."""
        self.submitted_for_refresh += submitted_for_refresh

    def record_mutation_metrics(
        self, mutations_created: int, mutations_creation_errors: int
    ) -> None:
        """Record metrics from mutation."""
        self.mutations_created += mutations_created
        self.mutations_creation_errors += mutations_creation_errors

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
