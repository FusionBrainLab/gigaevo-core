from datetime import datetime, timezone
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from src.programs.core_types import ProgramStageResult, StageState, _pickle_b64_deserialize, _pickle_b64_serialize
from src.programs.program_state import ProgramState
from src.programs.utils import pretty_print_error

if TYPE_CHECKING:
    from src.evolution.mutation.base import MutationSpec

MAX_METRIC_VALUE = 1e9
MIN_METRIC_VALUE = -1e9


class Lineage(BaseModel):
    """Represents the evolutionary lineage of a program."""

    parents: list[str] = Field(
        ..., min_length=1, description="List of parent program IDs"
    )
    mutation: str | None = Field(
        None, description="Description of the mutation applied"
    )
    generation: int | None = Field(
        None, ge=0, description="Generation number"
    )

    def get_parent_count(self) -> int:
        """Get the number of parents."""
        return len(self.parents)

    def is_root(self) -> bool:
        """Check if this is a root program (no parents)."""
        return not self.parents


class Program(BaseModel):
    """Represents a program in the evolutionary system."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique program identifier",
    )
    code: str = Field(..., min_length=1, description="The program code")

    name: str | None = Field(
        default=None,
        description="Optional human-readable label or experiment tag",
    )

    stage_results: dict[str, ProgramStageResult] = Field(
        default_factory=dict,
        description="Results of each processing stage",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # ------------------------------------------------------------------
    # Lifecycle state management
    # ------------------------------------------------------------------
    state: ProgramState = Field(
        default=ProgramState.FRESH,
        description="Lifecycle state of the program",
    )

    lineage: Lineage | None = Field(
        default=None, description="Lineage of the program"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the program was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the program was last updated",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    @field_validator("id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Validate that the ID is a valid UUID."""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError("Invalid UUID format")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate metric values."""
        out: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("Metric key must be a non-empty string")
            if not isinstance(val, (int, float)):
                raise ValueError(f"Metric value must be numeric: {k}={val}")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"Metric '{k}' must be finite, but got {val}")
            if not (MIN_METRIC_VALUE <= fval <= MAX_METRIC_VALUE):
                raise ValueError(
                    f"Metric value out of reasonable range: {k}={val}"
                )
            out[k] = fval
        return out

    @model_validator(mode="after")  # type: ignore[arg-type]
    def validate_timestamps(self) -> "Program":
        """Validate that timestamps are consistent."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self

    @field_serializer("metadata", when_used="json")
    def serialize_metadata(self, value: dict[str, Any]) -> str:
        """Serialize metadata to a JSON string, ensuring safety."""
        return _pickle_b64_serialize(value)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def add_metric(self, name: str, value: float | int) -> None:
        """Add or update a metric."""
        self.metrics[name] = float(value)
        self.update_timestamp()

    def add_metrics(self, metrics: dict[str, float | int]) -> None:
        """Add multiple metrics at once."""
        for k, v in metrics.items():
            self.metrics[k] = float(v)
        self.update_timestamp()

    def get_metric(
        self, name: str, default: float | None = None
    ) -> float | None:
        """Get a metric value by name."""
        return self.metrics.get(name, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this program."""
        self.metadata[key] = value
        self.update_timestamp()

    def get_metadata(self, key: str) -> Any | None:
        """Get metadata for this program."""
        return self.metadata.get(key)

    def set_stage_result(self, stage: str, result: ProgramStageResult) -> None:
        """Set the result for a specific stage."""
        self.stage_results[stage] = result
        self.update_timestamp()

    def get_stage_result(self, stage: str) -> ProgramStageResult | None:
        """Get the result for a specific stage."""
        return self.stage_results.get(stage)

    def get_stage_status(self, stage: str) -> StageState:
        """Get the status of a specific stage."""
        result = self.stage_results.get(stage)
        return result.status if result else StageState.PENDING

    def get_stage_error_summary(self, stage: str) -> str | None:
        """Get a formatted error summary for a failed stage that's LLM-parseable."""
        result = self.stage_results.get(stage)
        if not result or not result.is_failed():
            return None
        return pretty_print_error(result.error or {})

    def get_all_errors_summary(self) -> str:
        """Get a comprehensive summary of all stage errors for LLM analysis."""
        failed_stages = self.get_failed_stages()
        if not failed_stages:
            return "No stage errors found."

        error_summaries = []
        for stage in failed_stages:
            error_summary = self.get_stage_error_summary(stage)
            if error_summary:
                error_summaries.append(
                    f"=== Stage: {stage} ===\n{error_summary}"
                )

        return (
            "\n\n".join(error_summaries)
            if error_summaries
            else "Failed stages found but no error details available."
        )

    def get_age_seconds(self) -> float:
        """Get the age of the program in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def get_time_since_update_seconds(self) -> float:
        """Get the time since last update in seconds."""
        return (datetime.now(timezone.utc) - self.updated_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert the program to a dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        """Create a Program from a dictionary."""
        data = dict(data)
        for key in ("metadata",):
            if key in data and isinstance(data[key], str):
                data[key] = _pickle_b64_deserialize(data[key])
        if "stage_results" in data and isinstance(data["stage_results"], dict):
            data["stage_results"] = {
                k: ProgramStageResult.from_dict(v) if isinstance(v, dict) else v
                for k, v in data["stage_results"].items()
            }

        return cls(**data)

    @classmethod
    def create_child(
        cls,
        parents: list["Program"],
        code: str,
        mutation: str | None = None,
        name: str | None = None,
    ) -> "Program":
        """Create a child program from parent programs."""
        if not parents:
            raise ValueError("At least one parent is required")

        # Calculate generation - handle parents without lineage
        parent_generations = [
            (p.lineage.generation or 0) for p in parents if p.lineage
        ]
        if parent_generations:
            generation = max(parent_generations) + 1
        else:
            generation = 1  # If no parents have lineage, start at generation 1

        lineage = Lineage(
            parents=[p.id for p in parents],
            mutation=mutation,
            generation=generation,
        )

        return cls(code=code, lineage=lineage, name=name)

    @classmethod
    def from_mutation_spec(cls, spec: "MutationSpec") -> "Program":
        # Import here to avoid circular import
        from src.evolution.mutation.base import MutationSpec
        
        name = ""
        for i, parent in enumerate(spec.parents):
            name += f"{parent.id}"
            if i < len(spec.parents) - 1:
                name += " -> "
            else:
                name += f" (mutation: {spec.name})"
        return cls.create_child(
            parents=spec.parents, code=spec.code, mutation=spec.name, name=name
        )

    def copy_with_code(
        self, code: str, mutation: str | None = None
    ) -> "Program":
        """Create a copy of this program with new code."""
        return Program(
            code=code,
            lineage=Lineage(
                parents=[self.id],
                mutation=mutation,
                generation=(
                    (self.lineage.generation or 0) + 1 if self.lineage else 1
                ),
            ),
            name=self.name,
        )

    def to_dot_node(self, use_colors: bool = True) -> str:
        """Generate a DOT node representation."""
        label = f"{self.id[:6]}\\nProgram"
        if self.name:
            label = f"{self.name}\\n{label}"

        if use_colors:
            if self.is_failed():
                color = "red"
            elif self.is_complete:
                color = "green"
            elif self.is_pending():
                color = "gray"
            else:
                color = "blue"
            return f'  "{self.id}" [label="{label}", color="{color}", fontcolor="{color}"];'
        else:
            return f'  "{self.id}" [label="{label}"];'

    def to_dot_edges(self) -> list[str]:
        """Generate DOT edge representations."""
        if not self.lineage:
            return []
        return [
            f'  "{parent}" -> "{self.id}";' for parent in self.lineage.parents
        ]

    def __hash__(self) -> int:
        """Hash based on program ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on program ID."""
        return isinstance(other, Program) and self.id == other.id

    def summary(self) -> dict[str, Any]:
        """Get a summary of the program."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": self.metrics,
            "stages": {k: str(v.status) for k, v in self.stage_results.items()},
            "generation": self.generation,
        }

    def describe(self) -> str:
        """Get a human-readable description of the program."""
        name_part = f"'{self.name}' " if self.name else ""
        return (
            f"Program {name_part}{self.id[:8]} | "
            f"Generation: {self.generation or 'N/A'} | "
            f"Stages: {len(self.stage_results)} | "
            f"Metrics: {list(self.metrics.keys())}"
        )

    def is_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(
            result.status == StageState.FAILED
            for result in self.stage_results.values()
        )

    def is_pending(self) -> bool:
        """Check if all stages are pending."""
        return all(
            result.status == StageState.PENDING
            for result in self.stage_results.values()
        )

    def is_running(self) -> bool:
        """Check if any stage is running."""
        return any(
            result.status == StageState.RUNNING
            for result in self.stage_results.values()
        )

    def get_completed_stages(self) -> list[str]:
        """Get list of completed stages."""
        return [
            stage
            for stage, result in self.stage_results.items()
            if result.status == StageState.COMPLETED
        ]

    def get_failed_stages(self) -> list[str]:
        """Get list of failed stages."""
        return [
            stage
            for stage, result in self.stage_results.items()
            if result.status == StageState.FAILED
        ]

    def get_pending_stages(self) -> list[str]:
        """Get the names of all stages that are pending."""
        return [
            name
            for name, result in self.stage_results.items()
            if result.status == StageState.PENDING
        ]

    @property
    def generation(self) -> int | None:
        """Get the generation number."""
        return self.lineage.generation if self.lineage else None

    @property
    def parent_count(self) -> int:
        """Get the number of parents."""
        return self.lineage.get_parent_count() if self.lineage else 0

    @property
    def is_root(self) -> bool:
        """Check if this is a root program (no parents)."""
        return self.lineage.is_root() if self.lineage else True

    # ------------------------------------------------------------------
    # State-based properties (readonly) ----------------------------
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """Check if DAG processing is completed."""
        return self.state == ProgramState.DAG_PROCESSING_COMPLETED

    @property
    def is_discarded(self) -> bool:
        """Check if program is discarded."""
        return self.state == ProgramState.DISCARDED
