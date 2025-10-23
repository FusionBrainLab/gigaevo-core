from datetime import datetime, timezone
import math
import time
from typing import TYPE_CHECKING, Any
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from src.programs.core_types import ProgramStageResult, StageState, _pickle_b64_deserialize, _pickle_b64_serialize
from src.programs.program_state import ProgramState, FINAL_STATES
from src.programs.utils import pretty_print_error

if TYPE_CHECKING:
    from src.evolution.mutation.base import MutationSpec


class Lineage(BaseModel):
    """Represents the evolutionary lineage of a program."""

    parents: list[str] = Field(
        default_factory=list, description="List of parent program IDs"
    )
    children: list[str] = Field(
        default_factory=list, description="List of child program IDs (descendants)"
    )
    mutation: str | None = Field(
        None, description="Description of the mutation applied"
    )
    generation: int = Field(
        default=1, ge=1, description="Generation number"
    )

    def get_parent_count(self) -> int:
        """Get the number of parents."""
        return len(self.parents)
    
    def get_child_count(self) -> int:
        """Get the number of children."""
        return len(self.children)
    
    def add_child(self, child_id: str) -> None:
        """Add a child to this program's lineage."""
        if child_id not in self.children:
            self.children.append(child_id)

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

    state: ProgramState = Field(
        default=ProgramState.FRESH,
        description="Lifecycle state of the program",
    )

    lineage: Lineage = Field(
        default_factory=lambda: Lineage(parents=[]), description="Lineage of the program"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the program was created",
    )

    atomic_counter: int = Field(
        default=0,
        description="Monotonic storage-wide update counter",
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

    @field_serializer("metadata", when_used="json")
    def serialize_metadata(self, value: dict[str, Any]) -> str:
        """Serialize metadata to a JSON string, ensuring safety."""
        return _pickle_b64_serialize(value)

    def add_metrics(self, metrics: dict[str, float | int]) -> None:
        """Add multiple metrics at once."""
        for k, v in metrics.items():
            self.metrics[k] = float(v)

    def get_metadata(self, key: str) -> Any | None:
        """Get metadata for this program."""
        return self.metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this program."""
        self.metadata[key] = value

    def get_stage_error_summary(self, stage: str) -> str | None:
        """Get a formatted error summary for a failed stage that's LLM-parseable."""
        result = self.stage_results.get(stage)
        if not result or not result.is_failed():
            return None
        return pretty_print_error(result.error or {})

    def get_failed_stages(self) -> list[str]:
        """Get list of failed stages."""
        return [
            stage
            for stage, result in self.stage_results.items()
            if result.status == StageState.FAILED
        ]

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
            p.lineage.generation for p in parents
        ]

        generation = max(parent_generations) + 1


        lineage = Lineage(
            parents=[p.id for p in parents],
            mutation=mutation,
            generation=generation,
        )

        return cls(code=code, lineage=lineage, name=name)

    @classmethod
    def from_mutation_spec(cls, spec: "MutationSpec") -> "Program":
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

    def __hash__(self) -> int:
        """Hash based on program ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on program ID."""
        return isinstance(other, Program) and self.id == other.id

    def is_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(
            result.status == StageState.FAILED
            for result in self.stage_results.values()
        )

    @property
    def generation(self) -> int:
        """Get the generation number."""
        return self.lineage.generation

    @property
    def is_root(self) -> bool:
        """Check if this is a root program (no parents)."""
        return self.lineage.is_root()

    @property
    def is_complete(self) -> bool:
        """Check if DAG processing is completed."""
        return self.state in FINAL_STATES

