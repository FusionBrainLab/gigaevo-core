from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.program_state import (
    COMPLETE_STATES,
    TERMINAL_STATES,
    ProgramState,
)
from gigaevo.programs.utils import pickle_b64_deserialize, pickle_b64_serialize

if TYPE_CHECKING:
    from gigaevo.evolution.mutation.base import MutationSpec


GENESIS_GENERATION: int = 1
NO_STAGE_ERRORS_MSG: str = "<No stage errors found>"
NO_ERROR_DETAILS_MSG: str = "<Failed stages found but no error details available>"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Lineage(BaseModel):
    """Evolutionary lineage information for a program."""

    parents: list[str] = Field(default_factory=list, description="Parent program IDs.")
    children: list[str] = Field(default_factory=list, description="Child program IDs.")
    mutation: str | None = Field(None, description="Description of applied mutation.")
    generation: int = Field(
        default=GENESIS_GENERATION, ge=1, description="Generation index."
    )

    @property
    def parent_count(self) -> int:
        return len(self.parents)

    @property
    def child_count(self) -> int:
        return len(self.children)

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children:
            self.children.append(child_id)

    def is_root(self) -> bool:
        return not self.parents


class Program(BaseModel):
    """A single program in the evolutionary system."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable unique program identifier (UUID string).",
    )
    code: str = Field(..., min_length=1, description="Program source code.")
    name: str | None = Field(
        default=None, description="Optional label / experiment tag."
    )

    stage_results: dict[str, ProgramStageResult] = Field(
        default_factory=dict, description="Per-stage execution results."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Scalar performance metrics."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Auxiliary, storage-opaque metadata."
    )

    state: ProgramState = Field(
        default=ProgramState.FRESH, description="Lifecycle state."
    )
    lineage: Lineage = Field(
        default_factory=Lineage, description="Evolutionary lineage."
    )

    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation time (UTC)."
    )
    atomic_counter: int = Field(
        default=0,
        description="Monotonic storage-wide update counter (revision).",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=False,
    )

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_and_validate_uuid(cls, v: Any) -> str:
        """Accept UUID or str; store as canonical UUID string."""
        if isinstance(v, uuid.UUID):
            return str(v)
        s = str(v)
        try:
            uuid.UUID(s)
        except Exception as e:
            raise ValueError("Invalid UUID format") from e
        return s

    @field_serializer("metadata", when_used="json")
    def _serialize_metadata(self, value: dict[str, Any]) -> str:
        return pickle_b64_serialize(value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        d = dict(data)
        if "metadata" in d and isinstance(d["metadata"], str):
            d["metadata"] = pickle_b64_deserialize(d["metadata"])
        if "stage_results" in d and isinstance(d["stage_results"], dict):
            d["stage_results"] = {
                k: ProgramStageResult.from_dict(v) if isinstance(v, dict) else v
                for k, v in d["stage_results"].items()
            }
        return cls(**d)

    @classmethod
    def create_child(
        cls,
        parents: list["Program"],
        code: str,
        mutation: str | None = None,
        name: str | None = None,
    ) -> "Program":
        if not parents:
            raise ValueError("At least one parent is required")
        generation = max((p.lineage.generation for p in parents), default=0) + 1
        lineage = Lineage(
            parents=[p.id for p in parents],
            mutation=mutation,
            generation=generation,
        )
        return cls(code=code, lineage=lineage, name=name)

    @classmethod
    def from_mutation_spec(cls, spec: "MutationSpec") -> "Program":
        name = " -> ".join(p.id for p in spec.parents) + f" (mutation: {spec.name})"
        program = cls.create_child(
            parents=spec.parents,
            code=spec.code,
            mutation=spec.name,
            name=name,
        )
        # Store mutation metadata (structured output) if available
        if spec.metadata:
            for key, value in spec.metadata.items():
                program.set_metadata(key, value)
        return program

    def add_metrics(self, metrics: Mapping[str, float | int]) -> None:
        for k, v in metrics.items():
            self.metrics[k] = float(v)

    def get_metadata(self, key: str) -> Any | None:
        return self.metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def format_stage_error(
        self, *, stage: str, include_traceback: bool = False
    ) -> str | None:
        result = self.stage_results.get(stage)
        if not result or result.status != StageState.FAILED or result.error is None:
            return None
        return result.error.pretty(include_traceback=include_traceback)

    def format_errors(self, *, include_traceback: bool = False) -> str:
        """Aggregate LLM-friendly error summaries across all failed stages."""
        failed = self.failed_stages
        if not failed:
            return NO_STAGE_ERRORS_MSG
        summaries: list[str] = []
        for s in failed:
            r = self.stage_results.get(s)
            if r and r.error:
                summaries.append(
                    f"=== Stage: {s} ===\n{r.error.pretty(include_traceback=include_traceback)}"
                )
        return "\n\n".join(summaries) if summaries else NO_ERROR_DETAILS_MSG

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Program) and self.id == other.id

    @property
    def generation(self) -> int:
        return self.lineage.generation

    @property
    def is_root(self) -> bool:
        return self.lineage.is_root()

    @property
    def failed_stages(self) -> list[str]:
        return [
            s for s, r in self.stage_results.items() if r.status == StageState.FAILED
        ]

    @property
    def is_failed(self) -> bool:
        return any(r.status == StageState.FAILED for r in self.stage_results.values())

    @property
    def is_complete(self) -> bool:
        return self.state in (COMPLETE_STATES | TERMINAL_STATES)
