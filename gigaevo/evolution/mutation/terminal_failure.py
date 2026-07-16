"""Typed causal classification for failed child evaluations."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from gigaevo.evolution.mutation.constants import (
    MUTATION_TERMINAL_FAILURE_METADATA_KEY,
)

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


class MutationTerminalFailureStage(StrEnum):
    """Failure sites whose causal interpretation is fixed by the scheduler."""

    DAG_TIMEOUT = "dag_timeout"
    DAG_EXECUTION = "dag_execution"
    DAG_BUILD = "dag_build"
    SCHEDULER_ORPHAN = "scheduler_orphan"


_INVALID_STAGES = frozenset(
    {
        MutationTerminalFailureStage.DAG_TIMEOUT,
        MutationTerminalFailureStage.DAG_EXECUTION,
    }
)


class MutationTerminalFailure(BaseModel):
    """Persisted reason and causal disposition for a discarded child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: MutationTerminalFailureStage
    status: Literal["invalid", "censored"]

    @model_validator(mode="after")
    def _status_matches_taxonomy(self) -> MutationTerminalFailure:
        expected = "invalid" if self.stage in _INVALID_STAGES else "censored"
        if self.status != expected:
            raise ValueError(
                f"{self.stage.value} must be classified as {expected}, got {self.status}"
            )
        return self

    @classmethod
    def for_stage(cls, stage: MutationTerminalFailureStage) -> MutationTerminalFailure:
        status: Literal["invalid", "censored"] = (
            "invalid" if stage in _INVALID_STAGES else "censored"
        )
        return cls(stage=stage, status=status)


def set_mutation_terminal_failure(
    program: Program, stage: MutationTerminalFailureStage
) -> MutationTerminalFailure:
    """Stamp a typed failure record before the terminal state is persisted."""

    failure = MutationTerminalFailure.for_stage(stage)
    program.set_metadata(
        MUTATION_TERMINAL_FAILURE_METADATA_KEY,
        failure.model_dump(mode="json"),
    )
    return failure


def get_mutation_terminal_failure(
    program: Program,
) -> MutationTerminalFailure | None:
    """Parse a persisted record; malformed or absent metadata fails closed."""

    raw = program.get_metadata(MUTATION_TERMINAL_FAILURE_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return MutationTerminalFailure.model_validate(raw)
    except ValueError:
        return None
