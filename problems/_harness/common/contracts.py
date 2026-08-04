"""Method contracts shared by every evolved representation.

Config is opaque here on purpose: this layer must stay benchmark-free so that
DirectEvolve, UnifiedEvolve, ImprovEvolve and Generic-BH are driven through one
identical code path. Anything benchmark-specific belongs in a validator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

Config = Any


class ProposalStatus(StrEnum):
    SUCCESS = "success"
    INVALID_RETURN = "invalid_return"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ProposalResult:
    """Outcome of one atomic proposal.

    Atomic means the whole controller step: for ImprovEvolve that is
    improve(perturb(x)), not one method call. Timing a single method call would
    make the comparison a "one call versus two calls" artifact.
    """

    status: ProposalStatus
    label: str
    config: Config | None = None
    elapsed_s: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ProposalStatus.SUCCESS


@runtime_checkable
class DirectProgram(Protocol):
    def solve(self, instance: Any, seed: int) -> Config: ...


@runtime_checkable
class UnifiedProgram(Protocol):
    def propose(
        self,
        input_config: Config | None,
        intensity: float,
        seed: int | None = None,
    ) -> Config: ...


@runtime_checkable
class ModularProgram(Protocol):
    def generate_config(self, seed: int) -> Config: ...

    def perturb(self, input_config: Config, intensity: float, seed: int) -> Config: ...

    def improve(self, input_config: Config, seed: int) -> Config: ...


@runtime_checkable
class SearchAdapter(Protocol):
    """What the fixed controller sees. Every representation reduces to this."""

    def initialize(self, seed: int) -> Config: ...

    def transition(
        self, input_config: Config, intensity: float, seed: int
    ) -> Config: ...


@runtime_checkable
class Budget(Protocol):
    @property
    def remaining_s(self) -> float: ...

    def exhausted(self) -> bool: ...


@runtime_checkable
class SeedSchedule(Protocol):
    def next(self) -> int: ...


SupervisedCall = Callable[[Callable[[], Config], str], ProposalResult]
