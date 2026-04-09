from __future__ import annotations

from typing import Literal


class MemoryStateError(Exception):
    """Raised when invalid state transition attempted."""

    pass


class MemoryState:
    """Explicit memory lifecycle state machine.

    States:
    - initializing: Construction in progress
    - ready: Fully initialized, accepting operations
    - building: Rebuilding GAM index (brief transient state)
    - error: Failed to initialize or unrecoverable error

    Valid transitions:
    - initializing -> ready (success)
    - initializing -> error (failure)
    - ready -> building (rebuild triggered)
    - building -> ready (rebuild succeeded)
    - building -> error (rebuild failed)
    - ready -> error (unrecoverable error)
    - error -> initializing (recovery attempt)
    """

    StateType = Literal["initializing", "ready", "building", "error"]

    _VALID_TRANSITIONS: dict[str, set[str]] = {
        "initializing": {"ready", "error"},
        "ready": {"building", "error"},
        "building": {"ready", "error"},
        "error": {"initializing"},
    }

    def __init__(self) -> None:
        self._current: MemoryState.StateType = "initializing"
        self._error_reason: str = ""

    @property
    def current(self) -> MemoryState.StateType:
        """Current state."""
        return self._current

    @property
    def is_ready(self) -> bool:
        """True if state is ready."""
        return self._current == "ready"

    @property
    def error_reason(self) -> str:
        """Error reason if state is error, empty string otherwise."""
        return self._error_reason

    def _transition(self, new_state: MemoryState.StateType, reason: str = "") -> None:
        """Perform state transition, raising MemoryStateError on invalid transition."""
        allowed = self._VALID_TRANSITIONS.get(self._current, set())
        if new_state not in allowed:
            raise MemoryStateError(
                f"Cannot transition {self._current!r} -> {new_state!r}. "
                f"Allowed: {sorted(allowed)}"
            )
        self._current = new_state
        self._error_reason = reason

    def mark_initializing(self) -> None:
        """Transition to initializing (error recovery only)."""
        self._transition("initializing")

    def mark_ready(self) -> None:
        """Transition to ready."""
        self._transition("ready")

    def mark_building(self) -> None:
        """Transition to building."""
        self._transition("building")

    def mark_error(self, reason: str = "") -> None:
        """Transition to error with optional reason."""
        self._transition("error", reason)
