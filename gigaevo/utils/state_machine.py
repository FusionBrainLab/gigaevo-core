"""Generic transition-table state-machine helpers.

A state machine here is just a ``StrEnum`` plus a ``VALID_TRANSITIONS`` table;
these helpers own the validation semantics (self-transitions are always legal)
so every machine (``ProgramState``, the memory store state, ...) shares one
implementation instead of hand-rolling it.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from enum import StrEnum
from typing import TypeVar

StateT = TypeVar("StateT", bound=StrEnum)

TransitionTable = Mapping[StateT, Set[StateT]]


def is_valid_transition(
    current: StateT, new: StateT, table: TransitionTable[StateT]
) -> bool:
    if current == new:
        return True
    return new in table.get(current, frozenset())


def validate_transition(
    current: StateT, new: StateT, table: TransitionTable[StateT]
) -> None:
    if not is_valid_transition(current, new, table):
        valid_next = table.get(current, frozenset())
        raise ValueError(
            f"Invalid state transition: {current.value} -> {new.value}. "
            f"Valid transitions from {current.value}: "
            f"{sorted(s.value for s in valid_next)}"
        )
