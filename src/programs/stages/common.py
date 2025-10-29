from __future__ import annotations

from typing import Any, Generic, TypeVar

from src.programs.core_types import StageIO

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class Box(StageIO, Generic[T]):
    """Generic single-value container: { data: T }."""

    data: T


class ListOf(StageIO, Generic[T]):
    """Generic list container: { items: list[T] }."""

    items: list[T]


String = Box[str]
AnyContainer = Box[Any]
StringContainer = Box[str]
FloatDictContainer = Box[dict[str, float]]
DictContainer = Box[dict[str, Any]]

StringList = ListOf[str]
FloatDictList = ListOf[dict[str, float]]
