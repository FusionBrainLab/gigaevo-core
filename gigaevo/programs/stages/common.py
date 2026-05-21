from __future__ import annotations

from typing import Any, Generic, TypeVar

from gigaevo.programs.core_types import StageIO

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class Box(StageIO, Generic[T]):
    """Generic single-value container: { data: T }."""

    data: T


class ListOf(StageIO, Generic[T]):
    """Generic list container: { items: list[T] }."""

    items: list[T]


class CacheOnlyInput(StageIO):
    """Input field whose only purpose is to fold into the cache-key hash.

    `compute()` ignores `cache_on`, but `content_hash` (defined on `StageIO`
    via `cloudpickle.dumps(self.model_dump())`) folds the value, so changing
    `cache_on` invalidates the cached output. Used to attach an external
    invalidation signal (e.g. opponent-id list) to LLM stages whose
    `compute()` would otherwise be value-stable across the rotation.
    """

    cache_on: Any | None = None


String = Box[str]
AnyContainer = Box[Any]
StringContainer = Box[str]
FloatDictContainer = Box[dict[str, float]]
DictContainer = Box[dict[str, Any]]
# Backward-compat alias for validator stage outputs stored via cloudpickle.
ValidatorOutput = Box[tuple[dict[str, float], Any]]

StringList = ListOf[str]
FloatDictList = ListOf[dict[str, float]]
