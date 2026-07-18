"""Shared read-side component protocols."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.storage.base import ResearchResult


def _class_digest(component: object) -> str:
    identity = f"{type(component).__module__}.{type(component).__qualname__}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@runtime_checkable
class PolicyDigestProvider(Protocol):
    """A component whose policy identity is durable enough to record."""

    @property
    def policy_digest(self) -> str: ...


def policy_digest(component: object) -> str:
    """Return an explicit policy digest, or the component-class fallback."""

    if not isinstance(component, PolicyDigestProvider):
        return _class_digest(component)
    return component.policy_digest


@runtime_checkable
class Shortlister(Protocol):
    """Turns the mutation context into researched candidate cards."""

    @property
    def policy_digest(self) -> str: ...

    async def shortlist(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> ResearchResult: ...
