"""The store abstraction — callers see :class:`Card`, never Chroma or the agent.

Retrieval is an implementation detail of storage: a store answers
:meth:`~MemoryStore.nearest` (embedding similarity) and
:meth:`~MemoryStore.research` (agentic multi-step retrieval) itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardKind


class ScoredCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card: Card
    distance: float = Field(description="Embedding distance; lower is closer.")


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(description="What the caller wants memories about.")
    planning_context: str = Field(
        default="",
        description="Extra situation framing for the planner (parent metrics, "
        "task summary); not itself a query.",
    )
    exclude_ids: frozenset[str] = Field(
        default=frozenset(),
        description="Card ids that must not appear among candidates.",
    )


class ResearchFailure(StrEnum):
    TIMEOUT = "timeout"
    SHORTLISTER_EXCEPTION = "shortlister_exception"
    STORE_EXCEPTION = "store_exception"


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cards: tuple[Card, ...] = ()
    summary: str = Field(
        default="", description="The agent's synthesis of why these cards."
    )
    iterations: int = 0
    failure: ResearchFailure | None = None


class MemoryStore(ABC):
    """Typed card storage with retrieval built in.

    Cards in, cards out — no dict unions, no conversion layer; bad data
    raises. Retrieval failures degrade to empty results, never to exceptions
    crossing this boundary; their neutral result retains a typed failure marker.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def save(self, card: Card) -> str:
        """Insert or overwrite; mints an id when ``card.id`` is empty.

        Returns the id under which the card is stored.
        """

    @abstractmethod
    def update(
        self, card_id: str, transform: Callable[[Card], Card | None]
    ) -> Card | None:
        """Atomically transform a fresh card, or delete it by returning ``None``.

        The load, transform, and persistence share one exclusive transaction.
        Returns the affected card, or ``None`` when ``card_id`` was absent.
        """

    @abstractmethod
    def get(self, card_id: str) -> Card | None: ...

    @abstractmethod
    def delete(self, card_id: str) -> bool:
        """Remove a card; False when the id was not present."""

    @abstractmethod
    def snapshot(self) -> tuple[Card, ...]:
        """A stable view of the whole bank, ordered by card id."""

    @abstractmethod
    def nearest(
        self,
        text: str,
        k: int,
        kind: CardKind | None = None,
        task_key: str | None = None,
    ) -> list[ScoredCard]:
        """The ``k`` closest cards under the configured nearest scope."""

    @abstractmethod
    async def research(self, request: ResearchRequest) -> ResearchResult:
        """Agentic retrieval: plan queries, search, reflect; empty on failure."""

    @asynccontextmanager
    async def authoring_transaction(self) -> AsyncIterator[None]:
        """Serialize semantic deduplication and admission when required.

        Non-shared stores need no coordination. Durable stores can override
        this with a cross-process transaction covering retrieve → judge → save.
        """

        yield

    @abstractmethod
    def rebuild(self) -> None:
        """Force a full vector-index rebuild from the current bank."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
