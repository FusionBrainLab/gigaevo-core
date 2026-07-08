"""The store abstraction — callers see :class:`Card`, never Chroma or the agent.

Retrieval is an implementation detail of storage: a store answers
:meth:`~MemoryStore.nearest` (embedding similarity) and
:meth:`~MemoryStore.research` (agentic multi-step retrieval) itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
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


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cards: tuple[Card, ...] = ()
    summary: str = Field(
        default="", description="The agent's synthesis of why these cards."
    )
    iterations: int = 0


class MemoryStore(ABC):
    """Typed card storage with retrieval built in.

    Cards in, cards out — no dict unions, no conversion layer; bad data
    raises. Retrieval failures degrade to empty results, never to exceptions
    crossing this boundary.
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
    def get(self, card_id: str) -> Card | None: ...

    @abstractmethod
    def delete(self, card_id: str) -> bool:
        """Remove a card; False when the id was not present."""

    @abstractmethod
    def snapshot(self) -> tuple[Card, ...]:
        """A stable view of the whole bank, ordered by card id."""

    @abstractmethod
    def apply_merges(self, merged: Sequence[Card]) -> list[str]:
        """Persist merge survivors (each carries its absorbed ids) in one
        batch; returns the ids saved."""

    @abstractmethod
    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]:
        """The ``k`` closest cards under the configured nearest scope."""

    @abstractmethod
    async def research(self, request: ResearchRequest) -> ResearchResult:
        """Agentic retrieval: plan queries, search, reflect; empty on failure."""

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
