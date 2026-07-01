"""Decision context and contextual gain for use-attributed card credit.

Leaf value objects, deliberately at the ``gigaevo.memory`` top level: the
``core`` package depends on ``shared_memory.models`` (which embeds
``ContextualGain``), so these cannot live under ``core`` without a load-time
import cycle. Only depends on pydantic and the standard library.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DecisionContext(BaseModel):
    """The state a card-injection decision was made in.

    The base parent's id and metrics plus the crediting child's creation time —
    enough to identify which parent the decision was made against and to order
    events over the run. Both new fields default empty/None so pre-timestamp
    events still validate. The extension point for richer contexting later.
    """

    model_config = ConfigDict(frozen=True)

    parent_metrics: dict[str, float] = Field(default_factory=dict)
    parent_id: str = Field(
        default="", description="Base parent's program id (whose metrics these are)."
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Crediting child's creation time (UTC); None for legacy events "
        "written before the field existed.",
    )


class ContextualGain(BaseModel):
    """One credited injection event: the gain a card earned in a context."""

    model_config = ConfigDict(frozen=True)

    context: DecisionContext
    gain: float
    invalid: bool = Field(default=False)
