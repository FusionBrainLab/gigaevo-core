"""Decision context and contextual gain for use-attributed card credit.

Leaf value objects, deliberately at the ``gigaevo.memory`` top level: the
``core`` package depends on ``shared_memory.models`` (which embeds
``ContextualGain``), so these cannot live under ``core`` without a load-time
import cycle. Only depends on pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DecisionContext(BaseModel):
    """The state a card-injection decision was made in.

    Currently just the base parent's metrics; the extension point for richer
    contexting later.
    """

    model_config = ConfigDict(frozen=True)

    parent_metrics: dict[str, float] = Field(default_factory=dict)


class ContextualGain(BaseModel):
    """One credited injection event: the gain a card earned in a context."""

    model_config = ConfigDict(frozen=True)

    context: DecisionContext
    gain: float
    invalid: bool = Field(default=False)
