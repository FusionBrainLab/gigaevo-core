"""Generation bucketing: which run-progress quarter a generation falls in."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.shared_memory.models import Quartile


class GenerationBucketer(BaseModel):
    """Single owner of the quartile policy: the run's generation range is split
    into four equal spans with exclusive upper bounds, and every consumer maps
    a generation to its :class:`Quartile` through :meth:`bucket`."""

    model_config = ConfigDict(frozen=True)

    b1: float = Field(description="Exclusive upper generation bound of Q1.")
    b2: float = Field(description="Exclusive upper generation bound of Q2.")
    b3: float = Field(description="Exclusive upper generation bound of Q3.")

    @classmethod
    def from_generations(cls, gens: Sequence[int]) -> GenerationBucketer:
        """Equal generation-span bounds over the observed generation range."""
        if not gens:
            raise ValueError("No generations available.")
        gmin, gmax = min(gens), max(gens)
        span = (gmax - gmin) + 1
        return cls(
            b1=gmin + 0.25 * span,
            b2=gmin + 0.50 * span,
            b3=gmin + 0.75 * span,
        )

    def bucket(self, gen: int) -> Quartile:
        if gen < self.b1:
            return Quartile.Q1
        if gen < self.b2:
            return Quartile.Q2
        if gen < self.b3:
            return Quartile.Q3
        return Quartile.Q4

    def generations_by_bucket(self, gens: Iterable[int]) -> dict[Quartile, set[int]]:
        out: dict[Quartile, set[int]] = {q: set() for q in Quartile.quarters()}
        for gen in gens:
            out[self.bucket(gen)].add(gen)
        return out
