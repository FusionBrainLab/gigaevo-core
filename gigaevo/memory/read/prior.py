"""Cold-card prior policies for the memory read bandit."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext, MemoryContextModel
from gigaevo.memory.context.beta import BetaPrior, coerce_beta_prior
from gigaevo.memory.storage.base import MemoryStore


@runtime_checkable
class MemoryPrior(Protocol):
    """Supplies cold-card priors for a read decision."""

    def cold_card_prior(
        self, card: Card, context: DecisionContext | None = None
    ) -> BetaPrior: ...


class FixedMemoryPrior(BaseModel):
    """Static prior policy; preserves the historical ``Beta(3, 3)`` default."""

    model_config = ConfigDict(frozen=True)

    cold_card: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="Fallback prior for cards with no context-relevant evidence.",
    )

    def cold_card_prior(
        self, card: Card, context: DecisionContext | None = None
    ) -> BetaPrior:
        del card, context
        prior = coerce_beta_prior(self.cold_card, source="fixed_cold")
        return prior.model_copy(update={"source": "fixed_cold"})


def _event_weight(event: ContextualGain) -> float:
    attr = event.attribution
    if attr is not None and attr.credit_weight is not None:
        weight = float(attr.credit_weight)
    elif event.founding:
        weight = 0.0
    else:
        weight = 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def _first_non_founding_exposure(
    events: Iterable[ContextualGain],
) -> tuple[float, bool] | None:
    """Return ``(weight, success)`` for the first causal non-founding exposure."""

    for event in events:
        if event.founding:
            continue
        weight = _event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            return (weight, False)
        if event.gain is not None and math.isfinite(float(event.gain)):
            return (weight, float(event.gain) >= 0.0)
    return None


def _all_exposures(events: Iterable[ContextualGain]) -> tuple[float, float]:
    success = 0.0
    failure = 0.0
    for event in events:
        if event.founding:
            continue
        weight = _event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            failure += weight
        elif event.gain is not None and math.isfinite(float(event.gain)):
            if float(event.gain) >= 0.0:
                success += weight
            else:
                failure += weight
    return success, failure


class EmpiricalBayesMemoryPrior(BaseModel):
    """Contextual hierarchical cold prior learned from banked first exposures.

    The prior is intentionally conservative: it uses the first non-founding
    exposure per card by default, shrinks toward a seed prior, and caps the
    concentration.  This avoids letting repeatedly sampled warm winners dictate
    what an untested card should believe.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    store: MemoryStore = Field(description="Memory bank providing prior evidence.")
    context_model: MemoryContextModel = Field(default_factory=GlobalMemoryContext)
    seed_prior: tuple[float, float] = Field(
        default=(1.0, 1.0),
        description="No-data seed prior for cold-card help probability.",
    )
    k_min: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum cold-prior concentration when cohort evidence is tiny.",
    )
    k_max: float = Field(
        default=6.0,
        gt=0.0,
        description="Maximum cold-prior concentration after enough cohort evidence.",
    )
    n_ref: float = Field(
        default=32.0,
        gt=0.0,
        description="Effective events needed to approach k_max concentration.",
    )
    shrink_events: float = Field(
        default=16.0,
        ge=0.0,
        description="Parent-cohort pseudo-count used when shrinking each cohort.",
    )
    first_exposure_only: bool = Field(
        default=True,
        description="Use one first exposure per card for hyper-prior statistics.",
    )
    min_parameter: float = Field(
        default=0.25,
        gt=0.0,
        description="Lower bound for each Beta parameter after shrinkage.",
    )

    @field_validator("k_max")
    @classmethod
    def _finite_kmax(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("k_max must be finite")
        return float(value)

    def cold_card_prior(
        self, card: Card, context: DecisionContext | None = None
    ) -> BetaPrior:
        seed = coerce_beta_prior(self.seed_prior, source="seed")
        parent_mu = seed.alpha / (seed.alpha + seed.beta)
        best = BetaPrior(
            alpha=max(self.min_parameter, self.k_min * parent_mu),
            beta=max(self.min_parameter, self.k_min * (1.0 - parent_mu)),
            source="eb_seed",
            support_n=0.0,
        )
        cohorts = [
            ("eb_global", self._counts(None, kind=None, category=None, local=False)),
            ("eb_kind", self._counts(None, kind=card.kind, category=None, local=False)),
            (
                "eb_kind_category",
                self._counts(None, kind=card.kind, category=card.category, local=False),
            ),
        ]
        if self.context_model.key_for(context).kind != "global":
            cohorts.extend(
                [
                    (
                        "eb_context",
                        self._counts(context, kind=None, category=None, local=True),
                    ),
                    (
                        "eb_context_kind",
                        self._counts(
                            context, kind=card.kind, category=None, local=True
                        ),
                    ),
                    (
                        "eb_context_kind_category",
                        self._counts(
                            context,
                            kind=card.kind,
                            category=card.category,
                            local=True,
                        ),
                    ),
                ]
            )
        for source, counts in cohorts:
            success, failure = counts
            n = success + failure
            if n <= 0.0:
                continue
            p_hat = (success + 0.5) / (n + 1.0)
            mu = (
                (n * p_hat + self.shrink_events * parent_mu) / (n + self.shrink_events)
                if self.shrink_events > 0.0
                else p_hat
            )
            kappa = self.k_min + (self.k_max - self.k_min) * min(1.0, n / self.n_ref)
            best = BetaPrior(
                alpha=max(self.min_parameter, kappa * mu),
                beta=max(self.min_parameter, kappa * (1.0 - mu)),
                source=source,
                support_n=n,
            )
            parent_mu = mu
        return best

    def _counts(
        self,
        context: DecisionContext | None,
        *,
        kind: CardKind | None,
        category: str | None,
        local: bool,
    ) -> tuple[float, float]:
        success = 0.0
        failure = 0.0
        try:
            cards = self.store.snapshot()
        except Exception:
            return (0.0, 0.0)
        for other in cards:
            if kind is not None and other.kind is not kind:
                continue
            if category is not None and other.category != category:
                continue
            events = (
                self.context_model.local_evidence_events(other, context)
                if local
                else self.context_model.evidence_events(other, context)
            )
            if self.first_exposure_only:
                first = _first_non_founding_exposure(events)
                if first is None:
                    continue
                weight, ok = first
                if ok:
                    success += weight
                else:
                    failure += weight
            else:
                s, f = _all_exposures(events)
                success += s
                failure += f
        return (success, failure)
