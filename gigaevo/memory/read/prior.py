"""Cold-card prior policies for the memory read bandit."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import math
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext, MemoryContextModel
from gigaevo.memory.context.beta import BetaPrior, coerce_beta_prior
from gigaevo.memory.context.evidence import event_weight, split_events_by_task
from gigaevo.memory.events import MemoryPriorCohort, emit_memory_event
from gigaevo.memory.prior_evidence import EvictedEvidenceSource
from gigaevo.memory.storage.base import MemoryStore


@runtime_checkable
class MemoryPrior(Protocol):
    """Supplies cold-card priors for a read decision."""

    def cold_card_prior(
        self, card: Card, context: DecisionContext | None = None
    ) -> BetaPrior: ...


class FixedMemoryPrior(BaseModel):
    """Static prior policy; preserves the historical ``Beta(3, 3)`` default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


def _first_non_founding_exposure(
    events: Iterable[ContextualGain],
) -> tuple[float, bool] | None:
    """Return ``(weight, success)`` for the first causal non-founding outcome.

    Unused exposures are skipped entirely: an ignored card says nothing about
    whether its advice helps when acted on, and counting ignores as failures
    would make the cold prior track the mutator use-rate instead of
    ``P(help | used)``.

    Success requires strictly positive causal gain. Zero gain is neutral, not
    help, so it contributes to the failure/complement mass with negative and
    invalid outcomes.
    """

    ordered = sorted(
        events,
        key=lambda event: (
            event.context.timestamp is None,
            event.context.timestamp or datetime.min,
        ),
    )
    for event in ordered:
        if event.founding or event.unused:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid:
            return (weight, False)
        if event.gain is not None and math.isfinite(float(event.gain)):
            return (weight, float(event.gain) > 0.0)
    return None


def _all_exposures(events: Iterable[ContextualGain]) -> tuple[float, float]:
    """Count weighted causal outcomes, requiring strictly positive gain for success.

    Zero gain is neutral rather than help and contributes to the
    failure/complement mass, as do negative and invalid outcomes.
    """

    success = 0.0
    failure = 0.0
    for event in events:
        if event.founding or event.unused:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid:
            failure += weight
        elif event.gain is not None and math.isfinite(float(event.gain)):
            if float(event.gain) > 0.0:
                success += weight
            else:
                failure += weight
    return success, failure


_COHORT_LEVEL_TOKENS = frozenset({"kind", "category", "context", "task"})


class EmpiricalBayesMemoryPrior(BaseModel):
    """Contextual hierarchical cold prior learned from banked first exposures.

    The cohort ladder is config-driven: the global cohort is the implicit top
    level, and ``levels`` names the refinements below it as ``'+'``-joined
    subsets of ``{kind, category, context, task}``. Context-bearing levels apply
    only when the context model resolves a non-global bucket, task-bearing
    levels apply only for stamped decisions, and a level whose
    counts equal the previously applied level's carries no new information and
    is skipped — the same evidence must not compound its shrinkage once per
    level.

    The prior is intentionally conservative: it uses the first non-founding
    exposure per card by default, shrinks toward a seed prior, and caps the
    concentration.  This avoids letting repeatedly sampled warm winners dictate
    what an untested card should believe.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    store: MemoryStore = Field(description="Memory bank providing prior evidence.")
    evicted_evidence: EvictedEvidenceSource | None = Field(
        default=None,
        description="Optional evidence source of harm-evicted cards; when set, the EB "
        "cohort counts union the live snapshot with evicted-card first-exposures to "
        "correct survivorship bias. Default None = snapshot-only (legacy, "
        "byte-identical).",
    )
    context_model: MemoryContextModel = Field(default_factory=GlobalMemoryContext)
    levels: tuple[str, ...] = Field(
        default=(
            "kind",
            "kind+category",
            "task+kind+category",
            "context",
            "context+kind",
            "context+kind+category",
            "task+context+kind+category",
        ),
        description="Cohort refinement ladder below the implicit global top level; "
        "each entry is a '+'-joined subset of {kind, category, context, task}.",
    )
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

    @model_validator(mode="after")
    def _monotonic_concentration_range(self) -> EmpiricalBayesMemoryPrior:
        if self.k_max < self.k_min:
            raise ValueError(
                f"k_max ({self.k_max}) must be greater than or equal to "
                f"k_min ({self.k_min})"
            )
        return self

    @field_validator("levels")
    @classmethod
    def _known_level_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[frozenset[str]] = set()
        previous: dict[bool, frozenset[str]] = {}
        saw_local = False
        for level in value:
            tokens = tuple(token.strip() for token in str(level).split("+"))
            if any(not token for token in tokens):
                raise ValueError(f"empty token in cohort level {level!r}")
            unknown = set(tokens) - _COHORT_LEVEL_TOKENS
            if unknown:
                raise ValueError(
                    f"unknown cohort token(s) {sorted(unknown)} in level {level!r}; "
                    f"valid tokens: {sorted(_COHORT_LEVEL_TOKENS)}"
                )
            key = frozenset(tokens)
            if len(key) != len(tokens) or key in seen:
                raise ValueError(
                    f"duplicate token or level in cohort ladder: {level!r}"
                )
            seen.add(key)
            # The parent_mu shrinkage chain is only well-founded when each
            # applied level refines the previous one, so global levels must
            # precede context-bearing levels and each block must be a strict
            # token-superset chain.
            local = "context" in key
            if not local and saw_local:
                raise ValueError(
                    f"global cohort level {level!r} after a context-bearing level"
                )
            saw_local = saw_local or local
            prev = previous.get(local)
            if prev is not None and not prev < key:
                raise ValueError(
                    f"cohort ladder must refine monotonically: {level!r} does not "
                    f"refine the previous {'context' if local else 'global'} level"
                )
            previous[local] = key
        return tuple(value)

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
        try:
            bank = tuple(self.store.snapshot())
        except Exception:
            return best
        live_card_count = len(bank)
        evicted_card_count = 0
        if self.evicted_evidence is not None:
            try:
                known_ids = {banked_card.id for banked_card in bank}
                evicted_cards: list[Card] = []
                for evicted_card in self.evicted_evidence.cards():
                    if evicted_card.id in known_ids:
                        continue
                    evicted_cards.append(evicted_card)
                    known_ids.add(evicted_card.id)
                bank = (*bank, *evicted_cards)
                evicted_card_count = len(evicted_cards)
            except Exception as exc:
                logger.warning(
                    "[Memory][Prior] failed to read evicted evidence: {}", exc
                )
        try:
            emit_memory_event(
                MemoryPriorCohort(
                    live_card_count=live_card_count,
                    evicted_card_count=evicted_card_count,
                    cohort_card_count=len(bank),
                )
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Prior] cohort telemetry emit failed; continuing"
            )
        # Per-card exposure outcomes are level-independent within the global /
        # local split, so one pass over one snapshot serves the whole ladder.
        counts_cache: dict[tuple[bool, bool], dict[str, tuple[float, float]]] = {}
        applied: tuple[float, float] | None = None
        for source, kind, category, local, task_local in self._cohort_specs(
            card, context
        ):
            counts = self._cohort_counts(
                bank,
                context,
                kind=kind,
                category=category,
                local=local,
                task_local=task_local,
                cache=counts_cache.setdefault((local, task_local), {}),
            )
            success, failure = counts
            n = success + failure
            if n <= 0.0:
                continue
            if counts == applied:
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
            # parent_mu deliberately carries across the global→local boundary:
            # the deepest informative global cohort is the shrinkage parent for
            # the first context-bearing level.
            parent_mu = mu
            applied = counts
        return best

    def _cohort_specs(
        self, card: Card, context: DecisionContext | None
    ) -> list[tuple[str, CardKind | None, str | None, bool, bool]]:
        specs: list[tuple[str, CardKind | None, str | None, bool, bool]] = [
            ("eb_global", None, None, False, False)
        ]
        context_is_local = self.context_model.key_for(context).kind != "global"
        has_task = context is not None and bool(context.task_key)
        for level in self.levels:
            tokens = tuple(token.strip() for token in level.split("+"))
            local = "context" in tokens
            if local and not context_is_local:
                continue
            task_local = "task" in tokens
            if task_local and not has_task:
                continue
            specs.append(
                (
                    "eb_" + "_".join(tokens),
                    card.kind if "kind" in tokens else None,
                    card.category if "category" in tokens else None,
                    local,
                    task_local,
                )
            )
        return specs

    def _cohort_counts(
        self,
        bank: tuple[Card, ...],
        context: DecisionContext | None,
        *,
        kind: CardKind | None,
        category: str | None,
        local: bool,
        task_local: bool,
        cache: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        success = 0.0
        failure = 0.0
        for other in bank:
            if kind is not None and other.kind is not kind:
                continue
            if category is not None and other.category != category:
                continue
            entry = cache.get(other.id)
            if entry is None:
                entry = self._card_counts(
                    other, context, local=local, task_local=task_local
                )
                cache[other.id] = entry
            success += entry[0]
            failure += entry[1]
        return (success, failure)

    def _card_counts(
        self,
        other: Card,
        context: DecisionContext | None,
        *,
        local: bool,
        task_local: bool,
    ) -> tuple[float, float]:
        scoped = other
        if task_local or local:
            assert context is not None
            native, _ = split_events_by_task(other.gain_events, context.task_key)
            scoped = other.model_copy(update={"gain_events": native})
        events = (
            self.context_model.local_evidence_events(scoped, context)
            if local
            else self.context_model.evidence_events(scoped, None)
        )
        if self.first_exposure_only:
            first = _first_non_founding_exposure(events)
            if first is None:
                return (0.0, 0.0)
            weight, ok = first
            return (weight, 0.0) if ok else (0.0, weight)
        return _all_exposures(events)
