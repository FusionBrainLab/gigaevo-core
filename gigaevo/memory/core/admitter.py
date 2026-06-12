from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.idea_stats import IdeaStats, coerce_metric
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import EvolutionStatistics, Quartile

_QUARTILE_PREF_RANK = {
    q: rank for rank, q in enumerate((Quartile.ALL, *reversed(Quartile.quarters())))
}
_MISSING_SCORE = -1e18


def harm_statistics(stats: IdeaStats) -> EvolutionStatistics:
    """Lift one origin-analysis row into the typed statistics shape the
    reputation harm predicate reads (the row becomes the ALL block)."""
    return EvolutionStatistics(ALL=stats.to_stats_block())


def _log_admission(admitter: BaseModel, kept: list[IdeaStats], total: int) -> None:
    if total:
        logger.debug(
            "[Memory][Admitter] {} admitted {}/{} idea(s): {}",
            type(admitter).__name__,
            len(kept),
            total,
            [s.idea_id for s in kept],
        )


def _dedup_sorted_by_idea(stats: list[IdeaStats]) -> list[IdeaStats]:
    """One row per idea, sorted by idea_id: prefer the ALL block, then the
    highest median (missing metrics rank last)."""

    def key(s: IdeaStats) -> tuple[str, int, float]:
        score = coerce_metric(s.IntroGain_best_median)
        return (
            s.idea_id,
            _QUARTILE_PREF_RANK[s.quartile],
            -(score if score is not None else _MISSING_SCORE),
        )

    out: list[IdeaStats] = []
    seen: set[str] = set()
    for s in sorted(stats, key=key):
        if s.idea_id in seen:
            continue
        seen.add(s.idea_id)
        out.append(s)
    return out


class TieredAdmitter(BaseModel):
    """Tiered admission gate over the per-idea origin-analysis rows.

    Base conditions (events >= min, relative median above floor, downside below
    cap) plus an event-count-tiered confidence ladder; one row per idea kept,
    preferring the ALL block then the highest median. Missing or NaN metrics
    never satisfy a condition (``coerce_metric`` semantics).
    """

    model_config = ConfigDict(frozen=True)

    min_intro_events: int = Field(
        default=1, description="Minimum intro events for any row to be considered."
    )
    min_rel_median: float = Field(
        default=0.01,
        description="Floor on the relative median IntroGain (base condition).",
    )
    max_downside: float = Field(
        default=0.4, description="Cap on the downside rate (base condition)."
    )
    min_sibling_win_ge3: float = Field(
        default=0.5,
        description="Sibling win-rate floor for the confident tier.",
    )
    confident_tier_events: int = Field(
        default=3,
        description="Intro-event count at which the confident tier applies.",
    )
    pair_tier_events: int = Field(
        default=2,
        description="Intro-event count at which the pair tier applies.",
    )
    eps: float = Field(
        default=1e-12,
        description="Tolerance when comparing rates against exactly 1.0.",
    )

    def select(self, stats: Sequence[IdeaStats]) -> list[IdeaStats]:
        kept = _dedup_sorted_by_idea([s for s in stats if self._keep(s)])
        _log_admission(self, kept, len(stats))
        return kept

    def _keep(self, s: IdeaStats) -> bool:
        rel_median = coerce_metric(s.IntroGain_best_rel_median)
        downside = coerce_metric(s.DownsideRate_best)
        base_ok = (
            s.intro_events >= self.min_intro_events
            and rel_median is not None
            and rel_median > self.min_rel_median
            and downside is not None
            and downside < self.max_downside
        )
        if not base_ok:
            return False
        sib_win_all = coerce_metric(s.SiblingWinRate_allgens)
        p10 = coerce_metric(s.IntroGain_best_p10)
        born_rate = coerce_metric(s.BornInElite_rate)
        cond_ge3 = (
            s.intro_events >= self.confident_tier_events
            and sib_win_all is not None
            and sib_win_all >= self.min_sibling_win_ge3
        )
        cond_eq2 = (
            s.intro_events == self.pair_tier_events
            and p10 is not None
            and p10 > 0
            and sib_win_all is not None
            and sib_win_all >= 1.0 - self.eps
        )
        cond_eq1 = (
            s.intro_events == 1
            and born_rate is not None
            and born_rate >= 1.0 - self.eps
        )
        return cond_ge3 or cond_eq2 or cond_eq1


class SignBasedAdmitter(BaseModel):
    """Variant-C admission: an idea enters the bank iff its ALL-block evidence
    exists (events >= min), its median gain is positive, and the reputation
    posterior does not confidently mark it harmful. Quartile rows are
    informational only; the ALL block decides."""

    model_config = ConfigDict(frozen=True)

    min_intro_events: int = Field(
        default=1, description="Minimum ALL-block intro events to consider an idea."
    )
    min_median: float = Field(
        default=0.0,
        description="Median IntroGain must strictly exceed this to admit.",
    )
    reputation: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Posterior model deciding the confidently-harmful veto.",
    )

    def select(self, stats: Sequence[IdeaStats]) -> list[IdeaStats]:
        out: list[IdeaStats] = []
        seen: set[str] = set()
        for s in stats:
            if s.quartile is not Quartile.ALL or s.idea_id in seen:
                continue
            if s.intro_events < self.min_intro_events:
                continue
            median = coerce_metric(s.IntroGain_best_median)
            if median is None or not median > self.min_median:
                continue
            if self.reputation.is_confidently_harmful(harm_statistics(s)):
                continue
            seen.add(s.idea_id)
            out.append(s)
        _log_admission(self, out, len(stats))
        return out


class PermissiveAdmitter(BaseModel):
    """Admit every idea with ALL-block evidence that is not confidently harmful
    (replay policy B); no sign condition on the median."""

    model_config = ConfigDict(frozen=True)

    min_intro_events: int = Field(
        default=1, description="Minimum ALL-block intro events to consider an idea."
    )
    reputation: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Posterior model deciding the confidently-harmful veto.",
    )

    def select(self, stats: Sequence[IdeaStats]) -> list[IdeaStats]:
        out: list[IdeaStats] = []
        seen: set[str] = set()
        for s in stats:
            if s.quartile is not Quartile.ALL or s.idea_id in seen:
                continue
            if s.intro_events < self.min_intro_events:
                continue
            if self.reputation.is_confidently_harmful(harm_statistics(s)):
                continue
            seen.add(s.idea_id)
            out.append(s)
        _log_admission(self, out, len(stats))
        return out
