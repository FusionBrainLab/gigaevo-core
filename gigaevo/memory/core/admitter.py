from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.core.idea_stats import IdeaStats, coerce_metric
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import CardStatsStamper
from gigaevo.memory.shared_memory.models import Quartile

_STAMPER = CardStatsStamper()


def _log_admission(admitter: BaseModel, kept: list[IdeaStats], total: int) -> None:
    if total:
        kept_ids = [s.idea_id for s in kept]
        emit_memory_event(
            component="Admitter",
            event_type="admission.select",
            payload={
                "admitter": type(admitter).__name__,
                "input_count": total,
                "admitted_count": len(kept),
                "admitted_ids": kept_ids,
            },
        )
        logger.debug(
            "[Memory][Admitter] {} admitted {}/{} idea(s): {}",
            type(admitter).__name__,
            len(kept),
            total,
            kept_ids,
        )


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
            if self.reputation.is_confidently_harmful(_STAMPER.harm_statistics(s)):
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
            if self.reputation.is_confidently_harmful(_STAMPER.harm_statistics(s)):
                continue
            seen.add(s.idea_id)
            out.append(s)
        _log_admission(self, out, len(stats))
        return out
