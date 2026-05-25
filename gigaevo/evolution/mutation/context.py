from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.evolution.mutation.constants import (  # noqa: F401 — re-export
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.llm.agents.insights import ProgramInsight, ProgramInsights
from gigaevo.llm.agents.lineage import TransitionAnalysis
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.collector import (
    MEDIAN_HORIZON,
    N_MIN_ARCHIVE,
    EvolutionaryStatistics,
)


def _archive_median(
    archive_fitnesses: Sequence[float],
) -> float | None:
    """Return the median of the archive's valid fitnesses, or None if below
    ``N_MIN_ARCHIVE`` samples."""
    valid = [f for f in archive_fitnesses if f is not None]
    if len(valid) < N_MIN_ARCHIVE:
        return None
    s = sorted(valid)
    return s[len(s) // 2]


def _archive_percentile_of_focal(
    focal: float,
    archive_fitnesses: Sequence[float],
    higher_is_better: bool,
) -> int | None:
    """Return integer 0..100 *quality* percentile of focal within the archive.

    The percentile is direction-aware so that **100 = best in archive** and
    **0 = worst in archive** regardless of ``higher_is_better``. Concretely:

    * ``higher_is_better=True``: percentile = fraction of archive entries at or
      below ``focal``.
    * ``higher_is_better=False`` (loss-style metric, lower is better):
      percentile = fraction of archive entries at or above ``focal``.

    Returns ``None`` when the archive is below ``N_MIN_ARCHIVE`` valid samples
    (early bootstrap — too few samples for a meaningful position).
    """
    valid = [f for f in archive_fitnesses if f is not None]
    if len(valid) < N_MIN_ARCHIVE:
        return None
    n = len(valid)
    if higher_is_better:
        rank = sum(1 for f in valid if f <= focal)
    else:
        rank = sum(1 for f in valid if f >= focal)
    return int(round(rank / n * 100))


class MutationContext(BaseModel, ABC):
    """Base class for mutation prompt context."""

    @abstractmethod
    def format(self) -> str:
        """Format context into readable string for mutation prompt."""
        pass


class MetricsMutationContext(MutationContext):
    """Context with program metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: dict[str, float]
    metrics_formatter: MetricsFormatter

    def format(self) -> str:
        lines = ["## Program Metrics", ""]
        formatted = self.metrics_formatter.format_metrics_block(self.metrics)
        lines.append(formatted)
        return "\n".join(lines)


class InsightsMutationContext(MutationContext):
    """Context with program insights."""

    insights: ProgramInsights

    def format(self) -> str:
        if not self.insights.insights:
            return "<No insights available>"

        lines = ["## Program Insights", ""]
        for insight in self.insights.insights:
            header = f"- **[{insight.type}][{insight.tag}][{insight.severity}]**"
            structured_body = self._render_structured(insight)
            if structured_body:
                lines.append(f"{header} {structured_body}")
            else:
                # Legacy free-string fallback (used by the off-path InsightsAgent)
                lines.append(f"{header} — {insight.insight}")

        return "\n".join(lines)

    @staticmethod
    def _render_structured(insight: ProgramInsight) -> str:
        """Render the v2 structured fields when populated; '' when empty.

        The mutator reads this as compact prose; refs/relations are appended
        only when non-empty so the line stays scannable for the simpler
        suggestions while preserving the full grounding chain when present.
        """
        anchor = (insight.anchor_quote or "").strip()
        mechanism = (insight.mechanism or "").strip()
        substitute = (insight.substitute or "").strip()
        if not (anchor or mechanism or substitute):
            return ""

        parts: list[str] = []
        if anchor:
            source = (insight.evidence_source or "").strip()
            parts.append(
                f"anchor `{anchor}` ({source})" if source else f"anchor `{anchor}`"
            )
        if mechanism:
            parts.append(f"mechanism: {mechanism}")
        if substitute:
            parts.append(f"substitute: {substitute}")
        relation = (insight.relation_to_lineage or "").strip()
        if relation:
            parts.append(f"vs lineage: {relation}")
        refs = [r for r in (insight.evidence_refs or []) if r and r.strip()]
        if refs:
            parts.append(f"refs: {', '.join(refs)}")
        return " — " + " | ".join(parts)


class FamilyTreeMutationContext(MutationContext):
    """Context with aggregated ancestor/descendant lineage analyses.

    Receives pre-collected lineage analyses from collector stages
    and formats them into a comprehensive family tree view.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ancestors: list[TransitionAnalysis]
    descendants: list[TransitionAnalysis]
    metrics_formatter: MetricsFormatter

    def format(self) -> str:
        """Format family tree with ancestors and descendants."""
        lines = ["## Lineage", ""]

        logger.debug(
            "[FamilyTreeMutationContext] Formatting with {} ancestors and {} descendants",
            len(self.ancestors),
            len(self.descendants),
        )

        if self.ancestors:
            lines.append("### Parents (transitions that produced this program)")
            lines.append("")
            for i, analysis in enumerate(self.ancestors):
                lines.append(
                    f"#### Parent {i + 1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append("")
                lines.append(self._format_lineage_block(analysis))
                lines.append("")

        if self.descendants:
            lines.append("### Children (mutations already attempted from this program)")
            lines.append("")
            for i, analysis in enumerate(self.descendants):
                lines.append(
                    f"#### Child {i + 1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append("")
                lines.append(self._format_lineage_block(analysis))
                lines.append("")

        result = "\n".join(lines) if len(lines) > 2 else ""
        return result

    def _format_lineage_block(self, analysis: TransitionAnalysis) -> str:
        """Format a single lineage analysis block using format_delta_block for consistency."""
        lines = []

        # Format all metrics using format_delta_block (includes primary + additional)
        formatted_deltas = self.metrics_formatter.format_delta_block(
            parent=analysis.parent_metrics,
            child=analysis.child_metrics,
            include_primary=True,
        )
        lines.append(formatted_deltas)

        if analysis.insights:
            lines.append("")
            lines.append("**Transition insights**:")
            for insight in analysis.insights.insights:
                lines.append(f"  - **[{insight.strategy}]** — {insight.description}")

        return "\n".join(lines)


class EvolutionaryStatisticsMutationContext(MutationContext):
    """Context with evolutionary statistics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    evolutionary_statistics: EvolutionaryStatistics
    metrics_context: MetricsContext

    def format(self) -> str:
        stats = self.evolutionary_statistics
        primary_key = self.metrics_context.get_primary_key()
        decimals = self.metrics_context.get_decimals(primary_key)

        def fmt(v: float | None) -> str:
            return f"{v:.{decimals}f}" if v is not None else "-"

        lines: list[str] = ["## Evolutionary Statistics", ""]

        if stats.iter_window_lo is None or stats.iter_window_hi is None:
            lines.append(
                f"This program: gen={stats.generation}  "
                f"(no iteration recorded — window unavailable)"
            )
            lines.append(
                f"Population: total={stats.total_program_count}  "
                f"valid_rate={stats.valid_rate * 100:.1f}%"
            )
            return "\n".join(lines)

        focal_valid = self.metrics_context.is_valid(stats.current_program_metrics)
        focal_fit = stats.current_program_metrics.get(primary_key)
        focal_fit_str = (
            fmt(focal_fit) if focal_valid and focal_fit is not None else "INVALID"
        )

        # R7+R8 (v3.1): pure distributional render.
        #
        # We emit:
        #   - `Archive: N=… worst=… median=… best=…` — the archive's empirical
        #     distribution (universal: no per-task constants, no target).
        #   - `archive-percentile pXX of N=Y` on the rank line — focal's
        #     direction-aware quality percentile (100=best by quality,
        #     regardless of higher_is_better).
        #
        # We deliberately DO NOT emit:
        #   - A `Regime:` literal (pre-baked classifier — flagged 2026-05-18).
        #   - A `Target:` line (the task description already states the target;
        #     rendering it as a regex-matchable token duplicates the task
        #     description and re-creates the same pre-baking flaw — flagged
        #     2026-05-18).
        #
        # The R9 prose gate in mutation/system.txt uses the archive-percentile
        # as the universal deterministic gate; the task description + archive
        # distribution are read by the LLM for qualitative target/non-linearity
        # judgment.
        primary_spec = self.metrics_context.specs.get(primary_key)
        higher_is_better = (
            primary_spec.higher_is_better if primary_spec is not None else True
        )

        archive_percentile: int | None = None
        archive_line: str | None = None
        if focal_valid and focal_fit is not None:
            archive_percentile = _archive_percentile_of_focal(
                focal_fit, stats.archive_valid_fitnesses, higher_is_better
            )
            archive_median = _archive_median(stats.archive_valid_fitnesses)
            if archive_percentile is not None and archive_median is not None:
                sorted_archive = sorted(
                    f for f in stats.archive_valid_fitnesses if f is not None
                )
                archive_worst = (
                    sorted_archive[0] if higher_is_better else sorted_archive[-1]
                )
                archive_best = (
                    sorted_archive[-1] if higher_is_better else sorted_archive[0]
                )
                n_archive = len(sorted_archive)
                archive_line = (
                    f"Archive: N={n_archive}  worst={fmt(archive_worst)}  "
                    f"median={fmt(archive_median)}  best={fmt(archive_best)}"
                )

        rank_part = ""
        if stats.iter_window_rank is not None and stats.iter_window_valid > 0:
            rank_part = (
                f" (rank {stats.iter_window_rank}/{stats.iter_window_valid} in window"
            )
            if archive_percentile is not None:
                n_archive = len(stats.archive_valid_fitnesses)
                rank_part += (
                    f", archive-percentile p{archive_percentile} of N={n_archive}"
                )
            rank_part += ")"

        iteration_str = stats.iteration if stats.iteration is not None else "?"
        lines.append(
            f"This program: iter={iteration_str}  gen={stats.generation}  "
            f"fitness={focal_fit_str}{rank_part}"
        )
        lines.append(
            f"Window: iters [{stats.iter_window_lo}..{stats.iter_window_hi}]  "
            f"programs={stats.iter_window_programs}  valid={stats.iter_window_valid}"
        )
        if archive_line is not None:
            lines.append(archive_line)

        # No Target: line — the task description states the target/bound, and
        # the LLM reads it from there. Rendering it here would duplicate the
        # task description as a pre-baked regex token (same flaw as the
        # categorical Regime: literal we dropped in v3).

        t1, t2, t3 = stats.iter_window_trend_thirds
        if t1 is not None and t2 is not None and t3 is not None:
            trend_detail = f"[medians of thirds: {fmt(t1)} → {fmt(t2)} → {fmt(t3)}]"
        else:
            trend_detail = (
                f"[only {stats.iter_window_valid} valid programs — too few for trend]"
            )
        lines.append(f"Trend (window): {stats.iter_window_trend}  {trend_detail}")

        if (
            stats.iter_window_best_fitness is not None
            and stats.iter_window_best_iter is not None
        ):
            offset = stats.iter_window_best_iter - (stats.iteration or 0)
            lines.append(
                f"Best in window: {fmt(stats.iter_window_best_fitness)} "
                f"at iter {stats.iter_window_best_iter}  "
                f"(offset {offset:+d} from this program)"
            )

        lines.append(
            f"Iters since last new best (global): {stats.iters_since_last_new_best}"
        )

        if stats.iter_window_median_before is not None:
            lines.append(
                f"Median fitness, {MEDIAN_HORIZON} iters before this program: "
                f"{fmt(stats.iter_window_median_before)}"
            )
        if stats.iter_window_median_after is not None:
            lines.append(
                f"Median fitness, {MEDIAN_HORIZON} iters after this program:  "
                f"{fmt(stats.iter_window_median_after)}"
            )

        lines.append(
            "Invalid streak (max consecutive in window): "
            f"{stats.iter_window_invalid_streak_max}"
        )
        if stats.iter_window_programs > 0:
            pct = stats.iter_window_invalid_count / stats.iter_window_programs * 100
            lines.append(
                f"Recent failure rate: {stats.iter_window_invalid_count}/"
                f"{stats.iter_window_programs} invalid ({pct:.1f}%)"
            )

        return "\n".join(lines)


class ArtifactMutationContext(MutationContext):
    """Context with an execution artifact from the validator.

    The artifact is assumed to be data produced by the validator (e.g. arrays,
    images, structured data). The default :meth:`format`
    simply prints a string representation of the artifact for the mutation
    prompt. You can subclass this class and override :meth:`format` to implement
    your own logic for processing or formatting the artifact (e.g. image
    descriptions, array summaries, or extracting specific fields) before it is
    shown to the LLM.
    """

    artifact: Any

    def format(self) -> str:
        """Format the artifact for the mutation prompt.

        Default: use a string representation of the artifact (e.g. repr for
        arrays/dicts). Override in a subclass to implement custom processing
        (e.g. array summaries, image descriptions, or structured output).
        """
        body = self.artifact if self.artifact is not None else "<no artifact>"
        if not isinstance(body, str):
            body = repr(body)
        return f"## Execution Artifact\n\n{body}"


class MemoryMutationContext(MutationContext):
    """Context with memory-selected idea cards."""

    memory_block: str

    def format(self) -> str:
        if not self.memory_block.strip():
            return ""
        return f"## Memory Instructions\n{self.memory_block}"


class PreformattedMutationContext(MutationContext):
    """Preformatted string block (e.g. from FormatterStage) included as-is in the mutation prompt."""

    content: str

    def format(self) -> str:
        return self.content


class CompositeMutationContext(MutationContext):
    """Aggregator that composes multiple mutation contexts."""

    contexts: list[MutationContext]

    def format(self) -> str:
        formatted_parts = [ctx.format() for ctx in self.contexts]
        non_empty = [part for part in formatted_parts if part.strip()]
        # Use a clear separator between different context types
        return "\n\n---\n\n".join(non_empty) if non_empty else "No context available."
