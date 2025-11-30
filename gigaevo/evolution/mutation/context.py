from abc import ABC, abstractmethod

from loguru import logger
from pydantic import BaseModel

from gigaevo.llm.agents.insights import ProgramInsights
from gigaevo.llm.agents.lineage import TransitionAnalysis
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.collector import EvolutionaryStatistics

MUTATION_CONTEXT_METADATA_KEY = "mutation_context"


class MutationContext(BaseModel, ABC):
    """Base class for mutation prompt context."""

    @abstractmethod
    def format(self) -> str:
        """Format context into readable string for mutation prompt."""
        pass


class MetricsMutationContext(MutationContext):
    """Context with program metrics."""

    metrics: dict[str, float]
    metrics_formatter: MetricsFormatter

    class Config:
        arbitrary_types_allowed = True

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
            lines.append(
                f"- **[{insight.type}]**[{insight.tag}]**[{insight.severity}]**: {insight.insight}"
            )

        return "\n".join(lines)


class FamilyTreeMutationContext(MutationContext):
    """Context with aggregated ancestor/descendant lineage analyses.

    Receives pre-collected lineage analyses from collector stages
    and formats them into a comprehensive family tree view.
    """

    ancestors: list[TransitionAnalysis]
    descendants: list[TransitionAnalysis]
    metrics_formatter: MetricsFormatter

    class Config:
        arbitrary_types_allowed = True

    def format(self) -> str:
        """Format family tree with ancestors and descendants."""
        lines = ["## Family Tree (Current State)", ""]

        logger.debug(
            f"[FamilyTreeMutationContext] Formatting with {len(self.ancestors)} ancestors and {len(self.descendants)} descendants"
        )

        if self.ancestors:
            lines.append("### Parents")
            lines.append("")
            for i, analysis in enumerate(self.ancestors):
                lines.append(
                    f"#### Parent {i + 1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append("")
                lines.append(self._format_lineage_block(analysis))
                lines.append("")

        if self.descendants:
            lines.append("### Children")
            lines.append("")
            for i, analysis in enumerate(self.descendants):
                lines.append(
                    f"#### Child {i + 1}: {analysis.from_id[:4]} → {analysis.to_id[:4]}"
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
            lines.append("**Transition Insights**:")
            for insight in analysis.insights.insights:
                strategy = insight.strategy
                desc = insight.description
                lines.append(f"  - **[{strategy}]**: {desc}")

        return "\n".join(lines)


class EvolutionaryStatisticsMutationContext(MutationContext):
    """Context with evolutionary statistics."""

    evolutionary_statistics: EvolutionaryStatistics
    metrics_context: MetricsContext

    class Config:
        arbitrary_types_allowed = True

    def format(self) -> str:
        """Format evolutionary statistics into readable string for mutation prompt."""
        stats = self.evolutionary_statistics
        lines = ["## Evolutionary Statistics", ""]

        # Current generation (← marks it in table)
        lines.append(
            f"**Generation**: {stats.generation} | **Total Programs**: {stats.total_program_count}"
        )
        lines.append("")

        # Generation history table - LLM interprets trends directly
        lines.append(self._format_generation_history_table())

        return "\n".join(lines)

    def _format_generation_history_table(self) -> str:
        """Format generation history as a compact table (window around current generation).

        Shows past generations for context and future generations to peek ahead.
        Window: [-3, +3] around current generation.
        """
        history = self.evolutionary_statistics.generation_history
        if not history:
            return "_No generation history available_"

        primary_key = self.metrics_context.get_primary_key()
        decimals = self.metrics_context.get_decimals(primary_key)

        lines = []
        lines.append(
            "| Gen | Best | Avg | Worst | Valid % | #Progs | Avg Children | Max Children |"
        )
        lines.append(
            "|-----|------|-----|-------|---------|--------|--------------|--------------|"
        )

        # Show window around current generation: [-3, +3]
        current_gen = self.evolutionary_statistics.generation
        window_start = current_gen - 3
        window_end = current_gen + 3

        # Get generations within window that exist in history
        gens_in_window = sorted(
            g for g in history.keys() if window_start <= g <= window_end
        )

        for gen_num in gens_in_window:
            metrics = history[gen_num]
            marker = " ←" if gen_num == current_gen else ""

            # Format fitness values, handling None
            best_str = (
                f"{metrics.best:.{decimals}f}" if metrics.best is not None else "-"
            )
            avg_str = (
                f"{metrics.average:.{decimals}f}"
                if metrics.average is not None
                else "-"
            )
            worst_str = (
                f"{metrics.worst:.{decimals}f}" if metrics.worst is not None else "-"
            )

            lines.append(
                f"| {gen_num}{marker} | {best_str} | {avg_str} | "
                f"{worst_str} | {metrics.valid_rate * 100:.1f}% | "
                f"{metrics.program_count} | {metrics.avg_num_children:.2f} | {metrics.max_num_children} |"
            )

        return "\n".join(lines)


class CompositeMutationContext(MutationContext):
    """Aggregator that composes multiple mutation contexts."""

    contexts: list[MutationContext]

    def format(self) -> str:
        formatted_parts = [ctx.format() for ctx in self.contexts]
        non_empty = [part for part in formatted_parts if part.strip()]
        # Use a clear separator between different context types
        return "\n\n---\n\n".join(non_empty) if non_empty else "No context available."
