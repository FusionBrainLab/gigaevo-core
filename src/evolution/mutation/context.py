from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel

from src.llm.agents.insights import ProgramInsights
from src.llm.agents.lineage import TransitionAnalysis
from src.programs.metrics.formatter import MetricsFormatter

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
            lines.append("### Ancestors")
            lines.append("")
            for i, analysis in enumerate(self.ancestors):
                lines.append(
                    f"#### Ancestor {i+1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append("")
                lines.append(self._format_lineage_block(analysis))
                lines.append("")

        if self.descendants:
            logger.debug(
                f"[FamilyTreeMutationContext] Adding descendants section with {len(self.descendants)} descendants"
            )
            lines.append("### Descendants")
            lines.append("")
            for i, analysis in enumerate(self.descendants):
                logger.debug(
                    f"[FamilyTreeMutationContext] Adding descendant {i+1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append(
                    f"#### Descendant {i+1}: {analysis.from_id[:8]} → {analysis.to_id[:8]}"
                )
                lines.append("")
                lines.append(self._format_lineage_block(analysis))
                lines.append("")
        else:
            logger.debug(f"[FamilyTreeMutationContext] No descendants to add")

        result = "\n".join(lines) if len(lines) > 2 else ""
        logger.debug(
            f"[FamilyTreeMutationContext] Formatted result length: {len(result)}"
        )
        logger.debug(
            f"[FamilyTreeMutationContext] Formatted result contains '### Descendants': {'### Descendants' in result}"
        )

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


class CompositeMutationContext(MutationContext):
    """Aggregator that composes multiple mutation contexts."""

    contexts: list[MutationContext]

    def format(self) -> str:
        formatted_parts = [ctx.format() for ctx in self.contexts]
        non_empty = [part for part in formatted_parts if part.strip()]
        return "\n\n".join(non_empty) if non_empty else "No context available."
