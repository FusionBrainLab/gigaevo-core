"""Mutation-suggestion agent: parent + intra card + memory cards + ancestral trail → suggestions.

Architectural split (see ``lineage_memory.py`` for the descriptive half):

* ``IntraMemoryStage`` builds a purely *descriptive* per-parent lineage card.
* ``MutationSuggestionStage`` (this module's consumer) reads that card +
  the global memory cards + a backward trail through the parent's ancestry
  + the parent's own code/metrics and produces *actionable* suggestions
  the mutator can consume directly.

Output schema reuses ``ProgramInsights`` (type / insight / tag / severity) so
the result is wire-compatible with ``MutationContextStage.insights`` — the
mutator's PROGRAM INSIGHTS scaffold renders the same fields regardless of
which analyst produced them.

The ``ancestral_trail`` slot replaces the prior plateau/stats block: instead
of telling the suggester about the population at large, we tell it about
*this lineage's* recent step_deltas (oriented so positive = improvement) so
it can decide between "extend the winning direction" and "pivot to an
orthogonal mechanism" based on whether the lineage is on a streak or
stalled.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from gigaevo.evolution.mutation.context import EvolutionaryStatisticsMutationContext
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.insights import ProgramInsights
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import OPTIMIZATION_STAGES, Program
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.lineage_memory import INTRA_MEMORY_SIGNAL_METADATA_KEY


class MutationSuggestionState(TypedDict):
    """LangGraph state for mutation-suggestion analysis."""

    program: Program
    intra_card: str | None
    memory_cards: str | None
    ancestral_trail: list[dict[str, Any]] | None
    evolutionary_statistics: EvolutionaryStatistics | None
    mutation_mode: str | None

    messages: list[BaseMessage]
    llm_response: AIMessage | ProgramInsights | None

    insights: ProgramInsights | None
    metadata: dict


class MutationSuggestionAgent(LangGraphAgent):
    """Generate actionable mutation suggestions from program + memory + trail.

    Mirrors ``InsightsAgent`` shape (single program in, structured
    ``ProgramInsights`` out) but extends ``build_prompt`` to splice in three
    optional blocks:

    - ``intra_card`` — per-parent lineage summary (already-tried strategies
      with structured failure inventory). Rendered into ``{intra_block}``.
    - ``memory_cards`` — cross-population top-program excerpts. Rendered into
      ``{memory_cards_block}``.
    - ``ancestral_trail`` — list of ancestor entries (depth_back,
      ancestor_fitness, step_delta) emitted by ``collect_ancestral_trail``.
      Rendered into ``{trail_block}``. `step_delta` is already oriented so
      positive ALWAYS means improvement regardless of metric direction.

    Each slot is the EMPTY STRING when the corresponding input is absent —
    not a placeholder — so the user template collapses cleanly for seed
    programs / no-memory-hook runs.
    """

    StateSchema = MutationSuggestionState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        max_insights: int,
        metrics_formatter: MetricsFormatter,
        metrics_context: MetricsContext,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_insights = max_insights
        self.metrics_formatter = metrics_formatter
        self.metrics_context = metrics_context
        structured_llm = llm.with_structured_output(ProgramInsights)
        super().__init__(structured_llm)

    def build_prompt(self, state: MutationSuggestionState) -> MutationSuggestionState:
        program = state["program"]

        metrics_text = (
            self.metrics_formatter.format_metrics_block(program.metrics)
            if program.metrics
            else "No metrics available"
        )
        errors = program.format_errors(
            include_traceback=True, exclude_stages=set(OPTIMIZATION_STAGES)
        )
        error_section = f"\n\n## ERRORS\n\n{errors}" if errors else ""

        intra = (state.get("intra_card") or "").strip()
        intra_block = (
            f"\n\n## Intra Memory — Per-Parent Lineage Card\n\n{intra}" if intra else ""
        )

        cards = (state.get("memory_cards") or "").strip()
        memory_cards_block = (
            f"\n\n## Memory Cards — Top Evolved Programs from the Global Bank\n\n{cards}"
            if cards
            else ""
        )

        trail_block = self._format_trail_block(state.get("ancestral_trail"))
        stats_block = self._format_stats_block(
            state.get("evolutionary_statistics"), self.metrics_context
        )
        signal = program.metadata.get(INTRA_MEMORY_SIGNAL_METADATA_KEY)
        intra_signal_block = self._format_intra_signal_block(signal)
        mutation_mode_block = self._format_mutation_mode_block(
            state.get("mutation_mode")
        )
        logger.info(
            "[MutationSuggestionAgent] parent={} intra_signal={} mutation_mode={}",
            (program.id[:8] if program.id else "?"),
            (signal.get("severity") if isinstance(signal, dict) else "absent"),
            (state.get("mutation_mode") or "absent"),
        )

        user_prompt = self.user_prompt_template.format(
            code=program.code,
            metrics=metrics_text,
            error_section=error_section,
            intra_block=intra_block,
            memory_cards_block=memory_cards_block,
            trail_block=trail_block,
            stats_block=stats_block,
            intra_signal_block=intra_signal_block,
            mutation_mode_block=mutation_mode_block,
        )

        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return state

    @staticmethod
    def _format_trail_block(trail: list[dict[str, Any]] | None) -> str:
        """Render the ancestral trail as a compact markdown block.

        Returns the empty string when the trail is absent or empty (seed
        programs / programs whose parents lacked the primary metric) so the
        prompt slot collapses without leaving an orphan header.

        ``step_delta`` is rendered with an explicit sign so the LLM cannot
        miss the direction at a glance; the system prompt already documents
        that it is oriented (positive = improvement in the metric's
        direction, regardless of higher_is_better).
        """
        if not trail:
            return ""

        lines: list[str] = [
            (
                "`step_delta` is **oriented**: positive = improvement in the "
                "metric's direction (sign flip for lower-is-better metrics "
                "is already applied). `ancestor_fitness` is given in the "
                "metric's native units."
            ),
            "",
        ]
        for entry in trail:
            depth = entry.get("depth_back")
            fitness = entry.get("ancestor_fitness")
            step_delta = entry.get("step_delta")
            delta_part = (
                "n/a (root ancestor)" if step_delta is None else f"{step_delta:+g}"
            )
            lines.append(
                f"- depth_back={depth}: ancestor_fitness={fitness}, "
                f"step_delta={delta_part}"
            )
        body = "\n".join(lines)
        return (
            "\n\n## Ancestral Momentum — backward trail through this parent's lineage"
            f"\n\n{body}"
        )

    @staticmethod
    def _format_stats_block(
        stats: EvolutionaryStatistics | None, metrics_context: MetricsContext
    ) -> str:
        """Render the population-level evolutionary stats as a markdown block.

        Returns the empty string when no stats snapshot is attached (seed
        programs or pre-collector stages) so the user template collapses
        cleanly. The underlying context renderer already emits a leading
        ``## Evolutionary Statistics`` header, so we only prepend the blank
        lines that separate this block from the preceding slot.
        """
        if stats is None:
            return ""
        body = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=stats,
            metrics_context=metrics_context,
        ).format()
        return f"\n\n{body}"

    @staticmethod
    def _format_mutation_mode_block(mutation_mode: str | None) -> str:
        """Render the downstream MUTATION MODE banner.

        Tells the suggester which mutation operator will consume its output
        (``rewrite`` vs ``diff``) so it can shape substitutes to fit.
        Returns the empty string when unspecified so the slot collapses.
        """
        mode = (mutation_mode or "").strip()
        if not mode:
            return ""
        return f"## MUTATION MODE\nmutation_mode: {mode}\n\n"

    @staticmethod
    def _format_intra_signal_block(signal: dict[str, Any] | None) -> str:
        """Render the structured INTRA STRATEGY SIGNAL block.

        Reads the upstream ``IntraMemoryStage`` signal dict
        (``severity``, ``n_clusters``, ``n_negative``, ``clusters``,
        ``delta_dist``) and emits a compact informational block ALWAYS
        when data exists — severity-tiered guidance lives in system.txt,
        not in this banner.

        Returns the empty string when the signal is absent or empty so
        seed programs / pre-intra runs collapse cleanly.
        """
        if not signal:
            return ""

        severity = signal.get("severity", "healthy")
        clusters = signal.get("clusters") or []
        delta = signal.get("delta_dist") or {}
        n_neg = signal.get("n_negative", 0)
        n_total = signal.get("n_clusters", len(clusters))

        cluster_lines = (
            "\n".join(
                f"- `{c.get('label', '?')}` ({c.get('verdict', '?')}, "
                f"n={c.get('n_attempts', 0)})"
                for c in clusters
            )
            or "- (none)"
        )
        delta_line = (
            f"improving={delta.get('improving', 0)}, "
            f"neutral={delta.get('neutral', 0)}, "
            f"catastrophic={delta.get('catastrophic', 0)}, "
            f"n_failed={delta.get('n_failed', 0)}"
        )
        return (
            "## INTRA STRATEGY SIGNAL\n"
            f"intra_signal: {severity} ({n_neg}/{n_total} clusters negative)\n"
            f"{cluster_lines}\n"
            f"Delta: {delta_line}\n\n"
        )

    def parse_response(self, state: MutationSuggestionState) -> MutationSuggestionState:
        llm_response = state["llm_response"]
        if not isinstance(llm_response, ProgramInsights):
            raise ValueError(
                f"Expected ProgramInsights, got {type(llm_response).__name__}"
            )
        state["insights"] = llm_response
        return state

    async def arun(
        self,
        program: Program,
        intra_card: str | None = None,
        memory_cards: str | None = None,
        ancestral_trail: list[dict[str, Any]] | None = None,
        evolutionary_statistics: EvolutionaryStatistics | None = None,
        mutation_mode: str | None = None,
    ) -> ProgramInsights:
        initial_state: MutationSuggestionState = {
            "program": program,
            "intra_card": intra_card,
            "memory_cards": memory_cards,
            "ancestral_trail": ancestral_trail,
            "evolutionary_statistics": evolutionary_statistics,
            "mutation_mode": mutation_mode,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {
                "program_id": program.id,
                "intra_present": bool((intra_card or "").strip()),
                "memory_cards_present": bool((memory_cards or "").strip()),
                "trail_len": len(ancestral_trail or []),
                "stats_present": evolutionary_statistics is not None,
                "mutation_mode": mutation_mode,
            },
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["insights"]
