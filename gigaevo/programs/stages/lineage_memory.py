"""In-run lineage memory stages: per-program (intra) cards.

IntraMemoryStage: runs on the CURRENT program X — the program whose lineage
the card summarises. Receives X's children ids as a stage input
(``children_ids: StringList``, produced upstream by ``DescendantProgramIds``)
plus an optional ``memory_cards: StringContainer`` block from
``MemoryContextStage``. Asks an LLM to render a per-parent lineage card and
writes it to ``X.metadata['intra_memory_card']``. The framework cache
(``InputHashCache``) keys on the children_ids + memory_cards hash, so the
LLM call is skipped whenever the input set is unchanged. When the engine's
``ParentRefresher`` flips X from DONE→QUEUED after a new child eval, the
upstream collector returns a new id list → hash changes → card re-renders.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StageIO, StringContainer, StringList
from gigaevo.programs.stages.stage_registry import StageRegistry

INTRA_MEMORY_CARD_METADATA_KEY = "intra_memory_card"
# Legacy signature key kept for back-compat reads of programs persisted by the
# pre-DAG-native stage; new runs do not write this field.
INTRA_MEMORY_CARD_SIGNATURE_KEY = "intra_memory_card_signature"

# Context lines around each diff hunk. Five gives the analyst enough surrounding
# code to locate the edit without bloating the payload.
_DIFF_CONTEXT_LINES = 5

# Trail size caps (depth-vs-payload tradeoff). Not task-tuning knobs: depth
# is bounded so the BFS doesn't walk the whole DAG, and the ancestor cap is
# a defensive size limit on prompt payload. Both are deliberately loose and
# kept here as constants rather than as constructor parameters because they
# are mechanism limits, not behavioural knobs to tune per problem.
DEFAULT_TRAIL_MAX_DEPTH = 8
DEFAULT_TRAIL_MAX_ANCESTORS = 32


async def collect_ancestral_trail(
    descendant: Program,
    storage: ProgramStorage,
    metrics_context: MetricsContext,
    *,
    max_depth: int = DEFAULT_TRAIL_MAX_DEPTH,
    max_total_ancestors: int = DEFAULT_TRAIL_MAX_ANCESTORS,
) -> list[dict[str, Any]]:
    """BFS the descendant's parent DAG backward, summarising each ancestor's step.

    The framework supports `num_parents > 1` (crossover). We BFS through the
    full parent DAG rather than following `parents[0]` so crossover-derived
    lineages don't lose their second parent's history. At each depth level we
    visit every parent of every node from the previous level, deduplicated by
    `Program.id` (a diamond pattern never double-counts), and emit one entry
    per ancestor:

    * ``depth_back`` — BFS layer (1 = direct parent(s), 2 = grandparents, ...).
    * ``ancestor_fitness`` — primary-metric value on the ancestor (raw, in the
      metric's native direction).
    * ``step_delta`` — ORIENTED gain the ancestor made over its OWN parents,
      taken as MAX across the ancestor's parents of
      ``(ancestor_fitness − one_parent.fitness) * sign``, where
      ``sign = +1`` for higher-is-better metrics and ``-1`` for
      lower-is-better metrics. After orientation, positive ALWAYS means
      "ancestor improved on its parent" regardless of metric direction, so
      downstream consumers can reason uniformly. `None` for root ancestors
      (no further parents in DAG).

    Orientation is critical: a raw subtraction would be correct only for
    higher-is-better metrics. For lower-is-better metrics (e.g. minimising
    error / area / loss), `ancestor_fit - grand_fit > 0` would mean the
    ancestor REGRESSED, and a naive max-aggregation would surface the worst
    sibling step instead of the best. Orienting the delta and then taking
    max-across-parents always picks the genuinely-best contribution.

    We deliberately do NOT precompute a "breakthrough" flag: deciding what
    delta counts as a meaningful gain is task-dependent, so the downstream
    LLM judges it from the oriented `step_delta` against the metric's scale
    rather than us hardcoding a threshold here.

    Walk terminates when: the frontier is empty (DAG root reached on every
    branch), ``max_depth`` BFS layers exhausted, or ``max_total_ancestors``
    entries collected (hard size cap on the payload). Storage faults on
    individual reads are absorbed silently — the walk continues with whatever
    ancestors did load.

    Returns an empty list when the descendant has no parents (seed program);
    callers should omit the trail block entirely in that case.
    """
    primary_key = metrics_context.get_primary_key()
    if not descendant.lineage or not descendant.lineage.parents:
        return []

    # +1 for higher-is-better, -1 for lower-is-better. Applied to raw deltas
    # so that "positive step_delta = improvement" holds uniformly downstream.
    sign: float = 1.0 if metrics_context.is_higher_better(primary_key) else -1.0

    visited: set[str] = set()
    trail: list[dict[str, Any]] = []
    frontier: list[str] = list(descendant.lineage.parents)

    for depth_back in range(1, max_depth + 1):
        if not frontier or len(trail) >= max_total_ancestors:
            break
        next_frontier: list[str] = []
        for ancestor_id in frontier:
            if ancestor_id in visited:
                continue
            visited.add(ancestor_id)
            try:
                ancestor = await storage.get(ancestor_id)
            except Exception:
                continue
            if ancestor is None:
                continue
            ancestor_fit = ancestor.metrics.get(primary_key)
            if ancestor_fit is None:
                continue

            grand_ids: list[str] = []
            if ancestor.lineage and ancestor.lineage.parents:
                grand_ids = list(ancestor.lineage.parents)

            step_delta: float | None = None
            for grand_id in grand_ids:
                try:
                    grand = await storage.get(grand_id)
                except Exception:
                    continue
                if grand is None:
                    continue
                grand_fit = grand.metrics.get(primary_key)
                if grand_fit is None:
                    continue
                # Orient so positive = improvement in the metric's native
                # direction, then keep the best (max) contribution across
                # this ancestor's parents.
                oriented = (ancestor_fit - grand_fit) * sign
                if step_delta is None or oriented > step_delta:
                    step_delta = oriented
                if grand_id not in visited:
                    next_frontier.append(grand_id)

            trail.append(
                {
                    "depth_back": depth_back,
                    "ancestor_fitness": ancestor_fit,
                    "step_delta": step_delta,
                }
            )
            if len(trail) >= max_total_ancestors:
                break
        frontier = next_frontier

    return trail


def _select_code_form(
    *, parent_code: str, child_code: str, is_valid: bool
) -> dict[str, Any]:
    """Pick the compact form (unified diff) or fall back to full code.

    The fallback rule covers three cases that would defeat the point of
    diffing:

    1. ``is_valid=False`` — keep full child source so ``error_summary`` line
       references read against the same buffer the analyst sees.
    2. Empty diff (identical sources) — there is nothing for the analyst to
       inspect in a diff; ship the full body once so clustering can still
       see "no-op" children.
    3. Diff no smaller than the child source — structural rewrites where
       every line differs; the diff is just parent+child catenated and the
       full body is strictly more readable.
    """
    if not is_valid:
        return {"change_form": "full_code", "code": child_code}
    diff_text = "".join(
        difflib.unified_diff(
            parent_code.splitlines(keepends=True),
            child_code.splitlines(keepends=True),
            fromfile="parent",
            tofile="child",
            n=_DIFF_CONTEXT_LINES,
            lineterm="",
        )
    )
    if not diff_text.strip() or len(diff_text) >= len(child_code):
        return {"change_form": "full_code", "code": child_code}
    return {"change_form": "diff", "diff": diff_text}


INTRA_SYSTEM_PROMPT_TEMPLATE = """\
TASK BEING SOLVED
{task_description}

AVAILABLE METRICS
{metrics_description}

YOUR ROLE
Lineage analyst for an LLM-guided evolutionary algorithm that mutates Python
programs to maximise fitness on the task above. You see ONE parent program and
EVERY evaluated child the algorithm produced from it so far. Distil that
experience into ONE compact, **purely descriptive** card: what was tried,
how each cluster fared (including HOW failures failed). A separate
downstream stage turns this card into actionable mutation suggestions —
your job is the history, not the prescription.

USER MESSAGE STRUCTURE (JSON payload)
| Field                              | Meaning |
|------------------------------------|---------|
| parent.code                        | Python source of the parent |
| parent.fitness                     | parent's primary metric value |
| children[i].change_form            | "diff" or "full_code"; discriminates which field below carries the child source |
| children[i].diff                   | unified diff of child vs parent.code (present iff change_form=="diff"); read this to see exactly what the child changed |
| children[i].code                   | full child Python source (present iff change_form=="full_code"; used when the diff would not be smaller than the file, or when the child is invalid so error_summary line refs remain readable) |
| children[i].delta                  | child.fitness − parent.fitness (+ better, − worse) |
| children[i].is_valid               | whether the child compiled AND ran cleanly |
| children[i].error_summary          | LLM-friendly stage-error text for FAILED children (empty for successes) |
| children[i].mutation_archetype     | mutator self-report (may be absent) |
| children[i].mutation_justification | mutator rationale (may be absent) |

When you cluster children by "what the code actually changed" (rule 1 below),
read `diff` for `change_form=="diff"` children — it already isolates the
edit — and compare against `parent.code` for `change_form=="full_code"`
children.

CARD CONSTRUCTION RULES
1. Cluster children by what their CODE actually changed (vs. the parent), not
   by their self-reported archetype. Each child belongs to EXACTLY ONE cluster.
   Sum of `tried_strategies[*].n_attempts` MUST equal top-level `n_attempts`.
2. For each cluster, set `verdict` ∈ {{"improved", "neutral", "regressed"}}
   using the delta bucketing below. If a cluster is failure-dominated, set
   verdict="regressed" AND fill `notes` with the dominant failure mode named
   concretely from `error_summary` (e.g. "IndexError when k=4 reached", "NaN
   from sqrt of negative", "timeout in inner loop"). For successful clusters,
   `notes` can be a brief mechanism note or left empty.
3. Delta bucketing: near-zero = within the noise floor of one evaluation;
   positive = measurably above; catastrophic = measurably below. EXCLUDE every
   `is_valid=false` child from `delta_distribution` (min / median / max /
   improving / neutral / catastrophic) AND from each cluster's `mean_delta`.
   Invalid programs carry a fitness sentinel (e.g. -1000) that would otherwise
   swamp the central tendency. Route them instead to `delta_distribution.n_failed`
   and, inside each cluster they belong to, to `tried_strategies[*].n_failed`.
   If a cluster has zero valid children, set its `mean_delta` to null and its
   `verdict` to "failed"; put the dominant failure mode in `notes`. Report
   `delta_distribution` directly from the input deltas; never invent values.
4. Labels: short (≤5 words), mutually exclusive, discriminating.
5. `summary`: one sentence, names the dominant outcome of this lineage so far
   (e.g. "Three threshold tweaks all regressed; loop-structure rewrites
   neutral; nothing has lifted fitness above parent yet").

OUTPUT
The schema (field names, types) is enforced by the system. Fill the fields;
no prose, no preamble, no fences. Do NOT include forward-looking hints or
suggestions — those are produced by a downstream stage.
"""


class IntraDeltaDistribution(BaseModel):
    """Aggregate stats over child fitness deltas for one parent.

    All distribution fields cover ONLY children with ``is_valid=true``. Invalid
    children are excluded from min/median/max/improving/neutral/catastrophic
    and counted on ``n_failed`` instead so the invalid-program sentinel (e.g.
    -1000 for heilbron) does not pollute the central tendency.
    """

    min: float | None = Field(
        description="Minimum delta across VALID children, or null if none."
    )
    median: float | None = Field(
        description="Median delta across VALID children, or null if none."
    )
    max: float | None = Field(
        description="Maximum delta across VALID children, or null if none."
    )
    improving: int = Field(
        description="Count of VALID children with delta measurably > 0."
    )
    neutral: int = Field(
        description=("Count of VALID children with delta in the noise floor of zero.")
    )
    catastrophic: int = Field(
        description="Count of VALID children with delta measurably << 0."
    )
    n_failed: int = Field(
        default=0,
        description=(
            "Count of INVALID children (is_valid=false) excluded from the "
            "delta stats above. Invalid programs carry a fitness sentinel "
            "(e.g. -1000) that would otherwise swamp median / mean — they are "
            "tracked separately here and surfaced as a failure summary."
        ),
    )


class IntraTriedStrategy(BaseModel):
    """One strategy already tried on this parent's lineage."""

    label: str = Field(
        description="Short (≤5 words), discriminating name for the strategy class."
    )
    n_attempts: int = Field(description="Children assigned to this strategy.")
    mean_delta: float | None = Field(
        default=None,
        description=(
            "Mean delta across the VALID children in this cluster. Null when "
            "the cluster has zero valid children (all attempts failed)."
        ),
    )
    verdict: str = Field(
        description=(
            "One of: improved | neutral | regressed | failed. Use 'failed' iff "
            "every attempt in this cluster was invalid (n_failed == n_attempts)."
        )
    )
    n_failed: int = Field(
        default=0,
        description=(
            "Invalid children inside this cluster, excluded from mean_delta. "
            "Counts toward n_attempts; must satisfy n_failed <= n_attempts."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "For regressed/failure-dominated strategies, the concrete failure "
            "mode named from children's error_summary (e.g. 'IndexError when "
            "k reached limit', 'timeout in inner loop'). For improved/neutral "
            "strategies, a brief mechanism note or empty."
        ),
    )


class IntraCardStructuredOutput(BaseModel):
    """Structured intra (per-parent) lineage card returned by the analyst LLM."""

    parent_id: str = Field(description="Parent program id (copied from input).")
    parent_fitness: float = Field(description="Parent's evaluated primary fitness.")
    n_attempts: int = Field(description="Total evaluated children for this parent.")
    delta_distribution: IntraDeltaDistribution = Field(
        description="Aggregate distribution of fitness deltas across children."
    )
    tried_strategies: list[IntraTriedStrategy] = Field(
        default_factory=list,
        description=(
            "Strategies already tried, clustered by code change. Sum of "
            "n_attempts MUST equal the top-level n_attempts."
        ),
    )
    summary: str = Field(
        description=(
            "One-line descriptive takeaway about this lineage's outcomes so "
            "far. Forward-looking suggestions are produced by a separate "
            "stage and MUST NOT appear here."
        )
    )


def _render_intra_card_text(
    card: dict[str, Any], metrics_context: MetricsContext | None = None
) -> str:
    """Render the JSON card into a compact markdown block for the mutator prompt.

    Numerical fields (parent_fitness, delta distribution, mean_delta) are
    formatted using the primary metric's ``decimals`` from metrics.yaml when a
    ``MetricsContext`` is provided; otherwise we fall back to ``repr()``
    (helpful for unit tests that don't bother wiring a context).
    """
    decimals: int | None = None
    if metrics_context is not None:
        primary = metrics_context.get_primary_key()
        decimals = metrics_context.specs[primary].decimals

    def _fmt(v: Any, signed: bool = False) -> str:
        if v is None:
            return "n/a"
        if decimals is None or not isinstance(v, (int, float)):
            return str(v)
        spec = f"+.{decimals}f" if signed else f".{decimals}f"
        return format(float(v), spec)

    lines: list[str] = ["## Intra Memory — Per-Parent Lineage Card", ""]

    parent_id = card.get("parent_id", "?")
    parent_fitness = card.get("parent_fitness")
    n_attempts = card.get("n_attempts", 0)
    lines.append(
        f"Parent `{parent_id[:8] if isinstance(parent_id, str) else parent_id}` "
        f"(fitness={_fmt(parent_fitness)}) has been mutated {n_attempts} time(s)."
    )

    dist = card.get("delta_distribution") or {}
    if dist:
        dist_line = (
            f"Delta distribution (valid children only): "
            f"min={_fmt(dist.get('min'), signed=True)}, "
            f"median={_fmt(dist.get('median'), signed=True)}, "
            f"max={_fmt(dist.get('max'), signed=True)}; "
            f"improving={dist.get('improving', 0)}, "
            f"neutral={dist.get('neutral', 0)}, "
            f"catastrophic={dist.get('catastrophic', 0)}"
        )
        n_failed = int(dist.get("n_failed", 0) or 0)
        if n_failed > 0:
            dist_line += f"; n_failed={n_failed} (excluded from stats above)"
        lines.append(dist_line)

    tried = card.get("tried_strategies") or []
    if tried:
        lines.append("")
        lines.append("**Already tried:**")
        for s in tried:
            n_attempts = s.get("n_attempts", 0)
            cluster_failed = int(s.get("n_failed", 0) or 0)
            attempt_part = f"{n_attempts} attempt(s)"
            if cluster_failed > 0:
                attempt_part += f" ({cluster_failed} failed)"
            mean_delta = s.get("mean_delta")
            mean_part = (
                "mean delta n/a"
                if mean_delta is None
                else f"mean delta {_fmt(mean_delta, signed=True)}"
            )
            line = (
                f"- *{s.get('label', '?')}* — {attempt_part}, "
                f"{mean_part}, verdict: {s.get('verdict', '?')}"
            )
            notes = (s.get("notes") or "").strip()
            if notes:
                line += f" — {notes}"
            lines.append(line)

    summary = card.get("summary")
    if summary:
        lines.append("")
        lines.append(f"_{summary}_")

    return "\n".join(lines)


class IntraMemoryInputs(StageIO):
    """Inputs to ``IntraMemoryStage``.

    ``children_ids`` is supplied by ``DescendantProgramIds`` upstream and lists
    the program's already-evaluated children to summarise. The card is purely
    descriptive — cross-population memory cards and ancestral-trail signals
    are consumed by the downstream ``MutationSuggestionStage`` instead.
    """

    children_ids: StringList | None = None


class _ConcatMemoryInputs(StageIO):
    """Optional inputs combined into a single memory block."""

    intra: StringContainer | None
    extra: StringContainer | None
    cards: StringContainer | None


@StageRegistry.register(
    description="Concat intra/extra/cards memory blocks into a single memory string"
)
class ConcatMemoryStage(Stage):
    """Joins intra/extra/cards memory blocks into a single ``memory`` string.

    Routes through the existing ``MutationContextStage.memory`` input so the
    prompt structure is unchanged — only the source of the memory block grows.
    Empty inputs are dropped; if everything is empty the output is an empty
    string (which MutationContextStage already skips).
    """

    InputsModel: type[StageIO] = _ConcatMemoryInputs
    OutputModel: type[StageIO] = StringContainer
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> StageIO:  # noqa: ARG002 — program unused
        params = cast(_ConcatMemoryInputs, self.params)
        blocks: list[str] = []
        for slot in (params.intra, params.extra, params.cards):
            if slot is not None and slot.data.strip():
                blocks.append(slot.data.strip())
        return StringContainer(data="\n\n".join(blocks))


@StageRegistry.register(description="Per-program lineage card (intra memory)")
class IntraMemoryStage(Stage):
    """Build a per-program lineage card from this program's evaluated children.

    DAG-native design. Runs on the CURRENT program X — the parent whose lineage
    the card summarises. ``children_ids`` is supplied upstream by
    ``DescendantProgramIds`` (an ``AncestrySelector`` over ``X.lineage.children``),
    and ``memory_cards`` (optional) by ``MemoryContextStage``. The framework
    cache (``InputHashCache`` — default) keys on the validated inputs, so the
    LLM call is skipped whenever the upstream selector returns the same id set
    and the memory_cards block is unchanged. When the engine's
    ``ParentRefresher`` flips X DONE→QUEUED after a new child finishes
    evaluating, the upstream selector now emits a new id list, the input hash
    changes, and this stage re-renders the card.

    Writes the rendered card to ``program.metadata['intra_memory_card']`` and
    emits a ``StringContainer`` so ``ConcatMemoryStage`` / ``MutationContextStage``
    can splice it into the next mutation prompt for X's future children.
    """

    InputsModel: type[StageIO] = IntraMemoryInputs
    OutputModel: type[StageIO] = StringContainer
    # Framework default (InputHashCache): re-run only when children_ids or
    # memory_cards change. No bespoke parent-side signature needed.

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        storage: ProgramStorage,
        metrics_context: MetricsContext,
        max_children: int = 32,
        task_description: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._structured_llm = llm.with_structured_output(IntraCardStructuredOutput)
        self._storage = storage
        self._metrics_context = metrics_context
        self._max_children = max_children
        metrics_desc = MetricsFormatter(metrics_context).format_metrics_description()
        self._system_prompt = INTRA_SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description.strip() or "(not provided)",
            metrics_description=metrics_desc,
        )

    def _collect_evaluated(
        self,
        children: list[Program],
        parent_fitness: float,
        parent_code: str,
    ) -> list[dict[str, Any]]:
        """Filter to evaluated children and serialise them for the LLM payload.

        Each child carries EITHER a unified diff against ``parent_code`` (the
        common case for typical small mutations) OR the full child source
        (for invalid children — full context for error_summary line refs —
        and for structural rewrites where the diff would be no smaller than
        the file itself). ``change_form`` discriminates the two.
        """
        primary_key = self._metrics_context.get_primary_key()
        evaluated: list[dict[str, Any]] = []
        for child in children:
            child_fitness = child.metrics.get(primary_key)
            if child_fitness is None:
                continue
            mutation_meta = child.metadata.get("mutation", {}) or {}
            archetype = None
            justification = None
            if isinstance(mutation_meta, dict):
                archetype = mutation_meta.get("archetype")
                justification = mutation_meta.get("justification")
            is_valid = self._metrics_context.is_valid(child.metrics)
            error_summary = ""
            if not is_valid:
                try:
                    error_summary = child.format_errors(include_traceback=True)
                except Exception:
                    error_summary = ""
            entry: dict[str, Any] = {
                "delta": child_fitness - parent_fitness,
                "is_valid": is_valid,
                "error_summary": error_summary,
                "mutation_archetype": archetype,
                "mutation_justification": justification,
            }
            code_form = _select_code_form(
                parent_code=parent_code,
                child_code=child.code,
                is_valid=is_valid,
            )
            entry.update(code_form)
            evaluated.append(entry)
        return evaluated

    async def compute(self, program: Program) -> StageIO:
        params = cast(IntraMemoryInputs, self.params)
        child_ids = list(params.children_ids.items) if params.children_ids else []
        if not child_ids:
            logger.debug(
                "[IntraMemoryStage] {} has no children to summarise; no-op",
                program.id[:8],
            )
            return StringContainer(data="")

        primary_key = self._metrics_context.get_primary_key()
        parent_fitness = program.metrics.get(primary_key)
        if parent_fitness is None:
            logger.debug(
                "[IntraMemoryStage] {} has no '{}' metric; no-op",
                program.id[:8],
                primary_key,
            )
            return StringContainer(data="")

        children = await self._storage.mget(child_ids[-self._max_children :])
        evaluated = self._collect_evaluated(children, parent_fitness, program.code)
        if not evaluated:
            logger.debug(
                "[IntraMemoryStage] {} has no evaluated children; no-op",
                program.id[:8],
            )
            return StringContainer(data="")

        payload: dict[str, Any] = {
            "parent": {
                "id": program.id,
                "fitness": parent_fitness,
                "code": program.code,
            },
            "children": evaluated,
        }
        user = (
            f"Produce the lineage card for the following parent and its "
            f"{len(evaluated)} evaluated children. Input JSON follows:\n\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

        try:
            card_obj = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=user),
                ]
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[IntraMemoryStage] LLM call failed for {}", program.id[:8]
            )
            return StringContainer(data="")

        if not isinstance(card_obj, IntraCardStructuredOutput):
            logger.warning(
                "[IntraMemoryStage] structured_llm returned {} (expected "
                "IntraCardStructuredOutput) for {}; dropping card",
                type(card_obj).__name__,
                program.id[:8],
            )
            return StringContainer(data="")

        card = card_obj.model_dump()
        rendered = _render_intra_card_text(card, self._metrics_context)
        program.set_metadata(INTRA_MEMORY_CARD_METADATA_KEY, rendered)
        try:
            await self._storage.update(program)
        except Exception:
            logger.opt(exception=True).warning(
                "[IntraMemoryStage] storage.update failed for {}", program.id[:8]
            )
        logger.info(
            "[IntraMemoryStage] Built intra card for {} ({} evaluated children)",
            program.id[:8],
            len(evaluated),
        )
        return StringContainer(data=rendered)
