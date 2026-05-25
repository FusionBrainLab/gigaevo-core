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
from typing import Any, Literal, cast

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
INTRA_MEMORY_SIGNAL_METADATA_KEY = "intra_memory_signal"
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
## ROLE

You are the **lineage analyst** in an LLM-guided evolutionary algorithm. You read ONE parent Python program plus EVERY child the algorithm has produced from it so far (with deltas, validity, error_summary). You emit ONE compact, **purely descriptive** clustering: which children share the same code-level move, and HOW failures failed.

You do NOT solve the task in CONTEXT below. You do NOT prescribe what to try next — a separate downstream stage does that. Your job is the HISTORY.

**You do NOT compute statistics.** Python computes `mean_delta`, `n_attempts`, `n_failed`, `verdict`, `delta_distribution`, and best/worst child deltas from your cluster membership. Your job is membership + qualitative labels + concrete anchors + failure mode.

## USER MESSAGE (JSON payload)

| Field | Meaning |
|---|---|
| parent.code | parent's Python source |
| parent.fitness | parent's primary metric value |
| children[i].index | stable integer index into the children array — USE THESE in `child_indices` |
| children[i].change_form | "diff" or "full_code"; discriminates which field below carries the child source |
| children[i].diff | unified diff of child vs parent.code (present iff change_form=="diff") |
| children[i].code | full child source (present iff change_form=="full_code"; used when diff isn't smaller, or when child is invalid) |
| children[i].delta | child.fitness − parent.fitness (+ better, − worse) — informational only; Python re-aggregates |
| children[i].is_valid | whether the child compiled AND ran cleanly |
| children[i].error_summary | LLM-friendly stage-error text for FAILED children (empty for successes) |
| children[i].mutation_archetype | mutator self-report (may be absent) |
| children[i].mutation_justification | mutator rationale (may be absent) |

When clustering by what the CODE actually changed, read `diff` for `change_form=="diff"` and compare `code` against `parent.code` for `change_form=="full_code"`.

## CARD CONSTRUCTION

1. **Cluster by what the CODE actually changed**, not by self-reported archetype. Each child's index MUST appear in EXACTLY ONE cluster's `child_indices`. Union of `child_indices` MUST cover every index 0..n-1 once.
2. **Labels** — short (≤5 words), mutually exclusive, discriminating (e.g. `threshold tighten`, `loop unroll`, `early termination`).
3. **representative_anchors** — for each cluster, 1–3 LITERAL strings copied verbatim from the cluster's diff/code: numeric constants that changed, identifiers added/removed, or distinctive expressions. These anchor the cluster so the downstream suggester can talk about specific code, not vague labels.
4. **failure_signature** — when ANY child in the cluster has `is_valid=false`, set this to a concrete failure-mode string named from `error_summary` (e.g. "IndexError when k=4 reached", "NaN from sqrt of negative", "timeout in inner loop"). Empty when all children in the cluster were valid.
5. **mechanism_note** — for clusters with at least one valid child, a brief sentence explaining what the cluster's code change does (mechanistically) and how it would move the primary metric. Empty when the entire cluster failed (use `failure_signature` instead).
6. **Summary** — one sentence naming the dominant outcome qualitatively (e.g. "Three threshold tweaks; loop-structure rewrites; one early-termination attempt; nothing structural has shifted yet").

## OUTPUT

Schema enforced. No prose, no preamble, no fences. Do NOT emit `mean_delta`, `verdict`, `n_attempts`, `n_failed`, or `delta_distribution` — Python computes these from `child_indices`.

## CONTEXT (the TASK the PROGRAM is solving — background for interpreting code changes)

{task_description}

Available metrics:
{metrics_description}
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
    """One strategy already tried on this parent's lineage.

    Post-merge shape: ``n_attempts``/``mean_delta``/``verdict``/``n_failed``/
    ``best_delta``/``worst_delta`` are Python-computed from the LLM-emitted
    ``child_indices`` (see ``IntraTriedStrategyLLM``) and the raw children
    deltas. ``label``/``notes``/``representative_anchors`` are the LLM's
    qualitative contributions.
    """

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
    best_delta: float | None = Field(
        default=None,
        description=(
            "Best (most-improving, sign-oriented) valid child delta in this "
            "cluster; null when no valid child exists."
        ),
    )
    worst_delta: float | None = Field(
        default=None,
        description=(
            "Worst (most-regressing) valid child delta in this cluster; null "
            "when no valid child exists. Distinct from invalid children, "
            "which are excluded and counted in n_failed."
        ),
    )
    representative_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Literal code snippets the LLM extracted from this cluster's "
            "diffs/code — concrete grounding for the downstream suggester."
        ),
    )


class IntraCardStructuredOutput(BaseModel):
    """Structured intra (per-parent) lineage card.

    Post-merge representation: ``tried_strategies`` carries Python-computed
    ``mean_delta``/``verdict``/``n_attempts``/``n_failed`` plus the LLM's
    qualitative ``label``/``notes``. The downstream renderer + signal derivation
    read this shape; only the LLM-side input schema (``IntraCardLLMOutput``)
    changed in the A2+F8 split.
    """

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


class IntraTriedStrategyLLM(BaseModel):
    """LLM-emitted descriptive cluster (membership + qualitative labels only).

    Python computes ``mean_delta``/``verdict``/``n_attempts``/``n_failed`` from
    ``child_indices`` and the raw children deltas — see ``_merge_intra_card``.
    """

    label: str = Field(
        description="Short (≤5 words), discriminating name for the strategy class."
    )
    child_indices: list[int] = Field(
        description=(
            "Indices into the children array (0..n-1) of every child belonging "
            "to this cluster. Disjoint across clusters; union covers every "
            "child exactly once."
        )
    )
    representative_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "1–3 literal strings copied verbatim from the cluster's diff/code "
            "(numeric constants, identifiers, distinctive expressions) so the "
            "downstream suggester can ground on concrete code."
        ),
    )
    mechanism_note: str = Field(
        default="",
        description=(
            "Brief mechanism description for clusters with at least one valid "
            "child. Empty when the cluster has zero valid children — use "
            "`failure_signature` instead."
        ),
    )
    failure_signature: str = Field(
        default="",
        description=(
            "Concrete failure mode named from children's `error_summary` when "
            "ANY child in the cluster was invalid (e.g. 'IndexError when k "
            "reached limit', 'timeout in inner loop'). Empty when every "
            "child in the cluster was valid."
        ),
    )


class IntraCardLLMOutput(BaseModel):
    """Structured payload the analyst LLM emits.

    Membership + qualitative labels + anchors only. All numeric aggregation
    happens in Python (``_merge_intra_card``).
    """

    tried_strategies: list[IntraTriedStrategyLLM] = Field(
        default_factory=list,
        description=(
            "Cluster membership of every evaluated child. Disjoint and "
            "covering: union of child_indices covers every index 0..n-1 once."
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
            best = s.get("best_delta")
            worst = s.get("worst_delta")
            if best is not None or worst is not None:
                line += (
                    f" (best {_fmt(best, signed=True)}, "
                    f"worst {_fmt(worst, signed=True)})"
                )
            anchors = [a for a in (s.get("representative_anchors") or []) if a]
            if anchors:
                line += f" — anchors: {', '.join(f'`{a}`' for a in anchors)}"
            notes = (s.get("notes") or "").strip()
            if notes:
                line += f" — {notes}"
            lines.append(line)

    summary = card.get("summary")
    if summary:
        lines.append("")
        lines.append(f"_{summary}_")

    return "\n".join(lines)


_NEGATIVE_VERDICTS = frozenset({"regressed", "failed"})

# Delta bucket thresholds (in primary-metric units, oriented so positive =
# improvement). Tighter than display precision so weak signals are not
# rounded into "neutral". The current pipelines work in fitness ≈ 0.02–0.05
# bands; 1e-4 captures noise vs signal at that scale without becoming a
# behavioural knob.
_DELTA_NOISE_FLOOR = 1e-4


def _bucket_delta(delta: float) -> str:
    if delta > _DELTA_NOISE_FLOOR:
        return "improving"
    if delta < -_DELTA_NOISE_FLOOR:
        return "catastrophic"
    return "neutral"


def _verdict_from_buckets(buckets: dict[str, int], n_valid: int) -> str:
    """Map per-cluster bucket counts → categorical verdict.

    Pure aggregation: 'failed' iff every child was invalid; otherwise the
    plurality of improving/neutral/catastrophic with improving and
    catastrophic tie-broken in favour of the worse direction so latent
    regression isn't masked by a single improvement.
    """
    if n_valid == 0:
        return "failed"
    improving = buckets.get("improving", 0)
    neutral = buckets.get("neutral", 0)
    catastrophic = buckets.get("catastrophic", 0)
    if improving > catastrophic and improving > neutral:
        return "improved"
    if catastrophic >= improving and catastrophic > 0:
        return "regressed"
    return "neutral"


def _merge_intra_card(
    llm_output: IntraCardLLMOutput,
    evaluated: list[dict[str, Any]],
    parent_id: str,
    parent_fitness: float,
) -> IntraCardStructuredOutput:
    """Fold LLM-emitted cluster membership + raw child deltas → numeric card.

    Python owns every aggregation (counts, mean_delta, verdict bucketing,
    delta distribution, best/worst per cluster). The LLM contributes
    membership + qualitative labels + anchors + mechanism/failure notes only.

    Membership validation: indices outside ``0..n-1`` are dropped with a
    warning; missing indices land in an ``unassigned`` catch-all cluster so
    aggregate counts always reconcile with the input. Duplicates across
    clusters are awarded to the first cluster that claimed them.
    """
    n_children = len(evaluated)
    assigned: set[int] = set()
    clusters: list[IntraTriedStrategy] = []

    delta_dist_buckets = {"improving": 0, "neutral": 0, "catastrophic": 0}
    valid_deltas: list[float] = []
    n_failed_total = 0

    for cluster_llm in llm_output.tried_strategies:
        idxs: list[int] = []
        for raw_idx in cluster_llm.child_indices:
            try:
                i = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if i < 0 or i >= n_children or i in assigned:
                continue
            idxs.append(i)
            assigned.add(i)

        cluster = _build_cluster(
            label=cluster_llm.label,
            indices=idxs,
            evaluated=evaluated,
            notes=cluster_llm.failure_signature or cluster_llm.mechanism_note,
            anchors=cluster_llm.representative_anchors,
        )
        clusters.append(cluster)

    unassigned = [i for i in range(n_children) if i not in assigned]
    if unassigned:
        clusters.append(
            _build_cluster(
                label="unassigned",
                indices=unassigned,
                evaluated=evaluated,
                notes=(
                    "Children whose cluster the LLM did not assign — Python "
                    "fallback bucket to keep aggregate counts consistent."
                ),
                anchors=[],
            )
        )

    for entry in evaluated:
        delta = float(entry["delta"])
        if entry.get("is_valid"):
            valid_deltas.append(delta)
            delta_dist_buckets[_bucket_delta(delta)] += 1
        else:
            n_failed_total += 1

    distribution = IntraDeltaDistribution(
        min=min(valid_deltas) if valid_deltas else None,
        median=(sorted(valid_deltas)[len(valid_deltas) // 2] if valid_deltas else None),
        max=max(valid_deltas) if valid_deltas else None,
        improving=delta_dist_buckets["improving"],
        neutral=delta_dist_buckets["neutral"],
        catastrophic=delta_dist_buckets["catastrophic"],
        n_failed=n_failed_total,
    )

    return IntraCardStructuredOutput(
        parent_id=parent_id,
        parent_fitness=parent_fitness,
        n_attempts=n_children,
        delta_distribution=distribution,
        tried_strategies=clusters,
        summary=llm_output.summary,
    )


def _build_cluster(
    *,
    label: str,
    indices: list[int],
    evaluated: list[dict[str, Any]],
    notes: str,
    anchors: list[str],
) -> IntraTriedStrategy:
    valid_deltas: list[float] = []
    buckets = {"improving": 0, "neutral": 0, "catastrophic": 0}
    n_failed = 0
    for i in indices:
        entry = evaluated[i]
        delta = float(entry["delta"])
        if entry.get("is_valid"):
            valid_deltas.append(delta)
            buckets[_bucket_delta(delta)] += 1
        else:
            n_failed += 1
    n_attempts = len(indices)
    mean_delta = sum(valid_deltas) / len(valid_deltas) if valid_deltas else None
    verdict = _verdict_from_buckets(buckets, n_valid=len(valid_deltas))
    return IntraTriedStrategy(
        label=label,
        n_attempts=n_attempts,
        mean_delta=mean_delta,
        verdict=verdict,
        n_failed=n_failed,
        notes=(notes or "").strip(),
        best_delta=max(valid_deltas) if valid_deltas else None,
        worst_delta=min(valid_deltas) if valid_deltas else None,
        representative_anchors=[a for a in (anchors or []) if a and a.strip()],
    )


class IntraStrategyEntry(BaseModel):
    """Compact view of one tried cluster — the only fields the downstream
    suggester needs to weight a cluster (full notes live in the card text)."""

    label: str
    verdict: str
    n_attempts: int


class IntraMemorySignal(BaseModel):
    """Structured plateau signal derived from the intra card.

    Severity tiers, all derived from the existing IntraCardStructuredOutput:

    * ``healthy``   — no clusters with verdict in {regressed, failed}
    * ``negative``  — >=1 negative cluster, neither cond_a nor cond_b fires
    * ``exhausted`` — cond_a (>=2 negative clusters) OR cond_b
                      (improving == 0 AND catastrophic + n_failed >= 2)

    Always emitted when an intra card exists, so the downstream prompt can
    grade guidance by severity instead of toggling on a binary predicate.
    """

    severity: Literal["healthy", "negative", "exhausted"]
    n_clusters: int
    n_negative: int
    clusters: list[IntraStrategyEntry]
    delta_dist: dict[str, Any]


def _derive_intra_signal(card: dict[str, Any]) -> IntraMemorySignal:
    """Compute the three-tier plateau signal from an IntraCardStructuredOutput dict.

    Replaces ``MutationSuggestionAgent._format_exhaustion_block``'s
    regex-on-rendered-markdown predicate with a single source of truth that
    reads the structured Pydantic fields the LLM already produced.
    """
    tried = card.get("tried_strategies") or []
    dist = card.get("delta_distribution") or {}

    clusters = [
        IntraStrategyEntry(
            label=str(s.get("label", "?")),
            verdict=str(s.get("verdict", "?")),
            n_attempts=int(s.get("n_attempts", 0) or 0),
        )
        for s in tried
    ]
    n_clusters = len(clusters)
    n_negative = sum(1 for c in clusters if c.verdict in _NEGATIVE_VERDICTS)

    improving = int(dist.get("improving", 0) or 0)
    neutral = int(dist.get("neutral", 0) or 0)
    catastrophic = int(dist.get("catastrophic", 0) or 0)
    n_failed = int(dist.get("n_failed", 0) or 0)

    cond_a = n_negative >= 2
    cond_b = improving == 0 and (catastrophic + n_failed) >= 2
    if cond_a or cond_b:
        severity: Literal["healthy", "negative", "exhausted"] = "exhausted"
    elif n_negative >= 1:
        severity = "negative"
    else:
        severity = "healthy"

    return IntraMemorySignal(
        severity=severity,
        n_clusters=n_clusters,
        n_negative=n_negative,
        clusters=clusters,
        delta_dist={
            "min": dist.get("min"),
            "median": dist.get("median"),
            "max": dist.get("max"),
            "improving": improving,
            "neutral": neutral,
            "catastrophic": catastrophic,
            "n_failed": n_failed,
        },
    )


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


_SECTION_HEADERS = {
    "intra": "## PARENT LINEAGE CARD",
    "extra": "## EXTRA EVIDENCE",
    "cards": "## CROSS-POP MEMORY CARDS",
}


@StageRegistry.register(
    description="Concat intra/extra/cards memory blocks into a single memory string"
)
class ConcatMemoryStage(Stage):
    """Joins intra/extra/cards memory blocks into a single ``memory`` string.

    Each present slot is rendered under an explicit provenance header so the
    downstream mutator can tell parent-lineage evidence from cross-population
    memory cards (preserves the distinction the v2 architecture review flagged
    as lost by a raw ``\\n\\n`` join). Empty inputs are dropped; if everything
    is empty the output is an empty string (which MutationContextStage already
    skips).
    """

    InputsModel: type[StageIO] = _ConcatMemoryInputs
    OutputModel: type[StageIO] = StringContainer
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> StageIO:  # noqa: ARG002 — program unused
        params = cast(_ConcatMemoryInputs, self.params)
        sections: list[str] = []
        for slot_name, slot in (
            ("intra", params.intra),
            ("extra", params.extra),
            ("cards", params.cards),
        ):
            if slot is not None and slot.data.strip():
                header = _SECTION_HEADERS[slot_name]
                sections.append(f"{header}\n\n{slot.data.strip()}")
        return StringContainer(data="\n\n".join(sections))


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
        self._structured_llm = llm.with_structured_output(IntraCardLLMOutput)
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
                "index": len(evaluated),
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
                "[Memory][IntraStage] {} has no children to summarise; no-op",
                program.id[:8],
            )
            return StringContainer(data="")

        primary_key = self._metrics_context.get_primary_key()
        parent_fitness = program.metrics.get(primary_key)
        if parent_fitness is None:
            logger.debug(
                "[Memory][IntraStage] {} has no '{}' metric; no-op",
                program.id[:8],
                primary_key,
            )
            return StringContainer(data="")

        children = await self._storage.mget(child_ids[-self._max_children :])
        evaluated = self._collect_evaluated(children, parent_fitness, program.code)
        if not evaluated:
            logger.debug(
                "[Memory][IntraStage] {} has no evaluated children; no-op",
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
            llm_output = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=user),
                ]
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][IntraStage] LLM call failed for {}", program.id[:8]
            )
            return StringContainer(data="")

        if not isinstance(llm_output, IntraCardLLMOutput):
            logger.warning(
                "[Memory][IntraStage] structured_llm returned {} (expected "
                "IntraCardLLMOutput) for {}; dropping card",
                type(llm_output).__name__,
                program.id[:8],
            )
            return StringContainer(data="")

        merged = _merge_intra_card(
            llm_output=llm_output,
            evaluated=evaluated,
            parent_id=program.id,
            parent_fitness=parent_fitness,
        )
        card = merged.model_dump()
        rendered = _render_intra_card_text(card, self._metrics_context)
        signal = _derive_intra_signal(card)
        program.set_metadata(INTRA_MEMORY_CARD_METADATA_KEY, rendered)
        program.set_metadata(INTRA_MEMORY_SIGNAL_METADATA_KEY, signal.model_dump())
        try:
            await self._storage.update(program)
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][IntraStage] storage.update failed for {}", program.id[:8]
            )
        logger.info(
            "[Memory][IntraStage] Built intra card for {} ({} evaluated children, "
            "signal severity={})",
            program.id[:8],
            len(evaluated),
            signal.severity,
        )
        return StringContainer(data=rendered)
