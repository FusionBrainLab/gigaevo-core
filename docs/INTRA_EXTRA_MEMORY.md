# Intra/Extra Memory Pipeline (`pipeline=intra_extra_memory`)

> Live, dual-track LLM memory for MAP-Elites: a **per-parent lineage card** (intra) and a **live global card bank** (extra), both refreshed mid-run. The intra card reaches the mutator verbatim; the global cards reach it ONLY as suggester output — `MutationSuggestionStage` is the single source of hints.

This mode replaces the legacy lineage / insights stages with two strong-LLM analyst stages: a **descriptive** `IntraMemoryStage` (per-parent lineage card) and a **prescriptive** `MutationSuggestionStage` that digests the intra card, the cross-population memory cards, the ancestral-momentum trail, and run statistics into structured `ProgramInsights` for the mutation prompt.

- **Pipeline config:** [`config/pipeline/intra_extra_memory.yaml`](../config/pipeline/intra_extra_memory.yaml)
- **Builder:** [`gigaevo/entrypoint/lineage_memory_pipeline.py`](../gigaevo/entrypoint/lineage_memory_pipeline.py)
- **Stages:** [`gigaevo/programs/stages/lineage_memory.py`](../gigaevo/programs/stages/lineage_memory.py)
- **Live refresh hook:** [`gigaevo/memory/live_memory_hook.py`](../gigaevo/memory/live_memory_hook.py)
- **Related:** [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md), [DAG_SYSTEM.md](DAG_SYSTEM.md), [memory.md](memory.md)

---

## 1. Why this mode exists

Default GigaEvo passes the mutator a flat list of "insights" derived from a parent's ancestors and descendants. In practice the mutator either drowns in noise or re-tries strategies that have already been logged as regressed. Two known failure modes:

1. **No deduplication of strategies across siblings.** The same idea (e.g. "increase search radius") is re-tried because each sibling's diff looks novel under raw text inspection.
2. **No global cross-pollination.** Successful patterns discovered in one lineage stay confined to that lineage; nothing pulls them into a parent's prompt unless the operator manually wires it.

`intra_extra_memory` addresses both with **one stage and one hook**:

| Track | Scope | Stage | Trigger |
|-------|-------|-------|---------|
| **Intra** (per-parent lineage card) | One parent's evaluated children | `IntraMemoryStage` | Cache-invalidated when a new child completes |
| **Extra** (live global ideas) | All evaluated programs across runs | `LiveMemoryRefreshHook` → `IdeaTracker.run_increment` | Every `refresh_every` ingestor sweeps that landed ≥ 1 program (default `10`) |

The mutator sees the intra card verbatim (`MutationContextStage.memory`) plus the suggester's `ProgramInsights` (`MutationContextStage.insights`). The global cards are consumed only by the suggester — the mutator never sees card text.

---

## 2. Pipeline architecture

The builder (`IntraExtraMemoryPipelineBuilder`, subclassing the intra-only `IntraMemoryPipelineBuilder`) inherits from `DefaultPipelineBuilder` and **strips five legacy stages**:

```
                            ┌───────────────────────┐
                            │  EnsureMetricsStage   │
                            └──────────┬────────────┘
                                       │ (exec dep)
                                       ▼
 ┌────────────────────────┐
 │ DescendantProgramIds   │
 │ (max_selected=24,      │
 │  strategy=recent)      │
 └──────────┬─────────────┘
            │ "children_ids"
            ▼
 ┌────────────────────────────────────┐
 │   IntraMemoryStage (strong LLM)    │  DESCRIPTIVE: what was tried, how it fared
 │   → IntraCardStructuredOutput      │
 └───────┬───────────────────┬────────┘
         │ "intra_card"      │ "memory"  (card text, verbatim)
         ▼                   │
 ┌────────────────────────┐  │   ┌──────────────────────┐
 │ MutationSuggestionStage│◀─┼───┤  MemoryContextStage  │ ← live global cards,
 │ (strong LLM,           │  │   │  "memory_cards"      │   numbered [card N] id=…,
 │  PRESCRIPTIVE)         │  │   └──────────────────────┘   refreshed by hook
 └──────────┬─────────────┘  │
            │ "insights"     │
            ▼                ▼
 ┌─────────────────────────────────┐
 │      MutationContextStage       │  memory = intra card; insights = ProgramInsights
 └─────────────────────────────────┘
```

`MutationSuggestionStage` also consumes `EvolutionaryStatisticsCollector` output (`evolutionary_statistics`) and walks the ancestral-momentum trail from storage internally. **The `MemoryContextStage → MutationSuggestionStage` edge is the ONLY consumer of cross-population cards** — a card mechanism reaches the mutator only if the suggester transposes it into a suggestion (`mechanism_source: memory_cards`, `card_id` set).

**Stages stripped** (superseded by `IntraMemoryStage` + `MutationSuggestionStage`):

- `AncestorProgramIds`, `LineageStage`, `LineagesToDescendants`, `LineagesFromAncestors`, `InsightsStage`

**Stages kept and reconfigured:**

- `DescendantProgramIds` — widened from `max_selected=1` (the default builder's LineageStage-tuned setting) to `intra_max_children=24` with `strategy="recent"`: a chronological recency window, so failed and regressed children stay visible to the analyst (a fitness-ranked selection would silently drop exactly the failures the card exists to report).

---

## 3. The intra card

`IntraMemoryStage` emits a Pydantic-validated structured output and renders it to Markdown for the mutation prompt. The structured schema lives in `gigaevo/programs/stages/lineage_memory.py`:

```python
IntraCardStructuredOutput
├── parent_id: str
├── parent_fitness: float
├── n_attempts: int                 — children in the recency window, not lifetime total
├── delta_distribution: IntraDeltaDistribution
│   ├── min / median / max         (float | None)  — VALID children only
│   ├── improving / neutral / catastrophic         — VALID children only
│   └── n_failed                                    — INVALID children, tracked separately
├── tried_strategies: list[IntraTriedStrategy]
│   ├── label, n_attempts, mean_delta (float | None), verdict, n_failed, notes
└── summary: str
```

All deltas are **oriented**: positive always means the child improved on the parent, for maximize and minimize metrics alike (same convention as the ancestral trail's `step_delta`). Multi-parent (crossover) children additionally carry `crossover_role` in the analyst payload — `"base"` when this parent is the base the mutator transformed, `"donor"` when this parent only contributed mechanisms (a donor-side diff mostly reflects the other parent's code, and the prompt tells the analyst not to cluster it as a mutation move).

**Invalid-child handling (heilbron `-1000` sentinel, etc.):** any child with `is_valid=false` is **excluded** from every distribution field and from per-cluster `mean_delta`, then **counted separately** in `n_failed`. The rendered card surfaces failures as a parenthesised count, e.g.:

```
- *naive_loop* — 5 attempt(s) (2 failed), mean delta 0.018, verdict: improved
- *radius_blowup* — 3 attempt(s) (3 failed), mean delta n/a, verdict: failed
Delta distribution (valid children only; + = improvement): min=0.01, median=0.015, max=0.02;
  improving=2, neutral=0, catastrophic=0; n_failed=2 (excluded from stats above)
```

The system prompt explicitly instructs the LLM to follow this contract (rule 3 in `INTRA_SYSTEM_PROMPT_TEMPLATE`).

---

## 4. The live external memory

The `extra` half of the pipeline name is provided by `LiveMemoryRefreshHook`, wired as the engine's `post_step_hook`:

```yaml
post_step_hook:
  _target_: gigaevo.memory.live_memory_hook.LiveMemoryRefreshHook
  tracker: ${ref:memory::tracker}
  storage: ${ref:program_storage}
  refresh_every: 10
```

It wraps `IdeaTracker.run_increment(...)`, so the **mid-run hook and the existing end-of-run `post_run_hook` share state** via the tracker's `_run_lock`. After each refresh:

- New cards land in the local card store.
- `MemoryContextStage`'s reload-on-read selector picks them up on the next stage invocation.
- The framework's `InputHashCache` sees the cards block change and invalidates downstream stages (including `IntraMemoryStage` for any parent whose lineage card hadn't already been invalidated by a new child).

`refresh_every: 10` ≈ one refresh per 10 newly-evaluated mutants, which on heilbron's smoke (~45 programs) gave 4 mid-run refreshes plus the end-of-run pass.

---

## 5. Caching contract

Both new stages are pure cache-aware nodes — the LLM only runs when an input actually changed:

| Stage | Input that invalidates cache |
|-------|------------------------------|
| `IntraMemoryStage` | `children_ids` (`DescendantProgramIds` output) |
| `MutationSuggestionStage` | `intra_card`, `memory_cards` (`MemoryContextStage` block), or `evolutionary_statistics` |

See [`tests/stages/test_intra_memory_cache.py`](../tests/stages/test_intra_memory_cache.py) for the contract.

**Cache-miss triggers in practice:**

1. A new child of parent X finishes evaluating → `ParentRefresher` flips X `DONE → QUEUED` → `DescendantProgramIds` returns a new id list → intra invalidates for X (and the new intra card invalidates the suggester).
2. `LiveMemoryRefreshHook` writes new global cards → `MemoryContextStage` block changes → the suggester invalidates for **all** parents on their next visit (the intra card itself does not).

---

## 6. Required co-overrides

The mode depends on one upstream config node that Hydra's defaults-list cannot safely flip from inside `pipeline/`, so it must be passed on the CLI:

```
memory=full              # one preset, both sides on: LiveMemoryRefreshHook
                         # calls IdeaTracker.run_increment (writer) and
                         # MemoryReadPipeline reads the card store it writes
                         # to between refreshes (reader)
```

Under `pipeline=intra_extra_memory` the writer-off presets (`memory=none`,
`memory=reader`) fail fast at startup — the live-refresh hook needs a real
tracker. A true no-memory baseline is `pipeline=standard memory=none`.

---

## 7. Launching an experiment

### Smoke (matches the 2026-05-15 acceptance run — 40 mutants, ~50 min)

```bash
cd ~/gigaevo
HTTPS_PROXY=http://mathemage:jky5exmw@64.225.96.36:8888 \
NO_PROXY="localhost,127.0.0.1,INTERNAL_IP" \
OPENAI_API_KEY=sk-gigaevo \
python3 run.py \
  problem.name=heilbron \
  llm_base_url=http://INTERNAL_IP:4000 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  redis.db=10 \
  num_parents=1 \
  pipeline=intra_extra_memory \
  memory=full \
  max_mutants=40 \
  hydra.run.dir=output/smoke_intra_extra/$(date +%Y%m%d_%H%M%S)_smoke
```

### Full experiment (longer horizon, wider parallelism)

```bash
cd ~/gigaevo
HTTPS_PROXY=http://mathemage:jky5exmw@64.225.96.36:8888 \
NO_PROXY="localhost,127.0.0.1,INTERNAL_IP" \
OPENAI_API_KEY=sk-gigaevo \
python3 run.py \
  problem.name=heilbron \
  llm_base_url=http://INTERNAL_IP:4000 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  redis.db=11 \
  num_parents=4 \
  pipeline=intra_extra_memory \
  memory=full \
  max_mutants=500 \
  hydra.run.dir=output/intra_extra_memory/$(date +%Y%m%d_%H%M%S)_heilbron
```

**What changes for the full run vs. the smoke:**

| Knob | Smoke | Full | Why |
|------|------:|-----:|-----|
| `max_mutants` | 40 | 500 | Smoke only needs to prove the wiring; full needs convergence. |
| `num_parents` | 1 | 4 | More parents per iteration → more concurrent intra-card slots and broader exploration. |
| `redis.db` | 10 | 11 | Avoid clashing with the smoke's persisted state. |
| `hydra.run.dir` | `smoke_intra_extra/...` | `intra_extra_memory/...` | Separate report aggregation. |

### Background launch with Telegram completion notify

```bash
cd ~/gigaevo
HTTPS_PROXY=http://mathemage:jky5exmw@64.225.96.36:8888 \
NO_PROXY="localhost,127.0.0.1,INTERNAL_IP" \
OPENAI_API_KEY=sk-gigaevo \
nohup python3 run.py \
  problem.name=heilbron \
  llm_base_url=http://INTERNAL_IP:4000 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  redis.db=11 num_parents=4 \
  pipeline=intra_extra_memory memory=full \
  max_mutants=500 \
  hydra.run.dir=output/intra_extra_memory/$(date +%Y%m%d_%H%M%S)_heilbron \
  > /tmp/intra_extra_run.log 2>&1 &
echo "PID=$!"
```

Pair with `tools/telegram_notify.notify(...)` from your monitoring shell for milestone pings.

---

## 8. Tuning knobs

| Setting | Where | Default | Effect |
|---------|-------|--------:|--------|
| `intra_max_children` | `pipeline_builder` block | `24` | Cap on the descendant pool the analyst sees. Lower = cheaper but shallower context. |
| `refresh_every` | `post_step_hook` block | `10` | Ingestor sweeps between live refreshes. Lower = fresher cards, more LLM calls. |
| `post_step_hook_timeout_s` | top-level (recipe) | `900` (`300` global) | Wall-clock budget (s) for one live-refresh increment. Raised above the CPU-hook-sized 300 s global default so LLM enrichment under shared-endpoint load completes instead of being cancelled (frozen card bank). Bounds a hung hook. Override per run: `post_step_hook_timeout_s=<s>`. |
| `max_insights` | top-level | inherited | Bound on memory-card count `MemoryContextStage` surfaces. |
| `max_code_length` | top-level | inherited | Truncation guard for parent code in the intra prompt. |
| `stage_timeout` | top-level | inherited | Per-stage timeout; respected by `IntraMemoryStage` and `MutationSuggestionStage`. |

To override at launch:

```bash
... pipeline=intra_extra_memory \
    pipeline_builder.intra_max_children=12 \
    post_step_hook.refresh_every=5
```

---

## 9. Verifying the wiring is live

After a run finishes, four artefacts should all be non-empty:

1. **Intra cards on parent metadata** — for any parent X visited by the mutator:
   ```python
   from gigaevo.programs.program import Program
   ...
   p = program_storage.get(parent_id)
   assert p.metadata.get("intra_memory_card", "").startswith("# Intra Memory")
   ```
2. **Chroma embedding count** — should grow from 0 → ~5 × card-count by end of run.
3. **Mutation context** — at least one parent's `metadata["mutation_context"]["memory"]` contains the `## Intra Memory` lineage card, and `["insights"]` carries the suggester's `ProgramInsights` (cards reach the mutator only through these).
4. **Captured prompts** — `MutationSuggestionStage` request payloads contain a `## Memory Cards` block with `[card N] id=<card-id>` headers; `MutationAgent` payloads contain the intra card and numbered `## Program Insights`, and NO card text.

The 2026-05-15 smoke (40 mutants, heilbron, Qwen3-235B-A22B) — run on the pre-split wiring — produced a top fitness of `0.02487` (≈15× the best seed) with **0 `IntraMemoryStage` failures** over 78 invocations.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `MemoryContextStage` block always empty | `memory=full` not set | Add it to the CLI. |
| Intra card never appears on parents | The mutator never visited any parent twice (run too short) | `max_mutants >= 2 * num_parents`. |
| `IntraMemoryStage` runs every iteration (no cache hits) | Different `children_ids` order on each visit | Confirm `DescendantProgramIds` uses `strategy="recent"` (chronological, stable across visits). |
| Rendered card shows `min=-1000.0` etc. | Pre-fix build (`< 89f01be5`) | Pull main, rebuild. The current schema excludes invalid children from delta stats and routes them to `n_failed`. |
| `LiveMemoryRefreshHook` never fires | Ingestor never landed ≥ 1 program in `refresh_every` sweeps | Lower `refresh_every`, or check ingestor health. |

---

## 11. See also

- [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) — the global memory subsystem this mode plugs into
- [DAG_SYSTEM.md](DAG_SYSTEM.md) — `InputHashCache`, `ExecutionOrderDependency`, `add_data_flow_edge` semantics
- [memory.md](memory.md) — broader card / idea taxonomy
- [`config/pipeline/intra_extra_memory.yaml`](../config/pipeline/intra_extra_memory.yaml) — the canonical config

---

*Last updated: 2026-06-09 (single-source hint wiring: cards → suggester only). Pipeline introduced in commit `89f01be5`.*
