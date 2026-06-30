# Memory System: Complete Guide

This document explains GigaEvo's memory-augmented mutation system end-to-end.
Memory lets the evolutionary algorithm learn from past experiments by feeding
"ideas" (memory cards) into the mutation prompt.

---

## Table of Contents

1. [The 30-Second Version](#the-30-second-version)
2. [What Memory Does](#what-memory-does)
3. [The Two Phases: Writing and Reading](#the-two-phases-writing-and-reading)
4. [How Memory Flows Through the Pipeline](#how-memory-flows-through-the-pipeline)
5. [Architecture: The Provider Pattern](#architecture-the-provider-pattern)
6. [Configuration Reference](#configuration-reference)
   - [Preset selection (memory=...)](#preset-selection-memoryname)
   - [Component groups](#component-groups)
   - [Backend builder](#backend-builder)
   - [The memory LLM (config/memory/common/llm/)](#the-memory-llm-configmemorycommonllm)
7. [The Ideas Tracker (Write Phase)](#the-ideas-tracker-write-phase)
   - [What It Does](#what-it-does)
   - [Entry Point: PostRunHook](#entry-point-postrunhook)
   - [The Writer Side of memory= (config/memory/writer/tracker/)](#the-writer-side-of-memory-configmemorywritertracker)
   - [Pipeline Internals](#pipeline-internals)
   - [Card Bank and Backend](#card-bank-and-backend)
   - [What a Memory Card Looks Like](#what-a-memory-card-looks-like)
   - [Logs and Checkpoints](#logs-and-checkpoints)
8. [The Memory Search (Read Phase)](#the-memory-search-read-phase)
9. [Tracking: How to Know if Memory Was Used](#tracking-how-to-know-if-memory-was-used)
10. [Full Experiment Workflow](#full-experiment-workflow)
    - [Phase A: Build the Memory Bank](#phase-a-build-the-memory-bank)
    - [Phase B: Controlled Experiment](#phase-b-controlled-experiment)
    - [Analysis](#analysis)
11. [Key Files](#key-files)
12. [FAQ](#faq)

---

## The 30-Second Version

```bash
python run.py memory=none   ...  # No memory (default): reader off, writer off
python run.py memory=reader ...  # Inject cards from an existing bank; no extraction
python run.py memory=writer ...  # Extract/enrich cards for a LATER run; inject nothing
python run.py memory=full   ...  # Reader + writer share ONE card bank
```

One Hydra override assembles the whole `MemorySystem`. Everything else is automatic.

---

## What Memory Does

Without memory, the LLM mutation agent sees:
- The parent program code
- Metrics (fitness scores)
- Insights (what changed in recent mutations)
- Lineage (ancestor/descendant analysis)

With memory, the pipeline ALSO surfaces **memory cards** — short, actionable
mechanism levers extracted from previous experiments:

```
[card 1] id=idea-abc123
Sort evidence by relevance score before chain traversal
efficacy: introduced in 9 children; median improvement +0.012 (confident)

[card 2] id=idea-def456
Filter low-confidence hops using a learned threshold
```

In the LEGACY default pipeline this card text lands in the mutator prompt
verbatim (numbered `[card N] id=…` blocks, no wrapper header). In the current
experiment pipelines
(`pipeline=standard` / `intra_extra_memory`) the cards go to the
**mutation-suggestion analyst** instead, which transposes them into structured
suggestions — the mutator never sees raw card text.

These ideas come from a **memory database** that accumulates knowledge across
evolution runs. The hypothesis: if you tell the LLM "here are techniques that
worked before", it produces better mutations than starting from scratch.

---

## The Two Phases: Writing and Reading

The memory system has two completely separate phases:

```
╔═══════════════════════════════════════════════════════════════════╗
║                      WRITE PHASE                                  ║
║                                                                   ║
║  Evolution Run A (no memory) ──> produces top programs            ║
║                                       │                           ║
║                                       ▼                           ║
║                              Ideas Tracker (PostRunHook)          ║
║                              extracts generalizable ideas         ║
║                                       │                           ║
║                                       ▼                           ║
║                              Memory Database (disk or API)        ║
╠═══════════════════════════════════════════════════════════════════╣
║                      READ PHASE                                   ║
║                                                                   ║
║  Evolution Run B (memory=reader) ──> DAG pipeline                 ║
║                                       │                           ║
║                                       ▼                           ║
║                              MemoryContextStage                   ║
║                              queries memory database              ║
║                              returns top-N relevant cards         ║
║                                       │                           ║
║                                       ▼                           ║
║                              LLM sees cards in mutation prompt    ║
╚═══════════════════════════════════════════════════════════════════╝
```

**Write phase** = Ideas Tracker extracts knowledge from completed runs.
**Read phase** = Evolution reads that knowledge during mutation.

In the two-pass build-then-read flow the phases are separate: the ideas tracker
runs AFTER an evolution completes, and the next evolution reads from the
database. Under `pipeline=intra_extra_memory` they interleave within one run —
the `LiveMemoryRefreshHook` writes mid-run while mutation reads — sharing one
card bank under the tracker's write lock.

---

## How Memory Flows Through the Pipeline

> **Pipeline-dependent routing.** The flow below shows the LEGACY default
> pipeline, where `MemoryContextStage` output feeds the mutator directly via
> `MutationContextStage.memory`. Under `pipeline=standard` /
> `pipeline=intra_extra_memory` (the current experiment pipelines) the cards
> instead feed ONLY `MutationSuggestionStage.memory_cards` — the suggester
> digests them into structured `ProgramInsights` and is the mutator's single
> source of hints; `MutationContextStage.memory` carries the per-parent intra
> lineage card. See [INTRA_EXTRA_MEMORY.md](INTRA_EXTRA_MEMORY.md).

Memory flows through the DAG pipeline just like metrics, insights, and lineage.
Here is the exact data flow:

```
Program enters DAG pipeline
        │
        ▼
ValidateCodeStage ──(success)──► MemoryContextStage
                                       │
                                       │ calls provider.select_cards(program, task, metrics)
                                       │
                                       │ NullMemoryProvider: returns empty instantly
                                       │ SelectorMemoryProvider: queries memory DB
                                       │
                                       ▼
                                  StringContainer("1. Sort evidence...\n\n2. Filter noise...")
                                       │
                                       │ also writes card IDs to program.metadata
                                       │   key: "memory_selected_idea_ids"
                                       │   value: ["idea-abc", "idea-def"]
                                       │
                                       ▼
                                 MutationContextStage
                                       │
                                       │ receives "memory" input via data flow edge
                                       │ creates MemoryMutationContext
                                       │ composes with MetricsMutationContext,
                                       │   InsightsMutationContext, etc.
                                       │
                                       ▼
                                 program.metadata["mutation_context"] =
                                   "## Metrics\n...\n\n---\n\n[card 1] id=idea-abc\nSort evidence..."
                                       │
                                       ▼
                                 LLM Mutation Agent reads mutation_context
                                 and uses memory ideas to guide the mutation
```

When `memory=none`:
- MemoryContextStage uses NullMemoryProvider
- Returns empty string immediately (zero latency, no network calls)
- MutationContextStage skips the empty memory section
- Everything works exactly as if the stage didn't exist

When `memory=reader` or `memory=full`:
- MemoryContextStage uses SelectorMemoryProvider (the `MemorySystem.provider`)
- Queries the memory database for relevant cards
- Returns formatted card text
- MutationContextStage includes it in the composite context

---

## Architecture: The Provider Pattern

The key abstraction is `MemoryProvider` (`gigaevo/memory/provider.py`):

```python
class MemoryProvider(ABC):
    @abstractmethod
    async def select_cards(
        self, program: Program, *,
        task_description: str, metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        """Select memory cards relevant to this program.

        ``parent_context`` is the fresh this-pass lineage card + live
        evolutionary snapshot the selector conditions retrieval on (None
        falls back to the legacy per-parent metadata block)."""
```

Two implementations:

| Provider | Config | What it does |
|----------|--------|-------------|
| `NullMemoryProvider` | `memory=none` or `memory=writer` (reader off) | Returns empty. Zero overhead. |
| `SelectorMemoryProvider` | `memory=reader` or `memory=full` (reader on) | Queries memory DB via a `MemoryReadPipeline` (retrieve → shortlist → auction → budget → render) |

The provider is reached as `MemorySystem.provider`; consumers wire to it via
`${ref:memory::provider}`. When the reader is off (`memory=none`,
`memory=writer`) that ref resolves to a `NullMemoryProvider`.

### Why a provider instead of a flag?

Old design had `memory_enabled=True` in the engine config, checked with
`if/else` in the engine loop. Problems:
- Broken in steady-state engine (the flag wasn't checked there)
- `if/else` branches scattered across engine, operator, mutation functions
- Hard to add new memory backends

New design uses the **Null Object pattern**: the provider IS the behavior.
`NullMemoryProvider` is the "off" state — a real object that does nothing, not a
flag that gates code paths. Benefits:
- Works identically in generational AND steady-state engines
- No `if memory_enabled:` checks anywhere
- Adding a new backend = one new class + one YAML file

---

## Configuration Reference

Memory is configured entirely through Hydra — there is no side-loaded YAML and
no environment-variable cascade. ONE knob assembles everything:

1. **Preset selection** (`config/memory/{none,reader,writer,full}.yaml`) —
   selected via `memory=<name>` on the command line. Each preset assembles a
   single `MemorySystem` node (`gigaevo/memory/system.py`,
   `_target_: gigaevo.memory.system.MemorySystem`) that owns BOTH the read side and
   the write side. Two booleans inside that node — `reader_enabled` /
   `writer_enabled` — are what each preset flips.
2. **Per-component groups** (`config/memory/<group>/*.yaml`) — one group per
   pipeline stage; each nests under `memory.<group>` (Hydra's natural
   packaging, no `@package` splats) and is injected into the `MemorySystem` via
   `${ref:...}` so every consumer shares one instance per component.
3. **The memory LLM** (`config/memory/common/llm/*.yaml`) — a `MultiModelRouter`
   shared by every consumer. The writer side spends money on `memory/common/llm`.

The `MemorySystem` threads the shared memory LLM
into ONE backend in Python — not via a `${ref:memory.*}` YAML web. It exposes
`.provider` (a `NullMemoryProvider` when the reader is off) and `.tracker` (a
`NullPostRunHook` when the writer is off). Consumers wire to it via
`${ref:memory::provider}` (read side) and `${ref:memory::tracker}` (write
side). The `${ref:...}` resolver instantiates a node on first access and writes
the instance back into the config tree, so later refs return the same object.

### Preset selection (`memory=<name>`)

```
config/memory/
  none.yaml    →  reader off + writer off  (NullMemoryProvider + NullPostRunHook)
  reader.yaml  →  reader on  + writer off  (inject from an existing bank; no extraction)
  writer.yaml  →  reader off + writer on   (extract/enrich cards for a LATER run; inject nothing)
  full.yaml    →  reader on  + writer on   (reader + writer share ONE card bank)
```

| Preset | reader | writer | What runs | Cost |
|--------|--------|--------|-----------|------|
| `memory=none` | off | off | `NullMemoryProvider` + `NullPostRunHook` | none |
| `memory=reader` | on | off | injects cards from an existing bank; no extraction | read only |
| `memory=writer` | off | on | extracts/enriches cards into a bank for a LATER run; injects nothing | memory/common/llm spend |
| `memory=full` | on | on | reader + writer share ONE card bank | memory/common/llm spend |

Each preset wires the `MemorySystem` from the per-component groups — what you
see in the config is exactly what the system receives:

```yaml
defaults:
  - retriever: gam
  - selector: llm
  - auction: thompson_ev          # EV-bid auction (theta x magnitude), abstains <= ev_floor
  - budget: top_bid               # bid-ranked cap to max_cards
  - reputation: absolute_progress # value channel: median base-relative gain
  - evictor: harm
  - excluder: none
  - provider: selector
  - tracker: librarian
  - backend: local
  - llm: gemini
  - _self_

_target_: gigaevo.memory.system.MemorySystem
reader_enabled: true
writer_enabled: true
max_cards: 1
checkpoint_dir: ${checkpoint_dir}
```

The default stack is the **EV contextual bandit**: cards bid their *expected
fitness gain* (`θ × magnitude`) and the auction abstains rather than inject a
card it expects to hurt. It works on every algorithm. To revert to the legacy
probability-only behavior, override all three:
`memory/reader/auction=thompson memory/reader/budget=top_theta memory/common/reputation=beta_binomial`.

The `evictor` singleton is a write-side component: the read backend never
ingests, so the `MemorySystem` threads it into the `CardAdmissionGate`, which
the librarian routes every write through and which sweeps confidently harmful
cards after each ingest pass. The sweep verdict is **global** (the card's whole
gain-event pool, `context=None`) by design: an eviction sweep has no query
parent, so there is no cell to reweight toward — see the rationale note in
`docs/audits/gain_event_sparsity_smoke_2026-06-28.md`. Override
`memory/writer/evictor=none` (`NullEvictor`) to keep the harm sweep off and let
cards accumulate uncapped — the write-side twin of `memory=none` on the read
side, useful as an admit-only ablation.

Swap a stage by overriding its group, tune a knob by path, switch the writer
LLM by its group:

```bash
python run.py memory=full \
  memory/common/llm=qwen_instruct \
  memory.auction.baseline_prior=[5,2] \
  checkpoint_dir=/workspace/experiments/hover/memory_store \
  problem.name=chains/hover/static
```

#### Asymmetric guards under `pipeline=intra_extra_memory`

The two sides of the knob fail differently under this pipeline — this is the #1
trap:

- **Writer guard RAISES.** `LiveMemoryRefreshHook.__init__` raises `TypeError`
  unless its tracker is a real `IncrementalPostRunHook`. So the writer-off
  presets (`memory=none`, `memory=reader`) FAIL FAST at startup under this
  pipeline.
- **Reader guard only WARNS.** `IntraExtraMemoryPipelineBuilder.__init__` only
  warns (never raises) on a `NullMemoryProvider`, because `memory=writer` is a
  LEGITIMATE write-cost-controlled baseline (cards written by the tracker, never
  injected). `memory=full` turns the read path back on.
- **A true no-memory baseline is `pipeline=standard memory=none`** — both sides
  off, no guard fires.

### Component groups

The **default** variant of each group is in **bold** (this is what `memory=full`
/ `memory=reader` compose with no overrides):

| Group | Variants (default in bold) | Class | Role |
|-------|----------|-------|------|
| `memory/common/backend` | **`local`** | `build_local_backend` (partial) | Card-bank construction (lazy, fail-fast) |
| `memory/reader/retriever` | **`gam`** | `GamRetriever` | Agentic GAM search (tools, top-k) |
| `memory/reader/selector` | **`llm`** | `LLMCardSelector` | Picks cards from retrieval hits |
| `memory/reader/auction` | `thompson`, **`thompson_ev`**, `thompson_ev_calibrated` | `ThompsonAuctioneer` / `EVThompsonAuctioneer` / `CalibratedColdPriorAuctioneer` | Card auction (`thompson_ev` bids `θ × magnitude` and abstains below `ev_floor`; `thompson_ev_calibrated` is `thompson_ev` with the cold-card bid calibrated per slate; `thompson` bids `θ` only) |
| `memory/reader/budget` | `top_theta`, **`top_bid`** | `TopThetaBudgeter` / `TopBidBudgeter` | Caps cards per injection (`top_bid` ranks by EV bid, `top_theta` by `θ`) |
| `memory/common/reputation` | `beta_binomial`, **`absolute_progress`**, `bd_proximity` | `BetaBinomialReputation` / `BetaBinomialReputation` / `BDProximityReputation` | Per-card efficacy posterior + value channel (`absolute_progress` is now an alias of `beta_binomial`; `bd_proximity` = cell-local, single-island only) |
| `memory/writer/evictor` | **`harm`**, `none` | `HarmEvictor` / `NullEvictor` | Evicts confidently harmful cards on each write sweep (threaded into the `CardAdmissionGate`); `none` disables the sweep so cards are only ever admitted, never evicted |
| `memory/reader/excluder` | **`none`**, `lineage` | `NullExcluder` / `LineageExcluder` | Filter-first gate (`lineage` drops cards already applied in this lineage) |
| `memory/reader/provider` | **`selector`** | `SelectorMemoryProvider` | Read-side provider (`shortlist_k` recall width, `max_cards` budget) |
| `memory/writer/tracker` | **`librarian`** | `IdeaTracker` | Writer side of `memory=` (librarian authors cards + routes verdicts through `CardAdmissionGate`) |
| `memory/common/llm` | **`gemini`**, `qwen_instruct` | `MultiModelRouter` | The memory LLM (writer-side spend) |

### Contextual-bandit card selection (EV auction)

This is the **default** stack (`thompson_ev` + `top_bid` + `absolute_progress`).
The legacy `thompson` auction bids a card's success *probability* `θ` alone; the
`thompson_ev` default instead turns card injection into a contextual bandit: the
context is the query parent's MAP-Elites cell, the arms are the candidate cards,
and the bid is `θ × magnitude` where `magnitude` is the card's *expected fitness
gain* (a signed value channel), not just its hit rate. A card whose expected gain
is non-positive bids at or below `ev_floor` (default `0.0`) and the auction
**abstains** — it injects nothing rather than a neutral card.

The value channel comes from the reputation arm:

- `absolute_progress` (`BetaBinomialReputation`) — the card's magnitude is
  `IntroGain_best_median`, the median base-relative child gain it has produced,
  pooled across all cells. (`absolute_progress` and `beta_binomial` resolve to
  the same class; the name is kept for config back-compat.)
- `bd_proximity` (`BDProximityReputation`) — **single-island only.** Re-buckets
  each card's stored `gain_events` into the *query parent's current cell* and
  bids over the in-cell subset only, so a card that helped near cell A and hurt
  near cell B bids high in A and abstains in B from one stored list. A cell with
  no in-cell evidence delegates to `absolute_progress`. Requires a top-level
  `${ref:behavior_space}` (the `single_island*`, `topology_3d*`, and
  `tabular/2d_local_ood` algorithms); pairing it with `multi_island` fails fast
  with a `NotImplementedError`.

#### `auction=thompson_ev_calibrated` (calibrated cold prior)

Same gate and `ev_floor` as `thompson_ev`, but the cold-card bid is **calibrated
per slate** instead of fixed. A cold card (no stamped magnitude) bids the
`cold_quantile` (default `0.5` = median) of the proven magnitudes present on its
own slate, so the cold bid tracks the substrate's fitness scale rather than the
fixed `prior_magnitude=0.1` — which on heilbron was ~25–80× the learned card
magnitudes and let untested cards out-bid proven ones
(`docs/audits/bandit_health_report.md`, RQ1). When no proven magnitude is present
(early run or an all-cold slate) every cold card bids `cold_floor` (default
`1e-6`) and the EV auction degenerates to the plain Thompson safety gate.

| field | default | meaning |
|---|---|---|
| `cold_quantile` | `0.5` | quantile of the slate's present magnitudes used as the cold bid; lower = more conservative exploration |
| `cold_floor` | `1e-6` | strictly-positive cold bid for all-cold / non-positive-quantile slates; keeps a fresh card explorable |

Opt-in only; `thompson_ev` remains the default. Intended to be A/B'd against
`thompson_ev` (re-run `tools/analyze_bandit_health.py`) before any default flip.

The gain evidence is **use-attributed**: a base-parent fitness snapshot is frozen
at child birth, and a card is credited (`gain = child − base`) only when it was
both *offered* to the mutation and *cited* by the mutator as used — selection
alone earns no credit. `memory.provider.shortlist_k` controls how wide the GAM
recall is before the auction (distinct from `max_cards`, the injection budget).

The `memory/*` arms above are necessary but **not sufficient** — the
contextual-bandit path only fires with the matching **non-memory** settings:
`pipeline=intra_extra_memory` (the memory-augmented mutation pipeline; the
`standard` pipeline has no memory-context stage), `storage=disk`, `num_parents=2`
(so `base_parent` and the donor/base credit distinction are live), and the
Qwen-on-proxy LLM overrides (the default model 401s on the proxy). The full
single-island treatment recipe:

```bash
python run.py \
  problem.name=heilbron \
  storage=disk \
  pipeline=intra_extra_memory \
  num_parents=2 \
  algorithm=single_island \
  memory=full \
  memory/common/llm=qwen_instruct \
  memory/reader/excluder=lineage \
  memory/common/reputation=bd_proximity \
  memory/reader/auction=thompson_ev \
  memory/reader/budget=top_bid \
  memory.provider.shortlist_k=10 \
  post_step_hook.refresh_every=10 \
  pipeline_builder.fresh_context_reorder=true \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  llm_base_url=http://localhost:8000/v1 \
  max_mutants=800
```

Swap only `memory/common/reputation=absolute_progress` for the context-free control arm
(same auction/budget/shortlist stack). The full derivation (RL framing, Thompson
sampling, the absolute-fitness value function, abstention proof) and a
per-override breakdown live in `docs/reports/memory_bandit_system.pdf`.

### Backend builder

The `memory/common/backend` sub-group has exactly ONE backend.
`config/memory/common/backend/local.yaml` is a Hydra `_partial_` over
`gigaevo.memory.shared_memory.backend.build_local_backend` bound to a nested
`MemoryConfig` node — every knob (`checkpoint_path`, `embedding_model_name`,
`search_limit`, `rebuild_interval`) is a Hydra field on that config. The partial
runs lazily on first card selection and raises `MemoryStorageError` on failure
rather than degrading to a no-memory run.

### The memory LLM (`config/memory/common/llm/`)

The writer side spends money on `memory/common/llm`. The default is `gemini`; swap it
with `memory/common/llm=qwen_instruct`. The `MemorySystem` threads this one router into
the backend in Python — the read side changes fitness, the writer side spends
the tokens. The node is a `gigaevo.llm.models.MultiModelRouter` with
`name: memory`, so its token usage is tracked separately from the evolution LLM
under `llm/tokens/memory/<model>/...`. The only environment variable involved is
the credential, read as `${oc.env:OPENROUTER_API_KEY}` — model id, endpoint,
temperature, reasoning effort, and `structured_output_method` are all YAML
fields.

The ideas-tracker librarian (the writer side) shares the SAME router instance, so
writer traffic is also booked under `llm/tokens/memory/<model>/...`.

#### Which settings matter most?

| Setting | Default | Why it matters |
|---------|---------|---------------|
| `memory.max_cards` | `1` | How many cards reach the prompt per mutation (the injection budget) |
| `memory.provider.shortlist_k` | `10` | Recall width: cards the selector shortlists before the auction ranks and the budgeter caps to `max_cards`. Set to `1` to fuse shortlist with the budget (legacy collapse) |
| `memory.auction.ev_floor` | `0.0` | Min expected gain to inject (`thompson_ev`); raise to inject only confidently-positive cards, lower (negative) to inject more eagerly |
| `memory.auction.prior_magnitude` | `0.1` | Optimistic cold-gain prior for a card with no value evidence; tune to the substrate's fitness scale (e.g. `0.05`) |
| `memory.auction.baseline_prior` | `[3.0, 3.0]` | Beta prior on a cold card's success probability `θ` |
| `memory.retriever.allowed_tools` | `[page_index, vector]` | Which GAM search tools the agent may call |
| `checkpoint_dir` | — | Where the card bank lives on disk (command line) |
| `memory/common/llm` | `gemini` | Writer LLM group: `gemini` vs `qwen_instruct` (command line) |

Component-level swaps (`memory/common/reputation=bd_proximity`, `memory/reader/excluder=lineage`,
`memory/reader/auction=thompson`, …) are the coarse levers; the table above are the
fine knobs. Everything else has sane defaults.

---

## The Ideas Tracker (Write Phase)

The Ideas Tracker extracts generalizable ideas from programs produced by an
evolution run and writes them as memory cards. It lives in
`gigaevo/memory/ideas_tracker/`.

### What It Does

1. Loads programs from the engine's storage — the full set at run end
   (`on_run_complete`), or the newest window each increment under the live hook
2. Filters to eligible records: non-root programs with positive fitness that
   have not already been processed
3. For each new program, the **Librarian** (`ideas_tracker/librarian.py`)
   authors a clean memory card from the parent→child mutation diff,
   reconciling it against the nearest existing cards (the memory LLM decides
   *new card* vs. *merge into an existing one*)
4. Routes every verdict through the **`CardAdmissionGate`** (`admit` / `merge` /
   `bump_provenance`) — the sole harm gate, which records each verdict to
   `write_ledger.jsonl`
5. Authors a clean `ProgramCard` for each top-fitness exemplar (top
   `memory_write_best_programs_percent`% of the pool)
6. Stamps each credited card's use-attributed `gain_events` from the full pool
   (reputation derives the Beta-Binomial posterior from them at read time)
7. Sweeps confidently harmful cards (`gate.sweep()`)
8. Every `consolidation_every_n` cards written, schedules one background
   **consolidation** pass (`ideas_tracker/consolidation.py`) that runs the same
   nearest-card primitive over the whole bank and folds drifted near-duplicate
   cards (cosine distance ≤ `consolidation_eps`) into one — the standard fix for
   the greedy, order-dependent online pre-gate. Idempotent and off the hot path

### Entry Point: PostRunHook

The IdeaTracker runs as an engine PostRunHook:

```
                    ┌──────────────────────────────────┐
                    │       PostRunHook (automatic)     │
                    │                                   │
                    │  EvolutionEngine.run() completes  │
                    │          ↓ finally block          │
                    │  hook.on_run_complete(storage)    │
                    │          ↓                        │
                    │  IdeaTracker fetches all programs │
                    │  from storage and runs pipeline   │
                    └──────────────────────────────────┘
```

Turn the writer on with `memory=writer` (build-only) or `memory=full`
(read+write) in your Hydra command. The engine fires
`on_run_complete(storage)` in its `run()` method's `finally` block after
evolution completes; under `pipeline=intra_extra_memory` the
`LiveMemoryRefreshHook` also calls `run_increment` mid-run on a cadence.
Hook errors are caught and logged — they never crash the engine.

### The Writer Side of `memory=` (`config/memory/writer/tracker/`)

The IdeaTracker is the WRITER SIDE of the `memory=` knob. The `memory=writer`
and `memory=full` presets flip `writer_enabled: true`; `memory=none` and
`memory=reader` leave it off, so `MemorySystem.tracker` resolves to a
`NullPostRunHook`. Consumers wire to it via `${ref:memory::tracker}`.

There is a single tracker config — `config/memory/writer/tracker/librarian.yaml`:

```yaml
_target_: gigaevo.memory.ideas_tracker.ideas_tracker.IdeaTracker
_partial_: true
memory_write_enabled: true
memory_write_best_programs_percent: 5.0
ingest_call_timeout_s: 300.0
consolidation_every_n: 64
dedup_policy:
  _target_: gigaevo.memory.ideas_tracker.dedup_policy.DedupPolicy
  online_eps: 0.05
  online_top_k: 5
  max_cards_per_diff: 3
  consolidation_eps: 0.05
  consolidation_k: 5
fitness_higher_is_better: ${higher_is_better}
checkpoint_dir: ${checkpoint_dir}
task_description: ${ref:problem_context::task_description}
prompts_dir: ${prompts.dir}
```

The `MemorySystem` threads the shared `memory/common/llm` router, the `memory/common/backend`
`_partial_` (over `build_local_backend`), and the shared `evictor` /
`reputation` instances into the tracker in Python — the tracker does not
assemble its own `${ref:memory.*}` web. The tracker uses the same router and the
same backend builder (and on-disk checkpoint) as the read side, so its token
usage is booked under `llm/tokens/memory/<model>/...`.

#### Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_write_enabled` | bool | `true` | Author and admit cards into the memory bank |
| `memory_write_best_programs_percent` | float | `5.0` | Share of top programs (by fitness) the librarian converts into program cards |
| `ingest_call_timeout_s` | float | `300.0` | Per-call wall-clock bound on each librarian LLM call (reconcile/author). A call that stalls past this is skipped — the record is retried on a later sweep — so one hung memory-LLM request cannot freeze the whole write increment |
| `consolidation_every_n` | int | `64` | After this many cards are written across sweeps, schedule one background consolidation pass that folds drifted near-duplicate cards the greedy online pre-gate let in separately. `0` disables it |
| `dedup_policy` | `DedupPolicy` | `DedupPolicy()` | Frozen, Hydra-instantiable value object holding the dedup thresholds: `online_eps` / `online_top_k` / `max_cards_per_diff` for the online pre-gate, and `consolidation_eps` / `consolidation_k` for the batch pass |
| `fitness_higher_is_better` | bool | `${higher_is_better}` | Metric direction; gains are stored in "positive = improvement" space |
| `checkpoint_dir` | str | `${checkpoint_dir}` → `${hydra:runtime.output_dir}/memory` | Directory for memory card storage. Inherits the global `checkpoint_dir`, which defaults to the Hydra run's output dir (`config/config.yaml`). To reuse one bank across the build phase (writer) and read phase (reader), override it to a fixed path in both runs (e.g. `checkpoint_dir=experiments/hover/memory/memory_bank`). |
| `task_description` | str | `${ref:problem_context::task_description}` | Human-readable task text; condensed once per run into the one-line summary stamped on every card |
| `prompts_dir` | str or null | `${prompts.dir}` | Optional prompt-override directory for the librarian agents; `null` uses the packaged prompts |

### Pipeline Internals

The core pipeline (`run_increment`) runs this sequence:

```
1. Load programs
   │  PostRunHook: storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
   │
2. Filter to eligible records (ProgramRecordExtractor.extract)
   │  Skip: root programs (no parents)
   │  Skip: no validated fitness (missing/non-positive is_valid;
   │        missing, non-finite, or sentinel fitness)
   │  Skip: already-seen ids
   │
3. Reconcile each diff into idea cards (Librarian.ingest_idea)
   │  ReconcileAgent authors a clean card from the parent→child diff and
   │  decides new-card vs. merge against the nearest existing cards; the
   │  CardAdmissionGate records every verdict to write_ledger.jsonl
   │
4. Author exemplar program cards (_author_exemplars)
   │  ProgramAuthorAgent writes one ProgramCard per top-fitness exemplar
   │  (top memory_write_best_programs_percent% of the pool), admitted
   │  through the same gate
   │
5. Stamp gain events + harm-evict (CardStatsUpdater.update)
   │  Use-attributed base-relative gains from the full pool are stamped
   │  onto each credited card (reputation derives the Beta-Binomial
   │  posterior from them at read time); confidently harmful cards are
   │  then evicted via gate.sweep
   │
6. Schedule consolidation if due (ConsolidationScheduler.note_writes)
   │  After consolidation_every_n cards, dispatch one background
   │  near-duplicate consolidation pass under the run lock
```

The store, admission gate, and librarian are built once, lazily, on the first
sweep (`LibrarianWriteStack.ensure()`), so the heavy backend I/O is paid only
when there is something to write.

### Card Bank and Backend

Cards are written to the local memory backend (`memory/common/backend=local`, the only
backend): JSON files under `checkpoint_dir` with a search index, shared with the
read side. The backend builder is injected — the PostRunHook path gets the
`MemorySystem`'s shared backend partial (the same llm-bound partial the
read-side provider uses).

### What a Memory Card Looks Like

Internally, a memory card is a structured object with these fields:

```python
{
    "id": "idea-abc-123",
    "description": "Sort evidence by relevance score before traversing the chain",
    "category": "retrieval",
    "keywords": ["sort", "relevance", "evidence", "chain"],
    "task_description_summary": "Multi-hop fact verification using evidence chains",
    "gain_events": [                          # raw use-attributed child gains
        {"context": {"parent_metrics": {"min_area": 0.41}}, "gain": 0.03},
        {"context": {"parent_metrics": {"min_area": 0.55}}, "gain": 0.0, "invalid": True},
    ],                                        # reputation derives the posterior at read time
    "programs": ["prog-1", "prog-2"],       # programs that produced this idea
    "absorbed_ids": [],                       # ids merged into this survivor; re-alias absorbed cards' gain events at restamp
}
```

The `description` is the core idea. Everything else is metadata for search
ranking and efficacy-aware selection.

### Logs and Checkpoints

The write path leaves two append-only traces:

```
<output>/memory/memory_events.jsonl   # structured event trace: admission.select,
                                      # gam.search, read.selection,
                                      # injection_posterior.compute, store.* ...
<checkpoint_dir>/write_ledger.jsonl   # one row per card verdict:
                                      # incoming_id, final_id, outcome, reason
```

The card bank itself lives under `checkpoint_dir` (`api_index.json` plus the
page store and amem exports), shared between the writer and the reader.

---

## The Memory Search (Read Phase)

When the reader is on (`memory=reader` or `memory=full`), here's what happens on
each program evaluation:

1. **`MemoryContextStage`** calls `SelectorMemoryProvider.select_cards()`
2. The provider assembles a **`MemoryReadPipeline`** lazily on first call
   (retriever → shortlister → auctioneer → budgeter → renderer)
3. `LLMCardSelector` builds a query from the parent code, task description, and metrics
4. The query is sent to the memory backend (local `AmemGamMemory` or remote API)
5. The **GAM (Generative Agentic Memory) pipeline** runs:
   - Retrieval tools search the card indices (vector similarity, page index)
   - The selector LLM emits an ordered `final_decision` shortlist of card ids
6. The shortlist goes through a **Thompson auction** on each card's
   Beta-Binomial efficacy posterior, then a top-theta budget cap (`max_cards`)
7. Card text is rendered as numbered blocks headed `[card N] id=<card-id>`
   (selection-rank order), each with a mechanism description and — when the
   bank has confident evidence — a trailing `efficacy:` line
8. Card IDs are stored in program metadata for tracking

The GAM pipeline is configurable via the `memory/reader/retriever` Hydra group
(`config/memory/reader/retriever/gam.yaml`). The `allowed_tools` list controls which
retrieval strategies are used.

---

## Tracking: How to Know if Memory Was Used

### On individual programs

Every mutant has a `memory_used` metadata flag, auto-derived after mutation:

```python
program.get_metadata("memory_used")  # True or False
```

Logic: if ANY parent of the mutation had memory cards selected (i.e., the parent
has `memory_selected_idea_ids` in its metadata with a non-empty list), then
`memory_used=True` on the child.

The selected card IDs themselves:
```python
program.metadata["memory_selected_idea_ids"]  # ["idea-abc", "idea-def"]
```

### In experiments

Use `status.py --experiment` and the evolution data CSV to compare:
- Fitness trajectory of memory-augmented mutations vs. non-memory mutations
- Which specific ideas (card IDs) were most frequently selected
- Whether memory usage correlates with fitness improvements

---

## Full Experiment Workflow

A memory experiment has two phases: build the bank, then run a controlled
experiment with and without memory.

### Phase A: Build the Memory Bank

Run evolution with `memory=writer` (writer on, reader off). The IdeaTracker
fires as a PostRunHook after evolution completes and writes memory cards to
`checkpoint_dir`; nothing is injected during this run.

```bash
# Phase A: Run evolution with the writer enabled
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  memory=writer \
  checkpoint_dir=experiments/hover/memory/memory_bank \
  redis.db=3 \
  max_mutants=200
```

After the run completes, check the memory bank:
```bash
ls experiments/hover/memory/memory_bank/
```

### Phase B: Controlled Experiment

Run 2+ control runs (no memory) and 2+ treatment runs (with memory from
Phase A). All runs use the same problem, config, and model.

```bash
MEMORY_BANK="experiments/hover/memory/memory_bank"

# R1: control (no memory)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  redis.db=4

# R2: control (no memory)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  redis.db=5

# R3: treatment (reader enabled, consumes the Phase A bank)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  memory=reader \
  checkpoint_dir="$MEMORY_BANK" \
  redis.db=6

# R4: treatment (reader enabled, consumes the Phase A bank)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  memory=reader \
  checkpoint_dir="$MEMORY_BANK" \
  redis.db=7
```

### Analysis

```bash
# Monitor all runs
gigaevo status \
  -r "chains/hover/full7_no_deep@4:R1" \
  -r "chains/hover/full7_no_deep@5:R2" \
  -r "chains/hover/full7_no_deep@6:R3" \
  -r "chains/hover/full7_no_deep@7:R4"

# Compare fitness trajectories
gigaevo plot comparison \
  -r "chains/hover/full7_no_deep@4:control-1" \
  -r "chains/hover/full7_no_deep@5:control-2" \
  -r "chains/hover/full7_no_deep@6:memory-1" \
  -r "chains/hover/full7_no_deep@7:memory-2" \
  --output-folder experiments/hover/memory/plots/

# Check memory usage in treatment runs
gigaevo top \
  -r "chains/hover/full7_no_deep@6:memory-1" -n 5 --code
```

---

## Key Files

### Provider Layer (Hydra-injected, Read Phase)

| File | What it does |
|------|-------------|
| `gigaevo/memory/system.py` | `MemorySystem` — owns `.provider` (read side) + `.tracker` (write side) |
| `gigaevo/memory/provider.py` | `MemoryProvider` ABC, `NullMemoryProvider`, `SelectorMemoryProvider` |
| `config/memory/none.yaml` | Hydra preset: reader off + writer off (default) |
| `config/memory/reader.yaml` | Hydra preset: reader on + writer off |
| `config/memory/writer.yaml` | Hydra preset: reader off + writer on |
| `config/memory/full.yaml` | Hydra preset: reader on + writer on |

### DAG Pipeline (Read Phase)

| File | What it does |
|------|-------------|
| `gigaevo/programs/stages/memory_context.py` | `MemoryContextStage` — calls provider, returns numbered `[card N] id=…` blocks |
| `gigaevo/evolution/mutation/context.py` | `MemoryMutationContext` — wraps the memory slot (intra card in current pipelines) for the mutation prompt |
| `gigaevo/programs/stages/mutation_context.py` | `MutationContextStage` — composes all context types |
| `gigaevo/entrypoint/default_pipelines.py` | Wires MemoryContextStage → mutator (LEGACY default pipeline only) |
| `gigaevo/entrypoint/lineage_memory_pipeline.py` | Wires MemoryContextStage → MutationSuggestionStage (`pipeline=intra_extra_memory`) |
| `gigaevo/evolution/engine/mutation.py` | Auto-derives `memory_used` from parent metadata |

### Memory Backend

| File | What it does |
|------|-------------|
| `gigaevo/memory/provider.py` | `SelectorMemoryProvider` — assembles the read pipeline lazily |
| `gigaevo/memory/core/read_pipeline.py` | `MemoryReadPipeline` — retrieve → shortlist → auction → budget → render |
| `gigaevo/memory/shared_memory/backend.py` | Lazy fail-fast backend builder (`build_local_backend`) |
| `gigaevo/memory/shared_memory/memory.py` | `AmemGamMemory` — local memory backend with GAM search |
| `config/memory/common/backend/` | Hydra config for the backend builder (checkpoint dir, embedding model, memory LLM) |

### Shared Memory Module (`gigaevo/memory/shared_memory/`)

`AmemGamMemory` is the orchestrator; the rest are pluggable collaborators wired
via the `AgenticRuntime` DI container.

| File | Responsibility |
|------|---------------|
| `memory.py` | `AmemGamMemory` orchestrator — coordinates save / search / rebuild / delete |
| `memory_config.py` | Pydantic configs: `MemoryConfig`, `GamConfig`, `ApiConfig` |
| `card_store.py` | Card dict + entity mappings + JSON index persistence |
| `card_conversion.py` | Card <-> record/note/concept conversion + `RawCardRecord` validation |
| `note_sync.py` | Bridges cards to the A-MEM vector store (Chroma) |
| `neighbor_source.py` | `ChromaNeighborSource` — nearest existing cards (reuses the store's A-MEM Chroma index) for the online pre-gate and consolidation |
| `api_sync.py` | Paginated fetch / full sync / remote search via concept API |
| `gam_search.py` | GAM `ResearchAgent` build + invalidate lifecycle |
| `agentic_runtime.py` | `AgenticRuntime` factory: injects LLM + generator + agentic classes |
| `protocols.py` | DI protocols (`LLMServiceProtocol`, `AgenticMemorySystemProtocol`, …) |

Search order is three-tier: GAM `ResearchAgent` (vector retrievers) → concept API
(remote full-text + LLM synthesis) → in-memory keyword fallback. Each tier falls
through on failure or empty result.

### PostRunHook (Engine Integration)

| File | What it does |
|------|-------------|
| `gigaevo/evolution/engine/hooks.py` | `PostRunHook` ABC + `NullPostRunHook` (no-op default) |
| `gigaevo/evolution/engine/core.py` | `EvolutionEngine.run()` fires hook in `finally` block |

### Ideas Tracker (Write Phase)

| File | What it does |
|------|-------------|
| `gigaevo/memory/ideas_tracker/ideas_tracker.py` | `IdeaTracker(PostRunHook)` — thin orchestration shell over the four write-path collaborators |
| `config/memory/writer/tracker/librarian.yaml` | Hydra config: IdeaTracker (writer side of `memory=`) |

### Ideas Tracker Modules

| File | What it does |
|------|-------------|
| `record_extractor.py` | `ProgramRecordExtractor` — eligible programs → `ProgramRecord` (+ dedup bookkeeping); `program_to_record`, `record_note` |
| `write_stack.py` | `LibrarianWriteStack` — lazy, off-event-loop assembly of store/gate/neighbors/librarian/consolidation-agent plus the one-line task summary |
| `card_stats.py` | `CardStatsUpdater` — gain-event attribution + authoritative restamp + harm sweep |
| `consolidation_scheduler.py` | `ConsolidationScheduler` — cadence-gated background consolidation pass dispatched under the tracker's run lock |
| `librarian.py` | `Librarian` — authors clean cards from a mutation diff, reconciles NEW/DUPLICATE/MERGE, routes verdicts through the `CardAdmissionGate`; also `author_program` |
| `consolidation.py` | `consolidate` — batch pass that folds drifted near-duplicate cards into one via the same `NeighborSource` primitive; idempotent |
| `dedup_policy.py` | `DedupPolicy` — frozen, Hydra-instantiable value object unifying the online + consolidation dedup thresholds |
| `fitness.py` | `select_top_programs` / `valid_fitness` — exemplar slice + sentinel-fitness filtering |
| `hf_cache.py` | `ensure_writable_hf_cache` — redirect HF cache env vars to a writable dir before the embedding model loads |
| `models.py` | Data structures: `ProgramRecord`, `program_to_record` |

### Efficacy core (`gigaevo/memory/efficacy/`)

| File | What it does |
|------|-------------|
| `scorer.py` | `robust_noise_band`, `beta_binomial_posterior`, `block_from_events`: MAD noise band + Beta-Binomial posterior derived at read time from a card's raw `gain_events` |
| `stamping.py` | `CardStatsStamper`: single writer of card-side stats (`DecisionMetrics` vocabulary only) |

### Tests

| File | What it covers |
|------|---------------|
| `tests/memory/test_provider.py` | Provider abstraction (null, selector, lazy init) |
| `tests/memory/test_memory_context_stage.py` | MemoryContextStage + MemoryMutationContext |
| `tests/memory/test_dag_memory_flow.py` | End-to-end DAG flow, composite context, auto-derivation |
| `tests/memory/test_ideas_tracker.py` | IdeaTracker pipeline: record extraction, PostRunHook contract, program filtering, gain restamp, engine integration |
| `tests/memory/test_consolidation_trigger.py` | ConsolidationScheduler: cadence, non-blocking dispatch, run-lock serialization, shutdown drain |
| `tests/memory/test_data_components.py` | Data structures: RecordBank, RecordCardExtended, IncomingIdeas |
| `tests/memory/test_typed_card_stats.py` | Typed `CardStatsBlock`/`DecisionMetrics` reads, gain-event posteriors, typed auction slate, no dict-plumbing source scan |
| `tests/integration/test_memory_e2e.py` | Full-loop E2E with real EvolutionEngine + fakeredis |

---

## FAQ

### Memory Read Phase

**Q: Does memory add latency?**
With the reader off (`memory=none`, `memory=writer`), zero. With the reader on
(`memory=reader`, `memory=full`), search runs on the local disk bank
(~50-200ms depending on card count and GAM tools). The search runs in parallel
with other DAG stages (insights, lineage), so the wall-clock impact is often
hidden.

**Q: Can I use memory with the steady-state engine?**
Yes. This was the main reason for the refactor. The old implementation was
broken in steady-state because memory was hardcoded in the generational engine
loop. Now both engines use the same DAG pipeline.

**Q: What if the memory backend is unavailable?**
`MemoryReadPipeline` fails to an empty selection on every error path
(behaves like `NullMemoryProvider`). A warning is logged. The mutation proceeds
without memory guidance.

**Q: How many cards are selected per mutation?**
Configurable via `max_cards` in `config/memory/reader/provider/selector.yaml`. The memory
agent searches the database and returns the most relevant cards.

**Q: What's the difference between `memory=reader` and `memory=full`?**
Both turn the reader on, so both use `SelectorMemoryProvider` over the same
`config/memory/common/backend/local.yaml` on-disk `AmemGamMemory` bank. They differ on
the writer: `reader` leaves the writer off (it only consumes a bank built by an
earlier run), while `full` also turns the writer on so the same run injects AND
extracts cards into ONE shared bank.

**Q: How does the system decide which cards are "relevant"?**
The GAM pipeline sends the parent code + task description as a query, then
runs the configured retrieval tools (vector search, page index) to
find matching cards. The `allowed_tools` and `top_k_by_tool` settings in
`config/memory/reader/retriever/gam.yaml` control which tools run and how many
results each returns.

### Ideas Tracker (Write Phase)

**Q: How are cards authored?**
The **Librarian** authors one clean idea card per eligible mutation diff
(parent→child), and the memory LLM decides whether each diff is a *new card* or
a *merge* into the nearest existing card. Each top-fitness exemplar additionally
gets a clean `ProgramCard`. Every verdict is routed through the single
`CardAdmissionGate` (admit / merge / bump-provenance / evict).

**Q: When does the IdeaTracker run?**
As a PostRunHook after evolution completes (the writer side of `memory=writer`
or `memory=full`): the engine calls `on_run_complete(storage)` in its `run()`
finally block. Under `pipeline=intra_extra_memory` the `LiveMemoryRefreshHook`
also drives `run_increment` mid-run on a cadence so the bank fills as the run
progresses.

**Q: What happens if the IdeaTracker crashes during the PostRunHook?**
Nothing bad. The engine wraps the hook call in try/except — hook errors are
logged but never crash the engine. The evolution results are already saved.

**Q: What's `best_programs_percent` and why is it 5%?**
It bounds how many top programs (by fitness) the librarian turns into program
cards — idea cards still come from every eligible mutation diff. 5% is the
default; for a run with 200 programs, the top 10 become program cards.

**Q: How do I check what cards were written?**
Read `write_ledger.jsonl` under the card bank (`checkpoint_dir`): one row per
verdict (`incoming_id`, `final_id`, `outcome`, `reason`). The richer event trace
(`<output>/memory/memory_events.jsonl`) carries `admission.select`,
`store.insert`/`store.merge`, and `injection_posterior.compute`. The cards
themselves live in `checkpoint_dir/api_index.json`.

**Q: Can I disable the write path?**
Yes. Set `memory_write_enabled: false` in the Hydra config (`memory=none` /
`memory=reader` leave it off). Authoring and writing are one path now — with it
off, the tracker is a no-op.

### General

**Q: Can I add a new memory backend?**
Yes. Implement a backend builder (like `build_local_backend`), drop a new variant
under the `memory/common/backend` group (`config/memory/common/backend/your_backend.yaml`),
and select it with
`memory/common/backend=your_backend` alongside any reader/writer preset
(`memory=reader|writer|full`). The pipeline doesn't need any changes.

**Q: Where are cards stored on disk?**
At the path specified by `checkpoint_dir`. Inside that directory, the
`AmemGamMemory` backend stores cards as JSON files with an index for search.

**Q: Can two experiments share the same memory database?**
Yes, if they use the same `checkpoint_dir` and `namespace`. But be careful —
concurrent writes (from two ideas trackers) are not safe. Read-only sharing
during evolution is fine.
