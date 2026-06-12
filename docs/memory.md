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
   - [Provider selection (memory=...)](#provider-selection-memoryname)
   - [Component groups](#component-groups)
   - [Backend factories](#backend-factories)
   - [The memory LLM (config/llms/)](#the-memory-llm-configllms)
7. [The Ideas Tracker (Write Phase)](#the-ideas-tracker-write-phase)
   - [What It Does](#what-it-does)
   - [Two Entry Points: PostRunHook vs CLI](#two-entry-points-postrunhook-vs-cli)
   - [Hydra Config Group (ideas_tracker=...)](#ideas-tracker-hydra-config-group)
   - [CLI Reference](#cli-reference)
   - [CLI Examples](#cli-examples)
   - [Pipeline Internals](#pipeline-internals)
   - [Analyzer Types](#analyzer-types)
   - [Memory Write Pipeline](#memory-write-pipeline)
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
python run.py memory=none  ...   # No memory (default)
python run.py memory=local ...   # Memory from local backend
python run.py memory=legacy_api ...  # DEPRECATED: remote API backend
```

One Hydra override. Everything else is automatic.

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
efficacy: introduced in 9 children; median improvement +0.012; downside 11% (confident)

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
║                              Ideas Tracker (CLI tool)             ║
║                              extracts generalizable ideas         ║
║                                       │                           ║
║                                       ▼                           ║
║                              Memory Database (disk or API)        ║
╠═══════════════════════════════════════════════════════════════════╣
║                      READ PHASE                                   ║
║                                                                   ║
║  Evolution Run B (memory=local) ──> DAG pipeline                  ║
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

They never run at the same time. The ideas tracker runs AFTER an evolution
completes (or at checkpoints), and the next evolution reads from the database.

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

When `memory=local` or `memory=legacy_api`:
- MemoryContextStage uses SelectorMemoryProvider
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
    ) -> MemorySelection:
        """Select memory cards relevant to this program."""
```

Two implementations:

| Provider | Config | What it does |
|----------|--------|-------------|
| `NullMemoryProvider` | `memory=none` | Returns empty. Zero overhead. Default. |
| `SelectorMemoryProvider` | `memory=local` or `memory=legacy_api` | Queries memory DB via a `MemoryReadPipeline` (retrieve → shortlist → auction → budget → render) |

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
no environment-variable cascade. Three pieces compose together:

1. **Provider selection** (`config/memory/{none,local,legacy_api}.yaml`) —
   selected via `memory=<name>` on the command line.
2. **Per-component groups** (`config/memory/<group>/*.yaml`) — one group per
   pipeline stage; each nests under `memory.<group>` (Hydra's natural
   packaging, no `@package` splats) and is injected into the provider via
   `${ref:...}` so every consumer shares one instance per component.
3. **The memory LLM** (`config/llms/*.yaml`) — a `MultiModelRouter` composed
   ONCE at the root-registered `memory_llm` entry and shared by every
   consumer via `${ref:memory_llm}`.

Two singletons are registered as null base entries in `config/config.yaml`
(`llms@memory_llm: null`, `memory/backend: null`); the `memory=` and
`ideas_tracker=` groups override them, so the read side and the write side
share ONE router and ONE card-bank backend factory. The `${ref:...}` resolver
instantiates a node on first access and writes the instance back into the
config tree, so later refs return the same object.

### Provider selection (`memory=<name>`)

```
config/memory/
  none.yaml        →  NullMemoryProvider (default)
  local.yaml       →  SelectorMemoryProvider over the local card bank
  legacy_api.yaml  →  DEPRECATED: SelectorMemoryProvider over the HTTP API backend
```

`local.yaml` wires every stage explicitly — what you see in the config is
exactly what the provider receives:

```yaml
defaults:
  - retriever: gam
  - selector: llm
  - auction: thompson
  - budget: top_theta
  - reputation: beta_binomial
  - admitter: sign_based
  - dedup: llm
  - evictor: harm
  - _self_
  - override /memory/backend@_global_.memory.backend: local
  - override /llms@_global_.memory_llm: gemini_flash_openrouter

provider:
  _target_: gigaevo.memory.provider.SelectorMemoryProvider
  max_cards: 1
  checkpoint_dir: ${checkpoint_dir}
  backend: ${ref:memory.backend}
  retriever: ${ref:memory.retriever}
  selector: ${ref:memory.selector}
  auctioneer: ${ref:memory.auction}
  budgeter: ${ref:memory.budget}
  reputation: ${ref:memory.reputation}
```

The `dedup`/`evictor` singletons composed here are write-side components: the
provider's read backend never ingests, so `config/ideas_tracker/*.yaml` picks
them up (`evictor: ${ref:memory.evictor}`, `deduplicator: ${ref:memory.dedup}`)
and IdeaTracker threads them into the write pipeline, which sweeps confidently
harmful cards after each ingest pass.

Swap a stage by overriding its group, tune a knob by path:

```bash
python run.py memory=local \
  memory/admitter=tiered \
  memory.auction.baseline_prior=[5,2] \
  memory.retriever.pipeline_mode=experimental \
  checkpoint_dir=/workspace/experiments/hover/memory_store \
  problem.name=chains/hover/static
```

### Component groups

| Group | Variants | Class | Role |
|-------|----------|-------|------|
| `memory/backend` | `local`, `legacy_api` | `LocalMemoryBackendFactory` / `LegacyApiMemoryBackendFactory` | Card-bank construction (lazy, fail-fast) |
| `memory/retriever` | `gam` | `GamRetriever` | Agentic GAM search (tools, top-k, `pipeline_mode`) |
| `memory/selector` | `llm` | `LLMCardSelector` | Picks cards from retrieval hits |
| `memory/auction` | `thompson` | `ThompsonAuctioneer` | Thompson-sampling card auction |
| `memory/budget` | `top_theta` | `TopThetaBudgeter` | Caps cards per injection |
| `memory/reputation` | `beta_binomial` | `BetaBinomialReputation` | Per-card efficacy posterior |
| `memory/admitter` | `sign_based`, `tiered` | `SignBasedAdmitter` / `TieredAdmitter` | Write-side admission gate |
| `memory/dedup` | `llm`, `none` | `LLMDeduplicator` / `NullDeduplicator` | Write-side dedup (via `ideas_tracker` → write pipeline) |
| `memory/evictor` | `harm` | `HarmEvictor` | Evicts confidently harmful cards on each write sweep (via `ideas_tracker` → write pipeline) |

### Backend factories

`config/memory/backend/local.yaml` instantiates
`gigaevo.memory.backend_factory.LocalMemoryBackendFactory`; the deprecated
`legacy_api.yaml` instantiates `LegacyApiMemoryBackendFactory` (emits a
`DeprecationWarning`; kept only so historical experiments reproduce). Factories
are plain pydantic models — every knob (`checkpoint_dir`,
`embedding_model_name`, `search_limit`, `rebuild_interval`, legacy
`base_url`/`namespace`/`channel`/`sync_*`) is a Hydra field. `build()` runs
lazily on first card selection and raises `MemoryStorageError` on failure
rather than degrading to a no-memory run.

### The memory LLM (`config/llms/`)

The router composes ONCE at the root-registered `memory_llm` entry (the
`memory=` and `ideas_tracker=` groups both override
`/llms@_global_.memory_llm: gemini_flash_openrouter`); the backend factory
points at it via `llm: ${ref:memory_llm}`. The node is a
`gigaevo.llm.models.MultiModelRouter` with `name: memory`, so its token usage
is tracked separately from the evolution LLM under
`llm/tokens/memory/<model>/...`. The only environment variable involved is the
credential, read as `${oc.env:OPENROUTER_API_KEY}` — model id, endpoint,
temperature, reasoning effort, and `structured_output_method` are all YAML
fields. Switch the memory LLM off with `memory.backend.llm=null`.

The ideas-tracker analyzer shares the SAME router instance
(`llm: ${ref:memory_llm}` in `config/ideas_tracker/*.yaml`), so analyzer
traffic is also booked under `llm/tokens/memory/<model>/...`. Standalone CLI
runs build an equivalent router from `--model`/`--base-url`/`--api-key`.

#### Which settings matter most?

| Setting | Where | Why it matters |
|---------|-------|---------------|
| `memory.provider.max_cards` | `config/memory/local.yaml` | How many cards reach the prompt per mutation |
| `memory.retriever.pipeline_mode` | `memory/retriever/gam.yaml` | `experimental` = multi-tool agentic retrieval (required by the selector) |
| `memory.retriever.allowed_tools` | `memory/retriever/gam.yaml` | Which GAM search tools the agent may call |
| `memory.dedup.config.enabled` | `memory/dedup/llm.yaml` | LLM dedup on card writes |
| `checkpoint_dir` | command line | Where the card bank lives on disk |
| `memory/backend` | command line | `local` (canonical) vs `legacy_api` (deprecated) |

Everything else has sane defaults.

---

## The Ideas Tracker (Write Phase)

The Ideas Tracker extracts generalizable ideas from programs produced by an
evolution run and writes them as memory cards. It lives in
`gigaevo/memory/ideas_tracker/`.

### What It Does

1. Loads programs from a completed evolution run (via Redis or CSV)
2. Filters to non-root programs with positive fitness
3. Uses an LLM to analyze each program's improvements and classify them as
   **new ideas**, **updates** to existing ideas, or **rewrites** of existing ideas
4. Deduplicates ideas against existing cards in active/inactive idea banks
5. Enriches ideas with keywords, explanations, and task summaries (postprocessing)
6. Optionally tracks which memory cards were used and their fitness impact
7. Optionally writes the best ideas to the memory database for future runs

### Two Entry Points: PostRunHook vs CLI

The IdeaTracker has two ways to run:

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

                    ┌──────────────────────────────────┐
                    │       CLI (manual / standalone)   │
                    │                                   │
                    │  python -m gigaevo.memory         │
                    │    .ideas_tracker.cli             │
                    │    --redis-db 3                   │
                    │    --redis-prefix chains/hover/.. │
                    │          ↓                        │
                    │  IdeaTracker loads from Redis/CSV │
                    │  and runs the same pipeline       │
                    └──────────────────────────────────┘
```

**PostRunHook** (preferred for experiments): Set `ideas_tracker=default` or
`ideas_tracker=fast` in your Hydra command. The engine fires
`on_run_complete(storage)` in its `run()` method's `finally` block after
evolution completes. Hook errors are caught and logged — they never crash the
engine.

**CLI** (for re-running on existing data): Use when you want to re-extract
ideas from a run that's already in Redis, or from a CSV export. Useful for
debugging, re-processing, or running on archived data.

Both entry points call the same internal `_run_on_programs()` pipeline.

### Ideas Tracker Hydra Config Group

Located in `config/ideas_tracker/`. Selected via `ideas_tracker=<name>`.

```
config/ideas_tracker/
  none.yaml      →  NullPostRunHook (no-op, default)
  default.yaml   →  IdeaTracker with default LLM analyzer
  fast.yaml      →  IdeaTracker with fast embedding+DBSCAN analyzer
  true.yaml      →  backward compat alias for default.yaml
```

The default is set in `config/config.yaml`:
```yaml
defaults:
  - ideas_tracker: none
```

#### `config/ideas_tracker/none.yaml`
```yaml
_target_: gigaevo.evolution.engine.hooks.NullPostRunHook
```

#### `config/ideas_tracker/default.yaml`
```yaml
defaults:
  - _self_
  - override /memory/backend@_global_.memory.backend: local
  - override /llms@_global_.memory_llm: gemini_flash_openrouter

_target_: gigaevo.memory.ideas_tracker.ideas_tracker.IdeaTracker
llm: ${ref:memory_llm}
backend: ${ref:memory.backend}
analyzer_type: default
analyzer_max_concurrent_classifications: 8
description_rewriting: true
memory_write_enabled: true
memory_write_best_programs_percent: 5.0
fitness_higher_is_better: ${higher_is_better}
checkpoint_dir: ${checkpoint_dir}
redis_prefix: ${problem.name}
admitter: ${ref:memory.admitter}
```

The tracker shares the SAME `memory_llm` router and `memory.backend` factory
as the `memory=` read side — both are root-registered singletons resolved via
`${ref:...}`. With `memory=none` the `admitter` ref resolves to null
(`TieredAdmitter` fallback inside the tracker); with `memory=local` the
tracker rides the composed admitter (sign-based by default).

#### `config/ideas_tracker/fast.yaml`

Same structure as `default.yaml` but with:
- `analyzer_type: fast` — uses sentence embeddings + DBSCAN clustering
- `analyzer_fast_settings:` — embedding model, DBSCAN parameters, batch sizes

#### Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyzer_type` | str | `"default"` | `"default"` = LLM-based sequential analysis. `"fast"` = embedding+DBSCAN batched analysis. |
| `llm` | MultiModelRouter | `${ref:memory_llm}` | Shared analyzer LLM router (model, endpoint, credential, reasoning all live in `config/llms/gemini_flash_openrouter.yaml`) |
| `analyzer_max_concurrent_classifications` | int | `8` | Max concurrent classification LLM calls inside `ClassifyingAnalyzer` |
| `description_rewriting` | bool | `true` | Allow the LLM to rewrite idea descriptions |
| `memory_write_enabled` | bool | `true` | Write extracted ideas to the memory database |
| `memory_write_best_programs_percent` | float | `5.0` | Share of top programs (by fitness) the write pipeline converts into program cards |
| `fitness_higher_is_better` | bool | `${higher_is_better}` | Metric direction; gains are stored in "positive = improvement" space |
| `admitter` | MemoryAdmitter or null | from `memory=` group | Admission gate for `best_ideas.json` (null → `TieredAdmitter`) |
| `checkpoint_dir` | str or null | `null` | Directory for memory card storage. Defaults to `null` in `config/config.yaml`. **Not** resolved via Hydra output dir — must be set explicitly as a Hydra override (e.g. `checkpoint_dir=experiments/hover/memory/memory_bank`). The same path must be used in Phase A (write) and Phase B (read) so the memory bank persists between phases. |
| `redis_prefix` | str | `${problem.name}` | Redis key prefix for loading programs |

### CLI Reference

```
python -m gigaevo.memory.ideas_tracker.cli [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | str | `$OPENROUTER_API_KEY` | Analyzer LLM API key (required: flag or env var) |
| `--model` | str | `google/gemini-3-flash-preview` | Analyzer LLM model id |
| `--base-url` | str | `https://openrouter.ai/api/v1` | Analyzer LLM API endpoint |
| `--csv-path` | PATH | none | CSV exported by `tools/redis2pd.py`; when given, programs load from the CSV instead of Redis |
| `--higher-is-better` / `--no-higher-is-better` | bool | higher | Fitness direction of the analyzed run |
| `--checkpoint-dir` | PATH | none | Card-bank directory for the memory write pipeline (required unless `--no-memory-write`) |
| `--logs-dir` | PATH | `ideas_tracker/logs/` | Directory for session logs (timestamped subdir created) |
| `--memory-write` / `--no-memory-write` | bool | enabled | Toggle the final memory write pipeline |
| `--redis-host` | str | `localhost` | Redis host |
| `--redis-port` | int | `6379` | Redis port |
| `--redis-db` | int | `0` | Redis DB |
| `--redis-prefix` | str | `""` | Redis key prefix (usually matches `problem.name`) |
| `--redis-label` | str | none | Optional label for logging/debugging |

### CLI Examples

```bash
# Extract ideas from a Redis run (most common)
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --redis-db 3 \
  --redis-prefix "chains/hover/static_soft" \
  --checkpoint-dir experiments/hover/memory/memory_store \
  --memory-write

# Extract from a CSV export (offline analysis)
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --csv-path experiments/hover/memory/archives/M0/evolution_data.csv \
  --checkpoint-dir experiments/hover/memory/memory_store

# Dry run: extract ideas but don't write to memory DB
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --redis-db 3 \
  --redis-prefix "chains/hover/static_soft" \
  --no-memory-write

# Write logs to a specific directory
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --redis-db 3 \
  --redis-prefix "chains/hover/static_soft" \
  --logs-dir experiments/hover/memory/tracker_logs
```

### Pipeline Internals

The core pipeline runs the same sequence regardless of entry point:

```
1. Load programs
   │  PostRunHook: storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
   │  CLI/Redis:   RedisProgramStorage.get_all()
   │  CLI/CSV:     parse CSV rows → Program objects
   │
2. Filter programs
   │  Remove: root programs (no parents)
   │  Remove: fitness <= 0
   │  Remove: already-processed (tracked in programs_ids set)
   │
3. Compute injection-efficacy posteriors
   │  For each child with memory_selected_idea_ids:
   │    gain = child_fitness - parent-local baseline (sign-normalized
   │    so positive = improvement) → Beta-Binomial posterior per card
   │
4. Convert to ProgramRecords
   │  Extract: id, fitness, generation, parents, code
   │  Extract from metadata.mutation_output: insights, changes, archetype
   │
5. Run analyzer pipeline
   │  "default": sequential LLM classification (process_program per record)
   │  "fast":    batched embedding + DBSCAN clustering + async LLM refinement
   │
   │  For each program's improvements:
   │    Classify as: NEW idea | UPDATE existing | REWRITE existing
   │    Apply to active/inactive idea banks via RecordManager
   │
6. Attach efficacy statistics to idea banks
   │  Merge per-card posteriors into evolution_statistics.ALL
   │
7. Enrich ideas (postprocessing)
   │  For each idea in record bank:
   │    Generate: keywords, explanation summary, task description summary
   │
8. Log final state
   │  Write: idea banks, processed programs, evolutionary statistics
   │  Output: timestamped directory with JSON/YAML files
   │
9. Memory write pipeline (if enabled)
   │  Load cards from idea banks
   │  Harm-gate (admitter) + dedup, then ingest
   │  Write to memory backend (local disk or API)
```

### Analyzer Types

**Default analyzer** (`analyzer_type: default`):
- Sequential, one program at a time
- Uses the LLM to classify each improvement against existing idea banks
- The LLM sees: the improvement, all active ideas, all inactive ideas
- Decides: new idea, update to existing, or rewrite of existing
- Best for small runs (< 100 programs) where accuracy matters

**Fast analyzer** (`analyzer_type: fast`):
- Batched, processes all programs at once
- Step 1: Embed all improvements using a sentence transformer
- Step 2: Cluster similar improvements using DBSCAN
- Step 3: Use the LLM to refine clusters into idea cards
- Step 4: Import all cards into the record bank with forced dedup
- Best for large runs (100+ programs) where speed matters

### Memory Write Pipeline

When `memory_write_enabled=true`, after idea extraction completes:

1. The best ideas (from top `memory_write_best_programs_percent`% of programs)
   are selected from the idea banks
2. Per-idea efficacy statistics (Beta-Binomial injection posterior) are
   carried in each card's `evolution_statistics`
3. Cards are written to the memory backend:
   - **Local**: JSON files in `checkpoint_dir` with a search index
   - **API**: Posted to the memory API service via the configured namespace

The write pipeline receives its backend factory by injection: the PostRunHook
path gets the run's shared `memory.backend` singleton (the same factory the
read-side provider uses), and the ideas-tracker CLI constructs a
`LocalMemoryBackendFactory` directly.

### What a Memory Card Looks Like

Internally, a memory card is a structured object with these fields:

```python
{
    "id": "idea-abc-123",
    "description": "Sort evidence by relevance score before traversing the chain",
    "category": "retrieval",
    "keywords": ["sort", "relevance", "evidence", "chain"],
    "task_description_summary": "Multi-hop fact verification using evidence chains",
    "explanation": {
        "explanations": ["Sorting evidence before traversal ensures high-quality..."],
        "summary": "Pre-sort evidence to avoid low-quality chain hops",
    },
    "evolution_statistics": {
        "ALL": {
            "intro_events": 5,
            "IntroGain_best_median": 0.03,
            "IntroGain_best_adj_median": 0.012,
            "DownsideRate_best": 0.1,
            "posterior_a": 5.0,
            "posterior_b": 2.0,
            "p_help_lo20": 0.55,
            "efficacy_confident": True,
        }
    },
    "programs": ["prog-1", "prog-2"],       # programs that produced this idea
    "last_generation": 15,                    # last generation where idea was seen
    "strategy": "exploitation",               # mutation archetype
}
```

The `description` is the core idea. Everything else is metadata for search
ranking, deduplication, and efficacy-aware selection.

### Logs and Checkpoints

The Ideas Tracker writes detailed logs to a timestamped directory:

```
ideas_tracker/logs/2026-04-03_14-30-00/
  active_ideas.json        # Current active idea bank (final state)
  inactive_ideas.json      # Ideas moved to inactive bank
  programs_processed.json  # All ProgramRecord dicts
  evolution_stats.json     # Evolutionary statistics (origin analysis)
  init.json                # Initialization parameters (model, redis, etc.)
```

When running via CLI with `--logs-dir`, logs go into a timestamped subfolder
of the specified directory.

---

## The Memory Search (Read Phase)

When `memory=local` or `memory=legacy_api`, here's what happens on each program
evaluation:

1. **`MemoryContextStage`** calls `SelectorMemoryProvider.select_cards()`
2. The provider assembles a **`MemoryReadPipeline`** lazily on first call
   (retriever → shortlister → auctioneer → budgeter → renderer)
3. `LLMCardSelector` builds a query from the parent code, task description, and metrics
4. The query is sent to the memory backend (local `AmemGamMemory` or remote API)
5. The **GAM (Generative Agentic Memory) pipeline** runs:
   - Multiple retrieval tools search different indices (vector, keyword, etc.)
   - The selector LLM emits an ordered `final_decision` shortlist of card ids
6. The shortlist goes through a **Thompson auction** on each card's
   Beta-Binomial efficacy posterior, then a top-theta budget cap (`max_cards`)
7. Card text is rendered as numbered blocks headed `[card N] id=<card-id>`
   (selection-rank order), each with a mechanism description and — when the
   bank has confident evidence — a trailing `efficacy:` line
8. Card IDs are stored in program metadata for tracking

The GAM pipeline is configurable via the `memory/retriever` Hydra group
(`config/memory/retriever/gam.yaml`). The `allowed_tools` list controls which
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

Run evolution with `ideas_tracker=true` (or `ideas_tracker=default`). The
IdeaTracker fires as a PostRunHook after evolution completes and writes
memory cards to `checkpoint_dir`.

```bash
# Phase A: Run evolution with IdeaTracker enabled
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  ideas_tracker=true \
  checkpoint_dir=experiments/hover/memory/memory_bank \
  redis.db=3 \
  max_mutants=200
```

After the run completes, check the memory bank:
```bash
ls experiments/hover/memory/memory_bank/
```

**Alternative: Re-extract ideas from an existing run** (if the PostRunHook
didn't run, or you want to re-process):

```bash
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --redis-db 3 \
  --redis-prefix "chains/hover/full7_no_deep" \
  --checkpoint-dir experiments/hover/memory/memory_bank \
  --memory-write
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

# R3: treatment (memory enabled)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  memory=local \
  checkpoint_dir="$MEMORY_BANK" \
  redis.db=6

# R4: treatment (memory enabled)
python run.py \
  problem.name=chains/hover/full7_no_deep \
  pipeline=structural_metrics \
  evolution=steady_state \
  memory=local \
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
| `gigaevo/memory/provider.py` | `MemoryProvider` ABC, `NullMemoryProvider`, `SelectorMemoryProvider` |
| `config/memory/none.yaml` | Hydra config: NullMemoryProvider (default) |
| `config/memory/local.yaml` | Hydra config: SelectorMemoryProvider (local) |
| `config/memory/legacy_api.yaml` | Hydra config: SelectorMemoryProvider (legacy API backend) |

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
| `gigaevo/memory/backend_factory.py` | Lazy fail-fast backend factories (`LocalMemoryBackendFactory`, `LegacyApiMemoryBackendFactory`) |
| `gigaevo/memory/shared_memory/memory.py` | `AmemGamMemory` — local memory backend with GAM search |
| `config/memory/backend/` | Hydra configs for the backend factories (checkpoint dir, embedding model, memory LLM) |

### Shared Memory Module (`gigaevo/memory/shared_memory/`)

`AmemGamMemory` is the orchestrator; the rest are pluggable collaborators wired
via the `AgenticRuntime` DI container.

| File | Responsibility |
|------|---------------|
| `memory.py` | `AmemGamMemory` orchestrator — coordinates save / search / rebuild / delete |
| `memory_config.py` | Pydantic configs: `MemoryConfig`, `GamConfig`, `ApiConfig`, `CardUpdateDedupConfig` |
| `card_store.py` | Card dict + entity mappings + JSON index persistence |
| `note_sync.py` | Bridges cards to the A-MEM vector store (Chroma) |
| `api_sync.py` | Paginated fetch / full sync / remote search via concept API |
| `gam_search.py` | GAM `ResearchAgent` build + invalidate lifecycle |
| `card_dedup.py` | Vector scoring + LLM dedup decision + card merge |
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
| `gigaevo/memory/ideas_tracker/ideas_tracker.py` | `IdeaTracker(PostRunHook)` — core pipeline orchestrator |
| `gigaevo/memory/ideas_tracker/cli.py` | CLI entry point (`python -m gigaevo.memory.ideas_tracker.cli`) |
| `config/ideas_tracker/none.yaml` | Hydra config: NullPostRunHook (default) |
| `config/ideas_tracker/default.yaml` | Hydra config: IdeaTracker with default LLM analyzer |
| `config/ideas_tracker/fast.yaml` | Hydra config: IdeaTracker with fast embedding analyzer |
| `config/ideas_tracker/true.yaml` | Backward compat alias for `default.yaml` |
| `config/memory.yaml` | Unified memory config (backend + ideas_tracker sections) |

### Ideas Tracker Modules

| File | What it does |
|------|-------------|
| `analyzers.py` | LLM / embedding+DBSCAN idea classification |
| `idea_bank.py` | Active/inactive idea bank management |
| `models.py` | Data structures: `ProgramRecord`, banks, incoming ideas |
| `schemas.py` | Structured-output schemas for tracker LLM calls |
| `llm.py` | Tracker-side LLM service wiring |
| `redis_loader.py` | Load programs from `ProgramStorage` (Redis) |
| `csv_loader.py` | Load archived program dumps from CSV |
| `run_ideas_tracker_from_csv.py` | Offline tracker replay over a CSV dump |

### Origin Analysis (`utils/origin_analysis/`)

| File | What it does |
|------|-------------|
| `pipeline.py` | `analyse()` entry point: banks+programs JSON → `AnalysisResult` (lists of `IdeaStats`), CSV writer |
| `events.py` | Intro-event detection (child introduces idea absent from parents) |
| `aggregation.py` | Per idea×quartile `IdeaStats` rows (gains, posteriors via `injection_posterior`) |
| `loader.py` | Banks/programs JSON loading |
| `quartiles.py` | Generation → quartile bucketing |
| `siblings.py` | Sibling win-rate computation |
| `statistics.py` | NaN-aware medians/quantiles/rates (pure python) |
| `types.py` | Event/row type definitions |

### Tests

| File | What it covers |
|------|---------------|
| `tests/memory/test_provider.py` | Provider abstraction (null, selector, lazy init) |
| `tests/memory/test_memory_context_stage.py` | MemoryContextStage + MemoryMutationContext |
| `tests/memory/test_dag_memory_flow.py` | End-to-end DAG flow, composite context, auto-derivation |
| `tests/memory/test_ideas_tracker_pipeline.py` | IdeaTracker pipeline: records conversion, PostRunHook contract, program filtering, engine integration, Hydra composability, E2E |
| `tests/memory/test_data_components.py` | Data structures: RecordBank, RecordCardExtended, IncomingIdeas |
| `tests/integration/test_memory_e2e.py` | Full-loop E2E with real EvolutionEngine + fakeredis |

---

## FAQ

### Memory Read Phase

**Q: Does memory add latency?**
With `memory=none`, zero. With `memory=local`, search runs on local disk
(~50-200ms depending on card count and GAM tools). With `memory=legacy_api`, depends
on network latency. The search runs in parallel with other DAG stages
(insights, lineage), so the wall-clock impact is often hidden.

**Q: Can I use memory with the steady-state engine?**
Yes. This was the main reason for the refactor. The old implementation was
broken in steady-state because memory was hardcoded in the generational engine
loop. Now both engines use the same DAG pipeline.

**Q: What if the memory backend is unavailable?**
`MemoryReadPipeline` fails to an empty selection on every error path
(behaves like `NullMemoryProvider`). A warning is logged. The mutation proceeds
without memory guidance.

**Q: How many cards are selected per mutation?**
Configurable via `max_cards` in the Hydra config (default: 3). The memory
agent searches the database and returns the most relevant cards.

**Q: What's the difference between `memory=local` and `memory=legacy_api`?**
Both use `SelectorMemoryProvider`; they differ only in the composed backend
factory (`config/memory/backend/local.yaml` vs `legacy_api.yaml`). `local`
builds an on-disk `AmemGamMemory` bank; `legacy_api` builds the deprecated
remote-API client and additionally needs `base_url`/`namespace`.

**Q: How does the system decide which cards are "relevant"?**
The GAM pipeline sends the parent code + task description as a query, then
runs the configured retrieval tools (vector search, keyword search, etc.) to
find matching cards. The `allowed_tools` and `top_k_by_tool` settings in
`config/memory/retriever/gam.yaml` control which tools run and how many
results each returns.

### Ideas Tracker (Write Phase)

**Q: What's the difference between `ideas_tracker=default` and `ideas_tracker=fast`?**
`default` uses a sequential LLM-based analyzer that processes each program
one at a time. It classifies each improvement against the full bank of existing
ideas. Slower but more accurate for small runs.
`fast` uses sentence embeddings + DBSCAN clustering to batch-process all
programs at once, then uses the LLM to refine clusters into idea cards.
Much faster for large runs (100+ programs).

**Q: When does the IdeaTracker run?**
Two ways: (1) **Automatically**, as a PostRunHook after evolution completes
(`ideas_tracker=default` or `ideas_tracker=fast` in Hydra). The engine calls
`on_run_complete(storage)` in its `run()` finally block. (2) **Manually**, via
CLI (`python -m gigaevo.memory.ideas_tracker.cli`), typically to re-extract
ideas from a run that's already in Redis.

**Q: What happens if the IdeaTracker crashes during the PostRunHook?**
Nothing bad. The engine wraps the hook call in try/except — hook errors are
logged but never crash the engine. The evolution results are already saved.
You can re-run the tracker via CLI afterward.

**Q: Can I run the IdeaTracker on a run that's already finished?**
Yes, that's what the CLI is for. Point it at the Redis DB/prefix of the
completed run, and it extracts ideas just as the PostRunHook would have.
You can also pass `--csv-path` to run on a CSV export from `redis2pd.py`.

**Q: What's `best_programs_percent` and why is it 5%?**
The memory write pipeline only extracts ideas from the top N% of programs by
fitness. This filters out noise from poorly-performing mutations. 5% is the
default — for a run with 200 programs, only the top 10 contribute ideas.

**Q: How do I check what ideas were extracted?**
Look at the logs directory (default: `gigaevo/memory/ideas_tracker/logs/`).
The `active_ideas.json` file contains the final idea bank with all extracted
cards. Each card has `description`, `keywords`, `programs`, and
`evolution_statistics` fields.

**Q: Can I disable memory write but still extract ideas?**
Yes. Use `--no-memory-write` in the CLI, or set
`memory_write_enabled: false` in the Hydra config. The tracker will still
analyze programs and log ideas — it just won't write them to the memory backend.

### General

**Q: Can I add a new memory backend?**
Yes. Implement `MemoryProvider.select_cards()`, create a new
`config/memory/your_backend.yaml`, and use `memory=your_backend` on the
command line. The pipeline doesn't need any changes.

**Q: Where are cards stored on disk?**
At the path specified by `checkpoint_dir`. Inside that directory, the
`AmemGamMemory` backend stores cards as JSON files with an index for search.

**Q: Can two experiments share the same memory database?**
Yes, if they use the same `checkpoint_dir` and `namespace`. But be careful —
concurrent writes (from two ideas trackers) are not safe. Read-only sharing
during evolution is fine.
