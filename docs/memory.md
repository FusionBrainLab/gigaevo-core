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
   - [Hydra Config Group (memory=...)](#hydra-config-group)
   - [SelectorMemoryProvider Parameters](#selectormemoryprovider-parameters)
   - [Backend Config (memory_backend.yaml)](#backend-config-memory_backendyaml)
7. [The Ideas Tracker (Write Phase)](#the-ideas-tracker-write-phase)
   - [What It Does](#what-it-does)
   - [Two Entry Points: PostRunHook vs CLI](#two-entry-points-postrunhook-vs-cli)
   - [Hydra Config Group (ideas_tracker=...)](#ideas-tracker-hydra-config-group)
   - [CLI Reference](#cli-reference)
   - [CLI Examples](#cli-examples)
   - [Pipeline Internals](#pipeline-internals)
   - [Analyzer Types](#analyzer-types)
   - [Memory Write Pipeline](#memory-write-pipeline)
   - [Usage Tracking](#usage-tracking)
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
python run.py memory=api   ...   # Memory from remote API service
```

One Hydra override. Everything else is automatic.

---

## What Memory Does

Without memory, the LLM mutation agent sees:
- The parent program code
- Metrics (fitness scores)
- Insights (what changed in recent mutations)
- Lineage (ancestor/descendant analysis)

With memory, it ALSO sees **memory cards** — short, actionable ideas extracted
from previous experiments:

```
## Memory Instructions

1. Sort evidence by relevance score before chain traversal
2. Filter low-confidence hops using a threshold of 0.3
3. Limit retrieval depth to 3 hops maximum
```

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
                                   "## Metrics\n...\n## Memory Instructions\n1. Sort evidence..."
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

When `memory=local` or `memory=api`:
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
| `SelectorMemoryProvider` | `memory=local` or `memory=api` | Queries memory DB via `MemorySelectorAgent` |

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

There are two layers of configuration:

1. **Hydra config group** (`config/memory/*.yaml`) — which provider to use
2. **Backend config** (`config/memory_backend.yaml`) — how the memory backend itself works

### Hydra Config Group

Located in `config/memory/`. Selected via `memory=<name>` on the command line.

```
config/memory/
  none.yaml    →  NullMemoryProvider (default)
  local.yaml   →  SelectorMemoryProvider (local backend)
  api.yaml     →  SelectorMemoryProvider (API backend)
```

The default is set in `config/config.yaml`:
```yaml
defaults:
  - memory: none
```

#### `config/memory/none.yaml`
```yaml
# @package _global_
memory_provider:
  _target_: gigaevo.memory.provider.NullMemoryProvider
```

#### `config/memory/local.yaml`
```yaml
# @package _global_
memory_provider:
  _target_: gigaevo.memory.provider.SelectorMemoryProvider
  max_cards: 3
  checkpoint_dir: ${checkpoint_dir}
  namespace: ${namespace}
```

#### `config/memory/api.yaml`
Same as `local.yaml`. The difference between local and API is controlled by
`config/memory_backend.yaml` → `api.use_api`, not by the Hydra config group.
(Both use `SelectorMemoryProvider`; the agent decides local vs API internally.)

### SelectorMemoryProvider Parameters

These are the constructor parameters of `SelectorMemoryProvider`, set in the
Hydra YAML:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_cards` | int | 3 | Maximum number of memory cards to return per mutation |
| `checkpoint_dir` | str or None | None | Local disk path where memory cards are cached. Overrides `memory_backend.yaml` → `paths.checkpoint_dir`. Pass via Hydra override: `checkpoint_dir=/path/to/store` |
| `namespace` | str or None | None | Isolation key for the memory API. Different experiments use different namespaces so their cards don't mix. Like a database schema. Overrides `memory_backend.yaml` → `api.namespace`. Pass via: `namespace=hover-memory-exp-1` |

Example command line:
```bash
python run.py \
  memory=local \
  checkpoint_dir=/workspace/experiments/hover/memory/memory_store \
  namespace=hover-memory-exp-1 \
  problem.name=chains/hover/static \
  ...
```

### Backend Config (`memory_backend.yaml`)

Located at `config/memory_backend.yaml`. This is NOT a Hydra config group — it's
loaded directly by `MemorySelectorAgent` via `runtime_config.py`. You rarely
need to edit this for normal experiments.

#### Full reference with explanations:

```yaml
# ═══════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════
paths:
  # Default local directory for memory card storage.
  # Overridden by SelectorMemoryProvider's checkpoint_dir param.
  checkpoint_dir: memory_usage_store/api_exp4

  # Path to ideas_tracker output banks (used by ideas_tracker CLI).
  banks_dir: ../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02

# ═══════════════════════════════════════════════
# API Connection
# ═══════════════════════════════════════════════
api:
  # Base URL of the memory API service (Concept API).
  base_url: http://localhost:8000

  # Default namespace for card isolation.
  # Overridden by SelectorMemoryProvider's namespace param.
  namespace: exp9

  # true = use remote API service for memory storage/search
  # false = use local disk only (no network calls)
  # This is the actual switch between local and API backends.
  use_api: false

  # Card version channel (latest, draft, etc.)
  channel: latest

  # Author tag attached to saved cards (null = anonymous).
  author: null

# ═══════════════════════════════════════════════
# Runtime Behavior
# ═══════════════════════════════════════════════
runtime:
  # Use an LLM to synthesize/summarize search results.
  # false = return raw card text (faster, no LLM cost).
  enable_llm_synthesis: false

  # Run A-MEM evolution flow when writing new cards.
  # Evolves card descriptions and merges similar cards.
  should_evolve: false

  # Use LLM to fill missing card metadata (keywords, etc.)
  fill_missing_fields_with_llm: false

  # Max cards returned per search query.
  search_limit: 5

  # Rebuild search index every N card writes.
  rebuild_interval: 30

  # Number of cards to sync per API page (pagination batch size).
  sync_batch_size: 100

  # Sync cards from API on memory backend initialization.
  sync_on_init: true

# ═══════════════════════════════════════════════
# GAM (Generative Agentic Memory) Search Pipeline
# ═══════════════════════════════════════════════
gam:
  # Enable BM25 keyword matching in addition to vector search.
  enable_bm25: false

  # GAM pipeline mode.
  # "default" = standard retrieval
  # "experimental" = multi-tool agentic retrieval
  pipeline_mode: experimental

  # Which retrieval tools the GAM agent can use.
  # Each tool searches a different index/representation:
  #   page_index       - page-level index search
  #   keyword          - BM25 keyword search
  #   vector           - dense vector search on card content
  #   vector_description            - search by description embedding
  #   vector_task_description       - search by task description embedding
  #   vector_explanation_summary    - search by explanation summary embedding
  #   vector_description_explanation_summary
  #   vector_description_task_description_summary
  allowed_tools:
    - page_index
    - vector

  # Maximum hits (top_k) per retrieval tool.
  top_k_by_tool:
    keyword: 5
    vector: 3
    vector_description: 3
    vector_task_description: 0
    vector_explanation_summary: 3
    vector_description_explanation_summary: 3
    vector_description_task_description_summary: 3
    page_index: 5

# ═══════════════════════════════════════════════
# Card Deduplication
# ═══════════════════════════════════════════════
card_update_dedup:
  # Use LLM to deduplicate/merge similar cards during writes.
  enabled: true
  retrieval:
    top_k_per_query: 10
    final_top_n: 10
    min_final_score: 0.05
    weights:
      description: 0.35
      explanation_summary: 0.2
      description_explanation_summary: 0.3
      description_task_description_summary: 0.15
  llm:
    max_retries: 2

# ═══════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════
models:
  # LLM for card enrichment and synthesis.
  openai_base_url: https://openrouter.ai/api/v1
  openrouter_model_name: google/gemini-3-flash-preview

  # Embedding model for A-MEM card indexing.
  amem_embedding_model_name: all-MiniLM-L6-v2

  # Dense retriever model for GAM search.
  gam_dense_retriever_model_name: BAAI/bge-m3

# ═══════════════════════════════════════════════
# Ideas Tracker (Write Phase)
# ═══════════════════════════════════════════════
ideas_tracker:
  # Max ideas per RecordList (batching for LLM analysis).
  list_max_ideas: 20

  # LLM model for idea extraction.
  analyzer:
    type: default       # "default" or "fast"
    model: google/gemini-3-flash-preview
    base_url: "https://openrouter.ai/api/v1"
    reasoning:
      effort: "minimal"

  # Redis connection for reading evolution run data.
  redis:
    redis_host: "localhost"
    redis_port: 6379
    redis_db: 1
    redis_prefix: "heilbron"
    label: ""

  # Statistics extraction from evolution runs.
  statistics:
    enabled: false
    mode: "top_k"       # "top_k", "top_fitness", "delta_fitness"

  # Write extracted ideas back into the memory database.
  memory_write_pipeline:
    enabled: true
    best_programs_percent: 5.0  # Extract ideas from top 5% programs

  # Track which memory cards are used and their fitness impact.
  usage_tracking:
    enabled: true
```

#### Which settings matter most?

For a typical experiment, you only care about:

| Setting | Where | Why it matters |
|---------|-------|---------------|
| `api.use_api` | `memory_backend.yaml` | Local-only vs remote API |
| `runtime.enable_llm_synthesis` | `memory_backend.yaml` | false = faster, cheaper search |
| `runtime.search_limit` | `memory_backend.yaml` | How many candidate cards to retrieve |
| `gam.pipeline_mode` | `memory_backend.yaml` | "default" = simple, "experimental" = multi-tool |
| `max_cards` | `config/memory/local.yaml` | How many cards to include in the prompt |
| `checkpoint_dir` | Command line override | Where cards are stored on disk |
| `namespace` | Command line override | Isolation between experiments |

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
# @package _global_
ideas_tracker:
  _target_: gigaevo.evolution.engine.hooks.NullPostRunHook
```

#### `config/ideas_tracker/default.yaml`
```yaml
# @package _global_
ideas_tracker:
  _target_: gigaevo.memory.ideas_tracker.ideas_tracker.IdeaTracker
  analyzer_type: default
  analyzer_model: google/gemini-3-flash-preview
  analyzer_base_url: "https://openrouter.ai/api/v1"
  analyzer_reasoning:
    effort: "minimal"
  list_max_ideas: 20
  postprocessing_type: default
  description_rewriting: true
  record_conversion_type: default
  memory_write_enabled: true
  memory_write_best_programs_percent: 5.0
  memory_usage_tracking_enabled: true
  checkpoint_dir: ${checkpoint_dir}
  namespace: ${namespace}
  redis_prefix: ${problem.name}
```

#### `config/ideas_tracker/fast.yaml`

Same structure as `default.yaml` but with:
- `analyzer_type: fast` — uses sentence embeddings + DBSCAN clustering
- `postprocessing_type: fast` — async postprocessing
- `record_conversion_type: fast` — async record conversion
- `analyzer_fast_settings:` — embedding model, DBSCAN parameters, batch sizes

#### Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyzer_type` | str | `"default"` | `"default"` = LLM-based sequential analysis. `"fast"` = embedding+DBSCAN batched analysis. |
| `analyzer_model` | str | `"google/gemini-3-flash-preview"` | LLM model for idea classification and enrichment |
| `analyzer_base_url` | str | `"https://openrouter.ai/api/v1"` | LLM API endpoint |
| `analyzer_reasoning` | dict | `{effort: "minimal"}` | Reasoning config passed to the LLM |
| `list_max_ideas` | int | `20` | Maximum ideas per RecordList batch |
| `postprocessing_type` | str | `"default"` | `"default"` = sync enrichment. `"fast"` = async enrichment. |
| `description_rewriting` | bool | `true` | Allow the LLM to rewrite idea descriptions |
| `record_conversion_type` | str | `"default"` | `"default"` = sync conversion. `"fast"` = async conversion. |
| `memory_write_enabled` | bool | `true` | Write extracted ideas to the memory database |
| `memory_write_best_programs_percent` | float | `5.0` | Only extract ideas from the top N% of programs by fitness |
| `memory_usage_tracking_enabled` | bool | `true` | Track fitness deltas for each card that was used |
| `checkpoint_dir` | str or null | `null` | Directory for memory card storage. Defaults to `null` in `config/config.yaml`. **Not** resolved via Hydra output dir — must be set explicitly as a Hydra override (e.g. `checkpoint_dir=experiments/hover/memory/memory_bank`). When `null`, falls back to `memory_backend.yaml` → `paths.checkpoint_dir`. The same path must be used in Phase A (write) and Phase B (read) so the memory bank persists between phases. |
| `namespace` | str | `${namespace}` | Isolation key for the memory API |
| `redis_prefix` | str | `${problem.name}` | Redis key prefix for loading programs |

### CLI Reference

```
python -m gigaevo.memory.ideas_tracker.cli [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | `redis` or `csv` | `redis` | Where to load programs from |
| `--csv-path` | PATH | (required if `--source csv`) | Path to CSV exported by `tools/redis2pd.py` |
| `--config-path` | PATH | `config/memory.yaml` | YAML config (full memory config or tracker-only section) |
| `--checkpoint-dir` | PATH | from config | Override `paths.checkpoint_dir` for memory write output |
| `--logs-dir` | PATH | `ideas_tracker/logs/` | Directory for session logs (timestamped subdir created) |
| `--memory-write` / `--no-memory-write` | bool | from config | Override `memory_write_pipeline.enabled` |
| `--redis-host` | str | from config | Redis host override |
| `--redis-port` | int | from config (6379) | Redis port override |
| `--redis-db` | int | from config | Redis DB override |
| `--redis-prefix` | str | from config | Redis key prefix (usually matches `problem.name`) |
| `--redis-label` | str | from config | Optional label for logging/debugging |

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
  --source csv \
  --csv-path experiments/hover/memory/archives/M0/evolution_data.csv \
  --checkpoint-dir experiments/hover/memory/memory_store

# Use custom config file
PYTHONPATH=. python -m gigaevo.memory.ideas_tracker.cli \
  --config-path experiments/hover/memory/custom_memory.yaml \
  --redis-db 3 \
  --redis-prefix "chains/hover/static_soft"

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
3. Build memory usage updates (if usage tracking enabled)
   │  For each child with memory_selected_idea_ids:
   │    delta = child_fitness - max(parent_fitnesses)
   │    Record delta per card per task
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
6. Apply memory usage updates to idea banks
   │  Merge fitness deltas into each card's usage statistics
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
   │  Apply usage updates
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
2. Usage statistics are merged into each card (if tracking is enabled)
3. Cards are written to the memory backend:
   - **Local**: JSON files in `checkpoint_dir` with a search index
   - **API**: Posted to the memory API service via the configured namespace

The write pipeline uses `EVO_MEMORY_CONFIG_PATH` to find backend configuration.
The CLI sets this env var automatically; the PostRunHook path inherits it from
the run's environment.

### Usage Tracking

When `memory_usage_tracking_enabled=true`, the tracker computes fitness deltas
for every memory card that was used during evolution:

```
For each child program with memory_selected_idea_ids:
  parent_fitness = max(fitness of all parents)
  delta = child_fitness - parent_fitness

  For each card_id in memory_selected_idea_ids:
    Record: (card_id, task_summary, delta)
```

These deltas are aggregated per card per task, producing:
- `total_used` — how many times the card was used
- `median_delta_fitness` — median fitness delta when used
- `fitness_delta_per_use` — full list of deltas

This data is stored in the card's `usage` field and used to rank cards in
future searches (cards that consistently improve fitness rank higher).

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
    "usage": {
        "used": {
            "entries": [
                {
                    "task_description_summary": "HoVer fact verification",
                    "used_count": 5,
                    "fitness_delta_per_use": [0.03, -0.01, 0.05, 0.02, 0.04],
                    "median_delta_fitness": 0.03,
                }
            ],
            "total": {"total_used": 5, "median_delta_fitness": 0.03},
        }
    },
    "programs": ["prog-1", "prog-2"],       # programs that produced this idea
    "last_generation": 15,                    # last generation where idea was seen
    "strategy": "exploitation",               # mutation archetype
}
```

The `description` is the core idea. Everything else is metadata for search
ranking, deduplication, and usage tracking.

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

When `memory=local` or `memory=api`, here's what happens on each program
evaluation:

1. **`MemoryContextStage`** calls `SelectorMemoryProvider.select_cards()`
2. The provider delegates to **`MemorySelectorAgent`** (created lazily on first call)
3. The agent builds a query from the parent code, task description, and metrics
4. The query is sent to the memory backend (local `AmemGamMemory` or remote API)
5. The **GAM (Generative Agentic Memory) pipeline** runs:
   - Multiple retrieval tools search different indices (vector, keyword, etc.)
   - Results are ranked and deduplicated
   - The top-N cards are selected
6. Card text is returned as a numbered list
7. Card IDs are stored in program metadata for tracking

The GAM pipeline is configurable via `memory_backend.yaml` → `gam.*` settings.
The `allowed_tools` list controls which retrieval strategies are used.

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
  max_generations=25 \
  max_mutations_per_generation=8 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  llm_base_url="http://localhost:4000/v1"
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
PYTHONPATH=. python tools/status.py \
  --run "chains/hover/full7_no_deep@4:R1" \
  --run "chains/hover/full7_no_deep@5:R2" \
  --run "chains/hover/full7_no_deep@6:R3" \
  --run "chains/hover/full7_no_deep@7:R4"

# Compare fitness trajectories
PYTHONPATH=. python tools/comparison.py \
  --run "chains/hover/full7_no_deep@4:control-1" \
  --run "chains/hover/full7_no_deep@5:control-2" \
  --run "chains/hover/full7_no_deep@6:memory-1" \
  --run "chains/hover/full7_no_deep@7:memory-2" \
  --output-folder experiments/hover/memory/plots/

# Check memory usage in treatment runs
PYTHONPATH=. python tools/top_programs.py \
  --run "chains/hover/full7_no_deep@6:memory-1" -n 5 --code
```

---

## Key Files

### Provider Layer (Hydra-injected, Read Phase)

| File | What it does |
|------|-------------|
| `gigaevo/memory/provider.py` | `MemoryProvider` ABC, `NullMemoryProvider`, `SelectorMemoryProvider` |
| `config/memory/none.yaml` | Hydra config: NullMemoryProvider (default) |
| `config/memory/local.yaml` | Hydra config: SelectorMemoryProvider (local) |
| `config/memory/api.yaml` | Hydra config: SelectorMemoryProvider (API) |

### DAG Pipeline (Read Phase)

| File | What it does |
|------|-------------|
| `gigaevo/programs/stages/memory_context.py` | `MemoryContextStage` — calls provider, returns card text |
| `gigaevo/evolution/mutation/context.py` | `MemoryMutationContext` — wraps cards for mutation prompt |
| `gigaevo/programs/stages/mutation_context.py` | `MutationContextStage` — composes all context types |
| `gigaevo/entrypoint/default_pipelines.py` | Wires MemoryContextStage into all pipelines |
| `gigaevo/evolution/engine/mutation.py` | Auto-derives `memory_used` from parent metadata |

### Memory Backend

| File | What it does |
|------|-------------|
| `gigaevo/llm/agents/memory_selector.py` | `MemorySelectorAgent` — builds queries, parses results |
| `gigaevo/memory/shared_memory/memory.py` | `AmemGamMemory` — local memory backend with GAM search |
| `gigaevo/memory/runtime_config.py` | Loads `memory_backend.yaml` settings |
| `config/memory_backend.yaml` | All backend settings (API, GAM, models, etc.) |

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

### Ideas Tracker Components

| File | What it does |
|------|-------------|
| `components/analyzer.py` | `IdeaAnalyzer` — LLM-based sequential idea classification |
| `components/analyzer_f.py` | `IdeaAnalyzerFast` — embedding+DBSCAN batched classification |
| `components/data_components.py` | Data structures: `RecordCardExtended`, `RecordBank`, `IncomingIdeas`, `ProgramRecord` |
| `components/records_manager.py` | `RecordManager` — active/inactive idea bank management |
| `components/memory_pipeline.py` | Memory write pipeline: banks → memory backend |
| `components/postprocessing.py` | Enrichment: keywords, explanation summaries |
| `components/statistics.py` | Evolutionary statistics (origin analysis) |
| `components/summary.py` | Task description summarization via LLM |

### Ideas Tracker Utilities

| File | What it does |
|------|-------------|
| `utils/cfg_loader.py` | Config loading from YAML / `EVO_MEMORY_CONFIG_PATH` |
| `utils/dataframe_loader.py` | Load programs from Redis/CSV into DataFrames |
| `utils/records_converter.py` | DataFrame rows → `ProgramRecord` conversion |
| `utils/helpers.py` | `build_memory_usage_updates()`, `sort_ideas()`, usage payload builders |
| `utils/it_logger.py` | Timestamped session logging for ideas tracker |
| `utils/task_description_loader.py` | Load task description from Redis problem dir |

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
(~50-200ms depending on card count and GAM tools). With `memory=api`, depends
on network latency. The search runs in parallel with other DAG stages
(insights, lineage), so the wall-clock impact is often hidden.

**Q: Can I use memory with the steady-state engine?**
Yes. This was the main reason for the refactor. The old implementation was
broken in steady-state because memory was hardcoded in the generational engine
loop. Now both engines use the same DAG pipeline.

**Q: What if the memory backend is unavailable?**
`MemorySelectorAgent` catches backend errors and returns an empty selection
(behaves like `NullMemoryProvider`). A warning is logged. The mutation proceeds
without memory guidance.

**Q: How many cards are selected per mutation?**
Configurable via `max_cards` in the Hydra config (default: 3). The memory
agent searches the database and returns the most relevant cards.

**Q: What's the difference between `memory=local` and `memory=api`?**
Both use `SelectorMemoryProvider`. The actual backend switch (`use_api`) is in
`memory_backend.yaml`. `local` is for experiments where you pre-populate cards
on disk; `api` is for when you have a running memory API service. In practice,
both configs are identical — the distinction is cosmetic for experiment clarity.

**Q: How does the system decide which cards are "relevant"?**
The GAM pipeline sends the parent code + task description as a query, then
runs the configured retrieval tools (vector search, keyword search, etc.) to
find matching cards. The `gam.allowed_tools` and `gam.top_k_by_tool` settings
control which tools run and how many results each returns.

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
You can also use `--source csv` to run on a CSV export from `redis2pd.py`.

**Q: What's `best_programs_percent` and why is it 5%?**
The memory write pipeline only extracts ideas from the top N% of programs by
fitness. This filters out noise from poorly-performing mutations. 5% is the
default — for a run with 200 programs, only the top 10 contribute ideas.

**Q: How do I check what ideas were extracted?**
Look at the logs directory (default: `gigaevo/memory/ideas_tracker/logs/`).
The `active_ideas.json` file contains the final idea bank with all extracted
cards. Each card has `description`, `keywords`, `programs`, and `usage` fields.

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
