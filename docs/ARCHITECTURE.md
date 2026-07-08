# GigaEvo Architecture Guide for New Researchers

## Overview

This guide helps you understand GigaEvo's architecture from a bird's-eye view before diving into implementation details.

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Problem                             │
│  (validate.py + metrics.yaml + initial_programs/)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Main Evolution Loop                           │
│                    (run.py)                                      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Evolution    │───→│   Strategy   │───→│    Redis     │     │
│  │   Engine     │    │ (Islands)    │    │   Storage    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         ↓                    ↓                    ↓             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  DAG Runner  │    │ LLM Mutation │    │   Stages     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Program Lifecycle: The Most Critical Flow

Understanding this is **essential**. Every program goes through these states
(defined in `gigaevo/programs/program_state.py`):

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROGRAM STATE MACHINE                        │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  QUEUED  │ ← Program created or re-queued for JIT refresh
    └────┬─────┘
         │ DagRunner picks it up
         ↓
    ┌──────────┐
    │ RUNNING  │ ← DAG executing stages
    └────┬─────┘
         │ DAG completes
         ├─→ (success) ────────────────→ DONE
         └─→ (acceptor rejects) ────────→ DISCARDED

    ┌──────────┐
    │   DONE   │ ← Ingested + in archive (if accepted)
    └────┬─────┘
         │ ParentRefresher (JIT, when selected as a parent)
         └─→ Back to QUEUED (to refresh lineage-aware stages)
```

### Why This Matters

- **QUEUED** programs are picked up by `DagRunner`
- **RUNNING** programs are being evaluated
- **DONE** programs have metrics and live in the archive; they can be
  selected as parents
- **DISCARDED** is terminal (rejected by acceptor or invalid)
- Re-evaluation is **JIT** (just-in-time): a parent is re-queued only when
  `ParentRefresher` selects it for mutation — there is no global per-epoch
  archive refresh (see `gigaevo/evolution/engine/refresh.py`).

### The "Idle" State

The engine waits for "idle" (no QUEUED or RUNNING programs) only during
startup (to drain the initial seed population). After that, the
steady-state engine runs `dispatcher_loop` and `ingestor_loop`
concurrently — there is no per-generation barrier.

**Debugging tip**: If evolution is stuck, count programs by status set
(Redis schema: `{prefix}:status:{status}`):
```bash
redis-cli SCARD "gigaevo:status:queued"
redis-cli SCARD "gigaevo:status:running"
redis-cli SCARD "gigaevo:status:done"
```

## Evolution Flow: Steady-State Engine

`SteadyStateEvolutionEngine` is the only concrete engine. It runs two
concurrent loops (no per-generation barrier):

```
┌─────────────────────────────────────────────────────────────────┐
│                  STEADY-STATE EVOLUTION                         │
└─────────────────────────────────────────────────────────────────┘

dispatcher_loop                          ingestor_loop
─────────────────                        ──────────────
 1. acquire producer_sema                 1. poll Redis for DONE/DISCARDED
 2. (JIT) ParentRefresher.refresh()       2. apply Strategy.add / reject
    — re-queues parents that need it      3. release producer_sema + buffer_sema
 3. MutationOperator.mutate()             4. release ParentRefreshTicket
 4. store mutant (state: QUEUED)
 5. acquire buffer_sema
 6. emit BackpressureSample event

Stopping: stopper.should_stop(StopContext) is called once per dispatched
mutant. Built-in stoppers live in config/stopper/ (max_mutants,
wall_clock, fitness_plateau, ...).
```

Backpressure is governed by `max_in_flight` (sizes both producer and
buffer semaphores). See `gigaevo/evolution/engine/steady_state.py`.

### Why JIT Refresh

Parents are re-evaluated only when they are themselves selected as
parents (`ParentRefresher`). This refreshes lineage-aware stages
(insights, lineage) with the latest descendant statistics without paying
the cost of refreshing the entire archive each epoch.

**Performance note**: Cacheable stages (e.g. `ValidateCodeStage`,
`ComputeComplexityStage`) skip re-computation across refresh cycles via
the stage cache; non-cacheable stages (LineageStage,
MutationContextStage) re-run.

## The DAG Pipeline: How Programs Are Evaluated

```
┌─────────────────────────────────────────────────────────────────┐
│                        DAG EXECUTION                             │
└─────────────────────────────────────────────────────────────────┘

Program (state: QUEUED)
    ↓
DagRunner picks it up
    ↓
DAG built from blueprint (GuidedMutationPipelineBuilder by default)
    ↓
Stages execute in parallel (respecting dependencies)
    │
    ├─→ ValidateCodeStage (cacheable)
    │   ├─→ SUCCESS → Continue
    │   └─→ FAILED → Skip dependent stages
    │
    ├─→ CallProgramFunction (depends on ValidateCodeStage)
    │   ├─→ Runs user's entrypoint() function
    │   └─→ Captures output
    │
    ├─→ CallValidatorFunction (depends on CallProgramFunction)
    │   ├─→ Runs validate() from the problem
    │   └─→ Returns (metrics, artifact) tuple
    │
    ├─→ FetchMetrics / FetchArtifact
    │   └─→ Split validator output into metrics + optional artifact
    │
    ├─→ ComputeComplexityStage (independent, cacheable)
    │   └─→ Analyzes code structure
    │
    ├─→ MergeMetricsStage → EnsureMetricsStage
    │   └─→ Combines + sanitises all metrics
    │
    ├─→ ArchivePotentialGateStage (enabled in builder-backed guided pipelines)
    │   └─→ Skips expensive feedback/memory stages for archive-rejected programs
    │
    ├─→ DescendantProgramIds → IntraMemoryStage
    │   └─→ Builds a per-parent history summary
    │
    ├─→ MemoryContextStage (only pipeline=memory_guided)
    │   └─→ Selects external memory cards
    │
    ├─→ MutationSuggestionStage
    │   └─→ Converts parent history + optional cards into structured hints
    │
    └─→ MutationContextStage (non-cacheable)
        └─→ Formats context for future mutation
    ↓
All stages complete
    ↓
Program state: DONE
```

### Data Flow Example

```
CallProgramFunction.OutputModel = Box[np.ndarray]
    ↓ DataFlowEdge(source="CallProgramFunction",
                   destination="CallValidatorFunction",
                   input_name="payload")
CallValidatorFunction.InputsModel.payload: Box[np.ndarray]
```

**How to find input_name**: Look at the destination stage's `InputsModel` class.

### Stage Types by Cacheability

| Stage Type | Cacheable? | Why |
|------------|------------|-----|
| ValidateCodeStage | ✅ Yes | Code syntax doesn't change |
| CallProgramFunction | ✅ Yes | Deterministic execution |
| ComputeComplexityStage | ✅ Yes | Static code analysis |
| IntraMemoryStage | ✅ Yes | Parent-child id list is the cache key |
| MutationSuggestionStage | ✅ Yes | Fixed parent history/card context |
| MemoryContextStage | ✅ Yes | Fixed reader inputs/card bank snapshot |
| Legacy InsightsStage | ✅ Yes | Fixed LLM-based analysis in custom/legacy builders |
| MutationContextStage | ❌ No | Aggregates non-cacheable data |

## Multi-Island Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│                      MULTI-ISLAND SYSTEM                         │
└─────────────────────────────────────────────────────────────────┘

Island 1: "fitness_island"              Island 2: "simplicity_island"
┌──────────────────────────┐            ┌──────────────────────────┐
│ Behavior Space:          │            │ Behavior Space:          │
│  - fitness (0-100)       │            │  - fitness (0-100)       │
│  - validity (0-1)        │            │  - complexity (0-1000)   │
│                          │            │                          │
│ Archive: 20×5 = 100 cells│            │ Archive: 20×10 = 200 cells│
│                          │            │                          │
│ Selector: Maximize       │            │ Selector: Maximize       │
│           fitness        │            │  fitness / complexity    │
└────────────┬─────────────┘            └─────────────┬────────────┘
             │                                        │
             └────────────→ Migration ←───────────────┘
                         (every 50 gens)
```

### Program Metadata

Programs track their island membership:

```python
program.metadata = {
    "home_island": "fitness_island",      # Where created
    "current_island": "simplicity_island", # Where currently lives
    "iteration": 42,
    "mutation_context": "...",
}
```

### Migration Process

```
Every `migration_interval` mutants (formerly: every N generations):

1. Select Migrants
   ├─→ Island 1: Select top 5 by fitness
   └─→ Island 2: Select top 5 by fitness

2. Route Migrants
   ├─→ Island 1 migrants → Route to Island 2
   └─→ Island 2 migrants → Route to Island 1

3. Add to Destination
   ├─→ Try to add to destination archive
   └─→ Must improve a cell to be accepted

4. Remove from Source
   ├─→ If successfully added, remove from source
   └─→ If removal fails, rollback (remove from destination)
```

**Why rollback?** To maintain invariant: "No program exists in multiple islands simultaneously."

## Redis Data Model

Redis is the single source of truth. Understanding the key schema is essential for debugging.

```
┌─────────────────────────────────────────────────────────────────┐
│                       REDIS KEY SCHEMA                           │
└─────────────────────────────────────────────────────────────────┘

# Templates (defaults from gigaevo/database/redis/config.py)
{prefix}:program:{pid}        → Program object (JSON)
{prefix}:status:{status}      → SET of program IDs in that state
{prefix}:status_events        → STREAM of status-change events
{prefix}:archive              → HASH: cell → program_id (elite archive)
{prefix}:archive:reverse      → HASH: program_id → cell
{prefix}:ts                   → Atomic counter / timestamp
{prefix}:run_state            → HASH of run-level counters

# Multi-island runs use a distinct key_prefix per island.

# Example keys (default prefix "gigaevo", status set member is a program id):
gigaevo:program:a1b2c3d4-...
gigaevo:status:queued                   → SET membership
gigaevo:status:done                     → SET membership
gigaevo:archive                         → elite hash
```

### Debugging Commands

```bash
# Count programs in each state
for s in queued running done discarded; do
  echo -n "$s: "; redis-cli SCARD "gigaevo:status:$s"
done

# List program IDs currently QUEUED
redis-cli SMEMBERS "gigaevo:status:queued"

# Show archive size
redis-cli HLEN "gigaevo:archive"

# Get program details
redis-cli GET "gigaevo:program:a1b2c3d4-..." | jq .
```

## LLM Mutation Pipeline

The mutation process involves multiple stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MUTATION PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

1. MutationContextStage (runs on parents)
   ├─→ Formats metrics for LLM
   ├─→ Adds insights
   ├─→ Adds lineage info
   └─→ Stores in program.metadata[MUTATION_CONTEXT_METADATA_KEY]

2. Parent Selection
   ├─→ EvolutionEngine: Strategy.select_elites(N)
   ├─→ ParentSelector: Group elites into parent tuples
   └─→ Usually 1-2 parents per mutation

3. MutationAgent
   ├─→ Reads pre-formatted mutation context from parent metadata
   ├─→ Builds prompt:
   │   ├─→ System: task_description + metrics_description
   │   └─→ User: parent code + context
   ├─→ Calls LLM
   ├─→ Parses response (extracts code block or applies diff)
   └─→ Returns MutationSpec

4. Create Child Program
   ├─→ Program.from_mutation_spec()
   ├─→ Set lineage (parents, generation, mutation name)
   ├─→ Store in Redis (state: QUEUED)
   └─→ Update parent.lineage.children

5. Child Evaluation
   └─→ DAG pipeline runs (same as any program)
```

### Prompt Construction

Default prompt templates ship under `gigaevo/prompts/` (override per
experiment via the `prompts.dir` Hydra knob; see
`config/prompts/default.yaml`).

```
System Prompt (from gigaevo/prompts/mutation/system.txt):
    Task: {task_description}
    Metrics: {metrics_description}
    Instructions: ...

User Prompt (from gigaevo/prompts/mutation/user.txt):
    Mutate {count} parent programs:

    === Parent 1 ===
    ```python
    {parent.code}
    ```

    {parent.metadata[MUTATION_CONTEXT_METADATA_KEY]}
    ← This contains formatted metrics, insights, lineage
```

**Critical dependency**: If `MutationContextStage` is missing from your pipeline, mutation prompts will lack context and produce poor results.

### Structured diff mutation (non-Python genomes)

`StructuredDiffMutationOperator` (`gigaevo/evolution/mutation/structured_diff.py`) is an
alternative to `LLMMutationOperator` for genomes where free-form code generation is the
dominant failure mode. Instead of asking the LLM for the whole child genome, it constrains
the response to a per-call JSON schema of *representable changes* and applies the returned
diff mechanically. The genome-specific vocabulary lives behind the `AllowedChanges` contract
(`gigaevo/evolution/mutation/allowed_changes.py`): a subclass owns one genome family and
provides `build_schema` / `render_parents` / `apply` / `describe`. Two vocabularies exist:
`AllowedDagChanges` (`gigaevo/chains/dag_changes.py`, LLM-step CARL reasoning chains;
enable with `mutation=structured_diff_chains`) and `AllowedToolChainChanges`
(`problems/chains/tool_chain_diff.py`, chains mixing LLM and tool/retrieval steps for the
HoVer tasks; enable with `mutation=carl_with_retrieval_tools`). JSON-document genomes
evaluate via the `ParseJsonProgram` stage (`gigaevo/programs/stages/json_genome.py`),
wired by `program_format=json_document`.

## Configuration System (Hydra)

The config system uses Hydra with custom resolvers:

```yaml
# config/experiment/base.yaml
defaults:
  - /constants: base        # Load constants/base.yaml
  - /redis: default         # Load redis/default.yaml
  - /llm: single           # Load llm/single.yaml
  - /algorithm: single_island_no_distant_parents
  - /pipeline: guided
  - /program_format: python_source

# Hydra instantiation
dag_blueprint:
  _target_: gigaevo.runner.dag_blueprint.DAGBlueprint
  nodes:
    ValidateCode:
      _target_: gigaevo.programs.stages.validation.ValidateCodeStage
      _partial_: true       # Create factory, not instance
      timeout: 30.0

# Custom resolvers
${problem.dir}              # Resolves to problem directory path
${ref:program_storage}        # References another instantiated object
${metrics_context}          # Resolves to metrics context
```

### Understanding `_partial_`

```python
# _partial_: true
# Creates: lambda: ValidateCodeStage(timeout=30.0)
# Used when DAGBlueprint needs to create multiple instances

# _partial_: false (or omitted)
# Creates: ValidateCodeStage(timeout=30.0)
# Used for singletons
```

## Common Debugging Scenarios

### "Evolution is stuck"

**Check:**
1. Are there programs in QUEUED state waiting for DAG?
   ```bash
   redis-cli SCARD "gigaevo:status:queued"
   ```

2. Are there programs in RUNNING that never advance?
   ```bash
   redis-cli SCARD "gigaevo:status:running"
   ```

3. Check DagRunner metrics in logs:
   ```
   [DagRunner] active_count: 8, completed: 142
   ```

4. Look for stage timeouts or failures in logs

### "Island not accepting programs"

**Check:**
1. Do programs have required behavior metrics?
   ```python
   missing = set(island.behavior_space.behavior_keys) - program.metrics.keys()
   ```

2. Are bounds reasonable for metric values?
   ```python
   # All programs mapping to same cell?
   island.behavior_space.feature_bounds
   ```

3. Is archive selector too strict?

### "LLM generating invalid code"

**Check:**
1. Is `ValidateCode` stage in your pipeline?
2. Are error messages being passed to LLM in subsequent mutations?
3. Check prompt construction in logs:
   ```
   [MutationAgent] Built prompt with 2 parents (system: 1200 chars, user: 3400 chars)
   ```

4. Is `MutationContextStage` present and running?

### "Programs not being mutated"

**Check:**
1. Archive size: `await strategy.get_metrics()`
2. Elite selection: Are any elites being selected?
3. Parent selector: Is it producing valid parent tuples?
4. Stopper fired: Has the configured stopper (e.g. `max_mutants`,
   `wall_clock`, `fitness_plateau`) reported `stop=True`?

## Storage Abstraction

Program data is accessed via `ProgramStorage` (abstract interface in `gigaevo/database/program_storage.py`). The ABC defines CRUD operations, status set management, `key_prefix`, and async context manager support; read-only and write flags guard mutation methods.

Backends are selected via the `storage` Hydra group (`config/storage/redis.yaml` is the default; `storage=disk` switches to `DiskProgramStorage`, JSON files under the Hydra run dir). Each backend config defines the `program_storage` and `archive_storage_factory` nodes consumed by the evolution engine. Read-only CLI/analytics paths use `gigaevo/database/factory.py` — the single non-Hydra construction point — which exposes `build_readonly_redis_storage()` and `build_writable_redis_storage()` for analytics scripts and testing.

Archive storage (MAP-Elites elite cells) is accessed via `ArchiveStorageFactory` protocol, wired to Hydra and passed to strategies. Contract tests are backend-parametrized (`tests/database/storage_backends.py` registry) — every backend runs the same suite.

Metrics history follows the same pattern: the run's writer (Hydra `writer` node) feeds a metrics backend (`RedisMetricsBackend` or `DiskMetricsBackend`, both under `gigaevo/utils/trackers/backends/`), and readers — `live_frontier_compare` and the end-of-run frontier render — go through the `MetricsHistoryReader` protocol via `gigaevo.utils.trackers.get_default_history_reader()`, so monitors are storage-agnostic.

## Memory Subsystem

Cross-run memory (`gigaevo/memory/`) is an optional subsystem that distills each run's mutation diffs and top exemplars into reusable cards, then researches that bank and injects the most promising cards into later mutation prompts. It is built as four strict layers, imports flowing one way only (an AST test enforces it):

```
cards.py → events.py → storage/ → { read/ │ write/ } → provider.py / live_memory_hook.py
```

- **`cards.py`** — the one `Card` model (`kind ∈ {insight, program}`), `GainEvent`, `DecisionContext`. Frozen Pydantic, `extra="forbid"`, zero aliases.
- **`storage/`** — retrieval is an implementation detail of storage. `MemoryStore` answers `nearest(text, k, kind)` and `research(request)`; callers never see Chroma or the LangGraph research agent. `LocalMemoryStore` = card bank (atomic JSON persist under `checkpoint_dir`, cold-loaded from disk at construction) ∘ vector index ∘ research agent. Within a run the reader and writer share one store instance (`${ref:memory.store}`), so mid-run writes are visible to the reader with no reload.
- **`read/`** — `MemoryReader` facade: research shortlist → Beta-Binomial reputation → Thompson auction against a no-card baseline → budget → prompt render. Every stage fails to an empty selection, so a memory outage never sinks a mutation.
- **`write/`** — `MemoryWriter` facade: extract valid parent→child records → librarian LLM reconciles each diff into NEW/DUPLICATE/MERGE cards → author program exemplars → restamp gain events → configured eviction sweep → throttled background near-duplicate consolidation. `write/` never imports `read/`; eviction consumes `CardScorer` / `CardValueScorer` Protocols that `read/reputation` implements (dependency inversion).

Assembly is pure YAML — a flat `${ref:}` graph in `config/memory/full.yaml` (no Python assembler, no `_partial_`). Pipelines that read memory take `${ref:memory.provider}`; engines take `${ref:memory.write.post_run_hook}` as their memory finalizer. The `memory={none,reader,writer,full,static}` arms choose read/write components, while `memory/write={none,end_of_run,live}` chooses write cadence. Full guide: [`docs/memory.md`](memory.md); package internals: [`gigaevo/memory/README.md`](../gigaevo/memory/README.md); the memory-guided pipeline: [`docs/MEMORY_GUIDED_PIPELINE.md`](MEMORY_GUIDED_PIPELINE.md).

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point |
| `gigaevo/evolution/engine/core.py` | `EvolutionEngine` base class (shared helpers, snapshot, idle wait) |
| `gigaevo/evolution/engine/steady_state.py` | `SteadyStateEvolutionEngine` (only concrete engine) |
| `gigaevo/evolution/engine/dispatcher.py` | Dispatcher loop (mutate + enqueue) |
| `gigaevo/evolution/engine/ingestor.py` | Ingestor loop (poll DONE + acceptor) |
| `gigaevo/evolution/engine/refresh.py` | `ParentRefresher` (JIT parent refresh) |
| `gigaevo/evolution/engine/stopper.py` | Stoppers: `MaxMutantsStopper`, `WallClockStopper`, ... |
| `gigaevo/runner/dag_runner.py` | Picks up QUEUED programs, runs DAGs |
| `gigaevo/evolution/strategies/multi_island.py` | Multi-island strategy |
| `gigaevo/evolution/strategies/island.py` | Single island (archive) |
| `gigaevo/programs/dag/dag.py` | DAG execution engine |
| `gigaevo/programs/dag/automata.py` | Stage scheduling logic |
| `gigaevo/programs/program_state.py` | `ProgramState` enum + valid transitions |
| `gigaevo/database/redis_program_storage.py` | Redis interface |
| `gigaevo/database/state_manager.py` | Program state transitions |
| `gigaevo/database/redis/keys.py` | Redis key templates |
| `gigaevo/entrypoint/lineage_memory_pipeline.py` | `GuidedMutationPipelineBuilder` and `MemoryGuidedMutationPipelineBuilder` |
| `gigaevo/llm/agents/mutation.py` | LLM mutation agent |
| `gigaevo/evolution/mutation/structured_diff.py` | `StructuredDiffMutationOperator` (schema-constrained diff mutation) |
| `gigaevo/evolution/mutation/allowed_changes.py` | `AllowedChanges` contract (genome-family diff vocabularies) |
| `gigaevo/chains/dag_changes.py` | `AllowedDagChanges` (CARL chain positional-slot diffs, LLM steps) |
| `problems/chains/tool_chain_diff.py` | `AllowedToolChainChanges` (CARL chain diffs incl. tool/retrieval steps) |
| `gigaevo/programs/stages/json_genome.py` | `ParseJsonProgram` (JSON-document genomes) |
| `gigaevo/memory/` | Cross-run memory subsystem (cards → storage → read/write → provider); see [Memory Subsystem](#memory-subsystem) |

## Next Steps

1. **Quick Start**: Follow README.md to run your first evolution
2. **Create a Problem**: See `problems/heilbron/` as template
3. **Customize Evolution**: Modify `config/experiment/base.yaml`
4. **Add Custom Stages**: Read `docs/DAG_SYSTEM.md`
5. **Debug Issues**: Use Redis commands and logs

## Getting Help

- **DAG System**: See `docs/DAG_SYSTEM.md`
- **Evolution Strategies**: See `docs/EVOLUTION_STRATEGIES.md`
- **Configuration**: See `config/` directory structure
- **Tools**: See `../tools/README.md` for analysis utilities
