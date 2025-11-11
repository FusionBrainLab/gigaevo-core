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

Understanding this is **essential**. Every program goes through these states:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROGRAM STATE MACHINE                        │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  FRESH   │ ← Program created or refreshed
    └────┬─────┘
         │ DagRunner picks it up
         ↓
    ┌──────────────────────┐
    │ DAG_PROCESSING_      │ ← DAG executing stages
    │     STARTED          │
    └────┬─────────────────┘
         │ DAG completes successfully
         ↓
    ┌──────────────────────┐
    │ DAG_PROCESSING_      │ ← Ready for ingestion
    │    COMPLETED         │
    └────┬─────────────────┘
         │ EvolutionEngine processes it
         ├─→ (New program + accepted) ──→ EVOLVING
         ├─→ (Existing program) ────────→ EVOLVING (restored)
         └─→ (Rejected) ────────────────→ DISCARDED

    ┌──────────┐
    │ EVOLVING │ ← Active in island archive
    └────┬─────┘
         │ Refresh phase
         └─→ Back to FRESH (to update lineage-aware stages)
```

### Why This Matters

- **FRESH** programs are picked up by `DagRunner`
- **DAG_PROCESSING_STARTED** programs are being evaluated
- **DAG_PROCESSING_COMPLETED** programs await ingestion by `EvolutionEngine`
- **EVOLVING** programs are in the archive and can be selected as parents
- Programs cycle: `EVOLVING → FRESH → ... → EVOLVING` to update lineage info

### The "Idle" State

`EvolutionEngine` waits for "idle" (no FRESH or DAG_PROCESSING_STARTED programs) before:
- Selecting elites and creating mutants
- Ingesting completed programs
- Refreshing evolving programs

**Debugging tip**: If evolution is stuck, check Redis for programs in FRESH/DAG_PROCESSING_STARTED:
```bash
redis-cli KEYS "state:FRESH:*"
redis-cli KEYS "state:DAG_PROCESSING_STARTED:*"
```

## Evolution Generation Flow

One complete generation consists of 6 phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION PHASES                             │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Wait for Idle
    └─→ Block until no FRESH or DAG_PROCESSING_STARTED programs

Phase 2: Select & Mutate
    ├─→ Strategy.select_elites(N)
    ├─→ MutationOperator.mutate() → Create N mutants
    └─→ Store mutants in Redis (state: FRESH)

Phase 3: Wait for Mutant DAGs
    └─→ Block until mutants complete evaluation

Phase 4: Ingest Completed Programs
    ├─→ For each DAG_PROCESSING_COMPLETED program:
    │   ├─→ Already in archive? → Restore to EVOLVING
    │   ├─→ New + accepted by strategy? → Add, set EVOLVING
    │   └─→ Rejected? → Set DISCARDED

Phase 5: Refresh Evolving Programs
    ├─→ Get all programs in EVOLVING state
    ├─→ Flip them to FRESH
    └─→ Lineage-aware stages will re-run with updated family tree

Phase 6: Wait for Refresh DAGs
    └─→ Block until refresh completes

[Increment generation counter and repeat]
```

### Why Refresh Exists

Programs need to be refreshed because:
1. New children are added to their lineage
2. Descendant statistics change
3. LLM-based stages (insights, lineage analysis) should be recomputed with updated context

**Performance note**: This means stages run multiple times per program. Use `cacheable=True` for deterministic stages to avoid redundant computation.

## The DAG Pipeline: How Programs Are Evaluated

```
┌─────────────────────────────────────────────────────────────────┐
│                        DAG EXECUTION                             │
└─────────────────────────────────────────────────────────────────┘

Program (state: FRESH)
    ↓
DagRunner picks it up
    ↓
DAG built from blueprint
    ↓
Stages execute in parallel (respecting dependencies)
    │
    ├─→ ValidateCode (cacheable)
    │   ├─→ SUCCESS → Continue
    │   └─→ FAILED → Skip dependent stages
    │
    ├─→ ExecuteProgram (depends on ValidateCode)
    │   ├─→ Runs user's entrypoint() function
    │   └─→ Captures output
    │
    ├─→ ValidateOutput (depends on ExecuteProgram)
    │   ├─→ Runs validate() from problem
    │   └─→ Returns metrics (fitness, etc.)
    │
    ├─→ ComputeComplexity (independent, cacheable)
    │   └─→ Analyzes code structure
    │
    ├─→ MergeMetrics
    │   └─→ Combines all metrics
    │
    ├─→ InsightsStage (depends on metrics, non-cacheable)
    │   └─→ LLM generates insights
    │
    └─→ MutationContextStage (non-cacheable)
        └─→ Formats context for future mutation
    ↓
All stages complete
    ↓
Program state: DAG_PROCESSING_COMPLETED
```

### Data Flow Example

```
ExecuteProgram.OutputModel = Box[np.ndarray]
    ↓ DataFlowEdge(source="ExecuteProgram", dest="ValidateOutput", input_name="payload")
ValidateOutput.InputsModel.payload: Box[np.ndarray]
```

**How to find input_name**: Look at the destination stage's `InputsModel` class.

### Stage Types by Cacheability

| Stage Type | Cacheable? | Why |
|------------|------------|-----|
| ValidateCode | ✅ Yes | Code syntax doesn't change |
| ExecuteProgram | ✅ Yes | Deterministic execution |
| ComputeComplexity | ✅ Yes | Static code analysis |
| InsightsStage | ✅ Yes | Fixed LLM based analysis |
| LineageStage | ❌ No | Depends on evolving family tree |
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
Generation 50, 100, 150, ... (every migration_interval):

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

program:{program_id}                    → Program object (JSON)
state:{state_value}:{program_id}        → State index
counter                                 → Atomic counter for updates

# Island-specific
island_{island_id}:archive              → Hash: cell → program_id

# Example keys:
program:a1b2c3d4-...                    → Program data
state:FRESH:a1b2c3d4-...                → Index entry
state:EVOLVING:a1b2c3d4-...             → Index entry
island_fitness_island:archive           → Archive hash
```

### Debugging Commands

```bash
# Show all states
redis-cli --scan --pattern "state:*" | cut -d: -f2 | sort | uniq -c

# Show programs in FRESH state
redis-cli KEYS "state:FRESH:*"

# Show island archive size
redis-cli HLEN "island_fitness_island:archive"

# Get program details
redis-cli GET "program:a1b2c3d4-..." | jq .

# Show all island archives
redis-cli KEYS "island_*:archive"
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
   ├─→ Store in Redis (state: FRESH)
   └─→ Update parent.lineage.children

5. Child Evaluation
   └─→ DAG pipeline runs (same as any program)
```

### Prompt Construction

```
System Prompt (from prompts/mutation/system.txt):
    Task: {task_description}
    Metrics: {metrics_description}
    Instructions: ...

User Prompt (from prompts/mutation/user.txt):
    Mutate {count} parent programs:

    === Parent 1 ===
    ```python
    {parent.code}
    ```

    {parent.metadata[MUTATION_CONTEXT_METADATA_KEY]}
    ← This contains formatted metrics, insights, lineage
```

**Critical dependency**: If `MutationContextStage` is missing from your pipeline, mutation prompts will lack context and produce poor results.

## Configuration System (Hydra)

The config system uses Hydra with custom resolvers:

```yaml
# config/experiment/base.yaml
defaults:
  - /constants: base        # Load constants/base.yaml
  - /redis: default         # Load redis/default.yaml
  - /llm: single           # Load llm/single.yaml
  - /algorithm: single_island
  - /pipeline: auto

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
${ref:redis_storage}        # References another instantiated object
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
1. Are there programs in FRESH state waiting for DAG?
   ```bash
   redis-cli KEYS "state:FRESH:*" | wc -l
   ```

2. Are there programs in DAG_PROCESSING_STARTED?
   ```bash
   redis-cli KEYS "state:DAG_PROCESSING_STARTED:*" | wc -l
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
4. Generation limit: Has `max_generations` been reached?

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point |
| `gigaevo/evolution/engine/core.py` | Evolution generation loop |
| `gigaevo/runner/dag_runner.py` | Picks up FRESH programs, runs DAGs |
| `gigaevo/evolution/strategies/multi_island.py` | Multi-island strategy |
| `gigaevo/evolution/strategies/island.py` | Single island (archive) |
| `gigaevo/programs/dag/dag.py` | DAG execution engine |
| `gigaevo/programs/dag/automata.py` | Stage scheduling logic |
| `gigaevo/database/redis_program_storage.py` | Redis interface |
| `gigaevo/database/state_manager.py` | Program state transitions |
| `gigaevo/llm/agents/mutation.py` | LLM mutation agent |

## Next Steps

1. **Quick Start**: Follow README.md to run your first evolution
2. **Create a Problem**: See `problems/heilbron/` as template
3. **Customize Evolution**: Modify `config/experiment/base.yaml`
4. **Add Custom Stages**: Read `DAG_SYSTEM.md`
5. **Debug Issues**: Use Redis commands and logs

## Getting Help

- **DAG System**: See `DAG_SYSTEM.md`
- **Evolution Strategies**: See `EVOLUTION_STRATEGIES.md`
- **Configuration**: See `config/` directory structure
- **Tools**: See `../tools/README.md` for analysis utilities
