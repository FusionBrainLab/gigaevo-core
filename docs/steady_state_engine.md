# Steady-State Evolution Engine

The `SteadyStateEvolutionEngine` eliminates the generational barrier by running
mutation and evaluation as two concurrent async loops. Instead of producing N
mutants → waiting for ALL N DAGs → ingesting → repeating, mutants are produced
and ingested continuously.

## Quick Start

```bash
# Using the experiment preset (recommended)
python run.py experiment=steady_state problem.name=heilbron

# Or as a single Hydra override on any existing setup
python run.py evolution=steady_state problem.name=heilbron

# Combined with migration bus for maximum throughput
python run.py experiment=steady_state_bus problem.name=heilbron redis.db=0
```

Works with any problem — just change `problem.name`. All other config (strategy, pipeline, LLM) stays the same.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  SteadyStateEvolutionEngine              │
│                                                         │
│  ┌──────────────────┐       ┌──────────────────┐       │
│  │  Mutation Loop    │       │  Ingestion Loop   │       │
│  │  (producer)       │       │  (consumer)        │       │
│  │                   │       │                    │       │
│  │  select elites    │  ←──  │  poll DONE progs   │       │
│  │  LLM → 1 mutant  │ sema  │  ingest → strategy │       │
│  │  persist QUEUED   │  ──→  │  release slot      │       │
│  └──────────────────┘       └──────────────────┘       │
│           │                          │                   │
│           └────── Epoch Refresh ─────┘                   │
│                  (every K programs)                       │
│           pause mutation → drain in-flight →             │
│           refresh archive → bump epoch → resume          │
└─────────────────────────────────────────────────────────┘
```

**Mutation loop** selects elites, calls the LLM to produce one mutant at a time,
and persists it as QUEUED. DagRunner picks it up immediately.

**Ingestion loop** polls for DONE programs, ingests each one (accept/reject into
strategy), and releases a semaphore slot so the mutation loop can proceed.

**Backpressure**: An `asyncio.Semaphore(max_in_flight)` caps the number of
programs in the pipeline (produced but not yet ingested). The mutation loop
blocks when all slots are taken. As soon as one DAG finishes and its program is
ingested, a slot opens and the next mutation starts.

**Epoch refresh** is the only synchronization point. It pauses mutation, drains
all in-flight programs, refreshes the archive (so NO_CACHE stages see a
consistent population snapshot), and increments `total_generations`. Triggers
every `max_mutations_per_generation` processed programs.

## Configuration

The only new config knob is `max_in_flight`:

```yaml
# config/evolution/steady_state.yaml
engine_config:
  max_in_flight: 8       # max programs in the mutation→evaluation pipeline
```

All other fields are inherited from the base `EngineConfig`:

| Field | Meaning in Steady-State |
|-------|------------------------|
| `max_in_flight` | **(new)** Max concurrent mutants between "produced" and "ingested" |
| `max_mutations_per_generation` | Epoch size — trigger epoch refresh after this many programs processed |
| `max_generations` | Max epochs (None = unlimited) |
| `max_elites_per_generation` | Passed to `select_elites()` each call |
| `loop_interval` | Ingestion polling interval (seconds) |

### Tuning `max_in_flight`

- **Default: 8** — good for most workloads
- **Higher** (16-32): more pipeline depth, better utilization when DAG evaluation
  times vary widely. Risk: more wasted work if the archive changes significantly
  between mutation and ingestion.
- **Lower** (2-4): tighter feedback loop, each mutant uses the most recent
  archive. Risk: pipeline bubbles if LLM calls are slow.
- **1**: serial mode — equivalent to producing and evaluating one mutant at a
  time (useful for debugging).

## Generation Semantics

| Concept | Generational Engine | Steady-State Engine |
|---------|-------------------|-------------------|
| `total_generations` (Redis) | Step count | Epoch count |
| `metadata["iteration"]` | Step when created | Epoch when created |
| `Program.lineage.generation` | Lineage depth | Unchanged |
| `max_generations` | Max steps | Max epochs |
| Epoch / generation size | `max_mutations_per_generation` | Same field, same meaning |

Status tools (`tools/status.py`, watchdog) work unchanged.

## Why It's Faster

```
GENERATIONAL (current):
  t=0:     [LLM: produce 8 mutants ~3min]
  t=3min:  [Wait for ALL 8 DAGs ~8min each]
  t=11min: [Ingest + refresh ~15s]
  → 8 mutants in 11 minutes = 0.7 mutants/min

STEADY-STATE:
  t=0:     mut1 produced → DAG starts
  t=22s:   mut2 produced → DAG starts
  ...
  t=3min:  mut8 produced → DAG starts | mut9 BLOCKED (sema full)
  t=8min:  mut1 DAG done → ingest → slot frees → mut9 produced
  t=8m22s: mut2 DAG done → ingest → slot frees → mut10 produced
  → continuous pipeline, ~1 mutant/min throughput
```

The improvement comes from eliminating the idle gap: the LLM starts the next
mutant as soon as a slot opens, rather than waiting for all DAGs to finish.

## Epoch Refresh Details

An epoch refresh runs these steps:

1. **Pause mutation** — clear the mutation gate (Event)
2. **Drain in-flight** — wait for all in-flight programs to finish DAG
   evaluation, then ingest their results (600s timeout)
3. **Snapshot bump** — full cache invalidation
4. **Refresh archive** — transition DONE→QUEUED for lineage/insights stages
5. **Wait for refresh DAGs** — reindex archive
6. **Increment epoch** — bump `total_generations`, save to Redis
7. **Log summary** — archive size, delta, best metrics, ETA, stagnation
8. **Resume mutation** — set the mutation gate

The mutation gate uses try/finally to always reopen, even on errors.

## Safety Features

- **Backpressure guarantee**: At most `max_in_flight` mutants in the pipeline at
  any instant. The mutation loop physically cannot outpace ingestion.
- **Scoped state checks**: `_drain_in_flight` uses `mget` on in-flight IDs only,
  not global status queries. This prevents hangs when archive refresh programs
  are also QUEUED/RUNNING.
- **TOCTOU-safe ingestion**: `_ingest_batch` returns `(count, handled_ids)` so
  only confirmed-DONE programs get their semaphore slots released.
- **Ghost detection**: `_sweep_discarded` catches programs that DagRunner timed
  out or discarded, releasing their slots to prevent permanent backpressure.
- **Drain timeout**: 600s for normal epoch refresh, 120s for final shutdown.
  Force-releases remaining slots on timeout.
- **Lineage fault isolation**: If parent lineage update fails after program
  persistence, the program ID is still returned and tracked (non-critical
  failure doesn't orphan programs).

## Compatibility

- **Hydra config**: Drop-in replacement — just add `evolution=steady_state`
- **Status tools**: `tools/status.py`, watchdog, `tools/comparison.py` all work unchanged
- **DagRunner**: No changes — communicates via Redis state transitions
- **Strategies**: All strategies (MAP-Elites, islands, etc.) work unchanged
- **Not compatible with `BusedEvolutionEngine`** — both override `run()`.
  Future: extract bus as a mixin.
