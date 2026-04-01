# Prompt Co-Evolution

GigaEvo can co-evolve the mutation prompts alongside the programs they mutate.
Two paired GigaEvo processes run in lockstep: a **main run** evolves
task-specific programs (e.g. HotpotQA chains), and a **prompt run** evolves the
system prompt used by the mutation LLM. A Redis-based feedback loop connects
them so that prompts producing better mutations survive.

## Quick Start

Launch two processes — one for programs, one for prompts:

```bash
# 1. Main run — uses co-evolved prompts from DB 6
python run.py experiment=prompt_coevolution problem.name=heilbron \
    redis.db=4 prompt_fetcher.prompt_redis_db=6

# 2. Prompt run — evolves mutation prompts, reads outcomes from DB 4
python run.py problem.name=prompt_evolution pipeline=prompt_evolution \
    redis.db=6 main_redis_db=4 main_redis_prefix=heilbron
```

The main run fetches the current best mutation prompt from the prompt run's archive. The prompt run reads fitness outcomes from the main run to select for prompts that produce better mutations.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MAIN RUN  (e.g. DB 4)                     │
│                                                              │
│  1. Select elite programs from archive                       │
│  2. Fetch mutation prompt from prompt run's archive  ◄────┐  │
│     (GigaEvoArchivePromptFetcher, fitness-proportional)   │  │
│  3. Mutation LLM generates program variant                 │  │
│  4. Evaluate variant (validation set)                      │  │
│  5. Record outcome → prompt_stats:{prompt_id}  ──────┐    │  │
│     { trials: N, successes: M }                      │    │  │
│  6. Ingest into program archive                      │    │  │
│  7. Increment generation counter                     │    │  │
└──────────────────────────────────────────────────────│────┘  │
                                                       │       │
                                                       ▼       │
┌──────────────────────────────────────────────────────────┐   │
│                  PROMPT RUN  (e.g. DB 6)                 │   │
│                                                          │   │
│  0. MainRunSyncHook: block until main run gen advances   │   │
│  1. Select elite prompts from prompt archive             │   │
│  2. Mutation LLM generates prompt variant                │   │
│  3. PromptExecutionStage: exec entrypoint() → text       │   │
│  4. PromptFitnessStage: read prompt_stats from main DB ──┘   │
│     fitness = successes / trials                             │
│     (trials < min_trials → fitness = 0.01 default)           │
│  5. Ingest into MAP-Elites archive  ─────────────────────────┘
│     (2D grid: fitness x prompt_length)
│  6. Archive refresh: re-score ALL archived prompts
│     with latest stats from main run
└──────────────────────────────────────────────────────────┘
```

## Key Components

### Files

| File | Purpose |
|------|---------|
| `gigaevo/prompts/fetcher.py` | `GigaEvoArchivePromptFetcher` — reads prompt archive, selects prompt, records outcomes |
| `gigaevo/prompts/coevolution/stats.py` | `RedisPromptStatsProvider` — reads per-prompt stats from main run's Redis |
| `gigaevo/prompts/coevolution/stages.py` | `PromptExecutionStage` + `PromptFitnessStage` — DAG stages for prompt evaluation |
| `gigaevo/prompts/coevolution/sync.py` | `MainRunSyncHook` — blocks prompt run until main run advances |
| `gigaevo/prompts/coevolution/pipeline.py` | `PromptEvolutionPipelineBuilder` — assembles the prompt run DAG |
| `config/prompt_fetcher/coevolved.yaml` | Hydra config for `GigaEvoArchivePromptFetcher` |
| `config/pipeline/prompt_evolution.yaml` | Hydra config for the prompt run pipeline |
| `problems/prompt_evolution/` | Problem definition: seed programs, metrics, task description |

### Redis Key Schema

The feedback loop communicates through two Redis key patterns:

```
# Written by main run's fetcher (record_outcome)
{main_prefix}:prompt_stats:{prompt_id}  →  {"trials": N, "successes": M}

# Read by prompt run's PromptFitnessStage via RedisPromptStatsProvider
# prompt_id = sha256(prompt_text)[:16]  (prompt_text_to_id)
```

The `prompt_id` is derived from the prompt **text** (not the program UUID) via
`prompt_text_to_id()` so that both the write side (fetcher) and read side
(PromptFitnessStage) agree on the key.

### Prompt Selection

The main run does **not** always pick the best prompt. Instead,
`_refresh_champion()` uses **fitness-proportional sampling** with an epsilon
floor (0.01) for zero-fitness prompts. This ensures multiple prompts accumulate
trial data rather than starving all but the champion.

Selection is TTL-cached (default 30s) so not every mutation call hits Redis.

### Outcome Tracking

After each mutation, the main run's `LLMMutationOperator` calls
`fetcher.record_outcome()` with:

- `prompt_id` — which prompt was used
- `child_fitness` — fitness of the mutant
- `parent_fitness` — fitness of the best parent
- `outcome` — ACCEPTED, REJECTED_STRATEGY, or REJECTED_ACCEPTOR

A **success** is recorded when `child_fitness > parent_fitness`. Outcomes of
type REJECTED_ACCEPTOR are skipped (the mutant didn't produce reliable fitness).

### Synchronization

The prompt run is lightweight (no expensive validation) and runs ~5x faster
than the main run. Without throttling, it exhausts `max_generations` before the
main run accumulates any stats.

`MainRunSyncHook` is wired as a `pre_step_hook` on the prompt run's
`EvolutionEngine`. Before each generation step, it polls the main run's
`engine:total_generations` counter in Redis and blocks until it advances. This
keeps the two processes in lockstep.

### Bootstrap

New prompts have zero trials, so `PromptFitnessStage` assigns `fitness = 0.01`
(optimistic default) when `trials < min_trials` (default 3). This lets new
prompts enter the archive and start accumulating data. Once enough trials are
collected, real `success_rate` takes over.

The 2D MAP-Elites grid (fitness x prompt_length) ensures diversity — prompts of
different lengths coexist even if they have similar fitness.

## How to Launch

### Prerequisites

- Two or more available LLM endpoints (mutation LLMs)
- One or more chain LLM endpoints (for the main run's task evaluation)
- Redis server (single instance, multiple DBs)
- Dedicated Redis DB per process (4 DBs total for a paired experiment)

### Minimal Example (1 main + 1 prompt run)

```bash
PYTHON=$GIGAEVO_PYTHON
export PYTHONPATH=/path/to/gigaevo-core

# Main run (X1) — evolves HotpotQA chains, fetches prompts from DB 6
HOTPOTQA_CHAIN_URL="http://CHAIN_HOST:8001/v1" \
$PYTHON run.py \
    problem.name=chains/hotpotqa/static_f1_600 \
    pipeline=hotpotqa_asi \
    prompts=default \
    prompt_fetcher=coevolved \
    prompt_fetcher.prompt_redis_db=6 \
    redis.db=4 \
    max_generations=25 \
    llm_base_url="http://MUT_HOST_1:8777/v1"

# Prompt run (P1) — evolves mutation prompts, reads stats from DB 4
$PYTHON run.py \
    problem.name=prompt_evolution \
    pipeline=prompt_evolution \
    redis.db=6 \
    main_redis_db=4 \
    main_redis_prefix=chains/hotpotqa/static_f1_600 \
    max_generations=25 \
    llm_base_url="http://MUT_HOST_2:8777/v1"
```

### Full Paired Experiment (2 main + 2 prompt runs)

For a proper experiment with replication, launch two independent pairs
(X1+P1 and X2+P2) on separate Redis DBs:

| Process | Type | DB | Reads from | Writes to |
|---------|------|----|------------|-----------|
| X1 | Main run | 4 | P1 archive (DB 6) | prompt_stats in DB 4 |
| P1 | Prompt run | 6 | prompt_stats in DB 4 | P1 archive (DB 6) |
| X2 | Main run | 5 | P2 archive (DB 7) | prompt_stats in DB 5 |
| P2 | Prompt run | 7 | prompt_stats in DB 5 | P2 archive (DB 7) |

See `experiments/hotpotqa/prompt_coevolution/launch.sh` for a complete launch
script with preflight checks, config verification, and watchdog setup.

### Key Config Overrides

**Main run** (adds co-evolved prompt fetching to a normal run):

| Override | Purpose |
|----------|---------|
| `prompt_fetcher=coevolved` | Use `GigaEvoArchivePromptFetcher` instead of static files |
| `prompt_fetcher.prompt_redis_db=N` | Redis DB of the paired prompt run |

All other main run config is identical to a normal GigaEvo run.

**Prompt run** (dedicated pipeline):

| Override | Purpose |
|----------|---------|
| `problem.name=prompt_evolution` | Prompt evolution problem (seeds, metrics) |
| `pipeline=prompt_evolution` | Pipeline with `PromptFitnessStage` + `MainRunSyncHook` |
| `main_redis_db=N` | Redis DB of the paired main run (for stats reads) |
| `main_redis_prefix=...` | Key prefix of the main run |

### Seed Programs

The prompt run starts with 4 seed programs in
`problems/prompt_evolution/initial_programs/`:

| Seed | Strategy |
|------|----------|
| `generic.py` | General-purpose evolutionary optimization prompt |
| `hotpotqa.py` | HotpotQA-specific failure mode targeting |
| `minimal.py` | Minimal instruction prompt |
| `generalization.py` | Generalization-focused prompt |

Each seed is a Python file with an `entrypoint()` function that returns a
system prompt string. The prompt run mutates these to discover better
mutation instructions.

### Smoke Test

Before a full launch, run a 3-generation smoke test with one pair:

```bash
bash experiments/hotpotqa/prompt_coevolution/launch.sh --smoke-test
```

Verify:
1. `redis-cli -n 4 keys '*:prompt_stats:*'` returns at least 1 key
2. `redis-cli -n 6 hlen 'island_fitness_island:archive'` shows archive growth
3. Main run logs show `has_champion = True` in fetcher output

### Monitoring

Check generation progress:
```bash
redis-cli -n 4 hget "chains/hotpotqa/static_f1_600:run_state" "engine:total_generations"
redis-cli -n 6 hget "prompt_evolution:run_state" "engine:total_generations"
```

Check prompt archive size:
```bash
redis-cli -n 6 hlen "island_fitness_island:archive"
```

Check feedback loop (stats flowing):
```bash
redis-cli -n 4 keys "chains/hotpotqa/static_f1_600:prompt_stats:*"
```

## Extending to Other Tasks

The co-evolution system is task-agnostic. To use it with a different problem:

1. Create a task-specific prompt evolution problem directory (e.g.
   `problems/prompt_evolution_hover/`) with its own seed programs, metrics,
   and task description. Alternatively, reuse `problems/prompt_evolution/`.
2. Change the main run's `problem.name` and `pipeline` to your task
3. Add `prompt_fetcher=coevolved` and `prompt_fetcher.prompt_redis_db=N`
4. Set `main_redis_prefix` on the prompt run to match your main run's prefix
5. **Important**: Set `prompt_fetcher.prompt_prefix` on the main run to match
   the prompt run's Redis prefix. The default is `prompt_evolution`, which
   only works for HotpotQA. For HoVer, use `prompt_evolution_hover`.

The fitness signal (mutation success rate) is universal — it measures whether
the prompt helps produce better programs regardless of the downstream task.
