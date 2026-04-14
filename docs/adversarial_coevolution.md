# Adversarial Co-Evolution Guide

How to set up, run, and debug two-population adversarial co-evolution experiments in GigaEvo.

---

## What Is It

Two MAP-Elites populations evolve in parallel. Each population's fitness depends on how well it performs against the other population's archive. This creates an arms race that drives both populations toward better solutions.

**GAN analogy**: Pop A = Generator (produces solutions), Pop B = Discriminator (finds flaws). The adversarial pressure forces Pop A toward genuine local optima, not just solutions that look good in isolation.

**When to use**: When single-population evolution stagnates at local optima, or when you want to co-evolve complementary capabilities (optimizer vs landscape, constructor vs improver, attack vs defense).

---

## Architecture

### Pipeline: `pipeline=adversarial_coevo`

Extends the standard pipeline with one new stage and a sync hook:

```
Standard Pipeline                    Adversarial Pipeline
─────────────────                    ────────────────────
MutationStage                        MutationStage
ValidateCodeStage                    ValidateCodeStage
CallProgramFunction                  CallProgramFunction
                                     FetchOpponentResultsStage  ← NEW
CallValidatorFunction(validate.py)   CallValidatorFunction(evaluate.py)  ← MODIFIED
FetchMetrics                         FetchMetrics
EnsureMetricsStage                   EnsureMetricsStage
CollectorStage                       CollectorStage
                                     MainRunSyncHook  ← NEW (pre-step hook)
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `AdversarialPipelineBuilder` | `gigaevo/adversarial/pipeline.py` | Extends `DefaultPipelineBuilder`: replaces `validate.py` with `evaluate.py`, adds `FetchOpponentResultsStage`, wires opponent results as `context` to `CallValidatorFunction` |
| `FetchOpponentResultsStage` | `gigaevo/adversarial/stages.py` | Reads N opponent codes from archive via `OpponentArchiveProvider`, executes each `entrypoint()` in parallel subprocesses (via `run_exec_runner`), returns results as context |
| `RedisOpponentArchiveProvider` | `gigaevo/adversarial/opponent_provider.py` | Reads opponent programs from the other population's MAP-Elites archive in Redis. Fitness-proportional sampling. Cached (30s TTL) |
| `MainRunSyncHook` | `gigaevo/prompts/coevolution/sync.py` | Pre-step hook: blocks engine after each generation until the opponent population has also advanced by >= 1 generation. Polls `{prefix}:run_state engine:total_generations` |
| `config/pipeline/adversarial_coevo.yaml` | Config | Hydra config tying it all together |

### Data Flow

```
Pop A process (redis.db=1)              Pop B process (redis.db=2)
┌───────────────────────┐               ┌───────────────────────┐
│ CallProgramFunction   │               │ CallProgramFunction   │
│   → program_output    │               │   → program_output    │
│                       │               │                       │
│ FetchOpponentResults  │◄──── reads ───│ archive (Pop B DB 2)  │
│   → opponent_results  │               │                       │
│                       │               │ FetchOpponentResults  │
│ archive (Pop A DB 1) ─┼─── reads ────►│   → opponent_results  │
│                       │               │                       │
│ CallValidatorFunction │               │ CallValidatorFunction │
│   evaluate.py(        │               │   evaluate.py(        │
│     opponent_results, │               │     opponent_results, │
│     program_output)   │               │     program_output)   │
│   → metrics dict      │               │   → metrics dict      │
└───────────────────────┘               └───────────────────────┘
         ▲                                        ▲
         └──── MainRunSyncHook ────────────────────┘
               (lockstep: wait for opponent gen)
```

---

## How to Create a New Adversarial Problem

### Step 1: Create Two Population Directories

```
problems/<your_problem>/
├── pop_a/
│   ├── evaluate.py          # REQUIRED
│   ├── metrics.yaml         # REQUIRED
│   ├── task_description.txt # REQUIRED
│   ├── helper.py            # optional shared utilities
│   ├── initial_programs/    # REQUIRED: at least 1 seed .py
│   │   └── seed.py
│   └── fallback/            # RECOMMENDED: cold-start opponents
│       └── simple.py
│
└── pop_b/
    ├── evaluate.py
    ├── metrics.yaml
    ├── task_description.txt
    ├── helper.py
    ├── initial_programs/
    │   └── seed.py
    └── fallback/
        └── simple.py
```

### Step 2: Write evaluate.py

**Signature** (both populations):
```python
def evaluate(opponent_results: list, program_output: object) -> dict[str, float]:
    """
    Args:
        opponent_results: list of outputs from opponent population's entrypoint()
        program_output: output of this population's entrypoint()

    Returns:
        dict with at least 'fitness' and 'is_valid' keys.
        All values must be float.
    """
```

**Design rules**:
- `opponent_results` contains the raw return values of the opponent's `entrypoint()` function
- Handle empty `opponent_results` gracefully (cold start)
- Return a sentinel dict with `is_valid: 0` for invalid programs
- Track `actual_fitness` (raw objective) separately from `fitness` (selection metric)
- Ensure `fitness` is in [0, 1] for consistent MAP-Elites behavior

**Fitness design pattern** (Prover/Improver):
```python
# Pop A: quality + resistance to improvement
fitness = ALPHA * quality + (1 - ALPHA) * resistance

# Pop B: mean improvement achieved
fitness = mean(normalized_improvements)
```

**Zero-sum property**: For any (Pop A, Pop B) pair, `resistance + mean_improvement = 1.0`. This creates the adversarial pressure — what's good for one is bad for the other.

### Step 3: Write metrics.yaml

Each population needs its own `metrics.yaml`. Required metrics:

```yaml
specs:
  fitness:
    description: "Primary selection metric"
    is_primary: true
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
    # ... other fields
  is_valid:
    description: "Validity flag"
    is_primary: false
    # ...
```

Add any additional tracking metrics (quality, resistance, n_opponents, etc.). Only metrics declared in `metrics.yaml` pass through `EnsureMetricsStage` — extras are silently dropped.

### Step 4: Write Seeds and Fallbacks

**Seed programs** (`initial_programs/seed.py`): The first program each population starts with. Must define `entrypoint()`.

**Fallback programs** (`fallback/`): Used during cold start when the opponent archive is empty. Should provide a basic opponent so the population can begin evolving meaningfully.

- Pop A fallback = simple Pop B implementations (so Pop A has something to resist)
- Pop B fallback = simple Pop A implementations (so Pop B has something to improve)

### Step 5: Write task_description.txt

Describe the adversarial game, not implementation details:
- What role does this population play?
- What does the opponent do?
- What makes a good solution?
- What constraints must be satisfied?

Do NOT list strategies or hardcode timeouts — let the LLM discover approaches.

---

## How to Configure and Launch

### Required Hydra Overrides

Each run needs these overrides:

```bash
python run.py \
  problem.name=<your_problem>/pop_a \
  pipeline=adversarial_coevo \
  redis.db=<DB_A> \
  opponent_redis_db=<DB_B> \
  opponent_redis_prefix=<your_problem>/pop_b \
  pipeline_builder.per_opponent_timeout=<seconds>
```

**Critical**: `per_opponent_timeout` is nested under `pipeline_builder`, NOT top-level. Using `per_opponent_timeout=300` silently falls back to the default (10s).

### experiment.yaml Structure

For N=2 replicate pairs, you need 4 runs:

```yaml
runs:
  - label: P1_A
    db: 1
    prefix: <problem>/pop_a
    pipeline: adversarial_coevo
    problem_name: <problem>/pop_a
    condition: "Pair 1: Pop A"
    extra_overrides:
      - opponent_redis_db=2
      - opponent_redis_prefix=<problem>/pop_b
      - pipeline_builder.per_opponent_timeout=300

  - label: P1_B
    db: 2
    prefix: <problem>/pop_b
    pipeline: adversarial_coevo
    problem_name: <problem>/pop_b
    condition: "Pair 1: Pop B"
    extra_overrides:
      - opponent_redis_db=1
      - opponent_redis_prefix=<problem>/pop_a
      - pipeline_builder.per_opponent_timeout=300

  # Pair 2: same config, different DBs (3, 4)
```

**Label naming**: Use underscores not hyphens (e.g. `P1_A` not `P1-A`). Hyphens break bash variable names in `launch.sh`.

### Config Defaults (adversarial_coevo.yaml)

| Key | Default | Notes |
|-----|---------|-------|
| `opponent_provider.cache_ttl` | 30.0 | Seconds to cache opponent list |
| `pipeline_builder.n_opponents` | 5 | Opponents per evaluation |
| `pipeline_builder.per_opponent_timeout` | 10.0 | Seconds per opponent subprocess |
| `pre_step_hook.timeout` | 7200.0 | Max seconds to wait for opponent sync |
| `pre_step_hook.poll_interval` | 5.0 | Seconds between sync polls |

---

## Monitoring

### Status Check

```bash
gigaevo status -e <task>/<name>
```

Key things to watch:
- **Generation parity**: Both populations should be within ~2 generations of each other. Large gaps indicate sync hook issues.
- **n_opponents**: Should be > 0 after gen 1. If stuck at 0, the opponent archive is empty.
- **Invalid%**: High invalidity (>50%) is normal for gen 1-2, but if it persists, check evaluate.py error handling.

### Sync Diagnostics

```bash
grep "MainRunSyncHook" experiments/<task>/<name>/<logfile>.log
```

If you see timeout warnings, the opponent population is stuck. Check its log for errors.

### Fitness Interpretation

- `fitness` (selection metric): May be volatile due to arms race dynamics. Pop A fitness can drop when Pop B improves.
- `actual_fitness` (raw objective): Should trend upward over generations. This is the real measure of progress.
- `resistance` (Pop A): Should increase as Pop A finds harder-to-improve solutions.

---

## Gotchas and Troubleshooting

### 1. Hydra Config Nesting

`per_opponent_timeout` is nested under `pipeline_builder`:
```
pipeline_builder.per_opponent_timeout=300  # Correct
per_opponent_timeout=300                    # WRONG — silently uses default 10s
```

### 2. Cloudpickle Deserialization

When programs return callables (e.g. Pop B returns an `improve()` function), the callable is serialized with `cloudpickle` in the subprocess and deserialized in the parent. If the callable has closures over module-level imports (e.g. `from helper import foo`), `sys.path` must include the problem directory at deserialization time.

The `wrapper.py` worker pool handles this via `_prepend_sys_path(python_path)` before `cloudpickle.loads()`. If you see `ModuleNotFoundError` during opponent execution, check that `python_path` is being passed correctly.

### 3. Cold Start Race Condition

At gen 0, both populations have empty archives. Each tries to fetch opponents from the other's archive and gets nothing. This is handled by:
- Pop A: `if not opponent_results: fitness = quality` (quality-only, resistance=1.0)
- Pop B: `if not opponent_results: return INVALID` (can't evaluate without opponents)

Pop B seeds should still be able to run their `entrypoint()` to produce a callable, even if they can't be evaluated yet. The fallback directory provides simple opponents for initial evaluation.

### 4. Stale Processes from Worktrees

If you run experiments from git worktrees (`.claude/worktrees/`), stale `run.py` processes may write to Redis DBs that you're trying to flush. Use `tools/flush.py` which detects both main-repo and worktree processes.

### 5. EnsureMetricsStage Drops Extras

Only metrics declared in `metrics.yaml` pass through `EnsureMetricsStage`. If `evaluate.py` returns a metric not in `metrics.yaml`, it is silently dropped. Always verify your metrics.yaml matches your evaluate.py output keys.

---

## Existing Adversarial Problems

| Problem | Pop A Role | Pop B Role | Status |
|---------|-----------|-----------|--------|
| `adversarial/optimizer_v2` | Optimizer (minimize f) | Landscape designer (make deceptive f) | Pilot complete (PR #169) |
| `heilbron_adversarial` | Constructor (place 11 points) | Improver (improve point configs) | Running (PR #183) |

---

## Code Reference

| What | Where |
|------|-------|
| Pipeline builder | `gigaevo/adversarial/pipeline.py` |
| FetchOpponentResultsStage | `gigaevo/adversarial/stages.py` |
| OpponentArchiveProvider | `gigaevo/adversarial/opponent_provider.py` |
| Sync hook | `gigaevo/prompts/coevolution/sync.py` |
| Pipeline config | `config/pipeline/adversarial_coevo.yaml` |
| Tests | `tests/adversarial_pipeline/` |
