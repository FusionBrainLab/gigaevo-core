# Adversarial Co-Evolution in GigaEvo

## Overview

**Adversarial co-evolution** is a two-population optimization technique where two populations compete in a GAN-like dynamic. One population (Generator/Constructor) produces candidate solutions; the other (Discriminator/Improver) tries to prove the candidates are suboptimal. This mutual pressure drives both populations toward higher-quality solutions and genuine local optima.

This guide explains how adversarial co-evolution works in GigaEvo, its architecture, fitness dynamics, and practical usage.

---

## The Core Idea: GAN-Inspired Co-Evolution

### Structural Mapping

| GAN Component | GigaEvo Equivalent | Role |
|---|---|---|
| **Generator G(z)** | Pop A (Constructor) | Produces candidate solutions (programs/configs) |
| **Discriminator D(x)** | Pop B (Improver) | Attempts to improve (disprove optimality) |
| **G loss** | Pop A fitness | quality (solution merit) + resistance (resistance to improvement) |
| **D loss** | Pop B fitness | improvement achieved (how much can it improve) |
| **Adversarial loop** | Lockstep sync | Both populations advance by 1 gen simultaneously |

### The Zero-Sum Dynamic

For a given pair (Pop A candidate, Pop B improver):

```
δ(C, I) = min_area(I(C())) - min_area(C())    [improvement margin]
```

**Zero-sum constraint**: `resistance + improvement_ratio = 1.0`

- When Pop B finds **large improvements** (δ → big): Pop A's resistance → 0, Pop B's fitness → high
- When Pop A **resists improvements** (δ → 0): Pop A's resistance → 1, Pop B's fitness → low
- The evolutionary pressure creates an **arms race**: as Pop B gets better at improving, Pop A must evolve harder-to-improve solutions

---

## Fitness Formulas

### Population A (Constructor) — "Generator"

The constructor's fitness balances two objectives:

```python
quality       = min(min_area(points) / Q_MAX, 1.0)
resistance    = 1.0 - mean_k(min(delta_k / Q_MAX, 1.0))
fitness       = ALPHA * quality + (1 - ALPHA) * resistance
```

Where:
- **quality**: Raw performance (normalized to [0,1])
- **resistance**: How hard the solution is to improve (1 = impossible to improve)
- **ALPHA** (mixing weight): 0.5 = equal weighting
- **delta_k**: Improvement achieved by opponent k on this candidate
- **Q_MAX** (0.0365): Target constant (normalizes both components to [0,1])

**Cold start** (no opponents yet): `fitness = quality`, `resistance = 1.0`

### Population B (Improver) — "Discriminator"

The improver's fitness is the mean improvement achieved:

```python
fitness = mean_k(min(delta_k / Q_MAX, 1.0))
```

Where:
- **delta_k** = max(min_area(improve_fn(P_k)) - min_area(P_k), 0.0)
- Improvements are clamped at 0 (improving = positive, worsening = 0 credit)
- Normalized by Q_MAX to [0,1]

**actual_fitness** (paper metric): `max_k(min_area(improve_fn(P_k)))`

---

## The Adversarial Pipeline

### Overview

GigaEvo's `pipeline=adversarial_coevo` is built on top of the standard pipeline, adding:

1. **FetchOpponentResultsStage** — Fetch and execute opponent programs
2. **Sync Hook** — Lockstep generation advancement
3. **Modified CallValidatorFunction** — Calls `evaluate.py` instead of `validate.py`

### Architecture

```
┌─── Pop A Run ───────────────────┐     ┌─── Pop B Run ───────────────────┐
│                                 │     │                                 │
│  1. MutationStage               │     │  1. MutationStage               │
│     ↓                           │     │     ↓                           │
│  2. ValidateCodeStage           │     │  2. ValidateCodeStage           │
│     ↓                           │     │     ↓                           │
│  3a. CallProgramFunction        │     │  3a. CallProgramFunction        │
│  3b. FetchOpponentResults ◄─────┼─────┼──→ Archive (Pop A programs)    │
│      (Pop B from Pop B DB)      │     │                                 │
│     ↓ opponent_results          │     │  3b. FetchOpponentResults ◄─────┼─────┤
│  4. CallValidatorFunction       │     │      (Pop A from Pop A DB)      │
│     (evaluate.py with context)  │     │     ↓ opponent_results          │
│     ↓                           │     │  4. CallValidatorFunction       │
│  5. EnsureMetricsStage          │     │     (evaluate.py with context)  │
│     ↓                           │     │     ↓                           │
│  6. CollectorStage              │     │  5. EnsureMetricsStage          │
│     (ingest into MAP-Elites)    │     │     ↓                           │
│     ↓ update archive            │     │  6. CollectorStage              │
│                                 │     │                                 │
│  MainRunSyncHook ◄──────────────┼─────┼─→ Await Pop B gen advancement  │
│  (wait for Pop B gen +1)        │     │  MainRunSyncHook               │
│                                 │     │  (wait for Pop A gen +1)        │
└─────────────────────────────────┘     └─────────────────────────────────┘

Data flow: opponent_results → CallValidatorFunction context
Sync: lockstep — neither population advances until both reach end of generation
```

### Key Stages

#### FetchOpponentResultsStage

```python
class FetchOpponentResultsStage(Stage):
    """Fetch and execute opponent programs in parallel subprocesses."""
    
    async def compute(self, program: Program) -> Box[Any]:
        # 1. Get N opponents from opponent's MAP-Elites archive (Redis)
        opponents = await self._provider.get_opponents(n=self._n)
        codes = [o.code for o in opponents]
        
        # 2. Fallback if archive empty (cold start)
        if not codes and self._fallback_codes:
            codes = self._fallback_codes
        
        # 3. Execute each opponent in parallel subprocesses
        # - Each has its own timeout (per_opponent_timeout)
        # - One timeout doesn't block others (asyncio.gather)
        tasks = [self._exec_one(code) for code in codes]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. Filter out failures, return as context Box
        results = [r for r in raw if not isinstance(r, Exception) and r is not None]
        return Box[Any](data=results)
```

**Key property**: Opponents are executed in **subprocesses with isolated timeouts**. A slow/buggy opponent doesn't block others; it just times out individually and is filtered out.

#### CallValidatorFunction (Modified)

In adversarial mode:
- Calls `evaluate.py` instead of `validate.py`
- Receives `opponent_results` as context (array of opponent outputs)
- `evaluate.py` signature: `evaluate(opponent_results: list[Any], program_output: Any) -> dict[str, float]`

Example (Pop A constructor):
```python
def evaluate(opponent_results, points):
    # opponent_results = [improved_points_1, improved_points_2, ...]
    # points = output of current Pop A program
    
    quality = compute_quality(points)
    deltas = [min_area(improved) - min_area(points) for improved in opponent_results]
    resistance = 1.0 - mean(normalize(deltas))
    
    return {
        'fitness': 0.5 * quality + 0.5 * resistance,
        'actual_fitness': min_area(points),
        'quality': quality,
        'resistance': resistance,
        ...
    }
```

#### MainRunSyncHook

```yaml
pre_step_hook:
  _target_: gigaevo.prompts.coevolution.sync.MainRunSyncHook
  host: ${redis.host}
  port: ${redis.port}
  db: ${opponent_redis_db}        # Watch opponent's Redis DB
  prefix: ${opponent_redis_prefix}  # Watch opponent's Redis prefix
  timeout: 7200.0                   # Max wait time
  poll_interval: 5.0                # Check every 5 seconds
```

**Behavior**:
- After every generation completes, this hook blocks the engine
- It polls opponent's `{prefix}:run_state` watching `engine:total_generations`
- Blocks until opponent advances by ≥1 generation
- Maximum 2-hour wait; if timeout, raises error

This ensures **lockstep synchronization**: neither population races ahead.

---

## Configuration

### experiment.yaml

```yaml
runs:
  - label: P1_A
    pipeline: adversarial_coevo
    problem_name: heilbron_adversarial/pop_a
    redis.db: 1
    extra_overrides:
      - opponent_redis_db=2
      - opponent_redis_prefix=heilbron_adversarial/pop_b
      - pipeline_builder.per_opponent_timeout=300

  - label: P1_B
    pipeline: adversarial_coevo
    problem_name: heilbron_adversarial/pop_b
    redis.db: 2
    extra_overrides:
      - opponent_redis_db=1
      - opponent_redis_prefix=heilbron_adversarial/pop_a
      - pipeline_builder.per_opponent_timeout=300
```

### Config Hierarchy

```
config/pipeline/adversarial_coevo.yaml
├── opponent_redis_db: ??? (required at launch)
├── opponent_redis_prefix: ??? (required at launch)
├── pre_step_hook: MainRunSyncHook
├── opponent_provider: RedisOpponentArchiveProvider
└── pipeline_builder: AdversarialPipelineBuilder
    ├── n_opponents: 5 (default)
    ├── per_opponent_timeout: 10.0 (default, usually overridden)
    └── ... other timeout fields ...
```

**At launch time, you must provide**:
```bash
python run.py \
  problem.name=heilbron_adversarial/pop_a \
  pipeline=adversarial_coevo \
  redis.db=1 \
  opponent_redis_db=2 \
  opponent_redis_prefix=heilbron_adversarial/pop_b \
  pipeline_builder.per_opponent_timeout=300
```

---

## Metrics Tracking

### Pop A Metrics

| Metric | Range | Primary | Purpose |
|--------|-------|---------|---------|
| **fitness** | [0,1] | ✓ | Drives MAP-Elites selection (quality + resistance) |
| **actual_fitness** | [0,0.0365] | | Raw min_area (paper metric, independent of adversarial dynamics) |
| **quality** | [0,1] | | Normalized min_area |
| **resistance** | [0,1] | | 1 - mean normalized improvement |
| **mean_improvement** | [0,0.0365] | | Mean delta achieved by opponents |
| **best_post_improvement** | [0,0.0365] | | Best min_area any opponent achieved |
| **n_opponents** | [0,10] | | Opponent count in this evaluation |

### Pop B Metrics

| Metric | Range | Primary | Purpose |
|--------|-------|---------|---------|
| **fitness** | [0,1] | ✓ | Drives MAP-Elites selection (mean normalized improvement) |
| **actual_fitness** | [0,0.0365] | | Best post-improvement min_area (paper metric) |
| **mean_improvement_raw** | [0,0.0365] | | Mean raw improvement |
| **mean_pre_quality** | [0,0.0365] | | Mean opponent quality before improvement |
| **mean_post_quality** | [0,0.0365] | | Mean opponent quality after improvement |
| **max_post_quality** | [0,0.0365] | | Best quality achieved after improvement |
| **n_opponents** | [0,10] | | Opponent count in this evaluation |

**Critical distinction**:
- **fitness**: Drives selection, includes adversarial dynamics (may inflate/deflate)
- **actual_fitness**: Raw objective, independent of dynamics, reported in paper

---

## Problem Directory Structure

```
problems/heilbron_adversarial/
├── pop_a/                         # Constructor population
│   ├── evaluate.py               # evaluate(opponent_results, points) → dict[str, float]
│   ├── metrics.yaml              # 8 metrics (fitness, actual_fitness, quality, resistance, ...)
│   ├── task_description.txt      # Describes the adversarial game to the LLM
│   ├── helper.py                 # Shared validation/geometry helpers
│   ├── initial_programs/
│   │   └── grid.py               # Seed program
│   └── fallback/                 # Cold-start improvers (for when Pop B archive empty)
│       ├── jitter.py
│       └── local_search.py
│
└── pop_b/                         # Improver population
    ├── evaluate.py               # evaluate(opponent_results, improve_fn) → dict[str, float]
    ├── metrics.yaml              # 8 metrics (fitness, actual_fitness, improvement, ...)
    ├── task_description.txt      # Describes the adversarial game to the LLM
    ├── helper.py                 # Symlink/copy from pop_a
    ├── initial_programs/
    │   └── seed.py               # Seed improver
    └── fallback/                 # Cold-start constructors (for when Pop A archive empty)
        ├── grid.py
        └── random_arr.py
```

### evaluate.py Structure

**Pop A (Constructor)**:
```python
def evaluate(opponent_results, program_output):
    """
    Args:
        opponent_results: list of improved points from opponent improvers
        program_output: (n_points, 2) array of this constructor's points
    
    Returns:
        dict with keys: fitness, actual_fitness, quality, resistance, mean_improvement, ...
    """
    points = program_output
    
    # Compute raw quality
    actual_fitness = min_area(points)
    quality = min(actual_fitness / Q_MAX, 1.0)
    
    # Compute resistance from opponent improvements
    if opponent_results:
        deltas = [min_area(improved) - actual_fitness for improved in opponent_results]
        deltas = [max(d, 0) for d in deltas]  # Clamp negatives
        resistance = 1.0 - mean([min(d / Q_MAX, 1.0) for d in deltas])
        mean_improvement = mean(deltas)
        best_post = min_area(max(opponent_results, key=lambda p: min_area(p)))
    else:
        resistance = 1.0  # Cold start
        mean_improvement = 0.0
        best_post = actual_fitness
    
    # Fitness combines quality and resistance
    fitness = 0.5 * quality + 0.5 * resistance
    
    return {
        'fitness': fitness,
        'actual_fitness': actual_fitness,
        'quality': quality,
        'resistance': resistance,
        'mean_improvement': mean_improvement,
        'best_post_improvement': best_post,
        'n_opponents': len(opponent_results),
        'is_valid': 1,
    }
```

**Pop B (Improver)**:
```python
def evaluate(opponent_results, program_output):
    """
    Args:
        opponent_results: list of point configs from opponent constructors
        program_output: callable improve(points) -> improved_points
    
    Returns:
        dict with keys: fitness, actual_fitness, mean_improvement_raw, ...
    """
    improve_fn = program_output
    
    if not opponent_results:
        return {'fitness': 0.0, 'is_valid': 0, ...}  # Invalid without opponents
    
    # Try to improve each opponent's config
    improved_configs = []
    deltas = []
    
    for config in opponent_results:
        try:
            improved = improve_fn(config.copy())
            improved_quality = min_area(improved)
            original_quality = min_area(config)
            delta = max(improved_quality - original_quality, 0.0)
            
            improved_configs.append(improved)
            deltas.append(delta)
        except:
            # Improver failed on this config, skip
            pass
    
    if not improved_configs:
        return {'fitness': 0.0, 'is_valid': 0, ...}  # Invalid if can't improve any
    
    # Compute fitness and metrics
    fitness = mean([min(d / Q_MAX, 1.0) for d in deltas])
    mean_improvement = mean(deltas)
    best_quality = min(min_area(c) for c in improved_configs)
    
    return {
        'fitness': fitness,
        'actual_fitness': best_quality,
        'mean_improvement_raw': mean_improvement,
        'mean_pre_quality': mean(min_area(c) for c in opponent_results),
        'mean_post_quality': mean(min_area(c) for c in improved_configs),
        'max_post_quality': best_quality,
        'n_opponents': len(opponent_results),
        'is_valid': 1,
    }
```

---

## Execution Flow

### Generation N Lifecycle

Both populations run **in parallel** with lockstep sync:

```
Pop A                           Pop B
├─ Gen N-1 complete
│  ├─ MutationStage
│  ├─ ValidateCodeStage
│  ├─ CallProgramFunction
│  ├─ FetchOpponentResults
│  │   └─ Execute Pop B programs (call entrypoint())
│  │       from Pop B's Gen N-1 archive
│  ├─ CallValidatorFunction
│  │   └─ evaluate(opponent_results, pop_a_output)
│  ├─ EnsureMetricsStage
│  ├─ CollectorStage
│  │   └─ Update archive with new individuals
│  │
│  └─ Engine advances to Gen N
│
└─ MainRunSyncHook
   │   Blocks: wait for Pop B to reach gen N
   │   Polls: Pop B's run_state:engine:total_generations
   │   ↓
   │   (same time) ↓
│                    Pop A
│                    ├─ Gen N-1 complete
│                    ├─ MutationStage
│                    ├─ ValidateCodeStage
│                    ├─ CallProgramFunction
│                    ├─ FetchOpponentResults
│                    │   └─ Execute Pop A programs
│                    │       from Pop A's Gen N-1 archive
│                    ├─ CallValidatorFunction
│                    │   └─ evaluate(opponent_results, pop_b_output)
│                    ├─ EnsureMetricsStage
│                    ├─ CollectorStage
│                    │   └─ Update archive
│                    │
│                    └─ Engine advances to Gen N
│
│                    └─ MainRunSyncHook
│                       └─ Blocks: wait for Pop A to reach gen N
│
└─ MainRunSyncHook returns, both proceed to Gen N+1
```

**Key**: Both populations evaluate against **N-1 generation** opponents (previous generation), not the current one. This ensures reproducibility and prevents race conditions.

---

## Common Pitfalls

### 1. Config Nesting: per_opponent_timeout

❌ **Wrong**:
```bash
python run.py ... per_opponent_timeout=300
```

✓ **Correct**:
```bash
python run.py ... pipeline_builder.per_opponent_timeout=300
```

The timeout is nested under `pipeline_builder` in the Hydra config, not at the top level.

### 2. Cloudpickle and Module Imports

When Pop B programs return callables with closures:
```python
# Pop B seed.py
def entrypoint():
    from helper import get_unit_triangle  # Import in function scope
    
    def improve(points):
        triangle = get_unit_triangle()
        # ... improvement logic ...
        return improved_points
    
    return improve
```

The callable must be deserializable by `cloudpickle.loads()` **after** `sys.path` is updated to include the problem directory. The exec_runner wrapper handles this via `_prepend_sys_path(python_path)` before deserialization.

### 3. Task Descriptions

**Bad**: Lists strategies, timeouts, implementation details
```
Write a function that improves point configurations by:
1. Trying random perturbations
2. Using scipy.optimize.minimize
3. Greedy accept within 60 seconds
```

**Good**: Describes the game and constraints
```
You are an Improver in an adversarial optimization game.
The opponent (Constructor) places 11 points in a unit-area triangle.
Your goal: find ANY changes that increase the minimum distance to edges.
You have ~60 seconds per configuration to find improvements.
```

Let LLMs find their own strategies.

### 4. Opponent Archive Availability

**Cold start (first few generations)**: Use fallback programs from the `fallback/` subdirectory. Once each population has produced enough individuals, FetchOpponentResultsStage reads from the live archive.

If **fallback is empty** and archive is empty, FetchOpponentResultsStage logs a warning and returns an empty opponent list. The evaluate.py should handle this gracefully (return "invalid" or use a cold-start fallback fitness).

---

## Monitoring an Adversarial Run

### Check Generation Parity

```bash
PYTHONPATH=. python tools/status.py --experiment adversarial/heilbron-prover
```

Output shows each run's generation, fitness, and PID. **Both populations should advance roughly in lockstep** — if one is far ahead, check the logs for sync hook timeouts.

### Check Opponent Availability

```bash
PYTHONPATH=. python tools/top_programs.py --run heilbron_adversarial/pop_a@1:P1_A
```

This shows the best programs in Pop A from the archive — these are the "opponents" Pop B will face next.

### Check Fitness Metrics

```bash
PYTHONPATH=. redis-cli -n 1 HGETALL heilbron_adversarial/pop_a:metrics:latest
```

Shows all latest metric values. `actual_fitness` should climb over generations (raw quality improving). `fitness` (selection metric) may be more volatile due to adversarial dynamics.

### Diagnose Sync Issues

Check logs for `MainRunSyncHook` messages:
```bash
grep "MainRunSyncHook" experiments/adversarial/heilbron-prover/pop_a_pair1.log
```

If you see "timeout waiting for opponent", the other population is blocked (check its logs for errors).

---

## References

- **Code**:
  - `gigaevo/adversarial/` — Pipeline implementation
  - `gigaevo/prompts/coevolution/sync.py` — Sync hook
  - `config/pipeline/adversarial_coevo.yaml` — Config template

- **Experiments**:
  - `problems/adversarial/optimizer_v2/` — First adversarial pilot (optimizer vs landscapes)
  - `problems/heilbron_adversarial/` — Heilbron prover/improver (current)

- **Documentation**:
  - `experiments/adversarial/heilbron-prover/01_design.md` — Scientific design
  - `experiments/adversarial/heilbron-prover/02_review.md` — Adversarial review
  - `experiments/adversarial/heilbron-prover/03_plan.md` — Pre-registration

---

## Summary

Adversarial co-evolution in GigaEvo:

1. **Two populations compete** in a GAN-like dynamic (Generator vs Discriminator)
2. **Fitness design** balances solution quality with resistance-to-improvement
3. **Pipeline** adds FetchOpponentResultsStage (fetch + execute opponents) + sync hook (lockstep)
4. **Metrics tracking** separates selection fitness from true objective (actual_fitness)
5. **Cold start handling** via fallback programs until populations stabilize
6. **Lockstep sync** ensures neither population races ahead

The mutual pressure drives populations toward genuine local optima and higher-quality solutions.
