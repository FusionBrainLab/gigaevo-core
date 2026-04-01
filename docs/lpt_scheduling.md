# LPT Scheduling for DAG Evaluation

## Problem

The DagRunner launches programs for evaluation in arbitrary order (Redis SET
iteration is hash-based, not insertion-ordered).  With variable DAG evaluation
times (LogNormal distributed, range 8-40 min depending on problem), this
creates bursty throughput: long-running programs block server slots while short
ones finish early, leaving servers idle.

```
FIFO:     |=============================|  long DAG (25 min)
          |==========|                     short DAG (8 min) — server idle 17 min

LPT:      |=============================|  long DAG (started first)
          |==========|                     short DAG (started last, fills gap)
```

## Solution: Longest Processing Time (LPT) Scheduling

Predict each program's evaluation time before launching, then start
predicted-longest programs first.  This minimizes tail idle time and
maximizes server utilization.

LPT is optimal for minimizing makespan on identical parallel machines
(Graham, 1969 — "Bounds on multiprocessing timing anomalies").

## Architecture

```
                         User code
                            |
                   FeatureExtractor (Protocol)
                   extract(program) -> dict[str, float]
                            |
                   EvalTimePredictor (ABC)
                   predict(program) -> float
                   update(program, actual_duration)
                            |
                   ProgramPrioritizer (ABC)
                   prioritize(programs) -> list[Program]
                            |
                   DagRunner._launch()
                   (sorts candidates before creating tasks)
```

All components are in `gigaevo/evolution/scheduling/`.

### Flow

1. DagRunner fetches QUEUED programs from Redis via `SMEMBERS` + `mget`
2. Fetched `Program` objects are passed to `prioritizer.prioritize()`
3. Programs are launched as async tasks in the returned order
4. After each DAG completes, `predictor.update(program, actual_duration)` feeds
   back the real timing for online learning
5. The predictor improves over time, so ordering improves across batches

### No extra Redis cost

The prioritizer operates on `Program` objects already deserialized by `mget`.
The sort is O(K log K) where K = batch size (typically 8-64).  Zero additional
Redis round trips.

## Components

### FeatureExtractor (Protocol)

```python
class FeatureExtractor(Protocol):
    def extract(self, program: Program) -> dict[str, float]: ...
```

Structural subtyping — implement the method, no inheritance required.

**Built-in extractors**:

| Extractor | Features | Use case |
|---|---|---|
| `CodeFeatureExtractor` | `code_length`, `num_lines`, `num_function_defs`, `num_loop_constructs` | Any problem (generic) |
| `ChainFeatureExtractor` | `n_tool_steps`, `n_llm_steps`, `n_total_steps`, `total_string_content`, `n_deep_retrieval`, `n_examples`, `has_system_prompt`, `max_dependency_fan_in` | Chain-based problems (HoVer, HotpotQA, etc.) |

`ChainFeatureExtractor` works on real chain-definition programs — the Python
functions returning dicts with `system_prompt` and `steps`.  Tested against
actual HoVer and HotpotQA baselines:

```python
# HoVer baseline: 3 tool + 4 LLM steps, 1 deep retrieval, max fan-in 2
# HotpotQA baseline: 2 tool + 4 LLM steps, 0 deep retrieval, max fan-in 2
```

**Custom example** (problem-specific features):

```python
class MyChainExtractor:
    def extract(self, program: Program) -> dict[str, float]:
        code = program.code
        return {
            "n_or_operators": float(code.count(" OR ")),  # query expansion
            "has_contradiction_check": float("CONTRADICTION" in code),
        }
```

**Composing extractors**:

```python
from gigaevo.evolution.scheduling import (
    ChainFeatureExtractor,
    CompositeFeatureExtractor,
)

extractor = CompositeFeatureExtractor([
    ChainFeatureExtractor(),     # chain structure features
    MyChainExtractor(),          # problem-specific features
])
```

### EvalTimePredictor (ABC)

```python
class EvalTimePredictor(ABC):
    def predict(self, program: Program) -> float: ...   # seconds
    def update(self, program: Program, actual_duration: float) -> None: ...
    def is_warm(self) -> bool: ...
```

| Implementation | Description | Dependencies | Cold-start behavior |
|---|---|---|---|
| `ConstantPredictor` | Returns fixed value | None | FIFO (all equal) |
| `SimpleHeuristicPredictor` | `code_length * learned_rate` | None | Uses default rate, warms after 5 samples |
| `RidgePredictor` | sklearn Ridge regression over features | sklearn (soft) | Falls back to code-length heuristic |

**Online learning**: Both `SimpleHeuristicPredictor` and `RidgePredictor` update
from every completed DAG execution.  They learn from failures too (duration-until-failure
is informative) to avoid survivorship bias.

**Safety**:
- `SimpleHeuristicPredictor` uses median (not mean) and clips outliers at 10x median
- `RidgePredictor` guards against NaN/Inf predictions, falls back to default
- sklearn import is lazy and guarded — if unavailable, logs a warning once

### ProgramPrioritizer (ABC)

```python
class ProgramPrioritizer(ABC):
    def prioritize(self, programs: list[Program]) -> list[Program]: ...
    @property
    def predictor(self) -> EvalTimePredictor | None: ...
```

| Strategy | Description | When to use |
|---|---|---|
| `FIFOPrioritizer` | Preserves input order | Default (backward compatible) |
| `LPTPrioritizer` | Longest predicted first | Production (reduces makespan) |
| `SJFPrioritizer` | Shortest predicted first | Benchmarking control only |

All prioritizers fall back to FIFO when the predictor is cold (`is_warm() == False`).

## Usage

### Enabling LPT in DagRunner

```python
from gigaevo.evolution.scheduling import (
    LPTPrioritizer,
    SimpleHeuristicPredictor,
)

predictor = SimpleHeuristicPredictor()
prioritizer = LPTPrioritizer(predictor)

runner = DagRunner(
    storage=storage,
    dag_blueprint=blueprint,
    config=config,
    writer=writer,
    prioritizer=prioritizer,  # <-- opt-in
)
```

Without the `prioritizer` argument, DagRunner uses `FIFOPrioritizer` (unchanged
behavior).

### With custom features (e.g., for a specific problem)

```python
from gigaevo.evolution.scheduling import (
    CompositeFeatureExtractor,
    CodeFeatureExtractor,
    RidgePredictor,
    LPTPrioritizer,
)

class MyExtractor:
    def extract(self, program):
        return {"n_api_calls": float(program.code.count("api_call("))}

extractor = CompositeFeatureExtractor([CodeFeatureExtractor(), MyExtractor()])
predictor = RidgePredictor(feature_extractor=extractor)
prioritizer = LPTPrioritizer(predictor)
```

## Benchmarks

Run `benchmarks/bench_lpt.py` for a discrete-event simulation comparing
strategies.  No Redis or real DAGs needed — pure math.

```bash
# Default: 40 programs, 4 servers, noise=0.3
PYTHONPATH=. python benchmarks/bench_lpt.py

# Sweep noise levels (prediction quality impact)
PYTHONPATH=. python benchmarks/bench_lpt.py --sweep-noise

# Sweep server count
PYTHONPATH=. python benchmarks/bench_lpt.py --sweep-servers

# Sweep batch size
PYTHONPATH=. python benchmarks/bench_lpt.py --sweep-programs
```

### Results (40 programs, 4 servers, noise=0.3)

| Strategy | Makespan | Utilization | vs FIFO |
|---|---|---|---|
| FIFO | 15007s | 93.9% | — |
| Oracle LPT | 14154s | 99.5% | +5.7% |
| Predicted LPT | 14234s | 99.0% | **+5.2%** |
| SJF | 15304s | 92.0% | -2.0% |

Key findings:
- Predicted LPT captures 91% of oracle upper bound
- Higher eval-time variance = bigger LPT benefit (+21.8% at noise=1.0)
- Small batches benefit most (+24.1% for 8 programs) — matches `max_in_flight=5`

## Testing

### Test files

| File | Tests | What it covers |
|---|---|---|
| `tests/evolution/test_scheduling.py` | 37 | Unit tests: extractors, predictors, prioritizers, protocol compliance |
| `tests/evolution/test_scheduling_integration.py` | 15 | Integration: DagRunner calls prioritizer, predictor learns from execution, LPT beats FIFO on real async event loop, multi-cycle online learning |

### Running tests

```bash
# All scheduling tests (unit + integration)
$GIGAEVO_PYTHON -m pytest tests/evolution/test_scheduling*.py -v

# Just integration tests
$GIGAEVO_PYTHON -m pytest tests/evolution/test_scheduling_integration.py -v
```

### What the integration tests verify

1. **DagRunner._launch() calls prioritizer** — the sort actually happens on
   fetched programs, not just in unit tests
2. **LPT launches longest first** — with `max_concurrent_dags=1` (serialized),
   the longer-code program executes before the shorter one
3. **Predictor updated after execution** — `SimpleHeuristicPredictor._window`
   grows after `_execute_dag()` completes
4. **Predictor updated on failure too** — survivorship bias fix verified
5. **FIFO is default** — `DagRunner()` without `prioritizer` kwarg uses FIFO
6. **LPT reduces makespan** — real async event loop with controlled sleep
   durations, 6 programs on 2 servers, LPT measurably faster
7. **Online learning** — over 3 `_launch()` cycles, predictor warms up and
   subsequent batches get reordered (longest first)
8. **NaN safety** — degenerate Ridge training data doesn't crash or produce
   garbage predictions
9. **Outlier clipping** — extreme training outlier doesn't poison the heuristic

### Production validation path

To validate in production, compare throughput metrics between runs with and
without LPT:

```python
# In your launch script, enable LPT for treatment runs:
from gigaevo.evolution.scheduling import LPTPrioritizer, SimpleHeuristicPredictor

predictor = SimpleHeuristicPredictor()
prioritizer = LPTPrioritizer(predictor)
dag_runner = DagRunner(..., prioritizer=prioritizer)
```

Then compare `programs_ingested / wall_time` between FIFO and LPT runs.
The predictor learns online — throughput should improve after the first
~5 program evaluations as the predictor warms up.

## Files

| File | Purpose |
|---|---|
| `gigaevo/evolution/scheduling/__init__.py` | Public API re-exports |
| `gigaevo/evolution/scheduling/feature_extractor.py` | FeatureExtractor protocol + CodeFeatureExtractor |
| `gigaevo/evolution/scheduling/predictor.py` | EvalTimePredictor ABC + 3 implementations |
| `gigaevo/evolution/scheduling/prioritizer.py` | ProgramPrioritizer ABC + FIFO/LPT/SJF |
| `gigaevo/runner/dag_runner.py` | Integration point (prioritize in _launch, learn in _execute_dag) |
| `benchmarks/bench_lpt.py` | Discrete-event scheduling benchmark |
| `tests/evolution/test_scheduling.py` | 37 unit tests |
| `tests/evolution/test_scheduling_integration.py` | 15 integration tests |
| `docs/lpt_scheduling.md` | This file |
