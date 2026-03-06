# Optuna Optimization Pipeline

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline Diagram](#pipeline-diagram)
- [Internal Flow](#internal-flow)
- [Budget System](#budget-system)
- [Configuration Reference](#configuration-reference)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Optuna optimization pipeline** uses an LLM to identify tuneable hyperparameters in evolved programs, then runs [Optuna](https://optuna.readthedocs.io/) TPE trials to find optimal values. Unlike CMA-ES (which tunes floating-point literals directly), Optuna supports integer, log-scale, and categorical parameters and lets the LLM decide *which* constants matter most.

Use the Optuna pipeline when:

- Programs have discrete or categorical knobs (solver methods, iteration counts, boolean toggles)
- You want the LLM to reason about which parameters to tune
- The search space is heterogeneous (mix of floats, ints, categoricals)

Use CMA-ES (`pipeline=cma_opt`) when programs only have floating-point constants and you want fast, gradient-free numerical optimization without LLM involvement.

---

## Quick Start

**Default run** (auto-computes budget, eval timeout, and trial count):

```bash
python run.py problem.name=heilbron pipeline=optuna_opt
```

**Custom time budget** (30 min optimization, 40 min DAG timeout):

```bash
python run.py problem.name=heilbron pipeline=optuna_opt \
    optimization_time_budget=1800 \
    dag_timeout=2400
```

**Explicit trials and timeout** (override auto-computation):

```bash
python run.py problem.name=heilbron pipeline=optuna_opt \
    '_optuna_stage_kwargs={eval_timeout: 60, n_trials: 80}'
```

**With a multi-island experiment**:

```bash
python run.py problem.name=heilbron pipeline=optuna_opt \
    experiment=multi_island_complexity
```

---

## Pipeline Diagram

The Optuna pipeline extends the default DAG with an optimization stage and a payload bypass path. When Optuna succeeds, the expensive `CallProgramFunction` is skipped entirely.

```
                                           ┌─────────────────────┐
                                           │  OptunaPayloadBridge │
                                           │  (extract best       │
                                           │   program output)    │
                                           └──────────┬──────────┘
                                                      │
                                                      ↓
┌──────────────────┐  success  ┌──────────────────┐        ┌─────────────────┐
│ ValidateCodeStage ├─────────→│  OptunaOptStage   │───────→│ PayloadResolver │
└──────────────────┘           └────────┬─────────┘        └────────┬────────┘
                                        │ failure                   │
                                        ↓                          ↓
                               ┌──────────────────┐    ┌────────────────────────┐
                               │CallProgramFunction├───→│  CallValidatorFunction  │
                               └──────────────────┘    └────────────────────────┘
```

**Two execution paths:**

1. **Optuna succeeds** — `OptunaOptStage` → `OptunaPayloadBridge` → `PayloadResolver` → `CallValidatorFunction`. The best trial's program output is forwarded directly; `CallProgramFunction` is skipped.
2. **Optuna fails** — `CallProgramFunction` runs the (possibly modified) code and feeds its output through `PayloadResolver` → `CallValidatorFunction`.

If the problem has a `context.py`, an `AddContext` stage is wired automatically into both `OptunaOptStage` and `CallProgramFunction`.

---

## Internal Flow

`OptunaOptStage.compute()` runs in six phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OptunaOptStage.compute()                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 0 ─── Resolve context                                       │
│              Extract context dict from DAG inputs (if contextual)   │
│                          │                                          │
│                          ↓                                          │
│  Phase 1 ─── Measure baseline runtime                               │
│              Run original code once with hardcoded constants        │
│              → baseline_runtime_s (float or None)                   │
│                          │                                          │
│                          ↓                                          │
│  Phase 1b ── Adaptive budget resolution                             │
│              Derive eval_timeout and n_trials from baseline +       │
│              optimization_time_budget (skipped if explicit values)  │
│                          │                                          │
│                          ↓                                          │
│  Phase 2 ─── LLM analysis → search space proposal                  │
│              Send numbered code + runtime info to LLM               │
│              → OptunaSearchSpace (parameters + code patches)        │
│              Apply patches → parameterized_code                     │
│                          │                                          │
│                          ↓                                          │
│  Phase 3 ─── Optuna trials                                          │
│              n_startup random trials + n_trials TPE trials          │
│              Up to max_parallel concurrent evaluations              │
│              Dynamic importance freezing at 33% and 75% of TPE     │
│              Early stopping if configured                           │
│                          │                                          │
│                          ↓                                          │
│  Phase 4 ─── Desubstitution                                         │
│              Replace _optuna_params["name"] with best literal       │
│              values → clean optimized_code                          │
│                          │                                          │
│                          ↓                                          │
│  Phase 5 ─── Update program code                                    │
│              program.code = optimized_code (if update_program_code) │
│                          │                                          │
│                          ↓                                          │
│  Phase 6 ─── Return OptunaOptimizationOutput                        │
│              optimized_code, best_scores, best_params, n_trials,    │
│              search_space_summary, best_program_output              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Budget System

Two user-facing parameters control the time budget. Everything else is derived automatically.

### Top-Level Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dag_timeout` | `3600` (1 hour) | Total time limit for the entire DAG pipeline |
| `optimization_time_budget` | `null` → `0.75 * dag_timeout` | Time budget for the Optuna stage itself |

Set these in `config/constants/pipeline.yaml` or via CLI overrides:

```bash
python run.py dag_timeout=2400 optimization_time_budget=1800
```

### Derived Values

When `eval_timeout` and `n_trials` are not explicitly set, the stage auto-computes them from the optimization budget and baseline runtime.

**`compute_eval_timeout(baseline_s, budget)`**

```
if baseline is known:
    raw = baseline_s * 3.0          (3x safety multiplier)
else:
    raw = budget * 0.05             (5% of total budget)

eval_timeout = clamp(raw, min=30, max=budget/2)
```

**`compute_n_trials(budget, eval_timeout, max_parallel)`**

```
usable       = budget - 60          (subtract LLM overhead)
n_rounds     = floor(usable / eval_timeout)
raw          = n_rounds * max_parallel
n_trials     = clamp(raw, min=20, max=100)
```

**Startup trials** (always added on top of `n_trials`):

```
n_startup = clamp(n_trials // 2, min=10, max=25)
total_trials = n_startup + n_trials
```

### Example Budget Table

| Baseline Runtime | Budget | eval_timeout | n_trials | n_startup | Total Trials |
|-----------------|--------|-------------|----------|-----------|-------------|
| 0.5s | 2700s | 30s | 100 | 25 | 125 |
| 5s | 2700s | 30s | 100 | 25 | 125 |
| 30s | 2700s | 90s | 100 | 25 | 125 |
| 120s | 2700s | 360s | 70 | 25 | 95 |
| unknown | 2700s | 135s | 100 | 25 | 125 |
| 0.5s | 1800s | 30s | 100 | 25 | 125 |
| 30s | 1800s | 90s | 100 | 25 | 125 |
| 120s | 1800s | 360s | 40 | 20 | 60 |

---

## Configuration Reference

### Pipeline YAML (`config/pipeline/optuna_opt.yaml`)

```yaml
# @package _global_
evolution_context:
  _target_: gigaevo.entrypoint.evolution_context.EvolutionContext
  problem_ctx: ${problem_context}
  llm_wrapper: ${ref:llm}
  storage: ${redis_storage}

pipeline_builder:
  _target_: gigaevo.entrypoint.default_pipelines.OptunaOptPipelineBuilder
  ctx: ${evolution_context}
  dag_timeout: ${dag_timeout}
  optimization_time_budget: ${optimization_time_budget}

dag_blueprint:
  _target_: gigaevo.config.helpers.build_dag_from_builder
  builder: ${pipeline_builder}
```

### Constants YAML (`config/constants/pipeline.yaml`)

```yaml
# @package _global_
stage_timeout: 600               # lightweight stages only
dag_timeout: 3600
optimization_time_budget: null   # null = 0.75 * dag_timeout
dag_concurrency: 16
max_code_length: 30000
max_insights: 8
```

### OptunaOptimizationConfig Fields

These are set via the `config` parameter of `OptunaOptimizationStage`. Override them through `_optuna_stage_kwargs` in a pipeline builder subclass or via Hydra structured configs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `random_state` | `int \| None` | `None` | Random seed for reproducible runs |
| `n_startup_trials` | `int \| None` | `None` → `clamp(n_trials//2, 10, 25)` | Random trials before TPE begins |
| `multivariate` | `bool` | `True` | Multivariate TPE (models parameter correlations) |
| `max_params` | `int \| None` | `None` → `clamp(n_trials//15, 3, 5)` | Max parameters the LLM should propose |
| `importance_freezing` | `bool` | `True` | Enable dynamic importance freezing |
| `early_tpe_fraction` | `float` | `0.333` | Fraction of TPE phase for early importance check |
| `late_tpe_fraction` | `float` | `0.75` | Fraction of TPE phase for late importance check |
| `early_threshold_multiplier` | `float` | `0.5` | Conservative threshold for early check (0.5x) |
| `ped_anova_early_quantile` | `float` | `0.25` | PED-ANOVA quantile for early check |
| `ped_anova_late_quantile` | `float` | `0.10` | PED-ANOVA quantile for late check |
| `importance_check_at` | `int \| None` | `None` (auto) | Trial count for early importance check |
| `importance_check_late_at` | `int \| None` | `None` (auto) | Trial count for late importance check |
| `min_trials_for_importance` | `int` | `20` | Minimum completed trials before importance runs |
| `importance_threshold_ratio` | `float` | `0.1` | Relative threshold for freezing (fraction of avg importance) |
| `importance_absolute_threshold` | `float` | `0.01` | Absolute importance floor for freezing |
| `early_stopping_patience` | `int \| None` | `None` | Stop after N consecutive non-improving trials |

### Stage Constructor Parameters

These are passed directly to `OptunaOptimizationStage.__init__()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `MultiModelRouter` | (required) | LLM wrapper for search space proposal |
| `validator_path` | `Path` | (required) | Path to `validate.py` |
| `score_key` | `str` | (required) | Key in validator result dict to optimize |
| `minimize` | `bool` | `False` | Minimize score (default: maximize) |
| `n_trials` | `int \| None` | `None` (auto) | Number of TPE trials |
| `max_parallel` | `int` | `8` | Max concurrent trial evaluations |
| `eval_timeout` | `int \| None` | `None` (auto) | Timeout per trial evaluation (seconds) |
| `function_name` | `str` | `"run_code"` | Function to call in the program |
| `validator_fn` | `str` | `"validate"` | Function to call in the validator |
| `update_program_code` | `bool` | `True` | Overwrite `program.code` with optimized result |
| `add_tuned_comment` | `bool` | `True` | Add `# tuned (Optuna)` comments |
| `task_description` | `str \| None` | `None` | Task context forwarded to the LLM |
| `optimization_time_budget` | `float \| None` | `None` | Time budget (falls back to stage timeout) |
| `config` | `OptunaOptimizationConfig` | defaults | Advanced config (see table above) |

### CLI Overrides

Override stage kwargs via the `_optuna_stage_kwargs` dict syntax:

```bash
# Set explicit trial count and timeout
python run.py pipeline=optuna_opt \
    '_optuna_stage_kwargs={n_trials: 80, eval_timeout: 60}'

# Set explicit max_parallel
python run.py pipeline=optuna_opt \
    '_optuna_stage_kwargs={max_parallel: 4}'
```

---

## Advanced Features

### Dynamic Feature Importance (Two-Phase PED-ANOVA Freezing)

When `importance_freezing=True` (default), the stage evaluates parameter importance at two checkpoints during TPE and freezes low-impact parameters to their current best values.

```
                  n_startup            early check         late check
                  (random)             (~33% TPE)          (~75% TPE)
    ─────────────┼───────────────────┼───────────────────┼──────────→ trials
                  │                   │                   │
                  │ TPE begins        │ Conservative      │ Standard
                  │                   │ threshold (0.5x)  │ threshold (1.0x)
                  │                   │ quantile = 0.25   │ quantile = 0.10
```

**How freezing works:**

1. PED-ANOVA evaluates each parameter's discriminative power on score
2. Parameters below the threshold (relative or absolute) are frozen to their current best value
3. Frozen parameters are still recorded in trials (via `study.enqueue_trial`) so TPE's internal model stays consistent
4. If all parameters are frozen, optimization stops early

**Requirements:**

- At least 3 parameters (`_MIN_PARAMS_FOR_IMPORTANCE = 3`)
- At least 20 completed trials (`_IMPORTANCE_CHECK_MIN_TRIALS = 20`)
- At least 15 post-startup trials before the first check (`_MIN_POST_STARTUP_TRIALS = 15`)

### Early Stopping

Set `early_stopping_patience` to stop after N consecutive trials without improvement:

```bash
python run.py pipeline=optuna_opt \
    '_optuna_stage_kwargs={config: {early_stopping_patience: 30}}'
```

This is disabled by default (`None`).

### Payload Bypass

When Optuna succeeds, the best trial already captured the program's output. The pipeline uses `OptunaPayloadBridge` and `PayloadResolver` to forward this output directly to `CallValidatorFunction`, skipping the expensive `CallProgramFunction` re-execution.

The bypass works automatically via DAG execution dependencies:

- `CallProgramFunction` has an `on_failure("OptunaOptStage")` dependency — it only runs when Optuna fails
- `OptunaPayloadBridge` requires `OptunaOptStage` output as a data dependency — when Optuna fails, this becomes IMPOSSIBLE and the stage is automatically SKIPPED

### Trial Deduplication

Identical parameter combinations are detected and skipped (marked as PRUNED) to avoid wasting budget on redundant evaluations. This happens automatically and is not configurable.

### Constant Liar

The TPE sampler uses `constant_liar=True` by default, which enables concurrent trial suggestions. Without this, TPE would need to wait for each trial to complete before suggesting the next one, serializing the optimization.

---

## Troubleshooting

### Stage Timeouts

**Symptom:** `[Optuna][<pid>]` logs show the stage being killed before trials complete.

**Cause:** The stage timeout equals `optimization_time_budget` (default: `0.75 * dag_timeout`). If baseline measurement + LLM analysis + trials exceed this, the stage times out.

**Fix:**
- Increase `optimization_time_budget` and `dag_timeout`:
  ```bash
  python run.py dag_timeout=7200 optimization_time_budget=6000
  ```
- Or reduce trial count / eval timeout:
  ```bash
  python run.py '_optuna_stage_kwargs={n_trials: 40, eval_timeout: 30}'
  ```

### All Trials Fail

**Symptom:** `[Optuna][<pid>] No trials completed successfully; returning original code.`

**Cause:** The parameterized code crashes for every parameter combination. Common reasons:
- LLM introduced syntax errors in the parameterized code
- Parameter bounds cause runtime errors (e.g., `range(0)`, division by zero)
- `eval_timeout` is too short for the program

**Fix:**
- Check the failure reasons in DEBUG logs: `[Optuna][<pid>] Trial N/M pruned`
- Increase `eval_timeout` if trials are timing out
- The stage returns the original code unchanged, so the program is not harmed

### LLM Proposes Bad Parameters

**Symptom:** `[Optuna][<pid>] LLM analysis or patching failed` or `Parameterized code syntax error`.

**Cause:** The LLM produced overlapping line ranges, invalid syntax in patches, or parameter names that don't match `_optuna_params["name"]` references.

**Fix:** This is a single-attempt failure — the stage raises `RuntimeError` and the program falls through to `CallProgramFunction` with the original code. No user action is needed unless it happens consistently. If it does, check the LLM model and prompt templates.

### Reading Log Output

All Optuna logs use the format `[Optuna][<pid>]` where `<pid>` is the first 8 characters of the program ID.

Key log messages to look for:

| Log Level | Message Pattern | Meaning |
|-----------|----------------|---------|
| DEBUG | `Auto eval_timeout=Ns` | Adaptive timeout was computed |
| DEBUG | `Auto n_trials=N` | Adaptive trial count was computed |
| DEBUG | `Baseline runtime: N.NNs` | Baseline measurement succeeded |
| INFO | `Running N trials total (M random + K TPE)` | Optimization starting |
| INFO | `Progress: N/M trials run, best score=X` | Every 10 trials |
| INFO | `Freezing 'param_name'` | Importance freezing triggered |
| INFO | `Early stopping: no improvement for N trials` | Patience exhausted |
| INFO | `== Done == trials=N/M params=K score=X` | Optimization complete |
| WARNING | `No trials completed successfully` | All trials failed |

### Baseline Measurement Fails

**Symptom:** `[Optuna][<pid>] Baseline runtime measurement failed`

**Cause:** The program crashes with its original hardcoded constants. This is non-fatal — the stage proceeds with fallback timeout computation (`budget * 0.05`).

**Fix:** No action needed. The LLM is informed that baseline runtime is unavailable and adjusts its parameter proposals accordingly.

---

## Source Files

| File | Contents |
|------|----------|
| `gigaevo/programs/stages/optimization/optuna/stage.py` | Main stage: `compute()` flow, baseline measurement, LLM analysis, Optuna runner |
| `gigaevo/programs/stages/optimization/optuna/models.py` | `compute_eval_timeout()`, `compute_n_trials()`, `OptunaOptimizationConfig`, `ParamSpec` |
| `gigaevo/programs/stages/optimization/optuna/prompts.py` | LLM prompt templates for search space proposal |
| `gigaevo/programs/stages/optimization/optuna/routing.py` | `OptunaPayloadBridge`, `PayloadResolver` (bypass stages) |
| `gigaevo/programs/stages/optimization/optuna/desubstitution.py` | Parameter desubstitution (clean code generation) |
| `gigaevo/entrypoint/default_pipelines.py` | `OptunaOptPipelineBuilder` (pipeline wiring) |
| `config/constants/pipeline.yaml` | `dag_timeout`, `optimization_time_budget` defaults |
| `config/pipeline/optuna_opt.yaml` | Pipeline YAML config |
