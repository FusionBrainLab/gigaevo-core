# GigaEvo Tools

**Run format** (all operational and analysis tools): `prefix@db[:label]`
where `prefix` = `problem.name` from the Hydra config (e.g. `chains/hotpotqa/static`).

## Prerequisites

All tool commands use the project venv and require `PYTHONPATH=.`:

```bash
PYTHONPATH=. $GIGAEVO_PYTHON tools/<tool>.py ...
```

Shell scripts use `$GIGAEVO_PYTHON` (falls back to `python3`):

```bash
export GIGAEVO_PYTHON=python3  # adjust for your environment
```

Protocol gates to run before launch and before merge:

```bash
bash tools/experiment/check_phase_order.sh <experiment-name>   # pre-launch (Phase 4)
bash tools/experiment/check_experiment_complete.sh <experiment-name>  # pre-merge (Phase 5)
```

## Quick Reference

| Task | Tool | Command |
|---|---|---|
| Live status (experiment) | `status.py` | `PYTHONPATH=. python tools/status.py --experiment task/name` |
| Live status (single run) | `status.py` | `PYTHONPATH=. python tools/status.py --run prefix@db:label` |
| Gen-by-gen trajectory | `trajectory.py` | `PYTHONPATH=. python tools/trajectory.py --run prefix@db:label` |
| Top N programs | `top_programs.py` | `PYTHONPATH=. python tools/top_programs.py --run prefix@db:label -n 10` |
| Evolutionary lineage | `lineage.py` | `PYTHONPATH=. python tools/lineage.py --run prefix@db:label --top-n 1` |
| Fitness curves plot | `comparison.py` | `PYTHONPATH=. python tools/comparison.py --run prefix@db:label ... --output-folder /tmp/` |
| Export full CSV | `redis2pd.py` | `PYTHONPATH=. python tools/redis2pd.py --run prefix@db:label --output-file /tmp/o.csv` |
| Export frontier CSV | `redis2pd.py` | `PYTHONPATH=. python tools/redis2pd.py --run prefix@db:label --frontier-csv --output-file /tmp/f.csv` |
| Archive + upload | `archive_run.sh` | `bash tools/experiment/archive_run.sh --exp <name> --run "prefix@db:label" --upload` |
| Kill workers + flush | `flush.py` | `PYTHONPATH=. python tools/flush.py --db N [--confirm]` |
| Task-specific tools | `experiments/<task>/<name>/tools/` | e.g. `experiments/hotpotqa/val_gap/tools/gap_analysis.py` |

---

## Monitoring a Running Experiment

### `status.py` — Live run status

Shows generation, all metrics from `metrics.yaml`, invalidity rate, validator timing, and PID liveness.
Reads `metrics.yaml` from the problem directory to discover metric names and formatting (percentage vs raw value).

```bash
# From experiment manifest (recommended — auto-discovers runs, PIDs, watchdog, metrics)
PYTHONPATH=. python tools/status.py --experiment hover/prompt_coevolution

# Manual: one run (defaults to "fitness" metric)
PYTHONPATH=. python tools/status.py --run chains/hotpotqa/static@4:O

# Manual: multiple runs with PID and watchdog check
PYTHONPATH=. python tools/status.py \
    --run chains/hotpotqa/static@4:O \
    --run chains/hotpotqa/static_r@7:R \
    --pid O:3054746 --pid R:3054747 \
    --watchdog 3057704
```

Output (with `--experiment` — shows all metrics per problem):
```
Run       DB    Gen     Fitness  Prompt Length  Invalid%    Val dur(s)    Keys         PID  Status
------------------------------------------------------------------------------------------------------
C1         9      3      76.2%              ?        0%       639/980     157       49341  ALIVE
C2        10      4      75.8%              ?       20%      654/1223     158       49342  ALIVE
P1        11      4      25.0%          299.0        0%             ?     169       49343  ALIVE
P2        12      4      25.0%          299.0        0%             ?     169       49344  ALIVE

Watchdog PID 50073: ALIVE
```

Column notes:
- **Metric columns** — auto-discovered from `problems/{problem_name}/metrics.yaml`. Fractional metrics (upper_bound=1.0) show as percentages; others show raw values. Decimal precision from `metrics.yaml`.
- **Invalid%** — fraction of programs that failed validation; >75% at gen 3+ = stage_timeout too short
- **Val dur(s)** — validator stage mean/max duration in seconds (last 20 evaluations)

**Watchdog**: each experiment has `experiments/<task>/<name>/run_watchdog.py`, launched at
experiment start and kept alive throughout. The watchdog posts hourly PR comments.

---

### `trajectory.py` — Gen-by-gen trajectory (text mode)

Prints a gen-by-gen table of best (frontier), mean fitness, and valid program count.
Lightweight — reads metrics history keys directly, no full program fetch.

```bash
# Full trajectory
PYTHONPATH=. python tools/trajectory.py --run chains/hotpotqa/static@4:O

# Last 10 gens only
PYTHONPATH=. python tools/trajectory.py --run chains/hotpotqa/static@4:O --tail 10
```

Output:
```
Trajectory: O  (prefix=chains/hotpotqa/static, db=4)

Gen  1: best=42.3%  mean=39.1%  n_valid=  6
Gen  2: best=55.2%  mean=43.7%  n_valid=  7
...
Gen 42: best=66.0%  mean=57.3%  n_valid=  5

  Last improvement: gen 38 (63.4% → 66.0%, +2.6pp)
  Acceptance rate (gens 33–42): 8.5% (5 improvements / 59 valid programs)
```

Acceptance rate note: numerator = number of gens (in last 10) where the frontier improved
(0–1 per gen); denominator = total valid programs in those gens summed. This is a
per-valid-program improvement rate, not a per-mutation rate — invalid programs are excluded
from the denominator.

---

## Ending a Run

> **Required order** (skipping steps loses data permanently):
> 1. Run test evaluations → `bash experiments/<task>/<name>/run_test_eval.sh`
> 2. Archive all runs → `bash tools/experiment/archive_run.sh --exp <name> --run "prefix@db:label" --upload`
> 3. Flush Redis → `PYTHONPATH=. python tools/flush.py --db N --confirm`

### `run_test_eval.sh` — Test evaluation (per-experiment)

Each experiment has `experiments/<task>/<name>/run_test_eval.sh`. Run it while Redis is live
(before archiving or flushing). It evaluates the best-by-val program from each run on
the held-out test set and writes results to `test_evals/results.json`.

```bash
export GIGAEVO_PYTHON=python3  # adjust for your environment
bash experiments/hotpotqa/push/run_test_eval.sh
```

Preflight: verifies thinking mode on all chain endpoints before evaluating.
Results: `experiments/<task>/<name>/test_evals/results.json` (one entry per run).

---

### `archive_run.sh` — Archive and upload run data

**Run this before flushing Redis or rebooting. Redis is ephemeral — data not exported is gone.**

```bash
# Dry run: export locally only (verify output first)
bash tools/experiment/archive_run.sh --exp hotpotqa/push --run "chains/hotpotqa/static_f1_600@10:C"

# Export and upload to GitHub Release exp/hotpotqa/push
bash tools/experiment/archive_run.sh --exp hotpotqa/push --run "chains/hotpotqa/static_f1_600@10:C" --upload

# Archive all 4 runs
for SPEC in "chains/hotpotqa/static_f1@8:A" "chains/hotpotqa/static@9:B" \
            "chains/hotpotqa/static_f1_600@10:C" "chains/hotpotqa/static_f1_600@11:D"; do
  bash tools/experiment/archive_run.sh --exp hotpotqa/push --run "$SPEC" --upload
done
```

Each archive (`<label>_archive.tar.gz` on the GitHub Release) contains:
- `evolution_data.csv` — all programs, all generations, all metrics
- `programs/*.py` — source code of every evaluated program
- `top50.json` — top 50 programs with full metadata

Also uploads `environment.txt` (pip freeze, OS, GPU) once per experiment.

---

### `flush.py` — Safe Redis flush

Kills stale exec_runner workers first, then flushes each DB, then verifies 0 keys remain.
**Never flush manually with `redis-cli FLUSHDB` or `FLUSHALL`** — workers will repopulate Redis immediately.

```bash
# Preview (dry-run, default)
PYTHONPATH=. python tools/flush.py --db 0 1 2 3

# Execute (kills workers first, then flushes)
PYTHONPATH=. python tools/flush.py --db 0 1 2 3 --confirm
```

---

## Analyzing Results

### `top_programs.py` — Inspect top programs

```bash
# Top 5 by fitness (default)
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@4:O

# Top 1 with full code (the program to run test eval on)
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@4:O -n 1 --code

# Save top-3 source files to disk
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@4:O -n 3 --save-dir top_k/

# JSON output for scripting
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@4:O -n 1 --json
```

---

### `comparison.py` — Fitness curve plots

Plots rolling fitness vs iteration across multiple runs. Always emits all three formats
(png/pdf/svg). Output folder is created automatically. Default backend is headless (Agg) — no
display required.

```bash
PYTHONPATH=. python tools/comparison.py \
    --run chains/hotpotqa/static@4:O \
    --run chains/hotpotqa/static_r@7:R \
    --run chains/hotpotqa/static_r@6:Q \
    --run chains/hotpotqa/static_r@5:F \
    --output-folder experiments/hotpotqa/val_gap/plots/

# Interactive display (requires a display server)
PYTHONPATH=. python tools/comparison.py --run ... --output-folder /tmp/ --show
```

Output files: `evolution_runs_comparison.{png,pdf,svg}` in the output folder.

---

### `redis2pd.py` — Export evolution data to CSV

```bash
# Full program history (all programs, all metrics)
PYTHONPATH=. python tools/redis2pd.py \
    --run chains/hotpotqa/static@4:O \
    --output-file experiments/hotpotqa/val_gap/archives/O/evolution_data.csv

# Frontier-only CSV (gen,best_val) — for 05_results.md tables
PYTHONPATH=. python tools/redis2pd.py \
    --run chains/hotpotqa/static@4:O \
    --frontier-csv \
    --output-file experiments/hotpotqa/val_gap/frontier_O.csv
```

Frontier CSV format (paste directly into results tables). Dense format — one row per gen,
including gens without frontier improvement (value carries forward from last improvement):

```
gen,best_val
1,0.423
5,0.552
...
42,0.660
```

Legacy args (`--redis-db` / `--redis-prefix`) still work for `archive_run.sh` compatibility.

---

### `lineage.py` — Evolutionary ancestry trace

Traces the ancestor chain of a program back to the root seed. Useful for Phase 5 "Lessons
Learned" — which mutations led to the best result?

```bash
# Trace best program by fitness
PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --top-n 1

# Trace specific program by ID prefix
PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --program abc12345

# Limit depth to 5 ancestor hops
PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --top-n 1 --depth 5
```

Output:
```
Lineage of program abc12345 (run O, gen 32, fitness=66.0%)

  gen 32  abc12345  fitness=66.0%  mutation: rewrite_answer_extraction_step6
  gen 29  def45678  fitness=63.4%  mutation: add_coreference_resolution_step4
  gen 25  ghi90123  fitness=62.7%  [SEED — ddce37b4]
```

---

## Protocol Gates

### `check_phase_order.sh` — Pre-launch gate

Verifies all required protocol documents exist, are committed, and are in the correct state.
Run as the first step of Phase 4 (before any code changes, before launch).

```bash
bash tools/experiment/check_phase_order.sh <experiment-name>
```

Exit 0 = safe to proceed. Exit 1 = do not launch.

---

### `check_experiment_complete.sh` — Pre-merge gate

Verifies all five experiment phases are complete before the PR is merged.

```bash
bash tools/experiment/check_experiment_complete.sh <experiment-name>
```

Checks: all phase docs committed, `02_review.md` APPROVED, GitHub Release assets uploaded,
`05_results.md` filled, `experiments/INDEX.md` entry added.

---

## Experiment Automation (`tools/experiment/`)

Tools for the experiment lifecycle (used by Claude Code skills).

| Tool | Purpose | Command |
|---|---|---|
| `manifest.py` | Load/update `experiment.yaml` programmatically | `from tools.experiment.manifest import load_manifest, update_manifest` |
| `preflight_check.py` | 20-check validation gate before launch | `PYTHONPATH=. python tools/experiment/preflight_check.py --experiment task/name` |
| `generate_launch.py` | Generate `launch.sh` from experiment.yaml | `PYTHONPATH=. python tools/experiment/generate_launch.py --experiment task/name` |
| `record_pids.py` | Record launched PIDs into experiment.yaml | `PYTHONPATH=. python tools/experiment/record_pids.py --experiment task/name --pids-file pids.txt --labels R1 R2` |
| `reset_status.py` | Force-reset experiment status (escape hatch) | `PYTHONPATH=. python tools/experiment/reset_status.py --experiment task/name --status implemented` |

---

## Visual Tools

### `dag_builder/` — Visual DAG pipeline builder

A React + FastAPI app for building execution pipelines visually. Drag-and-drop
stages, connect data flow edges, export as Python code or Hydra YAML config.

```bash
# Start both backend (port 8081) and frontend (port 8082)
bash tools/dag_builder/start.sh
```

See `tools/dag_builder/README.md` for full documentation.

---

## Benchmarking

### `benchmark.py` — Throughput benchmark runner

CLI wrapper that runs the benchmark test suite (`tests/benchmarks/`).

```bash
# Quick run with fakeredis
PYTHONPATH=. python tools/benchmark.py

# Full run with real Redis
PYTHONPATH=. python tools/benchmark.py --redis-url redis://localhost:6379/15 --full

# Also run profiler
PYTHONPATH=. python tools/benchmark.py --profile
```

### `profiler.py` — Redis and DAG throughput profiler

Measures throughput of Redis ops, program serialization, DAG construction,
stage execution, and concurrent workloads.

```bash
PYTHONPATH=. python tools/profiler.py --redis-url redis://localhost:6379/15
```

---

## Scaffolding

### `wizard` — Problem directory generator

Generates a complete problem directory from a YAML config (validate.py, metrics.yaml,
initial_programs/, task_description.txt). See `tools/wizard/` for documentation.

```bash
python -m tools.wizard my_config.yaml
python -m tools.wizard my_config.yaml --overwrite
python -m tools.wizard my_config.yaml --validate-only
```

---

## Appendix: Redis Key Reference

Every run uses one Redis DB (0–15), set via `redis.db=N`.
All keys are prefixed with `problem.name` (e.g. `chains/hotpotqa/static`).

### Key format

```
{prefix}:metrics:history:program_metrics:{metric_name}
```

Each metrics history key is a Redis **list**. Each entry is a JSON object:
```json
{"s": <step>, "t": <unix_timestamp>, "v": <value>, "k": "scalar"}
```

### Canonical metric keys

| What you want | Redis key (after prefix:metrics:history:program_metrics:) | How to read |
|---|---|---|
| **Current generation** | `{prefix}:run_state` (hash) | `hget … "engine:total_generations"` |
| **Best frontier value** | `valid_frontier_{metric}` | last entry `"v"` field |
| **Per-gen mean** | `valid_gen_{metric}_mean` | entries with `"s"` == gen; last `"v"` = mean |
| **n_valid per gen** | `valid_gen_{metric}_mean` | count entries with `"s"` == gen |
| **Total programs (cumulative)** | `programs_total_count` | last entry `"v"` field |
| **Valid programs (cumulative)** | `programs_valid_count` | last entry `"v"` field |
| **Frontier improvements** | `valid_frontier_{metric}` | list entries in order; `"s"` = iteration at improvement |

Where `{metric}` is the metric name from `problems/{problem_name}/metrics.yaml` (e.g. `fitness`, `prompt_length`).
Use `status.py --experiment` to auto-discover all metrics.

### Rules

1. **Never use `llen(valid_frontier_fitness)` as the generation count**, and never use
   `valid_iter_fitness_mean` last `"s"` as the canonical gen count. The canonical source is
   `{prefix}:run_state` hash field `engine:total_generations` (hget, not lindex). Written by
   `EvolutionEngine` after every generation; survives restarts.
2. **Never write ad-hoc Redis queries** to answer questions the tools already answer.
   If a tool gives wrong results, fix the tool — don't work around it with inline Python.
3. **Archive is in-memory only** — `archive`/`archive:reverse` Redis keys are NOT persisted.
   Export with `archive_run.sh` before flushing or rebooting.

### Generation count

`{prefix}:run_state` hash field `engine:total_generations` is the **canonical generation
count** (commit `e7648ab`). Use `hget {prefix}:run_state engine:total_generations` in
watchdogs, scripts, and status checks. Written by `EvolutionEngine` after every generation;
survives restarts. See `tools/status.py` line ~99 for the reference implementation.

`valid_iter_fitness_mean` last `"s"` is still useful for **trend analysis** (per-gen mean
fitness curves) but should NOT be used as the canonical gen count — under high throughput
it can lag 5+ generations behind the true count.

**Do NOT use log grep for gen count.** The pattern
`grep -c "Phase 1: Idle confirmed" run.log` is brittle: it depends on exact log message
format, silently returns 0 if the log path is wrong, and has caused watchdog crashes
in production. It is listed here only as a historical warning — do not use it.

---

## Experiment-specific tools

Task-specific tools (problem eval, cross-metric gap tables, custom analysis) live in
`experiments/<task>/<name>/tools/`, not in this directory. Each experiment that needs them
creates its own `tools/` subdirectory.

**Convention**:
- A tool goes in `tools/` if it works for **any** GigaEvo run (just needs `--run prefix@db`)
- A tool goes in `experiments/<task>/<name>/tools/` if it imports problem-specific code,
  hardcodes experiment URLs/paths, or is only meaningful for one experiment's design

**Examples**:

| Tool | Location | Reason |
|---|---|---|
| `lineage.py` | `tools/` | Generic — works for any GigaEvo run |
| `gap_analysis.py` | `experiments/hotpotqa/val_gap/tools/` | Hardcodes O/R/Q/F run specs, gate criteria from 03_plan.md |
| `eval_checkpoint.py` | `experiments/hotpotqa/val_gap/tools/` | Imports HotpotQA chain infra; HotpotQA-specific |
