# GigaEvo Tools

## CLI (`gigaevo`)

Installed via `pip install -e .` (console_scripts entry in pyproject.toml). Handles PYTHONPATH, run resolution from experiment.yaml, and output formatting automatically.

**Run format**: `prefix@db[:label]` where `prefix` = `problem.name` from the Hydra config (e.g. `chains/hotpotqa/static`).

Shell scripts use `$GIGAEVO_PYTHON` (falls back to `python3`):
```bash
export GIGAEVO_PYTHON=python3  # adjust for your environment
```

### Global Flags

| Flag | Description |
|------|-------------|
| `-e/--experiment TASK/NAME` | Experiment name — auto-discovers all runs, PIDs, watchdog from `experiment.yaml` |
| `-r/--run SPEC` | Manual run spec (repeatable). See "Run spec formats" below. |
| `-f/--format FORMAT` | Output format: `table` (default for terminal), `json`, `csv`, `markdown` |
| `-q/--quiet` | Suppress output |
| `-v/--verbose` | Verbose output |
| `--redis-host HOST` | Redis hostname (default: localhost) |
| `--redis-port PORT` | Redis port (default: 6379) |

#### Run spec formats (`-r`)

The `-r/--run` flag accepts several shorthand forms. Prefix is auto-discovered from the Redis DB's `{prefix}:__instance_lock__` key when omitted.

| Form | Example | Meaning |
|------|---------|---------|
| `prefix@db:label` | `chains/hover/static@4:O` | Full: explicit prefix, db, and display label |
| `prefix@db` | `chains/hover/static@4` | Full without label (label defaults to `prefix@db`) |
| `db` | `4` | Bare DB number — prefix auto-discovered from `:__instance_lock__` |
| `@db` | `@4` | Same as bare `db` |
| `db:label` | `4:O` | Bare DB with custom label |
| `/path/to/storage[:label]` | `outputs/run/storage:disk1` | Disk storage path (`storage=disk` runs) — must start with `/`, `./`, `../`, or `~` |

The auto-discover path fails if the DB is empty or contains multiple prefixes. Run `gigaevo inspect --db N` first to see what's there.

**Disk-path specs** point at the storage root written by a `storage=disk` run (Hydra default: `<output_dir>/storage`). Accepts either the root (its single `<prefix>/programs/` subdirectory is auto-discovered; errors if there are zero or several) or the prefix directory itself. The label defaults to the resolved prefix directory name. Disk specs are read-only and lock-free, so it's safe to inspect a live run. Supported by program-reading commands: `top`, `export csv/frontier`, `plot`. Redis-only commands (`status`, `trajectory`, `metrics`, `checkpoint`) reject disk specs with a usage error — their data (live PIDs, metric history) lives in the disk run's `metrics/*.jsonl` and log files instead.

### Commands

#### Monitoring

```bash
# Status — live run monitoring (gen, metrics, PIDs, watchdog)
gigaevo -e hover/my-exp status
gigaevo -r chains/hotpotqa/static@4:O status

# Trajectory — gen-by-gen fitness table
gigaevo -r chains/hotpotqa/static@4:O trajectory
gigaevo -r chains/hotpotqa/static@4:O trajectory --tail 10 --metric fitness

# Top programs — inspect best programs by fitness
gigaevo -r chains/hotpotqa/static@4:O top
gigaevo -r chains/hotpotqa/static@4:O top -n 1 --code --save-dir top_k/

# Logs — show evolution logs
gigaevo -e hover/my-exp logs
```

#### Plotting

```bash
# Fitness comparison across runs (png/pdf/svg)
gigaevo -e adversarial/adversarial-vs-solo plot comparison -o plots/
gigaevo -r A@4:A -r B@5:B plot comparison -o plots/ --paper --smoothing lowess

# Suppress cummax frontier for adversarial Improver runs
gigaevo -e ... plot comparison -o plots/ --no-frontier-for D1,D2
gigaevo -e ... plot comparison -o plots/ --no-frontier  # suppress for ALL runs

# Annotate frontier jumps
gigaevo -e ... plot comparison -o plots/ --annotate-frontier --max-annotations 5

# Single-run trajectory plot
gigaevo -r chains/hover/static@4:O plot trajectory -o plots/ --pdf

# Arms-race dual-panel plot (Constructor top, Improver bottom)
gigaevo -e ... plot arms-race -o plots/ --paired C1_A:C1_B --paper
gigaevo -e ... plot arms-race -o plots/ --paired C1_A:C1_B,C2_A:C2_B --show-max
```

#### Data Export

```bash
# Full evolution data to CSV
gigaevo -r chains/hotpotqa/static@4:O export csv -o data/evolution.csv

# Frontier-only CSV (gen, best_val)
gigaevo -r chains/hotpotqa/static@4:O export frontier -o data/frontier.csv --metric fitness
```

#### Metrics — raw Redis metric dump

```bash
# Stream all metric records, one per line: <tag>\tstep=<n>\twall=<iso>\tvalue=<v>
gigaevo -r heilbron@0 metrics | grep tokens

# Filter by glob, tail last N records per tag
gigaevo -r heilbron@0 metrics --tag "valid/frontier/*"
gigaevo -r heilbron@0 metrics --tag "*tokens*" --tail 10

# Bounded step window, TSV output
gigaevo -r heilbron@0 metrics --since 50 --until 100 --format tsv

# Inspect a known histogram by exact tag (not enumerable from the `latest` hash)
gigaevo -r heilbron@0 metrics --kind hist --tag valid/iter/duration_s
```

Read-only. See "Redis Data Model" below for how metrics are keyed.

#### Events — canonical-event analytics

```bash
# Validate canonical events against the Pydantic registry
gigaevo -e hover/my-exp events audit

# Per-run event plots + summary.md (registry-driven)
gigaevo -e hover/my-exp events plot -o plots/events/

# Grouped plot (e.g., G vs D arms)
gigaevo -e ... events plot --paired G1:D1,G2:D2 -o plots/events/
```

#### Profiler — log → text summary + HTML dashboard

```bash
# Every run in the manifest
gigaevo -e hover/my-exp profiler

# One run (by run label)
gigaevo -e hover/my-exp profiler G1

# Multiple runs
gigaevo -e hover/my-exp profiler G1 D1

# Arbitrary log file (no experiment.yaml required)
gigaevo profiler --file /path/to/runner.log --out-dir /tmp/profile/
```

Outputs two files per run: `profile_<label>.txt` (terminal summary) and `profile_<label>.html` (interactive dashboard with caption + hover + stable per-tag colors). Default output directory is `experiments/<exp>/profiler/`.

#### Operations

```bash
# Inspect — discover which experiment prefix(es) live in a Redis DB
gigaevo inspect --db 4                     # single DB
gigaevo inspect --db 1 --db 2 --db 3       # multiple DBs

# Flush Redis DBs (kills workers first)
gigaevo flush --db 4 5 --confirm           # execute
gigaevo flush --db 4 5                     # dry-run (default)

# Checkpoint — status + notify (for experiment monitoring)
gigaevo -e hover/my-exp checkpoint

# Watchdog — start watchdog engine
gigaevo -e hover/my-exp watchdog
```

#### Manifest (read/write experiment.yaml)

```bash
# Read fields (dotted paths supported)
gigaevo -e hover/my-exp manifest get status
gigaevo -e hover/my-exp manifest get control_plane.watchdog_pid
gigaevo -e hover/my-exp manifest get runs --format json

# Write fields
gigaevo -e hover/my-exp manifest update status running        # state-machine validated
gigaevo -e hover/my-exp manifest update control_plane.watchdog_pid 12345

# Gate checks (exit non-zero if gate not satisfied)
gigaevo -e hover/my-exp manifest gate implemented
gigaevo -e hover/my-exp manifest gate running

# Generate and push PR description
gigaevo -e hover/my-exp manifest pr-description --push
```

#### Lifecycle (experiment management)

```bash
# Launch — checks + start runs (use experiment-launch skill for full workflow)
gigaevo -e hover/my-exp launch

# Closeout — archive + analyze + update PR
gigaevo -e hover/my-exp closeout --confirm

# Restart — kill runs + flush + re-launch
gigaevo -e hover/my-exp restart --confirm
```

---

## Tool Index

### General Tools (`tools/`)

Most read/write/plot functionality previously lived as standalone scripts here; it has been consolidated into the unified `gigaevo` CLI (see Commands above). What remains in `tools/` is helper modules and infrastructure scripts.

| Tool | Purpose | Key flags |
|---|---|---|
| `lineage.py` | Trace evolutionary ancestry chain back to seed | `python -m tools.lineage --run`, `--top-n 1`, `--depth N` |
| `profiler.py` | Log → text summary + HTML dashboard (called by `gigaevo profiler`) | invoked via `gigaevo profiler`; importable as `from tools.profiler import Profiler` |
| `resource_manager.py` | Auto-detect available GPU servers and free Redis DBs; assign runs to servers/DBs | `--check`, `--experiment task/name` |
| `telegram_notify.py` | Send Telegram notifications and wait for async approval at experiment gates | `import` — not a CLI tool |
| `utils.py` | Shared utilities: `parse_run_arg`, Redis helpers | imported by other tools |

The former scripts (`status.py`, `trajectory.py`, `top_programs.py`, `comparison.py`, `redis2pd.py`, `flush.py`, `fitness_vs_time.py`, `pareto_plot.py`, `throughput_plot.py`, `csv_memory_comparison.py`) now live under `gigaevo/cli/` and are invoked as `gigaevo <subcommand>`.

### Live Monitoring (`gigaevo/monitoring/`)

Library-level helpers a runner can start to surface metrics to the terminal while an experiment is in flight. Not CLI tools — embed via `from gigaevo.monitoring import ...`.

| Symbol | Purpose |
|---|---|
| `start_live_profiler(log_path, out_dir, ...)` | Periodically re-profile a runner's log file and refresh `profile_<label>.{txt,html}` on disk (default interval 60s) |
| `start_live_frontier_compare(redis_url, key_prefix, metrics, ...)` | Stream side-by-side frontier comparison across runs/metrics into the terminal — sibling of `live_profiler` for multi-run experiments |

### Experiment Lifecycle Tools (`tools/experiment/`)

Depend on `experiment.yaml`, protocol docs, or PRs. Used by Claude Code skills.

| Tool | Purpose | Key flags |
|---|---|---|
| `archive_run.sh` | Export Redis data to local files + upload as GitHub Release asset | `--exp task/name`, `--run "prefix@db:label"`, `--upload` |
| `check_phase_order.sh` | Pre-launch gate: verify protocol docs, experiment.yaml, launch.sh, N>=2 | `<experiment-name>` |
| `check_experiment_complete.sh` | Pre-merge gate: verify all 5 phases, archives, release assets, INDEX.md | `<experiment-name>` |
| ~~`preflight_check.py`~~ | Replaced by `gigaevo.experiment.checks` (10 principled checks) | `gigaevo -e task/name launch` |
| ~~`generate_launch.py`~~ | Replaced by `gigaevo.experiment.launch_generator` (called by `gigaevo launch`) | `gigaevo -e task/name launch` |
| `flush --kill-only` | Kill stale watchdog / run processes | `gigaevo flush --db N1 N2 ... --kill-only` |
| `check_all_watchdogs.sh` | Cron health check: scan Redis heartbeats, alert on stale watchdogs | standalone (no args) |
| `skill_env.sh` | Shared env vars for skills (`$PROJ`, `$GIGAEVO_PYTHON`, `$PYTHONPATH`) | `source` — not executable |

### Infrastructure Tools

| Tool | Purpose |
|---|---|
| `litellm.sh` | Start/stop/status LiteLLM proxy for chain server load balancing |
| `litellm_bench.py` | Benchmark LiteLLM proxy (latency, throughput, error rate) |
| `llm_contention_bench.py` | Measure LLM server contention under concurrent load |

### Benchmarking Tools

| Tool | Purpose |
|---|---|
| `benchmark.py` | Run throughput benchmark suite (`tests/benchmarks/`) |
| `bench_snapshot.py` | Before/after benchmark snapshots for comparison |
| `benchmark_capture.py` | Capture benchmark results to `benchmark_history.jsonl` |
| `canonical_benchmark/run_benchmark.py` | **Regression benchmark — run on every major breaking change.** 5 problems × 2 seeds, spawning `python run.py problem.name=<P> redis.db=<N> hydra.run.dir=<DIR> llm_base_url=<URL> model_name=<NAME>`. The LLM endpoint must be supplied via required `--llm-base-url` / `--model-name` CLI args — no default is shipped because the framework default in `config/constants/endpoints.yaml` (OpenRouter Gemini-3-Flash) is too slow for a 10-run sweep and the right replacement is environment-specific. Reads framework defaults: `pipeline=standard num_parents=1 max_mutants=250` (intra-memory pipeline, no cross-population channel, no IdeaTracker). Extracts best fitness via `gigaevo top`, appends to `BENCHMARK_HISTORY.md`. Uplift sweeps: opt in via `--override ideas_tracker=default --override memory=local` and/or `--override pipeline=intra_extra_memory`. See `tools/canonical_benchmark/README.md`. |
| `profiler.py` | Redis ops, DAG construction, stage execution profiling |

### Scaffolding Tools

| Tool | Purpose |
|---|---|
| `dag_builder/` | Visual DAG pipeline builder (React + FastAPI): drag-drop stages, export YAML |
| `wizard/` | Problem directory generator from YAML config |

---

## Monitoring a Running Experiment

### `gigaevo status` — Live run status

Shows generation, all metrics from `metrics.yaml`, invalidity rate, validator timing, and PID liveness.
Reads `metrics.yaml` from the problem directory to discover metric names and formatting (percentage vs raw value).

```bash
# From experiment manifest (recommended — auto-discovers runs, PIDs, watchdog, metrics)
gigaevo -e hover/prompt_coevolution status

# Manual: one run
gigaevo -r chains/hotpotqa/static@4:O status

# Manual: multiple runs
gigaevo -r chains/hotpotqa/static@4:O -r chains/hotpotqa/static_r@7:R status
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

**Watchdog**: started via `gigaevo -e <task>/<name> watchdog` at experiment launch.
The watchdog posts hourly PR comments with status, plots, and stagnation alerts.

---

### `gigaevo trajectory` — Gen-by-gen trajectory (text mode)

Prints a gen-by-gen table of best (frontier), mean fitness, and valid program count.
Lightweight — reads metrics history keys directly, no full program fetch.

```bash
# Full trajectory
gigaevo -r chains/hotpotqa/static@4:O trajectory

# Last 10 gens only
gigaevo -r chains/hotpotqa/static@4:O trajectory --tail 10
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

---

## Ending a Run

> **Required order** (skipping steps loses data permanently):
> 1. Run test evaluations → `bash experiments/<task>/<name>/run_test_eval.sh`
> 2. Archive all runs → `bash tools/experiment/archive_run.sh --exp <name> --run "prefix@db:label" --upload`
> 3. Flush Redis → `gigaevo flush --db N --confirm`

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

### `gigaevo flush` — Safe Redis flush

Kills stale exec_runner workers first, then flushes each DB, then verifies 0 keys remain.
**Never flush manually with `redis-cli FLUSHDB` or `FLUSHALL`** — workers will repopulate Redis immediately.

```bash
# Preview (dry-run, default)
gigaevo flush --db 0 1 2 3

# Execute (kills workers first, then flushes)
gigaevo flush --db 0 1 2 3 --confirm
```

---

## Analyzing Results

### `gigaevo top` — Inspect top programs

```bash
# Top 5 by fitness (default)
gigaevo -r chains/hotpotqa/static@4:O top

# Top 1 with full code (the program to run test eval on)
gigaevo -r chains/hotpotqa/static@4:O top -n 1 --code

# Save top-3 source files to disk
gigaevo -r chains/hotpotqa/static@4:O top -n 3 --save-dir top_k/

# JSON output for scripting
gigaevo -r chains/hotpotqa/static@4:O top -n 1 --json
```

---

### `csv_comparison.py` - Compare CSV Exports

Compares multiple exported CSVs by plotting rolling fitness statistics over iterations.

**Usage:**
```bash
python -m tools.csv_comparison \
  --run "outputs/runA.csv:Run_A" \
  --run "outputs/runB.csv:Run_B" \
  --iteration-rolling-window 5 \
  --output-folder results/comparison
```

**Notes:**
- Uses the same plotting logic as `comparison.py`
- `--run` format is `path[:label]` (label defaults to filename stem)

---

### `gigaevo plot comparison` — Fitness curve plots

Plots rolling fitness vs iteration across multiple runs. Always emits all three formats
(png/pdf/svg). Output folder is created automatically. Default backend is headless (Agg) — no
display required.

```bash
gigaevo -r chains/hotpotqa/static@4:O \
    -r chains/hotpotqa/static_r@7:R \
    -r chains/hotpotqa/static_r@6:Q \
    -r chains/hotpotqa/static_r@5:F \
    plot comparison -o experiments/hotpotqa/val_gap/plots/
```

Output files: `evolution_runs_comparison.{png,pdf,svg}` in the output folder.

**Offline mode (`--from-csv`).** Plot directly from CSVs produced by
`gigaevo export csv` without touching Redis — useful for archived runs whose
Redis DB has been flushed, or for sharing data across machines:

```bash
# Export once
gigaevo -r chains/hotpotqa/static@4:O export csv -o archives/O.csv
gigaevo -r chains/hotpotqa/static_r@7:R export csv -o archives/R.csv

# Plot later, no Redis needed
gigaevo plot comparison \
    --from-csv archives/O.csv:Original \
    --from-csv archives/R.csv:Retrieval \
    -o plots/
```

Label defaults to the file stem when `:LABEL` is omitted. `--from-csv` is
mutually exclusive with `-r/--run` and `-e/--experiment`.

---

### `gigaevo export` — Export evolution data to CSV

```bash
# Full program history (all programs, all metrics)
gigaevo -r chains/hotpotqa/static@4:O export csv \
    -o experiments/hotpotqa/val_gap/archives/O/evolution_data.csv

# Frontier-only CSV (gen,best_val) — for 05_results.md tables
gigaevo -r chains/hotpotqa/static@4:O export frontier \
    -o experiments/hotpotqa/val_gap/frontier_O.csv
```

---

### `lineage.py` — Evolutionary ancestry trace

Traces the ancestor chain of a program back to the root seed. Useful for Phase 5 "Lessons
Learned" — which mutations led to the best result?

```bash
# Trace best program by fitness
gigaevo -r chains/hotpotqa/static@4:O lineage --top-n 1

# Trace specific program by ID prefix
gigaevo -r chains/hotpotqa/static@4:O lineage --program abc12345

# Limit depth to 5 ancestor hops
gigaevo -r chains/hotpotqa/static@4:O lineage --top-n 1 --depth 5
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
| `launch` | Preflight + generate script + exec + set running + spawn watchdog | `gigaevo -e task/name launch [--dry-run] [--skip-preflight]` |
| `manifest record-pids` | Record launched PIDs into experiment.yaml | `gigaevo -e task/name manifest record-pids --pids-file pids.txt --labels "A B"` |
| `manifest reset-status` | Force-reset experiment status (escape hatch) | `gigaevo -e task/name manifest reset-status implemented --reason '...'` |

---

## Experiment Manifest (`experiment.yaml`)

Every experiment has a single source of truth: `experiments/<task>/<name>/experiment.yaml`.
It is the machine-readable declaration of the experiment — Pydantic-validated, read by every
CLI command, skill, watchdog, and plot. `launch.sh` is **generated** from it (never hand-edited).

Load with `gigaevo.experiment.manifest.load_manifest(exp)` → Pydantic `ExperimentManifest`.
Readers access fields through the four canonical sub-sections (`m.contract.*`,
`m.lifecycle.*`, `m.telemetry.*`, `m.control_plane.*`). There are no flat
compatibility views.

### Schema (Pydantic `ExperimentManifest`, schema_version 2)

Source: `gigaevo/experiment/manifest.py`. Top-level sub-sections:

| Key | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | `int` | always | `2` (only currently supported version) |
| `contract` | `ContractSection` | always | Pre-registered identity, problem, runs, servers, config, stopping rule, baseline |
| `lifecycle` | `LifecycleState` | always | `status`, `launch`, `smoke_test`, `treatment_verification` |
| `telemetry` | `TelemetryLog` | no (defaults) | `checkpoints`, `mid_run_test_eval`, `checkpoint_analysis`, `treatment_checks` |
| `control_plane` | `ControlPlane` | no (defaults) | `watchdog`, `notifications`, `watchdog_pid`, `anomaly_detector_cron_id`, `checkpoint_cron_id` |

#### `contract` section

| Field | Type | Required | Purpose |
|---|---|---|---|
| `identity.name` | `str` | yes | `<task>/<short-name>`, e.g. `heilbron/asymmetric-iterations-v2` |
| `identity.task` | `str` | yes | Top-level task folder (e.g. `heilbron`, `hover`, `hotpotqa`) |
| `identity.branch` | `str` | `""` | Git branch hosting the experiment |
| `identity.pr_number` | `int \| None` | no | GitHub PR tracking the experiment |
| `identity.tracking_issue` | `int \| None` | no | GitHub Issue ID |
| `identity.prereg_commit` | `str \| None` | no | Git SHA of the pre-registration commit |
| `problem` | `ProblemSpec` | defaults | Test set, fitness type, metric name |
| `runs` | `list[RunSpec]` | gated | Required when `lifecycle.status ≥ implemented` |
| `servers` | `list[str]` | gated | Required when `lifecycle.status ≥ implemented` |
| `config` | `ConfigSpec` | gated | Typed standard keys + `extra: dict[str, Any]` for Hydra overrides |
| `custom_env` | `dict[str, str]` | no | Env vars exported in generated `launch.sh` |
| `max_generations` | `int` | `25` | Stopping-rule target. **Naming note**: kept in the manifest schema for backwards-compat, but emitted as `max_mutants=<N>` at launch (since v2.0.0, the engine-side knob is `max_mutants`). |
| `stopping_rule` | `StoppingRule` | no | Structured conditions (see `conditions[]`) + prose `description` |
| `baseline` | `BaselineInfo` | no | Reference / mean / metric for comparison |
| `tools` | `list[ToolRef]` | no | Experiment-specific tool registry |

#### `lifecycle` section

| Field | Type | Required | Purpose |
|---|---|---|---|
| `status` | `str` | yes | `preregistered`, `implemented`, `running`, `complete`, `invalid` |
| `launch.time` | `str \| None` | gated | ISO timestamp, required when `status ≥ running` |
| `launch.commit` | `str \| None` | gated | Git SHA at launch, required when `status ≥ running` |
| `launch.confirmed_at` | `str \| None` | no | Researcher confirmation timestamp |
| `launch.attempt` | `int \| None` | no | Launch attempt number |
| `smoke_test.completed` | `bool` | gated | Must be `true` for `status ≥ implemented` |
| `smoke_test.completed_at` | `str \| None` | no | Smoke-test completion timestamp |
| `treatment_verification.completed` | `bool` | no | Treatment checks recorded |
| `treatment_verification.alignment_check_completed` | `bool` | no | Implementation-aligner verdict recorded |

#### `telemetry` section

| Field | Type | Purpose |
|---|---|---|
| `checkpoints` | `list[CheckpointEntry]` | Appended by `/experiment-checkpoint` (gen, timestamp, run_metrics, notes) |
| `mid_run_test_eval` | `MidRunTestEvalInfo` | `completed`, `completed_at` |
| `checkpoint_analysis` | `CheckpointAnalysisInfo` | `mid_run.completed`, `mid_run.completed_at` |
| `treatment_checks` | `TreatmentChecksInfo` | `completed`, `completed_at`, `results[]` |

#### `control_plane` section

| Field | Type | Purpose |
|---|---|---|
| `watchdog` | `WatchdogSection` | Plugin, plot commands, alert thresholds, poll interval |
| `notifications` | `NotificationsSection` | `pr` and `telegram` channel configs |
| `watchdog_pid` | `int \| None` | Live watchdog PID |
| `anomaly_detector_cron_id` | `str \| None` | Cron ID for the anomaly-detector recurring agent |
| `checkpoint_cron_id` | `str \| None` | Cron ID for the checkpoint recurring skill |

#### `contract.runs[]` — per-run specification

| Field | Type | Required | Purpose |
|---|---|---|---|
| `label` | `str` | yes | Display label (e.g. `A1_G`, `C2_D`) |
| `db` | `int` (≥0) | yes | Redis DB number (0–15) |
| `prefix` | `str` | yes | Redis key prefix (= Hydra `problem.name`) |
| `pipeline` | `str` | yes | Pipeline config name (`standard`, `adversarial_asymmetric`, …) |
| `problem_name` | `str` | yes | Hydra `problem.name` override |
| `condition` | `str` | yes | Human-readable arm/condition description |
| `chain_url` | `str \| None` | no | Chain LLM endpoint (null ⇒ shared LB) |
| `mutation_url` | `str \| None` | no | Mutation LLM endpoint |
| `model_name` | `str` | yes | Model ID (e.g. `Qwen3-235B-A22B-Thinking-2507`) |
| `pid` | `int \| None` | gated | Set by `launch.sh`; required when `status=running` |
| `log_path` | `str \| None` | no | Relative log file path (default: `run_<label>.log`) |
| `extra_overrides` | `list[str] \| None` | no | Per-run Hydra overrides (appended to `run.py` CLI) |
| `role` | `str \| None` | gated | Required when `control_plane.watchdog.plugin=adversarial` (values: `constructor`, `improver`) |
| `wave` | `int \| None` | no | Wave grouping for sequential launches |

### Status state machine

Forward transitions enforced by `set_status`:

```
preregistered ──► implemented ──► running ──► complete
                                          └─► invalid ──► preregistered  (retry)
```

Recovery transitions allowed only via `gigaevo manifest reset-status` (escape hatch):

```
running ──► implemented    (launch failed; re-launch needed)
running ──► preregistered  (invalid launch; release DB claims, re-implement)
```

Status gates (enforced at load time by `ExperimentManifest.validate_status_gates`):

| Status | Required fields |
|---|---|
| `preregistered` | `experiment.*`, `schema_version` |
| `implemented` | above + non-empty `runs[]`, `servers[]`, `config`, `smoke_test.completed=true` |
| `running` | above + `launch.time`, `launch.commit`, every `runs[].pid` set |
| `complete` | same as `running` (archival state) |
| `invalid` | no additional gates (terminal; `reset-status` clears) |

### CLI reference — `gigaevo manifest …`

All subcommands require `-e/--experiment TASK/NAME`.

| Subcommand | Purpose | Example |
|---|---|---|
| `get FIELD` | Read scalar field or dotted path | `gigaevo -e hover/foo manifest get status` |
| `get runs` | Pretty-print runs table | `gigaevo -e hover/foo manifest get runs` |
| `get <dotted.path>` | Traverse nested YAML | `gigaevo -e hover/foo manifest get control_plane.watchdog_pid` |
| `update status VALUE` | State-machine-validated status transition | `gigaevo -e hover/foo manifest update status running` |
| `update PATH VALUE` | Write any other field (auto-coerces int/float/bool/null) | `gigaevo -e hover/foo manifest update control_plane.watchdog_pid 12345` |
| `gate STATUS` | Assert status; exit 0 on match, 1 on mismatch | `gigaevo -e hover/foo manifest gate implemented` |
| `pr-description [--push]` | Render Markdown PR body; optionally push via `gh` | `gigaevo -e hover/foo manifest pr-description --push` |
| `record-pids --pids-file F --labels "A B C"` | Write launched PIDs into `runs[].pid` | Called by generated `launch.sh` |
| `reset-status TARGET --reason 'why'` | Force status transition (escape hatch) | `gigaevo -e hover/foo manifest reset-status implemented --reason 'launch crashed'` |

Notes:
- `update` auto-coerces: `true`/`false` → bool, `null`/`none` → `None`, integer/float literals, else string.
- `reset-status` from `running`: releases Redis DB claims and clears `launch.*` + `runs[].pid` (when target is `implemented`).
- `pr-description --push` requires `experiment.pr_number` to be set.

### Bijective mapping: `experiment.yaml` ↔ `launch.sh`

`launch.sh` is regenerated from `experiment.yaml` with `gigaevo -e <exp> launch --generate-script`.
The mapping is one-way deterministic — every field in the manifest corresponds to an observable
fragment of `launch.sh`; every fragment of `launch.sh` is traceable back to a field.

| Manifest field | `launch.sh` output |
|---|---|
| `experiment.name` | Banner header, regeneration comment, label in launch log |
| `experiment.branch` | Header comment only |
| `experiment.pr_number` | Header comment only |
| `experiment.prereg_commit` | Header comment + launch banner |
| `experiment.max_generations` | `max_mutants=<N>` Hydra override per run (manifest field name is historical; engine knob is `max_mutants` since v2.0.0) |
| `servers[]` | `NO_PROXY` export: `localhost,127.0.0.1,api.github.com,<servers...>` |
| `custom_env{}` | `export KEY="VALUE"` lines, then propagated to every run |
| `config.extra.*` | Every key emitted as a Hydra override `KEY=VALUE` (bool → `true`/`false`, `None` → `null`). No defaults imposed — absent keys fall through to the Hydra config hierarchy. Dotted keys (`pipeline_builder.archive_reeval`) pass through verbatim. |
| `runs[].label` | PID variable name `PID_<label>`, log file `run_<label>.log`, label in `pids.txt` |
| `runs[].db` | `redis.db=<N>` Hydra override |
| `runs[].pipeline` | `pipeline=<name>` Hydra override |
| `runs[].problem_name` | `problem.name=<path>` Hydra override |
| `runs[].condition` | Launch banner comment |
| `runs[].model_name` | `model_name=<id>` Hydra override |
| `runs[].mutation_url` | `llm_base_url="<url>"` Hydra override |
| `runs[].chain_url` (non-null) | Per-run `${CHAIN_URL_ENV_VAR}=<url> nohup ...` prefix |
| `runs[].extra_overrides` | Appended verbatim to the run's Hydra CLI (`${…}` refs single-quoted per KF-02) |
| `runs[].log_path` | Stdout/stderr redirection target (default `run_<label>.log`) |

#### Worked example: asymmetric adversarial run `A1_G` (composition variant)

Manifest entry:

```yaml
- label: A1_G
  db: 1
  prefix: adversarial/g_d/pop_a
  pipeline: adversarial_asymmetric
  problem_name: adversarial/g_d/pop_a
  condition: 'Composition variant: Constructor, pair 1'
  chain_url: null
  mutation_url: http://localhost:8000/v1
  model_name: Qwen3-235B-A22B-Thinking-2507
  extra_overrides:
    - evolution=steady_state
    - opponent_redis_db=2
    - opponent_redis_prefix=adversarial/g_d/pop_b
    - feedback_mode=composition
    - population_role=constructor
    - post_step_hook=${composition_injection_hook}
  role: constructor
```

Generated `launch.sh` fragment:

```bash
# ── Run A1_G: Composition variant: Constructor, pair 1
nohup "$PYTHON" "$PROJ/run.py" \
    problem.name=adversarial/g_d/pop_a \
    pipeline=adversarial_asymmetric \
    prompts=default \
    redis.db=1 \
    stage_timeout=2400 \
    dag_timeout=2400 \
    max_mutants=50 \
    max_elites_per_generation=8 \
    num_parents=1 \
    model_name=Qwen3-235B-A22B-Thinking-2507 \
    llm_base_url="http://localhost:8000/v1" \
    evolution=steady_state \
    opponent_redis_db=2 \
    opponent_redis_prefix=adversarial/g_d/pop_b \
    feedback_mode=composition \
    population_role=constructor \
    '${composition_injection_hook}' \
    > "$LOG_DIR/run_A1_G.log" 2>&1 &
PID_A1_G=$!
```

Notes:
- `chain_url: null` ⇒ no per-run `CHAIN_URL=…` prefix; the run uses the shared LiteLLM proxy via `custom_env`.
- `post_step_hook=${composition_injection_hook}` contains a Hydra interpolation ref; the generator single-quotes it so bash doesn't expand `${…}` as a shell variable (KF-02).
- `role: constructor` does not appear in `launch.sh` — it is consumed only by the adversarial watchdog plugin for G/D dispatch and frontier suppression.

### Manifest helpers

Python API (`gigaevo.monitoring.manifest`):

| Function | Purpose |
|---|---|
| `load_manifest(exp) -> ExperimentManifest` | Load + validate experiment.yaml (Pydantic) |
| `set_status(exp, new, *, allow_recovery=False)` | State-machine-enforced status write |
| `update_manifest(exp, updater)` | Atomic mutation under Redis lock |
| `claim_dbs(exp, [dbs])` | Reserve Redis DBs with TTL=7d |
| `refresh_db_claims(exp, [dbs])` | Extend existing claims |
| `release_db_claims([dbs])` | Drop claims (e.g. on reset-status) |
| `find_active_experiments()` | Discover all implemented/running experiments |
| `generate_pr_description(exp)` | Render Markdown PR body |

Atomicity: every write acquires a Redis lock (`experiments:<exp>:yaml_lock`, 30s TTL),
writes to `experiment.yaml.tmp`, `fsync`s, then renames — FUSE-safe.

---

## Benchmarking

### `benchmark.py` — Throughput benchmark runner

CLI wrapper that runs the benchmark test suite (`tests/benchmarks/`).

```bash
# Quick run with fakeredis
python -m tools.benchmark

# Full run with real Redis
python -m tools.benchmark --redis-url redis://localhost:6379/15 --full

# Also run profiler
python -m tools.benchmark --profile
```

### `profiler.py` — Redis and DAG throughput profiler

Measures throughput of Redis ops, program serialization, DAG construction,
stage execution, and concurrent workloads.

```bash
python -m tools.profiler --redis-url redis://localhost:6379/15
```

---

## Scaffolding

### `dag_builder/` — Visual DAG pipeline builder

A React + FastAPI app for building execution pipelines visually. Drag-and-drop
stages, connect data flow edges, export as Python code or Hydra YAML config.

```bash
# Start both backend (port 8081) and frontend (port 8082)
bash tools/dag_builder/start.sh
```

See `tools/dag_builder/README.md` for full documentation.

### `wizard/` — Problem directory generator

Generates a complete problem directory from a YAML config (validate.py, metrics.yaml,
initial_programs/, task_description.txt). See `tools/wizard/` for documentation.

```bash
python -m tools.wizard my_config.yaml
python -m tools.wizard my_config.yaml --overwrite
python -m tools.wizard my_config.yaml --validate-only
```

---

## Placement Convention

- `tools/` — works for **any** GigaEvo run (just needs `--run prefix@db`)
- `tools/experiment/` — depends on experiment.yaml, protocol docs, or PRs
- `experiments/<task>/<name>/tools/` — imports problem-specific code, hardcodes experiment-specific values

---

## Appendix: Redis Data Model

Every run uses one Redis DB (0–15), set via `redis.db=N`.
`{prefix}` below = `problem.name` (e.g. `chains/hotpotqa/static`).

### Key namespaces

There are three independent key namespaces per run:

| Namespace | Key pattern | Data type | Purpose |
|---|---|---|---|
| **Program storage** | `{prefix}:program:{id}` | string (JSON) | Serialized Program objects |
| **Program status** | `{prefix}:status:{state}` | set | Sets of program IDs by state (PENDING, RUNNING, DONE, ERROR) |
| **Status stream** | `{prefix}:status_events` | stream | Status change events |
| **Run state** | `{prefix}:run_state` | hash | Engine counters (generation, migration) |
| **Archive** | `{prefix}:archive` | hash | MAP-Elites archive: cell → program_id |
| **Archive reverse** | `{prefix}:archive:reverse` | hash | Reverse index: program_id → cell |
| **Timestamp** | `{prefix}:ts` | string (int) | Atomic counter |
| **Instance lock** | `{prefix}:__instance_lock__` | string | Distributed lock |
| **Metrics latest** | `{prefix}:metrics:latest` | hash | Latest value for each metric tag |
| **Metrics history** | `{prefix}:metrics:history:{tag}` | list | Time series (see below) |
| **Metrics meta** | `{prefix}:metrics:meta` | hash | Metadata (last_update timestamp) |

### Metrics history keys

The metrics backend writes history lists. Each entry is JSON:
```json
{"s": <step>, "t": <unix_timestamp>, "v": <value>, "k": "scalar"}
```

Tags are generated by `MetricsTracker` (bound to path `program_metrics`), then
sanitized (`/` → `_` within segments, segments joined by `:`). The full Redis key
for a metric tag is:

```
{prefix}:metrics:history:program_metrics:{sanitized_tag}
```

**Complete list of metric tags written:**

| Tag | Redis key suffix | Type | Written when |
|---|---|---|---|
| `is_valid` | `program_metrics:is_valid` | 0.0 or 1.0 | Every program |
| `programs_total_count` | `program_metrics:programs_total_count` | cumulative | Every program |
| `programs_valid_count` | `program_metrics:programs_valid_count` | cumulative | Every program |
| `programs_invalid_count` | `program_metrics:programs_invalid_count` | cumulative | Every program |
| `valid_program_{metric}` | `program_metrics:valid_program_{metric}` | per-program value | Each valid program |
| `valid_frontier_{metric}` | `program_metrics:valid_frontier_{metric}` | frontier best | On frontier improvement |
| `valid_iter_{metric}_mean` | `program_metrics:valid_iter_{metric}_mean` | running mean | Each valid program |
| `valid_iter_{metric}_std` | `program_metrics:valid_iter_{metric}_std` | running std | Each valid program |
| `valid_gen_{metric}_mean` | `program_metrics:valid_gen_{metric}_mean` | per-gen mean | Each valid program |
| `valid_gen_{metric}_std` | `program_metrics:valid_gen_{metric}_std` | per-gen std | Each valid program |

Where `{metric}` = metric name from `problems/{problem_name}/metrics.yaml` (e.g. `fitness`, `prompt_length`).

**DAG internals** (written by the DAG runner, not MetricsTracker):

| Redis key suffix | Purpose |
|---|---|
| `dag_runner:dag:internals:CallValidatorFunction:stage_duration` | Validator execution time |
| `dag_runner:dag:internals:{StageName}:{metric}` | Per-stage timing/error metrics |

### How to read common values

| What you want | Command | Notes |
|---|---|---|
| Current generation | `hget {prefix}:run_state engine:total_generations` | **Canonical** — never use other sources |
| Best frontier fitness | `lindex {prefix}:metrics:history:program_metrics:valid_frontier_fitness -1` | Parse JSON → `"v"` field |
| Per-gen mean fitness | `lrange {prefix}:metrics:history:program_metrics:valid_gen_fitness_mean 0 -1` | `"s"` = generation, `"v"` = mean |
| Total programs | `lindex {prefix}:metrics:history:program_metrics:programs_total_count -1` | `"v"` field |
| Valid programs | `lindex {prefix}:metrics:history:program_metrics:programs_valid_count -1` | `"v"` field |
| Validator duration | `lrange {prefix}:metrics:history:dag_runner:dag:internals:CallValidatorFunction:stage_duration -20 -1` | Mean/max of last 20 entries |
| Archive size | `hlen {prefix}:archive` | Number of occupied cells |
| All latest metrics | `hgetall {prefix}:metrics:latest` | Quick snapshot, no history |

Use `status.py --experiment` to auto-discover all metrics. **Never write ad-hoc Redis queries** — if a tool gives wrong results, fix the tool.

### Archive persistence

The archive is **dual-backed**:
- **Redis** (`{prefix}:archive` hash, `{prefix}:archive:reverse` hash) — persistent, survives engine restarts
- **In-memory cache** — write-through optimization, session-scoped only

Archive data persists in Redis until explicitly flushed. However, the **programs themselves** and their **metrics histories** are also in Redis and equally persistent. To preserve data before flushing, run `tools/experiment/archive_run.sh --upload`.

### Iteration vs. generation

- **Iteration**: monotonically increasing program evaluation counter (1, 2, 3, ...)
- **Generation**: MAP-Elites generation count (incremented after `max_mutations_per_generation` evaluations)

`valid_iter_*` keys track per-iteration running aggregates. `valid_gen_*` keys track per-generation aggregates. The canonical generation count is `{prefix}:run_state` field `engine:total_generations` — never derive it from metric step values.

### Rules

1. **Canonical generation count**: `hget {prefix}:run_state engine:total_generations`. Never use `llen(valid_frontier_fitness)` or `valid_iter_fitness_mean` last `"s"` — both can lag under high throughput.
2. **Never write ad-hoc Redis queries** to answer questions the tools already answer. If a tool gives wrong results, fix the tool.
3. **Never flush manually** with `redis-cli FLUSHDB` — use `gigaevo flush --db N --confirm` which kills workers first.
4. **Never use log grep for gen count** — `grep -c "Phase 1: Idle confirmed" run.log` is brittle and has caused production crashes. Use `hget` on `run_state`.
