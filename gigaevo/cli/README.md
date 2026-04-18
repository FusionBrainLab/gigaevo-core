# GigaEvo CLI Module

The CLI module provides **stateless query and control operations** over experiment manifests and live runs. All commands read from or write to the single source of truth (`experiment.yaml` and Redis run state), with no CLI-specific state.

## Overview

This module implements a **lazy-loaded command hierarchy** using Click, with a global context that carries experiment name, run specifications, and output formatting preferences. Commands are organized into logical groups based on their purpose.

## Design Principles

1. **Stateless** — CLI has no memory. All state lives in `experiment.yaml` (manifest) or Redis (run data).
2. **Transparent** — Each command maps 1:1 to manifest operations or monitoring queries. No business logic in CLI layer.
3. **Lazy-loaded** — Subcommand modules import only when invoked (5x faster startup).
4. **Composable** — Global flags (`-e`, `-r`, `-f`, etc.) are available to all subcommands without repetition.

## Global Flags

All commands inherit these global flags from the `main()` group:

```
-e, --experiment TEXT
    Experiment name (task/name). Reads experiment.yaml for run discovery.
    Required for manifest commands. Optional for monitoring commands.

-r, --run TEXT
    Run specification: prefix@db[:label] or shorthand db / @db / db:label.
    Repeatable. Optional for monitoring — filters to specific runs.

-f, --format [table|json|csv|markdown]
    Output format. Auto-detects: table for terminal, json for pipe.

-q, --quiet
    Suppress output (useful in scripts).

-v, --verbose
    Verbose output (debug-level logging).

--redis-host TEXT
    Redis server hostname. Default: localhost.

--redis-port INTEGER
    Redis server port. Default: 6379.
```

## Command Categories

### Manifest Commands (`gigaevo manifest ...`)

**Purpose**: Read and write experiment.yaml fields atomically with state machine enforcement.

**Subcommands**:

- `get FIELD` — Read a field from experiment.yaml using dotted path syntax (e.g., `lifecycle.status`, `contract.runs[0].db`, `lifecycle.launch.time`)
- `set FIELD VALUE` — Set a field and enforce state machine rules (e.g., set status → validates new status is in VALID_TRANSITIONS)
- `update FIELD VALUE` — Update a field via closure function (low-level; prefer `set` for status changes)
- `gate STATUS` — Verify experiment is in given status without reading output. Exit code 0 if match, non-zero otherwise. Used as a hard gate in shell scripts.
- `pr-description [--push]` — Generate PR description from manifest. Optionally push to GitHub PR.
- `record-pids --pids-file PATH --labels LABEL1 LABEL2 ...` — Record run PIDs from file into runs[] array after launch.
- `reset-status DESIRED_STATUS [--reason TEXT]` — Force status transition (debug only). Requires `--reason` for audit trail.

**Example workflow**:

```bash
# Check status is implemented
gigaevo -e hover/foo manifest gate implemented

# Read a nested field
gigaevo -e hover/foo manifest get runs --format json

# Update launch time
gigaevo -e hover/foo manifest update lifecycle.launch.time "2026-04-14T10:00:00Z"

# Transition status with state-machine validation
gigaevo -e hover/foo manifest update status running
```

**Key behaviors**:

- `get` — Returns field value as plain text (or JSON with `--format json`)
- `set` — Validates state machine transition rules. Fails if invalid. Writes atomically to experiment.yaml via Redis lock.
- `gate` — Hard gate for shell scripts. No output on success (exit 0), error message on failure (exit 1).

### Monitoring Commands

**Purpose**: Query live run state from Redis. Read-only, no manifest changes.

#### Status (`gigaevo status`)
Live experiment monitoring: generation, metrics, PIDs, watchdog status.

**Flags**:
- `-e, --experiment` — required
- `-r, --run` — optional; filter specific runs
- `--freq SECONDS` — refresh interval (0 = once)
- `--format` — table, json, csv

**Output**: Gen number, best fitness, run status (ALIVE/DEAD/STALLED), watchdog PID, last update timestamp.

#### Trajectory (`gigaevo trajectory`)
Generation-by-generation fitness history for each run.

**Flags**:
- `-e, --experiment` — required
- `-r, --run` — optional
- `--window INT` — show last N generations
- `--format` — table, json, csv

#### Top (`gigaevo top`)
Best programs by fitness. Displays program code, fitness, gen discovered, run label.

**Flags**:
- `-e, --experiment` — required
- `-r, --run` — optional
- `--limit INT` — how many programs to show (default: 10)
- `--format` — table, json, csv

#### Logs (`gigaevo logs`)
Tail per-run experiment log files at `experiments/<exp>/run_<label>.log`.

**Resolution priority** (highest first):
1. Explicit `--file PATH` (bypasses manifest entirely)
2. Positional `LABELS` (resolved via manifest to `experiments/<exp>/run_<label>.log`)
3. No labels + `--experiment` → list mode (table of candidate run logs)

**Flags**:
- `-e, --experiment` — required for label resolution and list mode
- `--file PATH` — explicit log file path (no manifest required)
- `-f, --follow` — live tail (`tail -f`)
- `-n, --tail LINES` — show last N lines (default: 50; ignored with `-f`)

**Examples**:
```bash
gigaevo -e hover/foo logs                  # List all run logs (sizes, mtimes)
gigaevo -e hover/foo logs A3_G             # Tail run_A3_G.log
gigaevo -e hover/foo logs A3_G -f          # Live tail
gigaevo -e hover/foo logs A3_G B5_D -f     # Multiplex live tail across runs
gigaevo logs --file /tmp/foo.log           # Tail an arbitrary file
```

Unknown labels exit non-zero with a list of known labels. Missing log files exit non-zero with the resolved path.

#### Plot (`gigaevo plot ...`)
Multi-run fitness visualization.

**Subcommands**:
- `comparison` — Multi-run fitness curves (one per run)
- `trajectory` — Single-run generation-by-generation trajectory
- `arms-race` — Dual-panel (defender vs adversary) for adversarial experiments

**Flags**:
- `-e, --experiment` — required
- `--paper` — 300 DPI + Okabe-Ito palette (publication-ready)
- `--no-frontier` — skip Pareto frontier for adversarial runs

#### Export (`gigaevo export ...`)
Bulk export of evolution data for analysis.

**Subcommands**:
- `csv` — Full evolution data to CSV (gen, fitness, run label, etc.)
- `frontier` — Frontier-only CSV (gen, best_val per generation)

**Selection semantics**:
1. No positional labels → operate on all runs resolved from `-e`/`-r`
2. Positional labels → filter resolved runs to only those labels (unknown → error)
3. **1 run in scope** → write to the exact `-o` path; emit flat JSON summary
4. **>1 run in scope** → fan out to `<stem>_<label><suffix>`; emit JSON list summary

**Flags**:
- `-e, --experiment` — experiment scope (or use `-r` directly)
- `-r, --run` — repeatable run spec (alternative to `-e`)
- `-o, --output-file PATH` — required; treated as a base path under fan-out
- `--metric NAME` — frontier only; metric used for `best_val` (default: `fitness`)

**Examples**:
```bash
# Export all runs in an experiment (fans out to results_A3_G.csv, results_A3_D.csv, ...)
gigaevo -e hover/foo export csv -o results.csv

# Export a single run (writes exactly to results.csv, flat JSON summary)
gigaevo -e hover/foo export csv A3_G -o results.csv

# Export selected runs (fans out only over the listed labels)
gigaevo -e hover/foo export csv A3_G B5_D -o results.csv

# Frontier with a custom metric
gigaevo -e hover/foo export frontier --metric actual_fitness -o frontier.csv
```

#### Inspect (`gigaevo inspect`)
Discover experiment prefix(es) in a Redis DB via `:__instance_lock__`.

**Flags**:
- `--db INT` — Redis DB number (required)

**Output**: Experiment name(s) in that DB.

### Infrastructure Commands

**Purpose**: Control experiment execution (watchdog, checkpoints, DB flushing).

#### Watchdog (`gigaevo watchdog`)
Start the per-experiment watchdog loop (manual invocation; usually started by `/experiment-launch` skill).

**Flags**:
- `-e, --experiment` — required
- `--freq SECONDS` — polling interval (default: 30)
- `--no-auto-restart` — don't auto-restart dead workers

**Behavior**: Runs indefinitely. Monitors PIDs, records metrics, flushes stalled generations, handles invalid runs.

#### Checkpoint (`gigaevo checkpoint`)
Snapshot current frontier and generation state.

**Flags**:
- `-e, --experiment` — required
- `-r, --run` — optional
- `--note TEXT` — human-readable checkpoint note

**Behavior**: Records checkpoint event in experiment.yaml under `checkpoints[]` with timestamp and note.

#### Flush (`gigaevo flush`)
Kill all run workers and flush Redis DBs.

**Flags**:
- `--db INT` — DB number(s) to flush (repeatable)
- `--confirm` — require explicit confirmation

**Behavior**: Finds all run processes in target DBs, kills them, flushes Redis (data lost permanently).

## Lazy Loading Architecture

Commands are **not** imported at CLI startup. Instead, the `LazyGroup` class delays module import until a subcommand is invoked.

**Flow**:

```
gigaevo status --experiment hover/foo
    ↓
main() (global context setup)
    ↓
LazyGroup.get_command('status')
    ↓ (only now)
importlib.import_module('gigaevo.cli.status')
    ↓
status.status (Click command object)
    ↓
status(...) (command execution)
```

**Benefits**:
- Startup time: 0.5s (not 2.5s with full import graph)
- No dependency on unused commands
- Scales with command count

**How it works** (`gigaevo/cli/__init__.py`):

```python
_LAZY_SUBCOMMANDS = {
    "status": ("gigaevo.cli.status", "status"),
    "trajectory": ("gigaevo.cli.trajectory", "trajectory"),
    ...
}

class LazyGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        if cmd_name in _LAZY_SUBCOMMANDS:
            module_path, attr_name = _LAZY_SUBCOMMANDS[cmd_name]
            mod = importlib.import_module(module_path)
            return getattr(mod, attr_name)
        return None
```

To add a new command:
1. Create `gigaevo/cli/my_cmd.py` with a Click command named `my_cmd`
2. Add entry to `_LAZY_SUBCOMMANDS`: `"my_cmd": ("gigaevo.cli.my_cmd", "my_cmd")`
3. That's it — no changes to imports or registration code

## Output Formatting

All commands respect the global `--format` flag via `OutputFormatter`:

```
table   — Human-readable table (default for terminal)
json    — Compact JSON (single line)
csv     — RFC 4180 CSV
markdown — Markdown table
```

Auto-detection: If stdout is a tty (interactive terminal), default to `table`. If piped, default to `json`.

**Custom formatting** in commands:

```python
formatter = ctx.obj["formatter"]
formatter.output(data)  # Formats according to --format flag
```

## Common Workflows

### Monitor a running experiment
```bash
gigaevo -e hover/foo status --freq 30  # Refresh every 30s
```

### Analyze best programs
```bash
gigaevo -e hover/foo top --limit 20 --format json | \
  python analyze_frontier.py
```

### Export data for paper
```bash
gigaevo -e hover/foo export csv -o results.csv         # fans out per run
gigaevo -e hover/foo export csv A3_G -o results.csv    # single run, exact path
gigaevo -e hover/foo plot comparison --paper --output fig1.png
```

### Checkpoint mid-run
```bash
gigaevo -e hover/foo checkpoint --note "Good frontier at gen 100"
```

### Debug a specific run
```bash
gigaevo -e hover/foo logs                 # List all run logs
gigaevo -e hover/foo logs A3_G -f         # Live tail one run by label
gigaevo -e hover/foo trajectory -r "prefix@5"
```

### Flush and restart
```bash
gigaevo flush --db 5 6 7 --confirm  # Kill workers + flush 3 DBs
gigaevo -e hover/foo manifest update status implemented  # Reset
/experiment-restart hover/foo  # Launch again with current code
```

## Error Handling

Commands validate inputs and provide actionable error messages:

- **Missing required flags** — "Error: manifest commands require --experiment / -e flag"
- **Invalid status transition** — "Error: invalid transition running → preregistered (not in VALID_TRANSITIONS)"
- **Redis unreachable** — "Cannot connect to Redis at localhost:6379. Fix: Start Redis or set REDIS_HOST/REDIS_PORT"
- **Manifest not found** — "FileNotFoundError: experiments/hover/foo/experiment.yaml"

## Integration with Skills

Skills invoke CLI commands for atomic operations:

```bash
# In /experiment-launch skill:
gigaevo -e "$EXP" manifest gate implemented    # Hard gate
gigaevo -e "$EXP" manifest get runs            # Discover run specs
gigaevo -e "$EXP" manifest update status running  # State transition

# In /experiment-checkpoint skill:
gigaevo -e "$EXP" checkpoint --note "frontier improved"
```

The skill layer is **orchestration** (human gates, error recovery); the CLI is **stateless operations**.

## Dependencies

- **Click**: Command-line interface framework
- **Pydantic**: Manifest schema validation (indirectly via `gigaevo.experiment.manifest`)
- **Redis**: Live run state (indirectly via CLI command modules)

## Testing

CLI commands are tested in `tests/cli/`:

- `test_manifest_cmd.py` — `manifest get/set/gate` operations
- `test_status_cmd.py` — Status monitoring
- `test_export_cmd.py` — Data export (positional labels + multi-run fan-out)
- `test_logs_cmd.py` — Log tailing (explicit --file / labels / list mode)
- `test_run_resolver.py` — `-e`/`-r` → RunConfig resolution
- etc.

Run all CLI tests:
```bash
/run-tests tests/cli/
```

## See Also

- `gigaevo/experiment/README.md` — Manifest module (source of truth for experiment state)
- `.claude/skills/experiment-launch/` — Multi-step launch workflow (uses CLI commands)
- `tools/README.md` — Lower-level run tools (scripts, not CLI)
