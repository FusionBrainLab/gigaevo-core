# Experiment Module

The experiment module is the **single source of truth** for all experiment lifecycle management. It provides schema validation, state machine enforcement, atomic file operations, and DB claim lifecycle management.

## Overview

This module is split into two focused submodules:

### `manifest.py` — Manifest Operations

Pydantic v2 schema + CRUD operations for experiment.yaml.

**Canonical import:**
```python
from gigaevo.experiment.manifest import load_manifest, set_status, update_manifest
```

**Purpose:**
- Load and validate experiment.yaml files
- Enforce status transitions with state machine rules
- Mutation operations with Redis locking
- DB claim lifecycle (claim, refresh, release)
- PR description generation
- Discovery of active experiments

### `preflight.py` — Pre-Launch Validation

22 checks covering schema validation, infrastructure, and configuration readiness.

**Import:**
```python
from gigaevo.experiment.preflight import run_checks
```

**Purpose:**
- Validate experiment configuration before launch (hard gate)
- Check database allocations, server connectivity, manifest syntax
- Report blocking vs warning severity
- Exit codes: 0 (pass), 1 (CRITICAL failures), 2 (WARNINGS)

### `launch_generator.py` — Launch Script Generation

Auto-generates launch.sh from experiment.yaml.

**Import:**
```python
from gigaevo.experiment.launch_generator import generate
```

**Purpose:**
- Eliminate hand-written launch scripts
- Generate runnable bash script from manifest
- Include config verification step
- Hydra parameter binding for all runs

## Schema: ExperimentManifest

Pydantic v2 dataclass-style model with strict validation.

**Top-level structure:**
```python
ExperimentManifest(
    schema_version: int,                    # Currently 1
    experiment: ExperimentSection,          # Name, task, status, branch, max_generations
    problem: ProblemSpec,                   # has_test_set, fitness_type, metric_name
    runs: list[RunSpec],                    # Each run's DB, pipeline, model, condition
    servers: list[str],                     # Server hostnames
    config: dict,                           # Shared Hydra config overrides
    custom_env: dict[str, str],             # Shared environment variables
    checkpoints: list[dict],                # Gen, timestamp, notes
    launch: LaunchInfo,                     # Time, commit, watchdog_pid, confirmed_at
    baseline: BaselineInfo,                 # Reference experiment, mean, metric
    smoke_test: SmokeTestInfo,              # completed, db, generations
    watchdog: WatchdogSection,              # Plugin, plot commands, alerts, polling
)
```

**Access patterns:**
```python
m = load_manifest("hover/feedback_softfit")

# Nested structure (canonical)
m.experiment.name                           # "hover/feedback_softfit"
m.experiment.status                         # "preregistered" | "implemented" | "running" | ...
m.experiment.max_generations                # 25 (or whatever)
m.runs[0].label                             # "A1"
m.runs[0].db                                # Redis DB number
m.launch.time                               # ISO timestamp or None
```

## State Machine

Four terminal/persistent states:

```
preregistered
    ├─ implemented          (after smoke test + config)
    │   ├─ running          (after launch + PID recording)
    │   │   ├─ complete     (terminal: final)
    │   │   └─ invalid      (terminal: broken mid-run)
    │   └─ implemented      (recovery: reset from running)
    └─ implemented          (recovery: back to draft)
```

**Status gates** (Pydantic validators):

| Status | Required Fields |
|--------|---|
| `preregistered` | experiment.name, experiment.task |
| `implemented` | + runs[], servers[], config, smoke_test.completed |
| `running` | + launch.time, launch.commit, all runs[].pid |
| `complete` | same as running |
| `invalid` | same as running (mid-experiment failure) |

## DB Claims

Redis-based mutual exclusion for DB allocations.

**Lifecycle:**
```python
# Claim DBs for an experiment
failed = claim_dbs("hover/feedback_softfit", [15, 16])
if failed:
    print(f"DBs already claimed by: {[owner for db, owner in failed]}")

# Refresh TTL (called by watchdog each cycle)
refresh_db_claims("hover/feedback_softfit", [15, 16])

# Release after experiment completes
release_db_claims([15, 16])
```

**Redis schema:**
- Key: `experiments:db_claim:{db_number}`
- Value: `experiment_name`
- TTL: 7 days (auto-cleans up stale claims)

## Locking

Atomic manifest writes use Redis locks + write-then-rename (FUSE-safe).

**Lock mechanics:**
- Key: `experiments:{experiment_name}:yaml_lock`
- Holder: PID of lock owner
- Expiry: 30 seconds (prevents zombie locks)
- Timeout: 5 seconds (fail fast if locked)

## Configuration

**Environment variables:**

```bash
export REDIS_HOST=localhost       # Default: localhost
export REDIS_PORT=6379            # Default: 6379
```

Redis connection failure provides actionable error messages:
```
Cannot connect to Redis at localhost:6379.
Fix: Start Redis with `redis-server` or set REDIS_HOST/REDIS_PORT.
Error: [Connection refused]
```

## Example: Full Workflow

```python
from gigaevo.experiment.manifest import (
    load_manifest, set_status, update_manifest, claim_dbs, release_db_claims
)

# 1. Load manifest
m = load_manifest("hover/my-exp")
print(f"Status: {m.experiment.status}")  # preregistered

# 2. Claim databases
failed = claim_dbs("hover/my-exp", [15, 16])
assert not failed, "DBs already claimed"

# 3. Run preflight checks
from gigaevo.experiment.preflight import run_checks
results = run_checks("hover/my-exp")
assert all(r.passed for r in results if r.severity == "CRITICAL")

# 4. Transition status
m = set_status("hover/my-exp", "implemented")
print(f"Status: {m.experiment.status}")  # implemented

# 5. Update manifest (e.g., add PIDs)
def record_pids(raw):
    for run in raw.get("runs", []):
        if run["label"] == "A":
            run["pid"] = 12345  # PID from launch

m = update_manifest("hover/my-exp", record_pids)

# 6. Transition to running
m = set_status("hover/my-exp", "running")

# 7. Refresh DB claims periodically (watchdog)
from gigaevo.experiment.manifest import refresh_db_claims
refresh_db_claims("hover/my-exp", [15, 16])

# 8. On experiment complete
release_db_claims([15, 16])
m = set_status("hover/my-exp", "complete")
```

## Testing

Run manifest tests:
```bash
pytest tests/test_tools/test_manifest.py -v
pytest tests/monitoring/test_manifest_ops.py -v
```

Coverage: Schema validation, state transitions, locking, atomic writes, DB claims, discovery, PR generation.

## See Also

- `gigaevo/cli/manifest_cmd.py` — CLI interface (manifest get/set/update)
- `.claude/skills/experiment-launch/` — Multi-step launch workflow
- `docs/protocol/` — Experimental protocol and phase lifecycle
