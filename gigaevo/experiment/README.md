# Experiment Module

Single source of truth for experiment lifecycle: schema, state machine, checks, launch, locking.

## Public API

Four modules, four concerns:

### `manifest.py` — Schema + CRUD

Pydantic v2 schema and all read/write operations on `experiment.yaml`.

```python
from gigaevo.experiment.manifest import load_manifest, set_status, update_manifest
from gigaevo.experiment.manifest import claim_dbs, refresh_db_claims, release_db_claims
```

- Load and validate experiment.yaml (strict Pydantic v2)
- Enforce status transitions via state machine
- Redis-locked atomic writes (write-then-rename, FUSE-safe)
- DB claim lifecycle (claim with SET NX, refresh TTL, release)
- PR description generation, experiment discovery

### `checks.py` — Pre-Launch Validation

10 principled checks targeting real operator failure modes.

```python
from gigaevo.experiment.checks import run_checks, CheckResult, Severity
```

Checks: status gate, `GIGAEVO_PYTHON`, server reachability, model IDs, Redis DBs empty, DB claims available, seed programs, test-set SHA, smoke test completed, treatment verification completed.

Returns `list[CheckResult]`. Any `CRITICAL` failure blocks launch.

### `launch.py` — Launch Orchestrator

Sequences the full launch atomically: gate → checks → claim DBs → generate script → exec → record PIDs → set status → spawn watchdog.

```python
from gigaevo.experiment.launch import run_launch, LaunchResult, LaunchStep
```

**CLI:** `gigaevo -e <exp> launch [--dry-run] [--skip-preflight]`

Returns typed `LaunchResult` with step-by-step progress. Rolls back DB claims on failure.

### `lock.py` — Internal Concurrency Primitives

Package-private (`_`-prefixed). Redis locks + atomic file writes. Used by `manifest.py`.

## Manifest Schema

Pydantic v2 model with four sub-sections:

| Sub-section | Writer | When it changes | Example fields |
|---|---|---|---|
| `contract` | Researcher | Frozen at `preregistered` | identity, problem, config, runs, servers, baseline |
| `lifecycle` | System | Mutates constantly | status, launch, smoke_test, treatment_verification |
| `telemetry` | System | Append-only during `running` | checkpoints, mid_run_test_eval, treatment_checks |
| `control_plane` | System | Mutates during setup + live ops | watchdog, notifications, watchdog_pid, cron IDs |

### Read paths

```python
m = load_manifest("hover/my-exp")

m.contract.identity.name                    # "hover/my-exp"
m.contract.runs[0].label                    # "A1"
m.lifecycle.status                          # "preregistered" | "implemented" | ...
m.lifecycle.launch.time                     # ISO timestamp or None
m.telemetry.mid_run_test_eval.completed     # bool
m.control_plane.watchdog.plugin             # "adversarial" | "solo" | ...
```

### Write paths (CLI)

```bash
gigaevo -e "$EXP" manifest update lifecycle.launch.time "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gigaevo -e "$EXP" manifest update control_plane.watchdog_pid 12345
```

## State Machine

```
preregistered
    ├─ implemented          (after smoke test + config)
    │   ├─ running          (after launch + PID recording)
    │   │   ├─ complete     (terminal: final)
    │   │   └─ invalid      (terminal: broken mid-run)
    │   └─ implemented      (recovery: reset from running)
    └─ implemented          (recovery: back to draft)
```

| Status | Required Fields |
|--------|---|
| `preregistered` | identity.name, identity.task |
| `implemented` | + runs[], servers[], config, smoke_test.completed |
| `running` | + launch.time, launch.commit, all runs[].pid |
| `complete` | same as running |
| `invalid` | same as running (mid-experiment failure) |

Hard gate pattern (used in skills):
```bash
gigaevo -e "$EXP" manifest gate implemented  # exits 0 on match, 1 otherwise
```

## DB Claims

Redis-based mutual exclusion for DB allocations.

```python
failed = claim_dbs("hover/my-exp", [15, 16])
refresh_db_claims("hover/my-exp", [15, 16])  # watchdog calls each cycle
release_db_claims([15, 16])                   # on experiment complete
```

Key: `experiments:db_claim:{db_number}`, Value: experiment name, TTL: 7 days.

## Locking

Key: `experiments:{name}:yaml_lock`, Expiry: 30s, Timeout: 5s.

## Testing

```bash
pytest tests/experiment/ tests/test_tools/test_manifest.py tests/monitoring/test_manifest_ops.py tests/monitoring/test_manifest_schema.py -v
```

## See Also

- `gigaevo/cli/manifest_cmd.py` — CLI for manifest get/set/update
- `gigaevo/cli/launch_cmd.py` — CLI for `gigaevo launch`
- `.claude/skills/experiment-launch/` — Launch workflow skill
- `docs/protocol/` — Experimental protocol and phase lifecycle
