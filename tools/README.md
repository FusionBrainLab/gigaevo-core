# GigaEvo Tools

Reusable scripts for any GigaEvo experiment — operational, analytical, and scaffolding.
These are known-working tools; prefer them over ad-hoc one-liners.

**Run format used by operational and analysis tools**: `prefix@db[:label]`
where `prefix` = `problem.name` from the Hydra config (e.g., `chains/hotpotqa/static`).

---

## Operational Tools

### `status.py` — Live run status

Shows generation count, best val EM, key count, and PID liveness for multiple runs.

```bash
# One run
PYTHONPATH=. python tools/status.py --run chains/hotpotqa/static@0:K

# Multiple runs with PID and watchdog check
PYTHONPATH=. python tools/status.py \
    --run chains/hotpotqa/static@0:K \
    --run chains/hotpotqa/static_r@1:L \
    --run chains/hotpotqa/static_r@2:M \
    --run chains/hotpotqa/static_r@3:N \
    --pid K:2616605 --pid L:2616606 --pid M:2616607 --pid N:2616608 \
    --watchdog 2716169
```

Output:
```
Run      DB    Gen   Best Val EM    Keys         PID  Status
---------------------------------------------------------------
K         0     26        66.0%     317     2616605  ✓ ALIVE
L         1     25        65.7%     329     2616606  ✓ ALIVE
M         2     27        70.7%     321     2616607  ✓ ALIVE
N         3     25        67.7%     321     2616608  ✓ ALIVE

Watchdog PID 2716169: ✓ ALIVE
```

---

### `flush.py` — Safe Redis flush

Kills stale exec_runner workers first (they repopulate Redis immediately after flush),
then flushes each DB, then verifies 0 keys remain.

**Dry-run by default** — shows what would happen without doing it.

```bash
# Preview (dry-run)
PYTHONPATH=. python tools/flush.py --db 0 1 2 3

# Execute
PYTHONPATH=. python tools/flush.py --db 0 1 2 3 --confirm

# P3 experiment DBs
PYTHONPATH=. python tools/flush.py --db 14 15 --confirm
```

**Always kill exec_runner workers before flushing.** Flushing first then killing leaves
a window where workers repopulate Redis. `flush.py` enforces the correct ordering.

---

### `archive_run.sh` — Archive and upload run data

**Run this before flushing Redis or rebooting. Redis is ephemeral — data not exported is gone.**

Exports all Redis data for a run to local files and uploads them as a GitHub Release asset.

```bash
# Dry run: export locally only (verify output)
bash tools/archive_run.sh --exp hotpotqa_nlp_prompts --run "chains/hotpotqa/static@0:K"

# Export and upload to GitHub Release exp/hotpotqa_nlp_prompts
bash tools/archive_run.sh --exp hotpotqa_nlp_prompts --run "chains/hotpotqa/static@0:K" --upload

# Archive all 4 runs
for SPEC in "chains/hotpotqa/static@0:K" "chains/hotpotqa/static_r@1:L" \
            "chains/hotpotqa/static_r@2:M" "chains/hotpotqa/static_r@3:N"; do
  bash tools/archive_run.sh --exp hotpotqa_nlp_prompts --run "$SPEC" --upload
done
```

Each archive (uploaded as `<label>_archive.tar.gz` to the GitHub Release) contains:
- `evolution_data.csv` — all programs, all generations, all metrics
- `programs/*.py` — source code of every evaluated program
- `top50.json` — top 50 programs with full metadata

Also uploads `environment.txt` (pip freeze, OS, GPU) once per experiment.

---

### `check_phase_order.sh` — Protocol phase gate

Verifies that all required protocol documents exist, are committed, and are in the correct
state before proceeding to launch. Run this as the first step of Phase 4.

```bash
bash tools/check_phase_order.sh <experiment-name>
```

Checks:
- `01_design.md` exists
- `02_review.md` exists and contains APPROVED verdict
- `03_plan.md` is committed to git (not just on disk)
- `run_test_eval.sh` sha256 hash matches what is pinned in `03_plan.md` (if applicable)

Exit code 0 = all passed. Exit code 1 = one or more failures (do not proceed to launch).

---

## Analysis Tools

### `top_programs.py` — Inspect top programs

Fetches all programs from a run, ranks by fitness, and prints a summary table
with optional full source code. Use at checkpoints to inspect what evolved.

```bash
# Top 5 by fitness (default)
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@0:K

# Top 1 with full code — the program to run test eval on
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@0:K -n 1 --code

# Save top-3 codes to files
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@0 -n 3 --save-dir top_k/

# JSON output for scripting
PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@0 -n 1 --json
```

---

### `redis2pd.py` - Export Evolution Data

Exports evolution run data from Redis to a pandas DataFrame (CSV format) for further analysis.

**Usage:**
```bash
python tools/redis2pd.py \
  --redis-host localhost \
  --redis-port 6379 \
  --redis-db 11 \
  --redis-prefix "heilbron" \
  --output-file results.csv
```

**Arguments:**
- `--redis-host`: Redis server hostname (default: localhost)
- `--redis-port`: Redis server port (default: 6379)
- `--redis-db`: Redis database number (required)
- `--redis-prefix`: Problem name used in the run (required, same as `problem.name`)
- `--output-file`: Output CSV file path (required)

**Output:**
CSV file containing program metrics, fitness scores, generation numbers, etc.

**Example:**
```bash
# Export evolution run from database 5
# Note: redis-prefix is just the problem name (e.g., problem.name=my_problem)
python tools/redis2pd.py \
  --redis-db 5 \
  --redis-prefix "my_problem" \
  --output-file my_run_data.csv
```

---

### `comparison.py` - Compare Multiple Runs

Compares multiple evolution runs by plotting rolling fitness statistics over iterations.

**Usage:**
```bash
python tools/comparison.py \
  --redis-host localhost \
  --redis-port 6379 \
  --run "heilbron@11:Run_A" \
  --run "heilbron@12:Run_B" \
  --iteration-rolling-window 5 \
  --output-folder results/comparison
```

**Arguments:**
- `--redis-host`: Redis server hostname (default: localhost)
- `--redis-port`: Redis server port (default: 6379)
- `--run`: Run specification in format `<prefix>@<db>:<label>` (can be repeated)
  - `<prefix>` is the problem name (same as `problem.name`)
- `--iteration-rolling-window`: Window size for rolling statistics (default: 5)
- `--output-folder`: Directory to save comparison plots (required)

**Run Format:**
- `prefix@db:label` - Full specification with custom label
- `prefix@db` - Label defaults to "Run_<db>"
- `prefix` is just the problem name (e.g., `heilbron`)

**Output:**
- PNG plots showing fitness evolution over iterations
- Rolling mean with ±1 standard deviation bands
- Multiple runs overlaid for easy comparison

**Example:**
```bash
# Compare three different experiments (all using problem.name=test)
python tools/comparison.py \
  --run "test@5:Baseline" \
  --run "test@6:Multi_Island" \
  --run "test@7:Multi_LLM" \
  --iteration-rolling-window 10 \
  --output-folder results/my_comparison
```

---

### `wizard.py` - Problem Scaffolding

Generates problem directory structure from YAML configuration.

**Usage:**
```bash
python -m tools.wizard heilbron.yaml
python -m tools.wizard my_config.yaml --overwrite
python -m tools.wizard my_config.yaml --validate-only
python -m tools.wizard my_config.yaml --output-dir custom/path
```

**Arguments:**
- `CONFIG_NAME`: YAML configuration filename (required), e.g., `heilbron.yaml`
- `--overwrite`: Overwrite existing problem directory if it exists
- `--validate-only`: Validate configuration without generating files
- `--output-dir PATH`: Override output directory (default: `problems/<problem.name>`)
- `--problem-type TYPE`: Problem type determining templates (default: `programs`)

**File Structure:**
- **Configuration files:** Store in `tools/wizard/config/` directory
- **Templates:** Located in `gigaevo/problems/types/{problem_type}/templates/`
- **Output:** Generated in `problems/<name>/` by default

**Configuration Example (`heilbron.yaml`):**
```yaml
name: "heilbron"
description: "Heilbronn triangle problem"

entrypoint:
  params: []
  returns: "(11, 2) array of coordinates"

validation:
  params: ["coordinates"]

metrics:
  fitness:
    description: "Area of smallest triangle"
    decimals: 5
    is_primary: true
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 0.0365
    include_in_prompts: true
    significant_change: !!float 1e-6

task_description:
  objective: |
    Return 11 distinct 2D coordinates inside unit-area equilateral triangle.
    Maximize the minimum area among all triangles formed by point triplets.

add_helper: true

initial_programs:
  - name: arc
    description: "Arc-based point distribution"
```

**Key Configuration Notes:**
- Exactly one metric must have `is_primary: true`
- `is_valid` metric is auto-generated (do NOT include in config)
- Use `!!float` tag for small scientific notation values (e.g., `!!float 1e-6`)
- `add_context: true` generates `context.py` (optional, requires `context` param in function signatures)
- `add_helper: true` generates `helper.py` (optional)

**Generated Structure:**
```
problems/heilbron/
├── task_description.txt
├── metrics.yaml
├── validate.py          # User must implement
├── helper.py            # Optional: User must implement utilities
└── initial_programs/
    └── arc.py           # User must implement strategy
```

**Required Implementation:**

After scaffolding, you **must implement** the following:

**1. `validate.py` - Validation and metrics computation:**
```python
"""
Validation function for: Heilbronn triangle problem
"""

from helper import *


def validate(coordinates):
    """
    Validate the solution and compute fitness metrics.

    Returns:
        dict with metrics:
        - fitness: Area of smallest triangle
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    # TODO: Validate constraints from task_description.txt

    # TODO: Compute metrics
    fitness = 0.0  # Area of smallest triangle
    is_valid = 1   # Set to 0 if any constraint violated

    return {
        "fitness": fitness,
        "is_valid": is_valid,
    }
```

**2. All `initial_programs/*.py` - Initial strategy implementations:**
```python
from helper import *


def entrypoint():
    """
    Arc-based point distribution

    Returns:
        (11, 2) array of coordinates
    """
    # TODO: Implement strategy

    pass
```

**Optional Implementation:**

**If `add_helper: true`** - `helper.py` with utility functions:
```python
"""
Helper functions for: Heilbronn triangle problem
"""

# TODO: Add helper functions here
# Example:
# def get_unit_triangle():
#     """Return vertices of unit-area equilateral triangle."""
#     unit_area_side = np.sqrt(4 / np.sqrt(3))
#     height = np.sqrt(3) / 2 * unit_area_side
#     A = np.array([0, 0])
#     B = np.array([unit_area_side, 0])
#     C = np.array([unit_area_side / 2, height])
#     return A, B, C
```

**If `add_context: true`** - `context.py` with runtime context builder:
```python
"""
Context builder for problem
"""


def build_context() -> dict:
    """
    Build runtime context data (called once at startup).

    Returns:
        dict: Context data passed to all programs
    """
    # TODO: Load or generate data

    return {}
```

---

## DAG Builder

Visual tool for designing and debugging DAG pipelines.

See `tools/dag_builder/README.md` for detailed documentation.

**Quick Start:**
```bash
cd tools/dag_builder
./start.sh
```

Opens a web interface for:
- Visually designing pipeline stages
- Configuring stage connections
- Debugging data flow
- Exporting pipeline YAML

---

## Common Workflows

### 1. Analyze a Single Run
```bash
# Export data (assuming you ran: python run.py problem.name=my_problem redis.db=5)
python tools/redis2pd.py \
  --redis-db 5 \
  --redis-prefix "my_problem" \
  --output-file run5.csv

# Analyze in Python/Jupyter
import pandas as pd
df = pd.read_csv('run5.csv')
print(df.describe())
```

### 2. Compare Experiments
```bash
# Run multiple experiments
python run.py problem.name=test redis.db=10 experiment=base
python run.py problem.name=test redis.db=11 experiment=multi_island_complexity
python run.py problem.name=test redis.db=12 experiment=full_featured

# Compare results (prefix is just problem.name which is "test")
python tools/comparison.py \
  --run "test@10:Base" \
  --run "test@11:Multi_Island" \
  --run "test@12:Full_Featured" \
  --output-folder results/experiment_comparison
```

### 3. Extract Best Programs
```bash
# Export data (for problem.name=test in database 5)
python tools/redis2pd.py --redis-db 5 --redis-prefix "test" --output-file run.csv

# In Python, find best program
import pandas as pd
df = pd.read_csv('run.csv')
best = df.loc[df['fitness'].idxmax()]
print(f"Best program ID: {best['program_id']}")
print(f"Fitness: {best['fitness']}")

# Retrieve from Redis (full key includes the prefix)
redis-cli -n 5 GET "test:program:<program_id>:code"
```

---

## Tips

### Redis Key Prefixes
GigaEvo stores data with the problem name as prefix:
```
<problem_name>:program:<program_id>:*
```

For example, if you run `python run.py problem.name=heilbron`, the keys will be:
```
heilbron:program:<uuid>:code
heilbron:program:<uuid>:metrics
heilbron:archive
...
```

Find your prefix:
```bash
# List all keys in database to see the prefix pattern
redis-cli -n <db> KEYS "*:program:*" | head -1
```

**Important:** The `--redis-prefix` argument for tools should be just the problem name (e.g., `heilbron`), NOT the full key pattern.

### Clearing Old Data
```bash
# Flush specific database (removes ALL data in that database)
redis-cli -n 5 FLUSHDB

# Or delete by pattern for specific problem (careful!)
redis-cli -n 5 --scan --pattern "old_problem:*" | xargs redis-cli -n 5 DEL
```

### Large Datasets
For very large evolution runs, consider:
- Using `--iteration-rolling-window` to smooth noisy plots
- Sampling the data before exporting
- Using databases with persistence enabled

---

## Requirements

These tools require additional dependencies:
```bash
pip install pandas matplotlib seaborn
```

For DAG Builder:
```bash
cd tools/dag_builder
pip install -r requirements.txt
```
