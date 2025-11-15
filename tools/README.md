# GigaEvo Tools

Utility scripts for analyzing, visualizing, and managing GigaEvo evolution experiments.

## Analysis Tools

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
