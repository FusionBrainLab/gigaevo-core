# MetaEvolve: LLM-based Evolutionary Optimization System

## Installation

Recommended Python version: 3.12+

```bash
# Clone the repository
git clone <repository-url>
cd metaevolve
pip install -e .

# Set up environment variables
export OPENAI_API_KEY=<your_llm_api_key_here> (required)
```
or using `.env` file 

## Quick Start

### Basic Usage

First we need to launch redis-server as a separate process.

```bash
redis-server
```

### Legacy approach

`run.py` is pure python example of launching the evolution which can be easily tweaked

```bash
# Run evolution on the hexagon packing problem
python run.py --problem-dir problems/hexagon_pack

# Use different Redis database
python run.py --problem-dir problems/hexagon_pack --redis-db 1
```

### Hydra-based configs (Recommended)

`run_hydra.py` utilizes composable hydra configs for experiments. See `config` folder to undertstand how the config is composed
example runs
```bash
python run_hydra.py problem.name=heilbron_simplified
python run_hydra.py problem.name=heilbron_simplified redis.db=1 constants.num_parents=1  constants.default_llm_base_url=<my_api_endpoint>
```


## Problem Directory Structure

Each problem must be organized in a specific directory structure:

```
problems/your_problem/
├── task_description.txt          # Problem description
├── task_hints.txt               # Optimization hints
├── validate.py                  # Validation function
├── mutation_system_prompt.txt   # LLM system prompt
├── mutation_user_prompt.txt     # LLM user prompt
├── helper.py                    # Helper functions (optional)
├── context.py                   # Context builder (optional)
└── initial_programs/            # Initial population strategies (required)
    ├── strategy1.py
    ├── strategy2.py
    └── ...
```

### Required Files and Directories

1. **`task_description.txt`**: Clear description of the optimization problem
2. **`task_hints.txt`**: Guidance and hints for the optimization process
3. **`validate.py`**: Must contain a `validate()` function that evaluates solutions
4. **`mutation_system_prompt.txt`**: System prompt for LLM-based mutations
5. **`mutation_user_prompt.txt`**: User prompt template for LLM mutations
6. **`initial_programs/`**: Directory with at least one Python file containing initial population strategies

### Optional Files

- **`helper.py / <any additional py files>`**: Auxiliary functions that solutions can import
- **`context.py`**: Context builder function for problems requiring external data

## Example Problems

The system includes three example problems demonstrating different types of optimization challenges:

### 1. Hexagon Packing (`problems/hexagon_pack/`)

**Problem**: Arrange 11 unit regular hexagons inside a larger enclosing hexagon to minimize the enclosing hexagon's side length.

**Type**: Geometric optimization without context

**Key Features**:
- Complex constraint satisfaction (non-overlapping)
- Geometric reasoning and spatial optimization
- Multiple initial strategies (hexagonal rings, spirals, clusters)

**Usage**:
```bash
python run.py --problem-dir problems/hexagon_pack
python run_hydra.py problem.name=hexagon_pack
```

### 2. Regression Optimization (`problems/optimization/`)

**Problem**: Learn a regression model from California housing dataset to predict house prices.

**Type**: Machine learning optimization with context

**Key Features**:
- Uses external data context (California housing dataset)
- Requires `--add-context` flag
- Demonstrates ML model evolution

**Usage**:
```bash
# Regression model optimization (note: requires --add-context)
python run.py --problem-dir problems/optimization \
    --add-context

python run_hydra.py problem.name=optimization
```

## Architecture

MetaEvolve uses a modular, high-performance architecture designed for scalability and flexibility:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Runner        │────│  Evolution       │────│  DAG Pipeline   │
│   Orchestrator  │    │  Engine          │    │  Executor       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────────────┐
                    │     Redis Storage       │
                    │   (Programs & State)    │
                    └─────────────────────────┘
```

### Core Components

#### 1. Evolution Engine
High-performance evolutionary loop with configurable strategies:
- **MapElitesMultiIsland**: Multi-island quality-diversity optimization with migration and specialization
- **LLM Integration**: Intelligent code generation using state-of-the-art language models
- **Adaptive Strategies**: Dynamic behavior space adjustment and fitness landscape exploration

#### 2. DAG Pipeline System
Flexible program execution pipeline with parallel processing:
- Execution-order deps: sequencing only (on_success/always_after)
- Dataflow via edges: edges carry data only, never gate readiness
- Mandatory/optional inputs: each stage declares (mandatory, optional_max)
- Code Validation: Syntax checking and compilation verification
- Sandboxed Execution: Safe program execution with resource limits
- Multi-Stage Evaluation: Custom fitness, behavior, and complexity evaluation
- Metrics Collection: Comprehensive performance and structural analysis

#### 3. Runner Orchestration
Coordinates evolution and execution with high concurrency:
- **Concurrent Processing**: Multiple DAG pipelines running in parallel
- **Resource Management**: Configurable concurrency limits and memory allocation
- **Monitoring**: Real-time metrics, performance tracking, and auto-optimization

#### 4. Redis Storage System
Persistent, high-performance program and state management:
- **Async Operations**: Non-blocking Redis operations for maximum throughput
- **Program Versioning**: Full program history and metadata tracking
- **State Persistence**: Evolution state survives restarts and failures

### Behavior Spaces

The system in default configuration uses one island:

**Fitness Island**: focuses on fitness purely

### Execution Pipeline

1. Validation: Check code compilation and syntax
2. Execution: Run the program to generate solutions
3. Domain Validation: Evaluate solution quality (fixed validator code)
4. Insights Generation: Generate LLM-based insights
5. Metrics Collection: Aggregate performance data

## 🔄 How It Works

MetaEvolve operates through a continuous cycle of evolution, evaluation, and optimization:

### 1. Initialization Phase
- Load initial programs from `initial_programs/` directory
- Populate Redis database with initial population
- Initialize multi-island MAP-Elites strategy with specialized behavior spaces

### 2. Evolution Loop
```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Evolution Loop                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Select Elite Programs  → 2. Generate Mutations              │
│    ↓                          ↓                                │
│ 4. Update Archives       ← 3. Evaluate via DAG Pipeline        │
│    ↓                                                           │
│ 5. Migrate Between Islands (periodically)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Creating New Problems

### Step 1: Scaffold with Wizard (recommended)

```bash
# Minimal scaffold
PYTHONPATH=. python tools/wizard.py problems/my_problem

# Include context.py and overwrite existing files
PYTHONPATH=. python tools/wizard.py problems/my_problem --add-context --overwrite

# With custom texts
PYTHONPATH=. python tools/wizard.py problems/my_problem \
  --task-description "Optimize X under Y" \
  --task-hints "Use A; consider B; avoid C" \
  --system-prompt "... {task_definition} ... {task_hints} ... {metrics_description} ..." \
  --user-prompt "=== Parents ({count}) ===\n{parent_blocks}\n"
```

### Manual Setup (alternative)

```bash
mkdir -p problems/my_problem/initial_programs
touch problems/my_problem/task_description.txt
touch problems/my_problem/task_hints.txt
touch problems/my_problem/validate.py
touch problems/my_problem/mutation_system_prompt.txt
touch problems/my_problem/mutation_user_prompt.txt
# Optional:
touch problems/my_problem/context.py
```

### Step 3: Implement Validation Function

```python
# problems/my_problem/validate.py
def validate(payload):
    """
    Validate and score the solution.
    
    Args:
        payload: For context problems: (context, solution_output)
                For non-context problems: solution_output
        
    Returns:
        dict: Metrics including 'fitness' and 'is_valid'
    """
    # Implement your validation logic here
    return {
        'fitness': your_fitness_score,
        'is_valid': 1 if valid else 0
    }
```

### Step 4: Create Initial Programs

Add at least one Python file to the `initial_programs/` directory. The expected function name is `entrypoint` (configurable in pipeline builder).

#### For Problems Without Context:
```python
# problems/my_problem/initial_programs/basic_solution.py
"""
Basic solution strategy for my_problem.
"""

def entrypoint():
    # Implement your basic solution here
    return solution_data
```

#### For Problems With Context:
```python
# problems/my_problem/initial_programs/basic_solution.py
"""
Basic solution strategy for my_problem.
"""

def entrypoint(context):
    # Implement your basic solution here
    return solution_data
```

### Step 5: Optional Context Implementation

For problems requiring external data, create a context builder:

```python
# problems/my_problem/context.py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def build_context() -> dict[str, np.ndarray]:
    """
    Build context data for the problem.
    
    Returns:
        dict: Context data that will be passed to entrypoint()
    """
    housing = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing[0], housing[1], test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test
    }
```

### Step 6: Run Evolution

#### For Non-Context Problems:
```bash
python run.py --problem-dir problems/my_problem
```

#### For Context Problems:
```bash
python run.py --problem-dir problems/my_problem --add-context
```

with hydra DAG is set automatically to include context generation by default

```bash
python run_hydra.py problem.name=my_problem
```
works for both cases

### Evolution analysis

There are several helper scripts included in `tools`
1) `redis2pd.py` converts evolution history stored in redis to .csv file which can be studied with pandas
2) `comparison.py` allows for comparing multiple / single evolution runs

#TODO add DAG tool decription