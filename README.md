# MetaEvolve: LLM-based Evolutionary Optimization System

MetaEvolve is a flexible, professional-grade evolutionary optimization system that uses Large Language Models (LLMs) for code generation and mutation. It's designed to solve complex optimization problems through multi-island evolution with diverse behavior spaces.

## Features

- **Flexible Problem Configuration**: Support for any optimization problem through configurable problem directories
- **Multi-Island Evolution**: MAP-Elites algorithm with multiple specialized islands
- **LLM-Based Mutation**: Intelligent code generation using state-of-the-art language models
- **Performance Monitoring**: Built-in system performance tracking and auto-optimization
- **Modular Architecture**: Clean separation of concerns with extensible DAG-based execution
- **Professional CLI**: Comprehensive command-line interface with extensive configuration options

## Installation

Recommended Python version: 3.12+

```bash
# Clone the repository
git clone <repository-url>
cd metaevolve

# Install the package in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# For examples and visualization
pip install -e ".[examples]"

# Set up environment variables
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Quick Start

### Basic Usage

First we need to launch redis-server as a separate process.

```bash
redis-server
```

```bash
# Run evolution on the hexagon packing problem
python run.py --problem-dir problems/hexagon_pack

# Use different Redis database
python run.py --problem-dir problems/hexagon_pack --redis-db 1
```

### Advanced Configuration

```bash
# Full configuration example
python run.py \
    --problem-dir problems/hexagon_pack \
    --redis-host localhost \
    --redis-port 6379 \
    --redis-db 2 \
    --max-concurrent-dags 8 \
    --log-level DEBUG \
    --log-dir logs/experiment_1
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

- **`helper.py`**: Utility functions that solutions can import
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
# Basic hexagon packing optimization
python run.py --problem-dir problems/hexagon_pack

# With high performance settings
python run.py --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 12 \
    --log-level INFO
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

# With extended evolution
python run.py --problem-dir problems/optimization \
    --add-context \
    --max-generations 100 \
    --log-level DEBUG
```

### 3. Circle Packing (`problems/toy_example/`)

**Problem**: Arrange 9 non-overlapping circles with variable radii in a unit square to maximize total radius sum.

**Type**: Geometric optimization with variable sizing

**Key Features**:
- Variable circle sizes (adaptive radius optimization)
- Unit square constraint
- Multiple initial strategies (simple, jittered, optimized)

**Usage**:
```bash
# Circle packing optimization
python run.py --problem-dir problems/toy_example

# With performance monitoring disabled for speed
python run.py --problem-dir problems/toy_example \
    --max-concurrent-dags 16
```

## Configuration Options

### Command Line Arguments

#### Required
- `--problem-dir`: Directory containing problem files

#### Context Configuration
- `--add-context`: Enable context mode (required for problems with `context.py`)

#### Redis Configuration
- `--redis-url`: Redis URL, e.g. `redis://host:port/db` (overrides host/port/db)
- `--redis-host`: Redis hostname (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--redis-db`: Redis database number (default: 0)

#### Evolution Configuration
- `--max-generations`: Maximum number of generations (default: unlimited)
- `--population-size`: Initial population size (default: auto-determined)

#### Redis Selection Mode
- `--use-redis-selection`: Use existing Redis programs instead of initial_programs/
- `--source-redis-url`: Source Redis URL for program selection (overrides source host/port/db)
- `--source-redis-db`: Source Redis database for program selection (default: 0)
- `--top-n`: Number of top programs to select by fitness (default: 50)

#### Performance Configuration
- `--max-concurrent-dags`: Maximum concurrent DAG executions (default: 6)

#### Logging Configuration
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-dir`: Directory for log files (default: logs)

### Environment Variables

- `OPENROUTER_API_KEY`: Required for LLM access
- `REDIS_HOST`: Override default Redis host
- `REDIS_PORT`: Override default Redis port
- `REDIS_DB`: Override default Redis database

## Advanced Usage Examples

### Continuing Evolution from Previous Run

```bash
# Run initial evolution
python run.py --problem-dir problems/hexagon_pack \
    --redis-db 1

# Continue evolution using best programs from previous run
python run.py --problem-dir problems/hexagon_pack \
    --redis-db 2 \
    --use-redis-selection \
    --source-redis-db 1 \
    --top-n 30
```

### High-Performance Configuration

```bash
# Maximum performance setup
python run.py --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 16 \
    --log-level WARNING \
    --redis-db 3
```

### Debugging and Development

```bash
# Full debugging information
python run.py --problem-dir problems/toy_example \
    --log-level DEBUG \
    --log-dir debug_logs \
    --max-concurrent-dags 2
```

### Remote Redis Configuration

```bash
# Using remote Redis server
python run.py --problem-dir problems/optimization \
    --add-context \
    --redis-url redis://redis-server.example.com:6380/5
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
- **Code Validation**: Syntax checking and compilation verification
- **Sandboxed Execution**: Safe program execution with resource limits
- **Multi-Stage Evaluation**: Custom fitness, behavior, and complexity evaluation
- **Metrics Collection**: Comprehensive performance and structural analysis

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

The system uses three specialized islands:

1. **Fitness Island**: Focuses on pure fitness optimization using fitness and validity dimensions
2. **Simplicity Island**: Balances performance and code elegance using Pareto optimization across fitness, complexity_score, and validity dimensions
3. **Entropy Island**: Rewards fitness combined with structural diversity using fitness, AST entropy, and validity dimensions

### Execution Pipeline

1. **Validation**: Check code compilation and syntax
2. **Execution**: Run the program to generate solutions
3. **Complexity Analysis**: Compute structural metrics
4. **Validation**: Evaluate solution quality
5. **Insights Generation**: Generate LLM-based insights
6. **Metrics Collection**: Aggregate performance data

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

### 3. Program Evaluation Pipeline
Each generated program goes through a multi-stage DAG pipeline:

1. **Validation Stage**: Check syntax and compilation
2. **Execution Stage**: Run the program with safety limits
3. **Complexity Analysis**: Compute structural metrics (AST entropy, node count, etc.)
4. **Domain Validation**: Run problem-specific validation function
5. **Insights Generation**: LLM-based analysis and feedback
6. **Metrics Collection**: Aggregate performance data

### 4. Multi-Island Strategy
The system maintains three specialized islands:

- **🏆 Fitness Island**: Pure optimization for best solutions using fitness proportional selection
- **⚖️ Simplicity Island**: Balance between performance and code elegance using Pareto front optimization (fitness vs. complexity)  
- **🌀 Entropy Island**: Reward fitness combined with structural diversity using Pareto front optimization (fitness vs. AST entropy)

### 5. Migration & Selection
- Programs migrate between islands based on performance
- Each island uses specialized selection criteria (Pareto fronts, fitness proportional, etc.)
- Archive management prevents overcrowding while preserving diversity

### 6. LLM-Driven Mutation
- Context-aware code generation using problem-specific prompts
- Incorporates insights from previous successful programs
- Supports both incremental improvements and radical rewrites

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

Add at least one Python file to the `initial_programs/` directory. The expected function name is `entrypoint`.

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

## Performance Monitoring

MetaEvolve includes comprehensive performance monitoring:

- **Real-time Metrics**: CPU, memory, and Redis usage
- **Auto-optimization**: Automatic resource adjustment
- **Alert System**: Performance degradation warnings
- **Dashboard Logging**: Periodic performance summaries

## 📊 Performance

MetaEvolve is designed for high performance and scalability. Benchmarks on standard hardware (8-core CPU, 16GB RAM):

| Metric | Value |
|--------|-------|
| Programs/second | 100-500+ |
| Concurrent DAGs | 6-16 (configurable) |
| Evolution cycles/second | 5-20 |
| Memory usage | <1GB for 10k programs |
| Redis operations/second | 1000+ |

### Performance Tuning

```bash
# High-performance configuration
python run.py \
    --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 12 \
    --redis-db 1

# Memory-efficient configuration  
python run.py \
    --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 4 \
    --log-level WARNING
```

## ⚙️ Advanced Configuration

### Evolution Engine Settings

The system can be fine-tuned by modifying the configuration in the source code:

```python
# In run.py - create_evolution_strategy()
engine_config = EngineConfig(
    loop_interval=1.0,                    # Evolution frequency (seconds)
    max_elites_per_generation=6,          # Selection pressure
    max_mutations_per_generation=8,       # Exploration rate
    required_behavior_keys=behavior_keys, # Required metrics
    log_validation_failures=True          # Debug failed programs
)
```

### Runner Configuration

```python
# In run.py - run_evolution_experiment()
runner_config = RunnerConfig(
    poll_interval=5.0,                    # DAG polling frequency
    max_concurrent_dags=6,                # Parallelism level
    log_interval=15                       # Status reporting interval
)
```

### Behavior Space Customization

```python
# In run.py - create_behavior_spaces()
fitness_validity_space = BehaviorSpace(
    feature_bounds={
        'fitness': (-7.0, -3.93),         # Expected fitness range
        'is_valid': (-0.01, 1.01)         # Binary validity flag
    },
    resolution={
        'fitness': 30,                     # Discretization granularity
        'is_valid': 2
    },
    binning_types={
        'fitness': BinningType.LINEAR,     # Linear or logarithmic binning
        'is_valid': BinningType.LINEAR
    }
)
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENROUTER_API_KEY` is set
2. **Redis Connection**: Check Redis server is running
3. **Problem Directory**: Verify all required files are present
4. **Permissions**: Ensure write access to log directory
5. **Context Problems**: Use `--add-context` flag for problems with `context.py`

### Debug Mode

Enable detailed debugging by setting:

```bash
--log-level DEBUG
```

### Log Files

Logs are automatically rotated and stored in:
- Default: `logs/` directory
- Custom: Use `--log-dir` option

## Advanced Usage

### Custom Behavior Spaces

Modify `create_behavior_spaces()` to define custom optimization criteria.

### Custom Mutation Operators

Extend `LLMMutationOperator` for problem-specific mutations.

### Custom Execution Stages

Add new stages to the DAG pipeline in `create_dag_stages()`.

