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

```bash
# Run evolution on the hexagon packing problem
python restart_llm_evolution_improved.py --problem-dir problems/hexagon_pack --min-fitness -7 --max-fitness -3.9

# Run with verbose logging
python restart_llm_evolution_improved.py --problem-dir problems/hexagon_pack --verbose --min-fitness -7 --max-fitness -3.9

# Use different Redis database
python restart_llm_evolution_improved.py --problem-dir problems/hexagon_pack --redis-db 1 --min-fitness -7 --max-fitness -3.9
```

### Advanced Configuration

```bash
# Full configuration example
python restart_llm_evolution_improved.py \
    --problem-dir problems/hexagon_pack \
    --redis-host localhost \
    --redis-port 6379 \
    --redis-db 2 \
    --max-concurrent-dags 8 \
    --log-level DEBUG \
    --log-dir logs/experiment_1 \
    --min-fitness -7 \
    --max-fitness -3.9 \
    --disable-monitoring
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

## Configuration Options

### Command Line Arguments

#### Required
- `--problem-dir`: Directory containing problem files

#### Redis Configuration
- `--redis-host`: Redis hostname (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--redis-db`: Redis database number (default: 0)

#### Evolution Configuration
- `--max-generations`: Maximum number of generations (default: unlimited)
- `--population-size`: Initial population size (default: auto-determined)
-- `--min-fitness`: Expected smallest value of fitness during the evolution
-- `--max-fitness`: Expected largest value of fitness during the evolution

#### Performance Configuration
- `--max-concurrent-dags`: Maximum concurrent DAG executions (default: 6)
- `--disable-monitoring`: Disable performance monitoring

#### Logging Configuration
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-dir`: Directory for log files (default: logs)
- `--verbose`: Enable verbose logging

### Environment Variables

- `OPENROUTER_API_KEY`: Required for LLM access
- `REDIS_HOST`: Override default Redis host
- `REDIS_PORT`: Override default Redis port
- `REDIS_DB`: Override default Redis database

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

The system uses four specialized islands:

1. **Fitness Island**: Focuses on pure fitness optimization
2. **Simplicity Island**: Balances performance and code elegance
3. **Structure Island**: Encourages scalable and diverse logic
4. **Entropy Island**: Rewards fitness with structural diversity

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
6. **Metrics Collection**: Aggregate all performance data

### 4. Multi-Island Strategy
The system maintains four specialized islands:

- **🏆 Fitness Island**: Pure optimization for best solutions
- **⚖️ Simplicity Island**: Balance between performance and code elegance  
- **🏗️ Structure Island**: Encourage scalable and diverse program logic
- **🌀 Entropy Island**: Reward fitness combined with structural diversity

### 5. Migration & Selection
- Programs migrate between islands based on performance
- Each island uses specialized selection criteria (Pareto fronts, fitness proportional, etc.)
- Archive management prevents overcrowding while preserving diversity

### 6. LLM-Driven Mutation
- Context-aware code generation using problem-specific prompts
- Incorporates insights from previous successful programs
- Supports both incremental improvements and radical rewrites

## Creating New Problems

### Step 1: Create Problem Directory

```bash
mkdir problems/my_problem
```

### Step 2: Create Required Files and Directories

```bash
# Create basic problem files
touch problems/my_problem/task_description.txt
touch problems/my_problem/task_hints.txt
touch problems/my_problem/validate.py
touch problems/my_problem/mutation_system_prompt.txt
touch problems/my_problem/mutation_user_prompt.txt

# Create initial programs directory
mkdir problems/my_problem/initial_programs
```

### Step 3: Implement Validation Function

```python
# problems/my_problem/validate.py
def validate(solution_output):
    """
    Validate and score the solution.
    
    Args:
        solution_output: Output from the solution program
        
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

Add at least one Python file to the `initial_programs/` directory with your problem-specific function (e.g., `construct_packing()` for hexagon packing):

```bash
# Example: Create a basic initial program
cat > problems/my_problem/initial_programs/basic_solution.py << 'EOF'
"""
Basic solution strategy for my_problem.
"""

def my_problem_function():
    # Implement your basic solution here
    return solution_data
EOF
```

### Step 5: Run Evolution

```bash
python restart_llm_evolution_improved.py --problem-dir problems/my_problem
```

## Performance Monitoring

MetaEvolve includes comprehensive performance monitoring:

- **Real-time Metrics**: CPU, memory, and Redis usage
- **Auto-optimization**: Automatic resource adjustment
- **Alert System**: Performance degradation warnings
- **Dashboard Logging**: Periodic performance summaries

Disable monitoring for maximum performance:

```bash
python restart_llm_evolution_improved.py --problem-dir problems/my_problem --disable-monitoring
```

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
python restart_llm_evolution_improved.py \
    --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 12 \
    --disable-monitoring \
    --redis-db 1

# Memory-efficient configuration  
python restart_llm_evolution_improved.py \
    --problem-dir problems/hexagon_pack \
    --max-concurrent-dags 4 \
    --log-level WARNING
```

## ⚙️ Advanced Configuration

### Evolution Engine Settings

The system can be fine-tuned by modifying the configuration in the source code:

```python
# In restart_llm_evolution_improved.py - create_evolution_strategy()
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
# In restart_llm_evolution_improved.py - run_evolution_experiment()
runner_config = RunnerConfig(
    poll_interval=5.0,                    # DAG polling frequency
    max_concurrent_dags=6,                # Parallelism level
    log_interval=15                       # Status reporting interval
)
```

### Behavior Space Customization

```python
# In restart_llm_evolution_improved.py - create_behavior_spaces()
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

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python restart_llm_evolution_improved.py --problem-dir problems/my_problem --verbose
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

