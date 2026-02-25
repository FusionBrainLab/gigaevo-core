# GigaEvo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-78%25-yellow)](https://github.com/KhrulkovV/gigaevo-core-internal/actions)

Evolutionary algorithm that uses Large Language Models (LLMs) to automatically improve programs through iterative mutation and selection.

## Demo

![Demo](./demos/demo-opt.gif)

## Getting Started

- **[Quick Start](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Understand the system design

## Documentation

- **[DAG System](docs/DAG_SYSTEM.md)** - Comprehensive guide to GigaEvo's execution engine
- **[Evolution Strategies](docs/EVOLUTION_STRATEGIES.md)** - MAP-Elites and multi-island evolution system
- **[Tools](tools/README.md)** - Helper utilities for analysis, debugging, and problem scaffolding
- **[Usage Guide](docs/USAGE.md)** - Detailed usage instructions
- **[Changelog](docs/CHANGELOG.md)** - Version history and changes
- **[Contributing](docs/CONTRIBUTING.md)** - Guidelines for contributors

## Quick Start

### 1. Install Dependencies

**Requirements:** Python 3.12+

```bash
pip install -e .
```

### 2. Set up Environment

Create a `.env` file with your OpenRouter API key:

```bash
OPENAI_API_KEY=sk-or-v1-your-api-key-here

# Optional: Langfuse tracing (for observability)
LANGFUSE_PUBLIC_KEY=<your_langfuse_public_key>
LANGFUSE_SECRET_KEY=<your_langfuse_secret_key>
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

### 3. Start Redis

```bash
redis-server
```

### 4. Run Evolution

```bash
python run.py problem.name=heilbron
```

That's it! Evolution will start and logs will be saved to `outputs/`.
To study results, check `tools` or start `tensorboard` / `wandb`.
Sample analysis code is available at `tools/playground.ipynb`.

## What Happens

1. **Loads initial programs** from `problems/heilbron/`
2. **Mutates programs** using LLMs (GPT, Claude, Gemini, etc.)
3. **Evaluates fitness** by running the programs
4. **Selects best solutions** using MAP-Elites algorithm
5. **Repeats** for multiple generations

## Customization

### Use a Different Experiment

```bash
# Multi-island evolution (explores diverse solutions)
python run.py experiment=multi_island_complexity problem.name=heilbron

# Multi-LLM exploration (uses multiple models)
python run.py experiment=multi_llm_exploration problem.name=heilbron
```

### Change Settings

```bash
# Limit generations
python run.py problem.name=heilbron max_generations=10

# Use different Redis database
python run.py problem.name=heilbron redis.db=5

# Change LLM model
python run.py problem.name=heilbron model_name=anthropic/claude-3.5-sonnet
```

## Configuration

GigaEvo uses a modular configuration system based on [Hydra](https://hydra.cc/). All configuration is in `config/`:

### Top-Level Configuration

- **`experiment/`** - Complete experiment templates (start here!)
  - `base.yaml` - Simple single-island evolution (default)
  - `full_featured.yaml` - Multi-island + multi-LLM exploration
  - `multi_island_complexity.yaml` - Two islands: performance + simplicity
  - `multi_llm_exploration.yaml` - Multiple LLMs for diverse mutations

### Component Configurations

- **`algorithm/`** - Evolution algorithms
  - `single_island.yaml` - Standard MAP-Elites
  - `multi_island.yaml` - Multiple independent populations with migration

- **`llm/`** - Language model setups
  - `single.yaml` - One LLM for all mutations
  - `heterogeneous.yaml` - Multiple LLMs (GPT, Claude, Gemini, etc.) for diverse mutations

- **`pipeline/`** - DAG execution pipelines
  - `auto.yaml` - Automatically selects pipeline (standard or contextual) based on problem
  - `standard.yaml` - Basic validation → execution → metrics
  - `with_context.yaml` - Includes contextual information extraction
  - `custom.yaml` - Template for custom pipelines

- **`constants/`** - Tunable parameters grouped by domain
  - `evolution.yaml` - Generation limits, mutation rates, selection pressure
  - `llm.yaml` - Temperature, max tokens, retry logic
  - `islands.yaml` - Island sizes, migration frequency, diversity settings
  - `pipeline.yaml` - Stage timeouts, parallelization settings
  - `redis.yaml` - Connection settings, key patterns
  - `logging.yaml` - Log levels, output formats
  - `runner.yaml` - DAG execution settings
  - `endpoints.yaml` - API endpoint defaults

### Supporting Configurations

- **`loader/`** - Program loading strategies
  - `directory.yaml` - Load initial programs from filesystem
  - `redis_selection.yaml` - Load from existing Redis archive

- **`logging/`** - Logging backends
  - `tensorboard.yaml` - TensorBoard integration
  - `wandb.yaml` - Weights & Biases tracking

- **`metrics/`** - Metric computation
  - `default.yaml` - Basic fitness metrics
  - `code_complexity.yaml` - Includes cyclomatic complexity, LOC, etc.

- **`redis/`** - Redis storage backend
- **`runner/`** - DAG runner configuration
- **`evolution/`** - Core evolution engine settings

### Configuration Overrides

Override any setting via command line:

```bash
# Override experiment
python run.py experiment=full_featured

# Override specific settings
python run.py problem.name=heilbron max_generations=50 temperature=0.8

# Override nested settings
python run.py constants.evolution.mutation_rate=0.3
```

See individual YAML files for detailed documentation on each component.

## Output

Results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:

- **Logs**: `evolution_YYYYMMDD_HHMMSS.log`
- **Programs**: Stored in Redis for fast access
- **Metrics**: TensorBoard logs (if enabled)

## Troubleshooting

### Redis Database Not Empty

If you see:
```
ERROR: Redis database is not empty!
```

Flush the database manually:
```bash
redis-cli -n 0 FLUSHDB
```

Or use a different database number:
```bash
python run.py redis.db=1
```

### LLM Connection Issues

Check your API key in `.env`:
```bash
echo $OPENAI_API_KEY
```

Verify OpenRouter is accessible:
```bash
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://openrouter.ai/api/v1/models
```

## Architecture

```
┌─────────────┐
│   Problem   │  Define task, initial programs, metrics
└──────┬──────┘
       │
       v
┌─────────────┐
│  Evolution  │  MAP-Elites algorithm
│   Engine    │  Selects parents, generates mutations
└──────┬──────┘
       │
       v
┌─────────────┐
│     LLM     │  Generates code mutations
│   Wrapper   │  (GPT, Claude, Gemini, etc.)
└──────┬──────┘
       │
       v
┌─────────────┐
│  Evaluator  │  Runs programs, computes fitness
│ (DAG Runner)│  Validates solutions
└──────┬──────┘
       │
       v
┌─────────────┐
│   Storage   │  Redis for fast program access
│   (Redis)   │  Maintains archive of solutions
└─────────────┘
```

## Key Concepts

- **MAP-Elites**: Algorithm that maintains diverse solutions across behavior dimensions
- **Islands**: Independent populations that can exchange solutions (migration)
- **DAG Pipeline**: Stages for validation, execution, complexity analysis, etc.
- **Behavior Space**: Multi-dimensional grid dividing solutions by characteristics

## Advanced Usage

### Generate Problem with Wizard

Create problem scaffolding from YAML configuration:

```bash
python -m tools.wizard heilbron.yaml
```

See `tools/README.md` for detailed wizard documentation.

### Create Your Own Problem Manually

1. Create directory in `problems/`:
   ```
   problems/my_problem/
     - validate.py           # Fitness evaluation function
     - metrics.yaml          # Metrics specification
     - task_description.txt  # Problem description
     - initial_programs/     # Directory with initial programs
       - strategy1.py        # Each contains entrypoint() function
       - strategy2.py
     - helper.py             # Optional: utility functions
     - context.py            # Optional: runtime context builder
   ```

2. Run:
   ```bash
   python run.py problem.name=my_problem
   ```

See `problems/heilbron/` for a complete example.

### Custom Experiment

Copy an existing experiment and modify:

```bash
cp config/experiment/base.yaml config/experiment/my_experiment.yaml
# Edit my_experiment.yaml...
python run.py experiment=my_experiment
```

## Tools

GigaEvo includes utilities for analysis and visualization:

- **`tools/redis2pd.py`** - Export evolution data to CSV
- **`tools/comparison.py`** - Compare multiple runs with plots
- **`tools/dag_builder/`** - Visual DAG pipeline designer
- **`tools/wizard/`** - Interactive problem setup

See `tools/README.md` for detailed documentation.

## Testing

GigaEvo uses [pytest](https://docs.pytest.org/) with [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) for async test support. Tests use `fakeredis` to avoid needing a running Redis server.

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run the full test suite
python -m pytest

# Run a specific subdirectory
python -m pytest tests/stages/
python -m pytest tests/evolution/

# Run a single test file
python -m pytest tests/evolution/test_elite_selectors.py

# Run a specific test by name
python -m pytest tests/evolution/test_elite_selectors.py::TestFitnessProportionalTemperature -v

# Run with verbose output
python -m pytest -v

# Run only tests matching a keyword
python -m pytest -k "optuna" -v

# Run with coverage
python -m pytest --cov=gigaevo --cov-report=term-missing
```

### Test Structure

Tests are organized into subdirectories that mirror the source layout:

```
tests/
├── conftest.py              # Shared fixtures (fakeredis, mock stages, factories)
├── stages/                  # Pipeline stage unit tests
│   ├── test_stage_execute.py            # Stage.execute() return dispatch, timeout, cleanup
│   ├── test_stage_base_extended.py      # __init_subclass__ validation, _is_optional_type,
│   │                                    #   VoidOutput, compute_hash_from_inputs exceptions
│   ├── test_metrics_stages.py           # EnsureMetricsStage, NormalizeMetricsStage
│   ├── test_complexity.py               # AST complexity analysis, code length
│   ├── test_json_processing.py          # MergeDictStage, ParseJSON, StringifyJSON
│   ├── test_formatter.py                # FormatterStage (None, string, repr paths)
│   ├── test_langgraph_stage.py          # LangGraphStage postprocess, preprocess, errors
│   ├── test_collector.py                # ProgramIds, descendants, ancestors, stats
│   ├── test_mutation_context.py         # MutationContextStage optional input combos
│   ├── test_lineage_stages.py           # LineagesToDescendants, LineagesFromAncestors
│   ├── test_validation_stage.py         # Code validation and syntax checking
│   ├── test_validation_extended.py      # Invalid regex, AST file ops, import edge cases
│   ├── test_python_executors.py         # Exec runner, worker pool, timeouts
│   ├── test_optuna_optimization.py      # Optuna search-space, trials, parameter freezing,
│   │                                    #   time-budget deadline
│   ├── test_cma_optimization.py         # CMA-ES numerical optimization
│   ├── test_cma_optimization_extended.py  # _should_extract, _extract_constants, _substitute,
│   │                                    #   adaptive penalty via _evaluate_population, sign convention
│   ├── test_optimization_utils.py       # format_value_for_source, make_numeric_const_node,
│   │                                    #   read_validator, build_eval_code
│   └── test_desubstitution_extended.py  # _coerce_param_value, _find_matching_close_paren,
│                                        #   _clean_eval_in_source, desubstitute_params
├── dag/                     # DAG runner and scheduling
│   ├── test_dag_automata.py             # Stage state machine transitions
│   ├── test_dag_automata_extended.py    # is_satisfied_historically, non-Stage validation,
│   │                                    #   duplicate input_name, _check_dataflow_gate, explain_blockers
│   ├── test_dag_execution.py            # Individual stage execution, timeouts, caching
│   ├── test_dag_integration.py          # End-to-end DAG pipeline runs
│   ├── test_dag_complex_integration.py  # Complex topologies, failure propagation
│   ├── test_dag_internals.py            # Dependency resolution, topological ordering
│   ├── test_dag_caching.py              # Stage result caching strategies
│   ├── test_dag_runner.py               # DagRunner cleanup, crash paths, scheduling
│   └── test_dag_compatibility_extended.py  # _normalize_annotation, _covariant_type_compatible
├── evolution/               # Evolution engine and strategies
│   ├── test_evolution_engine.py     # Generation loop, ingestion, exception handling
│   ├── test_island.py               # MapElitesIsland add, size limit, reindex, elites
│   ├── test_mutation_operator.py    # LLMMutationOperator with mocked LLM agent
│   ├── test_elite_selectors.py      # Fitness-proportional, tournament, Pareto selectors
│   ├── test_elite_selectors_extended.py  # RandomEliteSelector, inf/nan fallback, Pareto
│   │                                    #   constructor guards, custom tie-breaker
│   ├── test_strategy_utils.py       # weighted_sample_without_replacement, extract_fitness_values,
│   │                                #   dominates
│   ├── test_selectors.py            # Parent selection strategies
│   ├── test_acceptors.py            # Program acceptance criteria
│   ├── test_removers.py             # Archive removal strategies
│   ├── test_merge_strategies.py     # Program merge conflict resolution
│   ├── test_bandit.py               # Multi-armed bandit LLM model selector
│   ├── test_behavior_space.py       # Behavior space binning and dynamics
│   └── test_archive_storage.py      # Redis-backed archive operations
├── database/                # Storage and state management
│   ├── test_redis_storage.py        # Redis CRUD, locking, merge strategies
│   ├── test_redis_connection.py     # Connection pooling, retries, reconnection
│   ├── test_state_manager.py        # Program state transitions, concurrent updates
│   ├── test_state_consistency.py    # Cross-component state invariants
│   └── test_program_state.py        # Program state machine validation
└── llm/                     # LLM integration
    └── test_llm_routing.py          # MultiModelRouter, token tracking
```

### Shared Fixtures

`tests/conftest.py` provides reusable fixtures:

- `fakeredis_storage` — `RedisProgramStorage` backed by in-memory `fakeredis` (no Redis server needed)
- `state_manager` — `ProgramStateManager` wrapping the fake storage
- `make_program` — factory for creating `Program` objects with configurable state, metrics, and stage results
- `null_writer` — no-op `LogWriter` for tests that need a metrics sink
- Mock stages — `FastStage`, `FailingStage`, `SlowStage`, `VoidStage`, `SideEffectStage`, etc.

### Linting

```bash
# Run all pre-commit hooks (ruff format + lint, trailing whitespace, YAML check)
pre-commit run --all-files

# Or run individually
ruff check .       # lint
ruff format .      # format
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use GigaEvo in your research, please cite:

```bibtex
@misc{khrulkov2025gigaevoopensourceoptimization,
      title={GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms},
      author={Valentin Khrulkov and Andrey Galichin and Denis Bashkirov and Dmitry Vinichenko and Oleg Travkin and Roman Alferov and Andrey Kuznetsov and Ivan Oseledets},
      year={2025},
      eprint={2511.17592},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2511.17592},
}
```
