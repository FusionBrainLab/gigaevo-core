# GigaEvo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
