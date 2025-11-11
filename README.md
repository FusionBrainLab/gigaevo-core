# GigaEvo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Evolutionary algorithm that uses Large Language Models (LLMs) to automatically improve programs through iterative mutation and selection.

## Documentation

- **[DAG System](gigaevo/programs/dag/README.md)** - Comprehensive guide to GigaEvo's execution engine
- **[Evolution Strategies](gigaevo/evolution/strategies/README.md)** - MAP-Elites and multi-island evolution system
- **[Tools](tools/README.md)** - Helper utilities for analysis and debugging
- **[Contributing](CONTRIBUTING.md)** - Guidelines for contributors

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
To study results, check `tools` or start `tensorboard` / `wandb`

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

All configuration is in `config/`:

- **`experiment/`** - Complete experiment setups (start here!)
  - `base.yaml` - Simple single-island evolution (default)
  - `full_featured.yaml` - Multi-island + multi-LLM
  - `multi_island_complexity.yaml` - Two islands: performance + simplicity

- **`constants/`** - Tunable parameters split by domain
  - `evolution.yaml` - Generation limits, mutation rates
  - `llm.yaml` - Temperature, max tokens, etc.
  - `islands.yaml` - Island sizes, migration settings

- **`algorithm/`** - MAP-Elites configurations
- **`llm/`** - LLM provider setups

See `config/` for detailed documentation on each component.

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

### Create Your Own Problem

1. Create directory in `problems/`:
   ```
   problems/my_problem/
     - __init__.py
     - entrypoint.py    # Your function to evolve
     - validate.py      # Fitness evaluation
     - metrics.yaml     # Define metrics
     - initial_programs # Directory containing a number of initial programs
   ```

2. Run:
   ```bash
   python run.py problem.name=my_problem
   ```

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
- **`tools/wizard.py`** - Interactive problem setup

See `tools/README.md` for detailed documentation.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use GigaEvo in your research, please cite:

```bibtex
[Citation coming soon]
```
