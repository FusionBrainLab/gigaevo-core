# GigaEvo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Evolutionary algorithm framework that uses Large Language Models to automatically
improve programs through iterative mutation and selection (MAP-Elites). Programs
are Python functions; fitness is task performance. The framework is task-agnostic
and supports single runs, multi-island evolution, and prompt co-evolution.

## Demo

![Demo](./demos/demo-opt.gif)

## Getting Started

- **[Quick Start](docs/QUICKSTART.md)** — Get running in 5 minutes
- **[Architecture Guide](docs/ARCHITECTURE.md)** — System design overview

## Documentation

| Guide | Description |
|-------|-------------|
| [DAG System](docs/DAG_SYSTEM.md) | Execution engine: stages, dependencies, caching |
| [Evolution Strategies](docs/EVOLUTION_STRATEGIES.md) | MAP-Elites, multi-island, migration |
| [Prompt Co-Evolution](docs/COEVOLUTION.md) | Co-evolve mutation prompts alongside programs |
| [Tools](tools/README.md) | Analysis, debugging, and problem scaffolding utilities |
| [Usage Guide](docs/USAGE.md) | Detailed usage and Hydra configuration |
| [Contributing](docs/CONTRIBUTING.md) | Guidelines for contributors |
| [Changelog](CHANGELOG.md) | Version history |

## Quick Start

### 1. Install

**Requirements:** Python 3.12+, Redis

```bash
pip install -e .
```

Install Redis if not already available:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Or run via Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Configure LLM Access

Create a `.env` file with your API key:

```bash
OPENAI_API_KEY=sk-or-v1-your-api-key-here

# Optional: Langfuse tracing
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start Redis

```bash
redis-server
```

### 4. Run Evolution

```bash
python run.py problem.name=heilbron
```

Evolution starts immediately. Logs are saved to `outputs/`.

## How It Works

1. **Load initial programs** from `problems/<name>/initial_programs/`
2. **Mutate programs** using LLMs (GPT, Claude, Gemini, Qwen, etc.)
3. **Evaluate fitness** by running each program's `entrypoint()` + `validate()`
4. **Select solutions** using MAP-Elites across a behavior space
5. **Repeat** for N generations

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Problem   │────▶│  Evolution  │────▶│     LLM     │
│  (programs, │     │   Engine    │     │  (mutation)  │
│   metrics)  │     │ (MAP-Elites)│     └──────┬──────┘
└─────────────┘     └──────┬──────┘            │
                           │                   ▼
                    ┌──────┴──────┐     ┌─────────────┐
                    │   Storage   │◀────│  Evaluator   │
                    │   (Redis)   │     │ (DAG Runner) │
                    └─────────────┘     └─────────────┘
```

## Customization

### Different Experiments

```bash
# Multi-island evolution (diverse solution exploration)
python run.py experiment=multi_island_complexity problem.name=heilbron

# Multi-LLM exploration (uses multiple models)
python run.py experiment=multi_llm_exploration problem.name=heilbron
```

### Common Overrides

```bash
# Limit generations
python run.py problem.name=heilbron max_generations=10

# Use different Redis database
python run.py problem.name=heilbron redis.db=5

# Change LLM model
python run.py problem.name=heilbron model_name=anthropic/claude-3.5-sonnet

# Preview config without running
python run.py problem.name=heilbron --cfg job
```

### Prompt Co-Evolution

Co-evolve the mutation prompts alongside your programs. A paired prompt run
evolves the system prompt used by the mutation LLM, selecting for prompts that
produce better mutations:

```bash
# Main run — uses co-evolved prompts from DB 6
python run.py problem.name=my_task pipeline=my_pipeline \
    prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6 redis.db=4

# Prompt run — evolves mutation prompts, reads outcomes from DB 4
python run.py problem.name=prompt_evolution pipeline=prompt_evolution \
    redis.db=6 main_redis_db=4 main_redis_prefix=my_task
```

See [Prompt Co-Evolution Guide](docs/COEVOLUTION.md) for the full architecture,
launch instructions, and monitoring.

## Configuration

GigaEvo uses [Hydra](https://hydra.cc/) for modular configuration. All config
files are in `config/`:

| Directory | Purpose | Key files |
|-----------|---------|-----------|
| `experiment/` | Complete experiment templates | `base.yaml`, `full_featured.yaml`, `multi_island_complexity.yaml` |
| `algorithm/` | Evolution algorithms | `single_island.yaml`, `multi_island.yaml` |
| `llm/` | LLM setups | `single.yaml`, `heterogeneous.yaml` |
| `pipeline/` | DAG execution pipelines | `standard.yaml`, `with_context.yaml`, `prompt_evolution.yaml` |
| `prompt_fetcher/` | Prompt sourcing | `fixed.yaml`, `coevolved.yaml` |
| `constants/` | Tunable parameters | `evolution.yaml`, `llm.yaml`, `islands.yaml`, `pipeline.yaml` |
| `loader/` | Program loading | `directory.yaml`, `redis_selection.yaml` |
| `logging/` | Backends | `tensorboard.yaml`, `wandb.yaml` |

Override any setting via command line:
```bash
python run.py experiment=full_featured max_generations=50 temperature=0.8
```

## Creating a Problem

1. Create a directory under `problems/`:
   ```
   problems/my_problem/
   ├── validate.py           # Fitness evaluation
   ├── metrics.yaml          # Metric specifications
   ├── task_description.txt  # Problem description for the LLM
   └── initial_programs/     # Seed programs
       ├── strategy1.py      # Must define entrypoint()
       └── strategy2.py
   ```

2. Run:
   ```bash
   python run.py problem.name=my_problem
   ```

Or use the wizard: `python -m tools.wizard config.yaml`

See `problems/heilbron/` for a complete example.

## Output

Results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:
- **Logs**: `evolution_*.log`
- **Programs**: Stored in Redis (export with `tools/redis2pd.py`)
- **Metrics**: TensorBoard / W&B (if configured)

## Tools

| Tool | Purpose |
|------|---------|
| `tools/redis2pd.py` | Export evolution data to CSV/DataFrame |
| `tools/comparison.py` | Compare runs with fitness curve plots |
| `tools/top_programs.py` | Extract best programs from archive |
| `tools/flush.py` | Safely flush Redis DBs (kills workers first) |
| `tools/experiment/archive_run.sh` | Archive run data before flush |
| `tools/dag_builder/` | Visual DAG pipeline designer |
| `tools/wizard/` | Interactive problem scaffolding |

See [tools/README.md](tools/README.md) for full documentation and Redis key schema.

## Testing

```bash
# Full test suite (uses fakeredis, no Redis server needed)
python -m pytest

# Specific area
python -m pytest tests/stages/
python -m pytest tests/evolution/

# With coverage
python -m pytest --cov=gigaevo --cov-report=term-missing

# Linting
ruff check . && ruff format --check .
```

## Troubleshooting

**Redis database not empty:**
```bash
# Use tools/flush.py (kills exec_runner workers first):
PYTHONPATH=. python tools/flush.py --db 0 --confirm

# Or use a different DB:
python run.py redis.db=1
```

**LLM connection issues:**
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test OpenRouter
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://openrouter.ai/api/v1/models
```

## License

MIT License — see [LICENSE](LICENSE).

## Citation

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
