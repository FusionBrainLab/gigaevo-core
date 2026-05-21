# GigaEvo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Evolutionary algorithm framework that uses Large Language Models to automatically
improve programs through iterative mutation and selection (MAP-Elites). Programs
are Python functions; fitness is task performance. The framework is task-agnostic
and supports single runs, multi-island evolution, and prompt co-evolution.

## Demo

![Demo](./docs/demos/demo-opt.gif)

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
| [Usage Guide](docs/USAGE.md) | Typed CLI overrides and experiment-file authoring |
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

Each shipped experiment file under `experiments/` exports
`build() -> ExperimentConfig` and runs end-to-end via the typed CLI:

```bash
python run.py experiments/base.py
```

Use `--dry-run` to validate the configuration and dump the resolved
tree to `outputs/{experiment_id}/config.json` without invoking the
engine. Override any field through tyro:

```bash
python run.py experiments/base.py --seed 7 --engine.max_generations 200
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

### Experiment Presets

Each shipped preset lives at `experiments/<name>.py` and exports a `build()`
function returning a fully resolved `ExperimentConfig`:

```bash
# Steady-state: continuous mutation/evaluation, ~8x throughput
python run.py experiments/steady_state.py

# Migration bus: parallel runs share rejected programs via Redis stream
python run.py experiments/migration_bus.py
python run.py experiments/migration_bus.py --redis.db 1 --engine.migration_bus.run_id heilbron@db1

# Steady-state + bus: maximum throughput with cross-run sharing
python run.py experiments/steady_state_bus.py

# Multi-island evolution (fitness + simplicity islands)
python run.py experiments/multi_island_complexity.py

# Multi-LLM exploration (diverse mutation models)
python run.py experiments/multi_llm_exploration.py

# Prompt co-evolution (evolve mutation prompts alongside programs)
python run.py experiments/prompt_coevolution.py
```

### Common Overrides

Overrides are nested by `ExperimentConfig` field name, parsed by tyro:

```bash
# Limit generations
python run.py experiments/base.py --engine.max_generations 10

# Use different Redis database
python run.py experiments/base.py --redis.db 5

# Validate and dump the resolved config without invoking the engine
python run.py experiments/base.py --dry-run
```

The resolved config dumps to `outputs/{experiment_id}/config.json` on every
invocation, giving a reproducibility record per run.

### Prompt Co-Evolution

Co-evolve the mutation prompts alongside your programs. Paired runs use the
`experiments/prompt_coevolution.py` preset on the main side; the paired
prompt-evolution loop runs against a different Redis database:

```bash
# Main run — uses co-evolved prompts from DB 6
python run.py experiments/prompt_coevolution.py --redis.db 4 \
    --prompt_fetcher.prompt_redis_db 6

# Prompt run — evolves mutation prompts, reads outcomes from DB 4
# (uses a second experiment file dedicated to prompt evolution)
```

See [Prompt Co-Evolution Guide](docs/COEVOLUTION.md) for the full architecture,
launch instructions, and monitoring.

## Configuration

Configuration lives in `gigaevo/config/` as typed Pydantic models. The public
surface is:

| Module | Role |
|--------|------|
| `gigaevo/config/schemas/` | Discriminated-union Pydantic schemas (algorithm, engine, llm, pipeline, problem, runner, experiment) |
| `gigaevo/config/defaults.py` | `Final`-typed module-level scalars (timeouts, retry counts, behavior-space resolutions) |
| `gigaevo/config/algorithm_presets.py` | One-liner builders for single / multi-island MAP-Elites |
| `gigaevo/config/engine_presets.py` | Generational / steady-state / bus / ring engine builders |
| `gigaevo/config/llm_presets.py` | OpenRouter / Gemini / OpenAI / bandit ensembles |
| `gigaevo/config/pipeline_presets.py` | Default / context / structural-metrics / problem-specific pipelines |
| `gigaevo/config/problem_presets.py` | Heilbron, HotpotQA, AlgoTune, AlphaEvolve |
| `gigaevo/config/runner_presets.py` | DAG runner |

An experiment file under `experiments/` composes these presets into an
`ExperimentConfig`. The shipped files double as the reference for writing your
own.

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

2. Create `experiments/my_problem.py` (start from `experiments/base.py`),
   point `ProblemConfig.problem_dir` at the directory you just created,
   and tune the algorithm / pipeline / engine selectors as needed.

3. Run it:
   ```bash
   python run.py experiments/my_problem.py
   ```

See `experiments/base.py` and `problems/heilbron/` for a complete example.

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
python run.py experiments/base.py --redis.db 1
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
