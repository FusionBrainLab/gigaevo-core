# GigaEvo

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
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
| [Adversarial Co-Evolution](docs/adversarial_coevolution.md) | Two-population co-evolution guide (generator/discriminator pattern) |
| [DAG System](docs/DAG_SYSTEM.md) | Execution engine: stages, dependencies, caching |
| [Evolution Strategies](docs/EVOLUTION_STRATEGIES.md) | MAP-Elites, multi-island, migration |
| [Memory System](docs/memory.md) | How memory-augmented mutation works (writers, readers, providers, ideas tracker) |
| [Optuna Optimization](docs/OPTUNA_OPTIMIZATION.md) | LLM-driven hyperparameter sweeps for evolved programs |
| [Prompt Co-Evolution](docs/COEVOLUTION.md) | Co-evolve mutation prompts alongside programs |
| [Tools](tools/README.md) | Analysis, debugging, and problem scaffolding utilities |
| [Usage Guide](docs/USAGE.md) | Detailed usage and Hydra configuration |
| [Contributing](docs/CONTRIBUTING.md) | Guidelines for contributors |
| [Changelog](CHANGELOG.md) | Version history |

## Quick Start

### 1. Install

**Requirements:** Python 3.11+, Redis

GigaEvo ships with a minimal core and opt-in **extras** so installs stay fast
on firewalled/slow networks. Pick the install level that matches your use:

| Use case | Command |
|---|---|
| **Minimal** — engine + numpy exemplar problems + LLM mutation + core CLI (`status`, `top`, `trajectory`, `logs`, `flush`, `checkpoint`, `inspect`, `launch`, `watchdog`, `export`) | `pip install -e .` |
| **Common** — also runs chain/NLP problems (HoVer, HotpotQA, IFBench, gsm8k, …) + `gigaevo plot` / `gigaevo events` / `gigaevo profiler` | `pip install -e ".[chains,plotting]"` |
| **Full** — everything user-facing (chains, optimization, plotting, tracking, local-LLM runtime, memory platform) | `pip install -e ".[all]"` |
| **Developer** — full + linters, type-checkers, pytest, dag_builder dev API | `pip install -e ".[all,dev,test]"` |

À la carte mapping of features to extras:

| Feature / module | Required extras |
|---|---|
| `gigaevo plot`, `gigaevo events`, `gigaevo profiler` | `[plotting]` |
| Chain/prompt problems: HoVer, HotpotQA, IFBench, gsm8k, musique, papillon, pupa | `[chains]` |
| Optuna / CMA optimization stages | `[optimization]` |
| Alphaevolve / hexagon_improver / santa2025 problems (JAX, sympy, shapely) | `[optimization]` |
| W&B / TensorBoard tracker backends | `[tracking]` |
| sudoku local-runtime solver (torch + vllm) | `[local-llm]` |
| GAM memory **platform** backend (`use_api=True`) — local backend needs nothing | `[memory-platform]` |
| `tools/dag_builder` web API | `[dev]` (uvicorn) |

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
5. **Repeat** continuously (steady-state) until a `stopper` (e.g. `max_mutants`,
   wall-clock, fitness-plateau) fires

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

```bash
# Migration bus: parallel runs share rejected programs via Redis stream
python run.py experiment=migration_bus problem.name=heilbron redis.db=0
python run.py experiment=migration_bus problem.name=heilbron redis.db=1

# Multi-island evolution (fitness + simplicity islands)
python run.py experiment=multi_island_complexity problem.name=heilbron

# Multi-LLM exploration (diverse mutation models)
python run.py experiment=multi_llm_exploration problem.name=heilbron

# Prompt co-evolution (evolve mutation prompts alongside programs)
python run.py experiment=prompt_coevolution problem.name=heilbron \
    redis.db=4 prompt_fetcher.prompt_redis_db=6
```

### Common Overrides

```bash
# Cap total mutants (steady-state stopper budget)
python run.py problem.name=heilbron max_mutants=10

# Use different Redis database
python run.py problem.name=heilbron redis.db=5

# Change LLM model
python run.py problem.name=heilbron model_name=anthropic/claude-3.5-sonnet

# Pick a different stopper (wall-clock, fitness-plateau, ...)
python run.py problem.name=heilbron stopper=wall_clock

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
| `experiment/` | Complete experiment templates | `base.yaml`, `full_featured.yaml`, `migration_bus.yaml`, `multi_island_complexity.yaml`, `multi_llm_exploration.yaml`, `prompt_coevolution.yaml`, `steady_state_adversarial.yaml` |
| `algorithm/` | Evolution algorithms | `single_island.yaml`, `single_island_2d.yaml`, `multi_island.yaml`, `topology_3d.yaml` |
| `llm/` | LLM setups | `single.yaml`, `heterogeneous.yaml`, `heterogeneous_bandit.yaml`, `openrouter_bandit.yaml`, `openrouter_ensemble.yaml` |
| `pipeline/` | DAG execution pipelines | `auto.yaml` (default), `standard.yaml`, `with_context.yaml`, `custom.yaml`, `prompt_evolution.yaml` |
| `prompt_fetcher/` | Prompt sourcing | `fixed.yaml`, `coevolved.yaml` |
| `stopper/` | Stopping criteria | `max_mutants.yaml` (default), `wall_clock.yaml`, `fitness_plateau.yaml` |
| `constants/` | Tunable parameters | `evolution.yaml`, `llm.yaml`, `islands.yaml`, `pipeline.yaml`, `runner.yaml`, `endpoints.yaml`, `redis.yaml`, `logging.yaml` |
| `loader/` | Program loading | `directory.yaml`, `redis_selection.yaml` |
| `logging/` | Backends | `tensorboard.yaml`, `wandb.yaml` |

Override any setting via command line:
```bash
python run.py experiment=full_featured max_mutants=50 temperature=0.8
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
- **Programs**: Stored in Redis (export with `gigaevo export csv`)
- **Metrics**: TensorBoard / W&B (if configured)

## CLI Tools (`gigaevo`)

Installed via `pip install -e .`. Global flags: `-e/--experiment`, `-r/--run`, `-f/--format`.

| Command | Purpose |
|---------|---------|
| `gigaevo -e EXP status` | Live monitoring: gen, metrics, PIDs, watchdog |
| `gigaevo -r RUN trajectory` | Gen-by-gen fitness trajectory |
| `gigaevo -r RUN top` | Inspect best programs by fitness |
| `gigaevo -e EXP plot comparison -o DIR` | Multi-run fitness curve plots |
| `gigaevo -e EXP plot arms-race -o DIR` | Dual-panel adversarial arms-race plot |
| `gigaevo -e EXP profiler` | Profile runner logs into text summary + HTML dashboard |
| `gigaevo -e EXP manifest gate <status>` | Hard-gate on experiment status (preregistered/implemented/running/complete) |
| `gigaevo -r RUN export csv -o FILE` | Export evolution data to CSV |
| `gigaevo flush --db N --confirm` | Safely flush Redis DBs (kills workers first) |
| `gigaevo -e EXP launch` / `watchdog` | Launch + supervise an experiment |
| `tools/experiment/archive_run.sh` | Archive run data before flush |
| `tools/dag_builder/` | Visual DAG pipeline designer |
| `tools/wizard/` | Interactive problem scaffolding |

See [tools/README.md](tools/README.md) for full CLI reference and Redis key schema.

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
# Flush (kills exec_runner workers first):
gigaevo flush --db 0 --confirm

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
