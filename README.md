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
- **[Generic tabular FeatureGraph evolution](problems/dag_tab/README.md)** — Evolve dataset-parameterized JSON feature DAGs with fixed CatBoost evaluation
- **[Tabular DAG estimator baselines](problems/tabular_dag_baselines/README.md)** — Evolve or cross-evaluate the same FeatureGraphs with CatBoost, TabM, RealMLP, TabICLv2, TabPFN v3, TabFM 1.0, LightGBM, or XGBoost
- **[Interface-ablation arms](problems/aaai_submit/README.md)** — Fourteen ACI / hexagon / spherical-code families on one frozen grading harness, grouped as a study bundle

## Documentation

| Guide | Description |
|-------|-------------|
| [Adversarial Co-Evolution](docs/adversarial_coevolution.md) | Two-population co-evolution guide (generator/discriminator pattern) |
| [DAG System](docs/DAG_SYSTEM.md) | Execution engine: stages, dependencies, caching |
| [Evolution Strategies](docs/EVOLUTION_STRATEGIES.md) | MAP-Elites, multi-island, migration |
| [Memory System](docs/memory.md) | How memory-augmented mutation works (arms, read/write paths, cards, observability) |
| [Optuna Optimization](docs/OPTUNA_OPTIMIZATION.md) | LLM-driven hyperparameter sweeps for evolved programs |
| [Prompt Co-Evolution](docs/COEVOLUTION.md) | Co-evolve mutation prompts alongside programs |
| [Tools](tools/README.md) | Analysis, debugging, and problem scaffolding utilities |
| [Usage Guide](docs/USAGE.md) | Detailed usage and Hydra configuration |
| [Contributing](docs/CONTRIBUTING.md) | Guidelines for contributors |
| [Changelog](CHANGELOG.md) | Version history |

## Quick Start

### 1. Install

**Requirements:** Python 3.11+

GigaEvo ships with a minimal core and opt-in **extras** so installs stay fast
on firewalled/slow networks. Pick the install level that matches your use:

| Use case | Command |
|---|---|
| **Minimal** — engine + numpy exemplar problems + LLM mutation + core CLI (`status`, `top`, `trajectory`, `logs`, `flush`, `checkpoint`, `inspect`, `launch`, `watchdog`, `export`) | `pip install -e .` |
| **Common** — also runs chain/NLP problems (HoVer, HotpotQA, IFBench, gsm8k, …) + `gigaevo plot` / `gigaevo events` / `gigaevo profiler` | `pip install -e ".[chains,plotting]"` |
| **Full** — everything user-facing (chains, optimization, plotting, tracking, and TabM evaluation) | `pip install -e ".[all]"` |
| **Developer** — full + linters, type-checkers, pytest, dag_builder dev API | `pip install -e ".[all,dev,test]"` |

À la carte mapping of features to extras:

| Feature / module | Required extras |
|---|---|
| `gigaevo plot`, `gigaevo events`, `gigaevo profiler` | `[plotting]` |
| Chain/prompt problems: HoVer, HotpotQA, IFBench, gsm8k, musique, papillon, pupa | `[chains]` |
| Optuna / CMA optimization stages | `[optimization]` |
| Alphaevolve / hexagon_improver / santa2025 problems (JAX, sympy, shapely) | `[optimization]` |
| `problems/aaai_submit/` interface-ablation arms and their `_harness` (JAX seeds) | `[optimization]` |
| W&B / TensorBoard tracker backends | `[tracking]` |
| TabM FeatureGraph evaluator | `[tabm-eval]` |
| GAM memory **platform** backend (`use_api=True`) — local backend needs nothing | `[memory-platform]` |
| `tools/dag_builder` web API | `[dev]` (uvicorn) |

### 2. Configure LLM Access

GigaEvo uses two LLM routes:

| Route | Used for | Config group |
|---|---|---|
| Main mutation LLM | Writes child programs | `llm=...` |
| Memory LLM | Writes/reconciles memory cards (`memory=v2` by default) | `memory/llm=...` |

Create a `.env` file with the keys required by the configs you select:

```bash
# Main default `llm=single` reads this key and sends requests to llm_base_url.
OPENAI_API_KEY=sk-or-v1-your-openrouter-or-proxy-key

# Default `memory/llm=gemini` reads this key.
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key

# Shared OpenAI-compatible endpoint for locally served models and chain evaluators.
LOCAL_LLM_PROXY=http://127.0.0.1:4000/v1

# Optional: Langfuse tracing
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

Provider/OpenRouter example for the main run:

```bash
python run.py problem.name=heilbron \
    llm=single \
    llm_base_url=https://openrouter.ai/api/v1 \
    model_name=google/gemini-3-flash-preview
```

Local LiteLLM/vLLM-compatible proxy example (`LOCAL_LLM_PROXY` is read from
`.env`):

```bash
export NO_PROXY=127.0.0.1,localhost

python run.py problem.name=heilbron \
    llm=local_proxy \
    model_name=Qwen/Qwen3-235B-A22B-Thinking-2507
```

Memory v2 is the base-experiment default. It can use a different, usually
cheaper, instruct model:

```bash
python run.py problem.name=heilbron \
    memory/llm=qwen_instruct \
    memory_bank_dir=$PWD/SHARE_HEILBRON_MEMORY
```

`llm=local_proxy` and `memory/llm=qwen_instruct` both read
`LOCAL_LLM_PROXY`; the latter uses `LITELLM_MASTER_KEY` when the proxy has a
non-default key. `memory/llm=gemini` reads `OPENROUTER_API_KEY`.

### 3. Choose Storage

Disk storage is the default. Programs, archives, and metrics are written under
the Hydra run directory (`outputs/.../storage` and `outputs/.../metrics`), so a
basic run needs no Redis server.

Redis storage is still available when you want a shared Redis DB workflow:

```bash
python run.py problem.name=heilbron storage=redis redis.db=0
```

For Redis-backed runs, start Redis first:

```bash
redis-server
```

### 4. Run Evolution

```bash
python run.py problem.name=heilbron
```

Evolution starts with singleton-parent memory v2, live card writing, and the
metadata-routing memory-guided pipeline. Before Bayesian selection, one agentic
research pass forms a 12-card slate with at least four uniformly randomized
discovery positions; unused research capacity is randomized too. Conditional
on proposing a card, normal runs deliver it with
probability 0.70 and retain a 0.30 matched control arm.
Logs are saved to `outputs/`, with disk-backed programs under the same run
directory.

### Common Run Recipes

```bash
# Fast smoke test
python run.py problem.name=heilbron max_mutants=5

# Explicit no-external-memory baseline
python run.py problem.name=heilbron pipeline=guided memory=none

# Redis-backed storage, if you need the Redis workflow
python run.py problem.name=heilbron storage=redis redis.db=5

# Default read/write memory v2 run (shown explicitly)
python run.py problem.name=heilbron \
    pipeline=memory_guided memory=v2 memory/write=live num_parents=1 \
    memory_bank_dir=$PWD/SHARE_HEILBRON_MEMORY

# Balanced memory-v2 validation (50% delivery / 50% control)
python run.py problem.name=heilbron \
    memory.posterior_config.reference_offer_probability=0.50 \
    memory.policy_config.offer_probability=0.50

# Build a shared memory bank (memory v2 reads + writes it)
python run.py problem.name=heilbron \
    pipeline=memory_guided memory=v2 \
    memory_bank_dir=$PWD/SHARE_HEILBRON_MEMORY

# Reuse the same bank on a later run (also refreshes it)
python run.py problem.name=heilbron \
    pipeline=memory_guided memory=v2 \
    memory_bank_dir=$PWD/SHARE_HEILBRON_MEMORY

# Reuse one bank-wide deduplicated card set and scale-free usefulness evidence
python run.py problem.name=<another-task> \
    memory=v2_multitask \
    memory_bank_dir=$PWD/SHARED_TABULAR_MEMORY

# Parameterized problems should give each dataset its own statistical task key
python run.py problem.name=dag_tab problem.dataset=adult \
    memory=v2_multitask memory_task_key=dag_tab/adult \
    memory_bank_dir=$PWD/SHARED_TABULAR_MEMORY

# JSON-document genomes, e.g. CARL chain problems
python run.py problem.name=chains/hover/full7 program_format=json_document
```

Use `python run.py ... --cfg job` to inspect the exact resolved config before
spending LLM budget.

### Harness-Based Evolution (agentic coding CLIs)

Instead of an HTTP endpoint, the LLM backend can be an agentic coding CLI —
the harness's subscription then pays for mutations instead of API tokens:

```bash
# Claude Code as the mutation engine (needs `claude` on PATH, logged in)
python run.py problem.name=heilbron llm=harness

# Codex CLI with a cheap model (needs `codex` on PATH, logged in)
python run.py problem.name=heilbron llm=codex
```

The CLI is driven headless, one process per call, with structured output,
per-call audit workspaces (`SYSTEM.md`, `USER.md`, `STDOUT.log`, …), token and
cost accounting, and tool/egress containment. `config/llm/harness.yaml`
(Claude Code) and `config/llm/codex.yaml` (Codex) are pre-wired; adapting
another CLI is a config-only exercise. See
[docs/USAGE.md](docs/USAGE.md#agentic-coding-harness-backend) for the full
contract and every knob.

## How It Works

1. **Load initial programs** from `problems/<name>/initial_programs/`
2. **Mutate programs** using LLMs (GPT, Claude, Gemini, Qwen, etc.)
3. **Evaluate fitness** by running each program's `entrypoint()` + `validate()`
4. **Build mutation feedback** from the parent history; memory-guided runs also
   retrieve cross-run memory cards
5. **Select solutions** using MAP-Elites across a behavior space
6. **Repeat** continuously (steady-state) until a `stopper` (e.g. `max_mutants`,
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
                    │ (Disk/Redis)│     │ (DAG Runner) │
                    └─────────────┘     └─────────────┘
```

## Customization

### Experiment Presets

```bash
# Migration bus: parallel runs share rejected programs via Redis stream
python run.py migration_bus=bus problem.name=heilbron storage=redis redis.db=0
python run.py migration_bus=bus problem.name=heilbron storage=redis redis.db=1

# Multi-island evolution (fitness + simplicity islands)
python run.py algorithm=multi_island metrics=code_complexity problem.name=heilbron

# Multi-LLM exploration (diverse mutation models)
python run.py llm=heterogeneous problem.name=heilbron

# Prompt co-evolution (evolve mutation prompts alongside programs)
python run.py experiment=prompt_coevolution problem.name=heilbron \
    storage=redis redis.db=4 prompt_fetcher.prompt_redis_db=6
```

### Common Overrides

```bash
# Cap total mutants (steady-state stopper budget)
python run.py problem.name=heilbron max_mutants=10

# Use Redis-backed storage on a specific DB
python run.py problem.name=heilbron storage=redis redis.db=5

# Change LLM model
python run.py problem.name=heilbron model_name=anthropic/claude-3.5-sonnet

# Pick a different stopper (wall-clock, fitness-plateau, ...)
python run.py problem.name=heilbron stopper=wall_clock

# Preview config without running
python run.py problem.name=heilbron --cfg job
```

### Provider example: z.ai (rate-limited endpoint)

z.ai's coding-paas endpoint only allows one in-flight request per key.
Use `llm=zai` and cap every concurrency layer at 1 — the
`llm_max_concurrent` semaphore serializes both producer-side mutation
generation and DAG-stage LLM calls behind a single global lock:

```bash
OPENAI_API_KEY=<your-zai-key> \
python run.py \
    problem.name=heilbron \
    max_mutants=10 \
    llm=zai \
    model_name=glm-5.1 \
    llm_base_url=https://api.z.ai/api/coding/paas/v4 \
    max_in_flight=1 \
    max_concurrent_dags=1 \
    llm_max_concurrent=1
```

`llm_max_concurrent` defaults to `null` (no cap). Set it to a small
integer whenever the endpoint is rate-limited and per-key parallel
calls would be rejected.

### Prompt Co-Evolution

Co-evolve the mutation prompts alongside your programs. A paired prompt run
evolves the system prompt used by the mutation LLM, selecting for prompts that
produce better mutations:

```bash
# Main run — uses co-evolved prompts from DB 6
python run.py problem.name=my_task pipeline=my_pipeline \
    storage=redis prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6 redis.db=4

# Prompt run — evolves mutation prompts, reads outcomes from DB 4
python run.py problem.name=prompt_evolution pipeline=prompt_evolution \
    storage=redis redis.db=6 main_redis_db=4 main_redis_prefix=my_task
```

See [Prompt Co-Evolution Guide](docs/COEVOLUTION.md) for the full architecture,
launch instructions, and monitoring.

## Configuration

GigaEvo uses [Hydra](https://hydra.cc/) for modular configuration. All config
files are in `config/`:

| Directory | Purpose | Key files |
|-----------|---------|-----------|
| `experiment/` | Complete experiment templates | `base.yaml`, `full_featured.yaml`, `prompt_coevolution.yaml` |
| `algorithm/` | Evolution algorithms | `single_island_no_distant_parents.yaml` (default), `single_island.yaml`, `single_island_2d.yaml`, `multi_island.yaml`, `topology_3d.yaml` |
| `llm/` | LLM setups | `single.yaml`, `heterogeneous.yaml`, `heterogeneous_bandit.yaml`, `openrouter_bandit.yaml`, `openrouter_ensemble.yaml` |
| `pipeline/` | DAG execution pipelines | `guided.yaml` (default), `memory_guided.yaml`, `custom.yaml`, `prompt_evolution.yaml` |
| `program_format/` | Candidate representation | `python_source.yaml` (default), `json_document.yaml` |
| `prompt_fetcher/` | Prompt sourcing | `fixed.yaml`, `coevolved.yaml` |
| `stopper/` | Stopping criteria | `max_mutants.yaml` (default), `wall_clock.yaml`, `fitness_plateau.yaml` |
| `constants/` | Tunable parameters | `evolution.yaml`, `llm.yaml`, `islands.yaml`, `pipeline.yaml`, `runner.yaml`, `endpoints.yaml`, `redis.yaml`, `logging.yaml` |
| `loader/` | Program loading | `directory.yaml`, `top_programs.yaml` |
| `logging/` | Backends | `tensorboard.yaml`, `wandb.yaml` |

Override any setting via command line:
```bash
python run.py experiment=full_featured max_mutants=50 temperature=0.8
```

### Environment Variables

Hydra configs cover most behavior; a handful of environment variables control
credentials, the execution sandbox, and observability. Set them in `.env` or the
shell before launching a run.

**LLM access & routing**

| Variable | Effect |
|----------|--------|
| `OPENAI_API_KEY` | Primary LLM credential — read by the default mutation router (`config/llm/single.yaml`) and most mutation LLM configs in `config/llm/`. Holds the credential expected by the selected mutation endpoint. |
| `OPENROUTER_API_KEY` | OpenRouter credential — read by `config/memory/llm/gemini.yaml`, `config/memory/llm/gpt54_mini.yaml`, and explicitly OpenRouter-keyed mutation configs. |
| `LITELLM_MASTER_KEY` | Optional credential for the local LiteLLM proxy; read by `config/memory/llm/qwen_instruct.yaml`. |
| `LOCAL_LLM_PROXY` | Shared OpenAI-compatible base URL for `llm=local_proxy`, `memory/llm=qwen_instruct`, and chain evaluators unless a task-specific URL overrides it. Include the `/v1` suffix. |

The env-var name does **not** track the provider — e.g. the OpenRouter-targeted
`openrouter_bandit`/`openrouter_ensemble` main routers read `OPENAI_API_KEY`, not
`OPENROUTER_API_KEY`. Which key a run needs is determined by the selected `llm` and
`memory/llm` configs, not by the model vendor.

Memory-LLM model and structured-output method are Hydra knobs on the
`memory/llm` group (`config/memory/llm/*.yaml`). Locally served presets take
their endpoint from `LOCAL_LLM_PROXY` so an address change is made once.

**Execution sandbox**

| Variable | Effect |
|----------|--------|
| `EVO_EXEC_THREADS` | Per-mutant thread cap applied to `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `LOKY_MAX_CPU_COUNT`. Default `max(1, cpu_count // 8)` to stop concurrent mutants oversubscribing the box. |

**Experiment harness**

| Variable | Effect |
|----------|--------|
| `GIGAEVO_PYTHON` | Interpreter used to launch generated experiment runs (validated by `gigaevo manifest gate`). |
| `GIGAEVO_PROJ` | Project-root override for experiment-manifest resolution. |
| `GIGAEVO_PROMPT_LOG_DIR` | When set, the rendered mutation prompts are dumped to this directory (empty = off). |

**Observability & networking**

| Variable | Effect |
|----------|--------|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Langfuse LLM tracing (consumed by the Langfuse SDK). |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Status notifications via `tools.telegram_notify`. |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | Outbound proxy for LLM and Telegram traffic. |

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
- **Programs**: `storage/<problem name>/programs/*.json` by default
- **Metrics history**: `metrics/*.jsonl` by default
- **Metrics**: TensorBoard / W&B (if configured)

## CLI Tools (`gigaevo`)

Installed via `pip install -e .`. Global flags: `-e/--experiment`, `-r/--run`, `-f/--format`.

| Command | Purpose |
|---------|---------|
| `gigaevo -e EXP status` | Live monitoring: iteration, metrics, PIDs, watchdog |
| `gigaevo -r RUN trajectory` | Iteration-by-iteration fitness trajectory |
| `gigaevo -r RUN top` | Inspect best programs by fitness |
| `gigaevo -e EXP plot comparison -o DIR` | Multi-run fitness curve plots |
| `gigaevo -e EXP plot arms-race -o DIR` | Dual-panel adversarial arms-race plot |
| `gigaevo -e EXP profiler` | Profile runner logs into text summary + HTML dashboard |
| `gigaevo -f json memory calibrate-safety RUN...` | Replay memory-v2 safety priors and emit Hydra overrides |
| `gigaevo -e EXP manifest gate <status>` | Hard-gate on experiment status (preregistered/implemented/running/complete) |
| `gigaevo -r RUN export csv -o FILE` | Export evolution data to CSV |
| `gigaevo flush --db N --confirm` | Safely flush Redis DBs (kills workers first) |
| `gigaevo -e EXP launch` / `watchdog` | Launch + supervise an experiment |
| `tools/experiment/archive_run.sh` | Archive run data before flush |
| `tools/dag_builder/` | Visual DAG pipeline designer |
| `tools/wizard/` | Interactive problem scaffolding |

For disk runs, pass the storage directory as the run spec, e.g.
`gigaevo -r 'outputs/<date>/<time>/storage' top -n 5`. See
[tools/README.md](tools/README.md) for the full CLI reference and Redis key
schema.

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

**Redis database not empty** (only when using `storage=redis`):
```bash
# Flush (kills exec_runner workers first):
gigaevo flush --db 0 --confirm

# Or use a different DB:
python run.py storage=redis redis.db=1
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
