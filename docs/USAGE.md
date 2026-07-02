# Usage Guide

## Basic Usage

```bash
# Default configuration
python run.py problem.name=toy_example

# Override individual components
python run.py problem.name=toy_example llm=heterogeneous
python run.py problem.name=toy_example algorithm=multi_island
python run.py problem.name=toy_example constants=base
```

## Using Experiments

Experiments are preset configurations in `config/experiment/`. Use the
`experiment=` override to select one:

```bash
# Simple single-island evolution (default)
python run.py experiment=base problem.name=toy_example

# Everything enabled (multi-island + multi-LLM + complexity)
python run.py experiment=full_featured problem.name=toy_example
```

Experiments are starting points — override any setting after selecting one:

```bash
python run.py experiment=full_featured problem.name=toy_example \
    max_mutants=50 stage_timeout=300
```

## Common Overrides

```bash
# Cap total mutants (default stopper is max_mutants)
python run.py problem.name=toy_example max_mutants=50

# Switch stopper (e.g. wall-clock or fitness-plateau)
python run.py problem.name=toy_example stopper=wall_clock

# Change population size
python run.py problem.name=toy_example island_max_size=150

# Change LLM settings
python run.py problem.name=toy_example \
    temperature=0.7 \
    max_tokens=40960

# More parallelism
python run.py problem.name=toy_example \
    dag_concurrency=32 \
    max_concurrent_dags=20 \
    max_in_flight=12

# Use a different Redis database
python run.py problem.name=toy_example redis.db=5

# Disable the LLM I/O audit trail (on by default)
python run.py problem.name=toy_example llm_io_dump=false
```

### LLM I/O Audit Trail

Every LLM router call (mutation, memory, suggester — both plain and structured
output) is logged as a complete JSON record into `<run_dir>/llm_io/<router>.jsonl`.
Each line holds the exact prompt messages sent, the response text, the model
name, token usage, and any error — a durable record of what went into the LLM
and what came back. Enabled by default; set `llm_io_dump=false` to turn it off.

## Configuration Groups

Override individual config groups:

```bash
# Use different LLM config
python run.py problem.name=toy_example llm=heterogeneous

# Use different algorithm
python run.py problem.name=toy_example algorithm=multi_island

# Use custom pipeline
python run.py problem.name=toy_example pipeline=custom

# Co-evolved mutation prompts
python run.py problem.name=toy_example \
    prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6
```

### Available Config Groups

| Group | Options |
|-------|---------|
| `experiment` | `base`, `full_featured`, `prompt_coevolution` |
| `algorithm` | `single_island_no_distant_parents` (default), `single_island`, `single_island_2d`, `multi_island`, `topology_3d` (+ `_ret` variant) |
| `llm` | `single`, `heterogeneous`, `heterogeneous_bandit`, `balanced`, `openrouter_bandit`, `openrouter_ensemble`, `google`, `openai`, `gemini3_flash`, `gemini35_flash` |
| `pipeline` | `auto` (default), `standard`, `with_context`, `custom`, `structural_metrics`, `adversarial`, `adversarial_asymmetric`, `adversarial_coevo`, `intra_extra_memory` (see [INTRA_EXTRA_MEMORY.md](INTRA_EXTRA_MEMORY.md)), `prompt_evolution`, `optuna_opt` |
| `prompt_fetcher` | `fixed` (default), `coevolved` |
| `stopper` | `max_mutants` (default), `wall_clock`, `fitness_plateau`, `max_mutants_or_fitness_plateau` |
| `constants` | `base`, `evolution`, `llm`, `islands`, `pipeline`, `redis`, `logging`, `runner`, `endpoints` |
| `loader` | `directory`, `top_programs` (knobs: `loader.source_db` — Redis DB to read seeds from, default 0; `loader.top_n` — number of top programs to seed, default 50) |
| `logging` | `tensorboard`, `wandb` |
| `storage` | `redis` (default), `disk` |

### Disk Storage Backend

Programs and archives can be persisted to JSON files instead of Redis:

```bash
python run.py problem.name=toy_example storage=disk
```

Data lands under `<hydra run dir>/storage/<problem name>/` by default
(per-run, like `checkpoint_dir`). Override the root on the CLI
(`program_storage.config.root_dir=/abs/path`) to persist/resume across
runs. Single-process only — the instance lock is a PID file, and there is
no cross-process pub/sub. Metrics history also goes to disk (JSONL files
under `<hydra run dir>/metrics/`), and live monitors
(`live_frontier_compare`) read it through the same backend — `storage=disk`
runs are fully Redis-free.

Inspect a disk run with the CLI by passing the storage path as the run
spec (read-only, safe while the run is live):

```bash
gigaevo -r outputs/<run dir>/storage top -n 5
gigaevo -r outputs/<run dir>/storage:mylabel export csv -o out.csv
```

Supported by `top`, `export`, and `plot`; Redis-only commands (`status`,
`trajectory`, `metrics`, `checkpoint`) reject disk specs — see
[tools/README.md](../tools/README.md) for the full run-spec reference.

## Examples

### Quick Test Run
```bash
python run.py problem.name=toy_example max_mutants=5
```

### Production Run with Full-Featured Experiment
```bash
python run.py experiment=full_featured \
    problem.name=heilbron \
    max_mutants=100
```

### Prompt Co-Evolution
```bash
# See docs/COEVOLUTION.md for full details
python run.py problem.name=my_task pipeline=my_pipeline \
    prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6 redis.db=4
```

### Intra/Extra Memory (per-parent lineage card + live global ideas)
```bash
# See docs/INTRA_EXTRA_MEMORY.md for the full mode guide
python run.py problem.name=heilbron \
    pipeline=intra_extra_memory memory=full \
    num_parents=4 max_mutants=500
```

> **Required override:** launch with `memory=full` — the single preset that
> turns *both* the reader (injects cards) and the writer (`IdeaTracker`
> extracts + enriches them) on. Under `pipeline=intra_extra_memory` the
> writer-off presets (`memory=none`, `memory=reader`) **fail fast at
> startup**: the live-refresh hook needs a real tracker. A true no-memory
> baseline is `pipeline=standard memory=none`. The extra-memory (GAM) agents
> call OpenRouter directly, so `OPENROUTER_API_KEY` must be exported —
> without it every GAM call 401s and the extra channel ships zero cards
> silently. Verify the arm from the startup `[Memory][Arm]` banner before
> trusting results.

### Tabular Suite (regression + classification, 10 datasets)

See `problems/tabular/README.md`. Requires `$GIGAEVO_TABULAR_DATA` pointing at the tabm data root.

```bash
GIGAEVO_TABULAR_DATA=/path/to/data \
python run.py problem.name=tabular/california algorithm=tabular/2d_local_ood
```

## Viewing Configuration

```bash
# See the full resolved configuration (without running)
python run.py problem.name=toy_example --cfg job

# See resolved config for an experiment preset
python run.py experiment=full_featured problem.name=toy_example --cfg job
```

## Specific OpenAI API Parameters

Additional OpenAI API parameters can be specified by editing the `models` config
section in configuration files under `config/llm`. Parameters should be named
exactly as in the OpenAI API specification and placed under either `model_kwargs`
or `extra_body`.

### `model_kwargs` vs `extra_body`

**`model_kwargs`** — standard OpenAI API parameters merged into the top-level
request payload:

```yaml
model_kwargs:
  stream_options:
    include_usage: true
  max_completion_tokens: 300
```

**`extra_body`** — custom parameters for OpenAI-compatible providers (vLLM,
OpenRouter, etc.) nested under `extra_body` in the request:

```yaml
extra_body:
  provider:                       # OpenRouter-specific
    order: [google-vertex]
    allow_fallbacks: false
  top_k: 40                      # Provider-specific (Gemini, Claude)
  use_beam_search: true           # vLLM-specific
  reasoning:                      # OpenRouter-specific
    effort: high
    max_tokens: 5000
```

> **Warning:** Always use `extra_body` for non-standard parameters, **not**
> `model_kwargs`. Using `model_kwargs` for non-OpenAI parameters will cause API
> errors.

See [OpenAI API docs](https://platform.openai.com/docs/api-reference) and
[ChatOpenAI docs](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/)
for parameter references.

## Tips

1. **Start simple** — begin with the default config, add overrides as needed
2. **Experiments are starting points** — override anything after selecting one
3. **Check resolved config** — `--cfg job` shows exactly what will run
4. **Hydra saves config** — full resolved config is saved to `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`
5. **Use `experiment=` for presets** — don't need `--config-name`

## Troubleshooting

**Want to see available experiments?**
```bash
ls config/experiment/
```

**Want to see what an experiment does?**
```bash
cat config/experiment/base.yaml
```

**Want default config with one change?**
```bash
# Just override directly, no experiment needed
python run.py problem.name=toy_example llm=heterogeneous
```
