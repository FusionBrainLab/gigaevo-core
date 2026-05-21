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

# Two-island evolution (fitness + simplicity tradeoff)
python run.py experiment=multi_island_complexity problem.name=toy_example

# Multiple LLMs for diverse mutations
python run.py experiment=multi_llm_exploration problem.name=toy_example

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
```

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
| `experiment` | `base`, `full_featured`, `heilbron`, `migration_bus`, `multi_island_complexity`, `multi_llm_exploration`, `prompt_coevolution`, `steady_state`, `steady_state_adversarial`, `steady_state_bus` |
| `algorithm` | `single_island`, `single_island_2d` (+ `_d`, `_g` variants), `multi_island`, `topology_3d` (+ `_ret`, `_7step` variants), `single_island_fitness_prop_fixed_temp`, `single_island_weighted` |
| `llm` | `single`, `heterogeneous`, `heterogeneous_bandit`, `balanced`, `openrouter_bandit`, `openrouter_ensemble`, `google`, `openai`, `gemini25_pro`, `gemini31_pro`, `gemini3_flash` |
| `pipeline` | `auto` (default), `standard`, `with_context`, `custom`, `algotune_speed`, `structural_metrics`, `adversarial`, `adversarial_asymmetric`, `adversarial_coevo`, `hotpotqa_asi`, `hotpotqa_colbert`, `hotpotqa_reflective`, `hover_feedback`, `intra_extra_memory` (see [INTRA_EXTRA_MEMORY.md](INTRA_EXTRA_MEMORY.md)), `prompt_evolution`, `optuna_opt`, `cma_opt` |
| `prompt_fetcher` | `fixed` (default), `coevolved` |
| `stopper` | `max_mutants` (default), `wall_clock`, `fitness_plateau`, `max_mutants_or_fitness_plateau` |
| `constants` | `base`, `evolution`, `llm`, `islands`, `pipeline`, `redis`, `logging`, `runner`, `endpoints` |
| `loader` | `directory`, `redis_selection` |
| `logging` | `tensorboard`, `wandb` |

## Examples

### Quick Test Run
```bash
python run.py problem.name=toy_example max_mutants=5
```

### Production Run with Multi-Island
```bash
python run.py experiment=multi_island_complexity \
    problem.name=heilbron \
    max_mutants=100
```

### Multi-LLM Exploration
```bash
python run.py experiment=multi_llm_exploration \
    problem.name=heilbron \
    max_in_flight=12
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
    pipeline=intra_extra_memory ideas_tracker=default memory=local \
    num_parents=4 max_mutants=500
```

> **Required overrides:** `pipeline=intra_extra_memory` silently falls
> back to `NullMemoryProvider` + `NullPostRunHook` unless launched with
> **both** `ideas_tracker=default memory=local`. The extra-memory (GAM)
> agents also call OpenRouter directly, so `OPENROUTER_API_KEY` must be
> exported — without it every GAM call 401s and the extra channel ships
> zero cards silently. Verify the resolved config does not contain
> `Null*` targets before trusting results.

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
