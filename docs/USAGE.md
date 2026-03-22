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
    max_generations=50 stage_timeout=300
```

## Common Overrides

```bash
# Limit generations
python run.py problem.name=toy_example max_generations=50

# Change population size
python run.py problem.name=toy_example island_max_size=150

# Change LLM settings
python run.py problem.name=toy_example \
    default_temperature=0.7 \
    default_max_tokens=40960

# More parallelism
python run.py problem.name=toy_example \
    dag_concurrency=32 \
    max_concurrent_dags=20

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
| `experiment` | `base`, `full_featured`, `multi_island_complexity`, `multi_llm_exploration` |
| `algorithm` | `single_island`, `single_island_2d`, `multi_island`, `single_island_fitness_prop_fixed_temp`, `single_island_weighted` |
| `llm` | `single`, `heterogeneous`, `heterogeneous_bandit`, `openrouter_bandit`, `openrouter_ensemble`, `google`, `openai`, `gemini25_pro`, `gemini31_pro`, `gemini3_flash` |
| `pipeline` | `standard`, `with_context`, `auto`, `custom`, `hotpotqa_asi`, `hotpotqa_colbert`, `hotpotqa_reflective`, `hover_feedback`, `prompt_evolution`, `prompt_evolution_multi`, `mcts_evo`, `optuna_opt`, `cma_opt` |
| `prompt_fetcher` | `fixed` (default), `coevolved` |
| `constants` | `base`, `evolution`, `llm`, `islands`, `pipeline`, `redis`, `logging`, `runner`, `endpoints` |
| `loader` | `directory`, `redis_selection` |
| `logging` | `tensorboard`, `wandb` |

## Examples

### Quick Test Run
```bash
python run.py problem.name=toy_example max_generations=5
```

### Production Run with Multi-Island
```bash
python run.py experiment=multi_island_complexity \
    problem.name=heilbron \
    max_generations=100
```

### Multi-LLM Exploration
```bash
python run.py experiment=multi_llm_exploration \
    problem.name=heilbron \
    max_mutations_per_generation=12
```

### Prompt Co-Evolution
```bash
# See docs/COEVOLUTION.md for full details
python run.py problem.name=my_task pipeline=my_pipeline \
    prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6 redis.db=4
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
