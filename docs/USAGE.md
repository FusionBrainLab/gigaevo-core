# Quick Usage Guide

## Basic Usage

```bash
# Default configuration
python run_hydra.py problem.name=toy_example

# Override individual components
python run_hydra.py problem.name=toy_example llm=multi_model
python run_hydra.py problem.name=toy_example map_elites=multi_island auxiliary_metrics=code_complexity
python run_hydra.py problem.name=toy_example constants=base
```

## Using Profiles

Profiles are preset configurations located in `config/profiles/`. To use a profile:

```bash
# Use --config-name to select a profile as the base config
python run_hydra.py --config-name=profiles/local_dev problem.name=toy_example
```

### Available Profiles

**`profiles/base`** - Fast local iteration
```bash
python run_hydra.py --config-name=profiles/base problem.name=toy_example
```

**`profiles/multi_island_complexity`** - Two-island evolution
```bash
python run_hydra.py --config-name=profiles/multi_island_complexity problem.name=toy_example
```
- Fitness island + Simplicity island
- Code complexity metrics enabled
- Explores fitness-simplicity tradeoffs

**`profiles/multi_llm_exploration`** - Multi-LLM diversity
```bash
python run_hydra.py --config-name=profiles/multi_llm_exploration problem.name=toy_example
```
- 70% creative mutations (temp 0.8)
- 30% precise mutations (temp 0.4)
- Maximizes solution diversity

**`profiles/full_featured`** - Everything enabled
```bash
python run_hydra.py --config-name=profiles/full_featured problem.name=toy_example
```
- Multi-island + Multi-LLM + Complexity
- Production defaults
- Full feature set

## Combining Profiles with Overrides

```bash
# Start with a profile, then override specific settings
python run_hydra.py --config-name=profiles/local_dev problem.name=toy_example max_generations=20

# Use local_dev profile but with multi-LLM
python run_hydra.py --config-name=profiles/local_dev problem.name=toy_example llm=multi_model

# Full featured but with local_dev timeouts
python run_hydra.py --config-name=profiles/full_featured problem.name=toy_example \
    stage_timeout=300 dag_timeout=900
```

## Common Overrides

```bash
# Change population size
python run_hydra.py problem.name=toy_example island_max_size=150

# Limit generations
python run_hydra.py problem.name=toy_example max_generations=50

# Change LLM settings
python run_hydra.py problem.name=toy_example \
    default_temperature=0.7 \
    default_max_tokens=40960

# More parallelism
python run_hydra.py problem.name=toy_example \
    dag_concurrency=32 \
    max_concurrent_dags=20
```

## Configuration Groups

Override individual groups without using profiles:

```bash
# Use different constants
python run_hydra.py problem.name=toy_example constants=local_dev

# Use different LLM config
python run_hydra.py problem.name=toy_example llm=multi_model

# Use different MAP-Elites strategy
python run_hydra.py problem.name=toy_example map_elites=multi_island

# Enable auxiliary metrics
python run_hydra.py problem.name=toy_example auxiliary_metrics=code_complexity

# Use custom pipeline
python run_hydra.py problem.name=toy_example pipeline=custom
```

## Examples

### Quick Test Run
```bash
# Fast iteration for testing
python run_hydra.py --config-name=profiles/local_dev problem.name=toy_example
```

### Production Run with Multi-Island
```bash
# Full-featured evolution
python run_hydra.py --config-name=profiles/multi_island_complexity \
    problem.name=optimization \
    max_generations=null  # Unlimited
```

### Multi-LLM Exploration
```bash
# Diverse mutations for exploration
python run_hydra.py --config-name=profiles/multi_llm_exploration \
    problem.name=hexagon_pack \
    max_mutations_per_generation=12
```

### Custom Combination
```bash
# Mix and match
python run_hydra.py problem.name=heilbron \
    constants=base \
    llm=multi_model \
    map_elites=multi_island \
    auxiliary_metrics=code_complexity \
    max_generations=100
```

## Viewing Configuration

```bash
# See the full resolved configuration
python run_hydra.py --config-name=profiles/local_dev problem.name=toy_example --cfg job

# See only the config structure (no running)
python run_hydra.py problem.name=toy_example --cfg job | head -50
```

## Specific OpenAI API Parameters

Additional OpenAI API parameters can be specified by editing the `models` config section in configuration files under `config/llm`.
Parameters should be named exactly as in the OpenAI API specification and placed under either the `model_kwargs` or `extra_body` section.
Content from both of these sections will be placed at the top level of the OpenAI API request body.

### `model_kwargs` vs `extra_body` Parameters

Use the correct section for different types of API arguments:

**Use `model_kwargs` for:**

- Standard OpenAI API parameters not explicitly defined as class parameters
- Parameters that should be flattened into the top-level request payload
- Examples: `max_completion_tokens`, `stream_options`, `modalities`, `audio`

**Use `extra_body` for:**

- Custom parameters specific to OpenAI-compatible providers (vLLM, LM Studio, OpenRouter, etc.)
- Parameters that need to be nested under `extra_body` in the request
- Any non-standard OpenAI API parameters

**Key Differences:**

- `model_kwargs`: Parameters are **merged into top-level** request payload
- `extra_body`: Parameters are **nested under `extra_body`** key in request

>**Warning:**
>   Always use `extra_body` for custom parameters, **not** `model_kwargs`.
>   Using `model_kwargs` for non-OpenAI parameters will cause API errors.

More information can be found in the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/introduction) and
[ChatOpenAI documentation](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/?h=chatopenai#langchain_openai.chat_models.ChatOpenAI).

### Examples

#### Specifying `model_kwargs` section

```yaml
# Standard OpenAI parameters
llm:
  _target_: gigaevo.llm.models.MultiModelRouter
  _convert_: all
  models:
    - _target_: langchain_openai.ChatOpenAI
      model: "..."
      api_key: ${oc.env:OPENAI_API_KEY}
      temperature: ${temperature}
      max_tokens: ${max_tokens}
      top_p: ${top_p}
      base_url: ${llm_base_url}
      model_kwargs:
        # These are standard OpenAI API parameters or accepted sub-objects:
        stream_options:                 # Used by OpenAI API for streaming
          include_usage: true
        max_completion_tokens: 300      # Supported OpenAI parameter for completion length
        modalities: [text, audio]       # Example OpenAI API-supported parameter (multi-modal models)
        audio:                          # Audio config goes here, accepted by OpenAI's API for its voice models
          voice: alloy
          format: wav
  probabilities: [0.5, 0.5]
```

#### Specifying `extra_body` section

```yaml
llm:
  _target_: gigaevo.llm.models.MultiModelRouter
  _convert_: all
  models:
    - _target_: langchain_openai.ChatOpenAI
      model: ${model_name}
      api_key: ${oc.env:OPENAI_API_KEY}
      temperature: ${temperature}
      max_tokens: ${max_tokens}
      top_p: ${top_p}
      base_url: ${llm_base_url}
      extra_body:
        # These are non-OpenAI parameters:
        provider:                       # OpenRouter-specific (not OpenAI standard)
          order: [google-vertex]
          allow_fallbacks: false
          data_collection: deny
        top_k: ${top_k}                 # Provider-specific (e.g., Google Gemini, Anthropic Claude; not OpenAI standard)
        use_beam_search: true           # vLLM-specific parameter
        best_of: 4                      # vLLM-specific parameter
        ttl: 300                        # LM Studio-specific parameter
        reasoning:                      # OpenRouter-specific (not OpenAI standard)
          effort: high
          max_tokens: 5000
  probabilities: [0.5, 0.5]
```


## Tips

1. **Start simple**: Begin with default or `base` profile
2. **Profiles are starting points**: Override anything after selecting a profile
3. **Check outputs**: Hydra saves full config to `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`
4. **Use `--config-name` for profiles**: This is the correct way to select profile configs
5. **Regular overrides don't need `--config-name`**: Just `llm=multi_model` is enough

## Troubleshooting

**Profile not found?**
```bash
# List available profiles
ls config/profiles/
```

**Want default config with one change?**
```bash
# Don't use --config-name, just override
python run_hydra.py problem.name=toy_example llm=multi_model
```

**Want to see what a profile does?**
```bash
# Check the profile file
cat config/profiles/base.yaml
```
