# Usage Guide

## Basic Usage

```bash
# Run a shipped experiment preset
python run.py experiments/base.py

# Validate and dump the resolved config without invoking the engine
python run.py experiments/base.py --dry-run
```

`run.py` is a thin entry point that loads the experiment file, applies any
CLI overrides through tyro, dumps the resolved config to
`outputs/{experiment_id}/config.json`, and runs the engine.

## Shipped Experiments

Each shipped preset lives at `experiments/<name>.py` and exports a `build()`
returning an `ExperimentConfig`:

```bash
# Single-island MAP-Elites, Heilbron triangle problem (default)
python run.py experiments/base.py

# Steady-state engine (continuous mutation/evaluation, ~8x throughput)
python run.py experiments/steady_state.py

# Multi-island evolution (fitness + simplicity tradeoff)
python run.py experiments/multi_island_complexity.py

# Multiple LLMs for diverse mutations
python run.py experiments/multi_llm_exploration.py

# Cross-run program sharing through a Redis stream
python run.py experiments/migration_bus.py

# Steady-state engine plus cross-run sharing
python run.py experiments/steady_state_bus.py

# Co-evolve mutation prompts alongside programs
python run.py experiments/prompt_coevolution.py

# Multi-stage pipeline with prompt-fetcher + structural metrics
python run.py experiments/full_featured.py
```

## CLI Overrides

Overrides are parsed by tyro and dotted by `ExperimentConfig` field path:

```bash
# Limit generations
python run.py experiments/base.py --engine.max_generations 50

# Change Redis database
python run.py experiments/base.py --redis.db 5

# Tune LLM kwargs (single-router experiments)
python run.py experiments/base.py \
    --llm.models.0.temperature 0.7 \
    --llm.models.0.max_tokens 4096

# Stack overrides
python run.py experiments/full_featured.py \
    --engine.max_generations 50 \
    --pipeline.builder.stage_timeout 300
```

Every override is re-validated against the Pydantic schema, including any
cross-field invariants declared on `ExperimentConfig`. Invalid overrides
fail fast with the validator's error.

## Writing a Custom Experiment

For anything beyond a couple of overrides, copy `experiments/base.py` to a
new file under `experiments/` and edit its `build()`:

```python
# experiments/my_run.py
from pathlib import Path

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_generational
from gigaevo.config.llm_presets import build_openrouter_ensemble
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    RedisConfig,
)


def build() -> ExperimentConfig:
    redis = RedisConfig(db=2)
    return ExperimentConfig(
        name="my_run",
        seed=42,
        output_dir=Path("outputs"),
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix="gigaevo:my_run"),
        problem=build_heilbron(),
        algorithm=build_single_island(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_openrouter_ensemble(),
        runner=build_default_runner(),
    )
```

Then:

```bash
python run.py experiments/my_run.py --dry-run    # validate first
python run.py experiments/my_run.py              # run
```

## Resolved Config Output

Every invocation writes the post-validation `ExperimentConfig` to
`outputs/{experiment_id}/config.json`. The `experiment_id` is a content
hash of the resolved configuration, so two runs with the same inputs share
an output directory (idempotent) and any change in overrides produces a
new directory.

To preview without running, add `--dry-run`.

## Customizing LLM Endpoints

LLM endpoints are described by Pydantic schemas under
`gigaevo/config/schemas/llm.py`. Each `ChatOpenAIConfig` exposes the
standard knobs (`model`, `api_key`, `base_url`, `temperature`,
`max_tokens`, `request_timeout`) and can be assembled into routers
(`EnsembleRouterConfig`, `BanditRouterConfig`) via the preset functions in
`gigaevo/config/llm_presets.py`.

For provider-specific parameters not exposed on the schema, edit the
preset (or your custom experiment file) and pass them through whichever
`ChatOpenAI` field accepts them.
