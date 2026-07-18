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

# Bound the post-cap drain grace (seconds). Once the mutant cap is reached the
# engine drains in-flight evals up to this budget, then abandons any stragglers
# for teardown to kill and finalizes cleanly (default 30). Use null to restore
# the legacy "drain up to dag_timeout, then raise" contract.
python run.py problem.name=toy_example post_cap_drain_grace_s=60

# Use Redis-backed storage on a specific DB
python run.py problem.name=toy_example storage=redis redis.db=5

# Disable the LLM I/O audit trail (on by default)
python run.py problem.name=toy_example llm_io_dump=false
```

## LLM Setup

The main mutation LLM and the memory subsystem LLM are configured separately.

| Route | Config | Typical key |
|---|---|---|
| Main mutation | `llm=...`, `llm_base_url=...`, `model_name=...` | `OPENAI_API_KEY` for `llm=single` or `llm=local_proxy` |
| Memory research/writer | `memory/llm=...` | `OPENROUTER_API_KEY` for hosted presets; `LITELLM_MASTER_KEY` for local `qwen_instruct` |

Hosted/OpenRouter style:

```bash
export OPENAI_API_KEY=sk-or-v1-...
python run.py problem.name=toy_example \
    llm=single \
    llm_base_url=https://openrouter.ai/api/v1 \
    model_name=google/gemini-3-flash-preview
```

Local LiteLLM/vLLM-compatible proxy (`LOCAL_LLM_PROXY` is defined in `.env`):

```bash
export NO_PROXY=127.0.0.1,localhost

python run.py problem.name=toy_example \
    llm=local_proxy \
    model_name=Qwen/Qwen3-235B-A22B-Thinking-2507
```

Memory with a separate instruct model:

```bash
python run.py problem.name=toy_example \
    memory/llm=qwen_instruct \
    checkpoint_dir=$PWD/SHARE_TOY_MEMORY
```

The base experiment uses `memory=v2`, `pipeline=memory_guided_noise`,
`num_parents=1`, and 70% delivery / 30% matched control. Use
`pipeline=guided memory=none` for an explicit no-memory run.

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

# JSON-document genomes, such as CARL chain specs
python run.py problem.name=chains/hover/full7 program_format=json_document

# Co-evolved mutation prompts
python run.py problem.name=toy_example \
    storage=redis prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6
```

### Available Config Groups

| Group | Options |
|-------|---------|
| `experiment` | `base`, `full_featured`, `prompt_coevolution` |
| `algorithm` | `single_island_no_distant_parents` (default), `single_island`, `single_island_2d`, `multi_island`, `topology_3d` (+ `_ret` variant), `chains_bd3d` (chain-strategy 3D behavior space: hop_depth × passages_fetched × instr_chars; requires `enable_chain_structural_metrics=true` + `program_format=json_document` via `algorithm_requires`) |
| `llm` | `single`, `heterogeneous`, `heterogeneous_bandit`, `balanced`, `openrouter_bandit`, `openrouter_ensemble`, `google`, `openai`, `gemini3_flash`, `gemini35_flash` |
| `pipeline` | `guided` (default), `memory_guided` (see [MEMORY_GUIDED_PIPELINE.md](MEMORY_GUIDED_PIPELINE.md)), `memory_guided_noise` (memory_guided + per-sample score transport for the paired archive gate), `guided_noise` (same score transport on the plain guided DAG — the no-memory arm), `custom`, `structural_metrics`, `adversarial`, `adversarial_asymmetric`, `adversarial_coevo`, `prompt_evolution`, `optuna_opt` |
| `archive_selector` | `point` (default — replace elite iff weighted fitness sum is higher), `paired_bootstrap` (noise-aware paired bootstrap gate, knob: `archive_selector.p_accept`; needs a per-sample-score problem + `pipeline=memory_guided_noise` or `guided_noise`) |
| `program_format` | `python_source` (default), `json_document` |
| `prompt_fetcher` | `fixed` (default), `coevolved` |
| `stopper` | `max_mutants` (default), `wall_clock`, `fitness_plateau`, `max_mutants_or_fitness_plateau` |
| `constants` | `base`, `evolution`, `llm`, `islands`, `pipeline`, `redis`, `logging`, `runner`, `endpoints` |
| `loader` | `directory`, `top_programs` (knobs: `loader.source_db` — Redis DB to read seeds from, default 0; `loader.top_n` — number of top programs to seed, default 50) |
| `logging` | `tensorboard`, `wandb` |
| `storage` | `disk` (default), `redis` |

### Disk Storage Backend

Programs and archives are persisted to JSON files by default:

```bash
python run.py problem.name=toy_example
```

Data lands under `<hydra run dir>/storage/<problem name>/` by default
(per-run, like `checkpoint_dir`). Override the root on the CLI
(`program_storage.config.root_dir=/abs/path`) to persist/resume across
runs. Single-process only — the instance lock is a PID file, and there is
no cross-process pub/sub. Metrics history also goes to disk (JSONL files
under `<hydra run dir>/metrics/`), and live monitors
(`live_frontier_compare`) read it through the same backend — `storage=disk`
runs are fully Redis-free.

Use Redis explicitly when you want Redis-backed program/archive storage:

```bash
python run.py problem.name=toy_example storage=redis redis.db=5
```

Inspect a disk run with the CLI by passing the storage path as the run
spec (read-only, safe while the run is live):

```bash
gigaevo -r 'outputs/<run dir>' top -n 5
gigaevo -r 'outputs/<run dir>:mylabel' export csv -o out.csv
gigaevo -r 'outputs/<run dir>' trajectory --tail 20
gigaevo -r 'outputs/<run dir>' metrics --tag 'program_metrics/*'
```

Supported by `top`, `export`, `plot`, `trajectory`, and `metrics`. `status`
and `checkpoint` remain Redis-only because they require live process state.
See [the CLI reference](../gigaevo/cli/README.md) for the full run-spec and
backend matrix.

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

### Memory-Guided Mutation
```bash
# See docs/MEMORY_GUIDED_PIPELINE.md for the full mode guide
python run.py problem.name=heilbron \
    pipeline=memory_guided memory=full memory/write=live \
    num_parents=4 max_mutants=500
```

> `pipeline=memory_guided` reads cards; `memory/write=live` enables live
> writer sweeps. A true no-memory baseline is `pipeline=guided memory=none`.
> The default memory LLM (`memory/llm=gemini`) calls OpenRouter, so
> `OPENROUTER_API_KEY` must be exported.

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
