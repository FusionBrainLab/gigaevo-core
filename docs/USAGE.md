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

The base experiment uses `memory=v2`, `pipeline=memory_guided`,
`num_parents=1`, and 70% delivery / 30% matched control. Use
`pipeline=guided memory=none` for an explicit no-memory run.

### Agentic Coding Harness Backend

`llm=harness` replaces the HTTP chat endpoint with a coding CLI — `claude -p`,
`codex exec`, or anything else you can name on the command line. It substitutes
for the main LLM router, not just mutation: the mutation agent, the suggester
and the structured-diff operator all route through it, because it is a
`BaseChatModel` sitting in `MultiModelRouter.models` exactly where `ChatOpenAI`
normally sits.

```bash
python run.py problem.name=toy_example llm=harness

# Codex CLI, pre-wired (schema file flag + answer file + JSONL usage)
python run.py problem.name=toy_example llm=codex
```

`config/llm/harness.yaml` is `claude -p`; `config/llm/codex.yaml` is
`codex exec` on `gpt-5.6-luna`. Any other CLI is a `command` override plus
whichever of the knobs below its flags support.

**The memory LLM is separate.** The memory config groups bind their agents to
their own router (`memory.llm`), which is an HTTP model and stays one — so
`llm=harness` with `memory=v2` or `memory=full` still needs `OPENROUTER_API_KEY`
and reachable network for the insight and card agents. Add `memory=none` for a
run that uses no HTTP endpoint at all.

**The workspace contract.** There is no per-harness adapter code. Every call
gets a fresh directory, which is also the harness's working directory:

```
<workspace_root>/gigaevo-harness-<random>/<call_id>/
  SYSTEM.md     all SystemMessage content, joined
  USER.md       the remaining messages, role-tagged
  SCHEMA.json   the JSON Schema the answer must satisfy
  OUTPUT.json   written by the harness — the answer, unless `schema_flag` is set
  ANSWER.json   written by the CLI itself, under `answer_file_flag`
  STDOUT.log    the harness's stdout — token counts, and under `schema_flag`
                with `answer_key` the answer
  STDERR.log    the harness's stderr, kept for debugging a failed call
```

Workspaces are kept, not cleaned up: they are the audit trail for what a
harness was actually asked. A long run accumulates one small directory per LLM
call, so point `workspace_root` at a disk you are willing to fill.

The instruction telling the harness to do this is written to its stdin, and
lives in `gigaevo/prompts/harness/instruction.txt` — override it the same way as
any other prompt, with `prompts.dir`. Unstructured calls get a schema too (a
single `text` field), so the harness obeys one rule regardless of the caller.

**Skip the handshake if the harness has structured output of its own.** Writing
`OUTPUT.json` costs turns: the harness drafts the answer, then spends further
turns writing and checking a file. Set `schema_flag` to the option that takes a
JSON Schema on the command line and `answer_key` to where the answer comes back
in the harness's stdout envelope — for `claude -p` that is `--json-schema` and
`structured_output` — and the schema goes on argv instead, the answer is read
from the envelope, and no `OUTPUT.json` is involved. `instruction_native.txt`
replaces `instruction.txt` as the stdin instruction. Fleet-measured on heilbron
(~100 calls per arm, warm cache, matched pairs): mutation drops from ~8 turns
to 4 and suggestion from ~7 to 5, for 67% less input and 32% less cost per
call. Deny the
write tools when you do this — nothing reads the workspace back, so a harness
that writes there has burned a turn on a file no one opens. Set both fields or
neither; the file handshake is the default because it is the only contract every
harness can honour.

**Inline the prompts once the handshake is gone.** The remaining turns are the
harness reading `SYSTEM.md` and `USER.md`. Set `system_flag` to the option that
takes a system prompt on the command line — `--append-system-prompt` for
`claude -p` — and the backend sends the conversation directly: the system text
rides that flag, the user text goes to stdin verbatim, and no instruction is
sent at all, so the harness answers the prompt instead of reading files about
it. A call with no system text omits the flag rather than passing it empty.
The workspace files are still written — they stay the audit record — but
nothing directs the harness at them. This requires `schema_flag` and
`answer_key`: with the stdin instruction gone, the stdout envelope is the only
answer channel left. Mind the command-line limit — a system prompt is one argv
argument, and a kernel-refused exec surfaces as the usual `cannot start`
infrastructure failure. That argument is also visible to every process on the
box while the call runs (`ps`, `/proc/<pid>/cmdline`), unlike the `0700`
workspace files: GigaEvo prompts are not secrets, but do not inline one that
is.

**The codex shape: schema by file, answer by file.** `codex exec` has native
structured output too, but its flags speak files, not text. `--output-schema`
takes the path of a schema file, so set `schema_as_path: true` and the schema
flag carries the workspace `SCHEMA.json` path instead of the schema itself.
There is no answer key in its stdout either: `--output-last-message` names a
file the CLI itself writes the final schema-conformant message into — set
`answer_file_flag` to it and the answer is read from `ANSWER.json`,
mechanically written, no drafting turns spent. `answer_file_flag` and
`answer_key` are the same slot in the contract — set one, never both. Stdout
is then free to be codex's `--json` JSONL event stream, which the backend
mines for per-turn usage: OpenAI semantics, where `input_tokens` already
includes `cached_input_tokens`, and the number of usage-bearing events is
reported as `num_turns`. No `total_cost_usd` exists on this stream — compute
cost offline from the token counts and the model's prices. Two more wrinkles.
codex hands the schema to OpenAI *strict* structured output, which rejects
any schema whose objects are not closed and fully required — set
`strict_schema: true` and the backend rewrites the wire schema (optionals
become nullable) and strips the invited nulls from the answer, so pydantic
defaults apply as usual. And `codex exec` has no `--append-system-prompt`
equivalent while its sandboxed file reads flake — the first shell command
intermittently fails, after which the model returns a schema-valid *fallback*
answer rather than an error — so set `stdin_prompts: true`: the whole prompt
(system, then user) travels on stdin verbatim and no tool runs at all.
`config/llm/codex.yaml` wires all of this.

A harness that exits non-zero, fails its answer channel (a missing or invalid
`OUTPUT.json` or `ANSWER.json`; under `answer_key`, a bad stdout envelope), or
omits a required field raises `ValueError` — the same failure the HTTP path
raises, so `max_consecutive_mutation_failures` and the retry logic apply
unchanged. Infrastructure failures (a full disk, a binary that vanished) raise
`ValueError` too, with `infrastructure failure` in the message; they are counted
as failed mutations like any other error, so grep for that string before
reading a spike in invalid programs as a result.

`request_timeout` bounds each call. When it fires — and when a call is
cancelled, and when it simply finishes — the whole process group is killed, not
just the leader, since harnesses spawn MCP servers and tool subprocesses that
outlive it.

**Knobs that matter.**

| Field | Why |
|---|---|
| `command` | The CLI and its flags. Argument list, never a shell string. |
| `schema_flag`, `answer_key` | Native structured output — the schema flag plus exactly one answer channel. The CLI option that takes a JSON Schema on the command line, and the stdout-envelope key the answer comes back under (`--json-schema` / `structured_output` for `claude -p`). Unset, the `OUTPUT.json` file handshake applies. |
| `schema_as_path` | `schema_flag` passes the workspace `SCHEMA.json` path instead of the schema text, for a flag that wants a file (`--output-schema` for `codex exec`). |
| `answer_file_flag` | The other answer channel: a flag naming a file the CLI itself writes the final message into (`--output-last-message` for `codex exec`); the answer is then read from `ANSWER.json`. Set this or `answer_key`, never both. |
| `strict_schema` | Rewrite the wire schema into the OpenAI strict-mode subset (objects closed and fully required, optionals nullable) and strip the invited nulls from the answer. For a backend whose schema flag lands in strict structured output (`codex exec`). |
| `system_flag` | Inline prompts — requires `schema_flag`. The CLI option that takes a system prompt on the command line (`--append-system-prompt` for `claude -p`); the user text then goes to stdin and no instruction is sent. Unset, the harness is told to read `SYSTEM.md`/`USER.md` itself. |
| `stdin_prompts` | The other way to inline — requires `schema_flag`, excludes `system_flag`. The whole prompt (system, then user) travels on stdin verbatim, for a CLI with no system flag whose sandboxed file reads cannot be trusted (`codex exec`). |
| `model_name`, `llm_base_url` | Global, and the backend's identity. `config/memory/v2.yaml` builds the memory `LLMFingerprint` from them, so cards produced by a harness never pool with cards produced by an API model. Change `model_name` whenever you change `command`. |
| `llm_max_concurrent` | Each harness process costs hundreds of MB and spawns children. Defaults to 4 here, not `null`. |
| `prefetch_factor` | Defaults to 1: prefetched DAGs are not free when every call forks a process. |
| `workspace_root` | Defaults under the system temp dir, deliberately outside the repository. It is only the parent: each chat creates its own `0700` directory beneath it, so two of them can share a root without colliding. The prompts and the answers are kept there, and shared storage is the usual home for it. |
| `env` | Layered onto the parent environment — point the harness's own state directory somewhere disposable. |

**Containment is your job.** `--allowedTools` is an auto-approval allowlist, not
a sandbox: a harness launched with `--allowedTools Read` will still run shell
commands. Use `--disallowedTools` (Claude Code) or `--sandbox read-only`
(Codex), and rely on the working directory being outside the repo.
`config/llm/harness.yaml` denies the write, egress, subprocess and scheduling
tools by name rather than trusting any mode's defaults, since the prompt
carries model-authored text by design — and with `system_flag` set it denies
`Read` too, because the prompts arrive on argv and stdin and no call needs to
open a file. If you unset `system_flag`, restore `--allowedTools Read`: the
file modes deliver `SYSTEM.md` and `USER.md` through it. The
workspace deliberately contains no evaluator and no copy of the problem — a
harness that honours the stay-in-this-directory instruction has nothing to
score its answer against. That is an instruction, not a wall: a tool that takes
absolute paths can still read elsewhere, which is what `--disallowedTools` and
a sandbox are for.

**Run the harness hermetically.** A coding CLI loads the operator's own
configuration by default — MCP servers, plugins, skills, `CLAUDE.md`. That
config then reaches every mutation: its tool schemas are prepended to each turn,
so an unrelated MCP server is both a per-turn token cost and a live capability
the harness can call. It also makes a run irreproducible, since the prompt now
depends on what the operator happened to have installed. `config/llm/harness.yaml`
passes `--safe-mode` for this reason; measured on a one-turn probe it cut fixed
overhead from 30.4k to 4.8k tokens, and on real heilbron mutations at matched
turn count 15% off input and 17% off cost, with no change to the answer. Compare only at matched
turn count: an agentic call's total is dominated by how many turns it took, and
that varies run to run on identical input. The
equivalent on another CLI is whatever disables user-level config — do it, and
treat a harness that cannot as a harness that leaks. `--exclude-dynamic-system-prompt-sections`
completes the picture: it moves the per-call working directory out of the system
prompt, which otherwise differs on every call and breaks the cached prefix.

What the backend does enforce is narrow, and worth knowing exactly: the harness
gets its own session (keeping it out of GigaEvo's signal group — a same-UID
process can still signal by PID on purpose), its process group is killed when
the call ends however it ends, and an answer file that resolves outside its
workspace — `OUTPUT.json`, or `STDOUT.log` under `schema_flag`, which is also
refused if it is no longer a regular file — is refused. What it does **not** do is confine the filesystem or
the environment — the harness inherits this process's full environment,
including every API key in it, and `env` only layers on top. If that matters,
run the harness under a sandbox of its own; a `command` may name any wrapper.
Treat prompt content as untrusted by the same logic: `USER.md` holds
model-authored text, and a capable agent reads it as instructions.

**Token counts are optional, and opt-in on the harness's side.** If `STDOUT.log`
holds a single JSON object with a `usage` mapping, its counts are reported on
the normal metadata channels, so `TokenTracker`, the `llm/tokens/*` metrics, the
`LLMCall` events and the I/O dump all fill in as they do for an HTTP model.
`claude -p --output-format json` prints exactly that — which is why
`config/llm/harness.yaml` passes those flags — and adds a `total_cost_usd` and
a `num_turns` the backend carries on `response_metadata`: a CLI billed against
a subscription is the only thing that knows what the call cost, and the turn
count is the dominant cost variable of an agentic backend. `TokenTracker`
persists both — harness responses add `cost_usd`, `cumulative_cost_usd`,
`num_turns` and `cumulative_num_turns` scalars beside the token series, per
model and per stage. API models report neither and emit no such series, so the
harness panels are not buried under zero-filled noise.

Nothing else is required of a harness — under the default file handshake. One
that prints prose, or an event stream, or nothing at all reports zeros, exactly
as this backend did before counts existed; so does a `STDOUT.log` too large to
be a result envelope. There the answer never comes from stdout, so no harness
is degraded by staying quiet. Under `schema_flag` the opposite holds: the
envelope carries the answer, so a missing, oversized (the cap is 1 MiB),
unparseable or answer-less envelope is a failed call, reported with the
envelope's own text quoted — the oversized case reports its size instead — so
a quota or auth error surfaces in the run log.

Read the input total with the harness in mind: it includes cache reads and
writes, which are billed input and are most of an agentic CLI's spend. A single
call can book six figures of input for a one-line answer, and that is real —
`input_token_details` splits out `cache_read` and `cache_creation`, which are
priced differently from fresh input.

Keep `structured_output_method: json_schema` (as `config/llm/harness.yaml`
does). Under `structured_output_method=auto` the router retries a failed call
once per wire format, which for a harness means running the same subprocess
three times for one failure.

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
| `llm` | `single`, `heterogeneous`, `heterogeneous_bandit`, `balanced`, `openrouter_bandit`, `openrouter_ensemble`, `google`, `openai`, `gemini3_flash`, `gemini3_flash_high` (fast gemini-3-flash, reasoning=high), `gemini35_flash`, `gpt54_mini`, `zai`, `qwen_thinking`, `llama31_8b`, `local_proxy`, `harness` (an agentic coding CLI instead of an HTTP endpoint — see [Agentic Coding Harness Backend](#agentic-coding-harness-backend)) |
| `pipeline` | `guided` (default), `memory_guided` (see [MEMORY_GUIDED_PIPELINE.md](MEMORY_GUIDED_PIPELINE.md)), `custom`, `structural_metrics`, `adversarial`, `adversarial_asymmetric`, `adversarial_coevo`, `prompt_evolution`, `optuna_opt` |
| `archive_selector` | `point` (default — replace elite iff weighted fitness sum is higher), `paired_bootstrap` (noise-aware paired bootstrap gate, knob: `archive_selector.p_accept`; needs a validator that emits `per_sample_scores` through `_program_metadata`) |
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
    pipeline=memory_guided memory=v2 memory/write=live \
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
